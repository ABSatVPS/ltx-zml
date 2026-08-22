//! The LTX-2.5 video transformer block as a traced ZML model — the
//! shared definition consumed by //ltx:block_conformance (its 22-gate
//! oracle suite is this module's regression fence) and by the engine.
//! Extracted verbatim from block_conformance.zig at the E-STREAM-3
//! refactor (notebook 2026-08-21/22). Since the production-length rung
//! (2026-08-21 afternoon), all sequence geometry is derived from the
//! input tensors' dims at trace time; the T/S consts below are only the
//! conformance-harness defaults its binaries build their specs from.
const std = @import("std");
const zml = @import("zml");

pub const T: i64 = 64; // video tokens in the bundle
pub const S: i64 = 32; // context (prompt) tokens
pub const D: i64 = 4096;
pub const H: i64 = 32;
pub const HD: i64 = 128;
pub const FF: i64 = 16384;
pub const EPS: f32 = 1e-6;

// ---- the block as a traced model ------------------------------------------

pub fn lin(x: zml.Tensor, w: zml.Tensor, b: ?zml.Tensor) zml.Tensor {
    var y = x.dot(w.convert(.f32), .i);
    if (b) |bias| y = y.add(bias.convert(.f32).broad(y.shape()));
    return y.withTags(.{ .t, .i });
}

pub fn rmsNoW(x: zml.Tensor) zml.Tensor {
    return zml.nn.rmsNorm(x, .i, EPS);
}

pub fn rmsW(x: zml.Tensor, w: zml.Tensor) zml.Tensor {
    return rmsNoW(x).mul(w.convert(.f32).withTags(.{.i}).broad(x.shape()));
}

pub fn modulate(n: zml.Tensor, scale: zml.Tensor, shift: zml.Tensor) zml.Tensor {
    return n.mul(scale.addConstant(1.0)).add(shift);
}

/// ada value i: per-block table row + per-token timestep chunk. ts3 is
/// [.t, .n=9, .i], table [.n, .i].
pub fn adaVal(table: zml.Tensor, ts3: zml.Tensor, i: i64) zml.Tensor {
    const row = table.convert(.f32).slice1d(.n, .{ .start = i, .end = i + 1 }).squeeze(.n); // [.i]
    const tv = ts3.slice1d(.n, .{ .start = i, .end = i + 1 }).squeeze(.n); // [.t,.i]
    return tv.add(row.broad(tv.shape()));
}

/// split-half RoPE: x4 [.., .h, .p=2, .f=64] (leading tag .q or .k), cos/sin
/// tagged to match x4's leading tag: [lead, .h, .f].
pub fn rope(x4: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
    const a = x4.slice1d(.p, .{ .start = 0, .end = 1 }).squeeze(.p);
    const b = x4.slice1d(.p, .{ .start = 1, .end = 2 }).squeeze(.p);
    const ra = a.mul(cos).sub(b.mul(sin));
    const rb = b.mul(cos).add(a.mul(sin));
    return zml.Tensor.concatenate(&.{ ra, rb }, .f);
}

/// Which SDPA the block traces: XLA's dense softmax path, or the E3w
/// stablehlo.while online-softmax kernel — the production attention at
/// T≈28k, where dense scores would need ~100 GB (run 7/8b).
pub const Algo = enum { dense, blockwise };

/// E3w chunk for the f32 twin at HARNESS length. 16 gives 4 iterations
/// over T=64, so the rescale-correction path runs 3 times; a single chunk
/// of T would degenerate into plain softmax and test only loop plumbing.
pub const ACHUNK: i64 = 16;

/// E3w chunk at PRODUCTION length. E3's smoke measured 1.22 s at
/// WCHUNK=1024; the H-PROD walk proved why this cannot be ACHUNK (16 at
/// T=28,672 = 1,792 latency-bound sliver GEMMs, 27.9 s/block). 512 was
/// TRIED to halve the while workspace and measured 4x SLOWER
/// (5.9 s/block, run 6) — 1024 is the plateau's edge, keep it.
pub const PCHUNK: i64 = 1024;

/// The E3w kernel's f32 twin: the same loop-carried (step, m, l, acc)
/// while as smoke.zig's blockwiseWhileSdpaGraph, but f32 matmuls (the
/// harness dtype) and the 1/sqrt(hd) scale applied to K exactly where
/// zml.nn.sdpa applies it — so the agreement gate sees ONLY the
/// online-softmax algorithm. q3 [.q,.h,.hd], k3/v3 [.k,.h,.hd]; returns
/// [.h,.q,.hd] (dot leads with batch dims).
/// Trace-time chunk selection: production geometry when the K length
/// divides by it, harness geometry otherwise. Keeps every T=64 graph
/// BITWISE identical to the pre-PCHUNK record (the 22-gate fence).
pub fn whileSdpa(q3: zml.Tensor, k3: zml.Tensor, v3: zml.Tensor) zml.Tensor {
    if (@rem(k3.dim(.k), PCHUNK) == 0) return whileSdpaC(PCHUNK, q3, k3, v3);
    return whileSdpaC(ACHUNK, q3, k3, v3);
}

pub fn whileSdpaC(comptime chunk: i64, q3: zml.Tensor, k3: zml.Tensor, v3: zml.Tensor) zml.Tensor {
    const tq = q3.dim(.q); // trace-time; T is only the harness default
    std.debug.assert(@rem(k3.dim(.k), chunk) == 0);
    const n_chunks: i64 = @divExact(k3.dim(.k), chunk);
    const hd_f: f32 = @floatFromInt(HD);
    const k = k3.mul(zml.Tensor.scalar(1.0 / @sqrt(hd_f), .f32));

    const hq_shape = zml.Shape.init(.{ .h = H, .q = tq }, .f32);
    const acc_shape = zml.Shape.init(.{ .h = H, .q = tq, .hd = HD }, .f32);
    // Finite lowest (not -inf): exp(m0 - m_new) underflows cleanly to 0 on
    // the first iteration instead of -inf minus -inf = NaN.
    const m0 = zml.Tensor.scalar(-std.math.floatMax(f32), .f32).broad(hq_shape);
    const l0 = zml.Tensor.scalar(0, .f32).broad(hq_shape);
    const acc0 = zml.Tensor.scalar(0, .f32).broad(acc_shape);

    const Ctx = struct { q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, n: zml.Tensor };
    const Local = struct {
        fn cond(step: zml.Tensor, _: zml.Tensor, _: zml.Tensor, _: zml.Tensor, c: Ctx) zml.Tensor {
            return step.cmp(.LT, c.n);
        }
        fn body(step: zml.Tensor, m: zml.Tensor, l: zml.Tensor, acc: zml.Tensor, c: Ctx) [4]zml.Tensor {
            const off = step.mul(zml.Tensor.scalar(chunk, .i32));
            const kc = c.k.dynamicSlice(.{ .k = zml.Tensor.DynSlice{ .start = off, .len = chunk } });
            const vc = c.v.dynamicSlice(.{ .k = zml.Tensor.DynSlice{ .start = off, .len = chunk } });
            const s = c.q.dot(kc, .hd); // [.h,.q,.k]
            const cmax = s.max(.k).squeeze(.k); // [.h,.q]
            const m_new = m.maximum(cmax);
            const corr = m.sub(m_new).exp();
            const p = s.sub(m_new.broad(s.shape())).exp();
            const l_new = l.mul(corr).add(p.sum(.k).squeeze(.k));
            const pv = p.dot(vc, .k); // [.h,.q,.hd]
            const acc_new = acc.mul(corr.broad(acc.shape())).add(pv);
            return .{ step.addConstant(1), m_new, l_new, acc_new };
        }
    };

    const ctx: Ctx = .{ .q = q3, .k = k, .v = v3, .n = zml.Tensor.scalar(n_chunks, .i32) };
    const res = zml.ops.@"while"(
        .{ zml.Tensor.scalar(0, .i32), m0, l0, acc0 },
        Local.cond,
        Local.body,
        .{ctx},
    );
    return res[3].div(res[2].broad(acc_shape));
}

pub const Block = struct {
    q1_w: zml.Tensor,
    q1_b: zml.Tensor,
    k1_w: zml.Tensor,
    k1_b: zml.Tensor,
    v1_w: zml.Tensor,
    v1_b: zml.Tensor,
    o1_w: zml.Tensor,
    o1_b: zml.Tensor,
    qn1: zml.Tensor,
    kn1: zml.Tensor,
    g1_w: zml.Tensor,
    g1_b: zml.Tensor,
    q2_w: zml.Tensor,
    q2_b: zml.Tensor,
    k2_w: zml.Tensor,
    k2_b: zml.Tensor,
    v2_w: zml.Tensor,
    v2_b: zml.Tensor,
    o2_w: zml.Tensor,
    o2_b: zml.Tensor,
    qn2: zml.Tensor,
    kn2: zml.Tensor,
    g2_w: zml.Tensor,
    g2_b: zml.Tensor,
    ff1_w: zml.Tensor,
    ff2_w: zml.Tensor,
    sst: zml.Tensor, // [9, D] f32
    psst: zml.Tensor, // [2, D] f32

    const A = struct { // one attention's weights, picked at trace time
        qw: zml.Tensor,
        qb: zml.Tensor,
        kw: zml.Tensor,
        kb: zml.Tensor,
        vw: zml.Tensor,
        vb: zml.Tensor,
        ow: zml.Tensor,
        ob: zml.Tensor,
        qn: zml.Tensor,
        kn: zml.Tensor,
        gw: zml.Tensor,
        gb: zml.Tensor,
    };

    fn attn1w(self: @This()) A {
        return .{ .qw = self.q1_w, .qb = self.q1_b, .kw = self.k1_w, .kb = self.k1_b, .vw = self.v1_w, .vb = self.v1_b, .ow = self.o1_w, .ob = self.o1_b, .qn = self.qn1, .kn = self.kn1, .gw = self.g1_w, .gb = self.g1_b };
    }

    fn attn2w(self: @This()) A {
        return .{ .qw = self.q2_w, .qb = self.q2_b, .kw = self.k2_w, .kb = self.k2_b, .vw = self.v2_w, .vb = self.v2_b, .ow = self.o2_w, .ob = self.o2_b, .qn = self.qn2, .kn = self.kn2, .gw = self.g2_w, .gb = self.g2_b };
    }

    /// Attention per the reference: qk-norm (weighted, full 4096) then
    /// optional RoPE, sdpa, optional 2*sigmoid per-head gate from the
    /// attention INPUT, then output projection.
    fn attention(w: A, x: zml.Tensor, ctx: zml.Tensor, n_kv: i64, pe: ?struct { cos: zml.Tensor, sin: zml.Tensor }, comptime gated: bool, comptime algo: Algo) zml.Tensor {
        const tq = x.dim(.t);
        var q = rmsW(lin(x, w.qw, w.qb), w.qn);
        var k = rmsW(lin(ctx, w.kw, w.kb), w.kn);
        const v = lin(ctx, w.vw, w.vb);

        var q3: zml.Tensor = undefined;
        var k3: zml.Tensor = undefined;
        if (pe) |p| {
            const q4 = q.reshape(zml.Shape.init(.{ .q = tq, .h = H, .p = 2, .f = 64 }, .f32));
            const k4 = k.reshape(zml.Shape.init(.{ .k = n_kv, .h = H, .p = 2, .f = 64 }, .f32));
            const cos_k = p.cos.withTags(.{ .k, .h, .f });
            const sin_k = p.sin.withTags(.{ .k, .h, .f });
            q3 = rope(q4, p.cos, p.sin).withTags(.{ .q, .h, .hd });
            k3 = rope(k4, cos_k, sin_k).withTags(.{ .k, .h, .hd });
        } else {
            q3 = q.reshape(zml.Shape.init(.{ .q = tq, .h = H, .hd = HD }, .f32));
            k3 = k.reshape(zml.Shape.init(.{ .k = n_kv, .h = H, .hd = HD }, .f32));
        }
        const v3 = v.reshape(zml.Shape.init(.{ .k = n_kv, .h = H, .hd = HD }, .f32));

        var out = switch (algo) {
            .dense => zml.nn.sdpa(q3, k3, v3, .{}), // [.h,.q,.hd]-tagged
            .blockwise => whileSdpa(q3, k3, v3),
        };
        if (gated) {
            const glog = lin(x, w.gw, w.gb).withTags(.{ .q, .h }); // [T, 32]
            const gate = glog.sigmoid().scale(2.0);
            out = out.mul(gate.broad(out.shape()));
        }
        const merged = out.transpose(.{ .q, .h, .hd }).reshape(zml.Shape.init(.{ .t = tq, .i = D }, .f32));
        return lin(merged, w.ow, w.ob);
    }

    // ---- stages, each recomputing its prefix (pre-registered order) ----

    pub fn s1NormMsa(self: @This(), x: zml.Tensor, ts3: zml.Tensor) zml.Tensor {
        const shift = adaVal(self.sst, ts3, 0);
        const scl = adaVal(self.sst, ts3, 1);
        return modulate(rmsNoW(x), scl, shift);
    }

    pub fn s1bQnorm(self: @This(), x: zml.Tensor, ts3: zml.Tensor) zml.Tensor {
        const n = self.s1NormMsa(x, ts3);
        return rmsW(lin(n, self.q1_w, self.q1_b), self.qn1);
    }

    pub fn s2Attn1NoGate(self: @This(), x: zml.Tensor, ts3: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const n = self.s1NormMsa(x, ts3);
        return attention(self.attn1w(), n, n, n.dim(.t), .{ .cos = cos, .sin = sin }, false, .dense);
    }

    fn attn1Out(self: @This(), x: zml.Tensor, ts3: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor, comptime algo: Algo) zml.Tensor {
        const n = self.s1NormMsa(x, ts3);
        return attention(self.attn1w(), n, n, n.dim(.t), .{ .cos = cos, .sin = sin }, true, algo);
    }

    pub fn s3Attn1(self: @This(), x: zml.Tensor, ts3: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        return self.attn1Out(x, ts3, cos, sin, .dense);
    }

    /// E3w agreement probe: gated attn1 through the while kernel.
    pub fn wAttn1(self: @This(), x: zml.Tensor, ts3: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        return self.attn1Out(x, ts3, cos, sin, .blockwise);
    }

    fn afterSa(self: @This(), x: zml.Tensor, ts3: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor, comptime algo: Algo) zml.Tensor {
        const gate = adaVal(self.sst, ts3, 2);
        return x.add(self.attn1Out(x, ts3, cos, sin, algo).mul(gate));
    }

    pub fn s4AfterSa(self: @This(), x: zml.Tensor, ts3: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        return self.afterSa(x, ts3, cos, sin, .dense);
    }

    pub fn s4Normed(self: @This(), x: zml.Tensor, ts3: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        return rmsNoW(self.afterSa(x, ts3, cos, sin, .dense));
    }

    fn caOut(self: @This(), x_normed: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor) zml.Tensor {
        const shift_q = adaVal(self.sst, ts3, 6);
        const scale_q = adaVal(self.sst, ts3, 7);
        const gate_ca = adaVal(self.sst, ts3, 8);
        const attn_in = modulate(x_normed, scale_q, shift_q);
        // K/V modulation: per-block prompt table + prompt timestep, rows
        // (shift, scale).
        const kv = self.psst.convert(.f32).add(pts2); // [.n=2,.i]
        const shift_kv = kv.slice1d(.n, .{ .start = 0, .end = 1 }).squeeze(.n);
        const scale_kv = kv.slice1d(.n, .{ .start = 1, .end = 2 }).squeeze(.n);
        const enc = ctx.mul(scale_kv.broad(ctx.shape()).addConstant(1.0)).add(shift_kv.broad(ctx.shape()));
        // Cross-attention stays dense at every algo (pre-registered scope:
        // S=32 here, ~1k at deployment — the E3w memory argument is absent).
        const ca = attention(self.attn2w(), attn_in, enc, enc.dim(.t), null, true, .dense);
        return ca.mul(gate_ca);
    }

    fn s5CaOutA(self: @This(), x: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor, comptime algo: Algo) zml.Tensor {
        return self.caOut(rmsNoW(self.afterSa(x, ts3, cos, sin, algo)), ts3, ctx, pts2);
    }

    pub fn s5CaOut(self: @This(), x: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        return self.s5CaOutA(x, ts3, ctx, pts2, cos, sin, .dense);
    }

    fn afterCa(self: @This(), x: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor, comptime algo: Algo) zml.Tensor {
        return self.afterSa(x, ts3, cos, sin, algo).add(self.s5CaOutA(x, ts3, ctx, pts2, cos, sin, algo));
    }

    fn s6FfInA(self: @This(), x: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor, comptime algo: Algo) zml.Tensor {
        const shift = adaVal(self.sst, ts3, 3);
        const scl = adaVal(self.sst, ts3, 4);
        return modulate(rmsNoW(self.afterCa(x, ts3, ctx, pts2, cos, sin, algo)), scl, shift);
    }

    pub fn s6FfIn(self: @This(), x: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        return self.s6FfInA(x, ts3, ctx, pts2, cos, sin, .dense);
    }

    fn s7FfOutA(self: @This(), x: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor, comptime algo: Algo) zml.Tensor {
        const ff_in = self.s6FfInA(x, ts3, ctx, pts2, cos, sin, algo);
        const h = lin(ff_in, self.ff1_w, null).gelu();
        return lin(h, self.ff2_w, null);
    }

    pub fn s7FfOut(self: @This(), x: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        return self.s7FfOutA(x, ts3, ctx, pts2, cos, sin, .dense);
    }

    fn blockOutA(self: @This(), x: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor, comptime algo: Algo) zml.Tensor {
        const gate = adaVal(self.sst, ts3, 5);
        return self.afterCa(x, ts3, ctx, pts2, cos, sin, algo)
            .add(self.s7FfOutA(x, ts3, ctx, pts2, cos, sin, algo).mul(gate));
    }

    pub fn blockOut(self: @This(), x: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        return self.blockOutA(x, ts3, ctx, pts2, cos, sin, .dense);
    }

    /// E3w swap: the whole block with while attention in attn1.
    pub fn wBlockOut(self: @This(), x: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        return self.blockOutA(x, ts3, ctx, pts2, cos, sin, .blockwise);
    }

    /// f32 attn1 pre-softmax logits — probe 1's baseline for the int4 rung.
    pub fn fLogits(self: @This(), x: zml.Tensor, ts3: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const n = self.s1NormMsa(x, ts3);
        const q = rmsW(lin(n, self.q1_w, self.q1_b), self.qn1);
        const k = rmsW(lin(n, self.k1_w, self.k1_b), self.kn1);
        const q4 = q.reshape(zml.Shape.init(.{ .q = q.dim(.t), .h = H, .p = 2, .f = 64 }, .f32));
        const k4 = k.reshape(zml.Shape.init(.{ .k = k.dim(.t), .h = H, .p = 2, .f = 64 }, .f32));
        const q3 = rope(q4, cos, sin).withTags(.{ .q, .h, .hd });
        const k3 = rope(k4, cos.withTags(.{ .k, .h, .f }), sin.withTags(.{ .k, .h, .f })).withTags(.{ .k, .h, .hd });
        return q3.dot(k3, .hd);
    }
};

// ---- weight loading --------------------------------------------------------

pub const WSpec = struct {
    field: []const u8,
    file: []const u8,
    dims: []const i64,
    f32_: bool = false,
};

pub const WEIGHT_SPECS = [_]WSpec{
    .{ .field = "q1_w", .file = "attn1_to_q_weight.bin", .dims = &.{ D, D } },
    .{ .field = "q1_b", .file = "attn1_to_q_bias.bin", .dims = &.{D} },
    .{ .field = "k1_w", .file = "attn1_to_k_weight.bin", .dims = &.{ D, D } },
    .{ .field = "k1_b", .file = "attn1_to_k_bias.bin", .dims = &.{D} },
    .{ .field = "v1_w", .file = "attn1_to_v_weight.bin", .dims = &.{ D, D } },
    .{ .field = "v1_b", .file = "attn1_to_v_bias.bin", .dims = &.{D} },
    .{ .field = "o1_w", .file = "attn1_to_out_0_weight.bin", .dims = &.{ D, D } },
    .{ .field = "o1_b", .file = "attn1_to_out_0_bias.bin", .dims = &.{D} },
    .{ .field = "qn1", .file = "attn1_q_norm_weight.bin", .dims = &.{D} },
    .{ .field = "kn1", .file = "attn1_k_norm_weight.bin", .dims = &.{D} },
    .{ .field = "g1_w", .file = "attn1_to_gate_logits_weight.bin", .dims = &.{ H, D } },
    .{ .field = "g1_b", .file = "attn1_to_gate_logits_bias.bin", .dims = &.{H} },
    .{ .field = "q2_w", .file = "attn2_to_q_weight.bin", .dims = &.{ D, D } },
    .{ .field = "q2_b", .file = "attn2_to_q_bias.bin", .dims = &.{D} },
    .{ .field = "k2_w", .file = "attn2_to_k_weight.bin", .dims = &.{ D, D } },
    .{ .field = "k2_b", .file = "attn2_to_k_bias.bin", .dims = &.{D} },
    .{ .field = "v2_w", .file = "attn2_to_v_weight.bin", .dims = &.{ D, D } },
    .{ .field = "v2_b", .file = "attn2_to_v_bias.bin", .dims = &.{D} },
    .{ .field = "o2_w", .file = "attn2_to_out_0_weight.bin", .dims = &.{ D, D } },
    .{ .field = "o2_b", .file = "attn2_to_out_0_bias.bin", .dims = &.{D} },
    .{ .field = "qn2", .file = "attn2_q_norm_weight.bin", .dims = &.{D} },
    .{ .field = "kn2", .file = "attn2_k_norm_weight.bin", .dims = &.{D} },
    .{ .field = "g2_w", .file = "attn2_to_gate_logits_weight.bin", .dims = &.{ H, D } },
    .{ .field = "g2_b", .file = "attn2_to_gate_logits_bias.bin", .dims = &.{H} },
    .{ .field = "ff1_w", .file = "ff_net_0_proj_weight.bin", .dims = &.{ FF, D } },
    .{ .field = "ff2_w", .file = "ff_net_2_weight.bin", .dims = &.{ D, FF } },
    .{ .field = "sst", .file = "scale_shift_table.bin", .dims = &.{ 9, D }, .f32_ = true },
    .{ .field = "psst", .file = "prompt_scale_shift_table.bin", .dims = &.{ 2, D }, .f32_ = true },
};

pub fn weightShape(spec: WSpec) zml.Shape {
    const dt: zml.DataType = if (spec.f32_) .f32 else .bf16;
    return switch (spec.dims.len) {
        1 => if (std.mem.eql(u8, spec.field, "g1_b") or std.mem.eql(u8, spec.field, "g2_b"))
            zml.Shape.init(.{ .o = H }, dt)
        else
            zml.Shape.init(.{ .o = spec.dims[0] }, dt),
        2 => if (spec.f32_)
            zml.Shape.init(.{ .n = spec.dims[0], .i = spec.dims[1] }, dt)
        else
            zml.Shape.init(.{ .o = spec.dims[0], .i = spec.dims[1] }, dt),
        else => unreachable,
    };
}

/// Fresh tracer spec tensors for one block. MUST be called once per block
/// instance in a multi-block model: reusing one spec set across chain
/// fields would trip the tracer's duplicate-argument check (run 1 lesson).
pub fn makeBlockSpecs() Block {
    var m: Block = undefined;
    inline for (WEIGHT_SPECS) |spec| {
        @field(m, spec.field) = zml.Tensor.fromShape(weightShape(spec));
    }
    return m;
}

/// Phase 3 checkpoint 1: blocks 0, 23, 47 chained, boundary-gated. Shared
/// timestep/context/pe per LTXModel semantics; only x flows.
pub const Chain = struct {
    b0: Block,
    b23: Block,
    b47: Block,

    pub fn chainB23(self: @This(), x: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        return self.b23.blockOut(self.b0.blockOut(x, ts3, ctx, pts2, cos, sin), ts3, ctx, pts2, cos, sin);
    }

    pub fn chainB47(self: @This(), x: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        return self.b47.blockOut(self.chainB23(x, ts3, ctx, pts2, cos, sin), ts3, ctx, pts2, cos, sin);
    }

    /// E3w compounding probe: while attention in all three blocks.
    pub fn wChainB47(self: @This(), x: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const y0 = self.b0.wBlockOut(x, ts3, ctx, pts2, cos, sin);
        const y23 = self.b23.wBlockOut(y0, ts3, ctx, pts2, cos, sin);
        return self.b47.wBlockOut(y23, ts3, ctx, pts2, cos, sin);
    }
};

// ---- int4 rung (qkv class): dequant-in-graph per E1b ----------------------

/// Grouped-128 dequant (rung 2 of the escalation ladder; the per-row rung 1
/// measured 0.35/0.25/0.17 against budgets 5e-2/2e-2/2e-2 — the coli-zml
/// grouped-scale pattern, sc tagged [.o,.g]).
pub fn dq4(w: zml.Tensor, sc: zml.Tensor) zml.Tensor {
    var f = w.convert(.f32);
    if (w.dtype() == .u4) f = f.addConstant(-8.0); // i8 is signed already
    const o = f.shape().dim(0);
    const i_dim = f.shape().dim(1);
    const ng = sc.shape().dim(1);
    const f3 = f.reshape(zml.Shape.init(.{ .o = o, .g = ng, .e = @divExact(i_dim, ng) }, .f32));
    const m = f3.mul(sc.broad(f3.shape()));
    return m.reshape(zml.Shape.init(.{ .o = o, .i = i_dim }, .f32));
}

pub fn linQ(x: zml.Tensor, wq: zml.Tensor, sc: zml.Tensor, b: ?zml.Tensor) zml.Tensor {
    var y = x.dot(dq4(wq, sc), .i);
    if (b) |bias| y = y.add(bias.convert(.f32).broad(y.shape()));
    return y.withTags(.{ .t, .i });
}

/// Block plus int4 q/k/v for both attentions (rung 1 of the pre-registered
/// int4 ladder). Everything else rides on `base` in f32/bf16.
pub const QBlock = struct {
    base: Block,
    q1q: zml.Tensor,
    q1s: zml.Tensor,
    k1q: zml.Tensor,
    k1s: zml.Tensor,
    v1q: zml.Tensor,
    v1s: zml.Tensor,
    q2q: zml.Tensor,
    q2s: zml.Tensor,
    k2q: zml.Tensor,
    k2s: zml.Tensor,
    v2q: zml.Tensor,
    v2s: zml.Tensor,
    f1q: zml.Tensor,
    f1s: zml.Tensor,
    f2q: zml.Tensor,
    f2s: zml.Tensor,

    fn qkRoped(self: @This(), n: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) struct { q3: zml.Tensor, k3: zml.Tensor } {
        const q = rmsW(linQ(n, self.q1q, self.q1s, self.base.q1_b), self.base.qn1);
        const k = rmsW(linQ(n, self.k1q, self.k1s, self.base.k1_b), self.base.kn1);
        const q4 = q.reshape(zml.Shape.init(.{ .q = q.dim(.t), .h = H, .p = 2, .f = 64 }, .f32));
        const k4 = k.reshape(zml.Shape.init(.{ .k = k.dim(.t), .h = H, .p = 2, .f = 64 }, .f32));
        return .{
            .q3 = rope(q4, cos, sin).withTags(.{ .q, .h, .hd }),
            .k3 = rope(k4, cos.withTags(.{ .k, .h, .f }), sin.withTags(.{ .k, .h, .f })).withTags(.{ .k, .h, .hd }),
        };
    }

    /// Probe 1: attn1 pre-softmax logits, int4 q/k. Compared against the
    /// f32 twin so softmax amplification is separable from projection error.
    pub fn qLogits(self: @This(), x: zml.Tensor, ts3: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const n = self.base.s1NormMsa(x, ts3);
        const qk = self.qkRoped(n, cos, sin);
        return qk.q3.dot(qk.k3, .hd);
    }

    /// Probe 2: full gated attn1 with int4 q/k/v.
    pub fn qAttn1(self: @This(), x: zml.Tensor, ts3: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const n = self.base.s1NormMsa(x, ts3);
        const qk = self.qkRoped(n, cos, sin);
        const v = linQ(n, self.v1q, self.v1s, self.base.v1_b);
        const v3 = v.reshape(zml.Shape.init(.{ .k = v.dim(.t), .h = H, .hd = HD }, .f32));
        var out = zml.nn.sdpa(qk.q3, qk.k3, v3, .{});
        const glog = lin(n, self.base.g1_w, self.base.g1_b).withTags(.{ .q, .h });
        out = out.mul(glog.sigmoid().scale(2.0).broad(out.shape()));
        const merged = out.transpose(.{ .q, .h, .hd }).reshape(zml.Shape.init(.{ .t = n.dim(.t), .i = D }, .f32));
        return lin(merged, self.base.o1_w, self.base.o1_b);
    }

    fn qFf(self: @This(), ff_in: zml.Tensor) zml.Tensor {
        return linQ(linQ(ff_in, self.f1q, self.f1s, null).gelu(), self.f2q, self.f2s, null);
    }

    /// FFN-isolated probe: the f32 path up to the REAL ff_in, then the
    /// quantized FFN — so the comparison against f32 s7FfOut sees only
    /// this class's error.
    pub fn qFfOut(self: @This(), x: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        return self.qFf(self.base.s6FfIn(x, ts3, ctx, pts2, cos, sin));
    }

    /// The whole block with quantized qkv AND quantized FFN.
    pub fn qBlockOutAll(self: @This(), x: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const b = self.base;
        const gate_msa = adaVal(b.sst, ts3, 2);
        const x_sa = x.add(self.qAttn1(x, ts3, cos, sin).mul(gate_msa));
        const x_normed = rmsNoW(x_sa);
        const shift_q = adaVal(b.sst, ts3, 6);
        const scale_q = adaVal(b.sst, ts3, 7);
        const gate_ca = adaVal(b.sst, ts3, 8);
        const attn_in = modulate(x_normed, scale_q, shift_q);
        const kv = b.psst.convert(.f32).add(pts2);
        const shift_kv = kv.slice1d(.n, .{ .start = 0, .end = 1 }).squeeze(.n);
        const scale_kv = kv.slice1d(.n, .{ .start = 1, .end = 2 }).squeeze(.n);
        const enc = ctx.mul(scale_kv.broad(ctx.shape()).addConstant(1.0)).add(shift_kv.broad(ctx.shape()));
        const q = rmsW(linQ(attn_in, self.q2q, self.q2s, b.q2_b), b.qn2);
        const k = rmsW(linQ(enc, self.k2q, self.k2s, b.k2_b), b.kn2);
        const v = linQ(enc, self.v2q, self.v2s, b.v2_b);
        const q3 = q.reshape(zml.Shape.init(.{ .q = q.dim(.t), .h = H, .hd = HD }, .f32));
        const k3 = k.reshape(zml.Shape.init(.{ .k = k.dim(.t), .h = H, .hd = HD }, .f32));
        const v3 = v.reshape(zml.Shape.init(.{ .k = v.dim(.t), .h = H, .hd = HD }, .f32));
        var ca = zml.nn.sdpa(q3, k3, v3, .{});
        const glog = lin(attn_in, b.g2_w, b.g2_b).withTags(.{ .q, .h });
        ca = ca.mul(glog.sigmoid().scale(2.0).broad(ca.shape()));
        const merged = ca.transpose(.{ .q, .h, .hd }).reshape(zml.Shape.init(.{ .t = attn_in.dim(.t), .i = D }, .f32));
        const ca_out = lin(merged, b.o2_w, b.o2_b).mul(gate_ca);
        const x_ca = x_sa.add(ca_out);
        const shift_mlp = adaVal(b.sst, ts3, 3);
        const scale_mlp = adaVal(b.sst, ts3, 4);
        const gate_mlp = adaVal(b.sst, ts3, 5);
        const ff_in = modulate(rmsNoW(x_ca), scale_mlp, shift_mlp);
        return x_ca.add(self.qFf(ff_in).mul(gate_mlp));
    }

    /// Probe 3: the whole block with int4 qkv in BOTH attentions.
    pub fn qBlockOut(self: @This(), x: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const b = self.base;
        const gate_msa = adaVal(b.sst, ts3, 2);
        const x_sa = x.add(self.qAttn1(x, ts3, cos, sin).mul(gate_msa));
        const x_normed = rmsNoW(x_sa);
        // cross-attention with int4 q/k/v
        const shift_q = adaVal(b.sst, ts3, 6);
        const scale_q = adaVal(b.sst, ts3, 7);
        const gate_ca = adaVal(b.sst, ts3, 8);
        const attn_in = modulate(x_normed, scale_q, shift_q);
        const kv = b.psst.convert(.f32).add(pts2);
        const shift_kv = kv.slice1d(.n, .{ .start = 0, .end = 1 }).squeeze(.n);
        const scale_kv = kv.slice1d(.n, .{ .start = 1, .end = 2 }).squeeze(.n);
        const enc = ctx.mul(scale_kv.broad(ctx.shape()).addConstant(1.0)).add(shift_kv.broad(ctx.shape()));
        const q = rmsW(linQ(attn_in, self.q2q, self.q2s, b.q2_b), b.qn2);
        const k = rmsW(linQ(enc, self.k2q, self.k2s, b.k2_b), b.kn2);
        const v = linQ(enc, self.v2q, self.v2s, b.v2_b);
        const q3 = q.reshape(zml.Shape.init(.{ .q = q.dim(.t), .h = H, .hd = HD }, .f32));
        const k3 = k.reshape(zml.Shape.init(.{ .k = k.dim(.t), .h = H, .hd = HD }, .f32));
        const v3 = v.reshape(zml.Shape.init(.{ .k = v.dim(.t), .h = H, .hd = HD }, .f32));
        var ca = zml.nn.sdpa(q3, k3, v3, .{});
        const glog = lin(attn_in, b.g2_w, b.g2_b).withTags(.{ .q, .h });
        ca = ca.mul(glog.sigmoid().scale(2.0).broad(ca.shape()));
        const merged = ca.transpose(.{ .q, .h, .hd }).reshape(zml.Shape.init(.{ .t = attn_in.dim(.t), .i = D }, .f32));
        const ca_out = lin(merged, b.o2_w, b.o2_b).mul(gate_ca);
        const x_ca = x_sa.add(ca_out);
        // ffn (f32 this rung)
        const shift_mlp = adaVal(b.sst, ts3, 3);
        const scale_mlp = adaVal(b.sst, ts3, 4);
        const gate_mlp = adaVal(b.sst, ts3, 5);
        const ff_in = modulate(rmsNoW(x_ca), scale_mlp, shift_mlp);
        const ff_out = lin(lin(ff_in, b.ff1_w, null).gelu(), b.ff2_w, null);
        return x_ca.add(ff_out.mul(gate_mlp));
    }
};

