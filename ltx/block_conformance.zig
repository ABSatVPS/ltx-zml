//! Phase 2 conformance: one LTX-2.5 video transformer block (block 0, real
//! weights) against the torch/ltx-core oracle bundle, in the pre-registered
//! composition order (docs/lab-notebook.md, Phase 2 spec + amendments).
//!
//! Gate 0 (host, no GPU): RoPE tables generated here in f64 following the
//! reference numpy semantics, cast once to f32, compared BIT-EXACTLY against
//! the bundle's serialized tables. Stragglers (cross-libm trig ulps) are
//! reported as structured records and must stay within 1 f32 ulp and 0.01%
//! count; any straggler is grounds for review and coordinate-pinning.
//!
//! Gates 1..N (GPU): stage graphs traced in f32 (bf16 weights upcast
//! in-graph — exact) against the f64-computed oracle stages, rel-RMS gate
//! 2e-3 with ~1e-5 expected. The bf16 deployment-dtype block is Phase 3;
//! here f32 isolates SPEC errors from dtype noise.
const std = @import("std");
const zml = @import("zml");

const log = std.log;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const T: i64 = 64; // video tokens in the bundle
const S: i64 = 32; // context (prompt) tokens
const D: i64 = 4096;
const H: i64 = 32;
const HD: i64 = 128;
const FF: i64 = 16384;
const NF: usize = 682; // dim/6 rope frequencies
const EPS: f32 = 1e-6;
const MAXPOS = [3]f64{ 20, 2048, 2048 };
const THETA: f64 = 10000.0;

const BUNDLE = "/home/adam/Development/Experiments/Video-Generation/.work/oracle_bundle";
const WEIGHTS = "/home/adam/Development/Experiments/Video-Generation/.work/block0";

// ---- host I/O -------------------------------------------------------------

fn readBin(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u8 {
    var buf: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir, name });
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
}

fn loadF32(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, expect: usize) ![]f32 {
    const raw = try readBin(allocator, io, dir, name);
    defer allocator.free(raw);
    if (raw.len != expect * 4) {
        log.err("{s}: {d} bytes, expected {d}", .{ name, raw.len, expect * 4 });
        return error.BadSize;
    }
    const out = try allocator.alloc(f32, expect);
    @memcpy(std.mem.sliceAsBytes(out), raw);
    return out;
}

// ---- Gate 0: RoPE tables, host f64, bit-compared --------------------------

fn ulpDeltaF32(a: u32, b: u32) i64 {
    // Monotonic int mapping of f32 bit patterns (IEEE trick).
    const ia: i64 = if (a & 0x8000_0000 != 0) -@as(i64, a & 0x7fff_ffff) else @as(i64, a);
    const ib: i64 = if (b & 0x8000_0000 != 0) -@as(i64, b & 0x7fff_ffff) else @as(i64, b);
    return ia - ib;
}

fn ropeGate(allocator: std.mem.Allocator, io: std.Io) !bool {
    const Tu: usize = @intCast(T);
    const Hu: usize = @intCast(H);

    const grid = try loadF32(allocator, io, BUNDLE, "in_grid.bin", 3 * Tu * 2);
    defer allocator.free(grid);
    const ref_cos = try loadF32(allocator, io, BUNDLE, "rope_cos.bin", Hu * Tu * 64);
    defer allocator.free(ref_cos);
    const ref_sin = try loadF32(allocator, io, BUNDLE, "rope_sin.bin", Hu * Tu * 64);
    defer allocator.free(ref_sin);

    // Frequency grid: theta^linspace(0, 1, NF) * pi/2 computed in f64 —
    // then ROUNDED TO F32, because the reference's generate_freq_grid_np
    // returns dtype=torch.float32: the "f64 rope path" is f64 only INSIDE
    // the grid computation. Discovered by gate forensics in run 2 (ref
    // slot 2 held cos(f32(pi/2)), not cos(f64 pi/2)). The rounded values
    // then promote back to f64 for the position products and trig.
    var indices: [NF]f64 = undefined;
    for (&indices, 0..) |*v, k| {
        const x = @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(NF - 1));
        const f64_val = std.math.pow(f64, THETA, x) * (std.math.pi / 2.0);
        const f32_val: f32 = @floatCast(f64_val);
        v.* = @as(f64, f32_val);
    }

    // Tables in the bundle layout (B=1, H, T, 64): slot j = h*64 + f of the
    // 2048 per-token slots; j<2 are the PREPENDED identity pad; else the
    // freq-major (k, axis) pair with s = j-2, k = s/3, axis = s%3.
    var n_diff: usize = 0;
    var worst: i64 = 0;
    var pass = true;
    for (0..Hu) |h| {
        for (0..Tu) |t| {
            // fractional positions from the (start,end) grid, midpoint rule
            var arg: [3]f64 = undefined;
            for (0..3) |a| {
                const s0 = grid[(a * Tu + t) * 2];
                const s1 = grid[(a * Tu + t) * 2 + 1];
                const mid = (@as(f64, s0) + @as(f64, s1)) / 2.0;
                arg[a] = (mid / MAXPOS[a]) * 2.0 - 1.0;
            }
            for (0..64) |f| {
                const j = h * 64 + f;
                var c64: f64 = 1.0;
                var s64: f64 = 0.0;
                if (j >= 2) {
                    const s = j - 2;
                    const freq = indices[s / 3] * arg[s % 3];
                    c64 = @cos(freq);
                    s64 = @sin(freq);
                }
                const idx = (h * Tu + t) * 64 + f;
                inline for (.{ .{ c64, ref_cos, "cos" }, .{ s64, ref_sin, "sin" } }) |case| {
                    const ours: f32 = @floatCast(case[0]);
                    const theirs: f32 = case[1][idx];
                    const ob: u32 = @bitCast(ours);
                    const tb: u32 = @bitCast(theirs);
                    if (ob != tb) {
                        n_diff += 1;
                        const d = ulpDeltaF32(ob, tb);
                        if (@abs(d) > @abs(worst)) worst = d;
                        if (n_diff <= 8) {
                            log.warn("rope straggler: table={s} head={d} tok={d} slot={d} ref=0x{x:0>8} ours=0x{x:0>8} ulp={d}", .{
                                case[2], h, t, f, tb, ob, d,
                            });
                        }
                        if (@abs(d) > 1) pass = false;
                    }
                }
            }
        }
    }
    const total = Hu * Tu * 64 * 2;
    const limit = total / 10000; // 0.01%
    if (n_diff > limit) pass = false;
    log.info("GATE rope: {d}/{d} straggler bits, worst {d} ulp -> {s}", .{
        n_diff, total, worst, if (pass) "PASS" else "FAIL",
    });

    // Identity-slot probe: head 0 slots 0,1 must be exactly (1, 0).
    var ident_ok = true;
    for (0..Tu) |t| {
        for (0..2) |f| {
            const idx = t * 64 + f; // h=0
            if (ref_cos[idx] != 1.0 or ref_sin[idx] != 0.0) ident_ok = false;
        }
    }
    log.info("GATE rope identity slots: {s}", .{if (ident_ok) "PASS" else "FAIL"});
    return pass and ident_ok;
}

// ---- the block as a traced model ------------------------------------------

fn lin(x: zml.Tensor, w: zml.Tensor, b: ?zml.Tensor) zml.Tensor {
    var y = x.dot(w.convert(.f32), .i);
    if (b) |bias| y = y.add(bias.convert(.f32).broad(y.shape()));
    return y.withTags(.{ .t, .i });
}

fn rmsNoW(x: zml.Tensor) zml.Tensor {
    return zml.nn.rmsNorm(x, .i, EPS);
}

fn rmsW(x: zml.Tensor, w: zml.Tensor) zml.Tensor {
    return rmsNoW(x).mul(w.convert(.f32).withTags(.{.i}).broad(x.shape()));
}

fn modulate(n: zml.Tensor, scale: zml.Tensor, shift: zml.Tensor) zml.Tensor {
    return n.mul(scale.addConstant(1.0)).add(shift);
}

/// ada value i: per-block table row + per-token timestep chunk. ts3 is
/// [.t, .n=9, .i], table [.n, .i].
fn adaVal(table: zml.Tensor, ts3: zml.Tensor, i: i64) zml.Tensor {
    const row = table.convert(.f32).slice1d(.n, .{ .start = i, .end = i + 1 }).squeeze(.n); // [.i]
    const tv = ts3.slice1d(.n, .{ .start = i, .end = i + 1 }).squeeze(.n); // [.t,.i]
    return tv.add(row.broad(tv.shape()));
}

/// split-half RoPE: x4 [.., .h, .p=2, .f=64] (leading tag .q or .k), cos/sin
/// tagged to match x4's leading tag: [lead, .h, .f].
fn rope(x4: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
    const a = x4.slice1d(.p, .{ .start = 0, .end = 1 }).squeeze(.p);
    const b = x4.slice1d(.p, .{ .start = 1, .end = 2 }).squeeze(.p);
    const ra = a.mul(cos).sub(b.mul(sin));
    const rb = b.mul(cos).add(a.mul(sin));
    return zml.Tensor.concatenate(&.{ ra, rb }, .f);
}

const Block = struct {
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
    fn attention(w: A, x: zml.Tensor, ctx: zml.Tensor, n_kv: i64, pe: ?struct { cos: zml.Tensor, sin: zml.Tensor }, comptime gated: bool) zml.Tensor {
        var q = rmsW(lin(x, w.qw, w.qb), w.qn);
        var k = rmsW(lin(ctx, w.kw, w.kb), w.kn);
        const v = lin(ctx, w.vw, w.vb);

        var q3: zml.Tensor = undefined;
        var k3: zml.Tensor = undefined;
        if (pe) |p| {
            const q4 = q.reshape(zml.Shape.init(.{ .q = T, .h = H, .p = 2, .f = 64 }, .f32));
            const k4 = k.reshape(zml.Shape.init(.{ .k = n_kv, .h = H, .p = 2, .f = 64 }, .f32));
            const cos_k = p.cos.withTags(.{ .k, .h, .f });
            const sin_k = p.sin.withTags(.{ .k, .h, .f });
            q3 = rope(q4, p.cos, p.sin).withTags(.{ .q, .h, .hd });
            k3 = rope(k4, cos_k, sin_k).withTags(.{ .k, .h, .hd });
        } else {
            q3 = q.reshape(zml.Shape.init(.{ .q = T, .h = H, .hd = HD }, .f32));
            k3 = k.reshape(zml.Shape.init(.{ .k = n_kv, .h = H, .hd = HD }, .f32));
        }
        const v3 = v.reshape(zml.Shape.init(.{ .k = n_kv, .h = H, .hd = HD }, .f32));

        var out = zml.nn.sdpa(q3, k3, v3, .{}); // [.h,.q,.hd]-tagged
        if (gated) {
            const glog = lin(x, w.gw, w.gb).withTags(.{ .q, .h }); // [T, 32]
            const gate = glog.sigmoid().scale(2.0);
            out = out.mul(gate.broad(out.shape()));
        }
        const merged = out.transpose(.{ .q, .h, .hd }).reshape(zml.Shape.init(.{ .t = T, .i = D }, .f32));
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
        return attention(self.attn1w(), n, n, T, .{ .cos = cos, .sin = sin }, false);
    }

    pub fn s3Attn1(self: @This(), x: zml.Tensor, ts3: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const n = self.s1NormMsa(x, ts3);
        return attention(self.attn1w(), n, n, T, .{ .cos = cos, .sin = sin }, true);
    }

    fn afterSa(self: @This(), x: zml.Tensor, ts3: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const gate = adaVal(self.sst, ts3, 2);
        return x.add(self.s3Attn1(x, ts3, cos, sin).mul(gate));
    }

    pub fn s4AfterSa(self: @This(), x: zml.Tensor, ts3: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        return self.afterSa(x, ts3, cos, sin);
    }

    pub fn s4Normed(self: @This(), x: zml.Tensor, ts3: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        return rmsNoW(self.afterSa(x, ts3, cos, sin));
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
        const ca = attention(self.attn2w(), attn_in, enc, S, null, true);
        return ca.mul(gate_ca);
    }

    pub fn s5CaOut(self: @This(), x: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        return self.caOut(self.s4Normed(x, ts3, cos, sin), ts3, ctx, pts2);
    }

    fn afterCa(self: @This(), x: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        return self.afterSa(x, ts3, cos, sin).add(self.s5CaOut(x, ts3, ctx, pts2, cos, sin));
    }

    pub fn s6FfIn(self: @This(), x: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const shift = adaVal(self.sst, ts3, 3);
        const scl = adaVal(self.sst, ts3, 4);
        return modulate(rmsNoW(self.afterCa(x, ts3, ctx, pts2, cos, sin)), scl, shift);
    }

    pub fn s7FfOut(self: @This(), x: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const ff_in = self.s6FfIn(x, ts3, ctx, pts2, cos, sin);
        const h = lin(ff_in, self.ff1_w, null).gelu();
        return lin(h, self.ff2_w, null);
    }

    pub fn blockOut(self: @This(), x: zml.Tensor, ts3: zml.Tensor, ctx: zml.Tensor, pts2: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const gate = adaVal(self.sst, ts3, 5);
        return self.afterCa(x, ts3, ctx, pts2, cos, sin)
            .add(self.s7FfOut(x, ts3, ctx, pts2, cos, sin).mul(gate));
    }
};

// ---- weight loading --------------------------------------------------------

const WSpec = struct {
    field: []const u8,
    file: []const u8,
    dims: []const i64,
    f32_: bool = false,
};

const WEIGHT_SPECS = [_]WSpec{
    .{ .field = "q1_w", .file = "transformer_blocks_0_attn1_to_q_weight.bin", .dims = &.{ D, D } },
    .{ .field = "q1_b", .file = "transformer_blocks_0_attn1_to_q_bias.bin", .dims = &.{D} },
    .{ .field = "k1_w", .file = "transformer_blocks_0_attn1_to_k_weight.bin", .dims = &.{ D, D } },
    .{ .field = "k1_b", .file = "transformer_blocks_0_attn1_to_k_bias.bin", .dims = &.{D} },
    .{ .field = "v1_w", .file = "transformer_blocks_0_attn1_to_v_weight.bin", .dims = &.{ D, D } },
    .{ .field = "v1_b", .file = "transformer_blocks_0_attn1_to_v_bias.bin", .dims = &.{D} },
    .{ .field = "o1_w", .file = "transformer_blocks_0_attn1_to_out_0_weight.bin", .dims = &.{ D, D } },
    .{ .field = "o1_b", .file = "transformer_blocks_0_attn1_to_out_0_bias.bin", .dims = &.{D} },
    .{ .field = "qn1", .file = "transformer_blocks_0_attn1_q_norm_weight.bin", .dims = &.{D} },
    .{ .field = "kn1", .file = "transformer_blocks_0_attn1_k_norm_weight.bin", .dims = &.{D} },
    .{ .field = "g1_w", .file = "transformer_blocks_0_attn1_to_gate_logits_weight.bin", .dims = &.{ H, D } },
    .{ .field = "g1_b", .file = "transformer_blocks_0_attn1_to_gate_logits_bias.bin", .dims = &.{H} },
    .{ .field = "q2_w", .file = "transformer_blocks_0_attn2_to_q_weight.bin", .dims = &.{ D, D } },
    .{ .field = "q2_b", .file = "transformer_blocks_0_attn2_to_q_bias.bin", .dims = &.{D} },
    .{ .field = "k2_w", .file = "transformer_blocks_0_attn2_to_k_weight.bin", .dims = &.{ D, D } },
    .{ .field = "k2_b", .file = "transformer_blocks_0_attn2_to_k_bias.bin", .dims = &.{D} },
    .{ .field = "v2_w", .file = "transformer_blocks_0_attn2_to_v_weight.bin", .dims = &.{ D, D } },
    .{ .field = "v2_b", .file = "transformer_blocks_0_attn2_to_v_bias.bin", .dims = &.{D} },
    .{ .field = "o2_w", .file = "transformer_blocks_0_attn2_to_out_0_weight.bin", .dims = &.{ D, D } },
    .{ .field = "o2_b", .file = "transformer_blocks_0_attn2_to_out_0_bias.bin", .dims = &.{D} },
    .{ .field = "qn2", .file = "transformer_blocks_0_attn2_q_norm_weight.bin", .dims = &.{D} },
    .{ .field = "kn2", .file = "transformer_blocks_0_attn2_k_norm_weight.bin", .dims = &.{D} },
    .{ .field = "g2_w", .file = "transformer_blocks_0_attn2_to_gate_logits_weight.bin", .dims = &.{ H, D } },
    .{ .field = "g2_b", .file = "transformer_blocks_0_attn2_to_gate_logits_bias.bin", .dims = &.{H} },
    .{ .field = "ff1_w", .file = "transformer_blocks_0_ff_net_0_proj_weight.bin", .dims = &.{ FF, D } },
    .{ .field = "ff2_w", .file = "transformer_blocks_0_ff_net_2_weight.bin", .dims = &.{ D, FF } },
    .{ .field = "sst", .file = "transformer_blocks_0_scale_shift_table.bin", .dims = &.{ 9, D }, .f32_ = true },
    .{ .field = "psst", .file = "transformer_blocks_0_prompt_scale_shift_table.bin", .dims = &.{ 2, D }, .f32_ = true },
};

fn weightShape(spec: WSpec) zml.Shape {
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

// ---- comparison ------------------------------------------------------------

fn compare(name: []const u8, got: []const f32, want: []const f32, limit: f64) bool {
    var err: f64 = 0;
    var ref: f64 = 0;
    var maxd: f64 = 0;
    var arg: usize = 0;
    var nan: usize = 0;
    for (got, want, 0..) |g, w, i| {
        if (std.math.isNan(g) or std.math.isInf(g)) nan += 1;
        const d = @abs(@as(f64, g) - @as(f64, w));
        if (d > maxd) {
            maxd = d;
            arg = i;
        }
        err += d * d;
        ref += @as(f64, w) * w;
    }
    const rms = @sqrt(err / (ref + 1e-20));
    const pass = rms <= limit and nan == 0;
    const t = arg / @as(usize, @intCast(D));
    const d_idx = arg % @as(usize, @intCast(D));
    log.info("GATE {s}: rel-RMS {e:.3} max-abs-diff {e:.3} nan/inf {d} -> {s}", .{
        name, rms, maxd, nan, if (pass) "PASS" else "FAIL",
    });
    if (!pass) {
        log.err("  worst at token={d} chan={d} (head={d}): ref={d:.6} ours={d:.6} | neighborhood ref[{d:.5},{d:.5},{d:.5}] ours[{d:.5},{d:.5},{d:.5}]", .{
            t,                      d_idx,                  d_idx / @as(usize, @intCast(HD)),
            want[arg],              got[arg],               want[arg -| 1],
            want[arg],              want[@min(arg + 1, want.len - 1)], got[arg -| 1],
            got[arg],               got[@min(arg + 1, got.len - 1)],
        });
    }
    return pass;
}

// ---- main ------------------------------------------------------------------

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    // Gate 0 needs no GPU.
    const rope_ok = try ropeGate(allocator, io);
    if (!rope_ok) {
        log.err("RoPE gate failed — stopping before GPU stages (nothing downstream is diagnosable)", .{});
        return error.RopeGateFailed;
    }

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    log.info("platform: {s}", .{@tagName(platform.target)});

    const Tu: usize = @intCast(T);
    const Su: usize = @intCast(S);
    const Du: usize = @intCast(D);
    const Hu: usize = @intCast(H);

    // ---- weights to device ----
    var model: Block = undefined;
    var bufs: zml.Bufferized(Block) = undefined;
    inline for (WEIGHT_SPECS) |spec| {
        const raw = try readBin(allocator, io, WEIGHTS, spec.file);
        defer allocator.free(raw);
        const shape = weightShape(spec);
        const buf: zml.Buffer = try .fromBytes(io, platform, shape, .replicated, raw);
        @field(bufs, spec.field) = buf;
        @field(model, spec.field) = zml.Tensor.fromShape(shape);
    }
    log.info("28 weight tensors resident", .{});

    // ---- bundle inputs ----
    const x_h = try loadF32(allocator, io, BUNDLE, "in_x.bin", Tu * Du);
    defer allocator.free(x_h);
    const ctx_h = try loadF32(allocator, io, BUNDLE, "in_context.bin", Su * Du);
    defer allocator.free(ctx_h);
    const ts_h = try loadF32(allocator, io, BUNDLE, "in_timestep.bin", Tu * 9 * Du);
    defer allocator.free(ts_h);
    const pts_h = try loadF32(allocator, io, BUNDLE, "in_prompt_timestep.bin", 2 * Du);
    defer allocator.free(pts_h);
    // rope tables (B,H,T,64) -> host rearrange to (T,H,64) for graph layout
    const cos_bhtf = try loadF32(allocator, io, BUNDLE, "rope_cos.bin", Hu * Tu * 64);
    defer allocator.free(cos_bhtf);
    const sin_bhtf = try loadF32(allocator, io, BUNDLE, "rope_sin.bin", Hu * Tu * 64);
    defer allocator.free(sin_bhtf);
    const cos_thf = try allocator.alloc(f32, Tu * Hu * 64);
    defer allocator.free(cos_thf);
    const sin_thf = try allocator.alloc(f32, Tu * Hu * 64);
    defer allocator.free(sin_thf);
    for (0..Hu) |h| for (0..Tu) |t| for (0..64) |f| {
        cos_thf[(t * Hu + h) * 64 + f] = cos_bhtf[(h * Tu + t) * 64 + f];
        sin_thf[(t * Hu + h) * 64 + f] = sin_bhtf[(h * Tu + t) * 64 + f];
    };

    const x_shape = zml.Shape.init(.{ .t = T, .i = D }, .f32);
    const ctx_shape = zml.Shape.init(.{ .t = S, .i = D }, .f32);
    const ts_shape = zml.Shape.init(.{ .t = T, .n = 9, .i = D }, .f32);
    const pts_shape = zml.Shape.init(.{ .n = 2, .i = D }, .f32);
    const pe_shape = zml.Shape.init(.{ .q = T, .h = H, .f = 64 }, .f32);

    var x_buf: zml.Buffer = try .fromBytes(io, platform, x_shape, .replicated, std.mem.sliceAsBytes(x_h));
    defer x_buf.deinit();
    var ctx_buf: zml.Buffer = try .fromBytes(io, platform, ctx_shape, .replicated, std.mem.sliceAsBytes(ctx_h));
    defer ctx_buf.deinit();
    var ts_buf: zml.Buffer = try .fromBytes(io, platform, ts_shape, .replicated, std.mem.sliceAsBytes(ts_h));
    defer ts_buf.deinit();
    var pts_buf: zml.Buffer = try .fromBytes(io, platform, pts_shape, .replicated, std.mem.sliceAsBytes(pts_h));
    defer pts_buf.deinit();
    var cos_buf: zml.Buffer = try .fromBytes(io, platform, pe_shape, .replicated, std.mem.sliceAsBytes(cos_thf));
    defer cos_buf.deinit();
    var sin_buf: zml.Buffer = try .fromBytes(io, platform, pe_shape, .replicated, std.mem.sliceAsBytes(sin_thf));
    defer sin_buf.deinit();

    const x_spec: zml.Tensor = .fromShape(x_shape);
    const ctx_spec: zml.Tensor = .fromShape(ctx_shape);
    const ts_spec: zml.Tensor = .fromShape(ts_shape);
    const pts_spec: zml.Tensor = .fromShape(pts_shape);
    const cos_spec: zml.Tensor = .fromShape(pe_shape);
    const sin_spec: zml.Tensor = .fromShape(pe_shape);

    const out_len = Tu * Du;
    const got = try allocator.alloc(f32, out_len);
    defer allocator.free(got);

    var all_pass = true;
    const Stage = struct {
        method: []const u8,
        oracle: []const u8,
        args: enum { xt, xtcs, full },
    };
    const stages = [_]Stage{
        .{ .method = "s1NormMsa", .oracle = "s1_norm_msa.bin", .args = .xt },
        .{ .method = "s1bQnorm", .oracle = "s1b_qnorm.bin", .args = .xt },
        .{ .method = "s2Attn1NoGate", .oracle = "s2_attn1_nogate.bin", .args = .xtcs },
        .{ .method = "s3Attn1", .oracle = "s3_attn1.bin", .args = .xtcs },
        .{ .method = "s4AfterSa", .oracle = "s4_after_sa.bin", .args = .xtcs },
        .{ .method = "s4Normed", .oracle = "s4_normed.bin", .args = .xtcs },
        .{ .method = "s5CaOut", .oracle = "s5_ca_out.bin", .args = .full },
        .{ .method = "s6FfIn", .oracle = "s6_ff_in.bin", .args = .full },
        .{ .method = "s7FfOut", .oracle = "s7_ff_out.bin", .args = .full },
        .{ .method = "blockOut", .oracle = "block_out.bin", .args = .full },
    };

    inline for (stages) |st| {
        const method = comptime std.meta.stringToEnum(std.meta.DeclEnum(Block), st.method).?;
        var exe = switch (st.args) {
            .xt => try platform.compile(allocator, io, model, method, .{ x_spec, ts_spec }, .{}),
            .xtcs => try platform.compile(allocator, io, model, method, .{ x_spec, ts_spec, cos_spec, sin_spec }, .{}),
            .full => try platform.compile(allocator, io, model, method, .{ x_spec, ts_spec, ctx_spec, pts_spec, cos_spec, sin_spec }, .{}),
        };
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        switch (st.args) {
            .xt => args.set(.{ bufs, x_buf, ts_buf }),
            .xtcs => args.set(.{ bufs, x_buf, ts_buf, cos_buf, sin_buf }),
            .full => args.set(.{ bufs, x_buf, ts_buf, ctx_buf, pts_buf, cos_buf, sin_buf }),
        }
        exe.call(args, &results);
        var out: zml.Buffer = results.get(zml.Buffer);
        defer out.deinit();
        var slice = try out.toSliceAlloc(allocator, io);
        defer slice.free(allocator);
        @memcpy(got, slice.constItems(f32)[0..out_len]);

        const want = try loadF32(allocator, io, BUNDLE, st.oracle, out_len);
        defer allocator.free(want);
        if (!compare(st.method, got, want, 2e-3)) all_pass = false;
    }

    if (all_pass) {
        log.info("BLOCK CONFORMANCE: ALL GATES PASS", .{});
    } else {
        log.err("BLOCK CONFORMANCE: FAILURES ABOVE", .{});
        return error.ConformanceFailed;
    }
}
