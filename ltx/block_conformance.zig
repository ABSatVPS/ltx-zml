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

const blk = @import("block.zig");
const T = blk.T;
const S = blk.S;
const D = blk.D;
const H = blk.H;
const HD = blk.HD;
const FF = blk.FF;
const Block = blk.Block;
const Chain = blk.Chain;
const QBlock = blk.QBlock;
const WEIGHT_SPECS = blk.WEIGHT_SPECS;
const weightShape = blk.weightShape;
const makeBlockSpecs = blk.makeBlockSpecs;
const NF: usize = 682; // dim/6 rope frequencies
const MAXPOS = [3]f64{ 20, 2048, 2048 };
const THETA: f64 = 10000.0;

const BUNDLE = "/home/adam/Development/Experiments/Video-Generation/.work/oracle_bundle";
const CHAIN_BUNDLE = "/home/adam/Development/Experiments/Video-Generation/.work/chain_bundle";
const WEIGHTS = "/home/adam/Development/Experiments/Video-Generation/.work/block0";
const WEIGHTS23 = "/home/adam/Development/Experiments/Video-Generation/.work/block23";
const WEIGHTS47 = "/home/adam/Development/Experiments/Video-Generation/.work/block47";

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

// ---- weight loading (harness paths) ---------------------------------------

fn loadBlockBufs(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, dir: []const u8, block_idx: i64) !zml.Bufferized(Block) {
    var bufs: zml.Bufferized(Block) = undefined;
    inline for (WEIGHT_SPECS) |spec| {
        var namebuf: [256]u8 = undefined;
        const fname = try std.fmt.bufPrint(&namebuf, "transformer_blocks_{d}_{s}", .{ block_idx, spec.file });
        const raw = try readBin(allocator, io, dir, fname);
        defer allocator.free(raw);
        @field(bufs, spec.field) = try zml.Buffer.fromBytes(io, platform, weightShape(spec), .replicated, raw);
    }
    return bufs;
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
    // The nested Chain struct (3 x 28 tensors) pushes comptime reflection
    // past the default 1000-branch quota.
    @setEvalBranchQuota(1_000_000);
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
    const model = makeBlockSpecs();
    const bufs = try loadBlockBufs(allocator, io, platform, WEIGHTS, 0);
    log.info("block 0: 28 weight tensors resident", .{});

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

    // ---- Phase 3 checkpoint 1: the 0/23/47 chain --------------------------
    const b23_bufs = try loadBlockBufs(allocator, io, platform, WEIGHTS23, 23);
    const b47_bufs = try loadBlockBufs(allocator, io, platform, WEIGHTS47, 47);
    log.info("blocks 23 and 47 resident", .{});
    const chain_model: Chain = .{ .b0 = makeBlockSpecs(), .b23 = makeBlockSpecs(), .b47 = makeBlockSpecs() };
    const cbufs: zml.Bufferized(Chain) = .{ .b0 = bufs, .b23 = b23_bufs, .b47 = b47_bufs };

    const chain_stages = [_]struct { method: []const u8, oracle: []const u8 }{
        .{ .method = "chainB23", .oracle = "chain_b23_out.bin" },
        .{ .method = "chainB47", .oracle = "chain_b47_out.bin" },
    };
    inline for (chain_stages) |st| {
        const method = comptime std.meta.stringToEnum(std.meta.DeclEnum(Chain), st.method).?;
        var exe = try platform.compile(allocator, io, chain_model, method, .{ x_spec, ts_spec, ctx_spec, pts_spec, cos_spec, sin_spec }, .{});
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ cbufs, x_buf, ts_buf, ctx_buf, pts_buf, cos_buf, sin_buf });
        exe.call(args, &results);
        var out: zml.Buffer = results.get(zml.Buffer);
        defer out.deinit();
        var slice = try out.toSliceAlloc(allocator, io);
        defer slice.free(allocator);
        @memcpy(got, slice.constItems(f32)[0..out_len]);
        const want = try loadF32(allocator, io, CHAIN_BUNDLE, st.oracle, out_len);
        defer allocator.free(want);
        if (!compare(st.method, got, want, 2e-3)) all_pass = false;
    }

    // ---- E3w swap gates (pre-registered 2026-08-21) -----------------------
    // The production attention (stablehlo.while online softmax, run 8b) has
    // never run inside the block. Overlap domain at T=64: dense is
    // torch-anchored, the f32 while twin differs ONLY in algorithm.
    // H-E3W-1 budget 1e-5 (expected ~1e-7 f32 reordering noise); H-E3W-2/3
    // keep the standard 2e-3 oracle gates.
    {
        const dense_attn = try allocator.alloc(f32, out_len);
        defer allocator.free(dense_attn);
        inline for (.{ "s3Attn1", "wAttn1" }, .{ dense_attn, got }) |mname, dst| {
            const method = comptime std.meta.stringToEnum(std.meta.DeclEnum(Block), mname).?;
            var exe = try platform.compile(allocator, io, model, method, .{ x_spec, ts_spec, cos_spec, sin_spec }, .{});
            var args = try exe.args(allocator);
            defer args.deinit(allocator);
            var results = try exe.results(allocator);
            defer results.deinit(allocator);
            args.set(.{ bufs, x_buf, ts_buf, cos_buf, sin_buf });
            exe.call(args, &results);
            var out: zml.Buffer = results.get(zml.Buffer);
            defer out.deinit();
            var slice = try out.toSliceAlloc(allocator, io);
            defer slice.free(allocator);
            @memcpy(dst, slice.constItems(f32)[0..out_len]);
        }
        if (!compare("e3w-attn1-agreement", got, dense_attn, 1e-5)) all_pass = false;
    }
    {
        const wmethod = comptime std.meta.stringToEnum(std.meta.DeclEnum(Block), "wBlockOut").?;
        var exe = try platform.compile(allocator, io, model, wmethod, .{ x_spec, ts_spec, ctx_spec, pts_spec, cos_spec, sin_spec }, .{});
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ bufs, x_buf, ts_buf, ctx_buf, pts_buf, cos_buf, sin_buf });
        exe.call(args, &results);
        var out: zml.Buffer = results.get(zml.Buffer);
        defer out.deinit();
        var slice = try out.toSliceAlloc(allocator, io);
        defer slice.free(allocator);
        @memcpy(got, slice.constItems(f32)[0..out_len]);
        const want = try loadF32(allocator, io, BUNDLE, "block_out.bin", out_len);
        defer allocator.free(want);
        if (!compare("e3w-blockOut-vs-oracle", got, want, 2e-3)) all_pass = false;
    }
    {
        const wmethod = comptime std.meta.stringToEnum(std.meta.DeclEnum(Chain), "wChainB47").?;
        var exe = try platform.compile(allocator, io, chain_model, wmethod, .{ x_spec, ts_spec, ctx_spec, pts_spec, cos_spec, sin_spec }, .{});
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ cbufs, x_buf, ts_buf, ctx_buf, pts_buf, cos_buf, sin_buf });
        exe.call(args, &results);
        var out: zml.Buffer = results.get(zml.Buffer);
        defer out.deinit();
        var slice = try out.toSliceAlloc(allocator, io);
        defer slice.free(allocator);
        @memcpy(got, slice.constItems(f32)[0..out_len]);
        const want = try loadF32(allocator, io, CHAIN_BUNDLE, "chain_b47_out.bin", out_len);
        defer allocator.free(want);
        if (!compare("e3w-chainB47-vs-oracle", got, want, 2e-3)) all_pass = false;
    }

    // ---- int4 rung 1: qkv class, budgets pre-registered -------------------
    // Baselines are OUR f32 graphs (torch-anchored above); budgets from the
    // notebook: logits 5e-2, attention output 2e-2, block 2e-2.
    const Q4Spec = struct { qf: []const u8, sf: []const u8, dims: [2]i64 };
    const q4_specs = [_]struct { field: []const u8, s: Q4Spec }{
        .{ .field = "q1q", .s = .{ .qf = "attn1_to_q_weight_q8.bin", .sf = "attn1_to_q_weight_q8scale.bin", .dims = .{ D, D } } },
        .{ .field = "k1q", .s = .{ .qf = "attn1_to_k_weight_q8.bin", .sf = "attn1_to_k_weight_q8scale.bin", .dims = .{ D, D } } },
        .{ .field = "v1q", .s = .{ .qf = "attn1_to_v_weight_q8.bin", .sf = "attn1_to_v_weight_q8scale.bin", .dims = .{ D, D } } },
        .{ .field = "q2q", .s = .{ .qf = "attn2_to_q_weight_q8.bin", .sf = "attn2_to_q_weight_q8scale.bin", .dims = .{ D, D } } },
        .{ .field = "k2q", .s = .{ .qf = "attn2_to_k_weight_q8.bin", .sf = "attn2_to_k_weight_q8scale.bin", .dims = .{ D, D } } },
        .{ .field = "v2q", .s = .{ .qf = "attn2_to_v_weight_q8.bin", .sf = "attn2_to_v_weight_q8scale.bin", .dims = .{ D, D } } },
        .{ .field = "f1q", .s = .{ .qf = "ff_net_0_proj_weight_q8.bin", .sf = "ff_net_0_proj_weight_q8scale.bin", .dims = .{ FF, D } } },
        .{ .field = "f2q", .s = .{ .qf = "ff_net_2_weight_q8.bin", .sf = "ff_net_2_weight_q8scale.bin", .dims = .{ D, FF } } },
    };
    var qmodel: QBlock = undefined;
    var qbufs: zml.Bufferized(QBlock) = undefined;
    qmodel.base = makeBlockSpecs();
    qbufs.base = bufs;
    inline for (q4_specs) |qs| {
        var namebuf: [256]u8 = undefined;
        const qname = try std.fmt.bufPrint(&namebuf, "transformer_blocks_0_{s}", .{qs.s.qf});
        const qraw = try readBin(allocator, io, WEIGHTS, qname);
        defer allocator.free(qraw);
        const qshape = zml.Shape.init(.{ .o = qs.s.dims[0], .i = qs.s.dims[1] }, .i8);
        @field(qbufs, qs.field) = try zml.Buffer.fromBytes(io, platform, qshape, .replicated, qraw);
        @field(qmodel, qs.field) = zml.Tensor.fromShape(qshape);
        var namebuf2: [256]u8 = undefined;
        const sname = try std.fmt.bufPrint(&namebuf2, "transformer_blocks_0_{s}", .{qs.s.sf});
        const sraw = try readBin(allocator, io, WEIGHTS, sname);
        defer allocator.free(sraw);
        const sshape = zml.Shape.init(.{ .o = qs.s.dims[0], .g = @divExact(qs.s.dims[1], 128) }, .f32);
        const sfield = qs.field[0..2] ++ "s"; // q1q -> q1s etc.
        @field(qbufs, sfield) = try zml.Buffer.fromBytes(io, platform, sshape, .replicated, sraw);
        @field(qmodel, sfield) = zml.Tensor.fromShape(sshape);
    }
    log.info("int4 qkv: 6 quantized pairs resident", .{});

    const logits_len = @as(usize, @intCast(H * T * T));
    const lgot = try allocator.alloc(f32, logits_len);
    defer allocator.free(lgot);
    const lref = try allocator.alloc(f32, logits_len);
    defer allocator.free(lref);

    const QProbe = struct { fm: []const u8, qm: []const u8, budget: f64, len: usize, full: bool };
    const probes = [_]QProbe{
        .{ .fm = "fLogits", .qm = "qLogits", .budget = 5e-2, .len = logits_len, .full = false },
        .{ .fm = "s3Attn1", .qm = "qAttn1", .budget = 2e-2, .len = out_len, .full = false },
        .{ .fm = "blockOut", .qm = "qBlockOut", .budget = 2e-2, .len = out_len, .full = true },
        .{ .fm = "s7FfOut", .qm = "qFfOut", .budget = 5e-2, .len = out_len, .full = true },
        .{ .fm = "blockOut", .qm = "qBlockOutAll", .budget = 2e-2, .len = out_len, .full = true },
    };
    inline for (probes) |p| {
        const fmethod = comptime std.meta.stringToEnum(std.meta.DeclEnum(Block), p.fm).?;
        const qmethod = comptime std.meta.stringToEnum(std.meta.DeclEnum(QBlock), p.qm).?;
        const ref_buf = if (p.len == logits_len) lref else got;
        const q_buf_out = if (p.len == logits_len) lgot else got;
        // f32 baseline
        {
            var exe = if (p.full)
                try platform.compile(allocator, io, model, fmethod, .{ x_spec, ts_spec, ctx_spec, pts_spec, cos_spec, sin_spec }, .{})
            else
                try platform.compile(allocator, io, model, fmethod, .{ x_spec, ts_spec, cos_spec, sin_spec }, .{});
            var args = try exe.args(allocator);
            defer args.deinit(allocator);
            var results = try exe.results(allocator);
            defer results.deinit(allocator);
            if (p.full) args.set(.{ bufs, x_buf, ts_buf, ctx_buf, pts_buf, cos_buf, sin_buf }) else args.set(.{ bufs, x_buf, ts_buf, cos_buf, sin_buf });
            exe.call(args, &results);
            var out: zml.Buffer = results.get(zml.Buffer);
            defer out.deinit();
            var slice = try out.toSliceAlloc(allocator, io);
            defer slice.free(allocator);
            @memcpy(ref_buf[0..p.len], slice.constItems(f32)[0..p.len]);
        }
        // int4 variant — note: baseline for probe 2/3 briefly lives in `got`
        // before the int4 run overwrites it, so copy into place first.
        const ref_copy = try allocator.alloc(f32, p.len);
        defer allocator.free(ref_copy);
        @memcpy(ref_copy, ref_buf[0..p.len]);
        {
            var exe = if (p.full)
                try platform.compile(allocator, io, qmodel, qmethod, .{ x_spec, ts_spec, ctx_spec, pts_spec, cos_spec, sin_spec }, .{})
            else
                try platform.compile(allocator, io, qmodel, qmethod, .{ x_spec, ts_spec, cos_spec, sin_spec }, .{});
            var args = try exe.args(allocator);
            defer args.deinit(allocator);
            var results = try exe.results(allocator);
            defer results.deinit(allocator);
            if (p.full) args.set(.{ qbufs, x_buf, ts_buf, ctx_buf, pts_buf, cos_buf, sin_buf }) else args.set(.{ qbufs, x_buf, ts_buf, cos_buf, sin_buf });
            exe.call(args, &results);
            var out: zml.Buffer = results.get(zml.Buffer);
            defer out.deinit();
            var slice = try out.toSliceAlloc(allocator, io);
            defer slice.free(allocator);
            @memcpy(q_buf_out[0..p.len], slice.constItems(f32)[0..p.len]);
        }
        if (!compare("quant-" ++ p.qm, q_buf_out[0..p.len], ref_copy, p.budget)) all_pass = false;
    }

    if (all_pass) {
        log.info("BLOCK CONFORMANCE: ALL GATES PASS (single block + chain + E3w swap + fully quantized block)", .{});
    } else {
        log.err("BLOCK CONFORMANCE: FAILURES ABOVE", .{});
        return error.ConformanceFailed;
    }
}
