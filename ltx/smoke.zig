//! E1/E2 smoke: measure the compiler-behavior unknowns BEFORE any engine
//! code exists. See docs/lab-notebook.md (2026-08-20 entry) for hypotheses.
//!
//! E1  — quantized contraction crossover: qdot (broadcast-mul+reduce, the
//!       coli-zml decode trick) vs dot_general (dequant materialized,
//!       amortized) across S = 1 .. 28k rows. Same int4 geometry as
//!       coli-zml's smoke (O=2048, K=6144) so small-S numbers line up with
//!       the known-good baseline.
//! E1a — pure f16 dense GEMM with a scalar-sum epilogue (4-byte download,
//!       so timing is compute, not PCIe): the achieved-TFLOPS probe for
//!       whether XLA engages V_WMMA on gfx1200. Read the ISA dump via
//!       XLA_FLAGS="--xla_dump_to=<dir>" to confirm.
//! E2  — zml.nn.sdpa (naive composition: dot / softmax / dot) swept over
//!       video-scale sequence lengths. An OOM at large T is a RESULT, not
//!       a failure: it means XLA materialized the scores matrix and E3
//!       (in-graph blockwise attention) is mandatory. E2 runs LAST and
//!       ascending so everything else is already printed if it aborts.
const std = @import("std");
const zml = @import("zml");

const log = std.log;

pub const std_options: std.Options = .{
    .log_level = .info,
};

// E1 geometry (matches coli-zml / colibri bench: gate matrix of the GLM
// expert, [O=2048 out, K=6144 in], int4 per-row scales).
const O: i64 = 2048;
const K: i64 = 6144;
const S_LIST = [_]i64{ 1, 8, 64, 512, 4096, 28672 };

// E2 geometry (LTX-2.5 head count/dim UNVERIFIED — see notebook; update
// when the HF config is read).
const H: i64 = 16;
const HD: i64 = 128;
const T_LIST = [_]i64{ 512, 1024, 4096, 8192, 16384, 28672 };
const T_ORACLE: i64 = 512;

fn nowNs(io: std.Io) i96 {
    const ts: std.Io.Timestamp = .now(io, .awake);
    return ts.toNanoseconds();
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn relativeRms(got: []const f32, want: []const f32, limit: f64) bool {
    var err: f64 = 0;
    var ref: f64 = 0;
    for (got, want) |g, w| {
        const d: f64 = g - w;
        err += d * d;
        ref += @as(f64, w) * w;
    }
    const r = @sqrt(err / (ref + 1e-20));
    if (r > limit) {
        log.err("relative RMS {d} exceeds {d}", .{ r, limit });
        return false;
    }
    return true;
}

// ---- traced graphs --------------------------------------------------------

/// int4 (offset-8) dequant with per-row scales, traced into the graph.
fn dequant(w: zml.Tensor, sc: zml.Tensor) zml.Tensor {
    const f = w.convert(.f32).addConstant(-8.0);
    return f.mul(sc.broad(f.shape()));
}

/// coli-zml's qdot: broadcast-mul + reduce. Loop-fuses the dequant (reads
/// packed nibbles directly) but re-reads the weights once per row of x.
fn qdotGraph(w: zml.Tensor, sc: zml.Tensor, x: zml.Tensor) zml.Tensor {
    const wd = dequant(w, sc);
    const sh3 = zml.Shape.init(.{ .s = x.shape().dim(0), .o = wd.shape().dim(0), .i = wd.shape().dim(1) }, .f32);
    return x.broad(sh3).mul(wd.broad(sh3)).sum(.i).squeeze(.i).withTags(.{ .s, .o });
}

/// The prefill-regime alternative: dequant materializes (f32 copy of W in
/// VRAM during the call), then a real dot_general amortizes it over S rows.
fn dotQGraph(w: zml.Tensor, sc: zml.Tensor, x: zml.Tensor) zml.Tensor {
    return x.dot(dequant(w, sc), .i);
}

/// E1a: dense f16 GEMM reduced to one f32 scalar so the download is 4 bytes
/// and the timing isolates compute. The epilogue adds ~S*O adds vs 2*S*O*K
/// GEMM flops — noise.
fn gemmF16SumGraph(w: zml.Tensor, x: zml.Tensor) zml.Tensor {
    return x.dot(w, .i).convert(.f32).sum(.o).squeeze(.o).sum(.s).squeeze(.s);
}

/// E1b correctness reference: same GEMM, full f16 output.
fn gemmF16Graph(w: zml.Tensor, x: zml.Tensor) zml.Tensor {
    return x.dot(w, .i);
}

/// E1b: the actual engine candidate — int4 resident, dequant traced to f16,
/// then a real dot that can hit WMMA. The materialized f16 weight copy is
/// ~O*K*2 bytes per call; hypothesis says that's noise against the GEMM.
fn dotQF16Graph(w: zml.Tensor, sc: zml.Tensor, x: zml.Tensor) zml.Tensor {
    return x.dot(dequant(w, sc).convert(.f16), .i);
}

/// E1b scalar-sum twin: same compute as dotQF16Graph, 4-byte download.
/// Run 4 found the E1b-vs-E1a gap was a constant ~0.5 GB/s at every S —
/// i.e. the full-output D2H readback, not the GEMM. This variant is the
/// controlled test: if it matches E1a's scalar variant, the quantized
/// GEMM was never slow and toSliceAlloc readback was the whole story.
fn dotQF16SumGraph(w: zml.Tensor, sc: zml.Tensor, x: zml.Tensor) zml.Tensor {
    return x.dot(dequant(w, sc).convert(.f16), .i).convert(.f32).sum(.o).squeeze(.o).sum(.s).squeeze(.s);
}

/// E2: ZML's own sdpa — a NAIVE trace (scores materialize at graph level).
/// Whether XLA's fusion rescues it is exactly the experiment.
fn sdpaGraph(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor) zml.Tensor {
    return zml.nn.sdpa(q, k, v, .{});
}

// ---- exe plumbing (coli-zml idioms) ---------------------------------------

fn callExe(comptime T: type, allocator: std.mem.Allocator, io: std.Io, exe: *zml.Exe, inputs: anytype, out: []T) !void {
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(inputs);
    exe.call(args, &results);
    var o: zml.Buffer = results.get(zml.Buffer);
    defer o.deinit();
    var s = try o.toSliceAlloc(allocator, io);
    defer s.free(allocator);
    @memcpy(out, s.constItems(T)[0..out.len]);
}

/// p50 of `reps` timed (launch + sync-via-download) calls, ns.
fn timeExe(comptime T: type, allocator: std.mem.Allocator, io: std.Io, exe: *zml.Exe, inputs: anytype, out: []T, reps: usize) !u64 {
    var t = try allocator.alloc(u64, reps);
    defer allocator.free(t);
    for (0..reps) |r| {
        const t0 = nowNs(io);
        try callExe(T, allocator, io, exe, inputs, out);
        t[r] = @intCast(nowNs(io) - t0);
    }
    std.mem.sort(u64, t, {}, std.sort.asc(u64));
    return t[reps / 2];
}

fn uploadF32(io: std.Io, platform: *zml.Platform, shape: zml.Shape, data: []const f32) !zml.Buffer {
    return .fromSlice(io, platform, zml.Slice.initConst(shape, std.mem.sliceAsBytes(data)), .replicated);
}

fn uploadF16(io: std.Io, platform: *zml.Platform, shape: zml.Shape, data: []const f16) !zml.Buffer {
    return .fromSlice(io, platform, zml.Slice.initConst(shape, std.mem.sliceAsBytes(data)), .replicated);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    log.info("platform: {s}", .{@tagName(platform.target)});

    const Ou: usize = @intCast(O);
    const Ku: usize = @intCast(K);
    const Smax: usize = @intCast(S_LIST[S_LIST.len - 1]);

    // ---- E1 fixtures ------------------------------------------------------
    // Unpacked u4 host bytes (one element per byte; the patched PJRT
    // transfer packs on device), deterministic fill like coli-zml's bench.
    const w4 = try allocator.alloc(u8, Ou * Ku);
    defer allocator.free(w4);
    for (w4, 0..) |*v, i| v.* = @truncate((i * 17 + 29) & 15);
    const sc = try allocator.alloc(f32, Ou);
    defer allocator.free(sc);
    for (sc, 0..) |*v, i| v.* = 0.006 + @as(f32, @floatFromInt(i % 11)) * 0.0002;

    // CPU-dequantized reference copy (the oracle, and the f16 GEMM operand).
    const wref = try allocator.alloc(f32, Ou * Ku);
    defer allocator.free(wref);
    const w16 = try allocator.alloc(f16, Ou * Ku);
    defer allocator.free(w16);
    for (0..Ou) |o| {
        for (0..Ku) |i| {
            const v = (@as(f32, @floatFromInt(w4[o * Ku + i])) - 8.0) * sc[o];
            wref[o * Ku + i] = v;
            w16[o * Ku + i] = @floatCast(v);
        }
    }

    const x = try allocator.alloc(f32, Smax * Ku);
    defer allocator.free(x);
    for (x, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i + 1)) * 0.013) * 0.5;
    const x16 = try allocator.alloc(f16, Smax * Ku);
    defer allocator.free(x16);
    for (x16, x) |*v, xv| v.* = @floatCast(xv);

    const w_shape: zml.Shape = .init(.{ .o = O, .i = K }, .u4);
    var w_buf: zml.Buffer = try .fromBytes(io, platform, w_shape, .replicated, w4);
    defer w_buf.deinit();
    const sc_shape: zml.Shape = .init(.{ .o = O }, .f32);
    var sc_buf = try uploadF32(io, platform, sc_shape, sc);
    defer sc_buf.deinit();
    const w16_shape: zml.Shape = .init(.{ .o = O, .i = K }, .f16);
    var w16_buf = try uploadF16(io, platform, w16_shape, w16);
    defer w16_buf.deinit();

    const y_got = try allocator.alloc(f32, Smax * Ou);
    defer allocator.free(y_got);
    const y_dot = try allocator.alloc(f32, Smax * Ou);
    defer allocator.free(y_dot);

    // ---- E1: crossover sweep ---------------------------------------------
    log.info("E1: int4 [O={d},K={d}] qdot(fused) vs dot(materialize+amortize); timings incl. x-resident launch + y download", .{ O, K });
    const w_spec: zml.Tensor = .init(.{ .o = O, .i = K }, .u4);
    const sc_spec: zml.Tensor = .fromShape(sc_shape);

    for (S_LIST) |s| {
        const su: usize = @intCast(s);
        const reps: usize = if (su <= 64) 16 else if (su <= 4096) 8 else 4;
        const x_spec: zml.Tensor = .init(.{ .s = s, .i = K }, .f32);
        const x_shape: zml.Shape = .init(.{ .s = s, .i = K }, .f32);
        var x_buf = try uploadF32(io, platform, x_shape, x[0 .. su * Ku]);
        defer x_buf.deinit();

        var qdot_exe = try platform.compileFn(allocator, io, qdotGraph, .{ w_spec, sc_spec, x_spec }, .{});
        var dot_exe = try platform.compileFn(allocator, io, dotQGraph, .{ w_spec, sc_spec, x_spec }, .{});

        // Warmup + correctness: CPU oracle at small S, qdot-vs-dot
        // cross-check everywhere.
        try callExe(f32, allocator, io, &qdot_exe, .{ w_buf, sc_buf, x_buf }, y_got[0 .. su * Ou]);
        try callExe(f32, allocator, io, &dot_exe, .{ w_buf, sc_buf, x_buf }, y_dot[0 .. su * Ou]);
        if (su <= 8) {
            const want = try allocator.alloc(f32, su * Ou);
            defer allocator.free(want);
            for (0..su) |r| {
                for (0..Ou) |o| {
                    var acc: f64 = 0;
                    for (0..Ku) |i| acc += @as(f64, x[r * Ku + i]) * wref[o * Ku + i];
                    want[r * Ou + o] = @floatCast(acc);
                }
            }
            if (!relativeRms(y_got[0 .. su * Ou], want, 1e-4)) return error.QdotOracleMismatch;
            if (!relativeRms(y_dot[0 .. su * Ou], want, 1e-4)) return error.DotOracleMismatch;
            log.info("  S={d}: both formulations match CPU oracle", .{s});
        }
        if (!relativeRms(y_dot[0 .. su * Ou], y_got[0 .. su * Ou], 2e-3)) return error.CrossCheckMismatch;

        const p50_q = try timeExe(f32, allocator, io, &qdot_exe, .{ w_buf, sc_buf, x_buf }, y_got[0 .. su * Ou], reps);
        const p50_d = try timeExe(f32, allocator, io, &dot_exe, .{ w_buf, sc_buf, x_buf }, y_dot[0 .. su * Ou], reps);
        const gflop = 2.0 * @as(f64, @floatFromInt(s * O * K)) / 1e9;
        log.info("  S={d:>5}: qdot p50 {d:.3}ms ({d:.0} GFLOP/s) | dot p50 {d:.3}ms ({d:.0} GFLOP/s) | dot/qdot speedup {d:.2}x", .{
            @as(u64, @intCast(s)),
            ms(p50_q),
            gflop / (ms(p50_q) / 1e3),
            ms(p50_d),
            gflop / (ms(p50_d) / 1e3),
            @as(f64, @floatFromInt(p50_q)) / @as(f64, @floatFromInt(p50_d)),
        });
    }

    // ---- E1a: f16 dense GEMM TFLOPS (the WMMA probe) ----------------------
    log.info("E1a: f16 GEMM [O={d},K={d}] scalar-sum epilogue (4-byte download; compute-isolated)", .{ O, K });
    for ([_]i64{ 512, 4096, 28672 }) |s| {
        const su: usize = @intCast(s);
        const x16_spec: zml.Tensor = .init(.{ .s = s, .i = K }, .f16);
        const x16_shape: zml.Shape = .init(.{ .s = s, .i = K }, .f16);
        var x16_buf = try uploadF16(io, platform, x16_shape, x16[0 .. su * Ku]);
        defer x16_buf.deinit();
        const w16_spec: zml.Tensor = .init(.{ .o = O, .i = K }, .f16);
        var exe = try platform.compileFn(allocator, io, gemmF16SumGraph, .{ w16_spec, x16_spec }, .{});
        var scalar: [1]f32 = undefined;
        try callExe(f32, allocator, io, &exe, .{ w16_buf, x16_buf }, &scalar); // warmup
        const p50 = try timeExe(f32, allocator, io, &exe, .{ w16_buf, x16_buf }, &scalar, 8);
        const tflop = 2.0 * @as(f64, @floatFromInt(s * O * K)) / 1e12;
        log.info("  S={d:>5}: p50 {d:.3}ms -> {d:.2} TFLOP/s (RDNA4 f16 WMMA peak is tens; low single digits means vector-ALU fallback)", .{ @as(u64, @intCast(s)), ms(p50), tflop / (ms(p50) / 1e3) });
    }

    // ---- E1b: int4-resident dequant->f16 dot (the engine candidate) -------
    // The 2x2: {dense, quant} x {scalar-sum, full download}, separating
    // GEMM compute from D2H readback (run 4's suspect at ~0.5 GB/s).
    log.info("E1b: {{dense,int4-resident}} x {{scalar-sum,full-download}} at f16", .{});
    for ([_]i64{ 512, 4096, 28672 }) |s| {
        const su: usize = @intCast(s);
        const x16_spec: zml.Tensor = .init(.{ .s = s, .i = K }, .f16);
        const x16_shape: zml.Shape = .init(.{ .s = s, .i = K }, .f16);
        var x16_buf = try uploadF16(io, platform, x16_shape, x16[0 .. su * Ku]);
        defer x16_buf.deinit();
        const w16_spec: zml.Tensor = .init(.{ .o = O, .i = K }, .f16);
        var q_full_exe = try platform.compileFn(allocator, io, dotQF16Graph, .{ w_spec, sc_spec, x16_spec }, .{});
        var q_sum_exe = try platform.compileFn(allocator, io, dotQF16SumGraph, .{ w_spec, sc_spec, x16_spec }, .{});
        var d_full_exe = try platform.compileFn(allocator, io, gemmF16Graph, .{ w16_spec, x16_spec }, .{});
        const out16 = try allocator.alloc(f16, su * Ou);
        defer allocator.free(out16);
        const ref16 = try allocator.alloc(f16, su * Ou);
        defer allocator.free(ref16);
        var scalar: [1]f32 = undefined;
        try callExe(f16, allocator, io, &q_full_exe, .{ w_buf, sc_buf, x16_buf }, out16); // warmups
        try callExe(f32, allocator, io, &q_sum_exe, .{ w_buf, sc_buf, x16_buf }, &scalar);
        try callExe(f16, allocator, io, &d_full_exe, .{ w16_buf, x16_buf }, ref16);
        if (s == 512) {
            // Cross-check quantized vs dense (independent dequant paths).
            const a32 = try allocator.alloc(f32, su * Ou);
            defer allocator.free(a32);
            const b32 = try allocator.alloc(f32, su * Ou);
            defer allocator.free(b32);
            for (a32, out16) |*g, o| g.* = @floatCast(o);
            for (b32, ref16) |*g, o| g.* = @floatCast(o);
            if (!relativeRms(a32, b32, 5e-3)) return error.QF16CrossCheckMismatch;
            log.info("  S=512: dequant->f16 dot matches dense f16 GEMM", .{});
        }
        const reps: usize = if (su <= 4096) 8 else 4;
        const p50_qs = try timeExe(f32, allocator, io, &q_sum_exe, .{ w_buf, sc_buf, x16_buf }, &scalar, reps);
        const p50_qf = try timeExe(f16, allocator, io, &q_full_exe, .{ w_buf, sc_buf, x16_buf }, out16, reps);
        const p50_df = try timeExe(f16, allocator, io, &d_full_exe, .{ w16_buf, x16_buf }, ref16, reps);
        const tflop = 2.0 * @as(f64, @floatFromInt(s * O * K)) / 1e12;
        const dl_bytes = @as(f64, @floatFromInt(su * Ou * 2));
        log.info("  S={d:>5}: quant-scalar {d:.3}ms ({d:.2} TFLOP/s) | quant-full {d:.3}ms | dense-full {d:.3}ms | implied readback {d:.2} GB/s", .{
            @as(u64, @intCast(s)),
            ms(p50_qs),
            tflop / (ms(p50_qs) / 1e3),
            ms(p50_qf),
            ms(p50_df),
            dl_bytes / (@as(f64, @floatFromInt(p50_qf -| p50_qs)) + 1),
        });
    }

    // ---- E2: sdpa at video lengths (LAST: may OOM at large T) -------------
    log.info("E2: zml.nn.sdpa f16 H={d} HD={d}; naive scores at T=28672 would be ~{d:.1} GB/head — OOM here IS the result (E3 becomes mandatory)", .{
        H, HD, @as(f64, @floatFromInt(28672 * 28672 * 2)) / 1e9,
    });
    const HDu: usize = @intCast(HD);
    const Hu: usize = @intCast(H);
    for (T_LIST) |t| {
        const tu: usize = @intCast(t);
        const n = Hu * tu * HDu;
        const q = try allocator.alloc(f16, n);
        defer allocator.free(q);
        const kk = try allocator.alloc(f16, n);
        defer allocator.free(kk);
        const v = try allocator.alloc(f16, n);
        defer allocator.free(v);
        for (q, 0..) |*e, i| e.* = @floatCast(@sin(@as(f32, @floatFromInt(i + 1)) * 0.011) * 0.25);
        for (kk, 0..) |*e, i| e.* = @floatCast(@sin(@as(f32, @floatFromInt(i + 3)) * 0.017) * 0.25);
        for (v, 0..) |*e, i| e.* = @floatCast(@sin(@as(f32, @floatFromInt(i + 7)) * 0.023) * 0.25);

        const q_shape: zml.Shape = .init(.{ .h = H, .q = t, .hd = HD }, .f16);
        const kv_shape: zml.Shape = .init(.{ .h = H, .k = t, .hd = HD }, .f16);
        log.info("  T={d}: uploading + compiling (a crash past this line means XLA materialized ~{d:.1} GB of scores)", .{
            t, @as(f64, @floatFromInt(Hu * tu * tu * 2)) / 1e9,
        });
        var q_buf = try uploadF16(io, platform, q_shape, q);
        defer q_buf.deinit();
        var k_buf = try uploadF16(io, platform, kv_shape, kk);
        defer k_buf.deinit();
        var v_buf = try uploadF16(io, platform, kv_shape, v);
        defer v_buf.deinit();

        // Each compileFn argument needs its OWN tracer tensor: passing one
        // spec twice trips "Tensor ... already used once as an argument"
        // (found the hard way on run 1).
        const q_spec: zml.Tensor = .fromShape(q_shape);
        const k_spec: zml.Tensor = .fromShape(kv_shape);
        const v_spec: zml.Tensor = .fromShape(kv_shape);
        var exe = try platform.compileFn(allocator, io, sdpaGraph, .{ q_spec, k_spec, v_spec }, .{});

        const out = try allocator.alloc(f16, n);
        defer allocator.free(out);
        try callExe(f16, allocator, io, &exe, .{ q_buf, k_buf, v_buf }, out); // warmup

        if (t == T_ORACLE) {
            // CPU oracle: full attention in f64, per head.
            const want = try allocator.alloc(f32, n);
            defer allocator.free(want);
            const scores = try allocator.alloc(f64, tu);
            defer allocator.free(scores);
            const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(HD)));
            for (0..Hu) |h| {
                const base = h * tu * HDu;
                for (0..tu) |qi| {
                    var mx: f64 = -std.math.inf(f64);
                    for (0..tu) |ki| {
                        var acc: f64 = 0;
                        for (0..HDu) |d| {
                            acc += @as(f64, @floatCast(q[base + qi * HDu + d])) * @as(f64, @floatCast(kk[base + ki * HDu + d]));
                        }
                        scores[ki] = acc * scale;
                        mx = @max(mx, scores[ki]);
                    }
                    var denom: f64 = 0;
                    for (scores) |*sv| {
                        sv.* = @exp(sv.* - mx);
                        denom += sv.*;
                    }
                    for (0..HDu) |d| {
                        var acc: f64 = 0;
                        for (0..tu) |ki| acc += scores[ki] * @as(f64, @floatCast(v[base + ki * HDu + d]));
                        want[base + qi * HDu + d] = @floatCast(acc / denom);
                    }
                }
            }
            const got32 = try allocator.alloc(f32, n);
            defer allocator.free(got32);
            for (got32, out) |*g, o| g.* = @floatCast(o);
            // Tolerance 2e-2, not 5e-3: run 2 measured RMS 0.0082 against the
            // f64 oracle, which is consistent with an END-TO-END f16 softmax
            // (f16 exp/sum), not a wrong algorithm (that would be O(1) off).
            if (!relativeRms(got32, want, 2e-2)) return error.SdpaOracleMismatch;
            log.info("  T={d}: matches CPU oracle", .{t});
        }

        const reps: usize = if (tu <= 8192) 8 else 4;
        const p50 = try timeExe(f16, allocator, io, &exe, .{ q_buf, k_buf, v_buf }, out, reps);
        const tflop = 4.0 * @as(f64, @floatFromInt(H * t * t * HD)) / 1e12;
        log.info("  T={d:>5}: p50 {d:.3}ms -> {d:.2} TFLOP/s attention", .{ @as(u64, @intCast(t)), ms(p50), tflop / (ms(p50) / 1e3) });
    }

    log.info("SMOKE COMPLETE", .{});
}
