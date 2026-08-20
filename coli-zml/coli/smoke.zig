//! Correctness + latency driver for the colibri-on-ZML backend.
//! Vectors mirror c/tests/test_backend_cuda.cu; the bench mirrors
//! c/tests/bench_tensor_core.cu shapes (D=6144, I=2048, int4) so the
//! numbers compare directly against `make cuda-bench HIP=1`.
const std = @import("std");
const zml = @import("zml");
const coli = @import("backend.zig");

const log = std.log;

pub const std_options: std.Options = .{
    .log_level = .info,
};

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

// ---- fmt=6 reference, ported from upstream test_backend_cuda.cu's own
// reference decoder (NOT from quant.h) so a common-mode port mistake in the
// backend cannot hide here — the same independence upstream's test has.
var t6state: u32 = 0x2545F491;
fn t6Rng() u32 {
    t6state ^= t6state << 13;
    t6state ^= t6state >> 17;
    t6state ^= t6state << 5;
    return t6state;
}

fn t6Fill(q: []u8, i_dim: usize, o_dim: usize) void {
    const nb = (i_dim + 255) / 256;
    for (0..o_dim) |o| {
        for (0..nb) |b| {
            const blk = q[(o * nb + b) * 98 ..][0..98];
            for (0..96) |i| blk[i] = @truncate(t6Rng());
            const h: u16 = @intCast((t6Rng() & 0x03FF) | ((10 + t6Rng() % 6) << 10));
            blk[96] = @truncate(h);
            blk[97] = @intCast(h >> 8);
        }
    }
}

fn t6DecodeRowRef(row: []const u8, i_dim: usize, w: []f32, grid: *const [256][4]u8) void {
    const nb = (i_dim + 255) / 256;
    for (0..nb) |b| {
        const blk = row[b * 98 ..][0..98];
        const h: u16 = @as(u16, blk[96]) | (@as(u16, blk[97]) << 8);
        const sg: u32 = @as(u32, h & 0x8000) << 16;
        const ex: u32 = (h >> 10) & 0x1F;
        const mn: u32 = h & 0x3FF;
        const bits: u32 = if (ex == 0)
            (if (mn != 0) sg | ((127 - 15) << 23) | (mn << 13) else sg)
        else if (ex == 31)
            sg | 0x7F800000 | (mn << 13)
        else
            sg | ((ex + 112) << 23) | (mn << 13);
        const d: f32 = @bitCast(bits);
        for (0..8) |ib| {
            const base = b * 256 + ib * 32;
            if (base >= i_dim) return;
            const word = std.mem.readInt(u32, blk[64 + ib * 4 ..][0..4], .little);
            const db = d * (0.5 + @as(f32, @floatFromInt(word >> 28))) * 0.5;
            for (0..4) |l| {
                const sev: u32 = (word >> @intCast(7 * l)) & 0x7F;
                const ga = &grid[blk[ib * 8 + l * 2]];
                const gb = &grid[blk[ib * 8 + l * 2 + 1]];
                var parity: u32 = 0;
                for (0..8) |j| {
                    const idx = base + l * 8 + j;
                    if (idx >= i_dim) break;
                    var neg: u32 = undefined;
                    if (j < 7) {
                        neg = (sev >> @intCast(j)) & 1;
                        parity ^= neg;
                    } else neg = parity;
                    const mag = @as(f32, @floatFromInt(if (j < 4) ga[j] else gb[j - 4])) * 0.5;
                    w[idx] = if (neg != 0) -mag * db else mag * db;
                }
            }
        }
    }
}

fn t6SignsRef(bits: []u8, n: usize) void {
    var s: u64 = 417 +% @as(u64, n);
    for (0..(n + 7) / 8) |i| {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        bits[i] = @truncate((s *% 2685821657736338717) >> 56);
    }
}

fn t6RotRef(row: []f32, dim: usize) void {
    var off: usize = 0;
    while (off < dim) {
        const rem = dim - off;
        var n = rem & (~rem + 1);
        while (n > 4096) n >>= 1;
        var bits: [4096 / 8]u8 = undefined;
        t6SignsRef(bits[0 .. (n + 7) / 8], n);
        const a = row[off..][0..n];
        for (a, 0..) |*v, i| {
            if ((bits[i >> 3] >> @intCast(i & 7)) & 1 != 0) v.* = -v.*;
        }
        var len: usize = 1;
        while (len < n) : (len <<= 1) {
            var i: usize = 0;
            while (i < n) : (i += len << 1) {
                for (i..i + len) |j| {
                    const u = a[j];
                    const v = a[j + len];
                    a[j] = u + v;
                    a[j + len] = u - v;
                }
            }
        }
        const sc = 1.0 / @sqrt(@as(f32, @floatFromInt(n)));
        for (a) |*v| v.* *= sc;
        off += n;
    }
}

fn testFmt6(backend: *coli.Backend, allocator: std.mem.Allocator) !void {
    var grid: [256][4]u8 = undefined;
    for (0..256) |i| {
        for (0..4) |j| grid[i][j] = @intCast((i * 4 + j) % 17);
    }
    backend.setE8Grid(&grid);

    const I = 256;
    const O = 128;
    const S = 2;
    const q = try allocator.alloc(u8, O * 98);
    defer allocator.free(q);
    t6Fill(q, I, O);
    const x = try allocator.alloc(f32, S * I);
    defer allocator.free(x);
    for (x, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i + 1)) * 0.031);

    // matmul
    const want = try allocator.alloc(f32, S * O);
    defer allocator.free(want);
    var wrow: [I]f32 = undefined;
    for (0..O) |o| {
        t6DecodeRowRef(q[o * 98 ..][0..98], I, &wrow, &grid);
        for (0..S) |s| {
            var acc: f64 = 0;
            for (0..I) |i| acc += @as(f64, x[s * I + i]) * wrow[i];
            want[s * O + o] = @floatCast(acc);
        }
    }
    const got = try allocator.alloc(f32, S * O);
    defer allocator.free(got);
    var t6 = try backend.upload(q, null, 6, I, O, 0);
    defer t6.deinit();
    try backend.matmul(&t6, got, x, S);
    if (!relativeRms(got, want, 1e-4)) return error.E8MatmulMismatch;
    log.info("✅ e8 matmul (fmt=6)", .{});

    // expert MLP with the device-side down rotation
    const qu = try allocator.alloc(u8, O * 98);
    defer allocator.free(qu);
    t6Fill(qu, I, O);
    const qd = try allocator.alloc(u8, I * 98);
    defer allocator.free(qd);
    t6Fill(qd, O, I);

    const g = try allocator.alloc(f32, S * O);
    defer allocator.free(g);
    const u = try allocator.alloc(f32, S * O);
    defer allocator.free(u);
    var wu: [I]f32 = undefined;
    for (0..O) |o| {
        t6DecodeRowRef(q[o * 98 ..][0..98], I, &wrow, &grid);
        t6DecodeRowRef(qu[o * 98 ..][0..98], I, &wu, &grid);
        for (0..S) |s| {
            var a: f64 = 0;
            var b: f64 = 0;
            for (0..I) |i| {
                a += @as(f64, x[s * I + i]) * wrow[i];
                b += @as(f64, x[s * I + i]) * wu[i];
            }
            g[s * O + o] = @floatCast(a);
            u[s * O + o] = @floatCast(b);
        }
    }
    for (g, u) |*gv, uv| gv.* = (gv.* / (1.0 + @exp(-gv.*))) * uv;
    for (0..S) |s| t6RotRef(g[s * O ..][0..O], O);
    const want_e = try allocator.alloc(f32, S * I);
    defer allocator.free(want_e);
    var wr2: [O]f32 = undefined;
    for (0..I) |o| {
        t6DecodeRowRef(qd[o * 98 ..][0..98], O, &wr2, &grid);
        for (0..S) |s| {
            var acc: f64 = 0;
            for (0..O) |i| acc += @as(f64, g[s * O + i]) * wr2[i];
            want_e[s * I + o] = @floatCast(acc);
        }
    }
    var tg6 = try backend.upload(q, null, 6, I, O, 0);
    defer tg6.deinit();
    var tu6 = try backend.upload(qu, null, 6, I, O, 0);
    defer tu6.deinit();
    var td6 = try backend.upload(qd, null, 6, O, I, 0);
    defer td6.deinit();
    const got_e = try allocator.alloc(f32, S * I);
    defer allocator.free(got_e);
    try backend.expertMlp(&tg6, &tu6, &td6, got_e, x, S);
    if (!relativeRms(got_e, want_e, 2e-4)) return error.E8ExpertMlpMismatch;
    log.info("✅ e8 expert_mlp (down rotation folded)", .{});
}

fn closeEnough(got: []const f32, want: []const f32) bool {
    for (got, want) |g, w| {
        if (@abs(g - w) > 1e-4) {
            log.err("mismatch: got {d} want {d}", .{ g, w });
            return false;
        }
    }
    return true;
}

fn nowNs(io: std.Io) i96 {
    const ts: std.Io.Timestamp = .now(io, .awake);
    return ts.toNanoseconds();
}

fn us(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1000.0;
}

/// Phase-timed expert-MLP round trips at one (S, D, H) shape.
fn benchMlp(backend: *coli.Backend, gate: *const coli.DeviceTensor, up: *const coli.DeviceTensor, down: *const coli.DeviceTensor, x: []const f32, s: i64) !void {
    const reps = 64;
    const d_dim: usize = @intCast(gate.i);
    const exe = try backend.getExe(.{ .kind = .mlp, .s = s, .i = gate.i, .o = gate.o, .fmt = gate.fmt, .gs = gate.gs });
    const y = try backend.allocator.alloc(f32, @as(usize, @intCast(s)) * d_dim);
    defer backend.allocator.free(y);

    var ph = [_]u64{0} ** 3; // upload, launch, sync+download
    var totals: [reps]u64 = undefined;

    for (0..reps) |rep| {
        const t_start = nowNs(backend.io);
        var last = t_start;

        var args = try exe.args(backend.allocator);
        defer args.deinit(backend.allocator);
        var results = try exe.results(backend.allocator);
        defer results.deinit(backend.allocator);

        var x_buf = try backend.uploadX(x[0 .. @as(usize, @intCast(s)) * d_dim], s, gate.i);
        var now = nowNs(backend.io);
        ph[0] += @intCast(now - last);
        last = now;

        args.set(.{ gate.weights, gate.scales.?, up.weights, up.scales.?, down.weights, down.scales.?, x_buf });
        exe.call(args, &results);
        now = nowNs(backend.io);
        ph[1] += @intCast(now - last);
        last = now;

        var out: zml.Buffer = results.get(zml.Buffer);
        var out_slice = try out.toSliceAlloc(backend.allocator, backend.io);
        @memcpy(y, out_slice.constItems(f32)[0..y.len]);
        out_slice.free(backend.allocator);
        out.deinit();
        x_buf.deinit();
        now = nowNs(backend.io);
        ph[2] += @intCast(now - last);
        totals[rep] = @intCast(now - t_start);
    }

    std.mem.sort(u64, &totals, {}, std.sort.asc(u64));
    log.info("rows={d} zml_mlp mean={d:.3}ms | upload {d:.1}µs launch {d:.1}µs sync+dl {d:.1}µs | p50 {d:.1} p95 {d:.1} p99 {d:.1}µs", .{
        s,
        us((ph[0] + ph[1] + ph[2]) / reps) / 1000.0,
        us(ph[0] / reps),
        us(ph[1] / reps),
        us(ph[2] / reps),
        us(totals[reps / 2]),
        us(totals[reps * 95 / 100]),
        us(totals[reps * 99 / 100]),
    });
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var backend: coli.Backend = try .init(allocator, io);
    log.info("platform: {s}", .{@tagName(backend.platform.target)});

    const x = [8]f32{ 1, -2, 3, -4, 2, 1, -1, 0.5 };
    var got: [4]f32 = undefined;

    // fmt=1 int8 [O=2,I=4], per-row scales.
    const q8 = [8]i8{ 1, 2, 3, 4, -1, 2, -3, 4 };
    const s8 = [2]f32{ 0.5, 2.0 };
    const want8 = [4]f32{ -5.0, -60.0, 1.5, 10.0 };
    var t8 = try backend.upload(std.mem.sliceAsBytes(&q8), &s8, 1, 4, 2, 0);
    defer t8.deinit();
    try backend.matmul(&t8, &got, &x, 2);
    if (!closeEnough(&got, &want8)) return error.Q8Mismatch;
    log.info("✅ q8", .{});

    // fmt=2 int4, rows [-8,-1,0,7] and [1,2,3,4], packed low nibble first.
    const q4 = [4]u8{ 0x70, 0xf8, 0xa9, 0xcb };
    const s4 = [2]f32{ 1.0, 0.25 };
    const want4 = [2]f32{ -34.0, -2.5 };
    var t4 = try backend.upload(&q4, &s4, 2, 4, 2, 0);
    defer t4.deinit();
    try backend.matmul(&t4, got[0..2], &x, 1);
    if (!closeEnough(got[0..2], &want4)) return error.Q4Mismatch;
    log.info("✅ q4", .{});

    // fmt=3 int2, 4 values per byte.
    const q2 = [2]u8{ 0xe4, 0x1b };
    const s2 = [2]f32{ 0.5, 2.0 };
    const want2 = [2]f32{ -2.0, 12.0 };
    var t2 = try backend.upload(&q2, &s2, 3, 4, 2, 0);
    defer t2.deinit();
    try backend.matmul(&t2, got[0..2], &x, 1);
    if (!closeEnough(got[0..2], &want2)) return error.Q2Mismatch;
    log.info("✅ q2", .{});

    // fmt=0 f32.
    const wf = [8]f32{ 1, 0, -1, 2, 0.5, 0.5, 0.5, 0.5 };
    const wantf = [2]f32{ -10.0, -1.0 };
    var tf = try backend.upload(std.mem.sliceAsBytes(&wf), null, 0, 4, 2, 0);
    defer tf.deinit();
    try backend.matmul(&tf, got[0..2], &x, 1);
    if (!closeEnough(got[0..2], &wantf)) return error.F32Mismatch;
    log.info("✅ f32", .{});

    // fmt=4 grouped int4 (offset-8): I=4, gs=2, ng=2; scales[o*ng+g].
    // W row0 values [1,2,3,4] · scales {0.5,2} → [0.5,1,6,8]
    // W row1 values [-1,2,-3,4] · scales {1,0.25} → [-1,2,-0.75,1]
    const q4g = [4]u8{ 0xa9, 0xcb, 0xa7, 0xc5 };
    const s4g = [4]f32{ 0.5, 2.0, 1.0, 0.25 };
    const want4g = [2]f32{ -15.5, -11.25 };
    var t4g = try backend.upload(&q4g, &s4g, 4, 4, 2, 2);
    defer t4g.deinit();
    try backend.matmul(&t4g, got[0..2], &x, 1);
    if (!closeEnough(got[0..2], &want4g)) return error.Q4gMismatch;
    log.info("✅ q4 grouped (gs=2)", .{});

    // Fused expert MLP, f32 vectors from the upstream test: gate/up select
    // x0/x1, down mixes them.
    const eg = [8]f32{ 1, 0, 0, 0, 0, 1, 0, 0 };
    const ed = [8]f32{ 1, 0, 0, 1, 1, 1, 1, -1 };
    var tg = try backend.upload(std.mem.sliceAsBytes(&eg), null, 0, 4, 2, 0);
    defer tg.deinit();
    var tu = try backend.upload(std.mem.sliceAsBytes(&eg), null, 0, 4, 2, 0);
    defer tu.deinit();
    var td = try backend.upload(std.mem.sliceAsBytes(&ed), null, 0, 2, 4, 0);
    defer td.deinit();

    var want_expert: [8]f32 = undefined;
    for (0..2) |s| {
        var a = x[s * 4];
        var b = x[s * 4 + 1];
        a = (a / (1.0 + @exp(-a))) * a;
        b = (b / (1.0 + @exp(-b))) * b;
        want_expert[s * 4] = a;
        want_expert[s * 4 + 1] = b;
        want_expert[s * 4 + 2] = a + b;
        want_expert[s * 4 + 3] = a - b;
    }
    var expert: [8]f32 = undefined;
    try backend.expertMlp(&tg, &tu, &td, &expert, &x, 2);
    if (!closeEnough(&expert, &want_expert)) return error.ExpertMlpMismatch;
    log.info("✅ expert_mlp (fused)", .{});

    // Expert group: two experts, one row each, same tensors.
    var grouped: [8]f32 = undefined;
    const gates = [2]*const coli.DeviceTensor{ &tg, &tg };
    const ups = [2]*const coli.DeviceTensor{ &tu, &tu };
    const downs = [2]*const coli.DeviceTensor{ &td, &td };
    try backend.expertGroup(&gates, &ups, &downs, &.{ 1, 1 }, &grouped, &x);
    if (!closeEnough(&grouped, &want_expert)) return error.ExpertGroupMismatch;
    log.info("✅ expert_group (f32, loop path)", .{});

    // Same group through the FUSED single-executable path (needs quantized
    // tensors): identical selector weights as i8 with unit scales.
    const q8g = [8]i8{ 1, 0, 0, 0, 0, 1, 0, 0 };
    const q8d = [8]i8{ 1, 0, 0, 1, 1, 1, 1, -1 };
    const ones2 = [2]f32{ 1, 1 };
    const ones4 = [4]f32{ 1, 1, 1, 1 };
    var qg1 = try backend.upload(std.mem.sliceAsBytes(&q8g), &ones2, 1, 4, 2, 0);
    defer qg1.deinit();
    var qd1 = try backend.upload(std.mem.sliceAsBytes(&q8d), &ones4, 1, 2, 4, 0);
    defer qd1.deinit();
    var qgrouped: [8]f32 = undefined;
    const qgates = [2]*const coli.DeviceTensor{ &qg1, &qg1 };
    const qdowns = [2]*const coli.DeviceTensor{ &qd1, &qd1 };
    try backend.expertGroup(&qgates, &qgates, &qdowns, &.{ 1, 1 }, &qgrouped, &x);
    if (!closeEnough(&qgrouped, &want_expert)) return error.FusedGroupMismatch;
    log.info("✅ expert_group fused (i8, one executable)", .{});

    try testFmt6(&backend, allocator);

    // ---- Matched-shape bench vs `make cuda-bench HIP=1` ----
    // Same shapes and fill patterns as bench_tensor_core.cu:
    // D=6144, I=2048, int4 gate/up [I out, D in], down [D out, I in].
    const D = 6144;
    const I = 2048;
    log.info("bench shapes: D={d} I={d} fmt=int4 (matches HIP cuda-bench)", .{ D, I });

    const hidden = try allocator.alloc(u8, I * D / 2);
    defer allocator.free(hidden);
    const down_w = try allocator.alloc(u8, D * I / 2);
    defer allocator.free(down_w);
    for (hidden, 0..) |*v, i| v.* = @truncate(i * 17 + 29);
    for (down_w, 0..) |*v, i| v.* = @truncate(i * 13 + 41);
    const hs = try allocator.alloc(f32, I);
    defer allocator.free(hs);
    const ds = try allocator.alloc(f32, D);
    defer allocator.free(ds);
    for (hs, 0..) |*v, i| v.* = 0.006 + @as(f32, @floatFromInt(i % 11)) * 0.0002;
    for (ds, 0..) |*v, i| v.* = 0.006 + @as(f32, @floatFromInt(i % 7)) * 0.0002;
    const bx = try allocator.alloc(f32, 8 * D);
    defer allocator.free(bx);
    for (bx, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i + 1)) * 0.013) * 2.0;

    var bg = try backend.upload(hidden, hs, 2, D, I, 0);
    defer bg.deinit();
    var bu = try backend.upload(hidden, hs, 2, D, I, 0);
    defer bu.deinit();
    var bd = try backend.upload(down_w, ds, 2, I, D, 0);
    defer bd.deinit();

    for ([_]i64{ 1, 2, 4, 8 }) |rows| {
        // Warm the exe cache (XLA compile) outside the timed loop.
        const y = try allocator.alloc(f32, @as(usize, @intCast(rows)) * D);
        defer allocator.free(y);
        try backend.expertMlp(&bg, &bu, &bd, y, bx[0 .. @as(usize, @intCast(rows)) * D], rows);
        try benchMlp(&backend, &bg, &bu, &bd, bx, rows);
    }

    // ---- Whole-group bench: 8 DISTINCT int4 experts × 1 row each ----
    // Fused single executable vs a loop of 8 fused MLPs, outputs compared.
    var eg8: [8]coli.DeviceTensor = undefined;
    var eu8: [8]coli.DeviceTensor = undefined;
    var ed8: [8]coli.DeviceTensor = undefined;
    for (0..8) |e| {
        for (hidden, 0..) |*v, i| v.* = @truncate(i * 17 + 29 + e * 7);
        for (down_w, 0..) |*v, i| v.* = @truncate(i * 13 + 41 + e * 11);
        eg8[e] = try backend.upload(hidden, hs, 2, D, I, 0);
        eu8[e] = try backend.upload(hidden, hs, 2, D, I, 0);
        ed8[e] = try backend.upload(down_w, ds, 2, I, D, 0);
    }
    defer for (&eg8) |*t| t.deinit();
    defer for (&eu8) |*t| t.deinit();
    defer for (&ed8) |*t| t.deinit();
    var pg: [8]*const coli.DeviceTensor = undefined;
    var pu: [8]*const coli.DeviceTensor = undefined;
    var pd: [8]*const coli.DeviceTensor = undefined;
    for (0..8) |e| {
        pg[e] = &eg8[e];
        pu[e] = &eu8[e];
        pd[e] = &ed8[e];
    }
    const rows1 = [_]i32{1} ** 8;
    const gy_f = try allocator.alloc(f32, 8 * D);
    defer allocator.free(gy_f);
    const gy_l = try allocator.alloc(f32, 8 * D);
    defer allocator.free(gy_l);
    const gx = bx[0 .. 8 * D];

    {
        const c0: std.Io.Timestamp = .now(io, .awake);
        try backend.expertGroup(&pg, &pu, &pd, &rows1, gy_f, gx);
        log.info("group-of-8 first call (incl. XLA compile) [{f}]", .{c0.untilNow(io, .awake)});
    }
    for (0..8) |e| try backend.expertMlp(pg[e], pu[e], pd[e], gy_l[e * D ..][0..D], gx[e * D ..][0..D], 1);
    var maxdiff: f32 = 0;
    for (gy_f, gy_l) |a, b| maxdiff = @max(maxdiff, @abs(a - b));
    if (maxdiff > 1e-3) {
        log.err("fused group vs loop diverge: maxdiff {d}", .{maxdiff});
        return error.GroupBenchMismatch;
    }

    const reps = 32;
    var t_fused: [reps]u64 = undefined;
    var t_loop: [reps]u64 = undefined;
    for (0..reps) |r| {
        var t0 = nowNs(io);
        try backend.expertGroup(&pg, &pu, &pd, &rows1, gy_f, gx);
        t_fused[r] = @intCast(nowNs(io) - t0);
        t0 = nowNs(io);
        for (0..8) |e| try backend.expertMlp(pg[e], pu[e], pd[e], gy_l[e * D ..][0..D], gx[e * D ..][0..D], 1);
        t_loop[r] = @intCast(nowNs(io) - t0);
    }
    std.mem.sort(u64, &t_fused, {}, std.sort.asc(u64));
    std.mem.sort(u64, &t_loop, {}, std.sort.asc(u64));
    log.info("group-of-8: fused ONE exe p50 {d:.2}ms p95 {d:.2}ms | loop of 8 p50 {d:.2}ms p95 {d:.2}ms", .{
        us(t_fused[reps / 2]) / 1000.0,
        us(t_fused[reps * 95 / 100]) / 1000.0,
        us(t_loop[reps / 2]) / 1000.0,
        us(t_loop[reps * 95 / 100]) / 1000.0,
    });

    // Same group-of-8, but fmt=4 GROUPED scales (gs=128) — the format real
    // converted models use. Fusion behavior of the grouped-dequant subgraph
    // may differ from fmt=2's; this is the tripwire for it.
    {
        const gs: i32 = 128;
        const ng_gu = D / 128;
        const ng_d = I / 128;
        const hs4 = try allocator.alloc(f32, I * ng_gu);
        defer allocator.free(hs4);
        for (hs4, 0..) |*v, i| v.* = 0.006 + @as(f32, @floatFromInt(i % 11)) * 0.0002;
        const ds4 = try allocator.alloc(f32, D * ng_d);
        defer allocator.free(ds4);
        for (ds4, 0..) |*v, i| v.* = 0.006 + @as(f32, @floatFromInt(i % 7)) * 0.0002;

        var qg4: [8]coli.DeviceTensor = undefined;
        var qu4: [8]coli.DeviceTensor = undefined;
        var qd4: [8]coli.DeviceTensor = undefined;
        for (0..8) |e| {
            for (hidden, 0..) |*v, i| v.* = @truncate(i * 17 + 29 + e * 7);
            for (down_w, 0..) |*v, i| v.* = @truncate(i * 13 + 41 + e * 11);
            qg4[e] = try backend.upload(hidden, hs4, 4, D, I, gs);
            qu4[e] = try backend.upload(hidden, hs4, 4, D, I, gs);
            qd4[e] = try backend.upload(down_w, ds4, 4, I, D, gs);
        }
        defer for (&qg4) |*t| t.deinit();
        defer for (&qu4) |*t| t.deinit();
        defer for (&qd4) |*t| t.deinit();
        var pg4: [8]*const coli.DeviceTensor = undefined;
        var pu4: [8]*const coli.DeviceTensor = undefined;
        var pd4: [8]*const coli.DeviceTensor = undefined;
        for (0..8) |e| {
            pg4[e] = &qg4[e];
            pu4[e] = &qu4[e];
            pd4[e] = &qd4[e];
        }
        try backend.expertGroup(&pg4, &pu4, &pd4, &rows1, gy_f, gx);
        var t4b: [reps]u64 = undefined;
        for (0..reps) |r| {
            const t0 = nowNs(io);
            try backend.expertGroup(&pg4, &pu4, &pd4, &rows1, gy_f, gx);
            t4b[r] = @intCast(nowNs(io) - t0);
        }
        std.mem.sort(u64, &t4b, {}, std.sort.asc(u64));
        log.info("group-of-8 fmt=4 gs=128: p50 {d:.2}ms p95 {d:.2}ms", .{
            us(t4b[reps / 2]) / 1000.0,
            us(t4b[reps * 95 / 100]) / 1000.0,
        });

        // Same calls but with 15ms idle gaps (the engine's decode pattern) —
        // probes GPU downclocking on sparse dispatch.
        var tsp: [reps]u64 = undefined;
        for (0..reps) |r| {
            try std.Io.sleep(io, .fromMilliseconds(15), .awake);
            const t0 = nowNs(io);
            try backend.expertGroup(&pg4, &pu4, &pd4, &rows1, gy_f, gx);
            tsp[r] = @intCast(nowNs(io) - t0);
        }
        std.mem.sort(u64, &tsp, {}, std.sort.asc(u64));
        log.info("group-of-8 fmt=4 SPACED (15ms gaps): p50 {d:.2}ms p95 {d:.2}ms", .{
            us(tsp[reps / 2]) / 1000.0,
            us(tsp[reps * 95 / 100]) / 1000.0,
        });
    }

    log.info("SMOKE PASS", .{});
}
