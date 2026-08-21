//! E-STREAM-1/2 (docs/lab-notebook.md, 2026-08-21 pre-registration):
//! the streaming loader's transfer facts, measured before any loader
//! code exists.
//!
//! E-STREAM-1: host-to-device rate for one block-sized blob (~425 MB,
//! the 20 GB int8 DiT / 48 blocks) from pageable memory, and again
//! after PJRT_Client_DmaMap pins the same region (the ROCm plugin may
//! report Unimplemented — that is a result, not a failure).
//!
//! E-STREAM-2: overlap — enqueue a long compute, then an async upload
//! (wait=false), await both; compare wall time against the serial sum.
//! If the plugin runs H2D on its own stream the combined time is
//! ~max(compute, upload); if it serializes, ~the sum.
const std = @import("std");
const zml = @import("zml");

const log = std.log;

pub const std_options: std.Options = .{ .log_level = .info };

/// One quantized block: ~20 GB / 48. Page-aligned for dmaMap.
const BLOB_BYTES: usize = 425 * 1024 * 1024;
const GEMM_N: i64 = 4096;
const GEMM_CHAIN: usize = 128; // ~17.6 TFLOP -> ~300 ms at E1's 59 TFLOP/s

fn nowNs(io: std.Io) i96 {
    const ts: std.Io.Timestamp = .now(io, .awake);
    return ts.toNanoseconds();
}

fn gbps(bytes: usize, ns: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(ns));
}

/// Dependent GEMM chain with a scalar-sum epilogue (the E1b lesson:
/// never time through a full-output readback).
fn gemmChainGraph(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    var acc = a;
    for (0..GEMM_CHAIN) |_| {
        acc = acc.dot(b, .k).withTags(.{ .m, .k });
    }
    return acc.convert(.f32).sum(.m).squeeze(.m).sum(.k).squeeze(.k);
}

fn uploadOnce(io: std.Io, platform: *zml.Platform, shape: zml.Shape, blob: []const u8, wait: bool) !zml.Buffer {
    return zml.Buffer.fromBytesOpts(io, platform, shape, .replicated, blob, .{ .wait = wait });
}

fn p50Upload(io: std.Io, platform: *zml.Platform, shape: zml.Shape, blob: []const u8, reps: usize, t: []u64) !u64 {
    for (0..reps) |r| {
        const t0 = nowNs(io);
        var buf = try uploadOnce(io, platform, shape, blob, true);
        t[r] = @intCast(nowNs(io) - t0);
        buf.deinit();
    }
    std.mem.sort(u64, t[0..reps], {}, std.sort.asc(u64));
    return t[reps / 2];
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    log.info("platform: {s}", .{@tagName(platform.target)});

    // Page-aligned blob, deterministic fill.
    const blob = try std.heap.page_allocator.alignedAlloc(u8, .fromByteUnits(4096), BLOB_BYTES);
    defer std.heap.page_allocator.free(blob);
    var rng = std.Random.DefaultPrng.init(97);
    rng.random().bytes(blob[0 .. 1 << 20]);
    var off: usize = 1 << 20;
    while (off < BLOB_BYTES) : (off += 1 << 20) {
        const n = @min(BLOB_BYTES - off, 1 << 20);
        @memcpy(blob[off .. off + n], blob[0..n]);
    }
    const blob_shape = zml.Shape.init(.{ .n = @as(i64, @intCast(BLOB_BYTES)) }, .i8);

    var t: [9]u64 = undefined;

    // ---- compute exe, built early so warmup covers it ---------------------
    const a_spec: zml.Tensor = .fromShape(zml.Shape.init(.{ .m = GEMM_N, .k = GEMM_N }, .f16));
    const b_spec: zml.Tensor = .fromShape(zml.Shape.init(.{ .k = GEMM_N, .n = GEMM_N }, .f16));
    var exe = try platform.compileFn(allocator, io, gemmChainGraph, .{ a_spec, b_spec }, .{});

    const mat = try allocator.alloc(f16, @as(usize, @intCast(GEMM_N * GEMM_N)));
    defer allocator.free(mat);
    for (mat, 0..) |*v, i| v.* = @floatCast(@sin(@as(f32, @floatFromInt(i % 977))) * 0.01);
    var a_buf: zml.Buffer = try .fromBytes(io, platform, a_spec.shape(), .replicated, std.mem.sliceAsBytes(mat));
    defer a_buf.deinit();
    var b_buf: zml.Buffer = try .fromBytes(io, platform, b_spec.shape(), .replicated, std.mem.sliceAsBytes(mat));
    defer b_buf.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(.{ a_buf, b_buf });

    var scalar_out: [1]f32 = undefined;

    // Warmup: clocks ramp under load; the first cold batch otherwise poisons
    // whichever measurement runs first (caught in run 1 of this experiment —
    // "compute+upload" beat "compute alone" by 1.6x, which is impossible).
    for (0..3) |_| {
        exe.call(args, &results);
        var out: zml.Buffer = results.get(zml.Buffer);
        defer out.deinit();
        var s = try out.toSliceAlloc(allocator, io);
        defer s.free(allocator);
        var up = try uploadOnce(io, platform, blob_shape, blob, true);
        up.deinit();
    }

    // ---- E-STREAM-1a: pageable p50 ----------------------------------------
    const pageable_ns = try p50Upload(io, platform, blob_shape, blob, 9, &t);
    log.info("E-STREAM-1a pageable: {d} MB in p50 {d:.1} ms -> {d:.2} GB/s", .{
        BLOB_BYTES / (1024 * 1024),
        @as(f64, @floatFromInt(pageable_ns)) / 1e6,
        gbps(BLOB_BYTES, pageable_ns),
    });

    // ---- E-STREAM-1b: the same region, DmaMap-pinned ----------------------
    // Stays mapped through E-STREAM-2: pinned is the production path.
    var pinned_ns: ?u64 = null;
    if (platform.pjrt_client.dmaMap(platform.pjrt_api, blob)) |_| {
        pinned_ns = try p50Upload(io, platform, blob_shape, blob, 9, &t);
        log.info("E-STREAM-1b dmaMap-pinned: p50 {d:.1} ms -> {d:.2} GB/s ({d:.2}x pageable)", .{
            @as(f64, @floatFromInt(pinned_ns.?)) / 1e6,
            gbps(BLOB_BYTES, pinned_ns.?),
            @as(f64, @floatFromInt(pageable_ns)) / @as(f64, @floatFromInt(pinned_ns.?)),
        });
    } else |err| {
        log.info("E-STREAM-1b: dmaMap not available on this plugin: {t}", .{err});
    }
    defer if (pinned_ns != null) {
        platform.pjrt_client.dmaUnmap(platform.pjrt_api, blob) catch |err| {
            log.warn("dmaUnmap failed: {t}", .{err});
        };
    };

    // ---- E-STREAM-2: upload while a GEMM chain computes -------------------
    // Serial compute p50 (call + sync via 4-byte readback).
    const computeP50 = struct {
        fn go(io_: std.Io, allocator_: std.mem.Allocator, exe_: *zml.Exe, args_: anytype, results_: anytype, t_: []u64, out_scalar: *[1]f32) !u64 {
            for (0..5) |r| {
                const t0 = nowNs(io_);
                exe_.call(args_.*, results_);
                var out: zml.Buffer = results_.get(zml.Buffer);
                defer out.deinit();
                var s = try out.toSliceAlloc(allocator_, io_);
                defer s.free(allocator_);
                @memcpy(out_scalar, s.constItems(f32)[0..1]);
                t_[r] = @intCast(nowNs(io_) - t0);
            }
            std.mem.sort(u64, t_[0..5], {}, std.sort.asc(u64));
            return t_[2];
        }
    }.go;

    _ = computeP50; // interleaved protocol below supersedes the batch form

    // Interleaved A/B: alternate alone/together per repetition so both
    // conditions sample the SAME clock/thermal trajectory (run 1 measured
    // "together" faster than "alone" purely from clock ramp; run 2 still
    // drifted 297 -> 241 ms across batches). Upload is the pinned path
    // when dmaMap succeeded — the production configuration.
    var alone: [9]u64 = undefined;
    var together: [9]u64 = undefined;
    for (0..9) |r| {
        {
            const t0 = nowNs(io);
            exe.call(args, &results);
            var out: zml.Buffer = results.get(zml.Buffer);
            defer out.deinit();
            var s = try out.toSliceAlloc(allocator, io);
            defer s.free(allocator);
            @memcpy(&scalar_out, s.constItems(f32)[0..1]);
            alone[r] = @intCast(nowNs(io) - t0);
        }
        {
            const t0 = nowNs(io);
            exe.call(args, &results);
            var out: zml.Buffer = results.get(zml.Buffer);
            defer out.deinit();
            const t1 = nowNs(io);
            var up = try uploadOnce(io, platform, blob_shape, blob, false);
            const t2 = nowNs(io);
            try up.await(io);
            const t3 = nowNs(io);
            up.deinit();
            var s = try out.toSliceAlloc(allocator, io);
            defer s.free(allocator);
            @memcpy(&scalar_out, s.constItems(f32)[0..1]);
            together[r] = @intCast(nowNs(io) - t0);
            if (r == 4) log.info("  together breakdown: enqueue {d:.1} ms, fromBytes CALL {d:.1} ms, upload await {d:.1} ms, readback rest", .{
                @as(f64, @floatFromInt(@as(u64, @intCast(t1 - t0)))) / 1e6,
                @as(f64, @floatFromInt(@as(u64, @intCast(t2 - t1)))) / 1e6,
                @as(f64, @floatFromInt(@as(u64, @intCast(t3 - t2)))) / 1e6,
            });
        }
    }
    std.mem.sort(u64, &alone, {}, std.sort.asc(u64));
    std.mem.sort(u64, &together, {}, std.sort.asc(u64));
    const compute_ns = alone[4];
    const both_ns = together[4];
    const up_ns = pinned_ns orelse pageable_ns;

    log.info("E-STREAM-2 (interleaved, {s} upload): compute alone p50 {d:.1} ms, compute+upload p50 {d:.1} ms, upload alone {d:.1} ms (chain sum={e:.3})", .{
        if (pinned_ns != null) "pinned" else "pageable",
        @as(f64, @floatFromInt(compute_ns)) / 1e6,
        @as(f64, @floatFromInt(both_ns)) / 1e6,
        @as(f64, @floatFromInt(up_ns)) / 1e6,
        scalar_out[0],
    });
    const added = @as(f64, @floatFromInt(both_ns)) - @as(f64, @floatFromInt(compute_ns));
    const hidden = std.math.clamp(1.0 - added / @as(f64, @floatFromInt(up_ns)), 0.0, 1.0);
    log.info("E-STREAM-2 overlap: upload added {d:.1} ms of its {d:.1} ms -> {d:.0}% hidden under compute", .{
        added / 1e6, @as(f64, @floatFromInt(up_ns)) / 1e6, hidden * 100.0,
    });
}
