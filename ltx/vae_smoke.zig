//! E4 smoke: 3D convolution through XLA on ROCm, at the LTX-2.5 Conv VAE
//! decoder's real shapes (see docs/lab-notebook.md, Phase 1 entry, for
//! the pre-registered hypotheses). Shapes come from the checkpoint header:
//! 3x3x3 convs at 1024/512/256/128 channels across five stages, plus the
//! 1024->4096 pixel-shuffle upsample conv. Benchmarked at a latent tile
//! of 8x16x16 (t,h,w) and the corresponding per-stage resolutions.
//!
//! f16 rather than bf16: RDNA4 rates them identically and Zig has no
//! native bf16 host type. Symmetric SAME padding where the real decoder
//! is temporally causal: identical arithmetic, different edge semantics.
const std = @import("std");
const zml = @import("zml");

const log = std.log;

pub const std_options: std.Options = .{
    .log_level = .info,
};

fn nowNs(io: std.Io) i96 {
    const ts: std.Io.Timestamp = .now(io, .awake);
    return ts.toNanoseconds();
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// 3D convolution, NCTHW input, OI-ktkhkw kernel, stride 1, SAME padding.
fn conv3dGraph(x: zml.Tensor, w: zml.Tensor) zml.Tensor {
    return x.convolution(w, .{
        .window_strides = &.{ 1, 1, 1 },
        .pad_value = &.{ 1, 1, 1, 1, 1, 1 },
        .lhs_dilation = &.{ 1, 1, 1 },
        .rhs_dilation = &.{ 1, 1, 1 },
        .window_reversal = &.{ false, false, false },
        .input_batch_dimension = 0,
        .input_feature_dimension = 1,
        .input_spatial_dimensions = &.{ 2, 3, 4 },
        .kernel_output_feature_dimension = 0,
        .kernel_input_feature_dimension = 1,
        .kernel_spatial_dimensions = &.{ 2, 3, 4 },
        .output_batch_dimension = 0,
        .output_feature_dimension = 1,
        .output_spatial_dimensions = &.{ 2, 3, 4 },
        .feature_group_count = 1,
        .batch_group_count = 1,
    });
}

const Case = struct {
    name: []const u8,
    cin: i64,
    cout: i64,
    t: i64,
    h: i64,
    w: i64,
};

// The decoder's workhorse shapes at a latent tile of 8x16x16.
// Stage volumes: A (latent) 8x16x16; C (after two 2x2x2 upsamples)
// 32x64x64; E (after temporal x2 and spatial 2x2) 64x128x128.
const CASES = [_]Case{
    .{ .name = "stageA res 1024ch", .cin = 1024, .cout = 1024, .t = 8, .h = 16, .w = 16 },
    .{ .name = "up1 1024->4096", .cin = 1024, .cout = 4096, .t = 8, .h = 16, .w = 16 },
    .{ .name = "stageC res 512ch", .cin = 512, .cout = 512, .t = 32, .h = 64, .w = 64 },
    .{ .name = "stageE res 128ch", .cin = 128, .cout = 128, .t = 64, .h = 128, .w = 128 },
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    log.info("platform: {s}", .{@tagName(platform.target)});

    // ---- Oracle: tiny 3D conv against a CPU reference -------------------
    // Validates the dimension-number wiring (batch/feature/spatial roles,
    // padding) before any timing is believed.
    {
        const C = 4;
        const T = 4;
        const HW = 6;
        const n_in = C * T * HW * HW;
        var x: [n_in]f16 = undefined;
        for (&x, 0..) |*v, i| v.* = @floatCast(@sin(@as(f32, @floatFromInt(i + 1)) * 0.19) * 0.5);
        var w: [C * C * 27]f16 = undefined;
        for (&w, 0..) |*v, i| v.* = @floatCast(@sin(@as(f32, @floatFromInt(i + 3)) * 0.23) * 0.3);

        var want: [n_in]f64 = undefined;
        for (0..C) |co| {
            for (0..T) |t| {
                for (0..HW) |h| {
                    for (0..HW) |ww| {
                        var acc: f64 = 0;
                        for (0..C) |ci| {
                            for (0..3) |kt| {
                                for (0..3) |kh| {
                                    for (0..3) |kw| {
                                        const it = @as(i64, @intCast(t)) + @as(i64, @intCast(kt)) - 1;
                                        const ih = @as(i64, @intCast(h)) + @as(i64, @intCast(kh)) - 1;
                                        const iw = @as(i64, @intCast(ww)) + @as(i64, @intCast(kw)) - 1;
                                        if (it < 0 or it >= T or ih < 0 or ih >= HW or iw < 0 or iw >= HW) continue;
                                        const xi = ((ci * T + @as(usize, @intCast(it))) * HW + @as(usize, @intCast(ih))) * HW + @as(usize, @intCast(iw));
                                        const wi = ((co * C + ci) * 3 + kt) * 9 + kh * 3 + kw;
                                        acc += @as(f64, @floatCast(x[xi])) * @as(f64, @floatCast(w[wi]));
                                    }
                                }
                            }
                        }
                        want[((co * T + t) * HW + h) * HW + ww] = acc;
                    }
                }
            }
        }

        const x_shape: zml.Shape = .init(.{ .b = 1, .c = C, .t = T, .h = HW, .w = HW }, .f16);
        const w_shape: zml.Shape = .init(.{ .o = C, .i = C, .kt = 3, .kh = 3, .kw = 3 }, .f16);
        var x_buf: zml.Buffer = try .fromSlice(io, platform, zml.Slice.initConst(x_shape, std.mem.sliceAsBytes(&x)), .replicated);
        defer x_buf.deinit();
        var w_buf: zml.Buffer = try .fromSlice(io, platform, zml.Slice.initConst(w_shape, std.mem.sliceAsBytes(&w)), .replicated);
        defer w_buf.deinit();
        const x_spec: zml.Tensor = .fromShape(x_shape);
        const w_spec: zml.Tensor = .fromShape(w_shape);
        var exe = try platform.compileFn(allocator, io, conv3dGraph, .{ x_spec, w_spec }, .{});

        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ x_buf, w_buf });
        exe.call(args, &results);
        var out: zml.Buffer = results.get(zml.Buffer);
        defer out.deinit();
        var out_slice = try out.toSliceAlloc(allocator, io);
        defer out_slice.free(allocator);
        const got = out_slice.constItems(f16);

        var err: f64 = 0;
        var ref: f64 = 0;
        for (got, 0..) |g, i| {
            const d = @as(f64, @floatCast(g)) - want[i];
            err += d * d;
            ref += want[i] * want[i];
        }
        const rms = @sqrt(err / (ref + 1e-20));
        log.info("oracle: conv3d vs CPU f64 rms {d:.5}", .{rms});
        if (rms > 5e-3) return error.Conv3dOracleMismatch;
    }

    // ---- The decoder-shape sweep ----------------------------------------
    log.info("E4: 3x3x3 convs at Conv-VAE decoder shapes, latent tile 8x16x16, f16, SAME padding", .{});
    for (CASES) |c| {
        const nx: usize = @intCast(c.cin * c.t * c.h * c.w);
        const nw: usize = @intCast(c.cout * c.cin * 27);
        const x = try allocator.alloc(f16, nx);
        defer allocator.free(x);
        const w = try allocator.alloc(f16, nw);
        defer allocator.free(w);
        for (x, 0..) |*v, i| v.* = @floatCast(@sin(@as(f32, @floatFromInt(i + 1)) * 0.013) * 0.5);
        for (w, 0..) |*v, i| v.* = @floatCast(@sin(@as(f32, @floatFromInt(i + 7)) * 0.017) * 0.1);

        const x_shape: zml.Shape = .init(.{ .b = 1, .c = c.cin, .t = c.t, .h = c.h, .w = c.w }, .f16);
        const w_shape: zml.Shape = .init(.{ .o = c.cout, .i = c.cin, .kt = 3, .kh = 3, .kw = 3 }, .f16);
        var x_buf: zml.Buffer = try .fromSlice(io, platform, zml.Slice.initConst(x_shape, std.mem.sliceAsBytes(x)), .replicated);
        defer x_buf.deinit();
        var w_buf: zml.Buffer = try .fromSlice(io, platform, zml.Slice.initConst(w_shape, std.mem.sliceAsBytes(w)), .replicated);
        defer w_buf.deinit();
        const x_spec: zml.Tensor = .fromShape(x_shape);
        const w_spec: zml.Tensor = .fromShape(w_shape);
        var exe = platform.compileFn(allocator, io, conv3dGraph, .{ x_spec, w_spec }, .{}) catch |err| {
            log.err("  {s}: COMPILE failed ({s})", .{ c.name, @errorName(err) });
            continue;
        };

        const n_out: usize = @intCast(c.cout * c.t * c.h * c.w);
        const out = try allocator.alloc(f16, n_out);
        defer allocator.free(out);

        const reps = 6;
        var times: [reps]u64 = undefined;
        var ok = true;
        for (0..reps + 1) |r| {
            const t0 = nowNs(io);
            var args = try exe.args(allocator);
            defer args.deinit(allocator);
            var results = try exe.results(allocator);
            defer results.deinit(allocator);
            args.set(.{ x_buf, w_buf });
            exe.call(args, &results);
            var o: zml.Buffer = results.get(zml.Buffer);
            defer o.deinit();
            // Sync via a 2-element readback probe would still transfer the
            // whole buffer through toSliceAlloc; accept the known ~0.5 GB/s
            // on the output once per rep and report it separately.
            var s = o.toSliceAlloc(allocator, io) catch |err| {
                log.err("  {s}: EXEC failed ({s})", .{ c.name, @errorName(err) });
                ok = false;
                break;
            };
            @memcpy(out, s.constItems(f16)[0..out.len]);
            s.free(allocator);
            if (r > 0) times[r - 1] = @intCast(nowNs(io) - t0); // rep 0 is warmup
        }
        if (!ok) continue;
        std.mem.sort(u64, &times, {}, std.sort.asc(u64));
        const p50 = times[reps / 2];
        const flop = 2.0 * @as(f64, @floatFromInt(c.cin * c.cout * 27 * c.t * c.h * c.w));
        const out_mb = @as(f64, @floatFromInt(n_out * 2)) / 1e6;
        const dl_ms = out_mb / 500.0 * 1000.0; // known ~0.5 GB/s readback estimate
        log.info("  {s} @ {d}x{d}x{d}: p50 {d:.2}ms incl. ~{d:.0}ms est. readback of {d:.0} MB -> >= {d:.2} TFLOP/s", .{
            c.name, c.t, c.h, c.w,
            ms(p50),
            dl_ms,
            out_mb,
            flop / 1e12 / (ms(p50) / 1e3),
        });
    }

    log.info("VAE SMOKE COMPLETE", .{});
}
