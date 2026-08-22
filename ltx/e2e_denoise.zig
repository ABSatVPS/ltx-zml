//! The scheduler-driven multi-step denoising loop at harness T
//! (H-LOOP-1/2, notebook 2026-08-21 afternoon): stage 1's 8 distilled
//! steps, latent marching HOST-side through the bit-exact scheduler
//! (ltx/scheduler.zig), velocities from the card — patchify → adaln →
//! 48 ring-streamed blocks → tail per step, pts2 re-derived from each
//! step's sigma, the ring running its first repeating schedule
//! (48 blocks x 8 steps = 384 acquisitions).
//!
//! Gates per step: velocity vs the f64 oracle (budget 2e-3, expected
//! ~1e-4) and the marched latent vs the reference trajectory. Control:
//! the noised initial latent is recomputed from clean/noise/mask through
//! the scheduler's lerp chain and must match the oracle BITWISE.
const std = @import("std");
const zml = @import("zml");
const blk = @import("block.zig");
const core = @import("core_parts.zig");
const ldr = @import("loader.zig");
const sched = @import("scheduler.zig");

const log = std.log;

pub const std_options: std.Options = .{ .log_level = .info };

const ROOT = "/home/adam/Development/Experiments/Video-Generation/.work";
const BUNDLE = ROOT ++ "/oracle_bundle";
const DN_BUNDLE = ROOT ++ "/e2e_denoise_bundle";
const EXTRAS = ROOT ++ "/extras";
const N_BLOCKS: usize = 48;
const N_STEPS: usize = sched.SIGMAS_STAGE1.len - 1; // 8

const Tu: usize = @intCast(blk.T);
const Su: usize = @intCast(blk.S);
const Du: usize = @intCast(blk.D);
const Hu: usize = @intCast(blk.H);
const Cu: usize = @intCast(core.IN_CH);
const LAT: usize = Tu * Cu;

fn nowNs(io: std.Io) i96 {
    const ts: std.Io.Timestamp = .now(io, .awake);
    return ts.toNanoseconds();
}

fn readBin(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u8 {
    var buf: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir, name });
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
}

fn loadF32(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, expect: usize) ![]f32 {
    const raw = try readBin(allocator, io, dir, name);
    defer allocator.free(raw);
    if (raw.len != expect * 4) return error.BadSize;
    const out = try allocator.alloc(f32, expect);
    @memcpy(std.mem.sliceAsBytes(out), raw);
    return out;
}

fn relRms(got: []const f32, want: []const f32) f64 {
    var err: f64 = 0;
    var ref: f64 = 0;
    for (got, want) |g, w| {
        const d = @as(f64, g) - @as(f64, w);
        err += d * d;
        ref += @as(f64, w) * w;
    }
    return @sqrt(err / (ref + 1e-20));
}

fn deinitBlockBufs(bufs: *zml.Bufferized(blk.Block)) void {
    inline for (blk.WEIGHT_SPECS) |spec| @field(bufs.*, spec.field).deinit();
}

fn loadCoreBufs(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform) !zml.Bufferized(core.CoreParts) {
    var bufs: zml.Bufferized(core.CoreParts) = undefined;
    inline for (core.CORE_SPECS) |spec| {
        const raw = try readBin(allocator, io, EXTRAS, spec.file);
        defer allocator.free(raw);
        const sh = core.coreShape(spec);
        if (raw.len != sh.byteSize()) return error.BadSize;
        @field(bufs, spec.field) = try zml.Buffer.fromBytes(io, platform, sh, .replicated, raw);
    }
    return bufs;
}

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(1_000_000);
    const allocator = init.gpa;
    const io = init.io;

    // ---- blobs + ring -----------------------------------------------------
    const blobs = try allocator.alloc(ldr.Blob, N_BLOCKS);
    var max_bytes: usize = 0;
    for (blobs, 0..) |*b, i| {
        var dirbuf: [256]u8 = undefined;
        const dir = try std.fmt.bufPrint(&dirbuf, "{s}/block{d}", .{ ROOT, i });
        b.* = try ldr.Blob.open(allocator, io, dir);
        if (b.manifest.block != @as(i64, @intCast(i))) return error.WrongBlock;
        try b.checkLayout();
        max_bytes = @max(max_bytes, @as(usize, @intCast(b.manifest.total_bytes)));
    }
    log.info("48 blobs open, layouts OK", .{});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    log.info("platform: {s}", .{@tagName(platform.target)});

    const slot_stride = std.mem.alignForward(usize, max_bytes, 4096);
    const ring_mem = try std.heap.page_allocator.alignedAlloc(u8, .fromByteUnits(4096), ldr.HOST_SLOTS * slot_stride);
    defer std.heap.page_allocator.free(ring_mem);
    var pinned = true;
    platform.pjrt_client.dmaMap(platform.pjrt_api, ring_mem) catch |err| {
        pinned = false;
        log.warn("ring dmaMap failed ({t}) — pageable", .{err});
    };
    defer if (pinned) {
        platform.pjrt_client.dmaUnmap(platform.pjrt_api, ring_mem) catch {};
    };

    // ---- bundle inputs + host scheduler state -----------------------------
    const clean = try loadF32(allocator, io, DN_BUNDLE, "clean_v.bin", LAT);
    const mask = try loadF32(allocator, io, DN_BUNDLE, "mask_v.bin", Tu);
    const noise = try loadF32(allocator, io, DN_BUNDLE, "noise1_v.bin", LAT);
    const x0_want = try loadF32(allocator, io, DN_BUNDLE, "s1_x0.bin", LAT);
    const ctx_h = try loadF32(allocator, io, BUNDLE, "in_context.bin", Su * Du);
    const cos_bhtf = try loadF32(allocator, io, BUNDLE, "rope_cos.bin", Hu * Tu * 64);
    const sin_bhtf = try loadF32(allocator, io, BUNDLE, "rope_sin.bin", Hu * Tu * 64);
    const cos_thf = try allocator.alloc(f32, Tu * Hu * 64);
    const sin_thf = try allocator.alloc(f32, Tu * Hu * 64);
    for (0..Hu) |h| for (0..Tu) |t| for (0..64) |f| {
        cos_thf[(t * Hu + h) * 64 + f] = cos_bhtf[(h * Tu + t) * 64 + f];
        sin_thf[(t * Hu + h) * 64 + f] = sin_bhtf[(h * Tu + t) * 64 + f];
    };

    const latent = try allocator.alloc(f32, LAT);
    sched.noiseInit(latent, clean, noise, mask, Cu, 1.0);
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(latent), std.mem.sliceAsBytes(x0_want))) {
        log.err("CONTROL noised x0: engine lerp chain != oracle BITWISE — stopping", .{});
        return error.X0ControlFailed;
    }
    log.info("CONTROL noised x0: bitwise match (scheduler lerp chain composes)", .{});

    const t_h = try allocator.alloc(f32, Tu);
    const den = try allocator.alloc(f32, LAT);
    const post = try allocator.alloc(f32, LAT);
    const next = try allocator.alloc(f32, LAT);

    // ---- shapes/specs -----------------------------------------------------
    const lat_shape = zml.Shape.init(.{ .t = blk.T, .i = core.IN_CH }, .f32);
    const t_shape = zml.Shape.init(.{ .t = blk.T }, .f32);
    const sg_shape = zml.Shape.init(.{ .t = 1 }, .f32);
    const x_shape = zml.Shape.init(.{ .t = blk.T, .i = blk.D }, .f32);
    const ctx_shape = zml.Shape.init(.{ .t = blk.S, .i = blk.D }, .f32);
    const ts_shape = zml.Shape.init(.{ .k = blk.T, .n = 9, .i = blk.D }, .f32); // K=T table (identity index)
    const tidx_shape = zml.Shape.init(.{ .t = blk.T }, .i32);
    const pts_shape = zml.Shape.init(.{ .n = 2, .i = blk.D }, .f32);
    const pe_shape = zml.Shape.init(.{ .q = blk.T, .h = blk.H, .f = 64 }, .f32);

    var ctx_buf: zml.Buffer = try .fromBytes(io, platform, ctx_shape, .replicated, std.mem.sliceAsBytes(ctx_h));
    defer ctx_buf.deinit();
    var cos_buf: zml.Buffer = try .fromBytes(io, platform, pe_shape, .replicated, std.mem.sliceAsBytes(cos_thf));
    defer cos_buf.deinit();
    var sin_buf: zml.Buffer = try .fromBytes(io, platform, pe_shape, .replicated, std.mem.sliceAsBytes(sin_thf));
    defer sin_buf.deinit();
    const tidx_h = try allocator.alloc(i32, Tu);
    for (tidx_h, 0..) |*v, ii| v.* = @intCast(ii);
    var tidx_buf: zml.Buffer = try .fromBytes(io, platform, tidx_shape, .replicated, std.mem.sliceAsBytes(tidx_h));
    defer tidx_buf.deinit();

    const lat_spec: zml.Tensor = .fromShape(lat_shape);
    const t_spec: zml.Tensor = .fromShape(t_shape);
    const sg_spec: zml.Tensor = .fromShape(sg_shape);
    const x_spec: zml.Tensor = .fromShape(x_shape);
    const ctx_spec: zml.Tensor = .fromShape(ctx_shape);
    const ts_spec: zml.Tensor = .fromShape(ts_shape);
    const tidx_spec: zml.Tensor = .fromShape(tidx_shape);
    const pts_spec: zml.Tensor = .fromShape(pts_shape);
    const cos_spec: zml.Tensor = .fromShape(pe_shape);
    const sin_spec: zml.Tensor = .fromShape(pe_shape);

    // ---- executables, compiled once, reused all 8 steps -------------------
    const cmodel = core.makeCoreSpecs();
    const cbufs = try loadCoreBufs(allocator, io, platform);
    const patchify_m = comptime std.meta.stringToEnum(std.meta.DeclEnum(core.CoreParts), "patchify").?;
    const ts9r_m = comptime std.meta.stringToEnum(std.meta.DeclEnum(core.CoreParts), "ts9r").?;
    const pts2r_m = comptime std.meta.stringToEnum(std.meta.DeclEnum(core.CoreParts), "pts2r").?;
    const tail_m = comptime std.meta.stringToEnum(std.meta.DeclEnum(core.CoreParts), "tail").?;
    var patchify_exe = try platform.compile(allocator, io, cmodel, patchify_m, .{lat_spec}, .{});
    var ts9r_exe = try platform.compile(allocator, io, cmodel, ts9r_m, .{t_spec}, .{});
    var pts2r_exe = try platform.compile(allocator, io, cmodel, pts2r_m, .{sg_spec}, .{});
    var tail_exe = try platform.compile(allocator, io, cmodel, tail_m, .{ x_spec, t_spec }, .{});
    const model = blk.makeBlockSpecs();
    const blk_m = comptime std.meta.stringToEnum(std.meta.DeclEnum(blk.Block), "blockOut").?;
    var blk_exe = try platform.compile(allocator, io, model, blk_m, .{ x_spec, ts_spec, tidx_spec, ctx_spec, pts_spec, cos_spec, sin_spec }, .{});
    log.info("5 executables compiled, reused across {d} steps", .{N_STEPS});

    // ---- ring with the repeating 384-entry schedule -----------------------
    var schedule: [N_STEPS * N_BLOCKS]usize = undefined;
    for (&schedule, 0..) |*s, i| s.* = i % N_BLOCKS;
    const verified = try allocator.alloc(bool, N_BLOCKS);
    @memset(verified, false);
    var ring: ldr.Ring = .{
        .slots = .{
            .{ .mem = @alignCast(ring_mem[0..max_bytes]) },
            .{ .mem = @alignCast(ring_mem[slot_stride .. slot_stride + max_bytes]) },
        },
        .blobs = blobs,
        .schedule = &schedule,
        .verify_first_touch = false,
        .verified = verified,
    };
    try ring.staticCheck();
    var reader = try std.Thread.spawn(.{}, ldr.Ring.readerMain, .{ &ring, io });
    defer reader.join();
    defer ring.abort(io); // LIFO: abort runs before join

    var all_pass = true;
    var namebuf: [64]u8 = undefined;

    for (0..N_STEPS) |step| {
        errdefer log.err("denoise loop failed at step {d}", .{step});
        const step_t0 = nowNs(io);
        const sigma = sched.SIGMAS_STAGE1[step];
        const sigma_next = sched.SIGMAS_STAGE1[step + 1];

        // host conditioning; ts must equal the oracle's dump BITWISE
        sched.timesteps(t_h, mask, sigma);
        {
            const ts_want = try loadF32(allocator, io, DN_BUNDLE, try std.fmt.bufPrint(&namebuf, "s1_step{d}_ts.bin", .{step}), Tu);
            defer allocator.free(ts_want);
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(t_h), std.mem.sliceAsBytes(ts_want))) {
                log.err("step {d}: host timesteps != oracle bitwise", .{step});
                all_pass = false;
            }
        }

        var lat_buf: zml.Buffer = try .fromBytes(io, platform, lat_shape, .replicated, std.mem.sliceAsBytes(latent));
        defer lat_buf.deinit();
        var t_buf: zml.Buffer = try .fromBytes(io, platform, t_shape, .replicated, std.mem.sliceAsBytes(t_h));
        defer t_buf.deinit();
        var sg_buf: zml.Buffer = try .fromBytes(io, platform, sg_shape, .replicated, std.mem.sliceAsBytes(sched.SIGMAS_STAGE1[step .. step + 1]));
        defer sg_buf.deinit();

        var ts_buf: zml.Buffer = undefined;
        var pts_buf: zml.Buffer = undefined;
        var x_cur: zml.Buffer = undefined;
        {
            var args = try ts9r_exe.args(allocator);
            defer args.deinit(allocator);
            var results = try ts9r_exe.results(allocator);
            defer results.deinit(allocator);
            args.set(.{ cbufs, t_buf });
            ts9r_exe.call(args, &results);
            ts_buf = results.get(zml.Buffer);
        }
        {
            var args = try pts2r_exe.args(allocator);
            defer args.deinit(allocator);
            var results = try pts2r_exe.results(allocator);
            defer results.deinit(allocator);
            args.set(.{ cbufs, sg_buf });
            pts2r_exe.call(args, &results);
            pts_buf = results.get(zml.Buffer);
        }
        {
            var args = try patchify_exe.args(allocator);
            defer args.deinit(allocator);
            var results = try patchify_exe.results(allocator);
            defer results.deinit(allocator);
            args.set(.{ cbufs, lat_buf });
            patchify_exe.call(args, &results);
            x_cur = results.get(zml.Buffer);
        }
        defer ts_buf.deinit();
        defer pts_buf.deinit();

        {
            var args = try blk_exe.args(allocator);
            defer args.deinit(allocator);
            var results = try blk_exe.results(allocator);
            defer results.deinit(allocator);
            for (0..N_BLOCKS) |k| {
                const pos = step * N_BLOCKS + k;
                const slot = try ring.acquire(io, pos);
                const blob = &ring.blobs[schedule[pos]];
                var bufs = try ldr.buildBlockBufs(io, platform, blob, slot.mem, @intCast(schedule[pos]));
                ring.release(io, slot);
                args.set(.{ bufs, x_cur, ts_buf, tidx_buf, ctx_buf, pts_buf, cos_buf, sin_buf });
                blk_exe.call(args, &results);
                var out: zml.Buffer = results.get(zml.Buffer);
                try out.await(io);
                x_cur.deinit();
                deinitBlockBufs(&bufs);
                x_cur = out;
            }
        }

        var vel_buf: zml.Buffer = undefined;
        {
            var args = try tail_exe.args(allocator);
            defer args.deinit(allocator);
            var results = try tail_exe.results(allocator);
            defer results.deinit(allocator);
            args.set(.{ cbufs, x_cur, t_buf });
            tail_exe.call(args, &results);
            vel_buf = results.get(zml.Buffer);
        }
        x_cur.deinit();
        var vslice = try vel_buf.toSliceAlloc(allocator, io);
        const vel = vslice.constItems(f32)[0..LAT];

        {
            const want = try loadF32(allocator, io, DN_BUNDLE, try std.fmt.bufPrint(&namebuf, "s1_step{d}_vel.bin", .{step}), LAT);
            defer allocator.free(want);
            const rms = relRms(vel, want);
            const pass = rms <= 2e-3;
            if (!pass) all_pass = false;
            log.info("GATE step{d} velocity (sigma {d:.6}): rel-RMS {e:.3} -> {s}", .{ step, sigma, rms, if (pass) "PASS" else "FAIL" });
        }

        // host march: to_denoised -> post_process -> euler step
        sched.toDenoised(den, latent, vel, t_h, Cu);
        vslice.free(allocator);
        vel_buf.deinit();
        sched.postProcess(post, den, clean, mask, Cu);
        sched.eulerStep(next, latent, post, sigma, sigma_next);
        @memcpy(latent, next);

        {
            const want = try loadF32(allocator, io, DN_BUNDLE, try std.fmt.bufPrint(&namebuf, "s1_step{d}_next.bin", .{step}), LAT);
            defer allocator.free(want);
            const rms = relRms(latent, want);
            const pass = rms <= 2e-3;
            if (!pass) all_pass = false;
            log.info("GATE step{d} latent: rel-RMS {e:.3} vs reference trajectory -> {s}", .{ step, rms, if (pass) "PASS" else "FAIL" });
        }
        log.info("step {d}: {d:.2} s", .{ step, @as(f64, @floatFromInt(@as(u64, @intCast(nowNs(io) - step_t0)))) / 1e9 });
    }

    if (all_pass) {
        log.info("MULTI-STEP DENOISING AT HARNESS T: ALL GATES PASS ({d} steps)", .{N_STEPS});
    } else {
        log.err("MULTI-STEP DENOISING: FAILURES ABOVE", .{});
        return error.DenoiseLoopFailed;
    }
}
