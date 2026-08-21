//! The E2E composition at harness T=64 (H-CORE-5, notebook 2026-08-21
//! late morning): the full video-only forward — patchify → adaln-driven
//! 48 ring-streamed block calls → LayerNorm tail — every activation and
//! conditioning tensor produced ON DEVICE by the gated core parts, gated
//! against the f64 composition oracle (tools/make_e2e_bundle.py).
//!
//! Gates: e2e_x0 (after patchify), e2e_b23/e2e_b47 (walk checkpoints),
//! e2e_out (the velocity out of proj_out). Budget 2e-3, expectation
//! ~1.5e-5 carried from the stage-walk.
const std = @import("std");
const zml = @import("zml");
const blk = @import("block.zig");
const core = @import("core_parts.zig");
const ldr = @import("loader.zig");

const log = std.log;

pub const std_options: std.Options = .{ .log_level = .info };

const ROOT = "/home/adam/Development/Experiments/Video-Generation/.work";
const BUNDLE = ROOT ++ "/oracle_bundle";
const CORE_BUNDLE = ROOT ++ "/core_bundle";
const E2E_BUNDLE = ROOT ++ "/e2e_bundle";
const EXTRAS = ROOT ++ "/extras";
const N_BLOCKS: usize = 48;

const Tu: usize = @intCast(blk.T);
const Su: usize = @intCast(blk.S);
const Du: usize = @intCast(blk.D);
const Hu: usize = @intCast(blk.H);
const Cu: usize = @intCast(core.IN_CH);

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

fn gateBuffer(allocator: std.mem.Allocator, io: std.Io, name: []const u8, buf: *zml.Buffer, oracle_file: []const u8, len: usize) !bool {
    var slice = try buf.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    const want = try loadF32(allocator, io, E2E_BUNDLE, oracle_file, len);
    defer allocator.free(want);
    const rms = relRms(slice.constItems(f32)[0..len], want);
    const pass = rms <= 2e-3;
    log.info("GATE e2e {s}: rel-RMS {e:.3} vs oracle -> {s}", .{ name, rms, if (pass) "PASS" else "FAIL" });
    return pass;
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

    // ---- all 48 blobs -----------------------------------------------------
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

    // ---- host inputs ------------------------------------------------------
    const lat_h = try loadF32(allocator, io, CORE_BUNDLE, "in_latent.bin", Tu * Cu);
    const t_h = try loadF32(allocator, io, CORE_BUNDLE, "in_timesteps.bin", Tu);
    const sg_h = try loadF32(allocator, io, CORE_BUNDLE, "in_sigma.bin", 1);
    const ctx_h = try loadF32(allocator, io, BUNDLE, "in_context.bin", Su * Du);
    const cos_bhtf = try loadF32(allocator, io, BUNDLE, "rope_cos.bin", Hu * Tu * 64);
    const sin_bhtf = try loadF32(allocator, io, BUNDLE, "rope_sin.bin", Hu * Tu * 64);
    const cos_thf = try allocator.alloc(f32, Tu * Hu * 64);
    const sin_thf = try allocator.alloc(f32, Tu * Hu * 64);
    for (0..Hu) |h| for (0..Tu) |t| for (0..64) |f| {
        cos_thf[(t * Hu + h) * 64 + f] = cos_bhtf[(h * Tu + t) * 64 + f];
        sin_thf[(t * Hu + h) * 64 + f] = sin_bhtf[(h * Tu + t) * 64 + f];
    };

    const lat_shape = zml.Shape.init(.{ .t = blk.T, .i = core.IN_CH }, .f32);
    const t_shape = zml.Shape.init(.{ .t = blk.T }, .f32);
    const sg_shape = zml.Shape.init(.{ .t = 1 }, .f32);
    const x_shape = zml.Shape.init(.{ .t = blk.T, .i = blk.D }, .f32);
    const ctx_shape = zml.Shape.init(.{ .t = blk.S, .i = blk.D }, .f32);
    const ts_shape = zml.Shape.init(.{ .t = blk.T, .n = 9, .i = blk.D }, .f32);
    const pts_shape = zml.Shape.init(.{ .n = 2, .i = blk.D }, .f32);
    const pe_shape = zml.Shape.init(.{ .q = blk.T, .h = blk.H, .f = 64 }, .f32);

    var lat_buf: zml.Buffer = try .fromBytes(io, platform, lat_shape, .replicated, std.mem.sliceAsBytes(lat_h));
    defer lat_buf.deinit();
    var t_buf: zml.Buffer = try .fromBytes(io, platform, t_shape, .replicated, std.mem.sliceAsBytes(t_h));
    defer t_buf.deinit();
    var sg_buf: zml.Buffer = try .fromBytes(io, platform, sg_shape, .replicated, std.mem.sliceAsBytes(sg_h));
    defer sg_buf.deinit();
    var ctx_buf: zml.Buffer = try .fromBytes(io, platform, ctx_shape, .replicated, std.mem.sliceAsBytes(ctx_h));
    defer ctx_buf.deinit();
    var cos_buf: zml.Buffer = try .fromBytes(io, platform, pe_shape, .replicated, std.mem.sliceAsBytes(cos_thf));
    defer cos_buf.deinit();
    var sin_buf: zml.Buffer = try .fromBytes(io, platform, pe_shape, .replicated, std.mem.sliceAsBytes(sin_thf));
    defer sin_buf.deinit();

    const lat_spec: zml.Tensor = .fromShape(lat_shape);
    const t_spec: zml.Tensor = .fromShape(t_shape);
    const sg_spec: zml.Tensor = .fromShape(sg_shape);
    const x_spec: zml.Tensor = .fromShape(x_shape);
    const ctx_spec: zml.Tensor = .fromShape(ctx_shape);
    const ts_spec: zml.Tensor = .fromShape(ts_shape);
    const pts_spec: zml.Tensor = .fromShape(pts_shape);
    const cos_spec: zml.Tensor = .fromShape(pe_shape);
    const sin_spec: zml.Tensor = .fromShape(pe_shape);

    // ---- executables: 4 core parts + ONE block, all compiled up front -----
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
    var blk_exe = try platform.compile(allocator, io, model, blk_m, .{ x_spec, ts_spec, ctx_spec, pts_spec, cos_spec, sin_spec }, .{});
    log.info("5 executables compiled (patchify, ts9r, pts2r, tail, block)", .{});

    var all_pass = true;

    // ---- conditioning + input projections, on device ----------------------
    var ts_buf: zml.Buffer = undefined;
    var pts_buf: zml.Buffer = undefined;
    var x0_buf: zml.Buffer = undefined;
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
        x0_buf = results.get(zml.Buffer);
    }
    defer ts_buf.deinit();
    defer pts_buf.deinit();
    if (!try gateBuffer(allocator, io, "x0", &x0_buf, "e2e_x0.bin", Tu * Du)) all_pass = false;

    // ---- the 48-block ring-streamed walk ----------------------------------
    var schedule: [N_BLOCKS]usize = undefined;
    for (&schedule, 0..) |*s, i| s.* = i;
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

    const walk_t0 = nowNs(io);
    var x_cur = x0_buf;
    {
        var args = try blk_exe.args(allocator);
        defer args.deinit(allocator);
        var results = try blk_exe.results(allocator);
        defer results.deinit(allocator);

        for (0..N_BLOCKS) |pos| {
            errdefer log.err("e2e walk failed at block {d}", .{pos});
            const slot = try ring.acquire(io, pos);
            const blob = &ring.blobs[pos];
            var bufs = try ldr.buildBlockBufs(io, platform, blob, slot.mem, @intCast(pos));
            ring.release(io, slot);

            args.set(.{ bufs, x_cur, ts_buf, ctx_buf, pts_buf, cos_buf, sin_buf });
            blk_exe.call(args, &results);
            var out: zml.Buffer = results.get(zml.Buffer);

            if (pos == 23 or pos == 47) {
                var nb: [64]u8 = undefined;
                const oracle_file = try std.fmt.bufPrint(&nb, "e2e_b{d}_out.bin", .{pos});
                var nb2: [32]u8 = undefined;
                const gname = try std.fmt.bufPrint(&nb2, "b{d}", .{pos});
                if (!try gateBuffer(allocator, io, gname, &out, oracle_file, Tu * Du)) all_pass = false;
            } else {
                try out.await(io);
            }

            x_cur.deinit();
            deinitBlockBufs(&bufs);
            x_cur = out;
            if (pos % 8 == 0) log.info("e2e walk: block {d} done", .{pos});
        }
    }
    const walk_ns: u64 = @intCast(nowNs(io) - walk_t0);

    // ---- the output tail --------------------------------------------------
    var out_buf: zml.Buffer = undefined;
    {
        var args = try tail_exe.args(allocator);
        defer args.deinit(allocator);
        var results = try tail_exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ cbufs, x_cur, t_buf });
        tail_exe.call(args, &results);
        out_buf = results.get(zml.Buffer);
    }
    x_cur.deinit();
    if (!try gateBuffer(allocator, io, "out", &out_buf, "e2e_out.bin", Tu * Cu)) all_pass = false;
    out_buf.deinit();

    log.info("e2e walk: {d:.2} s wall for 48 blocks at T={d}", .{ @as(f64, @floatFromInt(walk_ns)) / 1e9, blk.T });

    if (all_pass) {
        log.info("E2E COMPOSITION AT HARNESS T: ALL GATES PASS", .{});
    } else {
        log.err("E2E COMPOSITION: FAILURES ABOVE", .{});
        return error.E2eFailed;
    }
}
