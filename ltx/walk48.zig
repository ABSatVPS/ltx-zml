//! The 48-block stage-walk (assembly checkpoint, notebook 2026-08-21
//! evening): ONE compiled block executable called 48 times, the ring
//! feeding each call's weights, the activation staying on device —
//! the production execution model, run for the first time.
//!
//! Pass 1 (f32, gated): engine x vs the torch f64 oracle at blocks
//! 7/15/23/31/39/47 — G-ASM-1..7, budget 2e-3, expected ~1e-5 at the
//! end if sub-linear growth holds. Pass 2 (int8 recipe, measured):
//! the fully-quantized walk vs pass 1's checkpoints and vs the oracle
//! — H-ASM-2, the compounding number that calibrates Phase 6.
const std = @import("std");
const zml = @import("zml");
const blk = @import("block.zig");
const ldr = @import("loader.zig");

const log = std.log;

pub const std_options: std.Options = .{ .log_level = .info };

const ROOT = "/home/adam/Development/Experiments/Video-Generation/.work";
const BUNDLE = ROOT ++ "/oracle_bundle";
const WALK_BUNDLE = ROOT ++ "/walk_bundle";
const N_BLOCKS: usize = 48;
const CHECKPOINTS = [_]usize{ 7, 15, 23, 31, 39, 47 };

const Tu: usize = @intCast(blk.T);
const Su: usize = @intCast(blk.S);
const Du: usize = @intCast(blk.D);
const Hu: usize = @intCast(blk.H);

fn nowNs(io: std.Io) i96 {
    const ts: std.Io.Timestamp = .now(io, .awake);
    return ts.toNanoseconds();
}

fn loadF32(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, expect: usize) ![]f32 {
    var buf: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir, name });
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
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

fn checkpointSlot(pos: usize) ?usize {
    for (CHECKPOINTS, 0..) |c, i| {
        if (c == pos) return i;
    }
    return null;
}

fn deinitBlockBufs(bufs: *zml.Bufferized(blk.Block)) void {
    inline for (blk.WEIGHT_SPECS) |spec| @field(bufs.*, spec.field).deinit();
}

fn deinitQuantOnly(qbufs: *zml.Bufferized(blk.QBlock)) void {
    inline for (ldr.Q8_SPECS) |qs| {
        @field(qbufs.*, qs.field).deinit();
        @field(qbufs.*, qs.field[0..2] ++ "s").deinit();
    }
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
    log.info("48 blobs open, layouts OK, max {d} MB", .{max_bytes / (1024 * 1024)});

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

    // ---- frozen inputs ----------------------------------------------------
    const x_h = try loadF32(allocator, io, BUNDLE, "in_x.bin", Tu * Du);
    const ctx_h = try loadF32(allocator, io, BUNDLE, "in_context.bin", Su * Du);
    const ts_h = try loadF32(allocator, io, BUNDLE, "in_timestep.bin", Tu * 9 * Du);
    const pts_h = try loadF32(allocator, io, BUNDLE, "in_prompt_timestep.bin", 2 * Du);
    const cos_bhtf = try loadF32(allocator, io, BUNDLE, "rope_cos.bin", Hu * Tu * 64);
    const sin_bhtf = try loadF32(allocator, io, BUNDLE, "rope_sin.bin", Hu * Tu * 64);
    const cos_thf = try allocator.alloc(f32, Tu * Hu * 64);
    const sin_thf = try allocator.alloc(f32, Tu * Hu * 64);
    for (0..Hu) |h| for (0..Tu) |t| for (0..64) |f| {
        cos_thf[(t * Hu + h) * 64 + f] = cos_bhtf[(h * Tu + t) * 64 + f];
        sin_thf[(t * Hu + h) * 64 + f] = sin_bhtf[(h * Tu + t) * 64 + f];
    };

    const x_shape = zml.Shape.init(.{ .t = blk.T, .i = blk.D }, .f32);
    const ctx_shape = zml.Shape.init(.{ .t = blk.S, .i = blk.D }, .f32);
    const ts_shape = zml.Shape.init(.{ .t = blk.T, .n = 9, .i = blk.D }, .f32);
    const pts_shape = zml.Shape.init(.{ .n = 2, .i = blk.D }, .f32);
    const pe_shape = zml.Shape.init(.{ .q = blk.T, .h = blk.H, .f = 64 }, .f32);

    var x_buf0: zml.Buffer = try .fromBytes(io, platform, x_shape, .replicated, std.mem.sliceAsBytes(x_h));
    var ctx_buf: zml.Buffer = try .fromBytes(io, platform, ctx_shape, .replicated, std.mem.sliceAsBytes(ctx_h));
    var ts_buf: zml.Buffer = try .fromBytes(io, platform, ts_shape, .replicated, std.mem.sliceAsBytes(ts_h));
    var pts_buf: zml.Buffer = try .fromBytes(io, platform, pts_shape, .replicated, std.mem.sliceAsBytes(pts_h));
    var cos_buf: zml.Buffer = try .fromBytes(io, platform, pe_shape, .replicated, std.mem.sliceAsBytes(cos_thf));
    var sin_buf: zml.Buffer = try .fromBytes(io, platform, pe_shape, .replicated, std.mem.sliceAsBytes(sin_thf));
    defer x_buf0.deinit();
    defer ctx_buf.deinit();
    defer ts_buf.deinit();
    defer pts_buf.deinit();
    defer cos_buf.deinit();
    defer sin_buf.deinit();

    const x_spec: zml.Tensor = .fromShape(x_shape);
    const ctx_spec: zml.Tensor = .fromShape(ctx_shape);
    const ts_spec: zml.Tensor = .fromShape(ts_shape);
    const pts_spec: zml.Tensor = .fromShape(pts_shape);
    const cos_spec: zml.Tensor = .fromShape(pe_shape);
    const sin_spec: zml.Tensor = .fromShape(pe_shape);

    const out_len = Tu * Du;
    const got = try allocator.alloc(f32, out_len);

    // Oracle checkpoints + pass-1 engine checkpoints (for H-ASM-2).
    var oracle_cp: [CHECKPOINTS.len][]f32 = undefined;
    var f32_cp: [CHECKPOINTS.len][]f32 = undefined;
    for (&oracle_cp, &f32_cp, CHECKPOINTS) |*ocp, *fcp, c| {
        var nb: [64]u8 = undefined;
        ocp.* = try loadF32(allocator, io, WALK_BUNDLE, try std.fmt.bufPrint(&nb, "walk_b{d}_out.bin", .{c}), out_len);
        fcp.* = try allocator.alloc(f32, out_len);
    }

    // ---- ONE block executable each for f32 and quant ----------------------
    const model = blk.makeBlockSpecs();
    const f32_method = comptime std.meta.stringToEnum(std.meta.DeclEnum(blk.Block), "blockOut").?;
    var f32_exe = try platform.compile(allocator, io, model, f32_method, .{ x_spec, ts_spec, ctx_spec, pts_spec, cos_spec, sin_spec }, .{});

    var qmodel: blk.QBlock = undefined;
    qmodel.base = blk.makeBlockSpecs();
    inline for (ldr.Q8_SPECS) |qs| {
        @field(qmodel, qs.field) = zml.Tensor.fromShape(zml.Shape.init(.{ .o = qs.dims[0], .i = qs.dims[1] }, .i8));
        @field(qmodel, qs.field[0..2] ++ "s") = zml.Tensor.fromShape(zml.Shape.init(.{ .o = qs.dims[0], .g = @divExact(qs.dims[1], 128) }, .f32));
    }
    const q_method = comptime std.meta.stringToEnum(std.meta.DeclEnum(blk.QBlock), "qBlockOutAll").?;
    var q_exe = try platform.compile(allocator, io, qmodel, q_method, .{ x_spec, ts_spec, ctx_spec, pts_spec, cos_spec, sin_spec }, .{});
    log.info("block executables compiled (f32 + quant), reused across all 48 calls", .{});

    var schedule: [N_BLOCKS]usize = undefined;
    for (&schedule, 0..) |*s, i| s.* = i;
    const verified = try allocator.alloc(bool, N_BLOCKS);

    var all_pass = true;

    // ======== PASS 1: f32, gated against the oracle ========================
    inline for (.{ false, true }) |quant| {
        @memset(verified, false);
        var ring: ldr.Ring = .{
            .slots = .{
                .{ .mem = @alignCast(ring_mem[0..max_bytes]) },
                .{ .mem = @alignCast(ring_mem[slot_stride .. slot_stride + max_bytes]) },
            },
            .blobs = blobs,
            .schedule = &schedule,
            .verify_first_touch = false, // pipeline digest-verified every blob
            .verified = verified,
        };
        try ring.staticCheck();
        var reader = try std.Thread.spawn(.{}, ldr.Ring.readerMain, .{ &ring, io });
        defer reader.join();
        defer ring.abort(io); // LIFO: abort runs before join

        var upload_ns: u64 = 0;
        var compute_ns: u64 = 0;
        const walk_t0 = nowNs(io);
        var x_cur = x_buf0;
        var args = if (quant) try q_exe.args(allocator) else try f32_exe.args(allocator);
        defer args.deinit(allocator);
        var results = if (quant) try q_exe.results(allocator) else try f32_exe.results(allocator);
        defer results.deinit(allocator);

        for (0..N_BLOCKS) |pos| {
            errdefer log.err("{s} walk failed at block {d}", .{ if (quant) "quant" else "f32", pos });
            const slot = try ring.acquire(io, pos);
            const blob = &ring.blobs[pos];
            const tu0 = nowNs(io);
            var bufs = try ldr.buildBlockBufs(io, platform, blob, slot.mem, @intCast(pos));
            var qbufs: zml.Bufferized(blk.QBlock) = undefined;
            if (quant) try ldr.buildQuantBufs(io, platform, blob, slot.mem, @intCast(pos), bufs, &qbufs);
            upload_ns += @intCast(nowNs(io) - tu0);
            ring.release(io, slot);

            if (quant) args.set(.{ qbufs, x_cur, ts_buf, ctx_buf, pts_buf, cos_buf, sin_buf }) else args.set(.{ bufs, x_cur, ts_buf, ctx_buf, pts_buf, cos_buf, sin_buf });
            const tc0 = nowNs(io);
            if (quant) q_exe.call(args, &results) else f32_exe.call(args, &results);
            var out: zml.Buffer = results.get(zml.Buffer);

            if (checkpointSlot(pos)) |ci| {
                var slice = try out.toSliceAlloc(allocator, io);
                defer slice.free(allocator);
                @memcpy(got, slice.constItems(f32)[0..out_len]);
                compute_ns += @intCast(nowNs(io) - tc0);
                if (!quant) {
                    @memcpy(f32_cp[ci], got);
                    const rms = relRms(got, oracle_cp[ci]);
                    const pass = rms <= 2e-3;
                    if (!pass) all_pass = false;
                    log.info("GATE asm walk_b{d}: rel-RMS {e:.3} vs oracle -> {s}", .{ pos, rms, if (pass) "PASS" else "FAIL" });
                } else {
                    log.info("H-ASM-2 quant walk_b{d}: rel-RMS {e:.3} vs f32 engine, {e:.3} vs oracle", .{
                        pos, relRms(got, f32_cp[ci]), relRms(got, oracle_cp[ci]),
                    });
                }
            } else {
                try out.await(io);
                compute_ns += @intCast(nowNs(io) - tc0);
            }

            if (pos > 0) x_cur.deinit();
            deinitBlockBufs(&bufs);
            if (quant) deinitQuantOnly(&qbufs);
            x_cur = out;
            if (pos % 8 == 0) log.info("{s} walk: block {d} done", .{ if (quant) "quant" else "f32", pos });
        }
        var x_final = x_cur;
        x_final.deinit();
        const wall_ns: u64 = @intCast(nowNs(io) - walk_t0);
        log.info("{s} walk: {d:.2} s wall, uploads {d:.2} s, compute+sync {d:.2} s, reader parked {d}x, consumer parked {d}x", .{
            if (quant) "quant" else "f32",
            @as(f64, @floatFromInt(wall_ns)) / 1e9,
            @as(f64, @floatFromInt(upload_ns)) / 1e9,
            @as(f64, @floatFromInt(compute_ns)) / 1e9,
            ring.reader_parks,
            ring.consumer_parks,
        });
    }

    if (all_pass) {
        log.info("48-BLOCK STAGE-WALK: ALL ORACLE GATES PASS", .{});
    } else {
        log.err("48-BLOCK STAGE-WALK: FAILURES ABOVE", .{});
        return error.WalkFailed;
    }
}
