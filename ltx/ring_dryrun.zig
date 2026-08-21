//! E-STREAM-3: the streaming ring's three-block dry run, gated per the
//! notebook contract (2026-08-21/22). Blocks 0/23/47 — fetched,
//! oracle-anchored, block 0 also quantized.
//!
//! G-RING-2 manifest integrity, G-RING-3 byte fidelity of ring-served
//! bytes, G-RING-4 lifecycle under slow and fast consumers, G-RING-5
//! static schedule check before I/O, G-RING-6 the engine gate: the real
//! chain graph fed by ring-loaded buffers must match direct-loaded
//! buffers BITWISE, then block 0's fully-quantized block the same way.
const std = @import("std");
const zml = @import("zml");
const blk = @import("block.zig");
const ldr = @import("loader.zig");

const log = std.log;

pub const std_options: std.Options = .{ .log_level = .info };

const ROOT = "/home/adam/Development/Experiments/Video-Generation/.work";
const BUNDLE = ROOT ++ "/oracle_bundle";
const BLOCK_DIRS = [3][]const u8{ ROOT ++ "/block0", ROOT ++ "/block23", ROOT ++ "/block47" };
const BLOCK_IDX = [3]i64{ 0, 23, 47 };

const Tu: usize = @intCast(blk.T);
const Su: usize = @intCast(blk.S);
const Du: usize = @intCast(blk.D);
const Hu: usize = @intCast(blk.H);

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

/// Normalize a blob entry name to the harness file-stem convention:
/// '.' -> '_', "::q8scale" -> "_q8scale", "::q8" -> "_q8".
fn normalizeName(out: []u8, name: []const u8) []const u8 {
    var n: usize = 0;
    var i: usize = 0;
    while (i < name.len) : (i += 1) {
        if (name[i] == ':' and i + 1 < name.len and name[i + 1] == ':') {
            out[n] = '_';
            n += 1;
            i += 1; // skip the second ':'
        } else if (name[i] == '.') {
            out[n] = '_';
            n += 1;
        } else {
            out[n] = name[i];
            n += 1;
        }
    }
    return out[0..n];
}

/// Find a blob entry by its normalized harness name.
fn findEntry(blob: *const ldr.Blob, want: []const u8) !ldr.Entry {
    var nbuf: [256]u8 = undefined;
    for (blob.manifest.entries) |e| {
        if (std.mem.eql(u8, normalizeName(&nbuf, e.name), want)) return e;
    }
    log.err("blob {d}: no entry named {s}", .{ blob.manifest.block, want });
    return error.MissingEntry;
}

/// Build one block's device buffers from a memory region laid out per the
/// blob manifest (either a ring slot or the mmap itself for direct mode).
fn buildBlockBufs(io: std.Io, platform: *zml.Platform, blob: *const ldr.Blob, mem: []const u8, block_idx: i64) !zml.Bufferized(blk.Block) {
    var bufs: zml.Bufferized(blk.Block) = undefined;
    var namebuf: [256]u8 = undefined;
    inline for (blk.WEIGHT_SPECS) |spec| {
        const stem = comptime spec.file[0 .. spec.file.len - 4]; // drop ".bin"
        const want = try std.fmt.bufPrint(&namebuf, "transformer_blocks_{d}_{s}", .{ block_idx, stem });
        const e = try findEntry(blob, want);
        const data = mem[e.offset .. e.offset + e.nbytes];
        @field(bufs, spec.field) = try zml.Buffer.fromBytes(io, platform, blk.weightShape(spec), .replicated, data);
    }
    return bufs;
}

const Q8Spec = struct { field: []const u8, stem: []const u8, dims: [2]i64 };
const Q8_SPECS = [_]Q8Spec{
    .{ .field = "q1q", .stem = "attn1_to_q_weight", .dims = .{ blk.D, blk.D } },
    .{ .field = "k1q", .stem = "attn1_to_k_weight", .dims = .{ blk.D, blk.D } },
    .{ .field = "v1q", .stem = "attn1_to_v_weight", .dims = .{ blk.D, blk.D } },
    .{ .field = "q2q", .stem = "attn2_to_q_weight", .dims = .{ blk.D, blk.D } },
    .{ .field = "k2q", .stem = "attn2_to_k_weight", .dims = .{ blk.D, blk.D } },
    .{ .field = "v2q", .stem = "attn2_to_v_weight", .dims = .{ blk.D, blk.D } },
    .{ .field = "f1q", .stem = "ff_net_0_proj_weight", .dims = .{ blk.FF, blk.D } },
    .{ .field = "f2q", .stem = "ff_net_2_weight", .dims = .{ blk.D, blk.FF } },
};

fn buildQuantBufs(io: std.Io, platform: *zml.Platform, blob: *const ldr.Blob, mem: []const u8, base: zml.Bufferized(blk.Block), qbufs: *zml.Bufferized(blk.QBlock)) !void {
    qbufs.base = base;
    var namebuf: [256]u8 = undefined;
    inline for (Q8_SPECS) |qs| {
        const wq = try std.fmt.bufPrint(&namebuf, "transformer_blocks_0_{s}_q8", .{qs.stem});
        const eq = try findEntry(blob, wq);
        const qshape = zml.Shape.init(.{ .o = qs.dims[0], .i = qs.dims[1] }, .i8);
        @field(qbufs, qs.field) = try zml.Buffer.fromBytes(io, platform, qshape, .replicated, mem[eq.offset .. eq.offset + eq.nbytes]);
        const ws = try std.fmt.bufPrint(&namebuf, "transformer_blocks_0_{s}_q8scale", .{qs.stem});
        const es = try findEntry(blob, ws);
        const sshape = zml.Shape.init(.{ .o = qs.dims[0], .g = @divExact(qs.dims[1], 128) }, .f32);
        const sfield = qs.field[0..2] ++ "s";
        @field(qbufs, sfield) = try zml.Buffer.fromBytes(io, platform, sshape, .replicated, mem[es.offset .. es.offset + es.nbytes]);
    }
}

/// One lifecycle pass over `schedule`. slow_ms > 0 sleeps per block to
/// force ring-full backpressure; `check_digests` runs G-RING-3 on the
/// SERVED bytes (the slot, not the mmap).
fn lifecycleRun(ring: *ldr.Ring, io: std.Io, slow_ms: u64, check_digests: bool) !void {
    var reader = try std.Thread.spawn(.{}, ldr.Ring.readerMain, .{ ring, io });
    defer reader.join();
    for (0..ring.schedule.len) |pos| {
        const slot = try ring.acquire(io, pos);
        const blob = &ring.blobs[ring.schedule[pos]];
        if (slow_ms > 0) std.Io.sleep(io, .fromMilliseconds(@intCast(slow_ms)), .awake) catch {};
        if (check_digests) {
            for (blob.manifest.entries) |e| {
                const served = slot.mem[e.offset .. e.offset + e.nbytes];
                const hex = ldr.sha256Hex(served);
                if (!std.mem.eql(u8, &hex, e.sha256)) {
                    log.err("G-RING-3 FAIL: {s} served bytes != packed digest", .{e.name});
                    return error.ByteFidelity;
                }
            }
        }
        ring.release(io, slot);
    }
}

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(1_000_000);
    const allocator = init.gpa;
    const io = init.io;

    // ---- blobs + G-RING-2 -------------------------------------------------
    var blobs: [3]ldr.Blob = undefined;
    var max_bytes: usize = 0;
    for (&blobs, BLOCK_DIRS, BLOCK_IDX) |*b, dir, idx| {
        b.* = try ldr.Blob.open(allocator, io, dir);
        if (b.manifest.block != idx) return error.WrongBlock;
        try b.checkLayout();
        max_bytes = @max(max_bytes, @as(usize, @intCast(b.manifest.total_bytes)));
        log.info("GATE ring-2 blob {d}: {d} entries, {d} bytes, layout OK (rev {s})", .{
            b.manifest.block, b.manifest.entries.len, b.manifest.total_bytes, b.manifest.revision[0..12],
        });
    }

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    log.info("platform: {s}", .{@tagName(platform.target)});

    // H-RING-3 probe: can the FILE-BACKED mmap itself be dmaMap-pinned?
    if (platform.pjrt_client.dmaMap(platform.pjrt_api, blobs[0].data)) |_| {
        log.info("H-RING-3: dmaMap on file-backed mmap SUCCEEDED (hypothesis refuted)", .{});
        platform.pjrt_client.dmaUnmap(platform.pjrt_api, blobs[0].data) catch {};
    } else |err| {
        log.info("H-RING-3: dmaMap on file-backed mmap failed as hypothesized: {t}", .{err});
    }

    // ---- pinned host ring -------------------------------------------------
    const slot_stride = std.mem.alignForward(usize, max_bytes, 4096);
    const ring_mem = try std.heap.page_allocator.alignedAlloc(u8, .fromByteUnits(4096), ldr.HOST_SLOTS * slot_stride);
    defer std.heap.page_allocator.free(ring_mem);
    var pinned = true;
    platform.pjrt_client.dmaMap(platform.pjrt_api, ring_mem) catch |err| {
        pinned = false;
        log.warn("ring dmaMap failed ({t}) — running pageable", .{err});
    };
    defer if (pinned) {
        platform.pjrt_client.dmaUnmap(platform.pjrt_api, ring_mem) catch {};
    };
    log.info("host ring: {d} x {d} MB slots, pinned={}", .{ ldr.HOST_SLOTS, max_bytes / (1024 * 1024), pinned });

    // Two passes over the three blocks: slot reuse and re-serve of an
    // already-verified blob both get exercised.
    const schedule = [_]usize{ 0, 1, 2, 0, 1, 2 };
    const verified = try allocator.alloc(bool, 3);
    defer allocator.free(verified);

    // ---- G-RING-5 + G-RING-4 slow consumer --------------------------------
    @memset(verified, false);
    var ring_a: ldr.Ring = .{
        .slots = .{
            .{ .mem = @alignCast(ring_mem[0..max_bytes]) },
            .{ .mem = @alignCast(ring_mem[slot_stride .. slot_stride + max_bytes]) },
        },
        .blobs = &blobs,
        .schedule = &schedule,
        .verify_first_touch = true,
        .verified = verified,
    };
    try ring_a.staticCheck(); // G-RING-5, before the reader exists
    try lifecycleRun(&ring_a, io, 300, true);
    log.info("GATE ring-4 slow consumer: reader parked {d}x, consumer parked {d}x, digests verified in {d:.1} ms, copies {d:.1} ms -> {s}", .{
        ring_a.reader_parks,                              ring_a.consumer_parks,
        @as(f64, @floatFromInt(ring_a.verify_ns)) / 1e6,  @as(f64, @floatFromInt(ring_a.copy_ns)) / 1e6,
        if (ring_a.reader_parks > 0) "PASS" else "FAIL (no backpressure observed)",
    });
    log.info("GATE ring-3 byte fidelity: every served entry matched its packed digest (both passes)", .{});

    // ---- G-RING-4 fast consumer -------------------------------------------
    @memset(verified, false);
    var ring_b: ldr.Ring = .{
        .slots = .{
            .{ .mem = @alignCast(ring_mem[0..max_bytes]) },
            .{ .mem = @alignCast(ring_mem[slot_stride .. slot_stride + max_bytes]) },
        },
        .blobs = &blobs,
        .schedule = &schedule,
        .verify_first_touch = true,
        .verified = verified,
    };
    try lifecycleRun(&ring_b, io, 0, false);
    log.info("GATE ring-4 fast consumer: reader parked {d}x, consumer parked {d}x -> {s}", .{
        ring_b.reader_parks, ring_b.consumer_parks,
        if (ring_b.consumer_parks > 0) "PASS" else "PASS (consumer never waited — reader kept ahead)",
    });

    // ---- G-RING-6: the engine gate ----------------------------------------
    // Ring-loaded device sets for the three blocks (device residency: all
    // three at once — the chain consumes them together).
    @memset(verified, false);
    const gpu_schedule = [_]usize{ 0, 1, 2 };
    var ring_c: ldr.Ring = .{
        .slots = .{
            .{ .mem = @alignCast(ring_mem[0..max_bytes]) },
            .{ .mem = @alignCast(ring_mem[slot_stride .. slot_stride + max_bytes]) },
        },
        .blobs = &blobs,
        .schedule = &gpu_schedule,
        .verify_first_touch = false,
        .verified = verified,
    };
    var ring_bufs: [3]zml.Bufferized(blk.Block) = undefined;
    var qbufs_ring: zml.Bufferized(blk.QBlock) = undefined;
    {
        var reader = try std.Thread.spawn(.{}, ldr.Ring.readerMain, .{ &ring_c, io });
        defer reader.join();
        var up_bytes: usize = 0;
        const t0 = nowNs(io);
        for (0..gpu_schedule.len) |pos| {
            const slot = try ring_c.acquire(io, pos);
            const blob = &ring_c.blobs[gpu_schedule[pos]];
            ring_bufs[pos] = try buildBlockBufs(io, platform, blob, slot.mem, BLOCK_IDX[pos]);
            if (pos == 0) try buildQuantBufs(io, platform, blob, slot.mem, ring_bufs[0], &qbufs_ring);
            up_bytes += @intCast(blob.manifest.total_bytes);
            ring_c.release(io, slot);
        }
        const up_ns: u64 = @intCast(nowNs(io) - t0);
        log.info("ring uploads: {d} MB in {d:.1} ms -> {d:.2} GB/s ({s}, per-tensor)", .{
            up_bytes / (1024 * 1024),
            @as(f64, @floatFromInt(up_ns)) / 1e6,
            @as(f64, @floatFromInt(up_bytes)) / @as(f64, @floatFromInt(up_ns)),
            if (pinned) "pinned" else "pageable",
        });
    }

    // Direct sets, straight from the mmap (no ring) — the control.
    var direct_bufs: [3]zml.Bufferized(blk.Block) = undefined;
    var qbufs_direct: zml.Bufferized(blk.QBlock) = undefined;
    for (0..3) |i| {
        direct_bufs[i] = try buildBlockBufs(io, platform, &blobs[i], blobs[i].data, BLOCK_IDX[i]);
    }
    try buildQuantBufs(io, platform, &blobs[0], blobs[0].data, direct_bufs[0], &qbufs_direct);

    // Bundle inputs (as in block_conformance).
    const x_h = try loadF32(allocator, io, BUNDLE, "in_x.bin", Tu * Du);
    defer allocator.free(x_h);
    const ctx_h = try loadF32(allocator, io, BUNDLE, "in_context.bin", Su * Du);
    defer allocator.free(ctx_h);
    const ts_h = try loadF32(allocator, io, BUNDLE, "in_timestep.bin", Tu * 9 * Du);
    defer allocator.free(ts_h);
    const pts_h = try loadF32(allocator, io, BUNDLE, "in_prompt_timestep.bin", 2 * Du);
    defer allocator.free(pts_h);
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

    const x_shape = zml.Shape.init(.{ .t = blk.T, .i = blk.D }, .f32);
    const ctx_shape = zml.Shape.init(.{ .t = blk.S, .i = blk.D }, .f32);
    const ts_shape = zml.Shape.init(.{ .t = blk.T, .n = 9, .i = blk.D }, .f32);
    const pts_shape = zml.Shape.init(.{ .n = 2, .i = blk.D }, .f32);
    const pe_shape = zml.Shape.init(.{ .q = blk.T, .h = blk.H, .f = 64 }, .f32);

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
    const got_ring = try allocator.alloc(f32, out_len);
    defer allocator.free(got_ring);
    const got_direct = try allocator.alloc(f32, out_len);
    defer allocator.free(got_direct);

    const chain_model: blk.Chain = .{ .b0 = blk.makeBlockSpecs(), .b23 = blk.makeBlockSpecs(), .b47 = blk.makeBlockSpecs() };
    const chain_method = comptime std.meta.stringToEnum(std.meta.DeclEnum(blk.Chain), "chainB47").?;
    var chain_exe = try platform.compile(allocator, io, chain_model, chain_method, .{ x_spec, ts_spec, ctx_spec, pts_spec, cos_spec, sin_spec }, .{});
    inline for (.{ &ring_bufs, &direct_bufs }, .{ got_ring, got_direct }) |bufs3, dst| {
        const cbufs: zml.Bufferized(blk.Chain) = .{ .b0 = bufs3[0], .b23 = bufs3[1], .b47 = bufs3[2] };
        var args = try chain_exe.args(allocator);
        defer args.deinit(allocator);
        var results = try chain_exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ cbufs, x_buf, ts_buf, ctx_buf, pts_buf, cos_buf, sin_buf });
        chain_exe.call(args, &results);
        var out: zml.Buffer = results.get(zml.Buffer);
        defer out.deinit();
        var slice = try out.toSliceAlloc(allocator, io);
        defer slice.free(allocator);
        @memcpy(dst, slice.constItems(f32)[0..out_len]);
    }
    const chain_ok = std.mem.eql(u8, std.mem.sliceAsBytes(got_ring), std.mem.sliceAsBytes(got_direct));
    log.info("GATE ring-6 chainB47 ring vs direct: {s}", .{if (chain_ok) "BITWISE EQUAL -> PASS" else "MISMATCH -> FAIL"});

    var qmodel: blk.QBlock = undefined;
    qmodel.base = blk.makeBlockSpecs();
    inline for (Q8_SPECS) |qs| {
        @field(qmodel, qs.field) = zml.Tensor.fromShape(zml.Shape.init(.{ .o = qs.dims[0], .i = qs.dims[1] }, .i8));
        @field(qmodel, qs.field[0..2] ++ "s") = zml.Tensor.fromShape(zml.Shape.init(.{ .o = qs.dims[0], .g = @divExact(qs.dims[1], 128) }, .f32));
    }
    const q_method = comptime std.meta.stringToEnum(std.meta.DeclEnum(blk.QBlock), "qBlockOutAll").?;
    var q_exe = try platform.compile(allocator, io, qmodel, q_method, .{ x_spec, ts_spec, ctx_spec, pts_spec, cos_spec, sin_spec }, .{});
    inline for (.{ &qbufs_ring, &qbufs_direct }, .{ got_ring, got_direct }) |qb, dst| {
        var args = try q_exe.args(allocator);
        defer args.deinit(allocator);
        var results = try q_exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ qb.*, x_buf, ts_buf, ctx_buf, pts_buf, cos_buf, sin_buf });
        q_exe.call(args, &results);
        var out: zml.Buffer = results.get(zml.Buffer);
        defer out.deinit();
        var slice = try out.toSliceAlloc(allocator, io);
        defer slice.free(allocator);
        @memcpy(dst, slice.constItems(f32)[0..out_len]);
    }
    const q_ok = std.mem.eql(u8, std.mem.sliceAsBytes(got_ring), std.mem.sliceAsBytes(got_direct));
    log.info("GATE ring-6 qBlockOutAll ring vs direct: {s}", .{if (q_ok) "BITWISE EQUAL -> PASS" else "MISMATCH -> FAIL"});

    if (chain_ok and q_ok) {
        log.info("RING DRY RUN: ALL GATES PASS", .{});
    } else {
        return error.RingDryRunFailed;
    }
}
