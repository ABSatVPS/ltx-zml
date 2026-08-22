//! The production-length pass (H-PROD-1/2, notebook 2026-08-21
//! afternoon): ONE full velocity forward at T=28,672 — patchify →
//! adaln-driven 48 ring-streamed wBlockOut calls (E3w while attention;
//! dense scores cannot exist at this length) → LayerNorm tail. Fits,
//! runs, timed. Numerical authority lives in the gated harness suites;
//! this run's own checks are structural: no NaN/Inf, finite magnitudes.
const std = @import("std");
const zml = @import("zml");
const blk = @import("block.zig");
const core = @import("core_parts.zig");
const ldr = @import("loader.zig");

const log = std.log;

pub const std_options: std.Options = .{ .log_level = .info };

const ROOT = "/home/adam/Development/Experiments/Video-Generation/.work";
const EXTRAS = ROOT ++ "/extras";
const N_BLOCKS: usize = 48;
const TP: i64 = 28672; // production tokens (E3's working length)
const SP: i64 = 32; // harness context length (prompt wiring is Phase 4)
const SIGMA: f32 = 0.909375;

const Tu: usize = @intCast(TP);
const Su: usize = @intCast(SP);
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

/// Pre-pass agreement gate (pre-registered with the chunk fix): chunk
/// 1024 has never run inside the full block, so before the production
/// pass, wAttn1 (auto-selecting PCHUNK at this length) must agree with
/// the dense twin — torch-anchored at the harness — on block-0 weights
/// at T=4096, budget 1e-5 per the E3w agreement precedent.
fn chunkAgreeGate(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, blob: *const ldr.Blob) !bool {
    const TG: i64 = 4096;
    const TGu: usize = @intCast(TG);
    var prng = std.Random.DefaultPrng.init(0x5a5a1234);
    const rnd = prng.random();

    const x_h = try allocator.alloc(f32, TGu * Du);
    defer allocator.free(x_h);
    for (x_h) |*x| x.* = (rnd.float(f32) - 0.5) * 1.4;
    const ts_h = try allocator.alloc(f32, TGu * 9 * Du);
    defer allocator.free(ts_h);
    for (ts_h) |*x| x.* = (rnd.float(f32) - 0.5) * 0.2;
    const cos_h = try allocator.alloc(f32, TGu * Hu * 64);
    defer allocator.free(cos_h);
    const sin_h = try allocator.alloc(f32, TGu * Hu * 64);
    defer allocator.free(sin_h);
    for (cos_h, sin_h) |*c, *s| {
        const a = rnd.float(f32) * std.math.tau;
        c.* = @cos(a);
        s.* = @sin(a);
    }

    const x_shape = zml.Shape.init(.{ .t = TG, .i = blk.D }, .f32);
    const ts_shape = zml.Shape.init(.{ .t = TG, .n = 9, .i = blk.D }, .f32);
    const pe_shape = zml.Shape.init(.{ .q = TG, .h = blk.H, .f = 64 }, .f32);
    var x_buf: zml.Buffer = try .fromBytes(io, platform, x_shape, .replicated, std.mem.sliceAsBytes(x_h));
    defer x_buf.deinit();
    var ts_buf: zml.Buffer = try .fromBytes(io, platform, ts_shape, .replicated, std.mem.sliceAsBytes(ts_h));
    defer ts_buf.deinit();
    var cos_buf: zml.Buffer = try .fromBytes(io, platform, pe_shape, .replicated, std.mem.sliceAsBytes(cos_h));
    defer cos_buf.deinit();
    var sin_buf: zml.Buffer = try .fromBytes(io, platform, pe_shape, .replicated, std.mem.sliceAsBytes(sin_h));
    defer sin_buf.deinit();

    var bufs = try ldr.buildBlockBufs(io, platform, blob, blob.data, 0);
    defer deinitBlockBufs(&bufs);

    const model = blk.makeBlockSpecs();
    const x_spec: zml.Tensor = .fromShape(x_shape);
    const ts_spec: zml.Tensor = .fromShape(ts_shape);
    const cos_spec: zml.Tensor = .fromShape(pe_shape);
    const sin_spec: zml.Tensor = .fromShape(pe_shape);

    const outs: [2][]f32 = .{ try allocator.alloc(f32, TGu * Du), try allocator.alloc(f32, TGu * Du) };
    defer allocator.free(outs[0]);
    defer allocator.free(outs[1]);
    inline for (.{ "s3Attn1", "wAttn1" }, 0..) |mname, oi| {
        const method = comptime std.meta.stringToEnum(std.meta.DeclEnum(blk.Block), mname).?;
        var exe = try platform.compile(allocator, io, model, method, .{ x_spec, ts_spec, cos_spec, sin_spec }, .{});
        // Release the gate executable before the production pass: run 12
        // localized the latched ResourceExhausted to block 0's FIRST
        // arena allocation retrying past gate leftovers — the failed
        // attempt poisons every later transfer-stream await.
        defer exe.deinit();
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
        @memcpy(outs[oi], slice.constItems(f32)[0 .. TGu * Du]);
    }
    var err: f64 = 0;
    var ref: f64 = 0;
    for (outs[1], outs[0]) |g, w| {
        const d = @as(f64, g) - @as(f64, w);
        err += d * d;
        ref += @as(f64, w) * w;
    }
    const rms = @sqrt(err / (ref + 1e-20));
    const pass = rms <= 1e-5;
    log.info("GATE chunk-agree@T4096 (PCHUNK vs dense): rel-RMS {e:.3} -> {s}", .{ rms, if (pass) "PASS" else "FAIL" });
    return pass;
}

fn finiteStats(name: []const u8, v: []const f32) !void {
    var nan: usize = 0;
    var sum2: f64 = 0;
    var maxa: f64 = 0;
    for (v) |x| {
        if (std.math.isNan(x) or std.math.isInf(x)) nan += 1;
        const a = @abs(@as(f64, x));
        if (a > maxa) maxa = a;
        sum2 += @as(f64, x) * x;
    }
    const rms = @sqrt(sum2 / @as(f64, @floatFromInt(v.len)));
    log.info("CHECK {s}: nan/inf {d}, rms {d:.4}, max_abs {d:.4} -> {s}", .{
        name, nan, rms, maxa, if (nan == 0) "PASS" else "FAIL",
    });
    if (nan != 0) return error.NonFinite;
}

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(1_000_000);
    const allocator = init.gpa;
    const io = init.io;

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

    // 0.90, NOT higher: this card drives the display. The 0.95 experiment
    // starved the compositor's framebuffer pin (amdgpu -12, journalctl
    // 2026-08-21/22) and took the desktop down with the run. 0.90 ran the
    // full pass in run 2 without touching the session.
    const platform: *zml.Platform = try .auto(allocator, io, .{
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = true, .memory_fraction = 0.90 } } },
    });
    log.info("platform: {s}", .{@tagName(platform.target)});

    if (!try chunkAgreeGate(allocator, io, platform, &blobs[0])) {
        log.err("chunk agreement gate failed — stopping before the production pass", .{});
        return error.ChunkAgreeFailed;
    }

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

    // ---- seeded synthetic inputs ------------------------------------------
    var prng = std.Random.DefaultPrng.init(0x1f2e3d4c);
    const rnd = prng.random();

    const lat_h = try allocator.alloc(f32, Tu * Cu);
    for (lat_h) |*x| x.* = (rnd.float(f32) - 0.5) * 1.4; // ~N(0,0.5)-ish spread
    const t_h = try allocator.alloc(f32, Tu);
    for (t_h, 0..) |*x, i| {
        // stripes: first 4096 conditioning (0.0), next 4096 half (0.5), rest full
        const m: f32 = if (i < 4096) 0.0 else if (i < 8192) 0.5 else 1.0;
        x.* = m * SIGMA;
    }
    const sg_h = [_]f32{SIGMA};
    const ctx_h = try allocator.alloc(f32, Su * Du);
    for (ctx_h) |*x| x.* = (rnd.float(f32) - 0.5) * 1.4;
    // unit-circle rope: cos/sin of random angles (grids are Phase 5 wiring)
    const cos_h = try allocator.alloc(f32, Tu * Hu * 64);
    const sin_h = try allocator.alloc(f32, Tu * Hu * 64);
    for (cos_h, sin_h) |*c, *s| {
        const a = rnd.float(f32) * std.math.tau;
        c.* = @cos(a);
        s.* = @sin(a);
    }
    log.info("synthetic inputs ready (T={d})", .{TP});

    const lat_shape = zml.Shape.init(.{ .t = TP, .i = core.IN_CH }, .f32);
    const t_shape = zml.Shape.init(.{ .t = TP }, .f32);
    const sg_shape = zml.Shape.init(.{ .t = 1 }, .f32);
    const x_shape = zml.Shape.init(.{ .t = TP, .i = blk.D }, .f32);
    const ctx_shape = zml.Shape.init(.{ .t = SP, .i = blk.D }, .f32);
    const ts_shape = zml.Shape.init(.{ .t = TP, .n = 9, .i = blk.D }, .f32);
    const pts_shape = zml.Shape.init(.{ .n = 2, .i = blk.D }, .f32);
    const pe_shape = zml.Shape.init(.{ .q = TP, .h = blk.H, .f = 64 }, .f32);

    var lat_buf: zml.Buffer = try .fromBytes(io, platform, lat_shape, .replicated, std.mem.sliceAsBytes(lat_h));
    defer lat_buf.deinit();
    var t_buf: zml.Buffer = try .fromBytes(io, platform, t_shape, .replicated, std.mem.sliceAsBytes(t_h));
    defer t_buf.deinit();
    var sg_buf: zml.Buffer = try .fromBytes(io, platform, sg_shape, .replicated, std.mem.sliceAsBytes(&sg_h));
    defer sg_buf.deinit();
    var ctx_buf: zml.Buffer = try .fromBytes(io, platform, ctx_shape, .replicated, std.mem.sliceAsBytes(ctx_h));
    defer ctx_buf.deinit();
    var cos_buf: zml.Buffer = try .fromBytes(io, platform, pe_shape, .replicated, std.mem.sliceAsBytes(cos_h));
    defer cos_buf.deinit();
    var sin_buf: zml.Buffer = try .fromBytes(io, platform, pe_shape, .replicated, std.mem.sliceAsBytes(sin_h));
    defer sin_buf.deinit();
    allocator.free(cos_h);
    allocator.free(sin_h);

    const lat_spec: zml.Tensor = .fromShape(lat_shape);
    const t_spec: zml.Tensor = .fromShape(t_shape);
    const sg_spec: zml.Tensor = .fromShape(sg_shape);
    const x_spec: zml.Tensor = .fromShape(x_shape);
    const ctx_spec: zml.Tensor = .fromShape(ctx_shape);
    const ts_spec: zml.Tensor = .fromShape(ts_shape);
    const pts_spec: zml.Tensor = .fromShape(pts_shape);
    const cos_spec: zml.Tensor = .fromShape(pe_shape);
    const sin_spec: zml.Tensor = .fromShape(pe_shape);

    // ---- compile at production shapes -------------------------------------
    const compile_t0 = nowNs(io);
    const cmodel = core.makeCoreSpecs();
    const cbufs = try loadCoreBufs(allocator, io, platform);
    const patchify_m = comptime std.meta.stringToEnum(std.meta.DeclEnum(core.CoreParts), "patchify").?;
    const ts9r_m = comptime std.meta.stringToEnum(std.meta.DeclEnum(core.CoreParts), "ts9r").?;
    const pts2r_m = comptime std.meta.stringToEnum(std.meta.DeclEnum(core.CoreParts), "pts2r").?;
    const emb_m = comptime std.meta.stringToEnum(std.meta.DeclEnum(core.CoreParts), "emb").?;
    const tail_m = comptime std.meta.stringToEnum(std.meta.DeclEnum(core.CoreParts), "tailFromEmb").?;
    var patchify_exe = try platform.compile(allocator, io, cmodel, patchify_m, .{lat_spec}, .{});
    var ts9r_exe = try platform.compile(allocator, io, cmodel, ts9r_m, .{t_spec}, .{});
    var pts2r_exe = try platform.compile(allocator, io, cmodel, pts2r_m, .{sg_spec}, .{});
    var emb_exe = try platform.compile(allocator, io, cmodel, emb_m, .{t_spec}, .{});
    var tail_exe = try platform.compile(allocator, io, cmodel, tail_m, .{ x_spec, zml.Tensor.fromShape(x_shape) }, .{});
    const vstats_m = comptime std.meta.stringToEnum(std.meta.DeclEnum(core.CoreParts), "velStats").?;
    const vel_shape = zml.Shape.init(.{ .t = TP, .i = core.IN_CH }, .f32);
    var vstats_exe = try platform.compile(allocator, io, cmodel, vstats_m, .{zml.Tensor.fromShape(vel_shape)}, .{});
    const model = blk.makeBlockSpecs();
    const wblk_m = comptime std.meta.stringToEnum(std.meta.DeclEnum(blk.Block), "wBlockOut").?;
    var blk_exe = try platform.compile(allocator, io, model, wblk_m, .{ x_spec, ts_spec, ctx_spec, pts_spec, cos_spec, sin_spec }, .{});
    log.info("7 executables compiled at T={d} in {d:.1} s", .{ TP, @as(f64, @floatFromInt(@as(u64, @intCast(nowNs(io) - compile_t0)))) / 1e9 });

    // emb once, resident for the whole pass (470 MB) — the tail takes it
    // as input. Run-4 receipts: the RECOMPUTING tail's fused graph needed
    // one contiguous 8.78 GiB temp arena (~20 [T,D] buffers), which fits
    // an empty pool (probe PASSed) but not the post-pass fragmented one.
    var e_buf: zml.Buffer = undefined;
    {
        var args = try emb_exe.args(allocator);
        defer args.deinit(allocator);
        var results = try emb_exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ cbufs, t_buf });
        emb_exe.call(args, &results);
        e_buf = results.get(zml.Buffer);
        try e_buf.await(io);
    }
    defer e_buf.deinit();

    // ---- ring -------------------------------------------------------------
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

    // ---- the pass ---------------------------------------------------------
    const step_t0 = nowNs(io);
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
        try ts_buf.await(io);
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
        try x_cur.await(io);
    }
    const cond_ns: u64 = @intCast(nowNs(io) - step_t0);
    log.info("conditioning + patchify on device: {d:.2} s", .{@as(f64, @floatFromInt(cond_ns)) / 1e9});

    var block_ns_sum: u64 = 0;
    {
        var args = try blk_exe.args(allocator);
        defer args.deinit(allocator);
        var results = try blk_exe.results(allocator);
        defer results.deinit(allocator);
        for (0..N_BLOCKS) |pos| {
            errdefer log.err("production pass failed at block {d}", .{pos});
            const b_t0 = nowNs(io);
            const slot = try ring.acquire(io, pos);
            const blob = &ring.blobs[pos];
            var bufs = try ldr.buildBlockBufs(io, platform, blob, slot.mem, @intCast(pos));
            ring.release(io, slot);
            args.set(.{ bufs, x_cur, ts_buf, ctx_buf, pts_buf, cos_buf, sin_buf });
            blk_exe.call(args, &results);
            var out: zml.Buffer = results.get(zml.Buffer);
            try out.await(io);
            x_cur.deinit();
            deinitBlockBufs(&bufs);
            x_cur = out;
            const b_ns: u64 = @intCast(nowNs(io) - b_t0);
            block_ns_sum += b_ns;
            if (pos % 8 == 0 or pos == 47) log.info("block {d}: {d:.2} s", .{ pos, @as(f64, @floatFromInt(b_ns)) / 1e9 });
        }
    }

    // The tail needs only x and e — release the 4.23 GB ts9 (and pts2),
    // and DESTROY THE BLOCK EXECUTABLE: the buffer-assignment dump
    // identified the constant 8.78 GiB post-pass ask as the block exe's
    // own preallocated-temp arena (allocation 75, 9,426,703,416 bytes),
    // re-requested at the next executable invocation. The tail's temp is
    // 476 MB; it was never the problem (H-PROD runs 2-8, notebook).
    ts_buf.deinit();
    pts_buf.deinit();
    blk_exe.deinit();
    log.info("ts9/pts2 + block exe released; invoking tail", .{});

    // Diagnostic ladder (H-PROD run 9's stack trace: the OOM surfaces at
    // the readback await, i.e. the tail EXECUTION fails async; the same
    // exe ran fine on a young pool in run 4's probe): attempt, and on
    // ResourceExhausted wait for deferred deallocations to land, retry.
    var vel_buf: zml.Buffer = undefined;
    var attempt: usize = 0;
    while (true) : (attempt += 1) {
        var args = try tail_exe.args(allocator);
        defer args.deinit(allocator);
        var results = try tail_exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ cbufs, x_cur, e_buf });
        tail_exe.call(args, &results);
        vel_buf = results.get(zml.Buffer);
        if (vel_buf.await(io)) |_| {
            if (attempt > 0) log.info("tail retry {d} SUCCEEDED — transient pool state confirmed", .{attempt});
            break;
        } else |err| {
            log.warn("tail attempt {d} failed: {t}", .{ attempt, err });
            vel_buf.deinit();
            if (attempt >= 3) return err;
            try std.Io.sleep(io, .fromMilliseconds(1000), .awake);
        }
    }
    x_cur.deinit();
    const wall_ns: u64 = @intCast(nowNs(io) - step_t0);

    // Tiny readback first (12 bytes of device-computed stats) — run-10
    // showed the FULL-velocity D2H is where the 8.78 GiB ask surfaces,
    // while the tail itself executes cleanly. If even 12 bytes fail, the
    // error is latched client state, not transfer staging.
    {
        var args = try vstats_exe.args(allocator);
        defer args.deinit(allocator);
        var results = try vstats_exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ cbufs, vel_buf });
        vstats_exe.call(args, &results);
        var st_buf: zml.Buffer = results.get(zml.Buffer);
        defer st_buf.deinit();
        // The client's async error queue delivers the pass's failed-then-
        // retried arena attempt to the FIRST post-pass D2H await (runs
        // 2-13). If the queue-delivery theory is right, the error is
        // consumed once and a retry reads clean.
        var st_slice = st_buf.toSliceAlloc(allocator, io) catch |err| blk2: {
            log.warn("stats readback attempt 0 failed ({t}) — retrying (stale async error theory)", .{err});
            break :blk2 try st_buf.toSliceAlloc(allocator, io);
        };
        defer st_slice.free(allocator);
        const st = st_slice.constItems(f32)[0..3];
        const n: f64 = @floatFromInt(Tu * Cu);
        const rms = @sqrt(@as(f64, st[1]) / n);
        log.info("CHECK velocity (device stats): nan/inf {d:.0}, rms {d:.4}, max_abs {d:.4} -> {s}", .{
            st[0], rms, st[2], if (st[0] == 0) "PASS" else "FAIL",
        });
        if (st[0] != 0) return error.NonFinite;
    }

    // Full readback, attempted second, outcome logged either way — the
    // diagnostic bit for the D2H mechanism.
    if (vel_buf.toSliceAlloc(allocator, io)) |vslice_v| {
        var vslice = vslice_v;
        try finiteStats("velocity (full readback)", vslice.constItems(f32)[0 .. Tu * Cu]);
        vslice.free(allocator);
    } else |err| {
        log.warn("full-velocity readback failed ({t}) — device stats above stand; readback path is the known Phase 6 item", .{err});
    }
    vel_buf.deinit();

    log.info("PRODUCTION PASS: {d:.2} s wall (blocks {d:.2} s, {d:.2} s/block avg), reader parks {d}, consumer parks {d}", .{
        @as(f64, @floatFromInt(wall_ns)) / 1e9,
        @as(f64, @floatFromInt(block_ns_sum)) / 1e9,
        @as(f64, @floatFromInt(block_ns_sum)) / 1e9 / N_BLOCKS,
        ring.reader_parks,
        ring.consumer_parks,
    });
    log.info("PRODUCTION-LENGTH PASS COMPLETE (T={d})", .{TP});
}
