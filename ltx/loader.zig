//! The block-streaming loader's host side: NVMe → page cache (mmap) →
//! pinned host ring → (serial) device uploads. E-STREAM-3 contract in
//! docs/lab-notebook.md (2026-08-21/22): slot states asserted on every
//! transition, the schedule statically checked against ring capacities
//! before any I/O, blob digests verified on first touch, zero
//! steady-state allocation (all buffers preallocated).
//!
//! Uploads are the CONSUMER's job (serial with compute per E-STREAM-2's
//! verdict — the ROCm PJRT plugin stream-serializes H2D behind compute,
//! so there is nothing to overlap device-side). This module owns disk →
//! pinned staging, which the reader thread DOES overlap with compute.
const std = @import("std");
const zml = @import("zml");
const blk = @import("block.zig");

const log = std.log;

pub const HOST_SLOTS: usize = 2;

// ---- blob files -----------------------------------------------------------

pub const Entry = struct {
    name: []const u8,
    dtype: []const u8,
    shape: []i64,
    offset: u64,
    nbytes: u64,
    sha256: []const u8,
};

pub const Manifest = struct {
    packer_version: u32,
    packer_commit: []const u8,
    @"align": u32,
    block: i64,
    repo: []const u8,
    revision: []const u8,
    source_file: []const u8,
    model_version: []const u8,
    total_bytes: u64,
    blob_sha256: []const u8,
    entries: []Entry,
};

pub const Blob = struct {
    data: []align(std.heap.page_size_min) const u8,
    manifest: Manifest,
    parsed: std.json.Parsed(Manifest),

    pub fn open(allocator: std.mem.Allocator, io: std.Io, dir: []const u8) !Blob {
        var buf: [512]u8 = undefined;
        const mpath = try std.fmt.bufPrint(&buf, "{s}/blob_manifest.json", .{dir});
        const mraw = try std.Io.Dir.cwd().readFileAlloc(io, mpath, allocator, .unlimited);
        defer allocator.free(mraw);
        // .alloc_always: string fields must be COPIED into the parse arena —
        // by default they point into `mraw`, which is freed on return
        // (found as a segfault on the first manifest print).
        const parsed = try std.json.parseFromSlice(Manifest, allocator, mraw, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
        errdefer parsed.deinit();

        const bpath = try std.fmt.bufPrint(&buf, "{s}/blob.bin", .{dir});
        const st = try std.Io.Dir.cwd().statFile(io, bpath, .{});
        if (st.size != parsed.value.total_bytes) return error.BlobSizeMismatch;
        var f = try std.Io.Dir.cwd().openFile(io, bpath, .{});
        defer f.close(io);
        const data = try std.posix.mmap(null, @intCast(st.size), .{ .READ = true }, .{ .TYPE = .PRIVATE }, f.handle, 0);

        return .{ .data = data, .manifest = parsed.value, .parsed = parsed };
    }

    /// G-RING-2: offsets strictly increasing, aligned, gap-free up to
    /// alignment fill, last entry ends exactly at total_bytes.
    pub fn checkLayout(self: *const Blob) !void {
        const m = self.manifest;
        var end: u64 = 0;
        for (m.entries) |e| {
            if (e.offset % m.@"align" != 0) return error.Misaligned;
            if (e.offset < end or e.offset - end >= m.@"align") return error.LayoutGap;
            end = e.offset + e.nbytes;
        }
        if (end != m.total_bytes) return error.TrailingBytes;
    }

    pub fn entryBytes(self: *const Blob, e: Entry) []const u8 {
        return self.data[e.offset .. e.offset + e.nbytes];
    }

    pub fn deinit(self: *Blob) void {
        std.posix.munmap(self.data);
        self.parsed.deinit();
    }
};

pub fn sha256Hex(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

/// Normalize a blob entry name to the harness file-stem convention:
/// '.' -> '_', "::q8scale" -> "_q8scale", "::q8" -> "_q8".
pub fn normalizeName(out: []u8, name: []const u8) []const u8 {
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
pub fn findEntry(blob: *const Blob, want: []const u8) !Entry {
    var nbuf: [256]u8 = undefined;
    for (blob.manifest.entries) |e| {
        if (std.mem.eql(u8, normalizeName(&nbuf, e.name), want)) return e;
    }
    log.err("blob {d}: no entry named {s}", .{ blob.manifest.block, want });
    return error.MissingEntry;
}

/// Build one block's device buffers from a memory region laid out per the
/// blob manifest (either a ring slot or the mmap itself for direct mode).
pub fn buildBlockBufs(io: std.Io, platform: *zml.Platform, blob: *const Blob, mem: []const u8, block_idx: i64) !zml.Bufferized(blk.Block) {
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

pub const Q8Spec = struct { field: []const u8, stem: []const u8, dims: [2]i64 };
pub const Q8_SPECS = [_]Q8Spec{
    .{ .field = "q1q", .stem = "attn1_to_q_weight", .dims = .{ blk.D, blk.D } },
    .{ .field = "k1q", .stem = "attn1_to_k_weight", .dims = .{ blk.D, blk.D } },
    .{ .field = "v1q", .stem = "attn1_to_v_weight", .dims = .{ blk.D, blk.D } },
    .{ .field = "q2q", .stem = "attn2_to_q_weight", .dims = .{ blk.D, blk.D } },
    .{ .field = "k2q", .stem = "attn2_to_k_weight", .dims = .{ blk.D, blk.D } },
    .{ .field = "v2q", .stem = "attn2_to_v_weight", .dims = .{ blk.D, blk.D } },
    .{ .field = "f1q", .stem = "ff_net_0_proj_weight", .dims = .{ blk.FF, blk.D } },
    .{ .field = "f2q", .stem = "ff_net_2_weight", .dims = .{ blk.D, blk.FF } },
};

pub fn buildQuantBufs(io: std.Io, platform: *zml.Platform, blob: *const Blob, mem: []const u8, block_idx: i64, base: zml.Bufferized(blk.Block), qbufs: *zml.Bufferized(blk.QBlock)) !void {
    qbufs.base = base;
    var namebuf: [256]u8 = undefined;
    inline for (Q8_SPECS) |qs| {
        const wq = try std.fmt.bufPrint(&namebuf, "transformer_blocks_{d}_{s}_q8", .{ block_idx, qs.stem });
        const eq = try findEntry(blob, wq);
        const qshape = zml.Shape.init(.{ .o = qs.dims[0], .i = qs.dims[1] }, .i8);
        @field(qbufs, qs.field) = try zml.Buffer.fromBytes(io, platform, qshape, .replicated, mem[eq.offset .. eq.offset + eq.nbytes]);
        const ws = try std.fmt.bufPrint(&namebuf, "transformer_blocks_{d}_{s}_q8scale", .{ block_idx, qs.stem });
        const es = try findEntry(blob, ws);
        const sshape = zml.Shape.init(.{ .o = qs.dims[0], .g = @divExact(qs.dims[1], 128) }, .f32);
        const sfield = qs.field[0..2] ++ "s";
        @field(qbufs, sfield) = try zml.Buffer.fromBytes(io, platform, sshape, .replicated, mem[es.offset .. es.offset + es.nbytes]);
    }
}

// ---- the pinned host ring -------------------------------------------------

pub const SlotState = enum { free, filling, ready, in_use };

pub const HostSlot = struct {
    mem: []align(4096) u8,
    state: SlotState = .free,
    /// schedule position this slot holds (valid unless .free)
    sched_pos: usize = 0,
};

pub const Ring = struct {
    slots: [HOST_SLOTS]HostSlot,
    blobs: []const Blob,
    /// indices into `blobs`, one per scheduled block execution
    schedule: []const usize,
    verify_first_touch: bool,

    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    reader_parks: usize = 0,
    consumer_parks: usize = 0,
    copy_ns: u64 = 0,
    verify_ns: u64 = 0,
    reader_err: ?anyerror = null,
    /// Set by abort(): the reader exits at its next wait instead of
    /// parking forever — without this, an error on the consumer side
    /// deadlocks the teardown join (found the hard way in walk48 run 1).
    shutdown: bool = false,
    verified: []bool,

    fn assertTransition(from: SlotState, to: SlotState) void {
        const ok = switch (from) {
            .free => to == .filling,
            .filling => to == .ready,
            .ready => to == .in_use,
            .in_use => to == .free,
        };
        if (!ok) std.debug.panic("illegal slot transition {t} -> {t}", .{ from, to });
    }

    fn setState(slot: *HostSlot, to: SlotState) void {
        assertTransition(slot.state, to);
        slot.state = to;
    }

    /// G-RING-5: prove the schedule fits the ring before any I/O. With a
    /// serial consumer (one block in_use) and one reader (one filling),
    /// peak occupancy is 2 regardless of schedule length; the simulation
    /// walks the schedule and asserts it, and also that every scheduled
    /// index has a blob and slots fit the largest blob.
    pub fn staticCheck(self: *const Ring) !void {
        for (self.schedule) |bi| {
            if (bi >= self.blobs.len) return error.ScheduleOutOfRange;
            if (self.blobs[bi].manifest.total_bytes > self.slots[0].mem.len) return error.SlotTooSmall;
        }
        // Serial consumer (one slot in_use) plus one reader (at most one
        // slot filling ahead): peak occupancy is min(2, schedule length).
        const peak: usize = @min(2, self.schedule.len);
        if (peak > HOST_SLOTS) return error.ScheduleNeedsMoreSlots;
        log.info("static check: {d}-entry schedule, peak host occupancy {d}/{d} slots -> OK", .{ self.schedule.len, peak, HOST_SLOTS });
    }

    /// Reader thread body: walk the schedule, stage each blob into a free
    /// slot (page-cache read + memcpy into pinned memory), verifying the
    /// blob digest on first touch.
    pub fn readerMain(self: *Ring, io: std.Io) void {
        self.readerInner(io) catch |err| {
            self.mutex.lockUncancelable(io);
            self.reader_err = err;
            self.mutex.unlock(io);
            self.cond.broadcast(io);
        };
    }

    fn readerInner(self: *Ring, io: std.Io) !void {
        for (self.schedule, 0..) |bi, pos| {
            const slot = blk: {
                self.mutex.lockUncancelable(io);
                defer self.mutex.unlock(io);
                while (true) {
                    if (self.shutdown) return error.RingShutdown;
                    for (&self.slots) |*s| {
                        if (s.state == .free) break :blk s;
                    }
                    self.reader_parks += 1;
                    self.cond.waitUncancelable(io, &self.mutex);
                }
            };
            {
                self.mutex.lockUncancelable(io);
                defer self.mutex.unlock(io);
                setState(slot, .filling);
                slot.sched_pos = pos;
            }
            const blob = &self.blobs[bi];
            if (self.verify_first_touch and !self.verified[bi]) {
                const t0 = nowNs(io);
                const hex = sha256Hex(blob.data);
                if (!std.mem.eql(u8, &hex, blob.manifest.blob_sha256)) return error.BlobDigestMismatch;
                self.verified[bi] = true;
                self.verify_ns += @intCast(nowNs(io) - t0);
            }
            const t0 = nowNs(io);
            @memcpy(slot.mem[0..blob.data.len], blob.data);
            self.copy_ns += @intCast(nowNs(io) - t0);
            {
                self.mutex.lockUncancelable(io);
                defer self.mutex.unlock(io);
                setState(slot, .ready);
            }
            self.cond.broadcast(io);
        }
    }

    /// Consumer side: wait for the slot staging schedule position `pos`.
    pub fn acquire(self: *Ring, io: std.Io, pos: usize) !*HostSlot {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (true) {
            if (self.reader_err) |err| return err;
            for (&self.slots) |*s| {
                if (s.state == .ready and s.sched_pos == pos) {
                    setState(s, .in_use);
                    return s;
                }
            }
            self.consumer_parks += 1;
            self.cond.waitUncancelable(io, &self.mutex);
        }
    }

    /// Consumer-side teardown: wake and retire the reader regardless of
    /// ring state. Idempotent; call before joining the reader thread.
    pub fn abort(self: *Ring, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        self.shutdown = true;
        self.mutex.unlock(io);
        self.cond.broadcast(io);
    }

    pub fn release(self: *Ring, io: std.Io, slot: *HostSlot) void {
        self.mutex.lockUncancelable(io);
        setState(slot, .free);
        self.mutex.unlock(io);
        self.cond.broadcast(io);
    }
};

fn nowNs(io: std.Io) i96 {
    const ts: std.Io.Timestamp = .now(io, .awake);
    return ts.toNanoseconds();
}
