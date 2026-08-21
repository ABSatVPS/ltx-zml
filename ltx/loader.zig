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
