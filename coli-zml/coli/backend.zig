//! colibri GPU backend on ZML/PJRT. Weights stay in colibri's packed
//! quantized encodings on device (u4/u2/i8 + f32 scales); dequantization is
//! part of the compiled graph so XLA can fuse it into the matmuls — the
//! in-graph equivalent of what backend_cuda.cu's packed kernels do by hand.
//! Mirrors the semantics of c/backend_cuda.h; grows into the coli_cuda_* ABI.
const std = @import("std");
const zml = @import("zml");

/// coli fmt codes (QT in colibri.c): 0=f32, 1=int8, 2=int4, 3=int2,
/// 4=grouped int4, 6=E8 (not yet ported).
/// int4 is OFFSET-8 (`(b&0xF)-8`, low nibble first) and int2 is offset-2 —
/// per quant.h, the canonical codec. XLA's u4/u2 packing is also low-first,
/// so colibri's packed bytes upload as-is (I must be even for u4, %4 for u2).
pub const Fmt = enum(i32) {
    f32 = 0,
    q8 = 1,
    q4 = 2,
    q2 = 3,
    q4g = 4,
    e8 = 6,
    _,
};

/// Mirrors ColiCudaTensor: persistent device copy created at upload,
/// independent of the host buffer's lifetime.
pub const DeviceTensor = struct {
    weights: zml.Buffer,
    scales: ?zml.Buffer,
    /// fmt=6 only: decoded rows with the down-input rotation Q folded in,
    /// used when this tensor is the `down` matrix of an expert MLP.
    down_folded: ?zml.Buffer = null,
    fmt: i32,
    i: i64,
    o: i64,
    gs: i32,

    pub fn deinit(self: *DeviceTensor) void {
        self.weights.deinit();
        if (self.scales) |*s| s.deinit();
        if (self.down_folded) |*f| f.deinit();
    }
};

pub const ExeKey = struct {
    kind: enum { matmul, mlp, group },
    s: i64,
    i: i64,
    o: i64,
    fmt: i32,
    gs: i32,
};

/// Whole expert group as ONE executable: N same-shaped quantized experts,
/// one x row each (the dominant decode pattern). One upload, one launch,
/// one sync for the entire group instead of N round trips.
fn Group(comptime N: usize) type {
    return struct {
        gates: [N]zml.Tensor,
        gscs: [N]zml.Tensor,
        ups: [N]zml.Tensor,
        uscs: [N]zml.Tensor,
        downs: [N]zml.Tensor,
        dscs: [N]zml.Tensor,

        pub fn forward(self: @This(), x: zml.Tensor) zml.Tensor {
            var ys: [N]zml.Tensor = undefined;
            for (0..N) |e| {
                const xe = x.slice1d(.s, .{ .start = @intCast(e), .end = @intCast(e + 1) });
                const g = qdot(xe, self.gates[e], self.gscs[e]);
                const u = qdot(xe, self.ups[e], self.uscs[e]);
                const h = g.silu().mul(u).withTags(.{ .s, .i });
                ys[e] = qdot(h, self.downs[e], self.dscs[e]);
            }
            return zml.Tensor.concatenate(&ys, .s);
        }
    };
}

fn weightDtype(fmt: i32) !zml.DataType {
    return switch (fmt) {
        0, 6 => .f32, // fmt=6 (E8) is host-decoded to f32 for now
        1 => .i8,
        2, 4 => .u4,
        3 => .u2,
        else => error.UnsupportedFormat,
    };
}

// ---- fmt=6 (E8/IQ3) host decode — ported from quant.h, the engine oracle ----
// A super-block is 98 bytes per 256 weights: 64 codebook indices, 8 words of
// (4x7 sign bits + 4-bit sub-scale in the top nibble), then an fp16 scale.
// Two grid indices feed each group of 8 weights; the 8th sign is the parity
// of the other 7. Scales live inside the blocks (no scale array).
const E8_QK = 256;
const E8_SUB = 32;
const E8_BBYTES = 98;

pub fn e8RowBytes(i_dim: usize) usize {
    return ((i_dim + E8_QK - 1) / E8_QK) * E8_BBYTES;
}

fn e8Fp16(h: u16) f32 {
    const sign: u32 = @as(u32, h >> 15) << 31;
    const exp: u32 = (h >> 10) & 0x1F;
    const man: u32 = h & 0x3FF;
    const bits: u32 = if (exp == 0)
        (if (man != 0) sign | ((127 - 15 + 1) << 23) | (man << 13) else sign)
    else if (exp == 0x1F)
        sign | 0x7F800000 | (man << 13)
    else
        sign | ((exp + 112) << 23) | (man << 13);
    return @bitCast(bits);
}

fn e8DecodeRow(row: []const u8, i_dim: usize, w: []f32, grid: *const [256][4]u8) void {
    const nb = (i_dim + E8_QK - 1) / E8_QK;
    for (0..nb) |b| {
        const blk = row[b * E8_BBYTES ..][0..E8_BBYTES];
        const d = e8Fp16(@as(u16, blk[96]) | (@as(u16, blk[97]) << 8));
        for (0..E8_QK / E8_SUB) |ib| {
            const base = b * E8_QK + ib * E8_SUB;
            if (base >= i_dim) return;
            const word = std.mem.readInt(u32, blk[64 + ib * 4 ..][0..4], .little);
            const db = d * (0.5 + @as(f32, @floatFromInt((word >> 28) & 0xF))) * 0.5;
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

/// Rotation sign bits, regenerated — xorshift64* seeded 417+n, matching
/// quant.h e8_signs and the converter exactly.
fn e8Signs(bits: []u8, n: usize) void {
    var s: u64 = 417 +% @as(u64, n);
    for (0..(n + 7) / 8) |i| {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        bits[i] = @truncate((s *% 2685821657736338717) >> 56);
    }
}

fn fwhtButterfliesAndScale(a: []f32) void {
    const n = a.len;
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
    const s = 1.0 / @sqrt(@as(f32, @floatFromInt(n)));
    for (a) |*v| v.* *= s;
}

fn applySigns(a: []f32, bits: []const u8) void {
    for (a, 0..) |*v, i| {
        if ((bits[i >> 3] >> @intCast(i & 7)) & 1 != 0) v.* = -v.*;
    }
}

/// Apply the fmt=6 rotation to one row, block-diagonally tiled like
/// quant.h e8_rot_rows (block = lowest set bit of the remainder, cap 32768).
/// dir .qt computes Qᵀv (signs → butterflies → scale) — what the device
/// kernel applies to the silu product before the down matmul.
/// dir .q computes Q·v (butterflies → scale → signs) — used to FOLD the
/// rotation into decoded down weight rows: row·(Qᵀh) == (Q·row)·h.
fn e8RotRow(row: []f32, dim: usize, comptime dir: enum { qt, q }) void {
    var bits: [32768 / 8]u8 = undefined;
    var off: usize = 0;
    while (off < dim) {
        const rem = dim - off;
        var b = rem & (~rem + 1);
        while (b > 32768) b >>= 1;
        e8Signs(bits[0 .. (b + 7) / 8], b);
        const a = row[off..][0..b];
        switch (dir) {
            .qt => {
                applySigns(a, &bits);
                fwhtButterfliesAndScale(a);
            },
            .q => {
                fwhtButterfliesAndScale(a);
                applySigns(a, &bits);
            },
        }
        off += b;
    }
}

/// Traced dequant: convert packed weights to f32, apply the format's fixed
/// offset, then per-row ([O]) or per-group ([O,G]) scales.
fn dequant(w: zml.Tensor, sc: ?zml.Tensor) zml.Tensor {
    var f = w.convert(.f32);
    f = switch (w.dtype()) {
        .u4 => f.addConstant(-8.0),
        .u2 => f.addConstant(-2.0),
        else => f,
    };
    const s = sc orelse return f;
    if (s.rank() == 1) return f.mul(s.broad(f.shape()));
    // Grouped scales: [O,I] -> [O,G,E], scale each group, back to [O,I].
    const o = f.shape().dim(0);
    const i_dim = f.shape().dim(1);
    const ng = s.shape().dim(1);
    const f3 = f.reshape(zml.Shape.init(.{ .o = o, .g = ng, .e = @divExact(i_dim, ng) }, .f32));
    const m = f3.mul(s.broad(f3.shape()));
    return m.reshape(f.shape()).withTags(.{ .o, .i });
}

/// Quantized contraction as broadcast-mul + reduce instead of dot_general:
/// XLA loop-fuses the dequant chain into the reduction, so the kernel reads
/// the packed u4/u2 bytes directly. A dot_general would force the f32
/// dequantized operand to materialize in VRAM (measured: slower than not
/// quantizing at all). Reads the weights once per row of x — the right trade
/// at decode-sized S; revisit (dot + materialize, amortized) for prefill S.
fn qdot(x: zml.Tensor, w: zml.Tensor, sc: ?zml.Tensor) zml.Tensor {
    const wd = dequant(w, sc); // [.o,.i]
    const sh3 = zml.Shape.init(.{ .s = x.shape().dim(0), .o = wd.shape().dim(0), .i = wd.shape().dim(1) }, .f32);
    return x.broad(sh3).mul(wd.broad(sh3)).sum(.i).squeeze(.i).withTags(.{ .s, .o });
}

/// y[S,O] = x[S,I] @ W[O,I]^T — coli_cuda_matmul semantics.
fn matmulGraph(w: zml.Tensor, x: zml.Tensor) zml.Tensor {
    return x.dot(w, .i);
}

fn matmulQGraph(w: zml.Tensor, sc: zml.Tensor, x: zml.Tensor) zml.Tensor {
    return qdot(x, w, sc);
}

/// y = down(silu(gate(x)) * up(x)) — coli_cuda_expert_mlp semantics, fused.
fn mlpBody(gw: zml.Tensor, uw: zml.Tensor, dw: zml.Tensor, x: zml.Tensor) zml.Tensor {
    const g = x.dot(gw, .i);
    const u = x.dot(uw, .i);
    const h = g.silu().mul(u).withTags(.{ .s, .i });
    return h.dot(dw, .i);
}

fn mlpGraph(gw: zml.Tensor, uw: zml.Tensor, dw: zml.Tensor, x: zml.Tensor) zml.Tensor {
    return mlpBody(gw, uw, dw, x);
}

fn mlpQGraph(gw: zml.Tensor, gsc: zml.Tensor, uw: zml.Tensor, usc: zml.Tensor, dw: zml.Tensor, dsc: zml.Tensor, x: zml.Tensor) zml.Tensor {
    const g = qdot(x, gw, gsc);
    const u = qdot(x, uw, usc);
    const h = g.silu().mul(u).withTags(.{ .s, .i });
    return qdot(h, dw, dsc);
}

pub const Backend = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    exes: std.AutoHashMap(ExeKey, zml.Exe),
    /// coli_cuda_e8_set_grid: the E8 codebook (256x4 bytes) must be published
    /// before any fmt=6 upload; kept here so decode cannot drift from quant.h.
    e8_grid: ?[256][4]u8 = null,

    pub fn init(allocator: std.mem.Allocator, io: std.Io) !Backend {
        const platform: *zml.Platform = try .auto(allocator, io, .{});
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .exes = .init(allocator),
        };
    }

    pub fn setE8Grid(self: *Backend, grid: *const [256][4]u8) void {
        self.e8_grid = grid.*;
    }

    fn scaleShape(fmt: i32, i_dim: i64, o_dim: i64, gs: i32) !?zml.Shape {
        return switch (fmt) {
            0, 6 => null, // fmt=6 scales live inside the blocks
            1, 2, 3 => zml.Shape.init(.{ .o = o_dim }, .f32),
            4 => blk: {
                if (gs <= 0) break :blk zml.Shape.init(.{ .o = o_dim }, .f32);
                if (@rem(i_dim, gs) != 0) return error.UnsupportedFormat;
                break :blk zml.Shape.init(.{ .o = o_dim, .g = @divExact(i_dim, gs) }, .f32);
            },
            else => error.UnsupportedFormat,
        };
    }

    /// coli_cuda_tensor_upload(_g): quantized values keep their width on
    /// device (u4/u2/i8), but PJRT wants sub-byte hosts buffers UNPACKED
    /// (one element per byte), so nibbles/crumbs are expanded here. The
    /// format's offset (-8/-2) stays in the graph, not here.
    /// weights/scales may be freed by the caller afterwards.
    pub fn upload(self: *Backend, weights: []const u8, scales: ?[]const f32, fmt: i32, i_dim: i64, o_dim: i64, gs: i32) !DeviceTensor {
        const wdt = try weightDtype(fmt);
        const iu: usize = @intCast(i_dim);
        const ou: usize = @intCast(o_dim);

        if (fmt == 6) {
            const grid = &(self.e8_grid orelse return error.E8GridNotSet);
            const rb = e8RowBytes(iu);
            const w_f32 = try self.allocator.alloc(f32, iu * ou);
            defer self.allocator.free(w_f32);
            for (0..ou) |o| e8DecodeRow(weights[o * rb ..][0..rb], iu, w_f32[o * iu ..][0..iu], grid);

            const w_shape: zml.Shape = .init(.{ .o = o_dim, .i = i_dim }, .f32);
            var w_buf: zml.Buffer = try .fromBytes(self.io, self.platform, w_shape, .replicated, std.mem.sliceAsBytes(w_f32));
            errdefer w_buf.deinit();

            // Folded copy for use as an expert-MLP down matrix (rotation
            // baked into rows). 2x VRAM for E8 tensors — acceptable until
            // the decode moves in-graph.
            for (0..ou) |o| e8RotRow(w_f32[o * iu ..][0..iu], iu, .q);
            var folded: zml.Buffer = try .fromBytes(self.io, self.platform, w_shape, .replicated, std.mem.sliceAsBytes(w_f32));
            errdefer folded.deinit();
            return .{ .weights = w_buf, .scales = null, .down_folded = folded, .fmt = fmt, .i = i_dim, .o = o_dim, .gs = gs };
        }

        // PJRT sub-byte transfer (with null device layout) wants host data
        // UNPACKED, one element per byte; the transfer packs on device.
        var host: []const u8 = undefined;
        var tmp: ?[]u8 = null;
        defer if (tmp) |t| self.allocator.free(t);
        switch (fmt) {
            0 => host = weights[0 .. iu * ou * 4],
            1 => host = weights[0 .. iu * ou],
            2, 4 => { // colibri packing: low nibble first, per-row byte padding
                const rb = (iu + 1) / 2;
                const t = try self.allocator.alloc(u8, iu * ou);
                tmp = t;
                for (0..ou) |o| {
                    const row = weights[o * rb ..][0..rb];
                    for (0..iu) |i| t[o * iu + i] = if (i % 2 == 1) row[i / 2] >> 4 else row[i / 2] & 15;
                }
                host = t;
            },
            3 => { // 4 values per byte, low-first
                const rb = (iu + 3) / 4;
                const t = try self.allocator.alloc(u8, iu * ou);
                tmp = t;
                for (0..ou) |o| {
                    const row = weights[o * rb ..][0..rb];
                    for (0..iu) |i| t[o * iu + i] = (row[i / 4] >> @intCast(2 * (i % 4))) & 3;
                }
                host = t;
            },
            else => return error.UnsupportedFormat,
        }

        const w_shape: zml.Shape = .init(.{ .o = o_dim, .i = i_dim }, wdt);
        var w_buf: zml.Buffer = try .fromBytes(self.io, self.platform, w_shape, .replicated, host);
        errdefer w_buf.deinit();

        var sc_buf: ?zml.Buffer = null;
        if (try scaleShape(fmt, i_dim, o_dim, gs)) |sc_shape| {
            const sc = scales orelse return error.MissingScales;
            const sc_n: usize = @intCast(sc_shape.count());
            sc_buf = try .fromSlice(self.io, self.platform, zml.Slice.initConst(sc_shape, std.mem.sliceAsBytes(sc[0..sc_n])), .replicated);
        }
        return .{ .weights = w_buf, .scales = sc_buf, .fmt = fmt, .i = i_dim, .o = o_dim, .gs = gs };
    }

    pub fn getExe(self: *Backend, key: ExeKey) !*zml.Exe {
        const gop = try self.exes.getOrPut(key);
        if (!gop.found_existing) {
            const wdt = try weightDtype(key.fmt);
            const x_spec: zml.Tensor = .init(.{ .s = key.s, .i = key.i }, .f32);
            const quant = key.fmt != 0 and key.fmt != 6; // fmt=6 is f32 on device
            gop.value_ptr.* = switch (key.kind) {
                .group => unreachable, // compiled in groupCall (needs comptime N)
                .matmul => blk: {
                    const w_spec: zml.Tensor = .init(.{ .o = key.o, .i = key.i }, wdt);
                    if (!quant) break :blk try self.platform.compileFn(self.allocator, self.io, matmulGraph, .{ w_spec, x_spec }, .{});
                    const sc_spec: zml.Tensor = .fromShape((try scaleShape(key.fmt, key.i, key.o, key.gs)).?);
                    break :blk try self.platform.compileFn(self.allocator, self.io, matmulQGraph, .{ w_spec, sc_spec, x_spec }, .{});
                },
                .mlp => blk: {
                    const g_spec: zml.Tensor = .init(.{ .o = key.o, .i = key.i }, wdt);
                    const u_spec: zml.Tensor = .init(.{ .o = key.o, .i = key.i }, wdt);
                    const d_spec: zml.Tensor = .init(.{ .o = key.i, .i = key.o }, wdt);
                    if (!quant) break :blk try self.platform.compileFn(self.allocator, self.io, mlpGraph, .{ g_spec, u_spec, d_spec, x_spec }, .{});
                    const gsc_spec: zml.Tensor = .fromShape((try scaleShape(key.fmt, key.i, key.o, key.gs)).?);
                    const usc_spec: zml.Tensor = .fromShape((try scaleShape(key.fmt, key.i, key.o, key.gs)).?);
                    const dsc_spec: zml.Tensor = .fromShape((try scaleShape(key.fmt, key.o, key.i, key.gs)).?);
                    break :blk try self.platform.compileFn(self.allocator, self.io, mlpQGraph, .{ g_spec, gsc_spec, u_spec, usc_spec, d_spec, dsc_spec, x_spec }, .{});
                },
            };
        }
        return gop.value_ptr;
    }

    pub fn runExe(self: *Backend, exe: *zml.Exe, y: []f32, input_buffers: anytype) !void {
        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        var results = try exe.results(self.allocator);
        defer results.deinit(self.allocator);

        args.set(input_buffers);
        exe.call(args, &results);

        var out: zml.Buffer = results.get(zml.Buffer);
        defer out.deinit();
        var out_slice = try out.toSliceAlloc(self.allocator, self.io);
        defer out_slice.free(self.allocator);
        @memcpy(y, out_slice.constItems(f32)[0..y.len]);
    }

    pub fn uploadX(self: *Backend, x: []const f32, s: i64, i_dim: i64) !zml.Buffer {
        const n: usize = @intCast(s * i_dim);
        const x_shape: zml.Shape = .init(.{ .s = s, .i = i_dim }, .f32);
        return .fromSlice(self.io, self.platform, zml.Slice.initConst(x_shape, std.mem.sliceAsBytes(x[0..n])), .replicated);
    }

    /// coli_cuda_tensor_update: replace a resident tensor's contents,
    /// keeping fmt/shape (re-created rather than updated in place).
    pub fn update(self: *Backend, t: *DeviceTensor, weights: []const u8, scales: ?[]const f32) !void {
        const nt = try self.upload(weights, scales, t.fmt, t.i, t.o, t.gs);
        t.deinit();
        t.* = nt;
    }

    /// coli_cuda_matmul: y[S,O] = x[S,I] @ W^T
    pub fn matmul(self: *Backend, t: *const DeviceTensor, y: []f32, x: []const f32, s: i64) !void {
        const exe = try self.getExe(.{ .kind = .matmul, .s = s, .i = t.i, .o = t.o, .fmt = t.fmt, .gs = t.gs });
        var x_buf = try self.uploadX(x, s, t.i);
        defer x_buf.deinit();
        if (t.scales) |sc| {
            try self.runExe(exe, y, .{ t.weights, sc, x_buf });
        } else {
            try self.runExe(exe, y, .{ t.weights, x_buf });
        }
    }

    /// coli_cuda_expert_mlp: y[S,D] = down(silu(gate(x)) * up(x)), fused.
    /// gate/up/down must share fmt and gs (they do in colibri's expert files).
    pub fn expertMlp(self: *Backend, gate: *const DeviceTensor, up: *const DeviceTensor, down: *const DeviceTensor, y: []f32, x: []const f32, s: i64) !void {
        if (gate.fmt != up.fmt or gate.fmt != down.fmt) return error.MixedFormats;
        const exe = try self.getExe(.{ .kind = .mlp, .s = s, .i = gate.i, .o = gate.o, .fmt = gate.fmt, .gs = gate.gs });
        var x_buf = try self.uploadX(x, s, gate.i);
        defer x_buf.deinit();
        if (gate.scales != null) {
            try self.runExe(exe, y, .{ gate.weights, gate.scales.?, up.weights, up.scales.?, down.weights, down.scales.?, x_buf });
        } else {
            // fmt=6: the down matmul consumes the rotation-folded rows
            // (device kernels instead rotate the silu product — same math).
            const dbuf = if (down.fmt == 6) (down.down_folded orelse return error.MissingFold) else down.weights;
            try self.runExe(exe, y, .{ gate.weights, up.weights, dbuf, x_buf });
        }
    }

    /// coli_cuda_expert_group: consecutive [D] row blocks per expert, in call
    /// order. Single fused executable when every expert takes one row and
    /// formats/shapes are uniform (the decode pattern); loop otherwise.
    pub fn expertGroup(self: *Backend, gates: []const *const DeviceTensor, ups: []const *const DeviceTensor, downs: []const *const DeviceTensor, rows: []const i32, y: []f32, x: []const f32) !void {
        if (fusableGroup(gates, ups, downs, rows)) {
            switch (gates.len) {
                inline 1...16 => |n| return self.groupCall(n, gates, ups, downs, y, x),
                else => {},
            }
        }
        var off: usize = 0;
        for (gates, ups, downs, rows) |g, u, d, r| {
            const n: usize = @intCast(r);
            const dim: usize = @intCast(g.i);
            try self.expertMlp(g, u, d, y[off * dim ..][0 .. n * dim], x[off * dim ..][0 .. n * dim], r);
            off += n;
        }
    }

    fn fusableGroup(gates: []const *const DeviceTensor, ups: []const *const DeviceTensor, downs: []const *const DeviceTensor, rows: []const i32) bool {
        if (gates.len == 0 or gates.len > 16) return false;
        const g0 = gates[0];
        if (g0.fmt == 0) return false;
        for (gates, ups, downs, rows) |g, u, d, r| {
            if (r != 1) return false;
            for ([_]*const DeviceTensor{ g, u }) |t| {
                if (t.fmt != g0.fmt or t.gs != g0.gs or t.i != g0.i or t.o != g0.o or t.scales == null) return false;
            }
            if (d.fmt != g0.fmt or d.gs != g0.gs or d.i != g0.o or d.o != g0.i or d.scales == null) return false;
        }
        return true;
    }

    /// An in-flight fused group call: launched on the device, output not yet
    /// read. Backs the coli_cuda_expert_group_issue/take async contract.
    pub const PendingGroup = struct {
        args: zml.Exe.Arguments,
        results: zml.Exe.Results,
        x_buf: zml.Buffer,
        n: usize,
    };

    fn groupLaunch(self: *Backend, comptime N: usize, gates: []const *const DeviceTensor, ups: []const *const DeviceTensor, downs: []const *const DeviceTensor, x: []const f32) !PendingGroup {
        const g0 = gates[0];
        const key: ExeKey = .{ .kind = .group, .s = N, .i = g0.i, .o = g0.o, .fmt = g0.fmt, .gs = g0.gs };
        const gop = try self.exes.getOrPut(key);
        if (!gop.found_existing) {
            const wdt = try weightDtype(key.fmt);
            var model: Group(N) = undefined;
            for (0..N) |e| {
                model.gates[e] = .init(.{ .o = key.o, .i = key.i }, wdt);
                model.gscs[e] = .fromShape((try scaleShape(key.fmt, key.i, key.o, key.gs)).?);
                model.ups[e] = .init(.{ .o = key.o, .i = key.i }, wdt);
                model.uscs[e] = .fromShape((try scaleShape(key.fmt, key.i, key.o, key.gs)).?);
                model.downs[e] = .init(.{ .o = key.i, .i = key.o }, wdt);
                model.dscs[e] = .fromShape((try scaleShape(key.fmt, key.o, key.i, key.gs)).?);
            }
            const x_spec: zml.Tensor = .init(.{ .s = @as(i64, N), .i = key.i }, .f32);
            gop.value_ptr.* = try self.platform.compile(self.allocator, self.io, model, .forward, .{x_spec}, .{});
        }
        const exe = gop.value_ptr;

        var x_buf = try self.uploadX(x, N, g0.i);
        errdefer x_buf.deinit();
        var args = try exe.args(self.allocator);
        errdefer args.deinit(self.allocator);
        var results = try exe.results(self.allocator);
        errdefer results.deinit(self.allocator);

        var bufs: zml.Bufferized(Group(N)) = undefined;
        for (0..N) |e| {
            bufs.gates[e] = gates[e].weights;
            bufs.gscs[e] = gates[e].scales.?;
            bufs.ups[e] = ups[e].weights;
            bufs.uscs[e] = ups[e].scales.?;
            bufs.downs[e] = downs[e].weights;
            bufs.dscs[e] = downs[e].scales.?;
        }
        args.set(.{ bufs, x_buf });
        exe.call(args, &results); // launches; no host sync until take
        return .{ .args = args, .results = results, .x_buf = x_buf, .n = N * @as(usize, @intCast(g0.i)) };
    }

    /// Complete an in-flight group: sync, copy the rows out, free resources.
    pub fn pendingTake(self: *Backend, p: *PendingGroup, y: []f32) !void {
        defer {
            p.x_buf.deinit();
            p.args.deinit(self.allocator);
            p.results.deinit(self.allocator);
        }
        var out: zml.Buffer = p.results.get(zml.Buffer);
        defer out.deinit();
        var out_slice = try out.toSliceAlloc(self.allocator, self.io);
        defer out_slice.free(self.allocator);
        @memcpy(y, out_slice.constItems(f32)[0..y.len]);
    }

    /// Async entry: launch a fused group and return the pending handle, or
    /// null when the group shape can't take the fused path (caller falls
    /// back to the synchronous loop).
    pub fn expertGroupIssue(self: *Backend, gates: []const *const DeviceTensor, ups: []const *const DeviceTensor, downs: []const *const DeviceTensor, rows: []const i32, x: []const f32) !?PendingGroup {
        if (!fusableGroup(gates, ups, downs, rows)) return null;
        switch (gates.len) {
            inline 1...16 => |n| return try self.groupLaunch(n, gates, ups, downs, x),
            else => return null,
        }
    }

    fn groupCall(self: *Backend, comptime N: usize, gates: []const *const DeviceTensor, ups: []const *const DeviceTensor, downs: []const *const DeviceTensor, y: []f32, x: []const f32) !void {
        var p = try self.groupLaunch(N, gates, ups, downs, x);
        try self.pendingTake(&p, y);
    }
};
