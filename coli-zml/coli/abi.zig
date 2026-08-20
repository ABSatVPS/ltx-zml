//! The coli_cuda_* C ABI (c/backend_cuda.h) implemented on the ZML backend.
//! Contract notes:
//! - Every function returns 1 on success, 0 on failure; Zig errors never
//!   cross the boundary.
//! - `ColiCudaTensor` handles are opaque pointers to `ColiTensor`.
//! - tensor bytes are accounted in QUANTIZED size (weights + scale arrays),
//!   mirroring upstream — the conformance test asserts exact totals.
//! - `pipe_*`, batch attention, and w4a16 entry points are honest stubs
//!   returning 0: the engine falls back to its CPU path.
//! - `COLI_GPU_FAIL_AFTER=N` fails GPU ops once N successful ops have run.
const std = @import("std");
const zml = @import("zml");
const coli = @import("backend.zig");

const alloc = std.heap.c_allocator;

const MAX_DEVICES = 16;

const ColiTensor = struct {
    dt: coli.DeviceTensor,
    bytes: usize,
    device: c_int,
    host_w: ?[]f32 = null, // lazy f32 copy for the host attention path
};

const State = struct {
    stio: std.Io.Threaded,
    backend: coli.Backend,
    devices: [MAX_DEVICES]c_int,
    ndev: c_int,
    count_per_dev: [MAX_DEVICES]usize,
    bytes_per_dev: [MAX_DEVICES]usize,
    grp_calls: u64 = 0,
    grp_experts: u64 = 0,
    grp_rows: u64 = 0,
    ops: u64 = 0,
    take_buf: ?[]f32 = null,
    take_valid: bool = false,
    pending: ?coli.Backend.PendingGroup = null,
};
var g: ?*State = null;

fn devSlot(device: c_int) ?usize {
    const s = g orelse return null;
    for (s.devices[0..@intCast(s.ndev)], 0..) |d, i| {
        if (d == device) return i;
    }
    return null;
}

fn failInjected() bool {
    const p = std.c.getenv("COLI_GPU_FAIL_AFTER") orelse return false;
    const s = std.mem.span(@as([*:0]const u8, @ptrCast(p)));
    const n = std.fmt.parseInt(u64, s, 10) catch return false;
    return g.?.ops >= n;
}

fn quantBytes(fmt: c_int, i_dim: usize, o_dim: usize, gs: c_int) usize {
    return switch (fmt) {
        0 => i_dim * o_dim * 4,
        1 => i_dim * o_dim + o_dim * 4,
        2 => ((i_dim + 1) / 2) * o_dim + o_dim * 4,
        3 => ((i_dim + 3) / 4) * o_dim + o_dim * 4,
        4 => ((i_dim + 1) / 2) * o_dim + o_dim * 4 * (if (gs > 0) (i_dim + @as(usize, @intCast(gs)) - 1) / @as(usize, @intCast(gs)) else 1),
        6 => coli.e8RowBytes(i_dim) * o_dim,
        else => 0,
    };
}

fn weightByteLen(fmt: c_int, i_dim: usize, o_dim: usize) usize {
    return switch (fmt) {
        0 => i_dim * o_dim * 4,
        1 => i_dim * o_dim,
        2, 4 => ((i_dim + 1) / 2) * o_dim,
        3 => ((i_dim + 3) / 4) * o_dim,
        6 => coli.e8RowBytes(i_dim) * o_dim,
        else => 0,
    };
}

fn scaleLen(fmt: c_int, i_dim: usize, o_dim: usize, gs: c_int) usize {
    return switch (fmt) {
        1, 2, 3 => o_dim,
        4 => if (gs > 0) o_dim * ((i_dim + @as(usize, @intCast(gs)) - 1) / @as(usize, @intCast(gs))) else o_dim,
        else => 0,
    };
}

// ---- lifecycle ------------------------------------------------------------

export fn coli_cuda_init(devices: [*c]const c_int, count: c_int) c_int {
    if (g != null) return 1;
    if (count <= 0 or count > MAX_DEVICES) return 0;
    const st = alloc.create(State) catch return 0;
    st.* = .{
        .stio = .init_single_threaded,
        .backend = undefined,
        .devices = @splat(0),
        .ndev = count,
        .count_per_dev = @splat(0),
        .bytes_per_dev = @splat(0),
    };
    st.backend = coli.Backend.init(alloc, st.stio.io()) catch {
        alloc.destroy(st);
        return 0;
    };
    const avail: c_int = @intCast(st.backend.platform.devices.len);
    for (0..@intCast(count)) |i| {
        if (devices[i] < 0 or devices[i] >= avail) {
            alloc.destroy(st);
            return 0;
        }
        st.devices[i] = devices[i];
    }
    g = st;
    return 1;
}

export fn coli_cuda_shutdown() void {
    const s = g orelse return;
    drainPending(s);
    if (s.take_buf) |b| alloc.free(b);
    g = null;
    alloc.destroy(s);
}

export fn coli_cuda_device_count() c_int {
    return if (g) |s| s.ndev else 0;
}

export fn coli_cuda_device_at(index: c_int) c_int {
    const s = g orelse return -1;
    if (index < 0 or index >= s.ndev) return -1;
    return s.devices[@intCast(index)];
}

export fn coli_cuda_mem_info(device: c_int, free_bytes: [*c]usize, total_bytes: [*c]usize) c_int {
    const s = g orelse return 0;
    const slot = devSlot(device) orelse return 0;
    const total: usize = 16 << 30; // TODO: query PJRT memory stats
    if (total_bytes != null) total_bytes.* = total;
    if (free_bytes != null) free_bytes.* = total -| s.bytes_per_dev[slot];
    return 1;
}

export fn coli_cuda_device_integrated(device: c_int) c_int {
    _ = device;
    return 0;
}

export fn coli_cuda_stats(device: c_int, tensor_count: [*c]usize, tensor_bytes: [*c]usize) void {
    var count: usize = 0;
    var bytes: usize = 0;
    if (g) |s| {
        if (device < 0) {
            for (s.count_per_dev[0..@intCast(s.ndev)], s.bytes_per_dev[0..@intCast(s.ndev)]) |c, b| {
                count += c;
                bytes += b;
            }
        } else if (devSlot(device)) |slot| {
            count = s.count_per_dev[slot];
            bytes = s.bytes_per_dev[slot];
        }
    }
    if (tensor_count != null) tensor_count.* = count;
    if (tensor_bytes != null) tensor_bytes.* = bytes;
}

export fn coli_cuda_group_stats(calls: [*c]u64, experts: [*c]u64, rows: [*c]u64, h2d_ms: [*c]f64, kernel_ms: [*c]f64, d2h_ms: [*c]f64) void {
    const s = g orelse return;
    if (calls != null) calls.* = s.grp_calls;
    if (experts != null) experts.* = s.grp_experts;
    if (rows != null) rows.* = s.grp_rows;
    if (h2d_ms != null) h2d_ms.* = 0;
    if (kernel_ms != null) kernel_ms.* = 0;
    if (d2h_ms != null) d2h_ms.* = 0;
}

export fn coli_cuda_e8_set_grid(grid: ?*const anyopaque) c_int {
    const s = g orelse return 0;
    const p = grid orelse return 0;
    s.backend.setE8Grid(@ptrCast(@alignCast(p)));
    return 1;
}

// ---- tensors --------------------------------------------------------------

fn doUpload(tensor: *?*ColiTensor, weights: ?*const anyopaque, scales: [*c]const f32, fmt: c_int, I: c_int, O: c_int, device: c_int, gs: c_int) c_int {
    const s = g orelse return 0;
    if (tensor.* != null) return 0;
    const wp = weights orelse return 0;
    if (I <= 0 or O <= 0) return 0;
    const slot = devSlot(device) orelse return 0;
    const iu: usize = @intCast(I);
    const ou: usize = @intCast(O);
    const wlen = weightByteLen(fmt, iu, ou);
    if (wlen == 0) return 0;
    const slen = scaleLen(fmt, iu, ou, gs);
    var sc: ?[]const f32 = null;
    if (slen > 0) {
        if (scales == null) return 0;
        sc = scales[0..slen];
    }
    const wbytes: [*]const u8 = @ptrCast(wp);
    const dt = s.backend.upload(wbytes[0..wlen], sc, fmt, I, O, gs) catch return 0;
    const t = alloc.create(ColiTensor) catch {
        var d = dt;
        d.deinit();
        return 0;
    };
    t.* = .{ .dt = dt, .bytes = quantBytes(fmt, iu, ou, gs), .device = device };
    s.count_per_dev[slot] += 1;
    s.bytes_per_dev[slot] += t.bytes;
    tensor.* = t;
    return 1;
}

export fn coli_cuda_tensor_upload_g(tensor: *?*ColiTensor, weights: ?*const anyopaque, scales: [*c]const f32, fmt: c_int, I: c_int, O: c_int, device: c_int, gs: c_int) c_int {
    return doUpload(tensor, weights, scales, fmt, I, O, device, gs);
}

export fn coli_cuda_tensor_upload(tensor: *?*ColiTensor, weights: ?*const anyopaque, scales: [*c]const f32, fmt: c_int, I: c_int, O: c_int, device: c_int) c_int {
    return doUpload(tensor, weights, scales, fmt, I, O, device, 0);
}

export fn coli_cuda_tensor_update(tensor: ?*ColiTensor, weights: ?*const anyopaque, scales: [*c]const f32) c_int {
    const s = g orelse return 0;
    const t = tensor orelse return 0;
    const wp = weights orelse return 0;
    const iu: usize = @intCast(t.dt.i);
    const ou: usize = @intCast(t.dt.o);
    const wlen = weightByteLen(t.dt.fmt, iu, ou);
    const slen = scaleLen(t.dt.fmt, iu, ou, t.dt.gs);
    var sc: ?[]const f32 = null;
    if (slen > 0) {
        if (scales == null) return 0;
        sc = scales[0..slen];
    }
    const wbytes: [*]const u8 = @ptrCast(wp);
    s.backend.update(&t.dt, wbytes[0..wlen], sc) catch return 0;
    if (t.host_w) |h| {
        alloc.free(h);
        t.host_w = null;
    }
    return 1;
}

export fn coli_cuda_tensor_free(tensor: ?*ColiTensor) void {
    const s = g orelse return;
    const t = tensor orelse return;
    if (devSlot(t.device)) |slot| {
        s.count_per_dev[slot] -= 1;
        s.bytes_per_dev[slot] -= t.bytes;
    }
    if (t.host_w) |h| alloc.free(h);
    t.dt.deinit();
    alloc.destroy(t);
}

export fn coli_cuda_tensor_bytes(tensor: ?*const ColiTensor) usize {
    return if (tensor) |t| t.bytes else 0;
}

export fn coli_cuda_tensor_device(tensor: ?*const ColiTensor) c_int {
    return if (tensor) |t| t.device else -1;
}

// ---- compute --------------------------------------------------------------

export fn coli_cuda_matmul(tensor: *?*ColiTensor, y: [*c]f32, x: [*c]const f32, weights: ?*const anyopaque, scales: [*c]const f32, fmt: c_int, S: c_int, I: c_int, O: c_int, device: c_int, gs: c_int) c_int {
    const s = g orelse return 0;
    if (y == null or x == null or S <= 0) return 0;
    if (failInjected()) return 0;
    if (tensor.* == null) {
        if (doUpload(tensor, weights, scales, fmt, I, O, device, gs) == 0) return 0;
    }
    const t = tensor.*.?;
    const su: usize = @intCast(S);
    s.backend.matmul(&t.dt, y[0 .. su * @as(usize, @intCast(t.dt.o))], x[0 .. su * @as(usize, @intCast(t.dt.i))], S) catch return 0;
    s.ops += 1;
    return 1;
}

export fn coli_cuda_expert_mlp(gate: ?*ColiTensor, up: ?*ColiTensor, down: ?*ColiTensor, y: [*c]f32, x: [*c]const f32, S: c_int) c_int {
    const s = g orelse return 0;
    const gt = gate orelse return 0;
    const ut = up orelse return 0;
    const dt = down orelse return 0;
    if (y == null or x == null or S <= 0) return 0;
    if (failInjected()) return 0;
    const n = @as(usize, @intCast(S)) * @as(usize, @intCast(gt.dt.i));
    s.backend.expertMlp(&gt.dt, &ut.dt, &dt.dt, y[0..n], x[0..n], S) catch return 0;
    s.ops += 1;
    return 1;
}

fn doGroup(gates: [*c]const ?*ColiTensor, ups: [*c]const ?*ColiTensor, downs: [*c]const ?*ColiTensor, rows: [*c]const c_int, count: c_int, y: []f32, x: [*c]const f32) c_int {
    const s = g orelse return 0;
    if (count <= 0 or x == null) return 0;
    if (failInjected()) return 0;
    drainPending(s);
    const cu: usize = @intCast(count);
    const ga = alloc.alloc(*const coli.DeviceTensor, cu * 3) catch return 0;
    defer alloc.free(ga);
    const ra = alloc.alloc(i32, cu) catch return 0;
    defer alloc.free(ra);
    var total_rows: usize = 0;
    for (0..cu) |i| {
        const gt = gates[i] orelse return 0;
        ga[i] = &gt.dt;
        ga[cu + i] = &(ups[i] orelse return 0).dt;
        ga[2 * cu + i] = &(downs[i] orelse return 0).dt;
        ra[i] = rows[i];
        total_rows += @intCast(rows[i]);
    }
    const dim: usize = @intCast(ga[0].i);
    s.backend.expertGroup(ga[0..cu], ga[cu .. 2 * cu], ga[2 * cu .. 3 * cu], ra, y[0 .. total_rows * dim], x[0 .. total_rows * dim]) catch return 0;
    s.grp_calls += 1;
    s.grp_experts += cu;
    s.grp_rows += total_rows;
    s.ops += 1;
    return 1;
}

export fn coli_cuda_expert_group(gates: [*c]const ?*ColiTensor, ups: [*c]const ?*ColiTensor, downs: [*c]const ?*ColiTensor, rows: [*c]const c_int, count: c_int, y: [*c]f32, x: [*c]const f32) c_int {
    if (y == null) return 0;
    var total: usize = 0;
    for (0..@intCast(count)) |i| total += @intCast(rows[i]);
    const gt = gates[0] orelse return 0;
    return doGroup(gates, ups, downs, rows, count, y[0 .. total * @as(usize, @intCast(gt.dt.i))], x);
}

/// Complete (and discard the need for) any outstanding async group. The ABI
/// allows one outstanding issue per device; a new issue drains the old one.
fn drainPending(s: *State) void {
    if (s.pending) |*p| {
        const n = p.n;
        if (s.take_buf != null and s.take_buf.?.len >= n) {
            s.backend.pendingTake(p, s.take_buf.?[0..n]) catch {};
        }
        s.pending = null;
    }
}

export fn coli_cuda_expert_group_issue(gates: [*c]const ?*ColiTensor, ups: [*c]const ?*ColiTensor, downs: [*c]const ?*ColiTensor, rows: [*c]const c_int, count: c_int, x: [*c]const f32) c_int {
    const s = g orelse return 0;
    if (count <= 0 or x == null) return 0;
    if (failInjected()) return 0;
    drainPending(s);
    s.take_valid = false;

    const cu: usize = @intCast(count);
    var total: usize = 0;
    const ga = alloc.alloc(*const coli.DeviceTensor, cu * 3) catch return 0;
    defer alloc.free(ga);
    const ra = alloc.alloc(i32, cu) catch return 0;
    defer alloc.free(ra);
    for (0..cu) |i| {
        ga[i] = &(gates[i] orelse return 0).dt;
        ga[cu + i] = &(ups[i] orelse return 0).dt;
        ga[2 * cu + i] = &(downs[i] orelse return 0).dt;
        ra[i] = rows[i];
        total += @intCast(rows[i]);
    }
    const n = total * @as(usize, @intCast(ga[0].i));
    if (s.take_buf == null or s.take_buf.?.len < n) {
        if (s.take_buf) |b| alloc.free(b);
        s.take_buf = alloc.alloc(f32, n) catch return 0;
    }

    // Async fast path: launch the fused group and return without syncing —
    // the engine overlaps CPU work until take. Non-fusable groups run the
    // synchronous loop (same numerics as the sync entry point either way).
    if (s.backend.expertGroupIssue(ga[0..cu], ga[cu .. 2 * cu], ga[2 * cu .. 3 * cu], ra, x[0..n]) catch null) |p| {
        s.pending = p;
    } else {
        s.backend.expertGroup(ga[0..cu], ga[cu .. 2 * cu], ga[2 * cu .. 3 * cu], ra, s.take_buf.?[0..n], x[0..n]) catch return 0;
        s.take_valid = true;
    }
    s.grp_calls += 1;
    s.grp_experts += cu;
    s.grp_rows += total;
    s.ops += 1;
    return 1;
}

export fn coli_cuda_expert_group_take(device: c_int) [*c]const f32 {
    _ = device;
    const s = g orelse return null;
    if (s.pending) |*p| {
        const n = p.n;
        s.backend.pendingTake(p, s.take_buf.?[0..n]) catch {
            s.pending = null;
            return null;
        };
        s.pending = null;
        s.take_valid = true;
    }
    if (!s.take_valid) return null;
    return s.take_buf.?.ptr;
}

// ---- decode-time MLA attention (host math for now; see EXPERIMENT.md) -----

fn hostWeights(t: *ColiTensor) ?[]const f32 {
    const s = g orelse return null;
    if (t.host_w == null) {
        var slice = t.dt.weights.toSliceAlloc(alloc, s.backend.io) catch return null;
        defer slice.free(alloc);
        const items = slice.constItems(f32);
        const copy = alloc.alloc(f32, items.len) catch return null;
        @memcpy(copy, items);
        t.host_w = copy;
    }
    return t.host_w;
}

export fn coli_cuda_attention_absorb(kv_b: ?*ColiTensor, ctx: [*c]f32, q: [*c]const f32, latent: [*c]const f32, rope: [*c]const f32, H: c_int, Q: c_int, R: c_int, V: c_int, K: c_int, T: c_int, attention_scale: f32) c_int {
    const s = g orelse return 0;
    _ = s;
    const kt = kv_b orelse return 0;
    if (ctx == null or q == null or latent == null or rope == null) return 0;
    if (H <= 0 or Q < 0 or V <= 0 or K <= 0 or T <= 0) return 0;
    if (failInjected()) return 0;
    const w = hostWeights(kt) orelse return 0;
    const hh: usize = @intCast(H);
    const qq: usize = @intCast(Q);
    const rr: usize = @intCast(R);
    const vv: usize = @intCast(V);
    const kk: usize = @intCast(K);
    const tt: usize = @intCast(T);

    const q_abs = alloc.alloc(f32, kk) catch return 0;
    defer alloc.free(q_abs);
    const score = alloc.alloc(f32, tt) catch return 0;
    defer alloc.free(score);

    for (0..hh) |h| {
        const qh = q[h * (qq + rr) ..][0 .. qq + rr];
        @memset(q_abs, 0);
        for (0..qq) |qi| {
            const wrow = w[(h * (qq + vv) + qi) * kk ..][0..kk];
            for (0..kk) |k| q_abs[k] += qh[qi] * wrow[k];
        }
        var mx: f32 = -std.math.floatMax(f32);
        for (0..tt) |t| {
            var sc: f32 = 0;
            for (0..kk) |k| sc += q_abs[k] * latent[t * kk + k];
            for (0..rr) |r| sc += qh[qq + r] * rope[t * rr + r];
            sc *= attention_scale;
            score[t] = sc;
            mx = @max(mx, sc);
        }
        var z: f32 = 0;
        for (score) |*sc| {
            sc.* = @exp(sc.* - mx);
            z += sc.*;
        }
        for (score) |*sc| sc.* /= z;
        for (0..vv) |v| {
            const wrow = w[(h * (qq + vv) + qq + v) * kk ..][0..kk];
            var acc: f32 = 0;
            for (0..tt) |t| {
                var lv: f32 = 0;
                for (0..kk) |k| lv += wrow[k] * latent[t * kk + k];
                acc += score[t] * lv;
            }
            ctx[h * vv + v] = acc;
        }
    }
    return 1;
}

// ---- honest stubs: engine falls back to its CPU implementations -----------

export fn coli_cuda_shared_mlp_w4a16(gate: ?*ColiTensor, up: ?*ColiTensor, down: ?*ColiTensor, y: [*c]f32, x: [*c]const f32, S: c_int) c_int {
    _ = gate;
    _ = up;
    _ = down;
    _ = y;
    _ = x;
    _ = S;
    return 0;
}

export fn coli_cuda_attention_absorb_batch(kv_b: ?*ColiTensor, ctx: [*c]f32, q: [*c]const f32, latent: [*c]const f32, rope: [*c]const f32, S: c_int, H: c_int, Q: c_int, R: c_int, V: c_int, K: c_int, T: c_int, attention_scale: f32) c_int {
    _ = kv_b;
    _ = ctx;
    _ = q;
    _ = latent;
    _ = rope;
    _ = S;
    _ = H;
    _ = Q;
    _ = R;
    _ = V;
    _ = K;
    _ = T;
    _ = attention_scale;
    return 0;
}

export fn coli_cuda_attention_project_batch(kv_b: ?*ColiTensor, o_proj: ?*ColiTensor, out: [*c]f32, q: [*c]const f32, latent: [*c]const f32, rope: [*c]const f32, S: c_int, H: c_int, Q: c_int, R: c_int, V: c_int, K: c_int, T: c_int, attention_scale: f32) c_int {
    _ = kv_b;
    _ = o_proj;
    _ = out;
    _ = q;
    _ = latent;
    _ = rope;
    _ = S;
    _ = H;
    _ = Q;
    _ = R;
    _ = V;
    _ = K;
    _ = T;
    _ = attention_scale;
    return 0;
}

export fn coli_cuda_attention_project_ragged(kv_b: ?*ColiTensor, o_proj: ?*ColiTensor, out: [*c]f32, q: [*c]const f32, keys: [*c]const ?*const anyopaque, latent: [*c]const [*c]const f32, rope: [*c]const [*c]const f32, lengths: [*c]const c_int, S: c_int, H: c_int, Q: c_int, R: c_int, V: c_int, K: c_int, max_t: c_int, attention_scale: f32) c_int {
    _ = kv_b;
    _ = o_proj;
    _ = out;
    _ = q;
    _ = keys;
    _ = latent;
    _ = rope;
    _ = lengths;
    _ = S;
    _ = H;
    _ = Q;
    _ = R;
    _ = V;
    _ = K;
    _ = max_t;
    _ = attention_scale;
    return 0;
}

export fn coli_cuda_pipe_scratch(device: c_int, slot: c_int, bytes: usize) [*c]f32 {
    _ = device;
    _ = slot;
    _ = bytes;
    return null;
}

export fn coli_cuda_pipe_alloc(device: c_int, bytes: usize) ?*anyopaque {
    _ = device;
    _ = bytes;
    return null;
}

export fn coli_cuda_pipe_free(device: c_int, p: ?*anyopaque) void {
    _ = device;
    _ = p;
}

export fn coli_cuda_pipe_upload(device: c_int, dst: ?*anyopaque, src: ?*const anyopaque, bytes: usize) c_int {
    _ = device;
    _ = dst;
    _ = src;
    _ = bytes;
    return 0;
}

export fn coli_cuda_pipe_download(device: c_int, src: ?*const anyopaque, dst: ?*anyopaque, bytes: usize) c_int {
    _ = device;
    _ = src;
    _ = dst;
    _ = bytes;
    return 0;
}

export fn coli_cuda_pipe_rmsnorm(device: c_int, y_dev: [*c]f32, x_dev: [*c]const f32, w_dev: [*c]const f32, S: c_int, D: c_int, eps: f32) c_int {
    _ = device;
    _ = y_dev;
    _ = x_dev;
    _ = w_dev;
    _ = S;
    _ = D;
    _ = eps;
    return 0;
}

export fn coli_cuda_pipe_rope(device: c_int, v_dev: [*c]f32, pos_dev: [*c]const c_int, rows: c_int, stride: c_int, offset: c_int, R: c_int, heads: c_int, theta: f32) c_int {
    _ = device;
    _ = v_dev;
    _ = pos_dev;
    _ = rows;
    _ = stride;
    _ = offset;
    _ = R;
    _ = heads;
    _ = theta;
    return 0;
}

export fn coli_cuda_pipe_silu_mul(device: c_int, gate_dev: [*c]f32, up_dev: [*c]const f32, n: usize) c_int {
    _ = device;
    _ = gate_dev;
    _ = up_dev;
    _ = n;
    return 0;
}

export fn coli_cuda_pipe_add(device: c_int, x_dev: [*c]f32, t_dev: [*c]const f32, n: usize) c_int {
    _ = device;
    _ = x_dev;
    _ = t_dev;
    _ = n;
    return 0;
}

export fn coli_cuda_pipe_rows_add(device: c_int, x_dev: [*c]f32, partial_dev: [*c]const f32, rows_dev: [*c]const c_int, nrows: c_int, D: c_int) c_int {
    _ = device;
    _ = x_dev;
    _ = partial_dev;
    _ = rows_dev;
    _ = nrows;
    _ = D;
    return 0;
}

export fn coli_cuda_pipe_gemm(t: ?*ColiTensor, y_dev: [*c]f32, x_dev: [*c]const f32, S: c_int) c_int {
    _ = t;
    _ = y_dev;
    _ = x_dev;
    _ = S;
    return 0;
}

export fn coli_cuda_pipe_rmsnorm_s(device: c_int, y_dev: [*c]f32, x_dev: [*c]const f32, w_dev: [*c]const f32, S: c_int, D: c_int, eps: f32, xstride: c_int, ystride: c_int) c_int {
    _ = device;
    _ = y_dev;
    _ = x_dev;
    _ = w_dev;
    _ = S;
    _ = D;
    _ = eps;
    _ = xstride;
    _ = ystride;
    return 0;
}

export fn coli_cuda_pipe_rope_base(device: c_int, v_dev: [*c]f32, pos_base: c_int, rows: c_int, stride: c_int, offset: c_int, R: c_int, heads: c_int, theta: f32) c_int {
    _ = device;
    _ = v_dev;
    _ = pos_base;
    _ = rows;
    _ = stride;
    _ = offset;
    _ = R;
    _ = heads;
    _ = theta;
    return 0;
}

export fn coli_cuda_expert_group_resident_issue(gates: [*c]const ?*ColiTensor, ups: [*c]const ?*ColiTensor, downs: [*c]const ?*ColiTensor, weights: [*c]const f32, count: c_int, home_device: c_int, x_src_dev: [*c]const f32, partial_slot_dev: [*c]f32) c_int {
    _ = gates;
    _ = ups;
    _ = downs;
    _ = weights;
    _ = count;
    _ = home_device;
    _ = x_src_dev;
    _ = partial_slot_dev;
    return 0;
}

export fn coli_cuda_expert_group_resident_take(home_device: c_int, devices: [*c]const c_int, n_issued: c_int, slots_dev: [*c]f32, acc_dev: [*c]f32, D: c_int) c_int {
    _ = home_device;
    _ = devices;
    _ = n_issued;
    _ = slots_dev;
    _ = acc_dev;
    _ = D;
    return 0;
}

export fn coli_cuda_pipe_router(device: c_int, x_dev: [*c]const f32, rw_dev: ?*const anyopaque, rb_dev: ?*const anyopaque, D: c_int, E: c_int, Ksel: c_int, topp: f32, norm_topk: c_int, routed_scale: f32, idx_host: [*c]c_int, w_host: [*c]f32, keff_host: [*c]c_int) c_int {
    _ = device;
    _ = x_dev;
    _ = rw_dev;
    _ = rb_dev;
    _ = D;
    _ = E;
    _ = Ksel;
    _ = topp;
    _ = norm_topk;
    _ = routed_scale;
    _ = idx_host;
    _ = w_host;
    _ = keff_host;
    return 0;
}

export fn coli_cuda_pipe_copy2d(device: c_int, dst: [*c]f32, dpitch: c_int, src: [*c]const f32, spitch: c_int, width: c_int, height: c_int) c_int {
    _ = device;
    _ = dst;
    _ = dpitch;
    _ = src;
    _ = spitch;
    _ = width;
    _ = height;
    return 0;
}

export fn coli_cuda_attention_project_batch_dev(kv_b: ?*ColiTensor, o_proj: ?*ColiTensor, out: [*c]f32, q_dev: [*c]const f32, latent_dev: [*c]const f32, rope_dev: [*c]const f32, S: c_int, H: c_int, Q: c_int, R: c_int, V: c_int, K: c_int, T: c_int, scale: f32) c_int {
    _ = kv_b;
    _ = o_proj;
    _ = out;
    _ = q_dev;
    _ = latent_dev;
    _ = rope_dev;
    _ = S;
    _ = H;
    _ = Q;
    _ = R;
    _ = V;
    _ = K;
    _ = T;
    _ = scale;
    return 0;
}

export fn coli_cuda_attention_absorb_batch_dev(kv_b_shard: ?*ColiTensor, ctx_dev: [*c]f32, q_dev: [*c]const f32, latent_dev: [*c]const f32, rope_dev: [*c]const f32, S: c_int, H: c_int, Q: c_int, R: c_int, V: c_int, K: c_int, T: c_int, scale: f32) c_int {
    _ = kv_b_shard;
    _ = ctx_dev;
    _ = q_dev;
    _ = latent_dev;
    _ = rope_dev;
    _ = S;
    _ = H;
    _ = Q;
    _ = R;
    _ = V;
    _ = K;
    _ = T;
    _ = scale;
    return 0;
}

export fn coli_cuda_attention_absorb_kvdev(kv_b: ?*ColiTensor, ctx: [*c]f32, q: [*c]const f32, latent_dev: [*c]const f32, rope_dev: [*c]const f32, H: c_int, Q: c_int, R: c_int, V: c_int, K: c_int, T: c_int, scale: f32) c_int {
    _ = kv_b;
    _ = ctx;
    _ = q;
    _ = latent_dev;
    _ = rope_dev;
    _ = H;
    _ = Q;
    _ = R;
    _ = V;
    _ = K;
    _ = T;
    _ = scale;
    return 0;
}

export fn coli_cuda_pipe_peer_copy(dst_dev: c_int, dst: [*c]f32, src_dev: c_int, src: [*c]const f32, bytes: usize) c_int {
    _ = dst_dev;
    _ = dst;
    _ = src_dev;
    _ = src;
    _ = bytes;
    return 0;
}

export fn coli_cuda_attention_project_batch_dev_out(kv_b: ?*ColiTensor, o_proj: ?*ColiTensor, out_dev: [*c]f32, q_dev: [*c]const f32, latent_dev: [*c]const f32, rope_dev: [*c]const f32, S: c_int, H: c_int, Q: c_int, R: c_int, V: c_int, K: c_int, T: c_int, scale: f32) c_int {
    _ = kv_b;
    _ = o_proj;
    _ = out_dev;
    _ = q_dev;
    _ = latent_dev;
    _ = rope_dev;
    _ = S;
    _ = H;
    _ = Q;
    _ = R;
    _ = V;
    _ = K;
    _ = T;
    _ = scale;
    return 0;
}

export fn coli_cuda_pipe_sync(device: c_int) c_int {
    _ = device;
    return 0;
}
