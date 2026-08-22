//! Conformance gates for the E2E-core parts (ltx/core_parts.zig) against
//! the f64 torch oracle bundle (tools/make_core_bundle.py). Pre-registered
//! notebook 2026-08-21, late morning: H-CORE-1..4. Seven gates:
//! sinusoid, embedded_timestep, ts9, pts2, patchify, output tail, and the
//! wrong-norm control pair (engine RMS-tail must MATCH the oracle's
//! RMS-tail dump and must FAIL against the true tail — gate teeth).
const std = @import("std");
const zml = @import("zml");
const core = @import("core_parts.zig");

const log = std.log.scoped(.core_conformance);
pub const std_options: std.Options = .{ .log_level = .info };

const BUNDLE = "/home/adam/Development/Experiments/Video-Generation/.work/core_bundle";
const EXTRAS = "/home/adam/Development/Experiments/Video-Generation/.work/extras";

const T: i64 = 64;
const DIM: i64 = 4096;
const IN_CH = core.IN_CH;
const TE = core.TE;

fn readBin(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u8 {
    var buf: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir, name });
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
}

fn loadF32(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, expect: usize) ![]f32 {
    const raw = try readBin(allocator, io, dir, name);
    defer allocator.free(raw);
    if (raw.len != expect * 4) {
        log.err("{s}: {d} bytes, expected {d}", .{ name, raw.len, expect * 4 });
        return error.BadSize;
    }
    const out = try allocator.alloc(f32, expect);
    @memcpy(std.mem.sliceAsBytes(out), raw);
    return out;
}

/// rel-RMS + max-abs-diff; width only affects the worst-element diagnostic.
fn compare(name: []const u8, got: []const f32, want: []const f32, width: usize, limit: f64) bool {
    var err: f64 = 0;
    var ref: f64 = 0;
    var maxd: f64 = 0;
    var arg: usize = 0;
    var nan: usize = 0;
    for (got, want, 0..) |g, w, i| {
        if (std.math.isNan(g) or std.math.isInf(g)) nan += 1;
        const d = @abs(@as(f64, g) - @as(f64, w));
        if (d > maxd) {
            maxd = d;
            arg = i;
        }
        err += d * d;
        ref += @as(f64, w) * w;
    }
    const rms = @sqrt(err / (ref + 1e-20));
    const pass = rms <= limit and nan == 0;
    log.info("GATE {s}: rel-RMS {e:.3} max-abs-diff {e:.3} nan/inf {d} -> {s}", .{
        name, rms, maxd, nan, if (pass) "PASS" else "FAIL",
    });
    if (!pass) {
        log.err("  worst at token={d} chan={d}: ref={d:.6} ours={d:.6}", .{
            arg / width, arg % width, want[arg], got[arg],
        });
    }
    return pass;
}

fn loadCoreBufs(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform) !zml.Bufferized(core.CoreParts) {
    var bufs: zml.Bufferized(core.CoreParts) = undefined;
    inline for (core.CORE_SPECS) |spec| {
        const raw = try readBin(allocator, io, EXTRAS, spec.file);
        defer allocator.free(raw);
        const sh = core.coreShape(spec);
        if (raw.len != sh.byteSize()) {
            log.err("{s}: {d} bytes, shape wants {d}", .{ spec.file, raw.len, sh.byteSize() });
            return error.BadSize;
        }
        @field(bufs, spec.field) = try zml.Buffer.fromBytes(io, platform, sh, .replicated, raw);
    }
    return bufs;
}

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(1_000_000);
    const allocator = init.gpa;
    const io = init.io;

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    log.info("platform: {s}", .{@tagName(platform.target)});

    const Tu: usize = @intCast(T);
    const Du: usize = @intCast(DIM);

    const model = core.makeCoreSpecs();
    const bufs = try loadCoreBufs(allocator, io, platform);
    log.info("E2E-core weights resident ({d} tensors)", .{core.CORE_SPECS.len});

    // ---- bundle inputs ----
    const t_h = try loadF32(allocator, io, BUNDLE, "in_timesteps.bin", Tu);
    defer allocator.free(t_h);
    const sg_h = try loadF32(allocator, io, BUNDLE, "in_sigma.bin", 1);
    defer allocator.free(sg_h);
    const lat_h = try loadF32(allocator, io, BUNDLE, "in_latent.bin", Tu * @as(usize, @intCast(IN_CH)));
    defer allocator.free(lat_h);
    const x_h = try loadF32(allocator, io, BUNDLE, "in_xfinal.bin", Tu * Du);
    defer allocator.free(x_h);

    const e_h = try loadF32(allocator, io, BUNDLE, "s2_embedded_timestep.bin", Tu * Du);
    defer allocator.free(e_h);

    const t_shape = zml.Shape.init(.{ .t = T }, .f32);
    const sg_shape = zml.Shape.init(.{ .t = 1 }, .f32);
    const lat_shape = zml.Shape.init(.{ .t = T, .i = IN_CH }, .f32);
    const x_shape = zml.Shape.init(.{ .t = T, .i = DIM }, .f32);

    var t_buf: zml.Buffer = try .fromBytes(io, platform, t_shape, .replicated, std.mem.sliceAsBytes(t_h));
    defer t_buf.deinit();
    var sg_buf: zml.Buffer = try .fromBytes(io, platform, sg_shape, .replicated, std.mem.sliceAsBytes(sg_h));
    defer sg_buf.deinit();
    var lat_buf: zml.Buffer = try .fromBytes(io, platform, lat_shape, .replicated, std.mem.sliceAsBytes(lat_h));
    defer lat_buf.deinit();
    var x_buf: zml.Buffer = try .fromBytes(io, platform, x_shape, .replicated, std.mem.sliceAsBytes(x_h));
    defer x_buf.deinit();
    var e_buf: zml.Buffer = try .fromBytes(io, platform, x_shape, .replicated, std.mem.sliceAsBytes(e_h));
    defer e_buf.deinit();

    const t_spec: zml.Tensor = .fromShape(t_shape);
    const sg_spec: zml.Tensor = .fromShape(sg_shape);
    const lat_spec: zml.Tensor = .fromShape(lat_shape);
    const x_spec: zml.Tensor = .fromShape(x_shape);

    var all_pass = true;
    const Stage = struct {
        method: []const u8,
        oracle: []const u8,
        len: usize,
        width: usize,
        limit: f64,
        args: enum { t, sg, lat, xt, xe },
    };
    const stages = [_]Stage{
        // sinusoid: the interesting gate (H-CORE-2) — both sides f32, but
        // XLA exp/sin vs torch's need not round identically, and phase
        // amplification (arg up to ~909) can turn freq ulps into ~5e-5 abs.
        .{ .method = "sinu", .oracle = "s1_sinusoid.bin", .len = Tu * 256, .width = 256, .limit = 1e-5, .args = .t },
        .{ .method = "emb", .oracle = "s2_embedded_timestep.bin", .len = Tu * Du, .width = Du, .limit = 2e-3, .args = .t },
        .{ .method = "ts9", .oracle = "s3_ts9.bin", .len = Tu * 9 * Du, .width = 9 * Du, .limit = 2e-3, .args = .t },
        .{ .method = "pts2", .oracle = "s4_pts2.bin", .len = 2 * Du, .width = 2 * Du, .limit = 2e-3, .args = .sg },
        .{ .method = "patchify", .oracle = "s5_patchify.bin", .len = Tu * Du, .width = Du, .limit = 2e-3, .args = .lat },
        .{ .method = "tail", .oracle = "s6_tail_out.bin", .len = Tu * 128, .width = 128, .limit = 2e-3, .args = .xt },
        // the production form (emb as input, fed the ORACLE's emb dump)
        // must hit the SAME oracle tail output
        .{ .method = "tailFromEmb", .oracle = "s6_tail_out.bin", .len = Tu * 128, .width = 128, .limit = 2e-3, .args = .xe },
        // the control must MATCH the oracle's wrong-norm dump...
        .{ .method = "tailWrong", .oracle = "s6_tail_wrongnorm.bin", .len = Tu * 128, .width = 128, .limit = 2e-3, .args = .xt },
    };

    const wrong_got: []f32 = try allocator.alloc(f32, Tu * 128);
    defer allocator.free(wrong_got);

    inline for (stages) |st| {
        const method = comptime std.meta.stringToEnum(std.meta.DeclEnum(core.CoreParts), st.method).?;
        var exe = switch (st.args) {
            .t => try platform.compile(allocator, io, model, method, .{t_spec}, .{}),
            .sg => try platform.compile(allocator, io, model, method, .{sg_spec}, .{}),
            .lat => try platform.compile(allocator, io, model, method, .{lat_spec}, .{}),
            .xt => try platform.compile(allocator, io, model, method, .{ x_spec, t_spec }, .{}),
            .xe => try platform.compile(allocator, io, model, method, .{ x_spec, zml.Tensor.fromShape(x_shape) }, .{}),
        };
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        switch (st.args) {
            .t => args.set(.{ bufs, t_buf }),
            .sg => args.set(.{ bufs, sg_buf }),
            .lat => args.set(.{ bufs, lat_buf }),
            .xt => args.set(.{ bufs, x_buf, t_buf }),
            .xe => args.set(.{ bufs, x_buf, e_buf }),
        }
        exe.call(args, &results);
        var out: zml.Buffer = results.get(zml.Buffer);
        defer out.deinit();
        var slice = try out.toSliceAlloc(allocator, io);
        defer slice.free(allocator);
        const got = slice.constItems(f32)[0..st.len];
        if (comptime std.mem.eql(u8, st.method, "tailWrong")) @memcpy(wrong_got, got);

        const want = try loadF32(allocator, io, BUNDLE, st.oracle, st.len);
        defer allocator.free(want);
        if (!compare(st.method, got, want, st.width, st.limit)) all_pass = false;
    }

    // ...and must FAIL against the TRUE tail (teeth for the norm-kind
    // distinction). If wrong-vs-true sits inside the real budget, the s6
    // gate cannot see the difference between LayerNorm and RMSNorm.
    {
        const truth = try loadF32(allocator, io, BUNDLE, "s6_tail_out.bin", Tu * 128);
        defer allocator.free(truth);
        var err: f64 = 0;
        var ref: f64 = 0;
        for (wrong_got, truth) |g, w| {
            const d = @as(f64, g) - @as(f64, w);
            err += d * d;
            ref += @as(f64, w) * w;
        }
        const rms = @sqrt(err / (ref + 1e-20));
        const teeth = rms > 1e-3;
        log.info("GATE tailWrongVsTrue: rel-RMS {e:.3} (must EXCEED 1e-3) -> {s}", .{ rms, if (teeth) "PASS" else "FAIL" });
        if (!teeth) all_pass = false;
    }

    if (!all_pass) {
        log.err("CORE CONFORMANCE: FAILURES PRESENT", .{});
        return error.GateFailed;
    }
    log.info("CORE CONFORMANCE: ALL GATES PASS", .{});
}
