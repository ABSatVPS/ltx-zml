//! Phase 3: the distilled scheduler vs the reference bundle, gated
//! update-by-update (docs/lab-notebook.md, 2026-08-21 pre-registration).
//!
//! Host-only, no GPU, no ZML: the scheduler is pure f32 arithmetic
//! between DiT calls, so the engine module (ltx/scheduler.zig) runs on
//! the CPU and is gated here BIT-EXACTLY against dumps produced by the
//! reference's own loop code (tools/make_scheduler_bundle.py). The only
//! tolerated deviation is the RoPE-style straggler protocol: <=1 ulp and
//! <=0.01% of elements, coordinate-pinned — which at these tensor sizes
//! rounds down to zero allowed diffs everywhere.
const std = @import("std");
const sched = @import("scheduler.zig");

const log = std.log;

pub const std_options: std.Options = .{ .log_level = .info };

const BUNDLE = "/home/adam/Development/Experiments/Video-Generation/.work/scheduler_bundle";
const TV: usize = 64;
const CV: usize = 128;
const TA: usize = 16;
const CA: usize = 32;

fn loadF32(allocator: std.mem.Allocator, io: std.Io, name: []const u8, expect: usize) ![]f32 {
    var buf: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&buf, "{s}/{s}.bin", .{ BUNDLE, name });
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(raw);
    if (raw.len != expect * 4) {
        log.err("{s}: {d} bytes, expected {d}", .{ name, raw.len, expect * 4 });
        return error.BadSize;
    }
    const out = try allocator.alloc(f32, expect);
    @memcpy(std.mem.sliceAsBytes(out), raw);
    return out;
}

fn ulpDeltaF32(a: u32, b: u32) i64 {
    const ia: i64 = if (a & 0x8000_0000 != 0) -@as(i64, a & 0x7fff_ffff) else @as(i64, a);
    const ib: i64 = if (b & 0x8000_0000 != 0) -@as(i64, b & 0x7fff_ffff) else @as(i64, b);
    return ia - ib;
}

const Tally = struct {
    gates: usize = 0,
    exact: usize = 0,
    failed: usize = 0,

    fn gate(self: *@This(), name: []const u8, got: []const f32, want: []const f32) void {
        self.gates += 1;
        var n_diff: usize = 0;
        var worst: i64 = 0;
        var worst_idx: usize = 0;
        for (got, want, 0..) |g, w, i| {
            const gb: u32 = @bitCast(g);
            const wb: u32 = @bitCast(w);
            if (gb != wb) {
                const d = ulpDeltaF32(gb, wb);
                n_diff += 1;
                if (@abs(d) > @abs(worst)) {
                    worst = d;
                    worst_idx = i;
                }
            }
        }
        if (n_diff == 0) {
            self.exact += 1;
            return;
        }
        const limit = want.len / 10_000;
        if (@abs(worst) <= 1 and n_diff <= limit) {
            log.warn("GATE {s}: {d}/{d} stragglers within 1 ulp (worst at flat={d}) — tolerated", .{ name, n_diff, want.len, worst_idx });
            return;
        }
        self.failed += 1;
        log.err("GATE {s}: {d}/{d} bit diffs, worst {d} ulp at flat={d}: want=0x{x:0>8} got=0x{x:0>8} -> FAIL", .{
            name,                          n_diff,                        want.len,
            worst,                         worst_idx,                     @as(u32, @bitCast(want[worst_idx])),
            @as(u32, @bitCast(got[worst_idx])),
        });
    }
};

/// One modality through one stage: init gate, then per step the four
/// sub-gates (ts, den, post, next), carrying OUR stepped latent forward
/// so trajectory drift cannot hide.
fn runStage(
    allocator: std.mem.Allocator,
    io: std.Io,
    tally: *Tally,
    stage: []const u8,
    mod: []const u8,
    sigmas: []const f32,
    x: []f32,
    clean_x0: []const f32,
    noise: []const f32,
    mask: []const f32,
    t: usize,
    c: usize,
    noise_scale: f32,
) !void {
    var nb: [128]u8 = undefined;

    sched.noiseInit(x, clean_x0, noise, mask, c, noise_scale);
    const x0_want = try loadF32(allocator, io, try std.fmt.bufPrint(&nb, "{s}_{s}_x0", .{ stage, mod }), t * c);
    defer allocator.free(x0_want);
    tally.gate(try std.fmt.bufPrint(&nb, "{s}/{s} init", .{ stage, mod }), x, x0_want);

    const ts = try allocator.alloc(f32, t);
    defer allocator.free(ts);
    const den = try allocator.alloc(f32, t * c);
    defer allocator.free(den);
    const post = try allocator.alloc(f32, t * c);
    defer allocator.free(post);
    const next = try allocator.alloc(f32, t * c);
    defer allocator.free(next);

    for (0..sigmas.len - 1) |i| {
        const vel = try loadF32(allocator, io, try std.fmt.bufPrint(&nb, "{s}_vel_{s}_{d}", .{ stage, mod, i }), t * c);
        defer allocator.free(vel);

        sched.timesteps(ts, mask, sigmas[i]);
        sched.toDenoised(den, x, vel, ts, c);
        sched.postProcess(post, den, clean_x0, mask, c);
        sched.eulerStep(next, x, post, sigmas[i], sigmas[i + 1]);

        inline for (.{ "ts", "den", "post", "next" }, .{ ts, den, post, next }) |part, ours| {
            const want = try loadF32(allocator, io, try std.fmt.bufPrint(&nb, "{s}_{s}_step{d}_{s}", .{ stage, mod, i, part }), ours.len);
            defer allocator.free(want);
            var gname: [128]u8 = undefined;
            tally.gate(try std.fmt.bufPrint(&gname, "{s}/{s} step{d} {s}", .{ stage, mod, i, part }), ours, want);
        }
        @memcpy(x, next);
    }
    log.info("stage {s}/{s}: init + {d} steps gated", .{ stage, mod, sigmas.len - 1 });
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    var tally: Tally = .{};

    // G-SCHED-1: the Zig comptime sigma tables vs the bundle's.
    const sig1 = try loadF32(allocator, io, "sigmas1", sched.SIGMAS_STAGE1.len);
    defer allocator.free(sig1);
    const sig2 = try loadF32(allocator, io, "sigmas2", sched.SIGMAS_STAGE2.len);
    defer allocator.free(sig2);
    tally.gate("sigmas stage1", &sched.SIGMAS_STAGE1, sig1);
    tally.gate("sigmas stage2", &sched.SIGMAS_STAGE2, sig2);

    // Shared inputs.
    const clean_v = try loadF32(allocator, io, "clean_v", TV * CV);
    defer allocator.free(clean_v);
    const clean_a = try loadF32(allocator, io, "clean_a", TA * CA);
    defer allocator.free(clean_a);
    const mask_v = try loadF32(allocator, io, "mask_v", TV);
    defer allocator.free(mask_v);
    const mask_a = try loadF32(allocator, io, "mask_a", TA);
    defer allocator.free(mask_a);

    const xv = try allocator.alloc(f32, TV * CV);
    defer allocator.free(xv);
    const xa = try allocator.alloc(f32, TA * CA);
    defer allocator.free(xa);

    // Stage 1: from clean (== initial latent) at noise_scale 1.0.
    {
        const n_v = try loadF32(allocator, io, "noise1_v", TV * CV);
        defer allocator.free(n_v);
        const n_a = try loadF32(allocator, io, "noise1_a", TA * CA);
        defer allocator.free(n_a);
        try runStage(allocator, io, &tally, "s1", "v", &sched.SIGMAS_STAGE1, xv, clean_v, n_v, mask_v, TV, CV, 1.0);
        try runStage(allocator, io, &tally, "s1", "a", &sched.SIGMAS_STAGE1, xa, clean_a, n_a, mask_a, TA, CA, 1.0);
    }

    // Stage 2: video from the upsampled latent, audio from its stage-1
    // result, both renoised to sigma 0.909375 (deployment wiring).
    {
        const up_v = try loadF32(allocator, io, "upsampled_v", TV * CV);
        defer allocator.free(up_v);
        const init_a = try loadF32(allocator, io, "s2_init_a", TA * CA);
        defer allocator.free(init_a);
        const n_v = try loadF32(allocator, io, "noise2_v", TV * CV);
        defer allocator.free(n_v);
        const n_a = try loadF32(allocator, io, "noise2_a", TA * CA);
        defer allocator.free(n_a);
        try runStage(allocator, io, &tally, "s2", "v", &sched.SIGMAS_STAGE2, xv, up_v, n_v, mask_v, TV, CV, sched.SIGMAS_STAGE2[0]);
        try runStage(allocator, io, &tally, "s2", "a", &sched.SIGMAS_STAGE2, xa, init_a, n_a, mask_a, TA, CA, sched.SIGMAS_STAGE2[0]);
    }

    log.info("SCHEDULER CONFORMANCE: {d} gates, {d} bit-exact, {d} failed -> {s}", .{
        tally.gates, tally.exact, tally.failed, if (tally.failed == 0) "PASS" else "FAIL",
    });
    if (tally.failed != 0) return error.ConformanceFailed;
}
