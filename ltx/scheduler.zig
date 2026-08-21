//! The distilled LTX-2.5 scheduler, mirrored from the reference read
//! line by line on 2026-08-21 (notebook entry): ltx-core 1.2.0
//! diffusion_steps/noisers/utils plus ltx-pipelines 1.0.0
//! samplers/helpers. Pure f32 host arithmetic in the reference's exact
//! operation order — Zig's default float semantics do not contract, so
//! there is no FMA on this side, matching torch's CPU kernels.
//!
//! This module is the engine's scheduler; //ltx:scheduler_conformance
//! gates every function bit-exactly against the reference bundle.

/// DISTILLED_SIGMA_VALUES: the distilled pipeline does NOT use the
/// LTX2Scheduler class — it ships these constants. Non-dyadic decimals
/// round to f32 identically from Zig and Python literals (both parse
/// decimal -> f64 correctly rounded, then cast).
pub const SIGMAS_STAGE1 = [_]f32{ 1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0 };

/// STAGE_2_DISTILLED_SIGMA_VALUES: the tail of stage 1's list.
pub const SIGMAS_STAGE2 = [_]f32{ 0.909375, 0.725, 0.421875, 0.0 };

/// torch.lerp — the two-branch ATen kernel: |w| < 0.5 selects the
/// a + w*(b-a) form, else b - (b-a)*(1-w). Every weight this pipeline
/// uses (noise scales 1.0 and 0.909375; mask values 0.5 and 1.0 — note
/// 0.5 is NOT < 0.5) takes the SECOND branch. And the CPU tensor kernel
/// FUSES the multiply-add — a single rounding, not the two the header's
/// scalar formula suggests. Found by the s2 init gate (stage 1 cannot
/// discriminate: w=1.0 returns b under any formula and w=0.5's product
/// is exact); confirmed bitwise against torch.lerp on 2000 random pairs
/// per branch. Hence @mulAdd here.
pub fn lerp(a: f32, b: f32, w: f32) f32 {
    const d = b - a;
    if (@abs(w) < 0.5) {
        return @mulAdd(f32, w, d, a);
    }
    return @mulAdd(f32, -d, 1.0 - w, b);
}

/// GaussianNoiser.__call__ at state initialization, where latent ==
/// clean_latent (create_initial_state clones): x = lerp(clean, noise,
/// noise_scale), then x = lerp(clean, x, mask). mask has one value per
/// token, broadcast over `stride` channels.
pub fn noiseInit(out: []f32, clean: []const f32, noise: []const f32, mask: []const f32, stride: usize, noise_scale: f32) void {
    for (out, clean, noise, 0..) |*o, c, n, i| {
        const x = lerp(c, n, noise_scale);
        o.* = lerp(c, x, mask[i / stride]);
    }
}

/// timesteps_from_mask: denoise_mask * sigma, one value per token.
pub fn timesteps(out: []f32, mask: []const f32, sigma: f32) void {
    for (out, mask) |*o, m| o.* = m * sigma;
}

/// The X0Model wrapper's to_denoised(sample, velocity, timesteps):
/// sample - velocity * ts, PER-TOKEN ts broadcast over channels. This
/// per-token/scalar asymmetry against eulerStep's scalar sigma is the
/// conditioning mechanism — not a simplification target.
pub fn toDenoised(out: []f32, x: []const f32, v: []const f32, ts: []const f32, stride: usize) void {
    for (out, x, v, 0..) |*o, xi, vi, i| o.* = xi - vi * ts[i / stride];
}

/// post_process_latent: denoised*mask + clean*(1-mask), per token.
pub fn postProcess(out: []f32, den: []const f32, clean: []const f32, mask: []const f32, stride: usize) void {
    for (out, den, clean, 0..) |*o, d, c, i| {
        const m = mask[i / stride];
        o.* = d * m + c * (1.0 - m);
    }
}

/// EulerDiffusionStep.step via to_velocity: v = (x - denoised)/sigma in
/// f32, then x' = x + v*dt with dt = sigma_next - sigma. Deliberately NO
/// special case at sigma_next == 0: the reference divides and multiplies
/// on the final step too, so the result equals `denoised` only up to
/// rounding — snapping would be bit-different from the reference.
pub fn eulerStep(out: []f32, x: []const f32, post: []const f32, sigma: f32, sigma_next: f32) void {
    const dt = sigma_next - sigma;
    for (out, x, post) |*o, xi, p| {
        const v = (xi - p) / sigma;
        o.* = xi + v * dt;
    }
}
