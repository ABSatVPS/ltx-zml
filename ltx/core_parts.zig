//! The E2E-core parts of the LTX-2.5 video stream — everything between
//! latent and latent that is NOT a transformer block: adaln_single (the
//! sinusoidal-embed MLP producing embedded_timestep and the 9-chunk
//! per-token ts), prompt_adaln_single (producing pts2 from scalar sigma),
//! patchify_proj, and the output tail (LayerNorm + scale_shift_table
//! modulation + proj_out). Traced ZML models gated by
//! //ltx:core_conformance against tools/make_core_bundle.py (notebook
//! 2026-08-21, late morning pre-registration).
const std = @import("std");
const zml = @import("zml");
const blk = @import("block.zig");

pub const TE: i64 = 256; // sinusoid width (DDPM/PixArt 256-dim)
pub const HALF: i64 = 128;
pub const IN_CH: i64 = 128; // latent channels (patchify in, proj_out out)
pub const TS_MULT: f64 = 1000.0; // timestep_scale_multiplier (config default)

const D = blk.D;
const EPS = blk.EPS;
const lin = blk.lin;

pub fn silu(x: zml.Tensor) zml.Tensor {
    return x.mul(x.sigmoid());
}

/// torch.nn.LayerNorm(D, elementwise_affine=false, eps=1e-6): TRUE
/// mean-subtracting LayerNorm with biased variance — NOT the RMSNorm the
/// blocks use (read receipt, fact three). The wrong-norm gate keeps this
/// distinction honest.
pub fn layerNormNoW(x: zml.Tensor) zml.Tensor {
    const m = x.mean(.i);
    const c = x.sub(m.broad(x.shape()));
    const v = c.mul(c).mean(.i);
    return c.mul(v.addConstant(EPS).rsqrt().broad(x.shape()));
}

/// The DDPM/PixArt sinusoid of t*1000, computed in f32 in-graph — matching
/// the reference, whose get_timestep_embedding computes f32 even under an
/// f64 oracle (read receipt, fact one). flip_sin_to_cos=true (cos half
/// FIRST), downscale_freq_shift=0 (divisor is HALF, not HALF-1).
/// t: [.t] f32 raw (mask*sigma); result [.t, .i=256].
pub fn sinusoid(t: zml.Tensor) zml.Tensor {
    const n = t.dim(.t);
    const sh = zml.Shape.init(.{ .t = n, .f = HALF }, .f32);
    const idx = zml.Tensor.arange(.{ .end = HALF }, .f32).withTags(.{.f});
    const freqs = idx.scale(comptime -@log(1e4) / @as(f64, @floatFromInt(HALF))).exp();
    const arg = t.scale(TS_MULT).broad(sh).mul(freqs.broad(sh));
    return zml.Tensor.concatenate(&.{ arg.cos(), arg.sin() }, .f).withTags(.{ .t, .i });
}

pub const WSpec = struct { field: []const u8, file: []const u8, dims: []const i64, f32_: bool = false };

pub const CORE_SPECS = [_]WSpec{
    .{ .field = "ada_l1_w", .file = "adaln_single_emb_timestep_embedder_linear_1_weight.bin", .dims = &.{ D, TE } },
    .{ .field = "ada_l1_b", .file = "adaln_single_emb_timestep_embedder_linear_1_bias.bin", .dims = &.{D} },
    .{ .field = "ada_l2_w", .file = "adaln_single_emb_timestep_embedder_linear_2_weight.bin", .dims = &.{ D, D } },
    .{ .field = "ada_l2_b", .file = "adaln_single_emb_timestep_embedder_linear_2_bias.bin", .dims = &.{D} },
    .{ .field = "ada_out_w", .file = "adaln_single_linear_weight.bin", .dims = &.{ 9 * D, D } },
    .{ .field = "ada_out_b", .file = "adaln_single_linear_bias.bin", .dims = &.{9 * D} },
    .{ .field = "pada_l1_w", .file = "prompt_adaln_single_emb_timestep_embedder_linear_1_weight.bin", .dims = &.{ D, TE } },
    .{ .field = "pada_l1_b", .file = "prompt_adaln_single_emb_timestep_embedder_linear_1_bias.bin", .dims = &.{D} },
    .{ .field = "pada_l2_w", .file = "prompt_adaln_single_emb_timestep_embedder_linear_2_weight.bin", .dims = &.{ D, D } },
    .{ .field = "pada_l2_b", .file = "prompt_adaln_single_emb_timestep_embedder_linear_2_bias.bin", .dims = &.{D} },
    .{ .field = "pada_out_w", .file = "prompt_adaln_single_linear_weight.bin", .dims = &.{ 2 * D, D } },
    .{ .field = "pada_out_b", .file = "prompt_adaln_single_linear_bias.bin", .dims = &.{2 * D} },
    .{ .field = "pat_w", .file = "patchify_proj_weight.bin", .dims = &.{ D, IN_CH } },
    .{ .field = "pat_b", .file = "patchify_proj_bias.bin", .dims = &.{D} },
    .{ .field = "po_w", .file = "proj_out_weight.bin", .dims = &.{ IN_CH, D } },
    .{ .field = "po_b", .file = "proj_out_bias.bin", .dims = &.{IN_CH} },
    // the checkpoint's lone f32 video tensor
    .{ .field = "fsst", .file = "scale_shift_table.bin", .dims = &.{ 2, D }, .f32_ = true },
};

pub fn coreShape(spec: WSpec) zml.Shape {
    const dt: zml.DataType = if (spec.f32_) .f32 else .bf16;
    return switch (spec.dims.len) {
        1 => zml.Shape.init(.{ .o = spec.dims[0] }, dt),
        2 => if (spec.f32_)
            zml.Shape.init(.{ .n = spec.dims[0], .i = spec.dims[1] }, dt)
        else
            zml.Shape.init(.{ .o = spec.dims[0], .i = spec.dims[1] }, dt),
        else => unreachable,
    };
}

pub const CoreParts = struct {
    ada_l1_w: zml.Tensor,
    ada_l1_b: zml.Tensor,
    ada_l2_w: zml.Tensor,
    ada_l2_b: zml.Tensor,
    ada_out_w: zml.Tensor,
    ada_out_b: zml.Tensor,
    pada_l1_w: zml.Tensor,
    pada_l1_b: zml.Tensor,
    pada_l2_w: zml.Tensor,
    pada_l2_b: zml.Tensor,
    pada_out_w: zml.Tensor,
    pada_out_b: zml.Tensor,
    pat_w: zml.Tensor,
    pat_b: zml.Tensor,
    po_w: zml.Tensor,
    po_b: zml.Tensor,
    fsst: zml.Tensor,

    // ---- staged methods, each recomputing its prefix (house style) --------

    pub fn sinu(self: @This(), t: zml.Tensor) zml.Tensor {
        _ = self;
        return sinusoid(t);
    }

    /// embedded_timestep [.t, .i=D]: sinusoid -> linear_1 -> SiLU -> linear_2.
    /// Enters the output tail RAW (fact four); adaln's own SiLU applies only
    /// in front of the chunk linears.
    pub fn emb(self: @This(), t: zml.Tensor) zml.Tensor {
        return lin(silu(lin(sinusoid(t), self.ada_l1_w, self.ada_l1_b)), self.ada_l2_w, self.ada_l2_b);
    }

    /// the 9-chunk per-token ts [.t, .i=9D] the blocks consume.
    pub fn ts9(self: @This(), t: zml.Tensor) zml.Tensor {
        return lin(silu(self.emb(t)), self.ada_out_w, self.ada_out_b);
    }

    /// pts2 [.t=1, .i=2D] from scalar sigma — same x1000 path (fact two).
    pub fn pts2(self: @This(), sg: zml.Tensor) zml.Tensor {
        const e = lin(silu(lin(sinusoid(sg), self.pada_l1_w, self.pada_l1_b)), self.pada_l2_w, self.pada_l2_b);
        return lin(silu(e), self.pada_out_w, self.pada_out_b);
    }

    /// ts9 in the block executable's input layout [.t, .n=9, .i=D].
    pub fn ts9r(self: @This(), t: zml.Tensor) zml.Tensor {
        return self.ts9(t).splitAxis(.i, .{ .n = 9, .i = D });
    }

    /// pts2 in the block executable's input layout [.n=2, .i=D].
    pub fn pts2r(self: @This(), sg: zml.Tensor) zml.Tensor {
        return self.pts2(sg).squeeze(.t).splitAxis(.i, .{ .n = 2, .i = D });
    }

    pub fn patchify(self: @This(), latent: zml.Tensor) zml.Tensor {
        return lin(latent, self.pat_w, self.pat_b);
    }

    fn tailFrom(self: @This(), normed: zml.Tensor, t: zml.Tensor) zml.Tensor {
        const e = self.emb(t); // [.t,.i]
        const shift = self.fsst.slice1d(.n, .{ .start = 0, .end = 1 }).squeeze(.n).broad(e.shape()).add(e);
        const scl = self.fsst.slice1d(.n, .{ .start = 1, .end = 2 }).squeeze(.n).broad(e.shape()).add(e);
        return lin(blk.modulate(normed, scl, shift), self.po_w, self.po_b);
    }

    /// output tail: LayerNorm -> x*(1+scale)+shift -> proj_out, with
    /// shift/scale = scale_shift_table row + RAW embedded_timestep.
    pub fn tail(self: @This(), x: zml.Tensor, t: zml.Tensor) zml.Tensor {
        return self.tailFrom(layerNormNoW(x), t);
    }

    /// deliberate wrong-norm control (RMSNorm in the tail) — must FAIL the
    /// s6 gate by orders of magnitude, proving the gate can tell the norms
    /// apart (H-CORE-4), and must MATCH the oracle's own wrong-norm dump.
    pub fn tailWrong(self: @This(), x: zml.Tensor, t: zml.Tensor) zml.Tensor {
        return self.tailFrom(blk.rmsNoW(x), t);
    }
};

pub fn makeCoreSpecs() CoreParts {
    var m: CoreParts = undefined;
    inline for (CORE_SPECS) |spec| {
        @field(m, spec.field) = zml.Tensor.fromShape(coreShape(spec));
    }
    return m;
}
