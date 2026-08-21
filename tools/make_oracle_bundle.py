#!/usr/bin/env python3
"""Build the Phase 2 oracle bundle: reference outputs for one LTX-2.5
video transformer block, staged per the pre-registered composition order.

Runs the upstream ltx-core block (torch CPU) on the fetched block-0
weights, in float64 for the reference (weights upcast from bf16), with
RoPE tables built by the reference f64 numpy path and cast once to f32 —
the runtime contract. Stages are computed by calling the block's OWN
submodules in forward order, then the full block.forward is run and the
staged composition is asserted to reproduce it exactly (f64), so the
staging cannot drift from the reference.

Outputs: raw .bin tensors (f32 unless noted) + manifest.json with shapes,
hashes, stats, versions, and the RoPE table metadata.

Usage: oracle-venv/bin/python tools/make_oracle_bundle.py WEIGHTS_DIR OUTDIR
"""

import hashlib
import json
import pathlib
import sys

import numpy as np
import torch

from ltx_core.model.transformer.rope import LTXRopeType, generate_freq_grid_np, precompute_freqs_cis
from ltx_core.model.transformer.transformer import (
    BasicAVTransformerBlock,
    TransformerConfig,
    TransformerOpsConfig,
)
from ltx_core.model.transformer.transformer_args import TransformerArgs

SEED = 417
B, T, S = 1, 64, 32
DIM, HEADS, HD = 4096, 32, 128
MAX_POS = [20, 2048, 2048]
THETA = 10000.0

DT = {"BF16": torch.bfloat16, "F32": torch.float32}


def load_weights(wdir: pathlib.Path) -> dict[str, torch.Tensor]:
    man = json.loads((wdir / "manifest.json").read_text())
    sd = {}
    for name, e in man["tensors"].items():
        raw = (wdir / e["file"]).read_bytes()
        t = torch.frombuffer(bytearray(raw), dtype=DT[e["dtype"]]).reshape(e["shape"])
        key = name.replace("transformer_blocks." + str(man["block"]) + ".", "")
        sd[key] = t
    return sd


def make_grid() -> torch.Tensor:
    """Probe-first indices grid [1, 3, T, 2] (start,end; end=start so the
    middle-grid midpoint is exact). Rows 0-7: only t active (0..max_t
    inclusive, endpoints hit). Rows 8-15: only h. Rows 16-23: only w.
    Remaining rows: a deterministic mesh over all three axes."""
    g = torch.zeros(1, 3, T, dtype=torch.float64)
    for i in range(8):
        g[0, 0, i] = MAX_POS[0] * i / 7.0
    for i in range(8):
        g[0, 1, 8 + i] = MAX_POS[1] * i / 7.0
    for i in range(8):
        g[0, 2, 16 + i] = MAX_POS[2] * i / 7.0
    k = 0
    for i in range(24, T):
        g[0, 0, i] = (k * 3) % MAX_POS[0]
        g[0, 1, i] = (k * 97) % MAX_POS[1]
        g[0, 2, i] = (k * 53) % MAX_POS[2]
        k += 1
    return torch.stack([g, g], dim=-1)  # start == end


def stats(t: torch.Tensor) -> dict:
    a = t.detach().to(torch.float64).flatten()
    return {
        "max_abs": float(a.abs().max()),
        "rms": float((a * a).mean().sqrt()),
        "nan": int(torch.isnan(a).sum()),
        "inf": int(torch.isinf(a).sum()),
    }


def main() -> int:
    wdir, outdir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    sd = load_weights(wdir)
    cfg = TransformerConfig(
        dim=DIM, heads=HEADS, d_head=HD, context_dim=DIM,
        apply_gated_attention=True, cross_attention_adaln=True, ff_bias=False,
    )

    def build(ops=None):
        blk = BasicAVTransformerBlock(video=cfg, audio=None, rope_type=LTXRopeType.SPLIT, norm_eps=1e-6, ops=ops)
        missing, unexpected = blk.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected}"
        assert all("audio" in m or "video_to" in m for m in missing) or not missing, f"missing video keys: {missing}"
        return blk.to(torch.float64).eval()

    blk = build()
    identity_ops = TransformerOpsConfig.from_functions(gated_attention=lambda x, out, mod: out)
    blk_nogate = build(identity_ops)

    # ---- inputs (f64, seeded) --------------------------------------------
    x = (torch.randn(B, T, DIM, dtype=torch.float64) * 0.5)
    context = (torch.randn(B, S, DIM, dtype=torch.float64) * 0.5)
    timestep = (torch.randn(B, T, 9 * DIM, dtype=torch.float64) * 0.1)
    prompt_timestep = (torch.randn(B, 1, 2 * DIM, dtype=torch.float64) * 0.1)
    grid = make_grid()

    # ---- RoPE tables: f64 numpy path, single cast to f32 (the contract) --
    cos_f32, sin_f32 = precompute_freqs_cis(
        grid, dim=DIM, out_dtype=torch.float32, theta=THETA, max_pos=MAX_POS,
        use_middle_indices_grid=True, num_attention_heads=HEADS,
        rope_type=LTXRopeType.SPLIT, freq_grid_generator=generate_freq_grid_np,
    )
    pe = (cos_f32.to(torch.float64), sin_f32.to(torch.float64))

    # ---- staged forward from the block's own submodules ------------------
    dumps: dict[str, torch.Tensor] = {
        "in_x": x, "in_context": context, "in_timestep": timestep,
        "in_prompt_timestep": prompt_timestep, "in_grid": grid,
        "rope_cos": cos_f32, "rope_sin": sin_f32,
    }

    shift_msa, scale_msa, gate_msa = blk.get_ada_values(blk.scale_shift_table, B, timestep, slice(0, 3))
    norm_x = blk.ada_zero_function(x, blk.norm_eps, scale_msa, shift_msa)
    dumps["s1_norm_msa"] = norm_x

    dumps["s1b_qnorm"] = blk.attn1.q_norm(blk.attn1.to_q(norm_x))
    dumps["s2_attn1_nogate"] = blk_nogate.attn1(norm_x, pe=pe)
    attn1_out = blk.attn1(norm_x, pe=pe)
    dumps["s3_attn1"] = attn1_out

    x_sa, x_normed = blk.post_sa_function(x, attn1_out, None, blk.norm_eps, gate_msa)
    dumps["s4_after_sa"] = x_sa
    dumps["s4_normed"] = x_normed

    ca = blk._apply_text_cross_attention(
        x_normed, context, blk.attn2, blk.scale_shift_table,
        blk.prompt_scale_shift_table, timestep, prompt_timestep, None,
        cross_attention_adaln=True,
    )
    dumps["s5_ca_out"] = ca
    x_ca = x_sa + ca

    shift_mlp, scale_mlp, gate_mlp = blk.get_ada_values(blk.scale_shift_table, B, timestep, slice(3, 6))
    ff_in = blk.ada_zero_function(x_ca, blk.norm_eps, scale_mlp, shift_mlp)
    dumps["s6_ff_in"] = ff_in
    ff_out = blk.ff(ff_in)
    dumps["s7_ff_out"] = ff_out
    staged_out = x_ca + ff_out * gate_mlp
    dumps["block_out"] = staged_out

    # ---- the real forward must agree with the staging exactly ------------
    args = TransformerArgs(
        x=x, context=context, context_mask=None, timesteps=timestep,
        embedded_timestep=torch.zeros(B, 1, DIM, dtype=torch.float64),
        positional_embeddings=pe, cross_positional_embeddings=None,
        cross_scale_shift_timestep=None, cross_gate_timestep=None,
        enabled=True, prompt_timestep=prompt_timestep,
    )
    with torch.no_grad():
        vout, _ = blk.forward(args, None)
    diff = (vout.x - staged_out).abs().max()
    assert diff < 1e-10, f"staging drifted from forward: max abs {diff}"
    print(f"staging == forward (f64): max abs diff {float(diff):.2e}")

    # Secondary reference: the same forward in bf16 (deployment dtype).
    blk16 = build().to(torch.bfloat16)
    args16 = TransformerArgs(
        x=x.to(torch.bfloat16), context=context.to(torch.bfloat16), context_mask=None,
        timesteps=timestep.to(torch.bfloat16),
        embedded_timestep=torch.zeros(B, 1, DIM, dtype=torch.bfloat16),
        positional_embeddings=(cos_f32.to(torch.bfloat16), sin_f32.to(torch.bfloat16)),
        cross_positional_embeddings=None, cross_scale_shift_timestep=None,
        cross_gate_timestep=None, enabled=True,
        prompt_timestep=prompt_timestep.to(torch.bfloat16),
    )
    with torch.no_grad():
        vout16, _ = blk16.forward(args16, None)
    dumps["block_out_bf16ref"] = vout16.x.to(torch.float64)
    rel = float(((vout16.x.to(torch.float64) - staged_out) ** 2).mean().sqrt()
                / ((staged_out ** 2).mean().sqrt() + 1e-20))
    print(f"bf16 reference vs f64 reference rel-RMS: {rel:.5f}")

    # ---- serialize -------------------------------------------------------
    manifest = {
        "seed": SEED, "B": B, "T": T, "S": S, "dim": DIM, "heads": HEADS, "hd": HD,
        "torch": torch.__version__, "numpy": np.__version__,
        "weights_manifest_header_sha": json.loads((wdir / "manifest.json").read_text())["header_sha256"],
        "rope": {
            "theta": THETA, "max_pos": MAX_POS, "use_middle_indices_grid": True,
            "freq_grid": "numpy_f64 theta^linspace(log_t(1)..log_t(theta), dim//6) * pi/2, endpoints inclusive",
            "positions": "fractional pos/max_pos mapped to 2f-1 in [-1,1]",
            "layout": "freqs (indices x axes) transposed -> flatten: axis-triplets adjacent, freq-major",
            "pad": "2 identity slots PREPENDED (cos=1, sin=0) to reach dim/2",
            "heads": "2048 slots reshaped (T, 32, 64): frequency bands split across heads",
            "pairs": "split-half rotation (i, i+64) within each head",
            "cast": "single f64->f32 cast at serialization; f32 tables are the runtime contract",
        },
        "tensors": {},
        "bf16ref_vs_f64_relrms": rel,
    }
    for name, t in dumps.items():
        arr = t.detach().to(torch.float32).contiguous().numpy() if name != "in_grid" else t.numpy().astype(np.float32)
        data = arr.tobytes()
        (outdir / f"{name}.bin").write_bytes(data)
        manifest["tensors"][name] = {
            "shape": list(arr.shape), "dtype": "f32",
            "sha256": hashlib.sha256(data).hexdigest(), **stats(t),
        }
        print(f"  {name} {list(arr.shape)} max_abs={manifest['tensors'][name]['max_abs']:.4g}")
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {outdir}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
