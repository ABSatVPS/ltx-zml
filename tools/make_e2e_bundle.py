#!/usr/bin/env python3
"""The E2E composition oracle (H-CORE-5, notebook 2026-08-21): the full
video-only forward — patchify → adaln-driven 48-block chain → output
tail — in torch CPU f64, at harness T=64.

Inputs bind the existing bundles together: latent/timesteps/sigma are
the CORE bundle's (so the staged core dumps double as controls here),
context and the f32 RoPE tables are the Phase 2 oracle bundle's.
Blocks load one at a time (f64 carry, f32-cast dumps), the same
convention as the walk bundle.

Controls, asserted before anything is written:
  - patchify(latent) f32-cast must equal core_bundle/s5_patchify.bin;
  - ts9 must equal s3_ts9.bin, pts2 must equal s4_pts2.bin (byte-for-byte);
  - the tail applied to core_bundle's in_xfinal must equal s6_tail_out.bin.

Usage: oracle-venv/bin/python tools/make_e2e_bundle.py \
           CORE_DIR ORACLE_DIR EXTRAS_DIR WORK_ROOT OUT_DIR
"""

import gc
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch

from ltx_core.model.transformer.adaln import AdaLayerNormSingle
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
from ltx_core.model.transformer.transformer_args import TransformerArgs, TransformerArgsPreprocessor

B, T, S, DIM, IN_CH = 1, 64, 32, 4096, 128
CHECKPOINTS = [23, 47]
NORM_EPS = 1e-6
DT = {"BF16": torch.bfloat16, "F32": torch.float32}


def load_block(wdir: pathlib.Path) -> BasicAVTransformerBlock:
    man = json.loads((wdir / "manifest.json").read_text())
    sd = {}
    for name, e in man["tensors"].items():
        raw = (wdir / e["file"]).read_bytes()
        t = torch.frombuffer(bytearray(raw), dtype=DT[e["dtype"]]).reshape(e["shape"])
        sd[name.replace(f"transformer_blocks.{man['block']}.", "")] = t
    cfg = TransformerConfig(dim=DIM, heads=32, d_head=128, context_dim=DIM,
                            apply_gated_attention=True, cross_attention_adaln=True, ff_bias=False)
    blk = BasicAVTransformerBlock(video=cfg, audio=None, rope_type=LTXRopeType.SPLIT, norm_eps=NORM_EPS)
    blk.load_state_dict(sd, strict=False)
    return blk.to(torch.float64).eval()


def load_f32(d: pathlib.Path, name: str, shape) -> torch.Tensor:
    a = np.frombuffer((d / name).read_bytes(), dtype=np.float32).reshape(shape).copy()
    return torch.from_numpy(a).to(torch.float64)


def load_extras(wdir: pathlib.Path) -> dict[str, torch.Tensor]:
    man = json.loads((wdir / "manifest.json").read_text())
    sd = {}
    for name, e in man["tensors"].items():
        raw = (wdir / e["file"]).read_bytes()
        sd[name] = torch.frombuffer(bytearray(raw), dtype=DT[e["dtype"]]).reshape(e["shape"])
    return sd


def sub_state(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def main() -> int:
    core, oracle, extras_dir, root, out = (pathlib.Path(p) for p in sys.argv[1:6])
    out.mkdir(parents=True, exist_ok=True)

    latent = load_f32(core, "in_latent.bin", (B, T, IN_CH))
    timesteps = load_f32(core, "in_timesteps.bin", (B, T))
    sigma = load_f32(core, "in_sigma.bin", (B, 1))
    x_final_ctl = load_f32(core, "in_xfinal.bin", (B, T, DIM))
    context = load_f32(oracle, "in_context.bin", (B, S, DIM))
    cos = load_f32(oracle, "rope_cos.bin", (B, 32, T, 64))
    sin = load_f32(oracle, "rope_sin.bin", (B, 32, T, 64))

    sd = load_extras(extras_dir)
    adaln = AdaLayerNormSingle(DIM, embedding_coefficient=9)
    adaln.load_state_dict(sub_state(sd, "adaln_single."), strict=True)
    prompt_adaln = AdaLayerNormSingle(DIM, embedding_coefficient=2)
    prompt_adaln.load_state_dict(sub_state(sd, "prompt_adaln_single."), strict=True)
    patchify = torch.nn.Linear(IN_CH, DIM, bias=True)
    patchify.load_state_dict(sub_state(sd, "patchify_proj."), strict=True)
    proj_out = torch.nn.Linear(DIM, IN_CH, bias=True)
    proj_out.load_state_dict(sub_state(sd, "proj_out."), strict=True)
    sst64 = sd["scale_shift_table"].clone().to(torch.float64)
    norm_out = torch.nn.LayerNorm(DIM, elementwise_affine=False, eps=NORM_EPS)
    for m in (adaln, prompt_adaln, patchify, proj_out):
        m.to(torch.float64).eval()

    pre = TransformerArgsPreprocessor(
        patchify_proj=patchify, adaln=adaln, inner_dim=DIM,
        max_pos=[20, 2048, 2048], num_attention_heads=32,
        use_middle_indices_grid=True, timestep_scale_multiplier=1000,
        double_precision_rope=True, positional_embedding_theta=10000.0,
        rope_type=LTXRopeType.SPLIT, prompt_adaln=prompt_adaln,
    )

    def f32bytes(t: torch.Tensor) -> bytes:
        return t.to(torch.float32).numpy().tobytes()

    with torch.no_grad():
        x0 = patchify(latent)
        ts9, emb_t = pre._prepare_timestep(timesteps, adaln, B, torch.float64)
        pts2, _ = pre._prepare_timestep(sigma, prompt_adaln, B, torch.float64)

        # ---- controls binding this bundle to the core bundle ----
        assert f32bytes(x0) == (core / "s5_patchify.bin").read_bytes(), "x0 != core s5_patchify"
        assert f32bytes(ts9) == (core / "s3_ts9.bin").read_bytes(), "ts9 != core s3_ts9"
        assert f32bytes(pts2) == (core / "s4_pts2.bin").read_bytes(), "pts2 != core s4_pts2"
        tail_ctl = LTXModel._process_output(None, sst64, norm_out, proj_out, x_final_ctl, emb_t)
        assert f32bytes(tail_ctl) == (core / "s6_tail_out.bin").read_bytes(), "tail != core s6_tail_out"
        print("controls: x0/ts9/pts2/tail reproduce the core bundle byte-for-byte")

        manifest = {
            "core_manifest_sha": hashlib.sha256((core / "manifest.json").read_bytes()).hexdigest(),
            "oracle_manifest_sha": hashlib.sha256((oracle / "manifest.json").read_bytes()).hexdigest(),
            "checkpoints": CHECKPOINTS,
            "torch": torch.__version__,
            "tensors": {},
        }

        def dump(name: str, t: torch.Tensor) -> None:
            arr = t.to(torch.float32).numpy()
            (out / f"{name}.bin").write_bytes(arr.tobytes())
            manifest["tensors"][name] = {
                "shape": list(arr.shape),
                "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                "max_abs": float(np.abs(arr).max()),
                "rms": float(np.sqrt((arr.astype(np.float64) ** 2).mean())),
            }
            print(f"  {name}: max_abs={np.abs(arr).max():.4g} rms={manifest['tensors'][name]['rms']:.4g}", flush=True)

        dump("e2e_x0", x0)

        h = x0
        for bi in range(48):
            blk = load_block(root / f"block{bi}")
            args = TransformerArgs(
                x=h, context=context, context_mask=None, timesteps=ts9,
                embedded_timestep=torch.zeros(B, 1, DIM, dtype=torch.float64),
                positional_embeddings=(cos, sin), cross_positional_embeddings=None,
                cross_scale_shift_timestep=None, cross_gate_timestep=None,
                enabled=True, prompt_timestep=pts2,
            )
            vout, _ = blk.forward(args, None)
            h = vout.x
            del blk
            gc.collect()
            if bi in CHECKPOINTS:
                dump(f"e2e_b{bi}_out", h)
            elif bi % 8 == 7:
                print(f"  block {bi} done", flush=True)

        velocity = LTXModel._process_output(None, sst64, norm_out, proj_out, h, emb_t)
        dump("e2e_out", velocity)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out}/manifest.json")
    print("E2E BUNDLE COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
