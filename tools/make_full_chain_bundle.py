#!/usr/bin/env python3
"""The 48-block stage-walk oracle (assembly checkpoint, notebook
2026-08-21 evening): all video blocks chained in torch CPU f64 on the
frozen Phase 2 inputs, dumping the running x after blocks 7, 15, 23,
31, 39, and 47, plus block 0's output as the control.

Blocks load ONE AT A TIME (peak memory one f64 block); x carries in
f64 between blocks and every dump is a single f64→f32 cast — the same
convention the 3-chain bundle used, so error scales stay comparable.

Controls, asserted before anything is written:
  - block 0's output must equal the Phase 2 bundle's block_out EXACTLY;
  - re-running the 0/23/47-only chain must reproduce the existing
    chain_bundle dumps byte-for-byte (the 3-chain stays the anchor).

Usage: oracle-venv/bin/python tools/make_full_chain_bundle.py \
           BUNDLE_DIR CHAIN_DIR WORK_ROOT OUT_DIR
"""

import gc
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch

from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
from ltx_core.model.transformer.transformer_args import TransformerArgs

B, T, S, DIM = 1, 64, 32, 4096
CHECKPOINTS = [7, 15, 23, 31, 39, 47]
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
    blk = BasicAVTransformerBlock(video=cfg, audio=None, rope_type=LTXRopeType.SPLIT, norm_eps=1e-6)
    blk.load_state_dict(sd, strict=False)
    return blk.to(torch.float64).eval()


def load_f32(d: pathlib.Path, name: str, shape) -> torch.Tensor:
    a = np.frombuffer((d / name).read_bytes(), dtype=np.float32).reshape(shape).copy()
    return torch.from_numpy(a).to(torch.float64)


def main() -> int:
    bundle, chain_dir, root, out = (pathlib.Path(p) for p in sys.argv[1:5])
    out.mkdir(parents=True, exist_ok=True)

    x0 = load_f32(bundle, "in_x.bin", (B, T, DIM))
    context = load_f32(bundle, "in_context.bin", (B, S, DIM))
    timestep = load_f32(bundle, "in_timestep.bin", (B, T, 9 * DIM))
    prompt_timestep = load_f32(bundle, "in_prompt_timestep.bin", (B, 1, 2 * DIM))
    cos = load_f32(bundle, "rope_cos.bin", (B, 32, T, 64))
    sin = load_f32(bundle, "rope_sin.bin", (B, 32, T, 64))
    parent_block_out = load_f32(bundle, "block_out.bin", (B, T, DIM))

    def fwd(blk, h):
        args = TransformerArgs(
            x=h, context=context, context_mask=None, timesteps=timestep,
            embedded_timestep=torch.zeros(B, 1, DIM, dtype=torch.float64),
            positional_embeddings=(cos, sin), cross_positional_embeddings=None,
            cross_scale_shift_timestep=None, cross_gate_timestep=None,
            enabled=True, prompt_timestep=prompt_timestep,
        )
        with torch.no_grad():
            vout, _ = blk.forward(args, None)
        return vout.x

    # Control 2: the 0/23/47 chain must reproduce the frozen chain bundle.
    h = x0
    for bi, name in ((0, "chain_b0_out"), (23, "chain_b23_out"), (47, "chain_b47_out")):
        blk = load_block(root / f"block{bi}")
        h = fwd(blk, h)
        want = (chain_dir / f"{name}.bin").read_bytes()
        got = h.to(torch.float32).numpy().tobytes()
        assert got == want, f"3-chain control drifted at {name}"
        del blk
        gc.collect()
    print("control: 0/23/47 chain reproduces the frozen chain bundle byte-for-byte")

    # The full walk.
    manifest = {
        "parent_bundle_manifest_sha": hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest(),
        "checkpoints": CHECKPOINTS,
        "torch": torch.__version__,
        "tensors": {},
    }
    h = x0
    for bi in range(48):
        blk = load_block(root / f"block{bi}")
        h = fwd(blk, h)
        del blk
        gc.collect()
        if bi == 0:
            # Compare f32-cast vs f32-cast, like the chain generator did:
            # parent_block_out IS an f32-rounded dump, so the raw f64 h
            # differs from it by f32 eps by construction.
            d0 = (h.to(torch.float32).to(torch.float64) - parent_block_out).abs().max()
            assert d0 == 0.0, f"block 0 != parent block_out: {d0}"
            print("control: block 0 output == Phase 2 block_out exactly (f32-cast)")
        if bi in CHECKPOINTS:
            arr = h.to(torch.float32).numpy()
            name = f"walk_b{bi}_out"
            (out / f"{name}.bin").write_bytes(arr.tobytes())
            manifest["tensors"][name] = {
                "shape": list(arr.shape),
                "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                "max_abs": float(np.abs(arr).max()),
                "rms": float(np.sqrt((arr.astype(np.float64) ** 2).mean())),
            }
            print(f"  {name}: max_abs={np.abs(arr).max():.4g} rms={manifest['tensors'][name]['rms']:.4g}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out}/manifest.json")
    print("FULL CHAIN BUNDLE COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
