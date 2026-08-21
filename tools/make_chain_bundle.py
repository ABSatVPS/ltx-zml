#!/usr/bin/env python3
"""Phase 3 checkpoint 1 oracle: blocks 0, 23, 47 chained in f64.

Consumes the FROZEN single-block bundle's inputs (same x, timestep,
context, prompt timestep, RoPE tables — regression control: the parent
bundle hash is recorded), runs the three real blocks sequentially the
way LTXModel does (x mutates; every other arg is shared), and dumps the
three boundary activations. Asserts the first boundary equals the
parent bundle's block_out exactly.

Usage: oracle-venv/bin/python tools/make_chain_bundle.py BUNDLE_DIR OUT_DIR W0 W23 W47
"""

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
    bundle, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    wdirs = [pathlib.Path(p) for p in sys.argv[3:6]]
    out.mkdir(parents=True, exist_ok=True)

    x = load_f32(bundle, "in_x.bin", (B, T, DIM))
    context = load_f32(bundle, "in_context.bin", (B, S, DIM))
    timestep = load_f32(bundle, "in_timestep.bin", (B, T, 9 * DIM))
    prompt_timestep = load_f32(bundle, "in_prompt_timestep.bin", (B, 1, 2 * DIM))
    cos = load_f32(bundle, "rope_cos.bin", (B, 32, T, 64))
    sin = load_f32(bundle, "rope_sin.bin", (B, 32, T, 64))
    parent_block_out = load_f32(bundle, "block_out.bin", (B, T, DIM))

    blocks = [load_block(w) for w in wdirs]
    manifest = {
        "parent_bundle_manifest_sha": hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest(),
        "blocks": [str(w) for w in wdirs],
        "torch": torch.__version__,
        "tensors": {},
    }

    h = x
    names = ["chain_b0_out", "chain_b23_out", "chain_b47_out"]
    for blk, name in zip(blocks, names):
        args = TransformerArgs(
            x=h, context=context, context_mask=None, timesteps=timestep,
            embedded_timestep=torch.zeros(B, 1, DIM, dtype=torch.float64),
            positional_embeddings=(cos, sin), cross_positional_embeddings=None,
            cross_scale_shift_timestep=None, cross_gate_timestep=None,
            enabled=True, prompt_timestep=prompt_timestep,
        )
        with torch.no_grad():
            vout, _ = blk.forward(args, None)
        h = vout.x
        arr = h.to(torch.float32).numpy()
        (out / f"{name}.bin").write_bytes(arr.tobytes())
        manifest["tensors"][name] = {
            "shape": list(arr.shape),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
            "max_abs": float(np.abs(arr).max()),
        }
        print(f"  {name} max_abs={np.abs(arr).max():.4g}")

    d0 = (load_f32(out, "chain_b0_out.bin", (B, T, DIM)) - parent_block_out).abs().max()
    assert d0 == 0.0, f"chain b0 != parent block_out: {d0}"
    print("chain_b0_out == parent block_out exactly")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
