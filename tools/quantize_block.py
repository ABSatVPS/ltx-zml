#!/usr/bin/env python3
"""Quantize selected projection-class weights of a fetched block to int4.

Rung-1 scheme (pre-registered in the notebook): per-output-row symmetric
absmax — q = clamp(round(w/s), -8, 7), s = absmax/7 per row. Output files
sit beside the source bins: NAME_q4.bin (unpacked u4, one value per byte,
offset-8 applied: stored 0..15) and NAME_q4scale.bin (f32 per row).
Reports weight-space rel-RMS per tensor (informative, not a gate).

Usage: python3 tools/quantize_block.py WEIGHTS_DIR [--class qkv|out|ff]
"""

import argparse
import json
import pathlib
import sys

import numpy as np

CLASSES = {
    "qkv": ("attn1.to_q.weight", "attn1.to_k.weight", "attn1.to_v.weight",
            "attn2.to_q.weight", "attn2.to_k.weight", "attn2.to_v.weight"),
    "out": ("attn1.to_out.0.weight", "attn2.to_out.0.weight"),
    "ff": ("ff.net.0.proj.weight", "ff.net.2.weight"),
}


def bf16_to_f32(raw: bytes, shape) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u16.view(np.float32).reshape(shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wdir")
    ap.add_argument("--cls", default="qkv", choices=CLASSES)
    ap.add_argument("--gs", type=int, default=0, help="group size (0 = per-row)")
    ap.add_argument("--bits", type=int, default=4, choices=(4, 8))
    args = ap.parse_args()
    qmax = 7 if args.bits == 4 else 127
    lo = -8 if args.bits == 4 else -128
    wdir = pathlib.Path(args.wdir)
    man = json.loads((wdir / "manifest.json").read_text())

    for name, e in man["tensors"].items():
        if not any(name.endswith(f"{suffix}") for suffix in
                   (f"transformer_blocks.{man['block']}." + s for s in CLASSES[args.cls])) \
           and not any(name.split(".", 2)[-1] == s for s in CLASSES[args.cls]):
            continue
        assert e["dtype"] == "BF16", f"{name}: expected BF16, got {e['dtype']}"
        w = bf16_to_f32((wdir / e["file"]).read_bytes(), e["shape"]).astype(np.float64)
        if args.gs:
            o, i = w.shape
            wg = w.reshape(o, i // args.gs, args.gs)
            absmax = np.abs(wg).max(axis=2, keepdims=True)
            scale3 = np.where(absmax > 0, absmax / qmax, 1.0)
            q = np.clip(np.round(wg / scale3), lo, qmax).astype(np.int8).reshape(o, i)
            deq = (q.reshape(o, i // args.gs, args.gs).astype(np.float64) * scale3).reshape(o, i)
            scale = scale3.squeeze(2)  # [O, G] f32 file
        else:
            absmax = np.abs(w).max(axis=1, keepdims=True)
            scale = np.where(absmax > 0, absmax / qmax, 1.0)
            q = np.clip(np.round(w / scale), lo, qmax).astype(np.int8)
            deq = q.astype(np.float64) * scale
        rel = float(np.sqrt(((deq - w) ** 2).mean()) / (np.sqrt((w ** 2).mean()) + 1e-20))

        stem = e["file"].removesuffix(".bin")
        if args.bits == 4:
            (wdir / f"{stem}_q4.bin").write_bytes((q + 8).astype(np.uint8).tobytes())
            (wdir / f"{stem}_q4scale.bin").write_bytes(scale.astype(np.float32).tobytes())
        else:
            (wdir / f"{stem}_q8.bin").write_bytes(q.astype(np.int8).tobytes())
            (wdir / f"{stem}_q8scale.bin").write_bytes(scale.astype(np.float32).tobytes())
        print(f"  {name}: weight rel-RMS {rel:.4f} scale range [{scale.min():.3e}, {scale.max():.3e}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
