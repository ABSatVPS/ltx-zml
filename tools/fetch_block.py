#!/usr/bin/env python3
"""Fetch a subset of LTX-2.5 transformer tensors by HTTP range request.

No full-checkpoint download: reads the safetensors header, selects the
requested tensors (default: video-stream tensors of transformer block 0),
validates that their ranges are non-overlapping and exactly sized, then
range-requests each one and writes raw .bin files plus a manifest.json
with shapes, dtypes, SHA-256, and first/last elements (bf16 decoded).

The HF revision is pinned by commit hash at fetch time and recorded in
the manifest, so the bundle is reproducible even if `main` moves.

Usage:
  python3 tools/fetch_block.py OUTDIR [--block 0] [--file diffusion_models/...]
"""

import argparse
import hashlib
import json
import pathlib
import struct
import sys
import urllib.request

REPO = "Lightricks/LTX-2.5"
DEFAULT_FILE = "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"

# The revision every fetched block must come from. The original script
# pinned whatever `main` pointed to AT FETCH TIME — fine for the first
# fetch, silently wrong for the 45-block assembly fetch if upstream
# moves main in between (audit finding, 2026-08-21). Pass --revision
# main to deliberately re-resolve.
PINNED_REVISION = "6c7e5e573ac1667efc83407806fe9b0b93730e60"

# Video-stream prefixes for one block; audio and the A/V bridges are
# Phase 7's problem and excluded from the Phase 2 bundle.
VIDEO_KEYS = (
    "attn1.",
    "attn2.",
    "ff.",
    "scale_shift_table",
    "prompt_scale_shift_table",
)
AUDIO_MARKERS = ("audio", "a2v", "v2a", "video_to_audio", "audio_to_video")


def token() -> str:
    return pathlib.Path.home().joinpath(".cache/huggingface/token").read_text().strip()


def api_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token()}", "User-Agent": "curl/8"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def ranged(url: str, a: int, b: int) -> bytes:
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={a}-{b}", "Authorization": f"Bearer {token()}", "User-Agent": "curl/8"}
    )
    with urllib.request.urlopen(req) as r:
        data = r.read()
    assert len(data) == b - a + 1, f"short read: wanted {b - a + 1}, got {len(data)}"
    return data


DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "U8": 1}


def bf16_to_f32(two: bytes) -> float:
    return struct.unpack("<f", b"\x00\x00" + two)[0]


def first_last(data: bytes, dtype: str) -> tuple[float, float]:
    if dtype == "BF16":
        return bf16_to_f32(data[0:2]), bf16_to_f32(data[-2:])
    if dtype == "F32":
        return struct.unpack("<f", data[0:4])[0], struct.unpack("<f", data[-4:])[0]
    return float("nan"), float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--revision", default=PINNED_REVISION,
                    help="commit sha to fetch from ('main' re-resolves via the API)")
    args = ap.parse_args()
    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    sha = args.revision
    if sha == "main":
        sha = api_json(f"https://huggingface.co/api/models/{REPO}")["sha"]
    url = f"https://huggingface.co/{REPO}/resolve/{sha}/{args.file}"
    print(f"revision: {sha}")

    # Header.
    hlen = struct.unpack("<Q", ranged(url, 0, 7))[0]
    hdr_bytes = ranged(url, 8, 7 + hlen)
    hdr = json.loads(hdr_bytes.decode())
    meta = hdr.pop("__metadata__", {})
    data_base = 8 + hlen
    print(f"header: {hlen} bytes, {len(hdr)} tensors")

    prefix = f"model.diffusion_model.transformer_blocks.{args.block}."
    selected = {}
    for name, entry in hdr.items():
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):]
        if any(m in rest for m in AUDIO_MARKERS):
            continue
        if rest.startswith(VIDEO_KEYS):
            selected[name] = entry
    if not selected:
        print("nothing selected — wrong block index or prefix?", file=sys.stderr)
        return 1

    # Validate ranges before any allocation: sized exactly, no overlap.
    spans = []
    for name, e in selected.items():
        b, en = e["data_offsets"]
        n = 1
        for d in e["shape"]:
            n *= d
        expect = n * DTYPE_BYTES[e["dtype"]]
        assert en - b == expect, f"{name}: range {en - b} != shape bytes {expect}"
        spans.append((b, en, name))
    spans.sort()
    for (b1, e1, n1), (b2, e2, n2) in zip(spans, spans[1:]):
        assert e1 <= b2, f"overlap: {n1} [{b1},{e1}) vs {n2} [{b2},{e2})"
    total = sum(e - b for b, e, _ in spans)
    print(f"selected {len(selected)} tensors, {total / 1e6:.1f} MB")

    manifest = {
        "repo": REPO,
        "revision": sha,
        "file": args.file,
        "block": args.block,
        "header_sha256": hashlib.sha256(hdr_bytes).hexdigest(),
        "model_version": meta.get("model_version"),
        "tensors": {},
    }
    for i, (b, en, name) in enumerate(spans):
        e = selected[name]
        data = ranged(url, data_base + b, data_base + en - 1)
        short = name[len("model.diffusion_model."):]
        fname = short.replace(".", "_") + ".bin"
        (out / fname).write_bytes(data)
        f0, fl = first_last(data, e["dtype"])
        manifest["tensors"][short] = {
            "file": fname,
            "shape": e["shape"],
            "dtype": e["dtype"],
            "offsets": [b, en],
            "sha256": hashlib.sha256(data).hexdigest(),
            "first": f0,
            "last": fl,
        }
        print(f"  [{i + 1}/{len(spans)}] {short} {e['shape']} {e['dtype']} first={f0:.6g} last={fl:.6g}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
