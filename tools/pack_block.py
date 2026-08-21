#!/usr/bin/env python3
"""Pack one fetched (and optionally quantized) block into a single
contiguous blob + manifest — the streaming loader's on-disk unit
(E-STREAM-3 contract, notebook 2026-08-21/22).

Deterministic to the byte: entries in canonical name-sorted order,
offsets 64-byte aligned with zero fill, no timestamps anywhere. Packing
the same inputs twice yields the same blob digest (G-RING-1).

Input: a block directory from tools/fetch_block.py (manifest.json plus
per-tensor .bin files), with any *_q8.bin / *_q8scale.bin companions
from tools/quantize_block.py included automatically.

Output: BLOCKDIR/blob.bin and BLOCKDIR/blob_manifest.json with per-entry
name/dtype/shape/offset/nbytes/sha256, the blob sha256, the source
checkpoint revision, and the packer version + git commit.

Usage: python3 tools/pack_block.py BLOCKDIR
"""

import hashlib
import json
import pathlib
import subprocess
import sys

PACKER_VERSION = 1
ALIGN = 64

Q8_SHAPES = {"q8": None, "q8scale": None}  # derived below


def main() -> int:
    wdir = pathlib.Path(sys.argv[1])
    man = json.loads((wdir / "manifest.json").read_text())

    entries = []  # (name, path, dtype, shape)
    for name, e in man["tensors"].items():
        entries.append((name, wdir / e["file"], e["dtype"], e["shape"]))
        # Companion quantized files (int8-g128 recipe): shape derived from
        # the source weight ([O, I] for q8, [O, I/128] f32 for scales).
        stem = e["file"].removesuffix(".bin")
        q8 = wdir / f"{stem}_q8.bin"
        if q8.exists():
            o, i = e["shape"]
            entries.append((name + "::q8", q8, "I8", [o, i]))
            entries.append((name + "::q8scale", wdir / f"{stem}_q8scale.bin", "F32", [o, i // 128]))

    entries.sort(key=lambda t: t[0])  # canonical order

    blob = bytearray()
    manifest_entries = []
    for name, path, dtype, shape in entries:
        data = path.read_bytes()
        if len(blob) % ALIGN:
            blob.extend(b"\x00" * (ALIGN - len(blob) % ALIGN))
        offset = len(blob)
        blob.extend(data)
        manifest_entries.append({
            "name": name, "dtype": dtype, "shape": shape,
            "offset": offset, "nbytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })

    # Integrity: offsets strictly increasing, no overlap (gaps are only
    # the deterministic alignment fill).
    end = 0
    for e in manifest_entries:
        assert e["offset"] >= end and e["offset"] - end < ALIGN, f"bad layout at {e['name']}"
        end = e["offset"] + e["nbytes"]
    assert end == len(blob), "trailing bytes"

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=pathlib.Path(__file__).parent,
    ).stdout.strip() or "unknown"

    out_manifest = {
        "packer_version": PACKER_VERSION,
        "packer_commit": commit,
        "align": ALIGN,
        "block": man["block"],
        "repo": man["repo"],
        "revision": man["revision"],
        "source_file": man["file"],
        "model_version": man.get("model_version", ""),
        "total_bytes": len(blob),
        "blob_sha256": hashlib.sha256(bytes(blob)).hexdigest(),
        "entries": manifest_entries,
    }
    (wdir / "blob.bin").write_bytes(bytes(blob))
    (wdir / "blob_manifest.json").write_text(json.dumps(out_manifest, indent=1))
    print(f"block {man['block']}: {len(manifest_entries)} entries, "
          f"{len(blob)} bytes, sha256 {out_manifest['blob_sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
