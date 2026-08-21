#!/usr/bin/env python3
"""Assembly fetch pipeline: for every video transformer block not yet on
disk, run fetch → quantize (int8-g128, qkv + ff classes) → pack, each
from the existing single-block tools, resumably.

A block is considered DONE when its blob_manifest.json exists, records
the pinned revision, and its blob.bin matches the recorded digest. Rerun
the script after any interruption; completed blocks are skipped.

Usage: python3 tools/assemble_blocks.py [--root .work] [--blocks 0-47]
"""

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

TOOLS = pathlib.Path(__file__).parent
PINNED_REVISION = "6c7e5e573ac1667efc83407806fe9b0b93730e60"


def done(bdir: pathlib.Path) -> bool:
    mf = bdir / "blob_manifest.json"
    if not mf.exists():
        return False
    m = json.loads(mf.read_text())
    if m.get("revision") != PINNED_REVISION:
        print(f"  {bdir.name}: blob from revision {m.get('revision')!r} != pinned — refetching")
        return False
    blob = bdir / "blob.bin"
    if not blob.exists() or blob.stat().st_size != m["total_bytes"]:
        return False
    digest = hashlib.sha256(blob.read_bytes()).hexdigest()
    return digest == m["blob_sha256"]


def run(desc: str, *cmd: str) -> None:
    t0 = time.monotonic()
    r = subprocess.run(list(cmd), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], file=sys.stderr)
        print(r.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"FAILED: {desc}")
    print(f"  {desc}: OK ({time.monotonic() - t0:.0f}s)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".work")
    ap.add_argument("--blocks", default="0-47")
    args = ap.parse_args()
    a, b = args.blocks.split("-")
    root = pathlib.Path(args.root)
    py = sys.executable

    todo = []
    for i in range(int(a), int(b) + 1):
        bdir = root / f"block{i}"
        if done(bdir):
            continue
        todo.append((i, bdir))
    print(f"{len(todo)} blocks to assemble")

    for n, (i, bdir) in enumerate(todo):
        print(f"[{n + 1}/{len(todo)}] block {i}")
        if not (bdir / "manifest.json").exists():
            run("fetch", py, str(TOOLS / "fetch_block.py"), str(bdir), "--block", str(i))
        run("quantize qkv", py, str(TOOLS / "quantize_block.py"), str(bdir), "--cls", "qkv", "--gs", "128", "--bits", "8")
        run("quantize ff", py, str(TOOLS / "quantize_block.py"), str(bdir), "--cls", "ff", "--gs", "128", "--bits", "8")
        run("pack", py, str(TOOLS / "pack_block.py"), str(bdir))
        if not done(bdir):
            raise SystemExit(f"block {i}: post-pack verification failed")
        print(f"  block {i}: verified")

    print("ASSEMBLY FETCH COMPLETE" if todo else "nothing to do")
    return 0


if __name__ == "__main__":
    sys.exit(main())
