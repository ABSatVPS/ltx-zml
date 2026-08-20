#!/usr/bin/env python3
"""Refresh the sha256 of every repo.amd.com .deb in ZML's ROCm lockfile.

AMD republishes the ROCm packages in place under the same version+filename, so
ZML's pinned checksums go stale and every `--@zml//platforms:rocm=true` build
dies mid-fetch. Rather than pin a second set of hashes that will also rot, read
the live APT index (dists/stable/.../Packages.gz, which the repo signs) and
rewrite the lockfile from it.

    python3 tools/fix_rocm_lockfile.py <path/to/packages.lock.json>
"""
import gzip
import io
import json
import sys
import urllib.request

INDEX = ("https://repo.amd.com/rocm/packages-multi-arch/ubuntu2204/"
         "dists/stable/main/binary-amd64/Packages.gz")


def live_checksums() -> dict[str, str]:
    """filename -> sha256, from the repository's own package index."""
    with urllib.request.urlopen(INDEX, timeout=120) as response:
        raw = response.read()
    index, current = {}, {}
    for line in gzip.open(io.BytesIO(raw), "rt"):
        line = line.rstrip("\n")
        if not line:
            if "Filename" in current:
                index[current["Filename"].rsplit("/", 1)[-1]] = current.get("SHA256")
            current = {}
        elif line.startswith(("Filename:", "SHA256:")):
            key, value = line.split(":", 1)
            current[key] = value.strip()
    if "Filename" in current:
        index[current["Filename"].rsplit("/", 1)[-1]] = current.get("SHA256")
    return index


def main() -> int:
    path = sys.argv[1]
    index = live_checksums()
    lock = json.load(open(path))
    stats = {"ok": 0, "fixed": 0, "missing": 0}

    def walk(node):
        if isinstance(node, dict):
            urls = node.get("urls")
            if urls and "sha256" in node and any("repo.amd.com" in u for u in urls):
                name = urls[0].rsplit("/", 1)[-1]
                live = index.get(name)
                if live is None:
                    stats["missing"] += 1
                    print(f"    not in index: {name}", file=sys.stderr)
                elif live != node["sha256"]:
                    node["sha256"] = live
                    stats["fixed"] += 1
                else:
                    stats["ok"] += 1
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(lock)
    if stats["fixed"]:
        with open(path, "w") as handle:
            json.dump(lock, handle, indent="\t")
    print(f"    checksums: {stats['ok']} current, {stats['fixed']} refreshed, "
          f"{stats['missing']} absent")
    return 1 if stats["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
