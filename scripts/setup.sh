#!/usr/bin/env bash
# Wire this repo into a pinned checkout of ZML. Nothing is vendored: we
# clone, apply one patch, and drop the ltx/ Bazel package into the workspace.
#
#   ./scripts/setup.sh [workdir]        # default: ./.work
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-$REPO/.work}"

# Pinned upstream revision — the same rev coli-zml was validated against.
ZML_URL=https://github.com/zml/zml.git
ZML_REV=bf7e277

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

mkdir -p "$WORK"

# ---- bazelisk (user-local; no sudo, no system Bazel) ----------------------
if [ ! -x "$WORK/bin/bazelisk" ]; then
    say "fetching bazelisk"
    mkdir -p "$WORK/bin"
    curl -fsSL -o "$WORK/bin/bazelisk" \
        https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
    chmod +x "$WORK/bin/bazelisk"
fi

# ---- ZML ------------------------------------------------------------------
if [ ! -d "$WORK/zml/.git" ]; then
    say "cloning ZML @ $ZML_REV"
    git clone "$ZML_URL" "$WORK/zml"
    git -C "$WORK/zml" checkout --quiet "$ZML_REV"
fi

say "patching ZML: PJRT sub-byte host transfers (u4/u2 weights)"
git -C "$WORK/zml" apply --reverse --check "$REPO/patches/zml-pjrt-subbyte-transfer.patch" 2>/dev/null \
    && echo "    already applied" \
    || git -C "$WORK/zml" apply "$REPO/patches/zml-pjrt-subbyte-transfer.patch"

say "patching ZML: expose the generic N-D convolution (conv3d for the VAE)"
git -C "$WORK/zml" apply --reverse --check "$REPO/patches/zml-pub-convolution.patch" 2>/dev/null \
    && echo "    already applied" \
    || git -C "$WORK/zml" apply "$REPO/patches/zml-pub-convolution.patch"

say "refreshing ROCm package checksums (AMD republishes .debs in place)"
python3 "$REPO/tools/fix_rocm_lockfile.py" "$WORK/zml/platforms/rocm/packages.lock.json"

# ---- the Bazel package ----------------------------------------------------
say "installing the ltx/ package into the ZML workspace"
mkdir -p "$WORK/zml/ltx"
cp "$REPO"/ltx/*.zig "$REPO"/ltx/BUILD.bazel "$WORK/zml/ltx/"

cat <<EOF

$(say "ready")
  cd $WORK/zml
  $WORK/bin/bazelisk run --@zml//platforms:rocm=true //ltx:smoke

First build downloads a hermetic ROCm userland and toolchains (~50 GB cache,
~10 min). Re-run this script after editing ltx/ in the repo root.
EOF
