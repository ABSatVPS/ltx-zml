#!/usr/bin/env bash
# Wire this repo into pinned checkouts of its two upstreams. Nothing from
# either project is vendored here: we clone them, apply small patches, and
# drop the coli/ Bazel package into the ZML workspace.
#
#   ./scripts/setup.sh [workdir]        # default: ./.work
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-$REPO/.work}"

# Pinned upstream revisions this experiment was validated against.
ZML_URL=https://github.com/zml/zml.git
ZML_REV=bf7e277
COLIBRI_URL=https://github.com/JustVugg/colibri.git
COLIBRI_REV=2d62381

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

say "refreshing ROCm package checksums (AMD republishes .debs in place)"
python3 "$REPO/tools/fix_rocm_lockfile.py" "$WORK/zml/platforms/rocm/packages.lock.json"

# ---- colibri --------------------------------------------------------------
if [ ! -d "$WORK/colibri/.git" ]; then
    say "cloning colibri @ $COLIBRI_REV"
    git clone "$COLIBRI_URL" "$WORK/colibri"
    git -C "$WORK/colibri" checkout --quiet "$COLIBRI_REV"
fi

say "patching colibri: benchmark validation + env isolation"
git -C "$WORK/colibri" apply --reverse --check "$REPO/patches/colibri-bench-validation.patch" 2>/dev/null \
    && echo "    already applied" \
    || git -C "$WORK/colibri" apply "$REPO/patches/colibri-bench-validation.patch"

say "adding the real-dimension fixture generator"
cp "$WORK/colibri/c/tools/make_glm_bench_model.py" "$WORK/colibri/c/tools/make_glm_real_fixture.py"
patch -s -N -r /dev/null "$WORK/colibri/c/tools/make_glm_real_fixture.py" \
    < "$REPO/patches/colibri-real-dims-fixture.patch" || true

# ---- the Bazel package ----------------------------------------------------
say "installing the coli/ package into the ZML workspace"
mkdir -p "$WORK/zml/coli/engine" "$WORK/zml/coli/tests"
cp "$REPO"/coli/*.zig "$REPO"/coli/BUILD.bazel "$WORK/zml/coli/"

# The engine and the conformance suite are colibri's own sources, copied in at
# build time (Apache-2.0, © the colibri authors) with two mechanical edits:
#   - main() renamed so a Zig main can call it
#   - uring.h guarded with __has_include (hermetic sysroots lack kernel UAPI)
say "importing colibri engine sources (renaming main, guarding io_uring)"
cp "$WORK"/colibri/c/*.h "$WORK/zml/coli/engine/"
cp "$WORK/colibri/c/colibri.c" "$WORK/zml/coli/engine/"
cp "$WORK/colibri/c/backend_cuda.h" "$WORK/zml/coli/"
cp "$WORK/colibri/c/tests/test_backend_cuda.cu" "$WORK/zml/coli/tests/test_backend_cuda.cc"

sed -i 's/^int main(int argc, char \*\*argv){/int coli_engine_main(int argc, char **argv){/' \
    "$WORK/zml/coli/engine/colibri.c"
sed -i 's/^extern "C" int coli_test_main/int coli_test_main/; s/^int main(int argc, char \*\*argv) {/extern "C" int coli_test_main(int argc, char **argv) {/' \
    "$WORK/zml/coli/tests/test_backend_cuda.cc"
python3 - "$WORK/zml/coli/engine/uring.h" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old = "#ifdef __linux__"
new = "#if defined(__linux__) && __has_include(<linux/io_uring.h>)"
if old in s and new not in s:
    s = s.replace(old, new, 1)
    s = s.replace("#endif /* __linux__ */", pathlib.Path(__file__).parent.as_posix() and """#else /* no <linux/io_uring.h> in this sysroot: stub API, init fails so the
       * engine falls back to positioned reads at runtime. */
#include <stdint.h>
#include <stddef.h>
struct io_uring_cqe { uint64_t user_data; int32_t res; uint32_t flags; };
typedef struct { int fd; } ColiUring;
static inline int coli_uring_init(ColiUring *r, unsigned e){ (void)r; (void)e; return -1; }
static inline void coli_uring_close(ColiUring *r){ (void)r; }
static inline int coli_uring_set_workers(ColiUring *r, unsigned w){ (void)r; (void)w; return -1; }
static inline int coli_uring_prep_read(ColiUring *r, int fd, void *b, size_t l, uint64_t o, uint64_t t){ (void)r; (void)fd; (void)b; (void)l; (void)o; (void)t; return -1; }
static inline int coli_uring_enter(ColiUring *r, unsigned m){ (void)r; (void)m; return -1; }
static inline int coli_uring_peek(ColiUring *r, struct io_uring_cqe *o){ (void)r; (void)o; return 0; }
#endif /* __linux__ */""", 1)
    p.write_text(s)
    print("    uring.h guarded")
else:
    print("    uring.h already guarded")
PY

# OpenMP: the hermetic clang ships no omp.h, so use the system one (the engine's
# CPU half — shared expert, attention, lm_head — is otherwise single-threaded).
OMP_H="$(ls /usr/lib/gcc/*/*/include/omp.h 2>/dev/null | head -1 || true)"
if [ -n "$OMP_H" ]; then
    cp "$OMP_H" "$WORK/zml/coli/engine/omp.h"
    say "vendored system omp.h from $OMP_H"
else
    say "WARNING: no system omp.h found — drop -fopenmp from coli/BUILD.bazel"
fi

cat <<EOF

$(say "ready")
  cd $WORK/zml
  $WORK/bin/bazelisk run --@zml//platforms:rocm=true //coli:conformance   # upstream's GPU suite
  $WORK/bin/bazelisk run --@zml//platforms:rocm=true //coli:smoke         # correctness + benches
  $WORK/bin/bazelisk build --@zml//platforms:rocm=true //coli:coli        # the engine

First build downloads a hermetic ROCm userland and toolchains (~50 GB cache,
~10 min). See docs/reproduce.md for the fixture and end-to-end run.
EOF
