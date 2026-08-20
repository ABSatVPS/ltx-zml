# Lab notebook: colibrì × ZML on ROCm

The working log of this experiment, kept as written — including the wrong
turns, because the refuted hypotheses are the most useful part. Entries are
chronological over two days (2026-07-29/30); the finished state and the
headline numbers live in [`../README.md`](../README.md).

Goal: replace colibrì's hand-written CUDA/HIP GPU backend with a Zig
implementation on ZML (PJRT/StableHLO), targeting an AMD RX 9060 XT (Navi 44,
gfx1200, RDNA 4) on Fedora 44 with 16 GB of RAM.

Paths below are as they were during the experiment: `colibri-upstream/` and
`zml/` were sibling clones. In this repository, `scripts/setup.sh` recreates
the same layout under `.work/`, and the two local patches mentioned in the
earliest entries were superseded — the rocWMMA one by upstream's own fix
(see the last entry) and the lockfile one by `tools/fix_rocm_lockfile.py`.

## Key findings (2026-07-29)

1. **Upstream already runs on ROCm.** `make -C c glm HIP=1` compiles the one
   CUDA source through `backend_gpu_compat.h` (macro mapping, hipcc). Tested
   upstream on gfx1201/ROCm 7.2.4. So the experiment's novelty is the
   ZML-based backend, not ROCm support per se.
2. **The GPU backend is a clean C ABI seam**: ~43 `coli_cuda_*` functions
   (`c/backend_cuda.h`). On Windows it is even a runtime-loaded DLL
   (`backend_loader.c`), proving the engine works against the ABI alone.
   A Zig `.so`/`.o` exporting these symbols can replace `backend_cuda.o`
   at link time.
3. **Conformance harness for free**: `c/tests/test_backend_cuda.cu` is pure
   host C++ (only includes `backend_cuda.h`) — compile with g++, link against
   the Zig backend, run. Covers init/stats, upload fmt 0/1/2/3/6 (f32, q8,
   q4, q2, E8), error paths, cached-tensor matmul, `tensor_update`,
   `expert_mlp`, `expert_group`, E8 codebook grid.
4. **Backend call-site tiers** (from grep of `colibri.c`):
   - Tier 1 "expert offload" (first milestone): init/shutdown/device info/
     stats, `tensor_upload(_g)`, `matmul`, `expert_mlp`, `expert_group`
     (+ `_issue`/`_take`), `tensor_free/bytes/device/update`, `e8_set_grid`.
   - Tier 2: MLA attention family (`attention_absorb*`, `attention_project*`).
   - Tier 3: `pipe_*` resident pipeline — raw device pointers, scratch slots,
     peer copies (gated behind `COLI_CUDA_PIPE=1` etc.). Hardest to map onto
     PJRT buffers; do last or never.
5. **ZML bundles its own ROCm 7.14 userland + PJRT plugin** via Bazel
   (hermetic, patchelf'd debs) — system ROCm install is irrelevant; only the
   amdgpu kernel driver matters. **gfx1200 is a fully supported family**
   (blas/dnn/fft/rccl/solver/sparse packages exist for it).
6. Machine: Fedora 44, ROCm 7.1 system packages (hipcc present but
   `rocm-hip-devel` + `rocwmma-devel` NOT installed — baseline `make hip-test`
   needs `sudo dnf install rocm-hip-devel rocwmma-devel`). No passwordless
   sudo for the agent.
7. Engine returns-0-fallback: backend calls return 0 on failure and the
   engine falls back to the CPU path, so a partial backend is viable from
   day one.

## Milestones

1. ✅ Recon + workspace bootstrap
2. ✅ Go/no-go (2026-07-29): MNIST ran on the RX 9060 XT via `libpjrt_rocm.so`
   (~9 min first build, 498 ms model compile). ZML on gfx1200 works.
3. ⏳ Zig backend skeleton: `zml/coli/smoke.zig` (`bazelisk run
   --@zml//platforms:rocm=true //coli:smoke`) — Backend struct with
   upload (host-side dequant, fmt 0/1) + matmul via per-(S,I,O) compiled
   executables cached in a hash map. **PASSES upstream q8/f32 conformance
   vectors on the GPU.** Measured: first call 51.6 ms (incl. XLA compile),
   then ~0.66 ms/call round-trip (naive: per-call x upload, sync y download,
   per-call args alloc — optimization headroom is real). Still to do: fmt
   2/3/4/6 dequant (port from `quant.h`, then move into the graph),
   `expert_mlp`/`expert_group`, tensor_update/free/stats accounting.
   Note: don't format `*Platform` with `{f}` — zml's `Platform.format` has a
   latent bug (`self.devices()` call on a field).
4. Conformance: g++-compiled `test_backend_cuda.cu` linked against the Zig
   backend; pass all cases.
5. Wire into colibri: `ZML=1` make variant linking the Zig `.so`; run engine
   with `COLI_CUDA=1` on a small MoE (`olmoe.c` path — OLMoE is ~7B, vs GLM-5.2
   which needs ~370 GB on disk).
6. Perf: batch/fuse at larger granularity (whole expert group as one XLA
   program), compare against upstream HIP backend baseline (milestone 0:
   `hip-test` + tok/s once headers are installed).

## Measurements (2026-07-29, RX 9060 XT)

Baseline needs `rocm-hip-devel` + `rocwmma-devel`; Fedora quirk: hipcc does
not auto-link the HIP runtime — append `-lamdhip64` (in /usr/lib64).

- **Upstream HIP baseline: PASS.** `make hip-test HIP_ARCH=gfx1200
  ROCM_HOME=/usr` + `-lamdhip64` → "q8/q4/q2/f32/e8 correctness ok".
  (`tests/bench_tensor_core.cu` had bitrot — old 8-arg `tensor_upload`;
  locally fixed to `tensor_upload_g`.)
- **HIP `expert_group` round trip** (full gate/up/down MLP at real expert
  shapes, H2D+kernels+D2H, `make cuda-bench HIP=1`): rows=1 scalar 0.61 ms,
  packed 0.32 ms. (Ignore the "tensor" column: HIP grouped_s4 WMMA is a
  documented no-op, BUG-002, and iterations=3 makes it noisy.)
- **ZML skeleton phase breakdown** (2×4 q8 matmul, 512 reps): mean 587 µs =
  args-alloc 40 + x-upload (`Buffer.fromSlice`) 221 + launch 25 + sync+
  download (`toSliceAlloc`) 250 + cleanup 50. p50 608, p95 729, p99 803 µs.
  Transfers are ~48 bytes → this is NOT DMA/copy bandwidth; it is per-call
  buffer-lifecycle + synchronization latency (two blocking sync points).
- **Persistent args/results containers alone made tails WORSE**: mean 944 µs,
  p50 499 (better), p99 9,487, max 38,339 µs — naive container reuse
  interacts badly with PJRT event lifecycles. The real fix is graph-level
  batching (whole expert group = one executable), resident activations, and
  pinned staging for the big (19 MB) expert uploads — not arg reuse.
- Context: the HIP backend does a full expert MLP in ~the same wall time our
  PJRT path spends on a toy matmul → both stacks share a ~0.3–0.6 ms
  round-trip latency floor; colibri upstream already amortizes it via group
  calls + async issue/take + the resident pipe. Our fusion story must do the
  equivalent.

## Milestone: formats + fused expert ops (2026-07-29, later)

`coli/backend.zig` + `coli/smoke.zig`: fmt 0/1/2/3/4 upload (host dequant),
matmul, **fused expert_mlp** (one executable: gate/up/silu/mul/down, no host
round trips inside), expert_group (loop of fused MLPs). All conformance
vectors pass on GPU, including a self-derived grouped-int4 (gs=2) case.

**Decode gotchas (authoritative: quant.h):** int4 fmt 2/4 is OFFSET-8
(`(b&0xF)-8`), low nibble first — NOT two's complement (the 2c code in
`weight_at`/W4A16 kernels is a different path). int2 fmt 3 is offset-2
(`(v&3)-2`). The upstream q2/q4 test vectors do NOT discriminate offset vs 2c
encodings (both give identical dots by coincidence) — verify against quant.h,
not just the vectors. ZML: each compileFn argument needs a DISTINCT Tensor
spec (same spec twice → "Tensor already used as argument" panic).

**Matched-shape bench** (D=6144, I=2048, int4, same fills as upstream
`cuda-bench`; HIP numbers at 100 iters — bench_tensor_core.cu locally patched
from 3):

| rows | HIP scalar | HIP packed | ZML fused MLP (mean, p50/p95/p99 µs) |
|---|---|---|---|
| 1 | 0.576 ms | 0.289 ms | 2.386 ms (2377/2457/2869) |
| 2 | 0.972 ms | 0.469 ms | 2.342 ms |
| 4 | 1.528 ms | 0.715 ms | 2.379 ms |
| 8 | 2.586 ms | 1.324 ms | 2.357 ms |

ZML phase split is flat vs rows: upload ~230–275 µs, launch ~65 µs, sync+dl
~2.05 ms. Diagnosis: host-dequant stores weights f32 on device → the GEMV
reads 8× the weight bytes of HIP's packed int4 (~150 MB vs ~19 MB per call);
at S≤8 this is weight-bandwidth-bound, hence flat and ~8× slower. Note ZML
already ~ties HIP *scalar* at rows=8 and its flat curve wins at larger S.

**Next optimization (clearly #1): device-side dequant** — upload packed
int4/int8 bytes + scales as u8/f32 buffers, express dequant in the StableHLO
graph so XLA fuses it into the dot (what HIP kernels do by hand). Target:
sync+dl 2.05 ms → ~0.3–0.5 ms. Then: E8 fmt=6 (host FWHT port; fold the down
rotation into dequantized down weights at upload — expert_mlp fmt6 requires
rotating the silu product, see test_fmt6), then the C ABI + conformance run.

## Milestone: device-side dequant (2026-07-29, evening) — DONE

Weights now stay in packed quantized form on device (`u4`/`u2`/`i8` dtypes +
f32 scale buffers); dequant is traced into the graph. All conformance
vectors still pass. Three hard-won findings:

1. **PJRT sub-byte transfer contract** (ROCm plugin): host buffer is
   UNPACKED (one element per byte, low bits), `byte_strides` = null, device
   layout = null (plugin default packs 4/2 bits per element). Passing an
   explicit strides layout errors; passing packed host bytes silently
   truncates (reads low nibble per byte, then garbage — diagnose via S=2
   probes). Required patching zml: optional `layout` in
   `pjrt/pjrt.zig` BufferFromHostBufferArgs + a sub-byte branch in
   `zml/buffer.zig` from().
2. **`dot_general` on a dequantized operand MATERIALIZES the f32 weights**
   in VRAM every call — measured 3.0–3.4 ms, WORSE than resident f32
   (2.36 ms). XLA/ROCm does not fuse convert+scale into gemm operands.
3. **Broadcast-mul + `sum(.i)` instead of dot fuses fully**: the generated
   kernel reads packed u4 directly. `qdot` in backend.zig; note
   `sum` keeps the reduced axis — `.squeeze(.i)` after.

**Result table** (fused expert MLP, D=6144 I=2048 int4, 64 reps):

| rows | HIP packed | ZML fused-reduce (mean) | p50/p95/p99 µs | vs HIP |
|---|---|---|---|---|
| 1 | 0.289 ms | 0.525 ms | 509/549/1282 | 1.8× slower |
| 2 | 0.469 ms | 0.623 ms | 615/675/739 | 1.3× slower |
| 4 | 0.715 ms | 0.831 ms | 813/907/1299 | 1.16× slower |
| 8 | 1.324 ms | **1.218 ms** | 1201/1314/1396 | **1.09× FASTER** |

Effective weight-read bandwidth ≈69 GB/s — same as the hand-written HIP
kernels (≈66 GB/s). The XLA-generated quantized GEMV is bandwidth-competitive
with hand-written HIP on gfx1200. The remaining rows=1 gap is per-call
x-upload latency (~214 µs of the 525) + launch (~37 µs): the pinned-staging /
persistent-args work (zml DmaAllocator) and batching group calls into one
executable are now the right next levers. qdot reads weights once per x row —
fine at decode S; switch to dot+materialize for prefill-sized S later.

## Milestone: fused group executable (2026-07-29, night)

`expertGroup` now compiles the WHOLE group (N ≤ 8 same-shaped quantized
experts × 1 row — the decode pattern) into ONE executable via a comptime
`Group(N)` model struct ([N]Tensor fields, per-expert qdot subgraphs,
concatenated output; inline-switch dispatch on runtime count). One upload,
one launch, one sync per group instead of N round trips. Fallback: loop.

**Decode-unit results** (8 DISTINCT int4 experts × 1 row, D=6144, I=2048):

| path | time |
|---|---|
| HIP grouped kernels (packed, output-validated) | 1.557 ms |
| ZML fused single executable | **1.67 ms p50** / 1.82 p95 |
| ZML loop of 8 fused MLPs | 5.43 ms p50 |

ZML is within **7%** of the hand-written HIP grouped kernels at the real
decode unit. Both sit at ~90–97 GB/s effective weight-read bandwidth
(151 MB packed per call). Run-to-run variance note: single-MLP numbers
drifted ~1.5× between runs (clock states?); comparisons within one run.

**Trap found in upstream's own bench**: `bench_tensor_core.cu`'s `tensor_ms`
column is meaningless under HIP — `COLI_CUDA_TC_INT4` routes to
`grouped_s4_wmma`, the documented BUG-002 no-op, and the stale pinned result
buffer makes the rms check compare leftovers with themselves (rms 0.00000,
looks perfect, computes nothing). Also: env vars set by earlier bench modes
leak into later ones — my first count=8 number (0.119 ms = 1.3 TB/s,
physically impossible) came from that leak. Sanity-check derived bandwidth
against hardware limits before believing any GPU benchmark number.
Upstreamable: bench validation + the env leak fix.

Also learned: zml's `DmaAllocator` is a PASSTHROUGH on ROCm (pinned-memory
machinery exists only for CUDA/TPU/oneAPI) — pinned x-staging needs plugin
work, deprioritized; the fused group made per-call upload overhead 1/8th
as relevant anyway.

## Milestone: E8 (fmt=6) — DONE (2026-07-29, late night)

Full E8/IQ3 support: host decode at upload (ported from quant.h: 98-byte
super-blocks, 256×4 codebook via `setE8Grid`, parity-completed sign groups,
fp16 block scales), plus the expert-MLP **down-input rotation folded into the
decoded down rows at upload** — `row·(Qᵀh) == (Q·row)·h`, where Q = D·H/√n
and Qᵀ is quant.h's `e8_fwht` (signs→butterfly→scale); the fold applies
butterfly→scale→signs. Block-diagonal tiling for non-pow2 dims (lowest set
bit, cap 32768). fmt=6 tensors carry a second `down_folded` buffer (2× VRAM,
acceptable until decode moves in-graph). Smoke has an INDEPENDENT reference
ported from the test's own decoder (not quant.h) — preserving upstream's
two-implementations cross-check. Both e8 matmul (rms ≤ 1e-4) and expert_mlp
(rms ≤ 2e-4) pass on the GPU.

Port-divergence traps documented: quant.h vs test reference differ on fp16
subnormals ((127-15+1) vs (127-15) exponent) and rotation block cap (32768
vs 4096) — invisible at test dims; backend follows quant.h (the oracle).

Status: all conformance formats implemented. Remaining: C ABI export
(`coli_cuda_*`), exact tensor_bytes accounting, expert_group_issue/take
async pair, then upstream test_backend_cuda.cu linked against the Zig lib.

## Milestone: C ABI + CONFORMANCE PASS (2026-07-29, end of day 1)

`coli/abi.zig` exports the full `coli_cuda_*` surface. Upstream's
conformance test — copied verbatim to `coli/tests/test_backend_cuda.cc`
except one line (main → coli_test_main) — links against it and passes:

    cuda backend: q8/q4/q2/f32/e8 correctness ok on 1 device(s)

Run: `bazelisk run --@zml//platforms:rocm=true //coli:conformance`.

ABI implementation notes:
- Owned io: `std.Io.Threaded = .init_single_threaded` global; no main needed.
- Errors never cross: every export returns 1/0; opaque `ColiTensor` handles
  wrap DeviceTensor + byte accounting + device ordinal.
- Byte accounting mirrors upstream exactly (quantized weight bytes + scale
  array bytes; the test asserts count==7, bytes==166 literally).
- Upload-into-populated-handle fails; bad fmt/device/null args fail; the
  16 TiB allocation fails gracefully through PJRT with accounting intact.
- `COLI_GPU_FAIL_AFTER=N`: fail GPU ops once N successful ops have run.
- `expert_group_issue/take`: synchronous under the hood into a persistent
  buffer — same code path as sync, hence memcmp-identical (test requirement);
  true async is a later optimization.
- `attention_absorb`: host math for now (lazy device→host weight copy);
  batch attention + `pipe_*` + w4a16 are honest 0-returning stubs → engine
  CPU fallback.
- Build shape: rules_zig zig_library has no CcInfo, so linkage is inverted —
  the C++ test is a cc_library, the final target a zig_binary whose Zig main
  calls `coli_test_main`, with `comptime { _ = @import("abi.zig"); }` forcing
  export emission.
- Zig 0.16: std.posix.getenv is gone; use std.c.getenv (libc is linked).

Next (engine integration): a `ZML=1` make variant linking a .so/.a built
from the abi module into `coli`, then an end-to-end run with COLI_CUDA=1 on
a small MoE (OLMoE path) before attempting GLM-5.2.

## Milestone: ENGINE INTEGRATION — END-TO-END ON GPU (2026-07-30)

The full colibri engine runs with the Zig/ZML backend. Bazel target
`//coli:coli` = engine/colibri.c (copy of upstream c/, main renamed) +
abi.zig, one binary. Build tweaks needed for the hermetic toolchain:
uring.h `__has_include` guard + stub fallback (engine falls back to pread),
`-Domp_in_parallel()=0` (only omp call colibri.c doesn't self-stub), engine
headers PRIVATE in cc_library (rules_zig translate-c chokes on them),
OpenMP off (single-threaded CPU side, idot scalar — CPU speed not
comparable to the make build).

Validation ladder, all passing:
1. `make_glm_oracle.py` tiny model (bf16): TF self-test vs HF oracle —
   CPU engine 32/32, ZML engine 32/32 (GPU tier empty: bf16 experts not
   offload-eligible; validated integration/lifecycle only).
2. `make_glm_bench_model.py --fp8` → int4 fixture (167 MB, vocab 8192,
   8 layers, 32 experts, ref_glm.json included): REPLAY decode works on
   both engines.
3. GPU expert tier (harness two-step: run with STATS=..., rerun with
   PIN=stats PIN_GB=1 COLI_CUDA=1 CUDA_EXPERT_GB=2): **378 tensors
   (126 experts) resident in VRAM, 45 group calls, 760 rows, routed CPU
   0.000s / routed GPU 0.292s — 100% of routed expert compute on the GPU
   through the Zig backend.** 8 tokens, p50 31.3 ms (vs 54.7 ms same-build
   CPU path; the properly built OpenMP+VNNI CPU engine does 5.1 ms on this
   cache-warm toy — GPU pays off at real expert sizes, not 0.9 MB ones).

Notes for the perf leg: avg 9.6 experts/call exceeds our fused-group cap
(N ≤ 8) → many groups take the loop fallback; raise Group(N) dispatch to 16.
issue/take is synchronous → the engine's async overlap serializes. max
forward 257 ms = mid-run XLA compiles for new shape keys (warms out).

## Milestone: perf leg 1 (2026-07-30) — 3.6× per-token

Three changes: Group(N) fused dispatch cap 8 → 16; `-march=native` on the
engine cc_library (restores `idot: avx512-vnni` — the VNNI path is
compile-time gated); **true async issue/take** (backend
`expertGroupIssue`/`pendingTake` split — `Exe.call` launches without
waiting, the sync lives in the output read; the ABI keeps one
`PendingGroup` in flight, drains on re-issue/sync-call/shutdown, falls back
to the sync loop for non-fusable groups).

Same fixture, same STATS→PIN setup: p50 **31.3 ms → 8.6 ms**/token
(16.7 → 27.0 tok/s), orchestration 108 → 19 ms, still 0 routed CPU rows,
100% VRAM hit. Conformance PASSES (incl. the memcmp async==sync case) and
the full smoke suite passes unchanged. Reference: the make-built CPU engine
(16 threads OpenMP + VNNI) does p50 5.1 ms on this cache-warm toy fixture —
the ZML build's CPU side is 1 thread; at real expert sizes (19 MB vs 0.9 MB)
the GPU tier's economics invert.

Remaining perf ideas: OpenMP (or thread pool) for the bazel build's CPU
side; real-size fixture (few layers at D=6144/I=2048); pipe_* resident
tier; batch attention on GPU.

## Milestone: real-size fixture — GPU tier WINS (2026-07-30)

`tools/make_glm_real_fixture.py` (local copy of make_glm_bench_model with
real expert geometry: hidden 6144, moe_intermediate 2048 → 18.9 MB int4
experts; 3 layers / 2 MoE, 16 experts, ~1.7B params, 800 MB int4 in
models/glm_real_i4, own ref_glm.json).

Debug chain worth remembering — two hypotheses REFUTED by measurement
before the real one held:
1. fmt=4 grouped dequant breaking XLA fusion? NO — group-of-8 bench at
   gs=128: 1.62 ms ≈ fmt=2's 1.61 ms (now a permanent smoke tripwire).
2. GPU downclocking on sparse dispatch? NO — 15 ms idle gaps cost only ~9%
   (also a smoke probe now).
3. Single-threaded CPU half (shared expert ~37 MB int4 matvec per MoE layer
   + attention + lm_head)? YES — OpenMP via system omp.h/libomp (vendored
   header, `-Wl,--allow-shlib-undefined` for glibc symbol versions; Bazel
   rejects absolute -I paths) recovered it exactly.

**Result, real-dims fixture, 8-token replay, STATS→PIN, CUDA_EXPERT_GB=4:**

| engine | decode p50 |
|---|---|
| make-built CPU (16 threads, VNNI, OpenMP) | 13.7 ms |
| ZML build, GPU experts + 16-thread CPU side | **9.2 ms (1.5× faster)** |

routed CPU 0.000 s, 100% VRAM hit, expert-matmul 90% of forward (GPU +
shared expert), oracle self-test still 32/32 on the OpenMP build.
Throughput line (20 tok/s) is depressed by one mid-run ~330 ms XLA compile
for a fresh group count — steady-state ≈ 108 tok/s; pre-warming group exes
for counts 1..16 at startup would remove it.

## Dashboard on the ZML backend (2026-07-30)

The web dashboard runs against OUR engine. Plain browser app — no Nix
(flake.nix is an optional dev shell); `web/` is Vite/React, built once with
`cd web && npm install && npm run build` → `web/dist`, then served by
`openai_server.py`, which spawns any binary named `colibri` next to it and
talks a \x01-framed line protocol over stdin/stdout (SERVE=1).

Setup lives in `serve/`: `openai_server.py` (copy), `web` → symlink to
upstream's, and `colibri` = a **shell wrapper** that execs
`zml/bazel-bin/coli/coli`. The wrapper is required: a symlink breaks Bazel
runfiles discovery ("Unable to initialize runfiles" — libpjrt_rocm.so).

Start (health/profile/dashboard all live at 127.0.0.1:8000):

    cd serve && SNAP=<model> CTX=1024 COLI_CUDA=1 COLI_GPU=0 \
      CUDA_EXPERT_GB=4 PIN=<model>/bench_stats.txt OMP_NUM_THREADS=16 \
      COLI_NO_OMP_TUNE=1 python3 openai_server.py --model <model> --cap 4

Confirmed: `/health` reports `tiers {vram: 31, ram: 1, disk: 0, vram_gb:
0.62}` and `gpu: "CUDA device x1"`; `/` serves the built UI; a real
`/v1/chat/completions` turn round-trips (43 prompt + 12 completion tokens,
expert_matmul 0.578 s of 0.662 s wall on the GPU); `/profile` returns the
per-turn phase breakdown the Profiling tab charts.

Two gotchas: (1) the engine's RAM guard refuses to start when PIN_GB + a
4096 context project a peak over available RAM — drop PIN_GB, lower CTX, and
free the Bazel JVM (`bazelisk shutdown`, 1.7 GB) on this 16 GB box; (2) the
synthetic fixtures ship no tokenizer.json, so
`scratchpad/mktok.py` (kept in the session scratchpad) writes a byte-level
BPE one sized to the fixture's vocab — chat output is filler-token gibberish
(random weights), which is expected and irrelevant to the metrics.

## Upstream sync (2026-07-30)

Upstream is at v1.2.0 (main); our pin matches. Related ROCm work landed on
**origin/dev**, not main: PR #681 / commit 0ff5bb8
"fix(hip): support GPUs without matrix cores (gfx9xx, gfx101x/103x)" —
adds `COLI_HIP_NO_WMMA` + a Makefile `NO_WMMA_ARCHS` list so
`backend_gpu_compat.h` skips rocWMMA on archs whose headers static_assert,
keeping the `#error` for WMMA-capable archs missing the headers. It is a
**superset of our local backend_gpu_compat.h patch** (ours just forced
COLI_GPU_HAS_WMMA=0), and its rationale matches what we hit: the host must
not dispatch kernels whose bodies were compiled out. Our patch can be
dropped in favor of upstream's once dev merges; gfx1200 has WMMA, so we
keep using rocwmma-devel. Nothing in dev affects the Zig backend's ABI.

## Open questions / risks

- PJRT executable dispatch overhead vs. colibri's many tiny decode-time calls
  (S=1 GEMVs). Mitigation: coarser fusion (Tier-1 group call = one executable),
  async issue/take maps naturally onto PJRT's async execute.
- `expert_group_take` returns a pinned host pointer valid until next issue —
  PJRT D2H into a persistent host buffer works.
- Numerics: GPU float matmul vs CPU int8-dot differs upstream already
  (documented non-token-identical greedy output); conformance test tolerances
  are 1e-4 abs / relative-RMS, achievable.
- Zig version skew: ZML pins its own Zig toolchain via Bazel (0.16 nightly
  series) — write backend code against that, not a system Zig.
