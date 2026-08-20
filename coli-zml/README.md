# coli-zml — a compiler-generated GPU backend for colibrì, on AMD

[colibrì](https://github.com/JustVugg/colibri) is a dependency-free C engine
that runs a 744B-parameter MoE model on a consumer machine by streaming experts
from disk. Its optional GPU backend is ~1,800 lines of hand-written CUDA
kernels, compiled for AMD through a macro shim.

This is a replacement for that backend written in [Zig](https://ziglang.org)
on top of [ZML](https://github.com/zml/zml): instead of kernels, it emits
StableHLO and lets XLA generate the machine code, reaching the GPU through
PJRT. No CUDA, no HIP, no hand-written kernel in the stack.

It passes colibrì's own GPU conformance suite, and the full engine decodes
tokens with every routed expert served from VRAM.

## Results

Measured on a **Radeon RX 9060 XT** (Navi 44, gfx1200, RDNA 4), ROCm 7.1,
Fedora 44. Comparisons are against colibrì's hand-written kernels compiled by
hipcc for the same GPU, same weights, same shapes, in the same session.

**One expert MLP** (gate/up/SiLU/down, int4, D=6144 I=2048 — GLM-5.2's expert
geometry), host→device→host round trip:

| rows | hand-written HIP | this backend | |
|---:|---:|---:|:--|
| 1 | 0.289 ms | 0.525 ms | 1.8× slower |
| 2 | 0.469 ms | 0.623 ms | 1.3× slower |
| 4 | 0.715 ms | 0.831 ms | 1.16× slower |
| 8 | 1.324 ms | **1.218 ms** | 1.09× faster |

**The decode unit** — 8 distinct int4 experts, one token row each, which is
what an MoE router actually asks for per layer:

| | time |
|:--|---:|
| hand-written HIP grouped kernels | 1.557 ms |
| this backend, one fused executable | **1.60–1.67 ms** (p50 across runs) |
| this backend, unfused loop of 8 | 5.06–5.43 ms |

Both stacks land at ~90–97 GB/s of effective weight-read bandwidth. The
compiler-generated kernel is within 7% of the hand-written one.

**End to end**, colibrì decoding on a real-dimension MoE fixture (3 layers,
16 experts/layer at full GLM-5.2 expert geometry, int4), 8-token replay:

| engine | decode p50 |
|:--|---:|
| colibrì CPU (16 threads, AVX-512 VNNI) | 13.7 ms |
| colibrì + this backend (experts in VRAM) | **9.2 ms** |

100% VRAM hit, zero routed expert rows on the CPU, and colibrì's
teacher-forcing self-test still scores 32/32 against its HuggingFace oracle.

## What's here

| | |
|:--|:--|
| `coli/backend.zig` | the backend: quantized weights resident on device, dequant traced into the graph, fused expert MLP and expert-group executables cached per shape |
| `coli/abi.zig` | colibrì's `coli_cuda_*` C ABI (~43 functions) exported from Zig — status codes, opaque handles, exact byte accounting |
| `coli/smoke.zig` | correctness against colibrì's test vectors plus the benchmarks above |
| `coli/conformance_main.zig` | runs colibrì's **unmodified** GPU test suite against this backend |
| `coli/engine_main.zig` | the colibrì engine and this backend as one binary |
| `patches/` | three small upstream patches (see below) |
| `docs/lab-notebook.md` | the full experiment log, including the wrong turns |

Nothing from either upstream is vendored. `scripts/setup.sh` clones both at
pinned revisions, applies the patches, and installs `coli/` as a Bazel package
inside the ZML workspace.

## Try it

Needs Linux, an AMD GPU with a supported `gfx` target, Python 3, and ~50 GB of
disk for Bazel's cache (ZML fetches a hermetic ROCm userland — your system ROCm
install is not used; only the `amdgpu` kernel driver matters).

```sh
./scripts/setup.sh                 # clone + patch upstreams, ~10 min first build
cd .work/zml
../bin/bazelisk run --@zml//platforms:rocm=true //coli:conformance   # colibrì's GPU suite
../bin/bazelisk run --@zml//platforms:rocm=true //coli:smoke         # vectors + benchmarks
```

`docs/reproduce.md` covers generating the model fixtures, the end-to-end engine
run, and serving colibrì's web dashboard on this backend.

## How it works, briefly

colibrì hands the backend quantized weight blocks and expects `y = xWᵀ` — or a
whole expert MLP, or a group of experts — back in host memory. Three decisions
carry most of the performance:

**Quantized weights stay quantized on the device.** Weights upload as `u4`/`u2`/
`i8` buffers with their scale arrays; the dequant (offset, per-row or per-group
scale) is part of the compiled graph. Dequantizing on the host instead inflates
int4 to f32, and since these ops are weight-bandwidth-bound it costs exactly the
8× you'd expect.

**The contraction is a broadcast-multiply and a reduction, not a dot.** XLA on
ROCm does *not* fuse a dequant chain into a `dot_general` operand — it
materializes the f32 weights in VRAM on every call, which measured *slower*
than not quantizing at all. Expressed as multiply-then-`sum`, the same
arithmetic loop-fuses, and the emitted kernel reads the packed nibbles
directly. This is the single most important line in the repo, and it is a
property of today's compiler, not a guarantee: `coli/smoke.zig` benchmarks it
on every run so a regression shows up as a number, not a mystery.

**The whole expert group is one executable.** A comptime-generated model struct
holds *N* experts' weights, so a router's *k* selected experts become a single
launch with one upload and one synchronization instead of *k* round trips —
3.2× faster than the loop, and it's what closes the gap with the hand-written
grouped kernels. `issue`/`take` map onto it as a genuine async split.

## Upstream patches

Small, and each one is a bug or gap found by this work:

- **`zml-pjrt-subbyte-transfer.patch`** — ZML can't create sub-byte (`u4`/`u2`)
  device buffers: it always sends an explicit tiled layout, which PJRT rejects
  for 4-bit element types. Makes the layout optional and adds the sub-byte host
  path (unpacked bytes in, packed on device).
- **`colibri-bench-validation.patch`** — colibrì's tensor-core benchmark reports
  a speedup for a code path that is a documented no-op under HIP, because its
  RMS check compares a stale result buffer against itself. Adds validation
  against an independently-computed reference and stops benchmark modes from
  leaking environment variables into each other. (I hit this as a "1.3 TB/s"
  reading on a 320 GB/s card.)
- **`colibri-real-dims-fixture.patch`** — a variant of colibrì's fixture
  generator with production expert geometry, so backend numbers reflect 19 MB
  experts instead of 0.9 MB toys.

## Honest limits

- **One GPU, one architecture.** Everything here is measured on gfx1200. The
  code is not architecture-specific — that's rather the point of compiling
  through XLA — but "portable" is a claim I haven't earned on other hardware.
- **The backend is partial.** Expert offload, matmul, the fused MLP/group path,
  all five quantization formats including E8, and decode-time MLA attention are
  implemented. Batch attention, the resident device-pointer pipeline
  (`pipe_*`), and the W4A16 tensor-core path return 0, which is colibrì's
  documented "unsupported" contract — the engine falls back to CPU for those.
- **Test models are synthetic.** Random weights at real dimensions, generated
  locally with colibrì's own tooling. Enough to validate numerics against an
  oracle and measure bandwidth; not a substitute for running the real 744B
  checkpoint, which I don't have the disk for.
- **The engine build is deliberately non-hermetic** in two places (system
  `omp.h`, system `libomp`) so the CPU half is multi-threaded.
- Expert dequant runs at upload for E8 only (host-side), and its rotation is
  folded into the down-projection weights rather than applied per call.

## Credit and license

Apache 2.0, matching both upstreams. colibrì is © the colibrì authors and ZML
is © the ZML authors; this repository contains neither project's source, only
patches against them and original code that links to both. The conformance
suite this backend is validated against is colibrì's, unmodified except for
renaming its `main`.
