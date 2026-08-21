# Roadmap

The forward-looking tracker. Evidence and the reasoning behind every
status claim live in [docs/lab-notebook.md](docs/lab-notebook.md) — this
file only says what is done, what is next, and what "done" means for
each phase. Kept in prose deliberately (see the notebook's conventions).

Target: LTX-2.5 (22 B dual-stream audio-video DiT) generating video on a
Radeon RX 9060 XT (gfx1200, 16 GB), through ZML/StableHLO/XLA with no
hand-written GPU kernels. Architecture facts below are pinned against
the shipped checkpoint's own metadata (model_version 2.5.0), not
inferred.

## Phase 0 — Feasibility smokes (E1–E3) — COMPLETE

Answered on the target card, 2026-08-20/21, all in `//ltx:smoke`:

- E1: the coli-zml mul+reduce contraction and a real dot_general cross
  over between 8 and 64 rows; prefill belongs to dot_general.
- E1a: f16 GEMMs reach ~59 TFLOP/s via XLA's hipBLASLt routing — matrix
  cores engaged; autotuning is worth 3× over the Triton fallback.
- E1b: int4-resident weights with in-graph dequant feed those GEMMs at
  full dense speed. Quantized residency is free at prefill sizes.
- E2: naive attention materializes f32 scores, dies at a quarter of
  video length on 16 GB, and is bandwidth-crippled where it fits.
- E3: blockwise online-softmax attention through `stablehlo.while` runs
  the full 28,672-token working length (1.22 s, 11 TFLOP/s), verified
  bit-for-bit against its unrolled twin and through it against an f64
  oracle.

Open residuals carried forward (tracked, not blocking): the ~1.34×
fusion-fed GEMM gap at S=28,672 (suspected autotuner algorithm choice);
f16 scores in the attention loop (~2× expected); a pinned-memory
readback path for final frames (naive readback measured at ~0.5 GB/s).

## Phase 1 — E4: VAE conv tiles — COMPLETE

Answered 2026-08-21 (notebook has the numbers): XLA routes 3D convs to
MIOpen (`__cudnn$convForward`), delivering 21–34 TFLOP/s f16 at the
decoder's checkpoint-exact shapes. Decode arithmetic: ~1.4 s per
8×16×16 latent tile, ~35–60 s per 10 s clip — comparable to one
denoising step's attention, so the VAE is a co-star, not the
bottleneck. Seam plan recorded: the decoder's receptive-field halo
(~15 latent voxels) makes exact tiling untenable; Phase 5 blends with
modest overlap. Cost of entry: a second one-line ZML patch exposing
the (generic but private) N-D convolution.

## Phase 2 — One conformant transformer block — COMPLETE

Closed 2026-08-21, same day it opened (notebook has the full arc).
The ZML-traced block matches the upstream ltx-core implementation on
real block-0 weights at **rel-RMS 1.85e-6** (f32 vs f64 oracle;
~3000× tighter than the reference's own bf16 floor of 5.9e-3), with
every intermediate stage gated and passing at ~1e-6 and the RoPE
tables **bit-perfect** (0/262,144 stragglers). One spec discovery en
route, caught by the bit-level gate: the reference's f64 RoPE path
rounds its frequency grid to f32 mid-pipeline. Tooling shipped:
revision-pinned range-request weight fetcher, torch-CPU oracle bundle
generator with staged dumps verified identical to the reference
forward, and `//ltx:block_conformance`.

## Phase 3 — Full video-stream DiT — IN PROGRESS

All 48 blocks at int8-g128 (~20 GB, streamed through the block-prefetch
pipeline — the int4-residency plan died on the model's outlier-heavy
weight statistics; ladder receipts in the notebook), static-shape
executables replayed across the distilled model's few steps. Text
conditioning arrives as precomputed embeddings (Phase 4 runs offline
first). Audio stream stubbed out — the checkpoint's bridges are
skipped, accepting divergence from joint generation until Phase 7.

Pre-assembly COMPLETE (2026-08-21): the 0/23/47 chain conforms with
sub-linear error growth (2.68e-6 at three blocks); the quantization
recipe is settled (int8-g128 large projections, fully quantized block
at the bf16 floor); the E3w while-loop attention — the only attention
that fits at T≈28k — is swapped into the block path and gated
(agreement with dense 1.09e-6, oracle conformance unchanged); the
distilled scheduler is reproduced BIT-EXACTLY by ltx/scheduler.zig
(94/94 gates, including the fused-lerp kernel semantics the
reference's own header doesn't show); and the streaming loader is
BUILT and gated — pack_block blobs, mmap → pinned ring → serial
uploads, lifecycle asserted, and the real chain and quantized-block
graphs fed through the ring bitwise-equal to direct loads.

ASSEMBLY CHECKPOINT PASSED (2026-08-21): all 48 blocks fetched,
quantized, packed, and digest-verified against the pinned revision;
the production execution model (ONE compiled block executable, 48
calls, ring-streamed weights) matches the torch f64 oracle at six
stage-walk checkpoints — 1.48e-5 at depth 48 vs a 2e-3 budget, error
plateauing rather than accumulating — and the fully-quantized walk
lands at 2.84e-2 end-to-end, DECLINING through the model's second
half: the int8-g128 recipe survives assembly.

STEP-BY-STEP CONFORMANCE CLOSED at harness T (2026-08-21 afternoon):
the E2E-core parts (adaln singles, patchify, the LayerNorm output
tail) each gated against their own f64 oracles (8/8); the COMPOSED
forward — patchify → adaln-driven 48 streamed blocks → tail, all on
device — matches a full-forward f64 oracle at 8.4e-5; and the
bit-exact scheduler driving the engine through stage 1's full
8-step distilled trajectory reproduces the reference latents at
every step (final drift 1.0e-5, 200x under budget, all of it
velocity-sourced — the scheduler seam is bitwise). Block geometry is
now trace-time-parametrized (22 gates bitwise-stable). Remaining:
the production-length clause — fits + runs + time-per-step at
T≈28k under the E3w kernel (//ltx:e2e_prod, pre-registered).

Done means: latents for a fixed seed and fixed precomputed conditioning
match the reference pipeline's within tolerance, step by step, and a
full denoising pass fits and runs on the card with time-per-step
recorded in the notebook.

## Phase 4 — Prompt path

Gemma 4 12B (the checkpoint names its source) plus the 8-block
128-register connector transformers per stream. Runs once per prompt;
evicted before denoising. Note from the checkpoint: this variant's
prompt K/V is timestep-dependent, so no across-step K/V caching here.

Done means: connector outputs match the reference for a test prompt,
and the load→embed→evict cycle leaves the DiT its full VRAM budget.

## Phase 5 — Tiled VAE decode: first pixels

The Phase 1 geometry becomes a real tiled decoder with seam blending;
latents stay on device from DiT to VAE; frames come back through a
readback path that is not the naive 0.5 GB/s one.

Done means: a short video-only clip, generated end to end on the card,
that a human can watch. Quality parity is Phase 6's problem; existence
is Phase 5's.

## Phase 6 — Performance and fidelity pass

The carried residuals, in expected-value order: f16 attention scores,
autotuner algorithm pinning for fusion-fed GEMMs, block streaming to
unlock bf16/dev variants beyond 16 GB, pinned-memory readback, and a
proper quality comparison against reference outputs.

Done means: notebook entries with before/after numbers for each item,
and an honest side-by-side against the reference pipeline's output.

## Phase 7 — Audio, and beyond

The audio stream, the bidirectional bridges, joint A/V generation with
modality-aware CFG. After that, the stretch list: the x2 spatial and
temporal upscalers, keyframe conditioning, LoRA loading.

Done means: joint audio-video out of the same graph, conformant against
the reference. This phase is deliberately unscoped until Phase 3 exists.
