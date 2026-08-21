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

## Phase 1 — E4: VAE conv tiles — NEXT

Question: what does XLA's convolution path on ROCm deliver on
VAE-decode-shaped 3D convolutions, and what tile geometry keeps decode
inside the VRAM budget next to resident DiT weights? LTX-2.5 ships two
video VAEs (DiffVAE, heavier; Conv VAE, lighter) — probe the Conv VAE
shapes first.

Done means: measured GB/s and ms per tile at two or three candidate
tile geometries, a chosen geometry with the arithmetic for why, and a
seam-overlap plan for the causal convs written down before any decoder
is built.

## Phase 2 — One conformant transformer block

Build one video-stream block exactly to the checkpoint spec: gated
self-attention (per-head sigmoid gates) with 3D RoPE (f64-precomputed
tables baked at trace time), gated cross-attention, bias-free 4× GELU
FFN, RMS norms, 9-entry AdaLN modulation. Weights come from the real
checkpoint by range-requesting only block 0's tensors (~600 MB) using
the offsets already captured in the header — no 42 GB download.

Done means: the block's output matches the reference implementation
(ltx-core / ComfyUI, CPU) at teacher-forced inputs within an f16-honest
tolerance, with the comparison methodology written down first — the
coli-zml 32/32 discipline with a new oracle.

## Phase 3 — Full video-stream DiT

All 48 blocks, int4-resident (~11 GB) via the sub-byte PJRT path,
static-shape executables replayed across the distilled model's few
steps. Text conditioning arrives as precomputed embeddings (Phase 4
runs offline first). Audio stream stubbed out — the checkpoint's
bridges are skipped, accepting divergence from joint generation until
Phase 7.

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
