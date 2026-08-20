# Lab notebook: video generation on the compiler stack

The working log of this experiment, kept as written — including the wrong
turns, because refuted hypotheses are the most useful part. The convention,
carried over from coli-zml and sharpened: **log the why before the what.**
Every experiment gets its question, a hypothesis with the reasoning behind
it, and only then the measurement, so the notebook doubles as a record of
why the system behaves the way it does rather than a pile of numbers.
Formatting is deliberately prose-first with almost no tables — the author
reads with a screen reader, and prose survives that (and `git diff`) far
better than ASCII grids.

Goal: adapt the coli-zml approach — no hand-written GPU kernels; Zig tracing
StableHLO through ZML, XLA generating the machine code, PJRT reaching the
GPU — from sparse MoE LLM decode to dense video diffusion. Target model:
LTX-2.5. Hardware: AMD RX 9060 XT (Navi 44, gfx1200, RDNA 4), 16 GB VRAM,
Fedora 44, Ryzen 7 8700F.

## 2026-08-20 — genesis: what carries over, and what inverts

### Provenance

This repo begins as a pivot from
[coli-zml](https://github.com/ABSatVPS/coli-zml), which replaced colibrì's
hand-written HIP kernels with compiler-generated code and landed within 7%
of them. A snapshot lives in `coli-zml/` at upstream commit `dbf70c6`
(2026-07-30), git history detached — a frozen reference, not a dependency.

The colibrì engine itself is dropped entirely, and the reason is worth
recording: colibrì was the host engine that owned the pipeline and called
the backend through a C ABI, which forced a host→device→host round trip on
every call and made "keep activations resident" an ABI-design problem (the
`pipe_*` opaque-pointer tier, never implemented). Here there is no host
engine — this project owns the whole pipeline, so device residency between
stages is the default state of the world, not a feature. Dropping colibrì
deletes a class of problems instead of solving them.

### The target

LTX-2.5 (Lightricks, open weights released 2026-08-11): a 22 B-parameter
asymmetric dual-stream diffusion transformer that generates video and audio
jointly via bidirectional cross-attention between the streams; the text
encoder is Gemma 4 12B with a learned projection. Ships as Base, Distilled,
and pre-quantized variants.

Why this model rather than Wan or Hunyuan: the Distilled variant means few
denoising steps, so iteration on a consumer card stays cheap; the LTX
lineage's aggressively-compressing VAE keeps latent token counts well below
Wan's for comparable output; and the release is three weeks old, so the
consumer-AMD story around it is unclaimed.

**To verify from the HF repo before E2 is finalized:** the actual VAE
compression factors in 2.5, the video/audio parameter split of the 22 B,
head count and head dimension, and typical latent token counts per
resolution/duration. The working estimate used below — roughly 28 k video
tokens for ~10 s at ~720p-class output, extrapolated from the LTX-1 32×32×8
VAE — is an estimate, and is flagged wherever it is load-bearing.

### The central fact: the regime flips

Everything in coli-zml was shaped by one property: MoE **decode** is
weight-bandwidth-bound at tiny row counts — S of 1 to 8 rows per expert,
the GPU starving for weight bytes. A video DiT step is the opposite regime,
effectively permanent prefill: every call carries tens of thousands of rows
and the workload is compute-bound. Four consequences, each of which inverts
a coli-zml conclusion:

1. **`qdot` does not transfer.** The broadcast-multiply-plus-reduce
   contraction reads the full weight matrix once per row of x — free at
   S=1, catastrophic at S=28 k. coli-zml's own source anticipated this: the
   comment on `qdot` (coli-zml/coli/backend.zig, lines 231–236) reads "the
   right trade at decode-sized S; revisit (dot + materialize, amortized)
   for prefill S." This project is that revisit.
2. **The dequant-materialization penalty inverts.** At S=1, XLA
   materializing f32 weights before a `dot_general` destroyed effective
   bandwidth — measured slower than not quantizing at all. At S=28 k, each
   materialized weight byte is amortized over roughly 2·S FLOPs of GEMM
   work, so the time cost rounds to zero. What remains is a transient VRAM
   spike (an f32 copy of one layer's weights exists during the call), which
   matters on 16 GB and needs measuring rather than guessing.
3. **Dispatch overhead becomes noise.** coli-zml's row=1 result (1.8×
   slower than hand-written HIP) was ~0.2 ms of PJRT orchestration under
   ~0.3 ms of actual work. A DiT block launch carries seconds of work
   behind the same fixed overhead. The "close the last 7%" program from
   the coli-zml README is therefore *not* a prerequisite for this project —
   that 7% lives in exactly the terms this workload amortizes away.
4. **Matrix cores become the first-order question.** In a bandwidth-bound
   GEMV, WMMA is irrelevant — both backends saturated at ~90–97 GB/s
   without it, which is why coli-zml never needed to check. In a
   compute-bound GEMM at S=28 k, whether XLA's RDNA 4 backend emits V_WMMA
   decides the throughput ceiling, several-fold. This is untested territory
   for this stack on this card.

### What transfers unchanged

1. **The sub-byte PJRT patch** (`patches/zml-pjrt-subbyte-transfer.patch`).
   22 B parameters is ~44 GB at f16. Quantized residency is not an
   optimization here, it is feasibility: ~11 GB at int4 is the difference
   between the DiT fitting on the card and not fitting. The patch that let
   colibrì's u4/u2 weights upload packed is the same patch that lets a
   quantized DiT live in VRAM.
2. **Executable caching per shape.** Diffusion replays one static shape for
   an entire run — 50 steps (or 4–8 distilled) times every block of
   identical geometry. Compile once, launch hundreds of times. The
   amortization story is better than decode's, where S at least varied.
3. **Streaming as block swap — and the sparse-to-dense inversion helps.**
   The expert-streaming architecture maps onto dense block streaming with
   one crucial simplification: MoE routing is data-dependent (which expert
   is needed is unknown until the router fires, so prefetch is speculative),
   while a dense DiT's block order is a compile-time constant. Prefetch is
   deterministic and can be perfect. The `issue`/`take` async split is the
   primitive; the router is gone.
4. **Oracle-first methodology.** colibrì's tensor-core benchmark once
   reported 1.3 TB/s on a 320 GB/s card because its validation compared a
   stale buffer against itself. That patch sits in the snapshot as a
   permanent reminder: no number enters this notebook without an
   independently computed reference, and every load-bearing compiler
   behavior gets re-benchmarked on every smoke run so a regression surfaces
   as a number, not a mystery.

### Planned experiments — hypotheses before measurements

**E1 — contraction crossover and ISA audit.** Question: at what S does
`dot_general` (dequant materialized, cost amortized) overtake `qdot` (fused
packed-nibble reads), and does the large-S kernel actually use V_WMMA on
gfx1200? Hypothesis: the crossover sits in the tens of rows; at S≈28 k,
dot wins by a large factor if and only if WMMA engages. If the ISA dump
shows only vector-ALU FMAs, the compute ceiling drops several-fold and the
plan needs a rethink at the top. Method: the same int4 expert-geometry
matrices as coli-zml's smoke (known-good upload path), S swept from 1 to
32 k, `XLA_FLAGS=--xla_dump_to=` plus reading the emitted .s for `v_wmma`.

**E2 — what XLA's attention does at video lengths on ROCm.** Question:
does the SDPA lowering produce something tiled with an online softmax, or
does it materialize the scores matrix? Stakes: at T=28 k, one head's f16
scores are ~1.6 GB; all heads batched is far beyond 16 GB — naive attention
is not slow here, it is impossible. Hypothesis: genuinely unknown. This is
this project's `dot_general` moment — the make-or-break compiler behavior
that documentation cannot answer, only a benchmark. Method: trace naive
attention at T in {1 k, 4 k, 8 k, 16 k, 28 k}, watch VRAM and time, read
the post-fusion HLO.

**E3 — blockwise attention written into the graph.** If E2 fails at the top
sizes (expected), the fallback that preserves the no-hand-written-kernels
thesis: express online-softmax blockwise attention (Rabe–Staats style — a
running max, running sum, and accumulator carried through a loop over K/V
chunks) explicitly in the traced graph. Memory is bounded by construction;
the only question is whether XLA's codegen of the loop body lands within a
small factor of the CK flash-attention kernels. Hypothesis, calibrated on
coli-zml's 7% result: within 2× is achievable, and within-2× at guaranteed
O(T) memory beats fast-but-does-not-fit. Escape hatch if it is much worse:
a PJRT custom call into the ROCm flash-attention kernels (the CK backend
covers RDNA 4) — impure, but pragmatic.

**E4 — 3D conv tiles for the VAE.** Question: what does XLA's convolution
path on ROCm deliver on VAE-decode-shaped 3D convolutions, and what tile
geometry keeps decode inside the VRAM budget alongside resident DiT
weights? Static tile shapes mean each geometry compiles once, and latents
never leave the device between DiT and VAE — the structural advantage over
orchestrators that juggle separate models through host memory. Seam
overlap and blending for causal convs is known-fiddly and deferred until
raw tile throughput is measured.

Non-goals for now: the audio stream (bring up video-only first; need to
verify from the repo how separable the two streams are at inference), the
Gemma 4 12B text encoder (runs once per prompt and produces embeddings —
treat as a separate offline phase initially), and anything end-to-end until
E1 through E4 have numbers.

### Milestones

1. ✅ Workspace bootstrap: `scripts/setup.sh` (pinned ZML plus the sub-byte
   patch, `ltx/` package installed), `//ltx:smoke` builds and enumerates
   the gfx1200 device. (Done same day — see the run 1 entry below. First
   build: 640 s, 5417 actions, hermetic ROCm userland.)
2. E1 and E2 measured, ISA dump read.
3. E3 if E2 demands it (expected).
4. E4.
5. One LTX-2.5 transformer block, numerically conformant against the
   HuggingFace implementation as oracle — coli-zml's 32/32 teacher-forcing
   discipline with a new oracle.
6. Full DiT: quantized resident weights, static-shape replay across steps.
7. Block streaming (for f8 or larger variants), tiled VAE, then audio.

## 2026-08-20, later — run 1: E1 and E1a have numbers, E2 found a tracer bug

The very first `//ltx:smoke` run on the RX 9060 XT (640 s cold build, then
`platform: rocm`, both E1 formulations matching the CPU oracle at S=1 and
S=8, and the two GPU formulations cross-checking against each other at
every S). Results, then what they mean.

### E1 — the crossover is where the hypothesis put it

qdot is the coli-zml broadcast-mul+reduce; dot is dequant-materialize plus
a real `dot_general`. Both in f32, per-row-scaled int4, O=2048, K=6144,
p50 with x resident and the y download included (identical overhead on
both sides, so the comparison is clean even though the absolute numbers
include it).

At S=1, qdot 0.405 ms against dot 1.329 ms — qdot 3.3× faster, which is
the coli-zml result reproduced from scratch on a fresh workspace. At S=8
qdot still leads (0.641 ms vs 1.351 ms). By S=64 the tables have turned:
dot 1.744 ms vs qdot 2.775 ms. From there dot settles at 2.3–2.5× faster:
at S=512, 8.5 ms vs 20.0 ms; at S=4096, 70.6 ms vs 159.3 ms; at S=28672,
500.7 ms vs 1230.1 ms. **The crossover sits between 8 and 64 rows** —
"tens of rows," as hypothesized. Both regimes are real, and each
formulation is correct on its own side of the line.

Why each side plateaus where it does: qdot flattens at ~590–650 GFLOP/s,
and the reason is visible in its memory arithmetic — at S=4096 it re-reads
the 6.29 MB packed weight matrix 4096 times, ~25.8 GB in 159 ms, which is
~162 GB/s of sustained VRAM traffic on a 320 GB/s card. It is
bandwidth-bound by construction, exactly as designed for decode, and
that design is precisely what makes it the wrong algorithm at prefill.
The f32 dot path flattens at ~1.45 TFLOP/s, which is also nowhere near
the card: RDNA 4's matrix cores have no f32 input mode, so an f32 GEMM
is stuck on the vector ALUs. Conclusion: **neither E1 formulation is the
engine path.** The engine wants a third thing, measured as E1b below.

### E1a — XLA engages the matrix cores on gfx1200

Dense f16 GEMM, scalar-sum epilogue so the download is 4 bytes and the
timing is compute. S=512: 22.1 TFLOP/s. S=4096: 33.0. **S=28672: 58.4
TFLOP/s.** The card's packed-vector f16 ceiling is ~51 TFLOP/s at boost
clocks, so 58 exceeds what the vector ALUs can do at all — the matrix
path is engaged. (Run 2 reads the ISA dump for `v_wmma` to close this
definitively rather than by arithmetic; recorded there.) Two further
observations worth keeping: utilization climbs steeply with S — tall
GEMMs are where this card shines, and a DiT step at S≈28 k lives exactly
there; and O=2048 is a smallish N, so mid-S numbers may improve with the
wider projections of a real transformer block.

This was the make-or-break unknown for the whole project (notebook entry
above, consequence 4), and it broke the right way.

### E1b — hypothesis, written before its measurement

The engine candidate is neither of E1's formulations: keep weights int4
resident (the VRAM feasibility requirement), trace dequant to **f16**,
then a real dot that can hit WMMA. Hypothesis: within ~15% of E1a's
dense-f16 number, because the materialized f16 weight copy is ~25 MB of
write+read per call (~0.3 ms at observed bandwidth) against a 12.4 ms
GEMM at S=28672. The risk that would falsify it: the dequant convert
chain blocks XLA's GEMM rewriter and the whole thing falls back to a
fused-but-vector-ALU loop — which would be the mirror image of coli-zml's
original dot_general surprise, and would force a two-executable split
(dequant pass, then GEMM) instead of one graph. Run 2 answers this.

### E2 — run 1's failure was mine, not XLA's

E2 panicked before measuring anything: "Tensor with id 364 has already
been used once as an argument," from `compileFn`. Cause: I passed one
spec tensor for both k and v. ZML's tracer requires a distinct tracer
tensor per function argument — a fact now recorded in a comment at the
call site, and the kind of lesson the notebook exists to keep. No
conclusion about XLA attention behavior can be drawn from run 1; E2 runs
properly in run 2.

## 2026-08-20, later still — run 2: E1b refuted, and the forensics are better than the hypothesis

Run 2 (warm build: 11.6 s) reproduced E1 and E1a within noise — crossover
still between 8 and 64 rows, dot still 2.2–2.5× over qdot at large S, E1a
peaking at 60.8 TFLOP/s this time — and delivered two new results.

### E1b measured: 3.0 TFLOP/s, flat across S. Hypothesis refuted — but not
### the way the hypothesis feared

The int4-resident dequant→f16→dot path ran at 3.07 / 3.01 / 3.04 TFLOP/s
at S=512/4096/28672 — twenty times below E1a's dense-f16 number, nowhere
near the hypothesized "within 15%." The correctness cross-check against
the dense f16 GEMM passed, so the math is right and only the speed is
wrong. Then the ISA dump made the story much more specific, in three
steps:

First, the named risk did NOT materialize. The after-optimization HLO for
the E1b module shows XLA already does exactly the two-stage split the
hypothesis said we might have to build by hand: a `loop_convert_fusion`
(u4 plus scales → f16[2048,6144]) followed by a `__cublas$lt$matmul`
custom call — which on ROCm is hipBLASLt, the vendor BLAS library. The
GEMM rewriter was not blocked. (This also answers how E1a gets its 60
TFLOP/s: XLA routes big GEMMs to hipBLASLt, whose kernels use the matrix
cores. The compiler stack's fast path on this card is "hand the GEMM to
the library," which is still zero hand-written kernels in this repo.)

Second, the slow part is the GEMM itself, not the dequant. The E1b-minus-
E1a time delta scales linearly with S (~7.5 µs per row at every size),
and the dequant fusion is a fixed-size job (~25 MB written regardless of
S) — a fixed cost cannot produce a linear-in-S delta.

Third, the smoking gun: the E1a and E1b GEMM custom calls are
byte-for-byte identical in every dimension that should matter — same
shapes, same {1,0} layouts, same strides, same contracting dims, same
workspace — and differ in exactly one field: `selected_algorithm` is
**73** in E1a and **91** in E1b. The autotuner, keyed on something that
includes the operand producer (parameter in E1a, fusion output in E1b),
ran twice for the same problem and picked a hipBLASLt algorithm ~20×
slower the second time.

**Prediction for run 3, written before its result:** with
`--xla_gpu_autotune_level=0` (autotuning off, library default algorithm
for everyone), E1a and E1b should converge — if the theory is right, E1b
jumps to roughly E1a's speed, or both land at whatever the default
algorithm gives. If E1b stays 20× slower with autotuning off, the theory
is wrong and something structural about consuming a fusion output is the
real cause. Either way this is the project's first "brittleness of the
modern compiled stack" finding, the direct descendant of coli-zml's
dot_general materialization: **on this stack, WHERE a GEMM operand comes
from can silently cost 20×.** The smoke must keep both E1a and E1b
forever as the tripwire pair.

### E2 at T=512: the oracle tripped on tolerance, and the HLO already
### answers the structural question

The sdpa output measured RMS 0.0082 against the f64 CPU oracle, over my
5e-3 limit — run aborted before any timing. Analysis says the tolerance
was wrong, not the kernel: an end-to-end f16 attention (f16 exp and sum
in the softmax) accumulates exactly this order of error at T=512, and a
wrong algorithm would be off by O(1), not 0.8%. Tolerance raised to 2e-2
with a comment. The leaked-allocations spew after the error return is
the smoke skipping platform teardown on early exit — cosmetic, ignored.

More usefully, the T=512 module's HLO settles E2's structural question
without waiting for the timing: `zml.nn.sdpa` lowers to **two hipBLASLt
matmuls with the f16[16,512,512] scores tensor materialized between
them** plus a softmax fusion. No flash-style rewrite exists on this
path. Scores at T=28672 are ~26 GB — the prediction stands that E2's top
size OOMs and E3 (blockwise attention in the traced graph) is mandatory.
Run 3 measures how far the naive form gets and what it costs.
