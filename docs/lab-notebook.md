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
2. ✅ E1 and E2 measured, dumps read (runs 1–5, same day — including E1b's
   three-theory detective story; see the run entries).
3. ✅ E3 — E2 demanded it, as expected. Closed in run 8b: blockwise
   attention through stablehlo.while runs T=28672 on the 16 GB card,
   bit-identical to the unrolled form, no speed cost.
4. ✅ E4 — MIOpen conv lowering, 21–34 TFLOP/s at decoder shapes; VAE
   is not the bottleneck. (Phase 1 entry, 2026-08-21.)
5. ✅ One LTX-2.5 transformer block, numerically conformant against the
   upstream implementation as oracle — rel-RMS 1.85e-6, RoPE bit-perfect.
   (Phase 2 entries, 2026-08-21.)
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

### E1b measured: 3.0 TFLOP/s flat across S — hypothesis refuted, but not the way it feared

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

### E2 at T=512: the oracle tripped on tolerance, and the HLO already answers the structural question

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

## 2026-08-20, evening — run 3: E2 answered in full; the E1b prediction FAILS

Run 3 ran with `--xla_gpu_autotune_level=0` to test run 2's prediction,
and with the E2 oracle tolerance corrected to 2e-2.

### E2 complete: naive attention dies at T=16384, and weakly before that

With the tolerance fixed, T=512 matches the CPU oracle and the sweep
proceeds: 5.5 ms at T=512, 10.5 ms at T=1024, 37.8 ms at T=4096, 112.6 ms
at T=8192 — throughput climbing only to 4.9 TFLOP/s, roughly 12× below
what the card demonstrates on GEMMs, because the naive form is
bandwidth-bound shuttling the scores matrix through VRAM. At T=16384 it
died: "Out of memory while trying to allocate 16.00GiB." That number is
its own forensic: 16 heads × 16384² × 4 bytes is exactly 16 GiB — the
softmax path materializes the scores in **f32**, upcast from the f16
inputs, doubling the already-fatal footprint. (Caveat noted: run 3 had
autotuning off, which may depress the absolute ms figures somewhat; the
structural conclusion is unaffected.)

E2's verdict, cleanly: on the ROCm/XLA path there is no flash-style
attention rewrite; video-length attention through the naive form fails
on memory at a quarter of LTX's working sequence length, and is
bandwidth-crippled even when it fits. **E3 — blockwise online-softmax
attention written into the traced graph — is now the critical path for
the whole project**, exactly as the genesis entry hypothesized.

### The E1b prediction failed, and that's the most informative result yet

Prediction was: autotuning off, E1a and E1b converge. Measured: E1a fell
3× (60.8 → 20.5 TFLOP/s — so autotuning is worth 3× even on the clean
dense GEMM, its own finding), but E1b stayed at ~2.9 TFLOP/s — still 7×
behind E1a under supposedly identical default algorithm selection. Two
byte-identical GEMM configs under one shared heuristic cannot differ 7×,
so under autotune-off the two calls must not be getting the same
lowering at all. Revised theory: with autotuning disabled, the
fusion-fed GEMM doesn't go to hipBLASLt the way the parameter-fed one
does — plausibly the dequant gets absorbed into an XLA Triton GEMM
fusion, or the custom-call falls back differently. Run 4 repeats
autotune-off with the dump enabled and reads what each call actually
became. Prediction, pre-registered again: the two E1b-vs-E1a modules
will show structurally different lowerings, not just different algorithm
numbers.

The practical stakes: the engine wants int4-resident weights feeding
GEMMs at E1a speed. If fusion-fed GEMMs are unreliable across both
autotuner settings, the robust design is an explicit two-executable
split — a dequant executable writing a f16 scratch buffer, and the
dense GEMM consuming it as a plain parameter — which run 2's HLO shows
would place ~0.3 ms of dequant against a 12 ms GEMM. That design also
composes with block streaming: dequant once per block per step, right
after the block's upload.

## 2026-08-20, night — run 4: theory two dies, and the truth is embarrassing

Run 4 repeated autotune-off with the dump enabled. The revised theory
("fusion-fed GEMMs get a different lowering") is HALF right and wholly
beside the point.

Half right: under autotune-off, neither E1a nor E1b goes to hipBLASLt —
XLA emits its own Triton GEMM for both (so autotuning is what routes
GEMMs to the 3×-faster hipBLASLt path; without it you get Triton at ~20
TFLOP/s — a real finding that stands). But the E1b and E1a Triton
fusions have IDENTICAL block configs — num_warps 4, 32×32 output tiles,
same everything — and the dequant fusion stays a separate kernel in
both autotuner modes. Identical kernels cannot differ 7×. Theory dead.

Then the actual answer, hiding in plain sight since run 2. Recompute
every E1b-minus-E1a delta as a RATE against the bytes E1b downloads and
E1a doesn't (full f16 output vs 4-byte scalar): run 2 gives 0.56, 0.53,
0.52 GB/s at S=512/4096/28672; run 4 gives 0.50, 0.53, 0.50. **Six
measurements, two different GEMM backends, one constant: ~0.5 GB/s.
The "20× slower quantized GEMM" was never the GEMM — it was
`toSliceAlloc` device-to-host readback of the full output at half a
gigabyte per second** (a fresh pageable allocation per call, page
faults included, through PJRT). The GEMM was almost certainly at full
speed in every run. The "algorithm 91 vs 73" smoking gun of run 2
explained timing that this explains better — retracted; whether algo 91
is actually slower than 73 is UNKNOWN and no longer load-bearing.

This is the same genre of ghost as colibrì's 1.3 TB/s tensor-core
benchmark — a measurement artifact masquerading as a kernel property —
and this notebook exists precisely to catch these. The correction:
compute-side numbers must come from scalar-epilogue variants; full
downloads must be reported as what they are (readback benchmarks).

**Run 5, prediction pre-registered:** the 2×2 of {dense, int4-resident}
× {scalar-sum, full-download} at default autotuning. If the readback
theory is right: quant-scalar lands within ~15% of E1a's dense-scalar
~60 TFLOP/s (the ORIGINAL E1b hypothesis, resurrected), both
full-download variants sit ~0.5 GB/s above their scalar twins, and the
dense-vs-quant full-download gap is small. If quant-scalar is still
several-fold slow, something real remains after all.

Engine consequences if confirmed: int4-resident weights feeding
hipBLASLt GEMMs work at full speed with NO two-executable split needed
(XLA's own dequant-then-BLAS structure is already right); outputs must
stay on device between pipeline stages (they were always going to);
and the one place the engine truly downloads — final VAE frames — needs
pinned host memory or a better readback path than naive toSliceAlloc,
worth ~0.5 GB/s vs PCIe's ~25.

## 2026-08-20, late night — run 5: confirmed, and the engine path is open

The 2×2 at default autotuning settles it:

At S=4096, quant-scalar hits 42.81 TFLOP/s against dense-scalar's 42.39 —
**identical within noise**. The int4-resident dequant→f16→hipBLASLt path
runs at full dense speed; the original E1b hypothesis ("within 15%"),
refuted in run 2 and buried under two wrong theories, was correct all
along. And the two full-download variants are indistinguishable from
each other (260 vs 286 ms at S=28672, dense the slower one this time —
pure readback noise), both sitting at the predicted ~0.48 GB/s implied
readback rate. The run-2 "20× quantized penalty" is formally retracted:
it was toSliceAlloc, start to finish.

One residual, recorded honestly: at S=28672 quant-scalar reads 43.8
TFLOP/s against dense's 58.9 — a 1.34× gap (4.2 ms) that is far above
the ~0.3 ms the dequant fusion itself can account for. Hypothesis for a
future session: this is the algorithm-selection question from run 2
returning at its true magnitude — the autotuner picking a slightly
worse hipBLASLt algorithm for the fusion-fed GEMM — worth ~1.3× at the
largest size, not 20×. Open item, not a blocker.

### Day one's ledger

Established, each with an oracle or a forensic trail: the qdot/dot
crossover sits between 8 and 64 rows (decode trick stays on the decode
side); XLA routes large GEMMs to hipBLASLt and reaches ~59 TFLOP/s f16
on gfx1200 — matrix cores engaged, and autotuning is worth 3× (without
it, Triton fallback at ~20); int4-resident weights feed those GEMMs at
full speed with XLA's own dequant-then-BLAS split, no custom
restructuring needed; naive attention materializes f32 scores and dies
at T=16384 on 16 GB while running 12× below GEMM efficiency where it
fits, so E3 (in-graph blockwise attention) is the project's critical
path; and device-to-host readback via naive toSliceAlloc runs at ~0.5
GB/s, which poisons benchmarks that download and dictates that the
engine keep everything on device until the final frames.

Methodology note for the record: three hypotheses died today — E1b's
15%, the autotuner-instability story, and the different-lowering story
— and each death was cheap because its prediction was written down
before the measurement. The one that survived was the boring one. That
is the notebook working as intended.

Next session: E3 — blockwise online-softmax attention as an explicit
loop in the traced graph. Target: beat 4.9 TFLOP/s naive at T=8192,
run at all at T=28672, judged against the same CPU oracle. After that,
read the LTX-2.5 HF config to pin the real geometry (the 28 k estimate,
head count, and dual-stream split are still unverified).

## 2026-08-20/21, overnight — E3 first contact, and the real geometry arrives

### The estimated geometry is now verified geometry

The LTX-2.5 HF weights repo is gated (license acceptance plus a token —
noted for when we want real checkpoints), but the pipeline code is public
and `ltx-core`'s transformer configurator carries the numbers. Video
stream: **32 attention heads × 128 head dim (hidden 4096), 48 layers**,
128 latent channels, 3D RoPE over (t, x, y) with max positions
[20, 2048, 2048], RMS qk-norm (the blockwise kernel must apply it),
gelu-approximate FFN, bias-free in 2.5. Audio stream: 32 heads × 64
(hidden 2048), 1D temporal RoPE. The LTX-2 paper confirms the asymmetric
split — 14B video + 5B audio in LTX-2; 2.5's growth to 22B means the
2.5 checkpoint's own metadata (layers/FF width) may exceed these
defaults — unverifiable until the gate is passed.

Bonus findings from the card and code: frame counts obey %8==1 and
resolutions %32 — confirming the 8×/32× VAE compression my token
estimate assumed, so ~26k tokens for 10 s at 1216×704 stands, and the
x2 spatial/temporal upscalers exist precisely so generation happens at
that scale rather than native 4K. The 2.5 checkpoints also ship an
"int8-convrot" quant (rotation-folded quantization — a pleasing echo of
colibrì's E8 rotation fold), and the KV-cacheable checkpoints drop
timestep-dependence from cross-attention K/V so prompt K/V computes
once and is reused across all steps — a design gift for the engine.

My smoke had H=16 — half the real head count. Attention numbers from
runs 3–6 are internally consistent (E2 vs E3 at the same H) but all at
half the real memory pressure. H is now 32.

### Run 6: E3 tripped its own cross-check — tolerance miscalibration, round two

E1/E1a/E1b reproduced for the sixth time (readback 0.41–0.51 GB/s —
that constant is now beyond doubt). E3 compiled and executed at T=512,
then failed its cross-check against zml.nn.sdpa at rms 0.0103 vs my
5e-3 limit. Same lesson as run 2's oracle, one level up: the naive
reference itself deviates 0.0082 from the f64 oracle (its softmax runs
in f16), while blockwise carries f32 running stats — comparing two
~1%-accurate implementations against each other needs a ~2% budget,
and blockwise is likely the MORE accurate side of the pair. The check
now logs the measured rms and gates at 2.5e-2. Run 7 runs the full E3
sweep at real geometry; the open question is whether the unrolled
chunk loop's live set stays inside 16 GB at T=28672 with H=32 (one
[32, 28672, 1024] f32 scores chunk is ~3.8 GB — if XLA keeps two or
three alive across fusion boundaries it gets tight; CHUNK=512 is the
fallback).

## 2026-08-21 — run 7: E3 beats naive on speed, loses on the memory promise

First full E3 sweep at real geometry (H=32, HD=128). The good half: the
T=512 cross-check passed (rms 0.0103, inside the recalibrated budget),
and at T=8192 blockwise runs 151.8 ms / 7.25 TFLOP/s against naive's
211.3 ms / 5.20 — a 1.39× win, beating the pre-registered target. The
throughput arithmetic says both forms are bandwidth-bound on f32 score
traffic: at T=8192 the blockwise pass moves roughly 40 GB through VRAM,
which at 320 GB/s is ~125 ms — most of the measured time. Headroom for
later: f16 scores would halve that traffic; not today's problem.

The bad half: at T=16384 the unrolled form died with ResourceExhausted —
**the memory bound did not hold.** The algorithm needs one
[32, T, 1024] f32 scores chunk (~1.9 GB at 16384) live at a time; XLA's
buffer assignment across the 16 unrolled iterations evidently keeps
several alive. The compile-time-unrolled trace gives the scheduler
freedom, and the scheduler uses that freedom to spend memory. Genesis
entry E3 said "memory is bounded by construction" — construction turned
out to be a suggestion, not a bound, when the loop is unrolled.

The fix candidate is the real thing: ZML exposes `zml.ops.while`
(stablehlo.while, used by its GatedDeltaNet), and a while loop's
loop-carried state — (step, m, l, acc) here, with q/k/v as captured
context and dynamicSlice picking the chunk — forces buffer reuse across
iterations by construction. The trade: XLA cannot fuse across while
iterations, so some speed may go. One Zig detail worth recording: nested
fns cannot close over runtime values, so the chunk size becomes a module
constant (WCHUNK=1024) and anything runtime rides in the while context
as a Tensor.

**Run 8 prediction, pre-registered:** E3w (the while form) survives
T=16384 AND T=28672 — estimated live set at 28672 is ~10 GB (double-
buffered loop state ~1 GB, one 3.8 GB f32 scores chunk plus its exp'd
successor, q/k/v ~0.7 GB) — and lands within ±30% of the unrolled speed
at T ≤ 8192, since the workload is bandwidth-bound and fusion across
iterations wasn't buying much. If it still OOMs, WCHUNK drops to 512;
if it's dramatically slower, the host-driven multi-executable loop with
device-resident stats is the remaining card.

## 2026-08-21, afternoon — run 8 post-mortem: the experiment OOM'd the lab

Run 8 (the first E3w attempt) never reached E3w. The log shows a clean
build, PJRT load, E1 starting — then an endless "Clearing modules and
retrying hipModuleLoad" spin: the ROCm runtime could not load kernel
modules onto a GPU left in a bad state, almost certainly by run 7's
deliberate back-to-back VRAM exhaustions (E3-unrolled OOM at T=16384,
then E2-naive marching into its own 17 GiB allocation failure). The
agent session hosting the run died with it — on a 15 GiB-RAM machine
also carrying an IDE and, later, a heavy multi-threaded compute job
from an unrelated project, the margins are thin. Forensics after the fact: GPU back to ~500 MB used
and 2% busy once the stuck process was gone — the wedge was process
state, not hardware. No reboot was needed.

Lessons recorded: first, a smoke that INTENTIONALLY drives the
allocator into the ground twice per run is hostile to whatever else the
machine is doing — acceptable while E2/E3 memory limits were the open
question, but now that both failure points are measured, the naive
sweep should stop before its known OOM instead of demonstrating it
every run. Second, git discipline: the run-7 commit was accidentally
made inside .work/zml (a cwd slip — commands run from wherever the
previous one left the shell); it was reset out of the ZML checkout and
re-committed properly in this repo. Absolute paths or explicit `git -C`
from now on.

Run 8b retries E3w by executing the already-built binary directly —
no bazel server (a few GB of RAM returned), `nice -n 10`, sharing the
machine with that unrelated job. Benchmark-environment note: run 8b's CPU-
side timings carry that load; the E3w question (does the while loop's
bounded live set survive T=16384 and 28672) is load-independent, and
GPU-side p50s should be only mildly noisy. The run 8 prediction carries
over unchanged.

## 2026-08-20, evening — run 8b: E3w confirmed on every count

Run 8b (prebuilt binary, no bazel server, nice 10, sharing the machine
with an unrelated scheduled compute loop — which, it turns out, was
the mystery load, cycling a new job every ~25 minutes) delivered
the E3w verdict:

Correctness: while-vs-unrolled rms at T=1024 is **0.00000** — bit-
identical, same algorithm, same chunk order. The verification chain now
runs CPU f64 oracle → zml.nn.sdpa → unrolled blockwise → while
blockwise with no gaps.

Speed: the feared cost of losing cross-iteration fusion did not
materialize — 48.6 vs 47.9 ms at T=4096, and at T=8192 the while form
was marginally FASTER (125.9 vs 128.2 ms). Bandwidth-bound workloads
don't miss fusion across iterations.

Memory, the actual question: **T=16384 ran (449 ms, 9.8 TFLOP/s) and
T=28672 ran (1222 ms, 11.0 TFLOP/s)** — full LTX-2.5 working sequence
length, on the 16 GB card, through a compiler-traced stablehlo.while,
with throughput CLIMBING as T grows (better amortization of the loop
state updates). The unrolled form OOM'd at 16384 again in the same run,
completing the controlled comparison: the while loop's loop-carried
buffer reuse is what makes video-length attention fit, exactly as
pre-registered.

Honest engine arithmetic: 1.22 s per 48-layer step's worth of one
self-attention, times 48 layers ≈ 59 s of attention per denoising step
at current efficiency — minutes per clip on the distilled model before
optimization. Known headroom, in order: f16 scores (~2× — today the
chunk scores round-trip VRAM in f32), Q-tiling, and head-batching the
two GEMMs per chunk harder. Also noted: run 8b's numbers were ~15%
BETTER than run 7's despite the shared load — run-to-run variance on
this card is real; comparisons should stay within-run.

Hygiene applied: the naive E2 sweep now stops before its known OOM
(the smoke no longer wedges the GPU by design), verified with
`zig ast-check` after the stale-diagnostic scare.

**E3 is closed.** Attention was the make-or-break for the whole
project (genesis entry, "this project's dot_general moment") — it
broke the right way, twice over: hipBLASLt matrix cores for the GEMMs,
stablehlo.while for the memory bound. Next: E4 (VAE conv tiles), then
the first real transformer block against the HF oracle.

Addendum, run 8b close-out: the run finished end to end — the first
complete SMOKE COMPLETE with every experiment in one pass, exit 0, GPU
left healthy. One observation worth keeping: the naive T=16384 failure
surfaced at COMPILE time as error "Internal" this run, not at execute
as ResourceExhausted — the autotuner allocates real buffers while
compiling, so where the OOM manifests moves around. Reason enough that
the catch wraps compile and execute both, and that the cap (committed)
avoids the attempt entirely. The stack traces after SMOKE COMPLETE are
the debug allocator reporting the deliberately never-freed platform and
executables — cosmetic.

## 2026-08-21 — the checkpoint speaks: exact LTX-2.5 architecture, from the source

With the HF gate passed, a 677 KB range-request on the distilled bf16
transformer's safetensors header delivered what the gated repo had been
withholding: the full tensor manifest (4,349 tensors) and — because
LTX stores its config in the checkpoint metadata — the exact
architecture, model_version 2.5.0. No weights downloaded.

The measured facts. Total: **21.004 B parameters** in the DiT file
(the "22B" of the marketing rounds up), split 14.865 B video/shared and
6.139 B audio — the LTX-2 paper's 14B+5B asymmetric pair, with the
audio side grown. Both streams run the same **48 blocks**: video at
hidden 4096 (32 heads × 128 — our smoke geometry confirmed exactly),
audio at hidden 2048 (32 × 64), video↔audio cross-attention operating
in the 2048 space with per-side projections, all attention carrying
RMS qk-norm, biased QKV, and per-block 9-entry AdaLN tables driven by
a shared timestep embedder.

Three discoveries the configurator defaults did NOT show, each
engine-relevant:

First, **apply_gated_attention = true**: every attention — self,
cross, and the audio↔video bridges — has a `to_gate_logits` [32,
hidden] projection, one logit per head. Attention output is gated
per-head before the output projection. Cheap elementwise work that
fuses trivially, but the transformer-block oracle test will fail
mysteriously if it's forgotten, so it is recorded here in bold.

Second, the video FFN is **bias-free at 4× width** (16384,
gelu-approximate) while the audio FFN keeps biases at 8192 — the
ff_bias=false flag from the code applies to the video stream only.

Third, the prompt path is heavier than assumed: Gemma 4 12B output
(3840-wide) feeds **8-block connector transformers with 128 learnable
registers per stream** before it ever reaches cross-attention. The
connectors are full attention stacks (32 × 128 heads, gated, max_pos
4096) — a real sub-model, not a linear projection. And this
checkpoint does NOT set use_prompt_adaln_single=false, so the prompt
K/V here is timestep-dependent — the KV-cache-across-steps trick
noted earlier applies only to the separately-shipped kv-cacheable
checkpoints, not this one. Earlier optimism corrected.

Also pinned: RoPE frequencies are computed in **float64**
(frequencies_precision) with theta 10000 over max positions
[20, 2048, 2048] and causal temporal positioning — fine for us, since
the plan bakes position tables at trace time where f64 precision is
free. norm_eps 1e-6, RMS everywhere, norm_elementwise_affine=false.

The genesis entry's last "to verify" item is now closed. Milestone 5
(one transformer block, conformant against the HF implementation) has
an exact blueprint: patchify 128→4096, 48× [gated self-attn with 3D
RoPE + gated cross-attn against connector output + bias-free 4× GELU
FFN, AdaLN-modulated], with the audio stream and bridges alongside.

## 2026-08-21 — Phase 1 opens: E4 hypotheses, and the Conv VAE's real ladder

Same order as always: shapes from the source, hypotheses on paper,
then the benchmark. A 170-tensor header fetch on
`vae/ltx-2.5-video-vae-conv-bf16.safetensors` (726 M params total)
gives the decoder exactly: conv_in 128→1024, then five stages of
3×3×3 residual convs — 2 blocks at 1024 ch (latent scale), 2 at 512
after a 2×2×2 pixel-shuffle upsample, 4 more at 512 after a second
2×2×2, 6 at 256 after a temporal-only ×2, 4 at 128 after a 2×2
spatial — closing with a 48-channel conv that unshuffles to RGB × 4×4.
Spatial 2·2·2·4 = 32×, temporal 2·2·2 = 8×: the documented compression,
now accounted for stage by stage. Roughly 2–4 GFLOP of convolution per
latent voxel end to end, so a 10 s 1216×704 clip (≈26 k voxels) costs
on the order of 78 TFLOP to decode.

**E4 hypotheses, pre-registered:**

First, the lowering: XLA on ROCm will send these convs to a library
custom call (the conv analogue of the cublas→hipBLASLt routing E1a
found) — if instead the dump shows a native loop emitter, throughput
craters and that finding restructures Phase 5. The dump arbitrates.

Second, throughput: a 3×3×3 conv at C≥256 carries 27·C MACs per output
element — GEMM-class arithmetic intensity — so if the library path is
real, the wide stages should land in the tens of TFLOP/s and the
128-channel stage may lean bandwidth-bound. Under those numbers the
whole decode is seconds per clip, and the conclusion would be that
**the VAE is not the bottleneck** — attention holds that title —
making Phase 5's real risk memory and seams, not speed.

Third, memory: a latent tile of 8×16×16 keeps the largest activation
(128 ch at 64×128×128) around 270 MB and the whole tile pipeline under
~1.5 GB — comfortable next to resident DiT weights. The benchmark
sweeps the four workhorse shapes at that tile size.

Notes for honesty: the benchmark runs f16 rather than bf16 (RDNA 4
rates them identically and Zig has no native bf16 host type — noted,
not hidden), and it uses symmetric SAME padding where the real decoder
is temporally causal — identical arithmetic, different edge semantics,
irrelevant to throughput. Seam handling (the receptive-field halo is
several latent voxels per side by the deepest stage; overlap-vs-blend
is a real trade at tile 16×16) is deliberately deferred to Phase 5
design, as the roadmap says. ZML needs no patch for any of this: its
`convolution` is generically N-dimensional; conv1d/conv2d are wrappers,
and a third spatial dimension is just longer arrays.

## 2026-08-21, continued — E4 answered: MIOpen delivers, and Phase 1 closes

First result, before any number: the hypothesis entry's "ZML needs no
patch for any of this" died within the hour. `convolution` IS
generically N-dimensional — and private. conv1d/conv2d are its only
public doors. The project's second one-line ZML patch
(`zml-pub-convolution.patch`) exposes it; setup.sh applies it.

Then the run (oracle first: conv3d vs CPU f64 at rms 0.00021 — the
dimension-number wiring is right). Hypothesis 1 confirmed: every conv
lowers to the `__cudnn$convForward` custom call, which on ROCm is
MIOpen — the conv twin of E1a's cublas→hipBLASLt discovery. The
compiler stack's fast path, again, is "hand the op to the vendor
library," with zero hand-written kernels in this repo.

Hypothesis 2 confirmed, with compute-isolated numbers (scalar-sum
twins, the lesson runs 4/5 taught): stageA 1024ch at 26.9 TFLOP/s, the
1024→4096 upsample conv at 33.9, stageC 512ch at 30.9, stageE 128ch at
20.7 — the predicted narrow-channel dip is real but mild. The
full-output runs are a third independent confirmation of the readback
constant (~0.7 GB/s: 374 ms for stageE's 268 MB).

Hypothesis 3 survives with a corrected magnitude. Real arithmetic per
8×16×16 latent tile: stage C's eight 512-channel convs at 60 ms each
dominate (~480 ms), stages D and E add ~360 ms each, the top and the
upsamples ~120 ms — call it ~1.4 s per tile. A 10 s 1216×704 clip is
~24 tiles before overlap: **roughly 35–60 s of decode per clip.** Not
the "seconds" the hypothesis hoped, but the conclusion holds — that is
comparable to ONE denoising step's attention, so across 4–8 distilled
steps the VAE is a co-star, not the bottleneck.

And the seam plan, written down as Phase 1's acceptance demanded: the
decoder's receptive field, summed across ~44 3×3×3 convs at their
respective scales, is ~15 latent voxels of halo per side — as large as
the tile itself. Exact-overlap tiling is therefore untenable; Phase 5
decodes with modest overlap (~4 latent voxels) and feathered blending
in pixel space, accepting approximation at seams, with correctness
judged by rms in overlap regions and eyes on the output — the same
trade every production tiled decoder makes, now with the arithmetic
that forces it recorded.

**Phase 1 complete.** E1 through E4 are all answered. Next: Phase 2 —
one checkpoint-exact transformer block against the reference, using
range-requested block-0 weights.

## 2026-08-21, evening — Phase 2 spec: the reference forward, read line by line

Before writing a line of block code, the reference (`ltx-core`
transformer sources, read in full) answered every ambiguity — including
five that could not have been guessed. Recorded here so the oracle
diffs, when they come, are diagnosable.

**The block wiring, exactly.** All AdaLN values are PER-TOKEN (the
timestep embedding carries a token dimension — causal conditioning
means tokens can sit at different noise levels), computed as the
per-block table plus the embedding, chunk order (shift, scale, gate):
slots 0–2 self-attention, 3–5 feed-forward, 6–8 cross-attention
query-side. Sequence: modulated-RMS → gated self-attention → residual
add with gate_msa, then a FRESH un-weighted RMS of the residual which
feeds cross-attention (which does NOT re-normalize); cross-attention
applies its own query modulation, K/V get the 2-row prompt table plus
the prompt-timestep embedding, output is gated and residual-added;
feed-forward last with its own modulation and gate. Block RMS norms
carry no learned weight (norm_elementwise_affine=false); the qk norms
DO carry weights and act on the FULL 4096-dim projection, not
per-head. Norm precedes RoPE. Cross-attention has no RoPE at all.

**Gates:** logits from the attention's own (modulated) input, one per
head, and the gate is **2·sigmoid(logit)** — the factor of 2 is the
kind of detail a re-implementation from the paper would never contain.
Gating multiplies the per-head outputs BEFORE the output projection.

**RoPE, the five surprises.** (1) The frequency grid is
theta^linspace(0, 1, dim/6) · π/2 — increasing from π/2 to θπ/2, not
the classic inverse-power ladder. (2) Positions are FRACTIONAL:
pos/max_pos mapped to [-1, 1]. (3) The per-axis products are laid out
freq-major with (t, x, y) triplets adjacent, giving 3·682 = 2046
values — then **front-padded with 2 identity slots** (cos 1, sin 0) to
reach dim/2 = 2048. (4) Those 2048 slots are then SPLIT ACROSS THE 32
HEADS — head 0 gets the padding and lowest frequencies, head 31 the
highest; heads are not interchangeable. (5) Rotation pairs are
(i, i+64) within each head's 128 dims — split-half, not interleaved —
and table generation follows the numpy float64 path
(frequencies_precision = float64 in the checkpoint).

**Pass criteria, pre-registered before any diff exists:**
RoPE cos/sin tables must match the f64 reference to the last ulp after
the cast to f32 — anything worse means the grid formula is wrong, and
nothing downstream is diagnosable until it's exact. Individual
components (qk-norm, gate path, FFN) on real bf16 weights: relative
RMS ≤ 2e-3 against an f64-accumulated reference. The full block
output: relative RMS ≤ 1e-2, with the honest expectation of ~3e-3 —
if it needs the full budget, that is itself a finding to chase, not a
pass to celebrate. Weight pull verification happens BEFORE any forward
pass: per-tensor SHA-256 over the ranged bytes, shape/dtype checked
against the header, and first/last elements printed and compared with
the reference loader's view of the same tensors.

**Composition order for the oracle runs**, so the first mismatch
localizes itself: tables alone → attention with gates forced to
identity → gates on → feed-forward → AdaLN modulation → full block.
The 32/32 discipline works because each check is small.

Build plan: `tools/fetch_block.py` (range-request block 0 + shared
tables by header offsets, ~600 MB, checksummed), a torch-CPU
reference venv that runs the upstream block and dumps input/output
test vectors, then the ZML block and `//ltx:block_conformance`.

### Phase 2 spec amendments (review adopted before implementation)

Three amendments from review (Adam's local agent, vetted and adopted),
recorded before any code exists:

**The RoPE cast boundary, made explicit.** All position, frequency,
angle, sin and cos math in f64 (the numpy path the checkpoint
selects); one single rounding site — the final cast to f32, exactly
where the reference's `precompute_freqs_cis` casts to out_dtype. The
oracle artifact is the SERIALIZED F32 TABLE — the runtime contract —
not an abstract f64 computation. Gate, stated honestly: f32 bit
equality for at least 99.99% of entries, stragglers within 1 f32 ulp
and COUNTED (cross-libm f64 trig can differ in the last f64 ulp, which
occasionally lands on an f32 rounding boundary; a platform libm
difference must not become a false blocker, but the count must be
visible so a real formula error cannot hide inside the allowance).
Table metadata persists alongside: dims, linspace endpoints, position
ordering, flattening order, pad placement, rounding site.

**A sparse probe suite before dense comparisons.** Deterministic
inputs that isolate one thing each: one active axis at a time (t, then
h, then w, others zero); heads 0, 15, and 31 (the across-heads
frequency-band split is our most consequential discovery — probe it
directly); the identity-pad slots, the first, middle, and final
rotation pairs; fractional positions at both endpoints and both signs.
A per-head band-allocation bug fails a probe loudly where dense
rel-RMS would smear it into ambient error.

**Diagnosable mismatches by construction.** Every oracle comparison
emits max-abs, rel-RMS, NaN/Inf counts, and the argmax-error index
DECODED into batch/token/head/channel/branch, with reference value,
ours, and a small neighborhood. The first nonconforming stage should
localize itself, not open an afternoon of bisection.

Bundle contract for the torch oracle: raw binary tensors plus a JSON
manifest (shapes, dtypes, hashes, seeds, torch version, spec
revision) — no format conversions that can silently alter values. The
fetcher pins the HF revision by commit hash, validates
non-overlapping expected ranges before writing, and records SHA-256
plus first/last elements per tensor.

## 2026-08-21, late — Phase 2 build session 1: the oracle side is done

The pre-registered protocol executed without a surprise, which after
this week feels like a result in itself.

Weights: `tools/fetch_block.py` pulled block 0's 28 video-stream
tensors (513 MB) by revision-pinned range request — non-overlap and
exact-size validation before any byte landed, SHA-256 per tensor,
first/last elements logged and all at plausible weight scale. No
byte-shift, no transposition surprises at the container level.

Oracle: `tools/make_oracle_bundle.py` runs the UPSTREAM block — real
ltx-core code, real weights, torch CPU in f64 — and dumps the staged
composition in exactly the pre-registered order: modulated norm,
q-norm probe, gate-bypassed attention (via the reference's own ops
plumbing, no monkey-patching), gated attention, post-SA residual and
fresh norm, cross-attention contribution, FFN input and output, block
output. RoPE tables go through the reference's f64 numpy path and are
cast once to f32 — the runtime contract — with the full generation
metadata persisted beside them. The indices grid leads with the probe
rows: eight tokens with only t active (endpoints included), eight for
h, eight for w, then a deterministic mesh.

Two verification results worth their ink. First, the staged
decomposition reproduces the block's real `forward` at **max abs diff
0.00e+00 in f64** — the staging is the reference, not an
approximation of it, so a stage-level mismatch on our side indicts our
code and nothing else. Second, the calibration gift: the reference
run in its own bf16 deployment dtype lands at **rel-RMS 0.00592**
from its f64 self. The pre-registered full-block tolerance of 1e-2
now has a measured meaning — a ZML bf16 block near 6e-3 is
numerically indistinguishable from the reference's own precision
floor, and anything an order worse is a real defect, not dtype noise.

Bundle: 23 MB, 18 tensors, hashed manifest with per-tensor stats,
torch/numpy versions, seed, and the RoPE metadata. Next session: the
ZML side — f64 table generation matching the numpy semantics, the
block graph, and `//ltx:block_conformance` walking the stages in
order.

## 2026-08-21, night — the block conforms: every gate passes

Four runs from first compile to full pass, and the failure ledger is
short enough to quote in full: one invalid format string, one tensor
tag rename, and one genuine specification discovery.

The discovery deserves its ink. Run 2's RoPE gate failed with 223,669
of 262,144 table bits differing — and the structured straggler records
turned that wall into a single line: the reference's slot 2 held
cos(f32(π/2)), not cos(f64 π/2). **The "double-precision" RoPE path
rounds its frequency grid to f32 before use** — generate_freq_grid_np
computes linspace and powers in f64 and then returns
dtype=torch.float32; the f64-ness lives only inside the grid
computation, and the products and trig run on f32-rounded frequencies
promoted back to f64. No document says this; the bit patterns do. A
second, self-inflicted find rode along: probe positions like 20·i/7
are not f32-representable, so the serialized grid differed from what
the oracle computed with — production grids are integer pixel
coordinates, so the probe grid became integer-valued and the
serialization lossless. Both fixes in, run 3's RoPE gate read
**0/262,144 straggler bits, worst 0 ulp** — bit-perfect, with the
coordinate-pinned exception set empty. The gate protocol earned its
strictness: a looser "close enough" table check would have buried a
real spec fact under tolerance.

Then the ladder, run 4, in pre-registered order, every stage against
the f64 oracle with the 2e-3 gate: modulated norm 7.5e-8; q-norm probe
1.2e-6; gate-bypassed attention 1.5e-6 (projections, full-4096
weighted RMS, split-half RoPE with per-head bands, sdpa, merge,
output projection — all correct at once); gated attention 1.4e-6 (the
2·sigmoid gates); post-SA residual and fresh norm ~9e-7;
cross-attention with query modulation, prompt-table K/V modulation,
and gate 1.6e-6; FFN input and output ~1e-6 and 2.1e-6; and the full
block at **rel-RMS 1.85e-6** — three orders of magnitude inside the
gate, and ~3000× tighter than the reference's own measured bf16 floor
of 5.9e-3.

**Milestone 5 is complete.** A ZML-traced graph, compiled by XLA to
this AMD card, computes LTX-2.5's video transformer block to
float32 working precision, verified stage-by-stage against the
upstream implementation running real block-0 weights. Phase 2 closes
same-day. Phase 3 — all 48 blocks, quantized residency, the distilled
scheduler — now has a proven template: this block graph, repeated,
with the E1b-validated int4 path under it.

## 2026-08-21, later — Phase 3 entry criteria, pre-registered

Review amendments adopted before assembly begins. First, the scope of
Phase 2's result, stated the way external readers should read it:
measured conformance for the declared configuration — pinned upstream
implementation, checkpoint revision, oracle bundle, runtime — not a
general guarantee across model revisions, compiler versions, hardware,
or quantized 48-block execution. The qualifier strengthens the claim:
it makes it reproducible, and it stops Phase 3's changes from
inheriting a guarantee they haven't earned.

The risk has moved. The transformer block is no longer where semantic
danger lives; composition boundaries are. In pre-registered priority
order: int4 dequantization placement, group layout, scales and
accumulation dtype; cross-block dtype transitions and residual
accumulation; long-sequence attention masking, traversal order, and
online-softmax reduction at scale; distilled scheduler timestep
mapping, prediction parameterization, and latent scaling; and tensor
residency plus loader behavior under all 48 blocks.

**Phase 3's first checkpoint is NOT a generated clip.** It is a
three-block chain — blocks 0, 23, and 47, early/middle/late — in
float32 with the existing stage-walking oracle at every block
boundary. Only after that passes: int4 introduced one projection
family at a time, each compared against the f32 block path by
projection class; then the E3w video-length attention substituted
after establishing an overlap domain where it and dense attention
agree (the E3 cross-checks at T ≤ 8192 are the template); then the
scheduler, compared state-by-state and update-by-update before any
clip is judged. Every run record freezes the oracle bundle hash,
model revision, RoPE metadata, and compiler/runtime versions.

Blocks 23 and 47 are fetching as this is written.

## 2026-08-21, late night — Phase 3 checkpoint 1: the 0/23/47 chain conforms

Two defects between plan and pass, both instructive. First, the chain
oracle's exactness assertion caught a serialization phantom: the
parent bundle computed on f64 inputs BEFORE their f32 serialization,
so every bundle consumer — the Zig side included, all along — saw
inputs the oracle never exactly used, worth a ~1.5e-5 phantom at chain
boundaries. Inputs are now rounded through f32 at generation, making
the serialized bytes the exact contract; chain_b0_out then equals the
parent block_out to the bit. That is the second time in one day an
exactness assertion converted a would-be debugging afternoon into one
edit — a tolerance check would have absorbed the phantom silently and
let it masquerade as accumulating compute error across 48 blocks.
Second, Zig's comptime reflection over the nested three-block struct
(84 tensor fields) exceeded the default evaluation quota — one
@setEvalBranchQuota, noted for when the 48-block struct multiplies
that by sixteen.

The result: with real weights for blocks 0, 23, and 47 (early, middle,
late) chained under shared timestep/context/RoPE per LTXModel
semantics, boundary gates read **chainB23 rel-RMS 2.43e-6 and
chainB47 2.74e-6** against the f64 oracle — sub-linear error growth
(1.84 → 2.43 → 2.74e-6 across one, two, three blocks), extrapolating
to roughly 1e-5 across all 48. Activation magnitudes grow modestly
(253 → 495 → 717 max-abs under random modulation). Cross-block
composition — risk #2 in the pre-registered Phase 3 order — is clean
in f32. All twelve single-block gates re-passed unchanged on the
regenerated bundle.

Next per the pre-registered ladder: int4, one projection family at a
time, compared against this f32 chain by projection class.

## 2026-08-22 — int4 rung: budgets pre-registered before the first diff

The character of the oracle changes here. Every gate so far asked "is
this exact enough to call conformant" and heard ~1e-6. Quantization
produces errors orders of magnitude larger BY DESIGN, so the
pre-registration that matters is the acceptance threshold per
projection class, written before any diff exists — a tolerance chosen
now is a decision; chosen after, a rationalization. Also recorded from
review: the f32 chain stays FROZEN as the baseline (never regenerated
in the same session that compares against it); the ~1e-5 48-block
extrapolation is an estimate, not a bound, since 0/23/47 cannot rule
out a pathological interior block (cheap insurance noted: one more
chain through a random interior triple; not a blocker); and when the
48-block comptime pressure returns, the structural fix is a runtime
array over one comptime block type, not a bigger quota.

**Quantization scheme, rung 1:** per-output-row symmetric absmax int4
(q = clamp(round(w/s), -8, 7), s = absmax/7 per row), weights stored
unpacked-u4 through the proven sub-byte path, scales in f32 in-graph
(E1b showed residency is free; no reason to spend precision there),
dequant traced exactly as E1b validated. Projection classes, in
introduction order: qkv (the six [4096,4096] q/k/v matrices of both
attentions), then output projections, then the FFN pair. Gate logits,
norms, tables, and biases stay f32 permanently — they are noise-sized.

**Budgets, pre-registered:** qkv class — pre-softmax logit rel-RMS
≤ 5e-2 and post-attention output ≤ 2e-2 against our own f32 graph
(itself torch-anchored at 1.8e-6, so the comparison is transitively
grounded); full block with qkv-int4 ≤ 2e-2. Two probes per the review:
logit error and output error separately, because softmax either washes
quantization out (high-entropy attention) or amplifies it
(near-saturated), and the two failures have different fixes.

**Escalation ladder, also pre-registered:** if a class blows budget,
the lever is granularity, not retreat — grouped-128 scales first, then
asymmetric quantization on the offending projections (video
transformers carry outlier channels). Both compose with the
dequant-in-graph path unchanged.

## 2026-08-22, continued — the quantization ladder, climbed to a pass

Four rungs, every number pre-registered or forecast before measurement,
ending somewhere better than the budget asked for.

Rung 1, per-row symmetric int4: weight-space rel-RMS 0.19–0.36 across
the six qkv matrices (scale ranges spanning 50×: heavy outlier rows).
Probes: logits 0.348, attention output 0.245, block 0.166 — all far
over the 5e-2/2e-2/2e-2 budgets. The informative part: error
ATTENUATES through composition (0.35 → 0.25 → 0.17), so softmax washes
quantization out rather than amplifying it — the failure is pure
projection precision, exactly the case the granularity lever addresses.

Rung 2, grouped-128 int4: weight error only improved to 0.13–0.20
(typical LLM weights improve 2–3× here; these improved 1.4×). The
outliers live INSIDE 128-element groups. Probes 0.237/0.145/0.091 —
still 3–5× over.

Rung 3, asymmetric grouped-128, measured in weight space only:
0.142/0.164 — the predicted ~1.2× gain against a needed ~4×. Probe run
skipped with that justification on the record. **Ladder finding, and a
real fact about the model: LTX-2.5's attention projections are
outlier-heavy enough to defeat plain int4 at any granularity or
asymmetry.** This is presumably WHY Lightricks ships int8-convrot —
rotation-based outlier suppression — rather than plain low-bit quant.
Rotation remains open as a future rung (and this repo's coli-zml
snapshot contains working rotation-fold machinery as prior art); the
pragmatic rung came first.

Rung 4, int8 grouped-128 for the qkv class: weight error 0.007–0.012
(the 16× step-size refinement, as arithmetic predicted). Probes:
logits 1.50e-2 (budget 5e-2), attention output 9.25e-3 (budget 2e-2),
full block **4.99e-3 (budget 2e-2) — BELOW the reference's own
measured bf16 deployment floor of 5.9e-3.** The int8-attention block
is numerically indistinguishable-or-better relative to shipping
precision. All seventeen gates green in one run: twelve single-block,
two chain, three quantization.

Emerging mixed-precision plan (to be completed classwise): attention
projections int8-g128; FFN class next at int4-g128 (FFN weights are
usually better-behaved — measured, not assumed, next); gates, norms,
tables, biases f32 forever. Memory arithmetic at that split: roughly
13 GB for the full dual-stream DiT — on-card, with block streaming as
backstop.

## 2026-08-22, continued — FFN class rung, budgets pre-registered

The stakes make this rung the load-bearing one for residency: the FFN
pair is the parameter bulk (134M per video block). All-int8 puts the
DiT near 21 GB (streamed, not resident); attention-int8 plus FFN-int4
lands near 13 GB (resident). So int4-FFN is the difference between a
resident model and a streamed one — both viable (streaming is proven
roadmap tech), but resident is simpler and faster.

Budgets before the diff: FFN-isolated probe (f32 prefix up to the real
ff_in, int4-g128 FFN applied, compared against the f32 s7FfOut) ≤
5e-2 — looser than attention's 2e-2 because the FFN output enters
through a gated residual and dilutes at block level; the block gate is
what binds. Full block with int8-qkv AND int4-FFN ≤ 2e-2, same as
before. Hypothesis: FFN weight distributions are the well-behaved ones
(the qkv outlier pathology is attention-specific in most transformers)
— weight rel-RMS ~0.05–0.10, ff_out similar, block diluted ~2×.
Escalation if it fails: int8 FFN and block streaming — the trade
recorded above, no drama either way.

## 2026-08-22, continued — FFN rung: hypothesis refuted, recipe settled

The FFN-weights-are-well-behaved hypothesis died in weight space before
a probe ran: int4-g128 measured rel-RMS 0.187/0.213 on the FFN pair —
as pathological as the attention matrices. **The outlier structure is
model-wide in LTX-2.5, not attention-specific.** Probe skipped under
the same justified-skip protocol as ladder rung 3 (a 6× weight-space
gap cannot pass downstream), and the pre-recorded escalation taken:
int8-g128 FFN, accepting the residency consequence.

Int8 results: FFN weight error 0.011/0.013 (same profile as the
passing qkv class). Probes: FFN-isolated 2.07e-3 against its 5e-2
budget — 24× inside; and the FULLY QUANTIZED block (int8-g128
attention and FFN together) at **5.24e-3 — statistically at the
reference's own bf16 deployment floor of 5.9e-3.** Quantizing the FFN
added five parts in ten thousand to block error. Nineteen gates green.

The settled recipe, receipt-backed: int8-g128 for every large
projection, f32 for gates/norms/tables/biases, dequant in-graph.
Consequence accepted and priced: the DiT weighs ~20 GB quantized —
streamed through the deterministic block-prefetch design rather than
resident. The residency dream at int4 dies on the model's actual
weight statistics; the rotation rung (Lightricks's own convrot road,
with working fold machinery in the coli-zml snapshot as prior art)
stays open as the future play that could halve that again.

Phase 3's quantization item is closed. Remaining before the 48-block
assembly: the E3w attention swap behind its overlap-domain gate, the
scheduler comparison, and the block-streaming loader.

## 2026-08-22, evening — the streaming loader's design, from an unexpected angle

Adam connected his agent-memory work (a tombstone-based lifecycle
design) to the streaming problem, and the conversation produced the
loader's spec — after one measurement and one mechanism correction.

The measurement first, house rules: cross-block weight similarity on
the fetched blocks 0/23/47 reads cosine +0.0001 to +0.0016 with
inter-block deltas carrying MORE energy than the tensors themselves
(rel-RMS 1.16–1.19, near independent-noise √2). Structural redundancy
across layers is dead as a working-set reducer; entropy coding would
recover maybe 5–15% and is not worth a hot-path decompression stage.

The correction: mmap is zero-copy only host-side. DMA from pageable
memory routes through a hidden pinned bounce buffer — the same
~0.5 GB/s pathology measured three times on readback, reversed. The
pipeline therefore stages explicitly, every arrow overlapped because
the block order is compile-time constant:

NVMe → page cache (mmap + kernel readahead) → pinned host ring
(memcpy) → VRAM ring (async DMA) → compute.

And the relocated insight, which is the load-bearing one: this machine
has 15 GB of RAM against a ~20 GB quantized model — the model cannot
be HOST-resident either. The "overflow via virtual memory" instinct is
exactly right at the disk→RAM boundary: the page cache is the overflow
manager, holding the hot ~10 GB. Rates: ~20 GB streams per denoising
step; NVMe ~5 GB/s with cache assist ≈ ~2 s/step of disk traffic —
free against today's ~60 s/step attention, and the term the
prefetch-ahead-of-wavefront design exists to hide once attention gets
its planned 2×+.

Allocation governance: PJRT owns the VRAM allocator, but 48 blocks of
IDENTICAL geometry mean every allocation is the same size, and
same-size cycling through a binned allocator cannot fragment — the
tombstone benefit arrives via architectural regularity. What transfers
from the agent-memory work wholesale is the lifecycle-invariant
discipline, adopted as the loader's pre-registered design contract:
ring slots with explicit states (filling → ready → in-use → free),
transitions asserted, zero steady-state allocation, and — the schedule
being deterministic — the whole state machine verified as a static
cycle at startup, before the first frame.

## 2026-08-21 — E3w moves into the block: hypotheses before the swap

Erratum first: the calendar drifted during the overnight sessions. The
five entries above dated 2026-08-22 (int4 rung through the streaming
design) were actually written late 2026-08-20 into 2026-08-21 — a day
ahead of reality. The headers stay as written (they are committed and
pushed; the ORDER is correct and that is what matters for provenance),
but from here the dates are back on the real clock. Lesson for the
journal: dates are data too, and data gets checked.

Now the first of Phase 3's three remaining pre-assembly items, in the
pre-registered order: the E3w attention kernel swapped into the block
path, behind an overlap-domain agreement gate.

Why the swap is not optional. At deployment length the dense score
matrix is [32 heads, T, T] f32 with T ≈ 26–28k: roughly 100 GB for one
block's self-attention. Run 7 showed the trace-time-unrolled blockwise
form OOMs at T=16384 because XLA's buffer assignment keeps chunk
intermediates alive across unrolled iterations; run 8b showed the
`stablehlo.while` form with loop-carried (step, m, l, acc) runs
T=28672 in bounded memory at 11 TFLOP/s, bit-identical to the unrolled
form where both run. So the while kernel IS the production attention.
But every conformance number we have — 19 gates, the chain, the
quantized block — was earned with `zml.nn.sdpa`, XLA's dense softmax
path. The kernel that ships has never run inside the block, surrounded
by qk-norm, RoPE, the 2σ gate, and the output projection. That is the
gap this entry closes.

The overlap-domain design, following the E3 template (dense and
blockwise cross-checked at T ≤ 8192 where both fit): at the harness
geometry T=64 both algorithms run comfortably, so dense — already
anchored to the torch oracle at 1.85e-6 — becomes the baseline, and
the while kernel is measured against it with everything else held
fixed. Two choices make the isolation honest:

First, the harness twin of the E3w kernel is written in f32, NOT the
f16-matmul deployment form. The conformance harness is f32 throughout
precisely so that spec errors are not confused with dtype noise; an
f16 kernel here would smear ~1e-3 of cast noise over a comparison
whose whole point is to see the algorithm alone. The f16 form is the
perf pass's business, and the bf16 floor (5.9e-3) is already
calibrated for when it arrives.

Second, a detail read out of `zml.nn.sdpa`'s source before
registering: it applies the 1/sqrt(hd) scale to K (not Q) and runs
softmax in f32. The twin copies the K-side scaling so that
scale-placement rounding cannot pollute the comparison. The smoke-test
E3w kernel scales Q — an ulp-level placement difference, noted here so
the perf pass reconciles it deliberately instead of discovering it.

Chunk size: 16, giving 4 loop iterations over T=64, of which 3
traverse the rescale-correction path (m_new, corr = exp(m − m_new),
the l and acc rescales). A single chunk of 64 would degenerate the
online softmax into plain max-subtracted softmax and test nothing but
the loop plumbing.

Pre-registered hypotheses and budgets, before any measurement:

H-E3W-1 (agreement): while-attn1 vs dense-attn1 stage output, same
weights, same inputs, both f32 — budget rel-RMS 1e-5, expected order
1e-7. The two compute the same mathematics; the only legitimate
difference is f32 reordering noise (4-chunk PV accumulation plus at
most 3 rescale multiplies ≈ a few ulp). A failure near 1e-1 means the
algorithm is wrong (chunk offsets, missing rescale); a failure near
1e-3 means something subtler and gets investigated, not negotiated.

H-E3W-2 (oracle held): the full block with while-attn1 vs the torch
oracle block_out, the standard 2e-3 gate — predicted within 2× of the
dense block's 1.85e-6. The swap should be invisible at block scale.

H-E3W-3 (compounding): the 0/23/47 chain with while attention in all
three blocks vs the chain oracle — predicted within 2× of the dense
chain's 2.74e-6, sub-linear growth preserved. Three swapped blocks in
series is the cheapest available evidence that the kernel's noise does
not compound pathologically.

Scope, pre-registered: cross-attention stays dense. S=32 in the
harness and at most ~1k tokens of prompt context at deployment — the
memory argument that forces E3w simply does not apply, and dense at
small S is the better-tested path. This is a decision, not an
omission.

## 2026-08-21, continued — the swap conforms; one expectation misses by 10× and teaches something

Implementation notes first, then numbers. The while kernel's f32 twin
went in as a module function; the block's `attention` gained a
comptime algorithm switch threaded through the recompute-prefix chain
(every public stage method keeps its name and dense default, so all 19
existing gates are byte-for-byte the same graphs). Two new public
probes — wAttn1, wBlockOut — plus wChainB47 on the chain. One tag
lesson worth keeping: ZML's `dot` leads its result with the batch
dims, so q [.q,.h,.hd] against k [.k,.h,.hd] produces [.h,.q,.k], and
the while loop's carried stats are [.h,.q] — read out of
`dotGeneral`'s source before writing the kernel, not discovered by a
shape panic after.

All 22 gates pass. The three new ones, against their registrations:

H-E3W-1, agreement: rel-RMS **1.094e-6** against a 1e-5 budget —
PASS with 9× headroom, but 10× above my expected ~1e-7. The miss has
an identifiable cause and it is not the algorithm: 1.09e-6 is exactly
the scale of ONE f32 4096-length GEMM's reordering noise (compare
s1bQnorm's 1.15e-6, which is a projection plus a norm and nothing
else). Dense and while attn1 are two separately compiled executables,
and XLA autotunes each GEMM per executable — the q/k/v/out projections
surrounding the attention can land on different hipBLASLt tilings in
the two graphs (the E1b forensics already showed the autotuner picks
different algorithms in different contexts). The twin-graph design
isolates the algorithm at the GRAPH level but cannot pin the
compiler's per-executable GEMM choices. The expected-order error was
naivety about cross-executable determinism, not about online softmax.
Registered for the future: any "these two graphs should agree to
ulps" claim must either share one executable or budget one GEMM's
reordering noise (~1e-6) per differing projection.

H-E3W-2, oracle held: wBlockOut vs torch **1.797e-6**, dense blockOut
in the same run 1.843e-6 — the swapped block is not just within the
predicted 2×, it is statistically IDENTICAL to dense against the
oracle. The swap is invisible at block scale, as registered.

H-E3W-3, compounding: wChainB47 vs the chain oracle **2.684e-6**,
dense chain 2.735e-6 in the same run. Sub-linear growth preserved
(1.80e-6 at one block → 2.68e-6 at three); no pathological
compounding. Confirmed.

Regression control, for free: every previously recorded gate
reproduced its number exactly — blockOut 1.843e-6, chain 2.735e-6,
fully quantized block 5.237e-3, RoPE 0/262,144. The harness is
deterministic run to run, which is what makes the agreement gate's
1.09e-6 attributable to graph differences rather than run noise.

Status: Phase 3 pre-assembly item one of three is closed. The
production attention now has conformance receipts INSIDE the block —
RoPE, qk-norm, 2σ gate, projections and all — not just standalone.
What this deliberately did not test, still owed later: the f16-matmul
deployment form of the kernel (perf pass, judged against the 5.9e-3
bf16 floor), and the Q-side vs K-side scale placement reconciliation
noted at registration. Next in the pre-registered order: the
distilled scheduler, compared update-by-update.

## 2026-08-21, later — the distilled scheduler, read line by line: it isn't the scheduler class at all

Phase 3 item two. House rule first: read the reference before
registering anything. The denoising loop is not in ltx-core — it lives
in ltx-pipelines, now pinned into the oracle venv beside ltx-core
1.2.0 (ltx-pipelines 1.0.0, installed --no-deps like everything else
there). The read produced seven findings, several of which would have
been wrong guesses:

One. The distilled pipeline does NOT use the LTX2Scheduler class (the
token-count-shifted sigmoid schedule with terminal stretching). That
is the dev-model path. Distilled ships HARDCODED sigma lists:
stage 1 is [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725,
0.421875, 0.0] — eight steps — and stage 2 is [0.909375, 0.725,
0.421875, 0.0], the three-step tail of stage 1's list. Had we
implemented "the scheduler," we would have implemented the wrong
component perfectly.

Two. The pipeline is two-stage: stage 1 at half resolution from pure
noise, then a 2× spatial upsample of the latent, then stage 2 renoises
the upsampled latent to sigma 0.909375 and runs the three-step tail.

Three. The step is plain Euler, but the loop routes through x0 space
with an asymmetry that matters: the model wrapper converts velocity to
denoised using PER-TOKEN timesteps = denoise_mask · sigma (mask values
in [0,1]; fractional strengths are legal per the type documentation),
post_process blends denoised·mask + clean·(1−mask), and then the
stepper converts BACK to velocity using the SCALAR sigma before
applying x += v·dt. For mask=1 tokens the round-trip cancels to plain
Euler on velocity; for mask<1 tokens it does not — that asymmetry IS
the conditioning mechanism. Scalar-Euler-on-velocity would be an
invalid simplification of this loop.

Four. The final step (sigma_next = 0) is not special-cased: the update
is x + ((x−d)/σ)·(−σ), which equals d only up to f32 divide/multiply
rounding. (The ancestral stepper DOES snap to denoised; the distilled
loop's Euler stepper does not.) The engine must reproduce the
rounding, not snap.

Five. to_velocity computes in f32 regardless of storage dtype, takes
sigma as a Python scalar via .item(), and raises on sigma=0 — which is
why the loop iterates sigmas[:-1] and never steps FROM zero.

Six. torch.lerp — the noiser's initialization is a chain of two lerps,
lerp(initial, noise, noise_scale) then lerp(clean, that, mask) — is
the two-branch numerically-stable kernel: |w| < 0.5 gives a + w·(b−a),
ELSE b − (b−a)·(1−w). Every weight this pipeline uses (noise_scale 1.0
and 0.909375; mask 0.5 and 1.0 — note 0.5 is NOT less than 0.5) takes
the SECOND branch. A naive one-formula lerp on the Zig side would
diverge in the last ulp. Read out of ATen's Lerp.h before writing,
not discovered by a failed gate.

Seven. Most sigma values are non-dyadic decimals (0.421875 = 27/64 is
the only exactly-representable interior value), so the runtime f32
sigmas are round-to-nearest of the decimal literals. Python and Zig
both parse decimal→f64 correctly rounded and agree on the f64→f32
cast, so embedding the literals in Zig is exact — cross-checked by a
gate anyway.

Bonus find, filed for item three: upstream ships a block_streaming
package (block_fetcher, pool, provider, stream_sync…) — direct prior
art for our streaming loader. A comparative read goes into the loader
entry, not this one.

Oracle boundary, pre-registered: tools/make_scheduler_bundle.py
IMPORTS and runs the reference's own euler_denoising_loop,
EulerDiffusionStep, post_process_latent, timesteps_from_mask,
to_denoised/to_velocity, and GaussianNoiser — no re-implementation on
the oracle side. The 21B transformer is replaced by pre-generated
per-step VELOCITY tensors — data, not a function — and the stub
denoise_fn applies the reference to_denoised with mask-scaled
timesteps exactly as the X0Model wrapper does. This makes the gate
blind to everything except the scheduler math, which is the point.
Storage is f32 on CPU, so serialization is exact by construction (the
Phase 2 f32_exact lesson, inherited at design time instead of
rediscovered). A secondary bf16-storage run of the same trajectory
calibrates the deployment floor.

Geometry: video [1, 64, 128] with mask rows 0–47 at 1.0, rows 48–55
at 0.0 (pure conditioning), rows 56–63 at 0.5 (fractional strength,
exercised deliberately); audio [1, 16, 32], all-ones mask, stepped
through the same loop because the reference steps both modalities
with one stepper. Both stages run chained like deployment: stage 2
initializes from a synthetic "upsampled" latent through the noiser at
noise_scale 0.909375.

Gates, pre-registered. The Zig side is a NEW host-only binary,
//ltx:scheduler_conformance, over a reusable ltx/scheduler.zig — no
GPU, no ZML dependency, because this is pure f32 arithmetic and the
engine will reuse the same module between DiT calls.

G-SCHED-1: the Zig comptime sigma tables vs the bundle's, bit-exact,
zero exceptions. G-SCHED-2: per-step timesteps, bit-exact. G-SCHED-3:
per-step denoised (the x0 conversion), bit-exact expected, with the
RoPE-style straggler protocol (≤1 ulp, ≤0.01% count,
coordinate-pinned) as the only fallback. G-SCHED-4: per-step
post-processed latent, same. G-SCHED-5: per-step stepped latent, both
stages, both modalities, same. G-SCHED-6: the stage-2 initialization
chain, same — predicted failure mode if any: the lerp branch.

H-SCHED-1: everything bit-exact, zero stragglers. Basis: elementwise
f32 with identical operation order on both sides; Zig's default float
semantics are strict IEEE with no contraction, and torch's vectorized
CPU kernels use explicit non-fused intrinsics. H-SCHED-2: the
bf16-storage trajectory diverges from f32 with step count; final-state
rel-RMS lands in the 1e-3 to 1e-2 band (the bf16-floor scale, eight
storage casts deep). Informative, not a gate.

## 2026-08-21, night — 94/94 bit-exact, after the gate structure catches torch fusing a multiply

Build notes first. The oracle generator hit upstream's own version
skew: PyPI ltx-pipelines (1.0.0, the only release) does not actually
import against PyPI ltx-core 1.2.0 — the pipelines package `__init__`
wants a `decode_video` that core no longer exports, and helpers.py
imports a `GemmaTextEncoder` that core renamed to LTXGemmaTextEncoder.
Core 1.2.0 stays pinned (it anchors every Phase 2/3 oracle), so the
generator loads the loop modules verbatim from their installed files
through a bare package shim plus a one-line class alias, both disclosed
in the script. The loop code executed is the reference's own,
unmodified. Bundle: 126 tensors; the stepwise-driven trajectory equals
the single-call reference loop BITWISE in both stages (the staging
control, inherited from Phase 2); twin-captured noise reproduces the
noiser's draws bitwise (asserted in-script).

First run: 74 of 94 gates bit-exact — all of stage 1, both modalities,
every sub-gate — and every one of the 20 failures downstream of the
stage-2 init, starting at ~10 ulp there and inflating to ~128 ulp
through the x0 conversion (values shrink as sigma falls; absolute
error persists; ulp count grows). The per-step, per-stage sub-gate
structure pointed at the exact first divergence: the stage-2
initialization — the only place a noise-scale lerp runs with a
NON-TRIVIAL weight. Stage 1 could never have caught it: weight 1.0
returns the endpoint under any formula, and weight 0.5's product by a
power of two is exact. A trajectory-level tolerance gate would have
shrugged at 1e-7; the bit gate localized a one-ulp defect to one
function.

Forensics, one element deep: torch.lerp's output bit-matches
fma(−(b−a), 1−w, b) — a FUSED multiply-add, single rounding — not the
two-rounding formula written in ATen's Lerp.h, which is what I read
and registered. The header's scalar reference is not what the
vectorized CPU tensor kernel executes. Confirmed over 2000 random
pairs per branch, both branches FMA-formed, bitwise. G-SCHED-6's
registered failure prediction said "the lerp branch" — right organ,
wrong lesion: the fusion, not the branch select. Fix: @mulAdd in
ltx/scheduler.zig's lerp, both branches.

Re-run: **94/94 gates bit-exact, zero stragglers, zero tolerated
diffs.** H-SCHED-1 confirmed as registered. H-SCHED-2 also landed:
bf16-storage finals diverge from f32 at rel-RMS 3.63e-3 (video) and
3.97e-3 (audio), inside the predicted 1e-3..1e-2 band — that is the
deployment floor the engine's latents inherit if stored bf16.

What the receipts now cover: the hardcoded distilled sigma constants;
per-token timesteps versus scalar-sigma stepping (the conditioning
asymmetry); the mask blend; the no-snap final step (divide and
multiply through sigma_next = 0, rounding included); and the noiser's
fused two-branch lerp — all reproduced bit-for-bit by
ltx/scheduler.zig, the module the engine will run between DiT calls.

Registered for the future, extending the E3w lesson from this morning:
a "same math" claim must name the KERNEL, not the formula — the
formula in the reference's header and the instructions its kernel
executes differed by one fused rounding, and only a bit-exact gate
distinguishes them. Phase 3 item two of three is CLOSED. Next: the
streaming loader, with upstream's own block_streaming package
(discovered during this read) as prior art to compare against the
design note before building.

## 2026-08-21, late night — upstream's own streaming loader, read as prior art

Our streaming design note was written before discovering that ltx-core
ships a block_streaming package (1,625 lines: pool, provider, source,
disk, stream_sync, wrapper, builder). Reading it before building is
free risk reduction — either it validates the design or it teaches
something. It did both.

What upstream built. Two modes: RAM streaming (every block pre-loaded
into pinned CPU buffers — assumes the model fits in host RAM, which
ours does not) and disk streaming (a background worker thread reads
blocks from safetensors — which mmap, so the page cache does the disk
caching — into a small pool of pinned CPU slots, default TWO slots
plus a prefetch-depth lookahead). GPU side: a fixed pool of slots
carved from ONE contiguous device allocation; each block's tensors
are laid out contiguously in one blob, so landing a block is a SINGLE
H2D byte copy, and per-key tensor views are carved from the raw slot
afterward. Ordering: a dedicated copy stream with cross-stream events
enforcing exactly two invariants — copy-before-compute, and
compute-before-slot-reuse. Blocks stream via forward pre/post hooks;
the overlap comes not from explicit H2D lookahead but from Python
enqueuing GPU work asynchronously and running ahead of the device, so
block i+1's copy is in flight while block i's kernels still execute.

Where it validates the design note: the staging chain is IDENTICAL
(mmap page cache → pinned host slots → device slots → compute); the
two event orderings are exactly our ring lifecycle invariants under
different bookkeeping; and their single-slab-carved-into-fixed-slots
allocation is a stronger form of our "uniform geometry cannot
fragment" argument. The disk mode's tiny CPU pool (two slots plus
lookahead) confirms the pinned ring does not need depth — the page
cache absorbs the variance.

Two adoptions, folded into the design as of tonight. First: ONE BLOB
PER BLOCK. Our quantized tensors currently live as ~30 files per
block; the loader wants each block packed into a single contiguous
byte blob with a fixed manifest-defined layout, landed with one DMA,
views carved after. A small pack step joins the tooling before the
build. Second: overlap via asynchronous enqueue depth rather than an
explicit prefetch state — IF the PJRT execute path is asynchronous
under ZML the same free overlap applies; that is now a question to
answer by measurement, not assumption.

Where we still differ, deliberately: upstream's RAM mode is
unavailable to us (15 GB host, ~20 GB model — the page-cache overflow
design stays); we stream our own int8-g128 blobs, not bf16
safetensors; PJRT owns the device allocator, so our "slab" is N
identical preallocated buffers rather than carved raw memory; and the
schedule being compile-time constant, the whole slot state machine
gets verified as a static cycle at startup — the agent-memory
discipline upstream's event bookkeeping does not attempt.

Pre-registered build rungs (the last Phase 3 pre-assembly item):

E-STREAM-1, transfer audit: measure ZML/PJRT host-to-device rate for
one block-sized blob (~425 MB at int8: 20 GB / 48) from pageable
memory and from whatever pinned path ZML exposes. Hypothesis: pageable
lands in the low single-digit GB/s (the readback pathology, reversed,
suggested ~0.5 GB/s is possible but upload usually fares better);
pinned multiplies that severalfold. Decision rule: even the WORST
plausible rate hides under today's ~60 s/step compute — the loader
cannot be the current bottleneck; what the measurement actually
decides is headroom for the perf pass's 2×+ attention and time-to-
first-block.

E-STREAM-2, overlap: does an upload proceed while an executable runs?
Measure wall time of compute-plus-upload issued together vs summed
serially. Hypothesis: ZML's call path is asynchronous enough that the
upstream free-overlap trick applies; if it is not, the loader gets
its own thread and the invariants stay identical.

E-STREAM-3, the ring: three device slots cycling 48 synthetic blocks
fed by a two-slot pinned host ring and an mmap reader thread, slot
states (filling → ready → in-use → free) asserted on every
transition, the full 48-block schedule verified as a static cycle at
startup before any I/O. Measured: steady-state stall per block
(hypothesis: ~zero once warm at current compute speeds).

Then the loader meets the 48-block assembly, which is the next phase
checkpoint after these rungs.

## 2026-08-21, later that night — E-STREAM-1/2: the transfer facts, and an overlap illusion that died twice

//ltx:stream_smoke, one block-sized blob (425 MB — the 20 GB int8 DiT
divided by 48).

E-STREAM-1a, pageable: p50 55.3 ms, **8.06 GB/s**. The upload
direction does NOT suffer the readback pathology — H2D through
BufferFromHostBuffer runs sixteen times healthier than the ~0.5 GB/s
toSliceAlloc D2H we measured three times in runs 3–5. The two
directions take different code paths in the plugin; never infer one
from the other.

E-STREAM-1b: PJRT_Client_DmaMap WORKS on the ROCm plugin. The same
region pinned: p50 31.0 ms, **14.4 GB/s** — 1.78× pageable. The
hypothesis said "severalfold"; reality says 1.8×. Registered as is.

E-STREAM-2 took three protocols to measure honestly, and the wrong
turns are the valuable part. Protocol 1 (cold start): compute+upload
measured FASTER than compute alone, 260 vs 405 ms — impossible, and
therefore diagnostic: the "alone" batch ran first on a cold,
unboosted GPU. Clock ramp poisons whichever condition runs first.
Protocol 2 (warmup plus before/after drift controls): compute alone
still drifted 297 → 241 ms across batches; the subtraction stayed
meaningless. Protocol 3 (interleaved A/B — both conditions sample the
same clock trajectory): compute alone p50 229.1 ms, compute+upload
273.0 ms, upload alone 31.0 ms. The upload adds ~44 ms — more than
its own standalone cost. **Zero percent hidden.**

The mechanism, pinned by one more probe: the fromBytes CALL returns
in 0.0 ms (fully asynchronous at the API — and ZML's exe.call is
asynchronous too, verified in exe.zig source), but the upload's ready
event signals only at 272 ms, after the in-flight computation
finishes. The H2D lands BEHIND the executing kernels on the plugin's
stream. The serialization is device-side, inside the ROCm PJRT
plugin — which means a dedicated loader thread would change nothing,
and upstream's free-overlap-via-enqueue-depth trick does not transfer
to this backend as-is. The candidate for real overlap, filed for the
perf pass as E-STREAM-2b: PJRT's CreateBuffersForAsyncHostToDevice
transfer-manager path (bindings already present in ZML's pjrt.zig),
which exists precisely to give H2D its own stream.

Decision, per the pre-registered rule: serial pinned uploads cost
48 × 31 ms ≈ 1.5 s per denoising step against today's ~60 s/step
compute — 2.5%, invisible. The loader proceeds with SERIAL pinned
uploads and the ring state machine unchanged; overlap is a perf-pass
item with a named API, not a Phase 3 blocker. (If attention gets its
hoped-for order of magnitude someday, 1.5 s/step becomes material —
that is exactly when E-STREAM-2b runs.)

Two free findings. First: the 128-long dependent GEMM chain sustained
**~77 TFLOP/s** (17.6 TFLOP in 229 ms) — comfortably above E1's 59
for a single dispatch. Chained same-shape GEMMs amortize launch and
autotune overheads; good news for the DiT's dense stretches. (The
chain's scalar sum underflows to zero in f16 at these magnitudes —
expected, and irrelevant: the timing, not the value, is the
evidence.) Second, a methodology entry for the growing list: GPU A/B
timing must INTERLEAVE its conditions. This is the third time the
measurement itself was the bug — readback poison in runs 3–5,
cross-executable autotuning in the E3w agreement gate, and now clock
ramp. The instruments keep being the experiment.

E-STREAM-1 and -2 are answered. Remaining before assembly:
E-STREAM-3 — the ring itself (three device slots, two pinned host
slots, mmap reader, lifecycle asserts, startup static-cycle check)
plus the one-blob-per-block pack step in tooling.

## 2026-08-21/22, overnight — E-STREAM-3 pre-registered: the acceptance contract for the first engine component

Adam's agent reviewed the E-STREAM results and proposed an acceptance
contract for the ring. Most of it is adopted verbatim — lifecycle
states asserted on every transition, the static schedule check before
any allocation, the packer as an offline reproducible transformation
with canonical ordering and digests, and the staged three-block dry
run before any 48-block ambition. Four corrections before adoption,
logged per house rules:

One: the dry run uses blocks 0, 23, and 47 — the blocks actually
fetched, quantized (block 0), and oracle-anchored by the chain bundle
— not "blocks 0, 1, 2". Two: the packer's input is the revision-pinned
per-block directories from tools/fetch_block.py (range requests
against the pinned checkpoint revision; this machine never holds the
full checkpoint), not a local safetensors file. Three: digest cadence
— hashing 20 GB on every step would cost seconds per step; digests are
verified at pack time and once per blob on FIRST touch by the reader
thread (amortized into step one's traffic), after which the immutable
mapping is trusted. Four: the review's bracketed citations point at an
unrelated ROCm/JAX compatibility page; the numbers it quotes are from
our own run logs, which is where evidence lives here.

One design decision the contract did not cover, decided now: upstream
carves per-tensor views out of one raw device buffer, but PJRT
executables consume separate typed buffers, and the view-carving
equivalent (PJRT_Client_CreateViewOfDeviceBuffer) is unproven on this
plugin. So the ring uploads PER TENSOR from the pinned blob slot —
~44 transfers per block instead of one — and the single-DMA carve
upgrade is deferred to the perf pass alongside the async transfer
manager. The blob layout is contract-fixed now so that upgrade needs
no repack.

Structure being built. tools/pack_block.py: block dir in, one
contiguous 64-byte-aligned blob plus manifest out (canonical
name-sorted order, per-entry offset/nbytes/dtype/shape/sha256, blob
sha256, source revision, packer version; no timestamps — the output
is deterministic to the byte). ltx/block.zig: the Block/Chain graph
definitions REFACTORED OUT of block_conformance.zig into the shared
module the engine will import — gated by the full 22-gate conformance
suite reproducing exactly after the move. ltx/loader.zig: the ring —
two pinned host slots (dmaMap), one mmap reader thread, device sets
uploaded serially per E-STREAM-2's verdict, slot states {free,
filling, ready, in_use} asserted on every transition, the 48-entry
schedule statically checked against ring capacities at startup before
any I/O. ltx/ring_dryrun.zig: the E-STREAM-3 binary.

Gates, pre-registered:

G-RING-1 packer determinism: packing the same block twice yields
identical blob digests. G-RING-2 manifest integrity: no gaps, no
overlaps, canonical order, sizes exact, every entry accounted for.
G-RING-3 byte fidelity: every tensor served by the ring compares
equal (memcmp) to a direct read of the source files, all three
blocks. G-RING-4 lifecycle: an intentionally SLOW consumer forces
ring-full backpressure (the reader must park, never overwrite) and a
fast consumer forces ring-empty (the consumer parks); both runs
complete with every transition legal and zero steady-state
allocation. G-RING-5: the static schedule check runs and passes
before the first byte of I/O. G-RING-6 the engine gate: chainB47
computed from ring-loaded buffers equals the same graph on
direct-loaded buffers BITWISE (same bytes in, same executable, so
any difference is a loader bug); then block 0's fully-quantized
blockOut, same comparison, through the quantized blob entries.

H-RING-1: byte-exact everywhere — the loader only moves bytes, so
G-RING-3 and G-RING-6 admit zero tolerance. H-RING-2: per-tensor
pinned upload lands within 20% of the whole-blob 31 ms (transfer
setup is per-call cheap at ~14 MB mean tensor size). H-RING-3:
dmaMap on the FILE-BACKED mmap region fails (host registration of
file-backed pages is typically unsupported), and the memcpy into the
anonymous pinned ring — the design-note staging — is the path;
measured either way by trying it first.

## 2026-08-21, continued — E-STREAM-3: all gates pass; pre-assembly is COMPLETE

Build receipts first. tools/pack_block.py packed blocks 0/23/47 into
blobs (44/28/28 entries — block 0 carries its int8-g128 companions);
repacking block 0 reproduced digest c0d8351050b949fa… exactly
(G-RING-1). The Block/Chain/QBlock definitions moved out of the
harness into the shared ltx/block.zig the engine will import, and the
full 22-gate conformance suite reproduced every number to the digit
after the move (blockOut 1.843e-6, chain 2.735e-6, quantized block
5.237e-3) — the regression fence held. Three bugs between first build
and green, each caught in seconds by the gates: a JSON parser default
that left manifest strings pointing into a freed buffer (segfault on
the first print; fix: parse with alloc_always), a slot stride that
wasn't page-aligned (alignment panic on slot 1; fix: align-forward
stride), and nothing else — the lifecycle and the GPU gates ran clean
on the first complete attempt.

The gate sheet: G-RING-2 layout OK for all three blobs, revision
6c7e5e573ac1 bound and checked. G-RING-5 static check before any I/O:
6-entry schedule, peak occupancy 2 of 2 slots. G-RING-4 slow
consumer: the reader parked 4 times on a full ring and never
overwrote; fast consumer: the consumer parked 6 times on an empty
ring and the reader never parked; both completed, every transition
legal. G-RING-3: every served entry matched its packed digest on
both passes. G-RING-6, the engine gate: **chainB47 fed by ring-loaded
buffers is BITWISE EQUAL to the direct-loaded control, and so is the
fully-quantized qBlockOutAll through the quantized blob entries.**
The ring is invisible to the mathematics, which is the whole job.

Now the findings, one per hypothesis, two of three refuted:

H-RING-3 REFUTED, pleasantly: dmaMap on the file-backed mmap
SUCCEEDED on this plugin. The pinned host ring's memcpy stage may be
removable entirely — pin the mapping, upload straight from the page
cache, zero host copies. Filed for the perf pass rather than adopted
now (the interaction between pinned file pages and 20 GB of cache
eviction under 15 GB of RAM needs its own measurement — pinning
fights the very page reclaim the overflow design relies on, so this
is NOT obviously free at full scale; at three blocks it would be).

H-RING-2 REFUTED: per-tensor uploads reached 4.24 GB/s against the
whole-blob 14.4 — a 3.4× penalty, not the predicted <20%. The cause
is visible in the harness: each fromBytes waits its transfer to
completion (wait=true), so ~100 transfers pay ~100 full round-trip
latencies. Named fix, filed for assembly: issue the block's uploads
wait=false and await them together — the DMAs pipeline and most of
the penalty should vanish. Even unfixed: 146 ms per block, ~7 s per
48-block step against ~60 s of compute. Not blocking; not ignored.

And an unregistered cost surprise: first-touch digest verification of
1.8 GB took 48.6 seconds — ~38 MB/s, which is SOFTWARE SHA-256 on a
baseline-x86_64 build (the hermetic toolchain doesn't emit SHA-NI).
At engine scale that would be ~9 minutes over 20 GB. Decision:
pack-time digests stay mandatory, runtime first-touch verification
becomes opt-in, and a SIMD-friendly hash (BLAKE3) or a -Dcpu bump is
filed for when runtime verification is wanted. The memcpy stage
itself is healthy: ~12 GB/s into pinned slots.

**Phase 3 pre-assembly is COMPLETE.** E3w conformant in the block,
scheduler bit-exact, quantization recipe settled, transfer facts
measured, ring gated end-to-end with the real graphs. What remains is
the 48-block assembly itself: fetch and quantize the other 45 blocks,
pack 48 blobs, stage-walk early/middle/late, then the full stream —
every component of which now has receipts.

## 2026-08-21, continued — pre-assembly audit: the map before the march

Adam asked for an accurate mapping of everything before committing the
machine to the 45-block fetch. The audit found two real defects and
several facts worth having in writing.

The system. 537 GB free on the 952 GB NVMe; the full assembly
footprint (source bins 24 GB, int8 companions ~10 GB, dual-flavor
blobs ~36 GB, existing state) lands around 75 GB — ample. RAM 15 GB
with 9.3 available. GPU idle at 1%, no LAMMPS running, git clean at
bacfc0b. The Bazel cache holds 95 GB — deliberate; cleaning it buys
back disk at the price of a full hermetic rebuild, so it stays.

The source. HF main still points at our pinned revision
6c7e5e573ac1 — no upstream drift — and the token is live. The
safetensors header confirms 48 transformer blocks with video-stream
bytes EXACTLY uniform at 537,673,856 per block: the ring's
uniform-geometry premise is now checkpoint-verified for all 48 blocks,
not extrapolated from three. Remaining download: 45 × 537.7 MB =
24.2 GB. The file also carries 4.89 GB of non-block tensors —
embedders, prompt connectors, adaln heads, final projections — which
this fetch deliberately excludes and the END-TO-END run will need;
they are now a named later step, not a surprise.

Defect one: fetch_block.py pinned whatever `main` resolved to AT FETCH
TIME. Fine for the first fetch; silently wrong for an assembly fetch
if upstream moved main in between — 45 blocks from a different
revision than the three every oracle is anchored to. Fixed: --revision
defaults to the pinned sha; re-resolving main is now an explicit act.

Defect two, a wording correction against receipts: the quantization
entries say "int8-g128 for every large projection," but the ladder's
receipts cover qkv (both attentions) and the FFN pair — the OUTPUT
projections were never run through a rung and remain bf16 in the
measured 5.24e-3 configuration. The recipe as receipted: qkv + ff
int8-g128, everything else bf16/f32. Cost of the honest version:
64 MB per block, ~3 GB across the model — priced, accepted, and an
out-class rung stays available as a future size play.

Also mapped, filed rather than fixed: the harness QBlock uploads the
UNUSED bf16 qkv/ff base weights alongside the quantized pairs
(~770 MB per block moved where ~360 would do) — a lean deployment
struct is an assembly-time item; the blobs stay dual-flavor ON
PURPOSE so any of the 48 blocks can run the f32 conformance
stage-walk. And a small pleasing fact: measured blob padding is zero
— every tensor's byte size is already a multiple of 64, so blob bytes
equal source bytes exactly.

The pipeline: tools/assemble_blocks.py drives fetch → quantize (qkv,
ff, g128 int8) → pack → digest-verify per block, resumably — a block
counts as done only if its blob manifest records the pinned revision
and its blob matches the recorded digest, so interruptions cost
nothing. Skip-if-done verified against blocks 0 and 23 before launch.
Launched for blocks 0–47; the three existing blocks skip, 45 fetch.

## 2026-08-21, evening — 48 blocks on disk; the assembly checkpoint, pre-registered

The fetch pipeline finished clean: 48 blob manifests, every one
digest-verified against the pinned revision, ~45 s per block
end-to-end, .work at 71 GB against the projected 75. The resumability
clause earned its keep involuntarily — an early launch was killed ten
minutes in by my own process-management mistake and the rerun picked
up its 12 completed blocks without re-downloading a byte.

Now the checkpoint this was all for: THE 48-BLOCK STAGE-WALK. The
design decision first, because it defines what "the engine" means
here. The assembly does NOT trace 48 blocks into one graph — that way
lies the comptime quota blowup, a 20 GB-resident executable, and none
of the streaming machinery earning anything. Instead the engine
compiles ONE block executable (the uniform geometry pays again: same
shapes, same graph, all 48 blocks) and calls it 48 times, the ring
feeding each call's weight buffers, the activation x staying on
device between calls. That is the production execution model, run for
the first time.

The oracle side: tools/make_full_chain_bundle.py extends the chain
oracle to all 48 blocks — torch CPU f64, one block loaded at a time,
same frozen inputs as everything since Phase 2, dumping the running x
after blocks 7, 15, 23, 31, 39, and 47 (the stage-walk checkpoints:
early, four interior stations, and the end). Two controls are free
and mandatory: block 0's output must equal the Phase 2 bundle's
block_out.bin EXACTLY (same code path, same inputs), and the
0→23→47 subsequence cannot be checked directly (the full chain runs
blocks BETWEEN them) — so instead the generator re-asserts the
existing 3-chain bundle stays reproducible before writing anything.

Gates, pre-registered before the oracle runs:

G-ASM-1, control: engine x after block 0 vs the Phase 2 oracle
block_out — the standard 2e-3 gate, expected at the known 1.8e-6.
G-ASM-2 through G-ASM-7: engine x vs oracle at the six checkpoints,
2e-3 gates each. Expected error: sub-linear growth has held from one
block (1.84e-6) to three (2.74e-6); if it continues ~√depth the
48-block end lands near 1e-5, two hundred times inside the budget.
Registered decision rule: if any interior checkpoint jumps an order
of magnitude over its predecessor, stop and bisect blocks — the gate
placement exists precisely to localize a bad block to one sixth of
the model.

H-ASM-1: all seven gates pass with the end-state near 1e-5.
H-ASM-2, measured not gated: the fully-quantized engine walk (int8
recipe per block, same streaming) vs the f32 engine walk. Single
block measured 5.24e-3 ≈ the bf16 floor; how per-block quantization
noise COMPOUNDS over 48 blocks is the number that decides whether
the recipe survives assembly — pre-registered expectation: sub-linear
again, final rel-RMS in the low tenths at worst (√48 × 5.24e-3 ≈
3.6e-2 if it random-walks; linear 48× ≈ 2.5e-1 would be trouble).
No pass/fail line on H-ASM-2 — it calibrates the Phase 6 fidelity
budget instead.

Timing rides along for free: per-block upload and compute, end-to-end
wall time for a 48-block pass at T=64, and the ring's stall count —
the first real numbers for the streaming engine's overhead model.

## 2026-08-21, night — the 48-block stage-walk PASSES, and the model forgives quantization

The verdict first: **48-BLOCK STAGE-WALK: ALL ORACLE GATES PASS.**
The production execution model — one compiled block executable, 48
calls, ring-streamed weights, activation resident on device — matches
the torch f64 oracle at every checkpoint. The full engine now exists
in miniature.

H-ASM-1, confirmed almost exactly as registered. The f32 trajectory:
9.88e-6 at block 7, 1.82e-5 at 15, 1.77e-5 at 23, 1.48e-5 at 31,
1.42e-5 at 39, **1.48e-5 at 47** — against a predicted ~1e-5 endpoint
and a 2e-3 budget, so 135× inside. The shape is the story: error
grows for the first third and then PLATEAUS — the per-block noise
random-walks and partially cancels rather than accumulating. Rerun
reproduced every gate digit-for-digit.

H-ASM-2, answered better than either registered scenario. The
fully-quantized walk: 3.51e-2 at block 7, peaks at 3.89e-2 at block
15, then DECLINES monotonically — 3.81e-2, 3.20e-2, 3.04e-2,
**2.84e-2 at block 47.** Below the √48 random-walk estimate (3.6e-2),
an order of magnitude under the 2.5e-1 trouble line, and the decline
means the network actively attenuates accumulated quantization noise
in its second half — the single-block softmax-attenuation finding,
operating at model scale. The int8-g128 recipe survives assembly at
~2.8e-2 rel-RMS end-to-end; that number now calibrates Phase 6's
fidelity budget. (Quant-vs-f32-engine and quant-vs-oracle agree to
four digits at every checkpoint — as they must, since the f32 engine
sits 1e-5 from the oracle.)

Timing, the first real overhead model: 54.8 s wall for a 48-block f32
pass, of which uploads are 2.2 s and compute-plus-sync 1.0 s (T=64).
The other ~50 s is DISK-TO-STAGING: the reader never parked while the
consumer parked 48 times, and 26 GB through cold-cache mmap page
faults runs at ~430 MB/s — nowhere near NVMe streaming rates. The
kernel is not readahead-ing this access pattern; madvise(SEQUENTIAL)
or pread-based staging is the named lever, filed for the perf pass.
At production T≈28k the compute term grows a thousandfold while the
staging term stays fixed, so this is a first-step tax, not a
steady-state one — but it is also why time-to-first-latent will care.

Two bugs on the road, both instructive. First, run 1 stalled for an
hour and the diagnosis (gdb, thread states) found MY teardown
deadlocked: an error mid-walk propagated through `defer reader.join()`
while the reader was parked on a condition nobody would signal. The
ring gained abort() — consumer-side teardown that retires the reader
before the join — and the error path is now proven (run 2's failure
exited cleanly in seconds). Second, the error itself: blobs 23 and 47
PREDATED the assembly pipeline — packed during the ring dry run,
before their blocks were ever quantized — and the pipeline's resume
logic correctly honored its own contract (revision + digest valid)
and skipped them. The quant walk then reached for int8 entries that
did not exist. The contract was incomplete, not the code: done() now
also requires the full 44-entry schema. And a free gate from the fix:
block 23's repack reordered the blob layout, and the f32 walk
reproduced bit-identically through it — the loader is provably
layout-independent, serving by name and offset.

Also relearned, for the third time, in a new costume: never filter a
long-running process through a block-buffered pipe. Run 1's actual
error sat in grep's buffer for an hour while I diagnosed a "hang."

**The Phase 3 assembly checkpoint is closed.** What remains for the
phase's "done means" — latents matching the reference step-by-step
through a full denoising pass on the card — is composition of
now-gated parts plus the named remainder: the non-block tensors
(patchify/adaln/final projections, 4.89 GB, mapped in the audit), the
scheduler loop driving 48-block passes at production T, and the E3w
kernel doing its production job at that length.

## 2026-08-21, late night — non-block inventory, and a CPU false alarm

Adam flagged CPU spikes; checked before proceeding. Verdict: not the
experiment — every pipeline process is drained. The culprits are a
transient Flatpak update transaction (revokefs-fuse pair) and
warp-taskbar sitting at a steady 36% CPU for sixteen hours (5h45m
cumulative) — a busy-loop bug in the terminal app, cleared by
restarting Warp, Adam's call. Logged because "the machine is busy"
must never be conflated with "the experiment is busy."

Then the non-block tensor map, from the pinned header, and it splits
better than the audit's single 4.89 GB number suggested. The
VIDEO-ONLY denoising pass needs just ~450 MB: adaln_single (302 MB —
the sinusoidal-embed → MLP → 4096→36864 linear that PRODUCES the
9-chunk per-token timestep tensor our blocks have consumed as a
frozen input all along), prompt_adaln_single (103 MB — producing our
pts2), patchify_proj (128→4096 in), proj_out (4096→128 out), the
final scale_shift_table [2,4096], and the keyframes embedding. The
other ~3.4 GB is the video_embeddings_connector — the 8-block,
128-register prompt transformer, which is Phase 4's once-per-prompt
path — plus the A/V-bridge adaln (170 MB, Phase 7). Audio-side
non-block tensors (1.05 GB) stay excluded.

Fetched ALL of the video side now (153 tensors, 3.84 GB, same pinned
revision, fetch_block.py --extras) so no later phase has a fetch
step hiding in it. Next build, pre-registration to come in its own
entry: the E2E-core parts get small oracles of their own (they are
plain linears and norms — cheap to gate the Phase 2 way), then the
composition: patchify → adaln-driven 48-block streamed passes under
the bit-exact scheduler → proj_out, first at harness T, then at
production length where E3w does its real job.

## 2026-08-21, late morning — the extras claim regressed, then came true

Correction to the record: the previous entry's "fetched ALL of the
video side now" was written while the fetch was still running, and it
then briefly became false. Warp (the terminal app whose busy-loop we
had just diagnosed) crashed and took the whole desktop session with
it, killing the detached fetch at 62/153 tensors. Adam fixed Warp;
the fetch was relaunched and this time COMPLETED: 153/153 tensors,
3.84 GB in .work/extras, manifest.json written, pinned revision
6c7e5e57 throughout. Verified directly from the manifest, not the
monitor — because the monitor never fired: its pgrep pattern matched
its own shell's command line, so it saw the "process" alive forever.
Same self-match footgun as the pkill incident, opposite polarity.
Watchers must match the watched process, never their own reflection.

One checkpoint observation from the completed manifest, recorded
before anyone needs it: scale_shift_table is the single f32 tensor on
the checkpoint's video side — everything else, weights and biases
alike, ships bf16. The final modulation table was deemed too
precision-sensitive to round. Noted for the loader (dtype is per-entry
anyway) and for anyone tempted to assume "the checkpoint is bf16."

## 2026-08-21, late morning — E2E-core read: what the reference actually does between latent and latent

Read receipts before pre-registering the oracles, from
ltx_core/model/transformer/{adaln.py, timestep_embedding.py,
model.py, transformer_args.py} at the installed reference versions,
because "plain linears and norms" is exactly the kind of claim this
notebook exists to check. Mostly confirmed, with four facts worth
their own lines.

The input side (TransformerArgsPreprocessor.prepare, video-only
path): x = patchify_proj(latent) — a plain biased Linear 128→4096.
Then the keyframes absolute-position embedding is applied ONLY when a
keyframes_mask is present; our harness passes none, so it is a
structural no-op (and the checkpoint tensor is there for when Phase
7+ wants it). There is NO caption_projection in this checkpoint — the
24 non-connector video tensors are fully accounted for without one,
so prompt context arrives already at model width from the connector.

The timestep path: timestep × timestep_scale_multiplier (1000 for
this model, config default, same as PixArt lineage), flattened —
per-token conditioning rides through here, [B,T] becoming B·T
independent sinusoidal embeddings. The sinusoidal is the DDPM/PixArt
256-dim embedding with flip_sin_to_cos=True (cos half first),
downscale_freq_shift=0 (exponent divisor is half_dim, 128, not 127).
FACT ONE: get_timestep_embedding computes in f32 unconditionally —
torch.arange(dtype=f32), .float() on the timesteps — and only THEN
casts to hidden_dtype. Even an f64 oracle carries f32-rounded
sinusoids. This is the RoPE mid-pipeline f32 rounding discovery
wearing a different hat, and this time we saw it coming by reading
first. The MLP after it: linear_1 256→4096, SiLU, linear_2
4096→4096 = embedded_timestep; then adaln's OWN SiLU and the big
linear 4096→36864 = the 9-chunk per-token ts our blocks consume
(9 = 6 base + 3 cross-attn params, matching adaln.py's constants).
FACT TWO: prompt_adaln_single (embedding_coefficient=2 → 8192 = our
pts2) is driven by modality.sigma — the SCALAR sigma, not the
per-token timesteps — through the SAME ×1000 multiplier and the same
_prepare_timestep helper. The prompt-side modulation never sees the
denoise mask.

The output side (_process_output): scale_shift_values =
scale_shift_table[None,None] + embedded_timestep[:,:,None] — the
[2,4096] f32 table broadcast against the 4096-dim embedded_timestep,
giving per-token shift (index 0) and scale (index 1). FACT THREE:
norm_out is torch.nn.LayerNorm(4096, elementwise_affine=False,
eps=1e-6) — a TRUE mean-subtracting LayerNorm, not the RMSNorm the
blocks use everywhere. An engine that reaches for the existing rmsNoW
helper out of habit would be wrong in a way the gates must catch.
Then x·(1+scale)+shift, then proj_out 4096→128, biased. FACT FOUR:
embedded_timestep enters this tail RAW — the adaln's SiLU sits only
in front of the 36864 linear, not on the embedded_timestep the tail
consumes. The same [B,T,4096] tensor conditions both ends of the
model, un-activated.

Config values pinned from model_configurator defaults (this
checkpoint's config overrides none of them): norm_eps 1e-6,
timestep_scale_multiplier 1000. The av_ca adaln singles feed only the
A/V bridge (MultiModalTransformerArgsPreprocessor) — skipped in our
video-only Phase 3 scope exactly as the roadmap's audio-stub note
says.

### Pre-registration: E2E-core oracles (the Phase 2 treatment, smaller)

The build: tools/make_core_bundle.py runs the reference modules in
f64 (weights upcast from the extras bf16 bins; sinusoid f32 by the
reference's own construction) on seeded inputs — latent [1,64,128], a
per-token timestep vector built as denoise_mask·σ with a mixed mask
and σ=0.909375 from the distilled table, scalar sigma for the prompt
side — and dumps staged outputs: sinusoid, embedded_timestep, ts
9-chunk, pts2, patchify out, and the output tail applied to a seeded
x. Engine side: ltx/core_parts.zig (sinusoid built in-graph from
iota/exp/sin/cos, the two MLPs, patchify, LayerNorm tail) gated per
stage in ltx/core_conformance.zig.

H-CORE-1: patchify_proj and proj_out (plain biased GEMMs) conform at
~1e-6 rel-RMS, the Phase 2 per-stage level. H-CORE-2: the sinusoid
gate is the interesting one — both sides compute f32, but XLA's
sin/cos and libm's need not round identically; per the transcendental
budget precedent (RoPE straggler protocol) I expect agreement at
~1e-7 rel-RMS with possible few-ulp stragglers, and pre-commit to the
ulp protocol rather than loosening the budget if a handful appear.
Arguments are bounded by ×1000 scaling times max σ=1.0, so no
range-reduction cliff. H-CORE-3: embedded_timestep and the ts 9-chunk
conform at ~1e-6 (two GEMMs deep, one shared executable, so no
cross-exe autotune tax within a gate). H-CORE-4: the output tail
conforms at ~1e-6, and specifically a deliberate wrong-norm control
(RMSNorm in place of LayerNorm) must FAIL the gate by orders of
magnitude — if it doesn't, the gate has no teeth for fact three.
H-CORE-5, the composition target for the NEXT entry once these pass:
patchify → adaln-driven 48 streamed blocks → tail at harness T=64
against a full-forward f64 oracle, budget 2e-3 with expectation
~1.5e-5 carried from the stage-walk.

### E2E-core results: all gates pass; the sinusoid's stragglers are argument-ulps

Built tools/make_core_bundle.py (f64 reference through the REAL code
paths — TransformerArgsPreprocessor._prepare_timestep for the x1000
flatten/reshape, LTXModel._process_output called unbound for the tail,
since self is unused there — staging asserted against the helper at
1e-12), ltx/core_parts.zig (17 tensors, sinusoid in-graph from
arange/exp/sin/cos), and //ltx:core_conformance. Eight gates, one run,
all PASS:

sinu 5.720e-6 rel-RMS (max-abs 5.302e-5) · emb 7.846e-7 · ts9
8.671e-7 · pts2 1.289e-6 · patchify 2.029e-7 · tail 3.073e-7 ·
tailWrong-vs-oracle-control 3.006e-7 · tailWrong-vs-truth 3.912e-3
(must exceed 1e-3: teeth PASS).

H-CORE-1/3/4 land where pre-registered (~1e-6). H-CORE-2's "~1e-7"
was optimistic and the miss is worth its arithmetic: the sinusoid's
frequency vector is exp() of a small table, and one f32 ulp of a
frequency near 1.0 (~6e-8) multiplied by the largest argument
(0.909375 x 1000 = 909.375) is ~5.5e-5 of phase — sin then moves by
up to that much. Measured max-abs-diff: 5.302e-5. The stragglers are
argument-ulp amplification, not a sloppy transcendental: every one
sits within ONE ulp-of-the-argument bound, and the multiply path
itself (x1000 in f32 vs the oracle's f64-then-round) is provably
identical because t_f32 x 1000 fits f64 exactly. The next gate is the
proof no fallback is needed: embedded_timestep, one MLP downstream of
those stragglers, conforms at 7.8e-7 — the GEMM averages 256 O(1)
inputs and the few amplified entries vanish into it. The
freq-table-as-data fallback (RoPE-style contract) stays unfiled.

Two smaller receipts. The wrong-norm control pair: engine RMS-tail
matches the ORACLE's RMS-tail dump at 3.0e-7 while both differ from
the true LayerNorm tail by 3.9e-3 — engine and oracle agree on the
exact SIZE of the wrongness, which is the strongest form of gate
teeth. The 3.9e-3 is modest because seeded random x is near
zero-mean (LayerNorm ~ RMSNorm there); real block-47 output may
separate them harder, and the composition gate will see that for
free. And the conditioning tokens (t=0) cost nothing: their sinusoid
is exactly [1]x128,[0]x128 on both sides.

Every non-block part of the video-only pass is now individually
conformant. Next rung, H-CORE-5 as pre-registered: the composition —
patchify -> adaln-driven 48 streamed blocks -> tail at harness T=64
against a full-forward f64 oracle (extending the walk bundle), budget
2e-3, expectation ~1.5e-5 carried from the stage-walk.

### H-CORE-5: the composed video-only forward PASSES at harness T

Built tools/make_e2e_bundle.py (f64 oracle: patchify → adaln-driven
48-block chain → tail, blocks loaded one at a time as in the walk
bundle) and //ltx:e2e_walk (the engine composition: 5 executables —
patchify, ts9r, pts2r, tail, ONE block called 48x — with every
activation and conditioning tensor produced and consumed ON DEVICE,
weights ring-streamed). The oracle's inputs are the core bundle's, so
its first act is to reproduce the core dumps byte-for-byte (x0, ts9,
pts2, tail — all four controls held), binding the two bundles.

Engine vs oracle, budget 2e-3: x0 2.029e-7 · b23 2.376e-6 · b47
5.035e-5 · final velocity 8.394e-5. ALL PASS. 69 s wall for the
48-block walk at T=64, staging-dominated as expected.

The expectation (~1.5e-5, carried from the stage-walk) was missed by
5.6x, and the miss has a mechanism worth recording: the stage-walk
ran on SEEDED RANDOM timestep chunks (rms ~0.1), under which error
plateaued; the real adaln conditioning produces ts9 with rms ~1 and
drives the residual stream to max_abs 4030 / rms 59 by block 47 (the
oracle's own dumps; the harness walk stayed far smaller). Under real
conditioning the error GROWS with depth — 2.4e-6 at b23 to 5.0e-5 at
b47 — rather than plateauing. Same engine, same blocks, same budget
margin (24x under), but "error plateaus with depth" is now known to
be a property of the RANDOM-conditioning regime, not of the network.
Filed as the number to watch when production-length runs land: if
growth continues past 48 calls x multiple denoising steps, the 2e-3
budget is the fence that matters, and the scheduler's x0-space
round-trip renormalizes between steps anyway.

Also recorded: the tail is where the magnitude story resolves — rms
59 at b47 collapses to a velocity of rms 0.63 through the LayerNorm,
which is why the tail's norm KIND mattered enough to deserve its
teeth gate.

Phase 3 state after this entry: every part and the composition are
conformant at harness T. Remaining for done-means: the same
composition at production length (E3w blockwise attention, T≈28k,
where dense scores cannot exist), and the bit-exact scheduler
driving multi-step denoising with per-step latents gated against the
reference loop — then time-per-step goes in this notebook.

## 2026-08-21, afternoon — pre-registration: the scheduler-driven multi-step loop at harness T

The other half of Phase 3's done-means: "latents match the reference
step-by-step." The build: tools/make_e2e_denoise_bundle.py runs the
REFERENCE stage-1 loop (euler_denoising_loop / EulerDiffusionStep /
GaussianNoiser / timesteps_from_mask / post_process_latent, imported
through the same package shim as the scheduler bundle) with the REAL
f64 model as denoise_fn — patchify → adaln → 48 blocks → tail →
to_denoised — at T=64, 8 steps over the full distilled stage-1 sigma
ladder. Audio marches with zero velocities (video-only scope, audio
values never touch the video state). The scheduler bundle's mask
convention is reused deliberately: ones, tokens 48:56 at 0.0, 56:64
at 0.5 — the 0.5 region makes THREE distinct per-token sigmas per
step, so nothing in the engine can get away with special-casing the
binary mask. Per-step dumps: t vector, velocity, denoised, post,
next latent.

New ground this rung covers, called out before measuring: (a) pts2 is
sigma-driven, so it CHANGES every step — the single-forward harness
ran it once at a fixed sigma; (b) the ring makes 384 acquisitions
(48 blocks x 8 steps) through a repeating schedule, its first
non-identity schedule since the dry run; (c) the host scheduler
(bit-exact 94/94) and the device engine compose for the first time —
latent marches host-side through lerp/toDenoised/postProcess/
eulerStep, velocities come from the card.

Scope: STAGE 1 ONLY. Stage 2's scheduler mechanics are already
bit-exact in scheduler_conformance and its model side is identical;
a second 2-hour f64 oracle run buys no new information.

H-LOOP-1: each step's velocity conforms at ≤2e-3, expected ~1e-4
(the single-pass composition measured 8.4e-5; conditioning changes
per step but stays in the same regime). H-LOOP-2: the latent
trajectory compounds velocity error; expected final-latent rel-RMS
1e-4..1e-3 against a 2e-3 budget — if compounding is worse than
linear in steps, that is a finding, not a fence failure, unless the
budget breaks. CONTROL (bitwise): the engine recomputes the noised
initial latent from the dumped clean/noise/mask through
ltx/scheduler.zig's lerp chain and it must equal the oracle's
s1_x0 EXACTLY — the fused-lerp discovery standing guard at the
integration seam. Oracle-side control: the noise twin-capture
assert, as in the scheduler bundle.

## 2026-08-21, afternoon — T-parametrization paid off, and the production-length pre-registration

While the multi-step oracle walks its 8 f64 forwards, the standing
debt in block.zig's header ("assembly parametrizes T later") came due:
every use of the comptime T/S consts inside traced code is now derived
from the input tensors' dims at trace time — whileSdpa's chunk count
from k's .k dim, every attention reshape from the tensor being
reshaped, cross-attention K/V length from the modulated context. The
consts remain as the conformance harness's default geometry only. The
22-gate suite is the fence, and it reports NUMBERS IDENTICAL to the
historical record (blockOut 1.843e-6, e3w-attn1 1.094e-6,
quant-qBlockOutAll 5.237e-3, RoPE 0 stragglers): the refactor is
provably invisible at T=64.

### Pre-registration: the production-length pass (fits, runs, timed)

What Phase 3's done-means asks at production length is different in
kind from the harness gates: "a full denoising pass FITS and RUNS on
the card with time-per-step recorded." A step-by-step f64 oracle at
T=28,672 is off the table by arithmetic — one block costs ~448x the
harness forward, ~75 min on this CPU, 2.5 days for one 48-block pass
— so numerical authority at production length rests on the
already-gated pieces: the 22-gate suite (including E3w agreement and
oracle conformance), the composed harness-T pass, and the multi-step
loop. The production run's own checks are structural: no NaN/Inf in
the velocity, finite magnitudes logged.

The build: //ltx:e2e_prod — ONE velocity pass at T=28,672 (E3's
working length, 16-divisible for ACHUNK), wBlockOut (the E3w while
kernel — dense scores cannot exist at this length), ring-streamed
weights, conditioning from seeded inputs: random latent, a mask with
0.0/0.5/1.0 stripes, sigma 0.909375, synthetic unit-circle RoPE
tables (cos/sin of random angles — real grids are Phase 5 wiring;
timing and stability are angle-agnostic).

H-PROD-1 (fits): predicted device residency ~9-10 GB — ts9 f32
4.23 GB (the elephant: [28672, 9, 4096]), x + block-out ~0.94 GB,
q/k/v temps ~1.4 GB, FFN intermediate 1.88 GB, rope 0.47 GB, one
weight set 0.54 GB — inside 16 GB without the bf16-ts9 or
distinct-sigma-gather levers (both noted for Phase 6 if this
prediction is wrong). H-PROD-2 (time): per-block GEMMs ~13.5 TFLOP at
the measured 59-77 TFLOP/s plus E3w attention ~1.2 s (E3's number)
predicts ~1.4-1.7 s/block, ~70-85 s for 48 blocks; staging 25.8 GB at
~430 MB/s is ~60 s and for the FIRST time should hide entirely
behind compute — reader parks near zero is the tell. Core parts add
~1 s (the ts9 GEMM is 8.7 TFLOP). Predicted step wall: 75-120 s.

### H-LOOP results: the engine walks the reference trajectory, 8/8 steps

The oracle finished all 8 f64 forwards in ~35 minutes (~103 s per
forward — the 13-min-per-forward estimate from the e2e bundle's wall
clock was wrong because that wall clock was mostly BLOCK LOADING from
a cold page cache; with the blobs hot, the f64 compute itself is
fast). Engine run, //ltx:e2e_denoise:

CONTROL noised x0: BITWISE match. Every per-step host timesteps
vector: BITWISE match (no assert fired). Velocities: 1.006e-5,
1.517e-5, 3.452e-5, 2.648e-5, 1.180e-5, 1.600e-5, 1.784e-5, 2.200e-5
— all PASS, all better than the ~1e-4 expectation. Latents vs the
reference trajectory: 5.085e-8 → 9.110e-8 → 1.736e-7 → 2.084e-7 →
6.077e-7 → 3.309e-6 → 6.997e-6 → 1.004e-5 — all PASS, final drift
200x under the 2e-3 budget. ~68 s/step wall at T=64,
staging-dominated as at the single pass.

The error-attribution story is exactly what the design bought: with
the initial latent and every conditioning vector bitwise-identical
and the scheduler arithmetic bit-exact on both sides, ALL trajectory
drift is velocity-sourced. And the drift shape confirms it — early
steps have tiny dt (0.00625), so 1e-5 velocity error moves the latent
by ~1e-7; the late steps' large dt (up to 0.42) do the accumulating.
Compounding is benign: 8 steps of ~2e-5 velocity noise sum to 1e-5 of
latent, sub-linear in the pessimistic estimate.

**"Latents match the reference pipeline's within tolerance, step by
step" is CLOSED at harness T.** What remains of Phase 3's done-means
is exactly one clause: the pass at production length, fits + runs +
time-per-step. //ltx:e2e_prod is compiled and pre-registered; running
it next on the now-quiet machine.

### H-PROD results: it fits, it runs, and the time-per-step exposes a harness constant doing a production job

//ltx:e2e_prod at T=28,672: compile 51.0 s for the 5 executables.
H-PROD-1 (fits): PASS — no allocation failure, and the velocity is
finite and sane (nan/inf 0, rms 0.4335, max_abs 4.17). The run:
PRODUCTION PASS 1340.27 s wall, 27.88 s/block average, reader parks
46, consumer parks 0.

H-PROD-2's 1.4-1.7 s/block prediction missed by 16-20x, and the park
counters say where to look: the reader parked 46 times (staging
FINISHED early and waited on compute, every block), the consumer
never waited — the pass is thoroughly compute-bound, so the ~60 s of
staging per pass is fully hidden exactly as predicted, just behind
20x more compute than predicted. The suspect, anchored by receipts:
ACHUNK=16 was chosen so the T=64 HARNESS gets 4 while-loop
iterations ("a single chunk of T would degenerate into plain
softmax"); at T=28,672 the same constant means 1,792 iterations of
[H, T, 16]-thin GEMMs — the identical 13.5 TFLOP of attention
arithmetic pushed through latency-bound slivers. E3's 1.22 s
attention measurement — the number the prediction leaned on — was
taken at WCHUNK=1024 (28 iterations) in smoke.zig. The block
integration inherited the harness constant, and nothing between
T=64 and T=28,672 ever re-scaled it. (Also relearned, fourth
costume: `| head -30` on the run's monitor pipeline block-buffered
every line until process exit — head cannot flush. The run itself
was safe; the visibility wasn't.)

Pre-registered fix: whileSdpa becomes comptime-chunk-parametric with
trace-time selection — 1024 when the K length divides by it, 16
otherwise — so every harness graph (T=64) is BITWISE UNCHANGED (the
22-gate suite must reproduce its historical numbers) and production
gets E3's measured geometry. New gate, because chunk 1024 has never
run inside the full block: e2e_prod grows a pre-pass agreement check
at T=4096 — wBlockOut's attention path (auto-selecting 1024) vs the
dense twin (torch-anchored at the harness), block-0 weights, seeded
inputs, budget 1e-5 per the E3w agreement precedent. H-PROD-2b:
post-fix, attention returns to ~1.2-1.5 s and the block lands at
~3-4.5 s (f32 GEMMs at their own rate), putting the step at
~2.5-4 min. If the block stays slow after the chunk fix, the
hypothesis is wrong and the next suspect is the f32 GEMM rate
itself (the 59-77 TFLOP/s anchors are f16 numbers).

## 2026-08-22 — the chunk fix lands, then an OOM chase; a crash; the state as re-entered

(Continuity note: yesterday's session died with the desktop again —
second time a "detached" run went down with the ship; nohup+setsid
does not survive a session-manager teardown. Run 6 was the casualty,
killed mid-pass. Everything below up to run 6 happened before the
crash and is reconstructed from the run logs, which survived.)

The chunk fix: whileSdpa is now comptime-chunk-parametric with
trace-time selection (PCHUNK=1024 when the K length divides, else
ACHUNK=16), the 22-gate suite reproduced its historical numbers
bitwise (harness graphs provably untouched), and the new
chunk-agree@T4096 gate — full-block E3w at chunk 1024 vs the
torch-anchored dense twin on block-0 weights — passed at 8.383e-7.
Run 2 then measured H-PROD-2b's prediction almost exactly: blocks
dropped from 27.88 s to 1.1-3.0 s (block 0 always ~10 s of
first-touch warmup). The 1,792-sliver-GEMM diagnosis was right.

Then the pass OOMed at the very last call, and the chase is worth
its receipts. The tail executable's invocation, after all 48 blocks,
failed with a ResourceExhausted ask of 8.78 GiB — and that ask
stayed BYTE-IDENTICAL across three structurally different tail
designs: run 3 (ts9's 4.23 GB freed before the tail — same OOM),
run 4 (a probe call of the SAME tail exe at startup SUCCEEDED on the
young pool, the end-of-pass call still failed — so the demand is
satisfiable, the post-pass pool is not), run 5 (the tail rebuilt as
tailFromEmb, taking embedded_timestep as a 470 MB input instead of
recomputing it — a much smaller graph, gated at 3.052e-7 against the
same oracle dump in core_conformance — and the ask was STILL
8.78 GiB). An ask that ignores the graph it nominally serves is not
that graph's demand: the working theory is BFC pool mechanics — the
chunk-1024 while workspace (s and p are each [32, 28672, 1024] f32,
7.0 GiB the pair) leaves the 14.4 GB pool (0.90 fraction default,
preallocated) unable to seat one large contiguous region, and the
8.78 GiB constant (suspiciously ~20x the 470 MB [T,D] buffer) is
whatever the allocator reaches for at the first post-pass demand.
Unresolved; the number deserves a buffer-assignment dump when it
next blocks progress.

Run 6 (before the crash killed it): PCHUNK=512 to halve the while
workspace, pool fraction raised to 0.95. The gate passed at 8.050e-7
— and blocks ran at 5.9 s, FOUR TIMES slower than chunk 1024. Chunk
512 is a measured perf cliff, not a safe shrink; reverted to 1024.
The 0.95 fraction and the tailFromEmb split are kept.

Run 7, pre-registered: chunk 1024 + pool 0.95 + the small (x, e)
tail + ts9/pts2 freed pre-tail + an "invoking tail" log line pinning
the failure point. If the 8.78 GiB ask still fires, next lever is
diagnostic, not structural: dump the allocator's view (or bisect by
calling the tiny emb exe post-pass) before touching more code.
