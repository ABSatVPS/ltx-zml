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
