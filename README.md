# ltx-zml — compiler-generated video generation on AMD

An experiment: adapt the [coli-zml](https://github.com/ABSatVPS/coli-zml)
approach — no hand-written GPU kernels; Zig traces StableHLO through
[ZML](https://github.com/zml/zml), XLA generates the machine code, PJRT
reaches the GPU — from sparse MoE LLM decode to dense video diffusion.
Target model: LTX-2.5. Hardware: Radeon RX 9060 XT (Navi 44, gfx1200,
RDNA 4), 16 GB VRAM, Fedora 44.

**Status: the real model is running — one LTX-2.5 transformer block is
numerically conformant** against the upstream implementation on real
checkpoint weights (rel-RMS 1.85e-6 vs an f64 oracle, RoPE tables
bit-perfect), on top of all four feasibility experiments (E1–E4)
answered. Scope stated precisely: this is measured conformance for the
declared configuration — pinned upstream implementation, checkpoint
revision, oracle bundle, and runtime — not a general guarantee across
model revisions, compiler versions, hardware, or the quantized
48-block execution that Phase 3 must earn separately. Measured on
the target card: XLA routes large f16 GEMMs to hipBLASLt and reaches
~59 TFLOP/s on gfx1200 (matrix cores engaged; autotuning is worth 3×);
in-graph dequant feeds those GEMMs at full dense speed (the settled
quantization recipe is int8 grouped-128 — plain int4 at any granularity
fails on this model's outlier-heavy weights, which is why upstream
ships int8 too); and **blockwise online-softmax attention traced
through `stablehlo.while` runs the full 28k-token LTX working length on
16 GB** — bit-identical to its unrolled twin, beating naive sdpa where
naive fits at all, and now carrying conformance receipts inside the
block itself (agreement with dense attention at 1.1e-6, block and
three-block-chain oracle gates unchanged by the swap). A 0/23/47
three-block chain conforms with sub-linear error growth, and the fully
int8-quantized block sits at the reference's own bf16 deployment
floor. Also learned the hard way: naive device-to-host readback runs
at ~0.5 GB/s, so everything stays on device until the final frames.
Attention geometry is verified against LTX-2.5's own code (32 heads ×
128, 48 layers). Next: the distilled scheduler compared
update-by-update, then the streaming loader (~20 GB DiT through a
15 GB-RAM machine), then 48-block assembly. The journal at
[docs/lab-notebook.md](docs/lab-notebook.md) is
the source of truth — it logs the why behind each decision, with
hypotheses written down before the measurements, including the ones
that died. [ROADMAP.md](ROADMAP.md) tracks the phases ahead and what
"done" means for each.

## Layout

- `coli-zml/` — frozen snapshot of the predecessor project (upstream commit
  `dbf70c6`, git history detached). Reference only, not a dependency.
- `patches/` — carried forward from coli-zml: the ZML PJRT sub-byte transfer
  patch (u4/u2 device buffers), still required for quantized weight residency.
- `scripts/setup.sh` — clones ZML at a pinned revision, applies the patch,
  installs the `ltx/` package into the workspace.
- `ltx/` — the experiment package. Smoke benchmarks before engine code.
- `tools/` — build helpers (ROCm lockfile checksum refresh).
- `docs/lab-notebook.md` — the journal.

## Try it

```sh
./scripts/setup.sh
cd .work/zml
../bin/bazelisk run --@zml//platforms:rocm=true //ltx:smoke
```

First build fetches a hermetic ROCm userland via Bazel (~50 GB cache, ~10 min);
only the `amdgpu` kernel driver is used from the system.

## License

Apache 2.0, matching ZML and the coli-zml snapshot it descends from.
