# ltx-zml — compiler-generated video generation on AMD (working title)

An experiment: adapt the [coli-zml](https://github.com/ABSatVPS/coli-zml)
approach — no hand-written GPU kernels; Zig traces StableHLO through
[ZML](https://github.com/zml/zml), XLA generates the machine code, PJRT
reaches the GPU — from sparse MoE LLM decode to dense video diffusion.
Target model: LTX-2.5. Hardware: Radeon RX 9060 XT (Navi 44, gfx1200,
RDNA 4), 16 GB VRAM, Fedora 44.

**Status: feasibility smoke complete (day one).** Measured on the target
card: XLA routes large f16 GEMMs to hipBLASLt and reaches ~59 TFLOP/s on
gfx1200 (matrix cores engaged; autotuning is worth 3×); int4-resident
weights with in-graph dequant feed those GEMMs at full dense speed, so
quantized residency costs nothing at prefill sizes; naive attention
materializes f32 scores and OOMs at T=16384 on 16 GB, making in-graph
blockwise attention (E3) the critical path; and naive device-to-host
readback runs at ~0.5 GB/s, so everything stays on device until the final
frames. The journal at [docs/lab-notebook.md](docs/lab-notebook.md) is the
source of truth — it logs the why behind each decision, with hypotheses
written down before the measurements, including the three that died on
day one.

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
