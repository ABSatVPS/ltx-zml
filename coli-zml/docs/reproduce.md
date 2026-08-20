# Reproducing the results

Everything below assumes `./scripts/setup.sh` has run and `$WORK` is its
workdir (default `.work`). Commands run from `$WORK/zml`; `bazelisk` means
`$WORK/bin/bazelisk`.

```sh
cd .work/zml
BZL=../bin/bazelisk
ROCM="--@zml//platforms:rocm=true"
```

## 1. colibrì's GPU conformance suite

Its own test, unmodified except for renaming `main`, linked against this
backend instead of `backend_cuda.o`:

```sh
$BZL run $ROCM //coli:conformance
```

Expect the last line colibrì's suite prints on success:

```
cuda backend: q8/q4/q2/f32/e8 correctness ok on 1 device(s)
```

It covers q8/q4/q2/f32/E8 matmul, grouped int4 scales, upload error paths,
sticky-error recovery, `tensor_update`, exact tensor/byte accounting, the
fused expert MLP and expert group, the `issue`/`take` async pair (asserted
byte-identical to the synchronous path), MLA attention absorption, fault
injection via `COLI_GPU_FAIL_AFTER`, and leak checks.

## 2. Correctness vectors and benchmarks

```sh
$BZL run $ROCM //coli:smoke
```

Prints a ✅ per quantization format and expert path, then the benchmark tables
from the README: single expert MLP at rows 1/2/4/8 with a phase breakdown
(upload / launch / sync+download) and p50/p95/p99, the fused-vs-loop group
comparison, the fmt=4 grouped-scale variant, and a spaced-dispatch run that
checks the numbers aren't an artifact of back-to-back timing.

The **fused vs loop** line and the **fmt=4** line are regression tripwires for
the XLA fusion behaviour the backend depends on (see the README). If the fused
number drifts toward the loop number, the compiler stopped fusing.

## 3. The HIP baseline, for comparison

Needs ROCm's `hipcc` plus `rocwmma-devel` on a WMMA-capable GPU:

```sh
cd ../colibri/c
make hip-test  HIP=1 ROCM_HOME=/usr HIP_ARCH=gfx1200 \
     HIPCCFLAGS='-O3 -std=c++17 -x hip --offload-arch=gfx1200 -Wall -Wextra -fPIE -lamdhip64'
make cuda-bench HIP=1 ROCM_HOME=/usr HIP_ARCH=gfx1200 HIPCCFLAGS='...same...'
```

`-lamdhip64` is needed on Fedora, whose `hipcc` does not link the HIP runtime
itself. `cuda-bench` prints the `rows=…` lines and the `group8x1` line quoted
in the README.

## 4. Model fixtures

Both need `torch` and `transformers` (≥5.11 — older versions apply the wrong
RoPE for this architecture and the oracle silently drifts).

**Tiny oracle** — colibrì's own generator, small enough to validate the whole
engine against a HuggingFace forward pass:

```sh
cd ../colibri/c && python3 tools/make_glm_oracle.py     # -> c/glm_tiny + c/ref_glm.json
```

**Real-dimension fixture** — production expert geometry (hidden 6144,
moe_intermediate 2048 → 18.9 MB int4 experts), 3 layers, 16 experts, ~1.7 B
parameters, ~800 MB as int4:

```sh
python3 tools/make_glm_real_fixture.py --fp8 --output /path/models/glm_real_fp8
python3 tools/convert_fp8_to_int4.py --indir /path/models/glm_real_fp8 \
        --outdir /path/models/glm_real_i4 --ebits 4 --group-size 128
cp /path/models/glm_real_fp8/ref_glm.json /path/models/glm_real_i4/
```

Generation peaks around 4 GB of RAM; the weights are random, so these fixtures
validate numerics and bandwidth, not model quality.

## 5. The engine, end to end

Back in `$WORK/zml` (`cd ../../zml` if you just built fixtures). Self-test
against the oracle (32/32 expected, CPU and GPU builds alike):

```sh
SNAP=../colibri/c/glm_tiny REF=../colibri/c/ref_glm.json TF=1 \
OMP_NUM_THREADS=16 COLI_NO_OMP_TUNE=1 \
  $BZL run $ROCM //coli:coli -- 64 16 16
```

Decode with experts resident in VRAM. colibrì ranks experts by measured routing
heat, so this is two passes — one to record `bench_stats.txt`, one to use it:

```sh
M=/path/models/glm_real_i4
export SNAP=$M REF=$M/ref_glm.json REPLAY=1 DRAFT=0 PROF=1

STATS=$M/bench_stats.txt $BZL run $ROCM //coli:coli -- 4 4 4      # record heat

COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB=4 PIN=$M/bench_stats.txt \
OMP_NUM_THREADS=16 COLI_NO_OMP_TUNE=1 \
  $BZL run $ROCM //coli:coli -- 4 4 4                             # use it
```

Look for `routed CPU 0.000s` with a non-zero `routed GPU critical` in the
`P0-EXEC` line, `[CUDA] resident set: N tensors`, and the `decode forwards …
latency p50` line. A single ~300 ms outlier in `max` is XLA compiling the first
executable for a group size it hasn't seen; it warms out.

On a 16 GB machine, run `bazelisk shutdown` first — the Bazel server holds
~1.7 GB, and colibrì's RAM guard will (correctly) refuse to start if the
projected peak exceeds what's available. Lower `CTX` or drop `PIN_GB` if it
does.

## 6. colibrì's web dashboard on this backend

Build the UI once, then point colibrì's OpenAI-compatible server at a wrapper
that execs the Bazel binary:

```sh
cd ../colibri/web && npm install && npm run build      # -> web/dist
mkdir -p ../../serve && cd ../../serve
cp ../colibri/c/openai_server.py .
ln -sf ../colibri/web web
printf '#!/bin/sh\nexec %s/zml/bazel-bin/coli/coli "$@"\n' "$PWD/.." > colibri
chmod +x colibri
```

The wrapper matters: the server looks for a binary named `colibri` beside
itself, and a *symlink* breaks Bazel's runfiles discovery (`libpjrt_rocm.so`
goes missing). Then:

```sh
M=/path/models/glm_real_i4
SNAP=$M CTX=1024 COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB=4 \
PIN=$M/bench_stats.txt OMP_NUM_THREADS=16 COLI_NO_OMP_TUNE=1 \
  python3 openai_server.py --model $M --host 127.0.0.1 --port 8000 --cap 4
```

`http://127.0.0.1:8000` serves the dashboard; `/health` reports the VRAM/RAM/
disk tier counts and the GPU it found; `/profile` feeds the per-turn timing
charts.

Synthetic fixtures ship no tokenizer, so generate one sized to the fixture's
vocabulary (`vocab_size` in the fixture's `config.json`):

```sh
python3 ../../tools/make_fixture_tokenizer.py $M/tokenizer.json 8192
```

Generated text will be gibberish — the weights are random. The tier counts and
timings on the dashboard are real.
