#!/usr/bin/env python3
"""Build the Phase 3 scheduler bundle: reference outputs for the DISTILLED
denoising loop, dumped per update so every stage of every step is gated
independently.

Oracle boundary (pre-registered in the notebook, 2026-08-21): this script
IMPORTS and RUNS the reference's own code — euler_denoising_loop,
post_process_latent, timesteps_from_mask (ltx-pipelines 1.0.0),
EulerDiffusionStep, GaussianNoiser, to_denoised/to_velocity, LatentState
(ltx-core 1.2.0) — no scheduler math is re-implemented here. The 21B
transformer is replaced by pre-generated per-step VELOCITY tensors (data,
not a function); the stub denoise_fn applies the reference to_denoised
with mask-scaled timesteps exactly as the X0Model wrapper does.

Provenance note: PyPI ltx-pipelines 1.0.0 skews against ltx-core 1.2.0 —
its package __init__ imports `decode_video`, which core 1.2.0 no longer
exports, so `import ltx_pipelines` fails outright. The loop modules we
need (utils/samplers.py, utils/helpers.py and their imports) are
compatible; they are loaded verbatim from their installed files through a
bare package shim that skips only the broken top-level __init__.

Storage is f32 on CPU throughout the primary run, so every serialized
tensor is exactly the value the oracle computed with (the Phase 2
f32_exact lesson, applied at design time). A secondary bf16-storage run
of the same trajectory calibrates the deployment floor (informative, not
a gate).

Usage: oracle-venv/bin/python tools/make_scheduler_bundle.py OUTDIR
"""

import hashlib
import importlib
import importlib.metadata
import json
import pathlib
import sys
import types

import torch

# ---- package shim (see provenance note) -----------------------------------
_site = pathlib.Path(torch.__file__).parent.parent
for _pkg in ("ltx_pipelines", "ltx_pipelines.utils"):
    _m = types.ModuleType(_pkg)
    _m.__path__ = [str(_site / _pkg.replace(".", "/"))]
    sys.modules[_pkg] = _m

# Second skew symptom: core 1.2.0 renamed GemmaTextEncoder to
# LTXGemmaTextEncoder; pipelines 1.0.0's helpers imports the old name but
# uses it only as a type annotation on the prompt path (never touched by
# the scheduler functions). Alias it so helpers.py loads unmodified.
import ltx_core.text_encoders.gemma as _gemma  # noqa: E402

if not hasattr(_gemma, "GemmaTextEncoder"):
    _gemma.GemmaTextEncoder = _gemma.LTXGemmaTextEncoder

samplers = importlib.import_module("ltx_pipelines.utils.samplers")
helpers = importlib.import_module("ltx_pipelines.utils.helpers")
constants = importlib.import_module("ltx_pipelines.utils.constants")

from ltx_core.components.diffusion_steps import EulerDiffusionStep  # noqa: E402
from ltx_core.components.noisers import GaussianNoiser  # noqa: E402
from ltx_core.types import LatentState  # noqa: E402
from ltx_core.utils import to_denoised  # noqa: E402

euler_denoising_loop = samplers.euler_denoising_loop
post_process_latent = helpers.post_process_latent
timesteps_from_mask = helpers.timesteps_from_mask

SEED = 733
TV, CV = 64, 128  # video tokens, channels
TA, CA = 16, 32  # audio tokens, channels

SIGMAS_1 = constants.DISTILLED_SIGMA_VALUES
SIGMAS_2 = constants.STAGE_2_DISTILLED_SIGMA_VALUES
assert SIGMAS_1 == [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
assert SIGMAS_2 == [0.909375, 0.725, 0.421875, 0.0]


def make_state(latent: torch.Tensor, clean: torch.Tensor, mask: torch.Tensor) -> LatentState:
    return LatentState(
        latent=latent,
        denoise_mask=mask,
        positions=torch.zeros(1, 3, latent.shape[1]),
        clean_latent=clean,
    )


def make_stub(vv: torch.Tensor, va: torch.Tensor):
    """One step's denoise_fn: the reference X0Model conversion applied to
    pre-generated velocities — to_denoised(latent, v, mask*sigma)."""

    def stub(video_state, audio_state, sigmas, idx):
        sigma = sigmas[idx]
        dv = to_denoised(video_state.latent, vv, timesteps_from_mask(video_state.denoise_mask, sigma))
        da = to_denoised(audio_state.latent, va, timesteps_from_mask(audio_state.denoise_mask, sigma))
        return dv, da

    return stub


def run_stage(sigmas, v_state, a_state, vv_all, va_all, stepper, dumps, tag):
    """Drive the reference loop one step at a time (each call IS the
    reference loop over a 2-sigma slice, so sigma/sigma_next/step math is
    the reference's own), dumping every intermediate. Returns final states."""
    n = len(sigmas) - 1
    for i in range(n):
        sig_slice = sigmas[i : i + 2]
        for mod, st, v in (("v", v_state, vv_all[i]), ("a", a_state, va_all[i])):
            ts = timesteps_from_mask(st.denoise_mask, sigmas[i])
            den = to_denoised(st.latent, v, ts)
            post = post_process_latent(den, st.denoise_mask, st.clean_latent)
            dumps[f"{tag}_{mod}_step{i}_ts"] = ts
            dumps[f"{tag}_{mod}_step{i}_den"] = den
            dumps[f"{tag}_{mod}_step{i}_post"] = post
        v_state, a_state = euler_denoising_loop(
            sig_slice, v_state, a_state, stepper, make_stub(vv_all[i], va_all[i])
        )
        dumps[f"{tag}_v_step{i}_next"] = v_state.latent
        dumps[f"{tag}_a_step{i}_next"] = a_state.latent
    return v_state, a_state


def run_all(dtype: torch.dtype, dumps: dict | None):
    """The full two-stage trajectory in the given storage dtype. All data
    tensors are generated in f32 and cast, so f32/bf16 runs share values."""
    g = torch.Generator().manual_seed(SEED)

    def rnd(*shape, scale):
        return (torch.randn(*shape, generator=g) * scale).to(dtype)

    clean_v = rnd(1, TV, CV, scale=0.5)
    mask_v = torch.ones(1, TV, 1)
    mask_v[0, 48:56] = 0.0
    mask_v[0, 56:64] = 0.5
    clean_a = rnd(1, TA, CA, scale=0.5)
    mask_a = torch.ones(1, TA, 1)
    vv1 = [rnd(1, TV, CV, scale=0.7) for _ in range(len(SIGMAS_1) - 1)]
    va1 = [rnd(1, TA, CA, scale=0.7) for _ in range(len(SIGMAS_1) - 1)]
    vv2 = [rnd(1, TV, CV, scale=0.7) for _ in range(len(SIGMAS_2) - 1)]
    va2 = [rnd(1, TA, CA, scale=0.7) for _ in range(len(SIGMAS_2) - 1)]
    upsampled = rnd(1, TV, CV, scale=0.4)  # synthetic stage-2 initial latent
    noise_g = torch.Generator().manual_seed(SEED + 1)
    noiser = GaussianNoiser(generator=noise_g)
    stepper = EulerDiffusionStep()

    sig1 = torch.tensor(SIGMAS_1, dtype=torch.float32)
    sig2 = torch.tensor(SIGMAS_2, dtype=torch.float32)

    # Stage 1: latent == clean (create_initial_state semantics), then the
    # reference noiser at noise_scale 1.0. Noise draws are captured by
    # re-seeding a twin generator: GaussianNoiser draws one randn per call.
    v_state = make_state(clean_v.clone(), clean_v.clone(), mask_v)
    a_state = make_state(clean_a.clone(), clean_a.clone(), mask_a)
    twin = torch.Generator().manual_seed(SEED + 1)
    noise1_v = torch.randn(1, TV, CV, generator=twin, dtype=dtype)
    noise1_a = torch.randn(1, TA, CA, generator=twin, dtype=dtype)
    v_state = noiser(v_state, 1.0)
    a_state = noiser(a_state, 1.0)

    if dumps is not None:
        # Twin-capture control: the captured noise, pushed through the
        # noiser's own lerp chain, must reproduce the noised latent bitwise.
        for st, cl, nz, mk in ((v_state, clean_v, noise1_v, mask_v), (a_state, clean_a, noise1_a, mask_a)):
            chk = torch.lerp(cl.float(), torch.lerp(cl.float(), nz.float(), 1.0), mk).to(dtype)
            assert torch.equal(chk, st.latent), "noise twin capture diverged (stage 1)"
        dumps["mask_v"], dumps["mask_a"] = mask_v, mask_a
        dumps["clean_v"], dumps["clean_a"] = clean_v, clean_a
        dumps["noise1_v"], dumps["noise1_a"] = noise1_v, noise1_a
        dumps["s1_v_x0"], dumps["s1_a_x0"] = v_state.latent, a_state.latent
        for i, (v, a) in enumerate(zip(vv1, va1)):
            dumps[f"s1_vel_v_{i}"], dumps[f"s1_vel_a_{i}"] = v, a
        for i, (v, a) in enumerate(zip(vv2, va2)):
            dumps[f"s2_vel_v_{i}"], dumps[f"s2_vel_a_{i}"] = v, a
        dumps["upsampled_v"] = upsampled
        v_state, a_state = run_stage(sig1, v_state, a_state, vv1, va1, stepper, dumps, "s1")
    else:
        v_state, a_state = euler_denoising_loop(
            sig1, v_state, a_state, stepper,
            lambda vs, as_, s, i: make_stub(vv1[i], va1[i])(vs, as_, s, i),
        )
    s1_final_v, s1_final_a = v_state.latent, a_state.latent

    # Stage 2: video initializes from the synthetic upsampled latent, audio
    # from its own stage-1 result (deployment wiring); both renoised through
    # the reference noiser at noise_scale = sigmas_2[0].
    v_state = make_state(upsampled.clone(), upsampled.clone(), mask_v)
    a_state = make_state(s1_final_a.clone(), s1_final_a.clone(), mask_a)
    twin2 = torch.Generator().manual_seed(SEED + 1)
    twin2.set_state(noise_g.get_state())
    noise2_v = torch.randn(1, TV, CV, generator=twin2, dtype=dtype)
    noise2_a = torch.randn(1, TA, CA, generator=twin2, dtype=dtype)
    v_state = noiser(v_state, sig2[0])
    a_state = noiser(a_state, sig2[0])

    if dumps is not None:
        for st, cl, nz, mk in ((v_state, upsampled, noise2_v, mask_v), (a_state, s1_final_a, noise2_a, mask_a)):
            chk = torch.lerp(cl.float(), torch.lerp(cl.float(), nz.float(), sig2[0]), mk).to(dtype)
            assert torch.equal(chk, st.latent), "noise twin capture diverged (stage 2)"
        dumps["s2_init_a"] = s1_final_a
        dumps["noise2_v"], dumps["noise2_a"] = noise2_v, noise2_a
        dumps["s2_v_x0"], dumps["s2_a_x0"] = v_state.latent, a_state.latent
        v_state, a_state = run_stage(sig2, v_state, a_state, vv2, va2, stepper, dumps, "s2")
    else:
        v_state, a_state = euler_denoising_loop(
            sig2, v_state, a_state, stepper,
            lambda vs, as_, s, i: make_stub(vv2[i], va2[i])(vs, as_, s, i),
        )
    return s1_final_v, s1_final_a, v_state.latent, a_state.latent


def stats(t: torch.Tensor) -> dict:
    a = t.detach().to(torch.float64).flatten()
    return {
        "max_abs": float(a.abs().max()),
        "rms": float((a * a).mean().sqrt()),
        "nan": int(torch.isnan(a).sum()),
        "inf": int(torch.isinf(a).sum()),
    }


def main() -> int:
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)

    # Primary f32 run, stepwise-driven with full dumps.
    dumps: dict[str, torch.Tensor] = {}
    s1v, s1a, s2v, s2a = run_all(torch.float32, dumps)

    # Staging control: the same trajectory driven by ONE loop call per
    # stage must reproduce the stepwise finals bit-exactly.
    s1v_f, s1a_f, s2v_f, s2a_f = run_all(torch.float32, None)
    assert torch.equal(s1v, s1v_f) and torch.equal(s1a, s1a_f), "stage-1 stepwise != full loop"
    assert torch.equal(s2v, s2v_f) and torch.equal(s2a, s2a_f), "stage-2 stepwise != full loop"
    print("staging == full loop (bitwise): stage 1 and stage 2")

    # Secondary bf16-storage run: deployment-dtype floor, informative.
    _, _, s2v_b, s2a_b = run_all(torch.bfloat16, None)
    rel_v = float(((s2v_b.float() - s2v) ** 2).mean().sqrt() / ((s2v**2).mean().sqrt() + 1e-20))
    rel_a = float(((s2a_b.float() - s2a) ** 2).mean().sqrt() / ((s2a**2).mean().sqrt() + 1e-20))
    print(f"bf16-storage final vs f32: video rel-RMS {rel_v:.5f}, audio rel-RMS {rel_a:.5f}")

    manifest = {
        "seed": SEED,
        "tv": TV, "cv": CV, "ta": TA, "ca": CA,
        "sigmas_stage1": SIGMAS_1,
        "sigmas_stage2": SIGMAS_2,
        "torch": torch.__version__,
        "ltx_core": importlib.metadata.version("ltx-core"),
        "ltx_pipelines": importlib.metadata.version("ltx-pipelines"),
        "lerp": "two-branch ATen kernel: |w|<0.5 ? a+w*(b-a) : b-(b-a)*(1-w)",
        "bf16_final_relrms": {"video": rel_v, "audio": rel_a},
        "tensors": {},
    }
    dumps["sigmas1"] = torch.tensor(SIGMAS_1, dtype=torch.float32)
    dumps["sigmas2"] = torch.tensor(SIGMAS_2, dtype=torch.float32)
    for name, t in dumps.items():
        arr = t.detach().contiguous().to(torch.float32).numpy()
        data = arr.tobytes()
        (outdir / f"{name}.bin").write_bytes(data)
        manifest["tensors"][name] = {
            "shape": list(arr.shape), "dtype": "f32",
            "sha256": hashlib.sha256(data).hexdigest(), **stats(t),
        }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(dumps)} tensors + manifest to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
