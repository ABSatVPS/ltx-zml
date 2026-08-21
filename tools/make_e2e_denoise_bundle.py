#!/usr/bin/env python3
"""The multi-step E2E denoising oracle (H-LOOP-1/2, notebook 2026-08-21
afternoon): the REFERENCE stage-1 distilled loop with the REAL f64 model
as denoise_fn, at harness T=64.

The loop is the reference's own — euler_denoising_loop /
EulerDiffusionStep / GaussianNoiser / timesteps_from_mask /
post_process_latent, imported through the scheduler bundle's package
shim — driven one 2-sigma slice at a time (the scheduler bundle proved
slicing == full loop bitwise). The model is the composition proven by
make_e2e_bundle: patchify -> adaln-driven 48-block chain -> tail, blocks
loaded one at a time in f64, velocity converted via to_denoised with
per-token timesteps exactly as X0Model does. Audio marches with zero
velocities (video-only scope; audio values never touch the video state).

Mask convention (scheduler bundle's, deliberately): ones, tokens 48:56
at 0.0, 56:64 at 0.5 — three distinct per-token sigmas per step.

Per-step dumps (video): ts, velocity, denoised, post, next latent.

Usage: oracle-venv/bin/python tools/make_e2e_denoise_bundle.py \
           ORACLE_DIR EXTRAS_DIR WORK_ROOT OUT_DIR
"""

import gc
import hashlib
import importlib
import json
import pathlib
import sys
import types

import numpy as np
import torch

# ---- package shim (see make_scheduler_bundle.py provenance note) ----------
_site = pathlib.Path(torch.__file__).parent.parent
for _pkg in ("ltx_pipelines", "ltx_pipelines.utils"):
    _m = types.ModuleType(_pkg)
    _m.__path__ = [str(_site / _pkg.replace(".", "/"))]
    sys.modules[_pkg] = _m
import ltx_core.text_encoders.gemma as _gemma  # noqa: E402

if not hasattr(_gemma, "GemmaTextEncoder"):
    _gemma.GemmaTextEncoder = _gemma.LTXGemmaTextEncoder

samplers = importlib.import_module("ltx_pipelines.utils.samplers")
helpers = importlib.import_module("ltx_pipelines.utils.helpers")
constants = importlib.import_module("ltx_pipelines.utils.constants")

from ltx_core.components.diffusion_steps import EulerDiffusionStep  # noqa: E402
from ltx_core.components.noisers import GaussianNoiser  # noqa: E402
from ltx_core.model.transformer.adaln import AdaLayerNormSingle  # noqa: E402
from ltx_core.model.transformer.model import LTXModel  # noqa: E402
from ltx_core.model.transformer.rope import LTXRopeType  # noqa: E402
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig  # noqa: E402
from ltx_core.model.transformer.transformer_args import TransformerArgs, TransformerArgsPreprocessor  # noqa: E402
from ltx_core.types import LatentState  # noqa: E402
from ltx_core.utils import to_denoised  # noqa: E402

euler_denoising_loop = samplers.euler_denoising_loop
timesteps_from_mask = helpers.timesteps_from_mask
post_process_latent = helpers.post_process_latent

SEED = 954
B, T, S, DIM, IN_CH = 1, 64, 32, 4096, 128
TA, CA = 16, 32
NORM_EPS = 1e-6
SIGMAS_1 = constants.DISTILLED_SIGMA_VALUES
assert SIGMAS_1 == [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
DT = {"BF16": torch.bfloat16, "F32": torch.float32}


def load_block(wdir: pathlib.Path) -> BasicAVTransformerBlock:
    man = json.loads((wdir / "manifest.json").read_text())
    sd = {}
    for name, e in man["tensors"].items():
        raw = (wdir / e["file"]).read_bytes()
        t = torch.frombuffer(bytearray(raw), dtype=DT[e["dtype"]]).reshape(e["shape"])
        sd[name.replace(f"transformer_blocks.{man['block']}.", "")] = t
    cfg = TransformerConfig(dim=DIM, heads=32, d_head=128, context_dim=DIM,
                            apply_gated_attention=True, cross_attention_adaln=True, ff_bias=False)
    blk = BasicAVTransformerBlock(video=cfg, audio=None, rope_type=LTXRopeType.SPLIT, norm_eps=NORM_EPS)
    blk.load_state_dict(sd, strict=False)
    return blk.to(torch.float64).eval()


def load_f32(d: pathlib.Path, name: str, shape) -> torch.Tensor:
    a = np.frombuffer((d / name).read_bytes(), dtype=np.float32).reshape(shape).copy()
    return torch.from_numpy(a).to(torch.float64)


def load_extras(wdir: pathlib.Path) -> dict[str, torch.Tensor]:
    man = json.loads((wdir / "manifest.json").read_text())
    return {
        name: torch.frombuffer(bytearray((wdir / e["file"]).read_bytes()), dtype=DT[e["dtype"]]).reshape(e["shape"])
        for name, e in man["tensors"].items()
    }


def sub_state(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def main() -> int:
    oracle, extras_dir, root, out = (pathlib.Path(p) for p in sys.argv[1:5])
    out.mkdir(parents=True, exist_ok=True)

    context = load_f32(oracle, "in_context.bin", (B, S, DIM))
    cos = load_f32(oracle, "rope_cos.bin", (B, 32, T, 64))
    sin = load_f32(oracle, "rope_sin.bin", (B, 32, T, 64))

    sd = load_extras(extras_dir)
    adaln = AdaLayerNormSingle(DIM, embedding_coefficient=9)
    adaln.load_state_dict(sub_state(sd, "adaln_single."), strict=True)
    prompt_adaln = AdaLayerNormSingle(DIM, embedding_coefficient=2)
    prompt_adaln.load_state_dict(sub_state(sd, "prompt_adaln_single."), strict=True)
    patchify = torch.nn.Linear(IN_CH, DIM, bias=True)
    patchify.load_state_dict(sub_state(sd, "patchify_proj."), strict=True)
    proj_out = torch.nn.Linear(DIM, IN_CH, bias=True)
    proj_out.load_state_dict(sub_state(sd, "proj_out."), strict=True)
    sst64 = sd["scale_shift_table"].clone().to(torch.float64)
    norm_out = torch.nn.LayerNorm(DIM, elementwise_affine=False, eps=NORM_EPS)
    for m in (adaln, prompt_adaln, patchify, proj_out):
        m.to(torch.float64).eval()

    pre = TransformerArgsPreprocessor(
        patchify_proj=patchify, adaln=adaln, inner_dim=DIM,
        max_pos=[20, 2048, 2048], num_attention_heads=32,
        use_middle_indices_grid=True, timestep_scale_multiplier=1000,
        double_precision_rope=True, positional_embedding_theta=10000.0,
        rope_type=LTXRopeType.SPLIT, prompt_adaln=prompt_adaln,
    )

    dumps: dict[str, torch.Tensor] = {}

    def velocity_forward(latent64: torch.Tensor, ts: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """patchify -> adaln-driven 48-block chain -> tail, f64. ts is the
        per-token mask*sigma [B,T,1]; sigma the scalar driving pts2."""
        ts9, emb_t = pre._prepare_timestep(ts, adaln, B, torch.float64)
        pts2, _ = pre._prepare_timestep(sigma.reshape(B, 1).to(torch.float64), prompt_adaln, B, torch.float64)
        h = patchify(latent64)
        for bi in range(48):
            blk = load_block(root / f"block{bi}")
            args = TransformerArgs(
                x=h, context=context, context_mask=None, timesteps=ts9,
                embedded_timestep=torch.zeros(B, 1, DIM, dtype=torch.float64),
                positional_embeddings=(cos, sin), cross_positional_embeddings=None,
                cross_scale_shift_timestep=None, cross_gate_timestep=None,
                enabled=True, prompt_timestep=pts2,
            )
            vout, _ = blk.forward(args, None)
            h = vout.x
            del blk
            gc.collect()
        return LTXModel._process_output(None, sst64, norm_out, proj_out, h, emb_t)

    step_no = [0]

    def denoise_fn(video_state, audio_state, sigmas, idx):
        sigma = sigmas[idx]
        ts = timesteps_from_mask(video_state.denoise_mask, sigma)
        with torch.no_grad():
            vel = velocity_forward(video_state.latent.to(torch.float64), ts.to(torch.float64), sigma)
        vel32 = vel.to(torch.float32)
        dv = to_denoised(video_state.latent, vel32, ts)
        da = audio_state.latent  # zero velocity: to_denoised(l, 0, t) == l
        i = step_no[0]
        dumps[f"s1_step{i}_ts"] = ts
        dumps[f"s1_step{i}_vel"] = vel32
        dumps[f"s1_step{i}_den"] = dv
        dumps[f"s1_step{i}_post"] = post_process_latent(dv, video_state.denoise_mask, video_state.clean_latent)
        print(f"  step {i} (sigma={float(sigma):.6g}): velocity rms={float((vel32.double()**2).mean().sqrt()):.4g}", flush=True)
        return dv, da

    # ---- stage-1 initial state (scheduler-bundle conventions) -------------
    g = torch.Generator().manual_seed(SEED)
    clean_v = (torch.randn(B, T, IN_CH, generator=g) * 0.5).to(torch.float32)
    mask_v = torch.ones(B, T, 1)
    mask_v[0, 48:56] = 0.0
    mask_v[0, 56:64] = 0.5
    clean_a = (torch.randn(B, TA, CA, generator=g) * 0.5).to(torch.float32)
    mask_a = torch.ones(B, TA, 1)

    noise_g = torch.Generator().manual_seed(SEED + 1)
    noiser = GaussianNoiser(generator=noise_g)
    twin = torch.Generator().manual_seed(SEED + 1)
    noise1_v = torch.randn(B, T, IN_CH, generator=twin, dtype=torch.float32)
    noise1_a = torch.randn(B, TA, CA, generator=twin, dtype=torch.float32)

    def make_state(latent, clean, mask):
        return LatentState(latent=latent, denoise_mask=mask,
                           positions=torch.zeros(1, 3, latent.shape[1]), clean_latent=clean)

    v_state = make_state(clean_v.clone(), clean_v.clone(), mask_v)
    a_state = make_state(clean_a.clone(), clean_a.clone(), mask_a)
    v_state = noiser(v_state, 1.0)
    a_state = noiser(a_state, 1.0)
    for st, cl, nz, mk in ((v_state, clean_v, noise1_v, mask_v), (a_state, clean_a, noise1_a, mask_a)):
        chk = torch.lerp(cl.float(), torch.lerp(cl.float(), nz.float(), 1.0), mk)
        assert torch.equal(chk, st.latent), "noise twin capture diverged"
    print("control: noise twin capture reproduces the noised latents bitwise")

    dumps["clean_v"], dumps["mask_v"], dumps["noise1_v"] = clean_v, mask_v, noise1_v
    dumps["s1_x0"] = v_state.latent.clone()

    # ---- the loop, one reference slice per step ---------------------------
    sig1 = torch.tensor(SIGMAS_1, dtype=torch.float32)
    stepper = EulerDiffusionStep()
    n = len(SIGMAS_1) - 1
    for i in range(n):
        v_state, a_state = euler_denoising_loop(sig1[i:i + 2], v_state, a_state, stepper, denoise_fn)
        dumps[f"s1_step{i}_next"] = v_state.latent.clone()
        print(f"  step {i} latent rms={float((v_state.latent.double()**2).mean().sqrt()):.4g}", flush=True)
        step_no[0] += 1

    # ---- serialize --------------------------------------------------------
    manifest = {
        "seed": SEED, "B": B, "T": T, "dim": DIM, "in_ch": IN_CH,
        "sigmas": SIGMAS_1, "mask": "ones; 48:56=0.0; 56:64=0.5",
        "oracle_manifest_sha": hashlib.sha256((oracle / "manifest.json").read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "note": "slice-wise loop == full loop bitwise, proven by the scheduler bundle control",
        "tensors": {},
    }
    for name, t in dumps.items():
        arr = t.detach().to(torch.float32).contiguous().numpy()
        data = arr.tobytes()
        (out / f"{name}.bin").write_bytes(data)
        manifest["tensors"][name] = {
            "shape": list(arr.shape), "dtype": "f32",
            "sha256": hashlib.sha256(data).hexdigest(),
            "max_abs": float(np.abs(arr).max()),
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out}/manifest.json")
    print("E2E DENOISE BUNDLE COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
