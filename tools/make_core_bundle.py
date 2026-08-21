#!/usr/bin/env python3
"""Build the E2E-core oracle bundle: reference outputs for the video
stream's non-block parts — adaln_single, prompt_adaln_single,
patchify_proj, and the output tail (norm_out + scale_shift_table +
proj_out) — staged per the pre-registered order (notebook 2026-08-21,
late morning).

Runs the upstream ltx-core modules (torch CPU) on the fetched extras
weights, in float64 (weights upcast from bf16; the 256-dim sinusoid is
f32 by the reference's own construction — get_timestep_embedding
computes in f32 unconditionally and casts up after). The timestep path
goes through the REAL TransformerArgsPreprocessor._prepare_timestep
(x1000 multiplier, flatten, reshape) and the output tail through the
REAL LTXModel._process_output (self is unused — called unbound), so the
staging cannot drift from the reference.

Also dumps a wrong-norm control (RMSNorm in place of the tail's true
LayerNorm) that the engine gate must NOT match — teeth for the
norm-kind distinction (H-CORE-4).

Usage: oracle-venv/bin/python tools/make_core_bundle.py EXTRAS_DIR OUTDIR
"""

import hashlib
import json
import pathlib
import sys

import torch

from ltx_core.model.transformer.adaln import AdaLayerNormSingle
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer_args import TransformerArgsPreprocessor

SEED = 471
B, T = 1, 64
DIM = 4096
IN_CH = 128
SIGMA = 0.909375  # first stage-2 value from the distilled table
TS_MULT = 1000  # timestep_scale_multiplier (config default, unoverridden)
NORM_EPS = 1e-6

DT = {"BF16": torch.bfloat16, "F32": torch.float32}


def load_extras(wdir: pathlib.Path) -> dict[str, torch.Tensor]:
    man = json.loads((wdir / "manifest.json").read_text())
    sd = {}
    for name, e in man["tensors"].items():
        raw = (wdir / e["file"]).read_bytes()
        sd[name] = torch.frombuffer(bytearray(raw), dtype=DT[e["dtype"]]).reshape(e["shape"])
    return sd


def sub_state(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def stats(t: torch.Tensor) -> dict:
    a = t.detach().to(torch.float64).flatten()
    return {
        "max_abs": float(a.abs().max()),
        "rms": float((a * a).mean().sqrt()),
        "nan": int(torch.isnan(a).sum()),
        "inf": int(torch.isinf(a).sum()),
    }


def main() -> int:
    wdir, outdir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    sd = load_extras(wdir)

    adaln = AdaLayerNormSingle(DIM, embedding_coefficient=9)
    missing, unexpected = adaln.load_state_dict(sub_state(sd, "adaln_single."), strict=True)
    prompt_adaln = AdaLayerNormSingle(DIM, embedding_coefficient=2)
    prompt_adaln.load_state_dict(sub_state(sd, "prompt_adaln_single."), strict=True)
    patchify = torch.nn.Linear(IN_CH, DIM, bias=True)
    patchify.load_state_dict(sub_state(sd, "patchify_proj."), strict=True)
    proj_out = torch.nn.Linear(DIM, IN_CH, bias=True)
    proj_out.load_state_dict(sub_state(sd, "proj_out."), strict=True)
    scale_shift_table = sd["scale_shift_table"].clone()
    assert scale_shift_table.dtype == torch.float32  # the checkpoint's lone f32 tensor
    norm_out = torch.nn.LayerNorm(DIM, elementwise_affine=False, eps=NORM_EPS)

    for m in (adaln, prompt_adaln, patchify, proj_out):
        m.to(torch.float64).eval()
    sst64 = scale_shift_table.to(torch.float64)

    # ---- inputs (seeded; rounded through f32 like every bundle since the
    # 1.5e-5 phantom) ------------------------------------------------------
    def f32_exact(t: torch.Tensor) -> torch.Tensor:
        return t.to(torch.float32).to(torch.float64)

    latent = f32_exact(torch.randn(B, T, IN_CH, dtype=torch.float64) * 0.5)
    # per-token timesteps: denoise_mask * sigma, first 16 tokens conditioning
    mask = torch.ones(B, T, dtype=torch.float64)
    mask[:, :16] = 0.0
    timesteps = f32_exact(mask * SIGMA)
    sigma = torch.full((B, 1), SIGMA, dtype=torch.float64)
    x_final = f32_exact(torch.randn(B, T, DIM, dtype=torch.float64) * 0.5)

    dumps: dict[str, torch.Tensor] = {
        "in_latent": latent, "in_timesteps": timesteps,
        "in_sigma": sigma, "in_xfinal": x_final,
    }

    # ---- timestep path through the REAL preprocessor helper --------------
    pre = TransformerArgsPreprocessor(
        patchify_proj=patchify, adaln=adaln, inner_dim=DIM,
        max_pos=[20, 2048, 2048], num_attention_heads=32,
        use_middle_indices_grid=True, timestep_scale_multiplier=TS_MULT,
        double_precision_rope=True, positional_embedding_theta=10000.0,
        rope_type=LTXRopeType.SPLIT, prompt_adaln=prompt_adaln,
    )
    with torch.no_grad():
        # staged: the sinusoid alone (f32 by reference construction)
        sinusoid = adaln.emb.time_proj((timesteps * TS_MULT).flatten())
        assert sinusoid.dtype == torch.float32
        dumps["s1_sinusoid"] = sinusoid

        ts9, emb_t = pre._prepare_timestep(timesteps, adaln, B, torch.float64)
        dumps["s2_embedded_timestep"] = emb_t  # [B, T, DIM]
        dumps["s3_ts9"] = ts9  # [B, T, 9*DIM]
        pts2, _ = pre._prepare_timestep(sigma, prompt_adaln, B, torch.float64)
        dumps["s4_pts2"] = pts2  # [B, 1, 2*DIM]

        # control: the staged sinusoid feeds the same MLP the helper ran
        emb_ctl = adaln.emb.timestep_embedder(sinusoid.to(torch.float64))
        drift = (emb_ctl.view(B, T, DIM) - emb_t).abs().max()
        assert drift < 1e-12, f"staging drifted from helper: {drift}"

        dumps["s5_patchify"] = patchify(latent)

        # ---- output tail through the REAL model method (self unused) -----
        tail = LTXModel._process_output(None, sst64, norm_out, proj_out, x_final, emb_t)
        dumps["s6_tail_out"] = tail  # [B, T, IN_CH]

        # wrong-norm control: RMSNorm instead of LayerNorm, same everything
        rms = x_final * torch.rsqrt((x_final ** 2).mean(-1, keepdim=True) + NORM_EPS)
        ssv = sst64[None, None] + emb_t[:, :, None]
        wrong = proj_out(rms * (1 + ssv[:, :, 1]) + ssv[:, :, 0])
        rel = float(((wrong - tail) ** 2).mean().sqrt() / (tail ** 2).mean().sqrt())
        assert rel > 1e-3, f"wrong-norm control indistinguishable ({rel:.2e}) — gate would have no teeth"
        dumps["s6_tail_wrongnorm"] = wrong
        print(f"wrong-norm control rel-RMS vs true tail: {rel:.4f} (gate teeth confirmed)")

    manifest = {
        "seed": SEED, "B": B, "T": T, "dim": DIM, "in_ch": IN_CH,
        "sigma": SIGMA, "timestep_scale_multiplier": TS_MULT, "norm_eps": NORM_EPS,
        "cond_tokens": 16, "torch": torch.__version__,
        "weights_manifest_header_sha": json.loads((wdir / "manifest.json").read_text())["header_sha256"],
        "tensors": {},
    }
    for name, t in dumps.items():
        arr = t.detach().to(torch.float32).contiguous().numpy()
        data = arr.tobytes()
        (outdir / f"{name}.bin").write_bytes(data)
        manifest["tensors"][name] = {
            "shape": list(arr.shape), "dtype": "f32",
            "sha256": hashlib.sha256(data).hexdigest(), **stats(t),
        }
        print(f"  {name} {list(arr.shape)} max_abs={manifest['tensors'][name]['max_abs']:.4g}")
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {outdir}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
