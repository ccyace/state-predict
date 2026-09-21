#!/usr/bin/env python3
"""Generate Q-Diffusion-style LDM calibration data (xs/ts[/cs]) from FP model."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import trange

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_TAMING = os.path.join(ROOT, "src", "taming-transformers")
if _TAMING not in sys.path:
    sys.path.insert(0, _TAMING)

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def load_ldm(config_path: str, ckpt_path: str, device: torch.device):
    config = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    model = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.to(device).eval()
    if hasattr(model, "model_ema"):
        model.model_ema.store(model.model.parameters())
        model.model_ema.copy_to(model.model)
    return model, config


@torch.no_grad()
def collect_cali(
    model,
    *,
    n_traj: int,
    batch_size: int,
    steps: int,
    eta: float,
    cfg_scale: float,
    device: torch.device,
    conditional: bool,
):
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=False)
    timesteps = sampler.ddim_timesteps
    C = model.model.diffusion_model.in_channels
    H = W = model.model.diffusion_model.image_size

    xs_by_step = {i: [] for i in range(len(timesteps))}
    ts_by_step = {i: [] for i in range(len(timesteps))}
    cs_by_step = {i: [] for i in range(len(timesteps))} if conditional else None
    ucs_by_step = {i: [] for i in range(len(timesteps))} if conditional else None

    time_range = np.flip(timesteps)
    total = len(timesteps)

    for bi in trange((n_traj + batch_size - 1) // batch_size, desc="cali collect"):
        b = min(batch_size, n_traj - bi * batch_size)
        x = torch.randn(b, C, H, W, device=device)
        if conditional:
            classes = torch.randint(0, 1000, (b,), device=device)
            c = model.get_learned_conditioning({model.cond_stage_key: classes})
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.full((b,), 1000, device=device, dtype=classes.dtype)}
            )
        else:
            c = None
            uc = None

        for i, step in enumerate(time_range):
            index = total - i - 1
            t = torch.full((b,), int(step), device=device, dtype=torch.long)
            xs_by_step[i].append(x.detach().cpu())
            ts_by_step[i].append(t.detach().cpu().float())
            if conditional:
                cs_by_step[i].append(c.detach().cpu())
                ucs_by_step[i].append(uc.detach().cpu())

            outs = sampler.p_sample_ddim(
                x,
                c,
                t,
                index=index,
                unconditional_guidance_scale=cfg_scale if conditional else 1.0,
                unconditional_conditioning=uc,
            )
            x = outs[0]

    payload = {
        "xs": [torch.cat(xs_by_step[i], 0) for i in range(len(timesteps))],
        "ts": [torch.cat(ts_by_step[i], 0) for i in range(len(timesteps))],
        "meta": {"steps": steps, "eta": eta, "n_traj": n_traj, "conditional": conditional},
    }
    if conditional:
        payload["cs"] = [torch.cat(cs_by_step[i], 0) for i in range(len(timesteps))]
        payload["ucs"] = [torch.cat(ucs_by_step[i], 0) for i in range(len(timesteps))]
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp_ckpt", required=True)
    p.add_argument("--ldm_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n_traj", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--scale", type=float, default=3.0)
    p.add_argument("--cond", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_ldm(args.ldm_config, args.fp_ckpt, device)
    payload = collect_cali(
        model,
        n_traj=args.n_traj,
        batch_size=args.batch_size,
        steps=args.steps,
        eta=args.eta,
        cfg_scale=args.scale,
        device=device,
        conditional=args.cond or model.cond_stage_model is not None,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    torch.save(payload, args.output)
    print(f"saved {args.output} steps={len(payload['xs'])} per_step={payload['xs'][0].shape[0]}", flush=True)


if __name__ == "__main__":
    main()
