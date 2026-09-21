#!/usr/bin/env python3
"""
Collect closed-loop (x, eps_q, eps_fp, t) on LDM-4 LSUN-Bedroom for D2 / learned corrector.

Requires FP ckpt at models/ldm/lsun_beds256/model.ckpt and quantized cali ckpt.
"""

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

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from qdiff import QuantModel
from qdiff.utils import resume_cali_model


def load_ldm(config_path: str, ckpt_path: str, device: torch.device):
    config = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    model = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.to(device)
    model.eval()
    if hasattr(model, "model_ema"):
        model.model_ema.store(model.model.parameters())
        model.model_ema.copy_to(model.model)
    return model, config


@torch.no_grad()
def collect(
    fp_model,
    qnn,
    *,
    n_traj: int,
    batch_size: int,
    steps: int,
    eta: float,
    device: torch.device,
    out_path: str,
):
    # Attach quantized UNet for sampling trajectory; keep FP UNet for labels.
    fp_unet = fp_model.model.diffusion_model
    fp_model.model.diffusion_model = qnn
    sampler = DDIMSampler(fp_model)
    sampler.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=False)
    timesteps = sampler.ddim_timesteps

    C = fp_model.model.diffusion_model.in_channels
    H = W = fp_model.model.diffusion_model.image_size

    xs, eqs, efs, ts, tids = [], [], [], [], []
    traj_id = 0
    n_batches = (n_traj + batch_size - 1) // batch_size

    for bi in trange(n_batches, desc="collect LDM traj"):
        b = min(batch_size, n_traj - traj_id)
        x = torch.randn(b, C, H, W, device=device)
        time_range = np.flip(timesteps)
        total = timesteps.shape[0]
        for i, step in enumerate(time_range):
            index = total - i - 1
            t = torch.full((b,), int(step), device=device, dtype=torch.long)
            # quant eps (sampling model)
            eq = fp_model.apply_model(x, t, None)
            # fp eps on same state
            fp_model.model.diffusion_model = fp_unet
            ef = fp_model.apply_model(x, t, None)
            fp_model.model.diffusion_model = qnn

            xs.append(x.detach().cpu().half())
            eqs.append(eq.detach().cpu().half())
            efs.append(ef.detach().cpu().half())
            ts.append(t.detach().cpu().float())
            tids.append(torch.arange(traj_id, traj_id + b))

            # DDIM step with quant eps (open-loop / no corrector)
            outs = sampler.p_sample_ddim(x, None, t, index=index, eta=eta)
            # p_sample_ddim signature doesn't take eta - eta is in schedule.
            # Actually look at signature - no eta in p_sample_ddim. Good.
            x = outs[0]

        traj_id += b

    # restore
    fp_model.model.diffusion_model = fp_unet

    payload = {
        "x": torch.cat(xs, 0),
        "eq": torch.cat(eqs, 0),
        "ef": torch.cat(efs, 0),
        "t": torch.cat(ts, 0),
        "traj_id": torch.cat(tids, 0).long(),
        "meta": {
            "n_traj": n_traj,
            "steps": steps,
            "eta": eta,
            "shape": [C, H, W],
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    torch.save(payload, out_path)
    print(f"saved {out_path} n={payload['x'].shape[0]}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp_ckpt", default="models/ldm/lsun_beds256/model.ckpt")
    p.add_argument("--config", default="models/ldm/lsun_beds256/config.yaml")
    p.add_argument("--cali_ckpt", default="bedroom_w4a8_ckpt.pth")
    p.add_argument("--n_traj", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--weight_bit", type=int, default=4)
    p.add_argument("--act_bit", type=int, default=8)
    p.add_argument("--quant_act", action="store_true", default=True)
    p.add_argument("--a_sym", action="store_true", default=True)
    p.add_argument("--output", default="output/bedroom_compare/traj_w4a8_n128.pt")
    args = p.parse_args()

    if not os.path.isfile(args.fp_ckpt):
        raise FileNotFoundError(
            f"FP checkpoint missing: {args.fp_ckpt}\n"
            "Download CompVis LDM LSUN-Bedrooms and place model.ckpt there."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp_model, config = load_ldm(args.config, args.fp_ckpt, device)

    wq = {"n_bits": args.weight_bit, "channel_wise": True, "scale_method": "max"}
    aq = {
        "n_bits": args.act_bit,
        "symmetric": args.a_sym,
        "channel_wise": False,
        "scale_method": "max",
        "leaf_param": args.quant_act,
    }
    qnn = QuantModel(model=fp_model.model.diffusion_model, weight_quant_params=wq, act_quant_params=aq)
    qnn.cuda().eval()
    image_size = config.model.params.image_size
    channels = config.model.params.channels
    cali_data = (torch.randn(1, channels, image_size, image_size), torch.randint(0, 1000, (1,)))
    resume_cali_model(qnn, args.cali_ckpt, cali_data, args.quant_act, "qdiff", cond=False)
    qnn.set_quant_state(True, True)

    # Keep a pristine FP UNet copy for teacher labels
    # QuantModel wraps the original module in-place; reload FP UNet separately.
    fp_model2, _ = load_ldm(args.config, args.fp_ckpt, device)
    # Replace wrapped net: attach qnn for sampling, use fp_model2's unet as teacher via collect()
    # Simpler: put FP unet back into fp_model for teacher path inside collect.
    # Actually QuantModel replaced modules inside diffusion_model. Reload clean FP:
    collect(
        fp_model2,
        qnn,
        n_traj=args.n_traj,
        batch_size=args.batch_size,
        steps=args.steps,
        eta=args.eta,
        device=device,
        out_path=args.output,
    )


if __name__ == "__main__":
    main()
