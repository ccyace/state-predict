#!/usr/bin/env python
"""Collect open-loop full-grid trajectory data for dt-oracle training.

Stores (x, eq, ef, t, traj_id) at every DDIM step on uncorrected quantized paths.
ef uses the same QuantModel with quant disabled at nominal t (teacher label).

Example:
  python noise_eps_corr/scripts/collect_fullstep_training_data.py \\
    --backbone uvit --cali_ckpt uvit_experiments/checkpoints/uvit_w8a8_ckpt.pth \\
    --num_trajectories 1000 --timesteps 100 --skip_type quad \\
    --output uvit_experiments/outputs/phase2_w8a8/traj_openloop_n1000.pt
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "uvit_experiments"))

import torch
import yaml
from pytorch_lightning import seed_everything

from calibrate_trajectory_qhat import dict2namespace
from qdiff.ddim_helpers import alpha_bar_at, ddim_update
from qdiff.trajectory_error import build_ddim_seq
from pipeline_loaders import add_backbone_args, load_quant_from_args
from sample_diffusion_ddim import get_beta_schedule


@torch.no_grad()
def collect_fullstep(
    quant_model,
    betas: torch.Tensor,
    seq,
    device: torch.device,
    *,
    num_trajectories: int,
    batch_size: int,
    eta: float = 0.0,
    channels: int = 3,
    image_size: int = 32,
    seed: int = 1234,
):
    quant_model.eval()
    seq_next = [-1] + list(seq[:-1])
    steps_order = list(zip(reversed(seq), reversed(seq_next)))
    n_grid_steps = len(steps_order)

    xs, eqs, efs, ts, traj_ids = [], [], [], [], []
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    n_done = 0

    while n_done < num_trajectories:
        cur_b = min(batch_size, num_trajectories - n_done)
        x = torch.randn(cur_b, channels, image_size, image_size, device=device, generator=rng)
        ids = torch.arange(n_done, n_done + cur_b, dtype=torch.int64, device="cpu")

        for i, j in steps_order:
            t = torch.full((cur_b,), float(i), device=device)
            t_next = torch.full((cur_b,), float(j), device=device)
            at = alpha_bar_at(betas, t.long(), device)
            at_next = alpha_bar_at(betas, t_next.long(), device)

            quant_model.set_quant_state(weight_quant=True, act_quant=True)
            eq = quant_model(x, t)
            quant_model.set_quant_state(weight_quant=False, act_quant=False)
            ef = quant_model(x, t)
            quant_model.set_quant_state(weight_quant=True, act_quant=True)

            xs.append(x.detach().cpu().half())
            eqs.append(eq.detach().cpu().half())
            efs.append(ef.detach().cpu().half())
            ts.append(t.detach().cpu().float())
            traj_ids.append(ids.clone())

            x = ddim_update(x, eq, at, at_next, eta=float(eta), generator=rng)

        n_done += cur_b
        print(f"  trajectories {n_done}/{num_trajectories}", flush=True)

    return {
        "x": torch.cat(xs, dim=0),
        "eq": torch.cat(eqs, dim=0),
        "ef": torch.cat(efs, dim=0),
        "t": torch.cat(ts, dim=0),
        "traj_id": torch.cat(traj_ids, dim=0).long(),
        "meta": {
            "num_trajectories": num_trajectories,
            "t_max": 999,
            "n_samples": int(torch.cat(ts, dim=0).shape[0]),
            "n_grid_steps": n_grid_steps,
            "closed_loop": False,
            "full_step": True,
            "channels": channels,
            "image_size": image_size,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_backbone_args(p)
    p.add_argument("--config", default="configs/cifar10.yml")
    p.add_argument("--ckpt", default="")
    p.add_argument("--cali_ckpt", default="cifar_w8a8_ckpt.pth")
    p.add_argument("--cali_data_path", default="cifar_sd1236_sample2048_allst.pt")
    p.add_argument("--ode_scale_json", default="")
    p.add_argument("--ode_absorb_mode", default="")
    p.add_argument("--brecq_ckpt", default="")
    p.add_argument("--num_trajectories", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--t_max", type=int, default=999, help="unused; kept for CLI compat")
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--skip_type", default="quad")
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output", required=True)
    p.add_argument("--split", action="store_true", default=True)
    p.add_argument("--quant_act", action="store_true", default=True)
    p.add_argument("--a_sym", action="store_true", default=True)
    p.add_argument("--weight_bit", type=int, default=8)
    p.add_argument("--act_bit", type=int, default=8)
    p.add_argument("--sm_abit", type=int, default=8)
    p.add_argument("--cali_st", type=int, default=10)
    p.add_argument("--cali_n", type=int, default=256)
    p.add_argument("--joint_sa_resume", action="store_true", default=False)
    args = p.parse_args()
    args.cond = False

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = None
    channels, image_size = 3, 32
    if args.backbone == "unet":
        with open(args.config, "r", encoding="utf-8") as f:
            config = dict2namespace(yaml.safe_load(f))
        config.split_shortcut = args.split
        channels = int(config.data.channels)
        image_size = int(config.data.image_size)

    betas_np = get_beta_schedule(
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        num_diffusion_timesteps=1000,
    )
    if config is not None:
        betas_np = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
    betas = torch.tensor(betas_np, dtype=torch.float32, device=device)
    seq = build_ddim_seq(len(betas_np), args.timesteps, args.skip_type)

    print(f"Loading {args.backbone} quant model on {device} ...", flush=True)
    quant_model = load_quant_from_args(args, device, config)
    gc.collect()

    print(
        f"Collecting {args.num_trajectories} open-loop trajectories, "
        f"{len(seq)} DDIM steps each ...",
        flush=True,
    )
    data = collect_fullstep(
        quant_model,
        betas,
        seq,
        device,
        num_trajectories=args.num_trajectories,
        batch_size=args.batch_size,
        eta=args.eta,
        channels=channels,
        image_size=image_size,
        seed=args.seed,
    )
    data["meta"]["backbone"] = args.backbone
    data["meta"]["timesteps"] = args.timesteps
    data["meta"]["skip_type"] = args.skip_type
    data["meta"]["seed"] = args.seed

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, out)
    print(
        f"Saved {data['meta']['n_samples']} samples ({data['meta']['n_grid_steps']} steps/traj) -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
