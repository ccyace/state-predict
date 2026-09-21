#!/usr/bin/env python
"""Offline epsilon-oracle dt* labels for the traj dataset.

For each (x, ef, t): search τ in a window so that quantized UNet eps_Q(x,τ)
best matches the stored FP label ef under mse / l1 / smooth_l1.

Example:
  python state_aware_temporal_joint/label_dt_eps_oracle.py \\
    --criterion smooth_l1 --window 40 --n_grid 11 --batch_size 256 \\
    --output output/noise_corr/train_data/dt_star_eps_oracle_smoothl1_w40.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch
import yaml
from pytorch_lightning import seed_everything

from calibrate_trajectory_qhat import dict2namespace
from state_aware_temporal_joint.eps_oracle_dt import eps_oracle_dt_star
from sample_diffusion_ddim import get_beta_schedule


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="output/noise_corr/train_data/traj_5k_fullstep_cl_cifar_w8a8.pt")
    p.add_argument(
        "--output",
        default="output/noise_corr/train_data/dt_star_eps_oracle_smoothl1_w40.pt",
    )
    p.add_argument("--config", default="configs/cifar10.yml")
    p.add_argument("--cali_ckpt", default="cifar_w8a8_ckpt.pth")
    p.add_argument("--cali_data_path", default="cifar_sd1236_sample2048_allst.pt")
    p.add_argument("--window", type=int, default=40)
    p.add_argument("--n_grid", type=int, default=0, help="0 => window+1")
    p.add_argument("--t_cutoff", type=int, default=0)
    p.add_argument("--dt_max", type=float, default=20.0)
    p.add_argument(
        "--criterion",
        choices=("mse", "l1", "smooth_l1", "update_mse"),
        default="update_mse",
        help="distance for matching eps_Q(x,τ) to ef",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--split", action="store_true", default=True)
    p.add_argument("--quant_act", action="store_true", default=True)
    p.add_argument("--a_sym", action="store_true", default=True)
    p.add_argument("--weight_bit", type=int, default=8)
    p.add_argument("--act_bit", type=int, default=8)
    p.add_argument("--sm_abit", type=int, default=8)
    p.add_argument("--cali_st", type=int, default=10)
    p.add_argument("--cali_n", type=int, default=256)
    p.add_argument("--ckpt", default="")
    p.add_argument("--ode_scale_json", default="")
    p.add_argument("--ode_absorb_mode", default="")
    p.add_argument("--backbone", choices=("unet", "uvit"), default="unet")
    p.add_argument("--fp_ckpt", default="")
    return p.parse_args()


def main():
    args = parse_args()
    args.cond = False
    args.joint_sa_resume = False
    args.brecq_ckpt = ""
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        config = dict2namespace(yaml.safe_load(f))
    config.split_shortcut = args.split

    print("Loading models...", flush=True)
    if args.backbone == "uvit":
        sys.path.insert(0, os.path.join(_ROOT, "uvit_experiments"))
        from pipeline_loaders import load_quant_from_args

        qnn = load_quant_from_args(args, device)
    else:
        from calibrate_trajectory_qhat import load_float_model, load_quant_model

        fp = load_float_model(config, device, args)
        qnn = load_quant_model(config, device, args, fp)
        del fp
        qnn.set_quant_state(True, True)
        qnn.eval()

    print(f"Loading traj mmap: {args.data}", flush=True)
    raw = torch.load(args.data, map_location="cpu", mmap=True)
    n = int(raw["t"].shape[0])
    if args.max_samples and args.max_samples > 0:
        n = min(n, int(args.max_samples))
    n_grid = args.n_grid if args.n_grid > 0 else (args.window + 1)
    betas = torch.tensor(
        get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        ), device=device, dtype=torch.float32,
    )
    # Recover the actual sampler grid from the collected trajectory instead of
    # assuming a particular number/spacing of DDIM steps.
    unique_t = sorted({float(v) for v in raw["t"][:n].tolist()}, reverse=True)
    next_by_t = {tv: (unique_t[k + 1] if k + 1 < len(unique_t) else -1.0)
                 for k, tv in enumerate(unique_t)}
    print(
        f"Labeling n={n} window={args.window} n_grid={n_grid} "
        f"batch={args.batch_size} t_cutoff={args.t_cutoff} criterion={args.criterion}",
        flush=True,
    )

    dt_all = torch.empty(n, dtype=torch.float32)
    eps_curve_all = torch.empty(n, n_grid, dtype=torch.float32)
    state_curve_all = torch.empty(n, n_grid, dtype=torch.float32)
    started = time.time()
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        x = raw["x"][start:end].float().to(device)
        ef = raw["ef"][start:end].float().to(device)
        t = raw["t"][start:end].float().to(device)
        t_next = torch.tensor([next_by_t[float(v)] for v in t.detach().cpu().tolist()], device=device)
        result = eps_oracle_dt_star(
            qnn, x, ef, t,
            window=args.window,
            n_grid=n_grid,
            t_cutoff=args.t_cutoff,
            criterion=args.criterion,
            betas=betas,
            t_next=t_next,
            return_curves=True,
        )
        dt = result["dt_star"]
        dt_all[start:end] = dt.detach().cpu().clamp(-args.dt_max, args.dt_max)
        eps_curve_all[start:end] = result["eps_mse_curve"].detach().cpu()
        state_curve_all[start:end] = result["state_mse_curve"].detach().cpu()
        if (start // args.batch_size) % 20 == 0 or end >= n:
            done = end
            rate = done / max(time.time() - started, 1e-6)
            eta = (n - done) / max(rate, 1e-6)
            m = dt_all[:done]
            print(
                f"  {done}/{n} rate={rate:.1f}/s eta={eta/60:.1f}min "
                f"dt_mean={m.mean():.3f} dt_std={m.std():.3f} "
                f"|dt|mean={m.abs().mean():.3f} edge%={(m.abs()>=args.dt_max-0.5).float().mean()*100:.1f}",
                flush=True,
            )

    payload = {
        "dt_star": dt_all,
        "eps_mse_curve": eps_curve_all,
        "state_mse_curve": state_curve_all,
        "offsets": torch.linspace(-args.window // 2, args.window // 2, n_grid),
        "meta": {
            "source_data": args.data,
            "n": n,
            "window": args.window,
            "n_grid": n_grid,
            "t_cutoff": args.t_cutoff,
            "dt_max": args.dt_max,
            "criterion": args.criterion,
            "method": f"eps_{args.criterion}_oracle_fp_label",
        },
    }
    torch.save(payload, args.output)
    with open(args.output + ".stats.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **payload["meta"],
                "dt_mean": float(dt_all.mean()),
                "dt_std": float(dt_all.std()),
                "dt_abs_mean": float(dt_all.abs().mean()),
                "edge_frac": float((dt_all.abs() >= args.dt_max - 0.5).float().mean()),
                "elapsed_sec": time.time() - started,
            },
            f,
            indent=2,
        )
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
