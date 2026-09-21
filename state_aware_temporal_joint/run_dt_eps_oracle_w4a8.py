#!/usr/bin/env python
"""W4A8 eps-oracle δt pipeline: collect traj -> label dt* -> train dt_only.

Usage:
  python state_aware_temporal_joint/run_dt_eps_oracle_w4a8.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

COLLECT = "noise_eps_corr/scripts/collect_fullstep_training_data.py"
LABEL = "state_aware_temporal_joint/label_dt_eps_oracle.py"
TRAIN = "state_aware_temporal_joint/train_dt_eps_oracle.py"


def run(name: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 72}\n[{name}] {datetime.now().isoformat()}\n  {' '.join(cmd)}\n{'=' * 72}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cali_ckpt", default="cifar_w4a8_ckpt.pth")
    p.add_argument("--ode_scale_json", default="", help="set ode_pre_scaling.json if ckpt needs it")
    p.add_argument("--num_trajectories", type=int, default=5000)
    p.add_argument("--collect_batch_size", type=int, default=64)
    p.add_argument("--label_batch_size", type=int, default=32)
    p.add_argument("--window", type=int, default=40)
    p.add_argument("--criterion", default="update_mse", choices=("mse", "l1", "smooth_l1", "update_mse"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--train_batch_size", type=int, default=256)
    p.add_argument("--work_dir", default="state_aware_temporal_joint/output/dt_eps_oracle_w4a8")
    p.add_argument("--traj", default="output/noise_corr/train_data/traj_5k_fullstep_ol_cifar_w4a8.pt")
    p.add_argument("--dt_star", default="")
    p.add_argument("--run_dir", default="state_aware_temporal_joint/runs/dt_eps_oracle_w4a8")
    p.add_argument("--skip_collect", action="store_true")
    p.add_argument("--skip_label", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    args = p.parse_args()

    os.makedirs(os.path.join(ROOT, args.work_dir), exist_ok=True)
    dt_star = args.dt_star or os.path.join(
        "output/noise_corr/train_data",
        f"dt_star_eps_oracle_w4a8_{args.criterion}_w{args.window}.pt",
    )

    ptq = [
        "--config", "configs/cifar10.yml",
        "--cali_ckpt", args.cali_ckpt,
        "--cali_data_path", "cifar_sd1236_sample2048_allst.pt",
        "--weight_bit", "4",
        "--act_bit", "8",
        "--sm_abit", "8",
        "--cali_st", "10",
        "--cali_n", "256",
        "--quant_act",
        "--a_sym",
        "--split",
    ]
    if args.ode_scale_json:
        ptq += ["--ode_scale_json", args.ode_scale_json]

    if not args.skip_collect:
        if os.path.isfile(os.path.join(ROOT, args.traj)):
            print(f"[collect] skip, exists: {args.traj}", flush=True)
        else:
            # open-loop (no warm corrector on this server)
            run(
                "collect_w4a8_ol",
                [
                    PY, COLLECT, *ptq,
                    "--num_trajectories", str(args.num_trajectories),
                    "--batch_size", str(args.collect_batch_size),
                    "--t_max", "999",
                    "--timesteps", "100",
                    "--skip_type", "quad",
                    "--output", args.traj,
                ],
            )

    if not args.skip_label:
        if os.path.isfile(os.path.join(ROOT, dt_star)):
            print(f"[label] skip, exists: {dt_star}", flush=True)
        else:
            run(
                "label_dt_eps_oracle_w4a8",
                [
                    PY, LABEL, *ptq,
                    "--data", args.traj,
                    "--output", dt_star,
                    "--window", str(args.window),
                    "--criterion", args.criterion,
                    "--batch_size", str(args.label_batch_size),
                ],
            )

    if not args.skip_train:
        run(
            "train_dt_eps_oracle_w4a8",
            [
                PY, TRAIN,
                "--data", args.traj,
                "--dt_star", dt_star,
                "--init_eps_ckpt", "",
                "--output_dir", args.run_dir,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.train_batch_size),
                "--dt_only",
                "--t_cutoff", "0",
                "--loss", "smooth_l1",
            ],
        )

    print(
        f"\nDone @ {datetime.now().isoformat()}\n"
        f"  traj={args.traj}\n  dt_star={dt_star}\n  ckpt={args.run_dir}/ckpt_best.pt\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
