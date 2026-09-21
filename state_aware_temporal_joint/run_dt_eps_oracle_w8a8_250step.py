#!/usr/bin/env python
"""W8A8 250-step eps-oracle δt: collect -> label 80k -> train dt_only -> refresh sample 50k.

Usage:
  python state_aware_temporal_joint/run_dt_eps_oracle_w8a8_250step.py
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
REAL = os.path.join(ROOT, "new_real_images/cifar10_python.npz")

COLLECT = "noise_eps_corr/scripts/collect_fullstep_training_data.py"
LABEL = "state_aware_temporal_joint/label_dt_eps_oracle.py"
TRAIN = "state_aware_temporal_joint/train_dt_eps_oracle.py"
SAMPLE = "state_aware_temporal_joint/sample_50k.py"


def run(name: str, cmd: list[str], log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    print(f"\n{'=' * 72}\n[{name}] {datetime.now().isoformat()}\n  {' '.join(cmd)}\n  log={log_path}\n{'=' * 72}", flush=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"CMD: {' '.join(cmd)}\n\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{name} failed exit={rc}; see {log_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cali_ckpt", default="cifar_w8a8_ckpt.pth")
    p.add_argument("--weight_bit", type=int, default=8)
    p.add_argument("--act_bit", type=int, default=8)
    p.add_argument("--timesteps", type=int, default=250)
    p.add_argument("--skip_type", default="quad")
    p.add_argument("--num_trajectories", type=int, default=2000)
    p.add_argument("--collect_batch_size", type=int, default=64)
    p.add_argument("--learned_corr_ckpt", default="noise_eps_corr/w8a8_corr_ckpt_best.pt",
                   help="empty string = open-loop collect")
    p.add_argument("--max_samples", type=int, default=80000)
    p.add_argument("--label_batch_size", type=int, default=256)
    p.add_argument("--window", type=int, default=40)
    p.add_argument("--n_grid", type=int, default=11)
    p.add_argument("--criterion", default="update_mse", choices=("mse", "l1", "smooth_l1", "update_mse"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--train_batch_size", type=int, default=256)
    p.add_argument("--dt_eta", type=float, default=0.5)
    p.add_argument("--t_cutoff", type=int, default=300)
    p.add_argument("--dt_refresh_n", type=int, default=8)
    p.add_argument("--max_images", type=int, default=50000)
    p.add_argument("--sample_batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--work_dir", default="state_aware_temporal_joint/output/dt_eps_oracle_w8a8_250step")
    p.add_argument(
        "--traj",
        default="output/noise_corr/train_data/traj_2k_fullstep_cl_cifar_w8a8_250step.pt",
    )
    p.add_argument("--dt_star", default="")
    p.add_argument("--run_dir", default="state_aware_temporal_joint/runs/dt_eps_oracle_w8a8_250step")
    p.add_argument("--sample_dir", default="")
    p.add_argument("--with_eps_corrector", action="store_true",
                   help="also apply ε corrector at inference")
    p.add_argument("--skip_collect", action="store_true")
    p.add_argument("--skip_label", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_sample", action="store_true")
    p.add_argument("--skip_fid", action="store_true")
    args = p.parse_args()

    work = args.work_dir
    os.makedirs(os.path.join(ROOT, work), exist_ok=True)
    tag = f"w{args.weight_bit}a{args.act_bit}"
    dt_star = args.dt_star or (
        f"output/noise_corr/train_data/dt_star_eps_oracle_{tag}_250step_{args.criterion}_w{args.window}.pt"
    )
    sample_dir = args.sample_dir or os.path.join(
        work, f"refresh_eta{args.dt_eta:g}_tc{args.t_cutoff}_50k"
    )
    run_dir = args.run_dir
    ckpt = os.path.join(run_dir, "ckpt_best.pt")

    ptq = [
        "--config", "configs/cifar10.yml",
        "--cali_ckpt", args.cali_ckpt,
        "--cali_data_path", "cifar_sd1236_sample2048_allst.pt",
        "--weight_bit", str(args.weight_bit),
        "--act_bit", str(args.act_bit),
        "--sm_abit", "8",
        "--cali_st", "10",
        "--cali_n", "256",
        "--quant_act",
        "--a_sym",
        "--split",
    ]

    if not args.skip_collect:
        traj_abs = os.path.join(ROOT, args.traj)
        if os.path.isfile(traj_abs):
            print(f"[collect] skip, exists: {args.traj}", flush=True)
        else:
            cmd = [
                PY, COLLECT, *ptq,
                "--num_trajectories", str(args.num_trajectories),
                "--batch_size", str(args.collect_batch_size),
                "--t_max", "999",
                "--timesteps", str(args.timesteps),
                "--skip_type", args.skip_type,
                "--output", args.traj,
            ]
            if args.learned_corr_ckpt:
                cmd += [
                    "--learned_corr_ckpt", args.learned_corr_ckpt,
                    "--collection_t_cut", "999",
                ]
            run("collect_250", cmd, os.path.join(work, "collect.log"))

    if not args.skip_label:
        if os.path.isfile(os.path.join(ROOT, dt_star)):
            print(f"[label] skip, exists: {dt_star}", flush=True)
        else:
            run(
                "label_250",
                [
                    PY, LABEL, *ptq,
                    "--data", args.traj,
                    "--output", dt_star,
                    "--window", str(args.window),
                    "--n_grid", str(args.n_grid),
                    "--criterion", args.criterion,
                    "--batch_size", str(args.label_batch_size),
                    "--max_samples", str(args.max_samples),
                ],
                os.path.join(work, "label.log"),
            )

    if not args.skip_train:
        if os.path.isfile(os.path.join(ROOT, ckpt)):
            print(f"[train] skip, exists: {ckpt}", flush=True)
        else:
            run(
                "train_250",
                [
                    PY, TRAIN,
                    "--data", args.traj,
                    "--dt_star", dt_star,
                    "--init_eps_ckpt", "",
                    "--output_dir", run_dir,
                    "--epochs", str(args.epochs),
                    "--batch_size", str(args.train_batch_size),
                    "--dt_only",
                    "--loss", "smooth_l1",
                ],
                os.path.join(work, "train.log"),
            )

    if not args.skip_sample:
        sample_cmd = [
            PY, SAMPLE,
            "--disable_adapter",
            "--dt_only_infer",
            "--dt_mode", "refresh",
            "--joint_dt_ckpt", ckpt,
            "--cali_ckpt", args.cali_ckpt,
            "--weight_bit", str(args.weight_bit),
            "--act_bit", str(args.act_bit),
            "--sm_abit", "8",
            "--timesteps", str(args.timesteps),
            "--skip_type", args.skip_type,
            "--dt_eta", str(args.dt_eta),
            "--dt_carry_max", "20",
            "--t_cutoff", str(args.t_cutoff),
            "--dt_refresh_n", str(args.dt_refresh_n),
            "--max_images", str(args.max_images),
            "--batch_size", str(args.sample_batch_size),
            "--seed", str(args.seed),
            "--output_dir", sample_dir,
        ]
        if args.with_eps_corrector and args.learned_corr_ckpt:
            sample_cmd += [
                "--allow_joint_plus_corrector",
                "--corrector_ckpt", args.learned_corr_ckpt,
            ]
        else:
            sample_cmd.append("--disable_corrector")
        run("sample_refresh_250", sample_cmd, os.path.join(work, "sample.log"))

    if not args.skip_fid:
        fid_txt = os.path.join(work, "fid_50k.txt")
        run(
            "fid",
            [PY, "-m", "pytorch_fid", REAL, os.path.join(ROOT, sample_dir), "--device", "cuda:0"],
            fid_txt,
        )
        text = open(os.path.join(ROOT, fid_txt), encoding="utf-8").read()
        m = re.search(r"FID:\s*([0-9.eE+-]+)", text)
        fid = float(m.group(1)) if m else None
        print(
            f"\n{tag.upper()} 250-step refresh FID={fid}\n"
            f"  sample_dir={sample_dir}\n  ckpt={ckpt}\n",
            flush=True,
        )


if __name__ == "__main__":
    main()
