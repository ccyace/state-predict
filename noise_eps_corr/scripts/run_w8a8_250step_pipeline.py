"""
250-step DDIM W8A8 noise corrector pipeline (resource-light defaults).

Collect closed-loop traj (250-step quad) -> train DeltaEpsNet -> DDIM sample.

Defaults tuned for time/storage vs full 5000-traj / 50-epoch / 100-step setup:
  - 3000 trajectories (~690k samples, ~230 grid steps each)
  - batch_size 128 (collect + train)
  - 35 epochs, warm-start, mmap, best_by val_loss
  - MSE + cos + sr loss (lambda_mse=1.0, lambda_cos=2.0, lambda_sr=0.3)
  - 250-step quad at collect and sample

Usage:
  python noise_eps_corr/scripts/run_w8a8_250step_pipeline.py
  python noise_eps_corr/scripts/run_w8a8_250step_pipeline.py --max_images 5000
  python noise_eps_corr/scripts/run_w8a8_250step_pipeline.py --skip_collect --skip_train
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable
COLLECT = "noise_eps_corr/scripts/collect_fullstep_training_data.py"
TRAIN = "noise_eps_corr/scripts/train_corrector.py"


def run_step(name: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 72}\n[{name}]\n  {' '.join(cmd)}\n{'=' * 72}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description="W8A8 250-step DDIM corrector pipeline (lightweight)")
    p.add_argument("--num_trajectories", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--timesteps", type=int, default=250)
    p.add_argument("--skip_type", type=str, default="quad")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument(
        "--warm_ckpt",
        type=str,
        default="output/noise_corr/learned_corr_fullstep_cifar_w8a8/ckpt_best.pt",
        help="warm-start for closed-loop collection + DeltaEpsNet init",
    )
    p.add_argument(
        "--data",
        type=str,
        default="output/noise_corr/train_data/traj_3k_fullstep_cl_cifar_w8a8_250step.pt",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="output/noise_corr/learned_corr_fullstep_cifar_w8a8_250step",
    )
    p.add_argument(
        "--sample_logdir",
        type=str,
        default="output/cifar_w8a8_250step_corr",
    )
    p.add_argument("--max_images", type=int, default=50000)
    p.add_argument("--learned_corr_t_cut", type=int, default=999)
    p.add_argument("--learned_corr_alpha", type=float, default=0.5)
    p.add_argument("--collection_t_cut", type=int, default=999)
    p.add_argument("--t_max", type=int, default=999)
    p.add_argument("--lambda_t_ge_50", type=float, default=0.35)
    p.add_argument("--lambda_t_ge_200", type=float, default=0.15)
    p.add_argument("--lambda_mse", type=float, default=1.0)
    p.add_argument("--lambda_cos", type=float, default=2.0)
    p.add_argument("--lambda_sr", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no_mmap", action="store_true")
    p.add_argument("--skip_collect", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_sample", action="store_true")
    args = p.parse_args()

    ckpt_best = os.path.join(args.output_dir, "ckpt_best.pt")

    if not args.skip_collect:
        if not os.path.isfile(os.path.join(ROOT, args.warm_ckpt)):
            raise FileNotFoundError(f"Missing warm ckpt: {args.warm_ckpt}")
        run_step(
            "collect_250step",
            [
                PY,
                COLLECT,
                "--timesteps",
                str(args.timesteps),
                "--skip_type",
                args.skip_type,
                "--learned_corr_ckpt",
                args.warm_ckpt,
                "--collection_t_cut",
                str(args.collection_t_cut),
                "--num_trajectories",
                str(args.num_trajectories),
                "--batch_size",
                str(args.batch_size),
                "--t_max",
                str(args.t_max),
                "--output",
                args.data,
            ],
        )

    if not args.skip_train:
        if not os.path.isfile(os.path.join(ROOT, args.data)):
            raise FileNotFoundError(f"Missing training data: {args.data}")
        train_cmd = [
            PY,
            TRAIN,
            "--data",
            args.data,
            "--output_dir",
            args.output_dir,
            "--init_ckpt",
            args.warm_ckpt,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--t_cut",
            str(args.learned_corr_t_cut),
            "--alpha",
            str(args.learned_corr_alpha),
            "--lambda_t_ge_50",
            str(args.lambda_t_ge_50),
            "--lambda_t_ge_200",
            str(args.lambda_t_ge_200),
            "--lambda_mse",
            str(args.lambda_mse),
            "--lambda_cos",
            str(args.lambda_cos),
            "--lambda_sr",
            str(args.lambda_sr),
            "--best_by",
            "val_loss",
        ]
        if not args.no_mmap:
            train_cmd.append("--mmap")
        run_step("train_250step", train_cmd)

    if not args.skip_sample:
        if not os.path.isfile(os.path.join(ROOT, ckpt_best)):
            raise FileNotFoundError(f"Missing trained ckpt: {ckpt_best}")
        run_step(
            "sample",
            [
                PY,
                "scripts/sample_diffusion_ddim.py",
                "--config",
                "configs/cifar10.yml",
                "--use_pretrained",
                "--timesteps",
                str(args.timesteps),
                "--eta",
                "0",
                "--skip_type",
                args.skip_type,
                "--ptq",
                "--resume",
                "--split",
                "--a_sym",
                "--quant_act",
                "--weight_bit",
                "8",
                "--act_bit",
                "8",
                "--cali_ckpt",
                "cifar_w8a8_ckpt.pth",
                "--cali_data_path",
                "cifar_sd1236_sample2048_allst.pt",
                "--enable_learned_noise_corr",
                "--learned_corr_ckpt",
                ckpt_best,
                "--learned_corr_t_cut",
                str(args.learned_corr_t_cut),
                "--learned_corr_alpha",
                str(args.learned_corr_alpha),
                "--max_images",
                str(args.max_images),
                "--seed",
                str(args.seed),
                "-l",
                args.sample_logdir,
            ],
        )

    print(
        f"\n250-step pipeline done @ {datetime.now().isoformat(timespec='seconds')}\n"
        f"  trajectories : {args.num_trajectories}\n"
        f"  ddim steps   : {args.timesteps} ({args.skip_type})\n"
        f"  data         : {args.data}\n"
        f"  ckpt         : {ckpt_best}\n"
        f"  samples      : {args.sample_logdir}\n"
        f"  loss         : mse={args.lambda_mse} cos={args.lambda_cos} sr={args.lambda_sr}\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
