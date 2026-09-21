"""
End-to-end pipeline: full-step closed-loop data -> train no_aug corrector -> 50k DDIM sample.

Usage:
  python noise_eps_corr/scripts/run_w8a8_fullstep_pipeline.py
  python noise_eps_corr/scripts/run_w8a8_fullstep_pipeline.py --skip_collect --skip_train
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


def run_step(name: str, cmd: list[str], *, cwd: str = ROOT) -> None:
    print(f"\n{'=' * 72}\n[{name}]\n  {' '.join(cmd)}\n{'=' * 72}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    p = argparse.ArgumentParser(description="Full-step learned corrector pipeline")
    p.add_argument("--num_trajectories", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--t_max", type=int, default=999)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--warm_ckpt",
        type=str,
        default="output/noise_corr/learned_corr_cifar_w8a8/ckpt_best.pt",
        help="warm-start for collection + training",
    )
    p.add_argument(
        "--data",
        type=str,
        default="output/noise_corr/train_data/traj_5k_fullstep_cl_cifar_w8a8.pt",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="output/noise_corr/learned_corr_fullstep_cifar_w8a8",
    )
    p.add_argument(
        "--sample_logdir",
        type=str,
        default="output/cifar_w8a8_learned_fullstep",
    )
    p.add_argument("--max_images", type=int, default=50000)
    p.add_argument("--learned_corr_t_cut", type=int, default=999)
    p.add_argument("--learned_corr_alpha", type=float, default=0.5)
    p.add_argument("--lambda_t_ge_50", type=float, default=0.35)
    p.add_argument("--lambda_t_ge_200", type=float, default=0.15)
    p.add_argument("--collection_t_cut", type=int, default=999)
    p.add_argument("--skip_collect", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_sample", action="store_true")
    args = p.parse_args()

    ckpt_best = os.path.join(args.output_dir, "ckpt_best.pt")

    if not args.skip_collect:
        run_step(
            "collect_fullstep",
            [
                PY,
                COLLECT,
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
        run_step(
            "train_fullstep",
            [
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
            ],
        )

    if not args.skip_sample:
        if not os.path.isfile(ckpt_best):
            raise FileNotFoundError(f"Missing trained ckpt: {ckpt_best}")
        run_step(
            "sample_50k",
            [
                PY,
                "scripts/sample_diffusion_ddim.py",
                "--config",
                "configs/cifar10.yml",
                "--use_pretrained",
                "--timesteps",
                "100",
                "--eta",
                "0",
                "--skip_type",
                "quad",
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
                "1234",
                "-l",
                args.sample_logdir,
            ],
        )

    print(
        f"\nPipeline done @ {datetime.now().isoformat(timespec='seconds')}\n"
        f"  data   : {args.data}\n"
        f"  ckpt   : {ckpt_best}\n"
        f"  samples: {args.sample_logdir}\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
