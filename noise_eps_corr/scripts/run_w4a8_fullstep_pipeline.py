"""
W4A8 full-schedule learned noise corrector pipeline:
  collect (closed-loop full-step) -> train (t_cut=999) -> sample 50k

Bootstrap note: first-pass collection uses late ckpt with collection_t_cut=50 so
high-noise DDIM steps stay uncorrected during rollout. After training a full-step
ckpt, optionally re-collect with --warm_ckpt <fullstep ckpt> --collection_t_cut 999.

Usage:
  python noise_eps_corr/scripts/run_w4a8_fullstep_pipeline.py
  python noise_eps_corr/scripts/run_w4a8_fullstep_pipeline.py --skip_collect --skip_train
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

W4A8_CALI_CKPT = "output/ode_ptq_w4a8_dilate/samples/2026-06-29-08-34-47/ckpt.pth"
W4A8_WARM_CKPT = "output/noise_corr/learned_corr_w4a8/ckpt_best.pt"
# Late ckpt is trained for t<50 only. During closed-loop collection, t_cut must match
# the warm ckpt domain; forcing 999 applies out-of-domain corrections on high-noise
# steps and poisons x_t for all later saved samples (W4A8 is more sensitive than W8A8).
W4A8_COLLECTION_T_CUT = 50


def run_step(name: str, cmd: list[str], *, cwd: str = ROOT) -> None:
    print(f"\n{'=' * 72}\n[{name}]\n  {' '.join(cmd)}\n{'=' * 72}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    p = argparse.ArgumentParser(description="W4A8: full-step collect -> train -> sample")
    p.add_argument("--num_trajectories", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=64, help="collection batch size")
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--warm_ckpt", type=str, default=W4A8_WARM_CKPT)
    p.add_argument(
        "--data",
        type=str,
        default="D:/qdiff_data/train_data/traj_5k_fullstep_cl_cifar_w4a8.pt",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="output/noise_corr/learned_corr_fullstep_cifar_w4a8",
    )
    p.add_argument(
        "--sample_logdir",
        type=str,
        default="output/cifar_w4a8_learned_corr_fullstep",
    )
    p.add_argument("--max_images", type=int, default=50000)
    p.add_argument("--learned_corr_t_cut", type=int, default=999)
    p.add_argument("--learned_corr_alpha", type=float, default=0.5)
    p.add_argument(
        "--collection_t_cut",
        type=int,
        default=W4A8_COLLECTION_T_CUT,
        help="corrector t_cut during closed-loop collection (default 50 = warm late ckpt domain)",
    )
    p.add_argument("--cali_ckpt", type=str, default=W4A8_CALI_CKPT)
    p.add_argument("--skip_collect", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_sample", action="store_true")
    args = p.parse_args()

    ckpt_best = os.path.join(args.output_dir, "ckpt_best.pt")
    common_ptq = [
        "--config", "configs/cifar10.yml",
        "--cali_ckpt", args.cali_ckpt,
        "--ode_scale_json", "ode_pre_scaling.json",
        "--cali_data_path", "cifar_sd1236_sample2048_allst.pt",
        "--weight_bit", "4",
        "--act_bit", "8",
        "--cali_st", "10",
        "--cali_n", "256",
    ]

    if not args.skip_collect:
        if not os.path.isfile(args.warm_ckpt):
            raise FileNotFoundError(f"Warm ckpt for collection not found: {args.warm_ckpt}")
        run_step(
            "collect_w4a8_fullstep",
            [
                PY,
                COLLECT,
                *common_ptq,
                "--learned_corr_ckpt",
                args.warm_ckpt,
                "--collection_t_cut",
                str(args.collection_t_cut),
                "--num_trajectories",
                str(args.num_trajectories),
                "--batch_size",
                str(args.batch_size),
                "--t_max",
                "999",
                "--output",
                args.data,
            ],
        )

    if not args.skip_train:
        if not os.path.isfile(args.data):
            raise FileNotFoundError(f"Training data not found: {args.data}")
        run_step(
            "train_w4a8_fullstep_corrector",
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
                str(args.train_batch_size),
                "--mmap",
                "--t_cut",
                str(args.learned_corr_t_cut),
                "--alpha",
                str(args.learned_corr_alpha),
                "--lambda_t_ge_50",
                "0.35",
                "--lambda_t_ge_200",
                "0.15",
            ],
        )

    if not args.skip_sample:
        if not os.path.isfile(ckpt_best):
            raise FileNotFoundError(f"Missing trained ckpt: {ckpt_best}")
        run_step(
            "sample_w4a8_50k",
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
                "4",
                "--act_bit",
                "8",
                "--ode_scale_json",
                "ode_pre_scaling.json",
                "--cali_ckpt",
                args.cali_ckpt,
                "--cali_data_path",
                "cifar_sd1236_sample2048_allst.pt",
                "--cali_st",
                "10",
                "--cali_n",
                "256",
                "--enable_learned_noise_corr",
                "--learned_corr_ckpt",
                ckpt_best,
                "--learned_corr_t_cut",
                str(args.learned_corr_t_cut),
                "--learned_corr_alpha",
                str(args.learned_corr_alpha),
                "--learned_corr_rmin",
                "0.8",
                "--learned_corr_rmax",
                "1.25",
                "--max_images",
                str(args.max_images),
                "--seed",
                "1234",
                "-l",
                args.sample_logdir,
            ],
        )

    print(
        f"\nW4A8 full-step pipeline done @ {datetime.now().isoformat(timespec='seconds')}\n"
        f"  data   : {args.data}\n"
        f"  ckpt   : {ckpt_best}\n"
        f"  samples: {args.sample_logdir}\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
