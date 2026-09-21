#!/usr/bin/env python
"""U-ViT-S/2 CIFAR-10 FP sampling with DDIM 100 quad (aligned with PTQD protocol)."""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time

import torch
import torchvision.utils as tvu
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qdiff.ddim_helpers import ddim_update
from qdiff.trajectory_error import build_ddim_seq
from sample_diffusion_ddim import get_beta_schedule
from state_aware_temporal_joint.sample_50k import _alpha_bar_batch, run_fid
from uvit_loader import DEFAULT_CKPT, load_uvit_fp, load_uvit_quant


def _to_uint8_images(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] latent/image -> [0,1] for save_image."""
    return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)


def existing_count(output_dir: str, max_images: int) -> int:
    if not os.path.isdir(output_dir):
        return 0
    n = len(glob.glob(os.path.join(output_dir, "*.png")))
    return min(n, max_images)


@torch.no_grad()
def ddim_sample_fp(
    model,
    x: torch.Tensor,
    betas: torch.Tensor,
    *,
    timesteps: int = 100,
    skip_type: str = "quad",
    eta: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    device = x.device
    seq = build_ddim_seq(len(betas), timesteps, skip_type)
    seq_next = [-1] + list(seq[:-1])
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = torch.full((x.shape[0],), float(i), device=device)
        t_next = torch.full((x.shape[0],), float(j), device=device)
        at = _alpha_bar_batch(betas, t)
        at_next = _alpha_bar_batch(betas, t_next)
        eps = model(x, t)
        x = ddim_update(x, eps, at, at_next, eta=float(eta), generator=generator)
    return x


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--output_dir", default="uvit_experiments/outputs/phase0_fp_ddim100_eta1_50k")
    p.add_argument("--num_images", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--skip_type", default="quad")
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--fid_ref", default="new_real_images/real47500_vsc2500_fid_stats.npz")
    p.add_argument("--log_dir", default="uvit_experiments/outputs/phase0_fp_ddim100_eta1_50k/logs")
    p.add_argument("--skip_fid", action="store_true")
    p.add_argument("--smoke", action="store_true", help="64 images, random-init model (pipeline test only)")
    p.add_argument("--cali_ckpt", default="", help="quantized PTQ ckpt (enables W8A8 inference)")
    p.add_argument("--cali_data_path", default="cifar_sd1236_sample2048_allst.pt")
    p.add_argument("--weight_bit", type=int, default=8)
    p.add_argument("--act_bit", type=int, default=8)
    p.add_argument("--sm_abit", type=int, default=8)
    p.add_argument("--cali_st", type=int, default=10)
    p.add_argument("--cali_n", type=int, default=256)
    p.add_argument("--quant_act", action="store_true", default=True)
    p.add_argument("--a_sym", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.num_images = 64
        args.batch_size = 16
        args.skip_fid = True
        args.output_dir = "uvit_experiments/outputs/smoke_random_init"

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke:
        from uvit_loader import UViTEpsModel, build_uvit

        model = UViTEpsModel(build_uvit()).to(device).eval()
        print("[smoke] random-init U-ViT (no checkpoint)", flush=True)
    else:
        if args.cali_ckpt:
            model = load_uvit_quant(
                args.cali_ckpt,
                os.path.join(_ROOT, args.cali_data_path),
                device,
                fp_ckpt=args.ckpt,
                weight_bit=args.weight_bit,
                act_bit=args.act_bit,
                sm_abit=args.sm_abit,
                cali_st=args.cali_st,
                cali_n=args.cali_n,
                quant_act=args.quant_act,
                a_sym=args.a_sym,
            )
            print(f"[quant] loaded {args.cali_ckpt}", flush=True)
        else:
            model = load_uvit_fp(args.ckpt, device)

    cfg = yaml.safe_load(open(os.path.join(_ROOT, "configs/cifar10.yml")))
    betas = torch.tensor(
        get_beta_schedule(
            beta_schedule=cfg["diffusion"]["beta_schedule"],
            beta_start=cfg["diffusion"]["beta_start"],
            beta_end=cfg["diffusion"]["beta_end"],
            num_diffusion_timesteps=cfg["diffusion"]["num_diffusion_timesteps"],
        ),
        dtype=torch.float32,
        device=device,
    )

    start_id = existing_count(args.output_dir, args.num_images)
    if start_id >= args.num_images:
        print(f"Already have {start_id} images in {args.output_dir}", flush=True)
    else:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(args.seed))
        t0 = time.time()
        image_id = start_id
        while image_id < args.num_images:
            bs = min(args.batch_size, args.num_images - image_id)
            x = torch.randn(bs, 3, 32, 32, device=device, generator=gen)
            x = ddim_sample_fp(
                model, x, betas,
                timesteps=args.timesteps,
                skip_type=args.skip_type,
                eta=args.eta,
                generator=gen,
            )
            images = _to_uint8_images(x.cpu())
            for k in range(bs):
                tvu.save_image(images[k], os.path.join(args.output_dir, f"{image_id}.png"))
                image_id += 1
            if image_id % 512 == 0 or image_id == args.num_images:
                elapsed = time.time() - t0
                rate = image_id / max(elapsed, 1e-6)
                print(f"  {image_id}/{args.num_images} ({rate:.2f} img/s)", flush=True)

    if not args.skip_fid:
        fid_log = "fid_w8a8_ddim100_eta1.log" if args.cali_ckpt else "fid_fp_ddim100_eta1.log"
        fid_txt = "fid_w8a8_ddim100_eta1.txt" if args.cali_ckpt else "fid_fp_ddim100_eta1.txt"
        fid = run_fid(
            args.output_dir,
            os.path.join(_ROOT, args.fid_ref),
            str(device),
            os.path.join(args.log_dir, fid_log),
        )
        print(f"FID = {fid:.4f}", flush=True)
        with open(os.path.join(args.log_dir, fid_txt), "w") as f:
            f.write(f"FID: {fid:.4f}\n")


if __name__ == "__main__":
    main()
