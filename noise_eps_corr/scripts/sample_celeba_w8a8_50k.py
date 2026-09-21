"""
Sample 50k CelebA 64x64 images with W8A8 quant UNet + learned noise corrector.

Example (from repo root, after placing ckpts):
  python noise_eps_corr/scripts/sample_celeba_w8a8_50k.py \\
    --ckpt assets/celeba_64/ckpt.pth \\
    --cali_ckpt celeba_w8a8_ckpt.pth \\
    --cali_data_path celeba_sd1236_sample2048_allst.pt \\
    --learned_corr_ckpt learned_corr_fullstep_celeba_w8a8/ckpt_best.pt \\
    --output_dir learned_corr_fullstep_celeba_w8a8/samples
"""

from __future__ import annotations

import argparse
import gc
import glob
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch
import torchvision.utils as tvu
import yaml
from pytorch_lightning import seed_everything
from tqdm import tqdm

from calibrate_trajectory_qhat import dict2namespace, load_float_model, load_quant_model
from ddim.datasets import inverse_data_transform
from ddim.functions.denoising import generalized_steps
from noise_eps_corr.learned_noise_corrector import load_corrector_from_ckpt
from qdiff.trajectory_error import build_ddim_seq
from sample_diffusion_ddim import get_beta_schedule


def _require_file(path: str, label: str) -> str:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing {label}: {path}\n"
            "CelebA DDIM float ckpt is usually from DDIM authors (Google Drive).\n"
            "Place celeba_w8a8_ckpt.pth and cali .pt in repo root or pass explicit paths."
        )
    return path


@torch.no_grad()
def sample_images(
    model,
    corrector,
    betas,
    seq,
    device,
    config,
    *,
    batch_size,
    max_images,
    eta,
    seed,
    channels,
    image_size,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)
    existing = len(glob.glob(os.path.join(output_dir, "*.png")))
    if existing >= max_images:
        print(f"Already have {existing} images >= {max_images}, skip.", flush=True)
        return existing

    betas_t = betas.to(device)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    img_id = existing
    n_rounds = math.ceil((max_images - img_id) / batch_size)
    for _ in tqdm(range(n_rounds), desc="sample rounds"):
        n = min(batch_size, max_images - img_id)
        x = torch.randn(
            n, channels, image_size, image_size, device=device, generator=rng
        )
        xs, _ = generalized_steps(
            x,
            seq,
            model,
            betas_t,
            eta=eta,
            noise_corrector=corrector,
        )
        x_out = xs[-1].to(device)
        x_out = inverse_data_transform(config, x_out)
        for i in range(n):
            tvu.save_image(x_out[i], os.path.join(output_dir, f"{img_id}.png"))
            img_id += 1
    return img_id


def main():
    p = argparse.ArgumentParser(description="CelebA W8A8 + learned corrector 50k sampling")
    p.add_argument("--config", default="configs/celeba.yml")
    p.add_argument("--ckpt", default="assets/celeba_64/ckpt.pth")
    p.add_argument("--cali_ckpt", default="celeba_w8a8_ckpt.pth")
    p.add_argument("--cali_data_path", default="celeba_sd1236_sample2048_allst.pt")
    p.add_argument(
        "--learned_corr_ckpt",
        default="learned_corr_fullstep_celeba_w8a8/ckpt_best.pt",
    )
    p.add_argument("--output_dir", default="learned_corr_fullstep_celeba_w8a8/samples")
    p.add_argument("--max_images", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--skip_type", choices=["uniform", "quad"], default="quad")
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--learned_corr_t_cut", type=int, default=999)
    p.add_argument("--learned_corr_alpha", type=float, default=0.5)
    p.add_argument("--weight_bit", type=int, default=8)
    p.add_argument("--act_bit", type=int, default=8)
    p.add_argument("--sm_abit", type=int, default=8)
    p.add_argument("--cali_st", type=int, default=10)
    p.add_argument("--cali_n", type=int, default=256)
    p.add_argument("--split", action="store_true", default=True)
    p.add_argument("--quant_act", action="store_true", default=True)
    p.add_argument("--a_sym", action="store_true", default=True)
    args = p.parse_args()
    args.cond = False
    args.joint_sa_resume = False
    args.ptq = True
    args.resume = True

    args.ckpt = _require_file(args.ckpt, "float CelebA UNet ckpt")
    args.cali_ckpt = _require_file(args.cali_ckpt, "CelebA W8A8 cali ckpt")
    args.cali_data_path = _require_file(args.cali_data_path, "CelebA PTQ cali data")
    args.learned_corr_ckpt = _require_file(args.learned_corr_ckpt, "learned corrector ckpt")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(_ROOT, args.config), "r", encoding="utf-8") as f:
        config = dict2namespace(yaml.safe_load(f))
    config.split_shortcut = args.split

    betas_np = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.tensor(betas_np, dtype=torch.float32)
    seq = build_ddim_seq(len(betas_np), args.timesteps, args.skip_type)

    print(f"Loading float + W{args.weight_bit}A{args.act_bit} models on {device} ...", flush=True)
    float_model = load_float_model(config, device, args)
    quant_model = load_quant_model(config, device, args, float_model)
    del float_model
    gc.collect()

    corrector = load_corrector_from_ckpt(args.learned_corr_ckpt, device)
    corrector.meta.t_cut = args.learned_corr_t_cut
    corrector.meta.alpha = args.learned_corr_alpha
    corrector.train_mode_off()
    print(
        f"Corrector: t_cut={corrector.meta.t_cut}, alpha={corrector.meta.alpha}",
        flush=True,
    )

    out_dir = os.path.join(_ROOT, args.output_dir)
    n_done = sample_images(
        quant_model,
        corrector,
        betas,
        seq,
        device,
        config,
        batch_size=args.batch_size,
        max_images=args.max_images,
        eta=args.eta,
        seed=args.seed,
        channels=int(config.data.channels),
        image_size=int(config.data.image_size),
        output_dir=out_dir,
    )
    print(f"Done. Saved {n_done} images -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
