#!/usr/bin/env python
"""W8A8 CIFAR-10 DDIM sampling with optional δt refresh, mean residual head, and VSC."""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torchvision.utils as tvu
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from calibrate_trajectory_qhat import dict2namespace
from ddim.datasets import inverse_data_transform
from ddim.functions.denoising import compute_alpha
from noise_eps_corr.learned_noise_corrector import load_learned_corrector
from qdiff.ddim_helpers import ddim_update, ddim_update_vsc
from qdiff.joint_eps_dt_corrector import load_joint_corrector
from qdiff.trajectory_error import build_ddim_seq
from sample_diffusion_ddim import get_beta_schedule
from state_aware_temporal_joint.joint_feature_corrector import JointFeatureHooks, load_feature_corrector
from state_aware_temporal_joint.temporal_path import keep_temporal_path_float
from state_aware_temporal_joint.time_corrected_residual import load_residual

DEFAULT_FID_REF = os.path.join(_ROOT, "new_real_images/cifar10_python.npz")


def _refresh_step_indices(seq: Sequence[int], t_cutoff: float, n_refresh: int) -> Set[int]:
    eligible = [k for k, t in enumerate(reversed(list(seq))) if float(t) > float(t_cutoff)]
    if n_refresh <= 0 or not eligible:
        return set()
    n_refresh = min(int(n_refresh), len(eligible))
    if n_refresh == 1:
        return {eligible[len(eligible) // 2]}
    idxs = []
    for i in range(n_refresh):
        pos = int(round(i * (len(eligible) - 1) / (n_refresh - 1)))
        idxs.append(eligible[pos])
    return set(idxs)


def _alpha_bar_batch(betas: torch.Tensor, t_val: torch.Tensor) -> torch.Tensor:
    betas = betas.to(t_val.device)
    beta_ext = torch.cat([torch.zeros(1, device=betas.device), betas], dim=0)
    log_ab = torch.log((1.0 - beta_ext).clamp(min=1e-12)).cumsum(0)
    tv = t_val.float().reshape(-1).clamp(0.0, betas.numel() - 1)
    t0 = tv.floor().long()
    t1 = (t0 + 1).clamp(max=betas.numel() - 1)
    w = tv - t0.float()
    log_v = (1.0 - w) * log_ab[t0 + 1] + w * log_ab[t1 + 1]
    return torch.exp(log_v).view(-1, 1, 1, 1)


def _resolve_sample_path(path: str) -> str:
    if os.path.isfile(path):
        return path
    if path.endswith(".npz") and os.path.isfile(path[:-4] + ".png"):
        return path
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    if os.path.isdir(folder):
        return folder
    raise FileNotFoundError(path)


def existing_count(output_dir: str, *, save_format: str, npz_name: str, max_images: int) -> int:
    if save_format == "npz":
        npz_path = os.path.join(output_dir, npz_name)
        if os.path.isfile(npz_path):
            arr = np.load(npz_path)["images"]
            n = int(arr.shape[0])
            if n > max_images:
                raise RuntimeError(f"{npz_path} has {n} images (> {max_images})")
            return n
        return 0
    files = glob.glob(os.path.join(output_dir, "*.png"))
    ids = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        try:
            ids.append(int(stem))
        except ValueError:
            continue
    if not ids:
        return 0
    if max(ids) + 1 != len(set(ids)):
        return len(set(ids))
    return max(ids) + 1


def _tensor_batch_to_uint8_nhwc(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).contiguous().numpy()


def _vsc_lookup_value(entry: Any, field: str) -> float:
    if isinstance(entry, dict):
        if field == "trimmed_mean":
            return float(entry.get("trimmed_mean", entry.get("mean", 0.0)))
        if field in entry:
            return float(entry[field])
        return float(entry.get("trimmed_mean", entry.get("mean", 0.0)))
    return float(entry)


@torch.no_grad()
def final_only_ddim(
    x: torch.Tensor,
    seq,
    model,
    betas,
    corrector=None,
    *,
    eta: float = 0.0,
    generator: Optional[torch.Generator] = None,
    time_residual=None,
    time_residual_strength: float = 0.1,
    vsc_stats: Optional[Dict[str, Any]] = None,
    vsc_absorb_strength: float = 1.0,
    vsc_max_budget_fraction: float = 0.9,
    vsc_var_field: str = "trimmed_mean",
) -> torch.Tensor:
    """DDIM with optional ε-corrector and/or Student-t VSC (no δt adapter)."""
    seq_next = [-1] + list(seq[:-1])
    use_vsc = (
        vsc_stats is not None
        and float(eta) > 0.0
        and float(vsc_absorb_strength) > 0.0
    )
    per_time = vsc_stats["per_time"] if use_vsc else None
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = torch.full((x.shape[0],), float(i), device=x.device)
        next_t = torch.full((x.shape[0],), float(j), device=x.device)
        at = compute_alpha(betas, t.long())
        at_next = compute_alpha(betas, next_t.long())
        eps = model(x, t)
        if corrector is not None:
            eps = corrector.correct(eps, t, at, xt=x)
        if time_residual is not None:
            rf = torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)
            eps = time_residual.correct(
                eps, x, t, t, rf, strength=time_residual_strength,
            )
        if use_vsc:
            entry = per_time.get(int(i)) or per_time.get(str(int(i)))
            residual_var = _vsc_lookup_value(entry, vsc_var_field) if entry is not None else 0.0
            residual_var_t = torch.as_tensor(float(residual_var), device=x.device, dtype=x.dtype)
            x, _ = ddim_update_vsc(
                x,
                eps,
                at,
                at_next,
                eta=float(eta),
                residual_var=residual_var_t,
                absorb_strength=float(vsc_absorb_strength),
                max_budget_fraction=float(vsc_max_budget_fraction),
                generator=generator,
            )
        else:
            x = ddim_update(x, eps, at, at_next, eta=float(eta), generator=generator)
    return x


@torch.no_grad()
def ddim_with_sparse_dt_refresh(
    x: torch.Tensor,
    seq,
    model,
    betas,
    joint_corrector,
    *,
    dt_eta: float,
    dt_max: float,
    t_cutoff: int,
    n_refresh: int,
    apply_eps: bool,
    sync_alpha: bool,
    eps_corrector=None,
    eta: float = 0.0,
    time_residual=None,
    time_residual_strength: float = 0.1,
    vsc_stats: Optional[Dict[str, Any]] = None,
    vsc_absorb_strength: float = 1.0,
    vsc_max_budget_fraction: float = 0.9,
    vsc_var_field: str = "trimmed_mean",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    device = x.device
    b = x.shape[0]
    t_max = float(betas.numel() - 1)
    refresh_at = _refresh_step_indices(seq, t_cutoff, n_refresh)
    seq_next = [-1] + list(seq[:-1])
    rev_steps = list(zip(reversed(seq), reversed(seq_next)))

    for step_k, (i, j) in enumerate(rev_steps):
        t_nom = float(i)
        t_nom_t = torch.full((b,), t_nom, device=device)
        next_t = torch.full((b,), float(j), device=device)
        at = compute_alpha(betas, t_nom_t.long())
        at_next = compute_alpha(betas, next_t.long())

        eps_q = model(x, t_nom_t)
        do_refresh = step_k in refresh_at and t_nom > float(t_cutoff)
        if do_refresh:
            _, delta_t = joint_corrector.net(x, eps_q, t_nom_t)
            delta_t = (float(dt_eta) * delta_t).clamp(-float(dt_max), float(dt_max))
            t_corr = (t_nom_t + delta_t).clamp(0.0, t_max)
            eps_used = model(x, t_corr)
            if sync_alpha:
                at = _alpha_bar_batch(betas, t_corr)
            t_for_eps = t_corr
            if apply_eps:
                eps_used, _ = joint_corrector.correct_with_dt(
                    eps_used, float(t_corr.mean().item()), at, xt=x,
                )
            elif eps_corrector is not None:
                eps_used = eps_corrector.correct(eps_used, t_for_eps, at, xt=x)
        else:
            eps_used = eps_q
            t_for_eps = t_nom_t
            if apply_eps:
                eps_used, _ = joint_corrector.correct_with_dt(eps_used, t_nom, at, xt=x)
            elif eps_corrector is not None:
                eps_used = eps_corrector.correct(eps_used, t_for_eps, at, xt=x)

        if time_residual is not None:
            rf = torch.full((b,), float(do_refresh), device=device)
            eps_used = time_residual.correct(
                eps_used, x, t_nom_t, t_for_eps, rf, strength=time_residual_strength,
            )

        residual_var = None
        if (
            vsc_stats is not None
            and float(eta) > 0.0
            and float(vsc_absorb_strength) > 0.0
        ):
            per_time = vsc_stats["per_time"]
            key = int(round(t_nom))
            entry = per_time.get(key, per_time.get(str(key)))
            if entry is not None:
                value = _vsc_lookup_value(entry, vsc_var_field)
                residual_var = torch.full((b, 1, 1, 1), float(value), device=device, dtype=x.dtype)

        if float(eta) > 0.0 and vsc_stats is not None and float(vsc_absorb_strength) > 0.0:
            x, _ = ddim_update_vsc(
                x, eps_used, at, at_next,
                eta=float(eta),
                residual_var=residual_var,
                absorb_strength=float(vsc_absorb_strength),
                max_budget_fraction=float(vsc_max_budget_fraction),
                generator=generator,
            )
        else:
            x = ddim_update(x, eps_used, at, at_next, eta=float(eta), generator=generator)
    return x


def run_fid(sample_path: str, fid_ref: str, device: str, log_path: str) -> float:
    sample_path = _resolve_sample_path(sample_path)
    cmd = [sys.executable, "-m", "pytorch_fid", fid_ref, sample_path, "--device", device]
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    print(f"  FID: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"FID failed; see {log_path}")
    m = re.search(r"FID:\s*([0-9.eE+-]+)", proc.stdout)
    if not m:
        raise RuntimeError(f"FID parse failed: {log_path}")
    return float(m.group(1))


def parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/cifar10.yml")
    p.add_argument("--cali_ckpt", default="cifar_w8a8_ckpt.pth")
    p.add_argument("--cali_data_path", default="cifar_sd1236_sample2048_allst.pt")
    p.add_argument("--weight_bit", type=int, default=8)
    p.add_argument("--act_bit", type=int, default=8)
    p.add_argument("--sm_abit", type=int, default=8)
    p.add_argument("--cali_st", type=int, default=10)
    p.add_argument("--cali_n", type=int, default=256)
    p.add_argument("--split", action="store_true", default=True)
    p.add_argument("--quant_act", action="store_true", default=True)
    p.add_argument("--a_sym", action="store_true", default=True)
    p.add_argument("--ckpt", default="")
    p.add_argument("--ode_scale_json", default="")
    p.add_argument("--ode_absorb_mode", default="")
    p.add_argument("--backbone", choices=("unet", "uvit"), default="unet")
    p.add_argument("--fp_ckpt", default="")

    p.add_argument("--disable_adapter", action="store_true")
    p.add_argument("--adapter_ckpt", default="state_aware_temporal_joint/runs/adapter_w8a8_stage_a/ckpt_best.pt")
    p.add_argument("--disable_corrector", action="store_true")
    p.add_argument("--corrector_ckpt", default="state_aware_temporal_joint/runs/corrector_w8a8_stage_c/ckpt_best.pt")
    p.add_argument("--allow_joint_plus_corrector", action="store_true")
    p.add_argument("--fp_temporal_path", action="store_true")
    p.add_argument("--joint_feature_ckpt", default="")
    p.add_argument("--joint_dt_ckpt", default="")
    p.add_argument("--time_residual_ckpt", default="")
    p.add_argument("--time_residual_strength", type=float, default=0.1)

    p.add_argument("--dt_mode", default="refresh", choices=("carry", "refresh", "paper", "paper_distill"))
    p.add_argument("--dt_only_infer", action="store_true")
    p.add_argument("--dt_eta", type=float, default=0.5)
    p.add_argument("--dt_carry_max", type=float, default=20.0)
    p.add_argument("--t_cutoff", type=int, default=300)
    p.add_argument("--sync_alpha", action="store_true")
    p.add_argument("--dt_refresh_n", type=int, default=8)
    p.add_argument("--ts_window", type=int, default=10)

    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--skip_type", default="quad")
    p.add_argument("--vsc_stats", default="")
    p.add_argument("--vsc_absorb_strength", type=float, default=1.0)
    p.add_argument("--vsc_max_budget_fraction", type=float, default=0.9)
    p.add_argument(
        "--vsc_var_field",
        default="trimmed_mean",
        choices=("trimmed_mean", "mean", "var_mle"),
        help="per-time variance field in --vsc_stats for budget absorption",
    )

    p.add_argument("--max_images", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output_dir", default="state_aware_temporal_joint/output/adapter_corrector_w8a8_50k")
    p.add_argument("--save_format", default="png", choices=("png", "npz"))
    p.add_argument("--npz_name", default="images.npz")
    p.add_argument("--skip_fid", action="store_true")
    p.add_argument("--fid_ref", default=DEFAULT_FID_REF)
    p.add_argument("--fid_device", default="cuda:0")
    p.add_argument("--fid_log", default="")
    return p.parse_args()


def main() -> None:
    args = parser()
    args.cond = False
    args.joint_sa_resume = False
    args.brecq_ckpt = ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    start_id = existing_count(
        args.output_dir, save_format=args.save_format, npz_name=args.npz_name, max_images=args.max_images,
    )
    if start_id >= args.max_images:
        print(f"Already complete: {start_id} images", flush=True)
        if not args.skip_fid:
            fid_log = args.fid_log or os.path.join(args.output_dir, "fid.log")
            fid = run_fid(args.output_dir, args.fid_ref, args.fid_device, fid_log)
            print(f"FID: {fid}", flush=True)
        return

    with open(args.config, "r", encoding="utf-8") as f:
        config = dict2namespace(yaml.safe_load(f))
    config.split_shortcut = bool(args.split)

    t0 = time.time()
    if args.backbone == "uvit":
        sys.path.insert(0, os.path.join(_ROOT, "uvit_experiments"))
        from pipeline_loaders import load_quant_from_args

        print("  [load] U-ViT W8A8 quant model ...", flush=True)
        qnn = load_quant_from_args(args, device)
        print(f"  [load] ready ({time.time()-t0:.1f}s)", flush=True)
    else:
        from calibrate_trajectory_qhat import load_float_model, load_quant_model

        print("  [load 1/4] building float UNet ...", flush=True)
        fp = load_float_model(config, device, args)
        print(f"  [load 2/4] float teacher ready ({time.time()-t0:.1f}s)", flush=True)
        print("  [load 3/4] wrapping QuantModel ...", flush=True)
        qnn = load_quant_model(config, device, args, fp)
        del fp
        qnn.set_quant_state(True, True)
        qnn.eval()

    if args.fp_temporal_path:
        n_mod = keep_temporal_path_float(qnn)
        print(f"FP temporal path enabled for {n_mod} modules", flush=True)

    feature_hooks = None
    if args.joint_feature_ckpt:
        feat, _meta = load_feature_corrector(args.joint_feature_ckpt, device)
        feature_hooks = JointFeatureHooks(qnn, feat)
        feature_hooks.attach()
        print(f"Joint feature correction enabled for {len(feat.config.block_channels)} blocks", flush=True)

    if not args.disable_adapter:
        raise NotImplementedError(
            "Adapter sampling helpers are unavailable in this checkout; pass --disable_adapter."
        )

    corrector = None
    if not args.disable_corrector and args.corrector_ckpt:
        corrector = load_learned_corrector(args.corrector_ckpt, device)
        corrector.train_mode_off()
        print(f"Loaded ε corrector from {args.corrector_ckpt}", flush=True)

    joint_corrector = None
    if args.joint_dt_ckpt:
        joint_corrector = load_joint_corrector(args.joint_dt_ckpt, device)
        joint_corrector.train_mode_off()

    time_residual = None
    if args.time_residual_ckpt:
        time_residual = load_residual(args.time_residual_ckpt, device)

    vsc_stats = torch.load(args.vsc_stats, map_location="cpu") if args.vsc_stats else None
    if vsc_stats is not None and (float(args.eta) <= 0.0 or float(args.vsc_absorb_strength) <= 0.0):
        print("VSC disabled: eta=0 has no stochastic variance budget", flush=True)
        vsc_stats = None

    if joint_corrector is not None:
        eps_on = "on" if (args.allow_joint_plus_corrector and corrector is not None) else "off"
        print(
            f"Joint δt enabled from {args.joint_dt_ckpt} "
            f"(mode={args.dt_mode}, dt_only_infer={args.dt_only_infer}, dt_eta={args.dt_eta}, "
            f"carry_max={args.dt_carry_max}, t_cutoff={args.t_cutoff}, sync_alpha={args.sync_alpha}, "
            f"dt_refresh_n={args.dt_refresh_n}, eps_corrector={eps_on})",
            flush=True,
        )
    elif args.time_residual_ckpt:
        print(
            f"Mean residual corrector only from {args.time_residual_ckpt} "
            f"(strength={args.time_residual_strength}, no δt)",
            flush=True,
        )
    elif args.dt_mode in ("paper", "paper_distill"):
        raise ValueError(f"--dt_mode {args.dt_mode} requires --joint_dt_ckpt")

    betas = torch.as_tensor(
        get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        ),
        dtype=torch.float32,
        device=device,
    )
    seq = build_ddim_seq(len(betas), args.timesteps, args.skip_type)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))

    image_id = start_id
    rounds = math.ceil((args.max_images - image_id) / args.batch_size)
    npz_chunks = []
    apply_eps = not bool(args.dt_only_infer)
    eps_for_dt = corrector if args.allow_joint_plus_corrector else None
    started = time.time()

    print(
        f"Generating {args.max_images} images; start={start_id}, batch={args.batch_size}, "
        f"steps={len(seq)}, save_format={args.save_format}",
        flush=True,
    )

    with torch.inference_mode():
        for round_id in range(rounds):
            n = min(args.batch_size, args.max_images - image_id)
            x = torch.randn(
                n, config.data.channels, config.data.image_size, config.data.image_size,
                device=device, generator=generator,
            )

            if joint_corrector is not None and args.dt_mode == "refresh":
                final = ddim_with_sparse_dt_refresh(
                    x, seq, qnn, betas, joint_corrector,
                    dt_eta=args.dt_eta,
                    dt_max=args.dt_carry_max,
                    t_cutoff=args.t_cutoff,
                    n_refresh=args.dt_refresh_n,
                    apply_eps=apply_eps,
                    sync_alpha=bool(args.sync_alpha),
                    eps_corrector=eps_for_dt,
                    eta=args.eta,
                    time_residual=time_residual,
                    time_residual_strength=args.time_residual_strength,
                    vsc_stats=vsc_stats,
                    vsc_absorb_strength=args.vsc_absorb_strength,
                    vsc_max_budget_fraction=args.vsc_max_budget_fraction,
                    vsc_var_field=args.vsc_var_field,
                    generator=generator,
                )
            else:
                final = final_only_ddim(
                    x,
                    seq,
                    qnn,
                    betas,
                    corrector=corrector,
                    eta=args.eta,
                    generator=generator,
                    time_residual=time_residual,
                    time_residual_strength=args.time_residual_strength,
                    vsc_stats=vsc_stats,
                    vsc_absorb_strength=args.vsc_absorb_strength,
                    vsc_max_budget_fraction=args.vsc_max_budget_fraction,
                    vsc_var_field=args.vsc_var_field,
                )

            images = inverse_data_transform(config, final.cpu())
            if args.save_format == "npz":
                npz_chunks.append(_tensor_batch_to_uint8_nhwc(images))
                image_id += n
            else:
                for k in range(n):
                    tvu.save_image(images[k], os.path.join(args.output_dir, f"{image_id}.png"))
                    image_id += 1

            elapsed = time.time() - started
            rate = (image_id - start_id) / max(elapsed, 1e-6)
            eta_s = (args.max_images - image_id) / max(rate, 1e-6)
            print(
                f"batch {round_id + 1}/{rounds}: images={image_id}/{args.max_images} "
                f"rate={rate:.2f}/s eta={eta_s / 60:.1f}min",
                flush=True,
            )

    if args.save_format == "npz" and npz_chunks:
        arr = np.concatenate(npz_chunks, axis=0)
        np.savez_compressed(os.path.join(args.output_dir, args.npz_name), images=arr)

    manifest = {
        "images": image_id,
        "adapter_ckpt": args.adapter_ckpt,
        "corrector_ckpt": args.corrector_ckpt,
        "timesteps": args.timesteps,
        "adapter_enabled": not args.disable_adapter,
        "corrector_enabled": not args.disable_corrector,
        "allow_joint_plus_corrector": bool(args.allow_joint_plus_corrector),
        "fp_temporal_path": bool(args.fp_temporal_path),
        "joint_feature_ckpt": args.joint_feature_ckpt,
        "joint_dt_ckpt": args.joint_dt_ckpt,
        "time_residual_ckpt": args.time_residual_ckpt,
        "time_residual_strength": args.time_residual_strength,
        "vsc_stats": args.vsc_stats,
        "vsc_absorb_strength": args.vsc_absorb_strength,
        "vsc_max_budget_fraction": args.vsc_max_budget_fraction,
        "vsc_var_field": args.vsc_var_field,
        "dt_mode": args.dt_mode,
        "dt_only_infer": bool(args.dt_only_infer),
        "dt_eta": args.dt_eta,
        "dt_carry_max": args.dt_carry_max,
        "t_cutoff": args.t_cutoff,
        "dt_refresh_n": args.dt_refresh_n,
        "sync_alpha": bool(args.sync_alpha),
        "effective_steps": len(seq),
        "skip_type": args.skip_type,
        "eta": args.eta,
        "seed": args.seed,
        "elapsed_seconds_this_run": time.time() - started,
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if not args.skip_fid:
        fid_log = args.fid_log or os.path.join(args.output_dir, "fid.log")
        fid = run_fid(args.output_dir, args.fid_ref, args.fid_device, fid_log)
        manifest["fid"] = fid
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"FID: {fid}", flush=True)


if __name__ == "__main__":
    main()
