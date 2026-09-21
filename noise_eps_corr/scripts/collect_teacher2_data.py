"""Collect coupled native eta-DDIM Teacher-2 data for a discrete-error head.

All result paths must be outside the legacy ``output`` directory.  The default
root is ``teacher2_exp``.  Fine paths share the same coarse Gaussian and differ
only in their conditional bridge variables; averaging M paths estimates the
part of the fine-teacher defect predictable from the deployed inputs.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch
import yaml
from pytorch_lightning import seed_everything

from calibrate_trajectory_qhat import dict2namespace, load_float_model, load_quant_model
from noise_eps_corr.learned_noise_corrector import load_corrector_from_ckpt
from noise_eps_corr.scripts.validate_teacher2_fine_ddim import (
    _coefficients,
    _fine_path,
    _fine_terminal_weights,
    _model_eps,
    _randn,
    _step,
)
from qdiff.ddim_helpers import alpha_bar_at
from qdiff.trajectory_error import build_ddim_seq
from sample_diffusion_ddim import get_beta_schedule


def _reject_legacy_output(path: str) -> None:
    abs_path = os.path.abspath(path)
    legacy = os.path.abspath(os.path.join(_ROOT, "output"))
    if abs_path == legacy or abs_path.startswith(legacy + os.sep):
        raise ValueError("Teacher-2 outputs must not be written under legacy output/")


def _eta_at(t: float, args) -> float:
    return args.eta if args.eta_t_min <= float(t) <= args.eta_t_max else 0.0


def _conditional_bridge(z_coarse, k, betas, t, s, eta, generator):
    """Draw K fine Gaussians conditional on their weighted aggregate z_coarse."""
    w = _fine_terminal_weights(k, betas, t, s, eta, z_coarse.device)
    norm = w.square().sum().sqrt().clamp(min=1e-12)
    wn = w / norm
    raw = _randn((k, *z_coarse.shape), device=z_coarse.device, generator=generator)
    view = wn.view(k, *([1] * z_coarse.ndim))
    aggregate = (view * raw).sum(0)
    return raw + view * (z_coarse - aggregate).unsqueeze(0)


def _matched_fine_eta(k, betas, t, s, coarse_eta, device):
    """Scale fine eta so its linearly propagated endpoint std equals coarse sigma."""
    _carry, _eps_coeff, coarse_sigma_t = _coefficients(
        betas, t, s, coarse_eta, device
    )
    weights = _fine_terminal_weights(k, betas, t, s, coarse_eta, device)
    fine_std = weights.square().sum().sqrt()
    if float(fine_std) <= 0.0:
        return 0.0, 1.0
    ratio = float((fine_std / coarse_sigma_t.clamp(min=1e-12)).item())
    return float(coarse_eta / ratio), ratio


@torch.no_grad()
def collect(args, config, model, corrector, betas, seq, device):
    intervals = [
        (int(i), int(j))
        for i, j in zip(reversed(seq), reversed([-1] + list(seq[:-1])))
        if j >= 0
    ]
    active_steps = sum(_eta_at(i, args) > 0 for i, _j in intervals)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    fields = {name: [] for name in ("x", "eps_corr", "u", "d_disc", "t", "t_next", "b_coeff", "sigma", "fine_eta", "traj_id")}
    variance_ratios = {}

    done = 0
    while done < args.num_trajectories:
        batch = min(args.batch_size, args.num_trajectories - done)
        x = _randn(
            (batch, config.data.channels, config.data.image_size, config.data.image_size),
            device=device,
            generator=rng,
        )
        ids = torch.arange(done, done + batch, dtype=torch.int32)

        for i, j in intervals:
            step_eta = _eta_at(i, args)
            eq = _model_eps(model, x, i, quantized=True)
            at = alpha_bar_at(betas, i, device)
            eps_corr = corrector.correct(eq, float(i), at, xt=x)
            zc = _randn(x.shape, device=device, generator=rng)
            x_next = _step(x, eps_corr, zc, betas, i, j, step_eta, device)

            if step_eta > 0:
                ef = _model_eps(model, x, i, quantized=False)
                _carry, b_coeff_t, sigma_t = _coefficients(
                    betas, i, j, step_eta, device
                )
                x_fp_coarse = _step(x, ef, zc, betas, i, j, step_eta, device)
                fine_eta = step_eta
                if args.variance_match:
                    fine_eta, raw_ratio = _matched_fine_eta(
                        args.fine_steps, betas, i, j, step_eta, device
                    )
                    variance_ratios[str(i)] = raw_ratio
                fine_sum = torch.zeros_like(x_fp_coarse)
                for _ in range(args.bridge_repeats):
                    z_fine = _conditional_bridge(
                        zc, args.fine_steps, betas, i, j, fine_eta, rng
                    )
                    fine_sum.add_(
                        _fine_path(
                            model, x, i, j, fine_eta, z_fine, betas, device
                        )
                    )
                d_disc = fine_sum / args.bridge_repeats - x_fp_coarse
                sigma = float(sigma_t.item())
                b_coeff = float(b_coeff_t.item())

                fields["x"].append(x.detach().cpu().half())
                fields["eps_corr"].append(eps_corr.detach().cpu().half())
                fields["u"].append((sigma_t * zc).detach().cpu().half())
                fields["d_disc"].append(d_disc.detach().cpu().half())
                fields["t"].append(torch.full((batch,), float(i)))
                fields["t_next"].append(torch.full((batch,), float(j)))
                fields["b_coeff"].append(torch.full((batch,), b_coeff))
                fields["sigma"].append(torch.full((batch,), sigma))
                fields["fine_eta"].append(torch.full((batch,), fine_eta))
                fields["traj_id"].append(ids)

            model.set_quant_state(weight_quant=True, act_quant=True)
            x = x_next

        done += batch
        print(f"trajectories {done}/{args.num_trajectories}", flush=True)

    data = {key: torch.cat(parts, 0) for key, parts in fields.items()}
    data["traj_id"] = data["traj_id"].long()
    data["meta"] = {
        "kind": "teacher2_native_eta_ddim",
        "strict_reverse_sde": False,
        "num_trajectories": args.num_trajectories,
        "samples": int(data["t"].numel()),
        "active_steps_per_trajectory": active_steps,
        "timesteps": args.timesteps,
        "skip_type": args.skip_type,
        "eta": args.eta,
        "eta_t_min": args.eta_t_min,
        "eta_t_max": args.eta_t_max,
        "fine_steps": args.fine_steps,
        "bridge_repeats": args.bridge_repeats,
        "variance_matched": bool(args.variance_match),
        "raw_fine_to_coarse_std_ratio_by_t": variance_ratios,
        "seed": args.seed,
        "learned_corr_ckpt": args.learned_corr_ckpt,
        "collection_t_cut": args.collection_t_cut,
    }
    return data


def main():
    p = argparse.ArgumentParser(description="Collect Teacher-2 fine-DDIM targets")
    p.add_argument("--config", default="configs/cifar10.yml")
    p.add_argument("--ckpt", default="")
    p.add_argument("--cali_ckpt", default="cifar_w8a8_ckpt.pth")
    p.add_argument("--cali_data_path", default="cifar_sd1236_sample2048_allst.pt")
    p.add_argument("--learned_corr_ckpt", default="noise_eps_corr/w8a8_corr_ckpt_best.pt")
    p.add_argument("--collection_t_cut", type=int, default=999)
    p.add_argument("--output", default="teacher2_exp/data/teacher2_pilot.pt")
    p.add_argument("--eta", type=float, default=0.2)
    p.add_argument("--eta_t_min", type=int, default=200)
    p.add_argument("--eta_t_max", type=int, default=800)
    p.add_argument("--fine_steps", type=int, default=4)
    p.add_argument("--bridge_repeats", type=int, default=2)
    p.add_argument(
        "--variance_match",
        action="store_true",
        help="scale fine-step eta so propagated fine endpoint std equals coarse sigma",
    )
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--skip_type", choices=["uniform", "quad"], default="quad")
    p.add_argument("--num_trajectories", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=6789)
    p.add_argument("--split", action="store_true", default=True)
    p.add_argument("--quant_act", action="store_true", default=True)
    p.add_argument("--a_sym", action="store_true", default=True)
    p.add_argument("--weight_bit", type=int, default=8)
    p.add_argument("--act_bit", type=int, default=8)
    p.add_argument("--sm_abit", type=int, default=8)
    p.add_argument("--cali_st", type=int, default=10)
    p.add_argument("--cali_n", type=int, default=256)
    p.add_argument("--ode_scale_json", default="")
    p.add_argument("--ode_absorb_mode", default="")
    p.add_argument("--joint_sa_resume", action="store_true", default=False)
    args = p.parse_args()
    args.cond = False

    _reject_legacy_output(args.output)
    if args.eta <= 0 or args.fine_steps < 2 or args.bridge_repeats < 1:
        raise ValueError("require eta>0, fine_steps>=2, bridge_repeats>=1")
    if not 0 <= args.eta_t_min <= args.eta_t_max <= 999:
        raise ValueError("require 0 <= eta_t_min <= eta_t_max <= 999")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, "r", encoding="utf-8") as f:
        config = dict2namespace(yaml.safe_load(f))
    config.split_shortcut = args.split
    betas_np = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.tensor(betas_np, dtype=torch.float32, device=device)
    seq = build_ddim_seq(len(betas_np), args.timesteps, args.skip_type)

    float_model = load_float_model(config, device, args)
    model = load_quant_model(config, device, args, float_model)
    del float_model
    gc.collect()
    corrector = load_corrector_from_ckpt(args.learned_corr_ckpt, device)
    corrector.meta.t_cut = args.collection_t_cut
    corrector.train_mode_off()
    data = collect(args, config, model, corrector, betas, seq, device)

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, out)
    print(f"saved {data['meta']['samples']} samples -> {out}", flush=True)


if __name__ == "__main__":
    main()
