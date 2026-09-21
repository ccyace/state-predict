"""Validate an engineering Teacher-2 target with coupled native eta-DDIM steps.

This is deliberately a diagnostic, not a training script.  For selected coarse
intervals it constructs, from the same closed-loop quantized state:

  * quantized coarse endpoint;
  * full-precision coarse endpoint with the same coarse Gaussian;
  * full-precision K-substep endpoint driven by a coupled Gaussian path.

Native eta-DDIM is a discrete kernel, so the resulting ``d_disc`` is reported as
a fine-teacher distillation defect.  The script also reports endpoint variance
mismatch; it must not be presented as a strict reverse-SDE discretization error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch
import yaml
from pytorch_lightning import seed_everything

from calibrate_trajectory_qhat import dict2namespace, load_float_model, load_quant_model
from qdiff.ddim_helpers import alpha_bar_at
from qdiff.trajectory_error import build_ddim_seq
from sample_diffusion_ddim import get_beta_schedule
from noise_eps_corr.learned_noise_corrector import load_corrector_from_ckpt


def _randn(shape, *, device, generator):
    return torch.randn(shape, device=device, generator=generator)


def _coefficients(betas, t, s, eta, device):
    at = alpha_bar_at(betas, float(t), device)
    a_s = alpha_bar_at(betas, float(s), device)
    sigma = eta * ((1.0 - at / a_s) * (1.0 - a_s) / (1.0 - at)).clamp(min=0).sqrt()
    carry = (a_s / at).sqrt()
    eps_coeff = ((1.0 - a_s) - sigma.square()).clamp(min=0).sqrt()
    eps_coeff = eps_coeff - carry * (1.0 - at).sqrt()
    return carry, eps_coeff, sigma


def _step(x, eps, z, betas, t, s, eta, device):
    carry, eps_coeff, sigma = _coefficients(betas, t, s, eta, device)
    return carry * x + eps_coeff * eps + sigma * z


def _model_eps(model, x, t, *, quantized):
    model.set_quant_state(weight_quant=quantized, act_quant=quantized)
    tt = torch.full((x.shape[0],), float(t), device=x.device)
    return model(x, tt)


def _fine_path(model, x, t, s, eta, z_steps, betas, device):
    k = z_steps.shape[0]
    grid = torch.linspace(float(t), float(s), k + 1, device=device).tolist()
    y = x
    model.set_quant_state(weight_quant=False, act_quant=False)
    for idx in range(k):
        eps = _model_eps(model, y, grid[idx], quantized=False)
        y = _step(y, eps, z_steps[idx], betas, grid[idx], grid[idx + 1], eta, device)
    return y


def _aggregate_base_noise(z_base, k):
    """Group a finest iid path into k normalized equal-duration increments."""
    k_max = z_base.shape[0]
    if k_max % k:
        raise ValueError(f"largest K={k_max} must be divisible by K={k}")
    group = k_max // k
    return z_base.view(k, group, *z_base.shape[1:]).sum(1) / math.sqrt(group)


def _fine_terminal_weights(k, betas, t, s, eta, device):
    """Linearized weights from each fine Gaussian to the interval endpoint."""
    grid = torch.linspace(float(t), float(s), k + 1, device=device).tolist()
    weights = []
    for idx in range(k):
        _carry, _eps_coeff, sigma = _coefficients(
            betas, grid[idx], grid[idx + 1], eta, device
        )
        propagated = sigma
        for later in range(idx + 1, k):
            carry, _ec, _sg = _coefficients(
                betas, grid[later], grid[later + 1], eta, device
            )
            propagated = propagated * carry
        weights.append(propagated)
    return torch.stack([v.reshape(()) for v in weights])


def _coarse_noise_and_std(z_steps, betas, t, s, eta, device):
    """Linearized terminal weighting of fine stochastic increments."""
    k = z_steps.shape[0]
    w = _fine_terminal_weights(k, betas, t, s, eta, device)
    fine_std = w.square().sum().sqrt()
    if float(fine_std) <= 0:
        return torch.zeros_like(z_steps[0]), 0.0
    z_coarse = (w.view(k, *([1] * (z_steps.ndim - 1))) * z_steps).sum(0) / fine_std
    return z_coarse, float(fine_std.item())


def _sample_rms(x):
    return x.float().square().flatten(1).mean(1).sqrt()


def _sample_cos(a, b):
    af, bf = a.float().flatten(1), b.float().flatten(1)
    return (af * bf).sum(1) / (af.norm(dim=1) * bf.norm(dim=1) + 1e-12)


def _eta_at(t, args):
    """Piecewise stochastic schedule: deterministic at both denoising ends."""
    return args.eta if args.eta_t_min <= float(t) <= args.eta_t_max else 0.0


@torch.no_grad()
def validate(args, config, model, corrector, betas, seq, device):
    ks = sorted(set(args.fine_steps))
    k_max = max(ks)
    if any(k_max % k for k in ks):
        raise ValueError("Every --fine_steps value must divide the largest value")

    intervals = [(int(i), int(j)) for i, j in zip(reversed(seq), reversed([-1] + list(seq[:-1]))) if j >= 0]
    stochastic_ids = [idx for idx, (i, _j) in enumerate(intervals) if _eta_at(i, args) > 0]
    if args.max_intervals > 0 and len(stochastic_ids) > args.max_intervals:
        positions = torch.linspace(0, len(stochastic_ids) - 1, args.max_intervals).round().long().tolist()
        selected = {stochastic_ids[pos] for pos in positions}
    else:
        selected = set(stochastic_ids)

    rng = torch.Generator(device=device).manual_seed(args.seed)
    values = defaultdict(list)
    per_t = defaultdict(lambda: defaultdict(list))
    done = 0
    while done < args.num_trajectories:
        batch = min(args.batch_size, args.num_trajectories - done)
        x = _randn((batch, config.data.channels, config.data.image_size, config.data.image_size), device=device, generator=rng)

        for step_idx, (i, j) in enumerate(intervals):
            step_eta = _eta_at(i, args)
            eq = _model_eps(model, x, i, quantized=True)
            at = alpha_bar_at(betas, i, device)
            eps_roll = corrector.correct(eq, float(i), at, xt=x) if corrector is not None else eq

            z_roll = _randn(x.shape, device=device, generator=rng)
            x_next = _step(x, eps_roll, z_roll, betas, i, j, step_eta, device)

            if step_idx in selected:
                ef = _model_eps(model, x, i, quantized=False)
                z_base = _randn((k_max, *x.shape), device=device, generator=rng)
                z_ind = _randn(x.shape, device=device, generator=rng)
                endpoints = {}

                for k in ks:
                    z_k = _aggregate_base_noise(z_base, k)
                    z_coarse, fine_std = _coarse_noise_and_std(
                        z_k, betas, i, j, step_eta, device
                    )
                    _ca, _ce, coarse_sigma_t = _coefficients(
                        betas, i, j, step_eta, device
                    )
                    coarse_sigma = float(coarse_sigma_t.item())
                    x_q_c = _step(x, eq, z_coarse, betas, i, j, step_eta, device)
                    x_fp_c = _step(x, ef, z_coarse, betas, i, j, step_eta, device)
                    x_fp_ind = _step(x, ef, z_ind, betas, i, j, step_eta, device)
                    x_fp_f = _fine_path(model, x, i, j, step_eta, z_k, betas, device)
                    d_q = x_fp_c - x_q_c
                    d_d = x_fp_f - x_fp_c
                    d_ind = x_fp_f - x_fp_ind
                    d_total = x_fp_f - x_q_c
                    endpoints[k] = x_fp_f

                    prefix = f"k{k}"
                    batch_metrics = {
                        "d_q_rms": _sample_rms(d_q),
                        "d_disc_rms": _sample_rms(d_d),
                        "d_total_rms": _sample_rms(d_total),
                        "independent_disc_rms": _sample_rms(d_ind),
                        "dq_ddisc_cos": _sample_cos(d_q, d_d),
                    }
                    for name, tensor in batch_metrics.items():
                        vals = tensor.cpu().tolist()
                        values[f"{prefix}.{name}"].extend(vals)
                        per_t[str(i)][f"{prefix}.{name}"].extend(vals)
                    values[f"{prefix}.fine_to_coarse_std_ratio"].append(
                        fine_std / max(coarse_sigma, 1e-12)
                    )

                ref = endpoints[k_max]
                for k in ks[:-1]:
                    values[f"k{k}_to_k{k_max}_rms"].extend(
                        _sample_rms(endpoints[k] - ref).cpu().tolist()
                    )

            model.set_quant_state(weight_quant=True, act_quant=True)
            x = x_next

        done += batch
        print(f"trajectories {done}/{args.num_trajectories}", flush=True)

    def summarize(v):
        t = torch.tensor(v, dtype=torch.float64)
        return {
            "mean": float(t.mean()),
            "std": float(t.std(unbiased=False)),
            "p50": float(t.quantile(0.5)),
            "p90": float(t.quantile(0.9)),
            "n": int(t.numel()),
        }

    summary = {key: summarize(val) for key, val in sorted(values.items())}
    time_summary = {
        time: {key: summarize(val) for key, val in sorted(metrics.items())}
        for time, metrics in per_t.items()
    }
    return {
        "meta": {
            "description": "engineering native eta-DDIM fine-teacher validation",
            "strict_reverse_sde": False,
            "eta": args.eta,
            "eta_t_min": args.eta_t_min,
            "eta_t_max": args.eta_t_max,
            "eta_schedule": "eta inside [eta_t_min, eta_t_max], zero outside",
            "fine_steps": ks,
            "timesteps": args.timesteps,
            "skip_type": args.skip_type,
            "num_trajectories": args.num_trajectories,
            "selected_intervals": len(selected),
            "seed": args.seed,
            "closed_loop_corrector": bool(corrector is not None),
        },
        "summary": summary,
        "per_t": time_summary,
    }


def main():
    p = argparse.ArgumentParser(description="Validate coupled full-precision fine DDIM teacher")
    p.add_argument("--config", default="configs/cifar10.yml")
    p.add_argument("--ckpt", default="")
    p.add_argument("--cali_ckpt", default="cifar_w8a8_ckpt.pth")
    p.add_argument("--cali_data_path", default="cifar_sd1236_sample2048_allst.pt")
    p.add_argument("--learned_corr_ckpt", default="")
    p.add_argument("--collection_t_cut", type=int, default=999)
    p.add_argument("--output", default="output/noise_corr/teacher2_validation_eta02.json")
    p.add_argument("--eta", type=float, default=0.2)
    p.add_argument("--eta_t_min", type=int, default=200)
    p.add_argument("--eta_t_max", type=int, default=800)
    p.add_argument("--fine_steps", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--skip_type", choices=["uniform", "quad"], default="quad")
    p.add_argument("--num_trajectories", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_intervals", type=int, default=12)
    p.add_argument("--seed", type=int, default=5678)
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

    if args.eta <= 0:
        raise ValueError("Teacher-2 stochastic validation requires --eta > 0")
    if not 0 <= args.eta_t_min <= args.eta_t_max <= 999:
        raise ValueError("require 0 <= --eta_t_min <= --eta_t_max <= 999")
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

    print(f"loading W{args.weight_bit}A{args.act_bit} model on {device} ...", flush=True)
    float_model = load_float_model(config, device, args)
    quant_model = load_quant_model(config, device, args, float_model)
    corrector = None
    if args.learned_corr_ckpt:
        corrector = load_corrector_from_ckpt(args.learned_corr_ckpt, device)
        corrector.meta.t_cut = args.collection_t_cut
        corrector.train_mode_off()

    result = validate(args, config, quant_model, corrector, betas, seq, device)
    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out)
    print(f"saved: {out}", flush=True)
    for key, stats in result["summary"].items():
        print(f"{key:38s} mean={stats['mean']:.6g} std={stats['std']:.6g}")


if __name__ == "__main__":
    main()
