#!/usr/bin/env python
"""Oracle single-pulse contraction and delayed-recovery feasibility test.

This script intentionally uses the FP model only to construct an *oracle*
quantization innovation and the paired terminal reference.  It is a mechanism
test, not a deployable sampler.
"""

from __future__ import annotations

import argparse
import gc
import json
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
from ddim.functions.denoising import compute_alpha
from noise_eps_corr.qir_innovation import (
    InnovationCalibration,
    match_rms,
    rms,
    score_orthogonal,
)
from qdiff.ddim_helpers import ddim_update
from qdiff.trajectory_error import build_ddim_seq
from sample_diffusion_ddim import get_beta_schedule


def quant_state(model, enabled: bool, act_quant: bool):
    model.set_quant_state(weight_quant=enabled, act_quant=enabled and act_quant)


@torch.no_grad()
def eps(model, x, t, *, quantized, act_quant):
    quant_state(model, quantized, act_quant)
    tt = torch.full((x.shape[0],), float(t), device=x.device)
    return model(x, tt)


@torch.no_grad()
def rollout(model, x, steps, betas, *, quantized, act_quant, recovery=None):
    for local_index, (i, j) in enumerate(steps):
        at = compute_alpha(betas, torch.full((x.shape[0],), i, device=x.device).long())
        an = compute_alpha(betas, torch.full((x.shape[0],), j, device=x.device).long())
        x = ddim_update(
            x, eps(model, x, i, quantized=quantized, act_quant=act_quant), at, an, eta=0.0
        )
        if recovery is not None and local_index + 1 == recovery["delay"]:
            x = x - recovery["mu"] * recovery["pulse"]
    return x


def parse_ints(value):
    return [int(v) for v in value.split(",") if v.strip()]


def parse_floats(value):
    return [float(v) for v in value.split(",") if v.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/cifar10.yml")
    p.add_argument("--ckpt", default="")
    p.add_argument("--cali_ckpt", default="cifar_w8a8_ckpt.pth")
    p.add_argument("--cali_data_path", default="cifar_sd1236_sample2048_allst.pt")
    p.add_argument("--innovation_stats", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num_trajectories", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--timesteps", type=int, default=50)
    p.add_argument("--skip_type", default="quad")
    p.add_argument("--pulse_ts", default="")
    p.add_argument("--pulse_fractions", default="0.2,0.35,0.5,0.65,0.8")
    p.add_argument("--directions", default="gaussian,error,innovation,orthogonal")
    p.add_argument("--strengths", default="0.05,0.1,0.2,0.4")
    p.add_argument(
        "--strength_reference", choices=("step", "state"), default="step",
        help="interpret strengths relative to one DDIM displacement or current-state RMS",
    )
    p.add_argument("--recovery_delays", default="0,4,8")
    p.add_argument("--recovery_mus", default="0,0.25,0.5,0.75")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--weight_bit", type=int, default=8)
    p.add_argument("--act_bit", type=int, default=8)
    p.add_argument("--quant_act", action="store_true", default=True)
    p.add_argument("--a_sym", action="store_true", default=True)
    p.add_argument("--split", action="store_true", default=True)
    p.add_argument("--sm_abit", type=int, default=8)
    p.add_argument("--cali_st", type=int, default=10)
    p.add_argument("--cali_n", type=int, default=256)
    p.add_argument("--ode_scale_json", default="")
    p.add_argument("--ode_absorb_mode", default="")
    args = p.parse_args()
    args.cond = False
    args.joint_sa_resume = False

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, "r", encoding="utf-8") as handle:
        config = dict2namespace(yaml.safe_load(handle))
    config.split_shortcut = args.split
    beta_np = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.tensor(beta_np, device=device, dtype=torch.float32)
    seq = build_ddim_seq(len(beta_np), args.timesteps, args.skip_type)
    pairs = list(zip(reversed(seq), reversed([-1] + list(seq[:-1]))))
    ordered_t = [int(i) for i, _ in pairs]
    if args.pulse_ts:
        requested = parse_ints(args.pulse_ts)
    else:
        fractions = parse_floats(args.pulse_fractions)
        requested = [ordered_t[min(round(f * (len(ordered_t) - 1)), len(ordered_t) - 1)] for f in fractions]
    pulse_ts = sorted({min(ordered_t, key=lambda t: abs(t - v)) for v in requested}, reverse=True)

    fp_model = load_float_model(config, device, args)
    model = load_quant_model(config, device, args, fp_model)
    del fp_model
    gc.collect()
    calibration = InnovationCalibration.load(args.innovation_stats)
    directions = [v.strip() for v in args.directions.split(",") if v.strip()]
    strengths = parse_floats(args.strengths)
    delays = parse_ints(args.recovery_delays)
    mus = parse_floats(args.recovery_mus)
    records = defaultdict(list)
    baseline_mses = []
    generator = torch.Generator(device=device).manual_seed(args.seed)

    done = 0
    while done < args.num_trajectories:
        batch = min(args.batch_size, args.num_trajectories - done)
        x_start = torch.randn(
            batch, config.data.channels, config.data.image_size, config.data.image_size,
            generator=generator, device=device,
        )
        fp_terminal = rollout(
            model, x_start.clone(), pairs, betas,
            quantized=False, act_quant=args.quant_act,
        )

        # One quantized baseline pass supplies closed-loop states and causal errors.
        states, innovations, errors, scores = {}, {}, {}, {}
        x = x_start.clone()
        previous_residual, previous_t = None, None
        for i, j in pairs:
            eq = eps(model, x, i, quantized=True, act_quant=args.quant_act)
            ef = eps(model, x, i, quantized=False, act_quant=args.quant_act)
            error = eq - ef
            innovation, residual = calibration.innovation(
                error, i, previous_residual=previous_residual, previous_t=previous_t
            )
            if int(i) in pulse_ts:
                states[int(i)] = x.clone()
                innovations[int(i)] = innovation.clone()
                errors[int(i)] = error.clone()
                scores[int(i)] = eq.clone()
            at = compute_alpha(betas, torch.full((batch,), i, device=device).long())
            an = compute_alpha(betas, torch.full((batch,), j, device=device).long())
            x = ddim_update(x, eq, at, an, eta=0.0)
            previous_residual, previous_t = residual, int(i)
        base_terminal = x
        base_mse = (base_terminal - fp_terminal).square().flatten(1).mean(1)
        baseline_mses.extend(base_mse.cpu().tolist())

        for pulse_t in pulse_ts:
            pulse_index = ordered_t.index(pulse_t)
            i, j = pairs[pulse_index]
            x_at = states[pulse_t]
            eq = scores[pulse_t]
            at = compute_alpha(betas, torch.full((batch,), i, device=device).long())
            an = compute_alpha(betas, torch.full((batch,), j, device=device).long())
            base_next = ddim_update(x_at, eq, at, an, eta=0.0)
            step_rms = rms(base_next - x_at)
            tail = pairs[pulse_index + 1:]
            source = {
                "gaussian": torch.randn(x_at.shape, generator=generator, device=device),
                "error": errors[pulse_t],
                "innovation": innovations[pulse_t],
                "orthogonal": score_orthogonal(innovations[pulse_t], eq),
            }
            source["negative_orthogonal"] = -source["orthogonal"]
            random_sign = torch.randint(
                0, 2, (batch, 1, 1, 1), generator=generator, device=device
            ).mul_(2).sub_(1).to(source["orthogonal"].dtype)
            source["random_sign_orthogonal"] = random_sign * source["orthogonal"]
            for direction in directions:
                if direction not in source:
                    raise ValueError(f"unknown direction {direction}")
                for strength in strengths:
                    reference_rms = step_rms if args.strength_reference == "step" else rms(x_at)
                    pulse = match_rms(source[direction], strength * reference_rms)
                    for delay in delays:
                        valid_delay = delay > 0 and delay <= len(tail)
                        mu_values = mus if valid_delay else [0.0]
                        for mu in mu_values:
                            recovery = None
                            if valid_delay and mu > 0:
                                recovery = {"delay": delay, "mu": mu, "pulse": pulse}
                            terminal = rollout(
                                model, base_next + pulse, tail, betas,
                                quantized=True, act_quant=args.quant_act, recovery=recovery,
                            )
                            terminal_mse = (terminal - fp_terminal).square().flatten(1).mean(1)
                            residual_ratio = rms(terminal - base_terminal) / (rms(pulse) + 1e-12)
                            gain = (base_mse - terminal_mse) / (base_mse + 1e-12)
                            key = f"t{pulse_t}/{direction}/s{strength:g}/d{delay}/m{mu:g}"
                            pulse_to_state = (rms(pulse) / (rms(x_at) + 1e-12)).flatten()
                            pulse_to_step = (rms(pulse) / (step_rms + 1e-12)).flatten()
                            records[key].extend(zip(
                                terminal_mse.cpu().tolist(), gain.cpu().tolist(),
                                residual_ratio.flatten().cpu().tolist(),
                                pulse_to_state.cpu().tolist(), pulse_to_step.cpu().tolist(),
                            ))
        done += batch
        print(f"trajectories {done}/{args.num_trajectories}", flush=True)

    summary = {}
    for key, values in records.items():
        tensor = torch.tensor(values)
        summary[key] = {
            "n": len(values),
            "terminal_mse_mean": float(tensor[:, 0].mean()),
            "paired_gain_mean": float(tensor[:, 1].mean()),
            "paired_gain_std": float(tensor[:, 1].std(unbiased=False)),
            "terminal_residual_ratio_mean": float(tensor[:, 2].mean()),
            "pulse_to_state_rms_mean": float(tensor[:, 3].mean()),
            "pulse_to_step_rms_mean": float(tensor[:, 4].mean()),
        }
    payload = {
        "meta": vars(args),
        "pulse_ts": pulse_ts,
        "baseline_terminal_mse_mean": float(torch.tensor(baseline_mses).mean()),
        "summary": summary,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"saved {len(summary)} configurations to {args.output}")


if __name__ == "__main__":
    main()
