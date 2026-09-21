#!/usr/bin/env python
"""Stage B: refresh closed-loop trajectories with the trained adapter enabled."""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch
import yaml

from calibrate_trajectory_qhat import dict2namespace, load_float_model, load_quant_model
from ddim.functions.denoising import compute_alpha
from noise_eps_corr.learned_noise_corrector import DeltaEpsNet
from qdiff.ddim_helpers import ddim_update
from qdiff.trajectory_error import build_ddim_seq
from sample_diffusion_ddim import get_beta_schedule
from state_aware_temporal_joint.framework import StateAwareJointFramework
from state_aware_temporal_joint.io import load_joint_checkpoint


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter_ckpt", default="state_aware_temporal_joint/runs/adapter_w8a8_stage_a/ckpt_best.pt")
    p.add_argument("--output", default="state_aware_temporal_joint/data/traj_adapter_w8a8_n256.pt")
    p.add_argument("--report", default="state_aware_temporal_joint/data/traj_adapter_w8a8_n256_report.json")
    p.add_argument("--num_trajectories", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--skip_type", choices=("uniform", "quad"), default="quad")
    p.add_argument("--seed", type=int, default=20260721)
    p.add_argument("--config", default="configs/cifar10.yml")
    p.add_argument("--cali_ckpt", default="cifar_w8a8_ckpt.pth")
    p.add_argument("--cali_data_path", default="cifar_sd1236_sample2048_allst.pt")
    p.add_argument("--split", action="store_true", default=True)
    p.add_argument("--quant_act", action="store_true", default=True)
    p.add_argument("--a_sym", action="store_true", default=True)
    p.add_argument("--weight_bit", type=int, default=8)
    p.add_argument("--act_bit", type=int, default=8)
    p.add_argument("--sm_abit", type=int, default=8)
    p.add_argument("--cali_st", type=int, default=10)
    p.add_argument("--cali_n", type=int, default=256)
    p.add_argument("--ckpt", default="")
    p.add_argument("--ode_scale_json", default="")
    p.add_argument("--ode_absorb_mode", default="")
    return p.parse_args()


@torch.no_grad()
def three_outputs(framework, qnn, x, t):
    qnn.set_quant_state(weight_quant=True, act_quant=True)
    eps_adapted = framework.adapted_eps(x, t)
    eps_base = framework.unadapted_eps(x, t)
    qnn.set_quant_state(weight_quant=False, act_quant=False)
    eps_teacher = framework.unadapted_eps(x, t)
    qnn.set_quant_state(weight_quant=True, act_quant=True)
    return eps_base, eps_adapted, eps_teacher


def main():
    args = args_parser()
    args.cond = False
    args.joint_sa_resume = False
    args.brecq_ckpt = ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, "r", encoding="utf-8") as f:
        config = dict2namespace(yaml.safe_load(f))
    config.split_shortcut = args.split

    float_model = load_float_model(config, device, args)
    qnn = load_quant_model(config, device, args, float_model)
    del float_model
    corrector = DeltaEpsNet().to(device)
    adapter, corrector, payload = load_joint_checkpoint(args.adapter_ckpt, corrector, device)
    framework = StateAwareJointFramework(qnn, adapter, corrector).eval()
    adapter.eval()

    betas_np = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.as_tensor(betas_np, dtype=torch.float32, device=device)
    seq = build_ddim_seq(len(betas_np), args.timesteps, args.skip_type)
    pairs = list(zip(reversed(seq), reversed([-1] + list(seq[:-1]))))
    rng = torch.Generator(device=device).manual_seed(args.seed)

    xs, eqs, efs, ts, traj_ids = [], [], [], [], []
    per_t = {}
    totals = {"sse_base": 0.0, "sse_adapted": 0.0, "elements": 0}
    done = 0
    print(f"Refreshing {args.num_trajectories} adapter-only trajectories ({len(pairs)} DDIM steps)", flush=True)
    while done < args.num_trajectories:
        b = min(args.batch_size, args.num_trajectories - done)
        x = torch.randn(b, config.data.channels, config.data.image_size, config.data.image_size, device=device, generator=rng)
        ids = torch.arange(done, done + b, dtype=torch.long)
        for i, j in pairs:
            t = torch.full((b,), float(i), device=device)
            next_t = torch.full((b,), float(j), device=device)
            base, adapted, teacher = three_outputs(framework, qnn, x, t)
            err_b = (base.float() - teacher.float()).square()
            err_a = (adapted.float() - teacher.float()).square()
            sse_b, sse_a, elems = float(err_b.sum()), float(err_a.sum()), err_b.numel()
            totals["sse_base"] += sse_b
            totals["sse_adapted"] += sse_a
            totals["elements"] += elems
            rec = per_t.setdefault(str(int(i)), {"sse_base": 0.0, "sse_adapted": 0.0, "elements": 0})
            rec["sse_base"] += sse_b; rec["sse_adapted"] += sse_a; rec["elements"] += elems

            xs.append(x.cpu().half())
            eqs.append(adapted.cpu().half())
            efs.append(teacher.cpu().half())
            ts.append(torch.full((b,), float(i)))
            traj_ids.append(ids)
            at = compute_alpha(betas, t.long())
            at_next = compute_alpha(betas, next_t.long())
            x = ddim_update(x, adapted, at, at_next, eta=0.0)
        done += b
        print(f"trajectories {done}/{args.num_trajectories}", flush=True)

    data = {
        "x": torch.cat(xs), "eq": torch.cat(eqs), "ef": torch.cat(efs),
        "t": torch.cat(ts), "traj_id": torch.cat(traj_ids),
        "meta": {"kind": "adapter_enabled_closed_loop", "adapter_ckpt": args.adapter_ckpt,
                 "num_trajectories": args.num_trajectories, "timesteps": args.timesteps,
                 "skip_type": args.skip_type, "seed": args.seed, "corrector_applied": False},
    }
    overall_base = totals["sse_base"] / totals["elements"]
    overall_adapted = totals["sse_adapted"] / totals["elements"]
    report = {
        "mse_base": overall_base, "mse_adapted": overall_adapted,
        "reduction": (overall_base - overall_adapted) / max(overall_base, 1e-12),
        "improved_timesteps": 0, "total_timesteps": len(per_t), "per_timestep": {},
    }
    for t, rec in per_t.items():
        mb = rec["sse_base"] / rec["elements"]; ma = rec["sse_adapted"] / rec["elements"]
        report["per_timestep"][t] = {"mse_base": mb, "mse_adapted": ma, "reduction": (mb-ma)/max(mb, 1e-12)}
        report["improved_timesteps"] += int(ma < mb)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(data, args.output)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    framework.close()
    print(f"MSE {overall_base:.8f} -> {overall_adapted:.8f}; reduction={report['reduction']:.2%}; improved steps={report['improved_timesteps']}/{report['total_timesteps']}", flush=True)
    print(f"Saved {len(data['t'])} pairs -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
