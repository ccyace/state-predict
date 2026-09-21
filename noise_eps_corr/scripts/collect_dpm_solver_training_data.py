#!/usr/bin/env python
"""Collect open-loop (x, eq, ef, t) on DPM-Solver++ paths for ε-corrector training.

Matches scripts/sample_diffusion_ddim.py --sample_type dpm_solver:
  algorithm_type=dpmsolver++, order=3, skip_type=time_uniform, method=singlestep.

At every noise-prediction call, store quantized eq and teacher ef (quant off)
on the same state x, then advance with uncorrected eq.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "uvit_experiments"))

import torch
import yaml
from pytorch_lightning import seed_everything

from calibrate_trajectory_qhat import dict2namespace
from ddim.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from pipeline_loaders import add_backbone_args, load_quant_from_args
from sample_diffusion_ddim import get_beta_schedule


@torch.no_grad()
def collect_dpm_pp(
    quant_model,
    betas: torch.Tensor,
    device: torch.device,
    *,
    num_trajectories: int,
    batch_size: int,
    steps: int,
    order: int,
    channels: int,
    image_size: int,
    seed: int,
):
    quant_model.eval()
    noise_schedule = NoiseScheduleVP(schedule="discrete", betas=betas)

    xs, eqs, efs, ts, traj_ids = [], [], [], [], []
    n_calls = [0]

    def raw_model(x, t_input, **_kwargs):
        t = t_input
        if not torch.is_tensor(t):
            t = torch.full((x.shape[0],), float(t), device=x.device, dtype=torch.float32)
        else:
            t = t.reshape(-1)
            if t.numel() == 1 and x.shape[0] > 1:
                t = t.expand(x.shape[0])
            elif t.numel() != x.shape[0]:
                t = t[:1].expand(x.shape[0])

        quant_model.set_quant_state(weight_quant=True, act_quant=True)
        eq = quant_model(x, t)
        quant_model.set_quant_state(weight_quant=False, act_quant=False)
        ef = quant_model(x, t)
        quant_model.set_quant_state(weight_quant=True, act_quant=True)

        xs.append(x.detach().cpu().half())
        eqs.append(eq.detach().cpu().half())
        efs.append(ef.detach().cpu().half())
        ts.append(t.detach().cpu().float())
        n_calls[0] += 1
        return eq

    model_fn = model_wrapper(raw_model, noise_schedule, model_type="noise")
    solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    n_done = 0

    while n_done < num_trajectories:
        cur_b = min(batch_size, num_trajectories - n_done)
        x0 = torch.randn(cur_b, channels, image_size, image_size, device=device, generator=rng)
        ids = torch.arange(n_done, n_done + cur_b, dtype=torch.int64)

        n_before = len(ts)
        _ = solver.sample(
            x0,
            steps=steps,
            order=order,
            skip_type="time_uniform",
            method="singlestep",
        )
        n_new = len(ts) - n_before
        # one traj_id row per stored state (batch expanded inside raw_model)
        for _ in range(n_new):
            traj_ids.append(ids.clone())

        n_done += cur_b
        print(
            f"  trajectories {n_done}/{num_trajectories}  "
            f"states={sum(v.shape[0] for v in xs)}  model_calls_batches={n_calls[0]}",
            flush=True,
        )

    return {
        "x": torch.cat(xs, dim=0),
        "eq": torch.cat(eqs, dim=0),
        "ef": torch.cat(efs, dim=0),
        "t": torch.cat(ts, dim=0),
        "traj_id": torch.cat(traj_ids, dim=0).long(),
        "meta": {
            "num_trajectories": num_trajectories,
            "n_samples": int(torch.cat(ts, dim=0).shape[0]),
            "closed_loop": False,
            "sampler": "dpm_solver++",
            "steps": steps,
            "order": order,
            "skip_type": "time_uniform",
            "method": "singlestep",
            "channels": channels,
            "image_size": image_size,
            "seed": seed,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_backbone_args(p)
    p.add_argument("--config", default="configs/cifar10.yml")
    p.add_argument("--ckpt", default="")
    p.add_argument("--cali_ckpt", default="cifar_w8a8_ckpt.pth")
    p.add_argument("--cali_data_path", default="cifar_sd1236_sample2048_allst.pt")
    p.add_argument("--ode_scale_json", default="")
    p.add_argument("--ode_absorb_mode", default="")
    p.add_argument("--brecq_ckpt", default="")
    p.add_argument("--num_trajectories", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--timesteps", type=int, default=100, help="DPM-Solver++ steps (=NFE budget)")
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output", required=True)
    p.add_argument("--split", action="store_true", default=True)
    p.add_argument("--quant_act", action="store_true", default=True)
    p.add_argument("--a_sym", action="store_true", default=True)
    p.add_argument("--weight_bit", type=int, default=8)
    p.add_argument("--act_bit", type=int, default=8)
    p.add_argument("--sm_abit", type=int, default=8)
    p.add_argument("--cali_st", type=int, default=10)
    p.add_argument("--cali_n", type=int, default=256)
    p.add_argument("--joint_sa_resume", action="store_true", default=False)
    args = p.parse_args()
    args.cond = False

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r", encoding="utf-8") as f:
        config = dict2namespace(yaml.safe_load(f))
    config.split_shortcut = args.split
    channels = int(config.data.channels)
    image_size = int(config.data.image_size)

    betas_np = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.tensor(betas_np, dtype=torch.float32, device=device)

    print(f"Loading {args.backbone} quant model on {device} ...", flush=True)
    quant_model = load_quant_from_args(args, device, config)
    gc.collect()

    print(
        f"Collecting {args.num_trajectories} DPM++ trajs, steps={args.timesteps}, "
        f"order={args.order} ...",
        flush=True,
    )
    data = collect_dpm_pp(
        quant_model,
        betas,
        device,
        num_trajectories=args.num_trajectories,
        batch_size=args.batch_size,
        steps=args.timesteps,
        order=args.order,
        channels=channels,
        image_size=image_size,
        seed=args.seed,
    )
    data["meta"]["backbone"] = args.backbone
    data["meta"]["weight_bit"] = args.weight_bit
    data["meta"]["act_bit"] = args.act_bit

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, out)
    print(
        f"saved {out}  n={data['meta']['n_samples']}  "
        f"trajs={data['meta']['num_trajectories']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
