#!/usr/bin/env python
"""Fair W4A8 vs W8A8 raw-error collect on shared FP-rollout states.

Roll DDIM with the float U-ViT so both bitwidths see identical x_t, then evaluate
each quantized checkpoint at the same (x, t):

  e_w = eps_fp(x,t) - eps_q_w(x,t)

This is the correct protocol for claiming "lowering W makes heavier tails".
Old per-bit Q-rollout open-loop paths use different x_t and are not comparable.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "uvit_experiments"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch
from pytorch_lightning import seed_everything

from qdiff.ddim_helpers import alpha_bar_at, ddim_update
from qdiff.trajectory_error import build_ddim_seq
from sample_diffusion_ddim import get_beta_schedule
from uvit_loader import DEFAULT_CKPT, load_uvit_fp, load_uvit_quant


@torch.no_grad()
def rollout_fp_states(
    fp_model,
    betas: torch.Tensor,
    seq,
    device: torch.device,
    *,
    num_trajectories: int,
    batch_size: int,
    eta: float,
    seed: int,
):
    fp_model.eval()
    steps = list(zip(reversed(seq), reversed([-1] + list(seq[:-1]))))
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    xs, efs, ts, tids = [], [], [], []
    n_done = 0
    while n_done < num_trajectories:
        b = min(batch_size, num_trajectories - n_done)
        x = torch.randn(b, 3, 32, 32, device=device, generator=rng)
        ids = torch.arange(n_done, n_done + b, dtype=torch.int64)

        for i, j in steps:
            t = torch.full((b,), float(i), device=device)
            t_next = torch.full((b,), float(j), device=device)
            at = alpha_bar_at(betas, t.long(), device)
            at_next = alpha_bar_at(betas, t_next.long(), device)

            ef = fp_model(x, t)
            xs.append(x.detach().cpu().half())
            efs.append(ef.detach().cpu().half())
            ts.append(t.detach().cpu().float())
            tids.append(ids.clone())

            x = ddim_update(x, ef, at, at_next, eta=float(eta), generator=rng)

        n_done += b
        print(f"  FP rollout {n_done}/{num_trajectories}", flush=True)

    return {
        "x": torch.cat(xs, 0),
        "ef": torch.cat(efs, 0),
        "t": torch.cat(ts, 0),
        "traj_id": torch.cat(tids, 0).long(),
        "meta": {
            "driver": "fp",
            "num_trajectories": num_trajectories,
            "n_grid_steps": len(steps),
            "n_samples": int(torch.cat(ts, 0).shape[0]),
            "eta": float(eta),
            "seed": int(seed),
        },
    }


@torch.no_grad()
def eval_quant_eq(qnn, x_cpu, t_cpu, device, batch: int = 64):
    qnn.eval()
    qnn.set_quant_state(True, True)
    eqs = []
    n = x_cpu.shape[0]
    for st in range(0, n, batch):
        x = x_cpu[st : st + batch].float().to(device)
        t = t_cpu[st : st + batch].float().to(device)
        eqs.append(qnn(x, t).detach().cpu().half())
        if (st // batch) % 50 == 0:
            print(f"    eval {min(st + batch, n)}/{n}", flush=True)
    return torch.cat(eqs, 0)


def load_q(weight_bit: int, cali_ckpt: str, args, device):
    print(f"Loading W{weight_bit}A{args.act_bit} from {cali_ckpt} ...", flush=True)
    qnn = load_uvit_quant(
        cali_ckpt=cali_ckpt,
        cali_data_path=args.cali_data_path,
        device=device,
        fp_ckpt=args.fp_ckpt,
        weight_bit=weight_bit,
        act_bit=args.act_bit,
        sm_abit=args.sm_abit,
        cali_st=args.cali_st,
        cali_n=args.cali_n,
        quant_act=True,
        a_sym=True,
    )
    qnn.eval()
    return qnn


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fp_ckpt", default=DEFAULT_CKPT)
    p.add_argument("--w4_ckpt", default="uvit_experiments/checkpoints/uvit_w4a8_ckpt.pth")
    p.add_argument("--w8_ckpt", default="uvit_experiments/checkpoints/uvit_w8a8_ckpt.pth")
    p.add_argument("--cali_data_path", default="cifar_sd1236_sample2048_allst.pt")
    p.add_argument("--cali_st", type=int, default=10)
    p.add_argument("--cali_n", type=int, default=256)
    p.add_argument("--act_bit", type=int, default=8)
    p.add_argument("--sm_abit", type=int, default=8)
    p.add_argument("--num_trajectories", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch", type=int, default=64)
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--skip_type", default="quad")
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--output",
        default="uvit_experiments/outputs/error_distributions/paired_fp_rollout_w4_w8.pt",
    )
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    betas_np = get_beta_schedule(
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        num_diffusion_timesteps=1000,
    )
    betas = torch.tensor(betas_np, dtype=torch.float32, device=device)
    seq = build_ddim_seq(len(betas_np), args.timesteps, args.skip_type)

    print("Loading FP U-ViT ...", flush=True)
    fp = load_uvit_fp(args.fp_ckpt, device)
    fp.eval()

    print(
        f"FP-rollout shared states: n={args.num_trajectories}, "
        f"steps={len(seq)}, eta={args.eta}, seed={args.seed}",
        flush=True,
    )
    base = rollout_fp_states(
        fp,
        betas,
        seq,
        device,
        num_trajectories=args.num_trajectories,
        batch_size=args.batch_size,
        eta=args.eta,
        seed=args.seed,
    )
    del fp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eqs = {}
    for bit, ckpt in ((4, args.w4_ckpt), (8, args.w8_ckpt)):
        qnn = load_q(bit, ckpt, args, device)
        eqs[bit] = eval_quant_eq(qnn, base["x"], base["t"], device, batch=args.eval_batch)
        del qnn
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ef = base["ef"].float()
    summary = {}
    pack = {
        "x": base["x"],
        "t": base["t"],
        "traj_id": base["traj_id"],
        "ef": base["ef"],
        "eq_w4a8": eqs[4],
        "eq_w8a8": eqs[8],
        "e_w4a8": (ef - eqs[4].float()).half(),
        "e_w8a8": (ef - eqs[8].float()).half(),
        "meta": {
            **base["meta"],
            "backbone": "uvit",
            "timesteps": args.timesteps,
            "skip_type": args.skip_type,
            "act_bit": args.act_bit,
            "w4_ckpt": os.path.abspath(args.w4_ckpt),
            "w8_ckpt": os.path.abspath(args.w8_ckpt),
            "protocol": "fp_rollout_shared_xt_eval_both_q",
        },
    }
    for tag, e in (("w4a8", pack["e_w4a8"]), ("w8a8", pack["e_w8a8"])):
        v = e.float().reshape(-1)
        # light subsample for stats
        if v.numel() > 2_000_000:
            g = torch.Generator().manual_seed(0)
            idx = torch.randperm(v.numel(), generator=g)[:2_000_000]
            v = v[idx]
        summary[tag] = {
            "mean": float(v.mean()),
            "std": float(v.std(unbiased=True)),
            "mse": float(v.square().mean()),
            "abs_mean": float(v.abs().mean()),
            "q001": float(torch.quantile(v, 0.001)),
            "q999": float(torch.quantile(v, 0.999)),
        }
        print(tag, json.dumps(summary[tag], indent=2), flush=True)

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"
    torch.save(pack, tmp)
    os.replace(tmp, out)
    with open(out.replace(".pt", "_summary.json"), "w") as f:
        json.dump({"summary": summary, "meta": pack["meta"]}, f, indent=2)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
