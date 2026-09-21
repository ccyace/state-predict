#!/usr/bin/env python3
"""Compute q, S, K skew/kurtosis propagation metrics for residual r.

Definitions (variance-budget absorption, DDIM step t -> s):
  v_t       = Var(r_t)
  q_{t,s}   = B_{t,s}^2 * v_t / sigma_{t,s}^2
  gamma1    = E[(r-mu)^3] / v_t^{3/2}
  gamma2    = E[(r-mu)^4] / v_t^2 - 3
  S_{t,s}   = |gamma1| * q_{t,s}^{3/2}
  K_{t,s}   = |gamma2| * q_{t,s}^2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from calibrate_trajectory_qhat import dict2namespace  # noqa: E402
from sample_diffusion_ddim import get_beta_schedule  # noqa: E402
from state_aware_temporal_joint.time_corrected_residual import load_residual  # noqa: E402

EPS = 1e-30


def ddim_coeffs(t: int, t_next: int, ab: torch.Tensor, eta: float):
    at = ab[int(t)]
    an = torch.tensor(1.0) if t_next < 0 else ab[int(t_next)]
    ratio = ((1.0 - at / an.clamp(min=at)) * (1.0 - an) / (1.0 - at).clamp_min(1e-12)).clamp_min(0.0)
    sigma2 = float(eta * eta * ratio)
    c2 = math.sqrt(max(float(1.0 - an) - sigma2, 0.0))
    b = c2 - math.sqrt(float(an / at)) * math.sqrt(max(float(1.0 - at), 0.0))
    return float(b), sigma2


def higher_moments(v: torch.Tensor):
    x = v.reshape(-1).double()
    n = x.numel()
    if n < 8:
        return {"mu": 0.0, "var": 0.0, "gamma1": 0.0, "gamma2": 0.0}
    mu = float(x.mean())
    xc = x - mu
    m2 = float((xc * xc).mean())
    if m2 < EPS:
        return {"mu": mu, "var": m2, "gamma1": 0.0, "gamma2": 0.0}
    m3 = float((xc**3).mean())
    m4 = float((xc**4).mean())
    return {
        "mu": mu,
        "var": m2,
        "gamma1": m3 / (m2 ** 1.5),
        "gamma2": m4 / (m2 * m2) - 3.0,
    }


def summarize(arr: np.ndarray) -> Dict[str, float]:
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(x.max()),
    }


@torch.no_grad()
def collect_residual_by_t(
    data_path: str,
    ckpt_path: str,
    device: torch.device,
    *,
    strength: float = 1.0,
    val_mod: int = 5,
    batch_size: int = 128,
):
    d = torch.load(data_path, map_location="cpu", mmap=True)
    ids = d["traj_id"].long()
    idx = torch.where(ids % val_mod == 0)[0]
    net = load_residual(ckpt_path, device).net.eval()

    buckets: Dict[int, List[torch.Tensor]] = {}
    for st in range(0, len(idx), batch_size):
        ii = idx[st : st + batch_size]
        x = d["x"][ii].float().to(device)
        eq = d["eq"][ii].float().to(device)
        ef = d["ef"][ii].float().to(device)
        tn = d["t_nom"][ii].float().to(device)
        tc = d["t_corr"][ii].float().to(device)
        rf = d["is_refresh"][ii].float().to(device)
        r = (ef - eq) - strength * net(x, eq, tn, tc, rf)
        for t in tn.unique():
            m = tn == t
            buckets.setdefault(int(t), []).append(r[m].cpu())
    return {t: torch.cat(v, dim=0) for t, v in buckets.items()}


def compute_metrics(
    buckets: Dict[int, torch.Tensor],
    seq: List[int],
    ab: torch.Tensor,
    eta: float,
) -> List[dict]:
    nxt = seq[1:] + [-1]
    rows = []
    for t, j in zip(seq, nxt):
        if t not in buckets:
            continue
        block = buckets[t]  # [N,3,H,W]
        b_coef, sigma2 = ddim_coeffs(t, j, ab, eta)
        row = {
            "t": int(t),
            "next_t": int(j),
            "B": b_coef,
            "sigma2": sigma2,
            "per_channel": [],
        }
        q_list, s_list, k_list = [], [], []
        for c in range(3):
            mom = higher_moments(block[:, c])
            if sigma2 > 0:
                q = (b_coef * b_coef) * mom["var"] / sigma2
                s = abs(mom["gamma1"]) * (q ** 1.5)
                k = abs(mom["gamma2"]) * (q * q)
            else:
                q = s = k = float("nan")
            ch = {
                "channel": c,
                **mom,
                "q": q,
                "S": s,
                "K": k,
            }
            row["per_channel"].append(ch)
            q_list.append(q)
            s_list.append(s)
            k_list.append(k)
        row["q"] = q_list
        row["S"] = s_list
        row["K"] = k_list
        rows.append(row)
    return rows


def plot_curves(rows: List[dict], out_png: str, title: str):
    ts = np.array([r["t"] for r in rows])
    fig, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    cols = ["#d62728", "#2ca02c", "#1f77b4"]
    chs = ["R", "G", "B"]
    for c in range(3):
        q = np.array([r["q"][c] for r in rows])
        s = np.array([r["S"][c] for r in rows])
        k = np.array([r["K"][c] for r in rows])
        axs[0].plot(ts, np.maximum(q, 1e-16), color=cols[c], lw=1.6, label=chs[c])
        axs[1].plot(ts, np.maximum(s, 1e-16), color=cols[c], lw=1.6, label=chs[c])
        axs[2].plot(ts, np.maximum(k, 1e-16), color=cols[c], lw=1.6, label=chs[c])
    axs[0].set_ylabel(r"$q_{t,s}=B^2 v_t/\sigma^2$")
    axs[1].set_ylabel(r"$S_{t,s}=|\gamma_1|q^{3/2}$")
    axs[2].set_ylabel(r"$K_{t,s}=|\gamma_2|q^2$")
    axs[2].set_xlabel("nominal timestep t")
    for ax in axs:
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Compute residual skew/kurt budget metrics S,K")
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--name", default="run")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=1.0, help="eta used for sigma^2 budget (q,S,K)")
    p.add_argument("--config", default="configs/cifar10.yml")
    p.add_argument("--batch_size", type=int, default=128)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    buckets = collect_residual_by_t(
        args.data, args.ckpt, device, strength=args.strength, batch_size=args.batch_size
    )
    d = torch.load(args.data, map_location="cpu", mmap=True)
    first = int(d["traj_id"].long()[0])
    seq = d["t_nom"][d["traj_id"].long() == first].long().tolist()

    cfg = dict2namespace(yaml.safe_load(open(args.config)))
    betas = torch.tensor(
        get_beta_schedule(
            beta_schedule=cfg.diffusion.beta_schedule,
            beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end,
            num_diffusion_timesteps=cfg.diffusion.num_diffusion_timesteps,
        )
    )
    ab = (1.0 - betas).cumprod(0)
    rows = compute_metrics(buckets, seq, ab, args.eta)

    q = np.array([v for r in rows for v in r["q"]])
    s = np.array([v for r in rows for v in r["S"]])
    k = np.array([v for r in rows for v in r["K"]])
    g1 = np.array([ch["gamma1"] for r in rows for ch in r["per_channel"]])
    g2 = np.array([ch["gamma2"] for r in rows for ch in r["per_channel"]])

    rep = {
        "name": args.name,
        "data": os.path.abspath(args.data),
        "ckpt": os.path.abspath(args.ckpt),
        "eta_budget": args.eta,
        "definitions": {
            "q": "B^2 * Var(r_t) / sigma_{t,s}^2",
            "S": "|gamma1(r_t)| * q^{3/2}",
            "K": "|gamma2(r_t)| * q^2",
            "gamma1": "E[(r-mu)^3] / Var(r)^{3/2}",
            "gamma2": "E[(r-mu)^4] / Var(r)^2 - 3",
        },
        "summary": {
            "q_budget_occupancy": summarize(q),
            "S_skew_propagation": summarize(s),
            "K_kurt_propagation": summarize(k),
            "gamma1_residual_skew": summarize(g1),
            "gamma2_residual_exkurt": summarize(g2),
        },
        "per_timestep": rows,
    }
    json_path = os.path.join(args.output_dir, f"{args.name}_skew_kurt_budget.json")
    with open(json_path, "w") as f:
        json.dump(rep, f, indent=2)
    plot_curves(
        rows,
        os.path.join(args.output_dir, f"{args.name}_skew_kurt_budget.png"),
        f"{args.name}: q, S, K vs t (eta_budget={args.eta})",
    )
    print(json.dumps({"name": args.name, "summary": rep["summary"]}, indent=2), flush=True)
    print("wrote", json_path, flush=True)


if __name__ == "__main__":
    main()
