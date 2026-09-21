#!/usr/bin/env python
"""U-ViT W4A8/W8A8: raw quantization error vs residual after mean corrector.

Same layout as output/bitwidth_error_dist_eta0/plot_error_distributions.py (a)(b).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "output", "bitwidth_error_dist_eta0"))

from plot_error_distributions import EPS, load_er, moments, per_t_channel_kappa, subsample

BITS = [4, 8]
COLORS = {4: "#ff7f0e", 8: "#1f77b4"}
OUT_DIR = os.path.join(REPO, "uvit_experiments", "outputs", "error_distributions")

CFG = {
    4: {
        "openloop": "uvit_experiments/outputs/phase2_w4a8/traj_openloop_n1000.pt",
        "dt_traj": "uvit_experiments/outputs/phase2_w4a8/traj_dt_only_n1000.pt",
        "mean_ckpt": "uvit_experiments/outputs/phase2_w4a8/mean_ckpt/ckpt_best.pt",
    },
    8: {
        "openloop": "uvit_experiments/outputs/phase2_w8a8/traj_openloop_n200.pt",
        "dt_traj": "uvit_experiments/outputs/phase2_w8a8/traj_dt_only_n200.pt",
        "mean_ckpt": "uvit_experiments/outputs/phase2_w8a8/mean_ckpt/ckpt_best.pt",
        "dt_ckpt": "uvit_experiments/outputs/phase2_w8a8/dt_ckpt/ckpt_best.pt",
        "q_ckpt": "uvit_experiments/checkpoints/uvit_w8a8_ckpt.pth",
    },
}


def _collect_w8a8_dt_traj(out_path: str) -> None:
    cmd = [
        sys.executable,
        "state_aware_temporal_joint/collect_dt_corrected_residual_data.py",
        "--backbone", "uvit",
        "--fp_ckpt", "/root/autodl-tmp/ODE-scale/cifar10_uvit_small.pth",
        "--cali_ckpt", CFG[8]["q_ckpt"],
        "--cali_data_path", "cifar_sd1236_sample2048_allst.pt",
        "--weight_bit", "8", "--act_bit", "8", "--sm_abit", "8",
        "--cali_st", "10", "--cali_n", "256", "--quant_act", "--a_sym",
        "--output", out_path,
        "--dt_ckpt", CFG[8]["dt_ckpt"],
        "--num_trajectories", "200",
        "--batch_size", "64",
        "--timesteps", "100", "--skip_type", "quad",
        "--dt_eta", "0.5", "--dt_max", "20", "--t_cutoff", "300", "--n_refresh", "8",
        "--seed", "1234",
    ]
    print("Collecting W8A8 dt-only traj for mean-residual panel ...", flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def ensure_dt_traj(bit: int) -> str:
    path = os.path.join(REPO, CFG[bit]["dt_traj"])
    if os.path.isfile(path):
        return path
    if bit == 8:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _collect_w8a8_dt_traj(CFG[8]["dt_traj"])
        return path
    raise FileNotFoundError(f"Missing dt traj for W{bit}A8: {path}")


@torch.no_grad()
def load_raw_e(data_path: str, *, max_states: int = 6000) -> torch.Tensor:
    d = torch.load(data_path, map_location="cpu", mmap=True)
    ids = d["traj_id"].long()
    idx = torch.where(ids % 5 == 0)[0]
    if idx.numel() > max_states:
        g = torch.Generator().manual_seed(0)
        idx = idx[torch.randperm(idx.numel(), generator=g)[:max_states]]
    return (d["ef"][idx].float() - d["eq"][idx].float())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    packs = {}
    summary = {}

    for b in BITS:
        ol = os.path.join(REPO, CFG[b]["openloop"])
        dt = ensure_dt_traj(b)
        ckpt = os.path.join(REPO, CFG[b]["mean_ckpt"])
        e = load_raw_e(ol)
        _e2, r, t = load_er(dt, ckpt, device)
        packs[b] = (e, r, t)
        me, mr = moments(f"e_w{b}a8", e), moments(f"r_w{b}a8", r)
        kap = per_t_channel_kappa(r, t)
        summary[f"w{b}a8"] = {"raw_e": me, "residual_r": mr, "kappa_t_channel": kap}
        print(f"W{b}A8", json.dumps({"e": me, "r": mr, "kappa": kap}, indent=2), flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    ax = axes[0]
    for b in BITS:
        v = subsample(packs[b][0], 150_000, seed=b)
        lo, hi = np.quantile(v, [0.001, 0.999])
        v = v[(v >= lo) & (v <= hi)]
        ax.hist(v, bins=120, density=True, histtype="step", lw=1.8, color=COLORS[b], label=f"W{b}A8")
    ax.set_yscale("log")
    ax.set_xlabel(r"element of $e=\varepsilon_f-\varepsilon_q$")
    ax.set_ylabel("density (log)")
    ax.set_title("(a) Raw quantization error: heavy-tailed, bit-dependent scale")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for b in BITS:
        v = subsample(packs[b][1], 150_000, seed=10 + b)
        lo, hi = np.quantile(v, [0.001, 0.999])
        v = v[(v >= lo) & (v <= hi)]
        ax.hist(v, bins=120, density=True, histtype="step", lw=1.8, color=COLORS[b], label=f"W{b}A8")
    ax.set_yscale("log")
    ax.set_xlabel(r"element of $r=e-\hat{m}$")
    ax.set_ylabel("density (log)")
    ax.set_title("(b) After mean corrector: centered, still heavy-tailed")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Quantization error distribution vs residual after mean corrector "
        "(U-ViT-S/2 CIFAR DDIM-100 η=0)",
        fontsize=12,
    )

    stem = os.path.join(OUT_DIR, "quant_error_distribution_panel")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(OUT_DIR, "quant_error_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for ext in ("png", "svg", "pdf"):
        print("saved", f"{stem}.{ext}", flush=True)


if __name__ == "__main__":
    main()
