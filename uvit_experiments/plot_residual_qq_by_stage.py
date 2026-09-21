#!/usr/bin/env python
"""Q-Q plots of standardized U-ViT PTQ errors e = eps_fp - eps_q by denoising stage.

Mirrors output/bitwidth_error_dist_eta0/plot_residual_qq_by_stage.py layout:
  3 panels @ t ≈ 783 / 399 / 20, W4A8 vs W8A8 on open-loop trajectories.
"""
from __future__ import annotations

import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.abspath(__file__))
_BITWIDTH_DIR = os.path.join(REPO, "output", "bitwidth_error_dist_eta0")
sys.path.insert(0, REPO)
if os.path.isdir(_BITWIDTH_DIR):
    sys.path.insert(0, _BITWIDTH_DIR)

from plot_gamma1_by_t import setup_cjk_font

EPS = 1e-12
STAGE_TARGETS = [783, 399, 20]
UNIFIED_TITLE = (
    "U-ViT-S/2 CIFAR-10: Q–Q plots of standardized PTQ residuals "
    "against the Gaussian reference"
)

BITS = [4, 8]
COLORS = {4: "#ff7f0e", 8: "#1f77b4"}
TRAJ = {
    4: "uvit_experiments/outputs/phase2_w4a8/traj_openloop_n1000.pt",
    8: "uvit_experiments/outputs/phase2_w8a8/traj_openloop_n200.pt",
}
COLLECT = {
    8: [
        sys.executable,
        "noise_eps_corr/scripts/collect_fullstep_training_data.py",
        "--backbone", "uvit",
        "--fp_ckpt", "/root/autodl-tmp/ODE-scale/cifar10_uvit_small.pth",
        "--cali_ckpt", "uvit_experiments/checkpoints/uvit_w8a8_ckpt.pth",
        "--cali_data_path", "cifar_sd1236_sample2048_allst.pt",
        "--weight_bit", "8", "--act_bit", "8", "--sm_abit", "8",
        "--cali_st", "10", "--cali_n", "256", "--quant_act", "--a_sym",
        "--num_trajectories", "200",
        "--batch_size", "64",
        "--timesteps", "100",
        "--skip_type", "quad",
        "--output", TRAJ[8],
    ],
}


def ensure_traj(bit: int) -> str:
    path = os.path.join(REPO, TRAJ[bit])
    if os.path.isfile(path):
        return path
    if bit not in COLLECT:
        raise FileNotFoundError(f"Missing trajectory for W{bit}A8: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Collecting open-loop traj for W{bit}A8 -> {path}", flush=True)
    subprocess.run(COLLECT[bit], cwd=REPO, check=True)
    return path


@torch.no_grad()
def load_quant_error(data_path: str, *, max_states: int = 12_000) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pooled quantization error e = ef - eq and nominal timestep t."""
    raw = torch.load(data_path, map_location="cpu", mmap=True)
    ids = raw["traj_id"].long()
    idx = torch.where(ids % 5 == 0)[0]
    if max_states > 0 and idx.numel() > max_states:
        g = torch.Generator().manual_seed(0)
        idx = idx[torch.randperm(idx.numel(), generator=g)[:max_states]]
    eq = raw["eq"][idx].float()
    ef = raw["ef"][idx].float()
    t = raw["t"][idx].float()
    return ef - eq, t


def load_packs(*, max_states: int = 12_000) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    packs = {}
    for b in BITS:
        path = ensure_traj(b)
        e, t = load_quant_error(path, max_states=max_states)
        packs[b] = (e, t)
        print(f"U-ViT W{b}A8: {e.numel():,} error elements from {path}", flush=True)
    return packs


def nearest_step(steps, target: int) -> int:
    arr = np.asarray(steps, dtype=int)
    return int(arr[np.argmin(np.abs(arr - target))])


def std_at_t(e: torch.Tensor, t: torch.Tensor, t_target: int, n: int = 40_000, seed: int = 0):
    steps = sorted(int(x) for x in torch.unique(t.long()).tolist())
    t_use = nearest_step(steps, t_target)
    m = t.long() == t_use
    v = e[m].reshape(-1).numpy()
    rng = np.random.default_rng(seed)
    if v.size > n:
        v = rng.choice(v, n, replace=False)
    z = (v - v.mean()) / (v.std() + EPS)
    return z, t_use


def draw_qq_panel(ax, packs, t_target: int, *, seed_base: int = 0) -> int:
    t_used = None
    for b in BITS:
        e, t = packs[b]
        z, t_use = std_at_t(e, t, t_target, seed=seed_base + b)
        t_used = t_use
        (osm, osr), _ = stats.probplot(z, dist="norm")
        ax.plot(
            osm,
            osr,
            ".",
            ms=1.0,
            alpha=0.28,
            color=COLORS[b],
            label=f"W{b}A8",
            rasterized=True,
        )
    lim = [-5, 5]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim([-12, 12])
    ax.set_xlabel("Gaussian theoretical quantiles")
    ax.set_ylabel("Empirical quantiles")
    ax.set_title(f"$t={t_used}$", fontsize=11)
    ax.legend(loc="upper left", frameon=False, markerscale=4, fontsize=8)
    ax.grid(alpha=0.25)
    return t_used


def main():
    setup_cjk_font()
    work = os.path.join(ROOT, "outputs/error_distributions")
    os.makedirs(work, exist_ok=True)

    packs = load_packs()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.4, h_pad=0.6, hspace=0.08, wspace=0.25)
    resolved = []
    for ax, tgt in zip(axes, STAGE_TARGETS):
        t_use = draw_qq_panel(ax, packs, tgt, seed_base=100 * tgt)
        resolved.append(t_use)

    fig.suptitle(UNIFIED_TITLE, fontsize=12, y=1.06)
    print("timesteps used:", resolved, flush=True)

    stem = os.path.join(work, "residual_qq_by_stage")
    save_kw = dict(bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, **save_kw)
    fig.savefig(f"{stem}.svg", **save_kw)
    fig.savefig(f"{stem}.pdf", **save_kw)
    for ext in ("png", "svg", "pdf"):
        print("saved", f"{stem}.{ext}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
