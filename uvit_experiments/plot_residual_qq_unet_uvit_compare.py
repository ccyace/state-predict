#!/usr/bin/env python
"""Side-by-side Q-Q @ t=399/20: UNet DDIM vs U-ViT-S/2 (mid/late only)."""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

import importlib.util

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BITWIDTH = os.path.join(REPO, "output", "bitwidth_error_dist_eta0")
_UVIT = os.path.join(REPO, "uvit_experiments")
sys.path.insert(0, REPO)
sys.path.insert(0, _BITWIDTH)

from plot_gamma1_by_t import setup_cjk_font
from plot_residual_tails import EPS, load_packs as load_unet_packs

_uvit_spec = importlib.util.spec_from_file_location(
    "uvit_plot_residual_qq",
    os.path.join(_UVIT, "plot_residual_qq_by_stage.py"),
)
_uvit_mod = importlib.util.module_from_spec(_uvit_spec)
_uvit_spec.loader.exec_module(_uvit_mod)
UVIT_BITS = _uvit_mod.BITS
UVIT_COLORS = _uvit_mod.COLORS
load_uvit_packs = _uvit_mod.load_packs
uvit_std_at_t = _uvit_mod.std_at_t

UNET_BITS = [6, 7, 8]
UNET_COLORS = {6: "#ff7f0e", 7: "#2ca02c", 8: "#1f77b4"}
STAGE_TARGETS = [399, 20]
GROUP_LEFT = "DDIM based on UNet"
GROUP_RIGHT = "U-ViT-S/2 based on Transformer"


def nearest_step(steps, target):
    arr = np.asarray(steps, dtype=int)
    return int(arr[np.argmin(np.abs(arr - target))])


def unet_std_at_t(r, t, t_target, n=40_000, seed=0):
    steps = sorted(int(x) for x in torch.unique(t.long()).tolist())
    t_use = nearest_step(steps, t_target)
    m = t.long() == t_use
    v = r[m].reshape(-1).numpy()
    rng = np.random.default_rng(seed)
    if v.size > n:
        v = rng.choice(v, n, replace=False)
    z = (v - v.mean()) / (v.std() + EPS)
    return z, t_use


def draw_unet_panel(ax, packs, t_target, *, seed_base=0):
    t_used = None
    for b in UNET_BITS:
        r, t = packs[b]
        z, t_use = unet_std_at_t(r, t, t_target, seed=seed_base + b)
        t_used = t_use
        (osm, osr), _ = stats.probplot(z, dist="norm")
        ax.plot(
            osm, osr, ".", ms=1.0, alpha=0.28, color=UNET_COLORS[b],
            label=f"W4A{b}", rasterized=True,
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


def draw_uvit_panel(ax, packs, t_target, *, seed_base=0):
    t_used = None
    for b in UVIT_BITS:
        e, t = packs[b]
        z, t_use = uvit_std_at_t(e, t, t_target, seed=seed_base + b)
        t_used = t_use
        (osm, osr), _ = stats.probplot(z, dist="norm")
        ax.plot(
            osm, osr, ".", ms=1.0, alpha=0.28, color=UVIT_COLORS[b],
            label=f"W{b}A8", rasterized=True,
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
    out_dir = os.path.join(_UVIT, "outputs", "error_distributions")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = load_unet_packs(device)
    uvit = load_uvit_packs()

    fig = plt.figure(figsize=(13.5, 5.0))
    gs = fig.add_gridspec(
        2, 4, height_ratios=[0.12, 1], hspace=0.08, wspace=0.28,
        left=0.06, right=0.98, top=0.88, bottom=0.12,
    )
    ax_gl = fig.add_subplot(gs[0, 0:2])
    ax_gr = fig.add_subplot(gs[0, 2:4])
    for ax in (ax_gl, ax_gr):
        ax.axis("off")

    ax_gl.text(0.5, 0.35, GROUP_LEFT, ha="center", va="center", fontsize=13, fontweight="bold")
    ax_gr.text(0.5, 0.35, GROUP_RIGHT, ha="center", va="center", fontsize=13, fontweight="bold")

    axes = [fig.add_subplot(gs[1, j]) for j in range(4)]
    resolved = {"unet": [], "uvit": []}
    for j, tgt in enumerate(STAGE_TARGETS):
        t_u = draw_unet_panel(axes[j], unet, tgt, seed_base=100 * tgt)
        resolved["unet"].append(t_u)
    for j, tgt in enumerate(STAGE_TARGETS):
        t_v = draw_uvit_panel(axes[2 + j], uvit, tgt, seed_base=200 * tgt + 50)
        resolved["uvit"].append(t_v)

    # y-label on first panel of each group only
    for ax in (axes[1], axes[3]):
        ax.set_ylabel("")

    print("UNet timesteps:", resolved["unet"], flush=True)
    print("U-ViT timesteps:", resolved["uvit"], flush=True)

    stem = os.path.join(out_dir, "residual_qq_by_stage_unet_uvit")
    save_kw = dict(bbox_inches="tight", dpi=300)
    fig.savefig(f"{stem}.png", **save_kw)
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    for ext in ("png", "svg", "pdf"):
        print("saved", f"{stem}.{ext}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
