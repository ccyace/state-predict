#!/usr/bin/env python
"""Side-by-side panel (a): raw quantization error, UNet vs U-ViT."""
from __future__ import annotations

import importlib.util
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BITWIDTH = os.path.join(REPO, "output", "bitwidth_error_dist_eta0")
sys.path.insert(0, REPO)
sys.path.insert(0, _BITWIDTH)

from plot_error_distributions import load_er, subsample

_uvit_spec = importlib.util.spec_from_file_location(
    "uvit_plot_error_distributions",
    os.path.join(REPO, "uvit_experiments", "plot_error_distributions.py"),
)
_uvit_mod = importlib.util.module_from_spec(_uvit_spec)
_uvit_spec.loader.exec_module(_uvit_mod)
load_uvit_raw_e = _uvit_mod.load_raw_e
UVIT_BITS = _uvit_mod.BITS
UVIT_COLORS = _uvit_mod.COLORS
UVIT_CFG = _uvit_mod.CFG

UNET_BITS = [5, 6, 7]
UNET_COLORS = {5: "#d62728", 6: "#ff7f0e", 7: "#2ca02c"}
GROUP_LEFT = "DDIM based on UNet"
GROUP_RIGHT = "U-ViT-S/2 based on Transformer"
OUT_DIR = os.path.join(REPO, "uvit_experiments", "outputs", "error_distributions")
X_LABEL = (
    r"$\mathbf{r}_t^a="
    r"\mathbf{\epsilon}_t^{\mathrm{corr}}"
    r"-\mathbf{\epsilon}_t^{\mathrm{fp}|a}$"
)


def load_unet_raw_e(device: torch.device) -> dict[int, torch.Tensor]:
    packs = {}
    for b in UNET_BITS:
        data = os.path.join(REPO, f"output/bitwidth_error_dist_eta0/w4a{b}/traj_eta0_n200.pt")
        ckpt = os.path.join(REPO, f"output/bitwidth_error_dist_eta0/w4a{b}/checkpoints/ckpt_best.pt")
        e, _r, _t = load_er(data, ckpt, device)
        packs[b] = e
        print(f"UNet W4A{b}: e shape {tuple(e.shape)}", flush=True)
    return packs


def load_uvit_raw_e_all() -> dict[int, torch.Tensor]:
    packs = {}
    for b in UVIT_BITS:
        ol = os.path.join(REPO, UVIT_CFG[b]["openloop"])
        e = load_uvit_raw_e(ol)
        packs[b] = e
        print(f"U-ViT W{b}A8: e shape {tuple(e.shape)}", flush=True)
    return packs


def draw_raw_e_panel(ax, packs: dict[int, torch.Tensor], bits: list[int], colors: dict[int, str], *, seed_base: int = 0):
    for b in bits:
        v = subsample(packs[b], 150_000, seed=seed_base + b)
        lo, hi = np.quantile(v, [0.001, 0.999])
        v = v[(v >= lo) & (v <= hi)]
        label = f"W4A{b}" if b in UNET_BITS else f"W{b}A8"
        ax.hist(v, bins=120, density=True, histtype="step", lw=1.8, color=colors[b], label=label)
    ax.set_yscale("log")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("density (log)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet = load_unet_raw_e(device)
    uvit = load_uvit_raw_e_all()

    fig = plt.figure(figsize=(11.0, 4.6))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[0.14, 1], hspace=0.10, wspace=0.28,
        left=0.08, right=0.98, top=0.90, bottom=0.14,
    )
    ax_gl = fig.add_subplot(gs[0, 0])
    ax_gr = fig.add_subplot(gs[0, 1])
    for ax in (ax_gl, ax_gr):
        ax.axis("off")
    ax_gl.text(0.5, 0.35, GROUP_LEFT, ha="center", va="center", fontsize=13, fontweight="bold")
    ax_gr.text(0.5, 0.35, GROUP_RIGHT, ha="center", va="center", fontsize=13, fontweight="bold")

    ax_l = fig.add_subplot(gs[1, 0])
    ax_r = fig.add_subplot(gs[1, 1])
    draw_raw_e_panel(ax_l, unet, UNET_BITS, UNET_COLORS, seed_base=0)
    draw_raw_e_panel(ax_r, uvit, UVIT_BITS, UVIT_COLORS, seed_base=100)
    ax_r.set_ylabel("")

    stem = os.path.join(OUT_DIR, "quant_error_a_unet_uvit")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    for ext in ("png", "svg", "pdf"):
        print("saved", f"{stem}.{ext}", flush=True)


if __name__ == "__main__":
    main()
