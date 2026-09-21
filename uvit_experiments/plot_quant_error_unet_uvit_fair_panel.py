#!/usr/bin/env python
"""Side-by-side: UNet W4A5/6/7 (left) + U-ViT fair shared-x_t W4A8/W8A8/W4A6 (right)."""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "output", "bitwidth_error_dist_eta0"))
from plot_error_distributions import subsample  # noqa: E402

OUT_DIR = os.path.join(REPO, "uvit_experiments", "outputs", "error_distributions")
UNET_CURVES = os.path.join(OUT_DIR, "unet_w4a567_hist_curves.json")
UVIT_DATA = os.path.join(OUT_DIR, "paired_fp_rollout_w4a8_w8a8_w4a6.pt")
STEM = os.path.join(OUT_DIR, "quant_error_unet_uvit_fair_shared_xt")

GROUP_LEFT = "DDIM based on UNet"
GROUP_RIGHT = "U-ViT-S/2 based on Transformer"
X_LABEL = (
    r"$\mathbf{r}_t^a="
    r"\mathbf{\epsilon}_t^{\mathrm{corr}}"
    r"-\mathbf{\epsilon}_t^{\mathrm{fp}|a}$"
)

UNET_ORDER = [("W4A5", "#d62728"), ("W4A6", "#ff7f0e"), ("W4A7", "#2ca02c")]
UVIT_TAGS = [("w4a8", "#ff7f0e"), ("w8a8", "#1f77b4"), ("w4a6", "#d62728")]


def draw_unet_from_curves(ax, curves: dict):
    for label, color in UNET_ORDER:
        c = curves[label]
        ax.plot(c["x"], c["y"], color=color, lw=1.8, drawstyle="steps-post", label=label)
    ax.set_yscale("log")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("density (log)")
    ax.legend(frameon=True, fontsize=9, loc="upper right")
    ax.grid(alpha=0.25)


def draw_uvit_fair(ax, data_path: str):
    d = torch.load(data_path, map_location="cpu", mmap=True)
    for i, (tag, color) in enumerate(UVIT_TAGS):
        key = f"e_{tag}"
        v = subsample(d[key].float(), 150_000, seed=100 + i)
        lo, hi = np.quantile(v, [0.001, 0.999])
        v = v[(v >= lo) & (v <= hi)]
        ax.hist(
            v,
            bins=120,
            density=True,
            histtype="step",
            lw=1.8,
            color=color,
            label=tag.upper(),
        )
    ax.set_yscale("log")
    ax.set_xlabel(X_LABEL)
    ax.legend(frameon=True, fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isfile(UNET_CURVES):
        raise FileNotFoundError(
            f"missing {UNET_CURVES}; regenerate from quant_error_a_unet_uvit.svg first"
        )
    if not os.path.isfile(UVIT_DATA):
        raise FileNotFoundError(UVIT_DATA)

    with open(UNET_CURVES) as f:
        curves = json.load(f)

    fig = plt.figure(figsize=(11.0, 4.6))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.14, 1],
        hspace=0.10,
        wspace=0.28,
        left=0.08,
        right=0.98,
        top=0.90,
        bottom=0.14,
    )
    ax_gl = fig.add_subplot(gs[0, 0])
    ax_gr = fig.add_subplot(gs[0, 1])
    for ax in (ax_gl, ax_gr):
        ax.axis("off")
    ax_gl.text(0.5, 0.35, GROUP_LEFT, ha="center", va="center", fontsize=13, fontweight="bold")
    ax_gr.text(0.5, 0.35, GROUP_RIGHT, ha="center", va="center", fontsize=13, fontweight="bold")

    ax_l = fig.add_subplot(gs[1, 0])
    ax_r = fig.add_subplot(gs[1, 1])
    draw_unet_from_curves(ax_l, curves)
    draw_uvit_fair(ax_r, UVIT_DATA)

    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{STEM}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
        print("saved", f"{STEM}.{ext}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
