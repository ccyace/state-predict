#!/usr/bin/env python
"""Plot fair W4A8 vs W8A8 raw-error densities on shared FP-rollout x_t."""
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
DEFAULT_DATA = os.path.join(OUT_DIR, "paired_fp_rollout_w4_w8.pt")
COLORS = {4: "#ff7f0e", 8: "#1f77b4"}
X_LABEL = r"element of $e=\varepsilon_{\mathrm{fp}}-\varepsilon_q$ (shared FP $x_t$)"


def moments(v: np.ndarray) -> dict:
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)),
        "mse": float(np.mean(v * v)),
        "skew": float(((v - v.mean()) ** 3).mean() / (v.std(ddof=1) ** 3 + 1e-12)),
        "q001": float(np.quantile(v, 0.001)),
        "q999": float(np.quantile(v, 0.999)),
        "left_span": float(abs(np.quantile(v, 0.001) - np.median(v))),
        "right_span": float(abs(np.quantile(v, 0.999) - np.median(v))),
    }


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Missing {data_path}\n"
            "Run first:\n"
            "  python uvit_experiments/collect_paired_weight_error_fp_rollout.py"
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    d = torch.load(data_path, map_location="cpu", mmap=True)
    packs = {4: d["e_w4a8"].float(), 8: d["e_w8a8"].float()}
    summary = {f"w{b}a8": moments(subsample(packs[b], 500_000, seed=b)) for b in (4, 8)}
    print(json.dumps(summary, indent=2), flush=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    for b in (4, 8):
        v = subsample(packs[b], 150_000, seed=100 + b)
        lo, hi = np.quantile(v, [0.001, 0.999])
        v = v[(v >= lo) & (v <= hi)]
        ax.hist(
            v,
            bins=120,
            density=True,
            histtype="step",
            lw=1.8,
            color=COLORS[b],
            label=f"W{b}A8",
        )
    ax.set_yscale("log")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("density (log)")
    ax.set_title("U-ViT-S/2: weight-bit ablation on shared FP-rollout $x_t$")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    stem = os.path.join(OUT_DIR, "quant_error_w4_vs_w8_fair_shared_xt")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
        print("saved", f"{stem}.{ext}", flush=True)
    plt.close(fig)

    with open(os.path.join(OUT_DIR, "quant_error_w4_vs_w8_fair_summary.json"), "w") as f:
        json.dump(
            {"summary": summary, "meta": d.get("meta", {}), "data": os.path.abspath(data_path)},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
