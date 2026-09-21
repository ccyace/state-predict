#!/usr/bin/env python
"""Plot fair raw-error densities on shared FP-rollout x_t (any e_<tag> keys)."""
from __future__ import annotations

import argparse
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
X_LABEL = r"element of $e=\varepsilon_{\mathrm{fp}}-\varepsilon_q$ (shared FP $x_t$)"

# stable colors for common tags
COLOR_MAP = {
    "w4a8": "#ff7f0e",
    "w8a8": "#1f77b4",
    "w4a6": "#d62728",
    "w4a7": "#2ca02c",
    "w4a5": "#9467bd",
    "w8a6": "#8c564b",
}
FALLBACK_COLORS = ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


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


def label_of(tag: str) -> str:
    t = tag.lower()
    if t.startswith("w") and "a" in t:
        return t.upper()
    return tag


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument(
        "--tags",
        default="",
        help="comma-separated tags (default: auto-detect all e_* keys)",
    )
    p.add_argument(
        "--stem",
        default="",
        help="output stem without extension (default under error_distributions/)",
    )
    p.add_argument("--title", default="U-ViT-S/2: quant error on shared FP-rollout $x_t$")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.data):
        raise FileNotFoundError(args.data)

    os.makedirs(OUT_DIR, exist_ok=True)
    d = torch.load(args.data, map_location="cpu", mmap=True)

    if args.tags.strip():
        tags = [t.strip().lower() for t in args.tags.split(",") if t.strip()]
    else:
        tags = sorted(k[2:] for k in d.keys() if k.startswith("e_") and torch.is_tensor(d[k]))
        # prefer canonical order
        prefer = ["w4a8", "w8a8", "w4a6", "w4a7", "w4a5", "w8a6"]
        tags = [t for t in prefer if t in tags] + [t for t in tags if t not in prefer]
    if not tags:
        raise RuntimeError("no e_<tag> tensors found in data")

    packs = {}
    summary = {}
    for i, tag in enumerate(tags):
        key = f"e_{tag}"
        if key not in d:
            raise KeyError(f"missing {key} in {args.data}")
        packs[tag] = d[key].float()
        summary[tag] = moments(subsample(packs[tag], 500_000, seed=10 + i))
    print(json.dumps(summary, indent=2), flush=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    for i, tag in enumerate(tags):
        v = subsample(packs[tag], 150_000, seed=100 + i)
        lo, hi = np.quantile(v, [0.001, 0.999])
        v = v[(v >= lo) & (v <= hi)]
        color = COLOR_MAP.get(tag, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
        ax.hist(
            v,
            bins=120,
            density=True,
            histtype="step",
            lw=1.8,
            color=color,
            label=label_of(tag),
        )
    ax.set_yscale("log")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("density (log)")
    ax.set_title(args.title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    stem = args.stem or os.path.join(
        OUT_DIR, "quant_error_fair_shared_xt_" + "_".join(tags)
    )
    if not os.path.isabs(stem):
        stem = os.path.join(OUT_DIR, stem)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
        print("saved", f"{stem}.{ext}", flush=True)
    plt.close(fig)

    with open(stem + "_summary.json", "w") as f:
        json.dump(
            {"summary": summary, "meta": d.get("meta", {}), "data": os.path.abspath(args.data), "tags": tags},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
