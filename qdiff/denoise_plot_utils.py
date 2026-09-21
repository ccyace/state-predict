"""Shared axis helpers: unify DDIM step index with diffusion timestep t."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def denoise_steps_table(seq: Sequence[int]) -> List[Dict[str, int]]:
    """DDIM denoise order: step 0 @ highest t -> step N-1 @ lowest t."""
    rev = list(reversed(seq))
    return [{"step": i, "t": int(rev[i])} for i in range(len(rev))]


def sort_by_denoise_step(records):
    return sorted(records, key=lambda r: int(r["step"]))


def denoise_axes(records) -> Tuple[np.ndarray, np.ndarray]:
    """Return (steps, t_vals) sorted in denoise order."""
    recs = sort_by_denoise_step(records)
    steps = np.array([int(r["step"]) for r in recs], dtype=int)
    t_vals = np.array([int(r["t"]) for r in recs], dtype=int)
    return steps, t_vals


def setup_denoise_xaxis(ax, steps: np.ndarray, t_vals: np.ndarray, n_ticks: int = 10):
    """
    Use step as x coordinate; tick labels show both step and diffusion t.
    Left -> right follows denoising: high t (noisy) to t=0 (clean).
    """
    n = len(steps)
    if n == 0:
        return
    idx = np.unique(np.round(np.linspace(0, n - 1, min(n_ticks, n))).astype(int))
    ax.set_xticks(steps[idx])
    ax.set_xticklabels([f"s{steps[i]}\nt={t_vals[i]}" for i in idx], fontsize=7)
    ax.set_xlabel("DDIM denoise order  (left: high t / noisy  →  right: t=0 / clean)")


# Regions in denoising order (early -> late along sampling)
DENOISE_REGIONS = [
    (200, 801, "t≥200\n(denoise early)"),
    (50, 200, "50≤t<200\n(denoise mid)"),
    (0, 50, "t<50\n(denoise late)"),
]
