#!/usr/bin/env python
"""Plot whether a timestep-only scalar variance captures residual scale."""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch

from state_aware_temporal_joint.time_corrected_residual import load_residual


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--strength", type=float, default=1.0)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(a.data, map_location="cpu", mmap=True)
    ids = d["traj_id"].long()
    use = torch.where((ids % 10 == 0) | (ids % 10 == 5))[0]
    net = load_residual(a.ckpt, device).net.eval()
    energy, times, trajectories = [], [], []
    for start in range(0, len(use), a.batch_size):
        ii = use[start:start + a.batch_size]
        x = d["x"][ii].float().to(device)
        eq = d["eq"][ii].float().to(device)
        ef = d["ef"][ii].float().to(device)
        tn = d["t_nom"][ii].float().to(device)
        tc = d["t_corr"][ii].float().to(device)
        rf = d["is_refresh"][ii].float().to(device)
        r = (ef - eq) - a.strength * net(x, eq, tn, tc, rf)
        energy.append(r.square().mean((1, 2, 3)).cpu().numpy())
        times.append(tn.long().cpu().numpy())
        trajectories.append(ids[ii].numpy())
    e = np.concatenate(energy)
    t = np.concatenate(times)
    traj = np.concatenate(trajectories)
    cal, test = traj % 10 == 0, traj % 10 == 5

    steps = np.asarray(sorted(np.unique(t)))
    v_cal = np.asarray([e[cal & (t == k)].mean() for k in steps])
    v_test = np.asarray([e[test & (t == k)].mean() for k in steps])

    # The reported state-level score uses the same log-energy target as the core ablation.
    lookup = {int(k): np.log(v) for k, v in zip(steps, v_cal)}
    y = np.log(np.clip(e[test], 1e-12, None))
    pred = np.asarray([lookup[int(k)] for k in t[test]])
    r2_state = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    # Agreement between independently estimated per-time variance curves.
    lc, lt = np.log(v_cal), np.log(v_test)
    r2_curve = 1 - np.sum((lt - lc) ** 2) / np.sum((lt - lt.mean()) ** 2)
    rel = np.abs(v_test - v_cal) / np.maximum(v_test, 1e-30)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25))
    axes[0].plot(steps, v_test, color="#2878b5", lw=2.1, label="held-out observed")
    axes[0].plot(steps, v_cal, color="#e07b39", lw=1.8, ls="--",
                 label=r"timestep estimate $v_t$")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("nominal diffusion timestep")
    axes[0].set_ylabel(r"residual variance $\mathbb{E}[\|r_t\|^2/D]$")
    axes[0].set_title("Residual variance is primarily timestep-dependent")
    axes[0].grid(alpha=.2); axes[0].legend(frameon=False)

    lo = min(lc.min(), lt.min()); hi = max(lc.max(), lt.max())
    axes[1].scatter(lc, lt, s=24, alpha=.75, color="#2878b5", edgecolors="none")
    axes[1].plot([lo, hi], [lo, hi], color="black", ls="--", lw=1, label="perfect agreement")
    axes[1].set_xlabel(r"calibration $\log v_t$")
    axes[1].set_ylabel(r"held-out $\log v_t$")
    axes[1].set_title("Generalization to unseen trajectories")
    axes[1].text(.04, .94,
                 rf"state-level $R^2={r2_state:.4f}$" + "\n" +
                 rf"time-curve $R^2={r2_curve:.4f}$",
                 transform=axes[1].transAxes, va="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=.85, edgecolor="0.8"))
    axes[1].grid(alpha=.2); axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    output = os.path.join(a.output_dir, "time_variance_sufficiency.png")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    report = {
        "hypothesis": "The dominant scalar residual variance is determined by timestep.",
        "variance_estimator": "v_t = E[||r_t||^2 / D | t] on calibration trajectories",
        "split": {"calibration_trajectories": int(len(np.unique(traj[cal]))),
                  "test_trajectories": int(len(np.unique(traj[test]))),
                  "calibration_states": int(cal.sum()), "test_states": int(test.sum())},
        "state_level_log_energy_r2": float(r2_state),
        "per_timestep_log_variance_curve_r2": float(r2_curve),
        "per_timestep_relative_error": {"median": float(np.median(rel)),
                                        "p95": float(np.quantile(rel, .95)),
                                        "max": float(np.max(rel))},
        "interpretation": "A timestep-only scalar variance captures the dominant residual-scale variation on unseen trajectories; this is a sufficiency result, not proof of strict state independence or isotropy.",
    }
    with open(os.path.join(a.output_dir, "time_variance_sufficiency_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
