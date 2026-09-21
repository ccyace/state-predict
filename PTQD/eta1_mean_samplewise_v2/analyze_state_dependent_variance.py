#!/usr/bin/env python
"""Core ablation: does observable state explain residual variance beyond timestep?"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT]

import matplotlib.pyplot as plt
import numpy as np
import torch

from state_aware_temporal_joint.time_corrected_residual import load_residual, lookup


def time_lookup_fit(t, y):
    return {int(k): float(np.mean(y[t == k])) for k in np.unique(t)}


def time_lookup_predict(table, t, fallback):
    return np.asarray([table.get(int(k), fallback) for k in t], dtype=np.float64)


def metrics(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return {
        "r2_log_variance": float(1 - ss_res / max(ss_tot, 1e-30)),
        "mae_log_variance": float(np.mean(np.abs(y - p))),
        "rmse_log_variance": float(np.mean((y - p) ** 2) ** 0.5),
    }


def bootstrap_delta(y, p0, p1, traj, seed=2026, repeats=2000):
    rng = np.random.default_rng(seed)
    unique = np.unique(traj)
    by_traj = {k: np.flatnonzero(traj == k) for k in unique}
    vals = []
    for _ in range(repeats):
        draw = rng.choice(unique, len(unique), replace=True)
        ii = np.concatenate([by_traj[k] for k in draw])
        vals.append(metrics(y[ii], p1[ii])["r2_log_variance"] -
                    metrics(y[ii], p0[ii])["r2_log_variance"])
    q = np.quantile(vals, [0.025, 0.5, 0.975])
    return {"repeats": repeats, "median": float(q[1]), "ci95": [float(q[0]), float(q[2])],
            "probability_delta_positive": float(np.mean(np.asarray(vals) > 0))}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--strength", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(args.data, map_location="cpu", mmap=True)
    ids = d["traj_id"].long()
    # Both halves come from the correction model's held-out trajectories.
    use = torch.where((ids % 10 == 0) | (ids % 10 == 5))[0]
    net = load_residual(args.ckpt, device).net.eval()
    features, targets, times, out_ids = [], [], [], []

    for start in range(0, len(use), args.batch_size):
        ii = use[start:start + args.batch_size]
        x = d["x"][ii].float().to(device)
        eq = d["eq"][ii].float().to(device)
        ef = d["ef"][ii].float().to(device)
        tn = d["t_nom"][ii].float().to(device)
        tc = d["t_corr"][ii].float().to(device)
        rf = d["is_refresh"][ii].float().to(device)
        residual = (ef - eq) - args.strength * net(x, eq, tn, tc, rf)
        # Scalar isotropic variance target: trace(Cov)/D, estimated per state.
        target = residual.square().mean((1, 2, 3)).clamp_min(1e-12).log()
        # All features are observable at inference; no FP teacher/residual feature.
        feat = torch.cat([
            tn[:, None] / 999.0,
            lookup(net.logsnr_table, tn)[:, None],
            (tc - tn)[:, None] / 20.0,
            rf[:, None],
            x.mean((-2, -1)), x.std((-2, -1), unbiased=False),
            eq.mean((-2, -1)), eq.std((-2, -1), unbiased=False),
            x.flatten(1).square().mean(1, keepdim=True).sqrt(),
            eq.flatten(1).square().mean(1, keepdim=True).sqrt(),
        ], 1)
        features.append(feat.cpu().numpy())
        targets.append(target.cpu().numpy())
        times.append(tn.long().cpu().numpy())
        out_ids.append(ids[ii].numpy())

    X = np.concatenate(features); y = np.concatenate(targets)
    t = np.concatenate(times); traj = np.concatenate(out_ids)
    train = traj % 10 == 0
    test = traj % 10 == 5

    table = time_lookup_fit(t[train], y[train])
    fallback = float(np.mean(y[train]))
    pred_time_train = time_lookup_predict(table, t[train], fallback)
    pred_time = time_lookup_predict(table, t[test], fallback)

    # Nested ablation: preserve the exact time lookup, then predict only its
    # remaining error from state. Interactions allow state effects to vary with t.
    state = X[:, 2:]
    time_coord = X[:, 0:1]
    state_aug = np.concatenate([state, state * time_coord, state * time_coord ** 2], 1)
    mu = state_aug[train].mean(0); scale = state_aug[train].std(0).clip(1e-8)
    xa = np.concatenate([(state_aug[train] - mu) / scale, np.ones((train.sum(), 1))], 1)
    xb = np.concatenate([(state_aug[test] - mu) / scale, np.ones((test.sum(), 1))], 1)
    gram = xa.T @ xa
    penalty = np.eye(xa.shape[1]) * 1e-2; penalty[-1, -1] = 0
    weights = np.linalg.solve(gram + penalty, xa.T @ (y[train] - pred_time_train))
    pred_state = pred_time + xb @ weights
    m0, m1 = metrics(y[test], pred_time), metrics(y[test], pred_state)
    boot = bootstrap_delta(y[test], pred_time, pred_state, traj[test])

    # Variance calibration in five predicted-risk groups.
    edges = np.quantile(pred_state, np.linspace(0, 1, 6))
    bins = []
    for k in range(5):
        mask = (pred_state >= edges[k]) & ((pred_state <= edges[k+1]) if k == 4 else (pred_state < edges[k+1]))
        bins.append({
            "n": int(mask.sum()),
            "predicted_variance": float(np.mean(np.exp(pred_state[mask]))),
            "observed_variance": float(np.mean(np.exp(y[test][mask]))),
        })
    observed = np.asarray([b["observed_variance"] for b in bins])
    dynamic_range = float(observed[-1] / max(observed[0], 1e-30))

    report = {
        "definition": {
            "target": "log(mean((epsilon_fp - epsilon_q - mean_net(c_t))^2)) per state",
            "time_only": "per-nominal-timestep lookup fitted on calibration trajectories",
            "state_time": "exact time lookup plus standardized ridge correction from inference-observable state summaries and time interactions",
            "variance_form_tested": "Sigma_u(c_t) = v(c_t) I",
            "strength": args.strength,
        },
        "split": {
            "calibration_rule": "traj_id % 10 == 0",
            "test_rule": "traj_id % 10 == 5",
            "calibration_trajectories": int(len(np.unique(traj[train]))),
            "test_trajectories": int(len(np.unique(traj[test]))),
            "calibration_states": int(train.sum()),
            "test_states": int(test.sum()),
        },
        "time_only": m0,
        "state_plus_time": m1,
        "improvement": {
            "delta_r2": float(m1["r2_log_variance"] - m0["r2_log_variance"]),
            "relative_mae_reduction": float(1 - m1["mae_log_variance"] / m0["mae_log_variance"]),
            "trajectory_bootstrap_delta_r2": boot,
        },
        "state_model_risk_quintiles": bins,
        "observed_top_to_bottom_variance_ratio": dynamic_range,
        "decision": {
            "state_information_beyond_time_supported": bool(boot["ci95"][0] > 0),
            "criterion": "trajectory-bootstrap 95% CI of Delta R2 is entirely above zero",
        },
        "scope": "This tests state-dependent scalar residual energy, not Gaussianity or exact spatial/channel isotropy.",
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    names = ["time only", "time + state"]
    vals = [m0["r2_log_variance"], m1["r2_log_variance"]]
    axes[0].bar(names, vals, color=["#9e9e9e", "#2878b5"])
    axes[0].axhline(0, color="black", lw=.8)
    axes[0].set_ylabel(r"held-out $R^2$ of $\log v$")
    axes[0].set_title(r"Does $c_t$ explain variance beyond $t$?")
    axes[0].text(.5, max(vals) * .55, rf"$\Delta R^2={vals[1]-vals[0]:.3f}$" + "\n" +
                 rf"95% CI [{boot['ci95'][0]:.3f}, {boot['ci95'][1]:.3f}]", ha="center")
    pred_b = np.asarray([b["predicted_variance"] for b in bins])
    obs_b = np.asarray([b["observed_variance"] for b in bins])
    axes[1].plot(range(1, 6), obs_b, "o-", label="observed")
    axes[1].plot(range(1, 6), pred_b, "s--", label="predicted")
    axes[1].set_yscale("log"); axes[1].set_xlabel("predicted variance quintile")
    axes[1].set_ylabel(r"residual variance $v$")
    axes[1].set_title(r"State-conditioned variance calibration")
    axes[1].legend(); axes[1].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "state_variance_ablation.png"), dpi=200)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
