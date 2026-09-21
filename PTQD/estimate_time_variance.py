#!/usr/bin/env python
"""Estimate per-time residual variance tables for PTQD-style VSC (eta=1)."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from ddim.functions.denoising import compute_alpha
from qdiff.trajectory_error import build_ddim_seq
from sample_diffusion_ddim import get_beta_schedule


def trimmed_mean(values: np.ndarray, trim_frac: float = 0.10) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    if x.size < 4:
        return float(x.mean())
    lo = int(np.floor(trim_frac * x.size))
    hi = int(np.ceil((1.0 - trim_frac) * x.size))
    xs = np.sort(x)
    return float(xs[lo:hi].mean())


def fit_student_t_var(values: np.ndarray) -> dict:
    z = np.asarray(values, dtype=np.float64)
    z = z[np.isfinite(z)]
    out = {"n": int(z.size), "nu_mle": None, "sigma_mle": None, "var_mle": None}
    if z.size < 32:
        return out
    nu, _loc, scale = stats.t.fit(z)
    if nu > 2.0:
        out["nu_mle"] = float(nu)
        out["sigma_mle"] = float(scale)
        out["var_mle"] = float(nu / (nu - 2.0) * scale**2)
    return out


def budget_ratio(eps_coef: float, var: float, sigma2: float) -> float:
    if sigma2 <= 0.0:
        return 0.0
    return float((eps_coef**2) * var / sigma2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--output_pt", required=True)
    p.add_argument("--output_json", default="")
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--skip_type", default="quad")
    p.add_argument("--linear_start", type=float, default=0.0001)
    p.add_argument("--linear_end", type=float, default=0.02)
    p.add_argument("--trim_frac", type=float, default=0.10)
    p.add_argument("--student_t_subsample", type=int, default=8000)
    p.add_argument("--student_t_seed", type=int, default=0)
    args = p.parse_args()

    traj = torch.load(args.data, map_location="cpu")
    mse = traj["residual_mse"].float().numpy()
    t_nom = traj["t_nom"].float().numpy()
    eq = traj.get("eq")
    ef = traj.get("ef")

    betas = get_beta_schedule(
        beta_schedule="linear",
        beta_start=args.linear_start,
        beta_end=args.linear_end,
        num_diffusion_timesteps=1000,
    )
    seq = build_ddim_seq(len(betas), args.timesteps, args.skip_type)
    rev = list(reversed(seq))
    seq_next = {int(t): int(n) for t, n in zip(rev[:-1], rev[1:])}
    seq_next[int(rev[-1])] = -1

    betas_t = torch.tensor(betas, dtype=torch.float32)
    per_time = {}
    rng = np.random.default_rng(args.student_t_seed)
    ratios_trim = []
    ratios_actual = []

    for t in sorted({int(round(v)) for v in np.unique(t_nom)}, reverse=True):
        mask = np.isclose(t_nom, float(t))
        m = mse[mask]
        if m.size == 0:
            continue
        next_t = seq_next.get(t, -1)
        if next_t < 0:
            sigma2 = 0.0
            eps_coef = 0.0
        else:
            at = compute_alpha(betas_t, torch.tensor([float(t)]).long())
            at_next = compute_alpha(betas_t, torch.tensor([float(next_t)]).long())
            ratio = (1.0 - at / at_next.clamp(min=at)) * (1.0 - at_next) / (1.0 - at).clamp(min=1e-12)
            sigma2 = float((float(args.eta) ** 2) * ratio.clamp(min=0.0).item())
            c2 = ((1.0 - at_next) - ratio * (float(args.eta) ** 2)).clamp(min=0.0).sqrt()
            eps_coef = float((c2 - (at_next / at).sqrt() * (1.0 - at).sqrt()).item())

        mean_v = float(m.mean())
        trim_v = trimmed_mean(m, args.trim_frac)
        entry = {
            "mean": mean_v,
            "trimmed_mean": trim_v,
            "samples": int(m.size),
            "next_t": int(next_t),
            "sigma2": sigma2,
            "eps_coef": eps_coef,
            "budget_ratio_trimmed": budget_ratio(eps_coef, trim_v, sigma2),
        }

        if eq is not None and ef is not None:
            r = (ef[mask] - eq[mask]).reshape(-1).numpy()
            if r.size > args.student_t_subsample:
                r = rng.choice(r, args.student_t_subsample, replace=False)
            tfit = fit_student_t_var(r)
            entry.update(tfit)
            if tfit["var_mle"] is not None:
                entry["budget_ratio_var_mle"] = budget_ratio(eps_coef, tfit["var_mle"], sigma2)

        per_time[t] = entry
        if sigma2 > 0.0:
            ratios_trim.append(entry["budget_ratio_trimmed"])
            ratios_actual.append(budget_ratio(eps_coef, float(m.mean()), sigma2))

    def summarize(xs):
        xs = np.asarray(xs, dtype=np.float64)
        if xs.size == 0:
            return {"median": 0.0, "p95": 0.0, "fraction_over_1": 0.0}
        return {
            "median": float(np.median(xs)),
            "p95": float(np.quantile(xs, 0.95)),
            "fraction_over_1": float((xs > 1.0).mean()),
        }

    audit = {
        "eta": float(args.eta),
        "states": int(mse.size),
        "trajectories": int(traj.get("meta", {}).get("num_trajectories", 0)),
        "time_only_budget_ratio": summarize(ratios_trim),
        "sample_actual_budget_ratio": summarize(ratios_actual),
        "note": "Final deterministic endpoint with zero sigma budget is excluded from ratios.",
    }

    out = {
        "eta": float(args.eta),
        "statistic": f"{int(args.trim_frac * 100)}% two-sided trimmed mean of per-sample epsilon residual MSE",
        "per_time": per_time,
        "audit": audit,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_pt)) or ".", exist_ok=True)
    torch.save(out, args.output_pt)
    json_path = args.output_json or os.path.splitext(args.output_pt)[0] + ".json"
    json_out = {
        "eta": out["eta"],
        "statistic": out["statistic"],
        "audit": audit,
        "per_time": {str(k): v for k, v in per_time.items()},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)

    n_var = sum(1 for v in per_time.values() if v.get("var_mle") is not None)
    print(f"saved {args.output_pt} ({len(per_time)} steps, var_mle on {n_var} steps)", flush=True)
    print(f"audit median budget (trimmed): {audit['time_only_budget_ratio']['median']:.3e}", flush=True)


if __name__ == "__main__":
    main()
