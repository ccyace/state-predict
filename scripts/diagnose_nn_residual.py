#!/usr/bin/env python3
"""Diagnose residual quantization error after a learned DeltaEpsNet.

Primary residual (network fit):
  Δ* = eps_fp - eps_q
  Δ̂ = net(x, eps_q, t)
  r  = Δ* - Δ̂ = eps_fp - (eps_q + Δ̂)

Optional inference residual (with blend / clip schedule):
  r_inf = eps_fp - corrector.correct(eps_q, t, ·, xt=x)

Writes REPORT.md, metrics.json, and diagnostic plots under --out_dir.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from noise_eps_corr.learned_noise_corrector import (  # noqa: E402
    load_learned_corrector,
)

EPS = 1e-12


def _betas_linear(T: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)


def ddim_b_coeff(t: int, t_prev: int, T: int = 1000) -> float:
    """η=0 DDIM: x_prev = a*x + b*eps; return b so δx ≈ b * δε (same-x linearization)."""
    betas = _betas_linear(T)
    alphas = 1.0 - betas
    a_bar = torch.cumprod(alphas, dim=0)
    at = a_bar[int(t)].clamp(min=EPS)
    atp = a_bar[int(t_prev)].clamp(min=EPS) if t_prev >= 0 else torch.tensor(1.0, dtype=torch.float64)
    # x_prev = sqrt(atp)/sqrt(at)*x + (sqrt(1-atp) - sqrt(atp*(1-at)/at)) * eps
    b = torch.sqrt(1.0 - atp) - torch.sqrt(atp * (1.0 - at) / at)
    return float(b.item())


@torch.no_grad()
def collect_residuals(
    raw: dict,
    net,
    device: torch.device,
    *,
    val_mod: int = 5,
    batch_size: int = 64,
    max_samples: int = 0,
    corrector=None,
) -> Dict[str, torch.Tensor]:
    traj_id = raw["traj_id"].long()
    mask = (traj_id % val_mod) == 0
    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    if max_samples > 0 and idx.numel() > max_samples:
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(idx.numel(), generator=g)[:max_samples]
        idx = idx[perm]

    chunks = {
        "t": [],
        "traj_id": [],
        "delta_star": [],
        "delta_hat": [],
        "r": [],
        "eq": [],
        "r_inf": [],
    }
    for start in range(0, idx.numel(), batch_size):
        ii = idx[start : start + batch_size]
        x = raw["x"][ii].float().to(device)
        eq = raw["eq"][ii].float().to(device)
        ef = raw["ef"][ii].float().to(device)
        t = raw["t"][ii].float().to(device)
        d_star = ef - eq
        d_hat = net(x, eq, t)
        r = d_star - d_hat
        chunks["t"].append(t.cpu())
        chunks["traj_id"].append(raw["traj_id"][ii].cpu())
        chunks["delta_star"].append(d_star.cpu())
        chunks["delta_hat"].append(d_hat.cpu())
        chunks["r"].append(r.cpu())
        chunks["eq"].append(eq.cpu())
        if corrector is not None:
            # per-sample t
            r_inf_list = []
            for j in range(eq.shape[0]):
                ec = corrector.correct(
                    eq[j : j + 1],
                    float(t[j].item()),
                    torch.ones(1, device=device),
                    xt=x[j : j + 1],
                )
                r_inf_list.append((ef[j : j + 1] - ec).cpu())
            chunks["r_inf"].append(torch.cat(r_inf_list, dim=0))
        print(f"  collected {min(start+batch_size, idx.numel())}/{idx.numel()}", flush=True)

    out = {k: torch.cat(v, dim=0) for k, v in chunks.items() if v}
    return out


def _flat(x: torch.Tensor) -> np.ndarray:
    return x.reshape(-1).float().numpy()


def pearson_rho(a: torch.Tensor, b: torch.Tensor, max_n: int = 2_000_000) -> float:
    af = a.reshape(-1).float()
    bf = b.reshape(-1).float()
    n = af.numel()
    if n > max_n:
        g = torch.Generator().manual_seed(1)
        sel = torch.randperm(n, generator=g)[:max_n]
        af, bf = af[sel], bf[sel]
    af = af - af.mean()
    bf = bf - bf.mean()
    den = af.norm() * bf.norm()
    if float(den) < EPS:
        return 0.0
    return float((af * bf).sum() / den)


def moments(x: torch.Tensor) -> Dict[str, float]:
    f = x.reshape(-1).float()
    mean = float(f.mean())
    var = float(f.var(unbiased=False))
    # subsample for kurtosis / qq
    n = f.numel()
    if n > 500_000:
        g = torch.Generator().manual_seed(2)
        f = f[torch.randperm(n, generator=g)[:500_000]]
    arr = f.numpy()
    # excess kurtosis
    m4 = float(np.mean((arr - arr.mean()) ** 4))
    exkurt = m4 / (var**2 + EPS) - 3.0
    # Shapiro-like via QQ correlation vs normal
    arr_s = np.sort(arr)
    n = arr_s.size
    theo = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    # downsample for corr
    step = max(1, n // 20000)
    qq_r = float(np.corrcoef(theo[::step], arr_s[::step])[0, 1])
    return {
        "mean": mean,
        "var": var,
        "std": math.sqrt(max(var, 0.0)),
        "mse": float((f**2).mean()) if f.numel() else 0.0,
        "exkurt": exkurt,
        "qq_r": qq_r,
        "p01": float(np.quantile(arr, 0.01)),
        "p50": float(np.quantile(arr, 0.50)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def energy_r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    """1 - ||pred-target||^2 / ||target||^2  (here pred=Δ̂, target=Δ*)."""
    num = (pred - target).pow(2).mean().item()
    den = target.pow(2).mean().item()
    return float(1.0 - num / max(den, EPS))


def per_t_stats(t: torch.Tensor, r: torch.Tensor, d_star: torch.Tensor, d_hat: torch.Tensor) -> List[dict]:
    t_keys = t.long()
    uniq = sorted(int(u) for u in torch.unique(t_keys).tolist())
    rows = []
    for tk in uniq:
        m = t_keys == tk
        rr = r[m]
        ds = d_star[m]
        dh = d_hat[m]
        e_r = float(rr.pow(2).mean())
        e_d = float(ds.pow(2).mean())
        rows.append(
            {
                "t": tk,
                "n": int(m.sum()),
                "mean_r": float(rr.mean()),
                "var_r": float(rr.var(unbiased=False)),
                "mse_r": e_r,
                "mse_delta": e_d,
                "r2_vs0": float(1.0 - e_r / max(e_d, EPS)),
                "rho_r_dhat": pearson_rho(rr, dh),
                "rho_r_dstar": pearson_rho(rr, ds),
            }
        )
    return rows


def per_t_rho_eq(t: torch.Tensor, r: torch.Tensor, eq: torch.Tensor) -> Dict[int, float]:
    t_keys = t.long()
    out = {}
    for tk in torch.unique(t_keys).tolist():
        m = t_keys == int(tk)
        out[int(tk)] = pearson_rho(r[m], eq[m])
    return out


def bin_hetero(r: torch.Tensor, ref: torch.Tensor, n_bins: int = 40) -> Dict[str, float]:
    """Heteroscedasticity of r vs ref (e.g. eq or Δ̂): CV of bin variances."""
    rf = ref.reshape(-1).float()
    rr = r.reshape(-1).float()
    n = rf.numel()
    if n > 1_000_000:
        g = torch.Generator().manual_seed(3)
        sel = torch.randperm(n, generator=g)[:1_000_000]
        rf, rr = rf[sel], rr[sel]
    qs = torch.linspace(0, 1, n_bins + 1)
    edges = torch.quantile(rf, qs)
    vars_ = []
    means_ = []
    centers = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            m = (rf >= lo) & (rf < hi)
        else:
            m = (rf >= lo) & (rf <= hi)
        if int(m.sum()) < 32:
            continue
        vals = rr[m]
        vars_.append(float(vals.var(unbiased=False)))
        means_.append(float(vals.mean()))
        centers.append(float(0.5 * (lo + hi)))
    if not vars_:
        return {"bin_var_cv": float("nan"), "bin_mean_r2": float("nan"), "n_bins_used": 0}
    v = np.asarray(vars_)
    mu = np.asarray(means_)
    c = np.asarray(centers)
    cv = float(v.std() / (v.mean() + EPS))
    # linearity of E[r|ref]
    if c.size >= 3 and np.std(c) > EPS:
        coef = np.polyfit(c, mu, 1)
        pred = coef[0] * c + coef[1]
        ss_res = np.sum((mu - pred) ** 2)
        ss_tot = np.sum((mu - mu.mean()) ** 2) + EPS
        bin_r2 = float(1.0 - ss_res / ss_tot)
    else:
        bin_r2 = float("nan")
    return {"bin_var_cv": cv, "bin_mean_r2": bin_r2, "n_bins_used": int(len(vars_))}


def temporal_corr_r(
    t: torch.Tensor,
    traj_id: torch.Tensor,
    r: torch.Tensor,
    max_pairs: int = 256,
) -> List[dict]:
    """Adjacent-grid temporal corr of residual, paired by traj_id."""
    t_keys = t.long()
    uniq = sorted((int(u) for u in torch.unique(t_keys).tolist()), reverse=True)
    rows = []
    for t_hi, t_lo in zip(uniq[:-1], uniq[1:]):
        m_hi = t_keys == t_hi
        m_lo = t_keys == t_lo
        map_lo = {
            int(tr): i
            for tr, i in zip(traj_id[m_lo].tolist(), torch.nonzero(m_lo, as_tuple=False).squeeze(1).tolist())
        }
        hi_idx, lo_idx = [], []
        for tr, i in zip(traj_id[m_hi].tolist(), torch.nonzero(m_hi, as_tuple=False).squeeze(1).tolist()):
            j = map_lo.get(int(tr))
            if j is not None:
                hi_idx.append(int(i))
                lo_idx.append(int(j))
        if len(hi_idx) < 8:
            continue
        if len(hi_idx) > max_pairs:
            hi_idx = hi_idx[:max_pairs]
            lo_idx = lo_idx[:max_pairs]
        rh = r[hi_idx].reshape(len(hi_idx), -1).float()
        rl = r[lo_idx].reshape(len(lo_idx), -1).float()
        rh = rh - rh.mean(dim=1, keepdim=True)
        rl = rl - rl.mean(dim=1, keepdim=True)
        # mean over spatial of per-sample pearson, then average
        num = (rh * rl).sum(dim=1)
        den = rh.norm(dim=1) * rl.norm(dim=1)
        corr = (num / den.clamp(min=EPS)).mean().item()
        rows.append({"t_hi": t_hi, "t_lo": t_lo, "corr": float(corr), "n_pairs": len(hi_idx)})
    return rows


def plot_suite(data: dict, per_t: List[dict], temp: List[dict], out_dir: str, tag: str):
    r = data["r"]
    d_star = data["delta_star"]
    d_hat = data["delta_hat"]
    eq = data["eq"]
    t = data["t"]

    # 1) hist Δ* vs r
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for axi, tens, name in [(ax[0], d_star, "Δ*"), (ax[1], r, "r=Δ*-Δ̂")]:
        arr = _flat(tens)
        if arr.size > 300_000:
            arr = np.random.default_rng(0).choice(arr, 300_000, replace=False)
        axi.hist(arr, bins=120, density=True, alpha=0.85, color="#3b6ea5")
        axi.set_title(f"{tag}: {name}")
        axi.set_xlabel("value")
        axi.set_ylabel("density")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_hist_delta_vs_r.png"), dpi=140)
    plt.close(fig)

    # 2) QQ of r
    arr = _flat(r)
    if arr.size > 200_000:
        arr = np.random.default_rng(1).choice(arr, 200_000, replace=False)
    fig, ax = plt.subplots(figsize=(5, 5))
    stats.probplot(arr, dist="norm", plot=ax)
    ax.set_title(f"{tag}: QQ of residual r")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_qq_r.png"), dpi=140)
    plt.close(fig)

    # 3) scatter r vs eq / r vs Δ̂
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    rng = np.random.default_rng(2)
    for axi, ref, name in [(ax[0], eq, "ε_q"), (ax[1], d_hat, "Δ̂")]:
        a = ref.reshape(-1).float().numpy()
        b = r.reshape(-1).float().numpy()
        if a.size > 80_000:
            sel = rng.choice(a.size, 80_000, replace=False)
            a, b = a[sel], b[sel]
        axi.scatter(a, b, s=1, alpha=0.05, c="#333")
        axi.set_xlabel(name)
        axi.set_ylabel("r")
        axi.set_title(f"{tag}: r vs {name}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_scatter_r.png"), dpi=140)
    plt.close(fig)

    # 4) per-t variance / R² / mean
    ts = [row["t"] for row in per_t]
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(ts, [row["var_r"] for row in per_t], label="Var(r)")
    ax[0].plot(ts, [row["mse_delta"] for row in per_t], label="E‖Δ*‖²", alpha=0.7)
    ax[0].set_ylabel("energy")
    ax[0].legend()
    ax[0].set_title(f"{tag}: per-t residual energy")
    ax[1].plot(ts, [row["r2_vs0"] for row in per_t], color="#c44")
    ax[1].set_ylabel("R² of Δ̂ vs 0")
    ax[1].set_ylim(-0.05, 1.05)
    ax[2].plot(ts, [row["mean_r"] for row in per_t], color="#2a7")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_ylabel("E[r]")
    ax[2].set_xlabel("t")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_per_t.png"), dpi=140)
    plt.close(fig)

    # 5) temporal corr
    if temp:
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot([row["t_hi"] for row in temp], [row["corr"] for row in temp], "-o", ms=3)
        ax.set_xlabel("t_hi (transition t_hi→t_lo)")
        ax.set_ylabel("corr(r_hi, r_lo)")
        ax.set_title(f"{tag}: temporal corr of residual (traj-paired)")
        ax.set_ylim(-0.2, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{tag}_temporal_corr.png"), dpi=140)
        plt.close(fig)

    # 6) next-step |b_t| * rms(r)
    # use sorted ascending unique t; pair with previous lower t in schedule
    uniq = sorted(set(ts))
    b_rms = []
    for i, tk in enumerate(uniq):
        t_prev = uniq[i - 1] if i > 0 else -1
        # sampling goes high→low; for each t, next state uses t_prev = next lower in schedule
        # find next lower
        lowers = [u for u in uniq if u < tk]
        t_next = max(lowers) if lowers else -1
        b = ddim_b_coeff(tk, t_next)
        row = next(x for x in per_t if x["t"] == tk)
        rms = math.sqrt(max(row["mse_r"], 0.0))
        b_rms.append({"t": tk, "b": b, "rms_r": rms, "rms_dx": abs(b) * rms})
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot([x["t"] for x in b_rms], [x["rms_dx"] for x in b_rms], label="|b|·rms(r)")
    ax.plot([x["t"] for x in b_rms], [x["rms_r"] for x in b_rms], alpha=0.6, label="rms(r)")
    ax.set_xlabel("t")
    ax.set_ylabel("RMS")
    ax.legend()
    ax.set_title(f"{tag}: linearized next-state residual impact (η=0 DDIM)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_next_state_impact.png"), dpi=140)
    plt.close(fig)
    return b_rms


def write_report(
    path: str,
    *,
    tag: str,
    args,
    overall: dict,
    mom_r: dict,
    mom_d: dict,
    hetero: dict,
    temp: List[dict],
    b_rms: List[dict],
    per_t: List[dict],
):
    # summarize temporal
    if temp:
        corrs = [x["corr"] for x in temp]
        temp_s = f"mean={np.mean(corrs):.3f}, median={np.median(corrs):.3f}, min={np.min(corrs):.3f}, max={np.max(corrs):.3f}"
    else:
        temp_s = "n/a"
    # peak next-state impact
    if b_rms:
        peak = max(b_rms, key=lambda x: x["rms_dx"])
        peak_s = f"t={peak['t']}, |b|·rms(r)={peak['rms_dx']:.4g} (rms(r)={peak['rms_r']:.4g}, |b|={abs(peak['b']):.4g})"
    else:
        peak_s = "n/a"

    # mid/low/high t slices
    def slice_mean(key, lo, hi):
        vals = [row[key] for row in per_t if lo <= row["t"] <= hi]
        return float(np.mean(vals)) if vals else float("nan")

    lines = [
        f"# Residual after NN correction ({tag})",
        "",
        "## Setup",
        f"- data: `{args.data}`",
        f"- nn_ckpt: `{args.ckpt}`",
        f"- split: traj_id % {args.val_mod} == 0 → val",
        f"- primary residual: r = Δ* − Δ̂,  Δ* = ε_fp−ε_q,  Δ̂ = net(x,ε_q,t)  (full network, no blend)",
        f"- device: {args.device}",
        "",
        "## Overall (val)",
        "",
        "| field | value |",
        "|---|---:|",
        f"| E‖Δ*‖² | {overall['energy_delta']:.6g} |",
        f"| E‖r‖² | {overall['energy_r']:.6g} |",
        f"| R²(Δ̂ vs 0) | {overall['r2']:.4f} |",
        f"| ρ(r, ε_q) | {overall['rho_r_eq']:.4f} |",
        f"| ρ(r, Δ̂) | {overall['rho_r_dhat']:.4f} |",
        f"| ρ(r, Δ*) | {overall['rho_r_dstar']:.4f} |",
        f"| ρ(Δ*, ε_q) | {overall['rho_d_eq']:.4f} |",
        f"| mean(r) | {mom_r['mean']:.4g} |",
        f"| std(r) | {mom_r['std']:.4g} |",
        f"| excess kurtosis(r) | {mom_r['exkurt']:.3g} |",
        f"| QQ-r vs Normal | {mom_r['qq_r']:.4f} |",
        f"| bin-var CV (r\\|ε_q) | {hetero['bin_var_cv']:.3g} |",
        f"| bin E[r\\|ε_q] R² | {hetero['bin_mean_r2']:.3g} |",
        "",
        "### vs raw Δ* moments",
        "",
        "| | Δ* | r |",
        "|---|---:|---:|",
        f"| mean | {mom_d['mean']:.4g} | {mom_r['mean']:.4g} |",
        f"| std | {mom_d['std']:.4g} | {mom_r['std']:.4g} |",
        f"| exkurt | {mom_d['exkurt']:.3g} | {mom_r['exkurt']:.3g} |",
        f"| QQ r | {mom_d['qq_r']:.4f} | {mom_r['qq_r']:.4f} |",
        "",
        "## Temporal structure",
        f"- adjacent-step corr(r_t, r_{{t-}}) (traj-paired): {temp_s}",
        "",
        "## Next-state impact (η=0 DDIM linearization δx ≈ b(t)·r)",
        f"- peak: {peak_s}",
        f"- mean |b|·rms(r) over t: {float(np.mean([x['rms_dx'] for x in b_rms])):.4g}" if b_rms else "",
        "",
        "## Per-band summary",
        "",
        "| band | mean Var(r) | mean R²(Δ̂) | mean ρ(r,Δ̂) |",
        "|---|---:|---:|---:|",
        f"| t∈[0,100] | {slice_mean('var_r',0,100):.4g} | {slice_mean('r2_vs0',0,100):.3f} | {slice_mean('rho_r_dhat',0,100):.3f} |",
        f"| t∈[100,400] | {slice_mean('var_r',100,400):.4g} | {slice_mean('r2_vs0',100,400):.3f} | {slice_mean('rho_r_dhat',100,400):.3f} |",
        f"| t∈[400,1000] | {slice_mean('var_r',400,1000):.4g} | {slice_mean('r2_vs0',400,1000):.3f} | {slice_mean('rho_r_dhat',400,1000):.3f} |",
        "",
        "## Takeaway checklist",
        "",
        "1. **Zero-mean innovation?** |mean(r)| ≪ std(r).",
        "2. **Orthogonal to prediction?** |ρ(r, Δ̂)| small ⇒ network absorbed conditional mean.",
        "3. **Independent of ε_q?** |ρ(r, ε_q)| small ⇒ D2-style affine-on-ε_q has little left to do.",
        "4. **Gaussian-like?** QQ-r high + mild exkurt ⇒ residual SDE white-noise model more plausible than for raw Δ*.",
        "5. **Temporally colored?** high adjacent corr ⇒ model r as OU / AR(1), not iid each step.",
        "6. **State impact** peaks where |b(t)|·rms(r) is large (usually mid/low noise).",
        "",
    ]
    if overall.get("energy_r_inf") is not None:
        lines += [
            "## Inference residual (with blend/clip)",
            f"- E‖r_inf‖² = {overall['energy_r_inf']:.6g}",
            f"- ratio E‖r_inf‖² / E‖Δ*‖² = {overall['energy_r_inf']/max(overall['energy_delta'],EPS):.4f}",
            "",
        ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tag", default="residual")
    p.add_argument("--val_mod", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_samples", type=int, default=0, help="0 = all val")
    p.add_argument("--device", default="cpu")
    p.add_argument("--with_infer", action="store_true", help="also eval blend/clip corrector residual")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"loading traj {args.data}", flush=True)
    raw = torch.load(args.data, map_location="cpu", mmap=True)
    print(f"loading ckpt {args.ckpt}", flush=True)
    corrector = load_learned_corrector(args.ckpt, device)
    net = corrector.net
    net.eval()

    print("collecting residuals ...", flush=True)
    data = collect_residuals(
        raw,
        net,
        device,
        val_mod=args.val_mod,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        corrector=corrector if args.with_infer else None,
    )

    r, d_star, d_hat, eq = data["r"], data["delta_star"], data["delta_hat"], data["eq"]
    overall = {
        "energy_delta": float(d_star.pow(2).mean()),
        "energy_r": float(r.pow(2).mean()),
        "r2": energy_r2(d_hat, d_star),
        "rho_r_eq": pearson_rho(r, eq),
        "rho_r_dhat": pearson_rho(r, d_hat),
        "rho_r_dstar": pearson_rho(r, d_star),
        "rho_d_eq": pearson_rho(d_star, eq),
        "energy_r_inf": float(data["r_inf"].pow(2).mean()) if "r_inf" in data else None,
    }
    mom_r = moments(r)
    mom_d = moments(d_star)
    # fix mse in moments to full
    mom_r["mse"] = overall["energy_r"]
    mom_d["mse"] = overall["energy_delta"]
    hetero = bin_hetero(r, eq)
    print("per-t stats ...", flush=True)
    per_t = per_t_stats(data["t"], r, d_star, d_hat)
    rho_eq_t = per_t_rho_eq(data["t"], r, eq)
    for row in per_t:
        row["rho_r_eq"] = rho_eq_t.get(row["t"], float("nan"))
    print("temporal corr ...", flush=True)
    temp = temporal_corr_r(data["t"], data["traj_id"], r)
    print("plots ...", flush=True)
    b_rms = plot_suite(data, per_t, temp, args.out_dir, args.tag)

    payload = {
        "overall": overall,
        "moments_r": mom_r,
        "moments_delta": mom_d,
        "hetero_r_vs_eq": hetero,
        "per_t": per_t,
        "temporal_corr": temp,
        "next_state_impact": b_rms,
        "meta": {
            "data": os.path.abspath(args.data),
            "ckpt": os.path.abspath(args.ckpt),
            "definition": "r = (eps_fp - eps_q) - net(x, eps_q, t)",
            "corrector_meta": vars(corrector.meta),
        },
    }
    with open(os.path.join(args.out_dir, f"{args.tag}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    write_report(
        os.path.join(args.out_dir, f"{args.tag}_REPORT.md"),
        tag=args.tag,
        args=args,
        overall=overall,
        mom_r=mom_r,
        mom_d=mom_d,
        hetero=hetero,
        temp=temp,
        b_rms=b_rms,
        per_t=per_t,
    )
    print(json.dumps(overall, indent=2))
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
