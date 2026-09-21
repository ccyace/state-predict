"""
Scatter analysis and learnability probe for trajectory noise error (Experiment A).

Scatter: relate per-sample cos / strength_ratio to simple x^q_t statistics.
Probe: train a small CNN to predict delta_eps = eps_fp - eps_q from (x, t, eps_q).
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

EPS = 1e-8

SCATTER_FEATURES = (
    ("x_norm", r"$\|x^q_t\|$"),
    ("spatial_var", "spatial var (mean over C)"),
    ("x_std", r"std($x^q_t$)"),
    ("x_mean_abs", r"mean $|x^q_t|$"),
)


@torch.no_grad()
def compute_x_features(x: torch.Tensor) -> Dict[str, np.ndarray]:
    """Per-sample latent statistics from x [B,C,H,W]."""
    b = x.shape[0]
    flat = x.reshape(b, -1)
    spatial_var = x.var(dim=(2, 3), unbiased=False).mean(dim=1)
    return {
        "x_norm": flat.norm(dim=1).cpu().numpy(),
        "x_mean_abs": flat.abs().mean(dim=1).cpu().numpy(),
        "spatial_var": spatial_var.cpu().numpy(),
        "x_std": flat.std(dim=1).cpu().numpy(),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) < EPS or np.std(y) < EPS:
        return float("nan")
    r, _ = stats.pearsonr(x, y)
    return float(r)


class SmallDeltaEpsProbe(nn.Module):
    """Lightweight CNN: (x, eps_q, t) -> delta_eps."""

    def __init__(self, max_t: float = 1000.0):
        super().__init__()
        self.max_t = max_t
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        self.head = nn.Sequential(
            nn.Conv2d(32 + 64, 32, 1),
            nn.SiLU(),
            nn.Conv2d(32, 3, 1),
        )

    def forward(self, x: torch.Tensor, eq: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = self.encoder(torch.cat([x, eq], dim=1))
        t_norm = (t.float().view(b, 1) / self.max_t).clamp(0.0, 1.0)
        t_emb = self.time_mlp(t_norm).view(b, 64, 1, 1).expand(-1, -1, h.shape[2], h.shape[3])
        return self.head(torch.cat([h, t_emb], dim=1))


def _batch_cos(eq: torch.Tensor, ef: torch.Tensor) -> torch.Tensor:
    b = eq.shape[0]
    q = eq.reshape(b, -1)
    f = ef.reshape(b, -1)
    nq = q.norm(dim=1).clamp(min=EPS)
    nf = f.norm(dim=1).clamp(min=EPS)
    return (q * f).sum(dim=1) / (nq * nf)


def _batch_strength_ratio(eq: torch.Tensor, ef: torch.Tensor) -> torch.Tensor:
    b = eq.shape[0]
    nq = eq.reshape(b, -1).norm(dim=1).clamp(min=EPS)
    nf = ef.reshape(b, -1).norm(dim=1).clamp(min=EPS)
    return nq / nf


def _merge_probe_batches(batches: List[dict]) -> dict:
    out: dict = {"x": [], "eq": [], "ef": [], "traj_id": [], "cos": [], "strength_ratio": [], "rel_l2": []}
    feats = {k: [] for k, _ in SCATTER_FEATURES}
    for batch in batches:
        n = batch["cos"].shape[0]
        out["x"].append(batch["x"])
        out["eq"].append(batch["eq"])
        out["ef"].append(batch["ef"])
        out["traj_id"].append(batch["traj_id"])
        out["cos"].append(batch["cos"])
        out["strength_ratio"].append(batch["strength_ratio"])
        out["rel_l2"].append(batch["rel_l2"])
        for k in feats:
            feats[k].append(batch["x_features"][k])
    merged = {
        "x": torch.cat(out["x"], dim=0),
        "eq": torch.cat(out["eq"], dim=0),
        "ef": torch.cat(out["ef"], dim=0),
        "traj_id": np.concatenate(out["traj_id"]),
        "cos": np.concatenate(out["cos"]),
        "strength_ratio": np.concatenate(out["strength_ratio"]),
        "rel_l2": np.concatenate(out["rel_l2"]),
        "x_features": {k: np.concatenate(feats[k]) for k in feats},
    }
    return merged


def _train_val_split(traj_ids: np.ndarray, train_frac: float, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    uniq = np.unique(traj_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_train = max(1, int(len(uniq) * train_frac))
    train_set = set(int(i) for i in uniq[:n_train])
    train_mask = np.array([int(i) in train_set for i in traj_ids])
    val_mask = ~train_mask
    if not val_mask.any():
        val_mask = ~train_mask
        val_mask[0] = True
    return train_mask, val_mask


def _eval_correction(eq: torch.Tensor, ef: torch.Tensor, delta: torch.Tensor) -> dict:
    corr = eq + delta
    cos_b = _batch_cos(eq, ef).cpu().numpy()
    cos_a = _batch_cos(corr, ef).cpu().numpy()
    sr_b = _batch_strength_ratio(eq, ef).cpu().numpy()
    sr_a = _batch_strength_ratio(corr, ef).cpu().numpy()
    rel = lambda a, b: (
        (a.reshape(a.shape[0], -1) - b.reshape(b.shape[0], -1))
        .norm(dim=1)
        .div(b.reshape(b.shape[0], -1).norm(dim=1).clamp(min=EPS))
    )
    rel_b = rel(eq, ef).cpu().numpy()
    rel_a = rel(corr, ef).cpu().numpy()
    return {
        "cos_before": float(cos_b.mean()),
        "cos_after": float(cos_a.mean()),
        "delta_cos": float(cos_a.mean() - cos_b.mean()),
        "sr_err_before": float(np.abs(sr_b - 1.0).mean()),
        "sr_err_after": float(np.abs(sr_a - 1.0).mean()),
        "delta_sr_err": float(np.abs(sr_a - 1.0).mean() - np.abs(sr_b - 1.0).mean()),
        "rel_l2_before": float(rel_b.mean()),
        "rel_l2_after": float(rel_a.mean()),
        "delta_rel_l2": float(rel_a.mean() - rel_b.mean()),
    }


@torch.no_grad()
def _per_t_mean_delta(data: dict, train_mask: np.ndarray) -> torch.Tensor:
    delta = data["ef"] - data["eq"]
    mean = delta[train_mask].mean(dim=0, keepdim=True)
    return mean


def run_learnability_probe(
    probe_by_t: Dict[int, List[dict]],
    device: torch.device,
    *,
    train_frac: float = 0.8,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    late_t_max: int = 50,
    seed: int = 0,
) -> Tuple[List[dict], dict]:
    """Train probe per key t; return rows for CSV and aggregate summary."""
    torch.manual_seed(seed)
    rows: List[dict] = []

    for t_val in sorted(probe_by_t.keys()):
        data = _merge_probe_batches(probe_by_t[t_val])
        train_mask, val_mask = _train_val_split(data["traj_id"], train_frac, seed=seed)

        x = data["x"].float()
        eq = data["eq"].float()
        ef = data["ef"].float()
        t_tensor = torch.full((x.shape[0],), int(t_val), dtype=torch.float32)

        eq_v = eq[val_mask].to(device)
        ef_v = ef[val_mask].to(device)
        x_v = x[val_mask].to(device)
        t_v = t_tensor[val_mask].to(device)

        zero_delta = torch.zeros_like(eq_v)
        mean_delta = _per_t_mean_delta(data, train_mask).expand_as(eq_v).to(device)

        base_zero = _eval_correction(eq_v, ef_v, zero_delta)
        base_mean = _eval_correction(eq_v, ef_v, mean_delta)

        model = SmallDeltaEpsProbe().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        train_idx = np.where(train_mask)[0]

        model.train()
        for _ in range(epochs):
            perm = train_idx.copy()
            np.random.default_rng(seed).shuffle(perm)
            for start in range(0, len(perm), batch_size):
                idx = perm[start : start + batch_size]
                xb = x[idx].to(device)
                eqb = eq[idx].to(device)
                efb = ef[idx].to(device)
                tb = t_tensor[idx].to(device)
                pred = model(xb, eqb, tb)
                loss = nn.functional.mse_loss(pred, efb - eqb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            pred_v = model(x_v, eq_v, t_v)
        probe_metrics = _eval_correction(eq_v, ef_v, pred_v)

        row = {
            "t": int(t_val),
            "n_total": int(x.shape[0]),
            "n_val": int(val_mask.sum()),
            "region": "late" if t_val < late_t_max else ("mid" if t_val < 200 else "early"),
            **{f"zero_{k}": v for k, v in base_zero.items()},
            **{f"mean_{k}": v for k, v in base_mean.items()},
            **{f"probe_{k}": v for k, v in probe_metrics.items()},
        }
        rows.append(row)

    late_rows = [r for r in rows if r["region"] == "late"]
    summary = {
        "n_timesteps": len(rows),
        "late_mean_delta_cos_probe": float(np.mean([r["probe_delta_cos"] for r in late_rows])) if late_rows else float("nan"),
        "late_mean_delta_cos_mean": float(np.mean([r["mean_delta_cos"] for r in late_rows])) if late_rows else float("nan"),
        "late_mean_delta_cos_zero": float(np.mean([r["zero_delta_cos"] for r in late_rows])) if late_rows else float("nan"),
        "probe_beats_mean_on_late": bool(
            late_rows
            and np.mean([r["probe_delta_cos"] for r in late_rows]) > np.mean([r["mean_delta_cos"] for r in late_rows])
        ),
        "learnable_late": bool(
            late_rows and np.mean([r["probe_delta_cos"] for r in late_rows]) > 0.02
        ),
    }
    return rows, summary


def plot_learnability_summary(rows: List[dict], out_path: str, title_suffix: str = "") -> str:
    if not rows:
        return ""
    ts = [r["t"] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    ax.plot(ts, [r["zero_cos_before"] for r in rows], "o--", color="gray", ms=4, label="before (no corr)")
    ax.plot(ts, [r["mean_cos_after"] for r in rows], "s-", color="#2563eb", ms=4, label="after per-t mean Δε")
    ax.plot(ts, [r["probe_cos_after"] for r in rows], "^-", color="#dc2626", ms=4, label="after probe CNN")
    ax.set_ylabel("val mean cos(ε_corr, ε_fp)")
    ax.set_title(f"Learnability probe — direction{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar([t - 3 for t in ts], [r["zero_delta_cos"] for r in rows], width=6, color="gray", alpha=0.4, label="zero")
    ax.bar([t for t in ts], [r["mean_delta_cos"] for r in rows], width=6, color="#2563eb", alpha=0.7, label="per-t mean")
    ax.bar([t + 3 for t in ts], [r["probe_delta_cos"] for r in rows], width=6, color="#dc2626", alpha=0.7, label="probe")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axhline(0.02, color="#059669", ls="--", lw=0.8, label="Δcos=0.02 ref")
    ax.set_xlabel("diffusion t")
    ax.set_ylabel("Δcos (after − before)")
    ax.set_title("Validation improvement by method")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_learnability_csv(rows: List[dict], path: str) -> str:
    if not rows:
        return ""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def plot_scatter_analysis(
    probe_by_t: Dict[int, List[dict]],
    key_t_ordered: Sequence[Tuple[str, int]],
    out_dir: str,
    title_suffix: str = "",
) -> Tuple[List[str], List[dict]]:
    """Scatter cos and strength_ratio vs x features at key timesteps."""
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    corr_rows: List[dict] = []

    for metric_name, metric_label, y_ideal in (
        ("cos", "cos(ε_q, ε_fp)", 1.0),
        ("strength_ratio", "strength ratio", 1.0),
    ):
        ncols = len(key_t_ordered)
        nrows = len(SCATTER_FEATURES)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
        if ncols == 1:
            axes = axes.reshape(nrows, 1)
        if nrows == 1:
            axes = axes.reshape(1, ncols)

        for col, (label, t_val) in enumerate(key_t_ordered):
            batches = probe_by_t.get(int(t_val), [])
            if not batches:
                continue
            data = _merge_probe_batches(batches)
            y = data[metric_name]
            for row_i, (feat_key, feat_label) in enumerate(SCATTER_FEATURES):
                ax = axes[row_i, col]
                xfeat = data["x_features"][feat_key]
                ax.scatter(xfeat, y, s=6, alpha=0.35, c="#2563eb", edgecolors="none")
                if metric_name == "cos":
                    ax.axhline(y_ideal, color="gray", ls="--", lw=0.8)
                if metric_name == "strength_ratio":
                    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
                r = _pearson(xfeat, y)
                corr_rows.append({
                    "t": int(t_val),
                    "region_label": label,
                    "metric": metric_name,
                    "feature": feat_key,
                    "pearson_r": r,
                    "n": int(y.size),
                })
                ax.set_title(f"t={t_val} ({label})\nr={r:.3f}" if not np.isnan(r) else f"t={t_val} ({label})")
                if col == 0:
                    ax.set_ylabel(f"{metric_label}\nvs {feat_label}" if row_i == 0 else f"vs {feat_label}")
                if row_i == nrows - 1:
                    ax.set_xlabel(feat_label)

        fig.suptitle(
            f"Scatter: {metric_label} vs x^q_t stats (teacher on x^q_t){title_suffix}",
            fontsize=12,
        )
        plt.tight_layout()
        fname = "traj_scatter_cos_vs_x.png" if metric_name == "cos" else "traj_scatter_strength_vs_x.png"
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_path)

    # 1-cos vs rel_l2 joint scatter at late t only
    late_items = [(lab, tv) for lab, tv in key_t_ordered if tv < 50]
    if late_items:
        fig, axes = plt.subplots(1, len(late_items), figsize=(4.5 * len(late_items), 4))
        if len(late_items) == 1:
            axes = [axes]
        for ax, (label, t_val) in zip(axes, late_items):
            batches = probe_by_t.get(int(t_val), [])
            if not batches:
                continue
            data = _merge_probe_batches(batches)
            x = 1.0 - data["cos"]
            y = data["rel_l2"]
            ax.scatter(x, y, s=8, alpha=0.4, c="#7c3aed", edgecolors="none")
            r = _pearson(x, y)
            ax.set_xlabel("1 - cos (direction error)")
            ax.set_ylabel("rel L2")
            ax.set_title(f"t={t_val} ({label})\nr={r:.3f}")
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Direction vs magnitude error (late t){title_suffix}", fontsize=12)
        plt.tight_layout()
        p = os.path.join(out_dir, "traj_scatter_direction_vs_rel_l2_late.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    corr_csv = os.path.join(out_dir, "scatter_pearson_corr.csv")
    if corr_rows:
        with open(corr_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()))
            w.writeheader()
            w.writerows(corr_rows)
        paths.append(corr_csv)

    return paths, corr_rows
