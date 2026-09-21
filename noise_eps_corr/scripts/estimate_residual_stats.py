"""Estimate structured post-correction residual statistics.

The residual is ``eps_fp_same_input - eps_corrected``.  In addition to the
legacy scalar statistics, this script records per-channel central variance and
the per-channel correlation between adjacent DDIM grid steps on the same
trajectory.  These are the calibration inputs for experiment D.
"""

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import torch

from noise_eps_corr.learned_noise_corrector import load_corrector_from_ckpt


def _channel_moments(r: torch.Tensor):
    """Return channel sums, squared sums and elements per channel."""
    return (
        r.sum(dim=(0, 2, 3), dtype=torch.float64),
        r.square().sum(dim=(0, 2, 3), dtype=torch.float64),
        int(r.shape[0] * r.shape[2] * r.shape[3]),
    )


@torch.no_grad()
def _residual(raw, indices, corrector, t_idx, device):
    x = raw["x"][indices].float().to(device)
    eq = raw["eq"][indices].float().to(device)
    ef = raw["ef"][indices].float().to(device)
    ec = corrector.correct(eq, float(t_idx), torch.ones(1, device=device), xt=x)
    return (ef - ec).float()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="trajectory .pt containing x/eq/ef/t")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--t_cut", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--rmin", type=float, default=None)
    p.add_argument("--rmax", type=float, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(args.data, map_location="cpu", mmap=True)
    corrector = load_corrector_from_ckpt(args.ckpt, device)
    for name, value in (("t_cut", args.t_cut), ("alpha", args.alpha),
                        ("r_min", args.rmin), ("r_max", args.rmax)):
        if value is not None:
            setattr(corrector.meta, name, value)

    unique_t = sorted({int(v) for v in raw["t"].tolist()})
    required = {"x", "eq", "ef", "t", "traj_id"}
    missing = sorted(required.difference(raw.keys()))
    if missing:
        raise KeyError(f"trajectory data is missing fields required for D statistics: {missing}")

    variance_by_t, mean_by_t, mse_by_t, count_by_t = {}, {}, {}, {}
    channel_mean_by_t, channel_variance_by_t = {}, {}
    indices_by_t = {}
    for t_idx in unique_t:
        indices = torch.nonzero(raw["t"].long() == t_idx, as_tuple=False).squeeze(1)
        indices_by_t[t_idx] = indices
        sum_r = 0.0
        sum_r2 = 0.0
        count = 0
        sum_c = None
        sum2_c = None
        count_c = 0
        for start in range(0, indices.numel(), args.batch_size):
            idx = indices[start:start + args.batch_size]
            r = _residual(raw, idx, corrector, t_idx, device)
            sum_r += r.sum().item()
            sum_r2 += r.square().sum().item()
            count += r.numel()
            s, s2, nc = _channel_moments(r)
            sum_c = s if sum_c is None else sum_c + s
            sum2_c = s2 if sum2_c is None else sum2_c + s2
            count_c += nc
        mean = sum_r / max(count, 1)
        second = sum_r2 / max(count, 1)
        mean_c = sum_c / max(count_c, 1)
        second_c = sum2_c / max(count_c, 1)
        var_c = (second_c - mean_c.square()).clamp(min=0.0)
        variance_by_t[str(t_idx)] = max(second - mean * mean, 0.0)
        mean_by_t[str(t_idx)] = mean
        mse_by_t[str(t_idx)] = second
        count_by_t[str(t_idx)] = count
        channel_mean_by_t[str(t_idx)] = mean_c.cpu().tolist()
        channel_variance_by_t[str(t_idx)] = var_c.cpu().tolist()

    # Sampling runs from high t to low t.  Pair residuals by trajectory id so
    # collection batch/order does not affect the temporal correlation.
    descending_t = sorted(unique_t, reverse=True)
    temporal_corr = {}
    temporal_count = {}
    for t_hi, t_lo in zip(descending_t[:-1], descending_t[1:]):
        idx_hi_all = indices_by_t[t_hi]
        idx_lo_all = indices_by_t[t_lo]
        map_lo = {
            int(traj): int(idx)
            for traj, idx in zip(raw["traj_id"][idx_lo_all].tolist(), idx_lo_all.tolist())
        }
        paired_hi, paired_lo = [], []
        for traj, idx_hi in zip(raw["traj_id"][idx_hi_all].tolist(), idx_hi_all.tolist()):
            idx_lo = map_lo.get(int(traj))
            if idx_lo is not None:
                paired_hi.append(int(idx_hi))
                paired_lo.append(idx_lo)

        cross = None
        pair_elements = 0
        mean_hi = torch.tensor(channel_mean_by_t[str(t_hi)], device=device).view(1, -1, 1, 1)
        mean_lo = torch.tensor(channel_mean_by_t[str(t_lo)], device=device).view(1, -1, 1, 1)
        for start in range(0, len(paired_hi), args.batch_size):
            hi_idx = torch.tensor(paired_hi[start:start + args.batch_size], dtype=torch.long)
            lo_idx = torch.tensor(paired_lo[start:start + args.batch_size], dtype=torch.long)
            r_hi = _residual(raw, hi_idx, corrector, t_hi, device) - mean_hi
            r_lo = _residual(raw, lo_idx, corrector, t_lo, device) - mean_lo
            value = (r_hi * r_lo).sum(dim=(0, 2, 3), dtype=torch.float64)
            cross = value if cross is None else cross + value
            pair_elements += int(r_hi.shape[0] * r_hi.shape[2] * r_hi.shape[3])

        var_hi = torch.tensor(channel_variance_by_t[str(t_hi)], dtype=torch.float64)
        var_lo = torch.tensor(channel_variance_by_t[str(t_lo)], dtype=torch.float64)
        covariance = cross.cpu() / max(pair_elements, 1)
        corr = covariance / (var_hi * var_lo).clamp(min=1e-16).sqrt()
        key = f"{t_hi}->{t_lo}"
        temporal_corr[key] = corr.clamp(-0.999, 0.999).tolist()
        temporal_count[key] = pair_elements

    payload = {
        "variance_by_t": variance_by_t,
        "mean_by_t": mean_by_t,
        "mse_by_t": mse_by_t,
        "count_by_t": count_by_t,
        "channel_mean_by_t": channel_mean_by_t,
        "channel_variance_by_t": channel_variance_by_t,
        "temporal_corr_by_transition": temporal_corr,
        "temporal_count_by_transition": temporal_count,
        "meta": {
            "format_version": 2,
            "data": os.path.abspath(args.data),
            "ckpt": os.path.abspath(args.ckpt),
            "definition": "r = eps_fp_same_input - corrector(eps_q, x, t)",
            "variance": "global and per-channel elementwise central variance per timestep",
            "temporal_corr": "per-channel same-position correlation for adjacent sampled timesteps, paired by traj_id",
            "sampled_timesteps_descending": descending_t,
            "corrector_meta": vars(corrector.meta),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"saved {len(unique_t)} timestep statistics to {args.output}")


if __name__ == "__main__":
    main()
