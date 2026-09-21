#!/usr/bin/env python
"""Estimate causal bias/AR innovation statistics from same-state Q/FP pairs."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import torch


def _channel_moments(x: torch.Tensor):
    dims = (0,) + tuple(range(2, x.ndim))
    mean = x.float().mean(dims)
    var = (x.float() - mean.view(1, -1, 1, 1)).square().mean(dims)
    return mean, var.clamp_min(0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = torch.load(args.data, map_location="cpu")
    error = data["eq"].float() - data["ef"].float()
    timesteps = data["t"].long()
    trajectory = data["traj_id"].long()
    unique_t = sorted((int(v) for v in timesteps.unique()), reverse=True)

    mean_by_t, std_by_t, residual_by_t = {}, {}, {}
    diagnostics = {}
    for t in unique_t:
        values = error[timesteps == t]
        mean, var = _channel_moments(values)
        residual = values - mean.view(1, -1, 1, 1)
        mean_by_t[t], std_by_t[t] = mean, var.sqrt()
        residual_by_t[t] = residual
        total_energy = values.square().mean().item()
        diagnostics[str(t)] = {
            "error_rms": total_energy ** 0.5,
            "bias_energy_fraction": float(mean.square().mean() / (total_energy + 1e-12)),
            "residual_rms": float(residual.square().mean().sqrt()),
        }

    # collect tensors in trajectory order so correlations remain paired
    index = defaultdict(dict)
    for row, (traj, t) in enumerate(zip(trajectory.tolist(), timesteps.tolist())):
        index[int(t)][int(traj)] = row

    rho_by_transition = {}
    innovation_fraction = {}
    for previous_t, t in zip(unique_t[:-1], unique_t[1:]):
        common = sorted(set(index[previous_t]) & set(index[t]))
        prev = torch.stack([
            error[index[previous_t][k]] - mean_by_t[previous_t].view(-1, 1, 1)
            for k in common
        ])
        cur = torch.stack([
            error[index[t][k]] - mean_by_t[t].view(-1, 1, 1)
            for k in common
        ])
        dims = (0, 2, 3)
        cov = (prev * cur).mean(dims)
        rho = cov / (prev.square().mean(dims) * cur.square().mean(dims) + 1e-20).sqrt()
        rho = rho.nan_to_num().clamp(-0.99, 0.99)
        key = f"{previous_t}->{t}"
        rho_by_transition[key] = rho
        pred = rho.view(1, -1, 1, 1) * (
            std_by_t[t] / (std_by_t[previous_t] + 1e-12)
        ).view(1, -1, 1, 1) * prev
        innovation = cur - pred
        innovation_fraction[key] = float(
            innovation.square().mean() / (cur.square().mean() + 1e-12)
        )

    payload = {
        "meta": {**data.get("meta", {}), "source": os.path.abspath(args.data)},
        "mean_by_t_channel": {str(k): v.tolist() for k, v in mean_by_t.items()},
        "std_by_t_channel": {str(k): v.tolist() for k, v in std_by_t.items()},
        "rho_by_transition_channel": {k: v.tolist() for k, v in rho_by_transition.items()},
        "innovation_energy_fraction_by_transition": innovation_fraction,
        "diagnostics_by_t": diagnostics,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"saved {len(unique_t)} timestep statistics to {args.output}")


if __name__ == "__main__":
    main()
