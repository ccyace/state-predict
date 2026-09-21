#!/usr/bin/env python3
"""Apply learned corrector offline; export post-correction residual stats for VSC."""
from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from noise_eps_corr.learned_noise_corrector import load_learned_corrector


@torch.no_grad()
def build_post_corr_traj(data_path, corrector_ckpt, alpha, device):
    traj = torch.load(data_path, map_location="cpu")
    corr = load_learned_corrector(corrector_ckpt, device)
    corr.meta.alpha = float(alpha)

    x = traj["x"].float()
    eq = traj["eq"].float()
    ef = traj["ef"].float()
    t = traj["t"].float()
    n = x.shape[0]
    eq_corr_list = []
    residual_mse = []

    bs = 64
    for i in range(0, n, bs):
        sl = slice(i, i + bs)
        xb = x[sl].to(device)
        eqb = eq[sl].to(device)
        efb = ef[sl].to(device)
        tb = t[sl]
        out = []
        for j in range(xb.shape[0]):
            tj = tb[j]
            eqj = eqb[j : j + 1]
            eqc = corr.correct(eqj, float(tj.item()), None, xt=xb[j : j + 1])
            out.append(eqc)
        eqc = torch.cat(out, 0)
        eq_corr_list.append(eqc.cpu().half())
        residual_mse.append((eqc - efb).pow(2).mean(dim=(1, 2, 3)).cpu())

    eq_corr = torch.cat(eq_corr_list, 0)
    residual_mse = torch.cat(residual_mse, 0)
    out = dict(traj)
    out["eq"] = eq_corr
    out["residual_mse"] = residual_mse.float()
    out["t_nom"] = t.clone()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--corrector_ckpt", required=True)
    p.add_argument("--corrector_alpha", type=float, default=1.0)
    p.add_argument("--output_traj", required=True)
    p.add_argument("--output_vsc", required=True)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--skip_type", default="uniform")
    p.add_argument("--linear_start", type=float, default=0.0015)
    p.add_argument("--linear_end", type=float, default=0.0195)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    post = build_post_corr_traj(args.data, args.corrector_ckpt, args.corrector_alpha, device)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_traj)) or ".", exist_ok=True)
    torch.save(post, args.output_traj)
    print(f"saved post-corr traj -> {args.output_traj}", flush=True)

    import subprocess
    cmd = [
        sys.executable,
        os.path.join(ROOT, "PTQD", "estimate_time_variance.py"),
        "--data", args.output_traj,
        "--output_pt", args.output_vsc,
        "--output_json", os.path.splitext(args.output_vsc)[0] + ".json",
        "--eta", str(args.eta),
        "--timesteps", str(args.timesteps),
        "--skip_type", args.skip_type,
        "--linear_start", str(args.linear_start),
        "--linear_end", str(args.linear_end),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
