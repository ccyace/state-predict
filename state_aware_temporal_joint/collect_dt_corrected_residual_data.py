#!/usr/bin/env python
"""Collect closed-loop data after frozen sparse dt refresh correction."""
from __future__ import annotations

import argparse
import gc
import os
import sys

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from calibrate_trajectory_qhat import dict2namespace
from qdiff.ddim_helpers import alpha_bar_at, ddim_update
from qdiff.joint_eps_dt_corrector import load_joint_corrector
from qdiff.trajectory_error import build_ddim_seq
from sample_diffusion_ddim import get_beta_schedule
from state_aware_temporal_joint.sample_50k import _refresh_step_indices
from state_aware_temporal_joint.time_corrected_residual import load_residual


def compute_alpha(beta, t):
    """Match ddim.functions.denoising.compute_alpha API: beta[T], t[B] long -> [B,1,1,1]."""
    return alpha_bar_at(beta, t, beta.device if torch.is_tensor(beta) else t.device)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--dt_ckpt", default="", help="empty = no sparse dt refresh")
    p.add_argument("--num_trajectories", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--skip_type", default="quad")
    p.add_argument("--dt_eta", type=float, default=0.5)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--time_residual_ckpt", default="")
    p.add_argument("--time_residual_strength", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dt_max", type=float, default=20.0)
    p.add_argument("--t_cutoff", type=int, default=300)
    p.add_argument("--n_refresh", type=int, default=8)
    p.add_argument("--config", default="configs/cifar10.yml")
    p.add_argument("--cali_ckpt", default="cifar_w8a8_ckpt.pth")
    p.add_argument("--cali_data_path", default="cifar_sd1236_sample2048_allst.pt")
    p.add_argument("--weight_bit", type=int, default=8)
    p.add_argument("--act_bit", type=int, default=8)
    p.add_argument("--sm_abit", type=int, default=8)
    p.add_argument("--cali_st", type=int, default=10)
    p.add_argument("--cali_n", type=int, default=256)
    p.add_argument("--split", action="store_true", default=True)
    p.add_argument("--quant_act", action="store_true", default=True)
    p.add_argument("--a_sym", action="store_true", default=True)
    p.add_argument("--backbone", choices=("unet", "uvit"), default="unet")
    p.add_argument("--fp_ckpt", default="")
    return p.parse_args()


@torch.no_grad()
def main():
    a = parse_args()
    # placeholders expected by load helpers
    a.cond = False
    a.joint_sa_resume = False
    a.brecq_ckpt = ""
    a.ckpt = ""
    a.ode_scale_json = ""
    a.ode_absorb_mode = ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict2namespace(yaml.safe_load(open(a.config)))
    cfg.split_shortcut = bool(getattr(a, "split", False))

    if a.backbone == "uvit":
        sys.path.insert(0, os.path.join(ROOT, "uvit_experiments"))
        from pipeline_loaders import load_quant_from_args

        q = load_quant_from_args(a, device)
    else:
        from calibrate_trajectory_qhat import load_float_model, load_quant_model

        fp = load_float_model(cfg, device, a)
        q = load_quant_model(cfg, device, a, fp)
        del fp
        gc.collect()
        torch.cuda.empty_cache()

    dt = None
    if a.dt_ckpt:
        dt = load_joint_corrector(a.dt_ckpt, device)
        dt.train_mode_off()
    mean_corr = None
    if a.time_residual_ckpt:
        mean_corr = load_residual(a.time_residual_ckpt, device)

    bn = torch.tensor(
        get_beta_schedule(
            beta_schedule=cfg.diffusion.beta_schedule,
            beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end,
            num_diffusion_timesteps=cfg.diffusion.num_diffusion_timesteps,
        ),
        dtype=torch.float32,
        device=device,
    )
    seq = build_ddim_seq(len(bn), a.timesteps, a.skip_type)
    nxt = [-1] + list(seq[:-1])
    steps = list(zip(reversed(seq), reversed(nxt)))
    refresh = _refresh_step_indices(seq, a.t_cutoff, a.n_refresh) if dt is not None else set()
    t_cap = float(bn.numel() - 1)

    xs, eqs, efs, mses, t_noms, t_corrs, rfs, tids = [], [], [], [], [], [], [], []
    gen = torch.Generator(device=device)
    gen.manual_seed(int(a.seed))
    done = 0
    while done < a.num_trajectories:
        bs = min(a.batch_size, a.num_trajectories - done)
        x = torch.randn(bs, 3, 32, 32, device=device, generator=gen)
        ids = torch.arange(done, done + bs, device="cpu")
        for sk, (i, j) in enumerate(steps):
            tn = torch.full((bs,), float(i), device=device)
            nt = torch.full((bs,), float(j), device=device)
            at = compute_alpha(bn, tn.long())
            an = compute_alpha(bn, nt.long())
            q.set_quant_state(True, True)
            en = q(x, tn)
            do_refresh = dt is not None and sk in refresh and float(i) > float(a.t_cutoff)
            if do_refresh:
                dlt = dt.predict_dt_only(en, float(i), xt=x)
                dlt = (a.dt_eta * dlt).clamp(-a.dt_max, a.dt_max)
                t_eff = (tn + dlt).clamp(0.0, t_cap)
                q.set_quant_state(True, True)
                en = q(x, t_eff)
                used = t_eff
                rf = True
            else:
                used = tn
                rf = False
            corrected = en
            if mean_corr is not None:
                rf_tensor = torch.full((bs,), float(rf), device=device)
                corrected = mean_corr.correct(
                    en, x, tn, used, rf_tensor, strength=a.time_residual_strength,
                )
            q.set_quant_state(False, False)
            ef = q(x, used)
            q.set_quant_state(True, True)

            xs.append(x.detach().cpu().half())
            eqs.append(en.detach().cpu().half())
            efs.append(ef.detach().cpu().half())
            mses.append((ef - en).square().mean(dim=(1, 2, 3)).detach().cpu())
            t_noms.append(tn.detach().cpu().float())
            t_corrs.append(used.detach().cpu().float())
            rfs.append(torch.full((bs,), rf, dtype=torch.bool))
            tids.append(ids.clone())

            x = ddim_update(x, corrected, at, an, eta=float(a.eta), generator=gen)

        done += bs
        print(f"trajectories {done}/{a.num_trajectories}", flush=True)

    out = {
        "x": torch.cat(xs, 0),
        "eq": torch.cat(eqs, 0),
        "ef": torch.cat(efs, 0),
        "residual_mse": torch.cat(mses, 0),
        "t_nom": torch.cat(t_noms, 0),
        "t_corr": torch.cat(t_corrs, 0),
        "is_refresh": torch.cat(rfs, 0),
        "traj_id": torch.cat(tids, 0).long(),
        "meta": {**vars(a)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.output)) or ".", exist_ok=True)
    torch.save(out, a.output)
    print(f"saved {a.output} {out['x'].shape[0]}", flush=True)


if __name__ == "__main__":
    main()
