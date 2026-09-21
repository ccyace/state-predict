#!/usr/bin/env python
"""Collect closed-loop (x, eq, ef, t) under refresh-δt + ε-corrector policy.

On refresh nodes: predict δt at t_nom, re-query UNet at t_eff, save (eq, ef, t_eff),
then apply corrector at t_eff for the DDIM state update.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch
import yaml
from pytorch_lightning import seed_everything

from calibrate_trajectory_qhat import dict2namespace, load_float_model, load_quant_model
from ddim.functions.denoising import compute_alpha
from noise_eps_corr.learned_noise_corrector import load_learned_corrector
from qdiff.ddim_helpers import ddim_update
from qdiff.joint_eps_dt_corrector import load_joint_corrector
from qdiff.trajectory_error import build_ddim_seq
from sample_diffusion_ddim import get_beta_schedule
from state_aware_temporal_joint.sample_50k import _alpha_bar_batch, _refresh_step_indices


def _eps_pair(quant_model, x: torch.Tensor, t: torch.Tensor):
    """t: [B] float times used by UNet."""
    quant_model.set_quant_state(weight_quant=True, act_quant=True)
    eq = quant_model(x, t)
    quant_model.set_quant_state(weight_quant=False, act_quant=False)
    ef = quant_model(x, t)
    quant_model.set_quant_state(weight_quant=True, act_quant=True)
    return eq, ef


@torch.no_grad()
def collect(
    quant_model,
    joint_corrector,
    eps_corrector,
    betas,
    seq,
    device,
    *,
    num_trajectories: int,
    batch_size: int,
    t_max_save: int,
    dt_eta: float,
    dt_max: float,
    t_cutoff: int,
    n_refresh: int,
    sync_alpha: bool,
    seed: int,
    channels: int,
    image_size: int,
):
    quant_model.eval()
    joint_corrector.train_mode_off()
    if eps_corrector is not None:
        eps_corrector.train_mode_off()

    t_cap = float(betas.numel() - 1)
    refresh_at = _refresh_step_indices(seq, t_cutoff, n_refresh)
    seq_next = [-1] + list(seq[:-1])
    rev_steps = list(zip(reversed(seq), reversed(seq_next)))

    xs, eqs, efs, ts, traj_ids = [], [], [], [], []
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
    n_done = 0
    log_every = max(64, batch_size)

    print(
        f"  refresh-CL collect  traj={num_trajectories} refresh_n={n_refresh} "
        f"dt_eta={dt_eta} t_cutoff={t_cutoff} t_max_save={t_max_save}",
        flush=True,
    )

    while n_done < num_trajectories:
        cur_b = min(batch_size, num_trajectories - n_done)
        x = torch.randn(cur_b, channels, image_size, image_size, device=device, generator=rng)
        ids = torch.arange(n_done, n_done + cur_b, dtype=torch.int32)

        for step_k, (i, j) in enumerate(rev_steps):
            t_nom = float(i)
            t_nom_t = torch.full((cur_b,), t_nom, device=device)
            next_t = torch.full((cur_b,), float(j), device=device)
            at = compute_alpha(betas, t_nom_t.long())
            at_next = compute_alpha(betas, next_t.long())

            eps_probe = quant_model(x, t_nom_t)
            do_refresh = step_k in refresh_at and t_nom > float(t_cutoff)
            if do_refresh:
                _de, delta_t = joint_corrector.net(x, eps_probe, t_nom_t)
                del _de
                delta_t = (dt_eta * delta_t).clamp(-dt_max, dt_max)
                t_eff = (t_nom_t + delta_t).clamp(0.0, t_cap)
                if sync_alpha:
                    at = _alpha_bar_batch(betas, t_eff)
            else:
                t_eff = t_nom_t

            eq, ef = _eps_pair(quant_model, x, t_eff)
            if int(t_nom) < int(t_max_save):
                xs.append(x.detach().cpu().half())
                eqs.append(eq.detach().cpu().half())
                efs.append(ef.detach().cpu().half())
                ts.append(t_eff.detach().cpu().float())
                traj_ids.append(ids.clone())

            eps_step = eq
            if eps_corrector is not None:
                eps_step = eps_corrector.correct(eq, t_eff, at, xt=x)
            x = ddim_update(x, eps_step, at, at_next, eta=0.0)

        n_done += cur_b
        if n_done % log_every == 0 or n_done >= num_trajectories:
            print(f"  trajectories {n_done}/{num_trajectories}", flush=True)

    t_cat = torch.cat(ts, dim=0)
    return {
        "x": torch.cat(xs, dim=0),
        "eq": torch.cat(eqs, dim=0),
        "ef": torch.cat(efs, dim=0),
        "t": t_cat,
        "traj_id": torch.cat(traj_ids, dim=0).long(),
        "meta": {
            "num_trajectories": num_trajectories,
            "t_max": int(t_max_save),
            "n_samples": int(t_cat.shape[0]),
            "closed_loop": eps_corrector is not None,
            "policy": "refresh_dt_plus_eps_corrector",
            "dt_eta": float(dt_eta),
            "t_cutoff": int(t_cutoff),
            "n_refresh": int(n_refresh),
            "sync_alpha": bool(sync_alpha),
            "channels": int(channels),
            "image_size": int(image_size),
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/cifar10.yml")
    p.add_argument("--cali_ckpt", default="cifar_w8a8_ckpt.pth")
    p.add_argument("--cali_data_path", default="cifar_sd1236_sample2048_allst.pt")
    p.add_argument("--joint_dt_ckpt", required=True)
    p.add_argument("--corrector_ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num_trajectories", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--t_max", type=int, default=999)
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--skip_type", default="quad")
    p.add_argument("--dt_eta", type=float, default=0.5)
    p.add_argument("--dt_carry_max", type=float, default=20.0)
    p.add_argument("--t_cutoff", type=int, default=300)
    p.add_argument("--dt_refresh_n", type=int, default=8)
    p.add_argument("--sync_alpha", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--split", action="store_true", default=True)
    p.add_argument("--quant_act", action="store_true", default=True)
    p.add_argument("--a_sym", action="store_true", default=True)
    p.add_argument("--weight_bit", type=int, default=8)
    p.add_argument("--act_bit", type=int, default=8)
    p.add_argument("--sm_abit", type=int, default=8)
    p.add_argument("--cali_st", type=int, default=10)
    p.add_argument("--cali_n", type=int, default=256)
    p.add_argument("--ckpt", default="")
    p.add_argument("--ode_scale_json", default="")
    p.add_argument("--ode_absorb_mode", default="")
    args = p.parse_args()
    args.cond = False
    args.joint_sa_resume = False
    args.brecq_ckpt = ""

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, "r", encoding="utf-8") as f:
        config = dict2namespace(yaml.safe_load(f))
    config.split_shortcut = args.split

    betas_np = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.as_tensor(betas_np, dtype=torch.float32, device=device)
    seq = build_ddim_seq(len(betas_np), args.timesteps, args.skip_type)

    print(f"Loading models on {device} ...", flush=True)
    fp = load_float_model(config, device, args)
    qnn = load_quant_model(config, device, args, fp)
    del fp
    gc.collect()
    qnn.set_quant_state(True, True)
    qnn.eval()

    joint = load_joint_corrector(args.joint_dt_ckpt, device)
    corr = load_learned_corrector(args.corrector_ckpt, device)
    # Apply corrector on the full grid during collection (match fullstep CL practice).
    if int(corr.meta.t_cut) < 999:
        print(f"  overriding corrector t_cut {corr.meta.t_cut} -> 999 for CL collect", flush=True)
        corr.meta.t_cut = 999

    data = collect(
        qnn, joint, corr, betas, seq, device,
        num_trajectories=args.num_trajectories,
        batch_size=args.batch_size,
        t_max_save=args.t_max,
        dt_eta=args.dt_eta,
        dt_max=args.dt_carry_max,
        t_cutoff=args.t_cutoff,
        n_refresh=args.dt_refresh_n,
        sync_alpha=bool(args.sync_alpha),
        seed=args.seed,
        channels=int(config.data.channels),
        image_size=int(config.data.image_size),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    tmp = os.path.abspath(args.output) + ".tmp"
    if os.path.isfile(tmp):
        os.remove(tmp)
    torch.save(data, tmp)
    os.replace(tmp, os.path.abspath(args.output))
    m = data["meta"]
    print(f"Saved {m['n_samples']} samples -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
