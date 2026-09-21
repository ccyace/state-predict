#!/usr/bin/env python
"""Train δt head for paper-style Time-Shift (destination-time shift).

Labels (online, no UNet):
  δt* = t_s - t_next,  t_s = soft variance match around nominal t_next

Inference (see sample_50k --dt_mode paper):
  ε = Q(x, t);  t_s = t_next + δt̂;  DDIM with ᾱ(t), ᾱ(t_s)

Example:
  python state_aware_temporal_joint/train_dt_paper_ts.py --mmap --dt_only \\
    --output_dir state_aware_temporal_joint/runs/dt_paper_ts_w8a8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

from calibrate_trajectory_qhat import dict2namespace
from noise_eps_corr.learned_noise_corrector import TrajectoryLateDataset
from qdiff.joint_eps_dt_corrector import (
    DeltaEpsDtNet,
    JointCorrectorMeta,
    freeze_eps_head,
    load_eps_weights_into_joint,
    save_joint_corrector,
)
from qdiff.trajectory_error import build_ddim_seq
from state_aware_temporal_joint.timeshift_dt import (
    build_alphas_cumprod_from_config,
    build_t_to_tnext_map,
    t_next_from_map,
    timeshift_dt_star_paper,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="output/noise_corr/train_data/traj_5k_fullstep_cl_cifar_w8a8.pt")
    p.add_argument("--config", default="configs/cifar10.yml")
    p.add_argument("--init_eps_ckpt", default="noise_eps_corr/w8a8_corr_ckpt_best.pt")
    p.add_argument("--output_dir", default="state_aware_temporal_joint/runs/dt_paper_ts_w8a8")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--window", type=int, default=40)
    p.add_argument("--t_cutoff", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--hard_labels", action="store_true")
    p.add_argument("--dt_max", type=float, default=20.0)
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--skip_type", choices=("uniform", "quad"), default="quad")
    p.add_argument("--dt_only", action="store_true", default=True)
    p.add_argument("--val_mod", type=int, default=5)
    p.add_argument("--mmap", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--val_batches", type=int, default=50)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def _labels(x, t, alphas, t2n, args):
    t_next = t_next_from_map(t, t2n)
    dt = timeshift_dt_star_paper(
        x, t, t_next, alphas,
        window=args.window,
        t_cutoff=args.t_cutoff,
        soft=not args.hard_labels,
        temperature=args.temperature,
    )
    return dt.clamp(-args.dt_max, args.dt_max)


def _masked_loss(delta_t, dt_star, t, t_cutoff: int):
    per = F.smooth_l1_loss(delta_t, dt_star, reduction="none")
    mask = (t.float() > float(t_cutoff)).float()
    denom = mask.sum().clamp_min(1.0)
    return (per * mask).sum() / denom, mask


@torch.no_grad()
def evaluate(net, loader, device, alphas, t2n, args):
    net.eval()
    mae_sum, loss_sum, n_mask = 0.0, 0.0, 0.0
    for bi, batch in enumerate(loader):
        if bi >= max(int(args.val_batches), 1):
            break
        x, eq, ef, t = [b.to(device) for b in batch[:4]]
        del ef
        dt_star = _labels(x, t, alphas, t2n, args)
        _de, delta_t = net(x, eq, t.float())
        del _de
        loss, mask = _masked_loss(delta_t, dt_star, t, args.t_cutoff)
        m = float(mask.sum().item())
        if m <= 0:
            continue
        loss_sum += float(loss.item()) * m
        mae_sum += float(((delta_t - dt_star).abs() * mask).sum().item())
        n_mask += m
    return {"val_loss": loss_sum / max(n_mask, 1.0), "val_dt_mae": mae_sum / max(n_mask, 1.0)}


def main():
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.log_every = 1
        args.val_batches = 2
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        config = dict2namespace(yaml.safe_load(f))
    alphas = build_alphas_cumprod_from_config(config).to(device)
    seq = build_ddim_seq(
        config.diffusion.num_diffusion_timesteps, args.timesteps, args.skip_type
    )
    t2n = build_t_to_tnext_map(seq)

    train_ds = TrajectoryLateDataset(args.data, "train", val_mod=args.val_mod, mmap=args.mmap)
    val_ds = TrajectoryLateDataset(args.data, "val", val_mod=args.val_mod, mmap=args.mmap)
    bs = min(8, args.batch_size) if args.smoke else args.batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    net = DeltaEpsDtNet(max_t=1000.0, dt_max=args.dt_max).to(device)
    if args.init_eps_ckpt and os.path.isfile(args.init_eps_ckpt):
        load_eps_weights_into_joint(net, args.init_eps_ckpt)
        print(f"Warm-started from {args.init_eps_ckpt}", flush=True)
    freeze_eps_head(net)
    trainable = list(net.dt_head.parameters())
    print(
        f"Mode: paper-TS dt_only  train={len(train_ds)} val={len(val_ds)} "
        f"params={sum(p.numel() for p in trainable):,} window={args.window} "
        f"t_cutoff={args.t_cutoff} soft={not args.hard_labels}",
        flush=True,
    )

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05,
    )
    meta = JointCorrectorMeta(
        t_cut=999, alpha=0.0, max_t=1000.0, dt_max=args.dt_max, joint_eps_dt=True,
    )
    # stash paper-mode hint for sampling
    meta_extra = {
        "dt_mode": "paper",
        "t_cutoff": args.t_cutoff,
        "window": args.window,
        "timesteps": args.timesteps,
        "skip_type": args.skip_type,
    }

    history, best, started, global_step = [], float("inf"), time.time(), 0
    for epoch in range(1, args.epochs + 1):
        net.train()
        net.encoder.eval()
        net.time_mlp.eval()
        net.eps_head.eval()
        running = running_mae = seen = 0.0
        for batch in train_loader:
            x, eq, ef, t = [b.to(device, non_blocking=True) for b in batch[:4]]
            del ef
            with torch.no_grad():
                dt_star = _labels(x, t, alphas, t2n, args)
            opt.zero_grad(set_to_none=True)
            _de, delta_t = net(x, eq, t.float())
            del _de
            loss, mask = _masked_loss(delta_t, dt_star, t, args.t_cutoff)
            if float(mask.sum().item()) <= 0:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            m = float(mask.sum().item())
            running += float(loss.detach()) * m
            running_mae += float(((delta_t.detach() - dt_star).abs() * mask).sum().item())
            seen += m
            global_step += 1
            if global_step % args.log_every == 0:
                print(
                    f"epoch {epoch} step {global_step} "
                    f"loss={running/max(seen,1):.6f} dt_mae={running_mae/max(seen,1):.4f} "
                    f"lr={sched.get_last_lr()[0]:.2e} time={time.time()-started:.1f}s",
                    flush=True,
                )
                running = running_mae = seen = 0.0
            if args.smoke and global_step >= 2:
                break
        sched.step()
        metrics = evaluate(net, val_loader, device, alphas, t2n, args)
        record = {"epoch": epoch, **metrics, "lr": sched.get_last_lr()[0]}
        history.append(record)
        print(
            f"VAL epoch={epoch} loss={metrics['val_loss']:.6f} dt_mae={metrics['val_dt_mae']:.4f}",
            flush=True,
        )
        if metrics["val_dt_mae"] < best:
            best = metrics["val_dt_mae"]
            path = os.path.join(args.output_dir, "ckpt_best.pt")
            save_joint_corrector(path, net, meta)
            payload = torch.load(path, map_location="cpu")
            payload["paper_ts"] = meta_extra
            torch.save(payload, path)
            with open(os.path.join(args.output_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
        path = os.path.join(args.output_dir, "ckpt_last.pt")
        save_joint_corrector(path, net, meta)
        payload = torch.load(path, map_location="cpu")
        payload["paper_ts"] = meta_extra
        torch.save(payload, path)
        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "history": history, "best_dt_mae": best, **meta_extra}, f, indent=2)
        if args.smoke:
            break
    print(f"Done. best val dt_mae={best:.4f}; output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
