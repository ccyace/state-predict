#!/usr/bin/env python
"""Train delta-t head with Time-Shift variance-matching labels.

Uses existing DeltaEpsDtNet (eps head + dt head). Default: freeze eps path,
train only dt head with

    L = SmoothL1(delta_t, dt_star)

where dt_star comes from soft variance matching (no UNet grid search).
Loss is masked to t > t_cutoff (paper disables shift below cutoff).

Example:
  python state_aware_temporal_joint/train_dt_timeshift.py \\
    --data output/noise_corr/train_data/traj_5k_fullstep_cl_cifar_w8a8.pt \\
    --init_eps_ckpt noise_eps_corr/w8a8_corr_ckpt_best.pt \\
    --output_dir state_aware_temporal_joint/runs/dt_timeshift_w8a8 \\
    --dt_only --window 40 --dt_max 20 --t_cutoff 300
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
    joint_correction_loss,
    load_eps_weights_into_joint,
    save_joint_corrector,
    unfreeze_all,
)
from state_aware_temporal_joint.timeshift_dt import (
    build_alphas_cumprod_from_config,
    timeshift_dt_star,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="output/noise_corr/train_data/traj_5k_fullstep_cl_cifar_w8a8.pt")
    p.add_argument("--config", default="configs/cifar10.yml")
    p.add_argument("--init_eps_ckpt", default="noise_eps_corr/w8a8_corr_ckpt_best.pt")
    p.add_argument("--output_dir", default="state_aware_temporal_joint/runs/dt_timeshift_w8a8")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--window", type=int, default=40, help="Time-Shift search window width")
    p.add_argument("--t_cutoff", type=int, default=300,
                   help="Time-Shift cutoff: dt_star=0 and no loss for t<=cutoff")
    p.add_argument("--soft_labels", action="store_true", default=True,
                   help="soft window expectation labels (default on)")
    p.add_argument("--hard_labels", action="store_true",
                   help="use hard argmin labels instead of soft")
    p.add_argument("--temperature", type=float, default=0.01,
                   help="softmax temperature for soft variance matching")
    p.add_argument("--signal_var", type=float, default=0.0,
                   help="if >0, match c*abar+(1-abar) instead of 1-abar")
    p.add_argument("--dt_max", type=float, default=20.0, help="tanh bound for delta_t head")
    p.add_argument("--lambda_dt", type=float, default=1.0)
    p.add_argument("--lambda_mse", type=float, default=1.0)
    p.add_argument("--lambda_cos", type=float, default=2.0)
    p.add_argument("--lambda_sr", type=float, default=0.3)
    p.add_argument("--t_cut", type=int, default=999)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--dt_only", action="store_true", default=True,
                   help="train only dt head (freeze encoder/eps); default on")
    p.add_argument("--joint", action="store_true",
                   help="also train eps loss (unfreezes all)")
    p.add_argument("--val_mod", type=int, default=5)
    p.add_argument("--mmap", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--val_batches", type=int, default=50, help="max validation batches")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def _label_kwargs(args):
    return dict(
        window=args.window,
        t_cutoff=args.t_cutoff,
        soft=not args.hard_labels,
        temperature=args.temperature,
        signal_var=args.signal_var,
    )


def _masked_dt_loss(delta_t, dt_star, t, t_cutoff: int):
    """Smooth-L1 on dt, averaged only over t > t_cutoff."""
    per = F.smooth_l1_loss(delta_t, dt_star, reduction="none")
    mask = (t.float() > float(t_cutoff)).float()
    denom = mask.sum().clamp_min(1.0)
    return (per * mask).sum() / denom, mask


@torch.no_grad()
def evaluate(net, loader, device, alphas, args):
    net.eval()
    mae_sum, loss_sum, n_mask = 0.0, 0.0, 0.0
    for bi, batch in enumerate(loader):
        if bi >= max(int(args.val_batches), 1):
            break
        x, eq, ef, t = [b.to(device) for b in batch[:4]]
        dt_star = timeshift_dt_star(x, t, alphas, **_label_kwargs(args))
        dt_star = dt_star.clamp(-args.dt_max, args.dt_max)
        delta_eps, delta_t = net(x, eq, t.float())
        if args.joint and not args.dt_only:
            loss, stats = joint_correction_loss(
                eq, ef, delta_eps, delta_t, dt_star, t,
                lambda_mse=args.lambda_mse,
                lambda_cos=args.lambda_cos,
                lambda_sr=args.lambda_sr,
                lambda_dt=args.lambda_dt,
                t_cut=args.t_cut,
                dt_only=False,
            )
            del stats
            mask = (t.float() > float(args.t_cutoff)).float()
        else:
            loss, mask = _masked_dt_loss(delta_t, dt_star, t, args.t_cutoff)
        m = float(mask.sum().item())
        if m <= 0:
            continue
        loss_sum += float(loss.item()) * m
        mae_sum += float(((delta_t - dt_star).abs() * mask).sum().item())
        n_mask += m
    return {
        "val_loss": loss_sum / max(n_mask, 1.0),
        "val_dt_mae": mae_sum / max(n_mask, 1.0),
    }


def main():
    args = parse_args()
    if args.joint:
        args.dt_only = False
    if args.hard_labels:
        args.soft_labels = False
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

    train_ds = TrajectoryLateDataset(args.data, "train", val_mod=args.val_mod, mmap=args.mmap)
    val_ds = TrajectoryLateDataset(args.data, "val", val_mod=args.val_mod, mmap=args.mmap)
    if args.smoke:
        train_loader = DataLoader(
            train_ds, batch_size=min(8, args.batch_size), shuffle=True,
            num_workers=0, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=min(8, args.batch_size), shuffle=False, num_workers=0,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        )

    net = DeltaEpsDtNet(max_t=1000.0, dt_max=args.dt_max).to(device)
    if args.init_eps_ckpt and os.path.isfile(args.init_eps_ckpt):
        load_eps_weights_into_joint(net, args.init_eps_ckpt)
        print(f"Warm-started eps path from {args.init_eps_ckpt}", flush=True)
    if args.dt_only:
        freeze_eps_head(net)
        trainable = list(net.dt_head.parameters())
        print("Mode: dt_only (encoder/eps frozen)", flush=True)
    else:
        unfreeze_all(net)
        trainable = list(net.parameters())
        print("Mode: joint eps+dt", flush=True)

    n_train = sum(p.numel() for p in trainable if p.requires_grad)
    print(
        f"train={len(train_ds)} val={len(val_ds)} trainable_params={n_train:,} "
        f"window={args.window} t_cutoff={args.t_cutoff} dt_max={args.dt_max} "
        f"soft={not args.hard_labels} temp={args.temperature} signal_var={args.signal_var}",
        flush=True,
    )

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05,
    )
    meta = JointCorrectorMeta(
        t_cut=args.t_cut,
        alpha=args.alpha,
        max_t=1000.0,
        dt_max=args.dt_max,
        joint_eps_dt=True,
    )

    history = []
    best = float("inf")
    started = time.time()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        net.train()
        if args.dt_only:
            net.encoder.eval()
            net.time_mlp.eval()
            net.eps_head.eval()
        running = 0.0
        running_mae = 0.0
        seen = 0.0
        for batch in train_loader:
            x, eq, ef, t = [b.to(device, non_blocking=True) for b in batch[:4]]
            with torch.no_grad():
                dt_star = timeshift_dt_star(x, t, alphas, **_label_kwargs(args))
                dt_star = dt_star.clamp(-args.dt_max, args.dt_max)
            opt.zero_grad(set_to_none=True)
            delta_eps, delta_t = net(x, eq, t.float())
            if args.dt_only:
                loss, mask = _masked_dt_loss(delta_t, dt_star, t, args.t_cutoff)
            else:
                loss, _ = joint_correction_loss(
                    eq, ef, delta_eps, delta_t, dt_star, t,
                    lambda_mse=args.lambda_mse,
                    lambda_cos=args.lambda_cos,
                    lambda_sr=args.lambda_sr,
                    lambda_dt=args.lambda_dt,
                    t_cut=args.t_cut,
                    dt_only=False,
                )
                mask = (t.float() > float(args.t_cutoff)).float()
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
                    f"loss={running / max(seen, 1):.6f} dt_mae={running_mae / max(seen, 1):.4f} "
                    f"lr={sched.get_last_lr()[0]:.2e} time={time.time() - started:.1f}s",
                    flush=True,
                )
                running = 0.0
                running_mae = 0.0
                seen = 0.0
            if args.smoke and global_step >= 2:
                break

        sched.step()
        metrics = evaluate(net, val_loader, device, alphas, args)
        record = {"epoch": epoch, **metrics, "lr": sched.get_last_lr()[0]}
        history.append(record)
        print(
            f"VAL epoch={epoch} loss={metrics['val_loss']:.6f} dt_mae={metrics['val_dt_mae']:.4f}",
            flush=True,
        )
        if metrics["val_dt_mae"] < best:
            best = metrics["val_dt_mae"]
            save_joint_corrector(os.path.join(args.output_dir, "ckpt_best.pt"), net, meta)
            with open(os.path.join(args.output_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
        save_joint_corrector(os.path.join(args.output_dir, "ckpt_last.pt"), net, meta)
        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "history": history, "best_dt_mae": best}, f, indent=2)
        if args.smoke:
            break

    print(f"Done. best val dt_mae={best:.4f}; output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
