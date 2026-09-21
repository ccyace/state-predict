#!/usr/bin/env python3
"""Train learned DeltaEpsNet on closed-loop LDM/CIFAR trajectory data."""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from noise_eps_corr.learned_noise_corrector import (
    DeltaEpsNet,
    TrajectoryLateDataset,
    correction_loss,
    save_learned_corrector,
    LearnedCorrectorMeta,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--t_cut", type=int, default=999)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--lambda_mse", type=float, default=1.0)
    p.add_argument("--lambda_cos", type=float, default=2.0)
    p.add_argument("--lambda_sr", type=float, default=0.3)
    p.add_argument("--lambda_t_ge_50", type=float, default=0.35)
    p.add_argument("--lambda_t_ge_200", type=float, default=0.15)
    p.add_argument("--init_ckpt", default="")
    p.add_argument("--best_by", choices=("val_loss", "val_dcos"), default="val_loss")
    p.add_argument("--mmap", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading train split from {args.data} (mmap={args.mmap}) ...", flush=True)
    t0 = time.time()
    train_ds = TrajectoryLateDataset(args.data, "train", mmap=args.mmap)
    val_ds = TrajectoryLateDataset(args.data, "val", mmap=args.mmap)
    print(f"  train n={len(train_ds)} val n={len(val_ds)} loaded in {time.time()-t0:.1f}s", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    sample_x, _, _, _ = train_ds[0]
    in_ch = sample_x.shape[0]
    meta = LearnedCorrectorMeta(
        t_cut=args.t_cut,
        alpha=args.alpha,
        lambda_t_ge_50=args.lambda_t_ge_50,
        lambda_t_ge_200=args.lambda_t_ge_200,
    )
    net = DeltaEpsNet(max_t=meta.max_t, in_channels=in_ch, out_channels=in_ch).to(device)

    if args.init_ckpt and os.path.isfile(args.init_ckpt):
        payload = torch.load(args.init_ckpt, map_location="cpu")
        net.load_state_dict(payload["state_dict"], strict=False)
        print(f"Warm-start from {args.init_ckpt}", flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    best_score = float("inf")
    best_path = os.path.join(args.output_dir, "ckpt_best.pt")

    for epoch in range(1, args.epochs + 1):
        net.train()
        train_loss = 0.0
        n_train = 0
        for x, eq, ef, t in train_loader:
            x, eq, ef, t = x.to(device), eq.to(device), ef.to(device), t.to(device)
            delta = net(x, eq, t)
            loss, _ = correction_loss(
                eq, ef, delta, t,
                lambda_mse=args.lambda_mse,
                lambda_cos=args.lambda_cos,
                lambda_sr=args.lambda_sr,
                t_cut=args.t_cut,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            train_loss += float(loss.item()) * x.shape[0]
            n_train += x.shape[0]
        sched.step()

        net.eval()
        val_loss = 0.0
        val_dcos = 0.0
        n_val = 0
        with torch.no_grad():
            for x, eq, ef, t in val_loader:
                x, eq, ef, t = x.to(device), eq.to(device), ef.to(device), t.to(device)
                delta = net(x, eq, t)
                loss, stats = correction_loss(
                    eq, ef, delta, t,
                    lambda_mse=args.lambda_mse,
                    lambda_cos=args.lambda_cos,
                    lambda_sr=args.lambda_sr,
                    t_cut=args.t_cut,
                )
                val_loss += float(loss.item()) * x.shape[0]
                val_dcos += stats["delta_cos"] * x.shape[0]
                n_val += x.shape[0]

        tr = train_loss / max(n_train, 1)
        vl = val_loss / max(n_val, 1)
        vd = val_dcos / max(n_val, 1)
        score = vl if args.best_by == "val_loss" else -vd
        print(
            f"epoch {epoch:3d}  loss={tr:.4f}  val_loss={vl:.4f}  val_dcos={vd:.4f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            save_learned_corrector(best_path, net, meta)
            print(f"  -> saved {best_path}", flush=True)

    print(f"DONE best -> {best_path}", flush=True)


if __name__ == "__main__":
    main()
