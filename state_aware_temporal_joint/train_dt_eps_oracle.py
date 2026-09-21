#!/usr/bin/env python
"""Train delta-t head with precomputed epsilon-MSE oracle labels.

Example:
  python state_aware_temporal_joint/train_dt_eps_oracle.py \\
    --data output/noise_corr/train_data/traj_5k_fullstep_cl_cifar_w8a8.pt \\
    --dt_star output/noise_corr/train_data/dt_star_eps_oracle_w40.pt \\
    --init_eps_ckpt noise_eps_corr/w8a8_corr_ckpt_best.pt \\
    --output_dir state_aware_temporal_joint/runs/dt_eps_oracle_w8a8 \\
    --dt_only
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
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import seed_everything

from qdiff.joint_eps_dt_corrector import (
    DeltaEpsDtNet,
    JointCorrectorMeta,
    freeze_eps_head,
    load_eps_weights_into_joint,
    save_joint_corrector,
    unfreeze_all,
)


class TrajWithDtStar(Dataset):
    def __init__(self, data_path: str, dt_star_path: str, split: str, val_mod: int = 5):
        print(f"Loading {split} from {data_path} + {dt_star_path}", flush=True)
        t0 = time.time()
        raw = torch.load(data_path, map_location="cpu", mmap=True)
        lab = torch.load(dt_star_path, map_location="cpu")
        dt = lab["dt_star"] if isinstance(lab, dict) and "dt_star" in lab else lab
        n_lab = int(dt.shape[0])
        n_raw = int(raw["t"].shape[0])
        if n_lab > n_raw:
            raise ValueError(f"dt_star len {n_lab} > traj len {n_raw}")
        if n_lab < n_raw:
            print(f"  Using first {n_lab}/{n_raw} traj samples (partial labels)", flush=True)
        traj_id = raw["traj_id"][:n_lab].long()
        is_val = (traj_id % val_mod) == 0
        mask = is_val if split == "val" else ~is_val
        self._indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        self._raw = raw
        self._dt = dt.float()
        self._eps_curve = lab.get("eps_mse_curve") if isinstance(lab, dict) else None
        self._state_curve = lab.get("state_mse_curve") if isinstance(lab, dict) else None
        self.offsets = lab.get("offsets") if isinstance(lab, dict) else None
        self._n_lab = n_lab
        print(f"  {split} n={len(self)} in {time.time()-t0:.1f}s", flush=True)

    def __len__(self):
        return int(self._indices.shape[0])

    def __getitem__(self, idx: int):
        i = int(self._indices[idx])
        row = (
            self._raw["x"][i].float(),
            self._raw["eq"][i].float(),
            self._raw["ef"][i].float(),
            self._raw["t"][i].float(),
            self._dt[i],
        )
        if self._eps_curve is not None and self._state_curve is not None:
            row += (self._eps_curve[i].float(), self._state_curve[i].float())
        return row


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="output/noise_corr/train_data/traj_5k_fullstep_cl_cifar_w8a8.pt")
    p.add_argument("--dt_star", default="output/noise_corr/train_data/dt_star_eps_oracle_w40.pt")
    p.add_argument("--init_eps_ckpt", default="noise_eps_corr/w8a8_corr_ckpt_best.pt")
    p.add_argument("--output_dir", default="state_aware_temporal_joint/runs/dt_eps_oracle_w8a8")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dt_max", type=float, default=20.0)
    p.add_argument("--lambda_dt", type=float, default=1.0)
    p.add_argument("--lambda_eps", type=float, default=0.1)
    p.add_argument("--lambda_state", type=float, default=0.1)
    p.add_argument("--t_cutoff", type=int, default=0,
                   help="only supervise t>t_cutoff; 0 = all timesteps")
    p.add_argument(
        "--loss", choices=("smooth_l1", "l1"), default="smooth_l1",
        help="regression loss on δt (original run used smooth_l1)",
    )
    p.add_argument("--dt_only", action="store_true", default=True)
    p.add_argument("--joint", action="store_true")
    p.add_argument("--val_mod", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--val_batches", type=int, default=50)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def _masked_loss(delta_t, dt_star, t, t_cutoff: int, loss_type: str = "smooth_l1"):
    if loss_type == "l1":
        per = F.l1_loss(delta_t, dt_star, reduction="none")
    else:
        per = F.smooth_l1_loss(delta_t, dt_star, reduction="none")
    if int(t_cutoff) > 0:
        mask = (t.float() > float(t_cutoff)).float()
    else:
        mask = torch.ones_like(per)
    denom = mask.sum().clamp_min(1.0)
    return (per * mask).sum() / denom, mask


def _interp_curve(curve: torch.Tensor, delta_t: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear differentiable lookup of a precomputed candidate-loss curve."""
    offsets = offsets.to(device=curve.device, dtype=curve.dtype)
    step = (offsets[-1] - offsets[0]) / max(offsets.numel() - 1, 1)
    pos = ((delta_t - offsets[0]) / step.clamp_min(1e-8)).clamp(0, offsets.numel() - 1)
    lo = pos.floor().long()
    hi = (lo + 1).clamp(max=offsets.numel() - 1)
    frac = pos - lo.float()
    y0 = curve.gather(1, lo[:, None]).squeeze(1)
    y1 = curve.gather(1, hi[:, None]).squeeze(1)
    return y0 + frac * (y1 - y0)


def combined_loss(delta_t, dt_star, t, eps_curve, state_curve, offsets, args):
    loss_dt, mask = _masked_loss(delta_t, dt_star, t, args.t_cutoff, args.loss)
    denom = mask.sum().clamp_min(1.0)
    # Normalize each curve by its nominal-time value so 0.1 has stable meaning
    # across diffusion timesteps and between epsilon/state units.
    zero_idx = int(torch.argmin(offsets.abs()))
    eps_scale = eps_curve[:, zero_idx].detach().clamp_min(1e-8)
    state_scale = state_curve[:, zero_idx].detach().clamp_min(1e-8)
    eps_per = _interp_curve(eps_curve, delta_t, offsets) / eps_scale
    state_per = _interp_curve(state_curve, delta_t, offsets) / state_scale
    loss_eps = (eps_per * mask).sum() / denom
    loss_state = (state_per * mask).sum() / denom
    total = args.lambda_dt * loss_dt + args.lambda_eps * loss_eps + args.lambda_state * loss_state
    return total, mask, loss_dt, loss_eps, loss_state


@torch.no_grad()
def evaluate(net, loader, device, args):
    net.eval()
    mae_sum, loss_sum, n_mask = 0.0, 0.0, 0.0
    for bi, batch in enumerate(loader):
        if bi >= max(int(args.val_batches), 1):
            break
        x, eq, ef, t, dt_star, eps_curve, state_curve = [b.to(device) for b in batch]
        dt_star = dt_star.clamp(-args.dt_max, args.dt_max)
        _de, delta_t = net(x, eq, t.float())
        del _de, ef
        loss, mask, *_ = combined_loss(
            delta_t, dt_star, t, eps_curve, state_curve, loader.dataset.offsets, args,
        )
        m = float(mask.sum().item())
        if m <= 0:
            continue
        loss_sum += float(loss.item()) * m
        mae_sum += float(((delta_t - dt_star).abs() * mask).sum().item())
        n_mask += m
    return {"val_loss": loss_sum / max(n_mask, 1.0), "val_dt_mae": mae_sum / max(n_mask, 1.0)}


def main():
    args = parse_args()
    if args.joint:
        args.dt_only = False
    if args.smoke:
        args.epochs = 1
        args.log_every = 1
        args.val_batches = 2
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds = TrajWithDtStar(args.data, args.dt_star, "train", val_mod=args.val_mod)
    val_ds = TrajWithDtStar(args.data, args.dt_star, "val", val_mod=args.val_mod)
    if train_ds.offsets is None:
        raise ValueError("Label file lacks loss curves; regenerate it with label_dt_eps_oracle.py")
    bs = min(8, args.batch_size) if args.smoke else args.batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    net = DeltaEpsDtNet(max_t=1000.0, dt_max=args.dt_max).to(device)
    if args.init_eps_ckpt and os.path.isfile(args.init_eps_ckpt):
        load_eps_weights_into_joint(net, args.init_eps_ckpt)
        print(f"Warm-started from {args.init_eps_ckpt}", flush=True)
    if args.dt_only:
        freeze_eps_head(net)
        trainable = [p for p in net.parameters() if p.requires_grad]
        print("Mode: dt_only", flush=True)
    else:
        unfreeze_all(net)
        trainable = list(net.parameters())
        print("Mode: joint", flush=True)

    print(
        f"train={len(train_ds)} val={len(val_ds)} params={sum(p.numel() for p in trainable):,} "
        f"dt_max={args.dt_max} t_cutoff={args.t_cutoff} loss={args.loss}",
        flush=True,
    )
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05,
    )
    meta = JointCorrectorMeta(
        t_cut=999, alpha=0.0, max_t=1000.0, dt_max=args.dt_max, joint_eps_dt=True,
    )

    history, best, started, global_step = [], float("inf"), time.time(), 0
    for epoch in range(1, args.epochs + 1):
        net.train()
        if args.dt_only:
            net.eps_head.eval()
        running = running_mae = seen = 0.0
        for batch in train_loader:
            x, eq, ef, t, dt_star, eps_curve, state_curve = [b.to(device, non_blocking=True) for b in batch]
            del ef
            dt_star = dt_star.clamp(-args.dt_max, args.dt_max)
            opt.zero_grad(set_to_none=True)
            _de, delta_t = net(x, eq, t.float())
            del _de
            loss, mask, loss_dt, loss_eps, loss_state = combined_loss(
                delta_t, dt_star, t, eps_curve, state_curve, train_ds.offsets, args,
            )
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
                    f"parts=({float(loss_dt):.4f},{float(loss_eps):.4f},{float(loss_state):.4f}) "
                    f"lr={sched.get_last_lr()[0]:.2e} time={time.time()-started:.1f}s",
                    flush=True,
                )
                running = running_mae = seen = 0.0
            if args.smoke and global_step >= 2:
                break
        sched.step()
        metrics = evaluate(net, val_loader, device, args)
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
