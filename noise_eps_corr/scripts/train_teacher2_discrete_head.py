"""Train and evaluate the Teacher-2 discrete-error head in state space."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import seed_everything

from noise_eps_corr.teacher2_corrector import Teacher2DiscreteNet, batch_cosine


def _reject_legacy_output(path: str) -> None:
    abs_path = os.path.abspath(path)
    legacy = os.path.abspath(os.path.join(_ROOT, "output"))
    if abs_path == legacy or abs_path.startswith(legacy + os.sep):
        raise ValueError("Teacher-2 outputs must not be written under legacy output/")


class Teacher2Dataset(Dataset):
    def __init__(self, path, split, *, mmap=True):
        kw = {"map_location": "cpu"}
        if mmap:
            kw["mmap"] = True
        self.raw = torch.load(path, **kw)
        ids = self.raw["traj_id"].long()
        rem = ids.remainder(10)
        mask = rem >= 2 if split == "train" else rem == (0 if split == "val" else 1)
        self.indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        self.eta = float(self.raw["meta"]["eta"])

    def __len__(self):
        return self.indices.numel()

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        r = self.raw
        t, tn = r["t"][i].float(), r["t_next"][i].float()
        return (
            r["x"][i].float(),
            r["eps_corr"][i].float(),
            r["u"][i].float(),
            r["d_disc"][i].float(),
            t,
            torch.tensor(self.eta),
            t - tn,
            r["sigma"][i].float(),
            r["b_coeff"][i].float(),
        )


@torch.no_grad()
def compute_time_means(dataset):
    sums, counts = {}, defaultdict(int)
    raw = dataset.raw
    for idx_t in dataset.indices.split(1024):
        idx = idx_t.long()
        ts = raw["t"].index_select(0, idx).long()
        ds = raw["d_disc"].index_select(0, idx).float()
        for t in ts.unique().tolist():
            part = ds[ts == t].sum(0)
            sums[t] = part if t not in sums else sums[t] + part
            counts[t] += int((ts == t).sum())
    return {t: sums[t] / counts[t] for t in sums}


@torch.no_grad()
def evaluate(net, loader, device, time_means):
    net.eval()
    total = zero_err = pred_err = mean_err = cos_sum = norm_ratio_sum = 0.0
    n = 0
    per_t = defaultdict(lambda: [0.0, 0.0])
    for x, eps, u, target, t, eta, h, sigma, b_coeff in loader:
        x, eps, u, target = x.to(device), eps.to(device), u.to(device), target.to(device)
        t, eta, h, sigma = t.to(device), eta.to(device), h.to(device), sigma.to(device)
        b_coeff = b_coeff.to(device).view(-1, 1, 1, 1)
        pred = b_coeff * net(x, eps, u, t, eta, h, sigma)
        mean_pred = torch.stack([time_means[int(v)] for v in t.tolist()]).to(device)
        target_e = target.float().square().flatten(1).sum(1)
        pred_e = (target - pred).float().square().flatten(1).sum(1)
        mean_e = (target - mean_pred).float().square().flatten(1).sum(1)
        total += target_e.sum().item()
        zero_err += target_e.sum().item()
        pred_err += pred_e.sum().item()
        mean_err += mean_e.sum().item()
        cos_sum += batch_cosine(pred, target).sum().item()
        norm_ratio_sum += (pred.flatten(1).norm(dim=1) / (target.flatten(1).norm(dim=1) + 1e-12)).sum().item()
        n += x.shape[0]
        for tv in t.unique().tolist():
            m = t == tv
            per_t[str(int(tv))][0] += target_e[m].sum().item()
            per_t[str(int(tv))][1] += pred_e[m].sum().item()
    return {
        "r_d": 1.0 - pred_err / max(total, 1e-12),
        "r_d_time_mean": 1.0 - mean_err / max(total, 1e-12),
        "mse_zero": zero_err / max(n, 1) / (3 * 32 * 32),
        "mse_pred": pred_err / max(n, 1) / (3 * 32 * 32),
        "cosine": cos_sum / max(n, 1),
        "norm_ratio": norm_ratio_sum / max(n, 1),
        "per_t_r_d": {t: 1.0 - e[1] / max(e[0], 1e-12) for t, e in per_t.items()},
    }


def main():
    p = argparse.ArgumentParser(description="Train Teacher-2 discrete head")
    p.add_argument("--data", default="teacher2_exp/data/teacher2_pilot.pt")
    p.add_argument("--run_dir", default="teacher2_exp/runs/pilot")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--norm_penalty", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=7890)
    p.add_argument("--no_mmap", action="store_true")
    args = p.parse_args()
    _reject_legacy_output(args.run_dir)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.run_dir, exist_ok=True)

    train_ds = Teacher2Dataset(args.data, "train", mmap=not args.no_mmap)
    val_ds = Teacher2Dataset(args.data, "val", mmap=not args.no_mmap)
    test_ds = Teacher2Dataset(args.data, "test", mmap=not args.no_mmap)
    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise ValueError("train/val/test trajectory splits must all be non-empty")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    time_means = compute_time_means(train_ds)

    net = Teacher2DiscreteNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    best = float("inf")
    history = []
    meta = dict(train_ds.raw["meta"])

    for epoch in range(1, args.epochs + 1):
        net.train()
        loss_sum = n = 0
        for x, eps, u, target, t, eta, h, sigma, b_coeff in train_loader:
            x, eps, u, target = x.to(device), eps.to(device), u.to(device), target.to(device)
            t, eta, h, sigma = t.to(device), eta.to(device), h.to(device), sigma.to(device)
            b_coeff = b_coeff.to(device).view(-1, 1, 1, 1)
            state_pred = b_coeff * net(x, eps, u, t, eta, h, sigma)
            mse = (state_pred - target).square().mean()
            rms_ratio = (
                (state_pred.square().mean() + 1e-12).sqrt()
                / (target.square().mean().detach().sqrt() + 1e-12)
            )
            excess = torch.relu(rms_ratio - 1.5).square()
            loss = mse * (1.0 + args.norm_penalty * excess)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss: mse={float(mse.detach())} "
                    f"rms_ratio={float(rms_ratio.detach())}"
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * x.shape[0]
            n += x.shape[0]
        sched.step()
        val = evaluate(net, val_loader, device, time_means)
        rec = {"epoch": epoch, "train_loss": loss_sum / n, "lr": opt.param_groups[0]["lr"], **{f"val_{k}": v for k, v in val.items() if k != "per_t_r_d"}}
        history.append(rec)
        print(f"epoch {epoch:03d} loss={rec['train_loss']:.6g} val_Rd={val['r_d']:.4%} mean_Rd={val['r_d_time_mean']:.4%} cos={val['cosine']:.4f} ratio={val['norm_ratio']:.4f}", flush=True)
        if val["mse_pred"] < best:
            best = val["mse_pred"]
            torch.save({"kind": "teacher2_discrete", "meta": meta, "state_dict": net.state_dict(), "val": val}, os.path.join(args.run_dir, "ckpt_best.pt"))

    payload = torch.load(os.path.join(args.run_dir, "ckpt_best.pt"), map_location=device)
    net.load_state_dict(payload["state_dict"])
    test = evaluate(net, test_loader, device, time_means)
    torch.save({"kind": "teacher2_discrete", "meta": meta, "state_dict": net.state_dict(), "test": test}, os.path.join(args.run_dir, "ckpt_final.pt"))
    with open(os.path.join(args.run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "sizes": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)}, "history": history, "test": test}, f, indent=2)
    print(f"TEST R_d={test['r_d']:.4%} time_mean_Rd={test['r_d_time_mean']:.4%} cos={test['cosine']:.4f} ratio={test['norm_ratio']:.4f}", flush=True)


if __name__ == "__main__":
    main()
