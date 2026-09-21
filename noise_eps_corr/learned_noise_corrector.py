"""Learned noise corrector (MVP): CNN predicts delta_eps on (x, eps_q, t)."""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
EPS = 1e-8


@dataclass
class LearnedCorrectorMeta:
    t_cut: int = 50
    alpha: float = 0.5
    r_min: float = 0.8
    r_max: float = 1.25
    max_t: float = 1000.0
    lambda_t_lt_20: float = 1.0
    lambda_t_lt_40: float = 0.5
    lambda_t_lt_cut: float = 0.2
    lambda_t_ge_50: float = 0.35
    lambda_t_ge_200: float = 0.15


class DeltaEpsNet(nn.Module):
    def __init__(self, max_t: float = 1000.0, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.max_t = max_t
        self.in_channels = in_channels
        self.out_channels = out_channels
        enc_in = in_channels * 2
        self.encoder = nn.Sequential(
            nn.Conv2d(enc_in, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
        )
        self.time_mlp = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
        self.head = nn.Sequential(nn.Conv2d(96, 32, 1), nn.SiLU(), nn.Conv2d(32, out_channels, 1))

    def forward(self, x: torch.Tensor, eq: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = self.encoder(torch.cat([x, eq], dim=1))
        t_norm = (t.float().view(b, 1) / self.max_t).clamp(0.0, 1.0)
        t_emb = self.time_mlp(t_norm).view(b, 64, 1, 1).expand(-1, -1, h.shape[2], h.shape[3])
        return self.head(torch.cat([h, t_emb], dim=1))


def blend_lambda_for_t(t_idx: float, meta) -> float:
    t = float(t_idx)
    if t >= float(meta.t_cut):
        return 0.0
    if t >= 200.0:
        lam = float(meta.lambda_t_ge_200)
    elif t >= 50.0:
        lam = float(meta.lambda_t_ge_50)
    elif t < 20.0:
        lam = float(meta.lambda_t_lt_20)
    elif t < 40.0:
        lam = float(meta.lambda_t_lt_40)
    else:
        lam = float(meta.lambda_t_lt_cut)
    return float(meta.alpha * lam)


def batch_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_f = a.reshape(a.shape[0], -1)
    b_f = b.reshape(b.shape[0], -1)
    na = a_f.norm(dim=1).clamp(min=EPS)
    nb = b_f.norm(dim=1).clamp(min=EPS)
    return (a_f * b_f).sum(dim=1) / (na * nb)


def correction_loss(
    eq, ef, delta_pred, t, *, lambda_mse=1.0, lambda_cos=2.0, lambda_sr=0.3, t_cut=50,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    delta_star = ef - eq
    eq_corr = eq + delta_pred
    l_mse = (delta_pred - delta_star).pow(2).mean(dim=(1, 2, 3))
    l_cos = 1.0 - batch_cos(eq_corr, ef)
    ne = delta_pred.reshape(delta_pred.shape[0], -1).norm(dim=1).clamp(min=EPS)
    nf = delta_star.reshape(delta_star.shape[0], -1).norm(dim=1).clamp(min=EPS)
    l_sr = (ne / nf - 1.0).abs()
    w = torch.ones_like(t, dtype=torch.float32)
    if int(t_cut) >= 999:
        w = torch.where(t >= 200, torch.full_like(w, 0.5), w)
    per = lambda_mse * l_mse + lambda_cos * l_cos + lambda_sr * l_sr
    loss = (per * w).mean()
    with torch.no_grad():
        stats = {
            "loss": float(loss.item()),
            "cos_before": float(batch_cos(eq, ef).mean().item()),
            "cos_after": float(batch_cos(eq_corr, ef).mean().item()),
            "delta_cos": float((batch_cos(eq_corr, ef) - batch_cos(eq, ef)).mean().item()),
        }
    return loss, stats


def _clip_eps_norm(eps_corr, eps_q, r_min, r_max):
    corr_norm = eps_corr.reshape(eps_corr.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1).clamp(min=EPS)
    q_norm = eps_q.reshape(eps_q.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1).clamp(min=EPS)
    ratio = corr_norm / q_norm
    scale = torch.clamp(r_max / ratio, max=1.0)
    scale = torch.where(ratio < r_min, r_min / ratio, scale)
    return eps_corr * scale


class LearnedNoiseCorrector:
    def __init__(self, net: DeltaEpsNet, meta: LearnedCorrectorMeta):
        self.net = net
        self.meta = meta

    def blend_lambda(self, t_idx: float) -> float:
        return blend_lambda_for_t(t_idx, self.meta)

    @torch.no_grad()
    def correct(self, eps_q, t_idx, alpha_bar, xt=None, **_kwargs):
        del alpha_bar
        if xt is None:
            return eps_q
        if torch.is_tensor(t_idx):
            t = t_idx.detach().float().reshape(-1)
            lam = torch.tensor(
                [self.blend_lambda(float(v)) for v in t.tolist()],
                device=eps_q.device, dtype=eps_q.dtype,
            ).view(-1, 1, 1, 1)
            if float(lam.max().item()) <= 0.0:
                return eps_q
            delta = self.net(xt, eps_q, t)
            return _clip_eps_norm(eps_q + lam * delta, eps_q, self.meta.r_min, self.meta.r_max)
        lam = self.blend_lambda(float(t_idx))
        if lam <= 0.0:
            return eps_q
        t = torch.full((eps_q.shape[0],), float(t_idx), device=eps_q.device, dtype=torch.float32)
        return _clip_eps_norm(eps_q + lam * self.net(xt, eps_q, t), eps_q, self.meta.r_min, self.meta.r_max)

    def train_mode_off(self) -> None:
        self.net.eval()


class TrajectoryLateDataset(torch.utils.data.Dataset):
    def __init__(self, path, split="train", val_mod=5, mmap=True):
        self._raw = torch.load(path, map_location="cpu", mmap=mmap)
        traj_id = self._raw["traj_id"].long()
        is_val = (traj_id % val_mod) == 0
        mask = is_val if split == "val" else ~is_val
        self._indices = torch.nonzero(mask, as_tuple=False).squeeze(1)

    def __len__(self):
        return int(self._indices.shape[0])

    def __getitem__(self, idx):
        i = int(self._indices[idx])
        return (
            self._raw["x"][i].float(),
            self._raw["eq"][i].float(),
            self._raw["ef"][i].float(),
            self._raw["t"][i].float() if "t" in self._raw else self._raw["t_nom"][i].float(),
        )


def save_learned_corrector(path, net, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save({"meta": asdict(meta), "state_dict": net.state_dict(), "kind": "learned_delta_eps"}, path)


def load_learned_corrector(path, device):
    payload = torch.load(path, map_location=device)
    meta_dict = payload.get("meta", {})
    fields = LearnedCorrectorMeta.__dataclass_fields__
    meta = LearnedCorrectorMeta(**{k: v for k, v in meta_dict.items() if k in fields})
    sd = payload["state_dict"]
    out_ch = sd["head.2.weight"].shape[0]
    in_ch = sd["encoder.0.weight"].shape[1] // 2
    net = DeltaEpsNet(max_t=meta.max_t, in_channels=in_ch, out_channels=out_ch).to(device)
    net.load_state_dict(sd)
    net.eval()
    return LearnedNoiseCorrector(net, meta)


def load_corrector_from_ckpt(path, device):
    payload = torch.load(path, map_location="cpu")
    kind = payload.get("kind", "")
    if kind == "joint_eps_dt" or payload.get("meta", {}).get("joint_eps_dt", False):
        from qdiff.joint_eps_dt_corrector import load_joint_corrector
        return load_joint_corrector(path, device)
    return load_learned_corrector(path, device)
