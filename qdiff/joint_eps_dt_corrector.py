"""Joint learned eps + delta_t corrector (FiLM / log-SNR backbone)."""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from noise_eps_corr.learned_noise_corrector import (
    LearnedCorrectorMeta,
    _clip_eps_norm,
    batch_cos,
    blend_lambda_for_t,
    correction_loss,
)

logger = logging.getLogger(__name__)


@dataclass
class JointCorrectorMeta(LearnedCorrectorMeta):
    dt_max: float = 2.0
    dt_eta: float = 1.0
    joint_eps_dt: bool = True


def _default_logsnr_table(max_t: int = 1000) -> torch.Tensor:
    betas = torch.linspace(1e-4, 2e-2, int(max_t), dtype=torch.float32)
    alpha_bar = (1.0 - betas).cumprod(0).clamp(1e-8, 1.0 - 1e-8)
    return torch.log(alpha_bar) - torch.log1p(-alpha_bar)


class TimeConditionedResBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(4, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = nn.Linear(time_dim, 2 * channels)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(temb).chunk(2, dim=1)
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        return x + self.conv2(F.silu(h))


class DeltaEpsDtNet(nn.Module):
    def __init__(self, max_t: float = 1000.0, dt_max: float = 2.0):
        super().__init__()
        self.max_t = max_t
        self.dt_max = float(dt_max)
        self.register_buffer("logsnr_table", _default_logsnr_table(int(max_t)), persistent=True)
        self.input_conv = nn.Conv2d(6, 32, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 64))
        self.blocks = nn.ModuleList([TimeConditionedResBlock(32, 64) for _ in range(3)])
        self.eps_head = nn.Sequential(nn.Conv2d(96, 32, 1), nn.SiLU(), nn.Conv2d(32, 3, 1))
        self.dt_head = nn.Sequential(nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor, eq: torch.Tensor, t: torch.Tensor):
        b = x.shape[0]
        t_flat = t.float().view(b).clamp(0.0, self.logsnr_table.numel() - 1)
        lo = t_flat.floor().long()
        hi = (lo + 1).clamp(max=self.logsnr_table.numel() - 1)
        w = t_flat - lo.float()
        logsnr = self.logsnr_table[lo] * (1.0 - w) + self.logsnr_table[hi] * w
        time_features = torch.stack(
            [logsnr, torch.sin(logsnr), torch.cos(logsnr),
             torch.sin(0.5 * logsnr), torch.cos(0.5 * logsnr)], dim=1,
        )
        t_emb = self.time_mlp(time_features)
        h = self.input_conv(torch.cat([x, eq], dim=1))
        for block in self.blocks:
            h = block(h, t_emb)
        t_spatial = t_emb.view(b, 64, 1, 1).expand(-1, -1, h.shape[2], h.shape[3])
        delta_eps = self.eps_head(torch.cat([h, t_spatial], dim=1))
        pooled = torch.cat([h.mean(dim=(2, 3)), t_emb], dim=1)
        delta_t = torch.tanh(self.dt_head(pooled).squeeze(-1)) * self.dt_max
        return delta_eps, delta_t


class JointEpsDtCorrector:
    def __init__(self, net: DeltaEpsDtNet, meta: JointCorrectorMeta):
        self.net = net
        self.meta = meta

    def blend_lambda(self, t_idx: float) -> float:
        return blend_lambda_for_t(t_idx, self.meta)

    @torch.no_grad()
    def predict_dt_only(self, eps_q, t_idx, xt=None):
        if xt is None:
            return torch.zeros(eps_q.shape[0], device=eps_q.device)
        t = torch.full((eps_q.shape[0],), float(t_idx), device=eps_q.device, dtype=torch.float32)
        _, delta_t = self.net(xt, eps_q, t)
        return delta_t

    @torch.no_grad()
    def correct_eps_at_t(self, eps_q, t_idx, xt=None):
        if xt is None:
            return eps_q
        if torch.is_tensor(t_idx):
            t = t_idx.to(device=eps_q.device, dtype=torch.float32).reshape(-1)
        else:
            t = torch.full((eps_q.shape[0],), float(t_idx), device=eps_q.device)
        lam = torch.tensor(
            [self.blend_lambda(float(v)) for v in t.detach().cpu()],
            device=eps_q.device, dtype=eps_q.dtype,
        ).view(-1, 1, 1, 1)
        if float(lam.max()) <= 0.0:
            return eps_q
        delta_eps, _ = self.net(xt, eps_q, t)
        return _clip_eps_norm(eps_q + lam * delta_eps, eps_q, self.meta.r_min, self.meta.r_max)

    @torch.no_grad()
    def correct_with_dt(self, eps_q, t_idx, alpha_bar, xt=None):
        del alpha_bar
        lam = self.blend_lambda(t_idx)
        if xt is None:
            return eps_q, torch.zeros(eps_q.shape[0], device=eps_q.device)
        t = torch.full((eps_q.shape[0],), float(t_idx), device=eps_q.device, dtype=torch.float32)
        delta_eps, delta_t = self.net(xt, eps_q, t)
        if lam <= 0.0:
            return eps_q, delta_t
        return _clip_eps_norm(eps_q + lam * delta_eps, eps_q, self.meta.r_min, self.meta.r_max), delta_t

    def train_mode_off(self) -> None:
        self.net.eval()


def save_joint_corrector(path, net, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save({"meta": asdict(meta), "state_dict": net.state_dict(), "kind": "joint_eps_dt"}, path)


def load_joint_corrector(path, device):
    payload = torch.load(path, map_location=device)
    meta_dict = payload.get("meta", {})
    fields = JointCorrectorMeta.__dataclass_fields__
    meta = JointCorrectorMeta(**{k: v for k, v in meta_dict.items() if k in fields})
    meta.joint_eps_dt = True
    net = DeltaEpsDtNet(max_t=meta.max_t, dt_max=meta.dt_max).to(device)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return JointEpsDtCorrector(net, meta)


def freeze_eps_head(net: DeltaEpsDtNet) -> None:
    for p in net.eps_head.parameters():
        p.requires_grad = False


def unfreeze_all(net: DeltaEpsDtNet) -> None:
    for p in net.parameters():
        p.requires_grad = True


def load_eps_weights_into_joint(net: DeltaEpsDtNet, eps_ckpt: str) -> None:
    payload = torch.load(eps_ckpt, map_location="cpu")
    src = payload.get("state_dict", payload)
    dst = net.state_dict()
    loaded = 0
    for k, v in src.items():
        if k in dst and dst[k].shape == v.shape:
            dst[k] = v
            loaded += 1
    net.load_state_dict(dst)
    logger.info("Loaded %d tensors from %s", loaded, eps_ckpt)


def joint_correction_loss(
    eq, ef, delta_eps, delta_t, dt_star, t, *,
    lambda_mse=1.0, lambda_cos=2.0, lambda_sr=0.3, lambda_dt=0.3, t_cut=50, dt_only=False,
):
    if dt_only:
        loss_dt = F.smooth_l1_loss(delta_t, dt_star)
        return loss_dt, {"loss": float(loss_dt.item()), "loss_dt": float(loss_dt.item()),
                         "dt_mae": float((delta_t - dt_star).abs().mean().item())}
    loss_eps, stats = correction_loss(
        eq, ef, delta_eps, t, lambda_mse=lambda_mse, lambda_cos=lambda_cos, lambda_sr=lambda_sr, t_cut=t_cut,
    )
    loss_dt = F.smooth_l1_loss(delta_t, dt_star)
    with torch.no_grad():
        mask = (batch_cos(eq + delta_eps, ef) > batch_cos(eq, ef)).float()
    loss_dt_masked = (F.smooth_l1_loss(delta_t, dt_star, reduction="none") * mask).mean()
    loss = loss_eps + lambda_dt * loss_dt_masked
    stats.update({"loss": float(loss.item()), "loss_dt": float(loss_dt.item()),
                  "loss_dt_masked": float(loss_dt_masked.item()),
                  "dt_mae": float((delta_t - dt_star).abs().mean().item())})
    return loss, stats
