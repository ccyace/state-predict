"""Time-Shift style variance-matching labels for delta-t supervision.

Follows Algorithm 3 of
``Alleviating Exposure Bias in Diffusion Models through Sampling with Shifted Time Steps``:
for state x at nominal t, choose

    t_s = argmin_{τ in [t-w/2, t+w/2]} |var(x) - (1 - alpha_bar_τ)|

and set dt_star = t_s - t. No extra UNet calls.

Hard argmin saturates at window edges on CIFAR trajectories (schedule is flat at
high t; signal variance dominates at low t). For training we default to a soft
window expectation, which yields interior continuous labels while staying
faithful to the same variance-matching criterion.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def alphas_cumprod_from_betas(betas: torch.Tensor) -> torch.Tensor:
    return torch.cumprod(1.0 - betas.float().reshape(-1), dim=0)


def schedule_noise_variance(alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """σ_t^2 = 1 - ᾱ_t used by Time-Shift matching."""
    return (1.0 - alphas_cumprod.float()).clamp(min=0.0)


def sample_spatial_variance(x: torch.Tensor) -> torch.Tensor:
    """Per-sample internal variance (paper's single-sample variance estimator)."""
    return x.float().reshape(x.shape[0], -1).var(dim=1, unbiased=False)


@torch.no_grad()
def timeshift_dt_star(
    x: torch.Tensor,
    t: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    *,
    window: int = 40,
    t_max: Optional[int] = None,
    t_cutoff: int = 300,
    soft: bool = True,
    temperature: float = 0.01,
    signal_var: float = 0.0,
) -> torch.Tensor:
    """Return dt_star [B] = t_s - t via variance matching in a sliding window.

    Args:
        soft: if True, use softmax-weighted expectation over the window
            (better training labels). If False, hard argmin (paper Algorithm 3).
        temperature: softmax temperature on |var(x) - σ_τ^2| (only if soft).
        signal_var: if >0, match against E[||x_τ||^2] ≈ c·ᾱ_τ + (1-ᾱ_τ)
            instead of pure noise variance 1-ᾱ_τ.
        t_cutoff: for t <= cutoff, return 0 (paper disables shift late / low-noise).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    device = x.device
    ab = alphas_cumprod.to(device=device, dtype=torch.float32).reshape(-1)
    tmax = int(ab.numel() - 1 if t_max is None else t_max)
    half = max(int(window) // 2, 0)

    v = sample_spatial_variance(x)  # [B]
    t_f = t.float().to(device)
    t_i = t_f.round().long().clamp(0, tmax)
    offsets = torch.arange(-half, half + 1, device=device, dtype=torch.long)
    tau = (t_i.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, tmax)  # [B, W]

    if signal_var and float(signal_var) > 0.0:
        c = float(signal_var)
        target = c * ab[tau] + (1.0 - ab[tau])
    else:
        target = schedule_noise_variance(ab)[tau]

    err = (v.unsqueeze(1) - target).abs()
    if soft:
        temp = max(float(temperature), 1e-6)
        weights = F.softmax(-err / temp, dim=1)
        dt = (weights * offsets.float().unsqueeze(0)).sum(dim=1)
    else:
        best = err.argmin(dim=1)
        t_s = tau[torch.arange(x.shape[0], device=device), best].float()
        dt = t_s - t_f

    if t_cutoff is not None and int(t_cutoff) >= 0:
        dt = torch.where(t_i > int(t_cutoff), dt, torch.zeros_like(dt))
    return dt


@torch.no_grad()
def timeshift_dt_star_paper(
    x: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    *,
    window: int = 40,
    t_max: Optional[int] = None,
    t_cutoff: int = 300,
    soft: bool = True,
    temperature: float = 0.01,
) -> torch.Tensor:
    """Paper-style label: δt* = t_s - t_next.

    Match var(x) to 1-ᾱ_τ in a window around the *nominal next* time (official
    TS-DPM code), for use as the DDIM destination time. Returns 0 when t <= cutoff.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    device = x.device
    ab = alphas_cumprod.to(device=device, dtype=torch.float32).reshape(-1)
    sig2 = schedule_noise_variance(ab)
    tmax = int(ab.numel() - 1 if t_max is None else t_max)
    half = max(int(window) // 2, 0)

    v = sample_spatial_variance(x)
    t_f = t.float().to(device)
    t_i = t_f.round().long().clamp(0, tmax)
    t_next_f = t_next.float().to(device)
    # terminal steps (t_next < 0): no shift
    terminal = t_next_f < 0
    t_next_i = t_next_f.round().long().clamp(0, tmax)

    offsets = torch.arange(-half, half + 1, device=device, dtype=torch.long)
    tau = (t_next_i.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, tmax)
    target = sig2[tau]
    err = (v.unsqueeze(1) - target).abs()
    if soft:
        temp = max(float(temperature), 1e-6)
        weights = F.softmax(-err / temp, dim=1)
        t_s = (weights * tau.float()).sum(dim=1)
    else:
        best = err.argmin(dim=1)
        t_s = tau[torch.arange(x.shape[0], device=device), best].float()
    dt = t_s - t_next_f
    dt = torch.where(terminal, torch.zeros_like(dt), dt)
    if t_cutoff is not None and int(t_cutoff) >= 0:
        dt = torch.where(t_i > int(t_cutoff), dt, torch.zeros_like(dt))
    return dt


def build_t_to_tnext_map(seq) -> dict:
    """Map nominal grid t -> next grid t for a DDIM seq (reverse sampling)."""
    seq_next = [-1] + list(seq[:-1])
    return {int(i): float(j) for i, j in zip(reversed(list(seq)), reversed(seq_next))}


def t_next_from_map(t: torch.Tensor, t2n: dict, t_max: int = 999) -> torch.Tensor:
    table = torch.full((t_max + 1,), -1.0, dtype=torch.float32)
    for k, v in t2n.items():
        if 0 <= int(k) <= t_max:
            table[int(k)] = float(v)
    idx = t.float().round().long().clamp(0, t_max)
    return table.to(t.device)[idx]


def build_alphas_cumprod_from_config(config) -> torch.Tensor:
    from sample_diffusion_ddim import get_beta_schedule

    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    return alphas_cumprod_from_betas(torch.as_tensor(np.asarray(betas), dtype=torch.float32))
