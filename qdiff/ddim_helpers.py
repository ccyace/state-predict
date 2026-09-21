"""Shared DDIM helpers: alpha interpolation, single-step update, eps eval, float-t drift."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch


def alpha_bar_at(betas: torch.Tensor, t_val, device: torch.device) -> torch.Tensor:
    """Continuous scalar or per-sample diffusion index -> broadcastable alpha_bar."""
    betas = betas.to(device)
    beta_ext = torch.cat([torch.zeros(1, device=device), betas], dim=0)
    log_ab = torch.log((1.0 - beta_ext).clamp(min=1e-12)).cumsum(0)
    tv = torch.as_tensor(t_val, device=device, dtype=torch.float32).reshape(-1)
    tv = tv.clamp(0.0, betas.numel() - 1)
    t0 = tv.floor().long()
    t1 = (t0 + 1).clamp(max=betas.numel() - 1)
    w = tv - t0.float()
    log_v = (1.0 - w) * log_ab[t0 + 1] + w * log_ab[t1 + 1]
    return torch.exp(log_v).view(-1, 1, 1, 1)


def lambda_at(betas: torch.Tensor, t_val: float, device: torch.device) -> float:
    """log-SNR lambda(t) = 0.5 log(alpha_bar / (1 - alpha_bar))."""
    at = alpha_bar_at(betas, t_val, device).clamp(min=1e-12, max=1.0 - 1e-12)
    return float(0.5 * torch.log(at / (1.0 - at)).item())


def dlambda_dt(
    betas: torch.Tensor,
    t_val: float,
    device: torch.device,
    *,
    eps: float = 1e-4,
) -> float:
    """Central difference d lambda / d t on the discrete schedule."""
    t_max = float(betas.numel() - 1)
    t_val = float(max(0.0, min(t_val, t_max)))
    h = min(eps, 0.5)
    t_lo = max(0.0, t_val - h)
    t_hi = min(t_max, t_val + h)
    if t_hi <= t_lo:
        return 0.0
    lam_hi = lambda_at(betas, t_hi, device)
    lam_lo = lambda_at(betas, t_lo, device)
    return (lam_hi - lam_lo) / (t_hi - t_lo)


def x0_from_eps(x: torch.Tensor, eps: torch.Tensor, at: torch.Tensor) -> torch.Tensor:
    return (x - eps * (1.0 - at).sqrt()) / at.sqrt()


def dx0_dlambda(
    x: torch.Tensor,
    eps: torch.Tensor,
    at: torch.Tensor,
) -> torch.Tensor:
    """
    d x0 / d lambda at fixed (x, eps), lambda = 0.5 log(a / (1-a)), a = alpha_bar.
    """
    a = at.clamp(min=1e-12, max=1.0 - 1e-12)
    sqrt_a = a.sqrt()
    sqrt_1ma = (1.0 - a).sqrt()
    w = 1.0 / sqrt_a
    da_dlam = 2.0 * a * (1.0 - a)
    dw_da = -0.5 * a ** (-1.5)
    dsqrt_1ma_da = -0.5 * (1.0 - a).clamp(min=1e-12) ** (-0.5)
    dx0_da = x * dw_da - eps * (dsqrt_1ma_da * w + sqrt_1ma * dw_da)
    return dx0_da * da_dlam


def delta_t_from_delta_eps(
    delta_eps: torch.Tensor,
    x: torch.Tensor,
    eps_q: torch.Tensor,
    t_val: float,
    betas: torch.Tensor,
    device: torch.device,
    *,
    lambda_scale: float = 0.1,
    dt_max: float = 2.0,
    eps: float = 1e-8,
) -> float:
    """
    Route A: delta_eps -> delta_x0 -> project on dx0/dlambda -> delta_t.

    Returns scalar delta_t (index units) averaged over batch.
    """
    if float(lambda_scale) <= 0.0:
        return 0.0
    mag = delta_eps.abs().max().item()
    if mag < 1e-12:
        return 0.0

    at = alpha_bar_at(betas, t_val, device)
    delta_x0 = -delta_eps * (1.0 - at).sqrt() / at.sqrt()
    jac = dx0_dlambda(x, eps_q, at)

    b = delta_x0.shape[0]
    d_lam = []
    for i in range(b):
        d_flat = delta_x0[i].reshape(-1)
        j_flat = jac[i].reshape(-1)
        denom = torch.dot(j_flat, j_flat).item() + eps
        d_lam.append(torch.dot(d_flat, j_flat).item() / denom)

    d_lam_mean = float(sum(d_lam) / max(b, 1))
    dlam_dt = dlambda_dt(betas, t_val, device)
    if abs(dlam_dt) < 1e-12:
        return 0.0

    dt = lambda_scale * d_lam_mean / dlam_dt
    dt = float(max(-dt_max, min(dt_max, dt)))
    return dt


def nominal_step_size(i: int, j: int, t_eff: float) -> float:
    """Grid step i -> j in diffusion index units (i decreases toward 0)."""
    if j >= 0:
        return float(i - j)
    return float(max(t_eff, 0.0))


def ddim_update(
    x: torch.Tensor,
    eps: torch.Tensor,
    at: torch.Tensor,
    at_next: torch.Tensor,
    *,
    eta: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    x0 = (x - eps * (1.0 - at).sqrt()) / at.sqrt()
    if float(eta) > 0.0:
        # Guard: at_next must be >= at for a reverse step; otherwise (1-at/at_next)<0.
        ratio = (1.0 - at / at_next.clamp(min=at)) * (1.0 - at_next) / (1.0 - at).clamp(min=1e-12)
        c1 = float(eta) * ratio.clamp(min=0.0).sqrt()
        c2 = ((1.0 - at_next) - c1 ** 2).clamp(min=0.0).sqrt()
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        return at_next.sqrt() * x0 + c2 * eps + c1 * noise
    c2 = (1.0 - at_next).clamp(min=0.0).sqrt()
    return at_next.sqrt() * x0 + c2 * eps


def ddim_update_vsc(
    x: torch.Tensor,
    eps: torch.Tensor,
    at: torch.Tensor,
    at_next: torch.Tensor,
    *,
    eta: float,
    residual_var: Optional[torch.Tensor] = None,
    absorb_strength: float = 1.0,
    max_budget_fraction: float = 0.9,
    generator: Optional[torch.Generator] = None,
):
    """Generalized DDIM update with PTQD-style stochastic variance absorption.

    The nominal DDIM mean coefficient c2 is intentionally kept unchanged.  Only
    the independently sampled variance c1^2 is reduced by the state variance
    already contributed by the remaining epsilon error.
    """
    x0 = (x - eps * (1.0 - at).sqrt()) / at.sqrt()
    ratio = (1.0 - at / at_next.clamp(min=at)) * (1.0 - at_next) / (1.0 - at).clamp(min=1e-12)
    sigma2 = (float(eta) ** 2) * ratio.clamp(min=0.0)
    c2 = ((1.0 - at_next) - sigma2).clamp(min=0.0).sqrt()
    eps_coef = c2 - (at_next / at).sqrt() * (1.0 - at).sqrt()
    required = torch.zeros_like(sigma2)
    absorbed = torch.zeros_like(sigma2)
    sigma2_used = sigma2
    if residual_var is not None and float(eta) > 0.0:
        residual_var = residual_var.to(device=x.device, dtype=x.dtype).reshape(-1, 1, 1, 1)
        required = eps_coef.square() * residual_var
        cap = float(max_budget_fraction) * sigma2
        absorbed = float(absorb_strength) * torch.minimum(required, cap)
        sigma2_used = (sigma2 - absorbed).clamp(min=0.0)
    noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    out = at_next.sqrt() * x0 + c2 * eps + sigma2_used.sqrt() * noise
    return out, {
        "sigma2": sigma2,
        "sigma2_used": sigma2_used,
        "eps_coef": eps_coef,
        "required": required,
        "absorbed": absorbed,
    }


@torch.no_grad()
def eval_eps(
    model,
    x: torch.Tensor,
    t_val: float,
    betas: torch.Tensor,
    *,
    noise_corrector=None,
    device: torch.device,
) -> torch.Tensor:
    n = x.shape[0]
    t_tensor = torch.full((n,), float(t_val), device=device)
    et = model(x, t_tensor)
    if noise_corrector is not None:
        at = alpha_bar_at(betas, t_val, device)
        et = noise_corrector.correct(et, t_val, at, xt=x)
    return et


@torch.no_grad()
def eval_eps_with_delta(
    model,
    x: torch.Tensor,
    t_val: float,
    betas: torch.Tensor,
    *,
    noise_corrector=None,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (eps_corr, delta_eps_applied) for float-t drift diagnostics."""
    n = x.shape[0]
    t_tensor = torch.full((n,), float(t_val), device=device)
    eps_q = model(x, t_tensor)
    if noise_corrector is None:
        return eps_q, torch.zeros_like(eps_q)
    at = alpha_bar_at(betas, t_val, device)
    eps_corr = noise_corrector.correct(eps_q, t_val, at, xt=x)
    return eps_corr, eps_corr - eps_q


@torch.no_grad()
def eval_eps_unet_carry(
    model,
    x: torch.Tensor,
    t_unet: float,
    t_corr: float,
    betas: torch.Tensor,
    *,
    noise_corrector=None,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Scheme B: UNet at t_unet (= t_nom + carry), corrector at nominal t_corr.
    Returns (eps_corr, eps_q, delta_eps_applied).
    """
    n = x.shape[0]
    eps_q = model(x, torch.full((n,), float(t_unet), device=device))
    if noise_corrector is None:
        z = torch.zeros_like(eps_q)
        return eps_q, eps_q, z
    at = alpha_bar_at(betas, float(t_corr), device)
    eps_corr = noise_corrector.correct(eps_q, float(t_corr), at, xt=x)
    return eps_corr, eps_q, eps_corr - eps_q


def clamp_time_index(t_val: float, t_max: float) -> float:
    return float(max(0.0, min(float(t_val), float(t_max))))


def append_unet_carry_log(
    log: Optional[List[Dict[str, Any]]],
    *,
    step: int,
    t_nom: int,
    t_next_nom: int,
    t_unet: float,
    carry: float,
    delta_t: float,
    carry_next: float,
) -> None:
    if log is None:
        return
    log.append(
        {
            "step": int(step),
            "t_nom": int(t_nom),
            "t_next_nom": int(t_next_nom),
            "t_unet": float(t_unet),
            "carry": float(carry),
            "delta_t": float(delta_t),
            "carry_next": float(carry_next),
            "mode": "unet_carry",
        }
    )


@torch.no_grad()
def compute_dt_star_grid(
    quant_model,
    x: torch.Tensor,
    ef: torch.Tensor,
    t_nom: float,
    *,
    dt_search: float = 2.0,
    n_grid: int = 17,
) -> torch.Tensor:
    """
    Per-sample dt* = t* - t_nom where t* minimizes ||eps_q(x,t) - ef||^2 on a grid.
    Returns float tensor [B]. Uses one batched forward over the time grid.
    """
    device = x.device
    b = x.shape[0]
    ng = int(n_grid)
    ts = torch.linspace(float(t_nom) - dt_search, float(t_nom) + dt_search, ng, device=device)
    quant_model.set_quant_state(weight_quant=True, act_quant=True)
    x_rep = x.unsqueeze(1).expand(b, ng, *x.shape[1:]).reshape(b * ng, *x.shape[1:])
    t_rep = ts.unsqueeze(0).expand(b, ng).reshape(b * ng)
    eq_all = quant_model(x_rep, t_rep).view(b, ng, *x.shape[1:])
    err = (eq_all - ef.unsqueeze(1)).pow(2).mean(dim=(2, 3, 4))
    best_t = ts[err.argmin(dim=1)]
    return best_t - float(t_nom)


def append_learned_float_t_log(
    log: Optional[List[Dict[str, Any]]],
    *,
    step: int,
    t_nom: int,
    t_next_nom: int,
    t_eff: float,
    t_next_eff: float,
    dt_pred: float,
    dt_applied: float,
    h_nom: float,
) -> None:
    if log is None:
        return
    log.append(
        {
            "step": int(step),
            "t_nom": int(t_nom),
            "t_next_nom": int(t_next_nom),
            "t_eff": float(torch.as_tensor(t_eff).float().mean()),
            "t_next_eff": float(torch.as_tensor(t_next_eff).float().mean()),
            "dt_pred": float(torch.as_tensor(dt_pred).float().mean()),
            "dt_applied": float(torch.as_tensor(dt_applied).float().mean()),
            "drift": float(torch.as_tensor(t_eff).float().mean() - t_nom),
            "h_nom": float(h_nom),
            "mode": "learned_float_t",
        }
    )


def append_float_t_log(
    log: Optional[List[Dict[str, Any]]],
    *,
    step: int,
    t_nom: int,
    t_next_nom: int,
    t_eff: float,
    t_next_eff: float,
    delta_t: float,
    h_nom: float,
) -> None:
    if log is None:
        return
    drift_next = float(t_next_eff - t_next_nom) if int(t_next_nom) >= 0 else 0.0
    log.append(
        {
            "step": int(step),
            "t_nom": int(t_nom),
            "t_next_nom": int(t_next_nom),
            "t_eff": float(t_eff),
            "t_next_eff": float(t_next_eff),
            "delta_t": float(delta_t),
            "drift": float(t_eff - t_nom),
            "drift_next": drift_next,
            "h_nom": float(h_nom),
        }
    )
