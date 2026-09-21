"""Epsilon oracle labels for delta-t (user Method).

For state (x_t^q, t) with FP noise label ef = eps_FP(x, t):

    dt* = argmin_{τ in [t-W/2, t+W/2]} dist(eps_Q(x, τ), ef)

where dist is mse / l1 / smooth_l1 over spatial+channel dims.

Uses the stored traj `ef` (same QuantModel with quant off at nominal t) so we
only re-run the quantized UNet over the time window.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _eps_dist(eq: torch.Tensor, ef: torch.Tensor, criterion: str) -> torch.Tensor:
    """Per-sample distance [B] between eq and ef."""
    reduce_dims = tuple(range(1, eq.ndim))
    if criterion == "mse":
        return (eq - ef).pow(2).mean(dim=reduce_dims)
    if criterion == "l1":
        return (eq - ef).abs().mean(dim=reduce_dims)
    if criterion == "smooth_l1":
        return F.smooth_l1_loss(eq, ef, reduction="none").mean(dim=reduce_dims)
    raise ValueError(f"unknown criterion: {criterion}")


@torch.no_grad()
def eps_oracle_dt_star(
    quant_model,
    x: torch.Tensor,
    ef: torch.Tensor,
    t: torch.Tensor,
    *,
    window: int = 40,
    n_grid: Optional[int] = None,
    t_max: int = 999,
    t_cutoff: int = 0,
    criterion: str = "smooth_l1",
    betas: Optional[torch.Tensor] = None,
    t_next: Optional[torch.Tensor] = None,
    return_curves: bool = False,
):
    """Per-sample dt* [B] via Q-UNet grid search vs FP label ef.

    Loops over the time grid (not B*ng expand) to keep memory bounded.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    criterion = str(criterion).lower()
    if (criterion == "update_mse" or return_curves) and (betas is None or t_next is None):
        raise ValueError("update/state curves require betas and per-sample t_next")
    device = x.device
    b = x.shape[0]
    half = max(int(window) // 2, 0)
    ng = int(n_grid) if n_grid is not None else (2 * half + 1)
    ng = max(ng, 3)
    t_f = t.float().to(device)
    offsets = torch.linspace(-float(half), float(half), ng, device=device)
    tau = (t_f.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0.0, float(t_max))  # [B, ng]

    quant_model.set_quant_state(weight_quant=True, act_quant=True)
    def alpha_bar(tv: torch.Tensor) -> torch.Tensor:
        ab = (1.0 - betas.to(device)).cumprod(0).clamp_min(1e-12)
        tv = tv.clamp(0.0, ab.numel() - 1)
        lo = tv.floor().long()
        hi = (lo + 1).clamp(max=ab.numel() - 1)
        w = tv - lo.float()
        logv = (1.0 - w) * ab[lo].log() + w * ab[hi].log()
        return logv.exp().view(-1, 1, 1, 1)

    def deterministic_update(xv, epsv, at, an):
        x0 = (xv - (1.0 - at).sqrt() * epsv) / at.sqrt()
        return an.sqrt() * x0 + (1.0 - an).sqrt() * epsv

    if criterion == "update_mse":
        t_next_f = t_next.float().to(device)
        at_ref = alpha_bar(t_f)
        # alpha_bar(-1)=1 for the terminal x0 update.
        an_ref = torch.where(
            (t_next_f < 0).view(-1, 1, 1, 1),
            torch.ones(b, 1, 1, 1, device=device),
            alpha_bar(t_next_f.clamp_min(0.0)),
        )
        x_next_ref = deterministic_update(x, ef, at_ref, an_ref)
    errs, eps_mse_curve, state_mse_curve = [], [], []
    for k in range(ng):
        eq_k = quant_model(x, tau[:, k])
        eps_mse_k = _eps_dist(eq_k, ef, "mse")
        eps_mse_curve.append(eps_mse_k)
        if criterion == "update_mse":
            # We correct only the UNet time conditioning. The DDIM integration
            # grid remains nominal, exactly matching sparse-refresh inference.
            x_next_k = deterministic_update(x, eq_k, at_ref, an_ref)
            state_mse_k = (x_next_k - x_next_ref).pow(2).mean(dim=(1, 2, 3))
            state_mse_curve.append(state_mse_k)
            errs.append(state_mse_k)
        else:
            errs.append(_eps_dist(eq_k, ef, criterion))
    err = torch.stack(errs, dim=1)  # [B, ng]
    best = err.argmin(dim=1)
    t_s = tau[torch.arange(b, device=device), best]
    dt = t_s - t_f
    if t_cutoff is not None and int(t_cutoff) > 0:
        dt = torch.where(t_f > float(t_cutoff), dt, torch.zeros_like(dt))
    if not return_curves:
        return dt
    return {
        "dt_star": dt,
        "offsets": offsets,
        "eps_mse_curve": torch.stack(eps_mse_curve, dim=1),
        "state_mse_curve": torch.stack(state_mse_curve, dim=1),
    }
