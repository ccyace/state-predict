"""
Trajectory state-error tracking for quantized DDIM sampling (Section 6, term ①).

Maintains e_hat ≈ x^q_t - x_t. Scheme A (damped state propagation):
  x_{t-1} = A_t (x^q_t - clamp(e_hat)) + c_t eps^q(x^q_t)
  e_hat_{t-1} = A_t e_hat_t + alpha * c_t * q_hat[t]  (q_hat spatially averaged per channel by default)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def ddim_step_coefficients(
    at: torch.Tensor,
    at_next: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (A_t, c_t) matching theory Section 6 for eta=0 DDIM."""
    a_t = (at_next / at).sqrt()
    c_t = (1.0 - at_next).sqrt() - a_t * (1.0 - at).sqrt()
    return a_t, c_t


class TrajectoryErrorTracker:
    """Online tracker for cumulative latent trajectory error e_t = x^q_t - x_t."""

    def __init__(
        self,
        q_hat_by_t: Dict[int, torch.Tensor],
        scale: float = 0.05,
        channel_mean_q: bool = True,
        clamp_ratio: float = 0.2,
    ):
        self.q_hat_by_t = {int(k): v.float() for k, v in q_hat_by_t.items()}
        self.scale = float(scale)
        self.channel_mean_q = channel_mean_q
        self.clamp_ratio = float(clamp_ratio)
        self.e_hat: Optional[torch.Tensor] = None

    def reset(self, x: torch.Tensor) -> None:
        self.e_hat = torch.zeros_like(x)

    def correct_input(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy helper; term-① correction applies e_hat to the DDIM state term only."""
        if self.e_hat is None:
            self.reset(x)
        return x - self.e_hat

    def applied_state_correction(self, xt: torch.Tensor) -> torch.Tensor:
        """Correction subtracted in DDIM state term, with optional norm clamp vs ||xt||."""
        if self.e_hat is None:
            raise RuntimeError("call reset() before applied_state_correction")
        corr = self.e_hat
        if self.clamp_ratio > 0:
            corr_norm = corr.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
            x_norm = xt.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp(min=1e-8)
            max_norm = self.clamp_ratio * x_norm
            factor = (max_norm / corr_norm.clamp(min=1e-8)).clamp(max=1.0)
            corr = corr * factor
        return corr

    @property
    def state_correction(self) -> torch.Tensor:
        """Deprecated alias; prefer applied_state_correction(xt)."""
        if self.e_hat is None:
            raise RuntimeError("call reset() before reading state_correction")
        return self.e_hat

    def _lookup_q_hat(self, t_idx: int, ref: torch.Tensor) -> torch.Tensor:
        q = self.q_hat_by_t.get(int(t_idx))
        if q is None:
            return torch.zeros_like(ref)
        q = q.to(device=ref.device, dtype=ref.dtype)
        if q.dim() == 1:
            q = q.view(1, -1, 1, 1)
        if q.shape[0] == 1 and ref.shape[0] > 1:
            q = q.expand(ref.shape[0], -1, -1, -1)
        if q.shape[-2:] == (1, 1) and ref.shape[-2:] != (1, 1):
            q = q.expand(-1, -1, ref.shape[-2], ref.shape[-1])
        if self.channel_mean_q and q.shape[-2] > 1:
            q = q.mean(dim=(-2, -1), keepdim=True).expand_as(ref)
        return q

    def update(self, at: torch.Tensor, at_next: torch.Tensor, t_idx: int) -> None:
        if self.e_hat is None:
            raise RuntimeError("call reset() before update()")
        a_t, c_t = ddim_step_coefficients(at, at_next)
        q = self._lookup_q_hat(t_idx, self.e_hat)
        self.e_hat = a_t * self.e_hat + self.scale * c_t * q


def save_q_hat(path: str, q_hat_by_t: Dict[int, torch.Tensor], meta: Optional[dict] = None) -> None:
    payload = {
        "meta": meta or {},
        "q_hat": {str(int(t)): v.detach().cpu().tolist() for t, v in q_hat_by_t.items()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved trajectory q_hat to %s (%d timesteps)", path, len(q_hat_by_t))


def load_q_hat(path: str) -> Tuple[Dict[int, torch.Tensor], dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    q_hat = {int(k): torch.tensor(v, dtype=torch.float32) for k, v in payload["q_hat"].items()}
    meta = payload.get("meta", {})
    logger.info("Loaded trajectory q_hat from %s (%d timesteps)", path, len(q_hat))
    return q_hat, meta


def build_ddim_seq(num_timesteps: int, sampling_steps: int, skip_type: str = "uniform") -> List[int]:
    if skip_type == "uniform":
        skip = num_timesteps // sampling_steps
        seq = list(range(0, num_timesteps, skip))
    elif skip_type == "quad":
        seq = (
            np.linspace(0, np.sqrt(num_timesteps * 0.8), sampling_steps) ** 2
        )
        seq = [int(s) for s in seq]
    else:
        raise NotImplementedError(f"skip_type={skip_type}")
    # Integer projection of a quadratic grid can repeat early timesteps.
    # These yield i == j identity updates and break transition-wise statistics.
    return list(dict.fromkeys(seq))


@torch.no_grad()
def calibrate_q_hat(
    quant_model: nn.Module,
    betas: torch.Tensor,
    seq: List[int],
    *,
    act_quant: bool = True,
    n_trajectories: int = 32,
    batch_size: int = 8,
    channels: int = 3,
    image_size: int = 32,
    device: torch.device,
    seed: int = 1234,
) -> Dict[int, torch.Tensor]:
    """
    Estimate per-timestep mean q_hat[t] = E[eps^q(x^q_t, t) - eps^fp(x^q_t, t)]
    along uncorrected quantized DDIM trajectories.

    Float vs quant forward uses the same QuantModel with set_quant_state toggled.
    """
    from ddim.functions.denoising import compute_alpha

    quant_model.eval()
    if not hasattr(quant_model, "set_quant_state"):
        raise TypeError("calibrate_q_hat expects a QuantModel (with set_quant_state)")

    accum: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    seq_next = [-1] + list(seq[:-1])

    n_done = 0
    while n_done < n_trajectories:
        cur_b = min(batch_size, n_trajectories - n_done)
        x = torch.randn(cur_b, channels, image_size, image_size, device=device)

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((cur_b,), i, device=device, dtype=torch.float32)
            next_t = torch.full((cur_b,), j, device=device, dtype=torch.float32)
            at = compute_alpha(betas, t.long())
            at_next = compute_alpha(betas, next_t.long())

            quant_model.set_quant_state(weight_quant=True, act_quant=act_quant)
            et_q = quant_model(x, t)
            quant_model.set_quant_state(weight_quant=False, act_quant=False)
            et_fp = quant_model(x, t)
            q = et_q - et_fp

            q_mean = q.mean(dim=0, keepdim=True)
            t_key = int(i)
            if t_key not in accum:
                accum[t_key] = q_mean.cpu()
                counts[t_key] = 1
            else:
                accum[t_key] = accum[t_key] + q_mean.cpu()
                counts[t_key] += 1

            x0 = (x - et_q * (1.0 - at).sqrt()) / at.sqrt()
            c2 = (1.0 - at_next).sqrt()
            x = at_next.sqrt() * x0 + c2 * et_q

        n_done += cur_b

    quant_model.set_quant_state(weight_quant=True, act_quant=act_quant)

    q_hat_by_t = {t: (accum[t] / counts[t]) for t in accum}
    norms = [float(v.pow(2).sum().sqrt()) for v in q_hat_by_t.values()]
    logger.info(
        "Calibrated q_hat over %d trajectories, %d timesteps "
        "(||q_hat|| mean=%.4f, max=%.4f)",
        n_trajectories,
        len(q_hat_by_t),
        sum(norms) / max(len(norms), 1),
        max(norms) if norms else 0.0,
    )
    return q_hat_by_t
