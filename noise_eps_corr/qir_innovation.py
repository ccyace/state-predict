"""Calibration utilities for Quantization-Innovation Restart (QIR).

The decomposition is deliberately causal in reverse-sampling order.  At step
``t`` it removes a per-timestep/channel bias and the component predictable from
the residual observed at the preceding (higher-noise) DDIM step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional

import torch


def _channel_view(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return value.to(reference.device, reference.dtype).view(1, -1, 1, 1)


@dataclass
class InnovationCalibration:
    mean_by_t: Dict[int, torch.Tensor]
    std_by_t: Dict[int, torch.Tensor]
    rho_by_transition: Dict[str, torch.Tensor]

    def innovation(
        self,
        error: torch.Tensor,
        t: int,
        *,
        previous_residual: Optional[torch.Tensor] = None,
        previous_t: Optional[int] = None,
    ):
        mean = _channel_view(self.mean_by_t[int(t)], error)
        residual = error - mean
        if previous_residual is None or previous_t is None:
            return residual, residual

        key = f"{int(previous_t)}->{int(t)}"
        if key not in self.rho_by_transition:
            return residual, residual
        rho = _channel_view(self.rho_by_transition[key], error).clamp(-0.99, 0.99)
        cur_std = _channel_view(self.std_by_t[int(t)], error)
        prev_std = _channel_view(self.std_by_t[int(previous_t)], error)
        predictable = rho * (cur_std / (prev_std + 1e-12)) * previous_residual
        return residual - predictable, residual

    @classmethod
    def load(cls, path: str) -> "InnovationCalibration":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            mean_by_t={int(k): torch.tensor(v) for k, v in payload["mean_by_t_channel"].items()},
            std_by_t={int(k): torch.tensor(v) for k, v in payload["std_by_t_channel"].items()},
            rho_by_transition={
                k: torch.tensor(v) for k, v in payload["rho_by_transition_channel"].items()
            },
        )


def score_orthogonal(value: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    """Remove the per-sample projection of ``value`` onto the score tensor."""
    dims = tuple(range(1, value.ndim))
    coeff = (value * score).sum(dims, keepdim=True) / (
        score.square().sum(dims, keepdim=True) + 1e-12
    )
    return value - coeff * score


def rms(value: torch.Tensor) -> torch.Tensor:
    return value.square().mean(tuple(range(1, value.ndim)), keepdim=True).sqrt()


def match_rms(value: torch.Tensor, target_rms: torch.Tensor) -> torch.Tensor:
    return value * (target_rms / (rms(value) + 1e-12))
