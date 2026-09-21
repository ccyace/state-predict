"""Structured stochastic sampling for post-correction epsilon residuals.

Experiment D replaces part of DDIM's explicit Gaussian state-noise budget by
an epsilon-space residual with calibrated per-channel variance and AR(1)
correlation.  The replacement is variance preserving in state space.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class ResidualStepInfo:
    requested_scale: float
    actual_scale_mean: float
    saturation_ratio: float
    budget_fraction_mean: float


class ChannelARResidualSampler:
    """Sample zero-mean, per-channel AR(1) epsilon residuals."""

    def __init__(
        self,
        stats_path: str,
        *,
        scale: float = 0.5,
        ar_clip: float = 0.95,
        mode: str = "budget",
    ):
        with open(stats_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        variance = payload.get("channel_variance_by_t")
        if variance is None:
            # Accept the early design name as a compatibility alias.
            variance = payload.get("variance_by_t_channel")
        if not variance:
            raise ValueError(
                "Residual stats must contain channel_variance_by_t. "
                "Re-run noise_eps_corr/scripts/estimate_residual_stats.py."
            )
        self.std_by_t: Dict[str, list] = {
            str(key): [max(float(value), 0.0) ** 0.5 for value in values]
            for key, values in variance.items()
        }
        self.rho_by_transition: Dict[str, list] = payload.get(
            "temporal_corr_by_transition", {}
        )
        self.meta = payload.get("meta", {})
        self.scale = float(scale)
        self.ar_clip = float(ar_clip)
        self.mode = str(mode)
        if self.scale < 0.0:
            raise ValueError(f"residual scale must be >= 0, got {self.scale}")
        if not 0.0 <= self.ar_clip < 1.0:
            raise ValueError(f"AR clip must be in [0, 1), got {self.ar_clip}")
        if self.mode not in ("budget", "residual_only"):
            raise ValueError(f"unsupported residual stochastic mode: {self.mode}")
        self._z_prev: Optional[torch.Tensor] = None
        self._previous_t_idx: Optional[int] = None

    def reset(self) -> None:
        self._z_prev = None
        self._previous_t_idx = None

    def validate_grid(self, seq) -> None:
        sampled = [int(value) for value in reversed(list(seq))]
        missing_t = [value for value in sampled if str(value) not in self.std_by_t]
        if missing_t:
            raise ValueError(
                "Residual statistics do not match the DDIM grid; missing timesteps: "
                f"{missing_t[:20]}"
            )
        missing_transitions = [
            f"{left}->{right}"
            for left, right in zip(sampled[:-1], sampled[1:])
            if f"{left}->{right}" not in self.rho_by_transition
        ]
        if missing_transitions:
            raise ValueError(
                "Residual statistics are missing adjacent-grid correlations: "
                f"{missing_transitions[:20]}"
            )

    @staticmethod
    def _channel_tensor(values, reference: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(values, device=reference.device, dtype=reference.dtype)
        if value.numel() != reference.shape[1]:
            raise ValueError(
                f"residual stats have {value.numel()} channels, "
                f"but current sample has {reference.shape[1]}"
            )
        return value.view(1, -1, 1, 1)

    def _std(self, t_idx: int, reference: torch.Tensor) -> torch.Tensor:
        key = str(int(t_idx))
        if key not in self.std_by_t:
            raise KeyError(
                f"No channel residual statistics for timestep {key}; "
                "the stats and sampling grids must match"
            )
        return self._channel_tensor(self.std_by_t[key], reference)

    def _rho(self, t_idx: int, next_t_idx: int, reference: torch.Tensor) -> torch.Tensor:
        key = f"{int(t_idx)}->{int(next_t_idx)}"
        values = self.rho_by_transition.get(key)
        if values is None:
            values = [0.0] * reference.shape[1]
        rho = self._channel_tensor(values, reference)
        return rho.clamp(min=-self.ar_clip, max=self.ar_clip)

    def _sample_ar(self, reference: torch.Tensor, t_idx: int) -> torch.Tensor:
        innovation = torch.randn_like(reference)
        if self._z_prev is None or self._z_prev.shape != reference.shape:
            z_ar = innovation
        else:
            rho = self._rho(self._previous_t_idx, t_idx, reference)
            innovation_scale = (1.0 - rho.square()).clamp_min(0.0).sqrt()
            z_ar = rho * self._z_prev + innovation_scale * innovation
        self._z_prev = z_ar
        self._previous_t_idx = int(t_idx)
        return z_ar

    @torch.no_grad()
    def sample_unbudgeted(
        self,
        reference: torch.Tensor,
        *,
        t_idx: int,
        next_t_idx: int,
    ) -> torch.Tensor:
        """Sample a pure structured epsilon residual without DDIM white noise."""
        del next_t_idx
        std = self._std(t_idx, reference)
        return self.scale * std * self._sample_ar(reference, t_idx)

    @torch.no_grad()
    def sample_budgeted(
        self,
        reference: torch.Tensor,
        *,
        t_idx: int,
        next_t_idx: int,
        eps_to_state_coeff: torch.Tensor,
        ddim_sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, ResidualStepInfo]:
        """Return ``(delta_eps, sigma_remaining, diagnostics)``.

        Per channel, the construction enforces

        ``(B * actual_scale * residual_std)^2 + sigma_remaining^2 = sigma^2``.
        """
        std = self._std(t_idx, reference)
        z_ar = self._sample_ar(reference, t_idx)

        requested = torch.full_like(std, self.scale)
        max_scale = ddim_sigma.abs() / (eps_to_state_coeff.abs() * std + 1e-12)
        actual = torch.minimum(requested, max_scale)
        delta_eps = actual * std * z_ar

        state_residual_var = (
            eps_to_state_coeff.square() * actual.square() * std.square()
        )
        sigma_remaining = (
            ddim_sigma.square() - state_residual_var
        ).clamp_min(0.0).sqrt()

        saturated = (actual + 1e-7 < requested).float()
        budget_fraction = state_residual_var / (ddim_sigma.square() + 1e-12)
        info = ResidualStepInfo(
            requested_scale=self.scale,
            actual_scale_mean=float(actual.mean().item()),
            saturation_ratio=float(saturated.mean().item()),
            budget_fraction_mean=float(budget_fraction.mean().item()),
        )
        return delta_eps, sigma_remaining, info


def load_channel_ar_residual_sampler(
    stats_path: str,
    *,
    scale: float = 0.5,
    ar_clip: float = 0.95,
    mode: str = "budget",
) -> ChannelARResidualSampler:
    return ChannelARResidualSampler(
        stats_path, scale=scale, ar_clip=ar_clip, mode=mode
    )
