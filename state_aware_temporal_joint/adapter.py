from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_key(name: str) -> str:
    return name.replace(".", "__")


@dataclass
class AdapterConfig:
    state_channels: int
    temb_channels: int
    block_channels: Dict[str, int]
    bottleneck: int = 32
    rho_init: float = 0.05
    eps: float = 1e-6

    def to_dict(self):
        return asdict(self)


class StateAwareTemporalAdapter(nn.Module):
    """One shared state/time encoder with a post-temb-projection head per block."""

    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        r = config.bottleneck
        self.state_proj = nn.Linear(2 * config.state_channels, r)
        self.time_proj = nn.Linear(config.temb_channels, r, bias=False)
        self.heads = nn.ModuleDict(
            {module_key(n): nn.Linear(r, c) for n, c in config.block_channels.items()}
        )
        self.rho_logits = nn.ParameterDict()
        rho = min(max(config.rho_init, 1e-5), 1.0 - 1e-5)
        rho_logit = torch.logit(torch.tensor(rho)).item()
        for name in config.block_channels:
            self.rho_logits[module_key(name)] = nn.Parameter(torch.tensor(rho_logit))
        self.reset_parameters()
        self._code = None
        self._residuals: Dict[str, torch.Tensor] = {}

    def reset_parameters(self) -> None:
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def begin_forward(self, h0: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        mean = h0.mean(dim=(2, 3))
        rms = (h0.float().square().mean(dim=(2, 3)) + self.config.eps).sqrt().to(h0.dtype)
        stats = torch.cat((mean, rms), dim=1)
        self._code = F.silu(self.state_proj(stats) + self.time_proj(temb))
        self._residuals = {}
        return self._code

    def residual(self, block_name: str, reference: torch.Tensor) -> torch.Tensor:
        if self._code is None:
            raise RuntimeError("Adapter state is unavailable; conv_in/temb hooks did not run")
        key = module_key(block_name)
        rho = torch.sigmoid(self.rho_logits[key])
        delta = rho * torch.tanh(self.heads[key](self._code))
        delta = delta.to(dtype=reference.dtype)
        self._residuals[block_name] = delta
        return delta

    def magnitude_loss(self) -> torch.Tensor:
        if not self._residuals:
            p = next(self.parameters())
            return p.new_zeros(())
        return torch.stack([x.float().square().mean() for x in self._residuals.values()]).mean()

    def diagnostics(self) -> Mapping[str, float]:
        return {n: float(x.detach().float().norm(dim=1).mean()) for n, x in self._residuals.items()}

    def clear(self) -> None:
        self._code = None
        self._residuals = {}

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return self.parameters()
