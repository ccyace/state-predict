"""Small state-supervised correction head for Teacher-2 DDIM defects."""

from __future__ import annotations

import torch
import torch.nn as nn


class Teacher2DiscreteNet(nn.Module):
    """(x, corrected eps, sigma*z, t, eta, h, sigma) -> delta eps."""

    def __init__(self, channels: int = 3, hidden: int = 32, max_t: float = 1000.0):
        super().__init__()
        self.channels = channels
        self.hidden = hidden
        self.max_t = max_t
        self.encoder = nn.Sequential(
            nn.Conv2d(3 * channels, hidden, 3, padding=1),
            nn.GroupNorm(4, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(4, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(4, hidden),
            nn.SiLU(),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        self.pre_head = nn.Sequential(nn.Conv2d(hidden + 64, hidden, 1), nn.SiLU())
        self.out = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, eps_corr, u, t, eta, h, sigma):
        batch = x.shape[0]
        feat = self.encoder(torch.cat((x, eps_corr, u), dim=1))
        cond = torch.stack(
            (
                t.float() / self.max_t,
                eta.float(),
                h.float() / self.max_t,
                sigma.float(),
            ),
            dim=1,
        )
        emb = self.cond_mlp(cond).view(batch, 64, 1, 1)
        emb = emb.expand(-1, -1, feat.shape[-2], feat.shape[-1])
        return self.out(self.pre_head(torch.cat((feat, emb), dim=1)))


def batch_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    af, bf = a.flatten(1).float(), b.flatten(1).float()
    return (af * bf).sum(1) / (af.norm(dim=1) * bf.norm(dim=1) + 1e-12)


class Teacher2OnlineCorrector:
    def __init__(self, net: Teacher2DiscreteNet, meta: dict):
        self.net = net
        self.meta = meta

    @torch.no_grad()
    def predict(self, x, eps_corr, u, t, eta, h, sigma):
        batch = x.shape[0]
        device = x.device
        def full(value):
            return torch.full((batch,), float(value), device=device)
        return self.net(x, eps_corr, u, full(t), full(eta), full(h), full(sigma))


def load_teacher2_corrector(path: str, device: torch.device) -> Teacher2OnlineCorrector:
    payload = torch.load(path, map_location=device)
    if payload.get("kind") != "teacher2_discrete":
        raise ValueError(f"not a Teacher-2 discrete checkpoint: {path}")
    net = Teacher2DiscreteNet().to(device)
    net.load_state_dict(payload["state_dict"], strict=True)
    net.eval()
    return Teacher2OnlineCorrector(net, dict(payload.get("meta", {})))
