"""DDIM / DDPM denoising helpers (minimal restore for sampling + data collection)."""

from __future__ import annotations

import torch


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1, device=beta.device, dtype=beta.dtype), beta], dim=0)
    return (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)


@torch.no_grad()
def generalized_steps(x, seq, model, b, **kwargs):
    """Standard DDIM loop with optional learned noise corrector."""
    eta = float(kwargs.get("eta", 0.0))
    noise_corrector = kwargs.get("noise_corrector")
    device = x.device
    betas = b.to(device)

    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    xs = [x]
    x0_preds = []

    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = torch.full((n,), float(i), device=device)
        next_t = torch.full((n,), float(j), device=device)
        xt = xs[-1].to(device)

        et = model(xt, t)
        if noise_corrector is not None:
            at = compute_alpha(betas, t.long())
            et = noise_corrector.correct(et, float(i), at, xt=xt)

        at = compute_alpha(betas, t.long())
        at_next = compute_alpha(betas, next_t.long())
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        x0_preds.append(x0_t.detach().cpu())

        c1 = eta * ((1 - at / at_next.clamp(min=at)) * (1 - at_next) / (1 - at).clamp(min=1e-12)).sqrt()
        c2 = ((1 - at_next) - c1 ** 2).clamp(min=0.0).sqrt()
        if eta > 0.0:
            noise = torch.randn_like(xt)
            xt_next = at_next.sqrt() * x0_t + c1 * noise + c2 * et
        else:
            xt_next = at_next.sqrt() * x0_t + c2 * et
        xs.append(xt_next.detach().cpu())

    return xs, x0_preds


@torch.no_grad()
def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to(x.device)

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to("cpu"))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to("cpu"))
        return xs, x0_preds
