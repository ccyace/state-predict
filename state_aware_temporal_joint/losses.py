from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _cos(a, b, eps=1e-8):
    a, b = a.flatten(1), b.flatten(1)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(eps)


def adapter_loss(eps_adapted, eps_teacher, magnitude, lambda_mag=1e-3):
    mse = F.mse_loss(eps_adapted, eps_teacher)
    loss = mse + lambda_mag * magnitude
    return loss, {"adapter_mse": mse.detach(), "adapter_mag": magnitude.detach()}


def alignment_loss(eps_unadapted, eps_adapted, eps_teacher):
    contribution = eps_adapted - eps_unadapted
    target = eps_teacher - eps_unadapted
    return (1.0 - _cos(contribution, target)).mean()


def corrector_loss(eps_adapted, eps_teacher, residual, lambda_mse=1.0, lambda_dir=2.0, lambda_str=0.3):
    target = eps_teacher - eps_adapted
    corrected = eps_adapted + residual
    mse = F.mse_loss(residual, target)
    direction = (1.0 - _cos(corrected, eps_teacher)).mean()
    strength = (corrected.flatten(1).norm(dim=1) / eps_teacher.flatten(1).norm(dim=1).clamp_min(1e-8) - 1).abs().mean()
    loss = lambda_mse * mse + lambda_dir * direction + lambda_str * strength
    return loss, {"residual_mse": mse.detach(), "direction": direction.detach(), "strength": strength.detach()}


def joint_loss(eps_adapted, eps_teacher, residual, magnitude, lambda_adapter=0.1, lambda_mag=1e-3, **kwargs):
    corr, stats = corrector_loss(eps_adapted, eps_teacher, residual, **kwargs)
    mse = F.mse_loss(eps_adapted, eps_teacher)
    total = corr + lambda_adapter * mse + lambda_mag * magnitude
    stats.update(adapter_mse=mse.detach(), adapter_mag=magnitude.detach())
    return total, stats
