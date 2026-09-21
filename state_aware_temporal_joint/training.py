"""Stage-wise training utilities; batches must be closed-loop ``(x_t, t)`` states."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import torch
import torch.nn as nn

from .framework import StateAwareJointFramework
from .losses import adapter_loss, alignment_loss, corrector_loss, joint_loss


class Stage(str, Enum):
    ADAPTER = "adapter"
    CORRECTOR = "corrector"
    JOINT = "joint"


@dataclass
class LossConfig:
    lambda_mag: float = 1e-3
    lambda_align: float = 0.0
    lambda_adapter: float = 0.1
    lambda_mse: float = 1.0
    lambda_dir: float = 2.0
    lambda_str: float = 0.3


def freeze(module: nn.Module, frozen=True):
    module.eval() if frozen else module.train()
    for p in module.parameters():
        p.requires_grad_(not frozen)


def configure_stage(framework: StateAwareJointFramework, teacher: nn.Module, stage: Stage):
    freeze(teacher, True)
    freeze(framework.quant_unet, True)
    freeze(framework.adapter, stage == Stage.CORRECTOR)
    freeze(framework.corrector_net, stage == Stage.ADAPTER)
    # Quantized UNet stays in eval mode, but gradients must cross it to reach adapter heads.
    framework.quant_unet.eval()


def train_step(
    framework: StateAwareJointFramework,
    teacher: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    stage: Stage,
    loss_cfg: LossConfig = LossConfig(),
    context=None,
):
    """Compute a same-input teacher loss. Caller owns zero_grad/backward/step."""
    with torch.no_grad():
        eps_teacher = teacher(x, t, context)

    eps_adapted = framework.adapted_eps(x, t, context)
    magnitude = framework.adapter.magnitude_loss()

    if stage == Stage.ADAPTER:
        loss, stats = adapter_loss(eps_adapted, eps_teacher, magnitude, loss_cfg.lambda_mag)
        if loss_cfg.lambda_align:
            with torch.no_grad():
                eps_base = framework.unadapted_eps(x, t, context)
            align = alignment_loss(eps_base, eps_adapted, eps_teacher)
            loss = loss + loss_cfg.lambda_align * align
            stats["alignment"] = align.detach()
        return loss, stats

    # Detaching here is essential in stage C: only the old corrector is fine-tuned.
    corr_input = eps_adapted.detach() if stage == Stage.CORRECTOR else eps_adapted
    residual = framework.corrector_net(x, corr_input, t)
    kwargs = dict(lambda_mse=loss_cfg.lambda_mse, lambda_dir=loss_cfg.lambda_dir, lambda_str=loss_cfg.lambda_str)
    if stage == Stage.CORRECTOR:
        return corrector_loss(corr_input, eps_teacher, residual, **kwargs)
    return joint_loss(
        eps_adapted, eps_teacher, residual, magnitude,
        lambda_adapter=loss_cfg.lambda_adapter,
        lambda_mag=loss_cfg.lambda_mag,
        **kwargs,
    )


@torch.no_grad()
def diagnostics(framework, teacher, x, t, context=None) -> Dict[str, float]:
    eps_base = framework.unadapted_eps(x, t, context)
    eps_adapted = framework.adapted_eps(x, t, context)
    eps_teacher = teacher(x, t, context)
    residual = framework.corrector_net(x, eps_adapted, t)
    before = (eps_teacher - eps_base).float().square().flatten(1).sum(1)
    after = (eps_teacher - eps_adapted).float().square().flatten(1).sum(1)
    final = (eps_teacher - eps_adapted - residual).float().square().flatten(1).sum(1)
    contribution = eps_adapted - eps_base
    target = eps_teacher - eps_base
    cos = torch.nn.functional.cosine_similarity(contribution.flatten(1), target.flatten(1)).mean()
    return {
        "mse_before": float(before.mean()),
        "mse_after_adapter": float(after.mean()),
        "mse_final": float(final.mean()),
        "temporal_reduction": float(((before - after) / before.clamp_min(1e-8)).mean()),
        "contribution_alignment": float(cos),
        **{f"block_norm/{k}": v for k, v in framework.adapter.diagnostics().items()},
    }
