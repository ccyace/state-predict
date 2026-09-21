"""Constrained temporal-path interventions for diagnosis and restoration."""
from __future__ import annotations

import torch.nn as nn

from qdiff.quant_block import QuantResnetBlock


def temporal_quant_modules(qnn: nn.Module):
    """Return the two time-MLP layers and every ResNet time projection."""
    model = getattr(qnn, "model", qnn)
    modules = list(model.temb.dense)
    modules.extend(
        block.temb_proj
        for block in model.modules()
        if isinstance(block, QuantResnetBlock)
    )
    return modules


def keep_temporal_path_float(qnn: nn.Module) -> int:
    """Disable simulated quantization only on the temporal path (B6)."""
    modules = temporal_quant_modules(qnn)
    for module in modules:
        if not hasattr(module, "set_quant_state"):
            raise TypeError(f"Temporal module is not quantized: {type(module).__name__}")
        module.set_quant_state(weight_quant=False, act_quant=False)
    return len(modules)
