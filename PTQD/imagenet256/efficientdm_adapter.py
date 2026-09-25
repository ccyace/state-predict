"""Load EfficientDM (QALoRA int) UNet into an ODE-scale LatentDiffusion for apply_model.

The released ImageNet ckpt (e.g. quantw4a4_20steps_efficientdm.pth) is NOT qdiff-format.
It expects EfficientDM's QuantModel_intnlora + TemporalActivationQuantizer with
``num_steps`` matching the DDIM length used at finetune/downsample time.

Usage (from sample_diffusion_ldm.py)::

    model = load_ldm_fp(...)
    attach_efficientdm(model, ckpt, num_steps=20)

After attach, ``model.apply_model`` / DDIMSampler work as usual (corrector / VSC ok).
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn


DEFAULT_EFFICIENTDM_ROOT = os.environ.get("EFFICIENTDM_HOME", os.environ.get("EFFICIENTDM_ROOT", ""))


def _ensure_efficientdm_on_path(efficientdm_root: str) -> str:
    root = os.path.abspath(efficientdm_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"EfficientDM root not found: {root}")
    # Prefer EfficientDM quant_scripts; keep ODE-scale ldm first (already on path).
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def reset_efficientdm_temporal_steps(qnn: nn.Module) -> None:
    """Reset TALSQ indices to the start of a trajectory (last step index)."""
    for m in qnn.modules():
        if hasattr(m, "current_step") and hasattr(m, "total_steps"):
            m.current_step = int(m.total_steps) - 1


def attach_efficientdm(
    ldm_model: nn.Module,
    efficientdm_ckpt: str,
    *,
    num_steps: int = 20,
    weight_bit: int = 4,
    act_bit: int = 4,
    efficientdm_root: str = "",
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Replace ``ldm_model.model.diffusion_model`` with EfficientDM quantized UNet.

    Parameters
    ----------
    ldm_model:
        Full-precision LatentDiffusion already loaded (cin256).
    efficientdm_ckpt:
        Full EfficientDM ``state_dict`` (includes diffusion schedule + qnn + VAE/cond).
    num_steps:
        TALSQ length; must match ckpt (20 for quantw4a4_20steps_efficientdm.pth).
        Sampling ``-c`` / DDIM steps should equal this value.
    """
    root = (efficientdm_root or DEFAULT_EFFICIENTDM_ROOT or "").strip()
    if not root:
        raise ValueError(
            "EfficientDM root not set; pass efficientdm_root=... or set EFFICIENTDM_HOME"
        )
    _ensure_efficientdm_on_path(root)
    from quant_scripts.quant_model import QuantModel_intnlora
    from quant_scripts.quant_layer import QuantModule_intnlora, SimpleDequantizer

    if device is None:
        device = next(ldm_model.parameters()).device

    dmodel = ldm_model.model.diffusion_model
    wq_params = {"n_bits": weight_bit, "channel_wise": True, "scale_method": "mse"}
    aq_params = {
        "n_bits": act_bit,
        "channel_wise": False,
        "scale_method": "max",
        "leaf_param": True,
    }
    qnn = QuantModel_intnlora(
        model=dmodel,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
        num_steps=num_steps,
    )
    print(
        f"[efficientdm] QuantModel_intnlora W{weight_bit}A{act_bit} "
        f"num_steps={num_steps} special_counts={qnn.special_module_count_list}",
        flush=True,
    )
    qnn.set_first_last_layer_to_8bit()
    qnn.set_quant_state(True, True)
    # SimpleDequantizer allocates CUDA buffers in __init__; move qnn first.
    qnn.to(device)

    for module in qnn.modules():
        if isinstance(module, QuantModule_intnlora) and not module.ignore_reconstruction:
            module.intn_dequantizer = SimpleDequantizer(
                uaq=module.weight_quantizer, weight=module.weight
            ).to(device)

    # Pack placeholder weights so the init forward matches EfficientDM's sample script.
    for module in qnn.modules():
        if isinstance(module, QuantModule_intnlora) and not module.ignore_reconstruction:
            module.weight.data = module.weight.data.byte()
            if module.weight.device != device:
                module.weight.data = module.weight.data.to(device)

    # Dummy forward allocates TemporalActivationQuantizer delta_list / zp_list.
    print("[efficientdm] init forward (allocate TALSQ buffers)...", flush=True)
    with torch.no_grad():
        dummy_x = torch.randn(2, dmodel.in_channels, dmodel.image_size, dmodel.image_size, device=device)
        dummy_t = torch.randint(0, 1000, (2,), device=device)
        dummy_c = torch.randn(2, 1, 512, device=device)
        _ = qnn(dummy_x, dummy_t, dummy_c)

    setattr(ldm_model.model, "diffusion_model", qnn)

    print(f"[efficientdm] loading ckpt: {efficientdm_ckpt}", flush=True)
    ckpt = torch.load(efficientdm_ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt and len(ckpt) < 10:
        ckpt = ckpt["state_dict"]
    missing, unexpected = ldm_model.load_state_dict(ckpt, strict=False)
    print(
        f"[efficientdm] load_state_dict strict=False "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if missing:
        print(f"[efficientdm] missing (first 8): {missing[:8]}", flush=True)
    if unexpected:
        print(f"[efficientdm] unexpected (first 8): {unexpected[:8]}", flush=True)

    # Expose UNet geometry for collect scripts that read qnn.in_channels / image_size.
    if not hasattr(qnn, "in_channels"):
        qnn.in_channels = dmodel.in_channels
    if not hasattr(qnn, "image_size"):
        qnn.image_size = dmodel.image_size

    reset_efficientdm_temporal_steps(qnn)
    ldm_model.to(device)
    ldm_model.eval()
    print(
        "[efficientdm] ready — use DDIM steps == "
        f"{num_steps} (TALSQ length). Call reset_efficientdm_temporal_steps(qnn) "
        "if you change trajectory length mid-process.",
        flush=True,
    )
    return qnn
