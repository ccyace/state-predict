"""Shared quant-model loading for U-ViT vsc_tvar / dt pipeline scripts."""
from __future__ import annotations

import gc
from typing import Any

import torch

from uvit_loader import DEFAULT_CKPT, load_uvit_quant


def add_backbone_args(parser: Any) -> None:
    parser.add_argument(
        "--backbone",
        choices=("unet", "uvit"),
        default="unet",
        help="diffusion backbone; uvit uses cifar10_uvit_small.pth + uvit_w8a8 ckpt",
    )
    parser.add_argument(
        "--fp_ckpt",
        default=DEFAULT_CKPT,
        help="float U-ViT weights (backbone=uvit only)",
    )


def load_quant_from_args(args: Any, device: torch.device, config: Any | None = None):
    """Return a calibrated QuantModel on device (eval mode)."""
    backbone = getattr(args, "backbone", "unet")
    if backbone == "uvit":
        qnn = load_uvit_quant(
            cali_ckpt=args.cali_ckpt,
            cali_data_path=args.cali_data_path,
            device=device,
            fp_ckpt=getattr(args, "fp_ckpt", None) or DEFAULT_CKPT,
            weight_bit=int(args.weight_bit),
            act_bit=int(args.act_bit),
            sm_abit=int(getattr(args, "sm_abit", 8)),
            cali_st=int(args.cali_st),
            cali_n=int(args.cali_n),
            quant_act=bool(getattr(args, "quant_act", True)),
            a_sym=bool(getattr(args, "a_sym", True)),
        )
        qnn.set_quant_state(True, bool(getattr(args, "quant_act", True)))
        qnn.eval()
        return qnn

    if config is None:
        raise ValueError("config is required when backbone=unet")
    from calibrate_trajectory_qhat import load_float_model, load_quant_model

    fp = load_float_model(config, device, args)
    qnn = load_quant_model(config, device, args, fp)
    del fp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    qnn.set_quant_state(True, bool(getattr(args, "quant_act", True)))
    qnn.eval()
    return qnn
