"""Load U-ViT-S/2 CIFAR checkpoint for DDIM sampling."""
from __future__ import annotations

import os
import sys
from typing import Any, Dict

import torch
import torch.nn as nn

UVIT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party", "U-ViT"))
DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "cifar10_uvit_small.pth")

CIFAR10_UVIT_SMALL: Dict[str, Any] = dict(
    name="uvit",
    img_size=32,
    patch_size=2,
    embed_dim=512,
    depth=12,
    num_heads=8,
    mlp_ratio=4,
    qkv_bias=False,
    mlp_time_embed=False,
    num_classes=-1,
)


class UViTEpsModel(nn.Module):
    """Wrapper: forward(x, t) -> eps, matching qdiff QuantModel / sampling interface."""

    def __init__(self, nnet: nn.Module):
        super().__init__()
        self.nnet = nnet
        self.in_channels = 3

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        del context
        return self.nnet(x, t)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        """No-op for FP model; keeps collect scripts compatible."""
        del weight_quant, act_quant


def _ensure_uvit_path() -> None:
    if UVIT_ROOT not in sys.path:
        sys.path.insert(0, UVIT_ROOT)


def build_uvit(nnet_kwargs: Dict[str, Any] | None = None) -> nn.Module:
    _ensure_uvit_path()
    from utils import get_nnet  # type: ignore  # noqa: WPS433

    kwargs = dict(CIFAR10_UVIT_SMALL)
    if nnet_kwargs:
        kwargs.update(nnet_kwargs)
    return get_nnet(**kwargs)


def load_uvit_fp(ckpt_path: str = DEFAULT_CKPT, device: torch.device | str = "cuda") -> UViTEpsModel:
    ckpt_path = os.path.abspath(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"U-ViT checkpoint not found: {ckpt_path}\n"
            "Run: bash uvit_experiments/download_ckpt.sh\n"
            "Or manually place cifar10_uvit_small.pth there (~120MB)."
        )
    nnet = build_uvit()
    state = torch.load(ckpt_path, map_location="cpu")
    nnet.load_state_dict(state)
    nnet.eval()
    return UViTEpsModel(nnet).to(device)


def load_uvit_quant(
    cali_ckpt: str,
    cali_data_path: str,
    device: torch.device | str = "cuda",
    *,
    fp_ckpt: str = DEFAULT_CKPT,
    weight_bit: int = 8,
    act_bit: int = 8,
    sm_abit: int = 8,
    cali_st: int = 10,
    cali_n: int = 256,
    quant_act: bool = True,
    a_sym: bool = True,
) -> "QuantModel":
    """Load calibrated U-ViT QuantModel (Q-Diffusion style)."""
    from argparse import Namespace

    from qdiff.quant_model import QuantModel
    from qdiff.utils import get_train_samples, resume_cali_model

    nnet = build_uvit()
    state = torch.load(os.path.abspath(fp_ckpt), map_location="cpu")
    nnet.load_state_dict(state)
    nnet.eval()
    wrapper = UViTEpsModel(nnet)

    wq_params = {"n_bits": weight_bit, "channel_wise": True, "scale_method": "max"}
    aq_params = {
        "n_bits": act_bit,
        "symmetric": a_sym,
        "channel_wise": False,
        "scale_method": "max",
        "leaf_param": quant_act,
    }
    qnn = QuantModel(model=wrapper, weight_quant_params=wq_params, act_quant_params=aq_params, sm_abit=sm_abit)
    qnn.to(device)
    qnn.eval()

    args = Namespace(
        cali_st=cali_st,
        cali_n=cali_n,
        custom_steps=0,
        cond=False,
    )
    sample_data = torch.load(cali_data_path, map_location="cpu")
    cali_data = get_train_samples(args, sample_data, custom_steps=0)
    resume_cali_model(qnn, cali_ckpt, cali_data, quant_act=quant_act, cond=False)
    qnn.set_quant_state(True, quant_act)
    qnn.eval()
    return qnn
