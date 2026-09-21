#!/usr/bin/env python
"""Post-training W8A8 calibration for U-ViT-S/2 CIFAR (Q-Diffusion style, no split-shortcut)."""
from __future__ import annotations

import argparse
import gc
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.layer_recon import layer_reconstruction
from qdiff.quant_layer import QuantModule, UniformAffineQuantizer
from qdiff.quant_model import QuantModel
from qdiff.utils import get_train_samples, resume_cali_model, sync_org_weights_to_device
from uvit_loader import DEFAULT_CKPT, UViTEpsModel, build_uvit

logger = logging.getLogger(__name__)


def _recon_layers(qnn: QuantModel, *, cali_data, batch_size: int, iters: int, act_quant: bool, lr: float = 4e-4, p: float = 2.4):
    kwargs = dict(
        cali_data=cali_data,
        batch_size=batch_size,
        iters=iters,
        weight=0.01,
        asym=True,
        b_range=(20, 2),
        warmup=0.2,
        act_quant=act_quant,
        opt_mode="mse",
        lr=lr,
        p=p,
    )

    def _walk(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, QuantModule):
                logger.info("Reconstruction for layer %s", full)
                layer_reconstruction(qnn, child, **kwargs)
            else:
                _walk(child, full)

    torch.set_grad_enabled(True)
    _walk(qnn.model)
    torch.set_grad_enabled(False)


def calibrate_w8a8(args) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_ckpt)), exist_ok=True)

    nnet = build_uvit()
    state = torch.load(args.ckpt, map_location="cpu")
    nnet.load_state_dict(state)
    nnet.eval()
    wrapper = UViTEpsModel(nnet)

    wq_params = {"n_bits": args.weight_bit, "channel_wise": True, "scale_method": "max"}
    aq_params = {
        "n_bits": args.act_bit,
        "symmetric": args.a_sym,
        "channel_wise": False,
        "scale_method": "max",
        "leaf_param": True,
    }
    qnn = QuantModel(model=wrapper, weight_quant_params=wq_params, act_quant_params=aq_params, sm_abit=args.sm_abit)
    qnn.to(device)
    qnn.eval()

    logger.info("Loading calibration data from %s", args.cali_data_path)
    sample_data = torch.load(args.cali_data_path, map_location="cpu")
    cali_data = get_train_samples(args, sample_data, custom_steps=0)
    del sample_data
    gc.collect()
    cali_xs, cali_ts = cali_data
    logger.info("Calibration tensors: xs=%s ts=%s", tuple(cali_xs.shape), tuple(cali_ts.shape))

    if args.resume and os.path.isfile(args.output_ckpt):
        logger.info("Resume existing ckpt: %s", args.output_ckpt)
        resume_cali_model(qnn, args.output_ckpt, cali_data, quant_act=args.quant_act, cond=False)
        qnn.set_quant_state(True, args.quant_act)
        return args.output_ckpt

    logger.info("Initialize weight quant scales")
    qnn.set_quant_state(True, False)
    with torch.no_grad():
        _ = qnn(cali_xs[:8].to(device), cali_ts[:8].to(device))

    if not args.skip_weight_recon:
        logger.info("Weight BRECQ (iters=%d)", args.cali_iters)
        sync_org_weights_to_device(qnn, device)
        _recon_layers(qnn, cali_data=cali_data, batch_size=args.cali_batch_size, iters=args.cali_iters, act_quant=False)
        qnn.set_quant_state(True, False)

    if args.quant_act:
        logger.info("Activation calibration")
        qnn.set_quant_state(True, True)
        with torch.no_grad():
            inds = np.random.choice(cali_xs.shape[0], min(64, cali_xs.shape[0]), replace=False)
            _ = qnn(cali_xs[inds].to(device), cali_ts[inds].to(device))
            if args.running_stat:
                qnn.set_running_stat(True)
                bs = 64
                for i in range(int(cali_xs.size(0) / bs)):
                    _ = qnn(
                        cali_xs[i * bs : (i + 1) * bs].to(device),
                        cali_ts[i * bs : (i + 1) * bs].to(device),
                    )
                qnn.set_running_stat(False)

        if not args.skip_act_recon:
            logger.info("Activation BRECQ (iters=%d)", args.cali_iters_a)
            sync_org_weights_to_device(qnn, device)
            _recon_layers(
                qnn,
                cali_data=cali_data,
                batch_size=args.cali_batch_size,
                iters=args.cali_iters_a,
                act_quant=True,
                lr=args.cali_lr,
                p=args.cali_p,
            )
        qnn.set_quant_state(True, True)

    for m in qnn.model.modules():
        if isinstance(m, AdaRoundQuantizer):
            m.zero_point = nn.Parameter(m.zero_point)
            m.delta = nn.Parameter(m.delta)
        elif isinstance(m, UniformAffineQuantizer):
            m.delta = nn.Parameter(m.delta)
            if m.zero_point is not None:
                if not torch.is_tensor(m.zero_point):
                    m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                else:
                    m.zero_point = nn.Parameter(m.zero_point.float())

    torch.save(qnn.state_dict(), args.output_ckpt)
    logger.info("Saved %s", args.output_ckpt)
    return args.output_ckpt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--output_ckpt", default="uvit_experiments/checkpoints/uvit_w8a8_ckpt.pth")
    p.add_argument("--cali_data_path", default="cifar_sd1236_sample2048_allst.pt")
    p.add_argument("--cali_st", type=int, default=10)
    p.add_argument("--cali_n", type=int, default=256)
    p.add_argument("--custom_steps", type=int, default=0)
    p.add_argument("--cali_batch_size", type=int, default=32)
    p.add_argument("--cali_iters", type=int, default=512, help="weight BRECQ iters per layer")
    p.add_argument("--cali_iters_a", type=int, default=256, help="activation BRECQ iters per layer")
    p.add_argument("--cali_lr", type=float, default=4e-4)
    p.add_argument("--cali_p", type=float, default=2.4)
    p.add_argument("--weight_bit", type=int, default=8)
    p.add_argument("--act_bit", type=int, default=8)
    p.add_argument("--sm_abit", type=int, default=8)
    p.add_argument("--quant_act", action="store_true", default=True)
    p.add_argument("--a_sym", action="store_true", default=True)
    p.add_argument("--running_stat", action="store_true", default=True)
    p.add_argument("--skip_weight_recon", action="store_true")
    p.add_argument("--skip_act_recon", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cond", action="store_true", default=False)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    calibrate_w8a8(args)


if __name__ == "__main__":
    main()
