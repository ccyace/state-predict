#!/usr/bin/env python3
"""
Stage A: offline calibration of DDIM denoise-formula-aware pre-quantization scales.

Usage (CIFAR-10 example):
  python scripts/compute_ode_pre_scaling.py \\
    --config configs/cifar10.yml \\
    --cali_data_path <path_to_cali.pt> \\
    --cali_st 10 --cali_n 256 \\
    --output_json ode_pre_scaling.json

Then run PTQ with the scaled float model:
  python scripts/sample_diffusion_ddim.py \\
    --config configs/cifar10.yml --ptq --quant_act \\
    --ode_scale_json ode_pre_scaling.json ...
"""

import argparse
import gc
import json
import logging
import os
import sys

import torch
import yaml
from pytorch_lightning import seed_everything

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddim.functions.ckpt_util import get_ckpt_path
from ddim.models.diffusion import Model
from qdiff.ode_pre_scaling import (
    absorb_ode_weights_into_float_model,
    apply_ode_pre_scaling,
    attach_ode_input_scales_to_quant_model,
    attach_ode_output_scales_to_quant_model,
    attach_resblock_shortcut_scales,
    calibrate_ode_pre_scaling,
    discover_strict_shortcut_edges,
    load_ode_scales,
    save_ode_scales,
)
from qdiff import QuantModel
from qdiff.utils import get_train_samples

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    import numpy as np

    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "quad":
        betas = (
            np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64)
            ** 2
        )
    else:
        raise NotImplementedError(beta_schedule)
    return betas


def dict2namespace(config):
    import argparse as ap

    namespace = ap.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_model(config, device):
    model = Model(config)
    if config.data.dataset == "CIFAR10":
        name = "cifar10"
    elif config.data.dataset == "LSUN":
        name = f"lsun_{config.data.category}"
    else:
        raise ValueError(config.data.dataset)
    ckpt = get_ckpt_path(f"ema_{name}")
    logger.info("Loading checkpoint %s", ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()
    return model


def build_quant_wrapper(model, device):
    """QuantModel shell for --split forward (nin_shortcut needs QuantModule.split)."""
    wq_params = {"n_bits": 8, "channel_wise": True, "scale_method": "max"}
    aq_params = {
        "n_bits": 8,
        "symmetric": False,
        "channel_wise": False,
        "scale_method": "max",
        "leaf_param": False,
    }
    qnn = QuantModel(
        model=model,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
    )
    qnn.to(device)
    qnn.eval()
    return qnn


def _write_ode_absorb_meta(payload: dict, model, absorbed, absorb_mode: str) -> dict:
    payload.setdefault("meta", {})
    payload["meta"]["absorbed_layers"] = sorted(absorbed)
    payload["meta"]["absorb_mode"] = absorb_mode
    shortcut_inv = getattr(model, "_ode_shortcut_inv_scales", {})
    payload["meta"]["shortcut_blocks"] = sorted(shortcut_inv.keys())
    payload["meta"]["shortcut_inv_scales"] = {
        k: v.cpu().tolist() for k, v in shortcut_inv.items()
    }
    return payload


def run_float_verify(config, deploy, device, cali_data_path, cali_st, cali_n, split, absorb_mode):
    sample_data = torch.load(cali_data_path, map_location="cpu")
    cali_args = argparse.Namespace(
        cali_n=min(4, cali_n), cali_st=cali_st, cond=False, custom_steps=0,
    )
    cali_xs, cali_ts = get_train_samples(cali_args, sample_data, custom_steps=0)
    x = cali_xs[:4].to(device)
    t = cali_ts[:4].to(device).float()

    ref_model = load_model(config, device)
    if split:
        ref_model = build_quant_wrapper(ref_model, device)
    with torch.no_grad():
        ref = ref_model(x, t)

    scaled_model = load_model(config, device)
    absorbed_v = absorb_ode_weights_into_float_model(
        scaled_model, deploy, mode=absorb_mode
    )
    qnn = build_quant_wrapper(scaled_model, device)
    if absorb_mode == "dilate":
        attach_ode_input_scales_to_quant_model(qnn, absorbed_layers=absorbed_v)
    else:
        attach_ode_output_scales_to_quant_model(qnn, deploy, absorbed_layers=absorbed_v)
    attach_resblock_shortcut_scales(qnn)
    with torch.no_grad():
        scaled = qnn(x, t)
    diff = (ref - scaled).abs().max().item()
    n_short = len(getattr(scaled_model, "_ode_shortcut_inv_scales", {}))
    scale_kind = "input /s" if absorb_mode == "dilate" else "output /s"
    logger.info(
        "Float equivalence [%s] max |Δoutput|: %.6e (%d conv %s, %d shortcut blocks)",
        absorb_mode,
        diff,
        len(absorbed_v),
        scale_kind,
        n_short,
    )
    return diff, absorbed_v


def main():
    parser = argparse.ArgumentParser(description="Calibrate ODE pre-quantization channel scales")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cali_data_path", type=str, required=True)
    parser.add_argument("--cali_st", type=int, default=10, help="number of timesteps in cali set")
    parser.add_argument("--cali_n", type=int, default=256, help="samples per timestep")
    parser.add_argument("--cali_batch_size", type=int, default=8)
    parser.add_argument("--output_json", type=str, default="ode_pre_scaling.json")
    parser.add_argument("--scaled_ckpt", type=str, default="", help="optional: save model state_dict after apply")
    parser.add_argument("--ode_alpha", type=float, default=0.5)
    parser.add_argument("--ode_smin", type=float, default=0.5)
    parser.add_argument("--ode_smax", type=float, default=2.0)
    parser.add_argument("--ode_eps", type=float, default=1e-8)
    parser.add_argument("--max_batches_per_t", type=int, default=None)
    parser.add_argument("--verify", action="store_true", help="verify float output after apply")
    parser.add_argument(
        "--split", action="store_true",
        help="split shortcut connection (must match PTQ / sampling script)",
    )
    parser.add_argument(
        "--absorb_only", action="store_true",
        help="skip calibration; load --output_json and only absorb weights (+ --verify)",
    )
    parser.add_argument(
        "--ode_absorb_mode",
        type=str,
        default="dilate",
        choices=["dilate", "strict", "ptq", "legacy"],
        help="dilate=DilateQuant-style input/s+same-layer W (float-equiv); "
        "strict=shortcuts only; ptq=ResBlock mid (not float-equiv); legacy=heuristic",
    )
    parser.add_argument(
        "--verify_strict",
        action="store_true",
        help="also run --verify with absorb_mode=strict (expect near-zero delta)",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r", encoding="utf-8") as f:
        config = dict2namespace(yaml.safe_load(f))
    config.split_shortcut = args.split

    betas = torch.from_numpy(
        get_beta_schedule(
            config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
    ).float().to(device)

    model = load_model(config, device)

    if args.absorb_only:
        deploy = load_ode_scales(args.output_json)
        logger.info("Loaded %d layers from %s (absorb_only)", len(deploy), args.output_json)
        edges = discover_strict_shortcut_edges(model)
        if edges:
            logger.info("Strict shortcut edges: %s", edges)
        absorbed = absorb_ode_weights_into_float_model(
            model, deploy, mode=args.ode_absorb_mode
        )
        with open(args.output_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload = _write_ode_absorb_meta(payload, model, absorbed, args.ode_absorb_mode)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        if args.scaled_ckpt:
            torch.save(model.state_dict(), args.scaled_ckpt)
            logger.info("Saved weight-absorbed float model to %s", args.scaled_ckpt)
        if args.verify:
            run_float_verify(
                config, deploy, device, args.cali_data_path,
                args.cali_st, args.cali_n, args.split, args.ode_absorb_mode,
            )
        if args.verify_strict and args.ode_absorb_mode != "strict":
            run_float_verify(
                config, deploy, device, args.cali_data_path,
                args.cali_st, args.cali_n, args.split, "strict",
            )
        return

    calib_model = model
    if args.split:
        calib_model = build_quant_wrapper(model, device)
        logger.info("Wrapped float UNet in QuantModel for --split forward")

    cali_args = argparse.Namespace(
        cali_n=args.cali_n,
        cali_st=args.cali_st,
        cond=False,
        custom_steps=0,
    )
    sample_data = torch.load(args.cali_data_path, map_location="cpu")
    cali_xs, cali_ts = get_train_samples(cali_args, sample_data, custom_steps=0)
    del sample_data
    gc.collect()
    logger.info("Calibration data: xs=%s ts=%s", tuple(cali_xs.shape), tuple(cali_ts.shape))

    deploy, m_stats = calibrate_ode_pre_scaling(
        calib_model,
        cali_xs,
        cali_ts,
        betas,
        device,
        batch_size=args.cali_batch_size,
        alpha=args.ode_alpha,
        s_min=args.ode_smin,
        s_max=args.ode_smax,
        eps=args.ode_eps,
        max_batches_per_t=args.max_batches_per_t,
    )

    save_ode_scales(
        args.output_json,
        deploy,
        m_stats=m_stats,
        meta={
            "alpha": args.ode_alpha,
            "s_min": args.ode_smin,
            "s_max": args.ode_smax,
            "cali_st": args.cali_st,
            "cali_n": args.cali_n,
            "split": args.split,
            "absorb_mode": args.ode_absorb_mode,
        },
    )

    absorbed = absorb_ode_weights_into_float_model(
        model, deploy, mode=args.ode_absorb_mode
    )
    # Record which layers have paired (W absorb + output /s) for PTQ
    with open(args.output_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload = _write_ode_absorb_meta(payload, model, absorbed, args.ode_absorb_mode)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if args.verify:
        run_float_verify(
            config, deploy, device, args.cali_data_path,
            args.cali_st, args.cali_n, args.split, args.ode_absorb_mode,
        )
    if args.verify_strict and args.ode_absorb_mode != "strict":
        run_float_verify(
            config, deploy, device, args.cali_data_path,
            args.cali_st, args.cali_n, args.split, "strict",
        )

    if args.scaled_ckpt:
        torch.save(model.state_dict(), args.scaled_ckpt)
        logger.info(
            "Saved weight-absorbed float model to %s "
            "(PTQ still needs --ode_scale_json for QuantModule output /s)",
            args.scaled_ckpt,
        )

    logger.info(
        "Done. PTQ: python scripts/sample_diffusion_ddim.py --ptq "
        "--ode_scale_json %s ...",
        args.output_json,
    )


if __name__ == "__main__":
    main()
