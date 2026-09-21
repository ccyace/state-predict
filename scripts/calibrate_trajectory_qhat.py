"""
Offline calibration of per-timestep q_hat for trajectory state-error correction.

q_hat[t] ≈ E[ eps^q(x^q_t, t) - eps^fp(x^q_t, t) ] along uncorrected quantized DDIM paths.
"""

import argparse
import gc
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
from pytorch_lightning import seed_everything

from ddim.functions.ckpt_util import get_ckpt_path, load_ddim_state_dict
from ddim.models.diffusion import Model
from qdiff import QuantModel
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.joint_sa_opt import JointSAConfig, load_joint_sa_for_sampling
from qdiff.ode_pre_scaling import (
    absorb_ode_weights_into_float_model,
    attach_ode_input_scales_to_quant_model,
    attach_ode_output_scales_to_quant_model,
    attach_resblock_shortcut_scales,
    load_ode_absorb_mode,
    load_ode_absorbed_layers,
    load_ode_scales,
    load_ode_shortcut_inv_scales,
)
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.trajectory_error import build_ddim_seq, calibrate_q_hat, save_q_hat
from qdiff.utils import get_train_samples, resume_cali_model
from sample_diffusion_ddim import get_beta_schedule

logger = logging.getLogger(__name__)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_float_model(config, device, args):
    import time

    t0 = time.perf_counter()
    print("  [load 1/4] building float UNet ...", flush=True)
    model = Model(config)
    ckpt = getattr(args, "ckpt", None)
    if ckpt:
        print(f"  [load 2/4] loading weights from --ckpt: {ckpt}", flush=True)
        state_dict = load_ddim_state_dict(ckpt, map_location=device, prefer_ema=True)
        model.load_state_dict(state_dict, strict=True)
    else:
        if config.data.dataset == "CIFAR10":
            name = "cifar10"
        elif config.data.dataset == "LSUN":
            name = f"lsun_{config.data.category}"
        elif config.data.dataset == "CELEBA":
            raise ValueError("CELEBA requires --ckpt (e.g. celeba_64/ckpt.pth)")
        else:
            raise ValueError(config.data.dataset)
        ckpt = get_ckpt_path(f"ema_{name}")
        print(f"  [load 2/4] loading EMA weights: {ckpt}", flush=True)
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()
    print(f"  [load 2/4] float teacher ready ({time.perf_counter() - t0:.1f}s)", flush=True)

    if args.ode_scale_json:
        deploy = load_ode_scales(args.ode_scale_json)
        absorb_mode = args.ode_absorb_mode or load_ode_absorb_mode(args.ode_scale_json)
        absorb_ode_weights_into_float_model(model, deploy, mode=absorb_mode or "dilate")
        logger.info("Applied ODE weight absorption to float teacher (mode=%s)", absorb_mode)
    return model


def load_quant_model(config, device, args, float_model):
    import time

    if args.joint_sa_resume:
        if not args.cali_ckpt or not args.ode_scale_json or not args.brecq_ckpt:
            raise ValueError("joint_sa_resume requires --cali_ckpt, --ode_scale_json, --brecq_ckpt")
        cfg = JointSAConfig(
            weight_bit=args.weight_bit,
            act_bit=args.act_bit,
            a_sym=args.a_sym,
        )
        qnn = load_joint_sa_for_sampling(
            config, device, args.brecq_ckpt, args.cali_ckpt, args.ode_scale_json, cfg
        )
        return qnn

    t0 = time.perf_counter()
    print("  [load 3/4] wrapping QuantModel ...", flush=True)
    wq_params = {"n_bits": args.weight_bit, "channel_wise": True, "scale_method": "max"}
    aq_params = {
        "n_bits": args.act_bit,
        "symmetric": args.a_sym,
        "channel_wise": False,
        "scale_method": "max",
        "leaf_param": args.quant_act,
    }
    qnn = QuantModel(
        model=float_model,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
        sm_abit=args.sm_abit,
    )
    qnn.to(device)
    qnn.eval()

    if args.ode_scale_json:
        deploy = load_ode_scales(args.ode_scale_json)
        absorb_mode = args.ode_absorb_mode or load_ode_absorb_mode(args.ode_scale_json)
        absorbed = load_ode_absorbed_layers(args.ode_scale_json) or set()
        shortcut_inv = load_ode_shortcut_inv_scales(args.ode_scale_json)
        if absorb_mode == "dilate":
            attach_ode_input_scales_to_quant_model(qnn, absorbed_layers=absorbed)
        else:
            attach_ode_output_scales_to_quant_model(qnn, deploy, absorbed_layers=absorbed)
        attach_resblock_shortcut_scales(qnn, shortcut_inv=shortcut_inv)

    if not args.cali_ckpt:
        raise ValueError("--cali_ckpt is required for quantized model loading")
    print(f"  [load 4/4] cali data: {args.cali_data_path}", flush=True)
    t_cali = time.perf_counter()
    sample_data = torch.load(args.cali_data_path, map_location="cpu")
    print(f"           cali file loaded ({time.perf_counter() - t_cali:.1f}s)", flush=True)
    cali_data = get_train_samples(args, sample_data, custom_steps=0)
    del sample_data
    gc.collect()
    print(f"           PTQ resume: {args.cali_ckpt} (re-init weight/act quant, ~1-3 min) ...", flush=True)
    resume_cali_model(qnn, args.cali_ckpt, cali_data, args.quant_act, "qdiff", cond=False)
    qnn.set_quant_state(weight_quant=True, act_quant=args.quant_act)
    qnn.eval()
    print(f"  [load 4/4] quant model ready ({time.perf_counter() - t0:.1f}s total)", flush=True)
    return qnn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="float UNet checkpoint path (required for CELEBA)",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--skip_type", type=str, default="uniform", choices=["uniform", "quad"])
    parser.add_argument("--n_trajectories", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cali_ckpt", type=str, default="")
    parser.add_argument("--brecq_ckpt", type=str, default="")
    parser.add_argument("--joint_sa_resume", action="store_true")
    parser.add_argument("--ode_scale_json", type=str, default="")
    parser.add_argument("--ode_absorb_mode", type=str, default="")
    parser.add_argument("--cali_data_path", type=str, default="cifar_sd1236_sample2048_allst.pt")
    parser.add_argument("--cali_st", type=int, default=10,
                        help="timesteps subsampling for loading cali ckpt (match PTQ run)")
    parser.add_argument("--cali_n", type=int, default=256,
                        help="samples per timestep when loading cali ckpt")
    parser.add_argument("--split", action="store_true",
                        help="split shortcut (must match PTQ ckpt)")
    parser.add_argument("--weight_bit", type=int, default=8)
    parser.add_argument("--act_bit", type=int, default=8)
    parser.add_argument("--quant_act", action="store_true")
    parser.add_argument("--a_sym", action="store_true")
    parser.add_argument("--sm_abit", type=int, default=8)
    args = parser.parse_args()
    args.cond = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r", encoding="utf-8") as f:
        config = dict2namespace(yaml.safe_load(f))
    config.split_shortcut = args.split

    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device)
    num_timesteps = betas.shape[0]
    seq = build_ddim_seq(num_timesteps, args.timesteps, args.skip_type)

    float_model = load_float_model(config, device, args)
    quant_model = load_quant_model(config, device, args, float_model)
    del float_model

    q_hat = calibrate_q_hat(
        quant_model,
        betas,
        seq,
        act_quant=args.quant_act,
        n_trajectories=args.n_trajectories,
        batch_size=args.batch_size,
        channels=config.data.channels,
        image_size=config.data.image_size,
        device=device,
        seed=args.seed,
    )

    save_q_hat(
        args.output_json,
        q_hat,
        meta={
            "source": "trajectory_qhat_calibration",
            "seq": seq,
            "n_trajectories": args.n_trajectories,
            "timesteps": args.timesteps,
            "skip_type": args.skip_type,
            "channels": config.data.channels,
            "cali_ckpt": args.cali_ckpt,
            "ode_scale_json": args.ode_scale_json,
        },
    )
    logger.info("Done. Enable with --enable_state_error_corr --trajectory_qhat_json %s", args.output_json)


if __name__ == "__main__":
    main()
