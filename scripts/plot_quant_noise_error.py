"""
Compare per-step noise MSE and latent trajectory L2 deviation between
full-precision and quantized (W8A8) CIFAR-10 DDIM denoising.

X-axis: diffusion timestep from T to 0 (high noise -> low noise).
No ODE pre-scaling — raw FP checkpoint vs calibrated quantized checkpoint.
"""

import argparse
import json
import logging
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from pytorch_lightning import seed_everything
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ddim.functions.ckpt_util import get_ckpt_path
from ddim.functions.denoising import compute_alpha
from ddim.models.diffusion import Model
from qdiff import QuantModel
import torch.nn as nn
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.utils import convert_adaround

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


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    else:
        raise NotImplementedError(beta_schedule)
    return betas


def build_ddim_seq(num_timesteps, sampling_steps, skip_type):
    if skip_type == "uniform":
        skip = num_timesteps // sampling_steps
        seq = list(range(0, num_timesteps, skip))
    elif skip_type == "quad":
        seq = (
            np.linspace(0, np.sqrt(num_timesteps * 0.8), sampling_steps) ** 2
        )
        seq = [int(s) for s in list(seq)]
    else:
        raise NotImplementedError(skip_type)
    return seq


def load_fp_model(config, device):
    fp_config = argparse.Namespace(**vars(config))
    fp_config.split_shortcut = False
    model = Model(fp_config)
    ckpt = get_ckpt_path("ema_cifar10")
    logger.info("Loading FP checkpoint: %s", ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()
    return model


def resume_cali_model_compat(qnn, ckpt_path, cali_data, quant_act=False, cond=False):
    """Load a legacy Q-Diffusion ckpt (no ODE buffers) into this repo's QuantModel."""
    logger.info("Loading quantized checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    device = next(qnn.parameters()).device

    qnn.set_quant_state(True, False)
    if not cond:
        cali_xs, cali_ts = cali_data
        _ = qnn(cali_xs[:1].to(device), cali_ts[:1].to(device))
    else:
        cali_xs, cali_ts, cali_cs = cali_data
        _ = qnn(cali_xs[:1].to(device), cali_ts[:1].to(device), cali_cs[:1].to(device))
    convert_adaround(qnn)

    for m in qnn.model.modules():
        if isinstance(m, AdaRoundQuantizer):
            m.zero_point = nn.Parameter(m.zero_point)
            m.delta = nn.Parameter(m.delta)

    weight_ckpt = {k: v for k, v in ckpt.items() if "act" not in k}
    qnn.load_state_dict(weight_ckpt, strict=False)
    qnn.set_quant_state(weight_quant=True, act_quant=False)

    for m in qnn.model.modules():
        if isinstance(m, AdaRoundQuantizer):
            zero_data = m.zero_point.data
            delattr(m, "zero_point")
            m.zero_point = zero_data
            delta_data = m.delta.data
            delattr(m, "delta")
            m.delta = delta_data

    if not quant_act:
        return

    qnn.set_quant_state(True, True)
    if not cond:
        _ = qnn(cali_xs[:1].to(device), cali_ts[:1].to(device))
    else:
        _ = qnn(cali_xs[:1].to(device), cali_ts[:1].to(device), cali_cs[:1].to(device))

    for m in qnn.model.modules():
        if isinstance(m, AdaRoundQuantizer):
            m.zero_point = nn.Parameter(m.zero_point)
            m.delta = nn.Parameter(m.delta)
        elif isinstance(m, UniformAffineQuantizer):
            if m.zero_point is not None:
                if not torch.is_tensor(m.zero_point):
                    m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                else:
                    m.zero_point = nn.Parameter(m.zero_point)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    qnn.load_state_dict(ckpt, strict=False)
    qnn.set_quant_state(weight_quant=True, act_quant=True)

    for m in qnn.model.modules():
        if isinstance(m, AdaRoundQuantizer):
            zero_data = m.zero_point.data
            delattr(m, "zero_point")
            m.zero_point = zero_data
            delta_data = m.delta.data
            delattr(m, "delta")
            m.delta = delta_data
        elif isinstance(m, UniformAffineQuantizer):
            if m.zero_point is not None:
                zero_data = m.zero_point.item()
                delattr(m, "zero_point")
                assert int(zero_data) == zero_data
                m.zero_point = int(zero_data)


def load_quant_model(config, device, cali_ckpt, weight_bit=8, act_bit=8):
    q_config = argparse.Namespace(**vars(config))
    q_config.split_shortcut = True
    base = Model(q_config)
    ckpt = get_ckpt_path("ema_cifar10")
    base.load_state_dict(torch.load(ckpt, map_location=device))

    wq_params = {
        "n_bits": weight_bit,
        "channel_wise": True,
        "scale_method": "max",
    }
    aq_params = {
        "n_bits": act_bit,
        "symmetric": True,
        "channel_wise": False,
        "scale_method": "max",
        "leaf_param": True,
    }
    qnn = QuantModel(
        model=base,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
        sm_abit=8,
    )
    qnn.to(device)
    qnn.eval()

    image_size = config.data.image_size
    channels = config.data.channels
    cali_data = (
        torch.randn(1, channels, image_size, image_size),
        torch.randint(0, 1000, (1,)),
    )
    resume_cali_model_compat(qnn, cali_ckpt, cali_data, quant_act=True, cond=False)
    qnn.set_quant_state(weight_quant=True, act_quant=True)
    return qnn


@torch.no_grad()
def measure_noise_error_over_steps(x_init, seq, fp_model, q_model, betas, eta=0.0, show_progress=True):
    """
    Run paired FP / quantized DDIM denoising and record:
      - step_noise_mse: MSE(eps_q, eps_fp) at the same x_t (FP trajectory)
      - traj_l2: L2 mean deviation between FP and quantized latents after each step
    """
    device = x_init.device
    n = x_init.size(0)
    seq_next = [-1] + list(seq[:-1])

    x_fp = x_init.clone()
    x_q = x_init.clone()

    timesteps = []
    step_noise_mse = []
    traj_l2 = []

    for i, j in tqdm(
        list(zip(reversed(seq), reversed(seq_next))),
        desc="Denoise steps",
        leave=False,
        disable=not show_progress,
    ):
        t = torch.full((n,), i, device=device, dtype=torch.float)
        next_t = torch.full((n,), j, device=device, dtype=torch.float)
        at = compute_alpha(betas, t.long())
        at_next = compute_alpha(betas, next_t.long())

        et_fp = fp_model(x_fp, t)
        et_q_same = q_model(x_fp, t)
        mse = (et_q_same - et_fp).pow(2).mean(dim=(1, 2, 3)).mean().item()
        step_noise_mse.append(mse)

        x0_fp = (x_fp - et_fp * (1 - at).sqrt()) / at.sqrt()
        et_q = q_model(x_q, t)
        x0_q = (x_q - et_q * (1 - at).sqrt()) / at.sqrt()

        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        x_fp = at_next.sqrt() * x0_fp + c2 * et_fp
        x_q = at_next.sqrt() * x0_q + c2 * et_q

        l2 = (x_fp - x_q).pow(2).mean(dim=(1, 2, 3)).sqrt().mean().item()
        traj_l2.append(l2)
        timesteps.append(int(i))

    return {
        "timesteps": timesteps,
        "step_noise_mse": step_noise_mse,
        "traj_l2": traj_l2,
    }


def aggregate_results(results_list):
    timesteps = results_list[0]["timesteps"]
    mse = np.array([r["step_noise_mse"] for r in results_list])
    traj = np.array([r["traj_l2"] for r in results_list])
    return {
        "timesteps": timesteps,
        "step_noise_mse_mean": mse.mean(axis=0),
        "step_noise_mse_std": mse.std(axis=0),
        "traj_l2_mean": traj.mean(axis=0),
        "traj_l2_std": traj.std(axis=0),
    }


def plot_error_curves(agg, save_path, quant_label="W8A8"):
    timesteps = agg["timesteps"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    legend_label = f"{quant_label} vs FP"

    ax0 = axes[0]
    ax0.plot(timesteps, agg["step_noise_mse_mean"], color="#1f77b4", linewidth=1.8, label=legend_label)
    ax0.set_ylabel("MSE($\\epsilon_q$, $\\epsilon_{fp}$)")
    ax0.set_title(f"Per-step noise MSE ({quant_label})")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right")

    ax1 = axes[1]
    ax1.plot(timesteps, agg["traj_l2_mean"], color="#d62728", linewidth=1.8, label=legend_label)
    ax1.set_xlabel("Diffusion timestep $t$ ($T \\rightarrow 0$)")
    ax1.set_ylabel("$\\|x_q - x_{fp}\\|_2$ (per-sample mean)")
    ax1.set_title(f"Latent trajectory L2 deviation ({quant_label})")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax0.invert_xaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    logger.info("Saved figure to %s", save_path)
    plt.close(fig)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Plot quantized vs FP noise error over DDIM denoising steps (CIFAR-10)."
    )
    parser.add_argument("--config", type=str, default="configs/cifar10.yml")
    parser.add_argument(
        "--cali_ckpt",
        type=str,
        default="cifar_w8a8_ckpt.pth",
        help="Calibrated W8A8 quantized checkpoint (e.g. cifar_w8a8_ckpt.pth)",
    )
    parser.add_argument("--weight_bit", type=int, default=8)
    parser.add_argument("--act_bit", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=100, help="DDIM sampling steps")
    parser.add_argument("--skip_type", type=str, default="quad", choices=["uniform", "quad"])
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_seeds", type=int, default=16, help="Number of random x_T to average")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per seed run")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/quant_noise_error_w8a8.png",
        help="Output figure path",
    )
    parser.add_argument(
        "--json_output",
        type=str,
        default="outputs/quant_noise_error_w8a8.json",
        help="Output JSON with per-step statistics",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
        force=True,
    )

    print("=" * 60, flush=True)
    print("Quant vs FP noise error plot - starting", flush=True)
    print("=" * 60, flush=True)

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(ROOT, config_path)
    with open(config_path, "r") as f:
        config = dict2namespace(yaml.safe_load(f))

    cali_ckpt = args.cali_ckpt
    if not os.path.isabs(cali_ckpt):
        for candidate in (
            os.path.join(ROOT, cali_ckpt),
            os.path.join(ROOT, "checkpoints", cali_ckpt),
        ):
            if os.path.isfile(candidate):
                cali_ckpt = candidate
                break
        else:
            cali_ckpt = os.path.join(ROOT, cali_ckpt)
    if not os.path.isfile(cali_ckpt):
        raise FileNotFoundError(
            f"Quantized checkpoint not found: {cali_ckpt}\n"
            "Download from Q-Diffusion Google Drive and pass --cali_ckpt <path>."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    betas_np = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas_np).float().to(device)
    num_timesteps = betas.shape[0]
    seq = build_ddim_seq(num_timesteps, args.timesteps, args.skip_type)
    logger.info("DDIM seq: %d steps, skip_type=%s, eta=%.2f", len(seq), args.skip_type, args.eta)

    fp_model = load_fp_model(config, device)
    logger.info("FP model loaded.")
    q_model = load_quant_model(
        config, device, cali_ckpt, weight_bit=args.weight_bit, act_bit=args.act_bit
    )
    logger.info("Quantized W8A8 model loaded.")

    channels = config.data.channels
    image_size = config.data.image_size
    all_results = []

    total_forwards = args.num_seeds * len(seq) * 3
    logger.info(
        "Measurement: %d seeds x %d steps x 3 forwards ≈ %d UNet calls",
        args.num_seeds,
        len(seq),
        total_forwards,
    )
    logger.info("Estimated time: ~%.0f–%.0f min on GPU (depends on GPU speed)", total_forwards / 1200, total_forwards / 400)

    t0 = time.time()
    for seed_idx in tqdm(range(args.num_seeds), desc="Seeds", unit="seed"):
        seed = args.seed + seed_idx
        seed_everything(seed)
        x_init = torch.randn(
            args.batch_size,
            channels,
            image_size,
            image_size,
            device=device,
        )
        result = measure_noise_error_over_steps(
            x_init, seq, fp_model, q_model, betas, eta=args.eta, show_progress=False
        )
        all_results.append(result)
        elapsed = time.time() - t0
        eta_left = elapsed / (seed_idx + 1) * (args.num_seeds - seed_idx - 1)
        logger.info(
            "Seed %d/%d done (%.0fs elapsed, ~%.0fs left) | final traj L2=%.6f | mean step MSE=%.6f",
            seed_idx + 1,
            args.num_seeds,
            elapsed,
            eta_left,
            result["traj_l2"][-1],
            np.mean(result["step_noise_mse"]),
        )

    agg = aggregate_results(all_results)

    out_png = args.output
    out_json = args.json_output
    if not os.path.isabs(out_png):
        out_png = os.path.join(ROOT, out_png)
    if not os.path.isabs(out_json):
        out_json = os.path.join(ROOT, out_json)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    json_payload = {
        "config": args.config,
        "cali_ckpt": cali_ckpt,
        "timesteps": args.timesteps,
        "skip_type": args.skip_type,
        "eta": args.eta,
        "num_seeds": args.num_seeds,
        "batch_size": args.batch_size,
        "seq": seq,
        **{k: v.tolist() if hasattr(v, "tolist") else v for k, v in agg.items()},
    }
    with open(out_json, "w") as f:
        json.dump(json_payload, f, indent=2)
    logger.info("Saved metrics to %s", out_json)

    quant_label = f"W{args.weight_bit}A{args.act_bit}"
    plot_error_curves(agg, out_png, quant_label=quant_label)
    logger.info("All done.")


if __name__ == "__main__":
    main()
