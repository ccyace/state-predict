"""
Collect late-t trajectory training data for learned noise corrector (MVP).

On uncorrected quantized DDIM paths, store (x^q_t, eps^q, eps^fp, t, traj_id) for t < t_max.

  python noise_eps_corr/scripts/collect_late_training_data.py ...
"""

import argparse
import gc
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch
import yaml
from pytorch_lightning import seed_everything

from ddim.functions.denoising import compute_alpha
from qdiff.trajectory_error import build_ddim_seq
from calibrate_trajectory_qhat import dict2namespace, load_float_model, load_quant_model
from sample_diffusion_ddim import get_beta_schedule


@torch.no_grad()
def collect_late_training_data(
    quant_model,
    betas,
    seq,
    device,
    num_trajectories,
    batch_size,
    t_max,
    eta=0.0,
    channels=3,
    image_size=32,
):
    quant_model.eval()
    betas_t = torch.tensor(betas, dtype=torch.float32, device=device)
    seq_next = [-1] + list(seq[:-1])
    steps_order = list(zip(reversed(seq), reversed(seq_next)))
    late_ts = {int(i) for i, _j in steps_order if int(i) < t_max}

    xs, eqs, efs, ts, traj_ids = [], [], [], [], []
    rng = torch.Generator(device=device)
    rng.manual_seed(1234)
    log_every = max(64, batch_size)
    n_done = 0

    while n_done < num_trajectories:
        cur_b = min(batch_size, num_trajectories - n_done)
        x = torch.randn(
            cur_b, channels, image_size, image_size, device=device, generator=rng
        )

        for i, j in steps_order:
            t_int = int(i)
            t = torch.full((cur_b,), float(i), device=device)
            next_t = torch.full((cur_b,), float(j), device=device)
            at = compute_alpha(betas_t, t.long())
            at_next = compute_alpha(betas_t, next_t.long())

            quant_model.set_quant_state(weight_quant=True, act_quant=True)
            eq = quant_model(x, t)
            quant_model.set_quant_state(weight_quant=False, act_quant=False)
            ef = quant_model(x, t)
            quant_model.set_quant_state(weight_quant=True, act_quant=True)

            if t_int in late_ts:
                ids = torch.arange(n_done, n_done + cur_b, dtype=torch.int32)
                xs.append(x.detach().cpu().half())
                eqs.append(eq.detach().cpu().half())
                efs.append(ef.detach().cpu().half())
                ts.append(torch.full((cur_b,), t_int, dtype=torch.int16))
                traj_ids.append(ids)

            x0 = (x - eq * (1 - at).sqrt()) / at.sqrt()
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            x = at_next.sqrt() * x0 + c2 * eq
            if eta > 0:
                x = x + c1 * torch.randn_like(x)

        n_done += cur_b
        if n_done % log_every == 0 or n_done >= num_trajectories:
            print(f"  trajectories {n_done}/{num_trajectories}", flush=True)

    return {
        "x": torch.cat(xs, dim=0),
        "eq": torch.cat(eqs, dim=0),
        "ef": torch.cat(efs, dim=0),
        "t": torch.cat(ts, dim=0).long(),
        "traj_id": torch.cat(traj_ids, dim=0).long(),
        "meta": {
            "num_trajectories": num_trajectories,
            "t_max": t_max,
            "n_samples": int(torch.cat(ts, dim=0).shape[0]),
        },
    }


def main():
    p = argparse.ArgumentParser(description="Collect late-t noise correction training data")
    p.add_argument("--config", type=str, default="configs/cifar10.yml")
    p.add_argument("--ckpt", type=str, default="", help="float UNet ckpt (required for CELEBA)")
    p.add_argument("--cali_ckpt", type=str, default="cifar_w8a8_ckpt.pth")
    p.add_argument("--cali_data_path", type=str, default="cifar_sd1236_sample2048_allst.pt")
    p.add_argument(
        "--ode_scale_json",
        type=str,
        default="",
        help="empty for official Q-Diffusion ckpt; set ode_pre_scaling.json for ODE-dilate PTQ",
    )
    p.add_argument("--ode_absorb_mode", type=str, default="")
    p.add_argument("--brecq_ckpt", type=str, default="")
    p.add_argument("--num_trajectories", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--t_max", type=int, default=50, help="save steps with diffusion t < t_max")
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--skip_type", type=str, default="quad")
    p.add_argument("--output", type=str, default="output/noise_corr/train_data/traj_5k_late_cifar_w8a8.pt")
    p.add_argument("--split", action="store_true", default=True)
    p.add_argument("--quant_act", action="store_true", default=True)
    p.add_argument("--a_sym", action="store_true", default=True)
    p.add_argument("--weight_bit", type=int, default=8)
    p.add_argument("--act_bit", type=int, default=8)
    p.add_argument("--sm_abit", type=int, default=8)
    p.add_argument("--cali_st", type=int, default=10)
    p.add_argument("--cali_n", type=int, default=256)
    args = p.parse_args()
    args.cond = False
    args.joint_sa_resume = False

    seed_everything(1234)
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
    seq = build_ddim_seq(len(betas), args.timesteps, args.skip_type)

    print(f"Loading models on {device} ...", flush=True)
    float_model = load_float_model(config, device, args)
    quant_model = load_quant_model(config, device, args, float_model)
    del float_model
    gc.collect()

    print(
        f"Collecting {args.num_trajectories} trajectories, t<{args.t_max} ...",
        flush=True,
    )
    data = collect_late_training_data(
        quant_model, betas, seq, device,
        num_trajectories=args.num_trajectories,
        batch_size=args.batch_size,
        t_max=args.t_max,
        channels=int(config.data.channels),
        image_size=int(config.data.image_size),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    torch.save(data, args.output)
    print(f"Saved {data['meta']['n_samples']} samples -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
