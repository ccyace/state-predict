#!/usr/bin/env python3
"""Collect closed-loop (x, eq, ef, t) on LDM cin256 (class-conditional + CFG)."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import trange

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_TAMING = os.path.join(ROOT, "src", "taming-transformers")
if _TAMING not in sys.path:
    sys.path.insert(0, _TAMING)

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from qdiff import QuantModel
from qdiff.utils import resume_cali_model


def load_ldm(config_path: str, ckpt_path: str, device: torch.device):
    config = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    model = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.to(device).eval()
    if hasattr(model, "model_ema"):
        model.model_ema.store(model.model.parameters())
        model.model_ema.copy_to(model.model)
    return model, config


def _snapshot_talsq_steps(qnn):
    """Save EfficientDM TemporalActivationQuantizer indices (no-op if absent)."""
    snaps = []
    for m in qnn.modules():
        if hasattr(m, "current_step") and hasattr(m, "total_steps"):
            snaps.append((m, int(m.current_step)))
    return snaps


def _restore_talsq_steps(snaps) -> None:
    for m, step in snaps:
        m.current_step = step


@torch.no_grad()
def collect(
    fp_model,
    qnn,
    *,
    n_traj: int,
    batch_size: int,
    steps: int,
    eta: float,
    cfg_scale: float,
    device: torch.device,
    out_path: str,
):
    fp_unet = fp_model.model.diffusion_model
    fp_model.model.diffusion_model = qnn
    sampler = DDIMSampler(fp_model)
    sampler.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=False)
    timesteps = sampler.ddim_timesteps

    C = qnn.in_channels
    H = W = qnn.image_size
    xs, eqs, efs, ts, tids, labels = [], [], [], [], [], []
    traj_id = 0
    n_batches = (n_traj + batch_size - 1) // batch_size
    time_range = np.flip(timesteps)
    total = timesteps.shape[0]

    # EfficientDM TALSQ: each UNet forward decrements current_step. Collect does an
    # extra q-forward to log eq before p_sample_ddim — restore indices so the real
    # step only advances once (otherwise 20-step ckpt wraps after ~10 DDIM steps).
    try:
        from PTQD.imagenet256.efficientdm_adapter import reset_efficientdm_temporal_steps
    except Exception:  # pragma: no cover
        reset_efficientdm_temporal_steps = None

    for _ in trange(n_batches, desc="collect LDM traj"):
        b = min(batch_size, n_traj - traj_id)
        if reset_efficientdm_temporal_steps is not None:
            reset_efficientdm_temporal_steps(qnn)
        x = torch.randn(b, C, H, W, device=device)
        classes = torch.randint(0, 1000, (b,), device=device)
        c = fp_model.get_learned_conditioning({fp_model.cond_stage_key: classes})
        uc = fp_model.get_learned_conditioning(
            {fp_model.cond_stage_key: torch.full((b,), 1000, device=device, dtype=classes.dtype)}
        )

        for i, step in enumerate(time_range):
            index = total - i - 1
            t = torch.full((b,), int(step), device=device, dtype=torch.long)

            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([uc, c])

            talsq_snap = _snapshot_talsq_steps(qnn)
            e_cat = fp_model.apply_model(x_in, t_in, c_in)
            eq_u, eq_c = e_cat.chunk(2)
            eq = eq_u + cfg_scale * (eq_c - eq_u)
            _restore_talsq_steps(talsq_snap)

            fp_model.model.diffusion_model = fp_unet
            ef_cat = fp_model.apply_model(x_in, t_in, c_in)
            ef_u, ef_c = ef_cat.chunk(2)
            ef = ef_u + cfg_scale * (ef_c - ef_u)
            fp_model.model.diffusion_model = qnn

            xs.append(x.detach().cpu().half())
            eqs.append(eq.detach().cpu().half())
            efs.append(ef.detach().cpu().half())
            ts.append(t.detach().cpu().float())
            tids.append(torch.arange(traj_id, traj_id + b))
            labels.append(classes.detach().cpu())

            outs = sampler.p_sample_ddim(
                x,
                c,
                t,
                index=index,
                unconditional_guidance_scale=cfg_scale,
                unconditional_conditioning=uc,
            )
            x = outs[0]

        traj_id += b

    fp_model.model.diffusion_model = fp_unet
    payload = {
        "x": torch.cat(xs, 0),
        "eq": torch.cat(eqs, 0),
        "ef": torch.cat(efs, 0),
        "t": torch.cat(ts, 0),
        "traj_id": torch.cat(tids, 0).long(),
        "class_label": torch.cat(labels, 0).long(),
        "meta": {
            "n_traj": n_traj,
            "steps": steps,
            "eta": eta,
            "cfg_scale": cfg_scale,
            "shape": [C, H, W],
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    torch.save(payload, out_path)
    print(f"saved {out_path} n={payload['x'].shape[0]}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp_ckpt", default="models/ldm/cin256/model.ckpt")
    p.add_argument("--ldm_config", default="configs/latent-diffusion/cin256-v2.yaml")
    p.add_argument("--cali_ckpt", default="imagenet_w8a8_ckpt.pth")
    p.add_argument("--n_traj", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--scale", type=float, default=3.0)
    p.add_argument("--weight_bit", type=int, default=8)
    p.add_argument("--act_bit", type=int, default=8)
    qa = p.add_mutually_exclusive_group()
    qa.add_argument("--quant_act", dest="quant_act", action="store_true")
    qa.add_argument("--no_quant_act", dest="quant_act", action="store_false")
    p.set_defaults(quant_act=True)
    p.add_argument("--a_sym", action="store_true", default=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--efficientdm_ckpt",
        type=str,
        default="",
        help="If set, load EfficientDM W4A4 (etc.) instead of qdiff --cali_ckpt",
    )
    p.add_argument("--efficientdm_steps", type=int, default=20)
    p.add_argument("--efficientdm_root", type=str, default="", help="EfficientDM checkout (or set EFFICIENTDM_HOME)")
    p.add_argument("--efficientdm_weight_bit", type=int, default=4)
    p.add_argument("--efficientdm_act_bit", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp_model, config = load_ldm(args.ldm_config, args.fp_ckpt, device)

    if args.efficientdm_ckpt:
        from PTQD.imagenet256.efficientdm_adapter import attach_efficientdm

        q_host, _ = load_ldm(args.ldm_config, args.fp_ckpt, device)
        attach_efficientdm(
            q_host,
            args.efficientdm_ckpt,
            num_steps=int(args.efficientdm_steps),
            weight_bit=int(args.efficientdm_weight_bit),
            act_bit=int(args.efficientdm_act_bit),
            efficientdm_root=args.efficientdm_root,
            device=device,
        )
        qnn = q_host.model.diffusion_model
        fp_teacher, _ = load_ldm(args.ldm_config, args.fp_ckpt, device)
        collect(
            fp_teacher,
            qnn,
            n_traj=args.n_traj,
            batch_size=args.batch_size,
            steps=args.steps,
            eta=args.eta,
            cfg_scale=args.scale,
            device=device,
            out_path=args.output,
        )
        return

    wq = {"n_bits": args.weight_bit, "channel_wise": True, "scale_method": "max"}
    aq = {
        "n_bits": args.act_bit,
        "symmetric": args.a_sym,
        "channel_wise": False,
        "scale_method": "max",
        "leaf_param": args.quant_act,
    }
    qnn = QuantModel(model=fp_model.model.diffusion_model, weight_quant_params=wq, act_quant_params=aq)
    qnn.set_first_last_layer_to_8bit()
    qnn.disable_network_output_quantization()
    qnn.cuda().eval()
    image_size = config.model.params.image_size
    channels = config.model.params.channels
    cali_data = (
        torch.randn(1, channels, image_size, image_size),
        torch.randint(0, 1000, (1,)),
        torch.randn(1, 1, 512),
    )
    resume_cali_model(qnn, args.cali_ckpt, cali_data, args.quant_act, "qdiff", cond=True)
    qnn.set_quant_state(True, args.quant_act)

    fp_teacher, _ = load_ldm(args.ldm_config, args.fp_ckpt, device)
    collect(
        fp_teacher,
        qnn,
        n_traj=args.n_traj,
        batch_size=args.batch_size,
        steps=args.steps,
        eta=args.eta,
        cfg_scale=args.scale,
        device=device,
        out_path=args.output,
    )


if __name__ == "__main__":
    main()
