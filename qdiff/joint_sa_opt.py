"""
Joint fine-tuning of ODE deploy scales (s) and activation quantizers (A_q)
after BRECQ, with weight quantizers frozen.

Stage 2 of the BRECQ -> JointSA -> optional refresh pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ddim.functions.ckpt_util import get_ckpt_path
from ddim.models.diffusion import Model
from qdiff import QuantModel
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.layer_recon import layer_reconstruction
from qdiff.block_recon import block_reconstruction
from qdiff.quant_block import BaseQuantBlock, QuantResnetBlock
from qdiff.quant_layer import QuantModule, UniformAffineQuantizer
from qdiff.utils import get_train_samples, resume_cali_model
from qdiff.ode_pre_scaling import (
    absorb_ode_weights_into_float_model,
    attach_ode_input_scales_to_quant_model,
    attach_resblock_shortcut_scales,
    export_learned_scales,
    list_ode_calibration_modules,
    load_ode_absorb_mode,
    load_ode_absorbed_layers,
    load_ode_scales,
    load_ode_shortcut_inv_scales,
    resolve_dilate_input_scale,
)

logger = logging.getLogger(__name__)


def _sanitize_key(name: str) -> str:
    return name.replace(".", "__")


def _desanitize_key(key: str) -> str:
    return key.replace("__", ".")


@dataclass
class JointSAConfig:
    joint_steps: int = 2000
    act_refresh_iters: int = 2000
    batch_size: int = 32
    lr_s: float = 3e-3
    lr_act: float = 1e-4
    lambda_reg: float = 0.05
    lambda_uni: float = 0.005
    uni_start_step: int = 500
    s_min: float = 0.5
    s_max: float = 2.0
    log_delta_clip: float = 0.5
    holdout_ratio: float = 0.1
    early_stop_patience: int = 100
    log_every: int = 50
    eval_every: int = 100
    seed: int = 1234
    weight_bit: int = 8
    act_bit: int = 8
    a_sym: bool = True
    split: bool = True
    cali_st: int = 10
    cali_n: int = 256
    cali_lr: float = 4e-4
    cali_p: float = 2.4
    freeze_act: bool = False


@dataclass
class CaliSplit:
    train_xs: torch.Tensor
    train_ts: torch.Tensor
    hold_xs: torch.Tensor
    hold_ts: torch.Tensor
    timestep_buckets: List[int] = field(default_factory=list)


def _deploy_inv_for_quant_input(
    name: str,
    mod: QuantModule,
    deploy: Dict[str, torch.Tensor],
) -> Optional[torch.Tensor]:
    """Match joint-training forward: inv = (1/s_layer)[:in_ch], not resolve_dilate."""
    s = deploy.get(name)
    if s is None:
        return None
    in_ch = mod.weight.shape[1]
    return (1.0 / s.float().clamp(min=1e-8))[:in_ch]


class LearnableScaleManager(nn.Module):
    """Learn log-delta offsets on top of closed-form s0 (absorbed layers only)."""

    def __init__(
        self,
        s0: Dict[str, torch.Tensor],
        absorbed_layers: Set[str],
        shortcut_inv0: Optional[Dict[str, torch.Tensor]] = None,
        s_min: float = 0.5,
        s_max: float = 2.0,
        log_delta_clip: float = 0.5,
    ):
        super().__init__()
        self.s_min = s_min
        self.s_max = s_max
        self.log_delta_clip = log_delta_clip
        self.layer_names: List[str] = []
        self.shortcut_block_names: List[str] = []

        deltas = {}
        for name in sorted(absorbed_layers):
            if name not in s0:
                continue
            s_ref = s0[name].float().flatten()
            deltas[_sanitize_key(name)] = nn.Parameter(torch.zeros_like(s_ref))
            self.layer_names.append(name)
        self.layer_log_delta = nn.ParameterDict(deltas)

        shortcut_inv0 = shortcut_inv0 or {}
        sdelts = {}
        for block_name, inv in shortcut_inv0.items():
            inv = inv.float().flatten()
            s_ref = (1.0 / inv.clamp(min=1e-8)).clamp(min=s_min, max=s_max)
            sdelts[_sanitize_key(block_name)] = nn.Parameter(torch.zeros_like(s_ref))
            self.shortcut_block_names.append(block_name)
            self.register_buffer(
                f"shortcut_s0_{_sanitize_key(block_name)}",
                s_ref.clone(),
                persistent=False,
            )
        self.shortcut_log_delta = nn.ParameterDict(sdelts)

        for name in self.layer_names:
            self.register_buffer(
                f"s0_{_sanitize_key(name)}",
                s0[name].float().flatten().clone(),
                persistent=False,
            )

    def _layer_s0(self, name: str) -> torch.Tensor:
        return getattr(self, f"s0_{_sanitize_key(name)}")

    def _shortcut_s0(self, block_name: str) -> torch.Tensor:
        return getattr(self, f"shortcut_s0_{_sanitize_key(block_name)}")

    def compute_layer_s(self, name: str) -> torch.Tensor:
        delta = self.layer_log_delta[_sanitize_key(name)].clamp(
            -self.log_delta_clip, self.log_delta_clip
        )
        s = self._layer_s0(name) * torch.exp(delta)
        return s.clamp(min=self.s_min, max=self.s_max)

    def compute_shortcut_inv(self, block_name: str) -> torch.Tensor:
        delta = self.shortcut_log_delta[_sanitize_key(block_name)].clamp(
            -self.log_delta_clip, self.log_delta_clip
        )
        s = self._shortcut_s0(block_name) * torch.exp(delta)
        s = s.clamp(min=self.s_min, max=self.s_max)
        return 1.0 / s.clamp(min=1e-8)

    def deploy_scales(self) -> Dict[str, torch.Tensor]:
        return {name: self.compute_layer_s(name).detach().cpu() for name in self.layer_names}

    def shortcut_inv_scales(self) -> Dict[str, torch.Tensor]:
        return {
            name: self.compute_shortcut_inv(name).detach().cpu()
            for name in self.shortcut_block_names
        }

    def reg_loss(self) -> torch.Tensor:
        terms = [p.pow(2).mean() for p in self.layer_log_delta.values()]
        terms += [p.pow(2).mean() for p in self.shortcut_log_delta.values()]
        if not terms:
            return torch.tensor(0.0)
        return torch.stack(terms).mean()

    def apply_live_scales(self, qnn: nn.Module) -> None:
        """Set differentiable /s buffers used during joint training forward."""
        qmods = {
            n: m
            for n, m in qnn.model.named_modules()
            if isinstance(m, QuantModule) and m.fwd_func is F.conv2d
        }
        for name in self.layer_names:
            if name not in qmods:
                continue
            inv = 1.0 / self.compute_layer_s(name).clamp(min=1e-8)
            qmods[name]._ode_input_inv_scale_live = inv

        blocks = {
            n: m for n, m in qnn.model.named_modules() if isinstance(m, QuantResnetBlock)
        }
        for block_name in self.shortcut_block_names:
            if block_name not in blocks:
                continue
            blocks[block_name]._ode_shortcut_inv_scale_live = self.compute_shortcut_inv(
                block_name
            )

    def clear_live_scales(self, qnn: nn.Module) -> None:
        for m in qnn.model.modules():
            if isinstance(m, QuantModule):
                m._ode_input_inv_scale_live = None
            elif isinstance(m, QuantResnetBlock):
                m._ode_shortcut_inv_scale_live = None

    def sync_static_scales(self, qnn: nn.Module) -> None:
        """Write final inv scales to registered buffers (post joint training)."""
        qmods = {
            n: m
            for n, m in qnn.model.named_modules()
            if isinstance(m, QuantModule) and m.fwd_func is F.conv2d
        }
        deploy = {name: self.compute_layer_s(name) for name in self.layer_names}
        for name in self.layer_names:
            if name not in qmods:
                continue
            inv = _deploy_inv_for_quant_input(name, qmods[name], deploy)
            if inv is None:
                continue
            qmods[name].set_ode_input_inv_scale(inv.detach())
            qmods[name]._ode_input_inv_scale_live = None

        blocks = {
            n: m for n, m in qnn.model.named_modules() if isinstance(m, QuantResnetBlock)
        }
        for block_name in self.shortcut_block_names:
            if block_name not in blocks:
                continue
            inv = self.compute_shortcut_inv(block_name).detach()
            blocks[block_name].set_ode_shortcut_inv_scale(inv)
            blocks[block_name]._ode_shortcut_inv_scale_live = None


def _clone_namespace(ns) -> argparse.Namespace:
    """Deep-copy argparse.Namespace (config tree)."""
    out = argparse.Namespace()
    for key, val in vars(ns).items():
        if isinstance(val, argparse.Namespace):
            setattr(out, key, _clone_namespace(val))
        else:
            setattr(out, key, val)
    return out


def load_fp_teacher(config, device: torch.device) -> Model:
    """Unquantized FP UNet for epsilon targets (no split-shortcut path)."""
    teacher_config = _clone_namespace(config)
    teacher_config.split_shortcut = False
    model = Model(teacher_config)
    if config.data.dataset == "CIFAR10":
        name = "cifar10"
    elif config.data.dataset == "LSUN":
        name = f"lsun_{config.data.category}"
    else:
        raise ValueError(config.data.dataset)
    ckpt = get_ckpt_path(f"ema_{name}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def build_student_from_brecq(
    config,
    device: torch.device,
    ode_scale_json: str,
    brecq_ckpt: str,
    cfg: JointSAConfig,
) -> Tuple[QuantModel, Set[str]]:
    s0 = load_ode_scales(ode_scale_json)
    absorb_mode = load_ode_absorb_mode(ode_scale_json, default="dilate")
    absorbed = load_ode_absorbed_layers(ode_scale_json) or set()

    model = Model(config)
    if config.data.dataset == "CIFAR10":
        name = "cifar10"
    else:
        name = f"lsun_{config.data.category}"
    ckpt = get_ckpt_path(f"ema_{name}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()

    absorbed = absorb_ode_weights_into_float_model(model, s0, mode=absorb_mode)
    shortcut_inv = load_ode_shortcut_inv_scales(ode_scale_json)

    wq_params = {
        "n_bits": cfg.weight_bit,
        "channel_wise": True,
        "scale_method": "max",
    }
    aq_params = {
        "n_bits": cfg.act_bit,
        "symmetric": cfg.a_sym,
        "channel_wise": False,
        "scale_method": "max",
        "leaf_param": True,
    }
    qnn = QuantModel(model=model, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.to(device)

    attach_ode_input_scales_to_quant_model(qnn, absorbed_layers=absorbed)
    attach_resblock_shortcut_scales(qnn, shortcut_inv=shortcut_inv)

    image_size = config.data.image_size
    channels = config.data.channels
    cali_stub = (
        torch.randn(1, channels, image_size, image_size),
        torch.randint(0, 1000, (1,)),
    )
    resume_cali_model(qnn, brecq_ckpt, cali_stub, quant_act=True, act_quant_mode="qdiff", cond=False)
    qnn.set_quant_state(weight_quant=True, act_quant=True)
    return qnn, absorbed


def _init_ode_json_from_learned(learned_ode_json: str) -> str:
    import json as _json

    with open(learned_ode_json, "r", encoding="utf-8") as f:
        meta = _json.load(f).get("meta", {})
    return meta.get("init_ode_json") or "ode_pre_scaling.json"


def apply_learned_deploy_scales(qnn: QuantModel, learned_ode_json: str) -> None:
    """Apply joint-learned deploy scales using the same path as joint training."""
    deploy = load_ode_scales(learned_ode_json)
    shortcut_inv = load_ode_shortcut_inv_scales(learned_ode_json)
    qmods = {
        n: m
        for n, m in qnn.model.named_modules()
        if isinstance(m, QuantModule) and m.fwd_func is F.conv2d
    }
    active = getattr(qnn, "_ode_absorbed_layer_names", None) or set()
    attached = 0
    for name in active:
        if name not in qmods:
            continue
        inv = _deploy_inv_for_quant_input(name, qmods[name], deploy)
        if inv is None:
            continue
        qmods[name].set_ode_input_inv_scale(inv.detach())
        attached += 1
    attach_resblock_shortcut_scales(qnn, shortcut_inv=shortcut_inv)
    logger.info("Applied %d learned ODE input /s + shortcut scales from %s", attached, learned_ode_json)


def load_joint_sa_for_sampling(
    config,
    device: torch.device,
    brecq_ckpt: str,
    joint_ckpt: str,
    learned_ode_json: str,
    cfg: Optional[JointSAConfig] = None,
) -> QuantModel:
    """Build QuantModel and load joint-SA checkpoint for FID sampling."""
    cfg = cfg or JointSAConfig()
    init_json = _init_ode_json_from_learned(learned_ode_json)
    qnn, _ = build_student_from_brecq(config, device, init_json, brecq_ckpt, cfg)

    ckpt = torch.load(joint_ckpt, map_location="cpu")
    model_sd = qnn.state_dict()
    filtered = {}
    for key, val in ckpt.items():
        if "ode_" in key and "inv_scale" in key:
            continue
        if "weight_quantizer.zero_point" in key or "weight_quantizer.delta" in key:
            continue
        if key not in model_sd or model_sd[key].shape != val.shape:
            continue
        filtered[key] = val
    missing, unexpected = qnn.load_state_dict(filtered, strict=False)
    if missing:
        bad = [k for k in missing if "act_quantizer" in k or "org_weight" in k or "alpha" in k]
        if bad:
            raise RuntimeError(f"joint ckpt missing required keys: {bad[:5]} ... ({len(bad)} total)")
        logger.warning("joint ckpt missing %d non-critical keys (first: %s)", len(missing), missing[0])
    if unexpected:
        logger.warning("joint ckpt ignored %d unexpected keys", len(unexpected))

    apply_learned_deploy_scales(qnn, learned_ode_json)
    qnn.set_quant_state(weight_quant=True, act_quant=True)
    qnn.eval()
    logger.info("Loaded joint-SA checkpoint from %s", joint_ckpt)
    return qnn


def prepare_cali_split(
    cali_data_path: str,
    cfg: JointSAConfig,
) -> CaliSplit:
    sample_data = torch.load(cali_data_path, map_location="cpu")
    cali_args = type(
        "Args",
        (),
        {
            "cali_n": cfg.cali_n,
            "cali_st": cfg.cali_st,
            "cond": False,
            "custom_steps": 0,
        },
    )()
    xs, ts = get_train_samples(cali_args, sample_data, custom_steps=0)
    del sample_data

    n = xs.shape[0]
    rng = np.random.RandomState(cfg.seed)
    perm = rng.permutation(n)
    n_hold = max(1, int(n * cfg.holdout_ratio))
    hold_idx = perm[:n_hold]
    train_idx = perm[n_hold:]

    train_ts = ts[train_idx]
    buckets = sorted(int(x) for x in torch.unique(train_ts).tolist())
    if not buckets:
        buckets = sorted(int(x) for x in torch.unique(ts).tolist())

    return CaliSplit(
        train_xs=xs[train_idx],
        train_ts=train_ts,
        hold_xs=xs[hold_idx],
        hold_ts=ts[hold_idx],
        timestep_buckets=buckets,
    )


def sample_batch_at_t(
    split: CaliSplit,
    t_value: int,
    batch_size: int,
    device: torch.device,
    rng: np.random.RandomState,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = split.train_ts == t_value
    idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        idx = torch.arange(split.train_xs.shape[0])
    choice = rng.choice(idx.numpy(), size=min(batch_size, idx.numel()), replace=False)
    xs = split.train_xs[choice].to(device)
    ts = split.train_ts[choice].to(device)
    return xs, ts


def freeze_weight_quant(qnn: QuantModel) -> None:
    for p in qnn.parameters():
        p.requires_grad = False
    for m in qnn.model.modules():
        if isinstance(m, (QuantModule, BaseQuantBlock)):
            if isinstance(m, QuantModule):
                m.org_weight.requires_grad = False
                if m.org_bias is not None:
                    m.org_bias.requires_grad = False
            if isinstance(m, AdaRoundQuantizer):
                if hasattr(m, "alpha") and m.alpha is not None:
                    m.alpha.requires_grad = False


def collect_act_quant_params(qnn: QuantModel) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    seen = set()
    for m in qnn.model.modules():
        if isinstance(m, UniformAffineQuantizer) and m.leaf_param and m.inited:
            if m.delta is not None and id(m.delta) not in seen:
                m.delta.requires_grad = True
                params.append(m.delta)
                seen.add(id(m.delta))
    return params


def freeze_act_quant(qnn: QuantModel) -> None:
    for m in qnn.model.modules():
        if isinstance(m, UniformAffineQuantizer) and m.leaf_param:
            if m.delta is not None:
                m.delta.requires_grad = False


def eps_mse_loss(
    student: nn.Module,
    teacher: nn.Module,
    xs: torch.Tensor,
    ts: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        target = teacher(xs, ts.float())
    pred = student(xs.clone(), ts.float())
    return (pred - target).pow(2).mean()


@torch.no_grad()
def eval_eps_loss(
    student: nn.Module,
    teacher: nn.Module,
    xs: torch.Tensor,
    ts: torch.Tensor,
    device: torch.device,
    batch_size: int = 32,
    scale_mgr: Optional[LearnableScaleManager] = None,
) -> float:
    student.eval()
    total, count = 0.0, 0
    for i in range(0, xs.shape[0], batch_size):
        xb = xs[i : i + batch_size].to(device)
        tb = ts[i : i + batch_size].to(device)
        if scale_mgr is not None:
            scale_mgr.apply_live_scales(student)
        total += eps_mse_loss(student, teacher, xb, tb).item() * xb.shape[0]
        if scale_mgr is not None:
            scale_mgr.clear_live_scales(student)
        count += xb.shape[0]
    return total / max(count, 1)


def _save_quant_ckpt(qnn: QuantModel, ckpt_path: str) -> None:
    """Save QuantModel state in the same format as BRECQ ckpt.pth for --resume."""
    for m in qnn.model.modules():
        if isinstance(m, AdaRoundQuantizer):
            m.zero_point = nn.Parameter(m.zero_point)
            m.delta = nn.Parameter(m.delta)
        elif isinstance(m, UniformAffineQuantizer):
            if m.zero_point is not None and not isinstance(m.zero_point, nn.Parameter):
                if not torch.is_tensor(m.zero_point):
                    m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                else:
                    m.zero_point = nn.Parameter(m.zero_point)
    torch.save(qnn.state_dict(), ckpt_path)


def train_joint_sa(
    qnn: QuantModel,
    teacher: Model,
    scale_mgr: LearnableScaleManager,
    split: CaliSplit,
    cfg: JointSAConfig,
    device: torch.device,
    output_dir: str,
    ode_scale_json: str,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    freeze_weight_quant(qnn)
    if cfg.freeze_act:
        freeze_act_quant(qnn)
        act_params: List[nn.Parameter] = []
        logger.info("Joint SA: optimizing ODE scales only (act quantizers frozen)")
    else:
        act_params = collect_act_quant_params(qnn)
    scale_params = list(scale_mgr.parameters())

    optim_groups = [{"params": scale_params, "lr": cfg.lr_s}]
    if act_params:
        optim_groups.insert(0, {"params": act_params, "lr": cfg.lr_act})
    optimizer = torch.optim.Adam(optim_groups, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.joint_steps, 1), eta_min=1e-5)

    rng = np.random.RandomState(cfg.seed + 7)
    buckets = split.timestep_buckets
    if not buckets:
        raise RuntimeError("No timestep buckets in calibration split")

    log = {"train_loss": [], "holdout_loss": [], "best_step": 0, "best_holdout": float("inf")}
    best_holdout = float("inf")
    best_state = None
    stale = 0

    qnn.train()
    teacher.eval()

    desc = "JointSA (s only)" if cfg.freeze_act else "JointSA (s + A_q)"
    pbar = tqdm(range(cfg.joint_steps), desc=desc)
    for step in pbar:
        t1 = buckets[step % len(buckets)]
        xs, ts = sample_batch_at_t(split, t1, cfg.batch_size, device, rng)

        scale_mgr.apply_live_scales(qnn)
        loss = eps_mse_loss(qnn, teacher, xs, ts)
        loss = loss + cfg.lambda_reg * scale_mgr.reg_loss()
        if step >= cfg.uni_start_step and cfg.lambda_uni > 0:
            loss = loss + cfg.lambda_uni * scale_mgr.reg_loss()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        scale_mgr.clear_live_scales(qnn)

        if (step + 1) % cfg.log_every == 0:
            log["train_loss"].append({"step": step + 1, "loss": loss.item()})
            pbar.set_postfix(loss=f"{loss.item():.4f}", t=t1)

        if (step + 1) % cfg.eval_every == 0:
            hold = eval_eps_loss(
                qnn, teacher, split.hold_xs, split.hold_ts, device, cfg.batch_size, scale_mgr
            )
            log["holdout_loss"].append({"step": step + 1, "loss": hold})
            if hold < best_holdout:
                best_holdout = hold
                log["best_step"] = step + 1
                log["best_holdout"] = hold
                best_state = {
                    "qnn": {k: v.cpu() for k, v in qnn.state_dict().items()},
                    "scale_mgr": scale_mgr.state_dict(),
                }
                stale = 0
            else:
                stale += cfg.eval_every
            if stale >= cfg.early_stop_patience:
                logger.info("Early stop at step %d (holdout=%.6f)", step + 1, hold)
                break

    if best_state is not None:
        qnn.load_state_dict(best_state["qnn"])
        scale_mgr.load_state_dict(best_state["scale_mgr"])

    scale_mgr.sync_static_scales(qnn)
    qnn.eval()

    full_deploy = load_ode_scales(ode_scale_json)
    full_deploy.update(scale_mgr.deploy_scales())
    shortcut_inv = scale_mgr.shortcut_inv_scales()

    learned_json = os.path.join(output_dir, "ode_pre_scaling_learned.json")
    source = "joint_sa_stage2_scale_only" if cfg.freeze_act else "joint_sa_stage2"
    export_learned_scales(
        learned_json,
        full_deploy,
        meta={
            "source": source,
            "freeze_act": cfg.freeze_act,
            "joint_steps": cfg.joint_steps,
            "best_holdout_eps_mse": log["best_holdout"],
            "best_step": log["best_step"],
            "shortcut_inv_scales": {k: v.tolist() for k, v in shortcut_inv.items()},
            "init_ode_json": ode_scale_json,
        },
        base_json=ode_scale_json,
    )

    ckpt_path = os.path.join(output_dir, "ckpt_joint_sa.pth")
    _save_quant_ckpt(qnn, ckpt_path)
    with open(os.path.join(output_dir, "joint_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    logger.info("Saved learned scales to %s", learned_json)
    logger.info("Saved joint checkpoint to %s", ckpt_path)
    return log


def refresh_act_quant(
    qnn: QuantModel,
    cali_data: Tuple[torch.Tensor, torch.Tensor],
    cfg: JointSAConfig,
    iters: Optional[int] = None,
) -> None:
    """Stage 3a: lightweight activation LSQ with frozen weights."""
    iters = iters or cfg.act_refresh_iters
    cali_xs, cali_ts = cali_data
    freeze_weight_quant(qnn)
    qnn.set_quant_state(weight_quant=True, act_quant=True)

    kwargs = dict(
        cali_data=cali_data,
        batch_size=cfg.batch_size,
        iters=iters,
        act_quant=True,
        opt_mode="mse",
        lr=cfg.cali_lr,
        p=cfg.cali_p,
    )

    def recon(module):
        for name, child in module.named_children():
            if isinstance(child, QuantModule):
                if not child.ignore_reconstruction:
                    layer_reconstruction(qnn, child, **kwargs)
            elif isinstance(child, BaseQuantBlock):
                if not child.ignore_reconstruction:
                    block_reconstruction(qnn, child, **kwargs)
            else:
                recon(child)

    logger.info("Stage 3a: activation refresh (%d iters)", iters)
    recon(qnn.model)
