"""
DDIM denoise-formula-aware pre-quantization channel scaling.

Pre-scale s is NOT the activation quantizer delta. It reparameterizes the float
model so ODE/DDIM output stays identical while cross-timestep activation
statistics become more uniform before PTQ.

Deploy modes:
  dilate — DilateQuant-style: input /s + same-layer W·s (float-equivalent)
  strict — shortcut-only cross-layer absorb (legacy minimal)
  ptq    — ResBlock mid-scale (NOT float-equivalent)
  legacy — module-order heuristic (NOT float-equivalent)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


def list_target_conv_modules(model: nn.Module) -> List[Tuple[str, nn.Conv2d]]:
    """Conv2d layers on a float Model (before QuantModel wrapping)."""
    layers: List[Tuple[str, nn.Conv2d]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            layers.append((name, module))
    return layers


def _is_conv2d_layer(module: nn.Module) -> bool:
    """True for Conv2d or QuantModule wrapping conv2d (exclude Linear / temb)."""
    from qdiff.quant_layer import QuantModule

    if isinstance(module, nn.Conv2d):
        return True
    if isinstance(module, QuantModule) and module.fwd_func is F.conv2d:
        return True
    return False


def list_ode_calibration_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Conv2d output layers for ODE energy stats and weight absorption.
    Excludes Linear QuantModules (e.g. temb.dense) which are not channel-scaled.
    """
    layers: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if _is_conv2d_layer(module):
            layers.append((name, module))
    return layers


def _unwrap_forward_model(model: nn.Module) -> nn.Module:
    """Return the module whose forward(x, t) runs the UNet."""
    if hasattr(model, "model") and callable(getattr(model, "forward", None)):
        from qdiff.quant_model import QuantModel

        if isinstance(model, QuantModel):
            return model
    return model


def _alpha_bar(betas: torch.Tensor) -> torch.Tensor:
    return (1.0 - betas).cumprod(dim=0)


def ddim_coefs_at_timestep(
    betas: torch.Tensor, t_index: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (a_t, b_t, a_{t-1}, b_{t-1}) for DDIM state at discrete index t_index.

    a_t = sqrt(alpha_bar[t]), b_t = sqrt(1 - alpha_bar[t])
    a_{t-1}, b_{t-1} use index t_index - 1 (clamped at 0).
    """
    ab = _alpha_bar(betas)
    t_index = int(max(0, min(t_index, ab.numel() - 1)))
    a_t = ab[t_index].sqrt()
    b_t = (1.0 - ab[t_index]).sqrt()
    t_prev = max(t_index - 1, 0)
    a_tm1 = ab[t_prev].sqrt()
    b_tm1 = (1.0 - ab[t_prev]).sqrt()
    return a_t, b_t, a_tm1, b_tm1


def _predict_x0_eps(
    model: nn.Module, x: torch.Tensor, t: torch.Tensor, betas: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Noise prediction and x0 from current latent (same as generalized_steps)."""
    et = model(x, t)
    beta = torch.cat([torch.zeros(1, device=betas.device), betas], dim=0)
    at = (1.0 - beta).cumprod(dim=0).index_select(0, t.long() + 1)
    while at.dim() < x.dim():
        at = at.unsqueeze(-1)
    x0 = (x - et * (1.0 - at).sqrt()) / at.sqrt()
    return x0, et


def accumulate_denoise_energy(
    model: nn.Module,
    cali_xs: torch.Tensor,
    cali_ts: torch.Tensor,
    betas: torch.Tensor,
    device: torch.device,
    batch_size: int = 8,
    eps: float = 1e-8,
    max_batches_per_t: Optional[int] = None,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Compute M_{ell,t,c} = E[(j * h)^2] with
    j = a_{t-1} * d(x0)/d(h) + b_{t-1} * d(eps)/d(h) from DDIM formula.

    Returns:
        stats[layer_name][t_index] -> Tensor[C] on CPU
    """
    model.eval()
    betas = betas.to(device)
    forward_model = _unwrap_forward_model(model)
    hook_root = forward_model.model if hasattr(forward_model, "model") else model
    targets = list_ode_calibration_modules(hook_root)
    if not targets:
        raise RuntimeError("No Conv2d / QuantModule layers found for ODE pre-scaling.")

    unique_ts = torch.unique(cali_ts).cpu().numpy().astype(int)
    stats: Dict[str, Dict[int, torch.Tensor]] = {
        name: {} for name, _ in targets
    }
    counts: Dict[str, Dict[int, int]] = {name: {} for name, _ in targets}

    for t_val in unique_ts:
        mask = (cali_ts.cpu() == t_val).numpy()
        xs_t = cali_xs[mask]
        if xs_t.numel() == 0:
            continue
        _, _, a_tm1, b_tm1 = ddim_coefs_at_timestep(betas, int(t_val))
        a_tm1 = a_tm1.to(device)
        b_tm1 = b_tm1.to(device)

        n_batches = (xs_t.shape[0] + batch_size - 1) // batch_size
        if max_batches_per_t is not None:
            n_batches = min(n_batches, max_batches_per_t)

        for bi in tqdm(range(n_batches), desc=f"ODE scale t={t_val}", leave=False):
            x = xs_t[bi * batch_size : (bi + 1) * batch_size].to(device).float()
            t = torch.full((x.shape[0],), int(t_val), device=device, dtype=torch.float32)

            activations: Dict[str, torch.Tensor] = {}
            hooks = []

            def _make_hook(layer_name: str):
                def _hook(_m, _inp, out):
                    if out.requires_grad:
                        out.retain_grad()
                    activations[layer_name] = out
                    return out

                return _hook

            for name, mod in targets:
                hooks.append(mod.register_forward_hook(_make_hook(name)))

            try:
                with torch.enable_grad():
                    x0, et = _predict_x0_eps(forward_model, x, t, betas)
                    for name, _ in targets:
                        h = activations.get(name)
                        if h is None or not h.requires_grad:
                            continue
                        g0 = torch.autograd.grad(
                            x0.sum(), h, retain_graph=True, allow_unused=True
                        )[0]
                        ge = torch.autograd.grad(
                            et.sum(), h, retain_graph=True, allow_unused=True
                        )[0]
                        if g0 is None:
                            g0 = torch.zeros_like(h)
                        if ge is None:
                            ge = torch.zeros_like(h)
                        j = a_tm1 * g0 + b_tm1 * ge
                        z = j * h
                        m_c = z.pow(2).mean(dim=(0, 2, 3)).detach().cpu()

                        if t_val not in stats[name]:
                            stats[name][int(t_val)] = m_c.clone()
                            counts[name][int(t_val)] = 1
                        else:
                            n = counts[name][int(t_val)]
                            stats[name][int(t_val)] = (
                                stats[name][int(t_val)] * n + m_c
                            ) / (n + 1)
                            counts[name][int(t_val)] = n + 1
            finally:
                for h in hooks:
                    h.remove()

    return stats


def compute_deploy_scales(
    m_stats: Dict[str, Dict[int, torch.Tensor]],
    alpha: float = 0.5,
    s_min: float = 0.5,
    s_max: float = 2.0,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Per-timestep s* = sqrt(M_{l,t,c} / M^ref_{l,c}), geometric mean, clip, alpha.
    """
    deploy: Dict[str, torch.Tensor] = {}
    for layer_name, t_dict in m_stats.items():
        if not t_dict:
            continue
        ts = sorted(t_dict.keys())
        stack = torch.stack([t_dict[t] for t in ts], dim=0)
        m_ref = stack.mean(dim=0).clamp(min=eps)

        s_stars = torch.sqrt((stack + eps) / (m_ref.unsqueeze(0) + eps))
        log_mean = torch.log(s_stars + eps).mean(dim=0)
        s_bar = torch.exp(log_mean)
        if alpha <= 0.0:
            s_dep = torch.ones_like(s_bar)
        else:
            s_dep = s_bar.pow(alpha)
        s_dep = s_dep.clamp(min=s_min, max=s_max)
        deploy[layer_name] = s_dep
    return deploy


def save_ode_scales(
    path: str,
    deploy_scales: Dict[str, torch.Tensor],
    m_stats: Optional[Dict[str, Dict[int, torch.Tensor]]] = None,
    meta: Optional[dict] = None,
) -> None:
    payload = {
        "meta": meta or {},
        "deploy": {k: v.cpu().tolist() for k, v in deploy_scales.items()},
    }
    if m_stats is not None:
        payload["m_per_timestep"] = {
            layer: {str(t): v.cpu().tolist() for t, v in td.items()}
            for layer, td in m_stats.items()
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved ODE pre-scaling factors to %s (%d layers)", path, len(deploy_scales))


def load_ode_scales(path: str) -> Dict[str, torch.Tensor]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    deploy = payload.get("deploy", payload)
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in deploy.items()}


def export_learned_scales(
    path: str,
    deploy_scales: Dict[str, torch.Tensor],
    meta: Optional[dict] = None,
    base_json: Optional[str] = None,
) -> None:
    """Save learned deploy scales; optionally merge meta from a base JSON."""
    merged_meta = dict(meta or {})
    if base_json and os.path.isfile(base_json):
        with open(base_json, "r", encoding="utf-8") as f:
            base_payload = json.load(f)
        base_meta = base_payload.get("meta", {})
        for key in ("absorbed_layers", "absorb_mode", "shortcut_blocks", "alpha", "s_min", "s_max"):
            if key in base_meta and key not in merged_meta:
                merged_meta[key] = base_meta[key]
        if "shortcut_inv_scales" not in merged_meta and "shortcut_inv_scales" in base_meta:
            merged_meta["shortcut_inv_scales"] = base_meta["shortcut_inv_scales"]
    merged_meta.setdefault("source", "joint_sa")
    save_ode_scales(path, deploy_scales, m_stats=None, meta=merged_meta)


def load_ode_absorbed_layers(path: str) -> Optional[Set[str]]:
    """Layer names with paired weight absorb + output /s (from JSON meta)."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    names = payload.get("meta", {}).get("absorbed_layers")
    if not names:
        return None
    return set(names)


def load_ode_absorb_mode(path: str, default: str = "dilate") -> str:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("meta", {}).get("absorb_mode", default)


def load_ode_shortcut_inv_scales(path: str) -> Dict[str, torch.Tensor]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw = payload.get("meta", {}).get("shortcut_inv_scales", {})
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in raw.items()}


def _module_out_channels(module: nn.Module) -> int:
    from qdiff.quant_layer import QuantModule

    if isinstance(module, (nn.Conv2d, QuantModule)):
        return int(module.weight.shape[0])
    raise TypeError(f"unsupported module type: {type(module)}")


def _module_in_channels(module: nn.Module) -> int:
    from qdiff.quant_layer import QuantModule

    if isinstance(module, (nn.Conv2d, QuantModule)):
        return int(module.weight.shape[1])
    if isinstance(module, nn.Linear):
        return int(module.weight.shape[1])
    raise TypeError(f"unsupported module type: {type(module)}")


def _get_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    for n, mod in model.named_modules():
        if n == name:
            return mod
    return None


def list_resnet_block_prefixes(model: nn.Module) -> List[str]:
    """All ResnetBlock module prefixes on a float DDIM UNet."""
    from ddim.models.diffusion import ResnetBlock

    return [
        name
        for name, mod in model.named_modules()
        if isinstance(mod, ResnetBlock) and name
    ]


def discover_strict_shortcut_edges(model: nn.Module) -> List[Tuple[str, str]]:
    """
    Strict Conv→Conv edges on the shortcut branch: previous downsample.conv
    output is passed by reference into block.0 nin_shortcut / conv_shortcut.
    """
    from ddim.models.diffusion import ResnetBlock

    edges: List[Tuple[str, str]] = []
    for name, block in model.named_modules():
        if not isinstance(block, ResnetBlock):
            continue
        if block.in_channels == block.out_channels:
            continue
        if not name.endswith(".block.0") or not name.startswith("down."):
            continue
        level = int(name.split(".")[1])
        if level == 0:
            continue
        src = f"down.{level - 1}.downsample.conv"
        dst = (
            f"{name}.conv_shortcut"
            if block.use_conv_shortcut
            else f"{name}.nin_shortcut"
        )
        src_mod = _get_module_by_name(model, src)
        dst_mod = _get_module_by_name(model, dst)
        if src_mod is None or dst_mod is None:
            continue
        if not isinstance(src_mod, nn.Conv2d) or not isinstance(dst_mod, nn.Conv2d):
            continue
        if _module_out_channels(src_mod) != block.in_channels:
            continue
        if _module_in_channels(dst_mod) != block.in_channels:
            continue
        edges.append((src, dst))
    return edges


def _shortcut_dst_layers(model: nn.Module) -> Set[str]:
    return {dst for _, dst in discover_strict_shortcut_edges(model)}


def resolve_dilate_input_scale(
    layer_name: str,
    module: nn.Module,
    deploy_scales: Dict[str, torch.Tensor],
    conv_layers: List[Tuple[str, nn.Module]],
) -> Optional[torch.Tensor]:
    """
    Map ODE deploy scales (on conv outputs) to per-input-channel s for layer_name.

    DilateQuant applies s on the matmul input; ODE stats are keyed by conv output.
    """
    in_c = _module_in_channels(module)
    name_to_idx = {name: i for i, (name, _) in enumerate(conv_layers)}

    if layer_name.endswith(".conv2"):
        conv1_key = f"{layer_name.rsplit('.', 1)[0]}.conv1"
        s = deploy_scales.get(conv1_key)
        if s is not None and s.numel() == in_c:
            return s.float().clamp(min=1e-8)

    if layer_name in deploy_scales:
        s = deploy_scales[layer_name]
        if s.numel() == in_c:
            return s.float().clamp(min=1e-8)

    if layer_name not in name_to_idx:
        return None
    idx = name_to_idx[layer_name]
    for j in range(idx - 1, -1, -1):
        pname, pmod = conv_layers[j]
        if _module_out_channels(pmod) != in_c:
            continue
        s = deploy_scales.get(pname)
        if s is not None and s.numel() == in_c:
            return s.float().clamp(min=1e-8)
    return None


def absorb_scale_into_same_weight(s: torch.Tensor, mod: nn.Module) -> bool:
    """W'[:, c, ...] *= s_c on input channels of the same layer."""
    from qdiff.quant_layer import QuantModule

    s = s.float().clamp(min=1e-8)
    if isinstance(mod, QuantModule):
        cin = mod.weight.shape[1]
        if s.numel() != cin:
            logger.warning(
                "Skip same-layer absorb: scale dim %d != input %d",
                s.numel(),
                cin,
            )
            return False
        sf = s.view(1, -1, 1, 1).to(mod.weight.device)
        mod.weight.data.mul_(sf)
        mod.org_weight.data.mul_(sf)
        return True
    if isinstance(mod, nn.Conv2d):
        cin = mod.weight.shape[1]
        if s.numel() != cin:
            logger.warning(
                "Skip same-layer absorb: scale dim %d != Conv2d input %d",
                s.numel(),
                cin,
            )
            return False
        sf = s.view(1, -1, 1, 1).to(mod.weight.device)
        mod.weight.data.mul_(sf)
        return True
    return False


def _absorb_dilate_scales(
    model: nn.Module, deploy_scales: Dict[str, torch.Tensor]
) -> Tuple[Set[str], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    DilateQuant-style: input /s + same-layer W·s for each conv with resolved s.
    Returns (layer names, input inv scales 1/s, input s tensors).
    """
    conv_layers = list_ode_calibration_modules(model)
    skip_dst = _shortcut_dst_layers(model)
    absorbed: Set[str] = set()
    inv_scales: Dict[str, torch.Tensor] = {}
    input_scales: Dict[str, torch.Tensor] = {}

    for name, mod in conv_layers:
        if name in skip_dst:
            continue
        s = resolve_dilate_input_scale(name, mod, deploy_scales, conv_layers)
        if s is None:
            continue
        if absorb_scale_into_same_weight(s, mod):
            absorbed.add(name)
            input_scales[name] = s.cpu()
            inv_scales[name] = (1.0 / s).cpu()

    return absorbed, inv_scales, input_scales


def absorb_scale_into_linear_output(s: torch.Tensor, linear: nn.Linear) -> bool:
    """Match temb branch to conv1 output /s: W_out,: /= s_c, b /= s."""
    s = s.float().clamp(min=1e-8)
    out_features = linear.weight.shape[0]
    if s.numel() != out_features:
        logger.warning(
            "Skip linear output absorb: scale dim %d != linear out %d",
            s.numel(),
            out_features,
        )
        return False
    linear.weight.data.div_(s.view(-1, 1).to(linear.weight.device))
    if linear.bias is not None:
        linear.bias.data.div_(s.to(linear.bias.device))
    return True


def absorb_resblock_mid_scale(block: nn.Module, s_mid: torch.Tensor) -> bool:
    """
    ResBlock-level linear fold after conv1: h1/s + (W/s)*temb absorbed into
    conv2 input and temb_proj output. Runtime /s is on conv1 (QuantModule).
    """
    from ddim.models.diffusion import ResnetBlock

    if not isinstance(block, ResnetBlock):
        return False
    s = s_mid.float().clamp(min=1e-8)
    ok_conv2 = absorb_scale_into_next_weight(s, block.conv2)
    ok_temb = absorb_scale_into_linear_output(s, block.temb_proj)
    return ok_conv2 and ok_temb


def _absorb_strict_shortcut_edges(
    model: nn.Module, deploy_scales: Dict[str, torch.Tensor]
) -> Tuple[Set[str], Dict[str, torch.Tensor]]:
    """
    Absorb into shortcut conv only; return block prefixes needing shortcut-path /s.
    Do NOT scale the full downsample output (main branch norm1 must see unscaled x).
    """
    output_layers: Set[str] = set()
    shortcut_inv: Dict[str, torch.Tensor] = {}
    for src, dst in discover_strict_shortcut_edges(model):
        if src not in deploy_scales:
            logger.debug("Strict shortcut: no deploy scale for %s", src)
            continue
        dst_mod = _get_module_by_name(model, dst)
        if dst_mod is None:
            continue
        s = deploy_scales[src].float().clamp(min=1e-8)
        if absorb_scale_into_next_weight(s, dst_mod):
            block_prefix = dst.rsplit(".", 1)[0]
            shortcut_inv[block_prefix] = (1.0 / s).cpu()
            logger.debug(
                "Strict shortcut absorb: %s -> %s (block %s shortcut /s)",
                src,
                dst,
                block_prefix,
            )
    return output_layers, shortcut_inv


def _absorb_resblock_mid_scales(
    model: nn.Module, deploy_scales: Dict[str, torch.Tensor]
) -> Set[str]:
    absorbed: Set[str] = set()
    for prefix in list_resnet_block_prefixes(model):
        conv1_key = f"{prefix}.conv1"
        if conv1_key not in deploy_scales:
            continue
        block = _get_module_by_name(model, prefix)
        if block is None:
            continue
        s = deploy_scales[conv1_key].float().clamp(min=1e-8)
        if absorb_resblock_mid_scale(block, s):
            absorbed.add(conv1_key)
    return absorbed


def _absorb_legacy_heuristic(
    model: nn.Module, deploy_scales: Dict[str, torch.Tensor]
) -> Set[str]:
    conv_layers = list_ode_calibration_modules(model)
    name_to_idx = {name: i for i, (name, _) in enumerate(conv_layers)}
    absorbed: Set[str] = set()
    for name, s in deploy_scales.items():
        if name not in name_to_idx:
            continue
        idx = name_to_idx[name]
        s = s.float().clamp(min=1e-8)
        nxt = _find_absorb_target(conv_layers, idx)
        if nxt is not None and absorb_scale_into_next_weight(s, nxt[1]):
            absorbed.add(name)
    return absorbed


def _find_absorb_target(
    conv_layers: List[Tuple[str, nn.Module]], index: int
) -> Optional[Tuple[str, nn.Module]]:
    """
    Find the next layer whose input channels match current layer output channels.
    Skips unrelated convs (e.g. skip branches, concat paths with different widths).
    """
    out_c = _module_out_channels(conv_layers[index][1])
    for j in range(index + 1, len(conv_layers)):
        nxt_name, nxt_mod = conv_layers[j]
        if _module_in_channels(nxt_mod) == out_c:
            return nxt_name, nxt_mod
    return None


def absorb_scale_into_next_weight(
    s: torch.Tensor, next_mod: nn.Module
) -> bool:
    """W'[:, c, ...] *= s_c on input channels of the next layer. Returns False if skipped."""
    from qdiff.quant_layer import QuantModule

    s = s.float()
    if isinstance(next_mod, QuantModule):
        cin = next_mod.weight.shape[1]
        if s.numel() != cin:
            logger.warning(
                "Skip weight absorb: scale dim %d != next QuantModule input %d",
                s.numel(), cin,
            )
            return False
        sf = s.view(1, -1, 1, 1).to(next_mod.weight.device)
        next_mod.weight.data.mul_(sf)
        next_mod.org_weight.data.mul_(sf)
        return True
    if isinstance(next_mod, nn.Conv2d):
        cin = next_mod.weight.shape[1]
        if s.numel() != cin:
            logger.warning(
                "Skip weight absorb: scale dim %d != next Conv2d input %d",
                s.numel(), cin,
            )
            return False
        sf = s.view(1, -1, 1, 1).to(next_mod.weight.device)
        next_mod.weight.data.mul_(sf)
        return True
    if isinstance(next_mod, nn.Linear):
        cin = next_mod.weight.shape[1]
        if s.numel() != cin:
            logger.warning(
                "Skip weight absorb: scale dim %d != next Linear input %d",
                s.numel(), cin,
            )
            return False
        sf = s.view(1, -1).to(next_mod.weight.device)
        next_mod.weight.data.mul_(sf)
        return True
    return False


def absorb_ode_weights_into_float_model(
    model: nn.Module,
    deploy_scales: Dict[str, torch.Tensor],
    skip_if_absorbed: bool = True,
    mode: str = "dilate",
) -> Set[str]:
    """
    Absorb deploy scales into float weights before QuantModel wrapping.

    Modes:
      dilate — DilateQuant-style input /s + same-layer W·s (float-equivalent)
      strict — only strict shortcut Conv→Conv edges (float-equivalent on CIFAR)
      ptq    — ResBlock mid-scale (conv1→conv2+temb) + strict shortcuts
      legacy — old module-order heuristic (not float-equivalent)

    Returns layer names where runtime /s must be enabled (paired absorption).
    """
    if skip_if_absorbed and getattr(model, "_ode_weights_absorbed", False):
        logger.info("ODE weight absorption already applied, skipping")
        return set(getattr(model, "_ode_absorbed_layer_names", set()))

    mode = (mode or "dilate").lower()
    if mode not in ("dilate", "strict", "ptq", "legacy"):
        raise ValueError(f"unknown ode absorb mode: {mode}")

    output_layers: Set[str] = set()
    shortcut_inv: Dict[str, torch.Tensor] = {}

    if mode == "dilate":
        output_layers, inv_scales, input_scales = _absorb_dilate_scales(
            model, deploy_scales
        )
        _, shortcut_inv = _absorb_strict_shortcut_edges(model, deploy_scales)
        model._ode_input_inv_scales = inv_scales
        model._ode_dilate_input_scales = input_scales
        model._ode_scale_deploy_mode = "input"
    elif mode == "strict":
        output_layers, shortcut_inv = _absorb_strict_shortcut_edges(model, deploy_scales)
        model._ode_scale_deploy_mode = "output"
    elif mode == "ptq":
        output_layers = _absorb_resblock_mid_scales(model, deploy_scales)
        _, shortcut_inv = _absorb_strict_shortcut_edges(model, deploy_scales)
        model._ode_scale_deploy_mode = "output"
    else:
        output_layers = _absorb_legacy_heuristic(model, deploy_scales)
        model._ode_scale_deploy_mode = "output"

    model._ode_weights_absorbed = True
    model._ode_absorb_mode = mode
    model._ode_absorbed_layer_names = output_layers
    model._ode_shortcut_inv_scales = shortcut_inv
    model._ode_pre_scale_factors = {k: v.cpu() for k, v in deploy_scales.items()}
    if mode == "dilate":
        logger.info(
            "ODE weight absorption [dilate]: %d conv input /s, %d shortcut blocks (of %d deploy)",
            len(output_layers),
            len(shortcut_inv),
            len(deploy_scales),
        )
    else:
        logger.info(
            "ODE weight absorption [%s]: %d conv output /s, %d shortcut blocks (of %d deploy)",
            mode,
            len(output_layers),
            len(shortcut_inv),
            len(deploy_scales),
        )
    return output_layers


def _resolve_absorbed_layers(
    model_or_qnn: nn.Module,
    absorbed_layers: Optional[Set[str]],
) -> Optional[Set[str]]:
    if absorbed_layers is not None:
        return absorbed_layers
    root = getattr(model_or_qnn, "model", model_or_qnn)
    names = getattr(root, "_ode_absorbed_layer_names", None)
    return set(names) if names is not None else None


def attach_ode_output_scales_to_quant_model(
    qnn: nn.Module,
    deploy_scales: Dict[str, torch.Tensor],
    absorbed_layers: Optional[Set[str]] = None,
) -> int:
    """
    Register per-channel 1/s on QuantModule outputs only when weight absorption
    succeeded for that layer (paired h/s + W*s preserves float forward).
    """
    from qdiff.quant_layer import QuantModule

    active = _resolve_absorbed_layers(qnn, absorbed_layers)
    if active is None:
        logger.warning(
            "ODE output /s: no absorbed_layers set; skipping all output scales. "
            "Run absorb_ode_weights_into_float_model first."
        )
        return 0

    attached = 0
    skipped = 0
    qmods = {
        name: m
        for name, m in qnn.model.named_modules()
        if isinstance(m, QuantModule) and _is_conv2d_layer(m)
    }
    for name, s in deploy_scales.items():
        if name not in active:
            skipped += 1
            continue
        if name not in qmods:
            logger.warning(
                "ODE scale: absorbed layer %s not found as conv QuantModule, skip /s",
                name,
            )
            skipped += 1
            continue
        inv = (1.0 / s.float().clamp(min=1e-8))
        qmods[name].set_ode_output_inv_scale(inv)
        attached += 1

    qnn._ode_absorbed_layer_names = set(active)
    qnn._ode_pre_scale_factors = {
        k: v.cpu() for k, v in deploy_scales.items() if k in active
    }
    logger.info(
        "ODE output /s attached to %d QuantModules (%d skipped, no paired absorb)",
        attached, skipped,
    )
    return attached


def attach_ode_input_scales_to_quant_model(
    qnn: nn.Module,
    absorbed_layers: Optional[Set[str]] = None,
    input_inv_scales: Optional[Dict[str, torch.Tensor]] = None,
) -> int:
    """
    Register per-input-channel 1/s on QuantModule inputs (DilateQuant-style).
    """
    from qdiff.quant_layer import QuantModule

    active = _resolve_absorbed_layers(qnn, absorbed_layers)
    if active is None:
        logger.warning(
            "ODE input /s: no absorbed_layers set; skipping. "
            "Run absorb_ode_weights_into_float_model(mode='dilate') first."
        )
        return 0

    if input_inv_scales is None:
        root = getattr(qnn, "model", qnn)
        input_inv_scales = getattr(root, "_ode_input_inv_scales", None) or {}

    attached = 0
    skipped = 0
    qmods = {
        name: m
        for name, m in qnn.model.named_modules()
        if isinstance(m, QuantModule) and _is_conv2d_layer(m)
    }
    for name in active:
        if name not in qmods:
            logger.warning(
                "ODE input /s: layer %s not found as conv QuantModule, skip", name
            )
            skipped += 1
            continue
        inv = input_inv_scales.get(name)
        if inv is None:
            skipped += 1
            continue
        qmods[name].set_ode_input_inv_scale(inv)
        attached += 1

    qnn._ode_absorbed_layer_names = set(active)
    qnn._ode_scale_deploy_mode = "input"
    logger.info(
        "ODE input /s attached to %d QuantModules (%d skipped)",
        attached,
        skipped,
    )
    return attached


def attach_resblock_shortcut_scales(
    qnn: nn.Module,
    shortcut_inv: Optional[Dict[str, torch.Tensor]] = None,
) -> int:
    """Apply /s on ResBlock shortcut input only (paired with nin weight absorb)."""
    from qdiff.quant_block import QuantResnetBlock

    if shortcut_inv is None:
        root = getattr(qnn, "model", qnn)
        shortcut_inv = getattr(root, "_ode_shortcut_inv_scales", None) or {}

    attached = 0
    blocks = {
        name: m
        for name, m in qnn.model.named_modules()
        if isinstance(m, QuantResnetBlock)
    }
    for block_name, inv in shortcut_inv.items():
        if block_name not in blocks:
            logger.warning(
                "ODE shortcut /s: block %s not found as QuantResnetBlock", block_name
            )
            continue
        blocks[block_name].set_ode_shortcut_inv_scale(inv)
        attached += 1

    qnn._ode_shortcut_inv_scales = {k: v.cpu() for k, v in shortcut_inv.items()}
    logger.info("ODE shortcut-path /s attached to %d ResBlocks", attached)
    return attached


def apply_ode_pre_scaling(
    model: nn.Module,
    deploy_scales: Dict[str, torch.Tensor],
    register_hooks: bool = True,
    mode: str = "dilate",
) -> List:
    """
    Apply deploy scales on a float Model (no QuantModel):
      dilate — same-layer W·s + forward_pre_hook input /s
      other  — next-layer W·s + forward_hook output /s (legacy modes)

    For PTQ use absorb_ode_weights_into_float_model + attach_ode_*_scales.
    """
    absorbed = absorb_ode_weights_into_float_model(
        model, deploy_scales, skip_if_absorbed=False, mode=mode
    )
    conv_layers = list_ode_calibration_modules(model)
    name_to_idx = {name: i for i, (name, _) in enumerate(conv_layers)}
    handles = []

    if register_hooks and mode == "dilate":
        inv_map = getattr(model, "_ode_input_inv_scales", {})
        for name in absorbed:
            inv = inv_map.get(name)
            if inv is None or name not in name_to_idx:
                continue
            _, mod = conv_layers[name_to_idx[name]]
            inv_buf = inv.view(1, -1, 1, 1).to(mod.weight.device).clone()

            def _in_pre_hook(_m, args, inv_s=inv_buf):
                x = args[0]
                c = min(inv_s.shape[1], x.shape[1])
                x = x.clone()
                x[:, :c, :, :] = x[:, :c, :, :] * inv_s[:, :c, :, :].to(
                    device=x.device, dtype=x.dtype
                )
                return (x,) + args[1:]

            handles.append(mod.register_forward_pre_hook(_in_pre_hook))
    elif register_hooks:
        for name in absorbed:
            s = deploy_scales.get(name)
            if s is None or name not in name_to_idx:
                continue
            idx = name_to_idx[name]
            _, mod = conv_layers[idx]
            s = s.float().clamp(min=1e-8)
            inv_buf = (1.0 / s).view(1, -1, 1, 1).to(mod.weight.device).clone()

            def _out_hook(_m, _inp, out, inv_s=inv_buf):
                c = min(inv_s.shape[1], out.shape[1])
                out = out.clone()
                out[:, :c, :, :] = out[:, :c, :, :] * inv_s[:, :c, :, :]
                return out

            handles.append(mod.register_forward_hook(_out_hook))

    if not hasattr(model, "_ode_pre_scale_hooks"):
        model._ode_pre_scale_hooks = handles
    else:
        model._ode_pre_scale_hooks.extend(handles)
    logger.info(
        "Applied ODE pre-scaling on float model [%s] (%d paired layers)",
        mode,
        len(handles),
    )
    return handles


def apply_ode_pre_scaling_for_ptq(
    float_model: nn.Module,
    qnn: nn.Module,
    deploy_scales: Dict[str, torch.Tensor],
    mode: str = "dilate",
) -> Set[str]:
    """
    Full PTQ pipeline: weight absorb on float model, runtime /s on QuantModules.
    """
    absorbed = absorb_ode_weights_into_float_model(
        float_model, deploy_scales, mode=mode
    )
    if mode == "dilate":
        attach_ode_input_scales_to_quant_model(qnn, absorbed_layers=absorbed)
    else:
        attach_ode_output_scales_to_quant_model(
            qnn, deploy_scales, absorbed_layers=absorbed
        )
    attach_resblock_shortcut_scales(qnn)
    return absorbed


def calibrate_ode_pre_scaling(
    model: nn.Module,
    cali_xs: torch.Tensor,
    cali_ts: torch.Tensor,
    betas: torch.Tensor,
    device: torch.device,
    batch_size: int = 8,
    alpha: float = 0.5,
    s_min: float = 0.5,
    s_max: float = 2.0,
    eps: float = 1e-8,
    max_batches_per_t: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[int, torch.Tensor]]]:
    """Full calibration: M stats -> deploy scales."""
    m_stats = accumulate_denoise_energy(
        model,
        cali_xs,
        cali_ts,
        betas,
        device,
        batch_size=batch_size,
        eps=eps,
        max_batches_per_t=max_batches_per_t,
    )
    deploy = compute_deploy_scales(
        m_stats, alpha=alpha, s_min=s_min, s_max=s_max, eps=eps
    )
    return deploy, m_stats


def verify_float_equivalence(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> bool:
    """Check output unchanged after apply_ode_pre_scaling (hooks must be active)."""
    model.eval()
    with torch.no_grad():
        ref = model(x, t)
        out = model(x, t)
    ok = torch.allclose(ref, out, rtol=rtol, atol=atol)
    return bool(ok)
