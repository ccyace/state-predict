"""
Channel-wise noise correction (噪声矫正.pdf §1.1–1.3).

Offline: fit per-timestep affine a_t,c, b_t,c and optional strength ratio r_t
on same-input pairs (eps^q, eps^fp). Online: affine -> strength/direction ->
lambda blend -> norm clip; DDIM uses x_{t-1} = A_t x^q + c_t eps_corr.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class NoiseCorrectionMeta:
    lambda_max: float = 1.0
    kappa: float = 0.1
    r_min: float = 0.8
    r_max: float = 1.25
    use_strength_ratio: bool = True
    eps: float = 1e-8
    # Inference schedule: t < t_cut -> lambda=0; mid/high t use lambda_mid/high on PDF lambda.
    # t_cut < 0 disables truncation (legacy PDF-only blend).
    t_cut: int = -1
    t_high: int = 200
    lambda_mid: float = 1.0
    lambda_high: float = 1.0
    alpha: float = 1.0
    strength_only: bool = False


def _fit_affine_channel(
    eps_q: torch.Tensor,
    eps_fp: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel affine on [N, C, H, W] -> a[C], b[C]."""
    n, c, h, w = eps_q.shape
    q = eps_q.permute(1, 0, 2, 3).reshape(c, -1)
    fp = eps_fp.permute(1, 0, 2, 3).reshape(c, -1)
    mu_q = q.mean(dim=1)
    mu_fp = fp.mean(dim=1)
    q_c = q - mu_q.unsqueeze(1)
    fp_c = fp - mu_fp.unsqueeze(1)
    var_q = (q_c * q_c).mean(dim=1)
    cov = (q_c * fp_c).mean(dim=1)
    a = cov / (var_q + eps)
    b = mu_fp - a * mu_q
    return a.float(), b.float()


def _fit_strength_ratio(eps_q: torch.Tensor, eps_fp: torch.Tensor, eps: float) -> float:
    nq = eps_q.reshape(eps_q.shape[0], -1).norm(dim=1)
    nf = eps_fp.reshape(eps_fp.shape[0], -1).norm(dim=1)
    return float(nf.mean() / (nq.mean() + eps))


@torch.no_grad()
def calibrate_noise_correction(
    quant_model: nn.Module,
    cali_xs: torch.Tensor,
    cali_ts: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int = 32,
    eps: float = 1e-8,
    use_strength_ratio: bool = True,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], NoiseCorrectionMeta]:
    if not hasattr(quant_model, "set_quant_state"):
        raise TypeError("calibrate_noise_correction expects QuantModel")

    quant_model.eval()
    unique_ts = sorted(int(t) for t in torch.unique(cali_ts).tolist())
    params: Dict[int, Dict[str, torch.Tensor]] = {}

    for t_val in unique_ts:
        mask = cali_ts == t_val
        xs_t = cali_xs[mask]
        if xs_t.numel() == 0:
            continue
        eps_q_parts, eps_fp_parts = [], []
        t_tensor = torch.full((batch_size,), t_val, device=device, dtype=torch.float32)
        for start in range(0, xs_t.shape[0], batch_size):
            xb = xs_t[start : start + batch_size].to(device)
            cur_b = xb.shape[0]
            tb = t_tensor[:cur_b]
            quant_model.set_quant_state(weight_quant=True, act_quant=True)
            eps_q_parts.append(quant_model(xb, tb).cpu())
            quant_model.set_quant_state(weight_quant=False, act_quant=False)
            eps_fp_parts.append(quant_model(xb, tb).cpu())

        eq = torch.cat(eps_q_parts, dim=0)
        ef = torch.cat(eps_fp_parts, dim=0)
        a, b = _fit_affine_channel(eq, ef, eps)
        entry: Dict[str, torch.Tensor] = {"a": a, "b": b}
        if use_strength_ratio:
            entry["r"] = torch.tensor(_fit_strength_ratio(eq, ef, eps))
        params[t_val] = entry
        r_str = f" r={float(entry['r']):.4f}" if use_strength_ratio else ""
        logger.info(
            "t=%d: n=%d ||eps_q||=%.4f ||eps_fp||=%.4f%s",
            t_val,
            eq.shape[0],
            float(eq.norm() / max(eq.shape[0], 1)),
            float(ef.norm() / max(ef.shape[0], 1)),
            r_str,
        )

    quant_model.set_quant_state(weight_quant=True, act_quant=True)
    meta = NoiseCorrectionMeta(use_strength_ratio=use_strength_ratio, eps=eps)
    logger.info("Calibrated noise correction for %d timesteps", len(params))
    return params, meta


def save_noise_correction(
    path: str,
    params: Dict[int, Dict[str, torch.Tensor]],
    meta: Optional[NoiseCorrectionMeta] = None,
    extra_meta: Optional[dict] = None,
) -> None:
    meta = meta or NoiseCorrectionMeta()
    payload = {
        "meta": {
            "source": "noise_correction_calibration",
            "lambda_max": meta.lambda_max,
            "kappa": meta.kappa,
            "r_min": meta.r_min,
            "r_max": meta.r_max,
            "use_strength_ratio": meta.use_strength_ratio,
            "eps": meta.eps,
            **(extra_meta or {}),
        },
        "params": {},
    }
    for t, d in params.items():
        entry = {"a": d["a"].tolist(), "b": d["b"].tolist()}
        if "r" in d:
            entry["r"] = float(d["r"].item() if torch.is_tensor(d["r"]) else d["r"])
        payload["params"][str(int(t))] = entry

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved noise correction to %s (%d timesteps)", path, len(params))


def load_noise_correction(path: str) -> Tuple[Dict[int, Dict[str, torch.Tensor]], NoiseCorrectionMeta]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw_meta = payload.get("meta", {})
    meta = NoiseCorrectionMeta(
        lambda_max=float(raw_meta.get("lambda_max", 1.0)),
        kappa=float(raw_meta.get("kappa", 0.1)),
        r_min=float(raw_meta.get("r_min", 0.8)),
        r_max=float(raw_meta.get("r_max", 1.25)),
        use_strength_ratio=bool(raw_meta.get("use_strength_ratio", True)),
        eps=float(raw_meta.get("eps", 1e-8)),
    )
    params = {}
    for k, v in payload["params"].items():
        params[int(k)] = {
            "a": torch.tensor(v["a"], dtype=torch.float32),
            "b": torch.tensor(v["b"], dtype=torch.float32),
        }
        if "r" in v:
            params[int(k)]["r"] = torch.tensor(float(v["r"]), dtype=torch.float32)
    logger.info("Loaded noise correction from %s (%d timesteps)", path, len(params))
    return params, meta


class NoiseCorrector:
    """Apply PDF §1.1–1.3 at inference (no state-error recursion)."""

    def __init__(
        self,
        params: Dict[int, Dict[str, torch.Tensor]],
        meta: NoiseCorrectionMeta,
    ):
        self.params = params
        self.meta = meta

    def _lookup(self, t_idx: int, ref: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        d = self.params.get(int(t_idx))
        if d is None:
            return None
        a = d["a"].to(device=ref.device, dtype=ref.dtype).view(1, -1, 1, 1)
        b = d["b"].to(device=ref.device, dtype=ref.dtype).view(1, -1, 1, 1)
        r = None
        if "r" in d and self.meta.use_strength_ratio:
            r = d["r"].to(device=ref.device, dtype=ref.dtype)
        return a, b, r

    def _lambda_t(self, t_idx: int, alpha_bar: torch.Tensor) -> torch.Tensor:
        t_val = int(t_idx)
        if self.meta.t_cut >= 0 and t_val < self.meta.t_cut:
            return torch.zeros_like(alpha_bar).view(-1, 1, 1, 1)

        one_m = (1.0 - alpha_bar).clamp(min=0.0)
        lam = self.meta.lambda_max * one_m / (one_m + self.meta.kappa)

        if self.meta.t_cut >= 0:
            if self.meta.t_high >= 0 and t_val >= self.meta.t_high:
                lam = lam * self.meta.lambda_high
            else:
                lam = lam * self.meta.lambda_mid

        return lam.view(-1, 1, 1, 1)

    def correct(
        self,
        eps_q: torch.Tensor,
        t_idx: int,
        alpha_bar: torch.Tensor,
        xt: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        looked = self._lookup(t_idx, eps_q)
        if looked is None:
            return eps_q
        a, b, r = looked
        eps = self.meta.eps

        if self.meta.strength_only:
            v = eps_q
        else:
            v = a * eps_q + b

        use_r = r is not None and (self.meta.use_strength_ratio or self.meta.strength_only)
        if use_r:
            v_norm = v.reshape(v.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1).clamp(min=eps)
            eq_norm = eps_q.reshape(eps_q.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1).clamp(min=eps)
            eps_sd = r * eq_norm * v / v_norm
        else:
            eps_sd = v

        lam = self._lambda_t(t_idx, alpha_bar)
        blend = (self.meta.alpha * lam).clamp(min=0.0, max=1.0)
        eps_corr = eps_q + blend * (eps_sd - eps_q)

        corr_norm = eps_corr.reshape(eps_corr.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1).clamp(min=eps)
        q_norm = eps_q.reshape(eps_q.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1).clamp(min=eps)
        ratio = corr_norm / q_norm
        scale = torch.clamp(self.meta.r_max / ratio, max=1.0)
        scale = torch.where(ratio < self.meta.r_min, self.meta.r_min / ratio, scale)
        eps_corr = eps_corr * scale
        return eps_corr
