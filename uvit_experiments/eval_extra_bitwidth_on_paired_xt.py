#!/usr/bin/env python
"""Eval an extra quantized ckpt on an existing FP-rollout paired .pt (reuse shared x_t).

Example (add W4A6 onto the fair W4/W8 file):
  python uvit_experiments/eval_extra_bitwidth_on_paired_xt.py \\
    --paired uvit_experiments/outputs/error_distributions/paired_fp_rollout_w4_w8.pt \\
    --tag w4a6 --weight_bit 4 --act_bit 6 --sm_abit 6 \\
    --cali_ckpt uvit_experiments/checkpoints/uvit_w4a6_ckpt.pth \\
    --output uvit_experiments/outputs/error_distributions/paired_fp_rollout_w4a8_w8a8_w4a6.pt
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "uvit_experiments"))

import torch

from uvit_loader import DEFAULT_CKPT, load_uvit_quant


@torch.no_grad()
def eval_quant_eq(qnn, x_cpu, t_cpu, device, batch: int = 64):
    qnn.eval()
    qnn.set_quant_state(True, True)
    eqs = []
    n = x_cpu.shape[0]
    for st in range(0, n, batch):
        x = x_cpu[st : st + batch].float().to(device)
        t = t_cpu[st : st + batch].float().to(device)
        eqs.append(qnn(x, t).detach().cpu().half())
        if (st // batch) % 50 == 0:
            print(f"    eval {min(st + batch, n)}/{n}", flush=True)
    return torch.cat(eqs, 0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paired", required=True, help="existing paired_fp_rollout_*.pt with x/t/ef")
    p.add_argument("--tag", default="w4a6", help="key suffix, stores eq_<tag> and e_<tag>")
    p.add_argument("--cali_ckpt", required=True)
    p.add_argument("--fp_ckpt", default=DEFAULT_CKPT)
    p.add_argument("--cali_data_path", default="cifar_sd1236_sample2048_allst.pt")
    p.add_argument("--weight_bit", type=int, default=4)
    p.add_argument("--act_bit", type=int, default=6)
    p.add_argument("--sm_abit", type=int, default=6)
    p.add_argument("--cali_st", type=int, default=10)
    p.add_argument("--cali_n", type=int, default=256)
    p.add_argument("--eval_batch", type=int, default=64)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tag.lower()

    print(f"Loading paired states: {args.paired}", flush=True)
    d = torch.load(args.paired, map_location="cpu")
    for k in ("x", "t", "ef"):
        if k not in d:
            raise KeyError(f"paired file missing '{k}'")

    print(
        f"Loading Q {tag} W{args.weight_bit}A{args.act_bit} from {args.cali_ckpt} ...",
        flush=True,
    )
    qnn = load_uvit_quant(
        cali_ckpt=args.cali_ckpt,
        cali_data_path=args.cali_data_path,
        device=device,
        fp_ckpt=args.fp_ckpt,
        weight_bit=args.weight_bit,
        act_bit=args.act_bit,
        sm_abit=args.sm_abit,
        cali_st=args.cali_st,
        cali_n=args.cali_n,
        quant_act=True,
        a_sym=True,
    )
    eq = eval_quant_eq(qnn, d["x"], d["t"], device, batch=args.eval_batch)
    del qnn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ef = d["ef"].float()
    e = (ef - eq.float()).half()
    out = dict(d)
    out[f"eq_{tag}"] = eq
    out[f"e_{tag}"] = e
    meta = dict(out.get("meta") or {})
    meta[f"{tag}_ckpt"] = os.path.abspath(args.cali_ckpt)
    meta[f"{tag}_weight_bit"] = args.weight_bit
    meta[f"{tag}_act_bit"] = args.act_bit
    meta[f"{tag}_sm_abit"] = args.sm_abit
    out["meta"] = meta

    v = e.float().reshape(-1)
    if v.numel() > 2_000_000:
        g = torch.Generator().manual_seed(0)
        v = v[torch.randperm(v.numel(), generator=g)[:2_000_000]]
    stats = {
        "mean": float(v.mean()),
        "std": float(v.std(unbiased=True)),
        "mse": float(v.square().mean()),
        "abs_mean": float(v.abs().mean()),
        "q001": float(torch.quantile(v, 0.001)),
        "q999": float(torch.quantile(v, 0.999)),
    }
    print(tag, json.dumps(stats, indent=2), flush=True)

    path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, path)
    with open(path.replace(".pt", f"_{tag}_summary.json"), "w") as f:
        json.dump({"tag": tag, "summary": stats, "meta": meta}, f, indent=2)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
