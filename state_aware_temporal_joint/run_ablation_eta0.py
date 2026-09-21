#!/usr/bin/env python
"""Run the five controlled eta=0 ablations and compute 10k FID."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "state_aware_temporal_joint" / "output" / "ablation_eta0_10k"
REAL = ROOT / "new_real_images" / "cifar10_python.npz"
SAMPLE = ROOT / "state_aware_temporal_joint" / "sample_50k.py"
OLD = ROOT / "noise_eps_corr" / "w8a8_corr_ckpt_best.pt"
NEW = ROOT / "state_aware_temporal_joint" / "runs" / "corrector_w8a8_stage_c" / "ckpt_best.pt"

CONFIGS = [
    ("b1_quant", ["--disable_adapter", "--disable_corrector"]),
    ("b2_old_corrector", ["--disable_adapter", "--corrector_ckpt", str(OLD)]),
    ("b3_adapter", ["--disable_corrector"]),
    ("b4_adapter_old_corrector", ["--corrector_ckpt", str(OLD)]),
    ("b5_adapter_new_corrector", ["--corrector_ckpt", str(NEW)]),
]


def run(cmd):
    print("RUN", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for name, extra in CONFIGS:
        folder = OUT / name
        count = len(list(folder.glob("*.png"))) if folder.exists() else 0
        if count < 10000:
            run([
                sys.executable, str(SAMPLE), "--max_images", "10000",
                "--batch_size", "384", "--eta", "0", "--seed", "1234",
                "--output_dir", str(folder), *extra,
            ])
        stats = folder / "fid_stats_10k.npz"
        if not stats.exists():
            run([
                sys.executable, "-m", "pytorch_fid", str(folder), str(stats),
                "--save-stats", "--batch-size", "128", "--num-workers", "8",
                "--device", "cuda",
            ])
        proc = subprocess.run(
            [sys.executable, "-m", "pytorch_fid", str(REAL), str(stats), "--device", "cpu"],
            cwd=ROOT, check=True, text=True, capture_output=True,
        )
        match = re.search(r"FID:\s+([0-9.eE+-]+)", proc.stdout)
        if not match:
            raise RuntimeError(proc.stdout)
        fid = float(match.group(1))
        rec = {"name": name, "images": 10000, "eta": 0.0, "seed": 1234, "fid": fid}
        results.append(rec)
        print("RESULT", rec, flush=True)
        with open(OUT / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    print("ALL RESULTS", json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
