#!/usr/bin/env python
"""Queue B6 after the running five-way ablation, then compute its 10k FID."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRENT_PID = 342362
OUT = ROOT / "state_aware_temporal_joint/output/ablation_eta0_10k/b6_fp_temporal"


def main():
    while Path(f"/proc/{CURRENT_PID}").exists():
        print("Waiting for five-way ablation to release GPU ...", flush=True)
        time.sleep(30)
    OUT.mkdir(parents=True, exist_ok=True)
    sample = [
        sys.executable, "state_aware_temporal_joint/sample_50k.py",
        "--max_images", "10000", "--batch_size", "384", "--eta", "0",
        "--seed", "1234", "--output_dir", str(OUT),
        "--disable_adapter", "--disable_corrector", "--fp_temporal_path",
    ]
    subprocess.run(sample, cwd=ROOT, check=True)
    stats = OUT / "fid_stats_10k.npz"
    subprocess.run([
        sys.executable, "-m", "pytorch_fid", str(OUT), str(stats),
        "--save-stats", "--batch-size", "128", "--num-workers", "8", "--device", "cuda",
    ], cwd=ROOT, check=True)
    proc = subprocess.run([
        sys.executable, "-m", "pytorch_fid", "new_real_images/cifar10_python.npz",
        str(stats), "--device", "cpu",
    ], cwd=ROOT, check=True, text=True, capture_output=True)
    fid = float(re.search(r"FID:\s+([0-9.eE+-]+)", proc.stdout).group(1))
    with open(OUT / "result.json", "w", encoding="utf-8") as f:
        json.dump({"name": "b6_fp_temporal", "images": 10000, "eta": 0.0, "seed": 1234, "fid": fid}, f, indent=2)
    print(f"B6 FID={fid}", flush=True)


if __name__ == "__main__":
    main()
