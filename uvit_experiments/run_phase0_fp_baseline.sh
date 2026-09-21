#!/usr/bin/env bash
# Phase 0: U-ViT FP baseline — DDIM 100 quad, eta=1, 50k, FID.
set -euo pipefail
cd /root/autodl-tmp/ODE-scale

echo "=== [0] Download checkpoint ==="
bash uvit_experiments/download_ckpt.sh

echo "=== [1] Sample 50k FP ==="
python uvit_experiments/sample_fp_ddim.py \
  --output_dir uvit_experiments/outputs/phase0_fp_ddim100_eta1_50k \
  --log_dir uvit_experiments/outputs/phase0_fp_ddim100_eta1_50k/logs \
  --num_images 50000 \
  --batch_size 128 \
  --timesteps 100 \
  --skip_type quad \
  --eta 1.0 \
  --seed 1234 \
  --fid_ref new_real_images/real47500_vsc2500_fid_stats.npz \
  2>&1 | tee uvit_experiments/outputs/phase0_fp_ddim100_eta1_50k/logs/sample.log

echo "=== Phase 0 done ==="
