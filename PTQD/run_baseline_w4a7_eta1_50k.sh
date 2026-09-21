#!/usr/bin/env bash
# W4A7 η=1 baseline 50k: pure PTQ DDIM, no dt/mean/VSC/corrector.
set -euo pipefail
cd /root/autodl-tmp/ODE-scale

OUT=PTQD/baseline_w4a7_eta1_50k
LOG=PTQD/logs/baseline_w4a7_eta1_50k.log
REF=new_real_images/real47500_vsc2500_fid_stats.npz
mkdir -p PTQD/logs "$OUT"

echo "[1/2] sample 50k baseline (W4A7, eta=1, seed=1234) ..."
python state_aware_temporal_joint/sample_50k.py \
  --disable_adapter \
  --disable_corrector \
  --cali_ckpt cifar_w4a7_ckpt.pth \
  --weight_bit 4 \
  --act_bit 7 \
  --sm_abit 8 \
  --eta 1.0 \
  --timesteps 100 \
  --skip_type quad \
  --max_images 50000 \
  --batch_size 64 \
  --seed 1234 \
  --skip_fid \
  --output_dir "$OUT" \
  2>&1 | tee "$LOG"

echo "[2/2] FID vs $REF ..."
python -m pytorch_fid "$REF" "$OUT" --device cuda:0 \
  2>&1 | tee PTQD/logs/fid_baseline_w4a7_eta1_50k.log

echo "done -> PTQD/logs/fid_baseline_w4a7_eta1_50k.log"
