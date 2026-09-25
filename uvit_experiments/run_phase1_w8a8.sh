#!/usr/bin/env bash
# Phase 1: U-ViT W8A8 PTQ + 50k DDIM sampling + FID.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FP_CKPT="${FP_CKPT:-cifar10_uvit_small.pth}"
CALI_DATA="cifar_sd1236_sample2048_allst.pt"
Q_CKPT="uvit_experiments/checkpoints/uvit_w8a8_ckpt.pth"
OUT="uvit_experiments/outputs/phase1_w8a8_ddim100_eta1_50k"
LOGDIR="${OUT}/logs"

mkdir -p uvit_experiments/checkpoints "${LOGDIR}"

echo "=== [1/2] W8A8 calibration ==="
python uvit_experiments/calibrate_w8a8.py \
  --ckpt "${FP_CKPT}" \
  --output_ckpt "${Q_CKPT}" \
  --cali_data_path "${CALI_DATA}" \
  --cali_st 10 \
  --cali_n 256 \
  --cali_batch_size 32 \
  --cali_iters 512 \
  --cali_iters_a 256 \
  2>&1 | tee "${LOGDIR}/calibrate.log"

echo "=== [2/2] Sample 50k W8A8 ==="
python uvit_experiments/sample_fp_ddim.py \
  --ckpt "${FP_CKPT}" \
  --cali_ckpt "${Q_CKPT}" \
  --cali_data_path "${CALI_DATA}" \
  --output_dir "${OUT}" \
  --log_dir "${LOGDIR}" \
  --num_images 50000 \
  --batch_size 128 \
  --timesteps 100 \
  --skip_type quad \
  --eta 1.0 \
  --seed 1234 \
  --fid_ref new_real_images/real47500_vsc2500_fid_stats.npz \
  2>&1 | tee "${LOGDIR}/sample.log"

echo "=== Phase 1 done ==="
