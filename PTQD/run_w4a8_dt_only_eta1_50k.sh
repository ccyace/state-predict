#!/usr/bin/env bash
# CIFAR-10 W4A8: DDIM-100, η=1, dt-head only (no ε-mean / VSC), 50k samples.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-PTQD/w4a8_dt_only_eta1_50k}"
LOGDIR="${OUT}/logs"
mkdir -p "$LOGDIR" PTQD/logs

python state_aware_temporal_joint/sample_50k.py \
  --disable_adapter \
  --disable_corrector \
  --dt_only_infer \
  --dt_mode refresh \
  --joint_dt_ckpt state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_simple_v1/checkpoints/ckpt_best.pt \
  --cali_ckpt cifar_w4a8_ckpt.pth \
  --weight_bit 4 --act_bit 8 --sm_abit 8 \
  --quant_act --a_sym --split \
  --eta 1.0 \
  --timesteps 100 \
  --skip_type quad \
  --dt_eta 0.5 \
  --dt_carry_max 20 \
  --t_cutoff 300 \
  --dt_refresh_n 8 \
  --max_images 50000 \
  --batch_size 64 \
  --seed 1234 \
  --skip_fid \
  --output_dir "$OUT" \
  2>&1 | tee "$LOGDIR/sample_w4a8_dt_only_eta1_50k.log"

REF="${REF:-origin-cifar-10-python_fid_mu_sigma.npz}"
echo "[FID] vs $REF ..."
python -m pytorch_fid "$REF" "$OUT" --device cuda:0 \
  2>&1 | tee "$LOGDIR/fid_vs_origin_cifar10_python_mu_sigma.log"
