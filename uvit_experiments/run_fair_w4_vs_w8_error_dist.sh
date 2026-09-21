#!/usr/bin/env bash
# Fair U-ViT W4A8 vs W8A8 raw-error density on shared FP-rollout x_t.
# Optional: re-calibrate both ckpts with identical BRECQ iters first.
set -euo pipefail
cd /root/autodl-tmp/ODE-scale

FP_CKPT="${FP_CKPT:-/root/autodl-tmp/ODE-scale/cifar10_uvit_small.pth}"
CALI_DATA="${CALI_DATA:-cifar_sd1236_sample2048_allst.pt}"
W4_CKPT="${W4_CKPT:-uvit_experiments/checkpoints/uvit_w4a8_ckpt.pth}"
W8_CKPT="${W8_CKPT:-uvit_experiments/checkpoints/uvit_w8a8_ckpt.pth}"
OUT_DIR="uvit_experiments/outputs/error_distributions"
PAIRED="${OUT_DIR}/paired_fp_rollout_w4_w8.pt"
N_TRAJ="${N_TRAJ:-200}"
RECALIBRATE="${RECALIBRATE:-0}"
CALI_ITERS="${CALI_ITERS:-512}"
CALI_ITERS_A="${CALI_ITERS_A:-256}"

mkdir -p "${OUT_DIR}"

if [[ "${RECALIBRATE}" == "1" ]]; then
  echo "=== re-calibrate W8A8 / W4A8 with identical iters ==="
  python uvit_experiments/calibrate_w8a8.py \
    --ckpt "${FP_CKPT}" \
    --output_ckpt "${W8_CKPT}" \
    --cali_data_path "${CALI_DATA}" \
    --weight_bit 8 --act_bit 8 --sm_abit 8 \
    --cali_iters "${CALI_ITERS}" --cali_iters_a "${CALI_ITERS_A}" \
    --quant_act --a_sym
  python uvit_experiments/calibrate_w8a8.py \
    --ckpt "${FP_CKPT}" \
    --output_ckpt "${W4_CKPT}" \
    --cali_data_path "${CALI_DATA}" \
    --weight_bit 4 --act_bit 8 --sm_abit 8 \
    --cali_iters "${CALI_ITERS}" --cali_iters_a "${CALI_ITERS_A}" \
    --quant_act --a_sym
fi

echo "=== [1/2] collect paired errors on shared FP x_t (n=${N_TRAJ}) ==="
python uvit_experiments/collect_paired_weight_error_fp_rollout.py \
  --fp_ckpt "${FP_CKPT}" \
  --w4_ckpt "${W4_CKPT}" \
  --w8_ckpt "${W8_CKPT}" \
  --cali_data_path "${CALI_DATA}" \
  --num_trajectories "${N_TRAJ}" \
  --batch_size 64 \
  --eval_batch 64 \
  --timesteps 100 \
  --skip_type quad \
  --eta 0.0 \
  --seed 1234 \
  --output "${PAIRED}"

echo "=== [2/2] plot fair density ==="
python uvit_experiments/plot_weight_bit_error_fair.py "${PAIRED}"

echo "done:"
echo "  ${OUT_DIR}/quant_error_w4_vs_w8_fair_shared_xt.{png,svg,pdf}"
echo "  ${OUT_DIR}/quant_error_w4_vs_w8_fair_summary.json"
