#!/usr/bin/env bash
# Calibrate U-ViT W4A6, eval on existing shared FP x_t, redraw density with A6.
set -euo pipefail
cd /root/autodl-tmp/ODE-scale

FP_CKPT="${FP_CKPT:-/root/autodl-tmp/ODE-scale/cifar10_uvit_small.pth}"
CALI_DATA="${CALI_DATA:-cifar_sd1236_sample2048_allst.pt}"
W4A6_CKPT="${W4A6_CKPT:-uvit_experiments/checkpoints/uvit_w4a6_ckpt.pth}"
PAIRED_IN="${PAIRED_IN:-uvit_experiments/outputs/error_distributions/paired_fp_rollout_w4_w8.pt}"
PAIRED_OUT="${PAIRED_OUT:-uvit_experiments/outputs/error_distributions/paired_fp_rollout_w4a8_w8a8_w4a6.pt}"
OUT_DIR="uvit_experiments/outputs/error_distributions"
LOGDIR="${OUT_DIR}/logs"
CALI_ITERS="${CALI_ITERS:-512}"
CALI_ITERS_A="${CALI_ITERS_A:-256}"
SKIP_CALIB="${SKIP_CALIB:-0}"

mkdir -p "${OUT_DIR}" "${LOGDIR}"

if [[ ! -f "${PAIRED_IN}" ]]; then
  echo "missing shared-x_t file: ${PAIRED_IN}"
  echo "run first: bash uvit_experiments/run_fair_w4_vs_w8_error_dist.sh"
  exit 1
fi

if [[ "${SKIP_CALIB}" != "1" && ! -f "${W4A6_CKPT}" ]]; then
  echo "=== [1/3] calibrate U-ViT W4A6 (same iters as W4A8/W8A8) ==="
  python uvit_experiments/calibrate_w8a8.py \
    --ckpt "${FP_CKPT}" \
    --output_ckpt "${W4A6_CKPT}" \
    --cali_data_path "${CALI_DATA}" \
    --weight_bit 4 --act_bit 6 --sm_abit 6 \
    --cali_iters "${CALI_ITERS}" --cali_iters_a "${CALI_ITERS_A}" \
    --quant_act --a_sym \
    2>&1 | tee "${LOGDIR}/calibrate_w4a6.log"
else
  echo "skip calibrate (SKIP_CALIB=${SKIP_CALIB} or exists: ${W4A6_CKPT})"
fi

echo "=== [2/3] eval W4A6 on shared FP x_t ==="
python uvit_experiments/eval_extra_bitwidth_on_paired_xt.py \
  --paired "${PAIRED_IN}" \
  --tag w4a6 \
  --weight_bit 4 --act_bit 6 --sm_abit 6 \
  --cali_ckpt "${W4A6_CKPT}" \
  --fp_ckpt "${FP_CKPT}" \
  --cali_data_path "${CALI_DATA}" \
  --output "${PAIRED_OUT}" \
  2>&1 | tee "${LOGDIR}/eval_w4a6_on_paired.log"

echo "=== [3/3] plot W4A8 / W8A8 / W4A6 ==="
python uvit_experiments/plot_fair_shared_xt_error.py \
  --data "${PAIRED_OUT}" \
  --tags w4a8,w8a8,w4a6 \
  --stem quant_error_w4a8_w8a8_w4a6_fair_shared_xt \
  --title 'U-ViT-S/2: W4A8 / W8A8 / W4A6 on shared FP-rollout $x_t$'

echo "done:"
echo "  ${OUT_DIR}/quant_error_w4a8_w8a8_w4a6_fair_shared_xt.{png,svg,pdf}"
