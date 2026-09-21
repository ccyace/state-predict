#!/usr/bin/env bash
# W8A8 CIFAR: BRECQ rebuild (cali_n=512, cali_iters_a=7000) -> 50k sample -> FID
set -euo pipefail
cd /root/autodl-tmp/ODE-scale

OUT_ROOT=output/cifar_w8a8_cali_n512_iters_a7000
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"
PIPELINE_LOG="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

REF=origin-cifar-10-python_fid_mu_sigma.npz
if [[ ! -f "${REF}" ]]; then
  python - <<'PY'
import numpy as np
z = np.load("origin-cifar-10-python_fid_stats.npz")
np.savez("origin-cifar-10-python_fid_mu_sigma.npz", mu=z["mu"], sigma=z["sigma"])
print("created origin-cifar-10-python_fid_mu_sigma.npz")
PY
fi

echo "==== $(date) START sample+cali ===="
python scripts/sample_diffusion_ddim.py \
  --config configs/cifar10.yml \
  --use_pretrained \
  --timesteps 100 \
  --eta 0 \
  --skip_type quad \
  --ptq \
  --quant_mode qdiff \
  --weight_bit 8 \
  --quant_act \
  --act_bit 8 \
  --a_sym \
  --split \
  --cali_st 20 \
  --cali_batch_size 32 \
  --cali_n 512 \
  --cali_iters_a 7000 \
  --cali_data_path /root/autodl-tmp/ODE-scale/cifar_sd1236_sample2048_allst.pt \
  --max_images 50000 \
  --seed 1234 \
  -l "${OUT_ROOT}"
echo "==== $(date) SAMPLE DONE ===="

# Newest run dir under OUT_ROOT/samples/<timestamp>/img
IMG_DIR=$(find "${OUT_ROOT}/samples" -type d -name img | sort | tail -1)
if [[ -z "${IMG_DIR}" ]]; then
  echo "ERROR: no img/ directory found under ${OUT_ROOT}/samples"
  exit 1
fi
N_PNG=$(find "${IMG_DIR}" -name '*.png' | wc -l)
echo "img_dir=${IMG_DIR}  n_png=${N_PNG}"
if [[ "${N_PNG}" -lt 50000 ]]; then
  echo "WARNING: expected 50000 images, got ${N_PNG}"
fi

FID_LOG="${LOG_DIR}/fid_50k.log"
SUMMARY="${OUT_ROOT}/fid_summary.txt"
echo "==== $(date) START FID ===="
set +e
python -m pytorch_fid "${REF}" "${IMG_DIR}" --device cuda:0 2>&1 | tee "${FID_LOG}"
FID_RC=${PIPESTATUS[0]}
set -e

FID_LINE=$(grep -E 'FID:|Frechet' "${FID_LOG}" | tail -1 || true)
{
  echo "run=$(date -Iseconds)"
  echo "img_dir=${IMG_DIR}"
  echo "n_png=${N_PNG}"
  echo "ref=${REF}"
  echo "fid_rc=${FID_RC}"
  echo "${FID_LINE}"
} | tee "${SUMMARY}"

# Also copy ckpt next to OUT_ROOT if present
CKPT=$(find "${OUT_ROOT}/samples" -name ckpt.pth | sort | tail -1 || true)
if [[ -n "${CKPT}" ]]; then
  cp -f "${CKPT}" "${OUT_ROOT}/cifar_w8a8_cali_n512_iters_a7000_ckpt.pth"
  echo "saved ckpt -> ${OUT_ROOT}/cifar_w8a8_cali_n512_iters_a7000_ckpt.pth"
fi

echo "==== $(date) ALL DONE ===="
echo "pipeline_log=${PIPELINE_LOG}"
echo "summary=${SUMMARY}"
