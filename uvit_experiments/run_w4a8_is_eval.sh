#!/usr/bin/env bash
# Compute IS for existing W4A8 50k PNG (no resampling).
set -euo pipefail
cd /root/autodl-tmp/ODE-scale

OUT="uvit_experiments/outputs/w4a8_is_eval"
LOGDIR="${OUT}/logs"
mkdir -p "${LOGDIR}"
SUMMARY="${OUT}/summary.txt"

run_is() {
  local tag="$1"
  local img_dir="$2"
  local n
  n=$(find "${img_dir}" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
  if [ "${n}" -lt 50000 ]; then
    echo "[skip] ${tag}: only ${n}/50000 png in ${img_dir}" | tee -a "${SUMMARY}"
    return 1
  fi
  echo "=== IS: ${tag} (${n} images) ===" | tee -a "${SUMMARY}"
  python compute_IS_torch_fidelity.py \
    --path "${img_dir}" \
    --batch_size 64 \
    --out "${OUT}/${tag}_is.txt" \
    2>&1 | tee "${LOGDIR}/is_${tag}.log"
  grep -E "IS mean=" "${OUT}/${tag}_is.txt" | tee -a "${SUMMARY}" || true
}

echo "W4A8 IS evaluation $(date -Iseconds)" > "${SUMMARY}"

run_is "phase1_w4a8_naive" "uvit_experiments/outputs/phase1_w4a8_ddim100_eta1_50k"
run_is "phase3_w4a8_vsc_tvar" "uvit_experiments/outputs/phase3_w4a8_vsc_tvar_50k"

echo "=== W4A8 IS done $(date -Iseconds) ===" | tee -a "${SUMMARY}"
cat "${SUMMARY}"
