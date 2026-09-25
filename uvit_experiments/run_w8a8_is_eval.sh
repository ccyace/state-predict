#!/usr/bin/env bash
# Re-sample W8A8 experiments (PNG deleted after FID) and compute Inception Score.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FP_CKPT="${FP_CKPT:-cifar10_uvit_small.pth}"
Q_CKPT="uvit_experiments/checkpoints/uvit_w8a8_ckpt.pth"
CALI_DATA="cifar_sd1236_sample2048_allst.pt"
PHASE2="uvit_experiments/outputs/phase2_w8a8"
OUT="uvit_experiments/outputs/w8a8_is_eval"
LOGDIR="${OUT}/logs"
SUMMARY="${OUT}/summary.txt"
mkdir -p "${LOGDIR}"

DT_CKPT="${PHASE2}/dt_ckpt/ckpt_best.pt"
MEAN_CKPT="${PHASE2}/mean_ckpt/ckpt_best.pt"
VSC_STATS="${PHASE2}/vsc_time_stats_eta1_tvar.pt"
FID_REF="new_real_images/real47500_vsc2500_fid_stats.npz"

run_is() {
  local tag="$1"
  local img_dir="$2"
  local is_out="${OUT}/${tag}_is.txt"
  echo "=== IS: ${tag} ===" | tee -a "${SUMMARY}"
  python compute_IS_torch_fidelity.py \
    --path "${img_dir}" \
    --batch_size 64 \
    --out "${is_out}" \
    2>&1 | tee "${LOGDIR}/is_${tag}.log"
  grep -E "IS mean=" "${is_out}" | tee -a "${SUMMARY}" || true
  echo "[cleanup] rm ${tag} png" | tee -a "${SUMMARY}"
  find "${img_dir}" -maxdepth 1 -name '*.png' -delete 2>/dev/null || true
}

echo "W8A8 IS evaluation $(date -Iseconds)" > "${SUMMARY}"

echo "=== [1/2] resample Phase 1 W8A8 naive (50k) ==="
P1="uvit_experiments/outputs/phase1_w8a8_ddim100_eta1_50k"
mkdir -p "${P1}"
python uvit_experiments/sample_fp_ddim.py \
  --ckpt "${FP_CKPT}" \
  --cali_ckpt "${Q_CKPT}" \
  --cali_data_path "${CALI_DATA}" \
  --weight_bit 8 --act_bit 8 --sm_abit 8 \
  --output_dir "${P1}" \
  --log_dir "${LOGDIR}/phase1_resample" \
  --num_images 50000 \
  --batch_size 128 \
  --timesteps 100 \
  --skip_type quad \
  --eta 1.0 \
  --seed 1234 \
  --skip_fid \
  2>&1 | tee "${LOGDIR}/resample_phase1_w8a8.log"
run_is "phase1_w8a8_naive" "${P1}"

echo "=== [2/2] resample Phase 3 W8A8 vsc_tvar (50k) ==="
P3="uvit_experiments/outputs/phase3_vsc_tvar_50k"
mkdir -p "${P3}"
python state_aware_temporal_joint/sample_50k.py \
  --backbone uvit \
  --fp_ckpt "${FP_CKPT}" \
  --cali_ckpt "${Q_CKPT}" \
  --cali_data_path "${CALI_DATA}" \
  --weight_bit 8 --act_bit 8 --sm_abit 8 \
  --cali_st 10 --cali_n 256 --quant_act --a_sym \
  --disable_adapter --disable_corrector \
  --dt_only_infer --dt_mode refresh \
  --joint_dt_ckpt "${DT_CKPT}" \
  --time_residual_ckpt "${MEAN_CKPT}" \
  --time_residual_strength 0.1 \
  --eta 1.0 \
  --timesteps 100 --skip_type quad \
  --dt_eta 0.5 --dt_carry_max 20 \
  --t_cutoff 300 --dt_refresh_n 8 \
  --vsc_stats "${VSC_STATS}" \
  --vsc_var_field var_mle \
  --vsc_absorb_strength 1.0 \
  --vsc_max_budget_fraction 0.9 \
  --max_images 50000 \
  --batch_size 64 \
  --seed 1234 \
  --skip_fid \
  --output_dir "${P3}" \
  2>&1 | tee "${LOGDIR}/resample_phase3_vsc_tvar.log"
run_is "phase3_vsc_tvar" "${P3}"

echo "=== W8A8 IS eval done $(date -Iseconds) ===" | tee -a "${SUMMARY}"
cat "${SUMMARY}"
