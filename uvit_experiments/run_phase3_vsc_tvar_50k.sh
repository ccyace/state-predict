#!/usr/bin/env bash
# Phase 3: U-ViT W8A8 vsc_tvar 50k sampling + FID.
set -euo pipefail
cd /root/autodl-tmp/ODE-scale

FP_CKPT="/root/autodl-tmp/ODE-scale/cifar10_uvit_small.pth"
Q_CKPT="uvit_experiments/checkpoints/uvit_w8a8_ckpt.pth"
CALI_DATA="cifar_sd1236_sample2048_allst.pt"
PHASE2="uvit_experiments/outputs/phase2_w8a8"
OUT="uvit_experiments/outputs/phase3_vsc_tvar_50k"
LOGDIR="${OUT}/logs"
mkdir -p "${LOGDIR}"

DT_CKPT="${PHASE2}/dt_ckpt/ckpt_best.pt"
MEAN_CKPT="${PHASE2}/mean_ckpt/ckpt_best.pt"
VSC_STATS="${PHASE2}/vsc_time_stats_eta1_tvar.pt"
FID_REF="new_real_images/real47500_vsc2500_fid_stats.npz"

if [ ! -f "${DT_CKPT}" ] || [ ! -f "${MEAN_CKPT}" ] || [ ! -f "${VSC_STATS}" ]; then
  echo "Phase 2 artifacts missing. Run uvit_experiments/run_phase2_uvit_w8a8.sh first."
  exit 1
fi

echo "=== [1/2] sample 50k vsc_tvar ==="
python state_aware_temporal_joint/sample_50k.py \
  --backbone uvit \
  --fp_ckpt "${FP_CKPT}" \
  --cali_ckpt "${Q_CKPT}" \
  --cali_data_path "${CALI_DATA}" \
  --weight_bit 8 --act_bit 8 --sm_abit 8 \
  --cali_st 10 --cali_n 256 --quant_act --a_sym \
  --disable_adapter \
  --disable_corrector \
  --dt_only_infer \
  --dt_mode refresh \
  --joint_dt_ckpt "${DT_CKPT}" \
  --time_residual_ckpt "${MEAN_CKPT}" \
  --time_residual_strength 0.1 \
  --eta 1.0 \
  --timesteps 100 \
  --skip_type quad \
  --dt_eta 0.5 \
  --dt_carry_max 20 \
  --t_cutoff 300 \
  --dt_refresh_n 8 \
  --vsc_stats "${VSC_STATS}" \
  --vsc_var_field var_mle \
  --vsc_absorb_strength 1.0 \
  --vsc_max_budget_fraction 0.9 \
  --max_images 50000 \
  --batch_size 64 \
  --seed 1234 \
  --fid_ref "${FID_REF}" \
  --fid_log "${LOGDIR}/fid.log" \
  --output_dir "${OUT}" \
  2>&1 | tee "${LOGDIR}/sample_vsc_tvar_50k.log"

echo "=== Phase 3 done -> ${OUT} ==="
