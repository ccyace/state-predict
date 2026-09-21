#!/usr/bin/env bash
# Phase 2: U-ViT W4A8 dt correction + mean residual + tvar stats.
set -euo pipefail
cd /root/autodl-tmp/ODE-scale

FP_CKPT="/root/autodl-tmp/ODE-scale/cifar10_uvit_small.pth"
Q_CKPT="uvit_experiments/checkpoints/uvit_w4a8_ckpt.pth"
CALI_DATA="cifar_sd1236_sample2048_allst.pt"
OUT="uvit_experiments/outputs/phase2_w4a8"
LOGDIR="${OUT}/logs"
mkdir -p "${LOGDIR}"

PTQ=(
  --backbone uvit
  --fp_ckpt "${FP_CKPT}"
  --cali_ckpt "${Q_CKPT}"
  --cali_data_path "${CALI_DATA}"
  --weight_bit 4
  --act_bit 8
  --sm_abit 8
  --cali_st 10
  --cali_n 256
  --quant_act
  --a_sym
)

TRAJ_OPEN="${OUT}/traj_openloop_n1000.pt"
DT_LABELS="${OUT}/dt_labels_update_mse.pt"
DT_CKPT="${OUT}/dt_ckpt/ckpt_best.pt"
TRAJ_DT="${OUT}/traj_dt_only_n1000.pt"
MEAN_CKPT="${OUT}/mean_ckpt/ckpt_best.pt"
TRAJ_MEAN="${OUT}/traj_dt_mean_n200.pt"
VSC_STATS="${OUT}/vsc_time_stats_eta1_tvar.pt"

if [ ! -f "${Q_CKPT}" ]; then
  echo "Missing ${Q_CKPT}; run uvit_experiments/run_phase1_w4a8.sh first."
  exit 1
fi

echo "=== [1/7] collect open-loop trajectories ==="
if [ -f "${TRAJ_OPEN}" ]; then
  echo "skip, exists: ${TRAJ_OPEN}"
else
  python noise_eps_corr/scripts/collect_fullstep_training_data.py \
    "${PTQ[@]}" \
    --num_trajectories 1000 \
    --batch_size 64 \
    --timesteps 100 \
    --skip_type quad \
    --output "${TRAJ_OPEN}" \
    2>&1 | tee "${LOGDIR}/collect_openloop.log"
fi

echo "=== [2/7] label dt* (update_mse) ==="
if [ -f "${DT_LABELS}" ]; then
  echo "skip, exists: ${DT_LABELS}"
else
  python state_aware_temporal_joint/label_dt_eps_oracle.py \
    "${PTQ[@]}" \
    --data "${TRAJ_OPEN}" \
    --output "${DT_LABELS}" \
    --window 40 \
    --n_grid 11 \
    --criterion update_mse \
    --batch_size 256 \
    --max_samples 80000 \
    2>&1 | tee "${LOGDIR}/label_dt.log"
fi

echo "=== [3/7] train dt-only head ==="
if [ -f "${DT_CKPT}" ]; then
  echo "skip, exists: ${DT_CKPT}"
else
  mkdir -p "${OUT}/dt_ckpt"
  python state_aware_temporal_joint/train_dt_eps_oracle.py \
    --data "${TRAJ_OPEN}" \
    --dt_star "${DT_LABELS}" \
    --init_eps_ckpt "" \
    --output_dir "${OUT}/dt_ckpt" \
    --epochs 20 \
    --batch_size 256 \
    --dt_only \
    --loss smooth_l1 \
    2>&1 | tee "${LOGDIR}/train_dt.log"
fi

echo "=== [4/7] collect dt-only closed-loop traj ==="
if [ -f "${TRAJ_DT}" ]; then
  echo "skip, exists: ${TRAJ_DT}"
else
  python state_aware_temporal_joint/collect_dt_corrected_residual_data.py \
    "${PTQ[@]}" \
    --output "${TRAJ_DT}" \
    --dt_ckpt "${DT_CKPT}" \
    --num_trajectories 1000 \
    --batch_size 64 \
    --timesteps 100 \
    --skip_type quad \
    --dt_eta 0.5 \
    --dt_max 20 \
    --t_cutoff 300 \
    --n_refresh 8 \
    --seed 1234 \
    2>&1 | tee "${LOGDIR}/collect_dt_only.log"
fi

echo "=== [5/7] train mean residual head ==="
if [ -f "${MEAN_CKPT}" ]; then
  echo "skip, exists: ${MEAN_CKPT}"
else
  mkdir -p "${OUT}/mean_ckpt"
  python state_aware_temporal_joint/train_dt_corrected_residual.py \
    --data "${TRAJ_DT}" \
    --output_dir "${OUT}/mean_ckpt" \
    --epochs 30 \
    --batch_size 128 \
    --lr 0.001 \
    2>&1 | tee "${LOGDIR}/train_mean.log"
fi

echo "=== [6/7] collect eta=1 traj for tvar ==="
if [ -f "${TRAJ_MEAN}" ]; then
  echo "skip, exists: ${TRAJ_MEAN}"
else
  python state_aware_temporal_joint/collect_dt_corrected_residual_data.py \
    "${PTQ[@]}" \
    --output "${TRAJ_MEAN}" \
    --dt_ckpt "${DT_CKPT}" \
    --time_residual_ckpt "${MEAN_CKPT}" \
    --time_residual_strength 0.1 \
    --num_trajectories 200 \
    --batch_size 64 \
    --timesteps 100 \
    --skip_type quad \
    --eta 1.0 \
    --dt_eta 0.5 \
    --dt_max 20 \
    --t_cutoff 300 \
    --n_refresh 8 \
    --seed 1234 \
    2>&1 | tee "${LOGDIR}/collect_tvar_traj.log"
fi

echo "=== [7/7] estimate per-time variance (var_mle) ==="
python PTQD/estimate_time_variance.py \
  --data "${TRAJ_MEAN}" \
  --output_pt "${VSC_STATS}" \
  --output_json "${VSC_STATS%.pt}.json" \
  --eta 1.0 \
  --timesteps 100 \
  --skip_type quad \
  2>&1 | tee "${LOGDIR}/estimate_tvar.log"

echo "=== Phase 2 W4A8 done ==="
echo "  dt_ckpt=${DT_CKPT}"
echo "  mean_ckpt=${MEAN_CKPT}"
echo "  vsc_stats=${VSC_STATS}"
