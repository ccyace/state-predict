#!/usr/bin/env bash
# W4A8 / W4A7 / W4A6: same vsc_tvar (Student-t var_mle) pipeline as W8A8.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DT_CKPT="state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_simple_v1/checkpoints/ckpt_best.pt"
REF="new_real_images/real47500_vsc2500_fid_stats.npz"

run_bitwidth() {
  local TAG="$1"       # w4a8 | w4a7 | w4a6
  local CALI="$2"
  local WBIT="$3"
  local ABIT="$4"
  local MEAN_CKPT="$5"

  local BASE="PTQD/vsc_tvar/cifar_${TAG}_vsc_tvar_50k"
  local LOGDIR="${BASE}/logs"
  local TRAJ="PTQD/vsc_tvar/traj_${TAG}_eta1_dt_mean_s01_n200.pt"
  local STATS="PTQD/vsc_tvar/vsc_time_stats_eta1_tvar_${TAG}.pt"
  local GEN="${BASE}/vsc_tvar_50k"
  mkdir -p "$LOGDIR"

  echo "========== ${TAG} (W${WBIT}A${ABIT}) =========="

  if [[ ! -f "$TRAJ" ]]; then
    echo "[${TAG} 1/4] collect closed-loop traj (200 x 96) ..."
    python state_aware_temporal_joint/collect_dt_corrected_residual_data.py \
      --output "$TRAJ" \
      --dt_ckpt "$DT_CKPT" \
      --time_residual_ckpt "$MEAN_CKPT" \
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
      --cali_ckpt "$CALI" \
      --weight_bit "$WBIT" \
      --act_bit "$ABIT" \
      --seed 1234 \
      2>&1 | tee "$LOGDIR/collect_traj.log"
  else
    echo "[${TAG} 1/4] skip collect: $TRAJ"
  fi

  echo "[${TAG} 2/4] estimate Student-t var_mle ..."
  python PTQD/estimate_time_variance.py \
    --data "$TRAJ" \
    --output_pt "$STATS" \
    --output_json "${STATS%.pt}.json" \
    --eta 1.0 \
    --timesteps 100 \
    --skip_type quad \
    2>&1 | tee "$LOGDIR/estimate_tvar.log"

  echo "[${TAG} 3/4] sample 50k ..."
  python state_aware_temporal_joint/sample_50k.py \
    --disable_adapter \
    --disable_corrector \
    --dt_only_infer \
    --dt_mode refresh \
    --joint_dt_ckpt "$DT_CKPT" \
    --time_residual_ckpt "$MEAN_CKPT" \
    --time_residual_strength 0.1 \
    --eta 1.0 \
    --timesteps 100 \
    --skip_type quad \
    --dt_eta 0.5 \
    --dt_carry_max 20 \
    --t_cutoff 300 \
    --dt_refresh_n 8 \
    --vsc_stats "$STATS" \
    --vsc_var_field var_mle \
    --vsc_absorb_strength 1.0 \
    --vsc_max_budget_fraction 0.9 \
    --cali_ckpt "$CALI" \
    --weight_bit "$WBIT" \
    --act_bit "$ABIT" \
    --max_images 50000 \
    --batch_size 64 \
    --seed 1234 \
    --skip_fid \
    --output_dir "$GEN" \
    2>&1 | tee "$LOGDIR/sample_vsc_tvar_50k.log"

  echo "[${TAG} 4/4] FID vs ${REF} ..."
  python -m pytorch_fid "$REF" "$GEN" --device cuda:0 \
    2>&1 | tee "$LOGDIR/fid_real47500_vsc2500.log"

  echo "[${TAG}] done -> $LOGDIR/fid_real47500_vsc2500.log"
}

prepare_w4a8_mean() {
  local DT_ONLY="PTQD/vsc_tvar/traj_w4a8_eta1_dt_only_n200.pt"
  local MEAN_DIR="PTQD/vsc_tvar/w4a8_mean_ckpt"
  local MEAN_CKPT="${MEAN_DIR}/ckpt_best.pt"

  if [[ -f "$MEAN_CKPT" ]]; then
    printf '%s\n' "$MEAN_CKPT"
    return
  fi

  mkdir -p PTQD/vsc_tvar/logs
  if [[ ! -f "$DT_ONLY" ]]; then
    echo "[w4a8 prep] collect dt-only traj ..."
    python state_aware_temporal_joint/collect_dt_corrected_residual_data.py \
      --output "$DT_ONLY" \
      --dt_ckpt "$DT_CKPT" \
      --num_trajectories 200 \
      --batch_size 64 \
      --timesteps 100 \
      --skip_type quad \
      --eta 1.0 \
      --dt_eta 0.5 \
      --dt_max 20 \
      --t_cutoff 300 \
      --n_refresh 8 \
      --cali_ckpt cifar_w4a8_ckpt.pth \
      --weight_bit 4 \
      --act_bit 8 \
      --seed 1234 \
      >> PTQD/vsc_tvar/logs/w4a8_collect_dt_only.log 2>&1
  fi

  echo "[w4a8 prep] train eta=1 mean head ..."
  python state_aware_temporal_joint/train_dt_corrected_residual.py \
    --data "$DT_ONLY" \
    --output_dir "$MEAN_DIR" \
    --epochs 20 \
    --batch_size 128 \
    --lr 0.001 \
    >> PTQD/vsc_tvar/logs/w4a8_train_mean.log 2>&1

  printf '%s\n' "$MEAN_CKPT"
}

W4A8_MEAN="PTQD/vsc_tvar/w4a8_mean_ckpt/ckpt_best.pt"
if [[ ! -f "$W4A8_MEAN" ]]; then
  prepare_w4a8_mean >/dev/null
fi
run_bitwidth w4a8 cifar_w4a8_ckpt.pth 4 8 "$W4A8_MEAN"
run_bitwidth w4a7 cifar_w4a7_ckpt.pth 4 7 PTQD/eta1_mean_samplewise_v2/w4a7_run/checkpoints/ckpt_best.pt
run_bitwidth w4a6 cifar_w4a6_ckpt.pth 4 6 PTQD/eta1_mean_samplewise_v2/w4a6_run/checkpoints/ckpt_best.pt

echo "All W4A* vsc_tvar 50k runs finished."
