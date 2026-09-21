#!/usr/bin/env bash
# W4A5: vsc_tvar (Student-t var_mle) 50k — same protocol as W4A6/7/8.
set -euo pipefail
cd /root/autodl-tmp/ODE-scale

DT_CKPT="state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_simple_v1/checkpoints/ckpt_best.pt"
REF="new_real_images/real47500_vsc2500_fid_stats.npz"
MEAN_CKPT="PTQD/eta1_mean_samplewise_v2/w4a5_run/checkpoints/ckpt_best.pt"

TAG=w4a5
CALI=cifar_w4a5_ckpt.pth
WBIT=4
ABIT=5
BASE="PTQD/vsc_tvar/cifar_${TAG}_vsc_tvar_50k"
LOGDIR="${BASE}/logs"
TRAJ="PTQD/vsc_tvar/traj_${TAG}_eta1_dt_mean_s01_n200.pt"
STATS="PTQD/vsc_tvar/vsc_time_stats_eta1_tvar_${TAG}.pt"
GEN="${BASE}/vsc_tvar_50k"
mkdir -p "$LOGDIR" "$GEN"

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
