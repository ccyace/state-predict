#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT=PTQD/fid50k_vsc_tvar
mkdir -p "$OUT/logs"

TRAJ=PTQD/traj_eta1_dt_mean_s01_n200.pt
STATS=PTQD/vsc_time_stats_eta1_tvar.pt
GEN="$OUT/vsc_tvar_50k"

if [ ! -f "$TRAJ" ]; then
  echo "[1/4] collecting closed-loop traj (200 x 96 states) ..."
  python state_aware_temporal_joint/collect_dt_corrected_residual_data.py \
    --output "$TRAJ" \
    --dt_ckpt state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_simple_v1/checkpoints/ckpt_best.pt \
    --time_residual_ckpt state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/checkpoints/ckpt_best.pt \
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
    2>&1 | tee "$OUT/logs/collect_traj.log"
else
  echo "[1/4] skip collect, exists: $TRAJ"
fi

echo "[2/4] estimating Student-t var_mle per time ..."
python PTQD/estimate_time_variance.py \
  --data "$TRAJ" \
  --output_pt "$STATS" \
  --output_json "${STATS%.pt}.json" \
  --eta 1.0 \
  --timesteps 100 \
  --skip_type quad \
  2>&1 | tee "$OUT/logs/estimate_tvar.log"

echo "[3/4] sampling 50k with VSC(var_mle) ..."
python state_aware_temporal_joint/sample_50k.py \
  --disable_adapter \
  --disable_corrector \
  --dt_only_infer \
  --dt_mode refresh \
  --joint_dt_ckpt state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_simple_v1/checkpoints/ckpt_best.pt \
  --time_residual_ckpt state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/checkpoints/ckpt_best.pt \
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
  --max_images 50000 \
  --batch_size 64 \
  --seed 1234 \
  --skip_fid \
  --output_dir "$GEN" \
  2>&1 | tee "$OUT/logs/sample_vsc_tvar_50k.log"

echo "[4/4] FID ..."
python -m pytorch_fid new_real_images/cifar10_python.npz "$GEN" --device cuda:0 \
  2>&1 | tee "$OUT/logs/fid_vsc_tvar_50k.log"

echo "done -> $OUT/logs/fid_vsc_tvar_50k.log"
