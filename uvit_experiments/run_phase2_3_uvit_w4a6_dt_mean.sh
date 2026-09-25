#!/usr/bin/env bash
# U-ViT W4A6: train mean residual corrector (on dt-corrected traj) + sample 50k + FID.
# Reuses W4A8 δt head (same weight bit); trains W4A6-specific mean corrector.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FP_CKPT="${FP_CKPT:-cifar10_uvit_small.pth}"
Q_CKPT="uvit_experiments/checkpoints/uvit_w4a6_ckpt.pth"
CALI_DATA="cifar_sd1236_sample2048_allst.pt"
DT_CKPT="${DT_CKPT:-uvit_experiments/outputs/phase2_w4a8/dt_ckpt/ckpt_best.pt}"
OUT="uvit_experiments/outputs/phase2_w4a6"
GEN="uvit_experiments/outputs/phase3_w4a6_dt_mean_50k"
LOGDIR="${OUT}/logs"
FID_REF="${FID_REF:-origin-cifar-10-python_fid_mu_sigma.npz}"
N_TRAJ="${N_TRAJ:-500}"
mkdir -p "${LOGDIR}" "${OUT}/mean_ckpt" "${GEN}/logs"

PTQ=(
  --backbone uvit
  --fp_ckpt "${FP_CKPT}"
  --cali_ckpt "${Q_CKPT}"
  --cali_data_path "${CALI_DATA}"
  --weight_bit 4
  --act_bit 6
  --sm_abit 6
  --cali_st 10
  --cali_n 256
  --quant_act
  --a_sym
)

TRAJ_DT="${OUT}/traj_dt_only_n${N_TRAJ}.pt"
MEAN_CKPT="${OUT}/mean_ckpt/ckpt_best.pt"

if [[ ! -f "${Q_CKPT}" ]]; then
  echo "Missing ${Q_CKPT}"
  exit 1
fi
if [[ ! -f "${DT_CKPT}" ]]; then
  echo "Missing dt ckpt ${DT_CKPT}"
  exit 1
fi

echo "=== [1/4] collect dt-only closed-loop traj (W4A6, n=${N_TRAJ}) ==="
if [[ -f "${TRAJ_DT}" ]]; then
  echo "skip, exists: ${TRAJ_DT}"
else
  python state_aware_temporal_joint/collect_dt_corrected_residual_data.py \
    "${PTQ[@]}" \
    --output "${TRAJ_DT}" \
    --dt_ckpt "${DT_CKPT}" \
    --num_trajectories "${N_TRAJ}" \
    --batch_size 64 \
    --timesteps 100 \
    --skip_type quad \
    --eta 0.0 \
    --dt_eta 0.5 \
    --dt_max 20 \
    --t_cutoff 300 \
    --n_refresh 8 \
    --seed 1234 \
    2>&1 | tee "${LOGDIR}/collect_dt_only.log"
fi

echo "=== [2/4] train mean residual corrector ==="
if [[ -f "${MEAN_CKPT}" ]]; then
  echo "skip, exists: ${MEAN_CKPT}"
else
  python state_aware_temporal_joint/train_dt_corrected_residual.py \
    --data "${TRAJ_DT}" \
    --output_dir "${OUT}/mean_ckpt" \
    --epochs 30 \
    --batch_size 128 \
    --lr 0.001 \
    2>&1 | tee "${LOGDIR}/train_mean.log"
fi

# free traj before 50k sampling
if [[ -f "${TRAJ_DT}" ]]; then
  echo "[cleanup] rm ${TRAJ_DT}"
  rm -f "${TRAJ_DT}"
fi

echo "=== [3/4] sample 50k (dt + mean corrector, no VSC) ==="
python state_aware_temporal_joint/sample_50k.py \
  "${PTQ[@]}" \
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
  --max_images 50000 \
  --batch_size 64 \
  --seed 1234 \
  --skip_fid \
  --output_dir "${GEN}" \
  2>&1 | tee "${GEN}/logs/sample_dt_mean_50k.log"

echo "=== [4/4] FID vs ${FID_REF} ==="
python -m pytorch_fid "${FID_REF}" "${GEN}" --device cuda:0 \
  2>&1 | tee "${GEN}/logs/fid_vs_origin_cifar10_python_mu_sigma.log"

python3 - <<PY
import re, json, pathlib
log = pathlib.Path("${GEN}/logs/fid_vs_origin_cifar10_python_mu_sigma.log").read_text()
m = re.search(r"FID:\s*([0-9.]+)", log)
fid = float(m.group(1)) if m else None
path = pathlib.Path("${GEN}/manifest_fid.json")
path.write_text(json.dumps({
  "setting": "uvit_w4a6_dt_mean",
  "n_images": 50000,
  "dt_ckpt": "${DT_CKPT}",
  "mean_ckpt": "${MEAN_CKPT}",
  "time_residual_strength": 0.1,
  "fid_ref": "${FID_REF}",
  "fid": fid,
}, indent=2) + "\n")
print("wrote", path, "fid=", fid)
PY

echo "=== W4A6 dt+mean done -> ${GEN} (FID in logs / manifest_fid.json) ==="
