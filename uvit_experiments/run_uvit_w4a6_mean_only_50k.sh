#!/usr/bin/env bash
# U-ViT W4A6: mean residual corrector ONLY (no δt) + 50k sample + FID.
set -euo pipefail
cd /root/autodl-tmp/ODE-scale

FP_CKPT="/root/autodl-tmp/ODE-scale/cifar10_uvit_small.pth"
Q_CKPT="uvit_experiments/checkpoints/uvit_w4a6_ckpt.pth"
CALI_DATA="cifar_sd1236_sample2048_allst.pt"
OUT="uvit_experiments/outputs/phase2_w4a6_mean_only"
GEN="uvit_experiments/outputs/phase3_w4a6_mean_only_50k"
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

TRAJ="${OUT}/traj_openloop_n${N_TRAJ}.pt"
MEAN_CKPT="${OUT}/mean_ckpt/ckpt_best.pt"

if [[ ! -f "${Q_CKPT}" ]]; then
  echo "Missing ${Q_CKPT}"
  exit 1
fi

echo "=== [1/4] collect open-loop traj (no dt, W4A6, n=${N_TRAJ}) ==="
if [[ -f "${TRAJ}" ]]; then
  echo "skip, exists: ${TRAJ}"
else
  # empty --dt_ckpt => no refresh; still writes t_nom/t_corr/is_refresh for mean trainer
  python state_aware_temporal_joint/collect_dt_corrected_residual_data.py \
    "${PTQ[@]}" \
    --output "${TRAJ}" \
    --dt_ckpt "" \
    --num_trajectories "${N_TRAJ}" \
    --batch_size 64 \
    --timesteps 100 \
    --skip_type quad \
    --eta 0.0 \
    --seed 1234 \
    2>&1 | tee "${LOGDIR}/collect_openloop.log"
fi

echo "=== [2/4] train mean residual corrector ==="
if [[ -f "${MEAN_CKPT}" ]]; then
  echo "skip, exists: ${MEAN_CKPT}"
else
  python state_aware_temporal_joint/train_dt_corrected_residual.py \
    --data "${TRAJ}" \
    --output_dir "${OUT}/mean_ckpt" \
    --epochs 30 \
    --batch_size 128 \
    --lr 0.001 \
    2>&1 | tee "${LOGDIR}/train_mean.log"
fi

if [[ -f "${TRAJ}" ]]; then
  echo "[cleanup] rm ${TRAJ}"
  rm -f "${TRAJ}"
fi

echo "=== [3/4] sample 50k (mean corrector only, no dt / no VSC) ==="
python state_aware_temporal_joint/sample_50k.py \
  "${PTQ[@]}" \
  --disable_adapter \
  --disable_corrector \
  --time_residual_ckpt "${MEAN_CKPT}" \
  --time_residual_strength 0.1 \
  --eta 1.0 \
  --timesteps 100 \
  --skip_type quad \
  --max_images 50000 \
  --batch_size 64 \
  --seed 1234 \
  --skip_fid \
  --output_dir "${GEN}" \
  2>&1 | tee "${GEN}/logs/sample_mean_only_50k.log"

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
  "setting": "uvit_w4a6_mean_only",
  "n_images": 50000,
  "mean_ckpt": "${MEAN_CKPT}",
  "time_residual_strength": 0.1,
  "dt_enabled": False,
  "fid_ref": "${FID_REF}",
  "fid": fid,
}, indent=2) + "\n")
print("wrote", path, "fid=", fid)
PY

echo "=== W4A6 mean-only done -> ${GEN} ==="
