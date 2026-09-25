#!/usr/bin/env bash
# U-ViT W4A6 naive baseline: no δt / mean / VSC / adapter / corrector + 50k + FID.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FP_CKPT="${FP_CKPT:-cifar10_uvit_small.pth}"
Q_CKPT="uvit_experiments/checkpoints/uvit_w4a6_ckpt.pth"
CALI_DATA="cifar_sd1236_sample2048_allst.pt"
GEN="uvit_experiments/outputs/phase3_w4a6_naive_50k"
FID_REF="${FID_REF:-origin-cifar-10-python_fid_mu_sigma.npz}"
mkdir -p "${GEN}/logs"

if [[ ! -f "${Q_CKPT}" ]]; then
  echo "Missing ${Q_CKPT}"
  exit 1
fi

echo "=== [1/2] sample 50k W4A6 naive (no corrector) ==="
python state_aware_temporal_joint/sample_50k.py \
  --backbone uvit \
  --fp_ckpt "${FP_CKPT}" \
  --cali_ckpt "${Q_CKPT}" \
  --cali_data_path "${CALI_DATA}" \
  --weight_bit 4 \
  --act_bit 6 \
  --sm_abit 6 \
  --cali_st 10 \
  --cali_n 256 \
  --quant_act \
  --a_sym \
  --disable_adapter \
  --disable_corrector \
  --eta 1.0 \
  --timesteps 100 \
  --skip_type quad \
  --max_images 50000 \
  --batch_size 64 \
  --seed 1234 \
  --skip_fid \
  --output_dir "${GEN}" \
  2>&1 | tee "${GEN}/logs/sample_naive_50k.log"

echo "=== [2/2] FID vs ${FID_REF} ==="
python -m pytorch_fid "${FID_REF}" "${GEN}" --device cuda:0 \
  2>&1 | tee "${GEN}/logs/fid_vs_origin_cifar10_python_mu_sigma.log"

python3 - <<PY
import re, json, pathlib
log = pathlib.Path("${GEN}/logs/fid_vs_origin_cifar10_python_mu_sigma.log").read_text()
m = re.search(r"FID:\s*([0-9.]+)", log)
fid = float(m.group(1)) if m else None
path = pathlib.Path("${GEN}/manifest_fid.json")
path.write_text(json.dumps({
  "setting": "uvit_w4a6_naive",
  "n_images": 50000,
  "corrector": None,
  "dt_enabled": False,
  "time_residual": False,
  "vsc": False,
  "eta": 1.0,
  "timesteps": 100,
  "skip_type": "quad",
  "weight_bit": 4,
  "act_bit": 6,
  "fid_ref": "${FID_REF}",
  "fid": fid,
}, indent=2) + "\n")
print("wrote", path, "fid=", fid)
PY

echo "=== W4A6 naive baseline done -> ${GEN} ==="
