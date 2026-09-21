#!/usr/bin/env bash
# Resume W4A8 from Phase 2 (Phase 1 already done: FID 14.7438).
set -euo pipefail
cd /root/autodl-tmp/ODE-scale

LOG="uvit_experiments/outputs/w4a8_resume.nohup"
mkdir -p uvit_experiments/outputs

exec > >(tee -a "${LOG}") 2>&1

echo "========== W4A8 resume $(date -Iseconds) =========="
echo "Phase 1 skip (ckpt + 50k + FID=14.7438 already done)"

bash uvit_experiments/run_phase2_uvit_w4a8.sh
bash uvit_experiments/run_phase3_w4a8_vsc_tvar_50k.sh

echo "========== W4A8 resume done $(date -Iseconds) =========="
