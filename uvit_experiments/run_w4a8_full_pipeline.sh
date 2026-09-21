#!/usr/bin/env bash
# U-ViT W4A8 full pipeline: Phase 1 (PTQ+50k) -> Phase 2 (dt/mean/tvar) -> Phase 3 (vsc_tvar 50k).
set -euo pipefail
cd /root/autodl-tmp/ODE-scale

ROOT_LOG="uvit_experiments/outputs/w4a8_full_pipeline.nohup"
mkdir -p uvit_experiments/outputs

exec > >(tee -a "${ROOT_LOG}") 2>&1

echo "========== U-ViT W4A8 full pipeline start $(date -Iseconds) =========="

bash uvit_experiments/run_phase1_w4a8.sh
bash uvit_experiments/run_phase2_uvit_w4a8.sh
bash uvit_experiments/run_phase3_w4a8_vsc_tvar_50k.sh

echo "========== U-ViT W4A8 full pipeline done $(date -Iseconds) =========="
