#!/usr/bin/env bash
# Wait for W4A8 full pipeline, then run W8A8 IS evaluation (requires GPU re-sampling).
set -euo pipefail
cd /root/autodl-tmp/ODE-scale

ORCH_LOG="uvit_experiments/outputs/w8a8_is_after_w4a8.nohup"
mkdir -p uvit_experiments/outputs

exec > >(tee -a "${ORCH_LOG}") 2>&1

echo "========== wait for W4A8 pipeline $(date -Iseconds) =========="

while pgrep -f "run_w4a8_full_pipeline.sh" >/dev/null 2>&1 \
  || pgrep -f "run_phase2_uvit_w4a8.sh" >/dev/null 2>&1 \
  || pgrep -f "run_phase3_w4a8_vsc_tvar_50k.sh" >/dev/null 2>&1 \
  || pgrep -f "run_phase1_w4a8.sh" >/dev/null 2>&1; do
  echo "[wait] W4A8 pipeline active ... $(date -Iseconds)"
  sleep 180
done

# Extra guard: wait for W4A8 sample/collect python jobs
while pgrep -f "uvit_w4a8_ckpt.pth" >/dev/null 2>&1; do
  echo "[wait] W4A8 python job ... $(date -Iseconds)"
  sleep 120
done

if grep -q "All W4A8" uvit_experiments/outputs/w4a8_full_pipeline.nohup 2>/dev/null \
  || grep -q "Phase 3 W4A8 done" uvit_experiments/outputs/w4a8_full_pipeline.nohup 2>/dev/null \
  || grep -q "U-ViT W4A8 full pipeline done" uvit_experiments/outputs/w4a8_full_pipeline.nohup 2>/dev/null; then
  echo "[wait] W4A8 pipeline finished"
else
  echo "[wait] W4A8 orchestrator idle; proceed with IS eval"
fi

echo "========== start W8A8 IS eval $(date -Iseconds) =========="
bash uvit_experiments/run_w8a8_is_eval.sh

echo "========== W8A8 IS orchestrator done $(date -Iseconds) =========="
