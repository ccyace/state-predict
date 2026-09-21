#!/usr/bin/env bash
# Wait for W8A8 Phase 3 to finish, free disk, then run U-ViT W4A8 full pipeline.
set -euo pipefail
cd /root/autodl-tmp/ODE-scale

W8A8_P3="uvit_experiments/outputs/phase3_vsc_tvar_50k"
W8A8_P3_LOG="${W8A8_P3}/pipeline.nohup"
W8A8_FID_LOG="${W8A8_P3}/logs/fid.log"
ORCH_LOG="uvit_experiments/outputs/w4a8_after_w8a8.nohup"
mkdir -p uvit_experiments/outputs

exec > >(tee -a "${ORCH_LOG}") 2>&1

echo "========== wait for W8A8 Phase 3 $(date -Iseconds) =========="

wait_for_w8a8_phase3() {
  while true; do
    if [ -f "${W8A8_FID_LOG}" ] && grep -qE "FID:\s*[0-9]" "${W8A8_FID_LOG}" 2>/dev/null; then
      echo "[wait] W8A8 Phase 3 FID done"
      return 0
    fi
    if [ -f "${W8A8_P3_LOG}" ] && grep -q "Phase 3 done" "${W8A8_P3_LOG}" 2>/dev/null; then
      echo "[wait] W8A8 Phase 3 script finished"
      return 0
    fi
    if pgrep -f "run_phase3_vsc_tvar_50k.sh" >/dev/null 2>&1 \
      || pgrep -f "sample_50k.py.*phase3_vsc_tvar_50k" >/dev/null 2>&1; then
      n=$(find "${W8A8_P3}" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
      echo "[wait] W8A8 Phase 3 running ... png=${n}/50000 $(date -Iseconds)"
      sleep 120
      continue
    fi
    n=$(find "${W8A8_P3}" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
    if [ "${n}" -ge 50000 ]; then
      echo "[wait] 50000 png present, proceed"
      return 0
    fi
    echo "[wait] no active Phase 3 process (png=${n}); sleeping 60s"
    sleep 60
  done
}

wait_for_w8a8_phase3

if [ -f "${W8A8_FID_LOG}" ]; then
  echo "[W8A8 Phase 3 FID] $(grep -E 'FID:' "${W8A8_FID_LOG}" | tail -1 || true)"
fi

echo "========== disk cleanup before W4A8 $(date -Iseconds) =========="
df -h /root/autodl-tmp | tail -1

# Free space: drop W8A8 large traj + phase1/phase3 png (keep ckpts, logs, README)
P2W8="uvit_experiments/outputs/phase2_w8a8"
for f in "${P2W8}/traj_openloop_n1000.pt" "${P2W8}/traj_dt_only_n1000.pt" "${P2W8}/traj_dt_mean_n200.pt"; do
  if [ -f "$f" ]; then
    echo "[cleanup] rm $f"
    rm -f "$f"
  fi
done
if [ -d "uvit_experiments/outputs/phase1_w8a8_ddim100_eta1_50k" ]; then
  echo "[cleanup] rm phase1 W8A8 png"
  find uvit_experiments/outputs/phase1_w8a8_ddim100_eta1_50k -maxdepth 1 -name '*.png' -delete 2>/dev/null || true
fi
if [ -d "${W8A8_P3}" ]; then
  echo "[cleanup] rm W8A8 Phase 3 png (keep logs)"
  find "${W8A8_P3}" -maxdepth 1 -name '*.png' -delete 2>/dev/null || true
fi

df -h /root/autodl-tmp | tail -1

echo "========== start W4A8 full pipeline $(date -Iseconds) =========="
bash uvit_experiments/run_w4a8_full_pipeline.sh

echo "========== all done $(date -Iseconds) =========="
