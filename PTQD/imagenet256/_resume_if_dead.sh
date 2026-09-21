#!/usr/bin/env bash
# Resume cin256 W4A8 learned_corr+tvar pipeline from last completed stage.
set -euo pipefail
ROOT="/root/autodl-tmp/ODE-scale"
OUT="${OUT:-PTQD/imagenet256/cin256_w4a8_learned_corr_tvar}"
cd "$ROOT"

if pgrep -f "run_cin256_learned_corr_tvar.sh" >/dev/null 2>&1; then
  echo "ALIVE pipeline still running"
  exit 0
fi
if pgrep -f "collect_ldm_imagenet_traj.py|train_corrector.py|estimate_ldm_vsc|sample_diffusion_ldm.py" >/dev/null 2>&1; then
  echo "ALIVE child still running"
  exit 0
fi

SKIP_COLLECT=0 SKIP_TRAIN=0 SKIP_ESTIMATE=0 SKIP_SAMPLE=0
[[ -f "${OUT}/traj_w4a8.pt" ]] && SKIP_COLLECT=1
[[ -f "${OUT}/learned_corr/ckpt_best.pt" ]] && SKIP_TRAIN=1
[[ -f "${OUT}/vsc_time_stats_eta1_tvar.pt" ]] && SKIP_ESTIMATE=1
# sample done if fid staging has enough or sample log says finished
if [[ -d "${OUT}/fid_staging/learned_vsc_gen" ]] && [[ "$(find "${OUT}/fid_staging/learned_vsc_gen" -type f 2>/dev/null | wc -l)" -ge 9900 ]]; then
  echo "DONE enough samples"
  exit 0
fi

echo "RESUME SKIP_COLLECT=${SKIP_COLLECT} SKIP_TRAIN=${SKIP_TRAIN} SKIP_ESTIMATE=${SKIP_ESTIMATE}"
WBIT=4 ABIT=8 QUANT_ACT=1 SKIP_PTQ=1 SKIP_CALI=1 \
  SKIP_COLLECT="${SKIP_COLLECT}" SKIP_TRAIN="${SKIP_TRAIN}" SKIP_ESTIMATE="${SKIP_ESTIMATE}" \
  CALI_CKPT=/root/ODEscale/Imagenet_W4A8_ckpt.pth \
  OUT="${OUT}" SAMPLE_BATCH=3 N_TRAJ=128 FID_N=10000 \
  nohup bash PTQD/imagenet256/run_cin256_learned_corr_tvar.sh \
  >> "${OUT}/pipeline_nohup.log" 2>&1 &
echo "RESTARTED PID=$!"
