#!/usr/bin/env bash
# Supervise cin256 W8-only pipeline; auto-resume on crash.
set -euo pipefail

ROOT="/root/autodl-tmp/ODE-scale"
OUT="${OUT:-PTQD/imagenet256/cin256_w8_learned_corr_tvar}"
INTERVAL="${INTERVAL:-120}"
SUP_LOG="${OUT}/logs/supervisor.log"

cd "${ROOT}"
mkdir -p "${OUT}/logs"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${SUP_LOG}"; }

detect_step() {
  if [[ ! -f "${OUT}/imagenet_w8_ckpt.pth" ]]; then echo "ptq"; return; fi
  if [[ ! -f "${OUT}/traj_w8.pt" ]]; then echo "collect"; return; fi
  if [[ ! -f "${OUT}/learned_corr/ckpt_best.pt" ]]; then echo "train"; return; fi
  if [[ ! -f "${OUT}/vsc_time_stats_eta1_tvar.pt" ]]; then echo "estimate"; return; fi
  if [[ ! -d "${OUT}/fid_staging/learned_vsc_gen" ]] || [[ $(find "${OUT}/fid_staging/learned_vsc_gen" -name '*.png' 2>/dev/null | wc -l) -lt 100 ]]; then
    echo "sample"; return
  fi
  echo "fid"
}

pipeline_running() {
  pgrep -f "bash PTQD/imagenet256/run_cin256_learned_corr_tvar.sh" >/dev/null 2>&1
}

restart_pipeline() {
  local step="$1"
  log "RESTART STEP=${step}"
  SKIP_CALI=1 QUANT_ACT=0 \
    PTQ_CALI_N="${PTQ_CALI_N:-64}" PTQ_CALI_BATCH="${PTQ_CALI_BATCH:-4}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    STEP="${step}" OUT="${OUT}" \
    nohup bash PTQD/imagenet256/run_cin256_learned_corr_tvar.sh \
    >> "${OUT}/pipeline_nohup.log" 2>&1 &
  log "restarted pid=$!"
}

log "supervisor start OUT=${OUT} interval=${INTERVAL}s"

while true; do
  if grep -q "DONE →" "${OUT}/pipeline_nohup.log" 2>/dev/null; then
    log "pipeline DONE — exit supervisor"
    exit 0
  fi

  if pipeline_running; then
    # heartbeat
    if pgrep -f "sample_diffusion_ldm.py -r models/ldm/cin256" >/dev/null 2>&1; then
      stage="ptq"
    elif pgrep -f "collect_ldm_imagenet_traj.py" >/dev/null 2>&1; then
      stage="collect"
    elif pgrep -f "train_corrector.py" >/dev/null 2>&1; then
      stage="train"
    elif pgrep -f "estimate_ldm_vsc_after_corrector.py" >/dev/null 2>&1; then
      stage="estimate"
    elif pgrep -f "sample_diffusion_ldm.py.*enable_learned_noise_corr" >/dev/null 2>&1; then
      stage="sample"
    elif pgrep -f "pytorch_fid" >/dev/null 2>&1; then
      stage="fid"
    else
      stage="shell"
    fi
    log "OK running stage=${stage}"
  else
    if tail -20 "${OUT}/pipeline_nohup.log" 2>/dev/null | grep -qE "Killed|Traceback|ERROR:"; then
      log "ERROR detected in pipeline log"
      tail -5 "${OUT}/pipeline_nohup.log" >> "${SUP_LOG}"
    fi
    step="$(detect_step)"
    if [[ "${step}" == "fid" ]] && [[ -f "${OUT}/logs/fid.log" ]]; then
      log "fid log exists — marking done"
      exit 0
    fi
    restart_pipeline "${step}"
  fi
  sleep "${INTERVAL}"
done
