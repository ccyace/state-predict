#!/usr/bin/env bash
# EfficientDM W4A4 (20-step TALSQ) + learned ε-corrector + Student-t VSC (tvar)
#
# Protocol (aligned with prior cin256 corr runs; steps fixed by EfficientDM ckpt):
#   DDIM-20, eta=1.0, CFG=3.0
#   η=1 is intentional: VSC absorbs into DDIM stochastic budget (η=0 → budget≈0).
#
# Usage:
#   bash PTQD/imagenet256/run_cin256_efficientdm_corr_tvar.sh
#   STEP=sample FID_N=10000 bash PTQD/imagenet256/run_cin256_efficientdm_corr_tvar.sh
#   STEP: collect | train | estimate | sample | fid | all

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src/taming-transformers:${ROOT}:${PYTHONPATH:-}"

FP_CKPT="${FP_CKPT:-models/ldm/cin256/model.ckpt}"
LDM_CONFIG="${LDM_CONFIG:-configs/latent-diffusion/cin256-v2.yaml}"
EFFICIENTDM_CKPT="${EFFICIENTDM_CKPT:-}"  # set to EfficientDM W4A4 ckpt path
EFFICIENTDM_STEPS="${EFFICIENTDM_STEPS:-20}"
EFFICIENTDM_ROOT="${EFFICIENTDM_ROOT:-${EFFICIENTDM_HOME:-../EfficientDM}}"

TAG="${TAG:-efficientdm_w4a4}"
OUT="${OUT:-PTQD/imagenet256/cin256_${TAG}_learned_corr_tvar}"
LOGDIR="${OUT}/logs"

DDIM_STEPS="${DDIM_STEPS:-${EFFICIENTDM_STEPS}}"
ETA="${ETA:-1.0}"
CFG_SCALE="${CFG_SCALE:-3.0}"
SEED="${SEED:-1234}"
LINEAR_START="${LINEAR_START:-0.0015}"
LINEAR_END="${LINEAR_END:-0.0195}"
DDIM_SKIP_TYPE="${DDIM_SKIP_TYPE:-uniform}"

# 20-step traj is cheap → more trajectories than W4A8@200 for similar sample count
N_TRAJ="${N_TRAJ:-512}"
COLLECT_BATCH="${COLLECT_BATCH:-2}"
SAMPLE_BATCH="${SAMPLE_BATCH:-3}"
CORR_EPOCHS="${CORR_EPOCHS:-20}"
CORR_BATCH="${CORR_BATCH:-8}"
CORR_T_CUT="${CORR_T_CUT:-999}"
CORR_ALPHA="${CORR_ALPHA:-1.0}"

TRAJ="${OUT}/traj_${TAG}.pt"
CORR_DIR="${OUT}/learned_corr"
CORR_CKPT="${CORR_DIR}/ckpt_best.pt"
VSC_STATS="${OUT}/vsc_time_stats_eta1_tvar.pt"
SAMPLE_LOG="${OUT}/sample_learned_vsc"
FID_STAGING="${OUT}/fid_staging/learned_vsc_gen"

FID_N="${FID_N:-10000}"
FID_REF="${FID_REF:-}"  # e.g. VIRTUAL_imagenet256_labeled.npz

STEP="${STEP:-all}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_ESTIMATE="${SKIP_ESTIMATE:-0}"
SKIP_SAMPLE="${SKIP_SAMPLE:-0}"
SKIP_FID="${SKIP_FID:-0}"

SAMPLE_LDM="scripts/sample_diffusion_ldm.py"
COLLECT_LDM="scripts/collect_ldm_imagenet_traj.py"
ESTIMATE_VSC="scripts/estimate_ldm_vsc_after_corrector.py"
TRAIN_CORR="noise_eps_corr/scripts/train_corrector.py"

mkdir -p "${LOGDIR}" "${CORR_DIR}" "${FID_STAGING}"

PIPELINE_LOG="${LOGDIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

log_step() {
  echo ""
  echo "========================================================================"
  echo "[$(date '+%F %T')] $*"
  echo "========================================================================"
}

require_file() { [[ -f "$1" ]] || { echo "ERROR: missing file: $1" >&2; exit 1; }; }
require_script() { [[ -f "$1" ]] || { echo "ERROR: missing script: $1" >&2; exit 1; }; }
should_run() { [[ "${STEP}" == "all" || "${STEP}" == "$1" ]]; }

if [[ "${DDIM_STEPS}" != "${EFFICIENTDM_STEPS}" ]]; then
  echo "ERROR: DDIM_STEPS=${DDIM_STEPS} must equal EFFICIENTDM_STEPS=${EFFICIENTDM_STEPS} (TALSQ)" >&2
  exit 1
fi

log_step "EfficientDM W4A4 + learned_corr + tvar | steps=${DDIM_STEPS} eta=${ETA} | OUT=${OUT}"
require_file "${FP_CKPT}"
require_file "${EFFICIENTDM_CKPT}"

# ── [1] collect traj ──────────────────────────────────────────────────────────
if should_run collect && [[ "${SKIP_COLLECT}" != "1" ]]; then
  require_script "${COLLECT_LDM}"
  if [[ -f "${TRAJ}" ]]; then
    log_step "[1/4] skip collect — exists: ${TRAJ}"
  else
    log_step "[1/4] collect traj (${N_TRAJ} × ${DDIM_STEPS}, eta=${ETA})"
    python "${COLLECT_LDM}" \
      --fp_ckpt "${FP_CKPT}" \
      --ldm_config "${LDM_CONFIG}" \
      --efficientdm_ckpt "${EFFICIENTDM_CKPT}" \
      --efficientdm_steps "${EFFICIENTDM_STEPS}" \
      --efficientdm_root "${EFFICIENTDM_ROOT}" \
      --n_traj "${N_TRAJ}" \
      --batch_size "${COLLECT_BATCH}" \
      --steps "${DDIM_STEPS}" \
      --eta "${ETA}" \
      --scale "${CFG_SCALE}" \
      --output "${TRAJ}" \
      2>&1 | tee "${LOGDIR}/collect_traj.log"
  fi
else
  log_step "[1/4] skip collect"
fi

# ── [2] train corrector ───────────────────────────────────────────────────────
if should_run train && [[ "${SKIP_TRAIN}" != "1" ]]; then
  require_file "${TRAJ}"
  require_script "${TRAIN_CORR}"
  if [[ -f "${CORR_CKPT}" ]]; then
    log_step "[2/4] skip train — exists: ${CORR_CKPT}"
  else
    log_step "[2/4] train learned corrector"
    python "${TRAIN_CORR}" \
      --data "${TRAJ}" \
      --output_dir "${CORR_DIR}" \
      --epochs "${CORR_EPOCHS}" \
      --batch_size "${CORR_BATCH}" \
      --t_cut "${CORR_T_CUT}" \
      --alpha "${CORR_ALPHA}" \
      --lambda_mse 1.0 --lambda_cos 2.0 --lambda_sr 0.3 \
      2>&1 | tee "${LOGDIR}/train_corrector.log"
  fi
else
  log_step "[2/4] skip train"
fi

# ── [3] estimate VSC (var_mle / tvar) ─────────────────────────────────────────
if should_run estimate && [[ "${SKIP_ESTIMATE}" != "1" ]]; then
  require_file "${TRAJ}"
  require_file "${CORR_CKPT}"
  require_script "${ESTIMATE_VSC}"
  if [[ -f "${VSC_STATS}" ]]; then
    log_step "[3/4] skip estimate — exists: ${VSC_STATS}"
  else
    log_step "[3/4] post-corrector Student-t var_mle (eta=${ETA})"
    python "${ESTIMATE_VSC}" \
      --data "${TRAJ}" \
      --corrector_ckpt "${CORR_CKPT}" \
      --corrector_alpha "${CORR_ALPHA}" \
      --output_traj "${OUT}/traj_${TAG}_after_corr.pt" \
      --output_vsc "${VSC_STATS}" \
      --eta "${ETA}" \
      --timesteps "${DDIM_STEPS}" \
      --skip_type "${DDIM_SKIP_TYPE}" \
      --linear_start "${LINEAR_START}" \
      --linear_end "${LINEAR_END}" \
      2>&1 | tee "${LOGDIR}/estimate_vsc.log"
  fi
else
  log_step "[3/4] skip estimate"
fi

# ── [4] sample ────────────────────────────────────────────────────────────────
if should_run sample && [[ "${SKIP_SAMPLE}" != "1" ]]; then
  [[ -f "${OUT}/fid_n.txt" ]] && FID_N="$(tr -d '[:space:]' < "${OUT}/fid_n.txt")"
  [[ -f "${OUT}/sample_batch.txt" ]] && SAMPLE_BATCH="$(tr -d '[:space:]' < "${OUT}/sample_batch.txt")"
  require_script "${SAMPLE_LDM}"
  require_file "${CORR_CKPT}"
  require_file "${VSC_STATS}"
  log_step "[4/4] sample ${FID_N} (EfficientDM + learned_corr + vsc tvar, eta=${ETA})"
  python "${SAMPLE_LDM}" \
    -r "${FP_CKPT}" \
    --cond --scale "${CFG_SCALE}" \
    -c "${DDIM_STEPS}" -e "${ETA}" --seed "${SEED}" \
    --batch_size "${SAMPLE_BATCH}" -n "${FID_N}" \
    --efficientdm_ckpt "${EFFICIENTDM_CKPT}" \
    --efficientdm_steps "${EFFICIENTDM_STEPS}" \
    --efficientdm_root "${EFFICIENTDM_ROOT}" \
    --enable_learned_noise_corr \
    --learned_corr_ckpt "${CORR_CKPT}" \
    --learned_corr_t_cut "${CORR_T_CUT}" \
    --learned_corr_alpha "${CORR_ALPHA}" \
    --vsc_stats "${VSC_STATS}" \
    --vsc_var_field var_mle \
    --vsc_absorb_strength 1.0 \
    --vsc_max_budget_fraction 0.9 \
    -l "${SAMPLE_LOG}" \
    2>&1 | tee "${LOGDIR}/sample_learned_vsc.log"

  log_step "[4b] stage PNGs → ${FID_STAGING}"
  rm -rf "${FID_STAGING}"
  mkdir -p "${FID_STAGING}"
  find "${SAMPLE_LOG}" -type f \( -name 'sample_*.png' -o -name '*.jpg' \) ! -name 'sample_-*.png' \
    | sort | head -n "${FID_N}" \
    | awk -v dst="${FID_STAGING}" '{ printf "cp \"%s\" \"%s/sample_%06d.png\"\n", $0, dst, NR }' | bash
  echo "staged=$(ls "${FID_STAGING}" | wc -l)"
fi

# ── [5] FID ───────────────────────────────────────────────────────────────────
if should_run fid && [[ "${SKIP_FID}" != "1" ]]; then
  require_file "${FID_REF}"
  if [[ ! -d "${FID_STAGING}" ]] || [[ "$(ls "${FID_STAGING}" 2>/dev/null | wc -l)" -lt 100 ]]; then
    echo "ERROR: FID staging empty/small: ${FID_STAGING}" >&2
    exit 1
  fi
  log_step "[5] FID vs ${FID_REF}"
  python -m pytorch_fid "${FID_REF}" "${FID_STAGING}" --device cuda:0 \
    2>&1 | tee "${LOGDIR}/fid.log"
fi

log_step "done"
echo "  traj:      ${TRAJ}"
echo "  corr:      ${CORR_CKPT}"
echo "  vsc:       ${VSC_STATS}"
echo "  samples:   ${SAMPLE_LOG}"
echo "  log:       ${PIPELINE_LOG}"
