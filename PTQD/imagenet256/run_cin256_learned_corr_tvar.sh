#!/usr/bin/env bash
# ImageNet 256 (LDM cin256): PTQ + learned ε-corrector + Student-t VSC
# Standalone pipeline — all artifacts live under PTQD/imagenet256/<run_dir>/.
#
# Protocol:
#   - FP: models/ldm/cin256/model.ckpt
#   - PTQ: Q-Diffusion qdiff (local cali → imagenet_w8_ckpt.pth or w8a8)
#   - DDIM-200, eta=1.0, CFG=3.0
#
# Usage:
#   QUANT_ACT=0 bash PTQD/imagenet256/run_cin256_learned_corr_tvar.sh   # W8-only (default)
#   QUANT_ACT=1 bash PTQD/imagenet256/run_cin256_learned_corr_tvar.sh   # W8A8
#   STEP=ptq SKIP_CALI=1 bash PTQD/imagenet256/run_cin256_learned_corr_tvar.sh
#   STEP=sample FID_N=3000 bash PTQD/imagenet256/run_cin256_learned_corr_tvar.sh
#
# STEP: cali | ptq | collect | train | estimate | sample | fid | all

set -euo pipefail

ROOT="/root/autodl-tmp/ODE-scale"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src/taming-transformers:${ROOT}:${PYTHONPATH:-}"

FP_CKPT="${FP_CKPT:-models/ldm/cin256/model.ckpt}"
LDM_CONFIG="${LDM_CONFIG:-configs/latent-diffusion/cin256-v2.yaml}"
WBIT="${WBIT:-8}"
ABIT="${ABIT:-8}"
QUANT_ACT="${QUANT_ACT:-0}"

if [[ "${QUANT_ACT}" == "1" ]]; then
  TAG="${TAG:-w${WBIT}a${ABIT}}"
else
  TAG="${TAG:-w${WBIT}}"
fi
OUT="${OUT:-PTQD/imagenet256/cin256_${TAG}_learned_corr_tvar}"
LOGDIR="${OUT}/logs"

# All cin256 artifacts stay under OUT (do not write to repo root).
if [[ "${QUANT_ACT}" == "1" ]]; then
  CALI_CKPT="${CALI_CKPT:-${OUT}/imagenet_w8a8_ckpt.pth}"
else
  CALI_CKPT="${CALI_CKPT:-${OUT}/imagenet_w8_ckpt.pth}"
fi
CALI_DATA="${CALI_DATA:-${OUT}/imagenet_cali.pt}"

DDIM_STEPS="${DDIM_STEPS:-200}"
ETA="${ETA:-1.0}"
CFG_SCALE="${CFG_SCALE:-3.0}"
SEED="${SEED:-1234}"
LINEAR_START="${LINEAR_START:-0.0015}"
LINEAR_END="${LINEAR_END:-0.0195}"
DDIM_SKIP_TYPE="${DDIM_SKIP_TYPE:-uniform}"

N_TRAJ="${N_TRAJ:-128}"
COLLECT_BATCH="${COLLECT_BATCH:-2}"
# FID sampling batch (12GB GPU: 2 safe, 3 often OK for W4A8+CFG+corrector; decoupled from collect)
SAMPLE_BATCH="${SAMPLE_BATCH:-3}"
CORR_EPOCHS="${CORR_EPOCHS:-20}"
CORR_BATCH="${CORR_BATCH:-8}"
CORR_T_CUT="${CORR_T_CUT:-999}"
CORR_ALPHA="${CORR_ALPHA:-1.0}"

# cali generation (FP DDIM collect for PTQ)
CALI_N_TRAJ="${CALI_N_TRAJ:-512}"
CALI_BATCH="${CALI_BATCH:-4}"

# PTQ memory knobs (cgroup ~48GB RAM; weight recon caches act — keep cali_n low)
if [[ "${QUANT_ACT:-0}" == "1" ]]; then
  PTQ_CALI_N="${PTQ_CALI_N:-128}"
  PTQ_CALI_BATCH="${PTQ_CALI_BATCH:-4}"
else
  PTQ_CALI_N="${PTQ_CALI_N:-64}"
  PTQ_CALI_BATCH="${PTQ_CALI_BATCH:-4}"
fi
PTQ_RUNNING_BATCH="${PTQ_RUNNING_BATCH:-4}"
PTQ_CALI_ITERS_A="${PTQ_CALI_ITERS_A:-2000}"

TRAJ="${OUT}/traj_${TAG}.pt"
CORR_DIR="${OUT}/learned_corr"
CORR_CKPT="${CORR_DIR}/ckpt_best.pt"
VSC_STATS="${OUT}/vsc_time_stats_eta1_tvar.pt"
SAMPLE_LOG="${OUT}/sample_learned_vsc"
FID_STAGING="${OUT}/fid_staging/learned_vsc_gen"

FID_N="${FID_N:-3000}"
FID_REF="${FID_REF:-/root/autodl-tmp/dit-hsq/checkpoints/VIRTUAL_imagenet256_labeled.npz}"

STEP="${STEP:-all}"
SKIP_CALI="${SKIP_CALI:-0}"
SKIP_PTQ="${SKIP_PTQ:-0}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_ESTIMATE="${SKIP_ESTIMATE:-0}"
SKIP_SAMPLE="${SKIP_SAMPLE:-0}"
SKIP_FID="${SKIP_FID:-0}"

SAMPLE_LDM="scripts/sample_diffusion_ldm.py"
GEN_CALI="scripts/generate_ldm_cali_data.py"
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

PTQ_FLAGS=(
  --ptq --resume --cond
  --weight_bit "${WBIT}"
  --cali_ckpt "${CALI_CKPT}"
  --scale "${CFG_SCALE}"
)
if [[ "${QUANT_ACT}" == "1" ]]; then
  PTQ_FLAGS+=( --quant_act --act_bit "${ABIT}" --a_sym --a_min_max --running_stat )
fi

COLLECT_QUANT_FLAGS=( --a_sym )
if [[ "${QUANT_ACT}" == "1" ]]; then
  COLLECT_QUANT_FLAGS+=( --quant_act --act_bit "${ABIT}" )
else
  COLLECT_QUANT_FLAGS+=( --no_quant_act )
fi

SAMPLE_FLAGS=(
  -r "${FP_CKPT}"
  --batch_size "${SAMPLE_BATCH}"
  -c "${DDIM_STEPS}"
  -e "${ETA}"
  --seed "${SEED}"
  --scale "${CFG_SCALE}"
  --cond
  "${PTQ_FLAGS[@]}"
)

log_step "cin256 PTQ + learned_corr + tvar | TAG=${TAG} QUANT_ACT=${QUANT_ACT} | OUT=${OUT}"

require_file "${FP_CKPT}"
mkdir -p "${OUT}"

# Reuse legacy root cali file without duplicating 5GB data.
LEGACY_CALI="${ROOT}/imagenet_cali.pt"
if [[ ! -f "${CALI_DATA}" && -f "${LEGACY_CALI}" ]]; then
  ln -sf "${LEGACY_CALI}" "${CALI_DATA}"
  echo "Linked ${CALI_DATA} -> ${LEGACY_CALI}"
fi

# ── [0a] generate FP calibration data ─────────────────────────────────────────
if should_run cali && [[ "${SKIP_CALI}" != "1" ]]; then
  require_script "${GEN_CALI}"
  if [[ -f "${CALI_DATA}" ]]; then
    log_step "[0a] skip cali — exists: ${CALI_DATA}"
  else
    log_step "[0a] generate LDM cali data (${CALI_N_TRAJ} traj × ${DDIM_STEPS} steps) → ${CALI_DATA}"
    python "${GEN_CALI}" \
      --fp_ckpt "${FP_CKPT}" \
      --ldm_config "${LDM_CONFIG}" \
      --output "${CALI_DATA}" \
      --n_traj "${CALI_N_TRAJ}" \
      --batch_size "${CALI_BATCH}" \
      --steps "${DDIM_STEPS}" \
      --eta 0.0 \
      --scale "${CFG_SCALE}" \
      --cond \
      2>&1 | tee "${LOGDIR}/generate_cali.log"
  fi
else
  log_step "[0a] skip cali (STEP=${STEP} SKIP_CALI=${SKIP_CALI})"
fi

# ── [0b] PTQ (weight-only or W8A8) ───────────────────────────────────────────
if should_run ptq && [[ "${SKIP_PTQ}" != "1" ]]; then
  require_script "${SAMPLE_LDM}"
  if [[ -f "${CALI_CKPT}" ]]; then
    log_step "[0b] skip PTQ — exists: ${CALI_CKPT}"
  else
    require_file "${CALI_DATA}"
    if [[ "${QUANT_ACT}" == "1" ]]; then
      PTQ_LABEL="W${WBIT}A${ABIT}"
    else
      PTQ_LABEL="W${WBIT}-only"
    fi
    log_step "[0b] ${PTQ_LABEL} PTQ → ${CALI_CKPT} (cali_n=${PTQ_CALI_N} batch=${PTQ_CALI_BATCH})"
    PTQ_CMD=(
      python "${SAMPLE_LDM}"
      -r "${FP_CKPT}"
      -n 4 --batch_size 1
      -c "${DDIM_STEPS}" -e 0.0
      --seed "${SEED}" --scale "${CFG_SCALE}" --cond
      --ptq --weight_bit "${WBIT}" --quant_mode qdiff
      --cali_st 20 --cali_batch_size "${PTQ_CALI_BATCH}" --cali_n "${PTQ_CALI_N}"
      --cali_data_path "${CALI_DATA}"
      --ptq_output_ckpt "${CALI_CKPT}"
      -l "${OUT}/ptq_calibrate"
    )
    if [[ "${QUANT_ACT}" == "1" ]]; then
      PTQ_CMD+=(
        --cali_iters_a "${PTQ_CALI_ITERS_A}" --cali_running_batch "${PTQ_RUNNING_BATCH}"
        --quant_act --act_bit "${ABIT}" --a_sym --a_min_max --running_stat
      )
    fi
    "${PTQ_CMD[@]}" 2>&1 | tee "${LOGDIR}/ptq_calibrate.log"
  fi
else
  log_step "[0b] skip PTQ (STEP=${STEP} SKIP_PTQ=${SKIP_PTQ})"
fi

require_file "${CALI_CKPT}"

# ── [1] collect traj ──────────────────────────────────────────────────────────
if should_run collect && [[ "${SKIP_COLLECT}" != "1" ]]; then
  require_script "${COLLECT_LDM}"
  if [[ -f "${TRAJ}" ]]; then
    log_step "[1/5] skip collect — exists: ${TRAJ}"
  else
    log_step "[1/5] collect LDM traj (${N_TRAJ} × ${DDIM_STEPS}, eta=${ETA})"
    python "${COLLECT_LDM}" \
      --fp_ckpt "${FP_CKPT}" \
      --ldm_config "${LDM_CONFIG}" \
      --cali_ckpt "${CALI_CKPT}" \
      --n_traj "${N_TRAJ}" \
      --batch_size "${COLLECT_BATCH}" \
      --steps "${DDIM_STEPS}" \
      --eta "${ETA}" \
      --scale "${CFG_SCALE}" \
      --weight_bit "${WBIT}" \
      "${COLLECT_QUANT_FLAGS[@]}" \
      --output "${TRAJ}" \
      2>&1 | tee "${LOGDIR}/collect_traj.log"
  fi
else
  log_step "[1/5] skip collect"
fi

# ── [2] train corrector ───────────────────────────────────────────────────────
if should_run train && [[ "${SKIP_TRAIN}" != "1" ]]; then
  require_file "${TRAJ}"
  require_script "${TRAIN_CORR}"
  if [[ -f "${CORR_CKPT}" ]]; then
    log_step "[2/5] skip train — exists: ${CORR_CKPT}"
  else
    log_step "[2/5] train learned corrector"
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
  log_step "[2/5] skip train"
fi

# ── [3] estimate VSC (var_mle) ────────────────────────────────────────────────
if should_run estimate && [[ "${SKIP_ESTIMATE}" != "1" ]]; then
  require_file "${TRAJ}"
  require_file "${CORR_CKPT}"
  require_script "${ESTIMATE_VSC}"
  if [[ -f "${VSC_STATS}" ]]; then
    log_step "[3/5] skip estimate — exists: ${VSC_STATS}"
  else
    log_step "[3/5] post-corrector Student-t var_mle"
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
  log_step "[3/5] skip estimate"
fi

# ── [4] sample + FID ──────────────────────────────────────────────────────────
if should_run sample && [[ "${SKIP_SAMPLE}" != "1" ]]; then
  # Mid-run overrides while collect/train is running:
  #   echo 3000 > ${OUT}/fid_n.txt
  #   echo 3 > ${OUT}/sample_batch.txt
  [[ -f "${OUT}/fid_n.txt" ]] && FID_N="$(tr -d '[:space:]' < "${OUT}/fid_n.txt")"
  [[ -f "${OUT}/sample_batch.txt" ]] && SAMPLE_BATCH="$(tr -d '[:space:]' < "${OUT}/sample_batch.txt")"
  SAMPLE_FLAGS=(
    -r "${FP_CKPT}"
    --batch_size "${SAMPLE_BATCH}"
    -c "${DDIM_STEPS}"
    -e "${ETA}"
    --seed "${SEED}"
    --scale "${CFG_SCALE}"
    --cond
    "${PTQ_FLAGS[@]}"
  )
  require_script "${SAMPLE_LDM}"
  require_file "${CORR_CKPT}"
  require_file "${VSC_STATS}"
  log_step "[4/5] sample ${FID_N} (learned_corr + vsc tvar)"
  python "${SAMPLE_LDM}" \
    "${SAMPLE_FLAGS[@]}" \
    -n "${FID_N}" \
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
  find "${SAMPLE_LOG}" -type f \( -name '*.png' -o -name '*.jpg' \) | head -n "${FID_N}" | \
    awk -v dst="${FID_STAGING}" '{ printf "cp \"%s\" \"%s/sample_%06d.png\"\n", $0, dst, NR }' | bash
else
  log_step "[4/5] skip sample"
fi

if should_run fid && [[ "${SKIP_FID}" != "1" ]]; then
  require_file "${FID_REF}"
  log_step "[5/5] FID"
  python -m pytorch_fid "${FID_REF}" "${FID_STAGING}" --device cuda:0 \
    2>&1 | tee "${LOGDIR}/fid.log"
else
  log_step "[5/5] skip FID"
fi

log_step "DONE → ${LOGDIR}"
echo "  cali_ckpt: ${CALI_CKPT}"
echo "  traj:      ${TRAJ}"
echo "  corrector: ${CORR_CKPT}"
echo "  vsc:       ${VSC_STATS}"
echo "  samples:   ${FID_STAGING}"
