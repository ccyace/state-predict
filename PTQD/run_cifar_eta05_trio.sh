#!/usr/bin/env bash
# CIFAR-10 DDIM η=0.5 trio:
#   1) FP32 baseline, 100 steps
#   2) W8A8 baseline, 50 steps
#   3) W8A8 + learned ε-corrector + Student-t VSC (tvar), 100 steps
#
# Usage:
#   MAX_IMAGES=50000 bash PTQD/run_cifar_eta05_trio.sh
#   STEP=fp|w8a8|corr_vsc|fid|all

set -euo pipefail
cd /root/autodl-tmp/ODE-scale
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

ETA="${ETA:-0.5}"
SEED="${SEED:-1234}"
MAX_IMAGES="${MAX_IMAGES:-50000}"
BATCH="${BATCH:-64}"
SKIP_TYPE="${SKIP_TYPE:-quad}"
STEP="${STEP:-all}"

OUT_ROOT="${OUT_ROOT:-output/cifar_ddim_eta05_trio}"
LOGDIR="${OUT_ROOT}/logs"
FP_OUT="${OUT_ROOT}/fp32_ddim100"
W8_OUT="${OUT_ROOT}/w8a8_ddim50"
CORR_OUT="${OUT_ROOT}/w8a8_corr_tvar_ddim100"
CORR_CKPT="${CORR_CKPT:-noise_eps_corr/w8a8_corr_ckpt_best.pt}"
CALI_CKPT="${CALI_CKPT:-cifar_w8a8_ckpt.pth}"
CALI_DATA="${CALI_DATA:-cifar_sd1236_sample2048_allst.pt}"
TRAJ_RAW="${TRAJ_RAW:-state_aware_temporal_joint/data/traj_w8a8_n256_ol.pt}"
TRAJ_AFTER="${OUT_ROOT}/traj_w8a8_after_corr.pt"
VSC_STATS="${OUT_ROOT}/vsc_time_stats_eta05_tvar.pt"
FID_REF="${FID_REF:-origin-cifar-10-python_fid_mu_sigma.npz}"

mkdir -p "${LOGDIR}" "${FP_OUT}" "${W8_OUT}" "${CORR_OUT}"
PIPELINE_LOG="${LOGDIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

should_run() { [[ "${STEP}" == "all" || "${STEP}" == "$1" ]]; }
log() { echo ""; echo "======== [$(date '+%F %T')] $* ========"; }

require() { [[ -f "$1" ]] || { echo "ERROR: missing $1" >&2; exit 1; }; }

fid_of() {
  local img="$1" name="$2"
  local n
  n=$(find "$img" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l)
  echo "FID input ${name}: dir=${img} n=${n}"
  python -m pytorch_fid "${FID_REF}" "${img}" --device cuda:0 2>&1 | tee "${LOGDIR}/fid_${name}.log"
}

# ── 0) estimate VSC at η=0.5 after applying learned corrector ────────────────
if should_run corr_vsc || should_run all; then
  require "${CORR_CKPT}"
  require "${TRAJ_RAW}"
  if [[ ! -f "${VSC_STATS}" ]]; then
    log "build post-corr traj + estimate VSC tvar (eta=${ETA})"
    python - <<PY
import os, torch
from noise_eps_corr.learned_noise_corrector import load_learned_corrector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
traj = torch.load("${TRAJ_RAW}", map_location="cpu")
corr = load_learned_corrector("${CORR_CKPT}", device)
corr.train_mode_off()

x, eq, ef, t = traj["x"].float(), traj["eq"].float(), traj["ef"].float(), traj["t"].float()
eq_corr, residual_mse = [], []
bs = 64
with torch.no_grad():
    for i in range(0, x.shape[0], bs):
        xb, eqb, efb, tb = x[i:i+bs].to(device), eq[i:i+bs].to(device), ef[i:i+bs].to(device), t[i:i+bs]
        out = []
        for j in range(xb.shape[0]):
            out.append(corr.correct(eqb[j:j+1], float(tb[j].item()), None, xt=xb[j:j+1]))
        eqc = torch.cat(out, 0)
        eq_corr.append(eqc.cpu().half())
        residual_mse.append((eqc - efb).pow(2).mean(dim=(1,2,3)).cpu())

post = dict(traj)
post["eq"] = torch.cat(eq_corr, 0)
post["residual_mse"] = torch.cat(residual_mse, 0).float()
post["t_nom"] = t.clone()
os.makedirs(os.path.dirname("${TRAJ_AFTER}") or ".", exist_ok=True)
torch.save(post, "${TRAJ_AFTER}")
print("saved", "${TRAJ_AFTER}", "n=", post["eq"].shape[0], flush=True)
PY
    python PTQD/estimate_time_variance.py \
      --data "${TRAJ_AFTER}" \
      --output_pt "${VSC_STATS}" \
      --output_json "${VSC_STATS%.pt}.json" \
      --eta "${ETA}" \
      --timesteps 100 \
      --skip_type "${SKIP_TYPE}" \
      2>&1 | tee "${LOGDIR}/estimate_vsc_eta05.log"
  else
    log "skip VSC estimate — exists ${VSC_STATS}"
  fi
fi

# ── 1) FP32 DDIM-100 η=0.5 ───────────────────────────────────────────────────
if should_run fp || should_run all; then
  log "FP32 baseline DDIM-100 eta=${ETA} n=${MAX_IMAGES}"
  if [[ $(find "${FP_OUT}" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l) -ge "${MAX_IMAGES}" ]]; then
    echo "skip FP sample — already have >= ${MAX_IMAGES} pngs"
  else
    # sample_diffusion_ddim writes under -l/samples/<ts>/img; we also want a flat dir.
    # Use sample_50k-style for FP? sample_diffusion_ddim without --ptq.
    python scripts/sample_diffusion_ddim.py \
      --config configs/cifar10.yml \
      --use_pretrained \
      --timesteps 100 \
      --eta "${ETA}" \
      --skip_type "${SKIP_TYPE}" \
      --max_images "${MAX_IMAGES}" \
      --seed "${SEED}" \
      -l "${FP_OUT}" \
      2>&1 | tee "${LOGDIR}/sample_fp32_ddim100.log"
    IMG=$(find "${FP_OUT}/samples" -type d -name img | sort | tail -1)
    # flatten hardlinks/symlinks into FP_OUT for FID convenience
    find "${IMG}" -maxdepth 1 -name '*.png' | sort | head -n "${MAX_IMAGES}" | \
      awk -v dst="${FP_OUT}/img_flat" 'BEGIN{system("mkdir -p \"" dst "\"")} {printf "cp \"%s\" \"%s/%06d.png\"\n",$0,dst,NR}' | bash
  fi
fi

# ── 2) W8A8 DDIM-50 η=0.5 ────────────────────────────────────────────────────
if should_run w8a8 || should_run all; then
  require "${CALI_CKPT}"
  require "${CALI_DATA}"
  log "W8A8 baseline DDIM-50 eta=${ETA} n=${MAX_IMAGES}"
  if [[ $(find "${W8_OUT}" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l) -ge "${MAX_IMAGES}" ]] \
     || [[ $(find "${W8_OUT}/img_flat" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l) -ge "${MAX_IMAGES}" ]]; then
    echo "skip W8A8 sample — already have enough pngs"
  else
    python scripts/sample_diffusion_ddim.py \
      --config configs/cifar10.yml \
      --use_pretrained \
      --timesteps 50 \
      --eta "${ETA}" \
      --skip_type "${SKIP_TYPE}" \
      --ptq --resume --split --a_sym --quant_act \
      --weight_bit 8 --act_bit 8 \
      --cali_ckpt "${CALI_CKPT}" \
      --cali_data_path "${CALI_DATA}" \
      --cali_st 10 --cali_n 256 \
      --max_images "${MAX_IMAGES}" \
      --seed "${SEED}" \
      -l "${W8_OUT}" \
      2>&1 | tee "${LOGDIR}/sample_w8a8_ddim50.log"
    IMG=$(find "${W8_OUT}/samples" -type d -name img | sort | tail -1)
    find "${IMG}" -maxdepth 1 -name '*.png' | sort | head -n "${MAX_IMAGES}" | \
      awk -v dst="${W8_OUT}/img_flat" 'BEGIN{system("mkdir -p \"" dst "\"")} {printf "cp \"%s\" \"%s/%06d.png\"\n",$0,dst,NR}' | bash
  fi
fi

# ── 3) W8A8 + corrector + tvar DDIM-100 η=0.5 ────────────────────────────────
if should_run corr_vsc || should_run all; then
  require "${CORR_CKPT}"
  require "${VSC_STATS}"
  require "${CALI_CKPT}"
  log "W8A8 + corrector + tvar DDIM-100 eta=${ETA} n=${MAX_IMAGES}"
  if [[ $(find "${CORR_OUT}" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l) -ge "${MAX_IMAGES}" ]]; then
    echo "skip corr+vsc sample — already have >= ${MAX_IMAGES} pngs"
  else
    python state_aware_temporal_joint/sample_50k.py \
      --disable_adapter \
      --corrector_ckpt "${CORR_CKPT}" \
      --cali_ckpt "${CALI_CKPT}" \
      --cali_data_path "${CALI_DATA}" \
      --eta "${ETA}" \
      --timesteps 100 \
      --skip_type "${SKIP_TYPE}" \
      --vsc_stats "${VSC_STATS}" \
      --vsc_var_field var_mle \
      --vsc_absorb_strength 1.0 \
      --vsc_max_budget_fraction 0.9 \
      --max_images "${MAX_IMAGES}" \
      --batch_size "${BATCH}" \
      --seed "${SEED}" \
      --skip_fid \
      --output_dir "${CORR_OUT}" \
      2>&1 | tee "${LOGDIR}/sample_w8a8_corr_tvar_ddim100.log"
  fi
fi

# ── FID ──────────────────────────────────────────────────────────────────────
if should_run fid || should_run all; then
  require "${FID_REF}"
  log "FID"
  FP_IMG=$(find "${FP_OUT}" -type d -name img_flat 2>/dev/null | head -1)
  [[ -z "${FP_IMG}" ]] && FP_IMG=$(find "${FP_OUT}/samples" -type d -name img 2>/dev/null | sort | tail -1)
  W8_IMG=$(find "${W8_OUT}" -type d -name img_flat 2>/dev/null | head -1)
  [[ -z "${W8_IMG}" ]] && W8_IMG=$(find "${W8_OUT}/samples" -type d -name img 2>/dev/null | sort | tail -1)
  fid_of "${FP_IMG}" "fp32_ddim100"
  fid_of "${W8_IMG}" "w8a8_ddim50"
  fid_of "${CORR_OUT}" "w8a8_corr_tvar_ddim100"
fi

log "done → ${OUT_ROOT}  log=${PIPELINE_LOG}"
