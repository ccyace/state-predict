#!/usr/bin/env bash
# Wait for W4A8 dt-only 50k sampling, then FID vs origin-cifar-10-python_fid_mu_sigma.npz.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="PTQD/w4a8_dt_only_eta1_50k"
LOG="PTQD/logs/sample_w4a8_dt_only_eta1_50k.log"
REF="origin-cifar-10-python_fid_mu_sigma.npz"
FIDLOG="$OUT/logs/fid_vs_origin_cifar10_python_mu_sigma.log"
MON="PTQD/logs/monitor_w4a8_dt_only_eta1_50k_fid.log"
mkdir -p "$OUT/logs"

sample_running() {
  # Match only the python sampler argv, not this watcher script.
  pgrep -f "python state_aware_temporal_joint/sample_50k.py .*--output_dir PTQD/w4a8_dt_only_eta1_50k" >/dev/null
}

echo "[$(date)] waiting for sample_50k to finish ..." | tee -a "$MON"
while sample_running; do
  n=$(find "$OUT" -name "*.png" 2>/dev/null | wc -l)
  last=$(grep -oE "images=[0-9]+/50000[^ ]*" "$LOG" 2>/dev/null | tail -1 || true)
  echo "[$(date "+%H:%M:%S")] pngs=$n $last" | tee -a "$MON"
  sleep 180
done

echo "[$(date)] sampling process exited" | tee -a "$MON"
n=$(find "$OUT" -name "*.png" 2>/dev/null | wc -l)
echo "png_count=$n" | tee -a "$MON"
tail -30 "$LOG" | tee -a "$MON"

if [[ "$n" -lt 50000 ]]; then
  echo "[$(date)] ERROR: expected 50000 pngs, got $n — skip FID" | tee -a "$MON"
  exit 2
fi

echo "[$(date)] computing FID vs $REF ..." | tee -a "$MON"
python -m pytorch_fid "$REF" "$OUT" --device cuda:0 2>&1 | tee "$FIDLOG" | tee -a "$MON"
echo "[$(date)] FID done -> $FIDLOG" | tee -a "$MON"

python3 - <<PY
import re, json, pathlib
log = pathlib.Path("$FIDLOG").read_text()
m = re.search(r"FID:\s*([0-9.]+)", log)
fid = float(m.group(1)) if m else None
path = pathlib.Path("$OUT/manifest_fid.json")
path.write_text(json.dumps({
  "n_images": $n,
  "fid_ref": "$REF",
  "fid": fid,
  "fid_log": "$FIDLOG",
  "sample_log": "$LOG",
}, indent=2) + "\n")
print("wrote", path, "fid=", fid)
PY
