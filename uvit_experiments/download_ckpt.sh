#!/usr/bin/env bash
# Download U-ViT-S/2 CIFAR-10 pretrained weights (~120MB).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p checkpoints
OUT="checkpoints/cifar10_uvit_small.pth"
FILE_ID="1yoYyuzR_hQYWU0mkTj659tMTnoCWCMv-"

if [[ -f "$OUT" ]] && [[ "$(file -b "$OUT")" == *"data"* || $(stat -c%s "$OUT") -gt 10000000 ]]; then
  echo "Checkpoint already present: $OUT ($(du -h "$OUT" | cut -f1))"
  exit 0
fi

echo "Trying gdown..."
if command -v gdown >/dev/null 2>&1; then
  gdown "${FILE_ID}" -O "$OUT" && exit 0
fi

echo "Trying pip gdown..."
pip install -q gdown
gdown "${FILE_ID}" -O "$OUT" && exit 0 || true

echo ""
echo "Automatic download failed (Google Drive may be blocked)."
echo "Please manually download and place the file at:"
echo "  $(pwd)/$OUT"
echo ""
echo "Official link:"
echo "  https://drive.google.com/file/d/${FILE_ID}/view"
echo ""
echo "On AutoDL: use 学术加速 or upload via Jupyter/SCP."
exit 1
