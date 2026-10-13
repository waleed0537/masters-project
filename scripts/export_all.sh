#!/usr/bin/env bash
# Export the ONNX models from the trained checkpoint and collect them in artifacts/.
#
#   frontend   : audio (1x64600) -> features (1x32x23x16)           (stays in the normal world)
#   classifier : features -> logits [spoof, bonafide]                (runs in the secure world)
#   full model : audio -> logits (used for the PyTorch/ONNX/TVM cross-checks)
#
# Needs PyTorch (see rawtfnet/requirements.txt). export_classifier.py also writes features.bin
# from one reference clip, LA_E_1000048.flac (ASVspoof 2021 LA eval) -- pass its path as $1.
#
# Usage:  scripts/export_all.sh /path/to/LA_E_1000048.flac
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAMPLE="${1:-}"

cd "$REPO/rawtfnet"
if [[ -n "$SAMPLE" ]]; then cp "$SAMPLE" ./LA_E_1000048.flac; fi
[[ -f LA_E_1000048.flac ]] || { echo "Missing reference clip. Usage: $0 /path/to/LA_E_1000048.flac" >&2; exit 1; }
[[ -f ckpts/Best_RawTFNet_16.pth ]] || { echo "Missing rawtfnet/ckpts/Best_RawTFNet_16.pth" >&2; exit 1; }

python export_frontend.py
python export_classifier.py
python export_onnx.py

mkdir -p "$REPO/artifacts"
mv -f rawtfnet_frontend.onnx rawtfnet_classifier.onnx rawtfnet.onnx features.bin "$REPO/artifacts/"
echo "Exported models are in $REPO/artifacts/. Next: scripts/generate_ta_model.sh"
