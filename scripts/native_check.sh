#!/usr/bin/env bash
# Correctness gate before anything goes near the secure world: compile the generated C classifier
# natively, run it on the reference features, and compare with the PyTorch / ONNX reference.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$REPO/build/rawtfnet_cls.c" ]] || { echo "Run scripts/generate_ta_model.sh first" >&2; exit 1; }

gcc -O2 -o "$REPO/build/test_cls" "$REPO/secure-world/native-test/test_cls.c" "$REPO/build/rawtfnet_cls.c" -lm
OUT="$("$REPO/build/test_cls" "$REPO/artifacts/features.bin")"
echo "$OUT"
echo "expected:  logits: 5.391142 -5.621588   (order: spoof, bonafide)"
[[ "$OUT" == "logits: 5.391142 -5.621588" ]] && echo "OK: matches the PyTorch reference" || { echo "MISMATCH" >&2; exit 1; }
