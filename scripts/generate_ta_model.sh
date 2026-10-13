#!/usr/bin/env bash
# Classifier ONNX -> clean ONNX -> standalone C -> Trusted Application model source.
#
#   artifacts/rawtfnet_classifier.onnx
#     -> scripts/clean_onnx.py          -> artifacts/rawtfnet_classifier_clean.onnx
#     -> emx-onnx-cgen compile          -> build/rawtfnet_cls.c            (plain C, for the native test)
#     -> patch the three libc includes  -> secure-world/trusted-app/ta/rawtfnet_model.c
#
# The TA has no libc, so <stdlib.h>, <math.h> and <float.h> are replaced by the small subset in
# ta/ta_math.h. Run inside the pipeline virtualenv (requirements-pipeline.txt).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART="$REPO/artifacts"
BUILD="$REPO/build"
mkdir -p "$BUILD"

python "$REPO/scripts/clean_onnx.py" "$ART/rawtfnet_classifier.onnx" "$ART/rawtfnet_classifier_clean.onnx"
emx-onnx-cgen verify "$ART/rawtfnet_classifier_clean.onnx" --cc gcc
emx-onnx-cgen compile "$ART/rawtfnet_classifier_clean.onnx" "$BUILD/rawtfnet_cls.c" --large-weight-threshold 0

sed -e 's|^#include <stdlib.h>|/* stdlib.h via ta_math.h */|' \
    -e 's|^#include <math.h>|#include "ta_math.h"|' \
    -e 's|^#include <float.h>|/* float.h via ta_math.h */|' \
    "$BUILD/rawtfnet_cls.c" > "$REPO/secure-world/trusted-app/ta/rawtfnet_model.c"

echo "Wrote build/rawtfnet_cls.c and secure-world/trusted-app/ta/rawtfnet_model.c"
echo "Next: scripts/native_check.sh"
