#!/usr/bin/env bash
# Copy the Trusted Application + client app into the OP-TEE tree and rebuild.
#
# IMPORTANT: run this with the Python virtualenv DEACTIVATED. TA signing needs the system Python's
# `cryptography` module (python3-cryptography); the model virtualenv does not have it.
set -euo pipefail
OPTEE_DIR="${OPTEE_DIR:-$HOME/optee}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "A virtualenv is active ($VIRTUAL_ENV). Run 'deactivate' first." >&2
  exit 1
fi
[[ -d "$OPTEE_DIR/build" ]] || { echo "OP-TEE tree not found at $OPTEE_DIR (set OPTEE_DIR)" >&2; exit 1; }

mkdir -p "$OPTEE_DIR/optee_examples/rawtfnet"
rsync -a --delete "$HERE/trusted-app/" "$OPTEE_DIR/optee_examples/rawtfnet/"
cd "$OPTEE_DIR/build"
make -j"$(nproc)" 2>&1 | tee "$HERE/../build-ta.log"
echo "Built. Next: secure-world/run_qemu.sh"
