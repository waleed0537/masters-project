#!/usr/bin/env bash
# Boot OP-TEE on QEMU (Armv8-A "virt") with a host folder shared into the guest over 9p.
set -euo pipefail
OPTEE_DIR="${OPTEE_DIR:-$HOME/optee}"
SHARE="${QEMU_SHARE:-$HOME/qemu_share}"
mkdir -p "$SHARE"

cat <<MSG
Booting QEMU. Then:
  1. At the (qemu) prompt, type:  c   and press Enter
  2. In the Normal World window, log in as: root   (no password)
  3. Mount the shared folder:
       mkdir -p /mnt/host && mount -t 9p -o trans=virtio host /mnt/host
  4. Run the Trusted Application client:   optee_example_rawtfnet
Host folder shared with the guest: $SHARE
MSG

cd "$OPTEE_DIR/build"
exec make run QEMU_VIRTFS_ENABLE=y QEMU_VIRTFS_HOST_DIR="$SHARE"
