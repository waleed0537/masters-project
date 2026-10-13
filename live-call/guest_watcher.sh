#!/bin/sh
# Runs INSIDE the QEMU Normal World (busybox sh). Waits for the host-side live_detect.py to drop a
# file called "go" in the shared folder, then runs one secure-world inference and appends the
# verdict to results.log (which is visible on the host in ~/qemu_share/results.log).
#
# In the guest:
#   mkdir -p /mnt/host && mount -t 9p -o trans=virtio host /mnt/host
#   sh /mnt/host/guest_watcher.sh        # copy this file into ~/qemu_share first
rm -f /mnt/host/go /mnt/host/results.log
while true; do
  if [ -f /mnt/host/go ]; then
    rm -f /mnt/host/go
    optee_example_rawtfnet | tee -a /mnt/host/results.log
  fi
  sleep 1
done
