#!/usr/bin/env bash
# Install Asterisk on the Ubuntu VM and add one softphone extension (100) and one detection line (600).
#
#   SIP_PASSWORD='choose-a-password' live-call/setup_asterisk.sh
#
# Dial 600 from the softphone: the call is answered, the caller's audio is recorded to
# /tmp/live.wav16 (16 kHz), and Asterisk stays silent. Use the same password in the softphone.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SIP_PASSWORD:?Set SIP_PASSWORD (it must match the softphone account)}"

sudo apt update
sudo apt install -y asterisk
sudo systemctl enable --now asterisk

sudo cp -n /etc/asterisk/pjsip.conf /etc/asterisk/pjsip.conf.bak
sudo cp -n /etc/asterisk/extensions.conf /etc/asterisk/extensions.conf.bak

if ! grep -q '^\[auth100\]' /etc/asterisk/pjsip.conf; then
  sed "s/CHANGE_ME/${SIP_PASSWORD}/" "$HERE/pjsip_additions.conf" | sudo tee -a /etc/asterisk/pjsip.conf >/dev/null
fi
if ! grep -q '^exten => 600,1' /etc/asterisk/extensions.conf; then
  sudo tee -a /etc/asterisk/extensions.conf < "$HERE/extensions_additions.conf" >/dev/null
fi

sudo systemctl restart asterisk
sleep 5
sudo asterisk -rx "pjsip show endpoints"          # expect endpoint 100 (Unavailable until the softphone registers)
sudo asterisk -rx "dialplan show from-internal"   # expect extension 600
sudo usermod -aG asterisk "$USER"                 # lets live_detect.py read /tmp/live.wav16 (log out/in once)
echo "VM IP addresses (use one as the SIP server in the softphone):"
ip -br a
