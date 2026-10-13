#!/usr/bin/env python3
"""
live_detect.py — Live call detection via 4-second sliding window.

Watches the growing Asterisk call recording, and every STRIDE seconds:
  1. takes the most recent 64,600 samples (~4 s) of call audio
  2. resamples to 16 kHz if needed, peak-normalizes
  3. runs the front_end ONNX model to get features
  4. writes features + label into ~/qemu_share and signals the guest
The guest-side watcher loop then runs the secure-world inference and
appends the verdict to ~/qemu_share/results.log.

Usage:
    source ~/tvm-env/bin/activate
    python3 live-call/live_detect.py                 # default: /tmp/live.wav16
    python3 live-call/live_detect.py /path/to/rec    # custom recording path
    STRIDE=2.0 python3 live-call/live_detect.py      # overlapping windows (verdict every 2 s)

Environment overrides: QEMU_SHARE (default ~/qemu_share), RAWTFNET_FRONTEND
(default artifacts/rawtfnet_frontend.onnx), STRIDE (default 4.0).
"""
import os
import sys
import time

import numpy as np
import librosa
import soundfile as sf
import onnxruntime as ort

WAV_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/live.wav16"
WIN = 64600          # model's fixed input length (~4.04 s @ 16 kHz)
SR = 16000
STRIDE = float(os.environ.get("STRIDE", "4.0"))   # seconds between verdicts (4.0 = non-overlapping)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARE = os.path.expanduser(os.environ.get("QEMU_SHARE", "~/qemu_share"))
FRONTEND = os.environ.get("RAWTFNET_FRONTEND",
                          os.path.join(REPO, "artifacts", "rawtfnet_frontend.onnx"))

sess = ort.InferenceSession(FRONTEND)


def read_latest_window():
    """Read raw 16 kHz int16 PCM straight from the growing file (skip 44-byte header)."""
    try:
        size = os.path.getsize(WAV_PATH)
    except OSError:
        return None, 0.0
    n_samples = max(0, (size - 44) // 2)
    total_s = n_samples / SR
    if n_samples < WIN:
        return None, total_s
    with open(WAV_PATH, "rb") as f:
        f.seek(44 + (n_samples - WIN) * 2)
        raw = f.read(WIN * 2)
    if len(raw) < WIN * 2:
        return None, total_s
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return x, total_s


def main():
    print(f"[live_detect] watching: {WAV_PATH}")
    print(f"[live_detect] window = {WIN} samples (~{WIN/SR:.1f}s), stride = {STRIDE:.0f}s")
    print("[live_detect] waiting for the call recording to appear...")
    while not os.path.exists(WAV_PATH):
        time.sleep(0.5)
    print("[live_detect] recording found — call is live.")

    n = 0
    while True:
        x, total_s = read_latest_window()
        if x is None:
            print(f"[live_detect] buffering audio... {total_s:.1f}s / {WIN/SR:.1f}s")
            time.sleep(1.0)
            continue

        # peak-normalize (matches the offline pipeline)
        peak = float(np.max(np.abs(x)))
        if peak > 0:
            x = x / peak

        feat = sess.run(None, {"audio": x.astype(np.float32)[None, :]})[0]
        feat = feat.astype(np.float32)
        assert feat.shape == (1, 32, 23, 16), f"unexpected feature shape {feat.shape}"

        n += 1
        start_s = max(0.0, total_s - WIN / SR)
        label = f"LIVE_win{n}_{start_s:.0f}s-{total_s:.0f}s"

        feat.tofile(os.path.join(SHARE, "current.bin"))
        with open(os.path.join(SHARE, "current_label.txt"), "w") as f:
            f.write(label)
        # signal the guest watcher to run one inference
        with open(os.path.join(SHARE, "go"), "w") as f:
            f.write("1")

        print(f"[window {n:3d}] {start_s:6.1f}s-{total_s:6.1f}s  -> sent to secure world")
        time.sleep(STRIDE)


if __name__ == "__main__":
    main()
