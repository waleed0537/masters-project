#!/usr/bin/env python3
"""Denoising variant of prepare_audio.py, used for the out-of-distribution elimination study.

audio -> denoise (noisereduce) -> peak-normalise -> features -> QEMU shared folder.

Usage:
    python3 scripts/prepare_audio_denoise.py /path/to/audio.wav

Environment overrides: QEMU_SHARE (default ~/qemu_share), RAWTFNET_FRONTEND
(default artifacts/rawtfnet_frontend.onnx).
"""
import os
import sys
from pathlib import Path

import numpy as np
import librosa
import noisereduce as nr
import onnxruntime as ort

MAX_LEN = 64600
REPO = Path(__file__).resolve().parent.parent


def pad(x, m=MAX_LEN):
    return x[:m] if x.shape[0] >= m else np.tile(x, int(m / x.shape[0]) + 1)[:m]


if len(sys.argv) != 2:
    sys.exit(__doc__)
audio_path = sys.argv[1]
X, sr = librosa.load(audio_path, sr=16000)

# --- DENOISE STEP (the experiment) ---
X = nr.reduce_noise(y=X, sr=16000)

# peak-normalise (same as the main pipeline)
peak = np.max(np.abs(X))
if peak > 0:
    X = X / peak

X = pad(X).astype(np.float32)[None, :]

frontend = os.environ.get("RAWTFNET_FRONTEND", str(REPO / "artifacts" / "rawtfnet_frontend.onnx"))
sess = ort.InferenceSession(frontend)
feat = sess.run(None, {"audio": X})[0].astype(np.float32)
assert feat.shape == (1, 32, 23, 16), f"unexpected feature shape {feat.shape}"

share = os.path.expanduser(os.environ.get("QEMU_SHARE", "~/qemu_share"))
os.makedirs(share, exist_ok=True)
feat.tofile(os.path.join(share, "current.bin"))
label = "DENOISED_" + os.path.basename(audio_path)
with open(os.path.join(share, "current_label.txt"), "w") as f:
    f.write(label)

print(f"Ready (denoised): {label}")
print("Now run in QEMU Normal World:  optee_example_rawtfnet")
