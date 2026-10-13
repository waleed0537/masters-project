#!/usr/bin/env python3
"""Any audio in -> features + label, ready for the secure-world Trusted Application.

The frontend (feature extraction) runs here, in the normal world, via ONNX Runtime. The result is
written to the QEMU shared folder as current.bin (+ current_label.txt); inside QEMU,
`optee_example_rawtfnet` reads them and runs the classifier in the secure world.

Usage:
    python3 scripts/prepare_audio.py /path/to/any_audio.wav
    python3 scripts/prepare_audio.py clip.flac --no-normalize     # skip peak normalisation

Environment overrides:
    QEMU_SHARE         shared folder mounted in the guest (default: ~/qemu_share)
    RAWTFNET_FRONTEND  frontend ONNX file (default: artifacts/rawtfnet_frontend.onnx)
"""
import argparse
import os
from pathlib import Path

import numpy as np
import librosa
import onnxruntime as ort

MAX_LEN = 64600
REPO = Path(__file__).resolve().parent.parent


def pad(x, m=MAX_LEN):
    return x[:m] if x.shape[0] >= m else np.tile(x, int(m / x.shape[0]) + 1)[:m]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--no-normalize", action="store_true",
                    help="do not peak-normalise (ASVspoof evaluation files are already peak-normalised)")
    args = ap.parse_args()

    X, _ = librosa.load(args.audio, sr=16000)          # resamples to 16 kHz
    if not args.no_normalize:
        peak = float(np.max(np.abs(X)))
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
    label = os.path.basename(args.audio)
    with open(os.path.join(share, "current_label.txt"), "w") as f:
        f.write(label)

    print(f"Ready: {label}")
    print("Now run this ONE command in the QEMU Normal World window:")
    print("  optee_example_rawtfnet")


if __name__ == "__main__":
    main()
