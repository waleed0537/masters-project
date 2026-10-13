#!/usr/bin/env python3
"""Score audio files with RawTFNet-16 using ONNX Runtime only -- no PyTorch, no TrustZone.

This is the quickest way to see what the model says about a clip, and it is the reference the
secure-world result is compared against. It runs the same two stages as the deployed system:
frontend (feature extraction) -> classifier.

Output order is [spoof, bonafide] (training labels: bonafide = 1, spoof = 0). A clip is classified
bonafide when the bonafide logit (second value) is positive, spoof otherwise.

Usage:
    python scripts/score_audio.py clip1.flac clip2.wav
    python scripts/score_audio.py clip.flac --no-normalize
"""
import argparse
from pathlib import Path

import numpy as np
import librosa
import onnxruntime as ort

MAX_LEN = 64600
ART = Path(__file__).resolve().parent.parent / "artifacts"


def pad(x, m=MAX_LEN):
    return x[:m] if x.shape[0] >= m else np.tile(x, int(m / x.shape[0]) + 1)[:m]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--no-normalize", action="store_true", help="skip peak normalisation")
    ap.add_argument("--frontend", default=str(ART / "rawtfnet_frontend.onnx"))
    ap.add_argument("--classifier", default=str(ART / "rawtfnet_classifier_clean.onnx"))
    args = ap.parse_args()

    fe = ort.InferenceSession(args.frontend)
    cl = ort.InferenceSession(args.classifier)

    for path in args.audio:
        x, _ = librosa.load(path, sr=16000)
        if not args.no_normalize:
            peak = float(np.max(np.abs(x)))
            if peak > 0:
                x = x / peak
        x = pad(x).astype(np.float32)[None, :]
        feat = fe.run(None, {"audio": x})[0].astype(np.float32)
        logits = cl.run(None, {cl.get_inputs()[0].name: feat})[0].reshape(-1)
        spoof, bona = float(logits[0]), float(logits[1])
        verdict = "BONAFIDE (real)" if bona > 0 else "SPOOF (fake)"
        print(f"{Path(path).name}\tspoof logit={spoof:+.6f}\tbonafide logit={bona:+.6f}\t{verdict}")


if __name__ == "__main__":
    main()
