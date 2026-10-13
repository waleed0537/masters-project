"""Independent cross-check: run the full ONNX model through Apache TVM (Relax, LLVM CPU) and print
the bona fide logit. Compare with PyTorch / ONNX Runtime.

Usage (inside the tvm-env virtualenv, from the repo root):
    python scripts/tvm_verify.py artifacts/rawtfnet.onnx /path/to/LA_E_1000048.flac
"""
import onnx
import numpy as np
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

import sys
ONNX_PATH = sys.argv[1] if len(sys.argv) > 1 else "artifacts/rawtfnet.onnx"
AUDIO_PATH = sys.argv[2] if len(sys.argv) > 2 else "LA_E_1000048.flac"

onnx_model = onnx.load(ONNX_PATH)

input_name = "audio"
shape_dict = {input_name: (1, 64600)}

mod = from_onnx(onnx_model, shape_dict=shape_dict, keep_params_in_input=False)
if isinstance(mod, tuple):
    mod, params = mod
else:
    params = None

mod = relax.transform.LegalizeOps()(mod)

target = tvm.target.Target("llvm")
ex = relax.build(mod, target=target)

dev = tvm.cpu(0)
vm = relax.VirtualMachine(ex, dev)

import librosa

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, num_repeats)[:max_len]

X, _ = librosa.load(AUDIO_PATH, sr=16000)
X = pad(X).astype("float32")[None, :]

x_tvm = tvm.runtime.tensor(X, device=dev)   # new Tensor API
out = vm["main"](x_tvm)
out_np = out.numpy() if hasattr(out, "numpy") else np.array(out)
score = out_np[1]
print("TVM (relax/llvm) bona fide logit:", score)
