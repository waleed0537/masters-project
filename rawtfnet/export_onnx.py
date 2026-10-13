"""Export the full RawTFNet-16 model (frontend + classifier) to a single ONNX file.

Run from this folder (rawtfnet/):
    python export_onnx.py                      # 64,600 samples (~4 s) -> rawtfnet.onnx
    python export_onnx.py --samples 32000      # 2 s variant           -> rawtfnet_2s.onnx
"""
import argparse

import torch, onnx
from predict import load_model

parser = argparse.ArgumentParser(description="Export the full RawTFNet-16 model to ONNX")
parser.add_argument("--samples", type=int, default=64600,
                    help="input length in samples at 16 kHz (64600 ~ 4 s, the model's native length)")
parser.add_argument("--output", default=None,
                    help="output file (default: rawtfnet.onnx, or rawtfnet_2s.onnx for --samples 32000)")
args = parser.parse_args()
out_path = args.output or ("rawtfnet_2s.onnx" if args.samples == 32000 else "rawtfnet.onnx")

model = load_model("./ckpts/Best_RawTFNet_16.pth", device="cpu")
model.eval()

dummy_input = torch.randn(1, args.samples)

torch.onnx.export(
    model, dummy_input, out_path,
    input_names=["audio"], output_names=["logits"],
    opset_version=17, dynamic_axes=None, dynamo=False,
)
m = onnx.load(out_path, load_external_data=True)
onnx.save_model(m, out_path, save_as_external_data=False)
print(f"exported {out_path} (input: 1 x {args.samples} samples)")
