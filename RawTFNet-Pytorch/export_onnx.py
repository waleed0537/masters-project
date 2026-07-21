"""
export_onnx.py
Step 2 + Step 3: export RawTFNet_small to ONNX, then verify the ONNX
output matches the PyTorch output on the same audio file.

Run from the root of RawTFNet-Pytorch (same folder as predict.py):
    python export_onnx.py --model_path ./ckpts/Best_RawTFNet_16.pth --audio LA_E_1000048.flac
"""
import argparse
import numpy as np
import torch
import onnxruntime as ort

from predict import load_model, load_audio


def export_to_onnx(model, max_len, onnx_path="rawtfnet.onnx"):
    dummy_input = torch.randn(1, max_len)  # static shape: batch=1, fixed length
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["audio"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes=None,  # force static shape, no dynamic batching
    )
    print(f"Exported to {onnx_path}")


def run_onnx_inference(onnx_path, x_numpy):
    sess = ort.InferenceSession(onnx_path)
    out = sess.run(None, {"audio": x_numpy})
    out = out[0].reshape(x_numpy.shape[0], -1)
    score = out[:, 1].item()
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--audio', type=str, required=True)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--max_len', type=int, default=64600)
    parser.add_argument('--onnx_path', type=str, default='rawtfnet.onnx')
    args = parser.parse_args()

    device = 'cpu'  # export on CPU for simplicity/portability

    # --- Step 2: export ---
    model = load_model(args.model_path, sample_rate=args.sample_rate,
                        max_len=args.max_len, device=device)
    export_to_onnx(model, args.max_len, args.onnx_path)

    # --- Step 3: verify against PyTorch on the same file ---
    x = load_audio(args.audio, sample_rate=args.sample_rate, max_len=args.max_len)

    with torch.no_grad():
        torch_out = model(x).view(x.shape[0], -1)
        torch_score = torch_out[:, 1].item()

    onnx_score = run_onnx_inference(args.onnx_path, x.numpy().astype(np.float32))

    print(f"PyTorch score: {torch_score:.4f}")
    print(f"ONNX score:    {onnx_score:.4f}")
    print(f"Difference:    {abs(torch_score - onnx_score):.6f}")


if __name__ == '__main__':
    main()