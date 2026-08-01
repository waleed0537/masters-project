import torch, onnx
from predict import load_model

model = load_model("./ckpts/Best_RawTFNet_16.pth", device="cpu")
model.eval()

dummy_input = torch.randn(1, 32000)   # 2 seconds @ 16kHz — within good-accuracy range

torch.onnx.export(
    model, dummy_input, "rawtfnet_2s.onnx",
    input_names=["audio"], output_names=["logits"],
    opset_version=17, dynamic_axes=None, dynamo=False,
)
m = onnx.load("rawtfnet_2s.onnx", load_external_data=True)
onnx.save_model(m, "rawtfnet_2s.onnx", save_as_external_data=False)
print("exported 2-second model")