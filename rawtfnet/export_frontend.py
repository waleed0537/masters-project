import torch, onnx
from predict import load_model

model = load_model("./ckpts/Best_RawTFNet_16.pth", device="cpu")
model.eval()

dummy = torch.randn(1, 64600)
torch.onnx.export(model.front_end, dummy, "rawtfnet_frontend.onnx",
                  input_names=["audio"], output_names=["features"],
                  opset_version=17, dynamic_axes=None, dynamo=False)
m = onnx.load("rawtfnet_frontend.onnx", load_external_data=True)
onnx.save_model(m, "rawtfnet_frontend.onnx", save_as_external_data=False)
print("exported rawtfnet_frontend.onnx")