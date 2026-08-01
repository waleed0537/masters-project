import torch, onnx, numpy as np
from predict import load_model, load_audio

model = load_model("./ckpts/Best_RawTFNet_16.pth", device="cpu")
model.eval()

x = load_audio("LA_E_1000048.flac")   # (1, 64600)
with torch.no_grad():
    feat = model.front_end(x)
print("front_end output shape:", tuple(feat.shape))

# save the features: this becomes the TA's input for verification
feat.numpy().astype(np.float32).tofile("features.bin")

# reference logits through the classifier alone
with torch.no_grad():
    out = model.classifier(feat).view(1, -1)
print("classifier logits:", out.numpy())

# export classifier-only ONNX with that fixed input shape
dummy = torch.randn(*feat.shape)
torch.onnx.export(model.classifier, dummy, "rawtfnet_classifier.onnx",
                  input_names=["features"], output_names=["logits"],
                  opset_version=17, dynamic_axes=None, dynamo=False)
m = onnx.load("rawtfnet_classifier.onnx", load_external_data=True)
onnx.save_model(m, "rawtfnet_classifier.onnx", save_as_external_data=False)
print("exported rawtfnet_classifier.onnx")