# sanity check for the onnx export: runs the same random input through the
# pytorch model and the onnx model, then compares the raw outputs and the
# predicted classes. small numerical differences are expected.

import torch
import torch.nn as nn
from torchvision import models
import onnxruntime as ort
import numpy as np

# configuration
CHECKPOINT_PATH = "./checkpoints/best_validation_full.pth"
ONNX_PATH = "./models/hagrid_mobilenetv2_epoch37.onnx"
NUM_CLASSES = 24
INPUT_SIZE = 224

# load checkpoint
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)
print("Checkpoint loaded.")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Validation accuracy: {checkpoint['val_accuracy']:.4f}")

# recreate pytorch model
# architecture only, the trained weights are loaded in the next step
model = models.mobilenet_v2(weights=None)
# classifier head with dropout, matching the saved weights' structure
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, NUM_CLASSES)
)
model.load_state_dict(checkpoint["model_state_dict"])
# eval mode disables dropout and uses batchnorm running statistics
model.eval()
model.cpu()
print("PyTorch model loaded successfully.")

# create identical test input
# fixed seed so the comparison is reproducible
torch.manual_seed(42)
test_input = torch.randn(
    1,
    3,
    INPUT_SIZE,
    INPUT_SIZE
)
print("\nTest input shape:")
print(test_input.shape)

# pytorch inference
with torch.no_grad():
    pytorch_output = model(test_input)

pytorch_output = pytorch_output.numpy()
print("\nPyTorch output shape:")
print(pytorch_output.shape)

# load onnx model
session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"]
)
print("\nONNX Runtime model loaded successfully.")
print("ONNX inputs:")
for inp in session.get_inputs():
    print(f"  {inp.name}: {inp.shape}")

print("ONNX outputs:")
for out in session.get_outputs():
    print(f"  {out.name}: {out.shape}")

# onnx runtime inference
# onnx runtime expects a float32 numpy array
onnx_input = test_input.numpy().astype(np.float32)
# "input" and "output" are the names chosen during export
onnx_output = session.run(
    ["output"],
    {"input": onnx_input}
)[0]
print("\nONNX output shape:")
print(onnx_output.shape)

# compare outputs
# element-wise absolute difference between the two logit vectors
difference = np.abs(
    pytorch_output - onnx_output
)
max_difference = difference.max()
mean_difference = difference.mean()
print("\n==============================")
print("OUTPUT COMPARISON")
print("==============================")
print(f"Maximum absolute difference: {max_difference:.10f}")
print(f"Mean absolute difference:    {mean_difference:.10f}")

# compare predicted classes
pytorch_class = np.argmax(pytorch_output, axis=1)[0]
onnx_class = np.argmax(onnx_output, axis=1)[0]
print("\n==============================")
print("PREDICTION COMPARISON")
print("==============================")
print(f"PyTorch predicted class: {pytorch_class}")
print(f"ONNX predicted class:    {onnx_class}")
if pytorch_class == onnx_class:
    print("\nPASS: Both models predict the same class.")
else:
    print("\nWARNING: Models predict different classes.")

# final numerical check
# strict check: all logits must match within the given tolerances
if np.allclose(
    pytorch_output,
    onnx_output,
    rtol=1e-3,
    atol=1e-5
):
    print("PASS: PyTorch and ONNX outputs match.")
else:
    print("WARNING: PyTorch and ONNX outputs differ.")
