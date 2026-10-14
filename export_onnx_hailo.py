# exports the trained pytorch model to onnx in a form suitable for the hailo
# dataflow compiler: older opset (13), fixed batch size, legacy exporter
# (dynamo=False) and a single self-contained file (no external data).
# the exported model is then validated with the onnx checker.

import torch
import torch.nn as nn
from torchvision import models
import onnx

# configuration
CHECKPOINT_PATH = "./checkpoints/best_validation_full.pth"
ONNX_PATH = "./models/hagrid_mobilenetv2_epoch37_hailo.onnx"
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
print(f"Best validation accuracy: {checkpoint['best_val_accuracy']:.4f}")

# recreate the mobilenetv2 architecture
# architecture only, the trained weights are loaded in the next step
model = models.mobilenet_v2(weights=None)
# classifier head with dropout, matching the saved weights' structure
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, NUM_CLASSES)
)

# load trained weights
model.load_state_dict(checkpoint["model_state_dict"])
# eval mode is required so that batchnorm and dropout are exported correctly
model.eval()
model.cpu()
print("\nModel weights loaded successfully.")
print(model.classifier)

# create dummy input
# onnx export traces the model with an example input (nchw layout)
dummy_input = torch.randn(
    1,
    3,
    INPUT_SIZE,
    INPUT_SIZE
)

# export to onnx
# no dynamic_axes here: hailo needs a fixed input shape
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    export_params=True,
    dynamo=False,
    external_data=False
)
print(f"\nONNX export completed: {ONNX_PATH}")

# verify onnx model
# raises an exception if the model structure is invalid
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

# print input/output information
# these names are needed later when parsing the model with hailo
print("\nONNX Inputs:")
for inp in onnx_model.graph.input:
    print(inp.name)

print("\nONNX Outputs:")
for out in onnx_model.graph.output:
    print(out.name)
