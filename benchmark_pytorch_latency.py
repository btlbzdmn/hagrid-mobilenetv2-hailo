# measures the pure inference latency of the pytorch model using a random
# input (no data loading, no preprocessing, no display).
# reports average / minimum / maximum latency and the theoretical fps.

import torch
import torchvision
import torch.nn as nn
import time
from pathlib import Path
from PIL import Image
from torchvision import models, transforms

# configuration
NUM_CLASSES = 24
# warm-up runs are excluded from the measurement
NUM_WARMUP = 20
NUM_ITERATIONS = 100
# all paths are relative to the folder that contains this script
PROJECT_ROOT = Path(__file__).resolve().parent
# latest checkpoint of the full fine-tuning run
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "checkpoint_full.pth"
)

# device
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA:", torch.cuda.is_available())
# use the gpu when available, otherwise fall back to the cpu
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print("\nBenchmark device:", device)
if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )
    print(
        "CUDA version:",
        torch.version.cuda
    )

# classes
# alphabetical order, identical to the training class indices
# (not used in the measurement itself)
classes = [
    "call",
    "dislike",
    "fist",
    "four",
    "grabbing",
    "grip",
    "like",
    "middle_finger",
    "mute",
    "no_gesture",
    "ok",
    "one",
    "palm",
    "peace",
    "peace_inverted",
    "point",
    "rock",
    "stop",
    "stop_inverted",
    "three",
    "three2",
    "three3",
    "two_up",
    "two_up_inverted"
]

# transform
# preprocessing pipeline (not used in the measurement, the input is random)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])

# model
print("\nLoading MobileNetV2...")
# architecture only, the trained weights come from the checkpoint
model = models.mobilenet_v2(
    weights=None
)
# the classifier head must match the one used during training
model.classifier[1] = nn.Linear(
    in_features=1280,
    out_features=NUM_CLASSES
)

# load checkpoint
print(
    "Loading checkpoint:",
    CHECKPOINT_PATH
)
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)
model.load_state_dict(
    checkpoint["model_state_dict"]
)
print(
    "Checkpoint epoch:",
    checkpoint["epoch"]
)
print(
    "Validation accuracy:",
    f"{checkpoint['val_accuracy']:.2f}%"
)

# move model to device
model = model.to(device)
model.eval()

# dummy input
# a single random 224x224 rgb image (batch size 1)
dummy_input = torch.randn(
    1,
    3,
    224,
    224
).to(device)

# warm-up
# the first runs are slower (cuda init, kernel selection), so run a few first
print("\nWarming up GPU...")
with torch.no_grad():
    for _ in range(NUM_WARMUP):
        _ = model(dummy_input)

# wait until all warm-up kernels have finished
if device.type == "cuda":
    torch.cuda.synchronize()

# inference benchmark
print(
    f"\nRunning {NUM_ITERATIONS} inference iterations..."
)
times = []
with torch.no_grad():
    for _ in range(NUM_ITERATIONS):
        # cuda calls are asynchronous, so synchronize before and after timing
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(
            end - start
        )

# results
total_time = sum(times)
average_latency = (
    total_time
    / NUM_ITERATIONS
)
# fps if the model alone were the only cost per frame
fps = (
    1.0
    / average_latency
)
min_latency = min(times)
max_latency = max(times)
print("\n")
print("=" * 60)
print("INFERENCE BENCHMARK")
print("=" * 60)
print(
    f"Model: MobileNetV2"
)
print(
    f"Checkpoint epoch: {checkpoint['epoch']}"
)
print(
    f"Validation Accuracy: "
    f"{checkpoint['val_accuracy']:.2f}%"
)
print(
    f"Iterations: {NUM_ITERATIONS}"
)
print(
    f"Average latency: "
    f"{average_latency * 1000:.2f} ms"
)
print(
    f"Minimum latency: "
    f"{min_latency * 1000:.2f} ms"
)
print(
    f"Maximum latency: "
    f"{max_latency * 1000:.2f} ms"
)
print(
    f"Theoretical inference FPS: "
    f"{fps:.2f}"
)
print("=" * 60)
