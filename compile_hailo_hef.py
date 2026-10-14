# compiles the model for the hailo-10h accelerator:
# 1. builds a calibration set from random training images
# 2. loads the parsed model (.har) into the hailo dataflow compiler
# 3. quantizes the model (optimize) using the calibration set
# 4. compiles it into a .hef file that runs on the device
# note: the onnx -> har parsing step must be done beforehand.

import os

# force tensorflow/hailo optimization to use cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

import numpy as np
from pathlib import Path
from PIL import Image
from hailo_sdk_client import ClientRunner

# configuration
# all paths are relative to the folder that contains this script
PROJECT_ROOT = Path(__file__).resolve().parent
HAR_PATH = PROJECT_ROOT / "models" / "hagrid_mobilenetv2.har"
HEF_PATH = PROJECT_ROOT / "models" / "hagrid_mobilenetv2.hef"
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
INPUT_SIZE = 224
# number of images used to calibrate the quantization ranges
CALIBRATION_SIZE = 512

# collect calibration images
print("Collecting calibration images...")
image_paths = []
# walk through every class folder and gather all image files
for class_dir in sorted(TRAIN_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    for image_path in class_dir.iterdir():
        if image_path.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp"
        ]:
            image_paths.append(image_path)

print("Total training images found:", len(image_paths))
# use a fixed subset for reproducibility
rng = np.random.default_rng(42)
selected_indices = rng.choice(
    len(image_paths),
    size=min(CALIBRATION_SIZE, len(image_paths)),
    replace=False
)
selected_paths = [
    image_paths[i]
    for i in selected_indices
]

# load and preprocess calibration images
calib_dataset = []
for image_path in selected_paths:
    image = Image.open(image_path).convert("RGB")
    image = image.resize(
        (INPUT_SIZE, INPUT_SIZE)
    )
    image = np.asarray(
        image,
        dtype=np.float32
    )
    # scale from [0,255] to [0,1]
    image = image / 255.0
    # apply the exact same normalization used by the onnx model
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )
    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )
    image = (image - mean) / std
    # keep nhwc layout for hailo
    calib_dataset.append(image)

# stack into a single array of shape (n, 224, 224, 3)
calib_dataset = np.stack(
    calib_dataset,
    axis=0
).astype(np.float32)
print("\nCalibration dataset:")
print("Shape:", calib_dataset.shape)
print("Dtype:", calib_dataset.dtype)
print(
    "Value range:",
    calib_dataset.min(),
    "to",
    calib_dataset.max()
)

# load har
print("\nLoading HAR...")
# hw_arch selects the target accelerator
runner = ClientRunner(
    har=str(HAR_PATH),
    hw_arch="hailo10h"
)
print("HAR loaded successfully.")

# quantization / optimization
print("\nStarting quantization...")
print("This may take some time.")
# converts the float model to int8 using the calibration images
runner.optimize(
    calib_dataset
)
print("\nQuantization completed successfully.")

# compile to hef
print("\nCompiling model to HEF...")
# returns the compiled binary as bytes
hef_buffer = runner.compile()
print("Compilation completed successfully.")

# save hef
with open(
    HEF_PATH,
    "wb"
) as f:
    f.write(hef_buffer)

print("\n==========================================")
print("HEF compilation successful!")
print("==========================================")
print("HEF:", HEF_PATH)
print("Size:", HEF_PATH.stat().st_size, "bytes")
