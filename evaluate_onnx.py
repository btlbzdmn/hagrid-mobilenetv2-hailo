# evaluates the exported onnx model with onnx runtime on the test set.
# preprocessing is done manually with pil and numpy (no torchvision),
# which mimics what a deployment environment would do.
# prints overall accuracy and per-class accuracy.

import onnxruntime as ort
import numpy as np
from PIL import Image
from pathlib import Path

# configuration
ONNX_PATH = "./models/hagrid_mobilenetv2_epoch37.onnx"
TEST_DIR = Path("./data/test")
INPUT_SIZE = 224
NUM_CLASSES = 24

# class names
# alphabetical order, identical to the training class indices
CLASS_NAMES = [
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

# image preprocessing
# imagenet statistics, same values as used during training
MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)
STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def preprocess_image(image_path):
    # load as rgb (drops alpha channel, converts grayscale)
    image = Image.open(
        image_path
    ).convert("RGB")
    image = image.resize(
        (INPUT_SIZE, INPUT_SIZE)
    )
    # scale pixel values from [0, 255] to [0, 1]
    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0
    # normalize each channel with the imagenet mean and std
    image = (
        image - MEAN
    ) / STD
    # hwc -> chw
    image = np.transpose(
        image,
        (2, 0, 1)
    )
    # add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )
    return image.astype(np.float32)

# load onnx runtime model
# run on the cpu so the result does not depend on a gpu
session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"]
)
# read the tensor names from the model instead of hard-coding them
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("ONNX Runtime model loaded.")
print("Input:", input_name)
print("Output:", output_name)

# collect test images
# list of (image path, class index) pairs
samples = []
for class_index, class_name in enumerate(CLASS_NAMES):
    class_dir = TEST_DIR / class_name
    # sorted for a deterministic evaluation order
    image_paths = sorted(
        class_dir.glob("*.jpg")
    )
    for image_path in image_paths:
        samples.append(
            (
                image_path,
                class_index
            )
        )

print("\n========================================")
print("TEST DATASET")
print("========================================")
print("Test images:", len(samples))
print("Classes:", NUM_CLASSES)

# evaluate
correct = 0
total = 0
# per-class counters
class_correct = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)
class_total = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)
all_predictions = []
all_targets = []
# images are processed one at a time (batch size 1)
for image_path, target in samples:
    input_tensor = preprocess_image(
        image_path
    )
    output = session.run(
        [output_name],
        {
            input_name: input_tensor
        }
    )[0]
    # the predicted class is the one with the highest logit
    prediction = int(
        np.argmax(output, axis=1)[0]
    )
    all_predictions.append(
        prediction
    )
    all_targets.append(
        target
    )
    class_total[target] += 1
    if prediction == target:
        correct += 1
        class_correct[target] += 1

    total += 1

# overall accuracy
accuracy = (
    100.0 * correct / total
)
print("\n========================================")
print("ONNX FP32 TEST RESULTS")
print("========================================")
print(f"Test images: {total}")
print(f"Correct:     {correct}")
print(f"Incorrect:   {total - correct}")
print(f"Accuracy:    {accuracy:.2f}%")

# per-class accuracy
print("\n========================================")
print("PER-CLASS ACCURACY")
print("========================================")
for index, class_name in enumerate(
    CLASS_NAMES
):
    # guard against division by zero for empty classes
    if class_total[index] > 0:
        class_accuracy = (
            100.0
            * class_correct[index]
            / class_total[index]
        )
    else:
        class_accuracy = 0.0

    print(
        f"{class_name:20s} "
        f"{class_correct[index]:4d}/"
        f"{class_total[index]:<4d} "
        f"{class_accuracy:6.2f}%"
    )
