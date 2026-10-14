# real-time hand gesture recognition demo using a webcam and the pytorch model.
# shows the predicted gesture, confidence, smoothed fps and inference time
# on the live video. predictions below the confidence threshold are shown as
# "uncertain". press 'q' to quit.

import cv2
import torch
import torchvision
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import time

# configuration
# minimum softmax probability to display a gesture name
CONFIDENCE_THRESHOLD = 0.50
CAMERA_INDEX = 0

# project paths
# all paths are relative to the folder that contains this script
PROJECT_ROOT = Path(__file__).resolve().parent
# best checkpoint selected by validation accuracy
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_validation_full.pth"
)
TRAIN_DIR = (
    PROJECT_ROOT
    / "data"
    / "train"
)

# classes
# class names come from the train folders, in the same sorted order as training
classes = sorted(
    folder.name
    for folder in TRAIN_DIR.iterdir()
    if folder.is_dir()
)
NUM_CLASSES = len(classes)

# device
# use the gpu when available, otherwise fall back to the cpu
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA:", torch.cuda.is_available())
print("Device:", device)
if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )
    print(
        "CUDA version:",
        torch.version.cuda
    )

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
    "\nLoading checkpoint:",
    CHECKPOINT_PATH
)
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)
model.load_state_dict(
    checkpoint["model_state_dict"]
)
model = model.to(device)
model.eval()
print(
    "Checkpoint epoch:",
    checkpoint["epoch"]
)
print(
    "Validation accuracy:",
    f"{checkpoint['val_accuracy']:.2f}%"
)

# image transformation
# converts a camera frame (numpy array) into a normalized model input
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(
        (224, 224)
    ),
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

# webcam
print("\nOpening webcam...")
cap = cv2.VideoCapture(
    CAMERA_INDEX
)
# stop early if the camera cannot be opened
if not cap.isOpened():
    print(
        "ERROR: Could not open webcam."
    )
    raise SystemExit

# optional camera resolution
cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)
cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)
print("\nReal-time recognition started.")
print("Press 'q' to quit.")

# fps variables
previous_time = time.perf_counter()
fps = 0.0
# smooth fps (exponential moving average factor)
fps_alpha = 0.1

# inference loop
while True:
    ret, frame = cap.read()
    if not ret:
        print(
            "ERROR: Failed to read frame."
        )
        break

    # prepare image
    # preprocess and add a batch dimension
    image = transform(frame)
    image = image.unsqueeze(0)
    image = image.to(
        device,
        non_blocking=True
    )

    # inference
    # wait for pending gpu work so only the inference itself is timed
    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(image)
        # convert logits to probabilities and take the top class
        probabilities = torch.softmax(
            outputs,
            dim=1
        )
        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_end = time.perf_counter()

    # metrics
    inference_time = (
        inference_end
        - inference_start
    )
    inference_ms = (
        inference_time * 1000
    )
    current_time = time.perf_counter()
    # time between two frames, includes capture and drawing
    frame_time = (
        current_time
        - previous_time
    )
    previous_time = current_time
    if frame_time > 0:
        current_fps = (
            1.0 / frame_time
        )
        # exponential moving average keeps the displayed fps stable
        fps = (
            fps_alpha * current_fps
            + (1 - fps_alpha) * fps
        )

    # prediction
    predicted_class = (
        classes[predicted.item()]
    )
    confidence_value = (
        confidence.item()
    )
    confidence_percent = (
        confidence_value * 100
    )

    # display prediction
    # only show the gesture name when the model is confident enough
    if (
        confidence_value
        >= CONFIDENCE_THRESHOLD
    ):
        gesture_text = (
            f"Gesture: "
            f"{predicted_class}"
        )
    else:
        gesture_text = (
            "Gesture: uncertain"
        )

    # draw information
    # black background box so the text is readable on any image
    cv2.rectangle(
        frame,
        (10, 10),
        (500, 145),
        (0, 0, 0),
        -1
    )
    cv2.putText(
        frame,
        gesture_text,
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )
    cv2.putText(
        frame,
        (
            f"Confidence: "
            f"{confidence_percent:.1f}%"
        ),
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    cv2.putText(
        frame,
        (
            f"FPS: "
            f"{fps:.1f}"
        ),
        (25, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    cv2.putText(
        frame,
        (
            f"Inference: "
            f"{inference_ms:.2f} ms"
        ),
        (25, 138),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # display frame
    cv2.imshow(
        "HaGRID Gesture Recognition - Epoch 37",
        frame
    )

    # exit
    # waitkey(1) also lets the window refresh, 'q' stops the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# cleanup
# release the camera and close the preview window
cap.release()
cv2.destroyAllWindows()
print("\nReal-time recognition stopped.")
