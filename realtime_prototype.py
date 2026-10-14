# early prototype of the webcam demo (superseded by demo_webcam_pytorch.py).
# it is simpler: no fps / latency display and no confidence threshold, but it
# converts frames from bgr to rgb before inference. it uses the latest
# checkpoint (checkpoint_full.pth) and a hard-coded class list.
# press 'q' to quit.

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path

# configuration
# all paths are relative to the folder that contains this script
PROJECT_ROOT = Path(__file__).resolve().parent
# latest checkpoint of the full fine-tuning run
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "checkpoint_full.pth"
)
NUM_CLASSES = 24
# alphabetical order, identical to the training class indices
CLASSES = [
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

# device
# use the gpu when available, otherwise fall back to the cpu
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# transform
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
model = model.to(device)

# load checkpoint
print("\nLoading checkpoint:")
print(CHECKPOINT_PATH)
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)
model.load_state_dict(
    checkpoint["model_state_dict"]
)
model.eval()
print("\nCheckpoint loaded successfully.")
# these keys are optional, so check before printing
if "epoch" in checkpoint:
    print(
        "Checkpoint epoch:",
        checkpoint["epoch"]
    )

if "val_accuracy" in checkpoint:
    print(
        "Validation accuracy:",
        f"{checkpoint['val_accuracy']:.2f}%"
    )

if "best_val_accuracy" in checkpoint:
    print(
        "Best validation accuracy:",
        f"{checkpoint['best_val_accuracy']:.2f}%"
    )

# webcam
print("\nOpening webcam...")
cap = cv2.VideoCapture(0)
# stop early if the camera cannot be opened
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam opened successfully.")
print("Press 'q' to quit.")

# real-time inference
while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read frame.")
        break

    # prepare image
    # opencv gives bgr frames, convert to rgb because the model was trained on rgb images
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )
    input_tensor = transform(
        rgb_frame
    )
    # add a batch dimension
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(
        device
    )

    # inference
    with torch.no_grad():
        outputs = model(
            input_tensor
        )
        # convert logits to probabilities and take the top class
        probabilities = torch.softmax(
            outputs,
            dim=1
        )
        confidence, predicted = torch.max(
            probabilities,
            1
        )

    predicted_class = CLASSES[
        predicted.item()
    ]
    confidence_value = (
        confidence.item() * 100
    )

    # display
    text = (
        f"{predicted_class}: "
        f"{confidence_value:.1f}%"
    )
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    # device information
    device_text = (
        f"Device: {device}"
    )
    cv2.putText(
        frame,
        device_text,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    # show frame
    cv2.imshow(
        "HaGRID Real-Time Gesture Recognition",
        frame
    )

    # quit
    # waitkey(1) also lets the window refresh, 'q' stops the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# cleanup
# release the camera and close the preview window
cap.release()
cv2.destroyAllWindows()
print("\nReal-time test finished.")
