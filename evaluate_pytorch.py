# evaluates the fp32 pytorch model on the test set and reports:
# overall accuracy, per-class accuracy, confusion matrix and the
# sklearn classification report (precision / recall / f1).

import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# configuration
# best checkpoint selected by validation accuracy
CHECKPOINT_PATH = "./checkpoints/best_validation_full.pth"
TEST_DIR = "./data/test"
NUM_CLASSES = 24
INPUT_SIZE = 224
BATCH_SIZE = 32

# class names
# alphabetical order, identical to the imagefolder class indices
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

# device
# use the gpu when available, otherwise fall back to the cpu
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)

# test preprocessing
# resize and imagenet normalization, no augmentation
test_transform = transforms.Compose([
    transforms.Resize(
        (INPUT_SIZE, INPUT_SIZE)
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

# load test dataset
test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)
# keep the original order (shuffle=False) so results are reproducible
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)
print("\nTest dataset loaded.")
print("Number of test images:", len(test_dataset))
print("Number of classes:", len(test_dataset.classes))
print("\nDataset class mapping:")
for index, class_name in enumerate(test_dataset.classes):
    print(f"{index:2d}: {class_name}")

# load checkpoint
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)
print("\nCheckpoint loaded.")
print(
    f"Checkpoint epoch: "
    f"{checkpoint['epoch']}"
)
print(
    f"Validation accuracy: "
    f"{checkpoint['val_accuracy']:.4f}"
)
if "best_val_accuracy" in checkpoint:
    print(
        f"Best validation accuracy: "
        f"{checkpoint['best_val_accuracy']:.4f}"
    )

# recreate mobilenetv2
# architecture only, the trained weights are loaded in the next step
model = models.mobilenet_v2(
    weights=None
)
# classifier head with dropout, matching the saved weights' structure
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(
        1280,
        NUM_CLASSES
    )
)

# load trained weights
model.load_state_dict(
    checkpoint["model_state_dict"]
)
model.to(device)
model.eval()
print("\nMobileNetV2 model loaded successfully.")

# evaluate
# collect every prediction and target for the metrics below
all_predictions = []
all_targets = []
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # the predicted class is the one with the highest logit
        predictions = torch.argmax(
            outputs,
            dim=1
        )
        correct += (
            predictions == labels
        ).sum().item()
        total += labels.size(0)
        all_predictions.extend(
            predictions.cpu().numpy()
        )
        all_targets.extend(
            labels.cpu().numpy()
        )

# overall accuracy
accuracy = 100.0 * correct / total
print("\n========================================")
print("FP32 MobileNetV2 TEST RESULTS")
print("========================================")
print(f"Test images: {total}")
print(f"Correct:     {correct}")
print(f"Incorrect:   {total - correct}")
print(f"Accuracy:    {accuracy:.2f}%")

# per-class accuracy
all_predictions = np.array(
    all_predictions
)
all_targets = np.array(
    all_targets
)
print("\n========================================")
print("PER-CLASS ACCURACY")
print("========================================")
for class_index, class_name in enumerate(
    test_dataset.classes
):
    # select only the samples whose true label is this class
    class_mask = (
        all_targets == class_index
    )
    class_total = class_mask.sum()
    class_correct = (
        all_predictions[class_mask]
        == class_index
    ).sum()
    # guard against division by zero for empty classes
    class_accuracy = (
        100.0 * class_correct / class_total
        if class_total > 0
        else 0.0
    )
    print(
        f"{class_name:20s} "
        f"{class_correct:4d}/{class_total:<4d} "
        f"{class_accuracy:6.2f}%"
    )

# confusion matrix
# rows are true classes, columns are predicted classes
cm = confusion_matrix(
    all_targets,
    all_predictions,
    labels=range(NUM_CLASSES)
)
print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")
print(cm)

# classification report
print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")
print(
    classification_report(
        all_targets,
        all_predictions,
        labels=range(NUM_CLASSES),
        target_names=test_dataset.classes,
        digits=4,
        zero_division=0
    )
)
