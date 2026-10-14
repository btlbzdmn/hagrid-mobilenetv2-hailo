# compares two checkpoints on the held-out test set:
# the latest epoch (checkpoint_full.pth) versus the epoch with the best
# validation accuracy (best_validation_full.pth).
# this shows which checkpoint generalizes better to unseen data.

import torch
import torchvision
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn


class loss(nn.Module):
    """Classification loss compatible with the evaluation loop."""

    # thin wrapper around cross entropy that validates the input shapes
    # (note: not used by the script below, the plain nn.CrossEntropyLoss is used)
    def __init__(self, weight=None, ignore_index=-100, label_smoothing=0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    def forward(self, outputs, labels):
        # outputs must be (batch, classes) and labels must be (batch,)
        if outputs.ndim < 2:
            raise ValueError("outputs must have a class dimension")

        if labels.ndim != 1:
            raise ValueError("labels must contain one class index per sample")

        if outputs.shape[0] != labels.shape[0]:
            raise ValueError("outputs and labels must have matching batch sizes")

        return self.criterion(outputs, labels)


def evaluate(model, data_loader, criterion, device):
    # switch to eval mode so dropout and batchnorm behave deterministically
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    # no gradients are needed during evaluation
    with torch.no_grad():
        for images, labels in data_loader:
            # move the batch to the target device (gpu or cpu)
            images = images.to(
                device,
                non_blocking=True
            )
            labels = labels.to(
                device,
                non_blocking=True
            )
            outputs = model(images)
            loss = criterion(
                outputs,
                labels
            )
            running_loss += loss.item()
            # the predicted class is the one with the highest logit
            _, predicted = torch.max(
                outputs,
                1
            )
            total += labels.size(0)
            correct += (
                predicted == labels
            ).sum().item()

    # average loss over batches and accuracy over all samples
    loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total
    return loss, accuracy


def load_model(checkpoint_path, num_classes, device):
    # rebuild the architecture without pretrained weights,
    # the trained weights come from the checkpoint
    model = models.mobilenet_v2(
        weights=None
    )
    # the classifier head must match the one used during training
    model.classifier[1] = nn.Linear(
        in_features=1280,
        out_features=num_classes
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model = model.to(device)
    model.eval()
    return model, checkpoint


def main():
    print("PyTorch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
    print("CUDA:", torch.cuda.is_available())

    # paths
    # all paths are relative to the folder that contains this script
    PROJECT_ROOT = Path(__file__).resolve().parent
    TEST_DIR = PROJECT_ROOT / "data" / "test"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    # last epoch of training
    LAST_CHECKPOINT = (
        CHECKPOINT_DIR / "checkpoint_full.pth"
    )
    # epoch with the highest validation accuracy
    BEST_CHECKPOINT = (
        CHECKPOINT_DIR / "best_validation_full.pth"
    )

    # device
    # use the gpu when available, otherwise fall back to the cpu
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("\nTesting device:", device)
    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )
        print(
            "CUDA version:",
            torch.version.cuda
        )

    # test dataset
    # same preprocessing as validation: resize and imagenet normalization
    test_transform = transforms.Compose([
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
    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=test_transform
    )
    print("\nTest dataset:")
    print(
        "Test images:",
        len(test_dataset)
    )
    print(
        "Number of classes:",
        len(test_dataset.classes)
    )
    print("\nClasses:")
    for i, class_name in enumerate(
        test_dataset.classes
    ):
        print(
            f"{i}: {class_name}"
        )

    # test dataloader
    BATCH_SIZE = 32
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print("\nTest batches:")
    print(len(test_loader))

    # loss function
    criterion = nn.CrossEntropyLoss()

    # check checkpoints
    # make sure both checkpoint files exist before loading them
    print("\nCheckpoint status:")
    print(
        "Latest checkpoint:",
        LAST_CHECKPOINT.exists()
    )
    print(
        "Best validation checkpoint:",
        BEST_CHECKPOINT.exists()
    )

    # test latest epoch checkpoint
    print("\n")
    print("=" * 60)
    print("TESTING LATEST CHECKPOINT")
    print("=" * 60)
    latest_model, latest_checkpoint = load_model(
        LAST_CHECKPOINT,
        len(test_dataset.classes),
        device
    )
    latest_loss, latest_accuracy = evaluate(
        latest_model,
        test_loader,
        criterion,
        device
    )
    print(
        "\nCheckpoint:",
        LAST_CHECKPOINT
    )
    print(
        "Epoch:",
        latest_checkpoint["epoch"]
    )
    print(
        "Validation Accuracy:",
        f'{latest_checkpoint["val_accuracy"]:.2f}%'
    )
    print(
        "Test Loss:",
        f"{latest_loss:.4f}"
    )
    print(
        "Test Accuracy:",
        f"{latest_accuracy:.2f}%"
    )

    # test best validation checkpoint
    print("\n")
    print("=" * 60)
    print("TESTING BEST VALIDATION CHECKPOINT")
    print("=" * 60)
    best_model, best_checkpoint = load_model(
        BEST_CHECKPOINT,
        len(test_dataset.classes),
        device
    )
    best_loss, best_accuracy = evaluate(
        best_model,
        test_loader,
        criterion,
        device
    )
    print(
        "\nCheckpoint:",
        BEST_CHECKPOINT
    )
    print(
        "Epoch:",
        best_checkpoint["epoch"]
    )
    print(
        "Validation Accuracy:",
        f'{best_checkpoint["val_accuracy"]:.2f}%'
    )
    print(
        "Test Loss:",
        f"{best_loss:.4f}"
    )
    print(
        "Test Accuracy:",
        f"{best_accuracy:.2f}%"
    )

    # comparison
    print("\n")
    print("=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(
        f"Latest epoch ({latest_checkpoint['epoch']}): "
        f"{latest_accuracy:.2f}% test accuracy"
    )
    print(
        f"Best validation epoch ({best_checkpoint['epoch']}): "
        f"{best_accuracy:.2f}% test accuracy"
    )
    # positive difference means the latest checkpoint is better
    difference = (
        latest_accuracy -
        best_accuracy
    )
    print(
        f"Difference: {difference:+.2f} percentage points"
    )
    if latest_accuracy > best_accuracy:
        print(
            "\n>>> Latest checkpoint performs better "
            "on the test set."
        )
    elif best_accuracy > latest_accuracy:
        print(
            "\n>>> Best-validation checkpoint performs "
            "better on the test set."
        )
    else:
        print(
            "\n>>> Both checkpoints have the same "
            "test accuracy."
        )


# run only when executed directly (required for dataloader workers)
if __name__ == "__main__":
    main()
