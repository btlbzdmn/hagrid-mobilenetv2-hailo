# stage 1 training: partial fine-tuning of an imagenet-pretrained mobilenetv2
# on the hagrid gesture dataset. the backbone is frozen except for the last
# four feature blocks, and the new classifier head is trained from scratch.
# training can be resumed: every run continues from the last saved checkpoint.

import torch
import torchvision
from pathlib import Path


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model on validation/test data.
    """

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
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            # the predicted class is the one with the highest logit
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (
                predicted == labels
            ).sum().item()

    # average loss over batches and accuracy over all samples
    loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total
    return loss, accuracy


def main():
    print("PyTorch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
    print("CUDA:", torch.cuda.is_available())

    # dataset paths
    # all paths are relative to the folder that contains this script
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_DIR = PROJECT_ROOT / "data"
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "val"
    TEST_DIR = DATA_DIR / "test"
    # checkpoints are written here
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    # latest checkpoint (overwritten after every epoch, used to resume)
    CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoint.pth"
    # checkpoint with the best validation accuracy so far
    BEST_CHECKPOINT_PATH = (
        CHECKPOINT_DIR / "best_validation_checkpoint.pth"
    )
    print("\nDataset paths:")
    print("Train:", TRAIN_DIR)
    print("Val:", VAL_DIR)
    print("Test:", TEST_DIR)

    # check dataset directories
    # quick sanity check so a wrong path is noticed immediately
    print("\nDataset directories exist:")
    print("Train:", TRAIN_DIR.exists())
    print("Val:", VAL_DIR.exists())
    print("Test:", TEST_DIR.exists())

    # find gesture classes
    # each sub-folder of the train directory is one gesture class
    classes = sorted(
        folder.name
        for folder in TRAIN_DIR.iterdir()
        if folder.is_dir()
    )
    print("\nNumber of classes:", len(classes))
    print("Classes:")
    for i, class_name in enumerate(classes):
        print(f"{i}: {class_name}")

    # load datasets
    from torchvision import datasets
    # first load without transforms, only to inspect sizes and class mapping
    train_dataset = datasets.ImageFolder(
        TRAIN_DIR
    )
    val_dataset = datasets.ImageFolder(
        VAL_DIR
    )
    test_dataset = datasets.ImageFolder(
        TEST_DIR
    )
    print("\nDataset information:")
    print(
        "Training images:",
        len(train_dataset)
    )
    print(
        "Validation images:",
        len(val_dataset)
    )
    print(
        "Test images:",
        len(test_dataset)
    )
    print(
        "Number of classes:",
        len(train_dataset.classes)
    )
    print("\nClass mapping:")
    print(train_dataset.class_to_idx)

    # image transformations
    from torchvision import transforms
    # training transform: resize + light augmentation + imagenet normalization
    train_transform = transforms.Compose([
        transforms.Resize(
            (224, 224)
        ),
        # augmentation: random mirroring and small rotations
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        # imagenet statistics, required by the pretrained weights
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
    # validation/test transform: same as training but without augmentation
    val_test_transform = transforms.Compose([
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
    # re-create datasets with transformations
    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        VAL_DIR,
        transform=val_test_transform
    )
    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=val_test_transform
    )
    # check one transformed sample
    image, label = train_dataset[0]
    print("\nAfter transformation:")
    print(
        "Image shape:",
        image.shape
    )
    print(
        "Label:",
        label
    )
    print(
        "Class:",
        train_dataset.classes[label]
    )

    # dataloaders
    from torch.utils.data import DataLoader
    BATCH_SIZE = 32
    # shuffle only the training data
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print("\nDataLoader information:")
    print(
        "Batch size:",
        BATCH_SIZE
    )
    print(
        "Training batches:",
        len(train_loader)
    )
    print(
        "Validation batches:",
        len(val_loader)
    )
    print(
        "Test batches:",
        len(test_loader)
    )

    # first training batch
    # pull one batch to verify shapes and labels
    images, labels = next(
        iter(train_loader)
    )
    print("\nFirst training batch:")
    print(
        "Images shape:",
        images.shape
    )
    print(
        "Labels shape:",
        labels.shape
    )
    print(
        "First 10 labels:",
        labels[:10]
    )

    # model
    from torchvision import models
    import torch.nn as nn
    # start from imagenet-pretrained weights
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )
    NUM_CLASSES = len(
        train_dataset.classes
    )
    # swap the final linear layer for one with the number of gesture classes
    model.classifier[1] = nn.Linear(
        in_features=1280,
        out_features=NUM_CLASSES
    )
    print("\nModified classifier:")
    print(model.classifier)

    # freeze backbone
    # freeze all backbone layers first
    for param in model.features.parameters():
        param.requires_grad = False

    # unfreeze last four feature blocks
    # then re-enable training only for the last four blocks
    for param in model.features[-4:].parameters():
        param.requires_grad = True

    print("\nUnfreezing:")
    print(
        "Last 4 MobileNetV2 feature blocks are trainable."
    )

    # parameter information
    # count trainable and frozen parameters for a quick overview
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )
    frozen_params = (
        total_params -
        trainable_params
    )
    print("\nModel parameters:")
    print(
        "Total parameters:",
        total_params
    )
    print(
        "Trainable parameters:",
        trainable_params
    )
    print(
        "Frozen parameters:",
        frozen_params
    )
    print(
        "Trainable ratio:",
        f"{100 * trainable_params / total_params:.2f}%"
    )

    # device
    # use the gpu when available, otherwise fall back to the cpu
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("\nTraining device:", device)
    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )
        print(
            "CUDA version:",
            torch.version.cuda
        )

    model = model.to(device)

    # loss
    # standard loss for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # optimizer
    # the new classifier learns faster than the pretrained backbone blocks
    BACKBONE_LR = 0.0001
    CLASSIFIER_LR = 0.001
    optimizer = torch.optim.Adam([
        {
            "params":
                model.features[-4:].parameters(),
            "lr":
                BACKBONE_LR
        },
        {
            "params":
                model.classifier[1].parameters(),
            "lr":
                CLASSIFIER_LR
        }
    ])
    print("\nTraining configuration:")
    print(
        "Loss function:",
        criterion
    )
    print(
        "Backbone learning rate:",
        BACKBONE_LR
    )
    print(
        "Classifier learning rate:",
        CLASSIFIER_LR
    )

    # checkpoint loading
    start_epoch = 0
    best_val_accuracy = 0.0
    # resume training if a checkpoint from a previous run exists
    if CHECKPOINT_PATH.exists():
        print(
            "\nCheckpoint found. Loading..."
        )
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device
        )
        # load model
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        # load optimizer if available
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )
            print(
                "Optimizer state loaded."
            )

        # load epoch
        start_epoch = checkpoint["epoch"]
        # load best validation accuracy
        if "best_val_accuracy" in checkpoint:
            best_val_accuracy = (
                checkpoint[
                    "best_val_accuracy"
                ]
            )

        print(
            "Checkpoint loaded."
        )
        print(
            "Continuing from epoch:",
            start_epoch + 1
        )
        print(
            "Best validation accuracy:",
            f"{best_val_accuracy:.2f}%"
        )
    else:
        print(
            "\nNo checkpoint found."
        )
        print(
            "Starting training from epoch 1."
        )

    # training configuration
    # number of epochs trained in each run of this script
    EPOCHS_PER_RUN = 5

    # training loop
    for epoch in range(
        start_epoch,
        start_epoch + EPOCHS_PER_RUN
    ):
        print(
            f"\n{'=' * 50}"
        )
        print(
            f"Epoch {epoch + 1}"
        )
        print(
            f"{'=' * 50}"
        )

        # training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (
            images,
            labels
        ) in enumerate(train_loader):
            images = images.to(
                device,
                non_blocking=True
            )
            labels = labels.to(
                device,
                non_blocking=True
            )
            # clear gradients
            optimizer.zero_grad()
            # forward pass
            outputs = model(images)
            # loss
            loss = criterion(
                outputs,
                labels
            )
            # backpropagation
            loss.backward()
            # update weights
            optimizer.step()

            # statistics
            running_loss += loss.item()
            _, predicted = torch.max(
                outputs,
                1
            )
            total += labels.size(0)
            correct += (
                predicted == labels
            ).sum().item()

            # progress
            # print running statistics every 50 batches and at the last batch
            if (
                (batch_idx + 1) % 50 == 0
                or
                (batch_idx + 1) ==
                len(train_loader)
            ):
                current_loss = (
                    running_loss /
                    (batch_idx + 1)
                )
                current_accuracy = (
                    100 *
                    correct /
                    total
                )
                print(
                    f"Batch [{batch_idx + 1}/"
                    f"{len(train_loader)}] "
                    f"Loss: {current_loss:.4f} "
                    f"Accuracy: "
                    f"{current_accuracy:.2f}%"
                )

        # epoch training results
        train_loss = (
            running_loss /
            len(train_loader)
        )
        train_accuracy = (
            100 *
            correct /
            total
        )
        print(
            f"\n>>> Epoch {epoch + 1} "
            f"training finished"
        )
        print(
            f">>> Training Loss: "
            f"{train_loss:.4f}"
        )
        print(
            f">>> Training Accuracy: "
            f"{train_accuracy:.2f}%"
        )

        # validation
        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device
        )
        print(
            f">>> Validation Loss: "
            f"{val_loss:.4f}"
        )
        print(
            f">>> Validation Accuracy: "
            f"{val_accuracy:.2f}%"
        )

        # check whether this is the best model
        is_best = (
            val_accuracy >
            best_val_accuracy
        )
        if is_best:
            best_val_accuracy = (
                val_accuracy
            )
            print(
                "\n>>> NEW BEST VALIDATION "
                "ACCURACY!"
            )
            print(
                f">>> Best Validation "
                f"Accuracy: "
                f"{best_val_accuracy:.2f}%"
            )

        # save regular checkpoint
        # the checkpoint stores everything needed to resume training
        checkpoint = {
            "epoch":
                epoch + 1,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "train_loss":
                train_loss,
            "train_accuracy":
                train_accuracy,
            "val_loss":
                val_loss,
            "val_accuracy":
                val_accuracy,
            "best_val_accuracy":
                best_val_accuracy
        }
        torch.save(
            checkpoint,
            CHECKPOINT_PATH
        )
        print(
            f">>> Checkpoint saved: "
            f"{CHECKPOINT_PATH}"
        )

        # save best validation checkpoint
        # keep a separate copy whenever validation accuracy improves
        if is_best:
            best_checkpoint = {
                "epoch":
                    epoch + 1,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "train_loss":
                    train_loss,
                "train_accuracy":
                    train_accuracy,
                "val_loss":
                    val_loss,
                "val_accuracy":
                    val_accuracy,
                "best_val_accuracy":
                    best_val_accuracy
            }
            torch.save(
                best_checkpoint,
                BEST_CHECKPOINT_PATH
            )
            print(
                f">>> BEST model saved: "
                f"{BEST_CHECKPOINT_PATH}"
            )

    # end of training session
    print(
        "\n=================================================="
    )
    print(
        "Training session completed."
    )
    print(
        "Latest epoch:",
        start_epoch + EPOCHS_PER_RUN
    )
    print(
        "Best validation accuracy:",
        f"{best_val_accuracy:.2f}%"
    )
    print(
        "=================================================="
    )


# run training only when executed directly (required for dataloader workers)
if __name__ == "__main__":
    main()
