# measures the computational complexity of the trained model with thop:
# number of parameters, macs and flops for a single 224x224 rgb image.

import torch
import torchvision
import torch.nn as nn
from pathlib import Path
from thop import profile


def main():
    print("PyTorch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)

    # paths
    # all paths are relative to the folder that contains this script
    PROJECT_ROOT = Path(__file__).resolve().parent
    # best checkpoint selected by validation accuracy
    CHECKPOINT_PATH = (
        PROJECT_ROOT
        / "checkpoints"
        / "best_validation_full.pth"
    )

    # device
    # use the gpu when available, otherwise fall back to the cpu
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("\nDevice:", device)
    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # classes
    DATA_DIR = PROJECT_ROOT / "data"
    TRAIN_DIR = DATA_DIR / "train"
    # each sub-folder of the train directory is one gesture class
    classes = sorted(
        folder.name
        for folder in TRAIN_DIR.iterdir()
        if folder.is_dir()
    )
    NUM_CLASSES = len(classes)
    print(
        "\nNumber of classes:",
        NUM_CLASSES
    )

    # load mobilenetv2
    print("\nLoading MobileNetV2...")
    # architecture only, the trained weights come from the checkpoint
    model = torchvision.models.mobilenet_v2(
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
    print(
        "Checkpoint epoch:",
        checkpoint["epoch"]
    )
    print(
        "Validation accuracy:",
        f"{checkpoint['val_accuracy']:.2f}%"
    )

    # evaluation mode
    model = model.to(device)
    model.eval()

    # dummy input
    # a single 224x224 rgb image, the cost is reported per image
    input_tensor = torch.randn(
        1,
        3,
        224,
        224
    ).to(device)

    # flops / macs analysis
    print(
        "\nCalculating model complexity..."
    )
    macs, params = profile(
        model,
        inputs=(input_tensor,),
        verbose=False
    )
    # thop reports macs, and one mac is counted as two flops
    flops = 2 * macs

    # convert units
    params_m = params / 1e6
    macs_m = macs / 1e6
    flops_m = flops / 1e6
    flops_g = flops / 1e9

    # results
    print(
        "\n"
        + "=" * 60
    )
    print(
        "MODEL COMPLEXITY ANALYSIS"
    )
    print(
        "=" * 60
    )
    print(
        "Model:",
        "MobileNetV2"
    )
    print(
        "Checkpoint epoch:",
        checkpoint["epoch"]
    )
    print(
        "Validation accuracy:",
        f"{checkpoint['val_accuracy']:.2f}%"
    )
    print(
        "Input shape:",
        "1 x 3 x 224 x 224"
    )
    print(
        "Number of classes:",
        NUM_CLASSES
    )
    print(
        f"\nParameters: {params_m:.3f} M"
    )
    print(
        f"MACs:       {macs_m:.3f} M"
    )
    print(
        f"FLOPs:      {flops_m:.3f} M"
    )
    print(
        f"FLOPs:      {flops_g:.3f} GFLOPs"
    )
    print(
        "\nFLOPs convention:"
    )
    print(
        "1 MAC = 2 FLOPs"
    )
    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()
