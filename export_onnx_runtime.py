# exports the trained pytorch model to onnx for use with onnx runtime.
# uses opset 17 and a dynamic batch dimension.
# (for the hailo compiler use export_onnx_hailo.py instead.)

import torch
import torchvision
import torch.nn as nn
from pathlib import Path


def main():
    print("PyTorch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
    print("CUDA:", torch.cuda.is_available())

    # paths
    # all paths are relative to the folder that contains this script
    PROJECT_ROOT = Path(__file__).resolve().parent
    # best checkpoint selected by validation accuracy
    CHECKPOINT_PATH = (
        PROJECT_ROOT
        / "checkpoints"
        / "best_validation_full.pth"
    )
    # the exported model is written to the onnx folder
    ONNX_DIR = PROJECT_ROOT / "onnx"
    ONNX_DIR.mkdir(exist_ok=True)
    ONNX_PATH = (
        ONNX_DIR
        / "mobilenetv2_hagrid_epoch37.onnx"
    )

    # device
    # use the gpu when available, otherwise fall back to the cpu
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("\nExport device:", device)

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
    print("\nNumber of classes:", NUM_CLASSES)
    print("Classes:")
    for i, class_name in enumerate(classes):
        print(f"{i}: {class_name}")

    # model
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
    # eval mode is required so that batchnorm and dropout are exported correctly
    model = model.to(device)
    model.eval()

    # dummy input
    # onnx export traces the model with an example input (nchw layout)
    dummy_input = torch.randn(
        1,
        3,
        224,
        224,
        device=device
    )

    # onnx export
    print("\nExporting model to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        # allow any batch size at inference time
        dynamic_axes={
            "input": {
                0: "batch_size"
            },
            "output": {
                0: "batch_size"
            }
        }
    )
    print("\nONNX export completed!")
    print(
        "ONNX model:",
        ONNX_PATH
    )
    print(
        "Model size:",
        f"{ONNX_PATH.stat().st_size / (1024 * 1024):.2f} MB"
    )


if __name__ == "__main__":
    main()
