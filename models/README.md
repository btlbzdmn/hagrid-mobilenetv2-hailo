# models

Exported and compiled models live here and are not tracked by git.

| File | Produced by |
|---|---|
| `hagrid_mobilenetv2_epoch37.onnx` | `export_onnx_runtime.py` (written to `onnx/`, copy it here), used by `verify_onnx_export.py` and `evaluate_onnx.py` |
| `hagrid_mobilenetv2_epoch37_hailo.onnx` | `export_onnx_hailo.py`, input for the Hailo parser |
| `hagrid_mobilenetv2.har` | Hailo parser (ONNX to HAR), input for `compile_hailo_hef.py` |
| `hagrid_mobilenetv2.hef` | `compile_hailo_hef.py`, the model that runs on the Hailo-10H |

Compiled models, when available, are attached to the GitHub Releases of this repository.
