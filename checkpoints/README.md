# checkpoints

Trained weights are written here by the training scripts and are not tracked by git.

| File | Written by | Content |
|---|---|---|
| `checkpoint.pth` | `train_partial_finetune.py` | latest epoch of stage 1 |
| `best_validation_checkpoint.pth` | `train_partial_finetune.py` | best validation epoch of stage 1 |
| `checkpoint_full.pth` | `train_full_finetune.py` | latest epoch of stage 2 |
| `best_validation_full.pth` | `train_full_finetune.py` | best validation epoch of stage 2, used by evaluation, export and the demo |

Each checkpoint contains the model weights (`model_state_dict`), the optimizer state and the epoch, loss and accuracy values.

Pretrained checkpoints, when available, are attached to the GitHub Releases of this repository. Download `best_validation_full.pth` and place it in this folder.
