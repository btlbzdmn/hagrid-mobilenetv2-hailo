# data

The dataset is not stored in this repository. Place a 24-class subset of [HaGRID](https://github.com/hukenovs/hagrid) here in the `torchvision` `ImageFolder` layout, with one sub-folder per gesture:

```
data/
├── train/
│   ├── call/*.jpg
│   ├── dislike/*.jpg
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

The class folders must be named exactly like the classes listed in the main README, because the scripts rely on the alphabetical folder order for the class indices.

Images are resized to 224x224 by the scripts, so no resizing is needed beforehand.

TODO: describe how the images were prepared (which HaGRID version, how the train / val / test split was made, how many images per class).
