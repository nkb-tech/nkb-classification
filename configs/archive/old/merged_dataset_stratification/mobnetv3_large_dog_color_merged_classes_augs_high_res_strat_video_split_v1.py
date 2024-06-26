from os.path import split

import albumentations as A
import cv2
import torch.nn as nn
from albumentations.pytorch import ToTensorV2

n_epochs = 51
device = "cuda:1"
model_path = "/home/denis/src/project/models/classification/dog_color/mobilenetv3_large_100_merged_classes_augs_high_res_strat_video_split_v1"

experiment = {
    "api_key": "F0EvCaEPI2bgMyLl6pLhZ2SoM",
    "project_name": "PetSearch",
    "workspace": "dentikka",
    "auto_metric_logging": False,
    "name": "dog_color_" + split(model_path)[-1],
}

train_pipeline = A.Compose(
    [
        A.LongestMaxSize(max_size=224, always_apply=True),
        A.PadIfNeeded(
            224,
            224,
            always_apply=True,
            border_mode=cv2.BORDER_REFLECT_101,
        ),
        A.Resize(224, 224),
        A.MotionBlur(blur_limit=3, allow_shifted=True, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2),
            contrast_limit=(0.1, -0.5),
            p=0.5,
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ]
)

val_pipeline = A.Compose(
    [
        A.LongestMaxSize(max_size=224, always_apply=True),
        A.PadIfNeeded(
            224,
            224,
            always_apply=True,
            border_mode=cv2.BORDER_REFLECT_101,
        ),
        A.Resize(224, 224),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ]
)

train_data = {
    "type": "AnnotatedMultilabelDataset",
    "ann_file": "/home/denis/nkbtech/data/Dog_expo_Vladimir_02_07_2023_mp4_frames/multiclass_v4/annotation_dog_color_merged_classes_high_res_strat_video_split_v1.csv",
    "target_name": "dog_color",
    "fold": "train",
    "classes": [
        "belyj",
        "bezhevyj",
        "cherno_belo_seraya",
        "chyorno_belyj",
        "chyornyj",
        "korichnevyj",
        "netipichnyj",
        "ryzhij",
        "seryj",
        "smeshannij",
    ],
    "weighted_sampling": True,
    "shuffle": True,
    "batch_size": 64,
    "num_workers": 2,
    "size": 224,
}

val_data = {
    "type": "AnnotatedMultilabelDataset",
    "ann_file": "/home/denis/nkbtech/data/Dog_expo_Vladimir_02_07_2023_mp4_frames/multiclass_v4/annotation_dog_color_merged_classes_high_res_strat_video_split_v1.csv",
    "target_name": "dog_color",
    "fold": "val",
    "classes": [
        "belyj",
        "bezhevyj",
        "cherno_belo_seraya",
        "chyorno_belyj",
        "chyornyj",
        "korichnevyj",
        "netipichnyj",
        "ryzhij",
        "seryj",
        "smeshannij",
    ],
    "weighted_sampling": False,
    "shuffle": True,
    "batch_size": 64,
    "num_workers": 2,
    "size": 224,
}

model = {
    "model": "mobilenetv3_large_100",
    "pretrained": True,
}

optimizer = {
    "type": "adam",
    "lr": 0.00001,
    "weight_decay": 0.001,
}
lr_policy = {
    "type": "multistep",
    "steps": [10, 30],
    "gamma": 0.1,
}

criterion = nn.CrossEntropyLoss()
