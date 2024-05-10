import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from os.path import split
import cv2

n_epochs = 51
device = "cuda:1"
model_path = "/home/denis/src/project/models/classification/dog_muzzle_len/mobilenetv3_large_100_baseline"

experiment = {
    "api_key": "F0EvCaEPI2bgMyLl6pLhZ2SoM",
    "project_name": "PetSearch",
    "workspace": "dentikka",
    "auto_metric_logging": False,
    "name": "dog_muzzle_len_" + split(model_path)[-1] + "_augs_strat_video_split_v1",
}

train_pipeline = A.Compose(
    [
        A.LongestMaxSize(max_size=224, always_apply=True),
        A.PadIfNeeded(224, 224, always_apply=True, border_mode=cv2.BORDER_REFLECT_101),
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
        A.PadIfNeeded(224, 224, always_apply=True, border_mode=cv2.BORDER_REFLECT_101),
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
    "ann_file": "/home/denis/nkbtech/data/Dog_expo_Vladimir_02_07_2023_mp4_frames/multiclass_v4/annotation_dog_muzzle_len_strat_video_split_v1.csv",
    "target_name": "dog_muzzle_len",
    "fold": "train",
    "weighted_sampling": False,
    "shuffle": True,
    "batch_size": 64,
    "num_workers": 2,
    "size": 224,
}

val_data = {
    "type": "AnnotatedMultilabelDataset",
    "ann_file": "/home/denis/nkbtech/data/Dog_expo_Vladimir_02_07_2023_mp4_frames/multiclass_v4/annotation_dog_muzzle_len_strat_video_split_v1.csv",
    "target_name": "dog_muzzle_len",
    "fold": "val",
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
    "lr": 0.0001,
    "weight_decay": 0.0001,
}
lr_policy = {
    "type": "multistep",
    "steps": [20, 80],
    "gamma": 0.2,
}

criterion = nn.CrossEntropyLoss()
