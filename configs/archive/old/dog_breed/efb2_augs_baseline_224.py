import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from os.path import split

n_epochs = 51
device = "cuda:0"
model_path = (
    "/home/alexander/src/project/models/classification/nkb-breed/efb2_augs_baseline_224"
)

experiment = {
    "api_key": "CfFmqDpTCtsdDkLooedZh7bs2",
    "project_name": "PetSearch-breed",
    "workspace": "alexandernevarko",
    "auto_metric_logging": False,
    "name": split(model_path)[-1],
}

train_pipeline = A.Compose(
    [
        A.LongestMaxSize(max_size=224, always_apply=True),
        A.PadIfNeeded(224, 224, always_apply=True),
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
        A.PadIfNeeded(224, 224, always_apply=True),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ]
)

train_data = {
    "type": "GroupsDataset",
    "root": "/home/alexander/src/project/data/tsinghua-dog",
    "ann_file": "OriginalTrainAndValSplit/train.pkl",
    "group_dict": "/home/alexander/src/project/nkb-classification/configs/dog_breed/breed_dicts/tsinghua.pkl",
    "weighted_sampling": False,
    "shuffle": True,
    "batch_size": 64,
    "num_workers": 2,
    "size": 224,
}

val_data = {
    "type": "GroupsDataset",
    "root": "/home/alexander/src/project/data/tsinghua-dog",
    "ann_file": "OriginalTrainAndValSplit/test.pkl",
    "group_dict": "/home/alexander/src/project/nkb-classification/configs/dog_breed/breed_dicts/tsinghua.pkl",
    "weighted_sampling": False,
    "shuffle": True,
    "batch_size": 32,
    "num_workers": 2,
    "size": 224,
}

model = {
    "model": "efficientnet_b2",
    "pretrained": True,
}

optimizer = {
    "type": "adam",
    "lr": 0.0003,
    "weight_decay": 0.001,
}
lr_policy = {
    "type": "multistep",
    "steps": [20, 80],
    "gamma": 0.2,
}

criterion = nn.CrossEntropyLoss()
