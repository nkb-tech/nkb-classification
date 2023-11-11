import pickle as pkl
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2

n_epochs = 51
device = "cuda:0"
model_path = "/home/alexander/src/project/models/classification/nkb-breed/mobnetv3l_augs_baseline_224/last.pth"


inference_pipeline = A.Compose(
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
classes_dict = Path(
    "/home/alexander/src/project/nkb-classification/configs/dog_breed/breed_dicts/tsinghua.pkl"
)
with classes_dict.open("rb") as f:
    classes_dict = pkl.load(f)
class_to_idx = {k: i for i, k in enumerate(classes_dict.keys())}
idx_to_class = {idx: lb for lb, idx in class_to_idx.items()}

inference_data = {
    "type": "GroupsDataset",
    "root": "/home/alexander/src/project/data/cam_dog_crops_dataset/images",
    "batch_size": 64,
    "num_workers": 2,
    "size": 224,
    "classes": idx_to_class,
}

model = {
    "model": "mobilenetv3_large_100",
    "pretrained": False,
    "checkpoint": "/home/alexander/src/project/models/classification/nkb-breed/mobnetv3l_augs_baseline_224/last.pth",
}

save_path = (
    "/home/alexander/src/project/data/cam_dog_crops_dataset/predictions"
)
