import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

device = "cuda:1"

compile = False  # Is not working correctly yet, so set to False
save_path = "/home/slava/nkb-classification/jupyters_exps/runs/demo_run1"

target_names = [
    "dog_size",
    "dog_fur",
    "dog_color",
    "dog_ear_type",
    "dog_muzzle_len",
    "dog_leg_len",
]

img_size = 224

val_pipeline = A.Compose(
    [
        A.LongestMaxSize(img_size, always_apply=True),
        A.PadIfNeeded(img_size, img_size, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ]
)

val_data = {
    "type": "AnnotatedMultitargetDataset",
    "ann_file": "/home/slava/nkb-classification/jupyters_exps/annotation_high_res_video_split_v2_slava.csv",
    "target_names": target_names,
    "fold": "val",
    "weighted_sampling": False,
    "shuffle": True,
    "batch_size": 128,
    "num_workers": 10,
    "size": img_size,
}


model = {
    "model": "Unicom ViT-B/16",
    "pretrained": True,
    "backbone_dropout": 0.1,
    "classifier_dropout": 0.1,
    "classifier_initialization": "kaiming_normal_",
    "checkpoint": "/home/slava/nkb-classification/runs/demo_train2/best.pth",
}

criterion = {"type": "CrossEntropyLoss"}
enable_mixed_presicion = True
