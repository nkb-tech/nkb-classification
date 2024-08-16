import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

device = "cuda:1"

save_path = "exp"

target_names = [
    "dog_size",
    "dog_fur",
    "dog_color",
    "dog_ear_type",
    "dog_muzzle_len",
    "dog_leg_len",
]

img_size = 224

inference_pipeline = A.Compose(
    [
        A.LongestMaxSize(img_size, always_apply=True),
        A.PadIfNeeded(
            img_size,
            img_size,
            always_apply=True,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ]
)

inference_data = {
    "root": "/home/slava/hdd/hdd4/Datasets/petsearch/Dog_expo_Vladimir_02_07_2023_mp4_frames/multiclass_v4/images",
    "train_annotations_file": "/home/slava/nkb-classification/jupyters_exps/annotation_high_res_video_split_v2_slava.csv",
    "target_names": target_names,
    "batch_size": 64,
    "num_workers": 4,
    "size": img_size,
}

model = {"checkpoint": "/home/denis/src/project/models/classification/multitask/mobilenetv3_large_100_dummy/last.pth"}
