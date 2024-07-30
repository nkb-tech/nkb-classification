from os.path import split

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

show_full_current_loss_in_terminal = False

compile = False  # Is not working correctly yet, so set to False
log_gradients = True
n_epochs = 20 + 1
device = "cuda:1"
enable_mixed_presicion = True
enable_gradient_scaler = True

task = "single"

target_column = "label"

model_path = f"/home/a.nevarko/projects/parking/models/occupancy/128_sputnik_6k_+spaces1000_resnet14t_focal_gamma_1_w"

experiment = {
    "comet_api_cfg_path": "configs/comet_api_cfg.yml",
    "auto_metric_logging": False,
    "name": split(model_path)[-1],
}

# experiment = None

img_size = 128

train_pipeline = A.Compose(
    [
        # A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.LongestMaxSize(img_size, always_apply=True),
        A.PadIfNeeded(
            img_size,
            img_size,
            always_apply=True,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        # A.MotionBlur(blur_limit=3, allow_shifted=True, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2),
            contrast_limit=(0.1, -0.5),
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=0,
            sat_shift_limit=10,
            val_shift_limit=50,
            p=0.5,
        ),
        # A.RandomShadow(p=0.5),
        # A.RandomFog(
        #     fog_coef_lower=0.3,
        #     fog_coef_upper=0.5,
        #     alpha_coef=0.28,
        #     p=0.5,
        # ),
        # A.RandomRain(p=0.5),
        A.CoarseDropout(
            max_holes=4,
            min_holes=1,
            max_height=0.2,
            min_height=0.05,
            max_width=0.2,
            min_width=0.05,
            fill_value=[0, 0.5, 1],
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
        # A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
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

"""
Here you describe train data.
type: AnnotatedSingletaskDataset, AnnotatedMultitaskDataset, GroupsDataset, default - ImageFolder.
annotations_file: Path to csv labels in for AnnotatedSingletaskDataset and AnnotatedMultitaskDataset.
image_base_dir: Base directory of images. Paths in 'path' column must be relative to this dir. Set None if you have global dirs in your csv file.
target_column / target_names : column names(-s) with class labels.
fold : train, val
weighted_sampling : works only for single task
"""

train_data = {
    "type": "AnnotatedSingletaskDataset",
    "annotations_file": "/home/a.nevarko/projects/datasets/parking/sputnik_parking/sputnik6k_+spaces100_tr_sp_1k_val.csv",
    # "image_base_dir": '/home/slava/hdd/hdd4/Datasets/petsearch/Dog_expo_Vladimir_02_07_2023_mp4_frames/multiclass_v4/images',
    "target_column": target_column,
    # "root": "/home/a.nevarko/projects/datasets/parking/kaggle/pklot_original/spaces1000/train",
    "fold": "train",
    "weighted_sampling": True,
    "shuffle": True,
    "batch_size": 32,
    "num_workers": 10,
    "size": img_size,
}

# train_data = {
#     "type": "AnnotatedSingletaskDataset",
#     "annotations_file": "/home/slava/hdd/hdd4/Datasets/petsearch/Dog_expo_Vladimir_02_07_2023_mp4_frames/demo_dataset.csv",
#     "image_base_dir": '/home/slava/hdd/hdd4/Datasets/petsearch/Dog_expo_Vladimir_02_07_2023_mp4_frames/multiclass_v4/images',
#     "target_column": target_column,
#     "fold": "train",
#     "weighted_sampling": True,
#     "shuffle": True,
#     "batch_size": 32,
#     "num_workers": 10,
#     "size": img_size,
# }

val_data = {
    "type": "AnnotatedSingletaskDataset",
    "annotations_file": "/home/a.nevarko/projects/datasets/parking/sputnik_parking/sputnik6k_+spaces100_tr_sp_1k_val.csv",
    # "image_base_dir": '/home/slava/hdd/hdd4/Datasets/petsearch/Dog_expo_Vladimir_02_07_2023_mp4_frames/multiclass_v4/images',
    # "root": "/home/a.nevarko/projects/datasets/parking/kaggle/pklot_original/spaces1000/val",
    "target_column": target_column,
    "fold": "val",
    "weighted_sampling": False,
    "shuffle": True,
    "batch_size": 32,
    "num_workers": 10,
    "size": img_size,
}

# val_data = {
#     "type": "AnnotatedSingletaskDataset",
#     "annotations_file": "/home/slava/hdd/hdd4/Datasets/petsearch/Dog_expo_Vladimir_02_07_2023_mp4_frames/demo_dataset.csv",
#     "image_base_dir": '/home/slava/hdd/hdd4/Datasets/petsearch/Dog_expo_Vladimir_02_07_2023_mp4_frames/multiclass_v4/images',
#     "target_column": target_column,
#     "fold": "val",
#     "weighted_sampling": True,
#     "shuffle": True,
#     "batch_size": 32,
#     "num_workers": 8,
#     "size": img_size,
# }

model = {
    "task": task,
    "model": "resnet14t",
    # "checkpoint": "/home/a.nevarko/projects/parking/models/occupancy/96_sputnik_3k_tr+spaces1000_resnet14_focal_gamma_1/last.pth",
    "pretrained": True,
    "backbone_dropout": 0.1,
    "classifier_dropout": 0.1,
    "classifier_initialization": "kaiming_normal_",
}

optimizer = {
    "type": "nadam",
    "lr": 1e-5,
    "weight_decay": 0.2,
    "backbone_lr": 1e-5,
    "classifier_lr": 1e-4,
}

lr_policy = {
    "type": "multistep",
    "steps": [
        20,
    ],
    "gamma": 0.1,
}

backbone_state_policy = {0: "freeze", 5: "unfreeze", 10: "freeze"}

criterion = {"task": task, "type": "FocalLoss", "gamma": 1}
