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

model_path = f"/root/nkb-classification/exp"

comet_api_cfg_path = "config/comet_api_cfg_path.yaml"  # should contain "api_key", "workspace" and "project_name" fields 

experiment = {
    comet_api_cfg_path: comet_api_cfg_path,
    "auto_metric_logging": False,
    "name": split(model_path)[-1],
}

img_size = 128

mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)

train_pipeline = A.Compose(
    [
        # A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.LongestMaxSize(img_size, always_apply=True),
        A.PadIfNeeded(
            img_size,
            img_size,
            # border_mode=cv2.BORDER_CONSTANT,
            # value=128,
            # mask_value = 64,
            always_apply=True,
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
            mean=mean,
            std=std,
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
            # border_mode=cv2.BORDER_CONSTANT,
        ),
        A.Normalize(
            mean=mean,
            std=std,
        ),
        ToTensorV2(),
    ]
)

"""
Here you describe train data.
type: AnnotatedSingletaskDataset, AnnotatedMultitaskDataset, GroupsDataset, defaut - ImageFolder.
annotations_file: Path to csv labels in for AnnotatedSingletaskDataset and AnnotatedMultitaskDataset.
image_base_dir: Base directory of images. Paths in 'path' column must be relative to this dir. Set None if you have global dirs in your csv file.
target_column / target_names : column names(-s) with class labels.
fold : train, val
weighted_sampling : works only for single task
"""

train_data = {
    "type": "AnnotatedYOLODataset",
    "annotations_file": "/root/Projects/nkb-classification/brain-tumor.yaml",  # dataset yaml config
    "image_base_dir": '/root/Projects/nkb-classification/nkb_classification/datasets',  # prefix for dataset config "path" field
    "fold": "train",
    "weighted_sampling": True,
    "shuffle": True,
    "batch_size": 32,
    "num_workers": 10,
    "min_box_size": 5,  # minimum bounding boox side size to include the box in the dataset
    "generate_backgrounds": True,  # whether to generate additional bounding
                                   # boxes corresponding to additional background class
    "background_generating_prob": None,  # probability of generation a random background box from an image
                                         # if None, then equals 1 / nun_classes (ignored if generate_backgrounds == False)
    "background_crop_sizes": (0.1, 0.3),  # background box size range (ignored if generate_backgrounds == False)
}


val_data = {
    "type": "AnnotatedYOLODataset",
    "annotations_file": "/root/Projects/nkb-classification/brain-tumor.yaml",
    "fold": "val",
    "weighted_sampling": False,
    "shuffle": True,
    "batch_size": 32,
    "num_workers": 10,
    "min_box_size": 5,
    "generate_backgrounds": True,
    "background_generating_prob": None,
    "background_crop_sizes": (0.1, 0.3),
}


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

backbone_state_policy = {
    0: "freeze",
    5: "unfreeze",
    10: "freeze"
}

criterion = {
    "task": task,
    "type": "FocalLoss",
    "gamma": 1
}
