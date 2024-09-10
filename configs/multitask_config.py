import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

device = "cuda:0"
enable_mixed_presicion = True
enable_gradient_scaler = True
compile = False  # not working correctly yet, so set to false


experiment_name = "train_multitask_run_1"

experiment = {
    "comet": {  # for logging to comet-ml service (optional, may be set to None)
        "comet_api_cfg_path": "configs/comet_api_cfg.yml",  # should contain 'api_key', 'workspace' and 'project_name' fields
        "auto_metric_logging": False,
        "name": experiment_name,
    },
    "local": {# to save model weights, metrics and class names config locally
        "path": f"data/runs/{experiment_name}",
    },
}

show_full_current_loss_in_terminal = False  # to show loss with respect to every task in progress bar
log_gradients = False  # to include model gradients in logs
show_all_classes_in_confusion_matrix = True  # show all classes in comet confusion matrix, if False then show at most 25

task = "multi"  # to indicate working in multi-task mode

"""
Here you describe train data.

type: AnnotatedSingletaskDataset, ImageFolder, AnnotatedYOLODataset, AnnotatedMultitaskDataset, GroupsDataset, default - ImageFolder.
"""

"""
AnnotatedMultitaskDataset

annotations_file: Path to csv labels in for AnnotatedSingletaskDataset and AnnotatedMultitaskDataset.
image_base_dir: Base directory of images. Paths in 'path' column must be relative to this dir. Set None if you have global dirs in your csv file.
target_names : column names with class labels.
classes: optional way to provide classnames. If not given, will be infered from annotations
fold : train, val

and some pytorch dataloader parameters
"""

annotations_path = "data/annotations.csv"
image_base_dir = "data/images"  # optional (may be not specified)

target_names = ["dog_size", "dog_color"]
classes = {  # optional (may be not specified)
    "dog_size": ["bolshoj", "malenkij"],
    "dog_color": ["chernyj", "belyj"]
}

train_data = {
    "type": "AnnotatedMultitaskDataset",
    "annotations_file": annotations_path,
    "image_base_dir": image_base_dir,   # optional (may be not specified)
    "target_names": target_names,
    "classes": classes,   # optional (may be not specified)
    "fold": "train",
    "shuffle": True,
    "batch_size": 64,
    "num_workers": 8,
    "drop_last": True,
}

val_data = {
    "type": "AnnotatedMultitaskDataset",
    "annotations_file": annotations_path,
    "image_base_dir": image_base_dir,   # optional (may be not specified)
    "target_names": target_names,
    "classes": classes,  # optional (may be not specified, in this case will be infered from train classes)
    "fold": "val",
    "shuffle": False,
    "batch_size": 64,
    "num_workers": 8,
    "drop_last": False,
}

"""
Here you describe the transformations applied to the processed images with albumentations library
"""

img_size = 128

train_pipeline = A.Compose(
    [
        A.LongestMaxSize(img_size, always_apply=True),
        A.PadIfNeeded(
            img_size,
            img_size,
            always_apply=True,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
        ),
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
        A.LongestMaxSize(img_size, always_apply=True),
        A.PadIfNeeded(img_size, img_size, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ]
)

"""
Here you describe the model and optimizers
"""

model = {
    "task": task,
    "model": "unicom ViT-B/32",  # to use models from unicom library, the format should be "model_name", for unicom - "unicom model_name"
    "pretrained": True,  # to load pretrained weights from timm or unicom library
    # "checkpoint": "previous_run/model_Weights/last.pth",  # optional (may be not specified)
    "backbone_dropout": 0.1,
    "classifier_dropout": 0.1,
    "classifier_initialization": "kaiming_normal_",
}

optimizer = {
    "type": "nadam",
    "lr": 1e-5,  # learning rate of the whole model. May be overriden by backbone_lr and classifier_lr parameters (if poth provided then initial value is ignored)
    "backbone_lr": 1e-5,  # optional learning rate value for model backbone to override base lr value (may be not specified)
    "classifier_lr": 1e-4,  # optional learning rate value for model backbone to override base lr value (may be not specified)
    "weight_decay": 0.2,  # weight decay of the whole model. May also be overriden by backbone_weight_decay and classifier_weight_decay parameters
    "backbone_weight_decay": 0.01,  # optional weight decay value for model backbone to override base weight_decay value (may be not specified)
    "classifier_weight_decay": 0.2,  # optional weight decay value for model backbone to override base weight_decay value (may be not specified)
}

n_epochs = 5

lr_policy = {
    "type": "multistep",
    "steps": [20,],
    "gamma": 0.1,
}

backbone_state_policy = {0: "freeze", 5: "unfreeze", 10: "freeze"}

criterion = {"task": task, "type": "FocalLoss", "gamma": 1}

