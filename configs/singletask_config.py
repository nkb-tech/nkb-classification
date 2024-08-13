import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

show_full_current_loss_in_terminal = False
log_gradients = False  # to include model gradients in loss

device = "cuda:0"
enable_mixed_presicion = True
enable_gradient_scaler = True
compile = False  # not working correctly yet, so set to false


experiment_name = "train_singletask_run_1"

experiment = {
    "comet": {  # for logging to comet-ml service (optional)
        "comet_api_cfg_path": "configs/comet_api_cfg.yml",  # should contain 'api_key', 'workspace' and 'project_name' fields
        "auto_metric_logging": False,
        "name": experiment_name,
    },
    "local": { # to save model weights and metrics locally
        "path": f"data/runs/{experiment_name}",
    },
}

show_all_classes_in_confusion_matrix = True  # show all classes in comet confusion matrix, if False then show at most 25

"""
Here you describe train data.

type: AnnotatedSingletaskDataset, AnnotatedMultitaskDataset, GroupsDataset, default - ImageFolder.

For AnnotatedSingletaskDataset and AnnotatedMultitaskDataset the argumatns basically are:
annotations_file: Path to csv labels in for AnnotatedSingletaskDataset and AnnotatedMultitaskDataset.
image_base_dir: Base directory of images. Paths in 'path' column must be relative to this dir. Set None if you have global dirs in your csv file.
target_column / target_names : column names(-s) with class labels.
classes: optional way to provide classnames. If not given, will be infered from annotations
fold : train, val
weighted_sampling : works only for single task

and some pytorch dataloader parameters
"""

task = "single"

annotations_path = "data/annotations.csv"
image_base_dir = "data/images"

target_column = "label"
classes = ["first_class", "second_class"]  # optional

train_data = {
    "type": "AnnotatedSingletaskDataset",
    "annotations_file": annotations_path,
    "image_base_dir": image_base_dir,
    "target_column": target_column,
    "classes": classes,
    "fold": "train",
    "weighted_sampling": True,
    "shuffle": True,
    "batch_size": 64,
    "num_workers": 8,
    "drop_last": True,
}

val_data = {
    "type": "AnnotatedSingletaskDataset",
    "annotations_file": annotations_path,
    "image_base_dir": image_base_dir,
    "target_column": target_column,
    "classes": classes,
    "fold": "val",
    "weighted_sampling": False,
    "shuffle": False,
    "batch_size": 64,
    "num_workers": 8,
    "drop_last": False,
}

"""
For the ImageFolder dataset the argument is:
root: path to the folder where subfolders with images of each class are present
"""

# train_data = {
#     "type": "ImageFolder",
#     "root": "data/by_folders/train",
#     "weighted_sampling": True,
#     "shuffle": True,
#     "batch_size": 64,
#     "num_workers": 8,
#     "drop_last": True,
# }

# val_data = {
#     "type": "ImageFolder",
#     "root": "data/by_folders/val",
#     "weighted_sampling": False,
#     "shuffle": False,
#     "batch_size": 64,
#     "num_workers": 8,
#     "drop_last": False,
# }

"""
Here you describe the transformations applied to the processed images
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
    "model": "resnet14t",
    "pretrained": True,
    # "checkpoint": "previous_run/model_Weights/last.pth",  # optional
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

n_epochs = 5

lr_policy = {
    "type": "cosine",
    "n_epochs": n_epochs,
}

backbone_state_policy = {0: "freeze", 5: "unfreeze", 10: "freeze"}

criterion = {"task": task, "type": "CrossEntropyLoss"}
