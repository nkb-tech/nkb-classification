import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

device = "cuda:0"
enable_mixed_presicion = True
compile = False  # not working correctly yet, so set to false

save_path = "data/runs/val_singletask_run_1"

train_run_path = "data/runs/train_singletask_run_1"

"""
Here you describe validation data.

type: AnnotatedSingletaskDataset, AnnotatedMultitaskDataset, GroupsDataset, default - ImageFolder.

For AnnotatedSingletaskDataset and AnnotatedMultitaskDataset the argumatns basically are:
annotations_file: Path to csv labels in for AnnotatedSingletaskDataset and AnnotatedMultitaskDataset.
image_base_dir: Base directory of images. Paths in 'path' column must be relative to this dir. Set None if you have global dirs in your csv file.
target_column / target_names : column names(-s) with class labels.
classes: python list (dict in multitask case) or path to json file where the classes config generated while training is stored
fold : train, val

and some pytorch dataloader parameters
"""

task = "single"

annotations_path = "data/annotations.csv"
image_base_dir = "data/images"

target_column = "label"
classes = f"{train_run_path}/classes.json"

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
Here you describe the model
"""

model = {
    "scripted": True,
    "checkpoint": f"{train_run_path}/weights/scripted_best.pt",
}

criterion = {"task": task, "type": "CrossEntropyLoss"}

