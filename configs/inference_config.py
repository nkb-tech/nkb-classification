import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

device = "cuda:0"
enable_mixed_presicion = True
compile = False  # not working correctly yet, so set to false

save_path = "data/runs/val_singletask_run_1"

train_run_path = "data/runs/train_singletask_run_1"

"""
Here you describe inference data.

folder_path: where the images to be classified are stored

and some pytorch dataloader parameters
"""

task = "single"

target_column = "label"
classes = f"{train_run_path}/classes.json"

inference_data = {
    "folder_path": "data/unknown_images",
    "batch_size": 64,
    "num_workers": 8,
}

"""
Here you describe the transformations applied to the processed images
"""

img_size = 128

inference_pipeline = A.Compose(
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
