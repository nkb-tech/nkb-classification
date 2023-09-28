import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from os.path import split
import cv2

compile = False
log_gradients = True
n_epochs = 21
device = 'cuda:1'
enable_mixed_presicion = False
enable_gradient_scaler = False
model_path = '/home/denis/src/project/models/classification/dog_color/mobilenetv3_large_100_merged_classes_augs_high_res_strat_video_split_v6'

experiment = {
    'api_key': 'F0EvCaEPI2bgMyLl6pLhZ2SoM',
    'project_name': 'PetSearch',
    'workspace': 'dentikka',
    'auto_metric_logging': False,
    'name': 'dog_color_'+split(model_path)[-1],
}

train_pipeline = A.Compose([
    A.Resize(224, 224), 
    A.MotionBlur(blur_limit=3,
                 allow_shifted=True,
                 p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=(-0.2, 0.2),
        contrast_limit=(0.1, -0.5),
        p=0.5,
    ),
    A.HueSaturationValue(hue_shift_limit=0, 
                        sat_shift_limit=10, 
                        val_shift_limit=50,
                        p=0.5),
    A.RandomShadow(always_apply=True),
    A.RandomFog(fog_coef_lower=0.3, 
                fog_coef_upper=0.5, 
                alpha_coef=0.28,
                p=0.5),
    A.RandomRain(p=0.5),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])

val_pipeline = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])

train_data = {
    'type': 'AnnotatedMultilabelDataset',
    'ann_file': '/home/denis/nkbtech/data/Dog_expo_Vladimir_02_07_2023_mp4_frames/multiclass_v4/annotation_dog_color_merged_classes_high_res_strat_video_split_v5.csv',
    'target_name': 'dog_color',
    'fold': 'train',
    'weighted_sampling': True,
    'shuffle': True,
    'batch_size': 64,
    'num_workers': 4,
    'size': 224,
}

val_data = {
    'type': 'AnnotatedMultilabelDataset',
    'ann_file': '/home/denis/nkbtech/data/Dog_expo_Vladimir_02_07_2023_mp4_frames/multiclass_v4/annotation_dog_color_merged_classes_high_res_strat_video_split_v5.csv',
    'target_name': 'dog_color',
    'fold': 'val',
    'weighted_sampling': False,
    'shuffle': True,
    'batch_size': 64,
    'num_workers': 4,
    'size': 224,
}

model = {
    'model': 'mobilenetv3_large_100',
    'pretrained': True,
    'layers_to_unfreeze': 7
}

optimizer = {
    'type': 'radam',
    'lr': 0.00001,
    'weight_decay': 0.001,
}
lr_policy = {
    'type': 'multistep',
    'steps': [10, ],
    'gamma': 0.1,
}

criterion = {
    'type': 'CrossEntropyLoss'
}
