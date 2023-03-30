import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

n_epochs = 51
device = 'cuda:0'
model_path = '/home/alex/a.nevarko/models/vagon_cls/efb2_augs_baseline_512'

experiment = {
    'api_key': 'CfFmqDpTCtsdDkLooedZh7bs2',
    'project_name': 'Aurorai-gpt-vagons-cls',
    'workspace': 'alexandernevarko',
    'auto_metric_logging': False,
    'name': 'efb2_augs_baseline_512',
}

train_pipeline = A.Compose([
    A.Resize(512, 512),
    A.MotionBlur(blur_limit=7,
                 allow_shifted=True,
                 p=0.3),
    A.RandomShadow(shadow_roi=(0, 0.0, 1, 1),
                   num_shadows_lower=0,
                   num_shadows_upper=2,
                   shadow_dimension=20,
                   p=0.3),
    A.RandomToneCurve(scale=0.8, p=0.5),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])

val_pipeline = A.Compose([
    A.Resize(512, 512),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])

train_data = {
    'root': '/home/alex/a.nevarko/data/gpt/vagons_cls/data_v2/train',
    'weighted_sampling': False,
    'shuffle': True,
    'batch_size': 8,
    'num_workers': 0,
    'size': 512,
}

val_data = {
    'root': '/home/alex/a.nevarko/data/gpt/vagons_cls/data_v2/test',
    'weighted_sampling': False,
    'shuffle': True,
    'batch_size': 2,
    'num_workers': 0,
    'size': 512,
}

model = {
    'model': 'efficientnet_b2',
    'pretrained': True,
}

optimizer = {
    'type': 'adam',
    'lr': 0.0003,
    'weight_decay': 0.001,
}
lr_policy = {
    'type': 'multistep',
    'steps': [20, 80],
    'gamma': 0.2,
}

criterion = nn.CrossEntropyLoss()
    
