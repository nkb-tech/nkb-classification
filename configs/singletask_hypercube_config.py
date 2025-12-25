import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

device = "cuda:0"
enable_mixed_presicion = True  # use mixed float precision for faster model training
enable_gradient_scaler = True  # use float gradient scaling for faster model training
compile = False  # not working correctly yet, so set to false (possible way of training process speedup)


experiment_name = "fine-tune_resnet14t_selected_chans_40_run_1"

experiment = {
    "comet": {  # for logging to comet-ml service (optional, may be set to None)
        "comet_api_cfg_path": "configs/comet_api_cfg.yml",  # should contain 'api_key', 'workspace' and 'project_name' fields
        "auto_metric_logging": False,
        "name": experiment_name,
    },
    "local": { # to save model weights, metrics and class names config locally
        "path": f"data/runs/{experiment_name}",
    },
}

log_gradients = False  # to include model gradients in logs
show_all_classes_in_confusion_matrix = True  # show all classes in comet confusion matrix, if False then show at most 25

task = "single"  # to indicate working in single-task mode

"""
Here you describe train data.

type: AnnotatedSingletaskDataset, AnnotatedMultitaskDataset, GroupsDataset, default - ImageFolder.
"""

"""
AnnotatedSingletaskHypercubeDataset

annotations_file: path to the annotation file, which contains a pandas dataframe
                  with paths to .npy hypercubes and their target values
target_column: name of the column containing class annotations
fold: which fold in the dataset to work with (train, val, test, -1)
transform: which transform to apply to the cube (as HxWxC array) before returning it
image_base_dir: base directory of .npy files. if = None, then absolute paths are expected in 'path' column.
classes: optional config of classes (list, dict, or path to json). if None, then will be inferred from annotations

and some pytorch dataloader parameters
"""

annotations_path = "/home/prozhkov/projects/spectrum/dataset.csv"
# absolute paths to .npy hypercubes are stored in 'path' column of CSV
image_base_dir = None  # keep None because paths in CSV are absolute

target_column = "class"
classes = ["Bad", "Good", "Middle"]  # optional (may be not specified)

selected_channels = [283, 220, 198, 178, 152, 151, 150, 146, 139, 137, 136, 109, 108, 105, 96, 94, 92, 72, 70, 63, 61, 56, 49, 28, 26, 23, 21, 20, 17, 15, 13, 12, 11, 10, 8, 7, 5, 4, 1, 0]
# selected_channels = [113, 11, 51, 3, 112, 24, 183, 211, 117, 34, 209, 134, 206, 128, 130, 10, 54, 118, 208, 0, 40, 16, 2, 189, 111, 77, 60, 44, 12, 13, 163, 9, 52, 37, 22, 92, 62, 5, 19, 218]
train_data = {
    "type": "AnnotatedSingletaskHypercubeDataset",
    "annotations_file": annotations_path,
    "image_base_dir": image_base_dir,  # optional (may be not specified)
    "target_column": target_column,
    "classes": classes,  # optional (may be not specified)
    "selected_channels": selected_channels,  # optional: список индексов каналов для использования
    "fold": "train",
    "weighted_sampling": True,
    "shuffle": True,
    "batch_size": 64,
    "num_workers": 8,
    "drop_last": True,
}

val_data = {
    "type": "AnnotatedSingletaskHypercubeDataset",
    "annotations_file": annotations_path,
    "image_base_dir": image_base_dir,  # optional (may be not specified)
    "target_column": target_column,
    "classes": classes,  # optional (may be not specified, in this case will be infered from train classes)
    "selected_channels": selected_channels,  # должно совпадать с train_data для консистентности
    "fold": "val",
    "weighted_sampling": False,  # it is reasonable not to use weighted sampling on validation
    "shuffle": False,
    "batch_size": 64,
    "num_workers": 8,
    "drop_last": False,
}



"""
Here you describe the transformations applied to the processed hypercubes
with albumentations library.

IMPORTANT: we avoid color-based augmentations (brightness/contrast, hue/saturation)
and ImageNet-specific normalization, because input has many spectral channels,
not just RGB. We keep only simple geometric transforms and tensor conversion.
"""

img_size = 480  # spatial size to which hypercubes will be resized/padded

train_pipeline = A.Compose(
    [
        # Geometric resize/pad applied identically to all spectral channels
        A.LongestMaxSize(img_size),
        A.PadIfNeeded(
            img_size,
            img_size,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        ToTensorV2(),
    ]
)

val_pipeline = A.Compose(
    [
        A.LongestMaxSize(img_size),
        A.PadIfNeeded(
            img_size,
            img_size,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        ToTensorV2(),
    ]
)

"""
Here you describe the model and optimizers
"""

model = {
    "task": task,
    "model": "resnet14t",  # to use models from unicom library, the format should be "model_name", for unicom - "unicom model_name"
    # For hyperspectral input we will usually train from scratch.
    # NOTE: to fully support 300-channel input, model.get_emb_model should be
    # extended to pass `in_chans` into timm.create_model.
    "pretrained": False,
    # Если selected_channels указан, in_chans должен быть равен len(selected_channels)
    # Если selected_channels = None, то используется полное количество каналов (300)
    "in_chans": len(selected_channels) if selected_channels is not None else 300,
    
    # ===== FINE-TUNING: Загрузка предобученной модели =====
    # Раскомментируйте и укажите путь к чекпоинту для fine-tuning:
    "checkpoint": "/home/prozhkov/projects/nkb-classification/data/runs/pretrained_resnet14t_selected_chans_40_run_1/weights/best.pth",
    # "checkpoint": "data/runs/pretrained_resnet14t_selected_chans_40_run_1/weights/last.pth",
    
    # Опции для fine-tuning:
    # "strict_loading": True,  # Если True, все ключи должны совпадать. Если False, игнорируются несовпадающие ключи
    "load_backbone_only": True,  # Если True, загружаются только веса backbone (emb_model), classifier инициализируется заново
    
    "backbone_dropout": 0.1,  
    "classifier_dropout": 0.1,
    "classifier_initialization": "kaiming_normal_",
}

optimizer = {
    "type": "nadam",
    "lr": 1e-5,  # learning rate of the whole model. May be overriden by backbone_lr and classifier_lr parameters (if poth provided then initial value is ignored)
    "backbone_lr": 1e-5,  # optional learning rate value for model backbone to override base lr value (may be not specified)
    "classifier_lr": 1e-4,  # optional learning rate value for model backbone to override base lr value (may be not specified)
    # Для fine-tuning рекомендуется использовать меньшие learning rates:
    # "backbone_lr": 1e-6,  # Меньший LR для backbone при fine-tuning
    # "classifier_lr": 1e-5,  # Больший LR для classifier при fine-tuning
    "weight_decay": 0.2,  # weight decay of the whole model. May also be overriden by backbone_weight_decay and classifier_weight_decay parameters
    "backbone_weight_decay": 0.01,  # optional weight decay value for model backbone to override base weight_decay value (may be not specified)
    "classifier_weight_decay": 0.2,  # optional weight decay value for model backbone to override base weight_decay value (may be not specified)
}

n_epochs = 30

lr_policy = {
    "type": "cosine",
    "n_epochs": n_epochs,
}

# Политика заморозки/разморозки backbone для fine-tuning
# Для fine-tuning рекомендуется начать с замороженного backbone:
backbone_state_policy = {0: "freeze", 10: "unfreeze"}  # Заморозить на первых 5 эпохах, затем разморозить
# Альтернативные варианты:
# backbone_state_policy = {0: "unfreeze"}  # Не замораживать вообще
# backbone_state_policy = {0: "freeze", 10: "unfreeze"}  # Заморозить на первых 10 эпохах

criterion = {"task": task, "type": "CrossEntropyLoss"}
