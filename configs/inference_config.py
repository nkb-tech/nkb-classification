import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

device = "cuda:0"
enable_mixed_presicion = True
compile = False  # not working correctly yet, so set to false

save_path = "data/runs/val_singletask_hypercube_selected_chans_run_5"

train_run_path = "data/runs/train_singletask_hypercube_selected_chans_run_5"

"""
Here you describe inference data.

Options for type:
1. "hypercube_csv" - для работы с CSV файлом (как при обучении)
   - annotations_file: путь к CSV файлу с колонкой 'path' (пути к .npy файлам)
   - selected_channels: список каналов (должен совпадать с обучением)
   - image_base_dir: опционально, базовый путь к файлам

2. "hypercube" - для работы с папкой .npy файлов
   - folder_path: путь к папке с .npy файлами
   - selected_channels: список каналов

3. "image" (default) - для обычных изображений
   - folder_path: путь к папке с изображениями

and some pytorch dataloader parameters
"""

task = "single"

target_column = "class"  # название колонки, куда будут записаны предсказания
classes = f"{train_run_path}/classes.json"

# Указываем те же каналы, что использовались при обучении
selected_channels = [283, 220, 198, 178, 152, 151, 150, 146, 139, 137, 136, 109, 108, 105, 96, 94, 92, 72, 70, 63, 61, 56, 49, 28, 26, 23, 21, 20, 17, 15, 13, 12, 11, 10, 8, 7, 5, 4, 1, 0]

# Вариант 1: Использование CSV файла (рекомендуется)
inference_data = {
    "type": "hypercube_csv",  # для работы с CSV файлом
    "annotations_file": "/home/prozhkov/projects/spectrum/inference_dataset.csv",  # путь к CSV с колонкой 'path'
    "image_base_dir": None,  # если None, то пути в CSV должны быть абсолютными
    "selected_channels": selected_channels,  # должен совпадать с обучением!
    "batch_size": 4,
    "num_workers": 4,
}

# Вариант 2: Использование папки с .npy файлами (раскомментируйте, если нужно)
# inference_data = {
#     "type": "hypercube",  # для работы с папкой
#     "folder_path": "data/unknown_hypercubes",  # путь к папке с .npy файлами
#     "selected_channels": selected_channels,
#     "batch_size": 4,
#     "num_workers": 4,
# }

"""
Here you describe the transformations applied to the processed hypercubes.

IMPORTANT: для гиперкубов не используем ImageNet нормализацию,
так как у нас много спектральных каналов, а не RGB.
Используем только геометрические трансформации и конвертацию в тензор.
"""

img_size = 480  # должно совпадать с размером при обучении

inference_pipeline = A.Compose(
    [
        # Для гиперкубов: только геометрические трансформации
        A.LongestMaxSize(img_size, always_apply=True),
        A.PadIfNeeded(img_size, img_size, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=0),
        ToTensorV2(),
    ]
)

# Если используется обычное изображение (не гиперкуб), раскомментируйте:
# inference_pipeline = A.Compose(
#     [
#         A.LongestMaxSize(img_size, always_apply=True),
#         A.PadIfNeeded(img_size, img_size, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=0),
#         A.Normalize(
#             mean=(0.485, 0.456, 0.406),
#             std=(0.229, 0.224, 0.225),
#         ),
#         ToTensorV2(),
#     ]
# )

"""
Here you describe the model

Если используется scripted модель, in_chans уже зашит в веса.
Если используется обычная модель (scripted=False), нужно указать in_chans.
"""

model = {
    "scripted": True,  # если True, использует torch.jit.load
    "checkpoint": f"{train_run_path}/weights/scripted_best.pt",
    # Если scripted=False, раскомментируйте и укажите параметры модели:
    # "task": task,
    # "model": "resnet14t",
    # "pretrained": False,
    # "in_chans": len(selected_channels) if selected_channels is not None else 300,
    # "backbone_dropout": 0.1,
    # "classifier_dropout": 0.1,
    # "classifier_initialization": "kaiming_normal_",
}
