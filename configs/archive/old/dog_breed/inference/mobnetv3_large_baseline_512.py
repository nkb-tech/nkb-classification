import albumentations as A
from albumentations.pytorch import ToTensorV2

device = "cuda:0"
model_path = "/home/alex/a.nevarko/models/vagon_cls/efb2_augs_baseline_512"

inference_pipeline = A.Compose(
    [
        A.Resize(512, 512),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ]
)

inference_data = {
    "root": "/home/alex/a.nevarko/data/gpt/vagons_cls/data_v2/test/par_cisterna",
    "batch_size": 64,
    "num_workers": 2,
    "size": 512,
    "classes": {
        0: "cisterna",
        1: "fon",
        2: "locomotiv",
        3: "mezhvagon",
        4: "par_cisterna",
        5: "vagon",
    },
}

model = {
    "model": "mobilenetv3_large_100",
    "pretrained": False,
    "checkpoint": "/home/alex/a.nevarko/models/vagon_cls/mobilenetv3_large_100_v2_balanced/last.pth",
}

save_path = "/home/alex/a.nevarko/data/gpt/vagons_cls/test_data/par_cisterna"
