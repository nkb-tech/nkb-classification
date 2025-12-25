import torch
from torch import nn
from nkb_classification.model import get_model
from configs import singletask_hypercube_config as cfg  # или как ты подключаешь конфиг

# 1. Инициализируем такую же модель
device = "cuda:0"
classes = cfg.classes
model = get_model(cfg.model, classes, device=device, compile=False)

# 2. Грузим нужные веса
state = torch.load(
    "/home/prozhkov/projects/nkb-classification/data/runs/train_singletask_hypercube_run_14/weights/best.pth",
    map_location=device,
)
model.load_state_dict(state)
model.eval()

# 3. Находим первый Conv2d в backbone (resnet14t)
first_conv = None
for m in model.emb_model.modules():
    if isinstance(m, nn.Conv2d):
        first_conv = m
        break

w = first_conv.weight.detach()  # shape: (out_channels, in_channels, k, k)

# 4. Важность канала как средняя L1-норма фильтров по всем выходным каналам и ядру
# importance[c] соответствует входному спектральному каналу c
importance = w.abs().mean(dim=(0, 2, 3))  # shape: (in_channels,)

# 5. Топ-каналы
sorted_idx = torch.argsort(importance, descending=True)
topk = 20
for rank, c in enumerate(sorted_idx[:topk]):
    print(f"#{rank+1}: channel={int(c)}, score={float(importance[c]):.6f}")