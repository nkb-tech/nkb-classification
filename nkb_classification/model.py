import timm
import torch
from torch import nn


class MultilabelModel(nn.Module):
    """
    A class to make a model consisting of an embedding model (backbone)
    and several classifiers (head)
    Currently maintained architectures are:
        MobileNet, EfficientNet, ConvNext, ResNet, ViT
    """

    def __init__(self, cfg_model: dict, classes: dict):
        super().__init__()
        self.emb_model, emb_size = self.get_emb_model(cfg_model)
        self.set_dropout(self.emb_model, cfg_model["backbone_dropout"])

        self.classifiers = nn.ModuleDict()
        for target_name in classes:
            self.classifiers[target_name] = nn.Sequential(
                nn.Dropout(cfg_model["classifier_dropout"]),
                nn.Linear(emb_size, len(classes[target_name])),
            )

    def forward(self, x):
        emb = self.emb_model(x)
        return {
            class_name: classifier(emb)
            for class_name, classifier in self.classifiers.items()
        }

    @staticmethod
    def set_dropout(model: nn.Module, drop_rate: float = 0.2) -> None:
        """Set new `drop_rate` for model"""
        for child in model.children():
            if isinstance(child, torch.nn.Dropout):
                child.p = drop_rate
            MultilabelModel.set_dropout(child, drop_rate=drop_rate)

    @staticmethod
    def get_emb_model(cfg_model: dict):
        name = cfg_model["model"]
        initial_model = timm.create_model(
            name, pretrained=cfg_model["pretrained"]
        )
        if name.startswith(("efficientnet", "mobilenet")):
            emb_size = initial_model.classifier.in_features
            emb_model = nn.Sequential(
                *[*initial_model.children()][:-1], nn.Flatten()
            )
        elif name.startswith("convnext"):
            emb_size = initial_model.head.in_features
            new_head = nn.Sequential(
                *[*[*initial_model.children()][-1].children()][:-2]
            )
            emb_model = nn.Sequential(
                *[*initial_model.children()][:-1], new_head
            )
        elif name.startswith("resnet"):
            emb_size = initial_model.fc.in_features
            emb_model = nn.Sequential(
                *[*initial_model.children()][:-1], nn.Flatten()
            )
        elif name.startswith("vit"):
            emb_size = (
                initial_model.patch_embed.num_patches
                * initial_model.head.in_features
            )
            emb_model = nn.Sequential(
                *[*initial_model.children()][:-2], nn.Flatten()
            )

        return emb_model, emb_size


def get_model(cfg_model, classes, device="cpu", compile: bool = True):
    model = MultilabelModel(cfg_model, classes)

    model.to(device)
    if compile:
        model = torch.compile(model, dynamic=True)

    return model
