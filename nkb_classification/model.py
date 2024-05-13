import timm
import torch
import unicom
from torch import nn


class MultilabelModel(nn.Module):
    """
    A class to make a model consisting of an embedding model (backbone)
    and several classifiers (head)
    Currently maintained architectures are:
        MobileNet, EfficientNet, ConvNext, ResNet, ViT,
        Unicom (ViT pretrained for metric learning)
    """

    def __init__(self, cfg_model: dict, classes: dict):
        super().__init__()
        self.emb_model, self.emb_size = self.get_emb_model(cfg_model)
        self.set_dropout(self.emb_model, cfg_model["backbone_dropout"])

        self.classifiers = nn.ModuleDict()

        for target_name in classes:
            self.classifiers[target_name] = nn.Sequential(
                nn.Dropout(cfg_model["classifier_dropout"]),
                nn.Linear(self.emb_size, len(classes[target_name])),
            )

        # TODO load state dict if pretrained

        self.initialize_classifiers(strategy=cfg_model["classifier_initialization"])

    def forward(self, x: torch.tensor):
        emb = self.emb_model(x)
        return {
            class_name: classifier(emb)
            for class_name, classifier in self.classifiers.items()
        }

    def initialize_classifiers(self, strategy="kaiming_normal_"):
        for classifier in self.classifiers.values():
            for param in classifier.parameters():
                if param.ndim >= 2:
                    if strategy == "kaiming_normal_":
                        nn.init.kaiming_normal_(param, nonlinearity="relu")
                    elif strategy == "kaiming_uniform_":
                        nn.init.kaiming_uniform_(param, nonlinearity="relu")
                    elif strategy == "xavier_normal_":
                        nn.init.xavier_normal_(param, nonlinearity="relu")
                    elif strategy == "xavier_uniform_":
                        nn.init.xavier_uniform_(param, nonlinearity="relu")
                else:
                    nn.init.zeros_(param)

    def set_backbone_state(self, state: str = "freeze"):
        for param in self.emb_model.parameters():
            if state == "freeze":
                param.requires_grad = False
            elif state == "unfreeze":
                param.requires_grad = True

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
        if name.lower().startswith("unicom"):
            emb_model, _ = unicom.load(name.split()[1])
            emb_size = emb_model.feature[-2].out_features

        else:
            emb_model = timm.create_model(name, pretrained=cfg_model["pretrained"], num_classes=0)
            emb_size = emb_model.num_features

        return emb_model, emb_size


def get_model(cfg_model, classes, device="cpu", compile: bool = True):
    model = MultilabelModel(cfg_model, classes)

    model.to(device)
    if compile:
        model = torch.compile(model, dynamic=True)

    return model
