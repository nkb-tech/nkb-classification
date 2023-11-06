import torch
from torch import nn
import timm


def get_activation(activ_name: str="relu"):
    """"""
    act_dict = {
        "relu": nn.ReLU(inplace=True),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "identity": nn.Identity()}
    if activ_name in act_dict:
        return act_dict[activ_name]
    else:
        raise NotImplementedError
        

class Conv2dBNActiv(nn.Module):
    """Conv1d -> (BN ->) -> Activation"""
    
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: int, stride: int=1, padding: int=0,
        bias: bool=False, use_bn: bool=True, activ: str="relu"
    ):
        """Initialize"""
        super(Conv2dBNActiv, self).__init__()
        layers = []
        layers.append(nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride, padding, bias=bias))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
            
        layers.append(get_activation(activ))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward"""
        return self.layers(x)
        

class SSEBlock(nn.Module):
    """channel `S`queeze and `s`patial `E`xcitation Block"""

    def __init__(self, in_channels: int):
        """Initialize"""
        super(SSEBlock, self).__init__()
        self.channel_squeeze = nn.Conv2d(
            in_channels=in_channels, out_channels=1,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward"""
        # # x: (bs, ch, h, w) => h: (bs, 1, h, w)
        h = self.sigmoid(self.channel_squeeze(x))
        # # x, h => return: (bs, ch, h, w)
        return x * h
    
    
class SpatialAttentionBlock(nn.Module):
    """Spatial Attention for (C, emb_size) feature maps"""
    
    def __init__(
        self, in_channels: int,
        out_channels_list: list[int],
    ):
        """Initialize"""
        super(SpatialAttentionBlock, self).__init__()
        self.n_layers = len(out_channels_list)
        channels_list = [in_channels] + out_channels_list
        assert self.n_layers > 0
        assert channels_list[-1] == 1
        
        for i in range(self.n_layers - 1):
            in_chs, out_chs = channels_list[i: i + 2]
            layer = Conv2dBNActiv(in_chs, out_chs, 3, 1, 1, activ="relu")
            setattr(self, f"conv{i + 1}", layer)
            
        in_chs, out_chs = channels_list[-2:]
        layer = Conv2dBNActiv(in_chs, out_chs, 3, 1, 1, activ="sigmoid")
        setattr(self, f"conv{self.n_layers}", layer)
    
    def forward(self, x):
        """Forward"""
        h = x
        for i in range(self.n_layers):
            h = getattr(self, f"conv{i + 1}")(h)
            
        h = h * x
        return h


class MultilabelModel(nn.Module):
    """
    A class to make a model consisting of an embedding model (backbone)
    and several classifiers (head)
    Currently maintained architectures are:
        MobileNet, EfficientNet, ConvNext, ResNet, ViT
    """
    def __init__(self,
                 cfg_model: dict, 
                 classes: dict):
        super().__init__()
        self.emb_model = timm.create_model(cfg_model['model'], pretrained=cfg_model['pretrained'])
        self.emb_size = self.emb_model.num_features
        self.emb_model.reset_classifier(0, '')
        self.set_dropout(self.emb_model, cfg_model['backbone_dropout'])

        self.classifiers = nn.ModuleDict()
        for target_name in classes:
            self.classifiers[target_name] = nn.Sequential(
                SpatialAttentionBlock(self.emb_size, [64, 32, 16, 1]),
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(start_dim=1),
                nn.Linear(self.emb_size, self.emb_size),
                nn.ReLU(),
                nn.Dropout(cfg_model['classifier_dropout']),
                nn.Linear(self.emb_size, len(classes[target_name]))
            )

    def forward(self, x):
        emb = self.emb_model(x)
        return {
            class_name: classifier(emb)
            for class_name, classifier in self.classifiers.items()
        }

    def set_backbone_state(self, state: str = 'freeze'):
        for param in self.emb_model.parameters():
            if state == 'freeze':
                param.requires_grad = False
            elif state == 'unfreeze':
                param.requires_grad = True
    
    @staticmethod
    def set_dropout(model: nn.Module, drop_rate: float = 0.2) -> None:
        '''Set new `drop_rate` for model'''
        for child in model.children():
            if isinstance(child, torch.nn.Dropout):
                child.p = drop_rate
            MultilabelModel.set_dropout(child, drop_rate=drop_rate)
    

def get_model(cfg_model, classes, device='cpu', compile: bool=True):
    model = MultilabelModel(cfg_model, classes)

    model.to(device)
    if compile:
        model = torch.compile(model, dynamic=True)

    return model
