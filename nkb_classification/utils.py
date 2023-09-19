from comet_ml import Experiment

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam, SparseAdam, SGD, RAdam, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchsampler import ImbalancedDatasetSampler
from torchvision.datasets import ImageFolder
import timm

from nkb_classification.dataset import Transforms, InferDataset, GroupsDataset, AnnotatedMultilabelDataset
    

def get_experiment(cfg_exp):
    if cfg_exp is None:
        return None
    name = cfg_exp.pop('name')
    exp = Experiment(**cfg_exp)
    exp.set_name(name)
    return exp

def get_optimizer(model, cfg_opt):
    if cfg_opt['type'] == 'adam':
        return Adam(model.parameters(), lr=cfg_opt['lr'], weight_decay=cfg_opt.get('weight_decay', 0.0))
    if cfg_opt['type'] == 'radam':
        return RAdam(model.parameters(), lr=cfg_opt['lr'], weight_decay=cfg_opt.get('weight_decay', 0.0), decoupled_weight_decay=True)
    if cfg_opt['type'] == 'sparse_adam':
        return SparseAdam(model.parameters(), lr=cfg_opt['lr'], weight_decay=cfg_opt.get('weight_decay', 0.0))
    if cfg_opt['type'] == 'sgd':
        return SGD(model.parameters(), lr=cfg_opt['lr'], weight_decay=cfg_opt.get('weight_decay', 0.0))
    else:
        raise NotImplementedError(f'Unknown optimizer in config: {cfg_opt["type"]}')

def get_scheduler(opt, lr_policy):
    if len(lr_policy) == 0:
        return None
    if lr_policy['type'] == 'step':
        scheduler = lr_scheduler.StepLR(
            opt,
            step_size=lr_policy['step_size'],
            gamma=lr_policy['gamma'])
    elif lr_policy['type'] == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(
            opt,
            milestones=lr_policy['steps'],
            gamma=lr_policy['gamma'])
    else:
        raise NotImplementedError('Learning rate policy {} not implemented.'.
                                   format(lr_policy['type']))
    return scheduler

def get_loss(cfg_loss, device):
    if cfg_loss['type'] == 'CrossEntropyLoss':
        weight = None
        if 'weight' in cfg_loss:
            weight = torch.tensor(cfg_loss['weight'], dtype=torch.float)
        return nn.CrossEntropyLoss(weight).to(device)
    elif cfg_loss['type'] == 'FocalLoss':
        alpha = None
        if 'alpha' in cfg_loss:
            alpha = torch.tensor(cfg_loss['alpha'], dtype=torch.float)
        gamma = 0
        if 'gamma' in cfg_loss:
            gamma = cfg_loss['gamma']
        return FocalLoss(alpha, gamma).to(device)
    else:
        raise NotImplementedError(f'Unknown loss type in config: {cfg_loss["type"]}')

def get_model(cfg_model, n_classes, device='cpu', compile: bool=True):
    model = timm.create_model(cfg_model['model'], pretrained=cfg_model['pretrained'])
    if cfg_model['model'].startswith('vit'):
        model.head = nn.Linear(in_features=model.head.in_features, out_features=n_classes)
    elif cfg_model['model'].startswith('resnet'):
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=n_classes)
    elif cfg_model['model'].startswith('efficientnet') or cfg_model['model'].startswith('mobilenet'):
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=n_classes)
    
    chkp = cfg_model.get('checkpoint', None)
    if chkp is not None:
        model.load_state_dict(torch.load(chkp, map_location=device))
    model.to(device)
    if compile:
        model = torch.jit.script(model)
    return model

def get_dataset(data, pipeline):
    transform = Transforms(pipeline)
    if data['type'] == 'GroupsDataset':
        dataset = GroupsDataset(data['root'],
                                data['ann_file'], 
                                data['group_dict'],
                                transform=transform)
    elif data['type'] == 'AnnotatedMultilabelDataset':
        dataset = AnnotatedMultilabelDataset(data['ann_file'],
                                             data['target_name'],
                                             data['fold'],
                                             transform=transform)
    else:
        dataset = ImageFolder(data['root'], transform=transform)
    if data.get('weighted_sampling', False):
        loader = DataLoader(dataset, batch_size=data['batch_size'],  
                            sampler=ImbalancedDatasetSampler(dataset),
                            num_workers=data['num_workers'], pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=data['batch_size'], 
                            shuffle=data['shuffle'], num_workers=data['num_workers'], pin_memory=True)
    return loader

def get_inference_dataset(data, pipeline):
    transform = Transforms(pipeline)
    dataset = InferDataset(data['root'], transform=transform)
    loader = DataLoader(dataset, batch_size=data['batch_size'], 
                        num_workers=data['num_workers'], pin_memory=True)
    return loader

class FocalLoss(nn.Module):
    '''
    inspired by https://github.com/AdeelH/pytorch-multi-class-focal-loss/tree/master
    '''

    def __init__(self,
                 alpha=None,
                 gamma=0.,):

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.CrossEntropyLoss = nn.CrossEntropyLoss(
            weight=alpha,
            reduction='none')

    def forward(self, x, y):
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        if len(y) == 0:
            return torch.tensor(0.)

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        ce = self.CrossEntropyLoss(x, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        pt = x[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = (focal_term * ce).mean()

        return loss
