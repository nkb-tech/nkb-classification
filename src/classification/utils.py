from comet_ml import Experiment

import torch
import torch.nn as nn
from torch.optim import Adam, SparseAdam, SGD, lr_scheduler
import timm

def get_experiment(cfg_exp):
    if cfg_exp is None:
        return None
    return Experiment(**cfg_exp)

def get_optimizer(model, cfg_opt):
    if cfg_opt['type'] == 'adam':
        return Adam(model.parameters(), lr=cfg_opt['lr'], weight_decay=cfg_opt.get('weight_decay', 0.0))
    if cfg_opt['type'] == 'sparse_adam':
        return SparseAdam(model.parameters(), lr=cfg_opt['lr'], weight_decay=cfg_opt.get('weight_decay', 0.0))
    if cfg_opt['type'] == 'sgd':
        return SGD(model.parameters(), lr=cfg_opt['lr'], weight_decay=cfg_opt.get('weight_decay', 0.0))
    else:
        raise NotImplementedError(f'Unknown optimizer in config: {cfg_opt["type"]}')

def get_scheduler(opt, cfg_opt):
    if 'lr_policy' not in cfg_opt:
        return None
    if cfg_opt['lr_policy']['type'] == 'step':
        scheduler = lr_scheduler.StepLR(
            opt,
            step_size=cfg_opt['lr_policy']['step_size'],
            gamma=cfg_opt['lr_policy']['gamma'])
    elif cfg_opt['lr_policy']['type'] == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(
            opt,
            milestones=cfg_opt['lr_policy']['steps'],
            gamma=cfg_opt['lr_policy']['gamma'])
    else:
        raise NotImplementedError('Learning rate policy {} not implemented.'.
                                   format(cfg_opt['lr_policy']['type']))
    return scheduler

def get_loss(cfg_loss, device):
    if cfg_loss['type'] == 'CrossEntropyLoss':
        weight = None
        if 'weight' in cfg_loss:
            weight = torch.tensor(cfg_loss.get('weight', None), dtype=torch.float)
        return nn.CrossEntropyLoss(weight).to(device)
    else:
        raise NotImplementedError(f'Unknown loss type in config: {cfg_loss["type"]}')

def get_model(cfg_model, n_classes, device='cpu'):
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
    return model
