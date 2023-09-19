from comet_ml import Experiment

import torch
import torch.nn as nn
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
            weight = torch.tensor(cfg_loss.get('weight', None), dtype=torch.float)
        return nn.CrossEntropyLoss(weight).to(device)
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
