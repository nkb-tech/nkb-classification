from comet_ml import Experiment

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam, SparseAdam, SGD, RAdam, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchsampler import ImbalancedDatasetSampler
from torchvision.datasets import ImageFolder
import timm

from nkb_classification.dataset import Transforms, InferDataset, GroupsDataset, AnnotatedMultilabelDataset

from sklearn.metrics import balanced_accuracy_score, roc_auc_score


class MultilabelModel(nn.Module):
    def __init__(self,
                 cfg_model, 
                 classes):
        super().__init__()
        self.emb_model = timm.create_model(cfg_model['model'], pretrained=cfg_model['pretrained'])
        if cfg_model['model'].startswith('efficientnet') or cfg_model['model'].startswith('mobilenet'):
            emb_size = self.emb_model.classifier.in_features
        elif cfg_model['model'].startswith('vit'):
            emb_size = self.emb_model.head.in_features
        elif cfg_model['model'].startswith('resnet'):
            emb_size = self.emb_model.fc.in_features
        self.emb_model = nn.Sequential(*[*self.emb_model.children()][:-1])

        self.classifiers = nn.ModuleDict()
        for target_name in classes:
            self.classifiers[target_name] = nn.Linear(emb_size, len(classes[target_name]))

    def forward(self, x):
        emb = self.emb_model(x)
        return (classifier(emb) for _, classifier in self.classifiers.items())


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
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            y = y.view(-1)

        if len(y) == 0:
            return torch.tensor(0.)

        ce = self.CrossEntropyLoss(x, y)
        all_rows = torch.arange(len(x))

        pt = x[all_rows, y]

        focal_term = (1 - pt)**self.gamma

        loss = (focal_term * ce).mean()

        return loss
    

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


def get_model(cfg_model, classes, device='cpu', compile: bool=True):
    model = MultilabelModel(cfg_model, classes)

    model.to(device)
    if compile:
        model = torch.compile(model, mode='reduce-overhead')
    else:
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
                                             data['target_names'],
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


def log_metrics(experiment, 
                target_name,
                label_names,
                epoch,
                train_running_loss,
                val_running_loss,
                train_confidences,
                train_predictions,
                train_ground_truth,
                val_confidences,
                val_predictions,
                val_ground_truth):
    train_acc = balanced_accuracy_score(train_ground_truth, train_predictions)
    val_acc = balanced_accuracy_score(val_ground_truth, val_predictions)
    if len(train_confidences[0]) > 2:
        train_roc_auc = roc_auc_score(train_ground_truth, train_confidences, average=None, multi_class='ovr')
        val_roc_auc = roc_auc_score(val_ground_truth, val_confidences, average=None, multi_class='ovr')
    else:
        train_roc_auc = roc_auc_score(train_ground_truth, np.array(train_confidences)[:, 1])
        val_roc_auc = roc_auc_score(val_ground_truth, np.array(val_confidences)[:, 1])
    print(f'{target_name} Epoch {epoch} train roc_auc {train_roc_auc}')
    print(f'{target_name} Epoch {epoch} train balanced accuracy {train_acc}')
    print(f'{target_name} Epoch {epoch} val roc_auc {val_roc_auc}')
    print(f'{target_name} Epoch {epoch} validation balanced accuracy {val_acc}')
    epoch_train_loss = np.mean(train_running_loss)
    epoch_val_loss = np.mean(val_running_loss)
    if experiment is not None:
        experiment.log_metric(f'{target_name} Average epoch train loss', epoch_train_loss, epoch=epoch, step=epoch)
        experiment.log_metric(f'{target_name} Average epoch val loss', epoch_val_loss, epoch=epoch, step=epoch)
        if len(train_confidences[0]) > 2:
            for roc_auc, class_name in zip(train_roc_auc, label_names):
                experiment.log_metric(f'{target_name} Train ROC AUC, {class_name}', roc_auc, epoch=epoch, step=epoch)
            experiment.log_metric(f'{target_name} Train ROC AUC', np.mean(train_roc_auc), epoch=epoch, step=epoch)
            for roc_auc, class_name in zip(val_roc_auc, label_names):
                experiment.log_metric(f'{target_name} Validation ROC AUC, {class_name}', roc_auc, epoch=epoch, step=epoch)
            experiment.log_metric(f'{target_name} Validation ROC AUC', np.mean(val_roc_auc), epoch=epoch, step=epoch)
        else:
            experiment.log_metric(f'{target_name} Train ROC AUC', train_roc_auc, epoch=epoch, step=epoch)
            experiment.log_metric(f'{target_name} Validation ROC AUC', val_roc_auc, epoch=epoch, step=epoch)
        experiment.log_metric(f'{target_name} Train balanced accuracy', train_acc, epoch=epoch, step=epoch)
        experiment.log_metric(f'{target_name} Validation balanced accuracy', val_acc, epoch=epoch, step=epoch)
        experiment.log_confusion_matrix(val_ground_truth, val_predictions, labels=label_names, epoch=epoch)
