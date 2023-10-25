from comet_ml import Experiment

from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam, SparseAdam, SGD, RAdam, lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms
from torchsampler import ImbalancedDatasetSampler
from torchvision.datasets import ImageFolder
import timm

from nkb_classification.dataset import Transforms, InferDataset, GroupsDataset, AnnotatedMultitargetDataset

from sklearn.metrics import balanced_accuracy_score, roc_auc_score


class MultilabelModel(nn.Module):
    """
    A class to make a model consisting of an embedding model (backbone)
    and several classifiers (head)
    Currently maintained architectures are:
        MobileNet, EfficientNet, ConvNext, ResNet, ViT
    """
    def __init__(self,
                 cfg_model, 
                 classes):
        super().__init__()
        self.name = cfg_model['model']
        self.get_emb_model(cfg_model)
        def set_dropout(model, drop_rate=0.2):
            for child in model.children():
                if isinstance(child, torch.nn.Dropout):
                    child.p = drop_rate
                set_dropout(child, drop_rate=drop_rate)
        set_dropout(self.emb_model, cfg_model['backbone dropout'])

        self.classifiers = nn.ModuleDict()
        for target_name in classes:
            self.classifiers[target_name] = nn.Sequential(nn.Dropout(cfg_model['classifier dropout']),
                                                          nn.Linear(self.emb_size, len(classes[target_name])))
        
        # if self.name.startswith('beit'):
        #     def classifiers_forward(x):
        #         return {
        #             class_name: classifier(x)
        #             for class_name, classifier in self.classifiers.items()
        #         }
        #     self.classifiers.forward = classifiers_forward
        #     self.model.head = self.classifiers
            
    def forward(self, x):
        # if self.name.startswith('beit'):
        #     return self.model(x)
        # else:
        emb = self.emb_model(x)
        return {
            class_name: classifier(emb)
            for class_name, classifier in self.classifiers.items()
        }
    
    def get_emb_model(self, cfg_model):
        initial_model = timm.create_model(cfg_model['model'], pretrained=cfg_model['pretrained'])
        if self.name.startswith('efficientnet') or self.name.startswith('mobilenet'):
            self.emb_size = initial_model.classifier.in_features
            self.emb_model = nn.Sequential(*[*initial_model.children()][:-1],
                                            nn.Flatten())
        elif self.name.startswith('convnext'):
            self.emb_size = initial_model.head.in_features
            new_head = nn.Sequential(*[*[*initial_model.children()][-1].children()][:-2])
            self.emb_model = nn.Sequential(*[*initial_model.children()][:-1],
                                            new_head)
        elif self.name.startswith('resnet'):
            self.emb_size = initial_model.fc.in_features
            self.emb_model = nn.Sequential(*[*initial_model.children()][:-1],
                                            nn.Flatten())
        elif self.name.startswith('vit'):
            self.emb_size = initial_model.patch_embed.num_patches * initial_model.head.in_features
            self.emb_model = nn.Sequential(*[*initial_model.children()][:-2],
                                            nn.Flatten())
        # elif self.name.startswith('beit'):
        #     emb_size = initial_model.head.in_features
        #     self.model = initial_model


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
    api_key_path = cfg_exp.pop('api_key_path')
    with open(api_key_path, 'r') as api_key_file:
        cfg_exp['api_key'] = api_key_file.readline()
    name = cfg_exp.pop('name')
    exp = Experiment(**cfg_exp)
    exp.set_name(name)
    return exp


def get_optimizers(model, cfg_back_opt, cfg_class_opt):
    if cfg_back_opt['type'] == 'adam':
        back_opt = Adam([*model.parameters()][:-2*len(model.classifiers)], lr=cfg_back_opt['lr'], weight_decay=cfg_back_opt.get('weight_decay', 0.0))
    elif cfg_back_opt['type'] == 'radam':
        back_opt = RAdam([*model.parameters()][:-2*len(model.classifiers)], lr=cfg_back_opt['lr'], weight_decay=cfg_back_opt.get('weight_decay', 0.0), decoupled_weight_decay=True)
    elif cfg_back_opt['type'] == 'sparse_adam':
        back_opt = SparseAdam([*model.parameters()][:-2*len(model.classifiers)], lr=cfg_back_opt['lr'], weight_decay=cfg_back_opt.get('weight_decay', 0.0))
    elif cfg_back_opt['type'] == 'sgd':
        back_opt = SGD([*model.parameters()][:-2*len(model.classifiers)], lr=cfg_back_opt['lr'], weight_decay=cfg_back_opt.get('weight_decay', 0.0))
    else:
        raise NotImplementedError(f'Unknown optimizer in config: {cfg_back_opt["type"]}')
    if cfg_class_opt['type'] == 'adam':
        class_opt = Adam([*model.parameters()][-2*len(model.classifiers):], lr=cfg_class_opt['lr'], weight_decay=cfg_class_opt.get('weight_decay', 0.0))
    elif cfg_class_opt['type'] == 'radam':
        class_opt = RAdam([*model.parameters()][-2*len(model.classifiers):], lr=cfg_class_opt['lr'], weight_decay=cfg_class_opt.get('weight_decay', 0.0), decoupled_weight_decay=True)
    elif cfg_class_opt['type'] == 'sparse_adam':
        class_opt = SparseAdam([*model.parameters()][-2*len(model.classifiers):], lr=cfg_class_opt['lr'], weight_decay=cfg_class_opt.get('weight_decay', 0.0))
    elif cfg_class_opt['type'] == 'sgd':
        class_opt = SGD([*model.parameters()][-2*len(model.classifiers):], lr=cfg_class_opt['lr'], weight_decay=cfg_class_opt.get('weight_decay', 0.0))
    else:
        raise NotImplementedError(f'Unknown optimizer in config: {cfg_class_opt["type"]}')
    return back_opt, class_opt


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
        model = torch.compile(model)

    return model


def get_dataset(data, pipeline):
    transform = Transforms(pipeline)
    if data['type'] == 'GroupsDataset':
        dataset = GroupsDataset(data['root'],
                                data['ann_file'], 
                                data['group_dict'],
                                transform=transform)
    elif data['type'] == 'AnnotatedMultitargetDataset':
        dataset = AnnotatedMultitargetDataset(data['ann_file'],
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
    dataset = InferDataset(data['root'], 
                           data['train_annotations_file'],
                           data['target_names'],
                           transform=transform)
    loader = DataLoader(dataset, batch_size=data['batch_size'], 
                        num_workers=data['num_workers'], pin_memory=True)
    return loader


def log_images(experiment, 
               epoch,
               batch_to_log):
    inv_transform = transforms.Compose([
        transforms.Normalize(mean = [ 0., 0., 0. ],
                             std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                             std = [ 1., 1., 1. ]),
        transforms.ToPILImage(),
    ])
    grid = inv_transform(make_grid(batch_to_log, nrow=8, padding=2))
    experiment.log_image(grid, name=f'Epoch {epoch}', step=epoch)


def compute_targetwise_metrics(epoch_results, 
                               target_name):
    running_loss = epoch_results['running_loss'][target_name]
    confidences = epoch_results['confidences'][target_name]
    predictions = epoch_results['predictions'][target_name]
    ground_truth = epoch_results['ground_truth'][target_name]
    n_classes = len(confidences[0])
    epoch_acc = balanced_accuracy_score(ground_truth, predictions)
    if n_classes > 2:
        epoch_roc_auc = roc_auc_score(ground_truth, confidences, average=None, multi_class='ovr')
    else:
        epoch_roc_auc = roc_auc_score(ground_truth, np.array(confidences)[:, 1])
    epoch_loss = np.mean(running_loss)
    metrics = {
        'epoch_acc': epoch_acc,
        'epoch_roc_auc': epoch_roc_auc,
        'epoch_loss': epoch_loss
    }
    return metrics
    

def compute_metrics(epoch_results,
                    target_names):
    metrics = {target_name: compute_targetwise_metrics(epoch_results,
                                                       target_name) 
                                                       for target_name in target_names}
    metrics['loss'] = epoch_results['running_loss']['loss']
    metrics['epoch_acc'] = np.mean([metrics[target_name]['epoch_acc'] for target_name in target_names])
    return metrics


def log_targetwise_metrics(experiment, 
                target_name,
                label_names,
                epoch,
                metrics,
                fold='Train'):
    acc = metrics['epoch_acc']
    roc_auc = metrics['epoch_roc_auc']
    epoch_loss = metrics['epoch_loss']
    n_classes = len(label_names)
    print(f'{target_name} Epoch {epoch} {fold.lower()} roc_auc {roc_auc}')
    print(f'{target_name} Epoch {epoch} {fold.lower()} balanced accuracy {acc}')
    experiment.log_metric(f'{target_name} Average epoch {fold} loss', epoch_loss, epoch=epoch, step=epoch)
    if n_classes > 2:
        for roc_auc, class_name in zip(roc_auc, label_names):
            experiment.log_metric(f'{target_name} {fold} ROC AUC, {class_name}', roc_auc, epoch=epoch, step=epoch)
        experiment.log_metric(f'{target_name} {fold} ROC AUC', np.mean(roc_auc), epoch=epoch, step=epoch)
    else:
        experiment.log_metric(f'{target_name} {fold} ROC AUC', roc_auc, epoch=epoch, step=epoch)
    experiment.log_metric(f'{target_name} {fold} balanced accuracy', acc, epoch=epoch, step=epoch)


def log_metrics(experiment, 
                target_names,
                label_names,
                epoch,
                metrics,
                fold='Train'):
    for target_name in target_names:
        log_targetwise_metrics(experiment, 
                    target_name,
                    label_names[target_name],
                    epoch,
                    metrics[target_name],
                    fold)
    experiment.log_metric(f'{fold} loss', np.mean(metrics['loss']), epoch=epoch, step=epoch)
    experiment.log_metric(f'{fold} balanced accuracy', metrics['epoch_acc'], epoch=epoch, step=epoch)


def log_confusion_matrices(experiment, 
                    target_names,
                    label_names,
                    epoch,
                    results,
                    fold='Validation'):
    for target_name in target_names:
        experiment.log_confusion_matrix(results['ground_truth'][target_name],
                                        results['predictions'][target_name],
                                        labels=label_names[target_name], 
                                        title=f'{fold} {target_name} confusion matrix',
                                        file_name=f'{fold}-{target_name}-confusion-matrix.json',
                                        epoch=epoch)


def log_grads(experiment, 
              epoch, 
              metrics_grad_log):
    for key, value in metrics_grad_log.items():
        experiment.log_metric(key, torch.nanmean(torch.stack(value)), epoch=epoch, step=epoch)
    metrics_grad_log = defaultdict(list)
    return metrics_grad_log
