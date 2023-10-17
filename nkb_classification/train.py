from pathlib import Path
import numpy as np
import sys

import comet_ml

from collections import defaultdict
from typing import Callable, Union

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import make_grid
from torch.cuda.amp import GradScaler

import argparse
from tqdm import tqdm

from nkb_classification.utils import get_experiment, get_model, \
    get_optimizer, get_scheduler, get_loss, get_dataset, log_metrics

import warnings
warnings.filterwarnings("ignore")


def train(model, 
          train_loader, 
          val_loader, 
          optimizer, 
          scheduler, 
          criterion, 
          experiment,
          device,
          cfg):
    model_path = Path(cfg.model_path)
    model_path.mkdir(exist_ok=True, parents=True)
    n_epochs = cfg.n_epochs
    best_val_acc = 0
    inv_transform = transforms.Compose([
        transforms.Normalize(mean = [ 0., 0., 0. ],
                             std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                             std = [ 1., 1., 1. ]),
        transforms.ToPILImage(),
    ])
    class_to_idx = train_loader.dataset.class_to_idx
    target_names = [*sorted(class_to_idx)]
    label_names = {target_name: [*class_to_idx[target_name].keys()] for target_name in target_names}

    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler(enabled=cfg.enable_gradient_scaler)

    model_saver: Callable[[nn.Module], None] = torch.save
    model_scripter: Callable[[nn.Module], Union[nn.Module, torch.jit.ScriptModule]] = lambda x: x

    for epoch in tqdm(range(n_epochs), desc='Training epochs'):
        model.train()

        train_running_loss = defaultdict(list) #{target_name: [] for target_name in target_names}
        train_confidences = defaultdict(list) # {target_name: [] for target_name in target_names}
        train_predictions = defaultdict(list) #{target_name: [] for target_name in target_names}
        train_ground_truth = defaultdict(list) #{target_name: [] for target_name in target_names}
        val_confidences = defaultdict(list) #{target_name: [] for target_name in target_names}
        val_predictions = defaultdict(list) # {target_name: [] for target_name in target_names}
        val_ground_truth = defaultdict(list) #{target_name: [] for target_name in target_names}
        val_running_loss = defaultdict(list) #{target_name: [] for target_name in target_names}

        if cfg.log_gradients:
            metrics_grad_log = defaultdict(list)

        for img, target in tqdm(train_loader, leave=False, desc='Training iters'):
            img = img.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.enable_mixed_presicion):
                preds = model(img)
                loss = 0

                for target_name in target_names:
                    target_loss = criterion(preds[target_name], target[target_name].to(device))
                    train_running_loss[target_name].append(target_loss.item())
                    loss += target_loss

            train_running_loss['loss'].append(loss.item())
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            for target_name in target_names:
                train_ground_truth[target_name].extend(list(target[target_name].cpu().numpy()))

                train_confidences[target_name].extend(
                    torch.nn.functional.softmax(preds[target_name], dim=-1, dtype=torch.float32)
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                preds[target_name] = preds[target_name].argmax(dim=-1)
                train_predictions[target_name].extend(list(preds[target_name].detach().cpu().numpy()))

            if cfg.log_gradients:
                total_grad = 0
                for tag, value in model.named_parameters():
                    assert tag != "Total"
                    if value.grad is not None:
                        grad = value.grad.norm()
                        metrics_grad_log[f"Gradients/{tag}"].append(grad)
                        total_grad += grad

                metrics_grad_log["Gradients/Total"].append(total_grad)

        if scheduler is not None:
            scheduler.step()

        model.eval()
        logged = False
        for img, target in tqdm(val_loader, leave=False, desc='Evaluating'):
            img = img.to(device)
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.enable_mixed_presicion):
                preds = model(img)
                loss = 0
                for target_name in target_names:
                    target_loss = criterion(preds[target_name], target[target_name].to(device))
                    val_running_loss[target_name].append(target_loss.item())
                    loss += target_loss

            val_running_loss['loss'].append(loss.item())
            
            for target_name in target_names:
                val_ground_truth[target_name].extend(list(target[target_name].cpu().numpy()))

                val_confidences[target_name].extend(
                    torch.nn.functional.softmax(preds[target_name], dim=-1, dtype=torch.float32)
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                preds[target_name] = preds[target_name].argmax(dim=-1)
                val_predictions[target_name].extend(list(preds[target_name].detach().cpu().numpy()))

            if experiment is not None and not logged:
                batch_img = img.to('cpu')
                grid = inv_transform(make_grid(batch_img, nrow=8, padding=2))
                experiment.log_image(grid, name=f'Epoch {epoch}', step=epoch)
                logged = True

        epoch_train_acc, epoch_val_acc = [], []
        for target_name in target_names:
            train_acc, val_acc = log_metrics(experiment, 
                target_name,
                label_names[target_name],
                epoch,
                train_running_loss[target_name],
                val_running_loss[target_name],
                train_confidences[target_name],
                train_predictions[target_name],
                train_ground_truth[target_name],
                val_confidences[target_name],
                val_predictions[target_name],
                val_ground_truth[target_name])
            epoch_train_acc.append(train_acc)
            epoch_val_acc.append(val_acc)

        experiment.log_metric('Train loss', np.mean(train_running_loss['loss']), epoch=epoch, step=epoch)
        experiment.log_metric('Validation loss', np.mean(val_running_loss['loss']), epoch=epoch, step=epoch)

        if experiment is not None:
            if cfg.log_gradients:
                for key, value in metrics_grad_log.items():
                    experiment.log_metric(key, torch.nanmean(torch.stack(value)), epoch=epoch, step=epoch)
                metrics_grad_log = defaultdict(list)

        m = model_scripter(model)

        epoch_train_acc, epoch_val_acc = np.mean(epoch_train_acc), np.mean(epoch_val_acc)
        experiment.log_metric('Train balanced accuracy', epoch_train_acc, epoch=epoch, step=epoch)
        experiment.log_metric('Validation balanced accuracy', epoch_val_acc, epoch=epoch, step=epoch)

        if epoch_val_acc > best_val_acc:
            best_val_acc = val_acc
            model_saver(m, Path(model_path, 'best.pth'))
        model_saver(m, Path(model_path, 'last.pth'))
        
def read_py_config(path):
    path = Path(path)
    sys.path.append(str(path.parent))
    line = f'import {path.stem} as cfg'
    return line

def main():
    parser = argparse.ArgumentParser(description='Train arguments')
    parser.add_argument('-cfg', '--config', help='Config file path', type=str, default='', required=True)
    args = parser.parse_args()
    cfg_file = args.config
    exec(read_py_config(cfg_file), globals(), globals())
    train_loader = get_dataset(cfg.train_data, cfg.train_pipeline)
    val_loader = get_dataset(cfg.val_data, cfg.val_pipeline)
    classes = train_loader.dataset.classes
    device = torch.device(cfg.device)
    model = get_model(cfg.model, classes, device, compile=cfg.compile)
    optimizer = get_optimizer(model, cfg.optimizer)
    scheduler = get_scheduler(optimizer, cfg.lr_policy)
    criterion = get_loss(cfg.criterion, cfg.device)
    experiment = get_experiment(cfg.experiment)
    experiment.log_code(cfg_file)
    train(model, train_loader, val_loader,
          optimizer, scheduler, criterion, experiment, 
          device, cfg)

if __name__ == '__main__':
    main()

