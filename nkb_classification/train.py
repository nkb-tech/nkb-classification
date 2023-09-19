from pathlib import Path
import numpy as np
import sys
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from collections import defaultdict
from typing import Callable

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import make_grid
from torch.cuda.amp import GradScaler

import argparse
from tqdm import tqdm

from nkb_classification.utils import get_experiment, get_model, \
    get_optimizer, get_scheduler, get_dataset

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
    epoch_train_loss, epoch_val_loss = [], []
    best_val_acc = 0
    inv_transform = transforms.Compose([
        transforms.Normalize(mean = [ 0., 0., 0. ],
                             std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                             std = [ 1., 1., 1. ]),
        transforms.ToPILImage(),
    ])
    class_to_idx = val_loader.dataset.class_to_idx
    label_names = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]

    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler(enabled=cfg.enable_gradient_scaler)

    model_saver: Callable[[nn.Module], None] = torch.jit.save if cfg.compile else torch.save

    for epoch in tqdm(range(n_epochs), desc='Training epochs'):
        model.train()

        train_running_loss = []
        val_running_loss = []
        train_confidences = []
        train_predictions = []
        train_ground_truth = []
        val_confidences = []
        val_predictions = []
        val_ground_truth = []

        if cfg.log_gradients:
            metrics_grad_log = defaultdict(list)

        for img, target in tqdm(train_loader, leave=False, desc='Training iters'):
            train_ground_truth.extend(list(target.numpy()))
            img, target = img.float().to(device), target.long().to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.enable_mixed_presicion):
                preds = model(img)
                loss = criterion(preds, target)

            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

            train_running_loss.append(loss.item())
            train_confidences.extend(
                torch.nn.functional.softmax(preds, dim=-1, dtype=preds.dtype)
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
            preds = preds.argmax(dim=1)
            train_predictions.extend(list(preds.detach().cpu().numpy()))

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
            val_ground_truth.extend(list(target.numpy()))
            img, target = img.float().to(device), target.long().to(device)
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.enable_mixed_presicion):
                preds = model(img)
                loss = criterion(preds, target)
            
            val_running_loss.append(loss.item())
            val_confidences.extend(
                torch.nn.functional.softmax(preds, dim=-1, dtype=preds.dtype)
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
            preds = preds.argmax(dim=1)
            val_predictions.extend(preds.detach().cpu().numpy().tolist())
            if experiment is not None and not logged:
                batch_img = img.to('cpu')
                grid = inv_transform(make_grid(batch_img, nrow=8, padding=2))
                experiment.log_image(grid, name=f'Epoch {epoch}', step=epoch)
                logged = True
        
        train_acc = balanced_accuracy_score(train_ground_truth, train_predictions)
        val_acc = balanced_accuracy_score(val_ground_truth, val_predictions)
        try:
            train_roc_auc = roc_auc_score(train_ground_truth, train_confidences, average=None, multi_class='ovr')
            val_roc_auc = roc_auc_score(val_ground_truth, val_confidences, average=None, multi_class='ovr')
        except:
            import pdb; pdb.set_trace()
        print(f'Epoch {epoch} train roc_auc {train_roc_auc}')
        print(f'Epoch {epoch} train balanced accuracy {train_acc}')
        print(f'Epoch {epoch} val roc_auc {val_roc_auc}')
        print(f'Epoch {epoch} validation balanced accuracy {val_acc}')
        epoch_train_loss = np.mean(train_running_loss)
        epoch_val_loss = np.mean(val_running_loss)
        if experiment is not None:
            experiment.log_metric(f'Average epoch train loss', epoch_train_loss, epoch=epoch, step=epoch)
            experiment.log_metric(f'Average epoch val loss', epoch_val_loss, epoch=epoch, step=epoch)
            for roc_auc, class_name in zip(train_roc_auc, train_loader.dataset.classes):
                experiment.log_metric(f'Train ROC AUC, {class_name}', roc_auc, epoch=epoch, step=epoch)
            experiment.log_metric(f'Train ROC AUC', np.mean(train_roc_auc), epoch=epoch, step=epoch)
            for roc_auc, class_name in zip(val_roc_auc, val_loader.dataset.classes):
                experiment.log_metric(f'Validation ROC AUC, {class_name}', roc_auc, epoch=epoch, step=epoch)
            experiment.log_metric(f'Validation ROC AUC', np.mean(val_roc_auc), epoch=epoch, step=epoch)
            experiment.log_metric(f'Train balanced accuracy', train_acc, epoch=epoch, step=epoch)
            experiment.log_metric(f'Validation balanced accuracy', val_acc, epoch=epoch, step=epoch)
            experiment.log_confusion_matrix(val_ground_truth, val_predictions, labels=label_names, epoch=epoch)

            if cfg.log_gradients:
                for key, value in metrics_grad_log.items():
                    experiment.log_metric(key, torch.nanmean(torch.stack(value)), epoch=epoch, step=epoch)

                metrics_grad_log = defaultdict(list)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_saver(model, Path(model_path, 'best.pth'))
        model_saver(model, Path(model_path, 'last.pth'))
        
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
    n_classes = len(train_loader.dataset.classes)
    device = torch.device(cfg.device)
    model = get_model(cfg.model, n_classes, device, compile=cfg.compile)
    optimizer = get_optimizer(model, cfg.optimizer)
    scheduler = get_scheduler(optimizer, cfg.lr_policy)
    experiment = get_experiment(cfg.experiment)
    experiment.log_code(cfg_file)
    train(model, train_loader, val_loader,
          optimizer, scheduler, cfg.criterion, experiment, 
          device, cfg)

if __name__ == '__main__':
    main()

