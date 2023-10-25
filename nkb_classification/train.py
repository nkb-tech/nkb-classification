from pathlib import Path
import numpy as np
import sys

import comet_ml

from collections import defaultdict

import torch
from torch.cuda.amp import GradScaler

import argparse
from tqdm import tqdm

from nkb_classification.utils import get_experiment, get_model, \
    get_optimizers, get_scheduler, get_loss, get_dataset, \
    log_images, compute_metrics, log_metrics, log_confusion_matrices, log_grads

def train_epoch(model,
                train_loader,
                backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler,
                scaler,
                criterion,
                target_names,
                device, cfg):
    train_running_loss = defaultdict(list)
    train_confidences = defaultdict(list)
    train_predictions = defaultdict(list)
    train_ground_truth = defaultdict(list)

    if cfg.log_gradients:
        metrics_grad_log = defaultdict(list)

    for img, target in tqdm(train_loader, leave=False, desc='Training iters'):
        img = img.to(device)
        backbone_optimizer.zero_grad()
        backbone_optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.enable_mixed_presicion):
            preds = model(img)
            loss = 0

            for target_name in target_names:
                target_loss = criterion(preds[target_name], target[target_name].to(device))
                train_running_loss[target_name].append(target_loss.item())
                loss += target_loss

        train_running_loss['loss'].append(loss.item())
        
        scaler.scale(loss).backward()
        scaler.step(backbone_optimizer)
        scaler.step(classifier_optimizer)
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

    if backbone_scheduler is not None:
        backbone_scheduler.step()
    if classifier_scheduler is not None:
        classifier_scheduler.step()

    results = {
        'running_loss': train_running_loss,
        'confidences': train_confidences,
        'predictions': train_predictions,
        'ground_truth': train_ground_truth
    }

    if cfg.log_gradients:
        results['metrics_grad_log'] = metrics_grad_log

    return results


def val_epoch(model,
              val_loader,
              criterion,
              target_names,
              device, cfg):
    
    val_confidences = defaultdict(list)
    val_predictions = defaultdict(list)
    val_ground_truth = defaultdict(list)
    val_running_loss = defaultdict(list)

    batch_to_log = None
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

        if batch_to_log is None:
            batch_to_log = img.to('cpu')

    results = {
        'running_loss': val_running_loss,
        'confidences': val_confidences,
        'predictions': val_predictions,
        'ground_truth': val_ground_truth,
        'images': batch_to_log
    }

    return results


def train(model,
          train_loader, val_loader,
          backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler,
          criterion,
          experiment, 
          device, cfg):
    model_path = Path(cfg.model_path)
    model_path.mkdir(exist_ok=True, parents=True)
    n_epochs = cfg.n_epochs
    best_val_acc = 0
    class_to_idx = train_loader.dataset.class_to_idx
    target_names = [*sorted(class_to_idx)]
    label_names = {target_name: [*class_to_idx[target_name].keys()] for target_name in target_names}

    scaler = GradScaler(enabled=cfg.enable_gradient_scaler)

    for epoch in tqdm(range(n_epochs), desc='Training epochs'):

        model.train()
        train_results = train_epoch(model,
                train_loader,
                backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler,
                scaler,
                criterion,
                target_names,
                device, cfg)

        model.eval()
        val_results = val_epoch(model,
              val_loader,
              criterion,
              target_names,
              device, cfg)
        
        # import ipdb; ipdb.set_trace()

        epoch_val_acc = None
        if experiment is not None:
            log_images(experiment, 
                    epoch, 
                    val_results['images'])

            train_metrics = compute_metrics(train_results,
                                            target_names)

            log_metrics(experiment, 
                    target_names,
                    label_names,
                    epoch,
                    train_metrics,
                    'Train')
            
            val_metrics = compute_metrics(val_results,
                                          target_names)
            epoch_val_acc = val_metrics['epoch_acc']

            log_metrics(experiment, 
                    target_names,
                    label_names,
                    epoch,
                    val_metrics,
                    'Validation')
            
            log_confusion_matrices(experiment, 
                    target_names,
                    label_names,
                    epoch,
                    val_results,
                    'Validation')
            
            if cfg.log_gradients:
                log_grads(experiment, 
                    epoch, 
                    train_results['metrics_grad_log'])

        if epoch_val_acc is not None:
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save(model, Path(model_path, 'best.pth'))
        torch.save(model, Path(model_path, 'last.pth'))

        
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
    backbone_optimizer, classifier_optimizer = get_optimizers(model, cfg.backbone_optimizer, cfg.classifier_optimizer)
    backbone_scheduler = get_scheduler(backbone_optimizer, cfg.backbone_lr_policy)
    classifier_scheduler = get_scheduler(classifier_optimizer, cfg.classifier_lr_policy)
    criterion = get_loss(cfg.criterion, cfg.device)
    experiment = get_experiment(cfg.experiment)
    experiment.log_code(cfg_file)
    train(model,
          train_loader, val_loader,
          backbone_optimizer, classifier_optimizer, backbone_scheduler, classifier_scheduler,
          criterion,
          experiment, 
          device, cfg)


if __name__ == '__main__':
    main()
