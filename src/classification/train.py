from pathlib import Path
import numpy as np
import yaml
import sys
from sklearn.metrics import balanced_accuracy_score

import torch
from torchvision import transforms
from torchvision.utils import make_grid

import argparse
from tqdm import tqdm

from src.classification.utils import get_experiment, get_model, get_optimizer, get_scheduler
from src.classification.dataset import get_dataset

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
    for epoch in tqdm(range(n_epochs)):
        model.train()
        train_running_loss = []
        val_running_loss = []
        train_predictions = []
        train_ground_truth = []
        val_predictions = []
        val_ground_truth = []
        for img, target in tqdm(train_loader, leave=False):
            train_ground_truth.extend(list(target.numpy()))
            img, target = img.float().to(device), target.long().to(device)
            optimizer.zero_grad()
            preds = model(img)
            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()

            train_running_loss.append(loss.item())
            preds = preds.argmax(dim=1)
            train_predictions.extend(list(preds.detach().cpu().numpy()))
        if scheduler is not None:
            scheduler.step()
        model.eval()
        logged = False
        for img, target in tqdm(val_loader, leave=False):
            val_ground_truth.extend(list(target.numpy()))
            img, target = img.float().to(device), target.long().to(device)
            preds = model(img)
            loss = criterion(preds, target)
            
            val_running_loss.append(loss.item())
            preds = preds.argmax(dim=1)
            val_predictions.extend(list(preds.detach().cpu().numpy()))
            if experiment is not None and not logged:
                batch_img = img.to('cpu')
                grid = inv_transform(make_grid(batch_img, nrow=8, padding=2))
                experiment.log_image(grid, name=f'Epoch {epoch}', step=epoch)
                logged = True

        train_acc = balanced_accuracy_score(train_ground_truth, train_predictions)
        val_acc = balanced_accuracy_score(val_ground_truth, val_predictions)
        print(f'Epoch {epoch} train balanced accuracy {train_acc}')
        print(f'Epoch {epoch} validation balanced accuracy {val_acc}')
        epoch_train_loss = np.mean(train_running_loss)
        epoch_val_loss = np.mean(val_running_loss)
        if experiment is not None:
            experiment.log_metric(f'Average epoch train loss', epoch_train_loss, epoch=epoch, step=epoch)
            experiment.log_metric(f'Average epoch val loss', epoch_val_loss, epoch=epoch, step=epoch)
            experiment.log_metric(f'Train balanced accuracy', train_acc, epoch=epoch, step=epoch)
            experiment.log_metric(f'Validation balanced accuracy', val_acc, epoch=epoch, step=epoch)
            experiment.log_confusion_matrix(val_ground_truth, val_predictions, labels=label_names, epoch=epoch)
        m = torch.jit.script(model)    
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.jit.save(m, Path(model_path, 'best.pth'))
        torch.jit.save(m, Path(model_path, 'last.pth'))
        
def read_py_config(path):
    path = Path(path)
    sys.path.append(str(path.parent))
    line = f'import {path.stem} as cfg'
    return line

def main():
    # import ipdb; ipdb.set_trace()
    parser = argparse.ArgumentParser(description='Train arguments')
    parser.add_argument('-cfg', '--config', help='Config file path', type=str, default='', required=True)
    args = parser.parse_args()
    cfg_file = args.config
    line = read_py_config(cfg_file)
    exec(line, globals(), globals())
    train_loader = get_dataset(cfg.train_data, cfg.train_pipeline)
    val_loader = get_dataset(cfg.val_data, cfg.val_pipeline)
    n_classes = len(train_loader.dataset.classes)
    device = torch.device(cfg.device)
    model = get_model(cfg.model, n_classes, device)
    optimizer = get_optimizer(model, cfg.optimizer)
    scheduler = get_scheduler(optimizer, cfg.lr_policy)
    experiment = get_experiment(cfg.experiment)
    train(model, train_loader, val_loader,
          optimizer, scheduler, cfg.criterion, experiment, 
          device, cfg)

if __name__ == '__main__':
    main()

