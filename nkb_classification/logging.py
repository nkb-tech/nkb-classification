from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.utils import make_grid


def log_targetwise_metrics(experiment, target_name, label_names, epoch, metrics, fold="Train"):
    if target_name is None:
        target_name = ""
    acc = metrics["epoch_acc"]
    roc_auc = metrics["epoch_roc_auc"]
    epoch_loss = metrics["epoch_loss"]
    n_classes = len(label_names)
    # print(f'{target_name} Epoch {epoch} {fold.lower()} roc_auc {roc_auc:.4f}')
    # print(f'{target_name} Epoch {epoch} {fold.lower()} balanced accuracy {acc:.4f}')
    experiment.log_metric(
        f"{target_name} Average epoch {fold} loss".lstrip(),
        epoch_loss,
        epoch=epoch,
        step=epoch,
    )
    if n_classes > 2:
        for roc_auc, class_name in zip(roc_auc, label_names):
            experiment.log_metric(
                f"{target_name} {fold} ROC AUC, {class_name}".lstrip(),
                roc_auc,
                epoch=epoch,
                step=epoch,
            )
        experiment.log_metric(
            f"{target_name} {fold} ROC AUC".lstrip(),
            np.mean(roc_auc),
            epoch=epoch,
            step=epoch,
        )
    else:
        experiment.log_metric(
            f"{target_name} {fold} ROC AUC".lstrip(),
            roc_auc,
            epoch=epoch,
            step=epoch,
        )
    experiment.log_metric(
        f"{target_name} {fold} balanced accuracy".lstrip(),
        acc,
        epoch=epoch,
        step=epoch,
    )


def log_metrics(
    experiment,
    target_names,
    label_names,
    epoch,
    metrics,
    fold="Train",
):
    if target_names is None:
        log_targetwise_metrics(
            experiment,
            None,
            label_names,
            epoch,
            metrics,
            fold,
        )

    else:
        for target_name in target_names:
            log_targetwise_metrics(
                experiment,
                target_name,
                label_names[target_name],
                epoch,
                metrics[target_name],
                fold,
            )
    experiment.log_metric(
        f"{fold} loss",
        np.mean(metrics["loss"]),
        epoch=epoch,
        step=epoch,
    )
    experiment.log_metric(
        f"{fold} balanced accuracy",
        metrics["epoch_acc"],
        epoch=epoch,
        step=epoch,
    )


def log_confusion_matrices(
    experiment,
    target_names,
    label_names,
    epoch,
    results,
    fold="Validation",
):
    if target_names is None:
        experiment.log_confusion_matrix(
            results["ground_truth"],
            results["predictions"],
            labels=label_names,
            title=f"{fold} confusion matrix",
            file_name=f"{fold}-confusion-matrix.json",
            epoch=epoch,
        )
    else:
        for target_name in target_names:
            experiment.log_confusion_matrix(
                results["ground_truth"][target_name],
                results["predictions"][target_name],
                labels=label_names[target_name],
                title=f"{fold} {target_name} confusion matrix",
                file_name=f"{fold}-{target_name}-confusion-matrix.json",
                epoch=epoch,
            )


def log_images(experiment, name, epoch, batch_to_log, locally=False):
    inv_transform = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
            transforms.ToPILImage(),
        ]
    )
    grid = inv_transform(make_grid(batch_to_log, nrow=8, padding=2))
    if locally:
        plt.imsave(Path(experiment) / name, grid)
    else:
        experiment.log_image(grid, name=name, step=epoch)


def log_grads(experiment, epoch, metrics_grad_log):
    for key, value in metrics_grad_log.items():
        experiment.log_metric(
            key,
            torch.nanmean(torch.stack(value)),
            epoch=epoch,
            step=epoch,
        )
    metrics_grad_log = defaultdict(list)
    return metrics_grad_log


class TrainLogger:
    __slots__ = (
        "cfg",
        "epoch_running_loss",
        "epoch_confidences",
        "epoch_predictions",
        "epoch_ground_truth",
        "target_names",
        "label_names",
        "experiment",
        "task",
        "class_to_idx",
    )

    def __init__(self, cfg, experiment, class_to_idx):

        assert cfg.task in ("single", "multi")

        self.cfg = cfg
        self.experiment = experiment
        self.task = cfg.task
        self.class_to_idx = class_to_idx

    def log_images_at_start(self, save_path, loader, n_batches=3):

        for batch_num, (img_batch, _) in enumerate(loader):
            if batch_num + 1 > n_batches:
                break
            log_images(
                experiment=save_path,
                name=f"train_batch_{batch_num + 1}.png",
                epoch=None,
                batch_to_log=img_batch,
                locally=True
            )

    def init_iter_logs(self):

        if self.task == "single":
            self.epoch_running_loss = []
            self.epoch_confidences = []
            self.epoch_predictions = []
            self.epoch_ground_truth = []

            self.target_names = None
            # import ipdb; ipdb.set_trace()
            self.label_names = list(self.class_to_idx.keys())

        elif self.task == "multi":
            self.epoch_running_loss = defaultdict(list)
            self.epoch_confidences = defaultdict(list)
            self.epoch_predictions = defaultdict(list)
            self.epoch_ground_truth = defaultdict(list)

            self.target_names = [*sorted(self.class_to_idx)]

            self.label_names = {
                target_name: [*self.class_to_idx[target_name].keys()] for target_name in self.target_names
            }

    def log_iter(self, pred, true, loss):
        assert type(pred) == type(true)

        # Multi-task case
        if isinstance(pred, dict):
            assert pred.keys() == true.keys()
            for target_name in pred.keys():
                self.epoch_ground_truth[target_name].extend(true[target_name].cpu().numpy().tolist())
                self.epoch_confidences[target_name].extend(
                    pred[target_name].softmax(dim=-1, dtype=torch.float32).detach().cpu().numpy().tolist()
                )
                self.epoch_predictions[target_name].extend(
                    pred[target_name].argmax(dim=-1).detach().cpu().numpy().tolist()
                )
                self.epoch_running_loss[target_name].append(loss[target_name].item())
            self.epoch_running_loss["loss"].append(loss["loss"].item())
        else:
            self.epoch_ground_truth.extend(true.cpu().numpy().tolist())
            self.epoch_confidences.extend(pred.softmax(dim=-1, dtype=torch.float32).detach().cpu().numpy().tolist())
            self.epoch_predictions.extend(pred.argmax(dim=-1).detach().cpu().numpy().tolist())
            self.epoch_running_loss.append(loss.item())

    def get_epoch_results(self):
        return {
            "running_loss": self.epoch_running_loss,
            "confidences": self.epoch_confidences,
            "predictions": self.epoch_predictions,
            "ground_truth": self.epoch_ground_truth,
        }

    def log_epoch(self, epoch, train_results, val_results):
        if self.experiment is not None:  # log metrics
            log_images(self.experiment, "Train", epoch, train_results["images"])

            log_images(self.experiment, "Validation", epoch, val_results["images"])

            log_metrics(
                self.experiment,
                self.target_names,
                self.label_names,
                epoch,
                train_results["metrics"],
                "Train",
            )

            log_metrics(
                self.experiment,
                self.target_names,
                self.label_names,
                epoch,
                val_results["metrics"],
                "Validation",
            )

            log_confusion_matrices(
                self.experiment,
                self.target_names,
                self.label_names,
                epoch,
                val_results,
                "Validation",
            )

            if self.cfg.log_gradients:
                log_grads(self.experiment, epoch, train_results["metrics_grad_log"])
