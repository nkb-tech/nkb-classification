from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
import json
from comet_ml import Experiment as CometExperiment
from torchvision import transforms
from torchvision.utils import make_grid

from nkb_classification.utils import save_classes, sort_df_columns_titled


class LocalExperiment:
    def __init__(self, path=""):
        self.path = Path(path)
        self.metrics = pd.DataFrame([], columns=["Epoch"])

    def log_image(self, image, name="", step=0):
        plt.imsave(self.path / f"{name}_{step}.png", np.array(image))

    def log_metric(self, name, value, epoch=0, step=None, prefix=None):
        if prefix is not None:
            name = f"{prefix}/{name}"
        if isinstance(value, Sequence):  # take mean of epoch history
            value = np.mean(value)
        self.metrics.loc[epoch, name] = value  # only epochs are supported as step numbers
        self.metrics.loc[:, "Epoch"] = range(len(self.metrics))
        self.metrics = sort_df_columns_titled(self.metrics)  # sort metric namse aplhabetically
        self.metrics.to_csv(self.path / "metrics.csv", index=False, sep="\t")

    def log_metrics(self, metrics_dict, epoch=0, step=None, prefix=None):
        for name, value in metrics_dict.items():
            self.log_metric(name, value, epoch=epoch, prefix=prefix)


def get_comet_experiment(cfg_exp):
    if cfg_exp is None:
        return None
    api_cfg_path = cfg_exp.pop("comet_api_cfg_path")
    with open(api_cfg_path, "r") as api_cfg_file:
        comet_cfg = yaml.safe_load(api_cfg_file)
        cfg_exp["api_key"] = comet_cfg["api_key"]
        cfg_exp["workspace"] = comet_cfg["workspace"]
        cfg_exp["project_name"] = comet_cfg["project_name"]
    name = cfg_exp.pop("name")
    exp = CometExperiment(**cfg_exp)
    exp.set_name(name)
    return exp


def get_local_experiment(cfg_exp):
    assert cfg_exp is not None and "path" in cfg_exp.keys()
    exp_path = Path(cfg_exp["path"])
    dir_duplicate_num = 1
    while exp_path.exists():  # update local logging path if already exists
        exp_path = Path(cfg_exp["path"] + str(dir_duplicate_num))
        dir_duplicate_num += 1
    exp_path.mkdir(parents=True)
    (exp_path / "weights").mkdir()
    exp = LocalExperiment(exp_path)
    return exp


def log_targetwise_metrics(experiment, target_name, classes, epoch, metrics, fold="train"):
    if target_name is None:
        target_name = ""
    acc = metrics["epoch_acc"]
    roc_auc = metrics["epoch_roc_auc"]
    epoch_loss = metrics["epoch_loss"]
    n_classes = len(classes)
    # print(f'{target_name} Epoch {epoch} {fold.lower()} roc_auc {roc_auc:.4f}')
    # print(f'{target_name} Epoch {epoch} {fold.lower()} balanced accuracy {acc:.4f}')
    experiment.log_metric(
        f"{target_name} Average epoch {fold} loss".lstrip(),
        epoch_loss,
        epoch=epoch,
        step=epoch,
    )
    if n_classes > 2:
        for roc_auc_, class_name in zip(roc_auc, classes):
            experiment.log_metric(
                f"{target_name} {fold} ROC AUC, {class_name}".lstrip(),
                roc_auc_,
                epoch=epoch,
                step=epoch,
            )
        mean_roc_auc = np.nan if np.all(np.isnan(roc_auc)) else np.nanmean(roc_auc)
        experiment.log_metric(
            f"{target_name} {fold} ROC AUC".lstrip(),
            mean_roc_auc,
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
    classes,
    epoch,
    metrics,
    fold="train",
):
    if target_names is None:
        log_targetwise_metrics(
            experiment,
            None,
            classes,
            epoch,
            metrics,
            fold,
        )

    else:
        for target_name in target_names:
            log_targetwise_metrics(
                experiment,
                target_name,
                classes[target_name],
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
    classes,
    epoch,
    results,
    fold="validation",
    show_all=False
):
    default_max_categories = 25
    if target_names is None:
        max_categories = len(classes) if show_all else default_max_categories
        experiment.log_confusion_matrix(
            results["ground_truth"],
            results["predictions"],
            labels=tuple(map(str, classes)),
            max_categories=len(classes),
            title=f"{fold} confusion matrix",
            file_name=f"{fold}-confusion-matrix.json",
            epoch=epoch,
        )
    else:
        for target_name in target_names:
            max_categories = len(classes[target_name]) if show_all else default_max_categories
            experiment.log_confusion_matrix(
                results["ground_truth"][target_name],
                results["predictions"][target_name],
                labels=tuple(map(str, classes[target_name])),
                max_categories=max_categories,
                title=f"{fold} {target_name} confusion matrix",
                file_name=f"{fold}-{target_name}-confusion-matrix.json",
                epoch=epoch,
            )


def log_images(experiment, name, epoch, batch_to_log):
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


class BaseLogger:
    __slots__ = (
        "cfg",
        "epoch_running_loss",
        "epoch_confidences",
        "epoch_predictions",
        "epoch_ground_truth",
        "epoch_images_example",
        "target_names",
        "classes",
        "task",
    )

    def __init__(self, cfg, classes):

        assert cfg.task in ("single", "multi")

        self.cfg = cfg
        self.task = cfg.task
        self.classes = classes

        if self.task == "single":
            self.target_names = None

        elif self.task == "multi":
            self.target_names = sorted(self.class_to_idx)

    def init_iter_logs(self):

        self.epoch_images_example = None

        if self.task == "single":
            self.epoch_running_loss = []
            self.epoch_confidences = []
            self.epoch_predictions = []
            self.epoch_ground_truth = []

        elif self.task == "multi":
            self.epoch_running_loss = defaultdict(list)
            self.epoch_confidences = defaultdict(list)
            self.epoch_predictions = defaultdict(list)
            self.epoch_ground_truth = defaultdict(list)
    
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

    def log_images_if_needed(self, images):
        if self.epoch_images_example is None:
            self.epoch_images_example = images.to("cpu")

    def get_epoch_results(self):
        return {
            "running_loss": self.epoch_running_loss,
            "confidences": self.epoch_confidences,
            "predictions": self.epoch_predictions,
            "ground_truth": self.epoch_ground_truth,
            "images": self.epoch_images_example,
        }


class TrainLogger(BaseLogger):
    __slots__ = (
        "cfg",
        "epoch_running_loss",
        "epoch_confidences",
        "epoch_predictions",
        "epoch_ground_truth",
        "epoch_images_example",
        "target_names",
        "classes",
        "task",
        "comet_experiment",
        "local_experiment",
        "show_full_conf_matrix"
    )

    def __init__(self, cfg, comet_experiment, local_experiment, classes):

        super().__init__(cfg, classes)

        self.comet_experiment = comet_experiment
        self.local_experiment = local_experiment

        if hasattr(cfg, "show_all_classes_in_confusion_matrix"):
            self.show_full_conf_matrix = cfg.show_all_classes_in_confusion_matrix

        save_classes(self.classes, self.local_experiment.path / "classes.json")

    def log_images_at_start(self, loader, n_batches=3):

        for batch_num, (img_batch, _) in enumerate(loader):
            if batch_num + 1 > n_batches:
                break
            log_images(
                experiment=self.local_experiment, name=f"train_batch", epoch=batch_num + 1, batch_to_log=img_batch
            )

    def log_epoch(self, epoch, train_results, val_results):

        # log metrics locally
        log_metrics(
            self.local_experiment, self.target_names, self.classes, epoch, train_results["metrics"], "train"
        )

        log_metrics(self.local_experiment, self.target_names, self.classes, epoch, val_results["metrics"], "Val")

        if self.comet_experiment is not None:  # log metrics to comet
            log_images(self.comet_experiment, "train", epoch, train_results["images"])

            log_images(self.comet_experiment, "validation", epoch, val_results["images"])

            log_metrics(
                self.comet_experiment,
                self.target_names,
                self.classes,
                epoch,
                train_results["metrics"],
                "train",
            )

            log_metrics(
                self.comet_experiment,
                self.target_names,
                self.classes,
                epoch,
                val_results["metrics"],
                "validation",
            )

            log_confusion_matrices(
                self.comet_experiment,
                self.target_names,
                self.classes,
                epoch,
                val_results,
                "validation",
                self.show_full_conf_matrix
            )

            if self.cfg.log_gradients:
                log_grads(self.comet_experiment, epoch, train_results["metrics_grad_log"])
