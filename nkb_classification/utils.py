import sys
from collections import defaultdict
from pathlib import Path
import yaml

import torch
from comet_ml import Experiment
from torch.optim import (
    SGD,
    Adam,
    NAdam,
    RAdam,
    SparseAdam,
    lr_scheduler,
)
from torchvision import transforms
from torchvision.utils import make_grid


def get_experiment(cfg_exp):
    if cfg_exp is None:
        return None
    api_cfg_path = cfg_exp.pop("comet_api_cfg_path")
    with open(api_cfg_path, "r") as api_cfg_file:
        comet_cfg = yaml.safe_load(api_cfg_file)
        cfg_exp["api_key"] = comet_cfg["api_key"]
        cfg_exp["workspace"] = comet_cfg["workspace"]
        cfg_exp["project_name"] = comet_cfg["project_name"]
    name = cfg_exp.pop("name")
    exp = Experiment(**cfg_exp)
    exp.set_name(name)
    return exp


def get_optimizer(parameters, cfg):
    if cfg["type"] == "adam":
        opt = Adam(
            parameters,
            lr=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 0.0),
        )
    elif cfg["type"] == "radam":
        opt = RAdam(
            parameters,
            lr=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 0.0),
        )
    elif cfg["type"] == "nadam":
        opt = NAdam(
            parameters,
            lr=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 0.0),
            decoupled_weight_decay=True,
        )
    elif cfg["type"] == "sparse_adam":
        opt = SparseAdam(
            parameters,
            lr=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 0.0),
        )
    elif cfg["type"] == "sgd":
        opt = SGD(
            parameters,
            lr=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 0.0),
        )
    else:
        raise NotImplementedError(
            f'Unknown optimizer in config: {cfg["type"]}'
        )
    return opt


def get_scheduler(opt, lr_policy):
    if len(lr_policy) == 0:
        return None
    if lr_policy["type"] == "step":
        scheduler = lr_scheduler.StepLR(
            opt,
            step_size=lr_policy["step_size"],
            gamma=lr_policy["gamma"],
        )
    elif lr_policy["type"] == "multistep":
        scheduler = lr_scheduler.MultiStepLR(
            opt, milestones=lr_policy["steps"], gamma=lr_policy["gamma"]
        )
    else:
        raise NotImplementedError(
            "Learning rate policy {} not implemented.".format(
                lr_policy["type"]
            )
        )
    return scheduler


def log_images(experiment, name, epoch, batch_to_log):
    inv_transform = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
            transforms.Normalize(
                mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
            ),
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


def read_py_config(path):
    path = Path(path)
    sys.path.append(str(path.parent))
    line = f"import {path.stem} as cfg"
    return line


def export_formats():
    """YOLOv8 export formats.
    Taken from https://github.com/ultralytics/ultralytics/blob/70c400ee158fc52361e6d38e4b93f55fff21edd7/ultralytics/engine/exporter.py#L79
    """

    import pandas

    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["TensorRT", "engine", ".engine", False, True],
    ]
    return pandas.DataFrame(
        x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"]
    )
