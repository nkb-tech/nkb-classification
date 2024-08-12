import sys
from pathlib import Path

import numpy as np
import pandas as pd
import json
from torch.optim import SGD, Adam, NAdam, RAdam, SparseAdam, lr_scheduler


def get_optimizer(parameters, cfg):
    match cfg["type"]:
        case "adam":
            opt = Adam(
                parameters,
                lr=cfg["lr"],
                weight_decay=cfg.get("weight_decay", 0.0),
            )
        case "radam":
            opt = RAdam(
                parameters,
                lr=cfg["lr"],
                weight_decay=cfg.get("weight_decay", 0.0),
            )
        case "nadam":
            opt = NAdam(
                parameters,
                lr=cfg["lr"],
                weight_decay=cfg.get("weight_decay", 0.0),
                decoupled_weight_decay=True,
            )
        case "sparse_adam":
            opt = SparseAdam(
                parameters,
                lr=cfg["lr"],
                weight_decay=cfg.get("weight_decay", 0.0),
            )
        case "sgd":
            opt = SGD(
                parameters,
                lr=cfg["lr"],
                weight_decay=cfg.get("weight_decay", 0.0),
            )
        case _:
            raise NotImplementedError(f'Unknown optimizer in config: {cfg["type"]}')
    return opt


def get_scheduler(opt, lr_policy):
    if len(lr_policy) == 0:
        return None
    match lr_policy["type"]:
        case "step":
            scheduler = lr_scheduler.StepLR(
                opt,
                step_size=lr_policy["step_size"],
                gamma=lr_policy["gamma"],
            )
        case "multistep":
            scheduler = lr_scheduler.MultiStepLR(opt, milestones=lr_policy["steps"], gamma=lr_policy["gamma"])
        case "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=lr_policy["n_epochs"])
        case _:
            raise NotImplementedError("Learning rate policy {} not implemented.".format(lr_policy["type"]))
    return scheduler


def save_classes(classes, save_path):
    if isinstance(classes, (list, dict)):
        with open(save_path, "w") as f:
            json.dump(classes, f)
    else:
        raise NotImplementedError(f"unknown classes config type {type(classes)}")


def load_classes(classes):
    if isinstance(classes, (list, dict)):
        return classes
    elif isinstance(classes, (str, Path)):
        with open(classes, "r") as f:
            return json.load(f)
    else:
        raise NotImplementedError(f"unknown classes config type {type(classes)}")


def get_classes_configs(classes):
    if isinstance(classes, list):
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        return class_to_idx, idx_to_class
    elif isinstance(classes, dict):
        class_to_idx = {
            target_name: {cls: idx for idx, cls in enumerate(classes[target_name])}
            for target_name in classes.keys()
        }
        idx_to_class = {
            target_name: {idx: cls for cls, idx in class_to_idx[target_name].items()}
            for target_name in classes.keys()
        }
        return class_to_idx, idx_to_class
    else:
        raise NotImplementedError(f"unknown classes config type {type(classes)}")


def read_py_config(path):
    path = Path(path)
    sys.path.append(str(path.parent))
    line = f"import {path.stem} as cfg"
    return line


def sort_df_columns_titled(df):
    first_column = df.iloc[:, 0]
    other_columns_sorted = df.iloc[:, 1:].reindex(sorted(df.columns[1:]), axis=1)
    df_sorted = pd.concat([first_column, other_columns_sorted], axis=1)
    return df_sorted


def convert_dict_types_recursive(_dict):
    for key in _dict:
        if isinstance(_dict[key], dict):
            _dict[key] = convert_dict_types_recursive(_dict[key])
        elif isinstance(_dict[key], np.ndarray):
            _dict[key] = list(_dict[key])
    return _dict


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
    return pandas.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])
