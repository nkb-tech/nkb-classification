import sys
from pathlib import Path

from torch.optim import SGD, Adam, NAdam, RAdam, SparseAdam, lr_scheduler


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
        raise NotImplementedError(f'Unknown optimizer in config: {cfg["type"]}')
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
        scheduler = lr_scheduler.MultiStepLR(opt, milestones=lr_policy["steps"], gamma=lr_policy["gamma"])
    else:
        raise NotImplementedError("Learning rate policy {} not implemented.".format(lr_policy["type"]))
    return scheduler


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
    return pandas.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])
