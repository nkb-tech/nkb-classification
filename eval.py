import argparse
import json
from pathlib import Path

import torch

from nkb_classification.engine import val_epoch
from nkb_classification.dataset import get_dataset
from nkb_classification.logging import BaseLogger
from nkb_classification.losses import get_loss
from nkb_classification.metrics import compute_metrics
from nkb_classification.model import get_model
from nkb_classification.utils import read_py_config, convert_dict_types_recursive


def evaluate(model, val_loader, criterion, device, cfg):
    
    class_to_idx = val_loader.dataset.class_to_idx
    val_logger = BaseLogger(cfg, class_to_idx)

    val_results = val_epoch(model, val_loader, criterion, device, cfg, val_logger)
    val_metrics = compute_metrics(cfg, val_results)

    return val_metrics


def main():
    parser = argparse.ArgumentParser(description="Train arguments")
    parser.add_argument("-cfg", "--config", help="Config file path", type=str, default="", required=True)
    args = parser.parse_args()
    cfg_file = args.config
    exec(read_py_config(cfg_file), globals(), globals())
    val_loader = get_dataset(cfg.val_data, cfg.val_pipeline)
    classes = val_loader.dataset.classes
    device = torch.device(cfg.device)
    model = get_model(cfg.model, classes, device=device, compile=cfg.compile, **cfg.model)
    criterion = get_loss(cfg.criterion, device)
    metrics = evaluate(model, val_loader, criterion, device, cfg)

    save_path = Path(cfg.save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    with open(save_path / Path("metrics.json"), "w") as f:
        json.dump(convert_dict_types_recursive(metrics), f)


if __name__ == "__main__":
    main()
