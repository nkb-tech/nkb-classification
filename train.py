import argparse
import os
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from nkb_classification.engine import train_epoch, val_epoch
from nkb_classification.dataset import get_dataset
from nkb_classification.logging import TrainLogger, get_comet_experiment, get_local_experiment
from nkb_classification.losses import get_loss
from nkb_classification.metrics import compute_metrics
from nkb_classification.model import get_model
from nkb_classification.utils import get_optimizer, get_scheduler, read_py_config



def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    comet_experiment,
    local_experiment,
    device,
    cfg,
):
    model_path = local_experiment.path / "weights"
    n_epochs = cfg.n_epochs
    best_val_acc = 0
    classes = train_loader.dataset.classes
    train_logger = TrainLogger(cfg, comet_experiment, local_experiment, classes)
    train_logger.log_images_at_start(train_loader)
    scaler = GradScaler(enabled=cfg.enable_gradient_scaler)

    for epoch in tqdm(range(n_epochs), desc="Training epochs"):
        if epoch in cfg.backbone_state_policy.keys():
            model.set_backbone_state(cfg.backbone_state_policy[epoch])

        train_results = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            criterion,
            device,
            cfg,
            train_logger,
        )

        val_results = val_epoch(model, val_loader, criterion, device, cfg, train_logger)

        train_results["metrics"] = compute_metrics(cfg, train_results)
        val_results["metrics"] = compute_metrics(cfg, val_results)
        epoch_val_acc = val_results["metrics"]["epoch_acc"]
        train_logger.log_epoch(
            epoch,
            train_results,
            val_results,
        )
        # TODO save chkpt each n epoch
        scripted = torch.jit.script(model)
        if epoch_val_acc is not None:
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save(model.state_dict(), Path(model_path, "best.pth"))
                scripted.save(Path(model_path, "scripted_best.pt"))
        torch.save(model.state_dict(), Path(model_path, "last.pth"))
        scripted.save(Path(model_path, "scripted_last.pt"))
        """
        To load scripted model use: 
        model = torch.jit.load(path, map_location=device)
        """


def main():
    parser = argparse.ArgumentParser(description="Train arguments")
    parser.add_argument(
        "-cfg",
        "--config",
        help="Config file path",
        type=str,
        default="",
        required=True,
    )
    args = parser.parse_args()
    cfg_file = args.config
    exec(read_py_config(cfg_file), globals(), globals())
    train_loader = get_dataset(cfg.train_data, cfg.train_pipeline)
    classes = train_loader.dataset.classes
    if "classes" not in cfg.val_data.keys():
        cfg.val_data = {**cfg.val_data, "classes": classes}
    val_loader = get_dataset(cfg.val_data, cfg.val_pipeline)
    device = torch.device(cfg.device)
    model = get_model(cfg.model, classes, device, compile=cfg.compile)
    optimizer = get_optimizer(model, cfg_optimizer=cfg.optimizer)
    scheduler = get_scheduler(optimizer, cfg.lr_policy)
    criterion = get_loss(cfg.criterion, cfg.device)
    comet_experiment = get_comet_experiment(cfg.experiment["comet"])
    if comet_experiment is not None:
        comet_experiment.log_code(cfg_file)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        comet_experiment.log_code(os.path.join(dir_path, "nkb_classification/model.py"))
    local_experiment = get_local_experiment(cfg.experiment["local"])
    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        comet_experiment,
        local_experiment,
        device,
        cfg,
    )


if __name__ == "__main__":
    main()
