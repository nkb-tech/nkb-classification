import argparse
import os
from collections import defaultdict
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from nkb_classification.dataset import get_dataset
from nkb_classification.losses import get_loss
from nkb_classification.metrics import compute_metrics
from nkb_classification.model import get_model
from nkb_classification.utils import get_optimizer, get_scheduler, read_py_config
from nkb_classification.logging import TrainLogger, get_comet_experiment, get_local_experiment


class TrainPbar(tqdm):
    def __init__(self, train_loader, leave, desc, cfg):
        super().__init__(train_loader, leave=leave, desc=desc)
        self.cfg = cfg

    def update_loss(self, loss):
        if (self.cfg.task == "multi") and self.cfg.show_full_current_loss_in_terminal:
            self.set_postfix_str(", ".join(f"loss {key}: {value:.4f}" for key, value in loss.items()))
        elif self.cfg.task == "multi":
            self.set_postfix_str(f"Loss: {loss['loss'].item():.4f}")
        else:
            self.set_postfix_str(f"Loss: {loss.item():.4f}")


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    criterion,
    device,
    cfg,
    epoch_logger,
):

    model.train()
    epoch_logger.init_iter_logs()

    if cfg.log_gradients:
        metrics_grad_log = defaultdict(list)
    pbar = TrainPbar(train_loader, leave=False, desc="Training", cfg=cfg)

    batch_to_log = None
    for img, target in pbar:
        img = img.to(device)
        optimizer.zero_grad()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=cfg.enable_mixed_presicion,
        ):
            preds = model(img)
            if isinstance(target, torch.Tensor):
                target = target.to(device)
            loss = criterion(preds, target)

        pbar.update_loss(loss)

        if isinstance(loss, dict):
            scaler.scale(loss["loss"]).backward()
        else:
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_logger.log_iter(preds, target, loss)

        if cfg.log_gradients:
            total_grad = 0
            for tag, value in model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_grad_log[f"Gradients/{tag}"].append(grad)
                    total_grad += grad

            metrics_grad_log["Gradients/Total"].append(total_grad)

        if batch_to_log is None:
            batch_to_log = img.to("cpu")

    if scheduler is not None:
        scheduler.step()

    results = epoch_logger.get_epoch_results()
    results["images"] = batch_to_log

    if cfg.log_gradients:
        results["metrics_grad_log"] = metrics_grad_log

    return results


@torch.no_grad()
def val_epoch(
    model,
    val_loader,
    criterion,
    device,
    cfg,
    epoch_logger,
):
    model.eval()
    epoch_logger.init_iter_logs()

    batch_to_log = None
    for img, target in tqdm(val_loader, leave=False, desc="Evaluating"):
        img = img.to(device)
        # target = target.to(device)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=cfg.enable_mixed_presicion,
        ):
            preds = model(img)
            if isinstance(target, torch.Tensor):
                target = target.to(device)
            loss = criterion(preds, target)

        epoch_logger.log_iter(preds, target, loss)

        if batch_to_log is None:
            batch_to_log = img.to("cpu")

    results = epoch_logger.get_epoch_results()
    results["images"] = batch_to_log

    return results


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
    class_to_idx = train_loader.dataset.class_to_idx
    train_logger = TrainLogger(cfg, comet_experiment, local_experiment, class_to_idx)
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
    val_loader = get_dataset(cfg.val_data, cfg.val_pipeline)
    classes = train_loader.dataset.classes
    device = torch.device(cfg.device)
    model = get_model(cfg.model, classes, device, compile=cfg.compile)

    optimizer = get_optimizer(
        parameters=[
            {
                "params": model.emb_model.parameters(),
                "lr": cfg.optimizer["backbone_lr"],
            },
            {
                "params": model.classifier.parameters(),
                "lr": cfg.optimizer["classifier_lr"],
            },
        ],
        cfg=cfg.optimizer,
    )
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
