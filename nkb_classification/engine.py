from collections import defaultdict
import torch
from tqdm import tqdm


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
