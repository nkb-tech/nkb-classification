import argparse
import os
from collections import defaultdict
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from nkb_classification.dataset import get_dataset
from nkb_classification.losses import get_loss
from nkb_classification.metrics import (
    compute_metrics,
    log_confusion_matrices,
    log_metrics,
)
from nkb_classification.model import get_model
from nkb_classification.utils import (
    get_experiment,
    get_optimizer,
    get_scheduler,
    log_grads,
    log_images,
    read_py_config,
)


class TrainPbar(tqdm):
    def __init__(
            self,
            train_loader,
            leave,
            desc,
            cfg):
        super().__init__(train_loader, leave=leave, desc=desc)
        self.cfg = cfg

    def update_loss(self, loss):
        if (self.cfg.task == 'multi') \
            and self.cfg.show_full_current_loss_in_terminal:
            self.set_postfix_str(
                ", ".join(
                    f"loss {key}: {value:.4f}"
                    for key, value in loss.items()
                )
            )
        elif self.cfg.task == 'multi':
            self.set_postfix_str(f"Loss: {loss['loss'].item():.4f}")
        else:
            self.set_postfix_str(f"Loss: {loss.item():.4f}")

class TrainLogger:
    __slots__ = \
        'cfg', \
        'epoch_running_loss', \
        'epoch_confidences', \
        'epoch_predictions', \
        'epoch_ground_truth', \
        'target_names', \
        'label_names', \
        'experiment', \


    def __init__(self, cfg, experiment, class_to_idx):

        assert cfg.task in ('single', 'multi')

        self.cfg = cfg
        self.experiment = experiment

        if cfg.task == 'single':
            self.epoch_running_loss = []
            self.epoch_confidences = []
            self.epoch_predictions = []
            self.epoch_ground_truth = []

            self.target_names = None
            # import ipdb; ipdb.set_trace()
            self.label_names = list(class_to_idx.keys())

        elif cfg.task == 'multi':
            self.epoch_running_loss = defaultdict(list)
            self.epoch_confidences = defaultdict(list)
            self.epoch_predictions = defaultdict(list)
            self.epoch_ground_truth = defaultdict(list)

            self.target_names = [*sorted(class_to_idx)]

            self.label_names = {
                target_name: [*class_to_idx[target_name].keys()]
                for target_name in self.target_names
            }

    def log_iter(self, pred, true, loss):
        assert type(pred) == type(true)

        # Multi-task case
        if isinstance(pred, dict):
            assert pred.keys() == true.keys()
            for target_name in pred.keys():
                self.epoch_ground_truth[target_name].extend(
                    true[target_name].cpu().numpy().tolist()
                )
                self.epoch_confidences[target_name].extend(
                    pred[target_name]
                    .softmax(dim=-1, dtype=torch.float32)
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                self.epoch_predictions[target_name].extend(
                    pred[target_name]
                    .argmax(dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                self.epoch_running_loss[target_name].append(
                        loss[target_name].item()
                    )
            self.epoch_running_loss['loss'].append(
                        loss['loss'].item()
                    )
        else:
            self.epoch_ground_truth.extend(
                true.cpu().numpy().tolist()
            )
            self.epoch_confidences.extend(
                pred
                .softmax(dim=-1, dtype=torch.float32)
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
            self.epoch_predictions.extend(
                pred
                .argmax(dim=-1)
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
            self.epoch_running_loss.append(
                loss.item()
            )

    def get_epoch_results(self):
        return {
            "running_loss": self.epoch_running_loss,
            "confidences": self.epoch_confidences,
            "predictions": self.epoch_predictions,
            "ground_truth": self.epoch_ground_truth,
        }

    def log_epoch(self, epoch, train_results, val_results):
        epoch_val_acc = None
        if self.experiment is not None:  # log metrics
            log_images(
                self.experiment, "Train", epoch, train_results["images"]
            )

            log_images(
                self.experiment, "Validation", epoch, val_results["images"]
            )

            log_metrics(
                self.experiment,
                self.target_names,
                self.label_names,
                epoch,
                train_results['metrics'],
                "Train",
            )

            log_metrics(
                self.experiment,
                self.target_names,
                self.label_names,
                epoch,
                val_results['metrics'],
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
                log_grads(
                    self.experiment, epoch, train_results["metrics_grad_log"]
                )


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    criterion,
    # target_names,
    device,
    cfg,
    epoch_logger,
):
    # train_running_loss = defaultdict(list)
    # train_confidences = defaultdict(list)
    # train_predictions = defaultdict(list)
    # train_ground_truth = defaultdict(list)

    # epoch_logger = EpochLogger(cfg)

    model.train()

    if cfg.log_gradients:
        metrics_grad_log = defaultdict(list)

    # pbar = tqdm(train_loader, leave=False, desc="Training")
    pbar = TrainPbar(train_loader, leave=False, desc="Training", cfg=cfg)

    batch_to_log = None
    for img, target in pbar:
        # target = target.to(device)
        img = img.to(device)
        optimizer.zero_grad()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=cfg.enable_mixed_presicion,
        ):
            preds = model(img)
            # loss = 0

            # for target_name in target_names:
            #     target_loss = criterion(
            #         preds[target_name], target[target_name].to(device)
            #     )
            #     train_running_loss[target_name].append(
            #         target_loss.item()
            #     )
            #     loss += target_loss
            if isinstance(target, torch.Tensor):
                target = target.to(device)

            # import ipdb; ipdb.set_trace()
            loss = criterion(preds, target)

        # train_running_loss["loss"].append(loss_item)

        # loss_item = loss.item()

        # if cfg.show_full_current_loss_in_terminal:
        #     pbar.set_postfix_str(
        #         ", ".join(
        #             f"loss {key}: {value[-1]:.4f}"
        #             for key, value in train_running_loss.items()
        #         )
        #     )
        # else:
        #     pbar.set_postfix_str(f"Loss: {loss_item:.4f}")

        pbar.update_loss(loss)

        if isinstance(loss, dict):
            scaler.scale(loss['loss']).backward()
        else:
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_logger.log_iter(preds, target, loss)

        # for target_name in target_names:
        #     train_ground_truth[target_name].extend(
        #         target[target_name].cpu().numpy().tolist()
        #     )
        #     train_confidences[target_name].extend(
        #         preds[target_name]
        #         .softmax(dim=-1, dtype=torch.float32)
        #         .detach()
        #         .cpu()
        #         .numpy()
        #         .tolist()
        #     )
        #     train_predictions[target_name].extend(
        #         preds[target_name]
        #         .argmax(dim=-1)
        #         .detach()
        #         .cpu()
        #         .numpy()
        #         .tolist()
        #     )

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

    # results = {
    #     "running_loss": train_running_loss,
    #     "confidences": train_confidences,
    #     "predictions": train_predictions,
    #     "ground_truth": train_ground_truth,
    #     "images": batch_to_log,
    # }
    results = epoch_logger.get_epoch_results()
    results['images'] = batch_to_log

    if cfg.log_gradients:
        results["metrics_grad_log"] = metrics_grad_log

    return results


@torch.no_grad()
def val_epoch(
    model,
    val_loader,
    criterion,
    # target_names,
    device,
    cfg,
    epoch_logger,
    ):
    # val_confidences = defaultdict(list)
    # val_predictions = defaultdict(list)
    # val_ground_truth = defaultdict(list)
    # val_running_loss = defaultdict(list)

    # epoch_logger = EpochLogger(cfg)

    model.eval()

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

            # loss = 0
            # for target_name in target_names:
            #     target_loss = criterion(
            #         preds[target_name], target[target_name].to(device)
            #     ).item()
            #     val_running_loss[target_name].append(target_loss)
            #     loss += target_loss

        epoch_logger.log_iter(preds, target, loss)

        # val_running_loss["loss"].append(loss)

        # for target_name in target_names:
        #     val_ground_truth[target_name].extend(
        #         target[target_name].cpu().numpy().tolist()
        #     )
        #     val_confidences[target_name].extend(
        #         preds[target_name]
        #         .softmax(dim=-1, dtype=torch.float32)
        #         .cpu()
        #         .numpy()
        #         .tolist()
        #     )
        #     val_predictions[target_name].extend(
        #         preds[target_name].argmax(dim=-1).cpu().numpy().tolist()
        #     )

        if batch_to_log is None:
            batch_to_log = img.to("cpu")

    # results = {
    #     "running_loss": val_running_loss,
    #     "confidences": val_confidences,
    #     "predictions": val_predictions,
    #     "ground_truth": val_ground_truth,
    #     "images": batch_to_log,
    # }
    results = epoch_logger.get_epoch_results()
    results['images'] = batch_to_log

    return results


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    experiment,
    device,
    cfg,
):
    model_path = Path(cfg.model_path)
    model_path.mkdir(exist_ok=True, parents=True)
    n_epochs = cfg.n_epochs
    best_val_acc = 0
    class_to_idx = train_loader.dataset.class_to_idx
    train_logger = TrainLogger(
        cfg,
        experiment,
        class_to_idx
    )
    # target_names = [*sorted(class_to_idx)]

    # import ipdb; ipdb.set_trace()
    # label_names = {
    #     target_name: [*class_to_idx[target_name].keys()]
    #     for target_name in target_names
    # }

    scaler = GradScaler(enabled=cfg.enable_gradient_scaler)

    for epoch in tqdm(range(n_epochs), desc="Training epochs"):
        train_results = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            criterion,
            # target_names,
            device,
            cfg,
            train_logger,
        )

        val_results = val_epoch(
            model, val_loader, criterion, device, cfg, train_logger
        )

        train_results['metrics'] = compute_metrics(cfg, train_results)
        val_results['metrics'] = compute_metrics(cfg, val_results)
        epoch_val_acc = val_results['metrics']["epoch_acc"]
        train_logger.log_epoch(
            epoch,
            train_results,
            val_results,
        )
        # epoch_val_acc = None
        # if experiment is not None:  # log metrics
        #     log_images(
        #         experiment, "Train", epoch, train_results["images"]
        #     )

        #     log_images(
        #         experiment, "Validation", epoch, val_results["images"]
        #     )

        #     train_metrics = compute_metrics(train_results, target_names)

        #     log_metrics(
        #         experiment,
        #         target_names,
        #         label_names,
        #         epoch,
        #         train_metrics,
        #         "Train",
        #     )

        #     val_metrics = compute_metrics(val_results, target_names)
        #     epoch_val_acc = val_metrics["epoch_acc"]

        #     log_metrics(
        #         experiment,
        #         target_names,
        #         label_names,
        #         epoch,
        #         val_metrics,
        #         "Validation",
        #     )

        #     log_confusion_matrices(
        #         experiment,
        #         target_names,
        #         label_names,
        #         epoch,
        #         val_results,
        #         "Validation",
        #     )

        #     if cfg.log_gradients:
        #         log_grads(
        #             experiment, epoch, train_results["metrics_grad_log"]
        #         )

        # TODO save chkpt each n epoch
        if epoch_val_acc is not None:
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save(
                    model.state_dict(), Path(model_path, "best.pth")
                )
        torch.save(model.state_dict(), Path(model_path, "last.pth"))


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
    model.set_backbone_state("freeze")

    import ipdb; ipdb.set_trace()
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
    experiment = get_experiment(cfg.experiment)
    experiment.log_code(cfg_file)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    experiment.log_code(
        os.path.join(dir_path, "nkb_classification/model.py")
    )
    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        experiment,
        device,
        cfg,
    )


if __name__ == "__main__":
    main()
