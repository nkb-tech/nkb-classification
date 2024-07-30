import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def compute_targetwise_metrics(epoch_results, target_name=None):
    if target_name is None:
        running_loss = epoch_results["running_loss"]
        confidences = epoch_results["confidences"]
        predictions = epoch_results["predictions"]
        ground_truth = epoch_results["ground_truth"]
    else:
        running_loss = epoch_results["running_loss"][target_name]
        confidences = epoch_results["confidences"][target_name]
        predictions = epoch_results["predictions"][target_name]
        ground_truth = epoch_results["ground_truth"][target_name]
    n_classes = len(confidences[0])

    epoch_acc = balanced_accuracy_score(ground_truth, predictions)
    if n_classes > 2:
        epoch_roc_auc = roc_auc_score(ground_truth, confidences, average=None, multi_class="ovr")
    else:
        epoch_roc_auc = roc_auc_score(ground_truth, np.array(confidences)[:, 1])
    epoch_loss = np.mean(running_loss)
    metrics = {
        "epoch_acc": epoch_acc,
        "epoch_roc_auc": epoch_roc_auc,
        "epoch_loss": epoch_loss,
    }
    return metrics


def compute_metrics(
    cfg: dict,
    epoch_results: dict,
):
    if cfg.task == "single":
        metrics = compute_targetwise_metrics(epoch_results)

        metrics["loss"] = epoch_results["running_loss"]
        return metrics
    elif cfg.task == "multi":
        target_names = cfg.target_names
        metrics = {target_name: compute_targetwise_metrics(epoch_results, target_name) for target_name in target_names}
        metrics["loss"] = epoch_results["running_loss"]["loss"]
        metrics["epoch_acc"] = np.mean([metrics[target_name]["epoch_acc"] for target_name in target_names])
        return metrics
    else:
        raise ValueError(f"Unknown task type {cfg.task} for metric computation")


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
