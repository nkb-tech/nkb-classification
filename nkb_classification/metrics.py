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
