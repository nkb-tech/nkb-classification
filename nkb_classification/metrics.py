import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import warnings


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
    confidences = np.array(confidences)
    n_classes = confidences.shape[1]
    gt_classes = np.unique(ground_truth)
    gt_n_classes = len(gt_classes)
    if gt_n_classes < n_classes:
        warnings.warn(
                "\n"
                "Number of classes in ground truth "
                "is less than number of classes in predicted confidences. \n"
                "Some of ROC AUC metric values will be NaN"
                "\n"
            )

    epoch_acc = balanced_accuracy_score(ground_truth, predictions)

    if n_classes > 2:
        epoch_roc_auc = np.full(n_classes, np.nan)
        if gt_n_classes > 1:
            ground_truth_bin = label_binarize(ground_truth, classes=range(n_classes))
            for gt_class in gt_classes:
                epoch_roc_auc[gt_class] = roc_auc_score(ground_truth_bin[:, gt_class], confidences[:, gt_class])
    else:
        epoch_roc_auc = np.nan
        if gt_n_classes > 1:
            epoch_roc_auc = roc_auc_score(ground_truth, confidences[:, 1])

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
