import argparse
import colorsys
import hashlib
import logging
from collections import defaultdict
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import auc
from torchvision.ops import box_iou
from tqdm import tqdm
from ultralytics import YOLO

from nkb_classification.dataset import Transforms

# Установим базовое логирование
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==============================================================================================
# Скрипт для обработки изображений и валидации результатов детектора и детектора+классификатора.
# ==============================================================================================


def img2label_path(img_path: Path):
    """
    Yolo dataset convention
    dir:
    ├── images/
    │   ├── image1.jpg
    │   └── ...
    └── labels/
        ├── image1.txt
        └── ...
    """
    return img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name


def generate_color(parameter_name):
    # Get a hash from string
    hash_object = hashlib.md5(parameter_name.encode())
    # Convert hash to integer
    hash_int = int(hash_object.hexdigest(), 16)
    # Use hash to generate hue
    hue = (hash_int % 360) / 360.0  # Normalize to [0, 1]
    # Set high saturation and neutral lightness for better contrast
    saturation = 0.9
    lightness = 0.5
    # Convert HSL to RGB
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    # Convert RGB to hex
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


class Evaluator:
    def __init__(
        self,
        detection_model_path: str,
        is_detector_single_class: bool,
        classification_model_path: str,
        dataset_cfg: dict,
        min_det_conf_threshold: float,
        nms_iou: float,
        match_iou: float,
        device: torch.device,
        cls_inf_size: int,
        pad: bool,
    ):
        self.detection_model_path = detection_model_path
        self.is_detector_single_class = is_detector_single_class
        self.classification_model_path = classification_model_path
        self.dataset_cfg = dataset_cfg
        self.min_det_conf_threshold = min_det_conf_threshold
        self.nms_iou = nms_iou
        self.match_iou = match_iou
        self.device = device
        self.detector, self.classifier = self.load_models()
        self.all_images, self.all_labels = self.process_yolo_dataset_cfg()
        if self.classifier is not None:
            resizing_method = (
                A.Compose(
                    [
                        A.LongestMaxSize(cls_inf_size, always_apply=True),
                        A.PadIfNeeded(
                            cls_inf_size, cls_inf_size, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=0
                        ),
                    ]
                )
                if pad
                else A.Resize(cls_inf_size, cls_inf_size)
            )
            self.classifier_preprocess = Transforms(
                A.Compose(
                    [
                        resizing_method,
                        A.Normalize(
                            mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225),
                        ),
                        ToTensorV2(),
                    ]
                )
            )

    def load_models(self):
        """
        Загружает модели детекции и классификации.

        Args:
            weights_detector (str): Путь к весам модели детекции YOLO.
            weights_classifier (str): Путь к весам модели классификации.
            device (torch.device): Устройство, на котором выполняются вычисления (CPU/GPU).

        Returns:
            YOLO: Загруженная модель YOLO.
            torch.nn.Module: Загруженная модель классификации.
        """
        detection_model = YOLO(self.detection_model_path)
        if self.classification_model_path is not None:
            classifier_model = torch.jit.load(self.classification_model_path, map_location=self.device)
            classifier_model.eval()
        else:
            classifier_model = None
        return detection_model, classifier_model

    def process_yolo_dataset_cfg(self):
        """
        Обрабатывает все файлы валидации, выполняя детекцию и классификацию объектов.

        Args:
            val_paths (list): Список файлов или путей датасетов валидации.
            base_path (Path): Путь к директории с изображениями.
            detection_model (YOLO): Модель детекции YOLO.
            classifier_model (torch.nn.Module): Модель классификации.
            inference_pipeline (callable): Пайплайн для предобработки изображений.
            device (torch.device): Устройство, на котором выполняются вычисления (CPU/GPU).
            iou_threshold (float): Порог IoU для сопоставления предсказанных и истинных боксов.
            nms_iou_threshold (float): Порог IoU для NMS в модели YOLO.
            conf_threshold (float): Порог confidence для фильтрации предсказаний.

        Returns:
            list: Список результатов обработки всех изображений.
        """
        self.base_path = Path(self.dataset_cfg["path"])
        self.val_paths = self.dataset_cfg["val"]
        self.nc = self.dataset_cfg["nc"]
        self.names = self.dataset_cfg["names"]
        if type(self.names) is list:
            self.names = {i: nm for i, nm in enumerate(self.names)}
        all_images = []
        all_labels = []
        for val_path in tqdm(self.val_paths, desc="Processing validation folds"):
            p = self.base_path / val_path
            if p.is_file() and p.suffix == ".txt":
                with open(p, "r") as f:
                    image_paths = f.readlines()
                all_images.extend(image_paths)
                all_labels.extend([img2label_path(i) for i in image_paths])
            elif p.is_dir() and p.name == "images":
                image_paths = list(p.iterdir())
                all_images.extend(image_paths)
                all_labels.extend([img2label_path(i) for i in image_paths])
            elif p.is_dir() and (p / "images").exists():
                image_paths = list((p / "images").iterdir())
                all_images.extend(image_paths)
                all_labels.extend([img2label_path(i) for i in image_paths])
            else:
                raise ValueError(f"Unsupported yolo dataset path: {p}")

        return all_images, all_labels

    @torch.no_grad
    def process_image(self, image_path, label_path):
        """
        Обрабатывает одно изображение, выполняя детекцию и классификацию объектов.

        Args:
            image_path (str): Путь к изображению.
            label_path (str): Путь к файлу с метками.
            detection_model (YOLO): Модель детекции YOLO.
            classifier_model (torch.nn.Module): Модель классификации.
            inference_pipeline (callable): Пайплайн для предобработки изображений.
            device (torch.device): Устройство, на котором выполняются вычисления (CPU/GPU).
            iou_threshold (float): Порог IoU для сопоставления предсказанных и истинных боксов.
            nms_iou_threshold (float): Порог IoU для NMS в модели YOLO.
            conf_threshold (float): минимальный порог confidence для фильтрации предсказаний.

        Returns:
            list: Список результатов обработки изображения.
        """
        results = {}
        try:
            yolo_results = self.detector(
                image_path, verbose=False, conf=self.min_det_conf_threshold, device=self.device, iou=self.nms_iou
            )
            detector_preds = yolo_results[0].boxes.data.to("cpu").numpy()
            # Normalize
            detector_preds[:, :4] = yolo_results[0].boxes.xyxyn.to("cpu").numpy()

            gt_boxes = []
            with open(label_path, "r") as lf:
                for line in lf:
                    class_label, x_center, y_center, width, height = map(float, line.split())
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    gt_boxes.append([x1, y1, x2, y2, class_label])
            if len(gt_boxes) > 0:
                gt_boxes = np.stack(gt_boxes)
        except Exception as e:
            logging.error(f"Ошибка обработки {label_path}: {e}")
            return results
        if self.classifier is not None:
            classifier_preds = self.classify_crops(image_path, detector_preds[:, :4])
        else:
            classifier_preds = None

        results = {"detector_preds": detector_preds, "classifier_preds": classifier_preds, "gt_boxes": gt_boxes}
        return results

    def classify_crops(self, image_path, boxes):
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img.shape
        boxes = boxes.copy()
        boxes[:, [0, 2]] *= img_w
        boxes[:, [1, 3]] *= img_h
        crops = []
        for x1, y1, x2, y2 in boxes.astype(int):
            crop = img[y1:y2, x1:x2]
            crop_tensor = self.classifier_preprocess(crop).to(self.device)
            crops.append(crop_tensor)
        if len(crops) == 0:
            return np.empty(0)
        crops = torch.stack(crops)
        predicted_classes = self.classifier(crops).argmax(dim=1).to("cpu").numpy()
        return predicted_classes

    def eval(self, output_folder):
        predictions_df = defaultdict(list)
        gt_df = defaultdict(list)
        for img_path, lb_path in tqdm(zip(self.all_images, self.all_labels), desc="Inference"):
            results = self.process_image(img_path, lb_path)
            for det_pred in results["detector_preds"]:
                predictions_df["image_path"].append(str(img_path))
                predictions_df["xmin"].append(det_pred[0])
                predictions_df["ymin"].append(det_pred[1])
                predictions_df["xmax"].append(det_pred[2])
                predictions_df["ymax"].append(det_pred[3])
                predictions_df["conf"].append(det_pred[-2])
                predictions_df["detection_label"].append(int(det_pred[-1]))
            if self.classifier is not None:
                predictions_df["classifier_label"].extend(results["classifier_preds"].astype(int).tolist())
            for gt_item in results["gt_boxes"]:
                gt_df["image_path"].append(str(img_path))
                gt_df["xmin"].append(gt_item[0])
                gt_df["ymin"].append(gt_item[1])
                gt_df["xmax"].append(gt_item[2])
                gt_df["ymax"].append(gt_item[3])
                gt_df["label"].append(int(gt_item[4]))
        predictions_df = pd.DataFrame(predictions_df)
        gt_df = pd.DataFrame(gt_df)
        try:
            predictions_df.to_csv(output_folder / "predictions.csv", index=False)
            gt_df.to_csv(output_folder / "gt.csv", index=False)
            logging.info(f"Successfully saved predictions to {output_folder}")
        except Exception as e:
            logging.error(f"Error while saving predictions to {output_folder}: {e}")
        logging.info("Calculating metrics")
        # Match boxes
        matched_dets = []
        unmatched_dets = []
        unmatched_gts = []
        matched_gts = []
        for img_path in tqdm(gt_df["image_path"].unique(), desc="Matching boxes"):
            gt_img_df = gt_df[gt_df["image_path"] == img_path]
            pred_img_df = predictions_df[predictions_df["image_path"] == img_path]
            gt_img_info = gt_img_df[["xmin", "ymin", "xmax", "ymax", "label"]].values
            if self.classifier is not None:
                pred_img_info = pred_img_df[
                    ["xmin", "ymin", "xmax", "ymax", "conf", "detection_label", "classifier_label"]
                ].values
            else:
                pred_img_info = pred_img_df[["xmin", "ymin", "xmax", "ymax", "conf", "detection_label"]].values
            gt_boxes = torch.tensor(gt_img_info[:, :4])
            pred_boxes = torch.tensor(pred_img_info[:, :4])
            iou = box_iou(pred_boxes, gt_boxes)
            gt_idxs = []
            for i, iou_pred in enumerate(iou):
                gt_idx = torch.argmax(iou_pred)
                if iou_pred[gt_idx] > self.match_iou:
                    matched_dets.append(np.concatenate([pred_img_info[i], gt_img_info[gt_idx]]))
                    matched_gts.append(gt_img_info[gt_idx])
                    gt_idxs.append(gt_idx)
                else:
                    unmatched_dets.append(pred_img_info[i])
            for j, gt_box in enumerate(gt_img_info):
                if j not in gt_idxs:
                    unmatched_gts.append(gt_box)
        matched_dets = np.stack(matched_dets) if len(matched_dets) else np.empty((0, 12))
        unmatched_dets = np.stack(unmatched_dets) if len(unmatched_dets) else np.empty((0, 7))
        unmatched_gts = np.stack(unmatched_gts) if len(unmatched_gts) else np.empty((0, 5))
        matched_gts = np.stack(matched_gts) if len(matched_gts) else np.empty((0, 5))
        thresholds = np.linspace(start=self.min_det_conf_threshold, stop=0.95, num=40, endpoint=True)
        metrics = defaultdict(list)
        skip_labels = []
        fp_label = self.nc
        for thr in thresholds:
            # 1. Compute single class detector precision, recall and mAP_match_iou
            matched_thr_filtered = matched_dets[matched_dets[:, 4] > thr]
            unmatched_thr_filtered = unmatched_dets[unmatched_dets[:, 4] > thr]
            metrics["det_pr"].append(
                len(matched_thr_filtered) / (len(matched_thr_filtered) + len(unmatched_thr_filtered) + 1e-6)
            )
            metrics["det_recall"].append(len(matched_thr_filtered) / (len(matched_gts) + len(unmatched_gts) + 1e-6))
            metrics["threshold"].append(thr)
            if self.classifier is not None:
                unmatched_thr_cls_filtered = unmatched_thr_filtered[unmatched_thr_filtered[:, 6] != fp_label]
                matched_thr_cls_filtered = matched_thr_filtered[matched_thr_filtered[:, 6] != fp_label]
                metrics["det_cls_pr"].append(
                    len(matched_thr_cls_filtered)
                    / (len(unmatched_thr_cls_filtered) + len(matched_thr_cls_filtered) + 1e-6)
                )
                metrics["det_cls_recall"].append(
                    len(matched_thr_cls_filtered) / (len(matched_gts) + len(unmatched_gts) + 1e-6)
                )
                lb_cls_matched_thr_filtered = matched_thr_cls_filtered[
                    matched_thr_cls_filtered[:, 6] == matched_thr_cls_filtered[:, 11]
                ]
                lb_cls_unmatched_thr_filtered = matched_thr_cls_filtered[
                    matched_thr_cls_filtered[:, 6] != matched_thr_cls_filtered[:, 11]
                ]
            if not self.is_detector_single_class:
                gt_lb_idx = 10 if self.classifier is None else 11
                lb_det_matched_thr_filtered = matched_thr_filtered[
                    matched_thr_filtered[:, 5] == matched_thr_filtered[:, gt_lb_idx]
                ]
                lb_det_unmatched_thr_filtered = matched_thr_filtered[
                    matched_thr_filtered[:, 5] != matched_thr_filtered[:, gt_lb_idx]
                ]
            # 2. Compute multiclass metrics
            for i, (_, label) in enumerate(self.names.items()):
                # Skip non existing classes
                if len(gt_df[gt_df["label"] == i]) == 0:
                    skip_labels.append(label)
                    continue
                if not self.is_detector_single_class:
                    tp_det = sum(lb_det_matched_thr_filtered[:, 5] == i)
                    fp_det = sum(
                        np.concatenate([lb_det_unmatched_thr_filtered[:, 5] == i, unmatched_thr_filtered[:, 5] == i])
                    )
                    metrics[f"{label}_det_pr"].append(tp_det / (tp_det + fp_det + 1e-6))
                    metrics[f"{label}_det_recall"].append(
                        tp_det / (sum(matched_gts[:, 4] == i) + sum(unmatched_gts[:, 4] == i) + 1e-6)
                    )
                if self.classifier is not None:
                    tp_cls = sum(lb_cls_matched_thr_filtered[:, 6] == i)
                    fp_cls = sum(
                        np.concatenate(
                            [
                                lb_cls_unmatched_thr_filtered[:, 6] == i,
                                unmatched_thr_cls_filtered[:, 6] == i,
                            ]
                        )
                    )
                    metrics[f"{label}_det_cls_pr"].append(tp_cls / (tp_cls + fp_cls + 1e-6))
                    metrics[f"{label}_det_cls_recall"].append(
                        tp_cls / (sum(matched_gts[:, 4] == i) + sum(unmatched_gts[:, 4] == i) + 1e-6)
                    )

        metrics = pd.DataFrame(metrics)
        metrics.to_csv(output_folder / "metrics.csv")
        det_ap = auc(metrics["det_recall"], metrics["det_pr"])
        det_lb_aps = {}
        if self.classifier is not None:
            det_cls_ap = auc(metrics["det_cls_recall"], metrics["det_cls_pr"])
            det_cls_lb_aps = {}
        for _, label in self.names.items():
            if label in skip_labels:
                continue
            if not self.is_detector_single_class:
                det_lb_aps[label] = auc(metrics[f"{label}_det_recall"], metrics[f"{label}_det_pr"])
            if self.classifier is not None:
                det_cls_lb_aps[label] = auc(metrics[f"{label}_det_cls_recall"], metrics[f"{label}_det_cls_pr"])

        # Plot PR curve
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(
            metrics["det_recall"],
            metrics["det_pr"],
            marker="o",
            linestyle="-",
            label=f"Single class detection AP@{self.match_iou:.2f}={det_ap:.3f}",
            color=generate_color("Single_class"),
        )
        if self.classifier is not None:
            ax.plot(
                metrics["det_cls_recall"],
                metrics["det_cls_pr"],
                marker="x",
                linestyle="-",
                label=f"Single class detection + classification AP@{self.match_iou:.2f}={det_cls_ap:.3f}",
                color=generate_color("Single class"),
            )
        for _, label in self.names.items():
            if label in skip_labels:
                continue
            if not self.is_detector_single_class:
                ax.plot(
                    metrics[f"{label}_det_recall"],
                    metrics[f"{label}_det_pr"],
                    marker="o",
                    linestyle="-",
                    label=f"{label} Detection AP@{self.match_iou:.2f}={det_lb_aps[label]:.3f}",
                    color=generate_color(label),
                )
            if self.classifier is not None:
                ax.plot(
                    metrics[f"{label}_det_cls_recall"],
                    metrics[f"{label}_det_cls_pr"],
                    marker="x",
                    linestyle="-",
                    label=f"{label} Detection + classification AP@{self.match_iou:.2f}={det_cls_lb_aps[label]:.3f}",
                    color=generate_color(label),
                )
        if self.classifier is not None:
            ax.set_title(
                f"Precision-Recall Curve, mAP@{self.match_iou:.2f}: det {np.mean([v for v in det_lb_aps.values()]):.3f} det+cls {np.mean([v for v in det_cls_lb_aps.values()]):.3f}"
            )
        else:
            ax.set_title(
                f"Precision-Recall Curve, mAP@{self.match_iou:.2f}: det {np.mean([v for v in det_lb_aps.values()]):.3f}"
            )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.grid(True)
        fig.savefig(output_folder / "PR_curves.png")

        # Plot recall curves
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(
            metrics["threshold"],
            metrics["det_recall"],
            marker="o",
            linestyle="-",
            label="Single class detection",
            color=generate_color("Single_class"),
        )
        if self.classifier is not None:
            ax.plot(
                metrics["threshold"],
                metrics["det_cls_recall"],
                marker="x",
                linestyle="-",
                label="Single class detection + classification",
                color=generate_color("Single_class"),
            )
        for _, label in self.names.items():
            if label in skip_labels:
                continue
            if not self.is_detector_single_class:
                ax.plot(
                    metrics["threshold"],
                    metrics[f"{label}_det_recall"],
                    marker="o",
                    linestyle="-",
                    label=f"{label} Detection",
                    color=generate_color(label),
                )
            if self.classifier is not None:
                ax.plot(
                    metrics["threshold"],
                    metrics[f"{label}_det_cls_recall"],
                    marker="x",
                    linestyle="-",
                    label=f"{label} Detection + classification",
                    color=generate_color(label),
                )
        ax.set_title("Recall Curve")
        ax.set_xlabel("threshold")
        ax.set_ylabel("Recall")
        ax.legend()
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.grid(True)
        fig.savefig(output_folder / "recall_curves.png")

        # Plot precision curves
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(
            metrics["threshold"],
            metrics["det_pr"],
            marker="o",
            linestyle="-",
            label="Single class detection",
            color=generate_color("Single_class"),
        )
        if self.classifier is not None:
            ax.plot(
                metrics["threshold"],
                metrics["det_cls_pr"],
                marker="x",
                linestyle="-",
                label="Single class detection + classification",
                color=generate_color("Single_class"),
            )
        for _, label in self.names.items():
            if label in skip_labels:
                continue
            if not self.is_detector_single_class:
                ax.plot(
                    metrics["threshold"],
                    metrics[f"{label}_det_pr"],
                    marker="o",
                    linestyle="-",
                    label=f"{label} Detection",
                    color=generate_color(label),
                )
            if self.classifier is not None:
                ax.plot(
                    metrics["threshold"],
                    metrics[f"{label}_det_cls_pr"],
                    marker="x",
                    linestyle="-",
                    label=f"{label} Detection + classification",
                    color=generate_color(label),
                )
        ax.set_title("Precision Curve")
        ax.set_xlabel("threshold")
        ax.set_ylabel("Precision")
        ax.legend()
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.grid(True)
        fig.savefig(output_folder / "precision_curves.png")
        if self.classifier is not None:
            print(f"Single class metrics AP: detection {det_ap:.3f}, detection + classification {det_cls_ap:.3f}")
        else:
            print(f"Single class metrics AP: detection {det_ap:.3f}")
        for _, label in self.names.items():
            if label in skip_labels:
                continue
            if not self.is_detector_single_class:
                if self.classifier is not None:
                    print(
                        f"Class {label} metrics AP: detection {det_lb_aps[label]:.3f}, detection + classification {det_cls_lb_aps[label]:.3f}"
                    )
                else:
                    print(f"Class {label} metrics AP: detection {det_lb_aps[label]:.3f}")
            elif self.classifier is not None:
                print(f"Class {label} metrics AP: detection + classification {det_cls_lb_aps[label]:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Script for image processing and validation of detection and classification results."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file")
    parser.add_argument("--weights_detector", type=str, required=True, help="Path to the YOLO detector weights")
    parser.add_argument(
        "--detector_single_class", action="store_true", help="Is detector a single class detector only?"
    )
    parser.add_argument("--weights_classifier", type=str, default=None, help="Path to the classifier weights")
    parser.add_argument(
        "--iou_threshold", type=float, default=0.5, help="IoU threshold for matching predicted and true boxes"
    )
    parser.add_argument("--img_size", type=int, default=192, help="Image size for inference pipeline")
    parser.add_argument("--nms_iou_threshold", type=float, default=0.2, help="IoU threshold for NMS")
    parser.add_argument(
        "--conf_threshold", type=float, default=0.1, help="Confidence threshold for filtering predictions"
    )
    parser.add_argument("--output_folder", type=str, default="runs/predict", help="Results folder path")
    parser.add_argument("-pad", action="store_true", help="Preprocessing method for crops classification")
    parser.add_argument("--device", type=int, default=0, help="Cuda device index (0, 1, 2, etc.)")
    args = parser.parse_args()

    # Загрузка конфигурации из YAML файла
    try:
        with open(args.config, "r") as file:
            dataset_config = yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Ошибка загрузки конфигурации: {e}")

    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    evaluator = Evaluator(
        detection_model_path=args.weights_detector,
        is_detector_single_class=args.detector_single_class,
        classification_model_path=args.weights_classifier,
        dataset_cfg=dataset_config,
        min_det_conf_threshold=args.conf_threshold,
        nms_iou=args.nms_iou_threshold,
        match_iou=args.iou_threshold,
        device=device,
        cls_inf_size=args.img_size,
        pad=args.pad,
    )
    evaluator.eval(output_folder)


if __name__ == "__main__":
    main()
