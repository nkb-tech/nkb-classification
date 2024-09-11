import torch
import yaml
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
import logging
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import argparse

# Установим базовое логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================================
# Скрипт для обработки изображений и валидации результатов детектора и детектора+классификатора.
# ==============================================================================================

def get_crops_from_pil(image, boxes, transform):
    """
    Извлекает и преобразует области изображения (кропы) на основе заданных боксов.

    Args:
        image (PIL.Image): Исходное изображение.
        boxes (Tensor): Тензор с координатами боксов в формате (x1, y1, x2, y2).
        transform (callable): Трансформация, применяемая к извлеченным кропам.

    Returns:
        list: Список преобразованных областей изображения в виде numpy-массивов.
    """
    crops = []
    for box in boxes:
        box = box.int().tolist()
        crop = image.crop((box[0], box[1], box[2], box[3]))  # x1, y1, x2, y2
        crop = crop.convert("RGB")
        crop = np.array(crop)
        crop = transform(image=crop)['image']  # Применение трансформации
        crops.append(crop)
    return crops

def predict_classifier(image_path, boxes, classifier_model, transform, device):
    """
    Прогоняет извлеченные из изображения кропы через классификатор и возвращает предсказанные метки.

    Args:
        image_path (str): Путь к изображению.
        boxes (Tensor): Тензор с координатами боксов в формате (x1, y1, x2, y2).
        classifier_model (torch.nn.Module): Модель классификации.
        transform (callable): Трансформация, применяемая к извлеченным кропам.
        device (torch.device): Устройство, на котором выполняются вычисления (CPU/GPU).

    Returns:
        Tensor: Тензор с предсказанными метками для каждого кропа.
    """
    try:
        raw_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logging.error(f"Ошибка загрузки изображения {image_path}: {e}")
        return torch.tensor([])

    if boxes.numel() == 0:
        return torch.tensor([])

    crops = get_crops_from_pil(raw_image, boxes, transform)

    if not crops:
        return torch.tensor([])

    crops = torch.stack(crops).to(device)
    with torch.no_grad():
        outputs = classifier_model(crops)

    _, predicted_labels = torch.max(outputs, 1)
    return predicted_labels

def box_iou(pred_boxes, gt_boxes):
    """
    Вычисляет матрицу IoU (Intersection over Union) для предсказанных и истинных боксов.

    Args:
        pred_boxes (Tensor): Тензор с координатами предсказанных боксов (x1, y1, x2, y2).
        gt_boxes (Tensor): Тензор с координатами истинных боксов (x1, y1, x2, y2).

    Returns:
        Tensor: Матрица IoU размером (предсказанные боксы, истинные боксы).
    """
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return torch.zeros((pred_boxes.size(0), gt_boxes.size(0)))

    inter_x1 = torch.max(pred_boxes[:, None, 0], gt_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, None, 1], gt_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, None, 2], gt_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, None, 3], gt_boxes[:, 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    union_area = pred_area[:, None] + gt_area - inter_area
    iou = inter_area / torch.clamp(union_area, min=1e-6)

    return iou

def determine_true_label_and_best_gt(pred_boxes, gt_boxes, iou_threshold=0.3):
    """
    Определяет истинные метки для предсказанных боксов на основе IoU с истинными боксами.

    Args:
        pred_boxes (Tensor): Тензор с координатами предсказанных боксов (x1, y1, x2, y2).
        gt_boxes (Tensor): Тензор с координатами истинных боксов (x1, y1, x2, y2).
        iou_threshold (float): Пороговое значение IoU для сопоставления боксов.

    Returns:
        list: Список истинных меток для каждого предсказанного бокса.
        Tensor: Маска для идентификации не сопоставленных истинных боксов.
    """
    iou_matrix = box_iou(pred_boxes, gt_boxes)
    true_labels = []
    best_gt_boxes = []
    matched_gt_boxes = torch.zeros(len(gt_boxes), dtype=torch.bool).to(gt_boxes.device)

    for i in range(iou_matrix.size(0)):
        if iou_matrix[i].numel() == 0:
            true_labels.append('background')
            best_gt_boxes.append([0, 0, 0, 0])
        else:
            max_iou, max_idx = iou_matrix[i].max(0)
            if max_iou > iou_threshold:
                true_labels.append('dog')
                matched_gt_boxes[max_idx] = True
                best_gt_boxes.append(gt_boxes[max_idx].tolist())
            else:
                true_labels.append('background')
                best_gt_boxes.append([0, 0, 0, 0])

    return true_labels, matched_gt_boxes, best_gt_boxes

def load_models(weights_detector, weights_classifier, device):
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
    detection_model = YOLO(weights_detector)
    classifier_model = torch.jit.load(weights_classifier, map_location=device)
    classifier_model.eval()
    return detection_model, classifier_model

def create_inference_pipeline(img_size):
    """
    Создает пайплайн для предобработки изображений перед инференсом.

    Args:
        img_size (int): Размер изображения для предобработки.

    Returns:
        callable: Пайплайн для предобработки изображений.
    """
    return A.Compose(
        [
            A.LongestMaxSize(img_size, always_apply=True),
            A.PadIfNeeded(img_size, img_size, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )

def process_image(image_path, label_path, detection_model, classifier_model, inference_pipeline, device, iou_threshold, nms_iou_threshold, conf_threshold):
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
        conf_threshold (float): Порог confidence для фильтрации предсказаний.

    Returns:
        list: Список результатов обработки изображения.
    """
    results = []
    try:
        yolo_results = detection_model(image_path, verbose=False, conf=conf_threshold, iou=nms_iou_threshold)
        pred_boxes = yolo_results[0].boxes.xyxy.to(device)
        conf = yolo_results[0].boxes.conf
        image_height, image_width = yolo_results[0].orig_shape

        with open(label_path, 'r') as lf:
            gt_boxes = []
            for line in lf:
                _, x_center, y_center, width, height = map(float, line.split())
                x1 = (x_center - width / 2) * image_width
                y1 = (y_center - height / 2) * image_height
                x2 = (x_center + width / 2) * image_width
                y2 = (y_center + height / 2) * image_height
                gt_boxes.append([x1, y1, x2, y2])

        gt_boxes = torch.tensor(gt_boxes).to(device)
    except Exception as e:
        logging.error(f"Ошибка обработки {label_path}: {e}")
        return results

    true_labels, matched_gt_boxes, best_gt_boxes = determine_true_label_and_best_gt(pred_boxes, gt_boxes, iou_threshold)
    predicted_labels = predict_classifier(image_path, pred_boxes, classifier_model, inference_pipeline, device)

    for pred_box, true_label, pred_label, best_gt_box in zip(pred_boxes, true_labels, predicted_labels, best_gt_boxes):
        results.append({
            'image_name': os.path.basename(image_path),
            'detector_label': true_label,
            'class_label': pred_label.item(),
            'gt_box': [int(coord) for coord in best_gt_box],
            'pred_box': [int(coord) for coord in pred_box.tolist()]
        })

    unmatched_gt_boxes = gt_boxes[~matched_gt_boxes]
    if unmatched_gt_boxes.numel() > 0:
        unmatched_predicted_labels = predict_classifier(image_path, unmatched_gt_boxes, classifier_model, inference_pipeline, device)
        for unmatched_box, unmatched_pred_label in zip(unmatched_gt_boxes, unmatched_predicted_labels):
            results.append({
                'image_name': os.path.basename(image_path),
                'detector_label': 'real_dog',
                'class_label': unmatched_pred_label.item(),
                'gt_box': [int(coord) for coord in unmatched_box.tolist()],
                'pred_box': None
            })

    return results

def process_validation_files(val_files, dataset_path, detection_model, classifier_model, inference_pipeline, device, iou_threshold, nms_iou_threshold, conf_threshold):
    """
    Обрабатывает все файлы валидации, выполняя детекцию и классификацию объектов.

    Args:
        val_files (list): Список файлов валидации.
        dataset_path (str): Путь к директории с изображениями.
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
    results = []
    for val_file in tqdm(val_files, desc="Processing validation files"):
        try:
            with open(os.path.join(dataset_path, val_file), 'r') as f:
                image_paths = f.readlines()
        except Exception as e:
            logging.error(f"Ошибка загрузки валидационного файла {val_file}: {e}")
            continue

        for image_path in tqdm(image_paths, desc=f"Processing images in {val_file}", leave=False):
            image_path = image_path.strip()
            label_path = image_path.replace('/images', '/labels').replace('.jpg', '.txt')
            results.extend(process_image(image_path, label_path, detection_model, classifier_model, inference_pipeline, device, iou_threshold, nms_iou_threshold, conf_threshold))

    return results

def save_results(results, output_file):
    """
    Сохраняет результаты обработки изображений в CSV файл.

    Args:
        results (list): Список результатов обработки.
        output_file (str): Путь для сохранения выходного CSV файла.
    """
    results_df = pd.DataFrame(results)
    try:
        results_df.to_csv(output_file, index=False)
        logging.info(f"Результаты успешно сохранены в {output_file}")
    except Exception as e:
        logging.error(f"Ошибка сохранения результатов в {output_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Script for image processing and validation of detection and classification results.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration YAML file")
    parser.add_argument('--weights_detector', type=str, required=True, help="Path to the YOLO detector weights")
    parser.add_argument('--weights_classifier', type=str, required=True, help="Path to the classifier weights")
    parser.add_argument('--iou_threshold', type=float, default=0.5, help="IoU threshold for matching predicted and true boxes")
    parser.add_argument('--img_size', type=int, default=192, help="Image size for inference pipeline")
    parser.add_argument('--nms_iou_threshold', type=float, default=0.2, help="IoU threshold for NMS")
    parser.add_argument('--conf_threshold', type=float, default=0.1, help="Confidence threshold for filtering predictions")
    parser.add_argument('--output_file', type=str, default='detection_results.csv', help="Path for the output CSV file")
    args = parser.parse_args()

    # Загрузка конфигурации из YAML файла
    try:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Ошибка загрузки конфигурации: {e}")
        raise

    dataset_path = config['path']
    val_files = config['val']
    nc = config['nc']
    names = config['names']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detection_model, classifier_model = load_models(args.weights_detector, args.weights_classifier, device)
    inference_pipeline = create_inference_pipeline(args.img_size)

    results = process_validation_files(val_files, dataset_path, detection_model, classifier_model, inference_pipeline, device, args.iou_threshold, args.nms_iou_threshold, args.conf_threshold)

    save_results(results, args.output_file)

if __name__ == "__main__":
    main()
