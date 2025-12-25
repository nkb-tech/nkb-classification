import argparse
from pathlib import Path
from typing import Union, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from nkb_classification.dataset import get_inference_dataset
from nkb_classification.model import get_model
from nkb_classification.utils import load_classes, get_classes_configs, read_py_config


@torch.no_grad()
def inference(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    classes: Union[list, dict],
    save_path: str,
    device: Union[torch.device, str],
    cfg: Any,
    dataset=None,
) -> None:

    _, idx_to_class = get_classes_configs(classes)

    task = cfg.task
    assert task in ("single", "multi")
    
    # Проверяем, используется ли CSV датасет
    use_csv_dataset = dataset is not None and hasattr(dataset, 'get_table')
    
    # Получаем список классов для создания колонок с вероятностями
    if task == "single":
        if isinstance(classes, list):
            class_names = classes
        elif isinstance(classes, dict):
            class_names = list(classes.values())
        else:
            class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    else:
        class_names = None  # Для multi-task будет обработано отдельно
    
    if use_csv_dataset:
        # Если используется CSV датасет, загружаем исходную таблицу
        inference_annotations = dataset.get_table()
        # Создаем колонку для предсказаний, если её еще нет
        target_column = cfg.target_column
        if target_column not in inference_annotations.columns:
            inference_annotations[target_column] = None
        
        # Создаем колонки для вероятностей каждого класса
        if task == "single":
            for class_name in class_names:
                prob_column = f"prob_{class_name}"
                if prob_column not in inference_annotations.columns:
                    inference_annotations[prob_column] = None
        
        # Создаем словари для хранения предсказаний и вероятностей по исходным индексам
        predictions_dict = {}
        probabilities_dict = {} if task == "single" else None
    else:
        # Старый способ: создаем новый DataFrame
        if task == "single":
            target_column = cfg.target_column
            columns = [target_column]
            # Добавляем колонки для вероятностей
            for class_name in class_names:
                columns.append(f"prob_{class_name}")
        elif task == "multi":
            target_names = cfg.target_names
            assert set(target_names) == set(classes.keys())
            columns = target_names.copy()
        columns.append("path")
        inference_annotations = pd.DataFrame(columns=columns)

    model.eval()

    for batch_data in tqdm(loader, leave=False, desc="Inference"):
        if use_csv_dataset:
            # Для CSV датасета: (imgs, paths, original_indices)
            imgs, img_paths, original_indices = batch_data
            # Преобразуем индексы в список, если это тензоры
            if isinstance(original_indices, torch.Tensor):
                original_indices = original_indices.cpu().numpy().tolist()
            elif not isinstance(original_indices, list):
                original_indices = list(original_indices)
        else:
            # Для обычного датасета: (imgs, paths)
            imgs, img_paths = batch_data
            original_indices = None
        
        imgs = imgs.float().to(device)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=cfg.enable_mixed_presicion,
        ):
            preds = model(imgs)
        
        if task == "single":
            # Получаем logits и применяем softmax для получения вероятностей
            logits = preds
            probs = F.softmax(logits, dim=-1)
            
            # Получаем предсказанные классы (argmax)
            pred_indices = logits.argmax(dim=-1).cpu().numpy().tolist()
            pred_classes = [idx_to_class[idx] for idx in pred_indices]
            
            # Получаем вероятности для каждого класса
            probs_np = probs.cpu().numpy()  # shape: (batch_size, num_classes)
            
            if use_csv_dataset:
                # Сохраняем предсказания и вероятности по исходным индексам
                for i, orig_idx in enumerate(original_indices):
                    orig_idx_int = int(orig_idx)
                    predictions_dict[orig_idx_int] = pred_classes[i]
                    
                    # Сохраняем вероятности для каждого класса
                    if probabilities_dict is not None:
                        prob_dict = {}
                        for class_idx, class_name in enumerate(class_names):
                            prob_dict[class_name] = float(probs_np[i, class_idx])
                        probabilities_dict[orig_idx_int] = prob_dict
            else:
                batch_annotations = []
                batch_annotations.append(pred_classes)
                
                # Добавляем вероятности для каждого класса
                for class_idx, class_name in enumerate(class_names):
                    batch_probs = probs_np[:, class_idx].tolist()
                    batch_annotations.append(batch_probs)
                
                batch_annotations.append(list(img_paths))
                batch_annotations = np.vstack(batch_annotations).T
                inference_annotations = pd.concat(
                    [
                        inference_annotations,
                        pd.DataFrame(batch_annotations, columns=columns),
                    ]
                )
        elif task == "multi":
            batch_annotations = []
            for target_name in target_names:
                pred = preds[target_name]
                pred = pred.argmax(dim=-1).cpu().numpy().tolist()
                pred = [idx_to_class[target_name][idx] for idx in pred]
                batch_annotations.append(pred)
            batch_annotations.append(list(img_paths))
            batch_annotations = np.vstack(batch_annotations).T
            inference_annotations = pd.concat(
                [
                    inference_annotations,
                    pd.DataFrame(batch_annotations, columns=columns),
                ]
            )
    
    # Если использовался CSV датасет, обновляем предсказания в исходной таблице
    if use_csv_dataset:
        inference_annotations[target_column] = inference_annotations['_original_idx'].map(predictions_dict)
        
        # Добавляем вероятности для каждого класса
        if task == "single" and probabilities_dict is not None:
            for class_name in class_names:
                prob_column = f"prob_{class_name}"
                # Создаем функцию для извлечения вероятности конкретного класса
                inference_annotations[prob_column] = inference_annotations['_original_idx'].map(
                    lambda idx: probabilities_dict.get(int(idx), {}).get(class_name, None)
                )
        
        # Удаляем служебную колонку
        inference_annotations = inference_annotations.drop(columns=['_original_idx'])
        # Сохраняем обновленный CSV
        output_csv = Path(save_path, "inference_results.csv")
        inference_annotations.to_csv(output_csv, index=False)
        print(f"Результаты сохранены в: {output_csv}")
        print(f"Сохранены предсказания и вероятности для {len(class_names)} классов")
    else:
        inference_annotations.to_csv(Path(save_path, "inference_annotations.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Inference arguments")
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

    # get dataloader and dataset (dataset is returned only for CSV-based datasets)
    result = get_inference_dataset(cfg.inference_data, cfg.inference_pipeline)
    if isinstance(result, tuple):
        data_loader, dataset = result
    else:
        data_loader = result
        dataset = None

    # load classes config
    classes = load_classes(cfg.classes)

    # get model
    device = torch.device(cfg.device)
    model = get_model(cfg.model, classes, device, compile=cfg.compile)

    save_path = Path(cfg.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    inference(model, data_loader, classes, save_path, device, cfg, dataset=dataset)


if __name__ == "__main__":
    main()
