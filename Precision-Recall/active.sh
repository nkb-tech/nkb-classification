#!/bin/bash

# Параметры для перебора
# nms_iou_values=(0.1 0.2 0.3 0.4 0.5)
# iou_threshold_values=(0.1 0.2 0.3 0.4 0.5)
# conf_threshold_values=(0.05 0.15 0.25 0.35 0.4)
nms_iou_values=(0.2 0.3)
iou_threshold_values=(0.5)
conf_threshold_values=(0.05 0.1 0.15 0.25 0.35 0.4)

# Пути
output_csv_dir="./csv"
output_img_dir="./confusion_matrix"

# Список классификаторов
#classifiers=("deit" "eca" "eva02" "resnet50a1" "seresnext26d" "seresnext50")
classifiers=("eva02")
# Создаем папки, если они не существуют
mkdir -p $output_csv_dir
mkdir -p $output_img_dir

# Перебор всех классификаторов
for classifier in "${classifiers[@]}"; do

  # Создаем поддиректории для классификатора
  mkdir -p "$output_csv_dir/$classifier"
  mkdir -p "$output_img_dir/$classifier"

  # Путь к весам классификатора
  weights_classifier="/home/aduleba/projects/hdd4/models/yolov8_dog_det/classificator/timm/${classifier}/scripted_best.pt"
  
  # Перебор всех комбинаций параметров
  for nms_iou in "${nms_iou_values[@]}"; do
    for iou in "${iou_threshold_values[@]}"; do
      for conf in "${conf_threshold_values[@]}"; do
        
        # Формируем имя файла
        csv_file_name="nms${nms_iou}_iou${iou}_conf${conf}.csv"
        csv_file_path="$output_csv_dir/$classifier/$csv_file_name"
        
        # Запуск первого скрипта с текущими параметрами
        python PR.py --config /home/aduleba/projects/pets/scripts/models_check/yml/8august.yml \
                     --weights_detector /home/aduleba/projects/pets/scripts/models_check/models/check2/detector.pt \
                     --weights_classifier $weights_classifier \
                     --iou_threshold $iou --img_size 336 --nms_iou_threshold $nms_iou --conf_threshold $conf \
                     --output_file $csv_file_path
        
        # Формируем имя файла для изображения
        img_file_name="confusion_nms${nms_iou}_iou${iou}_conf${conf}.jpg"
        img_file_path="$output_img_dir/$classifier/$img_file_name"
        
        # Запуск второго скрипта для создания confusion matrix
        python confusion.py $csv_file_path $img_file_path dog
        
      done
    done
  done

done