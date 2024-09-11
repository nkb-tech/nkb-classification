import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def plot_confusion_matrices(df, output_path, dog_label):
    # Создание меток для детектора
    detector_true_labels = []
    detector_pred_labels = []

    for _, row in df.iterrows():
        if row['detector_label'] == 'background':
            detector_true_labels.append('background')  # На самом деле это был задний фон
            detector_pred_labels.append(dog_label)
        elif row['detector_label'] == f'real_{dog_label}':
            detector_true_labels.append(dog_label)  # На самом деле это была собака
            detector_pred_labels.append('background')
        elif row['detector_label'] == dog_label:
            detector_true_labels.append(dog_label)  # На самом деле это была собака
            detector_pred_labels.append(dog_label)

    # Построение confusion матрицы для детектора
    detector_cm = confusion_matrix(detector_true_labels, detector_pred_labels, labels=[dog_label, 'background'])

    # Вычисление Precision и Recall для детектора
    detector_precision = precision_score(detector_true_labels, detector_pred_labels, pos_label=dog_label, average='binary')
    detector_recall = recall_score(detector_true_labels, detector_pred_labels, pos_label=dog_label, average='binary')

    # Создание меток для классификатора
    classifier_true_labels = []
    classifier_pred_labels = []

    for _, row in df.iterrows():
        if row['class_label'] == 0 and row['detector_label'] == 'background':
            classifier_true_labels.append('background')  # На самом деле это задний фон
            classifier_pred_labels.append(dog_label)  # Классификатор классифицировал как собаку
        elif row['class_label'] == 0 and row['detector_label'] in [dog_label]:
            classifier_true_labels.append(dog_label)  # На самом деле это собака
            classifier_pred_labels.append(dog_label)  # Классификатор классифицировал как собаку
        elif row['class_label'] == 1 and row['detector_label'] == 'background':
            classifier_true_labels.append('background')  # На самом деле это задний фон
            classifier_pred_labels.append('background')  # Классификатор классифицировал как задний фон
        elif row['class_label'] == 1 and row['detector_label'] in [dog_label]:
            classifier_true_labels.append(dog_label)  # На самом деле это собака
            classifier_pred_labels.append('background')  # Классификатор классифицировал как задний фон

    # Построение confusion матрицы для классификатора
    classifier_cm = confusion_matrix(classifier_true_labels, classifier_pred_labels, labels=[dog_label, 'background'])

    # Вычисление Precision и Recall для классификатора
    classifier_precision = precision_score(classifier_true_labels, classifier_pred_labels, pos_label=dog_label, average='binary')
    classifier_recall = recall_score(classifier_true_labels, classifier_pred_labels, pos_label=dog_label, average='binary')

    # Визуализация confusion матриц
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), gridspec_kw={'height_ratios': [5, 1]})

    sns.heatmap(detector_cm, annot=True, fmt="d", cmap="Blues", square=True, 
                xticklabels=[dog_label, 'background'], yticklabels=[dog_label, 'background'], ax=axes[0, 0])
    axes[0, 0].set_xlabel('Predicted Label')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_title(f'Confusion Matrix for Detector')

    axes[1, 0].axis('off')  # Отключаем оси, чтобы разместить текст
    axes[1, 0].text(0.4, 0.5, f"Precision: {detector_precision:.2f} | Recall: {detector_recall:.2f}", 
                    fontsize=16, ha='center', va='center', transform=axes[1, 0].transAxes)

    sns.heatmap(classifier_cm, annot=True, fmt="d", cmap="Greens", square=True, 
                xticklabels=[dog_label, 'background'], yticklabels=[dog_label, 'background'], ax=axes[0, 1])
    axes[0, 1].set_xlabel('Predicted Label')
    axes[0, 1].set_ylabel('True Label')
    axes[0, 1].set_title(f'Confusion Matrix for Classifier')

    axes[1, 1].axis('off')  # Отключаем оси, чтобы разместить текст
    axes[1, 1].text(0.4, 0.5, f"Precision: {classifier_precision:.2f} | Recall: {classifier_recall:.2f}", 
                    fontsize=16, ha='center', va='center', transform=axes[1, 1].transAxes)

    # Сохранение изображения
    plt.tight_layout(h_pad=2.0)
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot and save confusion matrices for detector and classifier")
    parser.add_argument("csv_file", help="Path to the input CSV file")
    parser.add_argument("output_image", help="Path to save the output image")
    parser.add_argument("dog_label", help="Label to use for 'dog' (e.g., 'dog', 'cat')")

    args = parser.parse_args()

    # Загрузка данных
    df = pd.read_csv(args.csv_file)

    # Построение и сохранение графиков
    plot_confusion_matrices(df, args.output_image, args.dog_label)
