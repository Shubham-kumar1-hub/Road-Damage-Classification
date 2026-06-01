from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from ml.config import DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR, DEFAULT_REPORT_DIR, TrainingConfig
from ml.dataset import load_split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a road damage classifier.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_DIR / "road_damage_resnet50.keras")
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_MODEL_DIR / "road_damage_resnet50_labels.json")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    y_true_batches = []
    y_pred_batches = []

    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        y_true_batches.append(np.argmax(labels.numpy(), axis=1))
        y_pred_batches.append(np.argmax(probabilities, axis=1))

    return np.concatenate(y_true_batches), np.concatenate(y_pred_batches)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    class_names = json.loads(args.labels_path.read_text(encoding="utf-8"))
    config = TrainingConfig(image_size=(args.image_size, args.image_size), batch_size=args.batch_size)
    dataset = load_split_dataset(args.data_dir / args.split, config, shuffle=False)
    model = tf.keras.models.load_model(args.model_path)

    y_true, y_pred = collect_predictions(model, dataset)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred)

    report_path = DEFAULT_REPORT_DIR / f"{args.model_path.stem}_{args.split}_classification_report.json"
    matrix_path = DEFAULT_REPORT_DIR / f"{args.model_path.stem}_{args.split}_confusion_matrix.png"

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_confusion_matrix(matrix, class_names, matrix_path)

    print(classification_report(y_true, y_pred, target_names=class_names))
    print(f"Saved report: {report_path}")
    print(f"Saved confusion matrix: {matrix_path}")


if __name__ == "__main__":
    main()

