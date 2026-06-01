from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from ml.config import DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR, DEFAULT_REPORT_DIR, TrainingConfig
from ml.dataset import prepare_datasets
from ml.models import build_basic_cnn, build_resnet50, compile_model, unfreeze_resnet_top


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train road damage classification models.")
    parser.add_argument("--model", choices=["cnn", "resnet50"], required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--fine-tune-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--output-name", type=str, default=None)
    return parser.parse_args()


def make_callbacks(model_name: str) -> list[tf.keras.callbacks.Callback]:
    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path("ml") / "checkpoints" / f"{model_name}.keras"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
        ),
    ]


def compute_class_weights(data_dir: Path, class_names: list[str]) -> dict[int, float]:
    counts = []
    for class_name in class_names:
        class_dir = data_dir / "train" / class_name
        count = len([path for path in class_dir.rglob("*") if path.is_file()])
        counts.append(max(count, 1))

    total = sum(counts)
    num_classes = len(class_names)
    return {
        index: total / (num_classes * count)
        for index, count in enumerate(counts)
    }


def save_training_artifacts(
    model: tf.keras.Model,
    history: tf.keras.callbacks.History,
    model_name: str,
    class_names: list[str],
) -> None:
    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = DEFAULT_MODEL_DIR / f"{model_name}.keras"
    labels_path = DEFAULT_MODEL_DIR / f"{model_name}_labels.json"
    history_path = DEFAULT_REPORT_DIR / f"{model_name}_history.csv"

    model.save(model_path)
    labels_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    pd.DataFrame(history.history).to_csv(history_path, index=False)

    print(f"Saved model: {model_path}")
    print(f"Saved labels: {labels_path}")
    print(f"Saved history: {history_path}")


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fine_tune_learning_rate=args.fine_tune_learning_rate,
    )

    train_ds, val_ds, test_ds, class_names = prepare_datasets(args.data_dir, config)
    model_name = args.output_name or f"road_damage_{args.model}"
    class_weights = compute_class_weights(args.data_dir, class_names)

    if args.model == "cnn":
        model = build_basic_cnn(len(class_names), config)
    else:
        model = build_resnet50(len(class_names), config)

    compile_model(model, config.learning_rate)
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=make_callbacks(model_name),
    )

    if args.model == "resnet50" and args.fine_tune_epochs > 0:
        print("Fine-tuning top ResNet50 layers...")
        unfreeze_resnet_top(model, trainable_layers=40)
        compile_model(model, config.fine_tune_learning_rate)
        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            class_weight=class_weights,
            callbacks=make_callbacks(f"{model_name}_finetuned"),
        )
        for metric, values in fine_tune_history.history.items():
            history.history.setdefault(metric, [])
            history.history[metric].extend(values)

    save_training_artifacts(model, history, model_name, class_names)

    if test_ds is not None:
        print("Final test metrics:")
        print(dict(zip(model.metrics_names, model.evaluate(test_ds, verbose=0), strict=False)))


if __name__ == "__main__":
    main()
