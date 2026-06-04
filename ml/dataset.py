# Loads training dataset.

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from ml.config import TrainingConfig


def load_split_dataset(
    split_dir: Path,
    config: TrainingConfig,
    shuffle: bool,
) -> tf.data.Dataset:
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")

    return tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=shuffle,
        seed=config.seed,
    )


def prepare_datasets(
    data_dir: str | Path,
    config: TrainingConfig,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset | None, list[str]]:
    data_path = Path(data_dir)
    train_ds = load_split_dataset(data_path / "train", config, shuffle=True)
    val_ds = load_split_dataset(data_path / "val", config, shuffle=False)
    test_dir = data_path / "test"
    test_ds = load_split_dataset(test_dir, config, shuffle=False) if test_dir.exists() else None

    class_names = list(train_ds.class_names)
    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    if test_ds is not None:
        test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names

