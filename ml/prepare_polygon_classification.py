from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

from ml.preprocessing import crop_box, read_image_bgr


CLASS_MAP = {
    0: "pothole",
    1: "crack",
    2: "manhole",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class PolygonObject:
    class_id: int
    points: list[tuple[float, float]]

    @property
    def label(self) -> str:
        return CLASS_MAP[self.class_id]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert polygon labels into classification crop folders.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-crop-size", type=int, default=20)
    return parser.parse_args()


def parse_label_file(label_path: Path) -> list[PolygonObject]:
    objects: list[PolygonObject] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 9:
            continue

        class_id = int(float(parts[0]))
        if class_id not in CLASS_MAP:
            continue

        values = [float(value) for value in parts[1:]]
        points = list(zip(values[0::2], values[1::2], strict=False))
        objects.append(PolygonObject(class_id=class_id, points=points))
    return objects


def polygon_to_pixel_box(
    polygon: PolygonObject,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x_values = [point[0] for point in polygon.points]
    y_values = [point[1] for point in polygon.points]

    xmin = int(min(x_values) * image_width)
    ymin = int(min(y_values) * image_height)
    xmax = int(max(x_values) * image_width)
    ymax = int(max(y_values) * image_height)
    return xmin, ymin, xmax, ymax


def list_images(images_dir: Path) -> list[Path]:
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def split_samples(
    samples: dict[str, list[tuple[Path, PolygonObject]]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[tuple[str, Path, PolygonObject]]]:
    random.seed(seed)
    result: dict[str, list[tuple[str, Path, PolygonObject]]] = defaultdict(list)

    for label, label_samples in samples.items():
        random.shuffle(label_samples)
        train_end = int(len(label_samples) * train_ratio)
        val_end = train_end + int(len(label_samples) * val_ratio)

        for split, split_items in {
            "train": label_samples[:train_end],
            "val": label_samples[train_end:val_end],
            "test": label_samples[val_end:],
        }.items():
            for image_path, polygon in split_items:
                result[label].append((split, image_path, polygon))
    return result


def main() -> None:
    args = parse_args()
    images_dir = args.dataset_dir / "images"
    labels_dir = args.dataset_dir / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    samples: dict[str, list[tuple[Path, PolygonObject]]] = defaultdict(list)
    for image_path in list_images(images_dir):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        for polygon in parse_label_file(label_path):
            samples[polygon.label].append((image_path, polygon))

    split_data = split_samples(samples, args.train_ratio, args.val_ratio, args.seed)
    written_counts: dict[str, int] = defaultdict(int)

    for label, split_samples_for_label in split_data.items():
        for index, (split, image_path, polygon) in enumerate(split_samples_for_label):
            image = read_image_bgr(image_path)
            image_height, image_width = image.shape[:2]
            xmin, ymin, xmax, ymax = polygon_to_pixel_box(polygon, image_width, image_height)

            if xmax - xmin < args.min_crop_size or ymax - ymin < args.min_crop_size:
                continue

            cropped = crop_box(image, xmin, ymin, xmax, ymax)
            output_dir = args.output_dir / split / label
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{image_path.stem}_{index:05d}.jpg"
            cv2.imwrite(str(output_path), cropped)
            written_counts[f"{split}/{label}"] += 1

    print(f"Prepared classification dataset at: {args.output_dir}")
    for key in sorted(written_counts):
        print(f"{key}: {written_counts[key]}")


if __name__ == "__main__":
    main()

