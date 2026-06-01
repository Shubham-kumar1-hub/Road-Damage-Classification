from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

from ml.config import RDD_LABEL_MAP
from ml.preprocessing import crop_box, read_image_bgr


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class DamageObject:
    label: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def area(self) -> int:
        return max(0, self.xmax - self.xmin) * max(0, self.ymax - self.ymin)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RDD-style XML annotations into classification folders.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--mode", choices=["crop", "image"], default="crop")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-no-damage", action="store_true")
    return parser.parse_args()


def parse_annotation(xml_path: Path) -> list[DamageObject]:
    root = ET.parse(xml_path).getroot()
    objects: list[DamageObject] = []

    for object_node in root.findall("object"):
        raw_label = object_node.findtext("name", default="").strip()
        label = RDD_LABEL_MAP.get(raw_label, raw_label)
        box = object_node.find("bndbox")
        if box is None or label not in set(RDD_LABEL_MAP.values()):
            continue

        objects.append(
            DamageObject(
                label=label,
                xmin=int(float(box.findtext("xmin", default="0"))),
                ymin=int(float(box.findtext("ymin", default="0"))),
                xmax=int(float(box.findtext("xmax", default="0"))),
                ymax=int(float(box.findtext("ymax", default="0"))),
            )
        )

    return objects


def list_images(images_dir: Path) -> list[Path]:
    return sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def split_by_class(samples: dict[str, list[tuple[Path, DamageObject | None]]], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[tuple[str, Path, DamageObject | None]]]:
    random.seed(seed)
    split_samples: dict[str, list[tuple[str, Path, DamageObject | None]]] = defaultdict(list)

    for label, label_samples in samples.items():
        random.shuffle(label_samples)
        total = len(label_samples)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        for split, items in {
            "train": label_samples[:train_end],
            "val": label_samples[train_end:val_end],
            "test": label_samples[val_end:],
        }.items():
            for image_path, damage_object in items:
                split_samples[label].append((split, image_path, damage_object))

    return split_samples


def write_sample(
    image_path: Path,
    damage_object: DamageObject | None,
    output_path: Path,
    mode: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "image" or damage_object is None:
        shutil.copy2(image_path, output_path)
        return

    image = read_image_bgr(image_path)
    crop = crop_box(
        image,
        damage_object.xmin,
        damage_object.ymin,
        damage_object.xmax,
        damage_object.ymax,
    )
    cv2.imwrite(str(output_path), crop)


def main() -> None:
    args = parse_args()
    samples: dict[str, list[tuple[Path, DamageObject | None]]] = defaultdict(list)

    for image_path in list_images(args.images_dir):
        xml_path = args.annotations_dir / f"{image_path.stem}.xml"
        objects = parse_annotation(xml_path) if xml_path.exists() else []

        if not objects and args.include_no_damage:
            samples["no_damage"].append((image_path, None))
            continue

        if args.mode == "image":
            dominant = max(objects, key=lambda item: item.area, default=None)
            if dominant is not None:
                samples[dominant.label].append((image_path, dominant))
        else:
            for damage_object in objects:
                samples[damage_object.label].append((image_path, damage_object))

    split_samples = split_by_class(samples, args.train_ratio, args.val_ratio, args.seed)

    for label, label_samples in split_samples.items():
        for index, (split, image_path, damage_object) in enumerate(label_samples):
            suffix = image_path.suffix.lower()
            output_path = args.output_dir / split / label / f"{image_path.stem}_{index:05d}{suffix}"
            write_sample(image_path, damage_object, output_path, args.mode)

    print(f"Prepared classification dataset at: {args.output_dir}")
    for label, label_samples in split_samples.items():
        print(f"{label}: {len(label_samples)} samples")


if __name__ == "__main__":
    main()

