from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def read_image_bytes(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a valid image.")
    return image


def enhance_road_image(image_bgr: np.ndarray) -> np.ndarray:
    """Apply light contrast enhancement without changing road texture too much."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_lightness = clahe.apply(lightness)
    enhanced_lab = cv2.merge((enhanced_lightness, channel_a, channel_b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return cv2.GaussianBlur(enhanced, (3, 3), 0)


def preprocess_for_model(
    image_bgr: np.ndarray,
    image_size: tuple[int, int] = (224, 224),
    use_enhancement: bool = True,
) -> np.ndarray:
    if use_enhancement:
        image_bgr = enhance_road_image(image_bgr)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, image_size, interpolation=cv2.INTER_AREA)
    return image_rgb.astype("float32")


def crop_box(
    image_bgr: np.ndarray,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    padding: float = 0.15,
) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    box_width = xmax - xmin
    box_height = ymax - ymin
    pad_x = int(box_width * padding)
    pad_y = int(box_height * padding)

    left = max(0, xmin - pad_x)
    top = max(0, ymin - pad_y)
    right = min(width, xmax + pad_x)
    bottom = min(height, ymax + pad_y)

    if right <= left or bottom <= top:
        raise ValueError("Invalid crop coordinates.")

    return image_bgr[top:bottom, left:right]

