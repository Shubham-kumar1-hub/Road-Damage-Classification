from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from ml.predict import PredictionResult, RoadDamageClassifier


@lru_cache
def get_classifier() -> RoadDamageClassifier:
    settings = get_settings()
    if not settings.model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {settings.model_path}. "
            "Train a model first or update MODEL_PATH in .env."
        )
    if not settings.labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found at {settings.labels_path}. "
            "Train a model first or update LABELS_PATH in .env."
        )
    return RoadDamageClassifier(settings.model_path, settings.labels_path)


def predict_damage(image_bytes: bytes) -> PredictionResult:
    return get_classifier().predict_bytes(image_bytes)

