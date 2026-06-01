from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.preprocessing import preprocess_for_model, read_image_bgr, read_image_bytes
from ml.severity import estimate_severity


@dataclass(frozen=True)
class PredictionResult:
    damage_type: str
    severity: str
    confidence: float
    probabilities: dict[str, float]


class RoadDamageClassifier:
    def __init__(
        self,
        model_path: str | Path,
        labels_path: str | Path,
        image_size: tuple[int, int] = (224, 224),
    ) -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.image_size = image_size
        self.model = tf.keras.models.load_model(self.model_path)
        self.class_names = json.loads(self.labels_path.read_text(encoding="utf-8"))

    def predict_array(self, image_bgr: np.ndarray) -> PredictionResult:
        image = preprocess_for_model(image_bgr, self.image_size, use_enhancement=False)
        batch = np.expand_dims(image, axis=0)
        probabilities_array = self.model.predict(batch, verbose=0)[0]
        class_index = int(np.argmax(probabilities_array))
        damage_type = self.class_names[class_index]
        confidence = float(probabilities_array[class_index])
        probabilities = {
            class_name: float(probabilities_array[index])
            for index, class_name in enumerate(self.class_names)
        }
        return PredictionResult(
            damage_type=damage_type,
            severity=estimate_severity(damage_type, confidence),
            confidence=confidence,
            probabilities=probabilities,
        )

    def predict_path(self, image_path: str | Path) -> PredictionResult:
        return self.predict_array(read_image_bgr(image_path))

    def predict_bytes(self, image_bytes: bytes) -> PredictionResult:
        return self.predict_array(read_image_bytes(image_bytes))
