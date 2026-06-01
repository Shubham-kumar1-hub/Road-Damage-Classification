from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "ml" / "models"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "ml" / "reports"


RDD_LABEL_MAP = {
    "D00": "longitudinal_crack",
    "D10": "transverse_crack",
    "D20": "alligator_crack",
    "D40": "pothole",
}


@dataclass(frozen=True)
class TrainingConfig:
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    seed: int = 42
    learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-5
    dropout_rate: float = 0.35

