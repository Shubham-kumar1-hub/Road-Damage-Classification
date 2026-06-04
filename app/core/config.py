# Stores all project settings in one place.

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Road Damage Detection API"
    environment: str = "local"
    database_url: str = "postgresql+psycopg2://road_user:road_password@localhost:5432/road_damage"
    model_path: Path = Path("ml/models/road_damage_resnet50.keras")
    labels_path: Path = Path("ml/models/road_damage_resnet50_labels.json")
    upload_dir: Path = Path("data/uploads")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()

