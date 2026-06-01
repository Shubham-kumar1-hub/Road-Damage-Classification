from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from ml.predict import PredictionResult


def save_upload(upload: UploadFile, image_bytes: bytes) -> str:
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "image.jpg").suffix or ".jpg"
    file_path = settings.upload_dir / f"{uuid4().hex}{suffix}"
    file_path.write_bytes(image_bytes)
    return str(file_path)


def create_damage_report(
    db: Session,
    prediction: PredictionResult,
    latitude: float,
    longitude: float,
    image_path: str | None,
) -> dict:
    query = text(
        """
        INSERT INTO damage_reports (
            damage_type,
            severity,
            confidence,
            latitude,
            longitude,
            location,
            image_path
        )
        VALUES (
            :damage_type,
            :severity,
            :confidence,
            :latitude,
            :longitude,
            ST_SetSRID(ST_MakePoint(:longitude, :latitude), 4326),
            :image_path
        )
        RETURNING id, damage_type, severity, confidence, latitude, longitude, image_path, created_at
        """
    )
    result = db.execute(
        query,
        {
            "damage_type": prediction.damage_type,
            "severity": prediction.severity,
            "confidence": prediction.confidence,
            "latitude": latitude,
            "longitude": longitude,
            "image_path": image_path,
        },
    ).mappings().one()
    db.commit()
    return dict(result)


def list_damage_reports(db: Session, limit: int = 50) -> list[dict]:
    query = text(
        """
        SELECT id, damage_type, severity, confidence, latitude, longitude, image_path, created_at
        FROM damage_reports
        ORDER BY created_at DESC
        LIMIT :limit
        """
    )
    return [dict(row) for row in db.execute(query, {"limit": limit}).mappings().all()]


def find_nearby_reports(
    db: Session,
    latitude: float,
    longitude: float,
    radius_km: float,
    severity: str | None = None,
) -> list[dict]:
    query = text(
        """
        SELECT
            id,
            damage_type,
            severity,
            confidence,
            latitude,
            longitude,
            image_path,
            created_at,
            ST_Distance(
                location::geography,
                ST_SetSRID(ST_MakePoint(:longitude, :latitude), 4326)::geography
            ) / 1000.0 AS distance_km
        FROM damage_reports
        WHERE
            ST_DWithin(
                location::geography,
                ST_SetSRID(ST_MakePoint(:longitude, :latitude), 4326)::geography,
                :radius_m
            )
            AND (:severity IS NULL OR severity = :severity)
        ORDER BY distance_km ASC, confidence DESC
        """
    )
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "radius_m": radius_km * 1000,
        "severity": severity,
    }
    return [dict(row) for row in db.execute(query, params).mappings().all()]

