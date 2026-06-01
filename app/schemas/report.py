from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    damage_type: str
    severity: str
    confidence: float
    probabilities: dict[str, float]


class DamageReportResponse(BaseModel):
    id: int
    damage_type: str
    severity: str
    confidence: float
    latitude: float
    longitude: float
    image_path: str | None
    created_at: datetime
    distance_km: float | None = None


class NearbyQuery(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(5.0, gt=0, le=100)
    severity: str | None = None

