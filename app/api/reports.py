from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.report import DamageReportResponse
from app.services.model_service import predict_damage
from app.services.report_service import (
    create_damage_report,
    find_nearby_reports,
    list_damage_reports,
    save_upload,
)


router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("", response_model=DamageReportResponse)
async def create_report(
    lat: float = Form(..., ge=-90, le=90),
    lon: float = Form(..., ge=-180, le=180),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DamageReportResponse:
    image_bytes = await file.read()
    try:
        prediction = predict_damage(image_bytes)
        image_path = save_upload(file, image_bytes)
        report = create_damage_report(db, prediction, lat, lon, image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DamageReportResponse(**report)


@router.get("", response_model=list[DamageReportResponse])
def get_reports(
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[DamageReportResponse]:
    return [DamageReportResponse(**report) for report in list_damage_reports(db, limit)]


@router.get("/nearby", response_model=list[DamageReportResponse])
def get_nearby_reports(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(5.0, gt=0, le=100),
    severity: str | None = Query(None),
    db: Session = Depends(get_db),
) -> list[DamageReportResponse]:
    reports = find_nearby_reports(db, lat, lon, radius_km, severity)
    return [DamageReportResponse(**report) for report in reports]

