from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.report import PredictionResponse
from app.services.model_service import predict_damage


router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    image_bytes = await file.read()
    try:
        prediction = predict_damage(image_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictionResponse(
        damage_type=prediction.damage_type,
        severity=prediction.severity,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
    )

