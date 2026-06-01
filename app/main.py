from __future__ import annotations

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.predict import router as predict_router
from app.api.reports import router as reports_router
from app.core.config import get_settings


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    description="Classify road damage from dashcam images and log GPS-based repair reports.",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(reports_router)

