from __future__ import annotations

from fastapi import APIRouter

from app.demos.orchflow.routes import router as orchflow_router

api_router = APIRouter()
api_router.include_router(orchflow_router, prefix="/demos/orchflow", tags=["orchflow"])
