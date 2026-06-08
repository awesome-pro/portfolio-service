from __future__ import annotations

from fastapi import APIRouter

from app.demos.agenteval.routes import router as agenteval_router
from app.demos.orchflow.routes import router as orchflow_router

api_router = APIRouter()
api_router.include_router(
    agenteval_router,
    prefix="/demos/agenteval",
    tags=["agenteval"],
)
api_router.include_router(orchflow_router, prefix="/demos/orchflow", tags=["orchflow"])
