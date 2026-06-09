from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        summary="Python backend for interactive portfolio project demos.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=settings.cors_origin_regex,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/healthz", tags=["health"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", tags=["health"])
    async def root() -> dict[str, object]:
        return {
            "status": "ok",
            "service": settings.app_name,
            "docs": "/docs",
            "health": "/healthz",
            "demos": {
                "orchflow": "/demos/orchflow/run",
                "agenteval": "/demos/agenteval/run",
                "guardloop": "/demos/guardloop/run",
                "smartmemo": "/demos/smartmemo/run",
            },
        }

    app.include_router(api_router)
    return app


app = create_app()
