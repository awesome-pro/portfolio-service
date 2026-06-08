from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.core.config import Settings, get_settings
from app.core.rate_limit import (
    client_rate_limit_key,
    orchflow_rate_limiter,
    raise_rate_limit_exceeded,
)
from app.demos.orchflow.schemas import OrchflowModelPreset, OrchflowRunRequest
from app.demos.orchflow.workflow import encode_ndjson, iter_orchflow_demo_events

router = APIRouter()


@router.post("/run")
async def run_orchflow_demo(
    request: OrchflowRunRequest,
    http_request: Request,
) -> StreamingResponse:
    settings = get_settings()
    decision = orchflow_rate_limiter.check(
        client_rate_limit_key(http_request),
        max_runs=settings.orchflow_rate_limit_max_runs,
        window_seconds=settings.orchflow_rate_limit_window_seconds,
    )
    if not decision.allowed:
        raise_rate_limit_exceeded(decision.retry_after_seconds)

    missing_keys = [
        name
        for name, value in _required_provider_keys(request, settings).items()
        if not value
    ]
    if missing_keys:
        raise HTTPException(
            status_code=503,
            detail=f"Missing LLM API key(s): {', '.join(missing_keys)}",
        )

    async def stream() -> AsyncIterator[str]:
        async for event in iter_orchflow_demo_events(request):
            yield encode_ndjson(event)

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store"},
    )


def _required_provider_keys(
    request: OrchflowRunRequest,
    settings: Settings,
) -> dict[str, str | None]:
    if request.model_preset is OrchflowModelPreset.haiku_only:
        return {"ANTHROPIC_API_KEY": settings.anthropic_api_key}
    if request.model_preset is OrchflowModelPreset.o4_mini_only:
        return {"OPENAI_API_KEY": settings.openai_api_key}
    return {
        "OPENAI_API_KEY": settings.openai_api_key,
        "ANTHROPIC_API_KEY": settings.anthropic_api_key,
    }
