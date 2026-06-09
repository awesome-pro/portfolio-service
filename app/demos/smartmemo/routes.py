from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.core.rate_limit import (
    client_rate_limit_key,
    raise_rate_limit_exceeded,
    smartmemo_rate_limiter,
)
from app.demos.smartmemo.schemas import SmartMemoRunRequest
from app.demos.smartmemo.workflow import encode_ndjson, iter_smartmemo_demo_events

router = APIRouter()


@router.post("/run")
async def run_smartmemo_demo(
    request: SmartMemoRunRequest,
    http_request: Request,
) -> StreamingResponse:
    settings = get_settings()
    decision = smartmemo_rate_limiter.check(
        client_rate_limit_key(http_request),
        max_runs=settings.smartmemo_rate_limit_max_runs,
        window_seconds=settings.smartmemo_rate_limit_window_seconds,
    )
    if not decision.allowed:
        raise_rate_limit_exceeded(decision.retry_after_seconds)

    async def stream() -> AsyncIterator[str]:
        async for event in iter_smartmemo_demo_events(request):
            yield encode_ndjson(event)

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store"},
    )
