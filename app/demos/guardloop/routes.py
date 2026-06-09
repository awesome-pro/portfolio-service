from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.core.rate_limit import (
    client_rate_limit_key,
    guardloop_rate_limiter,
    raise_rate_limit_exceeded,
)
from app.demos.guardloop.schemas import GuardLoopExecution, GuardLoopRunRequest
from app.demos.guardloop.workflow import encode_ndjson, iter_guardloop_demo_events

router = APIRouter()


@router.post("/run")
async def run_guardloop_demo(
    request: GuardLoopRunRequest,
    http_request: Request,
) -> StreamingResponse:
    settings = get_settings()
    decision = guardloop_rate_limiter.check(
        client_rate_limit_key(http_request),
        max_runs=settings.guardloop_rate_limit_max_runs,
        window_seconds=settings.guardloop_rate_limit_window_seconds,
    )
    if not decision.allowed:
        raise_rate_limit_exceeded(decision.retry_after_seconds)

    if request.execution is GuardLoopExecution.openai and not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="Missing LLM API key: OPENAI_API_KEY",
        )
    if (
        request.execution is GuardLoopExecution.anthropic
        and not settings.anthropic_api_key
    ):
        raise HTTPException(
            status_code=503,
            detail="Missing LLM API key: ANTHROPIC_API_KEY",
        )

    async def stream() -> AsyncIterator[str]:
        async for event in iter_guardloop_demo_events(request):
            yield encode_ndjson(event)

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store"},
    )
