from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.demos.orchflow.schemas import OrchflowRunRequest
from app.demos.orchflow.workflow import encode_ndjson, iter_orchflow_demo_events

router = APIRouter()


@router.post("/run")
async def run_orchflow_demo(request: OrchflowRunRequest) -> StreamingResponse:
    settings = get_settings()
    missing_keys = [
        name
        for name, value in {
            "OPENAI_API_KEY": settings.openai_api_key,
            "ANTHROPIC_API_KEY": settings.anthropic_api_key,
        }.items()
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
