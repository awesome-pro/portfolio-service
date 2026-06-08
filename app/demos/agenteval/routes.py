from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import AsyncIterator

from anyio.to_thread import run_sync as run_sync_in_worker_thread
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.config import Settings, get_settings
from app.demos.agenteval.schemas import AgentEvalProvider, AgentEvalRunRequest
from app.demos.agenteval.workflow import encode_ndjson, iter_agenteval_demo_events

router = APIRouter()


@router.post("/run")
async def run_agenteval_demo(request: AgentEvalRunRequest) -> StreamingResponse:
    settings = get_settings()
    missing_key = None
    if request.provider == "openai" and not settings.openai_api_key:
        missing_key = "OPENAI_API_KEY"
    if request.provider == "anthropic" and not settings.anthropic_api_key:
        missing_key = "ANTHROPIC_API_KEY"
    if missing_key:
        raise HTTPException(
            status_code=503,
            detail=f"Missing LLM API key: {missing_key}",
        )
    await run_sync_in_worker_thread(_validate_provider_credentials, request, settings)

    async def stream() -> AsyncIterator[str]:
        async for event in iter_agenteval_demo_events(request):
            yield encode_ndjson(event)

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store"},
    )


def _validate_provider_credentials(
    request: AgentEvalRunRequest,
    settings: Settings,
) -> None:
    if request.provider is AgentEvalProvider.openai:
        _check_openai_key(settings.openai_api_key or "")
        return
    _check_anthropic_key(settings.anthropic_api_key or "")


def _check_openai_key(api_key: str) -> None:
    http_request = urllib.request.Request(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(http_request, timeout=20):
            return
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise HTTPException(
                status_code=503,
                detail=(
                    "OpenAI authentication failed: the API key was rejected. "
                    "Update OPENAI_API_KEY or DEMOS_API_OPENAI_API_KEY in the "
                    "demo API environment, then restart the FastAPI server."
                ),
            ) from None
        detail = _provider_error_detail(exc)
    except Exception as exc:
        detail = f"OpenAI credential check failed: {type(exc).__name__}"
    raise HTTPException(status_code=503, detail=detail)


def _check_anthropic_key(api_key: str) -> None:
    http_request = urllib.request.Request(
        "https://api.anthropic.com/v1/models",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(http_request, timeout=20):
            return
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Anthropic authentication failed: the API key was rejected. "
                    "Update ANTHROPIC_API_KEY or DEMOS_API_ANTHROPIC_API_KEY in "
                    "the demo API environment, then restart the FastAPI server."
                ),
            ) from None
        detail = _provider_error_detail(exc)
    except Exception as exc:
        detail = f"Anthropic credential check failed: {type(exc).__name__}"
    raise HTTPException(status_code=503, detail=detail)


def _provider_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(exc.read().decode(errors="replace"))
        message = payload.get("error", {}).get("message")
        if isinstance(message, str) and message:
            return f"Provider credential check failed: {message}"
    except Exception:
        pass
    return f"Provider credential check failed with HTTP {exc.code}."
