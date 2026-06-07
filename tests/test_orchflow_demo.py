from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import pytest
from app.core.config import get_settings
from app.main import app
from httpx import ASGITransport, AsyncClient


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    get_settings.cache_clear()

    calls: list[dict[str, Any]] = []

    async def acompletion(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        prompt = kwargs["messages"][1]["content"]
        content = _fake_json_content(prompt)
        return {"choices": [{"message": {"content": json.dumps(content)}}]}

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(acompletion=acompletion, calls=calls),
    )


async def _stream_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/demos/orchflow/run",
            json=payload,
        ) as response:
            assert response.status_code == 200
            return [
                json.loads(line)
                async for line in response.aiter_lines()
                if line.strip()
            ]


def _event_types(events: Iterable[dict[str, Any]]) -> list[str]:
    return [event["type"] for event in events]


def _fake_json_content(prompt: str) -> dict[str, Any]:
    if "TASK: plan" in prompt:
        return {
            "topic": "AI code review assistant",
            "audience": "engineering managers",
            "sections": ["positioning", "proof", "risk"],
            "tone": "practical",
            "research_questions": ["what hurts", "what proves value"],
            "model_note": "fake-haiku",
        }
    if "TASK: market_research" in prompt:
        return {
            "angle": "Readable workflows make agent demos easier to trust.",
            "pain": "Hidden orchestration makes failures hard to explain.",
            "audience_fit": "Engineering managers can inspect the run.",
            "signals": ["trace visibility", "resume story"],
            "model_note": "fake-haiku",
        }
    if "TASK: technical_research" in prompt:
        return {
            "core_value": "Steps remain normal Python functions.",
            "mechanics": ["parallel fan-out", "conditions", "checkpoint resume"],
            "dependency_story": "Core Orchflow has zero runtime dependencies.",
            "implementation_note": "Events stream as the flow executes.",
            "model_note": "fake-haiku",
        }
    if "TASK: risk_review" in prompt:
        return {
            "risk": "Instant output can look like static demo data.",
            "mitigation": "Use real model calls and show lifecycle events.",
            "review_score": 0.92,
            "model_note": "fake-haiku",
        }
    if "TASK: synthesize" in prompt:
        return {
            "headline": "Launch brief for AI code review assistant",
            "positioning": "A readable live pipeline for agent work.",
            "proof": ["parallel branches", "flat traces", "checkpoint resume"],
            "risk": "Demo may feel fake without latency.",
            "mitigation": "Use real Haiku and o4-mini calls.",
            "quality_score": 0.91,
            "model_note": "fake-o4-mini",
        }
    if "TASK: publish_ready" in prompt:
        return {
            "status": "publish_ready",
            "summary": "A live Orchflow run turns the project page into proof.",
            "why_orchflow": [
                "It keeps the pipeline readable.",
                "It exposes parallel work as flat traces.",
                "It resumes from JSON checkpoints.",
            ],
            "model_note": "fake-o4-mini",
        }
    if "TASK: revise" in prompt:
        return {
            "status": "needs_revision",
            "summary": "The brief needs another pass.",
            "why_orchflow": ["Readable steps", "Traceable branches", "Resume support"],
            "model_note": "fake-haiku",
        }
    raise AssertionError(f"Unexpected prompt: {prompt}")


@pytest.mark.asyncio
async def test_healthz() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_orchflow_success_streams_parallel_traces_and_result() -> None:
    events = await _stream_events(
        {
            "topic": "AI code review assistant",
            "audience": "engineering managers",
            "mode": "success",
        }
    )

    types = _event_types(events)
    assert types[0] == "flow_started"
    assert types[-1] == "flow_completed"

    parallel_events = [
        event
        for event in events
        if event["step_name"]
        in {"market_research", "technical_research", "risk_review"}
        and event["type"] == "step_completed"
    ]
    group_ids = {event["parallel_group_id"] for event in parallel_events}

    assert len(parallel_events) == 3
    assert len(group_ids) == 1
    assert None not in group_ids

    final_result = events[-1]["final_result"]
    assert final_result["success"] is True
    assert final_result["output"]["status"] == "publish_ready"
    assert final_result["output"]["audience"] == "engineering managers"


@pytest.mark.asyncio
async def test_orchflow_failure_resume_streams_checkpoint_lifecycle() -> None:
    events = await _stream_events(
        {
            "topic": "Observability for agent workflows",
            "audience": "founders",
            "mode": "failure_resume",
        }
    )

    types = _event_types(events)
    phases = {event["phase"] for event in events}

    assert "initial" in phases
    assert "resume" in phases
    assert "flow_failed" in types
    assert "checkpoint_saved" in types
    assert "checkpoint_loaded" in types
    assert types[-1] == "flow_completed"

    failed = next(event for event in events if event["type"] == "flow_failed")
    assert failed["error"] == "simulated deploy interruption after research"

    final_result = events[-1]["final_result"]
    trace_names = [trace["step_name"] for trace in final_result["traces"]]

    assert final_result["success"] is True
    assert trace_names.count("plan") == 1
    assert trace_names.count("synthesize") == 3
    assert final_result["output"]["status"] == "publish_ready"


@pytest.mark.asyncio
async def test_orchflow_run_requires_real_provider_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/demos/orchflow/run",
            json={
                "topic": "AI code review assistant",
                "audience": "engineering managers",
                "mode": "success",
            },
        )

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]
