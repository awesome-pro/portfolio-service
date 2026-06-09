from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

import pytest
from app.core.config import get_settings
from app.core.rate_limit import guardloop_rate_limiter
from app.demos.guardloop import workflow as guardloop_workflow
from app.main import app
from httpx import ASGITransport, AsyncClient


@pytest.fixture(autouse=True)
def reset_guardloop_demo(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    guardloop_rate_limiter.reset()
    monkeypatch.setenv("DEMOS_API_GUARDLOOP_RATE_LIMIT_MAX_RUNS", "100")
    monkeypatch.setenv("DEMOS_API_GUARDLOOP_RATE_LIMIT_WINDOW_SECONDS", "600")
    get_settings.cache_clear()
    yield
    guardloop_rate_limiter.reset()
    get_settings.cache_clear()


async def _stream_events(payload: dict[str, object]) -> list[dict[str, Any]]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/demos/guardloop/run",
            json=payload,
        ) as response:
            assert response.status_code == 200
            return [
                json.loads(line)
                async for line in response.aiter_lines()
                if line.strip()
            ]


def _event_types(events: Iterable[dict[str, Any]]) -> list[str]:
    return [str(event["type"]) for event in events]


@pytest.mark.asyncio
async def test_guardloop_budget_scenario_trips_preflight_budget() -> None:
    events = await _stream_events({"scenario": "budget", "policy": "guarded"})
    types = _event_types(events)
    final = events[-1]["final_result"]

    assert types[0] == "demo_started"
    assert types[-1] == "demo_completed"
    assert types.count("llm_call_completed") == 3
    assert "guardrail_tripped" in types
    assert final["scenario"] == "budget"
    assert final["run_result"]["terminated_reason"] == "budget_exceeded"
    assert final["run_result"]["cost_usd"] == "0.01575"
    assert final["guardloop"]["actual_provider_calls"] == 3
    assert any(span["name"] == "agent_run" for span in final["spans"])
    assert any(
        span["name"] == "llm_call openai.responses.create"
        for span in final["spans"]
    )


@pytest.mark.asyncio
async def test_guardloop_circuit_breaker_blocks_after_repeated_tool_failures() -> None:
    events = await _stream_events(
        {"scenario": "circuit_breaker", "policy": "guarded"}
    )
    types = _event_types(events)
    final = events[-1]["final_result"]

    assert types.count("tool_call_failed") == 2
    assert "tool_call_blocked" in types
    assert "guardrail_tripped" in types
    assert final["run_result"]["terminated_reason"] == "circuit_breaker_open"
    assert final["guardloop"]["actual_invocations"] == 2
    assert final["circuit_breakers"]["vendor_search"]["state"] == "open"
    assert any(span["name"] == "tool_call vendor_search" for span in final["spans"])


@pytest.mark.asyncio
async def test_guardloop_verifier_retries_until_output_passes() -> None:
    events = await _stream_events({"scenario": "verifier", "policy": "guarded"})
    types = _event_types(events)
    final = events[-1]["final_result"]

    assert types.count("agent_attempt_completed") == 3
    assert types.count("verifier_checked") == 3
    assert final["run_result"]["success"] is True
    assert final["run_result"]["verification_passed"] is True
    assert final["run_result"]["verification_attempts"] == 3
    assert json.loads(final["run_result"]["output"])["answer"] == 42
    assert any(span["name"].startswith("verifier_run") for span in final["spans"])


@pytest.mark.asyncio
async def test_guardloop_relaxed_policy_shows_unprotected_projection() -> None:
    events = await _stream_events(
        {"scenario": "circuit_breaker", "policy": "relaxed"}
    )
    types = _event_types(events)
    final = events[-1]["final_result"]

    assert types.count("tool_call_failed") == 5
    assert "tool_call_blocked" not in types
    assert "guardrail_tripped" not in types
    assert final["run_result"]["success"] is True
    assert final["run_result"]["tool_calls"] == 5
    assert final["guardloop"]["actual_invocations"] == 5
    assert final["baseline"]["projected_tool_calls"] == 5


@pytest.mark.asyncio
async def test_guardloop_invalid_scenario_is_rejected() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/demos/guardloop/run",
            json={"scenario": "prompt_injection"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_guardloop_live_provider_requires_backend_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("DEMOS_API_OPENAI_API_KEY", "")
    get_settings.cache_clear()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/demos/guardloop/run",
            json={"scenario": "budget", "execution": "openai"},
        )

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


@pytest.mark.asyncio
async def test_guardloop_rate_limits_run_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEMOS_API_GUARDLOOP_RATE_LIMIT_MAX_RUNS", "1")
    monkeypatch.setenv("DEMOS_API_GUARDLOOP_RATE_LIMIT_WINDOW_SECONDS", "600")
    get_settings.cache_clear()
    guardloop_rate_limiter.reset()

    transport = ASGITransport(app=app, client=("203.0.113.44", 1234))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post("/demos/guardloop/run", json={})
        second = await client.post("/demos/guardloop/run", json={})

    assert first.status_code == 200
    assert second.status_code == 429
    assert "Retry-After" in second.headers
    assert "Demo rate limit reached" in second.json()["detail"]


@pytest.mark.asyncio
async def test_guardloop_stream_sanitizes_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_demo(*args: object, **kwargs: object) -> dict[str, Any]:
        raise RuntimeError("bad api_key=sk-proj-secret-secret-secret leaked")

    monkeypatch.setattr(guardloop_workflow, "_run_demo", fail_demo)

    events = await _stream_events({})

    assert events[-1]["type"] == "demo_failed"
    assert "api_key=...redacted" in events[-1]["error"]
    assert "sk-proj-secret" not in events[-1]["error"]
