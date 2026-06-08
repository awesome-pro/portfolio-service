from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import pytest
from app.core.config import get_settings
from app.demos.agenteval import routes as agenteval_routes
from app.main import app
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    get_settings.cache_clear()

    calls: list[dict[str, Any]] = []

    def validate_provider_credentials(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        agenteval_routes,
        "_validate_provider_credentials",
        validate_provider_credentials,
    )

    async def acompletion(**kwargs: Any) -> Any:
        calls.append(kwargs)
        messages = kwargs["messages"]
        system_prompt = messages[0]["content"]
        tool_names = [
            message.get("name")
            for message in messages
            if message.get("role") == "tool"
        ]
        regression = "buggy speed-optimized" in system_prompt

        if not tool_names:
            return _fake_tool_response("lookup_order", {"order_id": "A1007"})

        if regression:
            if "create_support_ticket" not in tool_names:
                return _fake_tool_response(
                    "create_support_ticket",
                    {
                        "order_id": "A1007",
                        "reason": "refund request without policy verification",
                        "priority": "normal",
                    },
                )
            return _fake_text_response(
                "I opened a ticket. A support specialist will review the request."
            )

        if "fetch_refund_policy" not in tool_names:
            return _fake_tool_response(
                "fetch_refund_policy",
                {
                    "country": "US",
                    "item": "Noise cancelling headphones",
                },
            )
        if "create_support_ticket" not in tool_names:
            return _fake_tool_response(
                "create_support_ticket",
                {
                    "order_id": "A1007",
                    "reason": "refund request within policy window",
                    "priority": "normal",
                },
            )
        return _fake_text_response(
            "Your Noise cancelling headphones is eligible for a refund under "
            "the 30-day policy. I created TICKET-NORMAL-A1007."
        )

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(acompletion=acompletion, calls=calls),
    )


async def _stream_events(payload: dict[str, object]) -> list[dict[str, object]]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/demos/agenteval/run",
            json=payload,
        ) as response:
            assert response.status_code == 200
            return [
                json.loads(line)
                async for line in response.aiter_lines()
                if line.strip()
            ]


def _event_types(events: Iterable[dict[str, object]]) -> list[str]:
    return [str(event["type"]) for event in events]


def _fake_tool_response(name: str, arguments: dict[str, object]) -> Any:
    tool_call = SimpleNamespace(
        id=f"call_{name}",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[tool_call])
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=4,
            total_tokens=16,
        ),
    )


def _fake_text_response(content: str) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None)
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=18,
            completion_tokens=8,
            total_tokens=26,
        ),
    )


@pytest.mark.asyncio
async def test_agenteval_openai_healthy_mode_uses_live_provider_path() -> None:
    events = await _stream_events(
        {
            "message": "I want a refund for order A1007",
            "mode": "healthy",
            "provider": "openai",
            "n_runs": 6,
            "threshold": 0.8,
        }
    )

    types = _event_types(events)
    run_events = [event for event in events if str(event["type"]).startswith("run_")]
    final = events[-1]["final_result"]

    assert types[0] == "demo_started"
    assert types[-1] == "gate_passed"
    assert len(run_events) == 6
    assert isinstance(final, dict)
    assert final["provider"] == "openai"
    assert final["n_passed"] == 6
    assert final["met_threshold"] is True
    assert final["exit_code"] == 0
    assert final["traces"][0]["metadata"]["live_llm"] is True
    assert final["traces"][0]["token_usage"]["total_tokens"] == 74


@pytest.mark.asyncio
async def test_agenteval_anthropic_regression_fails_threshold() -> None:
    events = await _stream_events(
        {
            "message": "I want a refund for order A1007",
            "mode": "regression",
            "provider": "anthropic",
            "n_runs": 6,
            "threshold": 0.8,
        }
    )

    final = events[-1]["final_result"]

    assert events[-1]["type"] == "gate_failed"
    assert isinstance(final, dict)
    assert final["provider"] == "anthropic"
    assert final["n_passed"] == 0
    assert final["met_threshold"] is False
    assert final["exit_code"] == 1

    failed_traces = [
        trace
        for trace in final["traces"]
        if isinstance(trace, dict) and trace["passed"] is False
    ]

    assert len(failed_traces) == 6
    assert failed_traces[0]["metadata"]["provider"] == "anthropic"
    assert any(trace["assertion_errors"] for trace in failed_traces)


@pytest.mark.asyncio
async def test_agenteval_run_requires_selected_provider_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("DEMOS_API_ANTHROPIC_API_KEY", "")
    get_settings.cache_clear()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/demos/agenteval/run",
            json={
                "message": "I want a refund for order A1007",
                "mode": "healthy",
                "provider": "anthropic",
                "n_runs": 6,
                "threshold": 0.8,
            },
        )

    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


@pytest.mark.asyncio
async def test_agenteval_run_reports_invalid_provider_key_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_provider_credentials(*args: Any, **kwargs: Any) -> None:
        raise HTTPException(
            status_code=503,
            detail="OpenAI authentication failed: the API key was rejected.",
        )

    monkeypatch.setattr(
        agenteval_routes,
        "_validate_provider_credentials",
        reject_provider_credentials,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/demos/agenteval/run",
            json={
                "message": "I want a refund for order A1007",
                "mode": "healthy",
                "provider": "openai",
                "n_runs": 6,
                "threshold": 0.8,
            },
        )

    assert response.status_code == 503
    assert "OpenAI authentication failed" in response.json()["detail"]
