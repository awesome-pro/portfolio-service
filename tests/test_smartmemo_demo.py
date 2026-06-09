from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import numpy as np
import pytest
from app.core.config import get_settings
from app.core.rate_limit import smartmemo_rate_limiter
from app.demos.smartmemo import workflow as smartmemo_workflow
from app.demos.smartmemo.workflow import SmartMemoRuntime
from app.main import app
from httpx import ASGITransport, AsyncClient
from smartmemo.types import FloatVector


class FixtureEmbeddingProvider:
    dim = 4

    def embed(self, text: str) -> FloatVector:
        if (
            "turned off" in text
            or "cannot be extended" in text
            or "Decrease" in text
        ):
            return np.array([0.95, 0.05, 0, 0], dtype=np.float32)
        return np.array([1, 0, 0, 0], dtype=np.float32)


class FixtureClassifier:
    threshold = 0.95

    def predict_batch(
        self,
        pairs: Sequence[tuple[FloatVector, FloatVector]],
    ) -> list[float]:
        scores = []
        for query_embedding, _candidate_embedding in pairs:
            scores.append(0.2 if float(query_embedding[1]) > 0.01 else 0.99)
        return scores


@pytest.fixture(autouse=True)
def fake_smartmemo_runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    smartmemo_rate_limiter.reset()
    monkeypatch.setenv("DEMOS_API_SMARTMEMO_RATE_LIMIT_MAX_RUNS", "100")
    get_settings.cache_clear()

    runtime = SmartMemoRuntime(
        embedding_provider=FixtureEmbeddingProvider(),
        classifier=FixtureClassifier(),
        use_faiss=False,
    )
    monkeypatch.setattr(smartmemo_workflow, "_get_runtime", lambda: runtime)
    yield
    smartmemo_rate_limiter.reset()
    get_settings.cache_clear()


async def _stream_events(payload: dict[str, object]) -> list[dict[str, Any]]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/demos/smartmemo/run",
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
async def test_smartmemo_opposite_action_blocks_cosine_false_positive() -> None:
    events = await _stream_events(
        {
            "scenario": "debug_logging",
            "query_variant": "opposite_action",
            "cosine_threshold": 0.9,
            "classifier_threshold": 0.95,
            "include_feedback": True,
        }
    )

    types = _event_types(events)
    assert types[:3] == ["demo_started", "runtime_loading", "runtime_ready"]
    assert "cosine_decision" in types
    assert "classifier_decision" in types
    assert "feedback_exported" in types
    assert types[-1] == "demo_completed"

    final = events[-1]["final_result"]
    assert final["comparison"]["cosine_false_positive"] is True
    assert final["comparison"]["blocked_unsafe_reuse"] is True
    assert final["baseline"]["was_cache_hit"] is True
    assert final["baseline"]["unsafe_reuse"] is True
    assert final["smartmemo"]["was_cache_hit"] is False
    assert final["smartmemo"]["query_llm_called"] is True
    assert final["smartmemo"]["classifier_score"] == 0.2

    feedback = final["feedback"]
    assert feedback["recorded"] is True
    assert feedback["label"] == 0
    assert feedback["exported_pairs"] == 1
    exported = json.loads(feedback["jsonl"][0])
    assert exported["label"] == 0
    assert exported["source"] == "smartmemo-feedback"


@pytest.mark.asyncio
async def test_smartmemo_paraphrase_preserves_safe_cache_hit() -> None:
    events = await _stream_events(
        {
            "scenario": "debug_logging",
            "query_variant": "paraphrase",
            "cosine_threshold": 0.9,
            "classifier_threshold": 0.95,
            "include_feedback": True,
        }
    )

    final = events[-1]["final_result"]
    assert final["comparison"]["cosine_false_positive"] is False
    assert final["comparison"]["preserved_safe_reuse"] is True
    assert final["baseline"]["was_cache_hit"] is True
    assert final["smartmemo"]["was_cache_hit"] is True
    assert final["smartmemo"]["safe_reuse"] is True
    assert final["smartmemo"]["query_llm_called"] is False
    assert final["feedback"]["label"] == 1


@pytest.mark.asyncio
async def test_smartmemo_invalid_preset_is_rejected() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/demos/smartmemo/run",
            json={"scenario": "arbitrary_prompt_injection"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_smartmemo_rate_limits_run_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEMOS_API_SMARTMEMO_RATE_LIMIT_MAX_RUNS", "1")
    monkeypatch.setenv("DEMOS_API_SMARTMEMO_RATE_LIMIT_WINDOW_SECONDS", "600")
    get_settings.cache_clear()
    smartmemo_rate_limiter.reset()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post("/demos/smartmemo/run", json={})
        second = await client.post("/demos/smartmemo/run", json={})

    assert first.status_code == 200
    assert second.status_code == 429
    assert "Demo rate limit reached" in second.json()["detail"]


@pytest.mark.asyncio
async def test_smartmemo_stream_sanitizes_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_runtime() -> SmartMemoRuntime:
        raise RuntimeError("bad key sk-proj-secret-secret-secret leaked")

    monkeypatch.setattr(smartmemo_workflow, "_get_runtime", fail_runtime)

    events = await _stream_events({})

    assert events[-1]["type"] == "demo_failed"
    assert "sk-...redacted" in events[-1]["error"]
    assert "sk-proj-secret" not in events[-1]["error"]
