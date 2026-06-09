from __future__ import annotations

# ruff: noqa: E402, I001

import json
import os
import re
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import anyio
from anyio.to_thread import run_sync as run_sync_in_worker_thread

# faiss-cpu and torch can load separate OpenMP runtimes on macOS. Set this
# before importing smartmemo's ML stack so local tests behave like Linux deploys.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from smartmemo import CacheConfig  # noqa: E402
from smartmemo.classifier import ClassifierService  # noqa: E402
from smartmemo.embedding import (  # noqa: E402
    EmbeddingService,
    FaissVectorIndex,
    InMemoryVectorIndex,
    SentenceTransformerEmbeddingProvider,
)
from smartmemo.exceptions import MissingDependencyError  # noqa: E402
from smartmemo.models import CacheResult, CacheStats  # noqa: E402
from smartmemo.orchestrator import CacheOrchestrator  # noqa: E402
from smartmemo.resources import bundled_classifier_path  # noqa: E402
from smartmemo.store import SQLiteCacheStore  # noqa: E402
from smartmemo.types import EmbeddingProvider, EquivalenceClassifier, FloatVector  # noqa: E402

from app.demos.smartmemo.schemas import (  # noqa: E402
    SmartMemoQueryVariant,
    SmartMemoRunRequest,
    SmartMemoScenario,
)

EventEnvelope = dict[str, Any]
MODEL_NAME = "deterministic-demo"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
ESTIMATED_LLM_COST_USD = Decimal("0.002")


@dataclass(frozen=True, slots=True)
class QueryPreset:
    prompt: str
    fresh_response: str
    expected_equivalent: bool


@dataclass(frozen=True, slots=True)
class ScenarioPreset:
    scenario: SmartMemoScenario
    label: str
    domain: str
    seed_prompt: str
    seed_response: str
    variants: dict[SmartMemoQueryVariant, QueryPreset]


@dataclass(frozen=True, slots=True)
class SmartMemoRuntime:
    embedding_provider: EmbeddingProvider
    classifier: EquivalenceClassifier
    use_faiss: bool = True


@dataclass(slots=True)
class DemoCache:
    orchestrator: CacheOrchestrator
    store: SQLiteCacheStore

    def close(self) -> None:
        self.store.close()


class ThresholdedClassifier:
    def __init__(self, classifier: EquivalenceClassifier, threshold: float) -> None:
        self._classifier = classifier
        self.threshold = threshold

    def predict_batch(
        self,
        pairs: Sequence[tuple[FloatVector, FloatVector]],
    ) -> list[float]:
        return self._classifier.predict_batch(pairs)


class DeterministicLLM:
    def __init__(self, *, scenario: ScenarioPreset, query: QueryPreset) -> None:
        self._scenario = scenario
        self._query = query
        self.calls: list[str] = []

    async def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        await anyio.sleep(0.12)
        if prompt == self._scenario.seed_prompt:
            return self._scenario.seed_response
        if prompt == self._query.prompt:
            return self._query.fresh_response
        return f"Fresh deterministic response for: {prompt}"


SCENARIOS: dict[SmartMemoScenario, ScenarioPreset] = {
    SmartMemoScenario.debug_logging: ScenarioPreset(
        scenario=SmartMemoScenario.debug_logging,
        label="Debug logging config",
        domain="software-engineering",
        seed_prompt="Change the config to enable debug logging.",
        seed_response="Debug logging is enabled in the staging configuration.",
        variants={
            SmartMemoQueryVariant.paraphrase: QueryPreset(
                prompt="Update the configuration to allow debug logging to be enabled.",
                fresh_response="Debug logging is enabled in the staging configuration.",
                expected_equivalent=True,
            ),
            SmartMemoQueryVariant.opposite_action: QueryPreset(
                prompt="Revise the configuration so that debug logging is turned off.",
                fresh_response=(
                    "Debug logging is disabled in the staging configuration."
                ),
                expected_equivalent=False,
            ),
        },
    ),
    SmartMemoScenario.web_scaling: ScenarioPreset(
        scenario=SmartMemoScenario.web_scaling,
        label="Web service scaling",
        domain="devops",
        seed_prompt="Increase the scale of the web service.",
        seed_response="The web service capacity has been increased.",
        variants={
            SmartMemoQueryVariant.paraphrase: QueryPreset(
                prompt="Boost the web service to a larger scale.",
                fresh_response="The web service capacity has been increased.",
                expected_equivalent=True,
            ),
            SmartMemoQueryVariant.opposite_action: QueryPreset(
                prompt="Decrease the size of the web service.",
                fresh_response="The web service capacity has been reduced.",
                expected_equivalent=False,
            ),
        },
    ),
    SmartMemoScenario.trial_extension: ScenarioPreset(
        scenario=SmartMemoScenario.trial_extension,
        label="Customer trial extension",
        domain="customer-support",
        seed_prompt="Reply to the customer agreeing to extend the trial period.",
        seed_response="Your trial period has been extended as requested.",
        variants={
            SmartMemoQueryVariant.paraphrase: QueryPreset(
                prompt=(
                    "Respond to the customer by agreeing to extend their trial period."
                ),
                fresh_response="Your trial period has been extended as requested.",
                expected_equivalent=True,
            ),
            SmartMemoQueryVariant.opposite_action: QueryPreset(
                prompt="Inform the customer that the trial period cannot be extended.",
                fresh_response="We cannot extend the trial period for this account.",
                expected_equivalent=False,
            ),
        },
    ),
}


def encode_ndjson(payload: EventEnvelope) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n"


async def iter_smartmemo_demo_events(
    request: SmartMemoRunRequest,
) -> AsyncIterator[EventEnvelope]:
    started_at = time.perf_counter()
    run_id = f"smartmemo-demo-{time.time_ns()}"
    scenario = SCENARIOS[request.scenario]
    query = scenario.variants[request.query_variant]

    yield {
        "type": "demo_started",
        "run_id": run_id,
        "timestamp": time.time(),
        "config": _request_config(request, scenario, query),
        "scenarios": _scenario_catalog(),
    }

    try:
        yield {
            "type": "runtime_loading",
            "run_id": run_id,
            "timestamp": time.time(),
            "runtime": _runtime_payload(loaded=False),
        }
        runtime = await run_sync_in_worker_thread(_get_runtime)
        yield {
            "type": "runtime_ready",
            "run_id": run_id,
            "timestamp": time.time(),
            "runtime": _runtime_payload(loaded=True),
        }
        await anyio.sleep(0.08)

        with TemporaryDirectory(prefix="smartmemo-demo-") as directory:
            result = await _run_demo(
                request=request,
                scenario=scenario,
                query=query,
                runtime=runtime,
                directory=Path(directory),
                run_id=run_id,
            )
            for event in result["events"]:
                yield event
                await anyio.sleep(0.08)

            yield {
                "type": "demo_completed",
                "run_id": run_id,
                "timestamp": time.time(),
                "final_result": {
                    **result["final_result"],
                    "duration_seconds": time.perf_counter() - started_at,
                },
            }
    except Exception as exc:
        yield {
            "type": "demo_failed",
            "run_id": run_id,
            "timestamp": time.time(),
            "error": _sanitize_error(exc),
        }


async def _run_demo(
    *,
    request: SmartMemoRunRequest,
    scenario: ScenarioPreset,
    query: QueryPreset,
    runtime: SmartMemoRuntime,
    directory: Path,
    run_id: str,
) -> dict[str, Any]:
    events: list[EventEnvelope] = []
    baseline = _make_cache(
        runtime=runtime,
        db_path=directory / "cosine-only.db",
        domain=scenario.domain,
        cosine_threshold=request.cosine_threshold,
        classifier=None,
    )
    guarded = _make_cache(
        runtime=runtime,
        db_path=directory / "classifier-gated.db",
        domain=scenario.domain,
        cosine_threshold=request.cosine_threshold,
        classifier=ThresholdedClassifier(
            runtime.classifier,
            threshold=request.classifier_threshold,
        ),
    )
    baseline_llm = DeterministicLLM(scenario=scenario, query=query)
    guarded_llm = DeterministicLLM(scenario=scenario, query=query)
    feedback_payload: dict[str, Any] | None = None

    try:
        events.append(
            {
                "type": "cache_seed_started",
                "run_id": run_id,
                "timestamp": time.time(),
                "seed_prompt": scenario.seed_prompt,
            }
        )
        baseline_seed = await baseline.orchestrator.get_or_call(
            prompt=scenario.seed_prompt,
            llm_function=baseline_llm,
            model=MODEL_NAME,
            metadata={"demo_branch": "cosine_only", "role": "seed"},
        )
        guarded_seed = await guarded.orchestrator.get_or_call(
            prompt=scenario.seed_prompt,
            llm_function=guarded_llm,
            model=MODEL_NAME,
            metadata={"demo_branch": "classifier_gated", "role": "seed"},
        )
        events.append(
            {
                "type": "cache_seeded",
                "run_id": run_id,
                "timestamp": time.time(),
                "seed_prompt": scenario.seed_prompt,
                "baseline": _cache_result_payload(baseline_seed),
                "smartmemo": _cache_result_payload(guarded_seed),
            }
        )

        events.append(
            {
                "type": "lookup_started",
                "run_id": run_id,
                "timestamp": time.time(),
                "branch": "cosine_only",
                "query_prompt": query.prompt,
            }
        )
        baseline_before = len(baseline_llm.calls)
        baseline_result = await baseline.orchestrator.get_or_call(
            prompt=query.prompt,
            llm_function=baseline_llm,
            model=MODEL_NAME,
            metadata={"demo_branch": "cosine_only", "role": "query"},
        )
        baseline_query_called = len(baseline_llm.calls) > baseline_before
        baseline_decision = _decision_payload(
            result=baseline_result,
            query=query,
            query_llm_called=baseline_query_called,
            branch="cosine_only",
            threshold=request.cosine_threshold,
        )
        events.append(
            {
                "type": "cosine_decision",
                "run_id": run_id,
                "timestamp": time.time(),
                "decision": baseline_decision,
            }
        )

        if request.include_feedback and baseline_result.was_cache_hit:
            feedback_payload = _record_feedback(
                cache=baseline,
                result=baseline_result,
                query=query,
                directory=directory,
            )
            events.append(
                {
                    "type": "feedback_exported",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "feedback": feedback_payload,
                }
            )

        events.append(
            {
                "type": "lookup_started",
                "run_id": run_id,
                "timestamp": time.time(),
                "branch": "classifier_gated",
                "query_prompt": query.prompt,
            }
        )
        guarded_before = len(guarded_llm.calls)
        guarded_result = await guarded.orchestrator.get_or_call(
            prompt=query.prompt,
            llm_function=guarded_llm,
            model=MODEL_NAME,
            metadata={"demo_branch": "classifier_gated", "role": "query"},
        )
        guarded_query_called = len(guarded_llm.calls) > guarded_before
        guarded_decision = _decision_payload(
            result=guarded_result,
            query=query,
            query_llm_called=guarded_query_called,
            branch="classifier_gated",
            threshold=request.classifier_threshold,
        )
        events.append(
            {
                "type": "classifier_decision",
                "run_id": run_id,
                "timestamp": time.time(),
                "decision": guarded_decision,
            }
        )

        final_result = {
            "config": _request_config(request, scenario, query),
            "seed": {
                "prompt": scenario.seed_prompt,
                "response": scenario.seed_response,
            },
            "query": {
                "prompt": query.prompt,
                "fresh_response": query.fresh_response,
                "expected_equivalent": query.expected_equivalent,
            },
            "baseline": {
                **baseline_decision,
                "stats": _stats_payload(baseline.orchestrator.stats(), baseline_llm),
            },
            "smartmemo": {
                **guarded_decision,
                "stats": _stats_payload(guarded.orchestrator.stats(), guarded_llm),
            },
            "comparison": _comparison_payload(
                query=query,
                baseline=baseline_result,
                smartmemo=guarded_result,
            ),
            "feedback": feedback_payload,
        }
        return {"events": events, "final_result": final_result}
    finally:
        baseline.close()
        guarded.close()


@lru_cache(maxsize=1)
def _get_runtime() -> SmartMemoRuntime:
    embedding_provider = SentenceTransformerEmbeddingProvider(
        model_name=EMBEDDING_MODEL,
        dim=EMBEDDING_DIM,
    )
    classifier = ClassifierService(bundled_classifier_path(), threshold=0.95)
    return SmartMemoRuntime(
        embedding_provider=embedding_provider,
        classifier=classifier,
        use_faiss=True,
    )


def _make_cache(
    *,
    runtime: SmartMemoRuntime,
    db_path: Path,
    domain: str,
    cosine_threshold: float,
    classifier: EquivalenceClassifier | None,
) -> DemoCache:
    store = SQLiteCacheStore(db_path)
    embedding_service = EmbeddingService(
        runtime.embedding_provider,
        _make_index(runtime),
    )
    orchestrator = CacheOrchestrator(
        domain=domain,
        config=CacheConfig(
            db_path=db_path,
            embedding_dim=runtime.embedding_provider.dim,
            candidate_k=3,
            cosine_threshold=cosine_threshold,
            estimated_llm_cost_usd=ESTIMATED_LLM_COST_USD,
        ),
        store=store,
        embedding_service=embedding_service,
        classifier_service=classifier,
    )
    return DemoCache(orchestrator=orchestrator, store=store)


def _make_index(runtime: SmartMemoRuntime) -> FaissVectorIndex | InMemoryVectorIndex:
    if runtime.use_faiss:
        try:
            return FaissVectorIndex(runtime.embedding_provider.dim)
        except MissingDependencyError:
            return InMemoryVectorIndex(runtime.embedding_provider.dim)
    return InMemoryVectorIndex(runtime.embedding_provider.dim)


def _record_feedback(
    *,
    cache: DemoCache,
    result: CacheResult,
    query: QueryPreset,
    directory: Path,
) -> dict[str, Any]:
    if query.expected_equivalent:
        recorded = cache.orchestrator.report_good_hit(result.query_id)
        label = 1
        reason = "demo: safe equivalent cache hit"
    else:
        recorded = cache.orchestrator.report_bad_hit(
            result.query_id,
            reason="demo: cosine-only false positive",
        )
        label = 0
        reason = "demo: cosine-only false positive"

    output_path = directory / "feedback_pairs.jsonl"
    exported = cache.orchestrator.export_feedback_pairs(str(output_path), split="train")
    lines = output_path.read_text().splitlines() if output_path.exists() else []
    return {
        "recorded": recorded,
        "label": label,
        "reason": reason,
        "exported_pairs": exported,
        "jsonl": lines[:3],
    }


def _request_config(
    request: SmartMemoRunRequest,
    scenario: ScenarioPreset,
    query: QueryPreset,
) -> dict[str, Any]:
    return {
        "scenario": request.scenario.value,
        "scenario_label": scenario.label,
        "domain": scenario.domain,
        "query_variant": request.query_variant.value,
        "cosine_threshold": request.cosine_threshold,
        "classifier_threshold": request.classifier_threshold,
        "include_feedback": request.include_feedback,
        "expected_equivalent": query.expected_equivalent,
        "package": "smartmemo[ml]",
        "embedding_model": EMBEDDING_MODEL,
        "classifier": "classifier-v2",
        "command": (
            "python demo.py --scenario "
            f"{request.scenario.value} --variant {request.query_variant.value}"
        ),
    }


def _runtime_payload(*, loaded: bool) -> dict[str, Any]:
    return {
        "loaded": loaded,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "classifier": "classifier-v2",
        "vector_index": "faiss-cpu",
    }


def _scenario_catalog() -> list[dict[str, Any]]:
    return [
        {
            "id": scenario.scenario.value,
            "label": scenario.label,
            "domain": scenario.domain,
            "seed_prompt": scenario.seed_prompt,
            "variants": {
                key.value: {
                    "prompt": value.prompt,
                    "expected_equivalent": value.expected_equivalent,
                }
                for key, value in scenario.variants.items()
            },
        }
        for scenario in SCENARIOS.values()
    ]


def _cache_result_payload(result: CacheResult) -> dict[str, Any]:
    return {
        "query_id": str(result.query_id),
        "cache_entry_id": str(result.cache_entry_id) if result.cache_entry_id else None,
        "was_cache_hit": result.was_cache_hit,
        "response": result.response,
        "similarity_score": _round_optional(result.similarity_score),
        "classifier_score": _round_optional(result.classifier_score),
        "latency_ms": round(result.latency_ms, 1),
        "cost_saved_usd": float(result.cost_saved_usd),
    }


def _decision_payload(
    *,
    result: CacheResult,
    query: QueryPreset,
    query_llm_called: bool,
    branch: str,
    threshold: float,
) -> dict[str, Any]:
    unsafe_reuse = result.was_cache_hit and not query.expected_equivalent
    safe_reuse = result.was_cache_hit and query.expected_equivalent
    return {
        **_cache_result_payload(result),
        "branch": branch,
        "threshold": threshold,
        "query_llm_called": query_llm_called,
        "expected_equivalent": query.expected_equivalent,
        "safe_reuse": safe_reuse,
        "unsafe_reuse": unsafe_reuse,
        "decision_label": _decision_label(
            result=result,
            query=query,
            branch=branch,
        ),
    }


def _decision_label(
    *,
    result: CacheResult,
    query: QueryPreset,
    branch: str,
) -> str:
    if result.was_cache_hit and query.expected_equivalent:
        return "safe cache hit"
    if result.was_cache_hit and not query.expected_equivalent:
        return "unsafe cache hit"
    if branch == "classifier_gated" and not query.expected_equivalent:
        return "blocked unsafe reuse"
    return "cache miss"


def _comparison_payload(
    *,
    query: QueryPreset,
    baseline: CacheResult,
    smartmemo: CacheResult,
) -> dict[str, Any]:
    return {
        "cosine_only_would_reuse": baseline.was_cache_hit,
        "smartmemo_reused": smartmemo.was_cache_hit,
        "blocked_unsafe_reuse": (
            baseline.was_cache_hit
            and not smartmemo.was_cache_hit
            and not query.expected_equivalent
        ),
        "preserved_safe_reuse": (
            baseline.was_cache_hit
            and smartmemo.was_cache_hit
            and query.expected_equivalent
        ),
        "cosine_false_positive": (
            baseline.was_cache_hit and not query.expected_equivalent
        ),
    }


def _stats_payload(stats: CacheStats, llm: DeterministicLLM) -> dict[str, Any]:
    return {
        "total_entries": stats.total_entries,
        "total_lookups": stats.total_lookups,
        "cache_hits": stats.cache_hits,
        "cache_misses": stats.cache_misses,
        "hit_rate": stats.hit_rate,
        "llm_calls": len(llm.calls),
        "total_cost_saved_usd": float(stats.total_cost_saved_usd),
    }


def _round_optional(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _sanitize_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    text = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "sk-...redacted", text)
    return text[:300]
