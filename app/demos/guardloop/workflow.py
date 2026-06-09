from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol, cast

import anyio
from guardloop import (
    BudgetConfig,
    CircuitBreakerConfig,
    CircuitBreakerPolicy,
    GuardLoop,
    RunContext,
    RunResult,
    TelemetryConfig,
    VerifierConfig,
    VerifierContext,
    VerifierResult,
    is_json_object,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.core.config import Settings, get_settings
from app.demos.guardloop.schemas import (
    GuardLoopExecution,
    GuardLoopPolicy,
    GuardLoopRunRequest,
    GuardLoopScenario,
)

EventEnvelope = dict[str, Any]

NO_KEY_MODEL = "gpt-5.2"
PROMPT = "Investigate agent runtime safety."
TOOL_NAME = "vendor_search"


@dataclass(slots=True)
class FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass(slots=True)
class FakeResponse:
    output_text: str
    usage: FakeUsage


class HasOutputText(Protocol):
    output_text: str


class FakeOpenAIResponses:
    def __init__(self, *, input_tokens: int = 600, output_tokens: int = 300) -> None:
        self.calls = 0
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    async def create(self, **_: object) -> FakeResponse:
        self.calls += 1
        await anyio.sleep(0.08)
        return FakeResponse(
            output_text=f"loop iteration {self.calls}: still researching...",
            usage=FakeUsage(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
            ),
        )


class FakeOpenAIClient:
    def __init__(self, responses: FakeOpenAIResponses | None = None) -> None:
        self.responses = responses or FakeOpenAIResponses()


def encode_ndjson(payload: EventEnvelope) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n"


async def iter_guardloop_demo_events(
    request: GuardLoopRunRequest,
) -> AsyncIterator[EventEnvelope]:
    started_at = time.perf_counter()
    run_id = f"guardloop-demo-{time.time_ns()}"

    yield {
        "type": "demo_started",
        "run_id": run_id,
        "timestamp": time.time(),
        "config": _request_config(request),
        "scenarios": _scenario_catalog(),
    }

    try:
        demo = await _run_demo(request, run_id)
        for event in demo["events"]:
            yield event
            await anyio.sleep(0.07)

        final_result = {
            **demo["final_result"],
            "duration_seconds": time.perf_counter() - started_at,
        }
        yield {
            "type": "demo_completed",
            "run_id": run_id,
            "timestamp": time.time(),
            "final_result": final_result,
        }
    except Exception as exc:
        yield {
            "type": "demo_failed",
            "run_id": run_id,
            "timestamp": time.time(),
            "error": _sanitize_error(exc),
        }


async def _run_demo(request: GuardLoopRunRequest, run_id: str) -> dict[str, Any]:
    if request.scenario is GuardLoopScenario.budget:
        return await _run_budget_demo(request, run_id)
    if request.scenario is GuardLoopScenario.circuit_breaker:
        return await _run_circuit_breaker_demo(request, run_id)
    return await _run_verifier_demo(request, run_id)


async def _run_budget_demo(
    request: GuardLoopRunRequest,
    run_id: str,
) -> dict[str, Any]:
    events: list[EventEnvelope] = []
    settings = get_settings()
    exporter, tracer = _tracer_pair()
    fake_responses = (
        FakeOpenAIResponses()
        if request.execution is GuardLoopExecution.no_key
        else None
    )
    max_iterations = 8 if request.policy is GuardLoopPolicy.guarded else 5
    provider_calls = 0
    model = _model_for_request(request, settings)
    cost_limit = _budget_cost_limit(request)
    runtime = GuardLoop(
        budget=BudgetConfig(
            cost_limit_usd=cost_limit,
            token_limit=10_000,
            time_limit_seconds=10,
            tool_call_limit=10,
        ),
        telemetry=TelemetryConfig(enabled=True),
        openai_client=_openai_client_for_request(request, settings, fake_responses),
        anthropic_client=_anthropic_client_for_request(request, settings),
        tracer=tracer,
    )

    async def runaway_agent(ctx: RunContext, topic: str) -> str:
        nonlocal provider_calls
        outputs: list[str] = []
        for iteration in range(1, max_iterations + 1):
            text = await _run_budget_llm_call(
                ctx=ctx,
                request=request,
                model=model,
                topic=topic,
            )
            provider_calls += 1
            outputs.append(text)
            events.append(
                _event(
                    run_id,
                    "llm_call_completed",
                    step_key=f"llm-{iteration}",
                    title=f"LLM call {iteration}",
                    status="completed",
                    detail=text,
                    metrics={
                        "execution": request.execution.value,
                        "model": model,
                        "actual_calls": provider_calls,
                    },
                )
            )
        return "\n".join(outputs)

    result = await runtime.run(runaway_agent, PROMPT)
    if result.terminated_reason:
        events.append(_guardrail_event(run_id, result))

    return _demo_payload(
        request=request,
        run_id=run_id,
        title="Runaway budget stop",
        result=result,
        events=events,
        baseline={
            "label": "Unprotected agent",
            "outcome": "would keep calling the model until external code stops it",
            "projected_calls": 5,
            "projected_cost_usd": _format_decimal(
                _estimated_budget_call_cost(request, model) * Decimal(5)
            ),
            "risk": (
                "The next call is sent even when it crosses the intended spend "
                "ceiling."
            ),
        },
        guardloop={
            "label": "GuardLoop runtime",
            "outcome": (
                "blocked before the next model request"
                if result.terminated_reason
                else "completed inside the relaxed budget"
            ),
            "saved_calls": max(0, 5 - provider_calls),
            "actual_provider_calls": provider_calls,
            "execution": request.execution.value,
            "model": model,
        },
        spans=_span_payloads(exporter),
        code=_code_snippet(request.scenario, request.policy),
    )


async def _run_circuit_breaker_demo(
    request: GuardLoopRunRequest,
    run_id: str,
) -> dict[str, Any]:
    events: list[EventEnvelope] = []
    exporter, tracer = _tracer_pair()
    tool_invocations = 0
    runtime = GuardLoop(
        budget=BudgetConfig(tool_call_limit=10, time_limit_seconds=10),
        telemetry=TelemetryConfig(enabled=True),
        circuit_breakers=CircuitBreakerConfig(
            enabled=request.policy is GuardLoopPolicy.guarded,
            default=CircuitBreakerPolicy(
                failure_threshold=2,
                recovery_timeout_seconds=30,
            ),
        ),
        tracer=tracer,
    )

    async def flaky_search(query: str) -> str:
        nonlocal tool_invocations
        tool_invocations += 1
        await anyio.sleep(0.05)
        raise RuntimeError(f"upstream search API returned HTTP 503 for {query}")

    async def agent(ctx: RunContext) -> str:
        for attempt in range(1, 6):
            try:
                await ctx.call_tool(TOOL_NAME, flaky_search, "agent runtime safety")
            except RuntimeError as exc:
                events.append(
                    _event(
                        run_id,
                        "tool_call_failed",
                        step_key=f"tool-{attempt}",
                        title=f"Tool attempt {attempt}",
                        status="failed",
                        detail=str(exc),
                        metrics={"actual_invocations": tool_invocations},
                    )
                )
                continue
        return "No useful answer after five failed vendor_search calls."

    result = await runtime.run(agent)
    if isinstance(result.error_type, str) and result.error_type == "CircuitBreakerOpen":
        events.append(
            _event(
                run_id,
                "tool_call_blocked",
                step_key="breaker-open",
                title="Circuit breaker opened",
                status="blocked",
                detail=(
                    result.error_message
                    or "Tool call blocked before user code ran."
                ),
                metrics={
                    "actual_invocations": tool_invocations,
                    "tool_calls_charged": result.tool_calls,
                },
            )
        )
    if result.terminated_reason:
        events.append(_guardrail_event(run_id, result))

    snapshots = {
        name: snapshot.model_dump(mode="json")
        for name, snapshot in runtime.circuit_breaker_snapshots().items()
    }
    return _demo_payload(
        request=request,
        run_id=run_id,
        title="Flaky tool circuit breaker",
        result=result,
        events=events,
        baseline={
            "label": "Unprotected agent",
            "outcome": "keeps retrying the same failing vendor_search tool",
            "projected_tool_calls": 5,
            "risk": (
                "Repeated failures create noisy traces and can amplify upstream "
                "incidents."
            ),
        },
        guardloop={
            "label": "GuardLoop runtime",
            "outcome": (
                "opened the per-tool breaker and rejected the next call"
                if result.terminated_reason
                else "allowed all retries because the relaxed policy disables breakers"
            ),
            "actual_invocations": tool_invocations,
            "tool_calls_charged": result.tool_calls,
        },
        spans=_span_payloads(exporter),
        code=_code_snippet(request.scenario, request.policy),
        circuit_breakers=snapshots,
    )


async def _run_verifier_demo(
    request: GuardLoopRunRequest,
    run_id: str,
) -> dict[str, Any]:
    attempt_events: list[EventEnvelope] = []
    exporter, tracer = _tracer_pair()
    runtime = GuardLoop(
        budget=BudgetConfig(time_limit_seconds=10, tool_call_limit=10),
        telemetry=TelemetryConfig(enabled=True),
        verifiers=(
            [no_todo_placeholder, is_json_object(required_keys=["answer"])]
            if request.policy is GuardLoopPolicy.guarded
            else []
        ),
        verifier_config=VerifierConfig(max_retries=2),
        tracer=tracer,
    )

    async def agent(ctx: RunContext, question: str) -> str:
        await anyio.sleep(0.05)
        if ctx.attempt == 1:
            output = '{"answer": "TODO"}'
        elif ctx.attempt == 2:
            output = "answer = 42"
        else:
            output = json.dumps({"answer": 42, "question": question})
        attempt_events.append(
            _event(
                run_id,
                "agent_attempt_completed",
                step_key=f"attempt-{ctx.attempt}",
                title=f"Agent attempt {ctx.attempt}",
                status="completed",
                detail=output,
                metrics={"retry_feedback": list(ctx.retry_feedback)},
            )
        )
        return output

    result = await runtime.run(agent, "what is six times seven?")
    events = _verifier_timeline(run_id, attempt_events, result, request.policy)
    if result.terminated_reason:
        events.append(_guardrail_event(run_id, result))

    return _demo_payload(
        request=request,
        run_id=run_id,
        title="Verifier self-repair loop",
        result=result,
        events=events,
        baseline={
            "label": "Unprotected agent",
            "outcome": "ships the first answer without checking the TODO placeholder",
            "first_output": '{"answer": "TODO"}',
            "risk": "A syntactically valid response can still be untrusted.",
        },
        guardloop={
            "label": "GuardLoop runtime",
            "outcome": (
                "fed verifier feedback back into the agent until JSON passed"
                if result.verification_passed
                else "returned an unverified result"
            ),
            "verification_attempts": result.verification_attempts,
            "feedback": list(result.verification_feedback),
        },
        spans=_span_payloads(exporter),
        code=_code_snippet(request.scenario, request.policy),
    )


def no_todo_placeholder(output: object, ctx: VerifierContext) -> VerifierResult:
    if "TODO" in str(output):
        return VerifierResult(
            passed=False,
            feedback="The answer still contains a TODO placeholder; use a real value.",
        )
    return VerifierResult(passed=True)


async def _run_budget_llm_call(
    *,
    ctx: RunContext,
    request: GuardLoopRunRequest,
    model: str,
    topic: str,
) -> str:
    prompt = (
        f"{topic}\n"
        "You are inside a bounded runtime demo. Produce one concise research "
        "note and say whether another model call would normally be tempting."
    )
    if request.execution is GuardLoopExecution.anthropic:
        message = await ctx.anthropic.messages.create(
            model=model,
            max_tokens=90,
            messages=[{"role": "user", "content": prompt}],
        )
        return _anthropic_content_text(message)

    max_output_tokens = 500 if request.execution is GuardLoopExecution.no_key else 90
    response = await ctx.openai.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_output_tokens,
    )
    return cast(HasOutputText, response).output_text


def _openai_client_for_request(
    request: GuardLoopRunRequest,
    settings: Settings,
    fake_responses: FakeOpenAIResponses | None,
) -> object | None:
    if request.execution is GuardLoopExecution.no_key:
        return FakeOpenAIClient(fake_responses or FakeOpenAIResponses())
    if request.execution is GuardLoopExecution.openai:
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key=settings.openai_api_key)
    return None


def _anthropic_client_for_request(
    request: GuardLoopRunRequest,
    settings: Settings,
) -> object | None:
    if request.execution is not GuardLoopExecution.anthropic:
        return None
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(api_key=settings.anthropic_api_key)


def _model_for_request(request: GuardLoopRunRequest, settings: Settings) -> str:
    if request.execution is GuardLoopExecution.openai:
        return settings.guardloop_openai_model
    if request.execution is GuardLoopExecution.anthropic:
        return settings.guardloop_anthropic_model
    return NO_KEY_MODEL


def _budget_cost_limit(request: GuardLoopRunRequest) -> str:
    if request.execution is GuardLoopExecution.no_key:
        return "0.02" if request.policy is GuardLoopPolicy.guarded else "0.06"
    if request.execution is GuardLoopExecution.openai:
        return "0.00016" if request.policy is GuardLoopPolicy.guarded else "0.001"
    return "0.00030" if request.policy is GuardLoopPolicy.guarded else "0.0015"


def _estimated_budget_call_cost(request: GuardLoopRunRequest, model: str) -> Decimal:
    if request.execution is GuardLoopExecution.no_key:
        return _call_cost()
    if request.execution is GuardLoopExecution.openai:
        return Decimal("0.00007")
    if request.execution is GuardLoopExecution.anthropic:
        return Decimal("0.00013")
    raise ValueError(f"Unsupported execution mode for model {model}")


def _anthropic_content_text(message: object) -> str:
    blocks = getattr(message, "content", [])
    texts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            texts.append(text)
    return "\n".join(texts) or "Anthropic returned an empty text response."


def _verifier_timeline(
    run_id: str,
    attempt_events: list[EventEnvelope],
    result: RunResult,
    policy: GuardLoopPolicy,
) -> list[EventEnvelope]:
    if policy is GuardLoopPolicy.relaxed:
        return attempt_events

    timeline: list[EventEnvelope] = []
    feedback = list(result.verification_feedback)
    for index, attempt_event in enumerate(attempt_events, 1):
        timeline.append(attempt_event)
        if index <= len(feedback):
            timeline.append(
                _event(
                    run_id,
                    "verifier_checked",
                    step_key=f"verifier-{index}",
                    title=f"Verifier rejected attempt {index}",
                    status="failed",
                    detail=feedback[index - 1],
                    metrics={"attempt": index},
                )
            )
        else:
            timeline.append(
                _event(
                    run_id,
                    "verifier_checked",
                    step_key=f"verifier-{index}",
                    title=f"Verifier passed attempt {index}",
                    status="completed",
                    detail="All configured verifiers passed.",
                    metrics={"attempt": index},
                )
            )
    return timeline


def _demo_payload(
    *,
    request: GuardLoopRunRequest,
    run_id: str,
    title: str,
    result: RunResult,
    events: list[EventEnvelope],
    baseline: dict[str, Any],
    guardloop: dict[str, Any],
    spans: list[dict[str, Any]],
    code: str,
    circuit_breakers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "events": events,
        "final_result": {
            "run_id": run_id,
            "title": title,
            "scenario": request.scenario.value,
            "policy": request.policy.value,
            "execution": request.execution.value,
            "run_result": _run_result_payload(result),
            "timeline": events,
            "baseline": baseline,
            "guardloop": guardloop,
            "circuit_breakers": circuit_breakers or {},
            "spans": spans,
            "code": code,
            "summary": _summary_payload(result, request),
        },
    }


def _event(
    run_id: str,
    event_type: str,
    *,
    step_key: str,
    title: str,
    status: str,
    detail: str,
    metrics: dict[str, Any] | None = None,
) -> EventEnvelope:
    return {
        "type": event_type,
        "run_id": run_id,
        "timestamp": time.time(),
        "step": {
            "key": step_key,
            "title": title,
            "status": status,
            "detail": _sanitize_text(detail),
            "metrics": metrics or {},
        },
    }


def _guardrail_event(run_id: str, result: RunResult) -> EventEnvelope:
    return _event(
        run_id,
        "guardrail_tripped",
        step_key="guardrail",
        title=_guardrail_title(result),
        status="blocked",
        detail=result.error_message or result.terminated_reason or "Guardrail tripped.",
        metrics={
            "terminated_reason": result.terminated_reason,
            "error_type": result.error_type,
        },
    )


def _guardrail_title(result: RunResult) -> str:
    if result.terminated_reason == "budget_exceeded":
        return "Pre-flight budget denied the next call"
    if result.terminated_reason == "circuit_breaker_open":
        return "Circuit breaker denied the next tool call"
    if result.terminated_reason == "verification_failed":
        return "Verifier exhausted retry attempts"
    return "Runtime guardrail stopped the run"


def _request_config(request: GuardLoopRunRequest) -> dict[str, str]:
    settings = get_settings()
    return {
        "scenario": request.scenario.value,
        "policy": request.policy.value,
        "execution": request.execution.value,
        "model": _model_for_request(request, settings),
        "package": "guardloop",
        "command": (
            f"guardloop-demo --scenario {request.scenario.value} "
            f"--policy {request.policy.value} "
            f"--execution {request.execution.value}"
        ),
    }


def _scenario_catalog() -> list[dict[str, str]]:
    return [
        {
            "value": GuardLoopScenario.budget.value,
            "label": "Budget",
            "description": "Stop a runaway model loop before the next expensive call.",
        },
        {
            "value": GuardLoopScenario.circuit_breaker.value,
            "label": "Breaker",
            "description": "Open a per-tool circuit breaker after repeated failures.",
        },
        {
            "value": GuardLoopScenario.verifier.value,
            "label": "Verifier",
            "description": "Reject bad output and retry with verifier feedback.",
        },
    ]


def _summary_payload(result: RunResult, request: GuardLoopRunRequest) -> dict[str, Any]:
    return {
        "outcome": _outcome_label(result, request),
        "success": result.success,
        "terminated_reason": result.terminated_reason,
        "cost_usd": _format_decimal(result.cost_usd),
        "tokens_used": result.tokens_used,
        "tool_calls": result.tool_calls,
        "verification_attempts": result.verification_attempts,
        "trace_id": result.trace_id,
    }


def _outcome_label(result: RunResult, request: GuardLoopRunRequest) -> str:
    if result.terminated_reason:
        return result.terminated_reason
    if request.scenario is GuardLoopScenario.verifier and result.verification_passed:
        return "verified"
    return "completed"


def _run_result_payload(result: RunResult) -> dict[str, Any]:
    payload = result.model_dump(mode="json")
    payload["error_message"] = _sanitize_text(
        cast(str | None, payload.get("error_message"))
    )
    return payload


def _span_payloads(exporter: InMemorySpanExporter) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for span in exporter.get_finished_spans():
        context = span.context
        span_id = f"{context.span_id:016x}" if context is not None else ""
        trace_id = f"{context.trace_id:032x}" if context is not None else ""
        payloads.append(
            {
                "name": span.name,
                "span_id": span_id,
                "trace_id": trace_id,
                "parent_span_id": (
                    f"{span.parent.span_id:016x}" if span.parent is not None else None
                ),
                "status": span.status.status_code.name,
                "attributes": _json_safe_mapping(dict(span.attributes or {})),
                "events": [
                    {
                        "name": event.name,
                        "attributes": _json_safe_mapping(dict(event.attributes or {})),
                    }
                    for event in span.events
                ],
            }
        )
    return payloads


def _json_safe_mapping(values: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in values.items():
        if "stacktrace" in key.lower():
            safe[key] = "...redacted"
            continue
        if isinstance(value, str):
            safe[key] = _sanitize_text(value)
            continue
        safe[key] = _json_safe(value)
    return safe


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return _format_decimal(value)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return str(value)


def _tracer_pair() -> tuple[InMemorySpanExporter, Any]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider.get_tracer("guardloop-portfolio-demo")


def _call_cost() -> Decimal:
    input_cost = Decimal(600) * Decimal("1.75") / Decimal("1000000")
    output_cost = Decimal(300) * Decimal("14.00") / Decimal("1000000")
    return input_cost + output_cost


def _format_decimal(value: Decimal) -> str:
    return format(value, "f")


def _code_snippet(scenario: GuardLoopScenario, policy: GuardLoopPolicy) -> str:
    if scenario is GuardLoopScenario.budget:
        limit = "0.02" if policy is GuardLoopPolicy.guarded else "0.06"
        return "\n".join(
            [
                "from guardloop import BudgetConfig, GuardLoop",
                "",
                "runtime = GuardLoop(",
                "    budget=BudgetConfig(",
                f'        cost_limit_usd="{limit}",',
                "        token_limit=10_000,",
                "    ),",
                ")",
                "result = await runtime.run(runaway_agent, topic)",
            ]
        )
    if scenario is GuardLoopScenario.circuit_breaker:
        if policy is GuardLoopPolicy.relaxed:
            return "\n".join(
                [
                    "from guardloop import CircuitBreakerConfig, GuardLoop",
                    "",
                    "runtime = GuardLoop(",
                    "    circuit_breakers=CircuitBreakerConfig(enabled=False),",
                    ")",
                    "result = await runtime.run(agent)",
                ]
            )
        return "\n".join(
            [
                "from guardloop import (",
                "    CircuitBreakerConfig,",
                "    CircuitBreakerPolicy,",
                "    GuardLoop,",
                ")",
                "",
                "runtime = GuardLoop(",
                "    circuit_breakers=CircuitBreakerConfig(",
                "        default=CircuitBreakerPolicy(failure_threshold=2),",
                "    ),",
                ")",
                "result = await runtime.run(agent)",
            ]
        )
    if policy is GuardLoopPolicy.relaxed:
        return "\n".join(
            [
                "from guardloop import GuardLoop",
                "",
                "runtime = GuardLoop()",
                "result = await runtime.run(agent, question)",
            ]
        )
    return "\n".join(
        [
            "from guardloop import GuardLoop, VerifierConfig, is_json_object",
            "",
            "runtime = GuardLoop(",
            "    verifiers=[",
            "        no_todo_placeholder,",
            "        is_json_object(required_keys=['answer']),",
            "    ],",
            "    verifier_config=VerifierConfig(max_retries=2),",
            ")",
            "result = await runtime.run(agent, question)",
        ]
    )


def _sanitize_error(exc: Exception) -> str:
    return _sanitize_text(f"{type(exc).__name__}: {exc}")[:300]


def _sanitize_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "sk-...redacted", text)
    cleaned = re.sub(
        r"(?i)(api[_-]?key|token|secret)=([A-Za-z0-9._-]{8,})",
        r"\1=...redacted",
        cleaned,
    )
    return cleaned
