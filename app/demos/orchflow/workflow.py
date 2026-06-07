from __future__ import annotations

import json
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

from orchflow import (
    Agent,
    AgentConfig,
    Flow,
    FlowEvent,
    JsonCheckpointStore,
    StepContext,
    condition,
    step,
)

from app.core.config import Settings, get_settings
from app.demos.orchflow.schemas import OrchflowRunMode, OrchflowRunRequest

EventEnvelope = dict[str, Any]


def encode_ndjson(payload: EventEnvelope) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n"


async def iter_orchflow_demo_events(
    request: OrchflowRunRequest,
) -> AsyncIterator[EventEnvelope]:
    if request.mode is OrchflowRunMode.failure_resume:
        async for event in _iter_failure_resume_events(request):
            yield event
        return

    flow = build_launch_brief_flow()
    async for event in flow.events(request.to_flow_input()):
        yield _event_envelope(event, phase="run")


async def _iter_failure_resume_events(
    request: OrchflowRunRequest,
) -> AsyncIterator[EventEnvelope]:
    with tempfile.TemporaryDirectory(prefix="orchflow-demo-") as temporary_dir:
        store = JsonCheckpointStore(Path(temporary_dir) / "checkpoint.json")
        flow = build_launch_brief_flow(fail_synthesize_once=True)

        async for event in flow.events(
            request.to_flow_input(),
            checkpoint=store,
            raise_on_error=False,
        ):
            yield _event_envelope(event, phase="initial")

        async for event in flow.resume_events(store, raise_on_error=False):
            yield _event_envelope(event, phase="resume")


def build_launch_brief_flow(*, fail_synthesize_once: bool = False) -> Flow:
    synthesize_calls = 0
    settings = get_settings()
    haiku_agent, reasoning_agent = _build_agents(settings)

    @step(name="plan", retry=2)
    async def plan(input: dict[str, str], context: StepContext) -> dict[str, Any]:
        topic = input["topic"]
        audience = input["audience"]
        context.state["topic"] = topic
        context.state["audience"] = audience
        return await _run_json_agent(
            haiku_agent,
            context=context,
            prompt=f"""
TASK: plan
Create a compact launch-brief plan for this project demo.

Topic: {topic}
Audience: {audience}

Return only JSON with keys:
topic, audience, sections, tone, research_questions, model_note.
sections and research_questions must be arrays of strings.
""",
        )

    @step(name="market_research", retry=2)
    async def market_research(
        input: dict[str, str], context: StepContext
    ) -> dict[str, Any]:
        return await _run_json_agent(
            haiku_agent,
            context=context,
            prompt=f"""
TASK: market_research
Act as a market researcher. Use the plan below to explain why an Orchflow demo
is useful for the audience.

Topic: {input["topic"]}
Audience: {input["audience"]}
Plan: {json.dumps(context.previous)}

Return only JSON with keys:
angle, pain, audience_fit, signals, model_note.
signals must be an array of short strings.
""",
        )

    @step(name="technical_research", retry=2)
    async def technical_research(
        input: dict[str, str], context: StepContext
    ) -> dict[str, Any]:
        context.state["runtime_model"] = "sequential + parallel + conditional"
        return await _run_json_agent(
            haiku_agent,
            context=context,
            prompt=f"""
TASK: technical_research
Act as a Python framework engineer. Explain the concrete Orchflow mechanics
that this live portfolio demo should expose.

Topic: {input["topic"]}
Plan: {json.dumps(context.previous)}

Return only JSON with keys:
core_value, mechanics, dependency_story, implementation_note, model_note.
mechanics must be an array of strings.
""",
        )

    @step(name="risk_review", retry=2)
    async def risk_review(
        input: dict[str, str], context: StepContext
    ) -> dict[str, Any]:
        context.state["risk_level"] = "medium"
        review = await _run_json_agent(
            haiku_agent,
            context=context,
            prompt=f"""
TASK: risk_review
Act as a product reviewer. Identify one risk that would make the Orchflow
portfolio demo feel fake, and how to avoid it.

Topic: {input["topic"]}
Audience: {input["audience"]}
Plan: {json.dumps(context.previous)}

Return only JSON with keys:
risk, mitigation, review_score, model_note.
review_score must be a number between 0 and 1.
""",
        )
        review["review_score"] = _coerce_score(review.get("review_score"), 0.82)
        return review

    @step(name="synthesize", retry=2)
    async def synthesize(
        input: dict[str, str], context: StepContext
    ) -> dict[str, Any]:
        nonlocal synthesize_calls
        synthesize_calls += 1
        if fail_synthesize_once and synthesize_calls <= 2:
            raise RuntimeError("simulated deploy interruption after research")

        branches = context.previous
        brief = await _run_json_agent(
            reasoning_agent,
            context=context,
            prompt=f"""
TASK: synthesize
Act as a reasoning-heavy launch strategist. Synthesize the parallel agent
outputs into one launch brief for the portfolio page.

Topic: {input["topic"]}
Audience: {input["audience"]}
Parallel outputs: {json.dumps(branches)}

Return only JSON with keys:
headline, positioning, proof, risk, mitigation, quality_score, model_note.
proof must be an array of strings.
quality_score must be a number between 0 and 1.
""",
        )
        brief["quality_score"] = _coerce_score(brief.get("quality_score"), 0.85)
        context.state["brief_ready"] = True
        return brief

    @step(name="publish_ready", retry=2)
    async def publish_ready(
        input: dict[str, str], context: StepContext
    ) -> dict[str, Any]:
        brief = context.previous
        output = await _run_json_agent(
            reasoning_agent,
            context=context,
            prompt=f"""
TASK: publish_ready
Write the final compact visitor-facing summary for this launch brief.

Brief: {json.dumps(brief)}

Return only JSON with keys:
status, summary, why_orchflow, model_note.
status must be "publish_ready".
why_orchflow must be an array of exactly three strings.
""",
        )
        output["status"] = "publish_ready"
        output["brief"] = brief
        return output

    @step(name="revise", retry=2)
    async def revise(input: dict[str, str], context: StepContext) -> dict[str, Any]:
        output = await _run_json_agent(
            haiku_agent,
            context=context,
            prompt=f"""
TASK: revise
Explain why this launch brief needs revision before publishing.

Audience: {input["audience"]}
Brief: {json.dumps(context.previous)}

Return only JSON with keys:
status, summary, why_orchflow, model_note.
status must be "needs_revision".
why_orchflow must be an array of exactly three strings.
""",
        )
        output["status"] = "needs_revision"
        output["brief"] = context.previous
        return output

    @step(name="finalize")
    async def finalize(input: dict[str, str], context: StepContext) -> dict[str, Any]:
        output = context.previous
        return {
            "title": output["brief"]["headline"],
            "status": output["status"],
            "audience": input["audience"],
            "summary": output["summary"],
            "why_orchflow": output["why_orchflow"],
            "models": {
                "fast_parallel_steps": settings.anthropic_model,
                "reasoning_steps": settings.openai_model,
            },
        }

    return Flow(
        [
            plan,
            [market_research, technical_research, risk_review],
            synthesize,
            condition(
                when=lambda ctx: ctx.previous["quality_score"] >= 0.8,
                then=publish_ready,
                otherwise=revise,
                name="quality_gate",
            ),
            finalize,
        ],
        name="orchflow-launch-brief",
    )


def _build_agents(settings: Settings) -> tuple[Agent, Agent]:
    haiku_agent = Agent(
        name="haiku_researcher",
        role=(
            "You are a precise portfolio-demo agent. Return only valid JSON. "
            "Do not wrap JSON in Markdown."
        ),
        config=AgentConfig(
            model=settings.anthropic_model,
            temperature=0.2,
            max_tokens=600,
            timeout=settings.llm_timeout_seconds,
            extra={"drop_params": True},
        ),
    )
    reasoning_agent = Agent(
        name="o4_reasoner",
        role=(
            "You are a concise reasoning agent for a live product demo. "
            "Return only valid JSON. Do not wrap JSON in Markdown."
        ),
        config=AgentConfig(
            model=settings.openai_model,
            max_tokens=700,
            timeout=settings.llm_timeout_seconds,
            extra={"drop_params": True},
        ),
    )
    return haiku_agent, reasoning_agent


async def _run_json_agent(
    agent: Agent,
    *,
    context: StepContext,
    prompt: str,
) -> dict[str, Any]:
    content = await agent.run(prompt.strip(), context=context)
    parsed = _parse_json_object(content)
    parsed["model_note"] = parsed.get("model_note") or agent.name
    return parsed


def _parse_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        stripped = stripped.removesuffix("```").strip()

    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM output did not contain a JSON object") from None
        value = json.loads(stripped[start : end + 1])

    if not isinstance(value, dict):
        raise ValueError("LLM output JSON must be an object")
    return cast(dict[str, Any], value)


def _coerce_score(value: Any, fallback: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return fallback
    return min(1.0, max(0.0, score))


def _event_envelope(event: FlowEvent, *, phase: str) -> EventEnvelope:
    payload = event.to_dict()
    metadata = payload.get("metadata") or {}
    return {
        "phase": phase,
        "type": payload["type"],
        "run_id": payload["run_id"],
        "flow_name": payload["flow_name"],
        "timestamp": payload["timestamp"],
        "step_name": payload.get("step_name"),
        "step_index": payload.get("step_index"),
        "attempt": payload.get("attempt"),
        "parallel_group_id": payload.get("parallel_group_id"),
        "output": payload.get("output"),
        "error": payload.get("error"),
        "retry_delay": payload.get("retry_delay"),
        "trace": payload.get("trace"),
        "checkpoint": _checkpoint_metadata(metadata, event_type=payload["type"]),
        "final_result": payload.get("result"),
    }


def _checkpoint_metadata(
    metadata: dict[str, Any],
    *,
    event_type: str,
) -> dict[str, Any] | None:
    if event_type not in {"checkpoint_saved", "checkpoint_loaded"}:
        return None
    if "checkpoint_path" not in metadata:
        return None
    return {
        "action": "loaded" if event_type == "checkpoint_loaded" else "saved",
        "status": metadata.get("status"),
        "next_step_index": metadata.get("next_step_index"),
    }
