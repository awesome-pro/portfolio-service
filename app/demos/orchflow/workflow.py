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
    StructuredOutputError,
    condition,
    step,
)

from app.core.config import Settings, get_settings
from app.demos.orchflow.schemas import (
    OrchflowModelPreset,
    OrchflowRunMode,
    OrchflowRunRequest,
)

EventEnvelope = dict[str, Any]
JSON_REPAIR_ATTEMPTS = 2

PLAN_SCHEMA: dict[str, Any] = {
    "title": "orchflow_plan",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "topic",
        "audience",
        "sections",
        "tone",
        "research_questions",
        "model_note",
    ],
    "properties": {
        "topic": {"type": "string"},
        "audience": {"type": "string"},
        "sections": {"type": "array", "items": {"type": "string"}},
        "tone": {"type": "string"},
        "research_questions": {"type": "array", "items": {"type": "string"}},
        "model_note": {"type": "string"},
    },
}

MARKET_RESEARCH_SCHEMA: dict[str, Any] = {
    "title": "orchflow_market_research",
    "type": "object",
    "additionalProperties": False,
    "required": ["angle", "pain", "audience_fit", "signals", "model_note"],
    "properties": {
        "angle": {"type": "string"},
        "pain": {"type": "string"},
        "audience_fit": {"type": "string"},
        "signals": {"type": "array", "items": {"type": "string"}},
        "model_note": {"type": "string"},
    },
}

TECHNICAL_RESEARCH_SCHEMA: dict[str, Any] = {
    "title": "orchflow_technical_research",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "core_value",
        "mechanics",
        "dependency_story",
        "implementation_note",
        "model_note",
    ],
    "properties": {
        "core_value": {"type": "string"},
        "mechanics": {"type": "array", "items": {"type": "string"}},
        "dependency_story": {"type": "string"},
        "implementation_note": {"type": "string"},
        "model_note": {"type": "string"},
    },
}

RISK_REVIEW_SCHEMA: dict[str, Any] = {
    "title": "orchflow_risk_review",
    "type": "object",
    "additionalProperties": False,
    "required": ["risk", "mitigation", "review_score", "model_note"],
    "properties": {
        "risk": {"type": "string"},
        "mitigation": {"type": "string"},
        "review_score": {"type": "number"},
        "model_note": {"type": "string"},
    },
}

SYNTHESIZE_SCHEMA: dict[str, Any] = {
    "title": "orchflow_synthesize",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "headline",
        "positioning",
        "proof",
        "risk",
        "mitigation",
        "quality_score",
        "model_note",
    ],
    "properties": {
        "headline": {"type": "string"},
        "positioning": {"type": "string"},
        "proof": {"type": "array", "items": {"type": "string"}},
        "risk": {"type": "string"},
        "mitigation": {"type": "string"},
        "quality_score": {"type": "number"},
        "model_note": {"type": "string"},
    },
}

FINAL_SUMMARY_SCHEMA: dict[str, Any] = {
    "title": "orchflow_final_summary",
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "summary", "why_orchflow", "model_note"],
    "properties": {
        "status": {"type": "string"},
        "summary": {"type": "string"},
        "why_orchflow": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {"type": "string"},
        },
        "model_note": {"type": "string"},
    },
}


def encode_ndjson(payload: EventEnvelope) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n"


async def iter_orchflow_demo_events(
    request: OrchflowRunRequest,
) -> AsyncIterator[EventEnvelope]:
    if request.mode is OrchflowRunMode.failure_resume:
        async for event in _iter_failure_resume_events(request):
            yield event
        return

    flow = build_launch_brief_flow(model_preset=request.model_preset)
    async for event in flow.events(request.to_flow_input()):
        yield _event_envelope(event, phase="run")


async def _iter_failure_resume_events(
    request: OrchflowRunRequest,
) -> AsyncIterator[EventEnvelope]:
    with tempfile.TemporaryDirectory(prefix="orchflow-demo-") as temporary_dir:
        store = JsonCheckpointStore(Path(temporary_dir) / "checkpoint.json")
        flow = build_launch_brief_flow(
            model_preset=request.model_preset,
            fail_synthesize_once=True,
        )

        async for event in flow.events(
            request.to_flow_input(),
            checkpoint=store,
            raise_on_error=False,
        ):
            yield _event_envelope(event, phase="initial")

        async for event in flow.resume_events(store, raise_on_error=False):
            yield _event_envelope(event, phase="resume")


def build_launch_brief_flow(
    *,
    model_preset: OrchflowModelPreset = OrchflowModelPreset.balanced,
    fail_synthesize_once: bool = False,
) -> Flow:
    synthesize_calls = 0
    settings = get_settings()
    fast_model, reasoning_model = _models_for_preset(settings, model_preset)
    fast_agent, reasoning_agent = _build_agents(
        settings,
        fast_model=fast_model,
        reasoning_model=reasoning_model,
    )

    @step(name="plan", retry=2)
    async def plan(input: dict[str, str], context: StepContext) -> dict[str, Any]:
        topic = input["topic"]
        audience = input["audience"]
        context.state["topic"] = topic
        context.state["audience"] = audience
        return await _run_json_agent(
            fast_agent,
            context=context,
            schema=PLAN_SCHEMA,
            prompt=f"""
TASK: plan
Create a compact launch-brief plan for this project demo.

Topic: {topic}
Audience: {audience}
Constraints: {input["constraints"]}

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
            fast_agent,
            context=context,
            schema=MARKET_RESEARCH_SCHEMA,
            prompt=f"""
TASK: market_research
Act as a market researcher. Use the plan below to explain why an Orchflow demo
is useful for the audience.

Topic: {input["topic"]}
Audience: {input["audience"]}
Constraints: {input["constraints"]}
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
            fast_agent,
            context=context,
            schema=TECHNICAL_RESEARCH_SCHEMA,
            prompt=f"""
TASK: technical_research
Act as a Python framework engineer. Explain the concrete Orchflow mechanics
that this live portfolio demo should expose.

Topic: {input["topic"]}
Constraints: {input["constraints"]}
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
            fast_agent,
            context=context,
            schema=RISK_REVIEW_SCHEMA,
            prompt=f"""
TASK: risk_review
Act as a product reviewer. Identify one risk that would make the Orchflow
portfolio demo feel fake, and how to avoid it.

Topic: {input["topic"]}
Audience: {input["audience"]}
Constraints: {input["constraints"]}
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
            schema=SYNTHESIZE_SCHEMA,
            prompt=f"""
TASK: synthesize
Act as a reasoning-heavy launch strategist. Synthesize the parallel agent
outputs into one launch brief for the portfolio page.

Topic: {input["topic"]}
Audience: {input["audience"]}
Constraints: {input["constraints"]}
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
            schema=FINAL_SUMMARY_SCHEMA,
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
            fast_agent,
            context=context,
            schema=FINAL_SUMMARY_SCHEMA,
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
                "preset": model_preset.value,
                "fast_parallel_steps": fast_model,
                "reasoning_steps": reasoning_model,
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


def _build_agents(
    settings: Settings,
    *,
    fast_model: str,
    reasoning_model: str,
) -> tuple[Agent, Agent]:
    fast_agent = Agent(
        name="fast_researcher",
        role=(
            "You are a precise portfolio-demo agent. Return only valid JSON. "
            "Do not wrap JSON in Markdown."
        ),
        config=AgentConfig(
            model=fast_model,
            temperature=0,
            max_tokens=600,
            api_key=_api_key_for_model(settings, fast_model),
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
            model=reasoning_model,
            max_tokens=700,
            api_key=_api_key_for_model(settings, reasoning_model),
            timeout=settings.llm_timeout_seconds,
            extra={"drop_params": True},
        ),
    )
    return fast_agent, reasoning_agent


def _models_for_preset(
    settings: Settings,
    model_preset: OrchflowModelPreset,
) -> tuple[str, str]:
    if model_preset is OrchflowModelPreset.haiku_only:
        return settings.anthropic_model, settings.anthropic_model
    if model_preset is OrchflowModelPreset.o4_mini_only:
        return settings.openai_model, settings.openai_model
    return settings.anthropic_model, settings.openai_model


def _api_key_for_model(settings: Settings, model: str) -> str | None:
    if model.startswith("anthropic/"):
        return settings.anthropic_api_key
    if model.startswith("openai/"):
        return settings.openai_api_key
    return None


async def _run_json_agent(
    agent: Agent,
    *,
    context: StepContext,
    prompt: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    structured_prompt = _json_contract_prompt(prompt, schema)
    try:
        parsed = await agent.run_structured(
            structured_prompt,
            schema=schema,
            context=context,
        )
        return _finalize_json_object(parsed, schema=schema, agent_name=agent.name)
    except Exception as exc:
        if not _can_retry_without_response_format(exc):
            raise

    content = await agent.run(structured_prompt, context=context)
    last_error: ValueError | None = None
    for repair_index in range(JSON_REPAIR_ATTEMPTS + 1):
        try:
            parsed = _parse_json_object(content)
            return _finalize_json_object(parsed, schema=schema, agent_name=agent.name)
        except ValueError as error:
            last_error = error
            if repair_index >= JSON_REPAIR_ATTEMPTS:
                raise
            content = await agent.run(
                _json_repair_prompt(
                    original_prompt=prompt,
                    invalid_content=content,
                    schema=schema,
                    error=error,
                ),
                context=context,
            )

    raise RuntimeError("JSON repair loop exited unexpectedly") from last_error


def _json_contract_prompt(prompt: str, schema: dict[str, Any]) -> str:
    return f"""
{prompt.strip()}

JSON CONTRACT:
Return exactly one JSON object and no prose, Markdown, code fences, or comments.
Use double-quoted JSON strings. Keep values concise.
Required JSON schema:
{json.dumps(schema, ensure_ascii=False)}
""".strip()


def _json_repair_prompt(
    *,
    original_prompt: str,
    invalid_content: str,
    schema: dict[str, Any],
    error: Exception,
) -> str:
    return f"""
Convert the model output below into exactly one valid JSON object.
Do not add prose, Markdown, code fences, or comments.

Original task:
{original_prompt.strip()}

Validation error:
{error}

Required JSON schema:
{json.dumps(schema, ensure_ascii=False)}

Model output to convert:
{invalid_content}
""".strip()


def _can_retry_without_response_format(exc: Exception) -> bool:
    if isinstance(exc, (StructuredOutputError, ValueError)):
        return True
    message = str(exc).lower()
    return (
        "response_format" in message
        or "json_schema" in message
        or "not supported" in message
        or "unsupported" in message
    )


def _finalize_json_object(
    value: Any,
    *,
    schema: dict[str, Any],
    agent_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("LLM output JSON must be an object")

    parsed = cast(dict[str, Any], value)
    _validate_required_schema(parsed, schema=schema)
    parsed["model_note"] = parsed.get("model_note") or agent_name
    return parsed


def _validate_required_schema(value: dict[str, Any], *, schema: dict[str, Any]) -> None:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return

    required = schema.get("required", [])
    if isinstance(required, list):
        for key in required:
            if isinstance(key, str) and key not in value:
                raise ValueError(f"LLM output missing required JSON key: {key}")

    for key, config in properties.items():
        if not isinstance(key, str) or key not in value or not isinstance(config, dict):
            continue
        expected_type = config.get("type")
        if expected_type == "string" and not isinstance(value[key], str):
            raise ValueError(f"JSON key {key} must be a string")
        if expected_type == "number" and not isinstance(value[key], int | float):
            raise ValueError(f"JSON key {key} must be a number")
        if expected_type == "array" and not isinstance(value[key], list):
            raise ValueError(f"JSON key {key} must be an array")


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
    trace = _public_trace(payload.get("trace"))
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
        "error": _public_error(payload.get("error")),
        "retry_delay": payload.get("retry_delay"),
        "trace": trace,
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


def _public_trace(trace: Any) -> Any:
    if not isinstance(trace, dict):
        return trace
    sanitized = dict(trace)
    sanitized["error"] = _public_error(sanitized.get("error"))
    return sanitized


def _public_error(error: Any) -> str | None:
    if not isinstance(error, str) or not error:
        return None
    if "Incorrect API key" in error or "AuthenticationError" in error:
        return (
            "OpenAI authentication failed. Check the backend "
            "DEMOS_API_OPENAI_API_KEY or OPENAI_API_KEY value."
        )
    return error
