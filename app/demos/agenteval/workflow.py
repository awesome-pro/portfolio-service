from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from itertools import count
from typing import Any

import agenteval
import anyio
from agenteval import Tracer
from agenteval.adapters.anthropic_adapter import wrap_tools as wrap_anthropic_tools
from agenteval.adapters.openai_adapter import wrap_tools as wrap_openai_tools
from anyio.to_thread import run_sync as run_sync_in_worker_thread

from app.core.config import Settings, get_settings
from app.demos.agenteval.schemas import AgentEvalRunMode, AgentEvalRunRequest

EventEnvelope = dict[str, Any]
TokenUsage = dict[str, int]

ASSERTION_CONTRACT = [
    'called_tool("lookup_order")',
    'called_tool("fetch_refund_policy")',
    'called_tool("create_support_ticket")',
    'tool_called_before("lookup_order", "fetch_refund_policy")',
    'tool_called_with_args("create_support_ticket", {"priority": "normal"})',
    "completed_within_steps(3)",
    'response_contains("eligible")',
    'response_contains("30-day policy")',
    "no_errors()",
]


def encode_ndjson(payload: EventEnvelope) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n"


async def iter_agenteval_demo_events(
    request: AgentEvalRunRequest,
) -> AsyncIterator[EventEnvelope]:
    started_at = time.perf_counter()
    run_id = f"agenteval-demo-{time.time_ns()}"

    yield {
        "type": "demo_started",
        "run_id": run_id,
        "timestamp": time.time(),
        "config": _request_config(request),
        "assertions": ASSERTION_CONTRACT,
    }

    result = await run_sync_in_worker_thread(_run_refund_eval, request)
    traces = sorted(
        result.traces,
        key=lambda trace: int(trace.metadata.get("run_index", 0)),
    )

    for index, trace in enumerate(traces, 1):
        yield {
            "type": "run_passed" if trace.passed else "run_failed",
            "run_id": run_id,
            "timestamp": time.time(),
            "index": index,
            "trace": _trace_payload(trace),
        }
        await anyio.sleep(0.06)

    summary = _result_payload(result, traces, request, time.perf_counter() - started_at)
    yield {
        "type": "gate_passed" if result.met_threshold else "gate_failed",
        "run_id": run_id,
        "timestamp": time.time(),
        "final_result": summary,
    }


def _run_refund_eval(request: AgentEvalRunRequest) -> agenteval.TestResult:
    settings = get_settings()
    run_numbers = count()

    async def test_refund_support_agent(tracer: Tracer) -> None:
        run_index = next(run_numbers)
        wrap_tools = (
            wrap_openai_tools
            if request.provider == "openai"
            else wrap_anthropic_tools
        )
        tools = wrap_tools(
            {
                "lookup_order": lookup_order,
                "fetch_refund_policy": fetch_refund_policy,
                "create_support_ticket": create_support_ticket,
            },
            tracer,
        )
        model = _model_for_request(request, settings)

        async with tracer.run(input=request.message) as run:
            run.add_metadata(
                mode=request.mode.value,
                provider=request.provider.value,
                model=model,
                run_index=run_index,
                variant=f"live_{request.mode.value}",
                live_llm=True,
            )
            result, token_usage, llm_calls = await refund_support_agent(
                request.message,
                request=request,
                settings=settings,
                tools=tools,
            )
            run.set_output(result)
            run.add_metadata(llm_calls=llm_calls)
            if token_usage:
                run.set_token_usage(token_usage)

        (
            tracer.assert_that()
            .called_tool("lookup_order")
            .called_tool("fetch_refund_policy")
            .called_tool("create_support_ticket")
            .tool_called_before("lookup_order", "fetch_refund_policy")
            .tool_called_with_args("create_support_ticket", {"priority": "normal"})
            .completed_within_steps(3)
            .response_contains("eligible", case_sensitive=False)
            .response_contains("30-day policy", case_sensitive=False)
            .no_errors()
            .check()
        )

    return agenteval.run(
        test_refund_support_agent,
        n=request.n_runs,
        concurrency=2,
        name="test_refund_policy_flow",
        threshold=request.threshold,
        tags=["support", "policy", "demo"],
    )


async def lookup_order(order_id: str) -> dict[str, Any]:
    await anyio.sleep(0.004)
    return {
        "order_id": order_id,
        "status": "delivered",
        "delivered_days_ago": 2,
        "country": "US",
        "item": "Noise cancelling headphones",
    }


async def fetch_refund_policy(country: str, item: str) -> dict[str, Any]:
    await anyio.sleep(0.004)
    return {
        "country": country,
        "item": item,
        "return_window_days": 30,
        "refund_method": "original payment method",
    }


async def create_support_ticket(order_id: str, reason: str, priority: str) -> str:
    await anyio.sleep(0.004)
    return f"TICKET-{priority.upper()}-{order_id}-{reason.replace(' ', '-').upper()}"


async def refund_support_agent(
    message: str,
    *,
    request: AgentEvalRunRequest,
    settings: Settings,
    tools: dict[str, Any],
) -> tuple[str, TokenUsage, int]:
    return await _run_live_tool_calling_agent(
        message=message,
        request=request,
        settings=settings,
        tools=tools,
    )


async def _run_live_tool_calling_agent(
    *,
    message: str,
    request: AgentEvalRunRequest,
    settings: Settings,
    tools: dict[str, Any],
) -> tuple[str, TokenUsage, int]:
    import litellm

    model = _model_for_request(request, settings)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(request.mode)},
        {
            "role": "user",
            "content": (
                f"{message}\n\n"
                "Order id: A1007. Use the available tools and then answer the customer."
            ),
        },
    ]
    token_usage: TokenUsage = {}
    llm_calls = 0

    for _ in range(6):
        llm_calls += 1
        response = await litellm.acompletion(
            model=model,
            api_key=_api_key_for_request(request, settings),
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            max_tokens=650,
            timeout=settings.llm_timeout_seconds,
            drop_params=True,
        )
        _merge_token_usage(token_usage, _extract_token_usage(response))

        choice = _first_choice(response)
        assistant_message = _choice_message(choice)
        tool_calls = _message_tool_calls(assistant_message)
        content = _message_content(assistant_message)

        if not tool_calls:
            return content, token_usage, llm_calls

        messages.append(_assistant_tool_call_message(assistant_message, tool_calls))
        for tool_call in tool_calls:
            name = _tool_call_name(tool_call)
            arguments = _tool_call_arguments(tool_call)
            if name not in tools:
                raise ValueError(f"Model requested unknown tool: {name}")
            result = await tools[name](**arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _tool_call_id(tool_call),
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    raise RuntimeError("LLM did not produce a final answer after tool calls")


def _request_config(request: AgentEvalRunRequest) -> dict[str, Any]:
    settings = get_settings()
    return {
        "message": request.message,
        "mode": request.mode.value,
        "provider": request.provider.value,
        "model": _model_for_request(request, settings),
        "n_runs": request.n_runs,
        "threshold": request.threshold,
        "command": (
            "agenteval run tests/refund_support.py "
            f"--n {request.n_runs} --threshold {request.threshold:.2f} "
            f"--tag {request.provider.value} --traces --output report.json"
        ),
    }


def _result_payload(
    result: agenteval.TestResult,
    traces: list[agenteval.AgentTrace],
    request: AgentEvalRunRequest,
    duration_seconds: float,
) -> dict[str, Any]:
    return {
        **_request_config(request),
        "test_name": result.test_name,
        "n_passed": result.n_passed,
        "n_failed": result.n_runs - result.n_passed,
        "pass_rate": result.pass_rate,
        "met_threshold": result.met_threshold,
        "avg_duration_seconds": result.avg_duration,
        "avg_steps": result.avg_steps,
        "duration_seconds": duration_seconds,
        "exit_code": 0 if result.met_threshold else 1,
        "assertions": ASSERTION_CONTRACT,
        "traces": [_trace_payload(trace) for trace in traces],
    }


def _trace_payload(trace: agenteval.AgentTrace) -> dict[str, Any]:
    return {
        "run_id": trace.run_id,
        "input": trace.input,
        "output": trace.output,
        "passed": trace.passed,
        "duration_seconds": trace.duration_seconds,
        "effective_steps": trace.effective_steps,
        "error": _sanitize_error(trace.error),
        "assertion_errors": [
            _sanitize_error(error) or "" for error in trace.assertion_errors
        ],
        "metadata": trace.metadata,
        "token_usage": trace.token_usage,
        "tool_calls": [
            {
                "name": call.name,
                "arguments": call.arguments,
                "result": call.result,
                "duration_seconds": call.duration_seconds,
                "error": call.error,
            }
            for call in trace.tool_calls
        ],
    }


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up order delivery details and customer country.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The customer order id.",
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_refund_policy",
            "description": "Fetch the refund policy for a country and product item.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string"},
                    "item": {"type": "string"},
                },
                "required": ["country", "item"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Create a customer support ticket for the refund request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high"],
                    },
                },
                "required": ["order_id", "reason", "priority"],
                "additionalProperties": False,
            },
        },
    },
]


def _system_prompt(mode: AgentEvalRunMode) -> str:
    if mode is AgentEvalRunMode.regression:
        return """
You are a buggy speed-optimized refund support agent release candidate.
Use tools, but intentionally take the shortcut this eval is designed to catch:
look up order A1007, then create a normal-priority ticket without fetching the
refund policy. Your final answer should be vague and should not mention
"eligible" or "30-day policy".
""".strip()

    return """
You are a careful refund support agent.
You must use the tools in this exact behavioral sequence:
1. Call lookup_order with order_id "A1007".
2. Call fetch_refund_policy using the country and item returned by lookup_order.
3. If delivered_days_ago is within return_window_days, call create_support_ticket
   with priority "normal" and reason "refund request within policy window".
4. Final answer must tell the customer they are eligible and must include the
   exact phrase "30-day policy".
Do not claim policy eligibility until the policy tool has been called.
""".strip()


def _model_for_request(request: AgentEvalRunRequest, settings: Settings) -> str:
    if request.provider == "anthropic":
        return settings.anthropic_model
    return settings.openai_model


def _api_key_for_request(
    request: AgentEvalRunRequest,
    settings: Settings,
) -> str | None:
    if request.provider == "anthropic":
        return settings.anthropic_api_key
    return settings.openai_api_key


def _first_choice(response: Any) -> Any:
    if isinstance(response, dict):
        return response["choices"][0]
    return response.choices[0]


def _choice_message(choice: Any) -> Any:
    if isinstance(choice, dict):
        return choice["message"]
    return choice.message


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", None) or "")


def _message_tool_calls(message: Any) -> list[Any]:
    if isinstance(message, dict):
        return list(message.get("tool_calls") or [])
    return list(getattr(message, "tool_calls", None) or [])


def _assistant_tool_call_message(
    message: Any,
    tool_calls: list[Any],
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": _message_content(message) or None,
        "tool_calls": [
            {
                "id": _tool_call_id(tool_call),
                "type": "function",
                "function": {
                    "name": _tool_call_name(tool_call),
                    "arguments": json.dumps(
                        _tool_call_arguments(tool_call),
                        ensure_ascii=False,
                    ),
                },
            }
            for tool_call in tool_calls
        ],
    }


def _tool_call_id(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("id") or f"call_{time.time_ns()}")
    return str(getattr(tool_call, "id", None) or f"call_{time.time_ns()}")


def _tool_call_name(tool_call: Any) -> str:
    function = _tool_call_function(tool_call)
    if isinstance(function, dict):
        return str(function["name"])
    return str(function.name)


def _tool_call_arguments(tool_call: Any) -> dict[str, Any]:
    function = _tool_call_function(tool_call)
    raw_arguments = (
        function.get("arguments") if isinstance(function, dict) else function.arguments
    )
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if not raw_arguments:
        return {}
    parsed = json.loads(str(raw_arguments))
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must decode to an object")
    return parsed


def _tool_call_function(tool_call: Any) -> Any:
    if isinstance(tool_call, dict):
        return tool_call["function"]
    return tool_call.function


def _extract_token_usage(response: Any) -> TokenUsage:
    usage = (
        response.get("usage")
        if isinstance(response, dict)
        else getattr(response, "usage", None)
    )
    if usage is None:
        return {}

    result: TokenUsage = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    ):
        value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
        if isinstance(value, int):
            result[key] = value
    return result


def _merge_token_usage(total: TokenUsage, update: TokenUsage) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0) + value


def _sanitize_error(error: str | None) -> str | None:
    if error is None:
        return None
    return re.sub(r"sk-[A-Za-z0-9_-]{12,}", "sk-...redacted", error)
