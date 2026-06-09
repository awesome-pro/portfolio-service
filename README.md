# Portfolio Demos API

FastAPI backend for interactive demos embedded in
[abhinandan.one](https://abhinandan.one). The Orchflow demo powers a live
workflow run with provider-backed agent steps, safe model presets, and basic
per-IP rate limiting. The AgentEval demo runs a live refund-support reliability
gate with repeated `agenteval-py` traces, provider adapters, and the same
rate-limited public API boundary. The GuardLoop demo runs a deterministic
guardrail incident lab with the real `guardloop` runtime, fake provider/tool
clients, and in-memory OpenTelemetry spans. The SmartMemo demo runs a
deterministic semantic-cache safety lab with the real `smartmemo[ml]`
classifier path.

## Stack

- `uv` for Python, dependency locking, and local commands
- FastAPI + Uvicorn
- `orchflow[litellm]==0.5.0` from PyPI
- `agenteval-py>=0.1.1` from PyPI
- `guardloop[otel]>=0.4.2` from PyPI
- `smartmemo[ml]>=0.3.0` from PyPI
- Anthropic Haiku for fast planning/research/review steps
- OpenAI o4-mini for synthesis/finalization steps
- Pytest, Ruff, and Pyright for quality checks

## Local Development

```bash
uv sync
uv run uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

```bash
curl -N http://127.0.0.1:8000/demos/orchflow/run \
  -H 'content-type: application/json' \
  -d '{
    "topic":"AI code review assistant",
    "audience":"engineering managers",
    "constraints":"Show why traces and resume matter.",
    "mode":"success",
    "model_preset":"balanced"
  }'
```

Allowed Orchflow model presets are `balanced`, `haiku_only`, and
`o4_mini_only`; arbitrary model names are intentionally not accepted from the
browser.

The AgentEval demo is also live and bounded:

```bash
curl -N http://127.0.0.1:8000/demos/agenteval/run \
  -H 'content-type: application/json' \
  -d '{
    "message":"I want a refund for order A1007",
    "provider":"openai",
    "mode":"healthy",
    "n_runs":6,
    "threshold":0.8
  }'
```

Allowed AgentEval providers are `openai` and `anthropic`; allowed modes are
`healthy` and `regression`. The browser cannot submit arbitrary model names.

The GuardLoop demo supports live provider-backed runs and a no-key fallback:

```bash
curl -N http://127.0.0.1:8000/demos/guardloop/run \
  -H 'content-type: application/json' \
  -d '{
    "scenario":"budget",
    "policy":"guarded",
    "execution":"openai"
  }'
```

Allowed scenarios are `budget`, `circuit_breaker`, and `verifier`. Policies are
`guarded` and `relaxed`. Execution modes are `openai`, `anthropic`, and
`no_key`. Live modes use backend-owned provider keys and server-configured model
names; the browser cannot submit arbitrary prompts, tools, or model names.

The SmartMemo demo uses safe presets and deterministic responses:

```bash
curl -N http://127.0.0.1:8000/demos/smartmemo/run \
  -H 'content-type: application/json' \
  -d '{
    "scenario":"debug_logging",
    "query_variant":"opposite_action",
    "cosine_threshold":0.9,
    "classifier_threshold":0.95,
    "include_feedback":true
  }'
```

Allowed scenarios are `debug_logging`, `web_scaling`, and `trial_extension`.
Allowed query variants are `opposite_action` and `paraphrase`.

## Checks

```bash
uv run pytest
uv run ruff check .
uv run pyright
```

## Railway

This app includes a Dockerfile based on `python:3.11-slim-bookworm` with the
`uv` binary copied from Astral's official GHCR image. Railway will build the
Dockerfile and start Uvicorn with the platform-provided `PORT`. The image
exposes port `8080`, matching Railway's default container port.

Set these environment variables as needed:

| Variable | Default |
| --- | --- |
| `OPENAI_API_KEY` | Required for o4-mini synthesis/finalization |
| `ANTHROPIC_API_KEY` | Required for Haiku planning/research/review |
| `DEMOS_API_OPENAI_API_KEY` | Optional app-specific override for `OPENAI_API_KEY` |
| `DEMOS_API_ANTHROPIC_API_KEY` | Optional app-specific override for `ANTHROPIC_API_KEY` |
| `DEMOS_API_OPENAI_MODEL` | `openai/o4-mini` |
| `DEMOS_API_ANTHROPIC_MODEL` | `anthropic/claude-haiku-4-5` |
| `DEMOS_API_LLM_TIMEOUT_SECONDS` | `45` |
| `DEMOS_API_ORCHFLOW_RATE_LIMIT_MAX_RUNS` | `5` |
| `DEMOS_API_ORCHFLOW_RATE_LIMIT_WINDOW_SECONDS` | `600` |
| `DEMOS_API_AGENTEVAL_RATE_LIMIT_MAX_RUNS` | `4` |
| `DEMOS_API_AGENTEVAL_RATE_LIMIT_WINDOW_SECONDS` | `600` |
| `DEMOS_API_GUARDLOOP_RATE_LIMIT_MAX_RUNS` | `12` |
| `DEMOS_API_GUARDLOOP_RATE_LIMIT_WINDOW_SECONDS` | `600` |
| `DEMOS_API_GUARDLOOP_OPENAI_MODEL` | `gpt-4o-mini` |
| `DEMOS_API_GUARDLOOP_ANTHROPIC_MODEL` | `claude-3-haiku-20240307` |
| `DEMOS_API_SMARTMEMO_RATE_LIMIT_MAX_RUNS` | `10` |
| `DEMOS_API_SMARTMEMO_RATE_LIMIT_WINDOW_SECONDS` | `600` |
| `DEMOS_API_CORS_ORIGINS` | `["http://localhost:3000", "https://abhinandan.one"]` |
| `DEMOS_API_CORS_ORIGIN_REGEX` | `https://.*\\.vercel\\.app` |
