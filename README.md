# Portfolio Demos API

FastAPI backend for interactive demos embedded in
[abhinandan.one](https://abhinandan.one). The first demo powers the Orchflow
case-study page with a live, no-key workflow run.

## Stack

- `uv` for Python, dependency locking, and local commands
- FastAPI + Uvicorn
- `orchflow[litellm]==0.5.0` from PyPI
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
  -d '{"topic":"AI code review assistant","audience":"engineering managers","mode":"success"}'
```

## Checks

```bash
uv run pytest
uv run ruff check .
uv run pyright
```

## Railway

This app includes a Dockerfile based on Astral's `uv` Python image. Railway will
build the Dockerfile and start Uvicorn with the platform-provided `PORT`.

Set these environment variables as needed:

| Variable | Default |
| --- | --- |
| `OPENAI_API_KEY` | Required for o4-mini synthesis/finalization |
| `ANTHROPIC_API_KEY` | Required for Haiku planning/research/review |
| `DEMOS_API_OPENAI_MODEL` | `openai/o4-mini` |
| `DEMOS_API_ANTHROPIC_MODEL` | `anthropic/claude-haiku-4-5` |
| `DEMOS_API_LLM_TIMEOUT_SECONDS` | `45` |
| `DEMOS_API_CORS_ORIGINS` | `["http://localhost:3000", "https://abhinandan.one"]` |
| `DEMOS_API_CORS_ORIGIN_REGEX` | `https://.*\\.vercel\\.app` |
