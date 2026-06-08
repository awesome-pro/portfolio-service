from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AgentEvalRunMode(StrEnum):
    healthy = "healthy"
    regression = "regression"


class AgentEvalProvider(StrEnum):
    openai = "openai"
    anthropic = "anthropic"


class AgentEvalRunRequest(BaseModel):
    message: str = Field(
        default="I want a refund for order A1007",
        min_length=8,
        max_length=180,
    )
    n_runs: int = Field(default=6, ge=3, le=12)
    threshold: float = Field(default=0.8, ge=0.5, le=1.0)
    provider: AgentEvalProvider = AgentEvalProvider.openai
    mode: AgentEvalRunMode = AgentEvalRunMode.healthy

    model_config = ConfigDict(str_strip_whitespace=True)
