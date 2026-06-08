from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class OrchflowRunMode(StrEnum):
    success = "success"
    failure_resume = "failure_resume"


class OrchflowModelPreset(StrEnum):
    balanced = "balanced"
    haiku_only = "haiku_only"
    o4_mini_only = "o4_mini_only"


class OrchflowRunRequest(BaseModel):
    topic: str = Field(
        default="AI code review assistant",
        min_length=3,
        max_length=160,
    )
    audience: str = Field(
        default="engineering managers",
        min_length=2,
        max_length=120,
    )
    constraints: str = Field(
        default="Keep the final brief practical, specific, and easy to scan.",
        max_length=240,
    )
    mode: OrchflowRunMode = OrchflowRunMode.success
    model_preset: OrchflowModelPreset = OrchflowModelPreset.balanced

    model_config = ConfigDict(str_strip_whitespace=True)

    def to_flow_input(self) -> dict[str, str]:
        return {
            "topic": self.topic,
            "audience": self.audience,
            "constraints": self.constraints,
            "model_preset": self.model_preset.value,
        }
