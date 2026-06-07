from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class OrchflowRunMode(StrEnum):
    success = "success"
    failure_resume = "failure_resume"


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
    mode: OrchflowRunMode = OrchflowRunMode.success

    model_config = ConfigDict(str_strip_whitespace=True)

    def to_flow_input(self) -> dict[str, str]:
        return {
            "topic": self.topic,
            "audience": self.audience,
        }
