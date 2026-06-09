from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SmartMemoScenario(StrEnum):
    debug_logging = "debug_logging"
    web_scaling = "web_scaling"
    trial_extension = "trial_extension"


class SmartMemoQueryVariant(StrEnum):
    paraphrase = "paraphrase"
    opposite_action = "opposite_action"


class SmartMemoRunRequest(BaseModel):
    scenario: SmartMemoScenario = SmartMemoScenario.debug_logging
    query_variant: SmartMemoQueryVariant = SmartMemoQueryVariant.opposite_action
    cosine_threshold: float = Field(default=0.9, ge=0.7, le=0.98)
    classifier_threshold: float = Field(default=0.95, ge=0.7, le=0.99)
    include_feedback: bool = True

    model_config = ConfigDict(str_strip_whitespace=True)
