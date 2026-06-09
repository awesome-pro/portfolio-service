from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class GuardLoopScenario(StrEnum):
    budget = "budget"
    circuit_breaker = "circuit_breaker"
    verifier = "verifier"


class GuardLoopPolicy(StrEnum):
    guarded = "guarded"
    relaxed = "relaxed"


class GuardLoopExecution(StrEnum):
    no_key = "no_key"
    openai = "openai"
    anthropic = "anthropic"


class GuardLoopRunRequest(BaseModel):
    scenario: GuardLoopScenario = GuardLoopScenario.budget
    policy: GuardLoopPolicy = GuardLoopPolicy.guarded
    execution: GuardLoopExecution = GuardLoopExecution.no_key

    model_config = ConfigDict(str_strip_whitespace=True)
