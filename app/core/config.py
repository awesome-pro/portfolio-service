from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Portfolio Demos API"
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DEMOS_API_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "DEMOS_API_ANTHROPIC_API_KEY",
            "ANTHROPIC_API_KEY",
        ),
    )
    openai_model: str = "openai/o4-mini"
    anthropic_model: str = "anthropic/claude-haiku-4-5"
    llm_timeout_seconds: float = 45.0
    orchflow_rate_limit_max_runs: int = 5
    orchflow_rate_limit_window_seconds: int = 600
    agenteval_rate_limit_max_runs: int = 4
    agenteval_rate_limit_window_seconds: int = 600
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "https://abhinandan.one"]
    )
    cors_origin_regex: str | None = r"https://.*\.vercel\.app"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DEMOS_API_",
        extra="ignore",
        str_strip_whitespace=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
