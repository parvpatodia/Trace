from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-6-20251001"

    apify_api_token: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "trace/0.1.0"

    scalekit_env_url: str | None = None
    scalekit_client_id: str | None = None
    scalekit_client_secret: str | None = None
    auth_callback_url: str = "http://localhost:8000/auth/callback"

    database_url: str = "sqlite+aiosqlite:///./trace_data.db"

    scraper_max_concurrent: int = Field(default=5, ge=1, le=20)
    context_token_budget: int = Field(default=8_000, ge=1_000, le=50_000)
    max_topics: int = Field(default=5, ge=1, le=20)
    max_articles_per_topic: int = Field(default=3, ge=1, le=10)
    debt_occurrence_threshold: int = Field(default=3, ge=2, le=20)
    recency_half_life_days: int = Field(default=14, ge=1, le=365)

    browser_history_path: Path = Path("./BrowserHistory.json")
    # Optional additional signal sources — set to enable
    chatgpt_export_path: Path | None = None
    filesystem_root_dir: Path | None = None
    audit_log_path: Path = Path("./trace_audit.jsonl")
    upload_dir: Path = Path("./uploads")

    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1024, le=65535)

    @field_validator("anthropic_model")
    @classmethod
    def must_be_claude_model(cls, v: str) -> str:
        if not v.startswith("claude-"):
            raise ValueError(f"anthropic_model must start with 'claude-', got: {v!r}")
        return v

    @field_validator("database_url")
    @classmethod
    def must_be_supported_scheme(cls, v: str) -> str:
        supported = ("sqlite+aiosqlite://", "postgresql+asyncpg://")
        if not any(v.startswith(s) for s in supported):
            raise ValueError(
                f"database_url scheme not supported. Use one of: {supported}"
            )
        return v

    @field_validator(
        "browser_history_path", "audit_log_path", "upload_dir", mode="before"
    )
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        return Path(v).resolve()

    @field_validator("chatgpt_export_path", "filesystem_root_dir", mode="before")
    @classmethod
    def resolve_optional_path(cls, v: str | Path | None) -> Path | None:
        if v is None:
            return None
        return Path(v).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
