from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SignalSource(str, Enum):
    CHROME_HISTORY = "chrome_history"
    GOOGLE_TAKEOUT = "google_takeout"
    REDDIT_POST = "reddit_post"
    REDDIT_COMMENT = "reddit_comment"
    REDDIT_SAVED = "reddit_saved"
    GMAIL = "gmail"
    FILESYSTEM = "filesystem"
    CHATGPT_EXPORT = "chatgpt_export"
    ENTIRE_IO = "entire_io"


class CuriosityType(str, Enum):
    SHALLOW = "shallow"
    RECURRING = "recurring"
    DEEP = "deep"
    RESOLVED = "resolved"


class ContentSource(str, Enum):
    ARXIV = "arxiv"
    HACKER_NEWS = "hacker_news"
    REDDIT = "reddit"
    BLOG = "blog"
    YOUTUBE = "youtube"
    WEB_SEARCH = "web_search"


class RawSignal(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: SignalSource
    content: str = Field(min_length=1, max_length=2000)
    url: str | None = None
    timestamp: datetime
    raw_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def must_be_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError(
                "RawSignal.timestamp must be timezone-aware. "
                "Use datetime.now(timezone.utc) or attach tzinfo explicitly."
            )
        return v.astimezone(timezone.utc)

    @field_validator("content")
    @classmethod
    def strip_and_validate_content(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("RawSignal.content cannot be empty or whitespace-only.")
        return stripped

    @field_validator("url")
    @classmethod
    def validate_url_scheme(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(
                f"RawSignal.url must start with http:// or https://. Got: {v!r}"
            )
        return v


class Topic(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(min_length=1, max_length=200)
    aliases: list[str] = Field(default_factory=list)
    signal_ids: list[str] = Field(default_factory=list)
    source_types: frozenset[SignalSource] = Field(default_factory=frozenset)

    first_seen: datetime | None = None
    last_seen: datetime | None = None

    frequency: int = Field(default=0, ge=0)
    recency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    depth_score: float = Field(default=0.0, ge=0.0)
    debt_score: float = Field(default=0.0, ge=0.0)
    curiosity_type: CuriosityType = CuriosityType.SHALLOW

    @model_validator(mode="after")
    def temporal_consistency(self) -> "Topic":
        if self.first_seen and self.last_seen:
            if self.first_seen > self.last_seen:
                raise ValueError(
                    f"Topic '{self.name}': first_seen ({self.first_seen}) "
                    f"must not be after last_seen ({self.last_seen})."
                )
        return self

    @field_validator("name")
    @classmethod
    def normalise_name(cls, v: str) -> str:
        return v.strip().lower()

    def span_days(self) -> int:
        if not self.first_seen or not self.last_seen:
            return 0
        return max(0, (self.last_seen - self.first_seen).days)

    def composite_score(self) -> float:
        return self.recency_score * self.frequency + self.debt_score * 2.0


class CuriosityGraph(BaseModel):
    model_config = ConfigDict(frozen=True)

    topics: tuple[Topic, ...] = Field(default_factory=tuple)
    built_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    signal_count: int = Field(default=0, ge=0)
    source_breakdown: dict[str, int] = Field(default_factory=dict)

    def top_n(self, n: int) -> list[Topic]:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        eligible = [t for t in self.topics if t.curiosity_type != CuriosityType.RESOLVED]
        return sorted(eligible, key=lambda t: t.composite_score(), reverse=True)[:n]

    def debt_topics(self) -> list[Topic]:
        return sorted(
            [t for t in self.topics if t.curiosity_type == CuriosityType.RECURRING],
            key=lambda t: t.debt_score,
            reverse=True,
        )

    def by_type(self, curiosity_type: CuriosityType) -> list[Topic]:
        return [t for t in self.topics if t.curiosity_type == curiosity_type]

    def is_empty(self) -> bool:
        return len(self.topics) == 0


class ScrapedArticle(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic_id: str
    topic_name: str
    source: ContentSource
    title: str = Field(min_length=1, max_length=500)
    url: str
    summary: str = Field(default="", max_length=3000)
    full_text: str | None = None
    published_at: datetime | None = None
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(f"ScrapedArticle.url must be http/https. Got: {v!r}")
        return v


class NewsletterSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1)
    section_type: str = Field(pattern=r"^(weekly_topics|curiosity_debt|rabbit_hole)$")
    content: str = Field(min_length=50)
    source_urls: list[str] = Field(default_factory=list)
    audit_reasoning: str = Field(min_length=10)

    @field_validator("audit_reasoning")
    @classmethod
    def reasoning_must_be_substantive(cls, v: str) -> str:
        trivial = {"n/a", "none", "na", "because", "included", "relevant"}
        if v.strip().lower() in trivial:
            raise ValueError(
                "audit_reasoning must explain WHY this section was included, "
                f"not just acknowledge it exists. Got: {v!r}"
            )
        return v


class Newsletter(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    subject_line: str = Field(min_length=10, max_length=100)
    sections: tuple[NewsletterSection, ...] = Field(default_factory=tuple)
    plain_text: str = Field(default="")
    html: str = Field(default="")

    @model_validator(mode="after")
    def must_have_at_least_one_section(self) -> "Newsletter":
        if len(self.sections) == 0:
            raise ValueError("Newsletter must contain at least one section.")
        return self


class AuditEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pipeline_step: str = Field(min_length=1)
    decision: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    inputs_summary: dict[str, Any] = Field(default_factory=dict)
    outputs_summary: dict[str, Any] = Field(default_factory=dict)
