"""
Trace FastAPI delivery layer.

Endpoints:
  GET  /health                 — liveness probe
  POST /newsletter/generate    — run the full pipeline and return a newsletter

Dependency injection:
  get_pipeline() is the FastAPI dependency that returns the TracePipeline.
  Tests override it via app.dependency_overrides[get_pipeline] = lambda: mock.

The TracePipeline is expensive to construct (loads PRAW, Anthropic client,
etc.) so it is built once at application startup via the lifespan context
manager and stored on app.state.

WHY pipeline not constructed per-request:
  Signal collectors (filesystem, Reddit) and scrapers hold long-lived HTTP
  clients. Re-constructing them per request wastes connections and would
  trigger PRAW's auth flow on every call.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from trace.pipeline.runner import PipelineError, PipelineResult, TracePipeline


# ── Request / response models ─────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    token_budget: int = Field(default=8_000, ge=1)
    max_topics: int = Field(default=5, ge=1)
    max_articles_per_topic: int = Field(default=3, ge=1)


class SectionResponse(BaseModel):
    title: str
    section_type: str
    content: str
    source_urls: list[str]
    audit_reasoning: str


class GenerateResponse(BaseModel):
    id: str
    subject_line: str
    sections: list[SectionResponse]
    plain_text: str
    html: str
    generated_at: str
    errors: list[str]


# ── Application lifecycle ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Build the pipeline on startup if settings are available.
    In tests, the pipeline is always injected via dependency override so this
    path is never exercised.
    """
    # Defer heavy imports so tests that override get_pipeline never touch them
    try:
        application.state.pipeline = _build_pipeline_from_settings()
    except Exception:
        application.state.pipeline = None
    yield
    application.state.pipeline = None


def _build_pipeline_from_settings() -> TracePipeline | None:
    """
    Construct a fully-wired TracePipeline from environment settings.
    Returns None if required settings are missing (dev/test mode).
    """
    try:
        from pathlib import Path

        import anthropic
        import praw

        from trace.composer.assembler import ContextWindowAssembler
        from trace.composer.newsletter import NewsletterComposer
        from trace.config import get_settings
        from trace.graph.builder import CuriosityGraphBuilder
        from trace.graph.extractor import TopicExtractor
        from trace.scraper.arxiv import ArXivScraper
        from trace.scraper.hackernews import HackerNewsScraper
        from trace.scraper.reddit import RedditSearchScraper
        from trace.signals.google_takeout import GoogleTakeoutCollector

        settings = get_settings()

        anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        reddit = praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
        )

        history_path = settings.browser_history_path
        collectors = [
            GoogleTakeoutCollector(history_path=history_path),
        ]
        scrapers = [
            ArXivScraper(),
            HackerNewsScraper(),
            RedditSearchScraper(reddit_client=reddit),
        ]
        extractor = TopicExtractor(client=anthropic_client, model=settings.anthropic_model)
        builder = CuriosityGraphBuilder(
            extractor=extractor,
            half_life_days=settings.recency_half_life_days,
        )
        assembler = ContextWindowAssembler(
            token_budget=settings.context_token_budget,
            max_topics=settings.max_topics,
            max_articles_per_topic=settings.max_articles_per_topic,
        )
        composer = NewsletterComposer(
            client=anthropic_client,
            model=settings.anthropic_model,
        )
        return TracePipeline(
            collectors=collectors,
            scrapers=scrapers,
            builder=builder,
            assembler=assembler,
            composer=composer,
        )
    except Exception:
        return None


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Trace",
    description="Curiosity inference newsletter agent",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Dependencies ──────────────────────────────────────────────────────────────

def get_pipeline(request: Request) -> TracePipeline:
    """
    FastAPI dependency returning the shared TracePipeline instance.
    In production: reads from app.state.pipeline (set by lifespan).
    In tests: override via app.dependency_overrides[get_pipeline] = lambda: mock
    """
    pipeline: TracePipeline | None = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized — check server configuration and environment variables",
        )
    return pipeline


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/newsletter/generate", response_model=GenerateResponse)
async def generate_newsletter(
    request: GenerateRequest = GenerateRequest(),
    pipeline: TracePipeline = Depends(get_pipeline),
) -> GenerateResponse:
    try:
        result: PipelineResult = await pipeline.run()
    except PipelineError as e:
        raise HTTPException(status_code=503, detail=str(e))

    newsletter = result.newsletter
    return GenerateResponse(
        id=newsletter.id,
        subject_line=newsletter.subject_line,
        sections=[
            SectionResponse(
                title=s.title,
                section_type=s.section_type,
                content=s.content,
                source_urls=s.source_urls,
                audit_reasoning=s.audit_reasoning,
            )
            for s in newsletter.sections
        ],
        plain_text=newsletter.plain_text,
        html=newsletter.html,
        generated_at=newsletter.generated_at.isoformat(),
        errors=result.errors,
    )
