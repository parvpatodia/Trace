"""
Trace FastAPI delivery layer.

Endpoints:
  GET  /health                 — liveness probe
  GET  /auth/login             — returns Scalekit OAuth authorization URL
  GET  /auth/callback          — exchanges OAuth code for tokens
  GET  /auth/me                — returns authenticated user's profile
  GET  /auth/logout            — returns Scalekit logout URL
  POST /newsletter/generate    — run the full pipeline and return a newsletter

Authentication:
  The newsletter endpoint accepts an optional Bearer token (Scalekit JWT).
  When present the response includes `generated_for` with the user's identity,
  demonstrating that the pipeline runs on behalf of the authenticated user.
  When absent the endpoint still works — useful for development and testing.

  Auth endpoints return 503 when SCALEKIT_* environment variables are unset.

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
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from trace.auth.scalekit import UserClaims, _require_client, build_login_url, exchange_code, verify_token
from trace.config import get_settings
from trace.pipeline.runner import PipelineError, PipelineResult, TracePipeline


# ── Request / response models ─────────────────────────────────────────────────

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
    generated_for: str = ""


class LoginResponse(BaseModel):
    authorization_url: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int | None = None
    user: dict[str, Any] = {}


class UserResponse(BaseModel):
    user_id: str
    email: str
    name: str
    organization_id: str


# ── Application lifecycle ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Build the pipeline on startup if settings are available.
    In tests, the pipeline is always injected via dependency override so this
    path is never exercised.
    """
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
        import anthropic
        import praw

        from trace.audit.writer import AuditWriter
        from trace.composer.assembler import ContextWindowAssembler
        from trace.composer.newsletter import NewsletterComposer
        from trace.graph.builder import CuriosityGraphBuilder
        from trace.graph.extractor import TopicExtractor
        from trace.scraper.apify import ApifyScraper
        from trace.scraper.arxiv import ArXivScraper
        from trace.scraper.hackernews import HackerNewsScraper
        from trace.scraper.reddit import RedditSearchScraper
        from trace.signals.google_takeout import GoogleTakeoutCollector

        settings = get_settings()

        anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        history_path = settings.browser_history_path
        collectors = [
            GoogleTakeoutCollector(history_path=history_path),
        ]
        scrapers: list[Any] = [
            ArXivScraper(),
            HackerNewsScraper(),
        ]
        if settings.reddit_client_id and settings.reddit_client_secret:
            reddit = praw.Reddit(
                client_id=settings.reddit_client_id,
                client_secret=settings.reddit_client_secret,
                user_agent=settings.reddit_user_agent,
            )
            scrapers.append(RedditSearchScraper(reddit_client=reddit))

        if settings.apify_api_token:
            scrapers.append(ApifyScraper(api_token=settings.apify_api_token))

        extractor = TopicExtractor(client=anthropic_client, model=settings.anthropic_model)
        builder = CuriosityGraphBuilder(
            extractor=extractor,
            half_life_days=settings.recency_half_life_days,
            debt_threshold_occurrences=settings.debt_occurrence_threshold,
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
        audit_writer = AuditWriter(path=settings.audit_log_path)
        return TracePipeline(
            collectors=collectors,
            scrapers=scrapers,
            builder=builder,
            assembler=assembler,
            composer=composer,
            max_concurrent_scrapers=settings.scraper_max_concurrent,
            audit_writer=audit_writer,
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

_bearer = HTTPBearer(auto_error=False)


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


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> UserClaims | None:
    """
    FastAPI dependency that extracts and validates the Bearer token if present.

    Returns None when no Authorization header is sent (unauthenticated access
    is allowed — the newsletter endpoint degrades gracefully by omitting the
    `generated_for` field). Returns UserClaims when a valid token is provided.

    Raises 401 if a token is present but invalid.
    """
    if credentials is None:
        return None
    return await verify_token(credentials.credentials)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
) -> UserClaims:
    """
    FastAPI dependency that REQUIRES a valid Bearer token.
    Used by /auth/me and other strictly protected endpoints.
    """
    return await verify_token(credentials.credentials)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/auth/login", response_model=LoginResponse)
async def auth_login(state: str | None = None) -> LoginResponse:
    """
    Begin the Scalekit OAuth flow.

    Returns the authorization URL. The client should redirect the user's
    browser to this URL. After authentication, Scalekit redirects the browser
    to SCALEKIT_AUTH_CALLBACK_URL with ?code=...&state=...

    Query params:
      state: optional opaque string, echoed back in the callback for CSRF
             protection. Callers should generate a random value, store it in
             session, and verify it in /auth/callback.
    """
    _require_client()  # raises 503 immediately if Scalekit is unconfigured
    callback_url = get_settings().auth_callback_url
    authorization_url = build_login_url(redirect_uri=callback_url, state=state)
    return LoginResponse(authorization_url=authorization_url)


@app.get("/auth/callback", response_model=TokenResponse)
async def auth_callback(code: str, state: str | None = None) -> TokenResponse:
    """
    Complete the Scalekit OAuth flow.

    Exchanges the authorization code (received from Scalekit's redirect) for
    tokens. Returns access_token for use in subsequent API calls as:
      Authorization: Bearer <access_token>

    Note: In production, verify `state` matches what was stored in the session
    before exchanging the code to prevent CSRF attacks.
    """
    _require_client()  # raises 503 immediately if Scalekit is unconfigured
    callback_url = get_settings().auth_callback_url
    result = await exchange_code(code=code, redirect_uri=callback_url)
    return TokenResponse(
        access_token=result["access_token"],
        expires_in=result.get("expires_in"),
        user=result.get("user", {}),
    )


@app.get("/auth/me", response_model=UserResponse)
async def auth_me(
    current_user: UserClaims = Depends(get_current_user),
) -> UserResponse:
    """
    Return the authenticated user's profile.

    Requires: Authorization: Bearer <access_token>
    """
    return UserResponse(
        user_id=current_user.user_id,
        email=current_user.email,
        name=current_user.name,
        organization_id=current_user.organization_id,
    )


@app.get("/auth/logout")
async def auth_logout(post_logout_redirect_uri: str | None = None) -> dict[str, str]:
    """
    Return the Scalekit logout URL.

    The client should redirect the user's browser to the returned `logout_url`
    to invalidate the Scalekit session and clear the SSO cookie.
    """
    from scalekit.client import LogoutUrlOptions

    client = _require_client()

    options = LogoutUrlOptions()
    if post_logout_redirect_uri:
        options.post_logout_redirect_uri = post_logout_redirect_uri

    logout_url = client.get_logout_url(options)
    return {"logout_url": logout_url}


@app.post("/newsletter/generate", response_model=GenerateResponse)
async def generate_newsletter(
    pipeline: TracePipeline = Depends(get_pipeline),
    current_user: UserClaims | None = Depends(get_optional_user),
) -> GenerateResponse:
    """
    Run the full Trace pipeline and return a personalized newsletter.

    Authentication is optional. When a valid Bearer token is provided the
    response includes `generated_for` identifying the user the newsletter was
    generated for — demonstrating that the pipeline acts on behalf of that
    specific user rather than a generic service account.
    """
    try:
        result: PipelineResult = await pipeline.run(
            user_id=current_user.user_id if current_user else "",
            user_email=current_user.email if current_user else "",
        )
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
        generated_for=current_user.display_name if current_user else "",
    )
