"""
Trace FastAPI delivery layer.

Endpoints:
  GET  /                       — web frontend (upload + display)
  GET  /health                 — liveness probe
  POST /upload                 — upload BrowserHistory.json or conversations.json
  POST /newsletter/generate    — run pipeline with pre-configured paths
  POST /newsletter/from-upload — run pipeline against an uploaded file
  GET  /newsletter/{id}        — retrieve a previously generated newsletter
  GET  /auth/login             — returns Scalekit OAuth authorization URL
  GET  /auth/callback          — exchanges OAuth code for tokens
  GET  /auth/me                — returns authenticated user's profile
  GET  /auth/logout            — returns Scalekit logout URL

Personalization:
  The core value: upload YOUR Google Takeout BrowserHistory.json and receive a
  newsletter that reflects YOUR actual curiosity patterns — topics Claude infers
  from what you actually visited, not generic trending content.

Authentication:
  The newsletter endpoint accepts an optional Bearer token (Scalekit JWT).
  Auth endpoints return 503 when SCALEKIT_* environment variables are unset.

Dependency injection:
  get_pipeline() is the FastAPI dependency that returns the TracePipeline.
  Tests override it via app.dependency_overrides[get_pipeline] = lambda: mock.

Pipeline construction:
  The TracePipeline is expensive to construct (loads PRAW, Anthropic client,
  etc.) so it is built once at application startup via the lifespan context
  manager and stored on app.state.
"""

from __future__ import annotations

import io
import json as _json
import logging
import re
import uuid
import zipfile
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.IGNORECASE)

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from trace.auth.scalekit import UserClaims, _require_client, build_login_url, exchange_code, verify_token
from trace.config import get_settings
from trace.models import CuriosityGraph
from trace.pipeline.runner import PipelineError, PipelineResult, TracePipeline

_log = logging.getLogger(__name__)

# In-memory newsletter cache — last 20 newsletters (LRU-style)
_NEWSLETTER_CACHE: OrderedDict[str, dict] = OrderedDict()
_CACHE_MAX = 20

# Curiosity profile cache — stores CuriosityGraph (NOT raw signals) for daily
# newsletter regeneration without re-uploading browsing history. Only inferred
# topic names + scores are stored, never raw browsing/conversation data.
_PROFILE_CACHE: OrderedDict[str, CuriosityGraph] = OrderedDict()
_PROFILE_CACHE_MAX = 50


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
    # profile_id: opaque token that stores the CuriosityGraph (not raw data).
    # Present after a full pipeline run. Use POST /newsletter/regenerate/{profile_id}
    # to generate a fresh newsletter from the same interests without re-uploading.
    profile_id: str = ""
    # topic_names: curiosity topics sorted by composite_score descending.
    topic_names: list[str] = []
    # topic_scores: composite_score for each topic (parallel to topic_names).
    # Normalised by the caller — [0,1] relative to the highest-scoring topic.
    topic_scores: list[float] = []


class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    size_bytes: int
    message: str


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
    try:
        application.state.pipeline = _build_pipeline_from_settings()
    except Exception as exc:
        _log.warning("Could not build pipeline at startup: %s", exc)
        application.state.pipeline = None

    # Start autonomous agent scheduler.
    scheduler = None
    try:
        from trace.agent.scheduler import build_scheduler
        scheduler = build_scheduler()
        if scheduler is not None:
            scheduler.start()
            _log.info("Agent scheduler started")
    except Exception as exc:
        _log.warning("Agent scheduler failed to start: %s", exc)

    yield

    # Shutdown scheduler cleanly.
    if scheduler is not None:
        try:
            scheduler.shutdown(wait=False)
            _log.info("Agent scheduler stopped")
        except Exception:
            pass
    application.state.pipeline = None


def _build_pipeline_from_settings(
    override_history_path: Path | None = None,
    override_chatgpt_path: Path | None = None,
    override_youtube_path: Path | None = None,
) -> TracePipeline | None:
    """
    Construct a fully-wired TracePipeline from environment settings.

    override_history_path: use this BrowserHistory.json instead of the configured one.
    override_chatgpt_path: use this conversations.json instead of the configured one.
    override_youtube_path: use this watch-history.json instead of the configured one.

    Returns None if required settings are missing (dev/test mode).
    """
    try:
        import anthropic

        from trace.audit.writer import AuditWriter
        from trace.composer.assembler import ContextWindowAssembler
        from trace.composer.newsletter import NewsletterComposer
        from trace.graph.builder import CuriosityGraphBuilder
        from trace.graph.extractor import TopicExtractor
        from trace.scraper.arxiv import ArXivScraper
        from trace.scraper.hackernews import HackerNewsScraper
        from trace.signals.google_takeout import GoogleTakeoutCollector

        settings = get_settings()
        anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        # ── Signal collectors ────────────────────────────────────────────────
        collectors: list[Any] = []

        history_path = override_history_path or settings.browser_history_path
        if history_path.exists():
            collectors.append(GoogleTakeoutCollector(history_path=history_path))
            _log.info("GoogleTakeout collector active: %s", history_path)
        else:
            _log.warning(
                "BrowserHistory.json not found at %s — GoogleTakeout collector skipped",
                history_path,
            )

        chatgpt_path = override_chatgpt_path or settings.chatgpt_export_path
        if chatgpt_path and chatgpt_path.exists():
            from trace.signals.chatgpt_export import ChatGPTExportCollector
            collectors.append(ChatGPTExportCollector(export_path=chatgpt_path))
            _log.info("ChatGPT export collector active: %s", chatgpt_path)

        yt_path = override_youtube_path or settings.youtube_watch_history_path
        if yt_path and yt_path.exists():
            from trace.signals.youtube_takeout import YouTubeWatchHistoryCollector
            collectors.append(YouTubeWatchHistoryCollector(history_path=yt_path))
            _log.info("YouTube watch history collector active: %s", yt_path)

        if settings.filesystem_root_dir and settings.filesystem_root_dir.exists():
            from trace.signals.filesystem import FileSystemCollector
            collectors.append(FileSystemCollector(root_dir=settings.filesystem_root_dir))
            _log.info("FileSystem collector active: %s", settings.filesystem_root_dir)

        if not collectors:
            _log.warning(
                "No signal collectors available — pipeline cannot start without signals. "
                "Upload a BrowserHistory.json or set BROWSER_HISTORY_PATH."
            )
            return None

        # ── Article scrapers ─────────────────────────────────────────────────
        scrapers: list[Any] = [
            ArXivScraper(),
            HackerNewsScraper(),
        ]

        if settings.reddit_client_id and settings.reddit_client_secret:
            try:
                import praw
                from trace.scraper.reddit import RedditSearchScraper
                reddit = praw.Reddit(
                    client_id=settings.reddit_client_id,
                    client_secret=settings.reddit_client_secret,
                    user_agent=settings.reddit_user_agent,
                )
                scrapers.append(RedditSearchScraper(reddit_client=reddit))
                _log.info("Reddit scraper active")
            except Exception as exc:
                _log.warning("Reddit scraper skipped: %s", exc)

        if settings.apify_api_token:
            try:
                from trace.scraper.apify import ApifyScraper
                scrapers.append(ApifyScraper(
                    api_token=settings.apify_api_token,
                    actor_id=settings.apify_actor_id,
                ))
                _log.info("Apify scraper active (actor: %s)", settings.apify_actor_id)
            except Exception as exc:
                _log.warning("Apify scraper skipped: %s", exc)

        # ── Pipeline components ──────────────────────────────────────────────
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
    except Exception as exc:
        _log.warning("Failed to build pipeline: %s", exc)
        return None


def _store_newsletter(response: GenerateResponse) -> None:
    """Cache newsletter in memory, evicting oldest when full."""
    _NEWSLETTER_CACHE[response.id] = response.model_dump()
    if len(_NEWSLETTER_CACHE) > _CACHE_MAX:
        _NEWSLETTER_CACHE.popitem(last=False)


def _profile_dir() -> Path:
    """Directory where CuriosityGraph JSON files are persisted."""
    return get_settings().upload_dir / "profiles"


def _store_profile(graph: CuriosityGraph) -> str:
    """Persist a CuriosityGraph to disk and cache it in memory.

    Only inferred topic names + scores + signal_samples are stored — never raw
    browsing URLs or conversation content.  Disk persistence means profile_ids
    survive server restarts, enabling daily regeneration without re-uploading.
    """
    profile_id = str(uuid.uuid4())
    # Memory cache (fast path for same-process requests)
    _PROFILE_CACHE[profile_id] = graph
    if len(_PROFILE_CACHE) > _PROFILE_CACHE_MAX:
        _PROFILE_CACHE.popitem(last=False)
    # Disk persistence (survives restarts)
    try:
        d = _profile_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{profile_id}.json").write_text(
            graph.model_dump_json(), encoding="utf-8"
        )
    except OSError as exc:
        _log.warning("Could not persist profile %s to disk: %s", profile_id, exc)
    return profile_id


def _load_profile(profile_id: str) -> CuriosityGraph | None:
    """Return a CuriosityGraph by profile_id, checking memory then disk."""
    if profile_id in _PROFILE_CACHE:
        return _PROFILE_CACHE[profile_id]
    try:
        path = _profile_dir() / f"{profile_id}.json"
        if path.exists():
            graph = CuriosityGraph.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            # Warm the memory cache so subsequent calls are instant
            _PROFILE_CACHE[profile_id] = graph
            if len(_PROFILE_CACHE) > _PROFILE_CACHE_MAX:
                _PROFILE_CACHE.popitem(last=False)
            return graph
    except Exception as exc:
        _log.warning("Could not load profile %s from disk: %s", profile_id, exc)
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

# Mount the Curiosity OS MCP server at /mcp.
# Claude Desktop connects to: http://localhost:8000/mcp  (Streamable HTTP)
# SSE fallback:               http://localhost:8000/mcp/sse
try:
    from trace.mcp.server import get_mcp_asgi_app
    app.mount("/mcp", get_mcp_asgi_app())
    _log.info("Trace MCP server mounted at /mcp")
except Exception as _mcp_err:
    _log.warning("MCP server mount failed (non-fatal): %s", _mcp_err)

_bearer = HTTPBearer(auto_error=False)


# ── Dependencies ──────────────────────────────────────────────────────────────

def get_pipeline(request: Request) -> TracePipeline:
    pipeline: TracePipeline | None = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized — set ANTHROPIC_API_KEY and ensure BrowserHistory.json exists",
        )
    return pipeline


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> UserClaims | None:
    if credentials is None:
        return None
    return await verify_token(credentials.credentials)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
) -> UserClaims:
    return await verify_token(credentials.credentials)


# ── Frontend ──────────────────────────────────────────────────────────────────

_FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trace — Your Curiosity, Distilled</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #13131a;
    --surface2: #1c1c26;
    --border: #252535;
    --border2: #2e2e42;
    --accent: #6366f1;
    --accent-hover: #818cf8;
    --accent-dim: rgba(99,102,241,0.12);
    --text: #e2e8f0;
    --muted: #8892a4;
    --muted2: #6b7280;
    --error: #f87171;
    --success: #34d399;
    --warn: #fbbf24;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2.5rem 1rem 4rem;
    -webkit-font-smoothing: antialiased;
  }
  .container { width: 100%; max-width: 780px; }

  /* ── Header ── */
  header { text-align: center; margin-bottom: 3rem; }
  .logo { font-size: 2.8rem; font-weight: 700; letter-spacing: -0.04em; line-height: 1; }
  .logo span { color: var(--accent); }
  .tagline { color: var(--muted); margin-top: 0.6rem; font-size: 1rem; font-weight: 400; }

  /* ── Cards ── */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.25rem;
  }
  .card-title {
    font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: var(--muted2); margin-bottom: 1.25rem;
  }

  /* ── Source cards (stacked, all visible) ── */
  .source-cards { display: flex; flex-direction: column; gap: 0.75rem; }
  .source-card {
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 10px;
    overflow: hidden;
    transition: border-color 0.2s;
  }
  .source-card.has-file { border-color: rgba(99,102,241,0.5); }
  .source-card-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.85rem 1.1rem; cursor: pointer; user-select: none;
    gap: 0.75rem;
  }
  .source-card-left { display: flex; align-items: center; gap: 0.75rem; }
  .source-icon { font-size: 1.3rem; flex-shrink: 0; line-height: 1; }
  .source-name { font-size: 0.92rem; font-weight: 600; }
  .source-desc { font-size: 0.78rem; color: var(--muted); margin-top: 0.1rem; }
  .source-status {
    font-size: 0.75rem; font-weight: 500;
    color: var(--muted2); white-space: nowrap; flex-shrink: 0;
  }
  .source-status.chosen { color: var(--success); }
  .expand-toggle {
    color: var(--muted2); font-size: 0.65rem;
    flex-shrink: 0; transition: transform 0.2s;
  }
  .source-card.expanded .expand-toggle { transform: rotate(180deg); }
  .source-body { display: none; border-top: 1px solid var(--border2); padding: 1rem 1.1rem; }
  .source-card.expanded .source-body { display: block; }
  .how-to {
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 8px;
    padding: 0.85rem 1rem;
    font-size: 0.82rem;
    color: var(--muted);
    margin-bottom: 1rem;
  }
  .how-to strong { color: var(--text); }
  .how-to ol { padding-left: 1.2rem; line-height: 2.1; }
  .how-to code {
    background: rgba(255,255,255,0.09);
    padding: 0.1em 0.4em;
    border-radius: 3px;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.77rem;
  }
  .upload-zone {
    border: 2px dashed var(--border2);
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    display: block;
  }
  .upload-zone:hover, .upload-zone.drag-over {
    border-color: var(--accent);
    background: rgba(99,102,241,0.06);
  }
  .upload-zone input { display: none; }
  .upload-zone-label { font-size: 0.88rem; font-weight: 500; color: var(--text); }
  .upload-zone-sub { font-size: 0.78rem; color: var(--muted); margin-top: 0.2rem; }
  .upload-zone-chosen { font-size: 0.82rem; color: var(--success); font-weight: 500; margin-top: 0.5rem; }

  /* ── Signal strength bar ── */
  .signal-bar-row {
    display: flex; align-items: center; gap: 0.6rem;
    margin-top: 1.25rem; margin-bottom: 0.25rem;
  }
  .signal-bar-label { font-size: 0.75rem; color: var(--muted); white-space: nowrap; }
  .signal-bar-track {
    flex: 1; height: 4px; background: var(--border2);
    border-radius: 2px; overflow: hidden;
  }
  .signal-bar-fill {
    height: 100%; width: 0%;
    background: linear-gradient(90deg, #6366f1, #818cf8);
    border-radius: 2px;
    transition: width 0.4s ease;
  }
  .signal-bar-count { font-size: 0.75rem; color: var(--muted2); white-space: nowrap; }

  /* ── Generate button & loading ── */
  .gen-section { margin-top: 1.5rem; }
  button.primary {
    width: 100%; padding: 0.9rem;
    background: var(--accent); color: white;
    border: none; border-radius: 9px;
    font-size: 0.97rem; font-weight: 600; letter-spacing: -0.01em;
    cursor: pointer; transition: background 0.2s, transform 0.1s;
    font-family: inherit;
  }
  button.primary:hover:not(:disabled) { background: var(--accent-hover); }
  button.primary:active:not(:disabled) { transform: scale(0.99); }
  button.primary:disabled { opacity: 0.45; cursor: not-allowed; }
  .loading-wrap { display: none; flex-direction: column; align-items: center; gap: 0.75rem; padding: 1.25rem 0; }
  .dots { display: flex; gap: 6px; align-items: center; }
  .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent); opacity: 0.3;
    animation: dotPulse 1.4s ease-in-out infinite;
  }
  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes dotPulse {
    0%, 80%, 100% { opacity: 0.3; transform: scale(0.85); }
    40% { opacity: 1; transform: scale(1); }
  }
  .stage-msg { font-size: 0.88rem; color: var(--muted); text-align: center; min-height: 1.4em; }
  .stage-msg.error { color: var(--error); }
  .timing-hint {
    font-size: 0.73rem; color: var(--muted2);
    text-align: center; margin-top: 0.4rem;
  }

  /* ── Privacy note ── */
  .privacy-note {
    font-size: 0.78rem; color: var(--muted);
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 1.25rem;
    line-height: 1.65;
  }
  .privacy-note strong { color: var(--text); }

  /* ── Result panel ── */
  @keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  #result { display: none; animation: fadeSlideUp 0.4s ease; }
  .result-header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.25rem;
    margin-bottom: 1.5rem;
  }
  .result-eyebrow {
    font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: var(--accent); margin-bottom: 0.4rem;
  }
  .result-subject { font-size: 1.55rem; font-weight: 700; line-height: 1.3; letter-spacing: -0.02em; }
  .result-meta { font-size: 0.78rem; color: var(--muted); margin-top: 0.45rem; }

  /* ── Curiosity profile block ── */
  .curiosity-profile {
    margin-bottom: 1.5rem;
    padding: 1.1rem 1.25rem;
    background: rgba(52,211,153,0.04);
    border: 1px solid rgba(52,211,153,0.18);
    border-radius: 10px;
  }
  .profile-label {
    font-size: 0.68rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: #34d399; margin-bottom: 0.75rem;
  }
  .topic-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.75rem; }
  .topic-chip {
    display: inline-block; font-size: 0.74rem; font-weight: 500;
    padding: 0.28em 0.75em; border-radius: 20px;
    background: rgba(99,102,241,0.15); color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.3);
    cursor: default; transition: background 0.15s;
  }
  .topic-chip:hover { background: rgba(99,102,241,0.25); }
  .regen-bar { display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; }
  .btn-regen {
    padding: 0.4rem 1rem;
    background: rgba(52,211,153,0.12);
    border: 1px solid rgba(52,211,153,0.35);
    border-radius: 7px;
    color: #34d399;
    font-size: 0.8rem; font-weight: 500;
    cursor: pointer; transition: all 0.2s;
    font-family: inherit;
  }
  .btn-regen:hover:not(:disabled) { background: rgba(52,211,153,0.22); border-color: #34d399; }
  .btn-regen:disabled { opacity: 0.45; cursor: not-allowed; }
  .regen-link { font-size: 0.74rem; color: var(--muted); font-family: 'SF Mono', monospace; }
  .regen-link a { color: var(--accent); text-decoration: none; }
  .regen-link a:hover { text-decoration: underline; }

  /* ── Download / share row ── */
  .action-row {
    display: flex; gap: 0.6rem; margin-bottom: 1.25rem; flex-wrap: wrap;
    align-items: center;
  }
  .btn-action {
    padding: 0.42rem 1rem;
    background: transparent;
    border: 1px solid var(--border2);
    border-radius: 7px;
    color: var(--muted);
    font-size: 0.8rem; font-weight: 500;
    cursor: pointer; transition: all 0.2s;
    font-family: inherit;
  }
  .btn-action:hover { border-color: var(--accent); color: var(--accent); }
  .share-link { font-size: 0.74rem; color: var(--muted); margin-left: auto; }
  .share-link a { color: var(--accent); text-decoration: none; }
  .share-link a:hover { text-decoration: underline; }

  /* ── Table of contents ── */
  .toc {
    margin-bottom: 1.5rem;
    padding: 1rem 1.25rem;
    background: rgba(99,102,241,0.05);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 9px;
  }
  .toc-label {
    font-size: 0.68rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: var(--accent); margin-bottom: 0.6rem;
  }
  .toc ol { padding-left: 1.2rem; }
  .toc li { margin-top: 0.3rem; font-size: 0.86rem; line-height: 1.5; }
  .toc a { color: var(--text); text-decoration: none; }
  .toc a:hover { color: var(--accent); }

  /* ── Newsletter sections ── */
  .section {
    margin-bottom: 2rem; padding-bottom: 1.75rem;
    border-bottom: 1px solid var(--border);
  }
  .section:last-child { border-bottom: none; margin-bottom: 0; }
  .section-meta { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.55rem; }
  .section-badge {
    display: inline-block;
    font-size: 0.67rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em;
    padding: 0.22em 0.65em; border-radius: 4px;
  }
  .badge-weekly_topics { background: rgba(99,102,241,0.18); color: #818cf8; }
  .badge-curiosity_debt { background: rgba(251,191,36,0.14); color: #fbbf24; }
  .badge-rabbit_hole { background: rgba(52,211,153,0.14); color: #34d399; }
  .section h3 { font-size: 1.12rem; font-weight: 700; margin-bottom: 0.7rem; line-height: 1.4; letter-spacing: -0.01em; }
  .section .content-body p { color: #c8d3e0; line-height: 1.8; font-size: 0.94rem; margin-bottom: 0.85rem; }
  .section .content-body p:last-child { margin-bottom: 0; }

  /* ── Sources ── */
  .sources { margin-top: 1rem; display: flex; flex-direction: column; gap: 0.35rem; }
  .sources a {
    display: flex; align-items: center; gap: 0.45rem;
    font-size: 0.79rem; color: var(--accent); text-decoration: none;
  }
  .sources a:hover { color: var(--accent-hover); }
  .src-badge {
    display: inline-block; font-size: 0.62rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em;
    padding: 0.15em 0.5em; border-radius: 3px;
    flex-shrink: 0;
  }
  .src-arxiv { background: rgba(180,120,255,0.2); color: #c084fc; }
  .src-hn    { background: rgba(251,146,60,0.2);  color: #fb923c; }
  .src-web   { background: rgba(56,189,248,0.2);  color: #38bdf8; }
  .link-text {
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    max-width: 500px;
  }

  /* ── Audit details ── */
  details.audit {
    margin-top: 0.85rem;
    border: 1px solid var(--border);
    border-radius: 7px;
    overflow: hidden;
  }
  details.audit summary {
    padding: 0.5rem 0.85rem;
    font-size: 0.76rem; color: var(--muted2);
    cursor: pointer; user-select: none; list-style: none;
  }
  details.audit summary::-webkit-details-marker { display: none; }
  details.audit summary::before { content: "▶  "; font-size: 0.58rem; }
  details[open].audit summary::before { content: "▼  "; }
  details.audit .audit-body {
    padding: 0.8rem;
    font-size: 0.8rem; color: var(--muted);
    border-top: 1px solid var(--border);
    background: rgba(0,0,0,0.18);
    line-height: 1.65;
  }

  /* ── Errors ── */
  .errors-box {
    margin-top: 1rem;
    padding: 0.85rem 1rem;
    background: rgba(248,113,113,0.07);
    border: 1px solid rgba(248,113,113,0.22);
    border-radius: 8px;
    font-size: 0.8rem; color: #fca5a5;
  }
  .errors-box h4 { margin-bottom: 0.4rem; font-weight: 600; }
  .errors-box li { margin-left: 1rem; margin-top: 0.25rem; line-height: 1.5; }

  footer {
    margin-top: 3rem; text-align: center;
    font-size: 0.74rem; color: var(--muted2);
    letter-spacing: 0.02em;
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <div style="display:inline-block;padding:0.3rem 0.7rem;background:var(--accent-dim);color:var(--accent);font-size:0.7rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;border-radius:999px;margin-bottom:1rem;">Curiosity OS &middot; MCP-native</div>
    <div class="logo">Tr<span>a</span>ce</div>
    <p class="tagline">The personal curiosity data plane for AI agents.</p>
    <p style="color:var(--muted2,#6b7280);margin-top:0.5rem;font-size:0.88rem;max-width:560px;margin-left:auto;margin-right:auto;line-height:1.5;">
      Every AI agent will need to know what <em>you</em> care about. Trace exposes your real interests &mdash; inferred from what you actually browse, search, and read &mdash; as MCP tools any agent can query.
    </p>
  </header>

  <!-- MCP Connection Banner -->
  <div class="card" style="background:linear-gradient(135deg,rgba(99,102,241,0.08),rgba(99,102,241,0.02));border-color:rgba(99,102,241,0.3);margin-bottom:1.25rem;">
    <div class="card-title" style="color:var(--accent)">Connect Trace to Claude Desktop</div>
    <div style="font-size:0.85rem;color:var(--muted);line-height:1.6;margin-bottom:0.85rem;">
      Trace is an MCP server. Add this snippet to your <code style="color:var(--text,#f1f5f9);background:var(--surface2,#1e293b);padding:0.1rem 0.4rem;border-radius:4px;font-size:0.8rem;">claude_desktop_config.json</code> and Claude can call Trace tools directly.
    </div>
    <div style="background:var(--surface2,#1e293b);border:1px solid var(--border2,#334155);border-radius:8px;padding:0.9rem 1rem;font-family:'SF Mono',Consolas,monospace;font-size:0.78rem;color:var(--muted);line-height:1.8;overflow-x:auto;">
      <div style="color:var(--muted2,#6b7280);font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem;font-family:Inter,sans-serif;">claude_desktop_config.json</div>
      <div><span style="color:#818cf8">"mcpServers"</span>: {</div>
      <div>&nbsp;&nbsp;<span style="color:#34d399">"trace"</span>: { <span style="color:#818cf8">"url"</span>: <span style="color:#fbbf24">"http://localhost:8000/mcp"</span>, <span style="color:#818cf8">"transport"</span>: <span style="color:#fbbf24">"http"</span> }</div>
      <div>}</div>
    </div>
    <div style="margin-top:0.85rem;display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:0.5rem;font-size:0.72rem;">
      <div style="background:var(--surface2,#1e293b);border:1px solid var(--border2,#334155);border-radius:6px;padding:0.5rem 0.7rem;">
        <div style="color:var(--accent);font-weight:600;font-size:0.68rem;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.2rem;">Tool</div>
        <code style="color:var(--text,#f1f5f9);font-size:0.75rem;">get_curiosity_topics</code>
      </div>
      <div style="background:var(--surface2,#1e293b);border:1px solid var(--border2,#334155);border-radius:6px;padding:0.5rem 0.7rem;">
        <div style="color:var(--accent);font-weight:600;font-size:0.68rem;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.2rem;">Tool</div>
        <code style="color:var(--text,#f1f5f9);font-size:0.75rem;">get_unresolved_questions</code>
      </div>
      <div style="background:var(--surface2,#1e293b);border:1px solid var(--border2,#334155);border-radius:6px;padding:0.5rem 0.7rem;">
        <div style="color:var(--accent);font-weight:600;font-size:0.68rem;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.2rem;">Tool</div>
        <code style="color:var(--text,#f1f5f9);font-size:0.75rem;">generate_briefing</code>
      </div>
      <div style="background:var(--surface2,#1e293b);border:1px solid var(--border2,#334155);border-radius:6px;padding:0.5rem 0.7rem;">
        <div style="color:var(--accent);font-weight:600;font-size:0.68rem;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.2rem;">Tool</div>
        <code style="color:var(--text,#f1f5f9);font-size:0.75rem;">track_signal</code>
      </div>
    </div>
  </div>

  <!-- Sponsor Stack -->
  <div class="card" style="margin-bottom:1.25rem;">
    <div class="card-title">Powered by</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:0.7rem;font-size:0.78rem;">
      <div style="background:var(--surface2,#1e293b);border:1px solid var(--border2,#334155);border-radius:8px;padding:0.85rem 0.9rem;">
        <div style="font-weight:600;color:var(--text,#f1f5f9);margin-bottom:0.2rem;">Apify MCP</div>
        <div style="color:var(--muted);font-size:0.72rem;line-height:1.4;">Dynamic Actor selection · 31k+ scrapers</div>
      </div>
      <div style="background:var(--surface2,#1e293b);border:1px solid var(--border2,#334155);border-radius:8px;padding:0.85rem 0.9rem;">
        <div style="font-weight:600;color:var(--text,#f1f5f9);margin-bottom:0.2rem;">Scalekit</div>
        <div style="color:var(--muted);font-size:0.72rem;line-height:1.4;">MCP Auth (OAuth 2.1) + Token Vault (Apify bridge)</div>
      </div>
      <div style="background:var(--surface2,#1e293b);border:1px solid var(--border2,#334155);border-radius:8px;padding:0.85rem 0.9rem;">
        <div style="font-weight:600;color:var(--text,#f1f5f9);margin-bottom:0.2rem;">Redis</div>
        <div style="color:var(--muted);font-size:0.72rem;line-height:1.4;">Sub-50ms curiosity graph reads via ZSET</div>
      </div>
      <div style="background:var(--surface2,#1e293b);border:1px solid var(--border2,#334155);border-radius:8px;padding:0.85rem 0.9rem;">
        <div style="font-weight:600;color:var(--text,#f1f5f9);margin-bottom:0.2rem;">Anthropic Claude</div>
        <div style="color:var(--muted);font-size:0.72rem;line-height:1.4;">Topic extraction + briefing composition</div>
      </div>
    </div>
  </div>

  <!-- Connect Services (Scalekit Token Vault) -->
  <div class="card" style="border-color:#6366f1;">
    <div class="card-title" style="color:#818cf8;">Connect Services &mdash; Scalekit acts as you</div>
    <p style="color:var(--muted);font-size:0.82rem;margin-bottom:1rem;">
      Trace uses <strong>Scalekit Token Vault</strong> to connect your accounts.
      Your OAuth tokens live in Scalekit's encrypted vault &mdash; never in Trace's env vars or memory.
      Once connected, the autonomous agent can read your Gmail newsletters and act on your behalf.
    </p>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:0.7rem;font-size:0.82rem;">
      <div style="background:var(--surface2,#1e293b);border:1px solid #6366f1;border-radius:8px;padding:1rem;">
        <div style="font-weight:600;color:#f1f5f9;margin-bottom:0.4rem;">📧 Gmail</div>
        <div style="color:var(--muted);font-size:0.75rem;margin-bottom:0.7rem;">
          Read newsletter subjects → build subscription-debt signals.
          Trace never reads email bodies or sends emails automatically.
        </div>
        <button onclick="connectService('gmail')" style="background:#6366f1;color:#fff;border:none;border-radius:5px;padding:0.4rem 0.8rem;cursor:pointer;font-size:0.78rem;">Connect Gmail</button>
        <span id="gmail-status" style="color:var(--muted);font-size:0.72rem;margin-left:0.5rem;"></span>
      </div>
      <div style="background:var(--surface2,#1e293b);border:1px solid #334155;border-radius:8px;padding:1rem;">
        <div style="font-weight:600;color:#f1f5f9;margin-bottom:0.4rem;">📝 Notion</div>
        <div style="color:var(--muted);font-size:0.75rem;margin-bottom:0.7rem;">
          Auto-save emerging topics as Notion pages (Tier A action).
        </div>
        <button onclick="connectService('notion')" style="background:#334155;color:#94a3b8;border:1px solid #475569;border-radius:5px;padding:0.4rem 0.8rem;cursor:pointer;font-size:0.78rem;">Connect Notion</button>
        <span id="notion-status" style="color:var(--muted);font-size:0.72rem;margin-left:0.5rem;"></span>
      </div>
      <div style="background:var(--surface2,#1e293b);border:1px solid #334155;border-radius:8px;padding:1rem;">
        <div style="font-weight:600;color:#f1f5f9;margin-bottom:0.4rem;">📅 Calendar</div>
        <div style="color:var(--muted);font-size:0.75rem;margin-bottom:0.7rem;">
          Schedule deep-dive time for emerging interests (Tier A).
        </div>
        <button onclick="connectService('google_calendar')" style="background:#334155;color:#94a3b8;border:1px solid #475569;border-radius:5px;padding:0.4rem 0.8rem;cursor:pointer;font-size:0.78rem;">Connect Calendar</button>
        <span id="google_calendar-status" style="color:var(--muted);font-size:0.72rem;margin-left:0.5rem;"></span>
      </div>
      <div style="background:var(--surface2,#1e293b);border:1px solid #334155;border-radius:8px;padding:1rem;">
        <div style="font-weight:600;color:#f1f5f9;margin-bottom:0.4rem;">💬 Slack</div>
        <div style="color:var(--muted);font-size:0.75rem;margin-bottom:0.7rem;">
          DM yourself when a pattern triggers (Tier A).
        </div>
        <button onclick="connectService('slack')" style="background:#334155;color:#94a3b8;border:1px solid #475569;border-radius:5px;padding:0.4rem 0.8rem;cursor:pointer;font-size:0.78rem;">Connect Slack</button>
        <span id="slack-status" style="color:var(--muted);font-size:0.72rem;margin-left:0.5rem;"></span>
      </div>
    </div>
  </div>

  <div class="privacy-note">
    <strong>Your data stays local.</strong>
    You export your own files from Google / ChatGPT — no passwords or OAuth tokens required.
    Files are sent only to this server, used once to generate your newsletter, then
    <strong>deleted immediately</strong>. Nothing is stored or shared.
    <br>OAuth tokens for Gmail/Notion/Calendar/Slack are stored in Scalekit's encrypted Token Vault — never in this server's memory or env vars.
  </div>

  <div class="card">
    <div class="card-title">Web UI &middot; Upload signals &rarr; get a briefing</div>

    <div class="source-cards">

      <!-- Chrome history -->
      <div class="source-card" id="card-google">
        <div class="source-card-header" onclick="toggleCard('google')">
          <div class="source-card-left">
            <div class="source-icon">🌐</div>
            <div>
              <div class="source-name">Chrome / Browser History</div>
              <div class="source-desc">BrowserHistory.json or Takeout ZIP · multiple files supported</div>
            </div>
          </div>
          <div style="display:flex;align-items:center;gap:0.6rem">
            <span class="source-status" id="status-google">Optional</span>
            <span class="expand-toggle">▼</span>
          </div>
        </div>
        <div class="source-body" id="body-google">
          <div class="how-to">
            <ol>
              <li>Go to <strong>takeout.google.com</strong></li>
              <li>Deselect all → select only <code>Chrome</code></li>
              <li>Export → Download (you can export multiple date ranges)</li>
              <li>Upload the <strong>ZIP directly</strong>, or extract and upload <code>BrowserHistory.json</code></li>
              <li>Select <strong>multiple files at once</strong> if you have several exports</li>
            </ol>
          </div>
          <label class="upload-zone" id="drop-google">
            <input type="file" id="file-google" accept=".json,.zip" multiple onchange="fileChosen('google')">
            <div class="upload-zone-label">Drop BrowserHistory.json or Takeout ZIP(s) here</div>
            <div class="upload-zone-sub">or click to browse · .json and .zip · multiple files OK</div>
            <div class="upload-zone-chosen" id="chosen-google"></div>
          </label>
        </div>
      </div>

      <!-- YouTube history -->
      <div class="source-card" id="card-youtube">
        <div class="source-card-header" onclick="toggleCard('youtube')">
          <div class="source-card-left">
            <div class="source-icon">▶</div>
            <div>
              <div class="source-name">YouTube Watch History</div>
              <div class="source-desc">watch-history.json from Google Takeout</div>
            </div>
          </div>
          <div style="display:flex;align-items:center;gap:0.6rem">
            <span class="source-status" id="status-youtube">Optional</span>
            <span class="expand-toggle">▼</span>
          </div>
        </div>
        <div class="source-body" id="body-youtube">
          <div class="how-to">
            <ol>
              <li>Go to <strong>takeout.google.com</strong></li>
              <li>Deselect all → select <code>YouTube and YouTube Music</code></li>
              <li>Export → Download → extract the ZIP</li>
              <li>Find <code>Takeout/YouTube and YouTube Music/history/watch-history.json</code></li>
            </ol>
            <p style="margin-top:0.5rem;font-size:0.79rem">
              Re-watched lectures count as curiosity debt signals — they show interests you keep returning to.
            </p>
          </div>
          <label class="upload-zone" id="drop-youtube">
            <input type="file" id="file-youtube" accept=".json" onchange="fileChosen('youtube')">
            <div class="upload-zone-label">Drop watch-history.json here</div>
            <div class="upload-zone-sub">or click to browse · .json only</div>
            <div class="upload-zone-chosen" id="chosen-youtube"></div>
          </label>
        </div>
      </div>

      <!-- ChatGPT export -->
      <div class="source-card" id="card-chatgpt">
        <div class="source-card-header" onclick="toggleCard('chatgpt')">
          <div class="source-card-left">
            <div class="source-icon">💬</div>
            <div>
              <div class="source-name">ChatGPT Export</div>
              <div class="source-desc">conversations.json — highest-signal source</div>
            </div>
          </div>
          <div style="display:flex;align-items:center;gap:0.6rem">
            <span class="source-status" id="status-chatgpt">Optional</span>
            <span class="expand-toggle">▼</span>
          </div>
        </div>
        <div class="source-body" id="body-chatgpt">
          <div class="how-to">
            <ol>
              <li>Open ChatGPT → click your avatar → <strong>Settings</strong></li>
              <li>Go to <strong>Data Controls → Export Data</strong> → click Export</li>
              <li>Wait for the confirmation email → click Download</li>
              <li>Extract the ZIP → upload <code>conversations.json</code></li>
            </ol>
            <p style="margin-top:0.5rem;font-size:0.79rem">
              ChatGPT conversations carry 1.5× weight — they represent your most explicit intellectual intent.
            </p>
          </div>
          <label class="upload-zone" id="drop-chatgpt">
            <input type="file" id="file-chatgpt" accept=".json" onchange="fileChosen('chatgpt')">
            <div class="upload-zone-label">Drop conversations.json here</div>
            <div class="upload-zone-sub">or click to browse · .json only</div>
            <div class="upload-zone-chosen" id="chosen-chatgpt"></div>
          </label>
        </div>
      </div>

    </div><!-- /.source-cards -->

    <!-- Signal strength bar -->
    <div class="signal-bar-row">
      <span class="signal-bar-label">Signal strength</span>
      <div class="signal-bar-track"><div class="signal-bar-fill" id="sig-fill"></div></div>
      <span class="signal-bar-count" id="sig-count">0 / 3 sources</span>
    </div>

    <div class="gen-section">
      <div id="inline-error" style="display:none;background:rgba(248,113,113,0.1);border:1px solid rgba(248,113,113,0.4);border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.75rem;font-size:0.85rem;color:#fca5a5;"></div>
      <button class="primary" id="gen-btn" onclick="generate()">Generate My Newsletter</button>
      <div class="loading-wrap" id="loading-wrap">
        <div class="dots">
          <div class="dot"></div><div class="dot"></div><div class="dot"></div>
        </div>
        <div class="stage-msg" id="stage-msg"></div>
      </div>
      <p class="timing-hint">Takes <strong style="color:var(--muted)">60–120 seconds</strong> — Claude reads your entire history and writes a personalised newsletter. Keep this tab open.</p>
    </div>
  </div><!-- /.card -->

  <div id="result" class="card">
    <div class="result-header">
      <div class="result-eyebrow">Your Personalised Newsletter</div>
      <div class="result-subject" id="subject"></div>
      <div class="result-meta" id="meta"></div>
    </div>
    <div id="curiosity-profile"></div>
    <div class="action-row" id="action-row">
      <button class="btn-action" onclick="downloadHtml()">⬇ HTML</button>
      <button class="btn-action" onclick="downloadText()">⬇ Plain Text</button>
      <div id="share-link-container"></div>
    </div>
    <div id="toc"></div>
    <div id="sections"></div>
    <div id="errors-container"></div>
  </div>
</div>

<footer>Trace &mdash; Curiosity OS &middot; MCP AI Agents Hackathon &middot; Claude &middot; Apify MCP &middot; Scalekit &middot; Redis</footer>

<script>
let newsletterData = null;
let fileCount = 0;

// ── Card expand/collapse ──────────────────────────────────────────────────────
function toggleCard(type) {
  var card = document.getElementById('card-' + type);
  card.classList.toggle('expanded');
}

// All cards start expanded so upload zones are immediately visible
document.getElementById('card-google').classList.add('expanded');
document.getElementById('card-youtube').classList.add('expanded');
document.getElementById('card-chatgpt').classList.add('expanded');

// ── File chosen ───────────────────────────────────────────────────────────────
function fileChosen(type) {
  const input = document.getElementById('file-' + type);
  const files = Array.from(input.files);
  const chosenEl = document.getElementById('chosen-' + type);
  const statusEl = document.getElementById('status-' + type);
  const card = document.getElementById('card-' + type);
  if (files.length > 0) {
    // For google (multi-file), list all names; for others just the one
    const label = files.length === 1
      ? '✓ ' + files[0].name
      : '✓ ' + files.length + ' files: ' + files.map(f => f.name).join(', ');
    chosenEl.textContent = label;
    statusEl.textContent = files.length > 1 ? '✓ ' + files.length + ' files' : '✓ Ready';
    statusEl.className = 'source-status chosen';
    card.classList.add('has-file');
  } else {
    chosenEl.textContent = '';
    statusEl.textContent = 'Optional';
    statusEl.className = 'source-status';
    card.classList.remove('has-file');
  }
  updateSignalBar();
}

function updateSignalBar() {
  const types = ['google','youtube','chatgpt'];
  const count = types.filter(t => document.getElementById('file-' + t).files.length > 0).length;
  fileCount = count;
  document.getElementById('sig-fill').style.width = (count / 3 * 100) + '%';
  document.getElementById('sig-count').textContent = count + ' / 3 sources';
}

// ── Drag-and-drop ─────────────────────────────────────────────────────────────
['google','youtube','chatgpt'].forEach(t => {
  const el = document.getElementById('drop-' + t);
  el.addEventListener('dragover', e => { e.preventDefault(); el.classList.add('drag-over'); });
  el.addEventListener('dragleave', () => el.classList.remove('drag-over'));
  el.addEventListener('drop', e => {
    e.preventDefault(); el.classList.remove('drag-over');
    const dt = e.dataTransfer;
    if (dt.files.length) {
      // For google, allow multiple dropped files; for others take the first
      if (t === 'google') {
        // DataTransfer.files is read-only — we can't directly assign multiple
        // dropped files to an input. Use a workaround via DataTransfer API.
        try {
          const dta = new DataTransfer();
          Array.from(dt.files).forEach(f => dta.items.add(f));
          document.getElementById('file-' + t).files = dta.files;
        } catch {
          document.getElementById('file-' + t).files = dt.files;
        }
      } else {
        document.getElementById('file-' + t).files = dt.files;
      }
      fileChosen(t);
    }
  });
});

// ── Stage ticker ──────────────────────────────────────────────────────────────
const STAGE_MSGS = [
  'Reading your history and signals…',
  'Clustering curiosity topics with Claude — this takes 20–40s…',
  'Fetching fresh articles from arXiv, Hacker News & web…',
  'Assembling your personalised context window…',
  'Writing your newsletter with Claude…',
  'Still working — large histories can take up to 2 minutes…',
  'Almost there — finalising your newsletter…',
];
const STAGE_DELAYS = [3000, 18000, 15000, 5000, 15000, 20000, 20000];
let stageIdx = 0;
let stageTimer;

function tickStage() {
  if (stageIdx < STAGE_MSGS.length) {
    document.getElementById('stage-msg').textContent = STAGE_MSGS[stageIdx];
    stageTimer = setTimeout(tickStage, STAGE_DELAYS[stageIdx] || 15000);
    stageIdx++;
  } else {
    document.getElementById('stage-msg').textContent = 'Still processing — complex histories can take a few minutes…';
    stageTimer = setTimeout(tickStage, 20000);
  }
}

function stopStages() { clearTimeout(stageTimer); stageIdx = 0; }

function setLoading(loading) {
  document.getElementById('gen-btn').style.display = loading ? 'none' : 'block';
  document.getElementById('loading-wrap').style.display = loading ? 'flex' : 'none';
  if (!loading) document.getElementById('stage-msg').className = 'stage-msg';
}

function setError(msg) {
  const el = document.getElementById('stage-msg');
  el.textContent = msg;
  el.className = 'stage-msg error';
  document.getElementById('loading-wrap').style.display = 'flex';
  document.getElementById('gen-btn').style.display = 'block';
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function safeHref(u) {
  if (typeof u !== 'string') return '#';
  const l = u.toLowerCase();
  return (l.startsWith('http://') || l.startsWith('https://')) ? u : '#';
}

function sourceBadge(url) {
  if (url.includes('arxiv.org')) return '<span class="src-badge src-arxiv">arXiv</span>';
  if (url.includes('ycombinator.com')) return '<span class="src-badge src-hn">HN</span>';
  return '<span class="src-badge src-web">Web</span>';
}

function badgeClass(type) { return 'section-badge badge-' + (type || 'weekly_topics'); }

function badgeLabel(type) {
  return { weekly_topics: 'This Week', curiosity_debt: 'Curiosity Debt', rabbit_hole: 'Rabbit Hole' }[type] || type;
}

// Split content on blank lines into separate <p> tags
function fmtContent(text) {
  var parts = String(text).split('\\n\\n').map(function(p) { return p.trim(); }).filter(Boolean);
  if (parts.length === 0) return '<p>' + esc(String(text).trim()) + '</p>';
  return parts.map(function(p) { return '<p>' + esc(p) + '</p>'; }).join('');
}

function downloadBlob(content, filename, mime) {
  const blob = new Blob([content], { type: mime });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function downloadHtml() {
  if (!newsletterData) return;
  const subj = newsletterData.subject_line.replace(/[^a-z0-9]+/gi, '-').toLowerCase();
  downloadBlob(newsletterData.html, `trace-${subj}.html`, 'text/html');
}

function downloadText() {
  if (!newsletterData) return;
  const subj = newsletterData.subject_line.replace(/[^a-z0-9]+/gi, '-').toLowerCase();
  downloadBlob(newsletterData.plain_text, `trace-${subj}.txt`, 'text/plain');
}

// ── Regenerate ────────────────────────────────────────────────────────────────
async function regenerate() {
  if (!newsletterData || !newsletterData.profile_id) return;
  const btn = document.getElementById('regen-btn');
  if (btn) btn.disabled = true;
  document.getElementById('result').style.display = 'none';
  setLoading(true);
  stageIdx = 0;
  tickStage();
  try {
    const r = await fetch('/newsletter/regenerate/' + encodeURIComponent(newsletterData.profile_id), { method: 'POST' });
    if (!r.ok) {
      let msg = 'Regeneration failed';
      try { const ej = await r.json(); msg = ej.detail || msg; } catch (_e) { try { msg = await r.text(); } catch (_e2) {} }
      throw new Error(msg);
    }
    const data = await r.json();
    stopStages();
    setLoading(false);
    renderNewsletter(data);
  } catch (err) {
    stopStages();
    setLoading(false);
    setError('Error: ' + err.message);
    if (btn) btn.disabled = false;
    document.getElementById('result').style.display = 'block';
  }
}

// ── Render newsletter ─────────────────────────────────────────────────────────
function renderNewsletter(data) {
  newsletterData = data;
  document.getElementById('subject').textContent = data.subject_line;
  const dt = new Date(data.generated_at);
  const forStr = data.generated_for ? ' · for ' + data.generated_for : '';
  document.getElementById('meta').textContent = dt.toLocaleString() + forStr;

  // Curiosity profile chips
  const profileEl = document.getElementById('curiosity-profile');
  if (data.topic_names && data.topic_names.length > 0) {
    const scores = data.topic_scores || [];
    const chips = data.topic_names.map((t, i) => {
      const s = scores[i] != null ? scores[i] : 0;
      const bg  = (0.12 + 0.33 * s).toFixed(2);
      const bdr = (0.20 + 0.55 * s).toFixed(2);
      const tip = scores[i] != null ? ` title="Curiosity strength: ${Math.round(s*100)}%"` : '';
      return `<span class="topic-chip"${tip} style="background:rgba(99,102,241,${bg});border-color:rgba(99,102,241,${bdr})">${esc(t)}</span>`;
    }).join('');
    const regenHtml = data.profile_id ? `
      <div class="regen-bar">
        <button class="btn-regen" id="regen-btn" onclick="regenerate()">↺ Regenerate with today's articles</button>
        <span class="regen-link">Bookmark: <a href="/newsletter/regenerate/${esc(data.profile_id)}" onclick="return false;">/regenerate/${esc(data.profile_id.substring(0,8))}…</a></span>
      </div>` : '';
    profileEl.innerHTML = `<div class="curiosity-profile">
      <div class="profile-label">Curiosity Profile · ${data.topic_names.length} topic${data.topic_names.length !== 1 ? 's' : ''} inferred</div>
      <div class="topic-chips">${chips}</div>
      ${regenHtml}
    </div>`;
  } else {
    profileEl.innerHTML = '';
  }

  // Share link
  const shareCont = document.getElementById('share-link-container');
  shareCont.innerHTML = data.id
    ? `<span class="share-link">Permalink: <a href="/newsletter/${esc(data.id)}" target="_blank">/newsletter/${esc(data.id)}</a></span>`
    : '';

  // Table of contents
  const tocEl = document.getElementById('toc');
  if (data.sections && data.sections.length > 1) {
    const items = data.sections.map((s, i) =>
      `<li><a href="#section-${i}">${esc(s.title)}</a></li>`
    ).join('');
    tocEl.innerHTML = `<div class="toc"><div class="toc-label">In this issue</div><ol>${items}</ol></div>`;
  } else {
    tocEl.innerHTML = '';
  }

  // Sections
  const secEl = document.getElementById('sections');
  secEl.innerHTML = '';
  data.sections.forEach((s, i) => {
    const div = document.createElement('div');
    div.className = 'section';
    div.id = `section-${i}`;
    const urls = (s.source_urls || []).map(u => {
      const href = esc(safeHref(u));
      return `<a href="${href}" target="_blank" rel="noopener noreferrer">${sourceBadge(u)}<span class="link-text">${esc(u)}</span></a>`;
    }).join('');
    div.innerHTML = `
      <div class="section-meta">
        <span class="${badgeClass(s.section_type)}">${esc(badgeLabel(s.section_type))}</span>
      </div>
      <h3>${esc(s.title)}</h3>
      <div class="content-body">${fmtContent(s.content)}</div>
      ${urls ? '<div class="sources">' + urls + '</div>' : ''}
      <details class="audit">
        <summary>Why this section?</summary>
        <div class="audit-body">${esc(s.audit_reasoning)}</div>
      </details>`;
    secEl.appendChild(div);
  });

  // Errors
  const errBox = document.getElementById('errors-container');
  errBox.innerHTML = '';
  if (data.errors && data.errors.length) {
    errBox.innerHTML = `<div class="errors-box"><h4>Non-fatal warnings (${data.errors.length})</h4><ul>${
      data.errors.map(e => '<li>' + esc(e) + '</li>').join('')
    }</ul></div>`;
  }

  const resultEl = document.getElementById('result');
  resultEl.style.display = 'block';
  resultEl.scrollIntoView({ behavior: 'smooth' });
}

// ── Generate ──────────────────────────────────────────────────────────────────
async function generate() {
  const googleFiles = Array.from(document.getElementById('file-google').files);
  const youtubeFiles = Array.from(document.getElementById('file-youtube').files);
  const chatgptFiles = Array.from(document.getElementById('file-chatgpt').files);

  if (googleFiles.length === 0 && youtubeFiles.length === 0 && chatgptFiles.length === 0) {
    setError('Please upload at least one file first.');
    document.getElementById('loading-wrap').style.display = 'flex';
    return;
  }

  document.getElementById('result').style.display = 'none';
  setLoading(true);
  stageIdx = 0;
  tickStage();

  try {
    // Upload all Chrome/Google history files (may be multiple ZIPs or JSONs)
    const historyUploadIds = [];
    for (const f of googleFiles) {
      const fd = new FormData();
      fd.append('file', f);
      fd.append('file_type', 'history');  // explicit — bypass filename heuristic
      const r = await fetch('/upload', { method: 'POST', body: fd });
      if (!r.ok) { const e = await r.json(); throw new Error(e.detail || 'Upload failed: ' + f.name); }
      historyUploadIds.push((await r.json()).upload_id);
    }

    let youtubeUploadId = null;
    if (youtubeFiles[0]) {
      const fd = new FormData();
      fd.append('file', youtubeFiles[0]);
      fd.append('file_type', 'youtube');
      const r = await fetch('/upload', { method: 'POST', body: fd });
      if (!r.ok) { const e = await r.json(); throw new Error(e.detail || 'Upload failed'); }
      youtubeUploadId = (await r.json()).upload_id;
    }

    let chatgptUploadId = null;
    if (chatgptFiles[0]) {
      const fd = new FormData();
      fd.append('file', chatgptFiles[0]);
      fd.append('file_type', 'chatgpt');
      const r = await fetch('/upload', { method: 'POST', body: fd });
      if (!r.ok) { const e = await r.json(); throw new Error(e.detail || 'Upload failed'); }
      chatgptUploadId = (await r.json()).upload_id;
    }

    const body = {};
    if (historyUploadIds.length > 0) body.history_upload_ids = historyUploadIds;
    if (youtubeUploadId) body.youtube_upload_id = youtubeUploadId;
    if (chatgptUploadId) body.chatgpt_upload_id = chatgptUploadId;

    const r2 = await fetch('/newsletter/from-upload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r2.ok) {
      let msg = 'Generation failed';
      try { const ej2 = await r2.json(); msg = ej2.detail || msg; } catch (_e) { try { msg = await r2.text(); } catch (_e2) {} }
      throw new Error(msg);
    }
    const data = await r2.json();
    stopStages();
    setLoading(false);
    renderNewsletter(data);
  } catch (err) {
    stopStages();
    setLoading(false);
    setError('Error: ' + err.message);
  }
}

// ── Scalekit Connect Service ───────────────────────────────────────────────────
async function connectService(connectionName) {
  const statusEl = document.getElementById(connectionName + '-status');
  if (statusEl) statusEl.textContent = 'Connecting…';
  try {
    const r = await fetch('/auth/connect?connection_name=' + encodeURIComponent(connectionName));
    const data = await r.json();
    if (data.link) {
      window.open(data.link, '_blank', 'width=600,height=700');
      if (statusEl) statusEl.textContent = '⟳ Complete in popup';
    } else if (data.message) {
      if (statusEl) statusEl.textContent = data.message.slice(0, 60);
    } else {
      if (statusEl) statusEl.textContent = 'No link returned';
    }
  } catch (err) {
    if (statusEl) statusEl.textContent = 'Error: ' + err.message.slice(0, 50);
  }
}

// ── Demo: inject signal ───────────────────────────────────────────────────────
async function injectDemoSignal(observation) {
  try {
    const r = await fetch('/demo/inject-signal', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({observation: observation || 'diffusion policy robotics research', source: 'demo-ui'})
    });
    const data = await r.json();
    return data;
  } catch (err) {
    console.error('demo inject failed:', err);
  }
}
</script>
</body>
</html>"""


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def frontend() -> str:
    return _FRONTEND_HTML


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    file_type: str = "auto",
) -> UploadResponse:
    """
    Upload a signal source file. Returns an upload_id for /newsletter/from-upload.

    Accepted files:
      BrowserHistory.json  — Google Takeout Chrome history
      watch-history.json   — Google Takeout YouTube watch history
      conversations.json   — ChatGPT data export

    file_type: "auto" (detect by filename), "history", "youtube", or "chatgpt"
    """
    settings = get_settings()
    upload_dir = settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    size = len(content)
    _MAX = 200 * 1024 * 1024  # 200 MB to accommodate ZIP archives
    if size > _MAX:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size / 1024 / 1024:.1f} MB). Maximum is 200 MB.",
        )

    fname = file.filename or "upload.json"

    # Extract BrowserHistory.json from a Google Takeout ZIP archive.
    # ZIP magic bytes: PK\x03\x04. We check bytes rather than extension
    # because users sometimes rename files, and takeout ZIPs are always PK.
    if content[:4] == b"PK\x03\x04" or fname.lower().endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                candidates = [n for n in zf.namelist() if n.lower().endswith("browserhistory.json")]
                if not candidates:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "ZIP does not contain BrowserHistory.json. "
                            "Make sure you exported Chrome data from Google Takeout "
                            "and look for Takeout/Chrome/BrowserHistory.json inside the ZIP."
                        ),
                    )
                # Prefer the canonical Takeout/Chrome/ path; fall back to first match
                canonical = next(
                    (n for n in candidates if "chrome" in n.lower()), candidates[0]
                )
                content = zf.read(canonical)
                size = len(content)
                _log.info("Extracted %s from ZIP (%d bytes)", canonical, size)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="File is not a valid ZIP or JSON")

    stripped = content.strip()
    if not stripped.startswith(b"{") and not stripped.startswith(b"["):
        raise HTTPException(status_code=400, detail="File must be valid JSON")

    # Basic JSON depth/structure guard — reject obviously malformed payloads
    try:
        _json.loads(content)
    except (ValueError, MemoryError):
        raise HTTPException(status_code=400, detail="File is not valid JSON")

    upload_id = str(uuid.uuid4())

    # Detect file type from filename when set to auto
    detected_type = file_type
    if file_type == "auto":
        fname_lower = fname.lower()
        if "browserhistory" in fname_lower or "browser_history" in fname_lower:
            detected_type = "history"
        elif "watch" in fname_lower or "youtube" in fname_lower:
            detected_type = "youtube"
        elif "conversation" in fname_lower:
            detected_type = "chatgpt"
        else:
            # Heuristic: YouTube watch-history is always a JSON array
            detected_type = "youtube" if stripped.startswith(b"[") else "history"

    # Save with type prefix so pipeline knows how to load it
    save_path = upload_dir / f"{detected_type}_{upload_id}.json"
    save_path.write_bytes(content)
    _log.info("Uploaded %s (%d bytes) → %s", fname, size, save_path)

    return UploadResponse(
        upload_id=upload_id,
        filename=fname,
        size_bytes=size,
        message=f"File uploaded successfully as {detected_type} source",
    )


class FromUploadRequest(BaseModel):
    # Single history file (legacy / convenience)
    history_upload_id: str | None = None
    # Multiple Chrome history files — merged before pipeline runs
    history_upload_ids: list[str] = []
    chatgpt_upload_id: str | None = None
    youtube_upload_id: str | None = None


def _merge_browser_history(paths: list[Path], upload_dir: Path) -> Path:
    """
    Merge multiple BrowserHistory.json files into a single file.
    Each file's 'Browser History' array is concatenated; duplicates are left
    for the collector's deduplication logic to handle (it already dedupes by URL).
    Returns path to the temporary merged file.
    """
    merged_entries: list[_json.Any] = []
    for p in paths:
        try:
            data = _json.loads(p.read_text(encoding="utf-8"))
            entries = data.get("Browser History", [])
            if isinstance(entries, list):
                merged_entries.extend(entries)
                _log.info("Merged %d entries from %s", len(entries), p.name)
        except Exception as exc:
            _log.warning("Could not read history file %s for merge: %s", p.name, exc)
    merge_id = str(uuid.uuid4())
    merged_path = upload_dir / f"history_{merge_id}.json"
    merged_path.write_text(
        _json.dumps({"Browser History": merged_entries}), encoding="utf-8"
    )
    _log.info("Merged %d total history entries → %s", len(merged_entries), merged_path.name)
    return merged_path


@app.post("/newsletter/from-upload", response_model=GenerateResponse)
async def generate_from_upload(
    body: FromUploadRequest,
    current_user: UserClaims | None = Depends(get_optional_user),
) -> GenerateResponse:
    """
    Run the full pipeline using previously uploaded file(s).
    This is the primary personalization endpoint — results are specific to
    the individual's actual browsing/conversation history.
    """
    # Collect all history IDs — merge history_upload_id (legacy) + history_upload_ids (multi)
    all_history_ids: list[str] = []
    if body.history_upload_id:
        all_history_ids.append(body.history_upload_id)
    all_history_ids.extend(body.history_upload_ids)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_history_ids = [i for i in all_history_ids if not (i in seen or seen.add(i))]  # type: ignore[func-returns-value]

    if not unique_history_ids and not body.chatgpt_upload_id and not body.youtube_upload_id:
        raise HTTPException(status_code=400, detail="Provide at least one upload ID")

    # Reject non-UUID upload IDs — prevents path traversal via "../" in the ID
    all_ids = unique_history_ids + [body.chatgpt_upload_id, body.youtube_upload_id]
    for uid in all_ids:
        if uid is not None and not _UUID_RE.match(uid):
            raise HTTPException(status_code=400, detail=f"Invalid upload ID format: {uid!r}")

    settings = get_settings()
    upload_dir = settings.upload_dir

    # Resolve history paths and merge if multiple were uploaded.
    # Try "history_" prefix first; fall back to any prefix in case the type
    # heuristic in /upload misclassified the file (e.g. a ZIP with a generic name).
    history_paths: list[Path] = []
    for uid in unique_history_ids:
        p: Path | None = None
        for prefix in ("history", "youtube", "chatgpt"):
            candidate = upload_dir / f"{prefix}_{uid}.json"
            if candidate.exists():
                p = candidate
                break
        if p is None:
            raise HTTPException(status_code=404, detail=f"Upload {uid} not found — the file may have expired or the server restarted")
        history_paths.append(p)

    history_path: Path | None = None
    merged_path: Path | None = None  # only set when we created a temp merge file
    if len(history_paths) == 1:
        history_path = history_paths[0]
    elif len(history_paths) > 1:
        merged_path = _merge_browser_history(history_paths, upload_dir)
        history_path = merged_path

    chatgpt_path: Path | None = None
    youtube_path: Path | None = None

    if body.chatgpt_upload_id:
        uid = body.chatgpt_upload_id
        found: Path | None = next(
            (upload_dir / f"{pfx}_{uid}.json" for pfx in ("chatgpt", "history", "youtube")
             if (upload_dir / f"{pfx}_{uid}.json").exists()), None
        )
        if found is None:
            raise HTTPException(status_code=404, detail=f"Upload {uid} not found")
        chatgpt_path = found

    if body.youtube_upload_id:
        uid = body.youtube_upload_id
        found = next(
            (upload_dir / f"{pfx}_{uid}.json" for pfx in ("youtube", "history", "chatgpt")
             if (upload_dir / f"{pfx}_{uid}.json").exists()), None
        )
        if found is None:
            raise HTTPException(status_code=404, detail=f"Upload {uid} not found")
        youtube_path = found

    pipeline = _build_pipeline_from_settings(
        override_history_path=history_path,
        override_chatgpt_path=chatgpt_path,
        override_youtube_path=youtube_path,
    )
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline could not be built — check ANTHROPIC_API_KEY and that the uploaded file is valid",
        )

    try:
        result: PipelineResult = await pipeline.run(
            user_id=current_user.user_id if current_user else "",
            user_email=current_user.email if current_user else "",
        )
    except PipelineError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        _log.exception("Unexpected pipeline error")
        raise HTTPException(status_code=503, detail=f"Pipeline error: {e}") from e
    finally:
        # Delete uploaded files immediately after pipeline use — they contain
        # sensitive personal data (browsing/conversation history) and must not
        # persist on the server longer than necessary.
        files_to_delete = list(history_paths) + [chatgpt_path, youtube_path]
        if merged_path:
            files_to_delete.append(merged_path)
        for p in files_to_delete:
            if p and p.exists():
                try:
                    p.unlink()
                    _log.info("Deleted uploaded file: %s", p.name)
                except OSError as exc:
                    _log.warning("Could not delete uploaded file %s: %s", p.name, exc)

    newsletter = result.newsletter
    # Store the curiosity graph (not raw signals) for daily regeneration
    profile_id = _store_profile(result.graph)
    sorted_topics = sorted(
        result.graph.topics, key=lambda t: t.composite_score(), reverse=True
    )
    raw_scores = [t.composite_score() for t in sorted_topics]
    max_score = max(raw_scores) if raw_scores else 1.0
    topic_names = [t.name for t in sorted_topics]
    topic_scores = [round(s / max_score, 4) for s in raw_scores]

    response = GenerateResponse(
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
        profile_id=profile_id,
        topic_names=topic_names,
        topic_scores=topic_scores,
    )
    _store_newsletter(response)
    return response


@app.post("/newsletter/generate", response_model=GenerateResponse)
async def generate_newsletter(
    pipeline: TracePipeline = Depends(get_pipeline),
    current_user: UserClaims | None = Depends(get_optional_user),
) -> GenerateResponse:
    """
    Run the full Trace pipeline using server-configured signal sources.
    Requires BROWSER_HISTORY_PATH to point to a valid BrowserHistory.json.
    """
    try:
        result: PipelineResult = await pipeline.run(
            user_id=current_user.user_id if current_user else "",
            user_email=current_user.email if current_user else "",
        )
    except PipelineError as e:
        raise HTTPException(status_code=503, detail=str(e))

    newsletter = result.newsletter
    sorted_topics_g = sorted(
        result.graph.topics, key=lambda t: t.composite_score(), reverse=True
    )
    raw_scores_g = [t.composite_score() for t in sorted_topics_g]
    max_score_g = max(raw_scores_g) if raw_scores_g else 1.0
    profile_id_g = _store_profile(result.graph)
    response = GenerateResponse(
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
        profile_id=profile_id_g,
        topic_names=[t.name for t in sorted_topics_g],
        topic_scores=[round(s / max_score_g, 4) for s in raw_scores_g],
    )
    _store_newsletter(response)
    return response


@app.get("/newsletter/{newsletter_id}", response_model=GenerateResponse)
async def get_newsletter(newsletter_id: str) -> GenerateResponse:
    """Retrieve a previously generated newsletter by ID."""
    if newsletter_id not in _NEWSLETTER_CACHE:
        raise HTTPException(status_code=404, detail="Newsletter not found")
    return GenerateResponse(**_NEWSLETTER_CACHE[newsletter_id])


@app.post("/newsletter/regenerate/{profile_id}", response_model=GenerateResponse)
async def regenerate_newsletter(
    profile_id: str,
    current_user: UserClaims | None = Depends(get_optional_user),
) -> GenerateResponse:
    """Generate a fresh newsletter from a stored curiosity profile.

    No file upload required. The user's curiosity interests (inferred topics)
    are stored from their initial upload. This endpoint runs only stages 3-5:
    scrape today's articles → assemble context → compose newsletter.

    Use this for daily newsletter generation after the first upload.
    profile_id is returned in the initial GenerateResponse.
    """
    if not _UUID_RE.match(profile_id):
        raise HTTPException(status_code=400, detail="Invalid profile ID format")

    graph = _load_profile(profile_id)
    if graph is None:
        raise HTTPException(
            status_code=404,
            detail="Profile not found. Please re-upload your history file.",
        )

    pipeline = _build_pipeline_from_settings()
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline could not be built — check ANTHROPIC_API_KEY",
        )

    try:
        result: PipelineResult = await pipeline.run_from_graph(
            graph=graph,
            user_id=current_user.user_id if current_user else "",
            user_email=current_user.email if current_user else "",
        )
    except PipelineError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        _log.exception("Unexpected pipeline error in regenerate")
        raise HTTPException(status_code=503, detail=f"Pipeline error: {e}") from e

    newsletter = result.newsletter
    sorted_topics_r = sorted(
        graph.topics, key=lambda t: t.composite_score(), reverse=True
    )
    raw_scores_r = [t.composite_score() for t in sorted_topics_r]
    max_score_r = max(raw_scores_r) if raw_scores_r else 1.0
    response = GenerateResponse(
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
        profile_id=profile_id,
        topic_names=[t.name for t in sorted_topics_r],
        topic_scores=[round(s / max_score_r, 4) for s in raw_scores_r],
    )
    _store_newsletter(response)
    return response


@app.get("/auth/login", response_model=LoginResponse)
async def auth_login(state: str | None = None) -> LoginResponse:
    _require_client()
    callback_url = get_settings().auth_callback_url
    authorization_url = build_login_url(redirect_uri=callback_url, state=state)
    return LoginResponse(authorization_url=authorization_url)


@app.get("/auth/callback", response_model=TokenResponse)
async def auth_callback(code: str, state: str | None = None) -> TokenResponse:
    _require_client()
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
    return UserResponse(
        user_id=current_user.user_id,
        email=current_user.email,
        name=current_user.name,
        organization_id=current_user.organization_id,
    )


@app.get("/auth/logout")
async def auth_logout(post_logout_redirect_uri: str | None = None) -> dict[str, str]:
    from scalekit.client import LogoutUrlOptions

    client = _require_client()
    options = LogoutUrlOptions()
    if post_logout_redirect_uri:
        options.post_logout_redirect_uri = post_logout_redirect_uri
    logout_url = client.get_logout_url(options)
    return {"logout_url": logout_url}


@app.get("/auth/connect")
async def auth_connect(
    connection_name: str = "apify-mcp",
    identifier: str | None = None,
    redirect_url: str | None = None,
) -> dict[str, Any]:
    """Return a Scalekit magic link for the user to connect a third-party account.

    After the user clicks the link and approves, Scalekit stores their OAuth
    token in the Token Vault and Trace can call that service on their behalf
    without ever seeing the raw credentials.

    Example: GET /auth/connect?connection_name=apify-mcp
    → {"link": "https://auth.yourdomain.scalekit.com/connect/...", "status": "ok"}
    """
    from trace.auth.scalekit import connect_ensure_account, connect_get_authorization_link

    settings = get_settings()
    user_id = identifier or settings.scalekit_default_identifier
    await connect_ensure_account(user_id, connection_name)
    link = await connect_get_authorization_link(
        identifier=user_id,
        connection_name=connection_name,
        redirect_url=redirect_url,
    )
    if not link:
        return {
            "status": "unconfigured",
            "message": "Scalekit is not configured — set SCALEKIT_ENV_URL, SCALEKIT_CLIENT_ID, SCALEKIT_CLIENT_SECRET",
        }
    return {"link": link, "status": "ok", "connection_name": connection_name}


@app.get("/agent/status")
async def agent_status() -> dict[str, Any]:
    """Return the autonomous agent scheduler status — jobs, intervals, demo mode."""
    try:
        from trace.agent.scheduler import scheduler_status
        return scheduler_status()
    except Exception as exc:
        return {"enabled": False, "error": str(exc)}


@app.post("/demo/inject-signal")
async def demo_inject_signal(request: Request) -> dict[str, Any]:
    """Inject a synthetic signal into the curiosity graph for live demos.

    Simulates what happens when the Gmail poller finds a new newsletter or the
    Chrome history watcher detects a new topic.  Judges can call this to trigger
    the autonomous loop without needing real credentials.

    Body (JSON): {"observation": str, "source": str, "profile_id": str}
    """
    import uuid as _uuid
    from datetime import datetime as _dt, timezone as _tz
    from trace.models import RawSignal, SignalSource
    from trace.mcp.server import _pending_signals

    body = await request.json()
    observation = str(body.get("observation", "AI safety research deep dive")).strip()[:500]
    source_str = str(body.get("source", "demo")).strip()
    profile_id = str(body.get("profile_id", "default")).strip()

    if not observation:
        raise HTTPException(status_code=422, detail="observation must not be empty")

    signal = RawSignal(
        id=str(_uuid.uuid4()),
        source=SignalSource.ENTIRE_IO,
        content=observation,
        timestamp=_dt.now(_tz.utc),
        metadata={"submitted_by": source_str, "injected_via": "demo_endpoint"},
    )
    bucket = _pending_signals.setdefault(profile_id, [])
    bucket.append(signal)
    _log.info("demo/inject-signal: profile=%s pending=%d", profile_id, len(bucket))
    return {
        "status": "ok",
        "signal_id": signal.id,
        "pending_count": len(bucket),
        "profile_id": profile_id,
        "message": "Signal injected. Run detect_and_act to process it into the curiosity graph.",
    }


# ── Approvals endpoints (Tier B: Gmail drafts, Reddit posts) ──────────────────

@app.get("/approvals")
async def list_approvals(profile_id: str = "default") -> dict[str, Any]:
    """List all pending Tier B actions awaiting user approval."""
    from trace.agent.approvals import get_approvals_queue
    queue = get_approvals_queue()
    pending = await queue.list_pending(profile_id)
    return {
        "pending": [a.to_dict() for a in pending],
        "count": len(pending),
        "profile_id": profile_id,
    }


@app.post("/approvals/{action_id}/approve")
async def approve_action(action_id: str) -> dict[str, Any]:
    """Approve a pending Tier B action (Gmail draft or Reddit post)."""
    from trace.agent.approvals import get_approvals_queue
    queue = get_approvals_queue()
    action = await queue.approve(action_id)
    if action is None:
        raise HTTPException(status_code=404, detail="Action not found or already resolved")
    return {"status": "approved", "action": action.to_dict()}


@app.post("/approvals/{action_id}/reject")
async def reject_action(action_id: str) -> dict[str, Any]:
    """Reject a pending Tier B action."""
    from trace.agent.approvals import get_approvals_queue
    queue = get_approvals_queue()
    action = await queue.reject(action_id)
    if action is None:
        raise HTTPException(status_code=404, detail="Action not found or already resolved")
    return {"status": "rejected", "action": action.to_dict()}


@app.get("/mcp-info")
async def mcp_info() -> dict[str, Any]:
    """Return everything needed to connect Claude Desktop to Trace MCP.

    Judges: paste the snippet into claude_desktop_config.json and Claude
    can call Trace curiosity tools as native MCP tools instantly.
    """
    settings = get_settings()
    base = settings.public_base_url
    return {
        "claude_desktop_config": {
            "mcpServers": {
                "trace": {
                    "url": f"{base}/mcp",
                    "transport": "http",
                }
            }
        },
        "tools": [
            "track_signal",
            "get_curiosity_topics",
            "get_unresolved_questions",
            "generate_briefing",
            "health",
        ],
        "auth": {
            "type": "oauth2",
            "provider": "Scalekit",
            "mcp_auth_enforced": bool(settings.scalekit_mcp_resource_id),
            "connect_link": f"{base}/auth/connect",
        },
        "sponsor_stack": {
            "apify_mcp": bool(settings.apify_api_token),
            "scalekit_connect": bool(
                settings.scalekit_env_url
                and settings.scalekit_client_id
                and settings.scalekit_client_secret
            ),
            "redis": bool(settings.redis_url),
            "anthropic": bool(settings.anthropic_api_key),
        },
    }
