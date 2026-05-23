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

import logging
import re
import uuid
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
    # topic_names: human-readable curiosity topics inferred from the user's signals.
    topic_names: list[str] = []


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
    yield
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


def _store_profile(graph: CuriosityGraph) -> str:
    """Cache the CuriosityGraph and return a new profile_id UUID.

    Only inferred topic names + scores are stored — never raw browsing signals.
    The profile lets users regenerate a fresh newsletter daily without re-uploading.
    """
    profile_id = str(uuid.uuid4())
    _PROFILE_CACHE[profile_id] = graph
    if len(_PROFILE_CACHE) > _PROFILE_CACHE_MAX:
        _PROFILE_CACHE.popitem(last=False)
    return profile_id


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
<style>
  :root {
    --bg: #0f0f11;
    --surface: #1a1a1f;
    --border: #2a2a35;
    --accent: #6366f1;
    --accent-hover: #818cf8;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --error: #f87171;
    --success: #34d399;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem 1rem;
  }
  .container { width: 100%; max-width: 760px; }
  header { text-align: center; margin-bottom: 3rem; }
  header h1 { font-size: 2.5rem; font-weight: 700; letter-spacing: -0.03em; }
  header h1 span { color: var(--accent); }
  header p { color: var(--muted); margin-top: 0.5rem; font-size: 1.05rem; }
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 1.5rem;
  }
  .card h2 { font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; }
  .upload-area {
    border: 2px dashed var(--border);
    border-radius: 8px;
    padding: 2.5rem;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
  }
  .upload-area:hover, .upload-area.drag-over {
    border-color: var(--accent);
    background: rgba(99,102,241,0.05);
  }
  .upload-area input { display: none; }
  .upload-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
  .upload-area p { color: var(--muted); font-size: 0.9rem; }
  .upload-area strong { color: var(--text); display: block; margin-bottom: 0.25rem; }
  .file-chosen { color: var(--success); font-weight: 600; margin-top: 0.5rem; }
  .tabs { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; }
  .tab {
    flex: 1; padding: 0.6rem; text-align: center; border-radius: 6px;
    cursor: pointer; font-size: 0.85rem; border: 1px solid var(--border);
    color: var(--muted); transition: all 0.2s;
  }
  .tab.active { background: var(--accent); color: white; border-color: var(--accent); }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  .instructions {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 1.25rem;
  }
  .instructions ol { padding-left: 1.25rem; line-height: 2; }
  .instructions code {
    background: rgba(255,255,255,0.08);
    padding: 0.1em 0.4em;
    border-radius: 3px;
    font-family: monospace;
    font-size: 0.8rem;
  }
  button.primary {
    width: 100%; padding: 0.875rem;
    background: var(--accent); color: white;
    border: none; border-radius: 8px;
    font-size: 1rem; font-weight: 600;
    cursor: pointer; transition: background 0.2s;
  }
  button.primary:hover:not(:disabled) { background: var(--accent-hover); }
  button.primary:disabled { opacity: 0.5; cursor: not-allowed; }
  .spinner {
    display: none; width: 20px; height: 20px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: white; border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 0 auto;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .status {
    text-align: center; padding: 1rem;
    font-size: 0.9rem; color: var(--muted);
    display: none;
  }
  .status.error { color: var(--error); }
  #result { display: none; }
  .newsletter-header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.25rem;
    margin-bottom: 1.5rem;
  }
  .newsletter-header .label {
    font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--accent); font-weight: 600;
    margin-bottom: 0.4rem;
  }
  .newsletter-header h2 { font-size: 1.5rem; font-weight: 700; line-height: 1.3; }
  .meta { font-size: 0.8rem; color: var(--muted); margin-top: 0.4rem; }
  .section-badge {
    display: inline-block;
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
    padding: 0.2em 0.6em; border-radius: 4px; font-weight: 600;
    margin-bottom: 0.5rem;
  }
  .badge-weekly_topics { background: rgba(99,102,241,0.2); color: #818cf8; }
  .badge-curiosity_debt { background: rgba(251,191,36,0.15); color: #fbbf24; }
  .badge-rabbit_hole { background: rgba(52,211,153,0.15); color: #34d399; }
  .section h3 { font-size: 1.15rem; font-weight: 700; margin-bottom: 0.6rem; line-height: 1.4; }
  .section p { color: #cbd5e1; line-height: 1.75; font-size: 0.95rem; }
  .sources { margin-top: 0.85rem; display: flex; flex-direction: column; gap: 0.35rem; }
  .sources a {
    font-size: 0.8rem; color: var(--accent);
    text-decoration: none;
  }
  .sources a:hover { color: var(--accent-hover); text-decoration: underline; }
  details.audit {
    margin-top: 0.75rem;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }
  details.audit summary {
    padding: 0.5rem 0.75rem;
    font-size: 0.78rem; color: var(--muted);
    cursor: pointer; user-select: none;
    list-style: none;
  }
  details.audit summary::-webkit-details-marker { display: none; }
  details.audit summary::before { content: "▶  "; font-size: 0.6rem; }
  details[open].audit summary::before { content: "▼  "; }
  details.audit .audit-body {
    padding: 0.75rem;
    font-size: 0.82rem; color: var(--muted);
    border-top: 1px solid var(--border);
    background: rgba(0,0,0,0.2);
    line-height: 1.6;
  }
  .errors-box {
    margin-top: 1rem;
    padding: 0.75rem 1rem;
    background: rgba(248,113,113,0.08);
    border: 1px solid rgba(248,113,113,0.25);
    border-radius: 8px;
    font-size: 0.82rem; color: #fca5a5;
  }
  .errors-box h4 { margin-bottom: 0.4rem; font-weight: 600; }
  .errors-box li { margin-left: 1rem; margin-top: 0.2rem; }
  .toc {
    margin-bottom: 1.5rem;
    padding: 1rem 1.25rem;
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 8px;
  }
  .toc-label {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--accent); font-weight: 600; margin-bottom: 0.6rem;
  }
  .toc ol { padding-left: 1.2rem; }
  .toc li { margin-top: 0.3rem; font-size: 0.88rem; line-height: 1.5; }
  .toc a { color: var(--text); text-decoration: none; }
  .toc a:hover { color: var(--accent); }
  .section { margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid var(--border); }
  .section:last-child { border-bottom: none; }
  .section-meta { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.5rem; }
  .src-badge {
    display: inline-block; font-size: 0.65rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em;
    padding: 0.15em 0.5em; border-radius: 3px;
    vertical-align: middle; flex-shrink: 0;
  }
  .src-arxiv { background: rgba(180,120,255,0.2); color: #c084fc; }
  .src-hn    { background: rgba(251,146,60,0.2);  color: #fb923c; }
  .src-web   { background: rgba(56,189,248,0.2);  color: #38bdf8; }
  .sources a { display: flex; align-items: center; gap: 0.45rem; }
  .sources a .link-text {
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    max-width: 520px;
  }
  .download-bar {
    display: flex; gap: 0.75rem; margin-bottom: 1.25rem; flex-wrap: wrap;
  }
  .download-bar button {
    padding: 0.45rem 1rem;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--muted);
    font-size: 0.82rem; cursor: pointer;
    transition: all 0.2s;
  }
  .download-bar button:hover {
    border-color: var(--accent); color: var(--accent);
  }
  .share-link {
    font-size: 0.78rem; color: var(--muted); margin-top: 0.5rem;
  }
  .share-link a { color: var(--accent); text-decoration: none; }
  .share-link a:hover { text-decoration: underline; }
  .curiosity-profile {
    margin-bottom: 1.5rem;
    padding: 1rem 1.25rem;
    background: rgba(52,211,153,0.05);
    border: 1px solid rgba(52,211,153,0.2);
    border-radius: 8px;
  }
  .curiosity-profile .profile-label {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: #34d399; font-weight: 600; margin-bottom: 0.6rem;
  }
  .topic-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.75rem; }
  .topic-chip {
    display: inline-block; font-size: 0.75rem;
    padding: 0.25em 0.7em; border-radius: 12px;
    background: rgba(99,102,241,0.15); color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.3);
  }
  .regen-bar { display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; margin-top: 0.5rem; }
  .regen-bar .regen-link {
    font-size: 0.78rem; color: var(--muted);
    font-family: monospace;
  }
  .btn-regen {
    padding: 0.4rem 1rem;
    background: rgba(52,211,153,0.15);
    border: 1px solid rgba(52,211,153,0.4);
    border-radius: 6px;
    color: #34d399;
    font-size: 0.82rem; cursor: pointer;
    transition: all 0.2s;
  }
  .btn-regen:hover:not(:disabled) {
    background: rgba(52,211,153,0.25);
    border-color: #34d399;
  }
  .btn-regen:disabled { opacity: 0.5; cursor: not-allowed; }
  .privacy-note {
    font-size: 0.78rem; color: var(--muted);
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 1.5rem;
    line-height: 1.6;
  }
  .privacy-note strong { color: var(--text); }
  footer { margin-top: 3rem; text-align: center; font-size: 0.78rem; color: var(--muted); }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Tr<span>a</span>ce</h1>
    <p>Upload your browsing history. Get a newsletter that reflects <em>your</em> curiosity.</p>
  </header>

  <div class="privacy-note">
    <strong>Your data stays yours.</strong>
    You export your own history file locally from Google or ChatGPT — you never hand over
    any password, OAuth token, or account access. The file is sent only to this server,
    used once to generate your newsletter, then <strong>deleted immediately</strong>.
    Nothing is stored, logged, or shared beyond the newsletter itself.
  </div>

  <div class="card">
    <div class="tabs">
      <div class="tab active" onclick="switchTab('google')">Chrome History</div>
      <div class="tab" onclick="switchTab('youtube')">YouTube History</div>
      <div class="tab" onclick="switchTab('chatgpt')">ChatGPT Export</div>
    </div>
    <p style="font-size:0.78rem;color:var(--muted);margin-bottom:1rem">
      Upload one or more — the more signals, the more accurate your curiosity profile.
    </p>

    <div id="tab-google" class="tab-content active">
      <div class="instructions">
        <ol>
          <li>Go to <strong>takeout.google.com</strong></li>
          <li>Deselect all → select only <code>Chrome</code></li>
          <li>Export → Download → extract the ZIP</li>
          <li>Find <code>BrowserHistory.json</code> and upload it below</li>
        </ol>
      </div>
      <label class="upload-area" id="drop-google">
        <input type="file" id="file-google" accept=".json" onchange="fileChosen('google')">
        <div class="upload-icon">📂</div>
        <strong>Drop BrowserHistory.json here</strong>
        <p>or click to browse</p>
        <div class="file-chosen" id="chosen-google"></div>
      </label>
    </div>

    <div id="tab-youtube" class="tab-content">
      <div class="instructions">
        <ol>
          <li>Go to <strong>takeout.google.com</strong></li>
          <li>Deselect all → select <code>YouTube and YouTube Music</code></li>
          <li>Export → Download → extract the ZIP</li>
          <li>Find <code>Takeout/YouTube and YouTube Music/history/watch-history.json</code></li>
          <li>Upload it below</li>
        </ol>
        <p style="margin-top:0.5rem">
          Rewatched videos signal deeper interest — if you opened the same lecture 3+ times,
          Trace flags it as a curiosity you're stuck on and prioritises related content.
        </p>
      </div>
      <label class="upload-area" id="drop-youtube">
        <input type="file" id="file-youtube" accept=".json" onchange="fileChosen('youtube')">
        <div class="upload-icon">▶️</div>
        <strong>Drop watch-history.json here</strong>
        <p>or click to browse</p>
        <div class="file-chosen" id="chosen-youtube"></div>
      </label>
    </div>

    <div id="tab-chatgpt" class="tab-content">
      <div class="instructions">
        <ol>
          <li>Go to ChatGPT → <strong>Settings → Data Controls → Export Data</strong></li>
          <li>Wait for the email → Download → extract the ZIP</li>
          <li>Find <code>conversations.json</code> and upload it below</li>
        </ol>
      </div>
      <label class="upload-area" id="drop-chatgpt">
        <input type="file" id="file-chatgpt" accept=".json" onchange="fileChosen('chatgpt')">
        <div class="upload-icon">💬</div>
        <strong>Drop conversations.json here</strong>
        <p>or click to browse</p>
        <div class="file-chosen" id="chosen-chatgpt"></div>
      </label>
    </div>

    <div style="margin-top:1.5rem">
      <button class="primary" id="gen-btn" onclick="generate()">Generate My Newsletter</button>
      <div class="spinner" id="spinner"></div>
      <div class="status" id="status"></div>
    </div>
  </div>

  <div id="result" class="card">
    <div class="newsletter-header">
      <div class="label">Your Personalized Newsletter</div>
      <h2 id="subject"></h2>
      <div class="meta" id="meta"></div>
    </div>
    <div id="curiosity-profile"></div>
    <div class="download-bar">
      <button onclick="downloadHtml()">⬇ Download HTML</button>
      <button onclick="downloadText()">⬇ Download Plain Text</button>
    </div>
    <div id="share-link-container"></div>
    <div id="toc"></div>
    <div id="sections"></div>
    <div id="errors-container"></div>
  </div>
</div>

<footer>Trace · Hackathon Build · Powered by Claude &amp; Anthropic</footer>

<script>
let activeTab = 'google';
let newsletterData = null;

function switchTab(tab) {
  activeTab = tab;
  const tabs = ['google','youtube','chatgpt'];
  document.querySelectorAll('.tab').forEach((t, i) => {
    t.classList.toggle('active', tabs[i] === tab);
  });
  tabs.forEach(id => {
    document.getElementById('tab-' + id).classList.toggle('active', id === tab);
  });
}

function fileChosen(type) {
  const f = document.getElementById('file-' + type).files[0];
  document.getElementById('chosen-' + type).textContent = f ? '✓ ' + f.name : '';
}

// Drag-and-drop
['google','youtube','chatgpt'].forEach(t => {
  const el = document.getElementById('drop-' + t);
  el.addEventListener('dragover', e => { e.preventDefault(); el.classList.add('drag-over'); });
  el.addEventListener('dragleave', () => el.classList.remove('drag-over'));
  el.addEventListener('drop', e => {
    e.preventDefault(); el.classList.remove('drag-over');
    const dt = e.dataTransfer;
    if (dt.files.length) {
      document.getElementById('file-' + t).files = dt.files;
      fileChosen(t);
    }
  });
});

function setStatus(msg, isError) {
  const s = document.getElementById('status');
  s.textContent = msg;
  s.className = 'status' + (isError ? ' error' : '');
  s.style.display = msg ? 'block' : 'none';
}

function setLoading(loading) {
  document.getElementById('gen-btn').style.display = loading ? 'none' : 'block';
  document.getElementById('spinner').style.display = loading ? 'block' : 'none';
}

const STAGE_MSGS = [
  'Reading your history…',
  'Extracting curiosity topics with Claude…',
  'Fetching fresh articles from arXiv, HN & more…',
  'Assembling your context window…',
  'Writing your newsletter with Claude…',
];
let stageIdx = 0;
let stageTimer;

function tickStage() {
  if (stageIdx < STAGE_MSGS.length) {
    setStatus(STAGE_MSGS[stageIdx++]);
    stageTimer = setTimeout(tickStage, 6000);
  }
}

function stopStages() { clearTimeout(stageTimer); stageIdx = 0; }

function badgeClass(type) {
  return 'section-badge badge-' + (type || 'weekly_topics');
}

function badgeLabel(type) {
  return { weekly_topics: 'This Week', curiosity_debt: 'Curiosity Debt', rabbit_hole: 'Rabbit Hole' }[type] || type;
}

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

async function regenerate() {
  if (!newsletterData || !newsletterData.profile_id) return;
  const btn = document.getElementById('regen-btn');
  if (btn) btn.disabled = true;
  document.getElementById('result').style.display = 'none';
  setLoading(true);
  stageIdx = 0;
  tickStage();
  try {
    const r = await fetch('/newsletter/regenerate/' + encodeURIComponent(newsletterData.profile_id), {
      method: 'POST',
    });
    if (!r.ok) {
      let msg = 'Regeneration failed';
      try { const e = await r.json(); msg = e.detail || msg; } catch { msg = await r.text().catch(() => msg); }
      throw new Error(msg);
    }
    const data = await r.json();
    stopStages();
    setLoading(false);
    setStatus('');
    renderNewsletter(data);
  } catch (err) {
    stopStages();
    setLoading(false);
    setStatus('Error: ' + err.message, true);
    if (btn) btn.disabled = false;
    document.getElementById('result').style.display = 'block';
  }
}

function renderNewsletter(data) {
  newsletterData = data;
  document.getElementById('subject').textContent = data.subject_line;
  const dt = new Date(data.generated_at);
  const forStr = data.generated_for ? ' · for ' + data.generated_for : '';
  document.getElementById('meta').textContent = dt.toLocaleString() + forStr;

  // Curiosity profile: show inferred topics + daily regen link
  const profileEl = document.getElementById('curiosity-profile');
  if (data.topic_names && data.topic_names.length > 0) {
    const chips = data.topic_names.map(t => `<span class="topic-chip">${esc(t)}</span>`).join('');
    const regenHtml = data.profile_id ? `
      <div class="regen-bar">
        <button class="btn-regen" id="regen-btn" onclick="regenerate()">↺ Regenerate with today's articles</button>
        <span class="regen-link">Bookmark: <a href="/newsletter/regenerate/${esc(data.profile_id)}" onclick="return false;">/regenerate/${esc(data.profile_id.substring(0,8))}…</a></span>
      </div>` : '';
    profileEl.innerHTML = `<div class="curiosity-profile">
      <div class="profile-label">Your Curiosity Profile · ${data.topic_names.length} topic${data.topic_names.length !== 1 ? 's' : ''} inferred</div>
      <div class="topic-chips">${chips}</div>
      ${regenHtml}
    </div>`;
  } else {
    profileEl.innerHTML = '';
  }

  // Share link
  const shareCont = document.getElementById('share-link-container');
  if (data.id) {
    shareCont.innerHTML = `<div class="share-link">Permalink: <a href="/newsletter/${esc(data.id)}" target="_blank">/newsletter/${esc(data.id)}</a></div>`;
  } else {
    shareCont.innerHTML = '';
  }

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

  const secEl = document.getElementById('sections');
  secEl.innerHTML = '';
  data.sections.forEach((s, i) => {
    const div = document.createElement('div');
    div.className = 'section';
    div.id = `section-${i}`;
    const urls = s.source_urls.map(u => {
      const href = esc(safeHref(u));
      const badge = sourceBadge(u);
      const display = esc(u);
      return `<a href="${href}" target="_blank" rel="noopener noreferrer">${badge}<span class="link-text">${display}</span></a>`;
    }).join('');
    div.innerHTML = `
      <div class="section-meta">
        <span class="${badgeClass(s.section_type)}">${esc(badgeLabel(s.section_type))}</span>
      </div>
      <h3>${esc(s.title)}</h3>
      <p>${esc(s.content)}</p>
      ${urls ? '<div class="sources">' + urls + '</div>' : ''}
      <details class="audit">
        <summary>Why this section?</summary>
        <div class="audit-body">${esc(s.audit_reasoning)}</div>
      </details>`;
    secEl.appendChild(div);
  });

  const errBox = document.getElementById('errors-container');
  errBox.innerHTML = '';
  if (data.errors && data.errors.length) {
    errBox.innerHTML = `<div class="errors-box"><h4>Non-fatal pipeline warnings (${data.errors.length})</h4><ul>${
      data.errors.map(e => '<li>' + esc(e) + '</li>').join('')
    }</ul></div>`;
  }

  document.getElementById('result').style.display = 'block';
  document.getElementById('result').scrollIntoView({ behavior: 'smooth' });
}

async function generate() {
  const fileGoogle  = document.getElementById('file-google').files[0];
  const fileYoutube = document.getElementById('file-youtube').files[0];
  const fileChatgpt = document.getElementById('file-chatgpt').files[0];

  if (!fileGoogle && !fileYoutube && !fileChatgpt) {
    setStatus('Please upload at least one file first.', true);
    return;
  }

  document.getElementById('result').style.display = 'none';
  setLoading(true);
  stageIdx = 0;
  tickStage();

  try {
    let historyUploadId = null, youtubeUploadId = null, chatgptUploadId = null;

    if (fileGoogle) {
      const fd = new FormData();
      fd.append('file', fileGoogle);
      const r = await fetch('/upload', { method: 'POST', body: fd });
      if (!r.ok) { const e = await r.json(); throw new Error(e.detail || 'Upload failed'); }
      historyUploadId = (await r.json()).upload_id;
    }

    if (fileYoutube) {
      const fd = new FormData();
      fd.append('file', fileYoutube);
      fd.append('file_type', 'youtube');
      const r = await fetch('/upload', { method: 'POST', body: fd });
      if (!r.ok) { const e = await r.json(); throw new Error(e.detail || 'Upload failed'); }
      youtubeUploadId = (await r.json()).upload_id;
    }

    if (fileChatgpt) {
      const fd = new FormData();
      fd.append('file', fileChatgpt);
      fd.append('file_type', 'chatgpt');
      const r = await fetch('/upload', { method: 'POST', body: fd });
      if (!r.ok) { const e = await r.json(); throw new Error(e.detail || 'Upload failed'); }
      chatgptUploadId = (await r.json()).upload_id;
    }

    const body = {};
    if (historyUploadId) body.history_upload_id = historyUploadId;
    if (youtubeUploadId) body.youtube_upload_id = youtubeUploadId;
    if (chatgptUploadId) body.chatgpt_upload_id = chatgptUploadId;

    const r2 = await fetch('/newsletter/from-upload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r2.ok) {
      let msg = 'Generation failed';
      try { const e = await r2.json(); msg = e.detail || msg; } catch { msg = await r2.text().catch(() => msg); }
      throw new Error(msg);
    }
    const data = await r2.json();

    stopStages();
    setLoading(false);
    setStatus('');
    renderNewsletter(data);
  } catch (err) {
    stopStages();
    setLoading(false);
    setStatus('Error: ' + err.message, true);
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
    _MAX = 100 * 1024 * 1024
    if size > _MAX:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size / 1024 / 1024:.1f} MB). Maximum is 100 MB.",
        )

    stripped = content.strip()
    if not stripped.startswith(b"{") and not stripped.startswith(b"["):
        raise HTTPException(status_code=400, detail="File must be valid JSON")

    # Basic JSON depth/structure guard — reject obviously malformed payloads
    try:
        import json as _json
        _json.loads(content)
    except (ValueError, MemoryError):
        raise HTTPException(status_code=400, detail="File is not valid JSON")

    upload_id = str(uuid.uuid4())
    fname = file.filename or "upload.json"

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
    history_upload_id: str | None = None
    chatgpt_upload_id: str | None = None
    youtube_upload_id: str | None = None


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
    # Validate inputs before touching settings (settings may be unavailable in some envs)
    if not body.history_upload_id and not body.chatgpt_upload_id and not body.youtube_upload_id:
        raise HTTPException(status_code=400, detail="Provide at least one upload ID")

    # Reject non-UUID upload IDs — prevents path traversal via "../" in the ID
    for uid in [body.history_upload_id, body.chatgpt_upload_id, body.youtube_upload_id]:
        if uid is not None and not _UUID_RE.match(uid):
            raise HTTPException(status_code=400, detail=f"Invalid upload ID format: {uid!r}")

    settings = get_settings()
    upload_dir = settings.upload_dir

    history_path: Path | None = None
    chatgpt_path: Path | None = None
    youtube_path: Path | None = None

    if body.history_upload_id:
        p = upload_dir / f"history_{body.history_upload_id}.json"
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Upload {body.history_upload_id} not found")
        history_path = p

    if body.chatgpt_upload_id:
        p = upload_dir / f"chatgpt_{body.chatgpt_upload_id}.json"
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Upload {body.chatgpt_upload_id} not found")
        chatgpt_path = p

    if body.youtube_upload_id:
        p = upload_dir / f"youtube_{body.youtube_upload_id}.json"
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Upload {body.youtube_upload_id} not found")
        youtube_path = p

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
        for p in [history_path, chatgpt_path, youtube_path]:
            if p and p.exists():
                try:
                    p.unlink()
                    _log.info("Deleted uploaded file: %s", p.name)
                except OSError as exc:
                    _log.warning("Could not delete uploaded file %s: %s", p.name, exc)

    newsletter = result.newsletter
    # Store the curiosity graph (not raw signals) for daily regeneration
    profile_id = _store_profile(result.graph)
    topic_names = [t.name for t in result.graph.topics]

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

    graph = _PROFILE_CACHE.get(profile_id)
    if graph is None:
        raise HTTPException(
            status_code=404,
            detail="Profile not found or expired. Please re-upload your history file.",
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
    topic_names = [t.name for t in graph.topics]
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
