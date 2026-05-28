# -*- coding: utf-8 -*-
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
<title>Trace — Curiosity OS</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#04040e;--s1:#0a0a1a;--s2:#0f0f1e;--s3:#161626;
  --border:rgba(148,163,255,0.07);--border-md:rgba(148,163,255,0.13);--border-hi:rgba(148,163,255,0.22);
  --accent:#7c3aed;--accent-b:#4f46e5;--accent-l:#a78bfa;--accent-glow:rgba(124,58,237,0.18);
  --cyan:#22d3ee;--green:#10b981;--yellow:#f59e0b;--red:#ef4444;--pink:#f472b6;
  --t1:#f0f0fa;--t2:#9090b8;--t3:#55556a;
  --font:'Plus Jakarta Sans',-apple-system,BlinkMacSystemFont,sans-serif;
  --mono:'JetBrains Mono','Fira Code',monospace;
  --r:14px;--r-sm:8px;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{
  background:var(--bg);color:var(--t1);font-family:var(--font);
  font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased;
  min-height:100vh;overflow-x:hidden;
}
a{color:var(--accent-l);text-decoration:none}
a:hover{color:var(--t1)}
button{font-family:var(--font);cursor:pointer}
code,pre{font-family:var(--mono)}

/* Orbs */
.orb-field{position:fixed;inset:0;pointer-events:none;overflow:hidden;z-index:0}
.orb{position:absolute;border-radius:50%;filter:blur(110px);opacity:0.22;animation:drift 22s ease-in-out infinite}
.orb-1{width:700px;height:700px;background:#4f46e5;top:-280px;left:-120px;animation-delay:0s}
.orb-2{width:550px;height:550px;background:#7c3aed;bottom:-180px;right:-80px;animation-delay:-8s}
.orb-3{width:420px;height:420px;background:#06b6d4;top:38%;left:42%;transform:translate(-50%,-50%);animation-delay:-16s;opacity:0.1}
@keyframes drift{0%,100%{transform:translateY(0) scale(1)}50%{transform:translateY(-50px) scale(1.06)}}
.orb-3{animation:drift3 22s ease-in-out infinite}
@keyframes drift3{0%,100%{transform:translate(-50%,-50%) scale(1)}50%{transform:translate(-50%,-50%) translateY(-50px) scale(1.06)}}

/* Nav */
nav{
  position:sticky;top:0;z-index:100;
  background:rgba(4,4,14,0.75);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
  border-bottom:1px solid var(--border);
}
.nav-inner{
  max-width:920px;margin:0 auto;padding:0 1.5rem;
  display:flex;align-items:center;height:54px;gap:0.75rem;
}
.nav-logo{
  font-size:1.15rem;font-weight:800;letter-spacing:-0.04em;
  background:linear-gradient(135deg,var(--accent-l),var(--cyan));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.nav-tag{
  font-size:0.6rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
  color:var(--t3);border:1px solid var(--border-md);padding:0.18em 0.5em;border-radius:4px;
}
.nav-status{
  display:flex;align-items:center;gap:0.4rem;margin-left:auto;
  font-size:0.7rem;color:var(--t2);
}
.sdot{
  width:7px;height:7px;border-radius:50%;background:var(--t3);flex-shrink:0;
  transition:background 0.3s;
}
.sdot.live{background:var(--green);animation:pdot 2s ease-in-out infinite}
@keyframes pdot{0%{box-shadow:0 0 0 0 rgba(16,185,129,0.5)}70%{box-shadow:0 0 0 8px rgba(16,185,129,0)}100%{box-shadow:0 0 0 0 rgba(16,185,129,0)}}
#loop-next-run{color:var(--t3);font-family:var(--mono);font-size:0.67rem}

/* Page */
.page{position:relative;z-index:1;max-width:920px;margin:0 auto;padding:0 1.5rem 6rem}

/* Hero */
.hero{text-align:center;padding:4.5rem 0 3.5rem}
.hero-pill{
  display:inline-flex;align-items:center;gap:0.45rem;
  font-size:0.67rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
  color:var(--accent-l);background:rgba(124,58,237,0.1);border:1px solid rgba(124,58,237,0.22);
  padding:0.32em 0.9em;border-radius:999px;margin-bottom:1.5rem;
}
.hero-pill::before{content:'';width:5px;height:5px;border-radius:50%;background:var(--accent-l);display:inline-block}
.hero-title{
  font-size:clamp(3rem,7.5vw,5.2rem);font-weight:800;
  letter-spacing:-0.045em;line-height:1.08;margin-bottom:1.25rem;
}
.grad{
  background:linear-gradient(135deg,var(--accent-l) 0%,var(--cyan) 55%,#38bdf8 100%);
  background-size:200% auto;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  animation:shim 5s linear infinite;
}
@keyframes shim{to{background-position:200% center}}
.hero-sub{font-size:1.05rem;color:var(--t2);max-width:540px;margin:0 auto;line-height:1.75}

/* Section divider */
.sec-hdr{display:flex;align-items:center;gap:0.75rem;margin:2rem 0 0.85rem}
.sec-hdr h2{font-size:0.62rem;font-weight:700;letter-spacing:0.13em;text-transform:uppercase;color:var(--t3);white-space:nowrap}
.sec-hdr::after{content:'';flex:1;height:1px;background:var(--border)}

/* Cards */
.card{
  background:var(--s1);border:1px solid var(--border);border-radius:var(--r);
  padding:1.5rem;margin-bottom:0.85rem;transition:border-color 0.2s;
}
.card:hover{border-color:var(--border-md)}
.card.accent-border{border-color:rgba(124,58,237,0.28);box-shadow:inset 0 0 0 1px rgba(124,58,237,0.08),0 8px 40px rgba(124,58,237,0.07)}
.clabel{
  font-size:0.62rem;font-weight:700;letter-spacing:0.13em;text-transform:uppercase;
  color:var(--t3);margin-bottom:1rem;display:flex;align-items:center;gap:0.45rem;
}
.clabel-dot{width:5px;height:5px;border-radius:50%;background:currentColor;opacity:0.6}

/* Privacy */
.privacy{
  font-size:0.75rem;color:var(--t2);
  background:rgba(255,255,255,0.015);border:1px solid var(--border);
  border-radius:var(--r-sm);padding:0.75rem 1rem;line-height:1.65;margin-bottom:0.85rem;
}
.privacy strong{color:var(--t1)}

/* Source cards */
.src-list{display:flex;flex-direction:column;gap:0.45rem}
.sc{background:var(--s2);border:1px solid var(--border);border-radius:var(--r-sm);overflow:hidden;transition:border-color 0.2s}
.sc.active{border-color:rgba(124,58,237,0.38)}
.sc-head{display:flex;align-items:center;gap:0.7rem;padding:0.85rem 1rem;cursor:pointer;user-select:none}
.sc-icon{font-size:1.1rem;flex-shrink:0;line-height:1}
.sc-info{flex:1}
.sc-name{font-size:0.86rem;font-weight:600;color:var(--t1)}
.sc-desc{font-size:0.7rem;color:var(--t3);margin-top:0.1rem}
.sc-badge{font-size:0.62rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;padding:0.18em 0.5em;border-radius:4px;color:var(--t3);border:1px solid var(--border-md);flex-shrink:0}
.sc-badge.ok{color:var(--green);border-color:rgba(16,185,129,0.3);background:rgba(16,185,129,0.07)}
.sc-chev{color:var(--t3);font-size:0.58rem;transition:transform 0.2s;flex-shrink:0}
.sc.open .sc-chev{transform:rotate(180deg)}
.sc-body{display:none;border-top:1px solid var(--border);padding:1rem}
.sc.open .sc-body{display:block}

/* Howto */
.howto{
  background:rgba(124,58,237,0.05);border:1px solid rgba(124,58,237,0.14);
  border-radius:var(--r-sm);padding:0.8rem 1rem;font-size:0.78rem;color:var(--t2);
  margin-bottom:0.85rem;line-height:1.7;
}
.howto ol{padding-left:1.15rem}
.howto li{margin-top:0.28rem}
.howto strong{color:var(--t1)}
.howto code{background:rgba(255,255,255,0.07);padding:0.1em 0.38em;border-radius:3px;font-size:0.76rem;color:var(--t1)}

/* Drop zone */
.drop{
  border:2px dashed var(--border-md);border-radius:var(--r-sm);
  padding:1.35rem;text-align:center;cursor:pointer;transition:all 0.2s;display:block;
}
.drop:hover,.drop.over{border-color:var(--accent);background:rgba(124,58,237,0.05)}
.drop input{display:none}
.drop-label{font-size:0.84rem;font-weight:500;color:var(--t1)}
.drop-sub{font-size:0.7rem;color:var(--t3);margin-top:0.18rem}
.drop-ok{font-size:0.78rem;color:var(--green);font-weight:500;margin-top:0.45rem}

/* Signal bar */
.sig-row{display:flex;align-items:center;gap:0.7rem;margin-top:1.2rem}
.sig-lbl{font-size:0.7rem;color:var(--t3);white-space:nowrap}
.sig-track{flex:1;height:3px;background:var(--s3);border-radius:2px;overflow:hidden}
.sig-fill{height:100%;width:0%;background:linear-gradient(90deg,var(--accent),var(--cyan));transition:width 0.4s ease}
.sig-count{font-size:0.7rem;color:var(--t3);white-space:nowrap}

/* Buttons */
.btn{
  display:inline-flex;align-items:center;justify-content:center;gap:0.35rem;
  font-family:var(--font);font-size:0.85rem;font-weight:600;
  border:none;border-radius:var(--r-sm);cursor:pointer;transition:all 0.15s;
  padding:0.58rem 1.15rem;letter-spacing:-0.01em;
}
.btn-primary{
  background:var(--accent);color:#fff;
  box-shadow:0 0 0 1px rgba(124,58,237,0.4),0 4px 20px rgba(124,58,237,0.22);
}
.btn-primary:hover:not(:disabled){background:#6d28d9;box-shadow:0 0 0 1px rgba(109,40,217,0.5),0 6px 24px rgba(124,58,237,0.32)}
.btn-primary:active:not(:disabled){transform:scale(0.98)}
.btn-primary:disabled{opacity:0.38;cursor:not-allowed}
.btn-full{width:100%;padding:0.85rem;font-size:0.95rem}
.btn-ghost{background:transparent;border:1px solid var(--border-md);color:var(--t2)}
.btn-ghost:hover{border-color:var(--border-hi);color:var(--t1);background:rgba(255,255,255,0.03)}
.btn-sm{font-size:0.76rem;padding:0.35rem 0.8rem;border-radius:6px}
.btn-v{background:rgba(124,58,237,0.13);border:1px solid rgba(124,58,237,0.3);color:var(--accent-l)}
.btn-v:hover{background:rgba(124,58,237,0.22)}
.btn-c{background:rgba(34,211,238,0.1);border:1px solid rgba(34,211,238,0.22);color:var(--cyan)}
.btn-c:hover{background:rgba(34,211,238,0.18)}
.btn-g{background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.28);color:var(--green)}
.btn-g:hover{background:rgba(16,185,129,0.22)}
.btn-r{background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.28);color:var(--red)}
.btn-r:hover{background:rgba(239,68,68,0.22)}

/* Loading */
.loading-wrap{
  display:none;flex-direction:column;align-items:center;
  padding:2.5rem 1.5rem;text-align:center;
}
.spinner{
  width:38px;height:38px;border-radius:50%;
  border:3px solid var(--border-md);border-top-color:var(--accent);
  animation:spin 0.75s linear infinite;margin-bottom:1.1rem;
}
@keyframes spin{to{transform:rotate(360deg)}}
.stage-msg{font-size:0.88rem;color:var(--t2);min-height:1.5em}
.stage-msg.err{color:var(--red)}
.timing-note{font-size:0.7rem;color:var(--t3);margin-top:0.38rem}

/* Result */
@keyframes fadeUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
#result{display:none;animation:fadeUp 0.4s ease}
.res-eyebrow{font-size:0.62rem;font-weight:700;letter-spacing:0.13em;text-transform:uppercase;color:var(--accent-l);margin-bottom:0.45rem}
.res-subject{font-size:1.6rem;font-weight:800;letter-spacing:-0.025em;line-height:1.22;margin-bottom:0.38rem}
.res-meta{font-size:0.72rem;color:var(--t3)}

/* Chips */
.chips{display:flex;flex-wrap:wrap;gap:0.38rem;margin-bottom:0.8rem}
.chip{
  font-size:0.7rem;font-weight:500;padding:0.24em 0.68em;border-radius:20px;
  background:rgba(124,58,237,0.12);color:var(--accent-l);
  border:1px solid rgba(124,58,237,0.24);cursor:default;transition:background 0.15s;
}
.chip:hover{background:rgba(124,58,237,0.22)}

/* Action row */
.act-row{display:flex;gap:0.45rem;flex-wrap:wrap;align-items:center;margin-bottom:0.9rem}

/* TOC */
.toc-box{background:rgba(124,58,237,0.04);border:1px solid rgba(124,58,237,0.13);border-radius:var(--r-sm);padding:0.85rem 1rem;margin-bottom:1.1rem}
.toc-lbl{font-size:0.6rem;font-weight:700;letter-spacing:0.13em;text-transform:uppercase;color:var(--accent-l);margin-bottom:0.45rem}
.toc-box ol{padding-left:1.15rem}
.toc-box li{margin-top:0.24rem;font-size:0.82rem;line-height:1.5}
.toc-box a{color:var(--t1)}.toc-box a:hover{color:var(--accent-l)}

/* Newsletter sections */
.nls{margin-bottom:1.65rem;padding-bottom:1.4rem;border-bottom:1px solid var(--border)}
.nls:last-child{border-bottom:none}
.nl-badge{display:inline-block;font-size:0.6rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;padding:0.2em 0.58em;border-radius:4px;margin-bottom:0.52rem}
.badge-weekly_topics{background:rgba(79,70,229,0.14);color:#818cf8}
.badge-curiosity_debt{background:rgba(245,158,11,0.12);color:var(--yellow)}
.badge-rabbit_hole{background:rgba(16,185,129,0.12);color:var(--green)}
.nls h3{font-size:1.08rem;font-weight:700;margin-bottom:0.55rem;letter-spacing:-0.01em;line-height:1.32}
.nl-body p{color:#c0cce0;line-height:1.8;font-size:0.91rem;margin-bottom:0.7rem}
.nl-body p:last-child{margin-bottom:0}

/* Sources */
.srcs{margin-top:0.8rem;display:flex;flex-direction:column;gap:0.28rem}
.srcs a{display:flex;align-items:center;gap:0.38rem;font-size:0.77rem;color:var(--accent-l)}
.srcs a:hover{color:var(--t1)}
.spill{display:inline-block;font-size:0.58rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;padding:0.14em 0.42em;border-radius:3px;flex-shrink:0}
.sp-arxiv{background:rgba(180,120,255,0.17);color:#c084fc}
.sp-hn{background:rgba(251,146,60,0.17);color:#fb923c}
.sp-web{background:rgba(56,189,248,0.17);color:#38bdf8}
.slink{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:460px}

/* Audit */
details.why{margin-top:0.7rem;border:1px solid var(--border);border-radius:6px;overflow:hidden}
details.why summary{padding:0.42rem 0.75rem;font-size:0.7rem;color:var(--t3);cursor:pointer;user-select:none;list-style:none}
details.why summary::-webkit-details-marker{display:none}
details.why summary::before{content:"▶  ";font-size:0.56rem}
details[open].why summary::before{content:"▼  "}
.why-body{padding:0.7rem;font-size:0.77rem;color:var(--t2);border-top:1px solid var(--border);background:rgba(0,0,0,0.12);line-height:1.6}

/* Errors */
.err-box{margin-top:0.9rem;padding:0.75rem 1rem;background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.2);border-radius:var(--r-sm);font-size:0.78rem;color:#fca5a5}
.err-box h4{margin-bottom:0.32rem;font-weight:600}
.err-box li{margin-left:0.95rem;margin-top:0.18rem;line-height:1.5}

/* Graph */
#graph-card{display:none}
#d3-graph{width:100%;height:380px;background:var(--s2);border-radius:var(--r-sm);overflow:hidden}

/* Code block */
.code-block{background:var(--s2);border:1px solid var(--border-md);border-radius:var(--r-sm);padding:0.85rem 1rem;font-family:var(--mono);font-size:0.75rem;color:var(--t2);line-height:1.8;overflow-x:auto}
.code-lbl{font-size:0.58rem;font-family:var(--font);letter-spacing:0.08em;text-transform:uppercase;color:var(--t3);margin-bottom:0.42rem;font-weight:600}

/* Terminal */
.terminal{
  background:#030308;border:1px solid var(--border-md);border-radius:var(--r-sm);
  padding:0.8rem 1rem;font-family:var(--mono);font-size:0.72rem;color:#5a6280;
  max-height:230px;overflow-y:auto;white-space:pre-wrap;line-height:1.7;
}
.terminal::-webkit-scrollbar{width:4px}
.terminal::-webkit-scrollbar-track{background:transparent}
.terminal::-webkit-scrollbar-thumb{background:var(--border-md);border-radius:2px}

/* Loop bar */
.loop-bar{
  display:flex;align-items:center;gap:0.55rem;flex-wrap:wrap;
  padding:0.55rem 0.8rem;background:var(--s2);border:1px solid var(--border);
  border-radius:var(--r-sm);font-size:0.72rem;margin-bottom:0.85rem;
}
#loop-status-text{color:var(--t2)}

/* Demo btn row */
.demo-btns{display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.85rem}

/* Approvals */
.ap-card{background:var(--s2);border:1px solid var(--border-md);border-radius:var(--r-sm);padding:0.8rem;margin-bottom:0.55rem}
.ap-hdr{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.38rem;gap:0.45rem}
.ap-title{color:var(--t1);font-size:0.82rem;font-weight:600;flex:1}
.ap-type{background:var(--s3);color:var(--t3);padding:0.15rem 0.45rem;border-radius:4px;font-size:0.62rem;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;flex-shrink:0}
.ap-prev{color:var(--t2);font-size:0.73rem;margin-bottom:0.6rem;line-height:1.5}
.ap-btns{display:flex;gap:0.4rem}

/* Sponsor / tools grid */
.sp-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(135px,1fr));gap:0.55rem}
.sp-card{background:var(--s2);border:1px solid var(--border);border-radius:var(--r-sm);padding:0.75rem 0.85rem}
.sp-name{font-weight:600;color:var(--t1);font-size:0.8rem;margin-bottom:0.18rem}
.sp-desc{color:var(--t3);font-size:0.67rem;line-height:1.4}

/* Connect grid */
.cn-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(165px,1fr));gap:0.55rem}
.cn-card{background:var(--s2);border:1px solid var(--border);border-radius:var(--r-sm);padding:0.85rem}
.cn-name{font-weight:600;color:var(--t1);font-size:0.83rem;margin-bottom:0.28rem}
.cn-desc{color:var(--t3);font-size:0.7rem;line-height:1.4;margin-bottom:0.6rem}
.cn-row{display:flex;align-items:center;gap:0.38rem}
.cn-st{font-size:0.68rem;color:var(--t3)}

/* Tools row */
.tools-row{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:0.45rem;margin-top:0.8rem}
.tool-card{background:var(--s2);border:1px solid var(--border);border-radius:6px;padding:0.48rem 0.7rem}
.tool-lbl{font-size:0.58rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:var(--accent-l);margin-bottom:0.12rem}
.tool-name{font-size:0.74rem;font-family:var(--mono);color:var(--t1)}

/* Regen */
.regen-row{display:flex;align-items:center;gap:0.7rem;flex-wrap:wrap;margin-top:0.38rem}
.regen-lnk{font-size:0.7rem;color:var(--t3);font-family:var(--mono)}
.regen-lnk a{color:var(--accent-l)}

/* Footer */
footer{text-align:center;padding:1.75rem 0 0.5rem;font-size:0.68rem;color:var(--t3);letter-spacing:0.04em;border-top:1px solid var(--border);margin-top:2rem}

/* Utils */
.mt1{margin-top:0.5rem}.mt2{margin-top:1rem}.mb1{margin-bottom:0.5rem}.mb2{margin-bottom:1rem}
.ml-auto{margin-left:auto}.flex{display:flex}.items-c{align-items:center}.gap2{gap:0.5rem}
</style>
</head>
<body>

<div class="orb-field">
  <div class="orb orb-1"></div>
  <div class="orb orb-2"></div>
  <div class="orb orb-3"></div>
</div>

<nav>
  <div class="nav-inner">
    <div class="nav-logo">Trace</div>
    <div class="nav-tag">Curiosity OS</div>
    <div class="nav-status">
      <div class="sdot" id="loop-dot"></div>
      <span id="loop-status-text">checking…</span>
      <span id="loop-next-run"></span>
    </div>
  </div>
</nav>

<div class="page">

<section class="hero">
  <div class="hero-pill">Agent-native · MCP-first · Autonomous</div>
  <h1 class="hero-title">Your curiosity,<br><span class="grad">always surfaced</span></h1>
  <p class="hero-sub">Trace infers what you genuinely care about from your browser history, YouTube, and ChatGPT — then exposes a live curiosity API that any AI agent can query.</p>
</section>

<!-- UPLOAD -->
<div class="sec-hdr"><h2>Upload Your Signals</h2></div>

<div class="privacy">
  <strong>Your data stays local.</strong> You export your own files from Google / ChatGPT — no passwords required. Files are used once to build your curiosity profile, then <strong>deleted immediately</strong>. OAuth tokens for connected services live in Scalekit's encrypted vault, never in this server.
</div>

<div class="card">
  <div class="src-list">
    <div class="sc" id="card-google">
      <div class="sc-head" onclick="toggleCard('google')">
        <div class="sc-icon">🌐</div>
        <div class="sc-info"><div class="sc-name">Chrome / Browser History</div><div class="sc-desc">BrowserHistory.json or Google Takeout ZIP</div></div>
        <span class="sc-badge" id="status-google">Optional</span>
        <span class="sc-chev">▼</span>
      </div>
      <div class="sc-body" id="body-google">
        <div class="howto"><ol>
          <li>Go to <strong>takeout.google.com</strong></li>
          <li>Deselect all → select only <code>Chrome</code></li>
          <li>Export → Download the ZIP (multiple date ranges OK)</li>
          <li>Upload the <strong>ZIP directly</strong>, or extract <code>BrowserHistory.json</code></li>
        </ol></div>
        <label class="drop" id="drop-google">
          <input type="file" id="file-google" accept=".json,.zip" multiple onchange="fileChosen('google')">
          <div class="drop-label">Drop BrowserHistory.json or Takeout ZIP(s)</div>
          <div class="drop-sub">click to browse · .json and .zip · multiple files OK</div>
          <div class="drop-ok" id="chosen-google"></div>
        </label>
      </div>
    </div>

    <div class="sc" id="card-youtube">
      <div class="sc-head" onclick="toggleCard('youtube')">
        <div class="sc-icon">▶</div>
        <div class="sc-info"><div class="sc-name">YouTube Watch History</div><div class="sc-desc">watch-history.json from Google Takeout</div></div>
        <span class="sc-badge" id="status-youtube">Optional</span>
        <span class="sc-chev">▼</span>
      </div>
      <div class="sc-body" id="body-youtube">
        <div class="howto"><ol>
          <li>Go to <strong>takeout.google.com</strong></li>
          <li>Deselect all → select <code>YouTube and YouTube Music</code></li>
          <li>Export → Download → extract ZIP → find <code>history/watch-history.json</code></li>
        </ol></div>
        <label class="drop" id="drop-youtube">
          <input type="file" id="file-youtube" accept=".json" onchange="fileChosen('youtube')">
          <div class="drop-label">Drop watch-history.json</div>
          <div class="drop-sub">click to browse · .json only</div>
          <div class="drop-ok" id="chosen-youtube"></div>
        </label>
      </div>
    </div>

    <div class="sc" id="card-chatgpt">
      <div class="sc-head" onclick="toggleCard('chatgpt')">
        <div class="sc-icon">💬</div>
        <div class="sc-info"><div class="sc-name">ChatGPT Export</div><div class="sc-desc">conversations.json — highest-signal source</div></div>
        <span class="sc-badge" id="status-chatgpt">Optional</span>
        <span class="sc-chev">▼</span>
      </div>
      <div class="sc-body" id="body-chatgpt">
        <div class="howto"><ol>
          <li>Open ChatGPT → avatar → <strong>Settings → Data Controls → Export Data</strong></li>
          <li>Wait for the email → Download → extract → upload <code>conversations.json</code></li>
          <li>ChatGPT conversations carry 1.5× weight — your most explicit intellectual intent</li>
        </ol></div>
        <label class="drop" id="drop-chatgpt">
          <input type="file" id="file-chatgpt" accept=".json,.zip" onchange="fileChosen('chatgpt')">
          <div class="drop-label">Drop conversations.json or ChatGPT ZIP</div>
          <div class="drop-sub">click to browse · .json or .zip</div>
          <div class="drop-ok" id="chosen-chatgpt"></div>
        </label>
      </div>
    </div>
  </div>

  <div class="sig-row">
    <span class="sig-lbl">Signal strength</span>
    <div class="sig-track"><div class="sig-fill" id="sig-fill"></div></div>
    <span class="sig-count" id="sig-count">0 / 3 sources</span>
  </div>

  <div class="mt2">
    <button class="btn btn-primary btn-full" id="gen-btn" onclick="generate()">Generate Intelligence Digest →</button>
    <div class="loading-wrap" id="loading-wrap">
      <div class="spinner"></div>
      <div class="stage-msg" id="stage-msg">Preparing…</div>
      <div class="timing-note">Large histories take 1–2 minutes — keep this tab open</div>
    </div>
  </div>
</div>

<!-- RESULT -->
<div id="result">
  <div class="sec-hdr"><h2>Intelligence Digest</h2></div>
  <div class="card">
    <div class="res-eyebrow">Your Curiosity Brief</div>
    <div class="res-subject" id="subject"></div>
    <div class="res-meta" id="meta"></div>
    <div id="curiosity-profile" class="mt2"></div>
    <div class="act-row mt1">
      <button class="btn btn-ghost btn-sm" onclick="downloadHtml()">↓ HTML</button>
      <button class="btn btn-ghost btn-sm" onclick="downloadText()">↓ Text</button>
      <span class="ml-auto" id="share-link-container"></span>
    </div>
    <div id="toc"></div>
    <div id="sections"></div>
    <div id="errors-container"></div>
  </div>
</div>

<!-- GRAPH -->
<div class="card" id="graph-card">
  <div class="clabel"><div class="clabel-dot" style="background:var(--cyan)"></div>Curiosity Graph · Semantic topic network</div>
  <div id="d3-graph"></div>
</div>

<!-- AUTONOMOUS LOOP DEMO -->
<div class="sec-hdr"><h2>Autonomous Agent Loop</h2></div>

<div class="card accent-border">
  <div class="clabel"><div class="clabel-dot" style="background:var(--accent-l)"></div>Demo Control Panel</div>
  <p style="font-size:0.8rem;color:var(--t2);margin-bottom:1.1rem;line-height:1.65">
    Trace runs three loops every 30 s in DEMO_MODE: <strong style="color:var(--t1)">poll Gmail signals → scrape fresh articles → detect patterns → dispatch AI actions</strong>. Use the buttons below to trigger a manual cycle and watch the agent reason in real-time.
  </p>

  <div class="loop-bar">
    <div class="sdot" id="loop-dot-demo"></div>
    <span id="loop-status-text-demo" style="color:var(--t2)">checking…</span>
    <span id="loop-next-run-demo" style="color:var(--t3);font-family:var(--mono);font-size:0.67rem;margin-left:auto"></span>
  </div>

  <div class="demo-btns">
    <button class="btn btn-v btn-sm" onclick="seedDemo()">1. Seed Demo Profile</button>
    <button class="btn btn-primary btn-sm" onclick="runLoopNow()">2. Run Autonomous Loop ▶</button>
    <button class="btn btn-c btn-sm" onclick="loadGraph()">3. Render Curiosity Graph</button>
    <button class="btn btn-ghost btn-sm" onclick="checkApprovals()">4. Check Pending Approvals</button>
  </div>

  <div class="terminal" id="demo-output">// Trace agent loop output will appear here...
</div>
</div>

<!-- PENDING APPROVALS -->
<div class="card" id="approvals-card" style="display:none">
  <div class="clabel"><div class="clabel-dot" style="background:var(--pink)"></div>Pending Actions — Tier B (require your approval)</div>
  <p style="font-size:0.77rem;color:var(--t2);margin-bottom:0.8rem;line-height:1.6">
    These drafts were prepared by the agent. Gmail drafts are only created after you approve. Reddit posts are never auto-published.
  </p>
  <div id="approvals-list"></div>
</div>

<!-- MCP -->
<div class="sec-hdr"><h2>Connect to Claude</h2></div>
<div class="card">
  <div class="clabel"><div class="clabel-dot" style="background:var(--accent-l)"></div>MCP Server · Add Trace to Claude Desktop</div>
  <p style="font-size:0.8rem;color:var(--t2);margin-bottom:0.9rem;line-height:1.65">
    Trace is an MCP server. Add this config to <code style="background:var(--s3);padding:0.1em 0.35em;border-radius:3px;font-size:0.77rem">claude_desktop_config.json</code> and Claude can query your live curiosity profile directly.
  </p>
  <div class="code-block">
    <div class="code-lbl">claude_desktop_config.json</div>
    <div><span style="color:#818cf8">"mcpServers"</span>: {</div>
    <div>&nbsp;&nbsp;<span style="color:#34d399">"trace"</span>: { <span style="color:#818cf8">"url"</span>: <span style="color:#fbbf24">"http://localhost:8000/mcp"</span>, <span style="color:#818cf8">"transport"</span>: <span style="color:#fbbf24">"http"</span> }</div>
    <div>}</div>
  </div>
  <div class="tools-row">
    <div class="tool-card"><div class="tool-lbl">Tool</div><div class="tool-name">get_curiosity_topics</div></div>
    <div class="tool-card"><div class="tool-lbl">Tool</div><div class="tool-name">get_unresolved_questions</div></div>
    <div class="tool-card"><div class="tool-lbl">Tool</div><div class="tool-name">generate_briefing</div></div>
    <div class="tool-card"><div class="tool-lbl">Tool</div><div class="tool-name">get_emerging_interests</div></div>
    <div class="tool-card"><div class="tool-lbl">Tool</div><div class="tool-name">track_signal</div></div>
    <div class="tool-card"><div class="tool-lbl">Tool</div><div class="tool-name">get_topic_neighbors</div></div>
  </div>
</div>

<!-- CONNECT SERVICES -->
<div class="sec-hdr"><h2>Connect Services</h2></div>
<div class="card">
  <div class="clabel"><div class="clabel-dot" style="background:var(--yellow)"></div>Scalekit Token Vault · Secure OAuth for agent actions</div>
  <p style="font-size:0.78rem;color:var(--t2);margin-bottom:0.9rem;line-height:1.65">Your OAuth tokens live in <strong style="color:var(--t1)">Scalekit's encrypted vault</strong> — never in Trace's env vars or memory. Once connected, the autonomous agent can act on your behalf.</p>
  <div class="cn-grid">
    <div class="cn-card" style="border-color:rgba(124,58,237,0.28)">
      <div class="cn-name">📧 Gmail</div>
      <div class="cn-desc">Read newsletter subjects → subscription-debt signals</div>
      <div class="cn-row"><button class="btn btn-v btn-sm" onclick="connectService('gmail')">Connect</button><span class="cn-st" id="gmail-status"></span></div>
    </div>
    <div class="cn-card"><div class="cn-name">📝 Notion</div><div class="cn-desc">Auto-save emerging topics as pages (Tier A)</div><div class="cn-row"><button class="btn btn-ghost btn-sm" onclick="connectService('notion-akG2REQU')">Connect</button><span class="cn-st" id="notion-akG2REQU-status"></span></div></div>
    <div class="cn-card"><div class="cn-name">📅 Calendar</div><div class="cn-desc">Schedule deep-dive sessions (Tier A)</div><div class="cn-row"><button class="btn btn-ghost btn-sm" onclick="connectService('googlecalendar-fe75NXhO')">Connect</button><span class="cn-st" id="googlecalendar-fe75NXhO-status"></span></div></div>
    <div class="cn-card"><div class="cn-name">💬 Slack</div><div class="cn-desc">DM when a pattern triggers (Tier A)</div><div class="cn-row"><button class="btn btn-ghost btn-sm" onclick="connectService('slack-RLnbqcmP')">Connect</button><span class="cn-st" id="slack-RLnbqcmP-status"></span></div></div>
    <div class="cn-card"><div class="cn-name">🟠 Reddit</div><div class="cn-desc">Prepare bridge-topic drafts (Tier B, never auto-posts)</div><div class="cn-row"><button class="btn btn-ghost btn-sm" onclick="connectService('reddit')">Connect</button><span class="cn-st" id="reddit-status"></span></div></div>
  </div>
</div>

<!-- BUILT WITH -->
<div class="sec-hdr"><h2>Built With</h2></div>
<div class="card">
  <div class="sp-grid">
    <div class="sp-card"><div class="sp-name">Anthropic Claude</div><div class="sp-desc">Topic extraction · briefing · significance gating</div></div>
    <div class="sp-card"><div class="sp-name">Apify</div><div class="sp-desc">rag-web-browser · quality-first article scraping</div></div>
    <div class="sp-card"><div class="sp-name">Scalekit</div><div class="sp-desc">OAuth 2.1 MCP Auth · Token Vault</div></div>
    <div class="sp-card"><div class="sp-name">Tigris Data</div><div class="sp-desc">S3-compatible global object storage</div></div>
    <div class="sp-card"><div class="sp-name">Kalibr</div><div class="sp-desc">Agent orchestration · retry · failure detection</div></div>
    <div class="sp-card"><div class="sp-name">Redis</div><div class="sp-desc">Sub-50ms curiosity graph reads via ZSET</div></div>
    <div class="sp-card"><div class="sp-name">Render</div><div class="sp-desc">One-click deploy with managed Redis</div></div>
  </div>
</div>

</div><!-- /page -->

<footer>Trace — Curiosity OS &nbsp;·&nbsp; Applied Intelligence Hackathon 2026 &nbsp;·&nbsp; Claude · Apify · Scalekit · Tigris Data · Kalibr · Redis · Render</footer>

<script>
let newsletterData = null, fileCount = 0;

function toggleCard(type) {
  document.getElementById('card-' + type).classList.toggle('open');
}

function fileChosen(type) {
  const input = document.getElementById('file-' + type);
  const files = Array.from(input.files);
  const chosenEl = document.getElementById('chosen-' + type);
  const statusEl = document.getElementById('status-' + type);
  const card = document.getElementById('card-' + type);
  if (files.length > 0) {
    chosenEl.textContent = files.length === 1 ? '✓ ' + files[0].name : '✓ ' + files.length + ' files: ' + files.map(f => f.name).join(', ');
    statusEl.textContent = files.length > 1 ? '✓ ' + files.length + ' files' : '✓ Ready';
    statusEl.className = 'sc-badge ok';
    card.classList.add('active');
  } else {
    chosenEl.textContent = ''; statusEl.textContent = 'Optional'; statusEl.className = 'sc-badge'; card.classList.remove('active');
  }
  updateSigBar();
}

function updateSigBar() {
  const types = ['google','youtube','chatgpt'];
  const count = types.filter(t => document.getElementById('file-'+t).files.length > 0).length;
  fileCount = count;
  document.getElementById('sig-fill').style.width = (count / 3 * 100) + '%';
  document.getElementById('sig-count').textContent = count + ' / 3 sources';
}

['google','youtube','chatgpt'].forEach(t => {
  const el = document.getElementById('drop-' + t);
  el.addEventListener('dragover', e => { e.preventDefault(); el.classList.add('over'); });
  el.addEventListener('dragleave', () => el.classList.remove('over'));
  el.addEventListener('drop', e => {
    e.preventDefault(); el.classList.remove('over');
    const dt = e.dataTransfer;
    if (dt.files.length) {
      if (t === 'google') {
        try { const dta = new DataTransfer(); Array.from(dt.files).forEach(f => dta.items.add(f)); document.getElementById('file-'+t).files = dta.files; }
        catch { document.getElementById('file-'+t).files = dt.files; }
      } else { document.getElementById('file-'+t).files = dt.files; }
      fileChosen(t);
    }
  });
});

const STAGE_MSGS = [
  'Reading your history and signals…',
  'Clustering curiosity topics with Claude — 20-40 seconds…',
  'Fetching fresh articles from arXiv, Hacker News & web…',
  'Assembling personalised context window…',
  'Writing your newsletter with Claude…',
  'Still working — large histories can take up to 2 minutes…',
  'Almost there — finalising…',
];
const STAGE_DELAYS = [3000,18000,15000,5000,15000,20000,20000];
let stageIdx = 0, stageTimer;

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

function setLoading(on) {
  document.getElementById('gen-btn').style.display = on ? 'none' : 'block';
  const w = document.getElementById('loading-wrap');
  w.style.display = on ? 'flex' : 'none';
  if (on) w.style.flexDirection = 'column';
  if (!on) document.getElementById('stage-msg').className = 'stage-msg';
}

function setError(msg) {
  const el = document.getElementById('stage-msg');
  el.textContent = msg; el.className = 'stage-msg err';
  document.getElementById('loading-wrap').style.display = 'flex';
  document.getElementById('loading-wrap').style.flexDirection = 'column';
  document.getElementById('gen-btn').style.display = 'block';
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}
function safeHref(u) {
  if (typeof u !== 'string') return '#';
  const l = u.toLowerCase();
  return (l.startsWith('http://') || l.startsWith('https://')) ? u : '#';
}
function sourceBadge(url) {
  if (url.includes('arxiv.org')) return '<span class="spill sp-arxiv">arXiv</span>';
  if (url.includes('ycombinator.com')) return '<span class="spill sp-hn">HN</span>';
  return '<span class="spill sp-web">Web</span>';
}
function badgeClass(type) { return 'nl-badge badge-' + (type || 'weekly_topics'); }
function badgeLabel(type) { return {weekly_topics:'This Week',curiosity_debt:'Curiosity Debt',rabbit_hole:'Rabbit Hole'}[type] || type; }
function fmtContent(text) {
  const parts = String(text).split('\\n\\n').map(p => p.trim()).filter(Boolean);
  return parts.length ? parts.map(p => '<p>'+esc(p)+'</p>').join('') : '<p>'+esc(String(text).trim())+'</p>';
}
function downloadBlob(content, filename, mime) {
  const blob = new Blob([content],{type:mime}), a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = filename; a.click(); URL.revokeObjectURL(a.href);
}
function downloadHtml() {
  if (!newsletterData) return;
  downloadBlob(newsletterData.html, 'trace-'+newsletterData.subject_line.replace(/[^a-z0-9]+/gi,'-').toLowerCase()+'.html','text/html');
}
function downloadText() {
  if (!newsletterData) return;
  downloadBlob(newsletterData.plain_text, 'trace-'+newsletterData.subject_line.replace(/[^a-z0-9]+/gi,'-').toLowerCase()+'.txt','text/plain');
}

async function regenerate() {
  if (!newsletterData?.profile_id) return;
  const btn = document.getElementById('regen-btn');
  if (btn) btn.disabled = true;
  document.getElementById('result').style.display = 'none';
  setLoading(true); stageIdx = 0; tickStage();
  try {
    const r = await fetch('/newsletter/regenerate/'+encodeURIComponent(newsletterData.profile_id),{method:'POST'});
    if (!r.ok) { let msg='Regeneration failed'; try{const e=await r.json();msg=e.detail||msg;}catch{}throw new Error(msg); }
    stopStages(); setLoading(false); renderNewsletter(await r.json());
  } catch(err) {
    stopStages(); setLoading(false); setError('Error: '+err.message);
    if (btn) btn.disabled = false;
    document.getElementById('result').style.display = 'block';
  }
}

function renderNewsletter(data) {
  newsletterData = data;
  document.getElementById('subject').textContent = data.subject_line;
  const dt = new Date(data.generated_at);
  document.getElementById('meta').textContent = dt.toLocaleString() + (data.generated_for ? ' · for '+data.generated_for : '');

  const profileEl = document.getElementById('curiosity-profile');
  if (data.topic_names?.length) {
    const scores = data.topic_scores || [];
    const chips = data.topic_names.map((t,i) => {
      const s = scores[i]??0, tip = scores[i]!=null ? ` title="Curiosity strength: ${Math.round(s*100)}%"` : '';
      return `<span class="chip"${tip}>${esc(t)}</span>`;
    }).join('');
    const regenHtml = data.profile_id ? `<div class="regen-row mt1">
      <button class="btn btn-ghost btn-sm" id="regen-btn" onclick="regenerate()">↺ Regenerate with today's articles</button>
      <span class="regen-lnk">Bookmark: <a href="/newsletter/regenerate/${esc(data.profile_id)}" onclick="return false">/regenerate/${esc(data.profile_id.substring(0,8))}…</a></span>
    </div>` : '';
    profileEl.innerHTML = `<div style="margin-bottom:0.7rem">
      <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.13em;text-transform:uppercase;color:var(--green);margin-bottom:0.5rem">Curiosity Profile · ${data.topic_names.length} topic${data.topic_names.length!==1?'s':''} inferred</div>
      <div class="chips">${chips}</div>${regenHtml}</div>`;
  } else { profileEl.innerHTML = ''; }

  const sc = document.getElementById('share-link-container');
  sc.innerHTML = data.id ? `<span style="font-size:0.7rem;color:var(--t3)">Permalink: <a href="/newsletter/${esc(data.id)}" target="_blank">/newsletter/${esc(data.id)}</a></span>` : '';

  const tocEl = document.getElementById('toc');
  if (data.sections?.length > 1) {
    tocEl.innerHTML = `<div class="toc-box"><div class="toc-lbl">In this issue</div><ol>${data.sections.map((s,i)=>`<li><a href="#section-${i}">${esc(s.title)}</a></li>`).join('')}</ol></div>`;
  } else tocEl.innerHTML = '';

  const secEl = document.getElementById('sections');
  secEl.innerHTML = '';
  data.sections.forEach((s, i) => {
    const div = document.createElement('div');
    div.className = 'nls'; div.id = 'section-'+i;
    const urls = (s.source_urls||[]).map(u => `<a href="${esc(safeHref(u))}" target="_blank" rel="noopener noreferrer">${sourceBadge(u)}<span class="slink">${esc(u)}</span></a>`).join('');
    div.innerHTML = `<span class="${badgeClass(s.section_type)}">${esc(badgeLabel(s.section_type))}</span>
      <h3>${esc(s.title)}</h3>
      <div class="nl-body">${fmtContent(s.content)}</div>
      ${urls?'<div class="srcs">'+urls+'</div>':''}
      <details class="why"><summary>Why this section?</summary><div class="why-body">${esc(s.audit_reasoning)}</div></details>`;
    secEl.appendChild(div);
  });

  const errBox = document.getElementById('errors-container');
  errBox.innerHTML = (data.errors?.length) ? `<div class="err-box"><h4>Non-fatal warnings (${data.errors.length})</h4><ul>${data.errors.map(e=>'<li>'+esc(e)+'</li>').join('')}</ul></div>` : '';

  const resultEl = document.getElementById('result');
  resultEl.style.display = 'block';
  resultEl.scrollIntoView({behavior:'smooth'});
}

async function generate() {
  const gf = Array.from(document.getElementById('file-google').files);
  const yf = Array.from(document.getElementById('file-youtube').files);
  const cf = Array.from(document.getElementById('file-chatgpt').files);
  if (!gf.length && !yf.length && !cf.length) {
    setError('Please upload at least one file first.');
    document.getElementById('loading-wrap').style.display = 'flex';
    document.getElementById('loading-wrap').style.flexDirection = 'column';
    return;
  }
  document.getElementById('result').style.display = 'none';
  setLoading(true); stageIdx = 0; tickStage();
  try {
    const histIds = [];
    for (const f of gf) {
      const fd = new FormData(); fd.append('file',f); fd.append('file_type','history');
      const r = await fetch('/upload',{method:'POST',body:fd});
      if (!r.ok) { const e=await r.json(); throw new Error(e.detail||'Upload failed: '+f.name); }
      histIds.push((await r.json()).upload_id);
    }
    let ytId=null, cgId=null;
    if (yf[0]) {
      const fd=new FormData(); fd.append('file',yf[0]); fd.append('file_type','youtube');
      const r=await fetch('/upload',{method:'POST',body:fd});
      if (!r.ok) { const e=await r.json(); throw new Error(e.detail||'Upload failed'); }
      ytId=(await r.json()).upload_id;
    }
    if (cf[0]) {
      const fd=new FormData(); fd.append('file',cf[0]); fd.append('file_type','chatgpt');
      const r=await fetch('/upload',{method:'POST',body:fd});
      if (!r.ok) { const e=await r.json(); throw new Error(e.detail||'Upload failed'); }
      cgId=(await r.json()).upload_id;
    }
    const body={};
    if (histIds.length) body.history_upload_ids=histIds;
    if (ytId) body.youtube_upload_id=ytId;
    if (cgId) body.chatgpt_upload_id=cgId;
    const r2=await fetch('/newsletter/from-upload',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if (!r2.ok) {
      let msg='Generation failed';
      try{const ej=await r2.json();msg=ej.detail||msg;}catch{try{msg=await r2.text();}catch{}}
      throw new Error(msg);
    }
    stopStages(); setLoading(false); renderNewsletter(await r2.json());
  } catch(err) { stopStages(); setLoading(false); setError('Error: '+err.message); }
}

async function connectService(connectionName) {
  const st = document.getElementById(connectionName+'-status');
  if (st) st.textContent = 'Connecting…';
  try {
    const r = await fetch('/auth/connect?connection_name='+encodeURIComponent(connectionName));
    const d = await r.json();
    if (d.link) { window.open(d.link,'_blank','width=600,height=700'); if(st) st.textContent='⟳ Complete in popup'; }
    else if (d.status==='connector_not_found') { if(st) st.textContent='⚠ Not configured in Scalekit'; }
    else if (d.message) { if(st) st.textContent=d.message.slice(0,70); }
    else { if(st) st.textContent='No link returned'; }
  } catch(err) { if(st) st.textContent='Error: '+err.message.slice(0,50); }
}

function demoLog(msg) {
  const el = document.getElementById('demo-output');
  if (!el) return;
  const ts = new Date().toISOString().slice(11,19);
  el.textContent = '['+ts+'] '+msg+'\n'+el.textContent;
}

async function _fetchJson(url, opts) {
  const r = await fetch(url, opts||{});
  if (!r.ok) {
    let detail='';
    try{const e=await r.json();detail=e.detail||JSON.stringify(e);}catch{}
    throw new Error('HTTP '+r.status+(detail?': '+detail:''));
  }
  return r.json();
}

async function seedDemo() {
  demoLog('Seeding demo profile with ML/robotics curiosity graph...');
  try {
    const d = await _fetchJson('/demo/seed',{method:'POST'});
    demoLog('✅ Seeded '+d.topic_count+' topics | profile='+d.profile_id);
    demoLog('   Topics: '+(d.topics||[]).join(', '));
    demoLog('   → Now click "Run Autonomous Loop" to see the agent in action');
  } catch(e) { demoLog('❌ Seed failed: '+e.message); }
}

async function runLoopNow() {
  demoLog('▶ Running pattern detection + autonomous dispatch...');
  const btn = document.querySelector('[onclick="runLoopNow()"]');
  if (btn) { btn.disabled=true; btn.textContent='⏳ Running...'; }
  try {
    const d = await _fetchJson('/demo/run-detect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({profile_id:'demo'})});
    demoLog('🔍 Patterns: '+d.patterns_detected+' | Actions: '+d.actions_dispatched);
    if (d.patterns?.length) d.patterns.forEach(p=>demoLog('  📌 '+p.pattern+' → "'+p.topic+'" (score='+p.score+')'));
    else demoLog('  ℹ No new patterns — seed the demo profile first');
    if (d.actions?.length) {
      demoLog('⚡ Agent actions:');
      d.actions.forEach(a => {
        const s=a.status||'?', icon=s==='sent'||s==='created'?'✅':s==='queued_for_approval'?'📬':s==='stub'?'🔵':'⚠️';
        demoLog('  '+icon+' '+(a.action_type||a.via||s)+(a.topic||a.title?' — '+(a.topic||(a.title||'').slice(0,40)):''));
      });
    }
    setTimeout(checkApprovals,300);
  } catch(e) { demoLog('❌ Detect failed: '+e.message+' — seed the demo profile first?'); }
  finally { if (btn) { btn.disabled=false; btn.textContent='2. Run Autonomous Loop ▶'; } }
}

async function loadGraph() {
  demoLog('Loading curiosity graph...');
  try {
    let d = await _fetchJson('/graph.json?profile_id=demo');
    if (!d.nodes?.length) { demoLog('Demo empty — trying default...'); d = await _fetchJson('/graph.json?profile_id=default'); }
    if (!d.nodes?.length) { demoLog('⚠️ No graph data. Seed the demo profile first.'); return; }
    const s=d.stats||{};
    demoLog('📊 Graph: '+(s.node_count||0)+' nodes · '+(s.link_count||0)+' edges · '+(s.community_count||0)+' communities');
    const card=document.getElementById('graph-card');
    card.style.display='';
    requestAnimationFrame(() => { try { renderD3Graph(d); } catch(err) { demoLog('❌ Graph render error: '+err.message); } });
  } catch(e) { demoLog('❌ Graph load failed: '+e.message); }
}

async function checkApprovals() {
  try {
    const d = await _fetchJson('/approvals?profile_id=demo');
    demoLog('📬 Pending approvals: '+d.count+(d.count===0?' — run the loop first':''));
    if (d.count > 0) { document.getElementById('approvals-card').style.display=''; renderApprovals(d.pending); }
  } catch(e) { demoLog('❌ Approvals check failed: '+e.message); }
}

function _syncDot(data, dotId, textId, nextId) {
  const dot=document.getElementById(dotId), txt=document.getElementById(textId), nxt=document.getElementById(nextId);
  if (!dot||!txt) return;
  if (data.enabled && data.running) {
    dot.className='sdot live';
    txt.style.color='var(--green)'; txt.textContent='⚡ Loop running'+(data.demo_mode?' · DEMO (30s)':' · production');
    const j=(data.jobs||[]).find(j=>j.id==='detect_and_act');
    if (j?.next_run && nxt) { const s=Math.max(0,Math.round((new Date(j.next_run)-Date.now())/1000)); nxt.textContent='next in '+s+'s'; }
  } else if (data.enabled) {
    dot.className='sdot'; dot.style.background='var(--yellow)';
    txt.style.color='var(--yellow)'; txt.textContent='⏸ Scheduler paused';
    if (nxt) nxt.textContent='';
  } else {
    dot.className='sdot'; dot.style.background='var(--t3)';
    txt.style.color='var(--t2)'; txt.textContent='○ Loop disabled — use manual trigger above';
    if (nxt) nxt.textContent='';
  }
}

async function refreshLoopStatus() {
  try {
    const d = await _fetchJson('/agent/status');
    _syncDot(d,'loop-dot','loop-status-text','loop-next-run');
    _syncDot(d,'loop-dot-demo','loop-status-text-demo','loop-next-run-demo');
  } catch { const t=document.getElementById('loop-status-text'); if(t) t.textContent='Could not reach /agent/status'; }
}
refreshLoopStatus();
setInterval(refreshLoopStatus, 5000);

function renderApprovals(items) {
  const el = document.getElementById('approvals-list');
  if (!el) return;
  el.innerHTML = items.map(a => `
    <div class="ap-card">
      <div class="ap-hdr"><div class="ap-title">${esc(a.title)}</div><div class="ap-type">${esc(a.action_type)}</div></div>
      <div class="ap-prev">${esc((a.preview||'').slice(0,200))}${(a.preview||'').length>200?'…':''}</div>
      <div class="ap-btns">
        <button class="btn btn-g btn-sm" onclick="approveAction('${esc(a.id)}')">✓ Approve</button>
        <button class="btn btn-r btn-sm" onclick="rejectAction('${esc(a.id)}')">✗ Reject</button>
      </div>
    </div>`).join('');
}

async function approveAction(id) {
  demoLog('Approving action '+id.slice(0,8)+'...');
  try {
    const d=await _fetchJson('/approvals/'+id+'/approve',{method:'POST'});
    demoLog('✅ Approved: '+(d.action?.action_type||'?')+' — '+(d.action?.title||'').slice(0,50));
    checkApprovals();
  } catch(e) { demoLog('❌ Approve failed: '+e.message); }
}

async function rejectAction(id) {
  demoLog('Rejecting '+id.slice(0,8)+'...');
  try {
    const d=await _fetchJson('/approvals/'+id+'/reject',{method:'POST'});
    demoLog('✗ Rejected: '+(d.action?.action_type||'?'));
    checkApprovals();
  } catch(e) { demoLog('❌ Reject failed: '+e.message); }
}

function renderD3Graph(data) {
  const container=document.getElementById('d3-graph');
  if (!container) return;
  container.innerHTML='';
  const W=container.offsetWidth||700, H=380;
  const colors=['#6366f1','#10b981','#f59e0b','#ef4444','#8b5cf6','#22d3ee','#f97316','#84cc16'];

  const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
  svg.setAttribute('width',W); svg.setAttribute('height',H); svg.style.cssText='width:100%;height:100%;';

  const defs=document.createElementNS('http://www.w3.org/2000/svg','defs');
  const filt=document.createElementNS('http://www.w3.org/2000/svg','filter');
  filt.setAttribute('id','glow');
  const feBlur=document.createElementNS('http://www.w3.org/2000/svg','feGaussianBlur');
  feBlur.setAttribute('stdDeviation','4'); feBlur.setAttribute('result','blur');
  const feMerge=document.createElementNS('http://www.w3.org/2000/svg','feMerge');
  const fn1=document.createElementNS('http://www.w3.org/2000/svg','feMergeNode'); fn1.setAttribute('in','blur');
  const fn2=document.createElementNS('http://www.w3.org/2000/svg','feMergeNode'); fn2.setAttribute('in','SourceGraphic');
  feMerge.append(fn1,fn2); filt.append(feBlur,feMerge); defs.appendChild(filt); svg.appendChild(defs);

  const nodes=data.nodes.map((n,i) => ({...n,x:W/2+(Math.random()-0.5)*200,y:H/2+(Math.random()-0.5)*200,vx:0,vy:0}));
  const nm={}; nodes.forEach(n=>nm[n.id]=n);

  for (let it=0;it<80;it++) {
    for (let i=0;i<nodes.length;i++) for (let j=i+1;j<nodes.length;j++) {
      const dx=nodes[j].x-nodes[i].x, dy=nodes[j].y-nodes[i].y, d=Math.sqrt(dx*dx+dy*dy)||1, f=5000/(d*d);
      nodes[i].vx-=dx/d*f; nodes[i].vy-=dy/d*f; nodes[j].vx+=dx/d*f; nodes[j].vy+=dy/d*f;
    }
    data.links.forEach(l=>{
      const a=nm[l.source], b=nm[l.target]; if (!a||!b) return;
      const dx=b.x-a.x, dy=b.y-a.y, d=Math.sqrt(dx*dx+dy*dy)||1, f=(d-100)*0.03*(l.value||0.5);
      a.vx+=dx/d*f; a.vy+=dy/d*f; b.vx-=dx/d*f; b.vy-=dy/d*f;
    });
    nodes.forEach(n=>{
      n.vx+=(W/2-n.x)*0.01; n.vy+=(H/2-n.y)*0.01;
      n.vx*=0.82; n.vy*=0.82;
      n.x=Math.max(40,Math.min(W-40,n.x+n.vx));
      n.y=Math.max(30,Math.min(H-30,n.y+n.vy));
    });
  }

  const eg=document.createElementNS('http://www.w3.org/2000/svg','g');
  data.links.forEach(l=>{
    const a=nm[l.source], b=nm[l.target]; if (!a||!b) return;
    const line=document.createElementNS('http://www.w3.org/2000/svg','line');
    line.setAttribute('x1',a.x); line.setAttribute('y1',a.y); line.setAttribute('x2',b.x); line.setAttribute('y2',b.y);
    line.setAttribute('stroke','rgba(148,163,255,0.1)'); line.setAttribute('stroke-width',Math.max(0.5,(l.value||0.5)*2.5));
    eg.appendChild(line);
  });
  svg.appendChild(eg);

  const ng=document.createElementNS('http://www.w3.org/2000/svg','g');
  nodes.forEach(n=>{
    const score=typeof n.blended_score==='number'?n.blended_score:0.3;
    const community=typeof n.community==='number'?n.community:0;
    const r=Math.max(5,Math.min(22,5+score*24));
    const color=colors[community%colors.length];
    const g=document.createElementNS('http://www.w3.org/2000/svg','g');
    g.style.cursor='pointer';
    const glow=document.createElementNS('http://www.w3.org/2000/svg','circle');
    glow.setAttribute('cx',n.x); glow.setAttribute('cy',n.y); glow.setAttribute('r',r+5);
    glow.setAttribute('fill',color); glow.setAttribute('opacity','0.13'); glow.setAttribute('filter','url(#glow)');
    const c=document.createElementNS('http://www.w3.org/2000/svg','circle');
    c.setAttribute('cx',n.x); c.setAttribute('cy',n.y); c.setAttribute('r',r);
    c.setAttribute('fill',color); c.setAttribute('opacity','0.88');
    c.setAttribute('stroke','rgba(255,255,255,0.13)'); c.setAttribute('stroke-width','1');
    const lbl=document.createElementNS('http://www.w3.org/2000/svg','text');
    lbl.setAttribute('x',n.x); lbl.setAttribute('y',n.y+r+12);
    lbl.setAttribute('text-anchor','middle'); lbl.setAttribute('font-size','8.5');
    lbl.setAttribute('font-family','Plus Jakarta Sans,sans-serif'); lbl.setAttribute('fill','rgba(200,210,255,0.65)');
    const name=(n.label||n.id||'');
    lbl.textContent=name.length>16?name.substring(0,14)+'…':name;
    const title=document.createElementNS('http://www.w3.org/2000/svg','title');
    title.textContent=name+(typeof n.blended_score==='number'?' · score: '+n.blended_score.toFixed(3):'');
    g.append(glow,c,lbl,title);
    g.addEventListener('mouseenter',()=>{c.setAttribute('opacity','1');c.setAttribute('r',r+2);glow.setAttribute('opacity','0.28')});
    g.addEventListener('mouseleave',()=>{c.setAttribute('opacity','0.88');c.setAttribute('r',r);glow.setAttribute('opacity','0.13')});
    ng.appendChild(g);
  });
  svg.appendChild(ng);
  container.appendChild(svg);
}
</script>
</body>
</html>"""


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def frontend() -> str:
    return _FRONTEND_HTML


@app.get("/health")
async def health() -> dict[str, Any]:
    """Enhanced liveness probe — includes per-sponsor status indicators."""
    from trace.agent.kalibr_guard import get_event_log
    from trace.agent.scheduler import scheduler_status
    from trace.mcp.server import _redis_store
    from trace.storage.tigris import get_tigris_store

    settings = get_settings()

    redis_ok = False
    if _redis_store.enabled:
        try:
            client = await _redis_store._ensure_client()
            redis_ok = client is not None
        except Exception:
            pass

    scalekit_ok = bool(
        settings.scalekit_env_url
        and settings.scalekit_client_id
        and settings.scalekit_client_secret
    )

    tigris = get_tigris_store()
    tigris_health = tigris.health()

    slack_delivery = "scalekit" if scalekit_ok else ("webhook" if settings.slack_webhook_url else "stub")
    recent_actions = get_event_log()[-10:]

    return {
        "status": "ok",
        "sponsors": {
            "anthropic": {
                "active": bool(settings.anthropic_api_key),
                "model": settings.anthropic_model,
                "indicator": "🟢" if settings.anthropic_api_key else "🔴",
            },
            "apify_mcp": {
                "active": bool(settings.apify_api_token),
                "indicator": "🟢" if settings.apify_api_token else "🟡",
                "note": "Dynamic Actor routing with 13 hint mappings",
            },
            "scalekit": {
                "mcp_auth": bool(settings.scalekit_mcp_resource_id),
                "connect": scalekit_ok,
                "token_vault": scalekit_ok,
                "indicator": "🟢" if scalekit_ok else "🟡",
                "note": "OAuth 2.1 + Token Vault for 5 service connections",
            },
            "redis": {
                "active": redis_ok,
                "enabled": _redis_store.enabled,
                "indicator": "🟢" if redis_ok else ("🟡" if _redis_store.enabled else "⚪"),
                "note": "Sub-50ms ZSET reads for curiosity graph",
            },
            "tigris_data": {
                **tigris_health,
                "indicator": "🟢" if tigris_health.get("status") == "ok" else ("🟡" if tigris.enabled else "⚪"),
                "note": "S3-compatible globally-distributed object storage for uploads + artifacts",
            },
            "kalibr": {
                "active": True,
                "indicator": "🟢",
                "recent_actions": len(recent_actions),
                "note": "Agent orchestration with failure detection + exponential backoff retry",
            },
            "slack_delivery": {
                "mode": slack_delivery,
                "indicator": "🟢" if slack_delivery != "stub" else "🟡",
                "note": f"Active delivery path: {slack_delivery}",
            },
        },
        "autonomous_loop": scheduler_status(),
        "capabilities": {
            "mcp_tools": 7,
            "pattern_detectors": 3,
            "tier_a_actions": ["notion", "calendar", "slack"],
            "tier_b_drafts": ["gmail_draft", "reddit_post"],
            "graph_algo": ["embeddings", "pagerank", "louvain"],
            "resilience": "kalibr_guard (3x retry + backoff)",
            "storage": "tigris_data (S3-compatible)",
        },
        "recent_kalibr_events": recent_actions,
    }


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

    # Mirror upload to Tigris Data (non-blocking best-effort — local save is authoritative).
    try:
        from trace.storage.tigris import get_tigris_store
        tigris = get_tigris_store()
        if tigris.enabled:
            tigris_uri = tigris.store_upload(upload_id, f"{detected_type}_{fname}", content)
            if tigris_uri:
                _log.info("[Tigris] Upload mirrored to %s", tigris_uri)
    except Exception as _tigris_exc:
        _log.debug("[Tigris] Upload mirror skipped: %s", _tigris_exc)

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
    try:
        from scalekit.client import LogoutUrlOptions  # type: ignore[import]
    except ImportError:
        return {"logout_url": post_logout_redirect_uri or "/", "note": "scalekit_not_installed"}

    try:
        client = _require_client()
    except Exception:
        return {"logout_url": post_logout_redirect_uri or "/", "note": "scalekit_not_configured"}

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
    from trace.auth.scalekit import get_scalekit_client
    if not link:
        if get_scalekit_client() is None:
            return {
                "status": "unconfigured",
                "message": "Scalekit is not configured — set SCALEKIT_ENV_URL, SCALEKIT_CLIENT_ID, SCALEKIT_CLIENT_SECRET",
            }
        return {
            "status": "connector_not_found",
            "message": f"Connector '{connection_name}' is not registered in your Scalekit workspace. "
                       f"Go to app.scalekit.com → Connect → Connectors → Add '{connection_name}'.",
        }
    return {"link": link, "status": "ok", "connection_name": connection_name}


@app.get("/graph.json")
async def graph_json(profile_id: str = "default") -> dict[str, Any]:
    """Return the curiosity graph as a D3 force-directed graph JSON.

    Format: {"nodes": [{"id": str, "score": float, "community": int, ...}],
              "links": [{"source": str, "target": str, "value": float}]}

    Falls back across profile IDs so both demo and real newsletter graphs work.
    """
    from trace.graph.graph_algo import enrich_graph
    from trace.mcp.server import _load_graph_async

    # Try the requested profile first, then fall back to checking PROFILE_CACHE
    graph = await _load_graph_async(profile_id)
    if (graph is None or graph.is_empty()) and profile_id != "default":
        graph = await _load_graph_async("default")
        if graph is not None and not graph.is_empty():
            profile_id = "default"

    # Also check in-memory cache directly (handles demo seed + newsletter runs)
    if graph is None or graph.is_empty():
        if profile_id in _PROFILE_CACHE:
            graph = _PROFILE_CACHE[profile_id]
        elif _PROFILE_CACHE:
            # Use most recently added profile
            graph = next(reversed(_PROFILE_CACHE.values()))
            profile_id = next(reversed(_PROFILE_CACHE.keys()))

    if graph is None or graph.is_empty():
        return {"nodes": [], "links": [], "profile_id": profile_id, "error": "no_graph"}

    enrichment = enrich_graph(graph)
    topic_map = {t.name: t for t in graph.topics}

    nodes = [
        {
            "id": t.name,
            "score": round(t.composite_score(), 4),
            "blended_score": round(enrichment.blended_scores.get(t.name, t.composite_score()), 4),
            "pagerank": round(enrichment.pagerank.get(t.name, 0.0), 4),
            "community": enrichment.communities.get(t.name, 0),
            "curiosity_type": t.curiosity_type.value,
            "frequency": t.frequency,
            "span_days": t.span_days(),
        }
        for t in graph.topics
    ]

    links = [
        {"source": a, "target": b, "value": round(w, 4)}
        for a, b, w in enrichment.edges
    ]

    return {
        "nodes": nodes,
        "links": links,
        "profile_id": profile_id,
        "stats": {
            "node_count": len(nodes),
            "link_count": len(links),
            "community_count": len(set(enrichment.communities.values())),
        },
    }


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


@app.post("/demo/run-detect")
async def demo_run_detect(request: Request) -> dict[str, Any]:
    """Immediately run the pattern detector and orchestrator on the demo graph.

    This is the on-demand equivalent of what the autonomous scheduler does every
    30 seconds in DEMO_MODE.  Calling this endpoint lets judges see the full
    agent loop — pattern detection → significance gate → Tier A/B action dispatch
    — without waiting for the scheduler interval.

    Body (JSON): {"profile_id": str}  (default: "demo")
    """
    from trace.agent.orchestrator import AgentOrchestrator
    from trace.agent.patterns import run_all_detectors
    from trace.agent.scheduler import _previous_graphs
    from trace.mcp.server import _load_graph_async, _pending_signals

    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass
    profile_id = str(body.get("profile_id", "demo")).strip() or "demo"

    graph = await _load_graph_async(profile_id)
    if graph is None or graph.is_empty():
        return {
            "status": "no_graph",
            "profile_id": profile_id,
            "hint": "Call POST /demo/seed first to populate the demo profile",
        }

    previous = _previous_graphs.get(profile_id)
    pending_count = len(_pending_signals.get(profile_id, []))

    events = run_all_detectors(
        current_graph=graph,
        previous_graph=previous,
        pending_signals_count=pending_count,
    )

    event_summaries = [
        {"pattern": e.pattern_type, "topic": e.topic_name, "score": round(e.score, 3)}
        for e in events
    ]

    actions: list[dict[str, Any]] = []
    if events:
        orchestrator = AgentOrchestrator()
        actions = await orchestrator.process_events(events, graph, profile_id)
        # Advance the snapshot so next call detects *new* changes (delta-based detection).
        _previous_graphs[profile_id] = graph

    _log.info(
        "demo/run-detect: profile=%s patterns=%d actions=%d",
        profile_id, len(events), len(actions),
    )
    return {
        "status": "ok",
        "profile_id": profile_id,
        "graph_topics": len(graph.topics),
        "patterns_detected": len(events),
        "patterns": event_summaries,
        "actions_dispatched": len(actions),
        "actions": actions,
        "pending_signals_consumed": pending_count,
    }


@app.post("/demo/seed")
async def demo_seed() -> dict[str, Any]:
    """Inject a realistic ML/robotics curiosity profile for hackathon demos.

    Populates profile_id='demo' with 8 topics:
      diffusion_policy, robot_learning, nuplan, karpathy_neural_networks,
      embodied_ai, rust_programming, pose_estimation, ai_safety

    After seeding, use profile_id='demo' in all MCP tool calls to see the
    full autonomous agent loop in action within 30 seconds (DEMO_MODE).
    """
    from trace.agent.demo_seed import inject_demo_seed
    return await inject_demo_seed()


@app.get("/demo/seed/status")
async def demo_seed_status() -> dict[str, Any]:
    """Check whether the demo seed profile is active."""
    from trace.agent.demo_seed import _SEED_PROFILE_ID
    from trace.delivery.api import _PROFILE_CACHE
    in_memory = _SEED_PROFILE_ID in _PROFILE_CACHE
    return {
        "active": in_memory,
        "profile_id": _SEED_PROFILE_ID,
        "topic_count": len(_PROFILE_CACHE[_SEED_PROFILE_ID].topics) if in_memory else 0,
    }


# ── Approvals endpoints (Tier B: Gmail drafts, Reddit posts) ──────────────────

@app.get("/approvals")
async def list_approvals(profile_id: str | None = None) -> dict[str, Any]:
    """List all pending Tier B actions awaiting user approval.

    If profile_id is omitted or 'all', returns pending actions from ALL profiles
    so the UI always shows the full queue regardless of which profile generated them.
    """
    from trace.agent.approvals import get_approvals_queue
    queue = get_approvals_queue()
    # When called with profile_id=demo but actions are under 'default' or vice versa,
    # return everything so judges see the full queue.
    if profile_id and profile_id != "all":
        pending = await queue.list_pending(profile_id)
        if not pending:
            # Fallback: return all profiles' pending items
            pending = await queue.list_pending(None)
    else:
        pending = await queue.list_pending(None)
    return {
        "pending": [a.to_dict() for a in pending],
        "count": len(pending),
        "profile_id": profile_id or "all",
    }


@app.post("/approvals/{action_id}/approve")
async def approve_action(action_id: str) -> dict[str, Any]:
    """Approve a pending Tier B action and execute it.

    gmail_draft  → calls gmail_create_draft via Scalekit (NEVER sends)
    reddit_post  → calls submit_approved_post via Scalekit (requires subreddit)
    """
    from trace.agent.approvals import get_approvals_queue
    queue = get_approvals_queue()
    action = await queue.approve(action_id)
    if action is None:
        raise HTTPException(status_code=404, detail="Action not found or already resolved")

    execution_result: dict[str, Any] = {"status": "no_executor"}

    if action.action_type == "gmail_draft":
        try:
            from trace.actions.gmail_draft import create_digest_draft
            payload = action.payload
            execution_result = await create_digest_draft(
                topic_name=payload.get("subject", action.title).replace("[Trace] Curiosity digest: ", "").replace("[Trace] Reading list: ", ""),
                briefing=payload.get("body", action.preview),
                profile_id=action.profile_id,
                source_urls=payload.get("source_urls", []),
                pattern_type=action.pattern_event_type,
            )
        except Exception as exc:
            execution_result = {"status": "error", "error": str(exc)}

    elif action.action_type == "reddit_post":
        try:
            from trace.actions.reddit_draft import submit_approved_post
            execution_result = await submit_approved_post(
                payload=action.payload,
                profile_id=action.profile_id,
            )
        except Exception as exc:
            execution_result = {"status": "error", "error": str(exc)}

    return {
        "status": "approved",
        "action": action.to_dict(),
        "execution": execution_result,
    }


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
            "get_topic_neighbors",
            "get_emerging_interests",
            "health",
        ],
        "auth": {
            "type": "oauth2",
            "provider": "Scalekit",
            "mcp_auth_enforced": bool(settings.scalekit_mcp_resource_id),
            "connect_link": f"{base}/auth/connect",
        },
        "sponsor_stack": {
            "anthropic": bool(settings.anthropic_api_key),
            "apify_mcp": bool(settings.apify_api_token),
            "scalekit_connect": bool(
                settings.scalekit_env_url
                and settings.scalekit_client_id
                and settings.scalekit_client_secret
            ),
            "tigris_data": bool(settings.tigris_access_key_id),
            "kalibr": True,
            "slack_webhook": bool(settings.slack_webhook_url),
            "redis": bool(settings.redis_url),
        },
    }
