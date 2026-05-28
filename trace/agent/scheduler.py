"""
APScheduler-based autonomous agent loop.

THREE RECURRING JOBS:

  poll_gmail_signals   — collect Gmail signals via Scalekit Token Vault
                         Production: every 15 min  |  DEMO_MODE: every 30 s
  dispatch_apify_scrape — scrape fresh articles for top curiosity topics
                         Production: every 6 h     |  DEMO_MODE: every 2 min
  detect_and_act       — run pattern detectors + orchestrator on latest graph
                         Production: every 5 min   |  DEMO_MODE: every 30 s

DEMO_MODE:
  Set TRACE_DEMO_MODE=true in env to use compressed intervals.
  Designed for live hackathon demos — shows autonomous behavior in <5 min.

GRACEFUL DEGRADATION:
  - APScheduler not installed → scheduler silently disabled (server still starts)
  - Redis unavailable → uses in-memory graph from api.py _PROFILE_CACHE
  - Anthropic unavailable → pattern detection runs without significance gating
  - All signals → structured log entries so judges can see the loop running
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

_log = logging.getLogger(__name__)

# ── APScheduler availability guard ────────────────────────────────────────────
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import]
    from apscheduler.triggers.interval import IntervalTrigger  # type: ignore[import]
    _APSCHEDULER_AVAILABLE = True
except ModuleNotFoundError:
    _APSCHEDULER_AVAILABLE = False
    AsyncIOScheduler = None  # type: ignore[assignment,misc]
    IntervalTrigger = None  # type: ignore[assignment]

_DEMO_MODE = os.getenv("TRACE_DEMO_MODE", "false").lower() in ("true", "1", "yes")

# Job intervals (seconds).
_GMAIL_INTERVAL = 30 if _DEMO_MODE else 900      # 30 s demo | 15 min prod
_APIFY_INTERVAL = 120 if _DEMO_MODE else 21600   # 2 min demo | 6 hr prod
_PATTERN_INTERVAL = 30 if _DEMO_MODE else 300    # 30 s demo | 5 min prod

_log.info(
    "Agent scheduler intervals: DEMO_MODE=%s gmail=%ds apify=%ds patterns=%ds",
    _DEMO_MODE, _GMAIL_INTERVAL, _APIFY_INTERVAL, _PATTERN_INTERVAL,
)

# Rotate through varied demo signals so the loop looks live, not looped.
_DEMO_GMAIL_SIGNALS = [
    "AI alignment weekly: RLHF vs constitutional AI trade-offs",
    "diffusion models for robot manipulation — new paper from Berkeley",
    "nuplan benchmark: closed-loop planning leaderboard updated",
    "Karpathy on minGPT v2: simplicity as a design constraint",
    "embodied AI survey: progress in sim-to-real transfer 2024",
    "Rust async book updated — new chapter on structured concurrency",
    "pose estimation in the wild — ECCV 2024 best paper nominee",
]
_demo_gmail_idx = 0


# ── Graph snapshot store for diff-based pattern detection ────────────────────
# Keyed by profile_id.  Previous graph retained so detectors can compute deltas.
_previous_graphs: dict[str, Any] = {}

_DEFAULT_PROFILE = "demo" if _DEMO_MODE else "default"


def _boost_graph_with_signals(graph: Any, signals: list[Any]) -> Any:
    """Apply pending signals to the graph by boosting matched topic recency scores.

    For each pending signal, keyword-matches against existing topic names and
    bumps their recency_score by a small amount (capped at 1.0). Topics not
    matched by any signal are unchanged.  Runs in O(topics × signals) — fast
    enough for the 30-second scheduler tick with typical demo-scale data.

    Returns a new CuriosityGraph (frozen model — original is never mutated).
    """
    if not signals or graph is None or graph.is_empty():
        return graph

    try:
        from trace.models import CuriosityGraph

        signal_texts = [s.content.lower() for s in signals if hasattr(s, "content")]
        if not signal_texts:
            return graph

        boosted_topics = []
        for topic in graph.topics:
            name_lower = topic.name.lower()
            name_words = set(name_lower.split())
            # Check if any signal text mentions this topic
            match_weight = 0.0
            for text in signal_texts:
                if name_lower in text or any(w in text for w in name_words if len(w) > 3):
                    match_weight += 0.04  # +4% recency per matching signal
            if match_weight > 0:
                new_score = min(1.0, topic.recency_score + match_weight)
                boosted_topics.append(topic.model_copy(update={"recency_score": new_score}))
            else:
                boosted_topics.append(topic)

        _log.info(
            "[scheduler] signal boost: %d signal(s) applied to graph (%d topics)",
            len(signals), len(boosted_topics),
        )
        return CuriosityGraph(
            topics=tuple(boosted_topics),
            signal_count=graph.signal_count + len(signals),
            source_breakdown=graph.source_breakdown,
        )
    except Exception as exc:
        _log.warning("[scheduler] _boost_graph_with_signals failed: %s", exc)
        return graph


# ── Job: Poll Gmail signals ───────────────────────────────────────────────────

async def _job_poll_gmail() -> None:
    """Collect new Gmail signals via Scalekit Token Vault and add to pending."""
    _log.info("[scheduler] poll_gmail_signals triggered at %s", datetime.now(timezone.utc).isoformat())
    try:
        from trace.mcp.server import _pending_signals
        from trace.models import RawSignal, SignalSource
        import uuid

        # Import GmailCollector lazily — only available when Gmail connection is wired.
        try:
            from trace.signals.gmail import GmailCollector
            collector = GmailCollector()
            signals = await collector.collect()
            bucket = _pending_signals.setdefault(_DEFAULT_PROFILE, [])
            bucket.extend(signals)
            _log.info("[scheduler] poll_gmail_signals: +%d signals (total pending=%d)", len(signals), len(bucket))
        except (ImportError, TypeError, AttributeError):
            # Gmail collector not yet built — Phase 2.  Synthesize varied stub signals in DEMO_MODE.
            if _DEMO_MODE:
                global _demo_gmail_idx
                content = "[demo] " + _DEMO_GMAIL_SIGNALS[_demo_gmail_idx % len(_DEMO_GMAIL_SIGNALS)]
                _demo_gmail_idx += 1
                stub = RawSignal(
                    id=str(uuid.uuid4()),
                    source=SignalSource.GMAIL,
                    content=content,
                    timestamp=datetime.now(timezone.utc),
                    metadata={"demo": True},
                )
                bucket = _pending_signals.setdefault(_DEFAULT_PROFILE, [])
                bucket.append(stub)
                _log.info("[scheduler] poll_gmail_signals (DEMO stub): %r pending=%d", content[:60], len(bucket))
        except Exception as exc:
            _log.warning("[scheduler] GmailCollector failed: %s", exc)
    except Exception as exc:
        _log.error("[scheduler] poll_gmail_signals job error: %s", exc)


# ── Job: Dispatch Apify scrape ────────────────────────────────────────────────

async def _job_dispatch_apify() -> None:
    """Scrape fresh articles for the top curiosity topics and push to Redis."""
    _log.info("[scheduler] dispatch_apify_scrape triggered at %s", datetime.now(timezone.utc).isoformat())
    try:
        from trace.mcp.server import _load_graph_async, _redis_store
        from trace.mcp.apify_client import ApifyMCPScraper, _pick_actor_for_topic
        from trace.config import get_settings

        s = get_settings()
        graph = await _load_graph_async(_DEFAULT_PROFILE)
        if graph is None or graph.is_empty():
            _log.info("[scheduler] dispatch_apify_scrape: no graph yet — skipping")
            return

        top_topics = graph.top_n(3)
        if not top_topics:
            return

        if not s.apify_api_token:
            _log.info("[scheduler] dispatch_apify_scrape: no Apify token — logging topics only")
            for t in top_topics:
                _log.info("[scheduler] [Apify stub] would scrape: %r (actor=%s)", t.name, _pick_actor_for_topic(t.name))
            return

        scraper = ApifyMCPScraper(
            api_token=s.apify_api_token,
            scalekit_connection_name=(
                s.scalekit_apify_connection_name
                if (s.scalekit_env_url and s.scalekit_client_id and s.scalekit_client_secret)
                else None
            ),
            scalekit_identifier=s.scalekit_default_identifier,
        )

        import asyncio
        results = await asyncio.gather(
            *[scraper.scrape(t, max_results=3) for t in top_topics],
            return_exceptions=True,
        )
        total_articles = sum(len(r) for r in results if isinstance(r, list))
        _log.info("[scheduler] dispatch_apify_scrape: fetched %d articles for %d topics", total_articles, len(top_topics))

    except Exception as exc:
        _log.error("[scheduler] dispatch_apify_scrape job error: %s", exc)


# ── Job: Detect patterns and act ─────────────────────────────────────────────

async def _job_detect_and_act() -> None:
    """Run pattern detectors on the latest curiosity graph, then dispatch actions."""
    _log.info("[scheduler] detect_and_act triggered at %s", datetime.now(timezone.utc).isoformat())
    try:
        from trace.mcp.server import _load_graph_async, _pending_signals
        from trace.agent.patterns import run_all_detectors
        from trace.agent.orchestrator import AgentOrchestrator

        graph = await _load_graph_async(_DEFAULT_PROFILE)
        if graph is None or graph.is_empty():
            _log.info("[scheduler] detect_and_act: no graph yet — skipping")
            return

        # Consume pending signals and apply them to the graph so recency scores
        # reflect agent observations collected since the last full pipeline run.
        pending = _pending_signals.pop(_DEFAULT_PROFILE, [])
        if pending:
            graph = _boost_graph_with_signals(graph, pending)

        previous = _previous_graphs.get(_DEFAULT_PROFILE)
        pending_count = len(pending)

        events = run_all_detectors(
            current_graph=graph,
            previous_graph=previous,
            pending_signals_count=pending_count,
        )

        if events:
            orchestrator = AgentOrchestrator()
            actions = await orchestrator.process_events(events, graph, _DEFAULT_PROFILE)
            _log.info(
                "[scheduler] detect_and_act: %d patterns → %d actions dispatched",
                len(events), len(actions),
            )
        else:
            _log.info("[scheduler] detect_and_act: no significant patterns found")

        # Store current as previous for next cycle.
        _previous_graphs[_DEFAULT_PROFILE] = graph

    except Exception as exc:
        _log.error("[scheduler] detect_and_act job error: %s", exc)


# ── Scheduler lifecycle ────────────────────────────────────────────────────────

_scheduler: Any | None = None


def build_scheduler() -> Any | None:
    """Construct and return an AsyncIOScheduler, or None if APScheduler is not installed."""
    global _scheduler

    if not _APSCHEDULER_AVAILABLE:
        _log.warning(
            "APScheduler not installed — autonomous agent loop disabled. "
            "Install apscheduler to enable: pip install apscheduler"
        )
        return None

    _scheduler = AsyncIOScheduler(timezone="UTC")

    _scheduler.add_job(
        _job_poll_gmail,
        trigger=IntervalTrigger(seconds=_GMAIL_INTERVAL),
        id="poll_gmail_signals",
        name="Poll Gmail signals",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=_GMAIL_INTERVAL // 2,
    )

    _scheduler.add_job(
        _job_dispatch_apify,
        trigger=IntervalTrigger(seconds=_APIFY_INTERVAL),
        id="dispatch_apify_scrape",
        name="Dispatch Apify scrape",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=_APIFY_INTERVAL // 2,
    )

    _scheduler.add_job(
        _job_detect_and_act,
        trigger=IntervalTrigger(seconds=_PATTERN_INTERVAL),
        id="detect_and_act",
        name="Detect patterns and act",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=_PATTERN_INTERVAL // 2,
    )

    _log.info(
        "Agent scheduler built: DEMO_MODE=%s | jobs: %s",
        _DEMO_MODE,
        [j.id for j in _scheduler.get_jobs()],
    )
    return _scheduler


def get_scheduler() -> Any | None:
    return _scheduler


def scheduler_status() -> dict[str, Any]:
    """Return a JSON-serialisable status dict for the /health endpoint."""
    if not _APSCHEDULER_AVAILABLE:
        return {"enabled": False, "reason": "apscheduler_not_installed"}
    if _scheduler is None:
        return {"enabled": False, "reason": "not_started"}
    jobs = []
    for job in _scheduler.get_jobs():
        next_run = job.next_run_time
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": next_run.isoformat() if next_run else None,
        })
    return {
        "enabled": True,
        "running": _scheduler.running,
        "demo_mode": _DEMO_MODE,
        "jobs": jobs,
        "intervals": {
            "poll_gmail_s": _GMAIL_INTERVAL,
            "dispatch_apify_s": _APIFY_INTERVAL,
            "detect_and_act_s": _PATTERN_INTERVAL,
        },
    }
