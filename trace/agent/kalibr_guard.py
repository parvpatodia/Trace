"""
Kalibr — agent orchestration layer with failure detection, routing, and recovery.

Uses the official Kalibr Python SDK (pip install kalibr) when credentials are
configured. Falls back to the in-process retry layer when they are not.

WHAT THE REAL KALIBR SDK ADDS:
  - Auto-instrumentation of every Anthropic API call (zero code changes needed).
    Just importing this module instruments all Claude calls with Kalibr telemetry.
  - Thompson Sampling router: routes each goal to the model+path that is actually
    succeeding in production, learning over time.
  - Self-healing loop: detects structural failures (bad JSON, truncated output),
    classifies root cause via LLM judge, auto-repairs prompts or swaps models.
  - Outcome reporting: `router.report(success=bool)` feeds back to the bandit.

INTEGRATION:
  - `_action_router`: guards action dispatch (Notion, Gmail, Slack, Calendar).
    Reports success when the action returns status in {created, sent, ok}.
  - `_compose_router`: used by NewsletterComposer to validate JSON output and
    trigger self-healing on truncation or malformed responses.
  - Auto-instrumentation fires on module import — no API key needed for tracing.

GRACEFUL DEGRADATION:
  If KALIBR_API_KEY / KALIBR_TENANT_ID are not set, Router init fails and we
  fall back to the in-process exponential-backoff retry. All telemetry continues
  to be logged locally via the event log.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from typing import Any, Awaitable, Callable

_log = logging.getLogger(__name__)

# ── Kalibr SDK bootstrap ─────────────────────────────────────────────────────
# Importing kalibr auto-instruments the Anthropic SDK. This happens even without
# an API key — all Claude calls are traced to /tmp/kalibr_otel_spans.jsonl.
_KALIBR_SDK_AVAILABLE = False
_action_router: Any = None
_compose_router: Any = None

try:
    import kalibr  # noqa: F401 — side effect: instruments Anthropic
    from kalibr import Router

    _KALIBR_SDK_AVAILABLE = True

    _action_router = Router(
        goal="agent_action_dispatch",
        paths=[os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")],
        success_when=lambda out: isinstance(out, dict) and out.get("status") in (
            "created", "sent", "ok", "draft_created"
        ),
    )
    _compose_router = Router(
        goal="newsletter_compose",
        paths=[os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")],
        success_when=lambda out: '"subject_line"' in out and '"sections"' in out,
    )
    _log.info("[Kalibr] SDK v%s loaded — action + compose routers active",
              getattr(kalibr, "__version__", "?"))

except Exception as exc:
    _log.info("[Kalibr] SDK not fully configured (%s) — using local retry fallback. "
              "Set KALIBR_API_KEY + KALIBR_TENANT_ID for full routing.", str(exc)[:120])


def get_action_router() -> Any:
    """Return the Kalibr action Router, or None if SDK not configured."""
    return _action_router


def get_compose_router() -> Any:
    """Return the Kalibr compose Router, or None if SDK not configured."""
    return _compose_router


def kalibr_sdk_active() -> bool:
    return _KALIBR_SDK_AVAILABLE and _action_router is not None


# ── In-memory event log ───────────────────────────────────────────────────────
# Last 200 action events surfaced on /health and the MCP health tool.
_event_log: deque[dict[str, Any]] = deque(maxlen=200)


def get_event_log() -> list[dict[str, Any]]:
    return list(_event_log)


def _record(action: str, topic: str, status: str, attempt: int, detail: str = "") -> None:
    _event_log.append({
        "ts": time.time(),
        "action": action,
        "topic": topic,
        "status": status,
        "attempt": attempt,
        "detail": detail,
    })


# ── Guard execution ───────────────────────────────────────────────────────────

async def execute_with_guard(
    action_fn: Callable[..., Awaitable[dict[str, Any]]],
    action_name: str,
    topic: str,
    *args: Any,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute an async action with Kalibr failure detection, retry, and outcome reporting.

    When the Kalibr SDK is configured:
      - Reports success/failure to the Kalibr bandit after each execution.
      - The Router learns which action paths succeed most reliably over time.

    When not configured:
      - Falls back to local exponential-backoff retry (unchanged behaviour).

    Args:
        action_fn:    Async callable to guard (e.g. notion.create_topic_page).
        action_name:  Human-readable name for logging and Kalibr goal grouping.
        topic:        Topic name — used for event log and Kalibr metadata.
        max_attempts: Max retry attempts before giving up.
        base_delay:   Exponential backoff base in seconds.
    """
    last_result: dict[str, Any] = {}
    for attempt in range(1, max_attempts + 1):
        try:
            result = await action_fn(*args, **kwargs)
            status = result.get("status", "unknown")

            # Stub / auth_required — not retryable, surface immediately.
            if status in ("stub", "auth_required"):
                _record(action_name, topic, status, attempt, result.get("reason", ""))
                if _action_router:
                    try:
                        _action_router.report(success=False, reason=status)
                    except Exception:
                        pass
                return result

            if status in ("created", "sent", "ok", "draft_created"):
                _record(action_name, topic, "success", attempt)
                if _action_router:
                    try:
                        _action_router.report(success=True)
                    except Exception:
                        pass
                return result

            # Error status — may be retryable.
            err_detail = result.get("error", str(result))
            _record(action_name, topic, "error", attempt, err_detail[:120])
            last_result = result

            if _is_transient(err_detail) and attempt < max_attempts:
                delay = base_delay * (2 ** (attempt - 1))
                _log.warning(
                    "[Kalibr] %s for topic=%r attempt=%d/%d — retrying in %.1fs: %s",
                    action_name, topic, attempt, max_attempts, delay, err_detail[:80],
                )
                await asyncio.sleep(delay)
                continue

            _log.warning(
                "[Kalibr] %s for topic=%r failed after %d attempt(s): %s",
                action_name, topic, attempt, err_detail[:120],
            )
            if _action_router:
                try:
                    _action_router.report(success=False, reason=err_detail[:80])
                except Exception:
                    pass
            return result

        except Exception as exc:
            err_detail = str(exc)
            _record(action_name, topic, "exception", attempt, err_detail[:120])
            if attempt < max_attempts:
                delay = base_delay * (2 ** (attempt - 1))
                _log.warning(
                    "[Kalibr] %s exception attempt=%d/%d — retrying in %.1fs: %s",
                    action_name, topic, attempt, max_attempts, delay, err_detail[:80],
                )
                await asyncio.sleep(delay)
                last_result = {"status": "error", "error": err_detail}
            else:
                _log.error(
                    "[Kalibr] %s for topic=%r exhausted retries: %s",
                    action_name, topic, err_detail,
                )
                if _action_router:
                    try:
                        _action_router.report(success=False, reason=err_detail[:80])
                    except Exception:
                        pass
                return {"status": "error", "error": err_detail, "attempts": max_attempts}

    return last_result or {"status": "error", "error": "unknown_failure"}


def _is_transient(error_str: str) -> bool:
    transient_markers = (
        "timeout", "connection", "rate_limit", "429", "503", "502",
        "temporarily", "try again", "network", "socket",
    )
    lower = error_str.lower()
    return any(m in lower for m in transient_markers)
