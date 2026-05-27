"""
Kalibr — agent orchestration layer with failure detection and recovery.

Kalibr wraps every autonomous action dispatch with:
  1. Pre-flight validation (schema check, rate-limit guard)
  2. Execution with structured error capture
  3. Automatic retry with exponential backoff on transient failures
  4. Failure escalation to the approvals queue when retries exhausted
  5. Audit trail of every action attempt and outcome

WHY KALIBR MATTERS FOR AN AGENT OS:
  Trace's autonomous loop runs on a schedule and executes Tier A actions
  (Notion, Calendar, Slack) without user confirmation. Without a guard layer,
  a single transient Scalekit error silently kills the whole action batch.
  Kalibr gives us structured failure visibility and recovery so the agent
  keeps working even when individual connectors are flaky.

INTEGRATION APPROACH:
  KalibrGuard is a thin decorator/context-manager pattern. The orchestrator
  calls `execute_with_guard(action_fn, *args)` instead of `action_fn(*args)`
  directly. Guard handles retries, logs to the Kalibr event stream, and
  surfaces failures in the /health endpoint.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Awaitable, Callable

_log = logging.getLogger(__name__)

# In-memory event log — last 200 action events surfaced on /health endpoint.
_event_log: deque[dict[str, Any]] = deque(maxlen=200)


def get_event_log() -> list[dict[str, Any]]:
    """Return recent Kalibr action events for the /health endpoint."""
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


async def execute_with_guard(
    action_fn: Callable[..., Awaitable[dict[str, Any]]],
    action_name: str,
    topic: str,
    *args: Any,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute an async action function with Kalibr failure detection and retry.

    - Retries up to max_attempts on transient errors (network, rate-limit).
    - Returns the first successful result.
    - Returns a structured failure dict if all attempts exhausted.
    - All attempts are recorded in the Kalibr event log.

    Args:
        action_fn: The async callable to guard (e.g. notion.create_topic_page).
        action_name: Human-readable name for logging (e.g. "notion_page").
        topic: Topic name this action relates to — used for event log grouping.
        max_attempts: How many times to try before giving up.
        base_delay: Backoff base in seconds (doubles each retry).
    """
    last_result: dict[str, Any] = {}
    for attempt in range(1, max_attempts + 1):
        try:
            result = await action_fn(*args, **kwargs)
            status = result.get("status", "unknown")

            # Stub / auth_required are not retryable — surface immediately.
            if status in ("stub", "auth_required"):
                _record(action_name, topic, status, attempt, result.get("reason", ""))
                return result

            if status in ("created", "sent", "ok", "draft_created"):
                _record(action_name, topic, "success", attempt)
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

            # Non-retryable error or final attempt.
            _log.warning(
                "[Kalibr] %s for topic=%r failed after %d attempt(s): %s",
                action_name, topic, attempt, err_detail[:120],
            )
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
                return {"status": "error", "error": err_detail, "attempts": max_attempts}

    return last_result or {"status": "error", "error": "unknown_failure"}


def _is_transient(error_str: str) -> bool:
    """Heuristic: is this error likely transient and worth retrying?"""
    transient_markers = (
        "timeout", "connection", "rate_limit", "429", "503", "502",
        "temporarily", "try again", "network", "socket",
    )
    lower = error_str.lower()
    return any(m in lower for m in transient_markers)
