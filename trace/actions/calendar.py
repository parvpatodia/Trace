"""
Calendar action — Tier A auto-execute via Scalekit Token Vault.

Creates a Google Calendar "deep-dive" event for an emerging topic.
The Calendar OAuth token lives in Scalekit's encrypted Vault.

EVENT STRUCTURE:
  Title:    "🧠 Deep dive: [topic]"
  Duration: 60 minutes
  Time:     Next available morning slot (9 AM, tomorrow)
  Notes:    briefing + pattern metadata

GRACEFUL DEGRADATION:
  - Scalekit not configured → stub response
  - Auth required → returns magic link
  - API failure → returns error dict, never raises
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

_log = logging.getLogger(__name__)

_CALENDAR_TOOL = "google_calendar_create_event"
_CONNECTION_NAME = "google_calendar"


def _next_morning_slot() -> tuple[str, str]:
    """Return ISO start/end strings for a 60-min slot at 9 AM tomorrow."""
    now = datetime.now(timezone.utc)
    tomorrow_9am = (now + timedelta(days=1)).replace(
        hour=9, minute=0, second=0, microsecond=0
    )
    tomorrow_10am = tomorrow_9am + timedelta(hours=1)
    return tomorrow_9am.isoformat(), tomorrow_10am.isoformat()


async def schedule_deep_dive(
    topic_name: str,
    briefing: str,
    profile_id: str = "default",
    pattern_type: str = "",
) -> dict[str, Any]:
    """Schedule a 60-minute deep-dive calendar block for an emerging topic.

    Never raises — all errors captured in response dict.
    """
    from trace.auth.scalekit import connect_execute_tool, connect_get_authorization_link, get_scalekit_client

    event_title = f"🧠 Deep dive: {topic_name.title()}"
    start_time, end_time = _next_morning_slot()

    description = (
        f"Trace Curiosity OS detected rising interest in '{topic_name}'.\n\n"
        f"{briefing[:800]}\n\n"
        f"Pattern: {pattern_type or 'emerging_interest'}\n"
        f"Scheduled by Trace automatically."
    )

    if get_scalekit_client() is None:
        _log.info("[Calendar] Scalekit not configured — stub for topic=%r", topic_name)
        return {
            "status": "stub",
            "reason": "scalekit_not_configured",
            "topic": topic_name,
            "event_title": event_title,
            "scheduled_for": start_time,
        }

    try:
        result = await connect_execute_tool(
            tool_name=_CALENDAR_TOOL,
            tool_input={
                "title": event_title,
                "start": start_time,
                "end": end_time,
                "description": description,
                "colorId": "9",  # blueberry — used for focus blocks
            },
            identifier=profile_id,
        )

        if "error" in result:
            err = str(result["error"])
            if "not_authorized" in err.lower() or "unauthorized" in err.lower():
                link = await connect_get_authorization_link(
                    identifier=profile_id,
                    connection_name=_CONNECTION_NAME,
                )
                return {
                    "status": "auth_required",
                    "connection": _CONNECTION_NAME,
                    "auth_link": link,
                    "topic": topic_name,
                }
            return {"status": "error", "error": err, "topic": topic_name}

        event_id = result.get("id") or result.get("event_id", "")
        event_url = result.get("htmlLink") or result.get("url", "")
        _log.info("[Calendar] Event created for topic=%r id=%s", topic_name, event_id)
        return {
            "status": "created",
            "event_id": event_id,
            "event_url": event_url,
            "event_title": event_title,
            "scheduled_for": start_time,
            "topic": topic_name,
            "via_scalekit": True,
        }

    except Exception as exc:
        _log.warning("[Calendar] schedule_deep_dive failed for %r: %s", topic_name, exc)
        return {"status": "error", "error": str(exc), "topic": topic_name}
