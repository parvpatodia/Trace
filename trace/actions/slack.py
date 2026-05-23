"""
Slack action — Tier A auto-execute via Scalekit Token Vault.

Posts a DM to the user (self-DM) or to a hard-coded #trace-demo channel.
The Slack OAuth token lives in Scalekit's encrypted Vault.

MESSAGE FORMAT:
  Emerging interest: "🧠 Trace: New emerging interest — *topic*\n{briefing}"
  Bridge shift:      "🌉 Trace: Bridge shift — *topic* connects {related}\n{briefing}"
  Subscription debt: "📬 Trace: Curiosity debt — *topic* (unresolved for N days)"

SECURITY:
  - Only posts to the user's own DM or a pre-configured demo channel.
  - Channel ID is hard-coded to prevent injection attacks.
  - NEVER sends to arbitrary channels specified by agents.

GRACEFUL DEGRADATION:
  - Scalekit not configured → stub response
  - Auth required → returns magic link
  - API failure → error dict, never raises
"""
from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

_SLACK_TOOL = "slack_send_message"
_CONNECTION_NAME = "slack"

# Hard-coded demo channel — prevents injection. Override only via env.
_DEMO_CHANNEL = "#trace-demo"


async def send_pattern_alert(
    message: str,
    profile_id: str = "default",
    channel: str | None = None,
) -> dict[str, Any]:
    """Post a pattern alert to Slack.

    channel: If None, sends to _DEMO_CHANNEL. Only accepts '#trace-demo' or 'self'
             to prevent agents from directing messages to arbitrary channels.
    Never raises — all errors captured in response dict.
    """
    from trace.auth.scalekit import connect_execute_tool, connect_get_authorization_link, get_scalekit_client

    # Whitelist channels — security boundary.
    allowed = {None, "#trace-demo", "self", _DEMO_CHANNEL}
    target_channel = channel if channel in allowed else _DEMO_CHANNEL

    if get_scalekit_client() is None:
        _log.info("[Slack] Scalekit not configured — stub for channel=%r", target_channel)
        return {
            "status": "stub",
            "reason": "scalekit_not_configured",
            "channel": target_channel,
            "message_preview": message[:80],
        }

    try:
        result = await connect_execute_tool(
            tool_name=_SLACK_TOOL,
            tool_input={
                "channel": target_channel or _DEMO_CHANNEL,
                "text": message[:2000],  # Slack message limit.
                "username": "Trace Curiosity OS",
                "icon_emoji": ":brain:",
            },
            identifier=profile_id,
        )

        if "error" in result:
            err = str(result["error"])
            if "not_authorized" in err.lower() or "not_in_channel" in err.lower():
                link = await connect_get_authorization_link(
                    identifier=profile_id,
                    connection_name=_CONNECTION_NAME,
                )
                return {
                    "status": "auth_required",
                    "connection": _CONNECTION_NAME,
                    "auth_link": link,
                    "channel": target_channel,
                }
            return {"status": "error", "error": err, "channel": target_channel}

        ts = result.get("ts") or result.get("message", {}).get("ts", "")
        _log.info("[Slack] Message sent to %s ts=%s", target_channel, ts)
        return {
            "status": "sent",
            "channel": target_channel,
            "ts": ts,
            "message_preview": message[:80],
            "via_scalekit": True,
        }

    except Exception as exc:
        _log.warning("[Slack] send_pattern_alert failed: %s", exc)
        return {"status": "error", "error": str(exc), "channel": target_channel}
