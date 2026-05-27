"""
Slack action — Tier A auto-execute via Scalekit Token Vault or direct webhook.

Posts a DM to the user (self-DM) or to a hard-coded #trace-demo channel.

DELIVERY PATHS (in priority order):
  1. Scalekit Token Vault — full OAuth, multi-tenant
  2. SLACK_WEBHOOK_URL env var — direct Incoming Webhook, zero-config demo path
  3. Stub response (logged visibly for demo panel)

MESSAGE FORMAT:
  Emerging interest: "🧠 Trace: New emerging interest — *topic*\n{briefing}"
  Bridge shift:      "🌉 Trace: Bridge shift — *topic* connects {related}\n{briefing}"
  Subscription debt: "📬 Trace: Curiosity debt — *topic* (unresolved for N days)"

SECURITY:
  - Only posts to the user's own DM or a pre-configured demo channel.
  - Channel ID is hard-coded to prevent injection attacks.
  - NEVER sends to arbitrary channels specified by agents.

GRACEFUL DEGRADATION:
  - Scalekit configured → use Token Vault
  - SLACK_WEBHOOK_URL set → direct webhook POST
  - Neither → stub response (visible in /health endpoint)
"""
from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

_SLACK_TOOL = "slack_send_message"

# Hard-coded demo channel — prevents injection. Override only via env.
_DEMO_CHANNEL = "#trace-demo"


async def _send_via_webhook(webhook_url: str, message: str, channel: str) -> dict[str, Any]:
    """POST to a Slack Incoming Webhook URL directly."""
    import httpx

    payload = {
        "text": message[:2000],
        "username": "Trace Curiosity OS",
        "icon_emoji": ":brain:",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(webhook_url, json=payload)
            if resp.status_code == 200 and resp.text == "ok":
                _log.info("[Slack] Webhook message sent to %s", channel)
                return {
                    "status": "sent",
                    "channel": channel,
                    "via": "webhook",
                    "message_preview": message[:80],
                }
            return {
                "status": "error",
                "error": f"webhook returned {resp.status_code}: {resp.text}",
                "channel": channel,
            }
    except Exception as exc:
        _log.warning("[Slack] Webhook POST failed: %s", exc)
        return {"status": "error", "error": str(exc), "channel": channel}


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
    from trace.config import get_settings

    settings = get_settings()
    connection_name = settings.scalekit_slack_connection

    # Whitelist channels — security boundary.
    allowed = {None, "#trace-demo", "self", _DEMO_CHANNEL}
    target_channel = channel if channel in allowed else _DEMO_CHANNEL

    if get_scalekit_client() is None:
        # Fall back to direct Incoming Webhook if configured.
        if settings.slack_webhook_url:
            return await _send_via_webhook(settings.slack_webhook_url, message, target_channel or _DEMO_CHANNEL)

        _log.info("[Slack] No delivery path configured — stub for channel=%r", target_channel)
        return {
            "status": "stub",
            "reason": "no_delivery_path",
            "hint": "Set SLACK_WEBHOOK_URL or configure Scalekit Connect",
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
            connection_name=connection_name,
        )

        if "error" in result:
            err = str(result["error"])
            if "not_authorized" in err.lower() or "not_in_channel" in err.lower():
                link = await connect_get_authorization_link(
                    identifier=profile_id,
                    connection_name=connection_name,
                )
                return {
                    "status": "auth_required",
                    "connection": connection_name,
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
