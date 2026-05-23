"""
Gmail draft action — Tier B, requires user approval before any send.

SECURITY CONTRACT:
  This module creates Gmail DRAFTS only via the Gmail drafts.create API.
  It NEVER calls drafts.send or messages.send under ANY circumstances.
  The Tier B approval gate in approvals.py ensures a human clicks "Approve"
  before this code executes.

  The Gmail OAuth token lives in Scalekit's encrypted Token Vault — it is
  never stored in env vars, application memory, or logs.

DRAFT STRUCTURE:
  Subject:  "[Trace] Curiosity digest: {topic}"
  Body:     Personalized briefing + source links + unsubscribe instruction
  To:       Empty by default — user fills in recipient before sending
  From:     User's own Gmail address (retrieved from connected account)

GRACEFUL DEGRADATION:
  - Scalekit not configured → stub (no draft created)
  - Auth required → returns magic link
  - API failure → error dict, never raises
"""
from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

_GMAIL_DRAFT_TOOL = "gmail_create_draft"


def _build_draft_body(topic_name: str, briefing: str, source_urls: list[str]) -> str:
    """Build the plain-text email body for the digest draft."""
    sources_section = ""
    if source_urls:
        links = "\n".join(f"  • {url}" for url in source_urls[:5])
        sources_section = f"\n\n📎 Sources:\n{links}"

    return (
        f"Hi,\n\n"
        f"Trace detected sustained curiosity about: {topic_name.title()}\n\n"
        f"Here's a curated briefing:\n\n"
        f"{briefing}"
        f"{sources_section}\n\n"
        f"---\n"
        f"This draft was prepared by Trace Curiosity OS.\n"
        f"You approved this draft before it was created. Edit freely before sending.\n"
        f"To stop receiving these drafts, disconnect Gmail from Trace."
    )


async def create_digest_draft(
    topic_name: str,
    briefing: str,
    profile_id: str = "default",
    source_urls: list[str] | None = None,
    pattern_type: str = "",
) -> dict[str, Any]:
    """Create a Gmail draft for the subscription-debt digest.

    NEVER sends the email — only creates a draft in the user's Gmail Drafts folder.
    User must open Gmail and manually send (or Trace shows an approve button).

    Returns: {"status": "created", "draft_id": str, "subject": str, ...}
    """
    from trace.auth.scalekit import connect_execute_tool, connect_get_authorization_link, get_scalekit_client
    from trace.config import get_settings

    connection_name = get_settings().scalekit_gmail_connection

    subject = f"[Trace] Curiosity digest: {topic_name.title()}"
    body = _build_draft_body(topic_name, briefing, source_urls or [])

    if get_scalekit_client() is None:
        _log.info("[GmailDraft] Scalekit not configured — stub for topic=%r", topic_name)
        return {
            "status": "stub",
            "reason": "scalekit_not_configured",
            "topic": topic_name,
            "subject": subject,
            "body_preview": body[:200],
        }

    try:
        result = await connect_execute_tool(
            tool_name=_GMAIL_DRAFT_TOOL,
            tool_input={
                "subject": subject,
                "body": body,
                "to": "",  # User fills in recipient — intentionally blank.
                "bodyType": "text/plain",
            },
            identifier=profile_id,
            connection_name=connection_name,
        )

        if "error" in result:
            err = str(result["error"])
            if "not_authorized" in err.lower() or "unauthorized" in err.lower():
                link = await connect_get_authorization_link(
                    identifier=profile_id,
                    connection_name=connection_name,
                )
                return {
                    "status": "auth_required",
                    "connection": connection_name,
                    "auth_link": link,
                    "topic": topic_name,
                }
            return {"status": "error", "error": err, "topic": topic_name}

        draft_id = result.get("id") or result.get("draft_id", "")
        _log.info(
            "[GmailDraft] Draft created for topic=%r draft_id=%s  (NOT sent)",
            topic_name, draft_id,
        )
        return {
            "status": "created",
            "draft_id": draft_id,
            "subject": subject,
            "body_preview": body[:200],
            "topic": topic_name,
            "via_scalekit": True,
            "note": "Draft created — not sent. User must approve and send manually.",
        }

    except Exception as exc:
        _log.warning("[GmailDraft] create_digest_draft failed for %r: %s", topic_name, exc)
        return {"status": "error", "error": str(exc), "topic": topic_name}
