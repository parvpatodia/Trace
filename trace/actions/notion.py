"""
Notion action — Tier A auto-execute via Scalekit Token Vault.

Creates a Notion page for an emerging topic using connect_execute_tool.
The Notion OAuth token lives in Scalekit's encrypted Vault — never in env vars.

PAGE STRUCTURE:
  Title:   "📚 [topic name]"
  Body:    briefing + source links + Trace metadata block
  Tags:    curiosity_type, pattern_that_triggered, built_at

GRACEFUL DEGRADATION:
  - Scalekit not configured → logs action, returns stub response
  - Notion tool call fails → logs warning, returns error response
  - Connection not yet authorized → returns auth_required response with magic link
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

_log = logging.getLogger(__name__)

_NOTION_TOOL = "notion_create_page"
_CONNECTION_NAME = "notion"


async def create_topic_page(
    topic_name: str,
    briefing: str,
    profile_id: str = "default",
    source_urls: list[str] | None = None,
    pattern_type: str = "",
    curiosity_type: str = "",
) -> dict[str, Any]:
    """Create a Notion page for an emerging topic.

    Returns a dict with status, page_url (if created), and metadata.
    Never raises — errors are captured and returned in the response dict.
    """
    from trace.auth.scalekit import connect_execute_tool, connect_get_authorization_link, get_scalekit_client
    from trace.config import get_settings

    s = get_settings()

    # Build page content.
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sources_md = ""
    if source_urls:
        sources_md = "\n\n**Sources:**\n" + "\n".join(f"- {url}" for url in source_urls[:5])

    page_title = f"📚 {topic_name.title()}"
    page_body = (
        f"# {page_title}\n\n"
        f"*Created by Trace Curiosity OS on {now_str}*\n\n"
        f"**Pattern:** {pattern_type or 'emerging_interest'}  \n"
        f"**Curiosity type:** {curiosity_type or 'unknown'}  \n"
        f"**Profile:** {profile_id}\n\n"
        f"---\n\n"
        f"{briefing}"
        f"{sources_md}"
    )

    if get_scalekit_client() is None:
        _log.info("[Notion] Scalekit not configured — stub response for topic=%r", topic_name)
        return {
            "status": "stub",
            "reason": "scalekit_not_configured",
            "topic": topic_name,
            "page_title": page_title,
        }

    try:
        result = await connect_execute_tool(
            tool_name=_NOTION_TOOL,
            tool_input={
                "title": page_title,
                "content": page_body,
                "properties": {
                    "Topic": topic_name,
                    "Pattern": pattern_type,
                    "Profile": profile_id,
                    "CreatedBy": "Trace Curiosity OS",
                },
            },
            identifier=profile_id,
        )

        if "error" in result:
            err = str(result["error"])
            # Detect auth-required error and return magic link.
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

        page_url = (
            result.get("url")
            or result.get("page_url")
            or result.get("id", "")
        )
        _log.info("[Notion] Created page for topic=%r url=%s", topic_name, page_url)
        return {
            "status": "created",
            "page_url": page_url,
            "page_title": page_title,
            "topic": topic_name,
            "via_scalekit": True,
        }

    except Exception as exc:
        _log.warning("[Notion] create_topic_page failed for %r: %s", topic_name, exc)
        return {"status": "error", "error": str(exc), "topic": topic_name}
