"""
Reddit post draft — Tier B, requires explicit user approval before publishing.

SECURITY CONTRACT:
  This module prepares a Reddit post draft and queues it in the approvals system.
  It NEVER calls the Reddit API to submit/publish a post automatically.
  Only after the user explicitly clicks "Approve" in the Trace UI does the
  draft get submitted — and even then, the final publish is a separate step.

  The Reddit OAuth token lives in Scalekit's encrypted Token Vault.

DRAFT STRUCTURE:
  Title:     Agent-generated title based on bridge topic + supporting topics
  Body:      Briefing text + Trace attribution + disclosure
  Subreddit: Left blank — user chooses subreddit before approving
  Flair:     "Discussion" by default

WHY REDDIT FOR BRIDGE TOPICS:
  When Trace detects a bridge_topic_shift (a new topic connecting two clusters),
  it's exactly the kind of cross-domain insight that makes for a good Reddit post.
  The user can share their synthesized perspective with a community.
"""
from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

_REDDIT_SUBMIT_TOOL = "reddit_submit_post"
_CONNECTION_NAME = "reddit"


def _build_post_body(
    topic_name: str,
    briefing: str,
    supporting_topics: list[str],
) -> str:
    """Build the post body markdown."""
    related = ", ".join(f"*{t}*" for t in supporting_topics[:3]) if supporting_topics else "adjacent topics"
    return (
        f"{briefing}\n\n"
        f"---\n\n"
        f"**Context:** I've been noticing connections between {topic_name} and {related}. "
        f"This was surfaced by [Trace](https://github.com/parvpatodia/Trace), "
        f"an autonomous curiosity OS that tracks my interests across sources.\n\n"
        f"*Draft prepared by Trace — manually reviewed and approved before posting.*"
    )


def build_reddit_draft(
    topic_name: str,
    briefing: str,
    supporting_topics: list[str] | None = None,
) -> dict[str, Any]:
    """Build a Reddit post payload (does NOT submit to Reddit API).

    Returns the payload dict that would be sent to reddit_submit_post if approved.
    """
    title = f"Bridge topic: how {topic_name} connects to {', '.join((supporting_topics or ['adjacent ideas'])[:2])}"
    if len(title) > 300:
        title = title[:297] + "..."

    body = _build_post_body(topic_name, briefing, supporting_topics or [])

    return {
        "title": title,
        "body": body,
        "subreddit": "",  # user fills in before approving
        "flair": "Discussion",
    }


async def submit_approved_post(
    payload: dict[str, Any],
    profile_id: str = "default",
) -> dict[str, Any]:
    """Submit an APPROVED Reddit post via Scalekit Token Vault.

    This function is called ONLY after the user has explicitly approved the draft
    via POST /approvals/{id}/approve. Never call this directly from pattern detectors.

    Returns: {"status": "submitted", "post_url": str, ...}
    """
    from trace.auth.scalekit import connect_execute_tool, connect_get_authorization_link, get_scalekit_client

    if get_scalekit_client() is None:
        _log.info("[RedditDraft] Scalekit not configured — stub submission")
        return {
            "status": "stub",
            "reason": "scalekit_not_configured",
            "title": payload.get("title", ""),
        }

    if not payload.get("subreddit"):
        return {
            "status": "error",
            "error": "subreddit is required before submitting",
            "title": payload.get("title", ""),
        }

    try:
        result = await connect_execute_tool(
            tool_name=_REDDIT_SUBMIT_TOOL,
            tool_input=payload,
            identifier=profile_id,
        )

        if "error" in result:
            err = str(result["error"])
            if "not_authorized" in err.lower():
                link = await connect_get_authorization_link(
                    identifier=profile_id,
                    connection_name=_CONNECTION_NAME,
                )
                return {
                    "status": "auth_required",
                    "connection": _CONNECTION_NAME,
                    "auth_link": link,
                }
            return {"status": "error", "error": err}

        post_url = result.get("url") or result.get("post_url", "")
        _log.info("[RedditDraft] Post submitted: %s (user-approved)", post_url)
        return {
            "status": "submitted",
            "post_url": post_url,
            "title": payload.get("title", ""),
            "via_scalekit": True,
        }

    except Exception as exc:
        _log.warning("[RedditDraft] submit_approved_post failed: %s", exc)
        return {"status": "error", "error": str(exc)}
