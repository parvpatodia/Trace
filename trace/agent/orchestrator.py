"""
Pattern → Action orchestrator.

Maps detected PatternEvent instances to concrete action chains:

  emerging_interest   → Tier A: Notion page + Calendar reminder + Slack DM
  subscription_debt   → Tier B: Gmail digest draft (curated links for debt topic)
  bridge_topic_shift  → Tier A: Slack alert + Notion brief + Tier B Reddit draft

The orchestrator also uses Claude to classify whether a pattern is significant
enough to warrant an action (prevents noise from marginal score changes).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from trace.agent.approvals import ApprovalsQueue, PendingAction, get_approvals_queue
from trace.agent.kalibr_guard import execute_with_guard
from trace.agent.patterns import PatternEvent
from trace.config import get_settings
from trace.models import CuriosityGraph

_log = logging.getLogger(__name__)


# ── Claude significance gate ───────────────────────────────────────────────────

async def _is_significant(event: PatternEvent, graph: CuriosityGraph) -> bool:
    """Ask Claude whether a pattern event is worth acting on.

    Returns True immediately if Claude is not configured (dev mode).
    Returns True if the API call fails (fail-open so the demo always works).
    """
    s = get_settings()
    if not s.anthropic_api_key:
        return True  # dev mode — act on everything

    top_topics = [t.name for t in graph.top_n(5)]
    prompt = (
        f"You are evaluating whether a detected curiosity pattern is significant "
        f"enough for an AI agent to take action on behalf of the user.\n\n"
        f"Pattern type: {event.pattern_type}\n"
        f"Topic: {event.topic_name}\n"
        f"Score: {event.score:.3f}\n"
        f"Metadata: {event.metadata}\n"
        f"User's top 5 topics: {', '.join(top_topics)}\n\n"
        f"Reply with a single JSON object: {{\"significant\": true/false, \"reason\": \"...\"}}\n"
        f"Be conservative — only mark significant if it clearly warrants an action."
    )

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=s.anthropic_api_key)
        response = await asyncio.to_thread(
            lambda: client.messages.create(
                model=s.anthropic_model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
        )
        import json
        text = response.content[0].text.strip() if response.content else "{}"
        # Extract JSON from possible markdown code block.
        if "```" in text:
            text = text.split("```")[1].lstrip("json").strip()
        result = json.loads(text)
        significant = bool(result.get("significant", True))
        _log.info(
            "Claude significance gate: %s/%s → %s (%s)",
            event.pattern_type, event.topic_name, significant, result.get("reason", ""),
        )
        return significant
    except Exception as exc:
        _log.warning("Claude significance gate failed (%s) — defaulting to True", exc)
        return True  # fail-open


# ── Tier A action dispatchers ────────────────────────────────────────────────

async def _dispatch_notion(
    topic_name: str, briefing: str, profile_id: str, pattern_type: str = ""
) -> dict[str, Any]:
    """Create a Notion page for a topic via Scalekit Token Vault."""
    from trace.actions.notion import create_topic_page
    return await create_topic_page(
        topic_name=topic_name,
        briefing=briefing,
        profile_id=profile_id,
        pattern_type=pattern_type,
    )


async def _dispatch_calendar(
    topic_name: str, briefing: str, profile_id: str, pattern_type: str = ""
) -> dict[str, Any]:
    """Schedule a deep-dive calendar event via Scalekit Token Vault."""
    from trace.actions.calendar import schedule_deep_dive
    return await schedule_deep_dive(
        topic_name=topic_name,
        briefing=briefing,
        profile_id=profile_id,
        pattern_type=pattern_type,
    )


async def _dispatch_slack(
    message: str, profile_id: str
) -> dict[str, Any]:
    """Post a pattern alert to Slack via Scalekit Token Vault."""
    from trace.actions.slack import send_pattern_alert
    return await send_pattern_alert(message=message, profile_id=profile_id)


async def _draft_gmail(
    topic_name: str,
    briefing: str,
    profile_id: str,
    queue: ApprovalsQueue,
    pattern_type: str,
) -> dict[str, Any]:
    """Enqueue a Gmail draft for user approval (NEVER sends automatically)."""
    subject = f"[Trace] Reading list: {topic_name}"
    body_preview = (
        f"You've been revisiting '{topic_name}' for a while without diving deep.\n\n"
        f"{briefing[:400]}\n\n---\nThis draft was prepared by Trace and requires your approval before sending."
    )
    action = PendingAction(
        action_type="gmail_draft",
        profile_id=profile_id,
        title=subject,
        preview=body_preview[:500],
        payload={"subject": subject, "body": briefing, "to": ""},
        pattern_event_type=pattern_type,
    )
    await queue.enqueue(action)
    return {"status": "queued_for_approval", "action_id": action.id, "type": "gmail_draft"}


async def _draft_reddit(
    topic_name: str,
    briefing: str,
    profile_id: str,
    queue: ApprovalsQueue,
    pattern_type: str,
) -> dict[str, Any]:
    """Enqueue a Reddit post draft for user approval (NEVER auto-publishes)."""
    title = f"New bridge between {topic_name} and adjacent topics — worth exploring?"
    body_preview = (
        f"Trace detected a bridge-topic shift around '{topic_name}'.\n\n"
        f"{briefing[:400]}\n\n---\nThis post draft requires your approval before publishing."
    )
    action = PendingAction(
        action_type="reddit_post",
        profile_id=profile_id,
        title=title,
        preview=body_preview[:500],
        payload={"title": title, "body": briefing, "subreddit": ""},
        pattern_event_type=pattern_type,
    )
    await queue.enqueue(action)
    return {"status": "queued_for_approval", "action_id": action.id, "type": "reddit_post"}


# ── Briefing helper ────────────────────────────────────────────────────────────

async def _quick_briefing(topic_name: str) -> str:
    """Get a 2-sentence briefing on a topic from Claude or fallback."""
    s = get_settings()
    if not s.anthropic_api_key:
        return f"Trace detected rising interest in '{topic_name}'. Consider exploring recent developments."
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=s.anthropic_api_key)
        response = await asyncio.to_thread(
            lambda: client.messages.create(
                model=s.anthropic_model,
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Write 2 concise sentences about why '{topic_name}' is worth "
                        "exploring right now. Be specific and actionable. No filler."
                    ),
                }],
            )
        )
        return response.content[0].text.strip() if response.content else ""
    except Exception as exc:
        _log.warning("Quick briefing failed: %s", exc)
        return f"Rising interest detected in '{topic_name}'. Explore the latest developments."


# ── Main orchestrator ──────────────────────────────────────────────────────────

class AgentOrchestrator:
    """Maps pattern events to action chains with Claude significance gating."""

    def __init__(self) -> None:
        self._queue = get_approvals_queue()

    async def process_events(
        self,
        events: list[PatternEvent],
        graph: CuriosityGraph,
        profile_id: str = "default",
    ) -> list[dict[str, Any]]:
        """Process all pattern events and return a log of dispatched actions."""
        results: list[dict[str, Any]] = []

        for event in events:
            try:
                if not await _is_significant(event, graph):
                    _log.debug("Pattern %s/%s skipped (not significant)", event.pattern_type, event.topic_name)
                    continue
                action_results = await self._dispatch(event, profile_id)
                results.extend(action_results)
            except Exception as exc:
                _log.error(
                    "Orchestrator error for %s/%s: %s",
                    event.pattern_type, event.topic_name, exc,
                )
        return results

    async def _dispatch(
        self, event: PatternEvent, profile_id: str
    ) -> list[dict[str, Any]]:
        briefing = await _quick_briefing(event.topic_name)

        if event.pattern_type == "emerging_interest":
            # Tier A: Notion + Calendar + Slack — all guarded by Kalibr for
            # automatic retry and failure visibility.
            tasks = await asyncio.gather(
                execute_with_guard(
                    _dispatch_notion, "notion_page", event.topic_name,
                    event.topic_name, briefing, profile_id, event.pattern_type,
                ),
                execute_with_guard(
                    _dispatch_calendar, "calendar_event", event.topic_name,
                    event.topic_name, briefing, profile_id, event.pattern_type,
                ),
                execute_with_guard(
                    _dispatch_slack, "slack_alert", event.topic_name,
                    f"🧠 Trace: New emerging interest — *{event.topic_name}*\n{briefing}",
                    profile_id,
                ),
                return_exceptions=True,
            )
            return [r for r in tasks if isinstance(r, dict)]

        elif event.pattern_type == "subscription_debt":
            # Tier B: Gmail draft — requires user approval
            result = await _draft_gmail(
                event.topic_name, briefing, profile_id, self._queue, event.pattern_type
            )
            return [result]

        elif event.pattern_type == "bridge_topic_shift":
            # Tier A: Slack + Notion guarded by Kalibr; Tier B: Reddit draft
            tier_a = await asyncio.gather(
                execute_with_guard(
                    _dispatch_slack, "slack_alert", event.topic_name,
                    f"🌉 Trace: Bridge shift — *{event.topic_name}* connects "
                    f"{', '.join(event.supporting_topics[:2])}.\n{briefing}",
                    profile_id,
                ),
                execute_with_guard(
                    _dispatch_notion, "notion_page", event.topic_name,
                    event.topic_name, briefing, profile_id, event.pattern_type,
                ),
                return_exceptions=True,
            )
            tier_b = await _draft_reddit(
                event.topic_name, briefing, profile_id, self._queue, event.pattern_type
            )
            results = [r for r in tier_a if isinstance(r, dict)]
            results.append(tier_b)
            return results

        _log.warning("Unknown pattern type: %s", event.pattern_type)
        return []
