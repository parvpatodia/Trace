"""
NewsletterComposer — calls Claude to generate a Newsletter from an AssemblyContext.

Output contract:
  Claude returns a JSON object with:
    {
      "subject_line": str,
      "sections": [
        {
          "title": str,
          "section_type": "weekly_topics" | "curiosity_debt" | "rabbit_hole",
          "content": str,           # ≥50 chars
          "source_urls": [str, ...],
          "audit_reasoning": str    # ≥10 chars
        }
      ]
    }

Plain text and HTML are rendered locally from the parsed sections — no second
API call.

Prompt caching:
  The system prompt is sent as a list of content blocks with
  cache_control: {"type": "ephemeral"} on the last block, so repeated calls
  (e.g. regeneration or retry) hit the cache tier.

WHY asyncio.to_thread:
  anthropic.Anthropic is a sync client. Wrapping the call in asyncio.to_thread
  keeps the async interface without forcing callers to manage threads.
"""

from __future__ import annotations

import asyncio
import html as _html
import json
import logging
import textwrap
from datetime import datetime, timezone
from typing import Any

import anthropic
from pydantic import ValidationError

from trace.composer.assembler import AssemblyContext
from trace.models import Newsletter, NewsletterSection
from trace.utils import strip_markdown_fence

_log = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_MAX_TOKENS = 4096

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert newsletter writer for a personalized curiosity digest.
    You write in a direct, insight-first style — no filler, no hollow praise, no em-dash abuse.

    Each topic in the payload includes:
      • name, frequency, curiosity_type
      • span_days — total days this interest has been active in the reader's history
      • days_since_last_seen — days since they last engaged with this topic
      • first_seen_days_ago — how long ago this interest first appeared
      • article_count — number of articles available (may be 0)
      • sample_signals — their EXACT search queries or ChatGPT questions that triggered this topic
      • debt_score > 0 → they keep returning to this without resolving it (curiosity debt)

    YOUR MOST IMPORTANT TASK: make every sentence feel written for THIS specific person.

    1. MIRROR sample_signals VOCABULARY
       Use their exact language. If they searched "how does attention work" — write
       "attention", not "self-attention mechanisms". If they asked "why is Rust fast" —
       stay in their register. Generic newsletter voice is the failure mode.

    2. USE TEMPORAL DATA for time-aware, personal copy
       span_days and days_since_last_seen unlock observations like:
         "You've been circling this for 6 weeks without resolving it"
         "This re-emerged 3 days ago after a 2-month gap — something triggered it"
         "Brand new this week — here's the fastest path from zero to depth"
       These are what make the reader feel seen rather than spammed.

    3. REQUIRED CURIOSITY FINGERPRINT — first section only
       The very first section (weekly_topics) must open with a 2–3 sentence
       "curiosity fingerprint" paragraph that names the reader's pattern for this period.
       Be specific and punchy — name topics, durations, and whether they look unresolved.
       Example: "Your signals cluster around transformer fine-tuning (8 weeks active,
       still circling) and Rust embedded systems (re-emerged this week after 2 months).
       The fine-tuning thread is curiosity debt — you keep returning without landing
       anywhere definitive. This issue goes deep on both."

    SCHEMA — return ONLY this JSON (no markdown fence, no preamble):
    {
      "subject_line": "<10–100 chars>",
      "sections": [
        {
          "title": "<headline>",
          "section_type": "<weekly_topics|curiosity_debt|rabbit_hole>",
          "content": "<≥50 chars of substantive insight>",
          "source_urls": ["<url>", ...],
          "audit_reasoning": "<≥10 chars explaining why this section was included>"
        }
      ]
    }

    RULES:
    • subject_line: specific and personal. Name the dominant topic. AVOID generic titles
      like "Your Weekly Digest". PREFER: "The transformer fine-tuning question you keep
      reopening" or "Rust is back — and so is that embedded systems thread".
    • First section must be weekly_topics and must open with the curiosity fingerprint.
    • section_type must be one of: weekly_topics, curiosity_debt, rabbit_hole.
    • content ≥50 chars. When articles are available, cite specific findings.
    • article_count == 0: write educational content from training knowledge. Use
      section_type rabbit_hole. Do NOT invent source_urls — leave the array empty.
    • curiosity_debt topics (debt_score > 0): call out span_days explicitly and frame
      the section as "here's what you need to finally close this loop."
    • source_urls: ONLY URLs present in the input articles. Never fabricate.
    • audit_reasoning: explain frequency, span, recency, and debt pattern.
""")


class NewsletterGenerationError(Exception):
    pass


class NewsletterComposer:
    """
    Generates a Newsletter by calling Claude with an AssemblyContext payload.

    Parameters:
        client: anthropic.Anthropic sync client (required).
        model: Claude model ID.
        max_tokens: maximum tokens in the Claude response.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        if client is None:
            raise ValueError("client must be a non-None anthropic.Anthropic instance")
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    async def compose(self, ctx: AssemblyContext) -> Newsletter:
        if not ctx.selected_topics:
            raise NewsletterGenerationError(
                "Cannot compose newsletter: AssemblyContext has no selected topics"
            )

        user_message = _build_user_message(ctx)
        raw_json = await self._call_claude(user_message)
        data = _parse_response(raw_json)
        return _build_newsletter(data)

    async def _call_claude(self, user_message: str) -> str:
        system = [
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        try:
            response = await asyncio.to_thread(
                self._client.messages.create,
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_message}],
            )
        except anthropic.APIError as e:
            raise NewsletterGenerationError(f"API error calling Claude: {e}") from e

        if not response.content or not hasattr(response.content[0], "text"):
            raise NewsletterGenerationError(
                f"Unexpected Claude response format — empty or non-text content: "
                f"{response.content!r}"
            )
        return response.content[0].text


def _build_user_message(ctx: AssemblyContext) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "topics": [],
        "debt_topic_ids": [t.id for t in ctx.debt_topics],
    }
    for topic in ctx.selected_topics:
        articles = ctx.articles_by_topic_id.get(topic.id, [])
        days_since = (
            max(0, (now - topic.last_seen).days)
            if topic.last_seen else None
        )
        first_seen_ago = (
            max(0, (now - topic.first_seen).days)
            if topic.first_seen else None
        )
        payload["topics"].append(
            {
                "id": topic.id,
                "name": topic.name,
                "frequency": topic.frequency,
                "span_days": topic.span_days(),
                "days_since_last_seen": days_since,
                "first_seen_days_ago": first_seen_ago,
                "recency_score": round(topic.recency_score, 3),
                "debt_score": round(topic.debt_score, 3),
                "curiosity_type": topic.curiosity_type.value,
                "article_count": len(articles),
                "articles": [
                    {
                        "title": a.title,
                        "url": a.url,
                        "summary": a.summary,
                        "relevance_score": a.relevance_score,
                        "source": a.source.value,
                    }
                    for a in articles
                ],
                "sample_signals": ctx.signal_samples.get(topic.id, []),
            }
        )
    return json.dumps(payload, indent=2)


def _parse_response(text: str) -> dict[str, Any]:
    text = strip_markdown_fence(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise NewsletterGenerationError(
            f"Claude returned invalid JSON: {e}\nRaw: {text[:200]}"
        ) from e

    if not isinstance(data.get("subject_line"), str) or not data["subject_line"].strip():
        raise NewsletterGenerationError(
            "Claude response missing or empty 'subject_line'"
        )

    sections = data.get("sections")
    if not isinstance(sections, list) or len(sections) == 0:
        raise NewsletterGenerationError(
            "Claude response missing or empty 'sections'"
        )

    return data


def _build_newsletter(data: dict[str, Any]) -> Newsletter:
    try:
        sections = tuple(
            NewsletterSection(
                title=s["title"],
                section_type=s["section_type"],
                content=s["content"],
                source_urls=s.get("source_urls", []),
                audit_reasoning=s["audit_reasoning"],
            )
            for s in data["sections"]
        )
        plain_text = _render_plain(data["subject_line"], sections)
        html = _render_html(data["subject_line"], sections)

        return Newsletter(
            generated_at=datetime.now(timezone.utc),
            subject_line=data["subject_line"],
            sections=sections,
            plain_text=plain_text,
            html=html,
        )
    except (KeyError, ValueError, ValidationError) as e:
        raise NewsletterGenerationError(
            f"Failed to build Newsletter from Claude response: {e}"
        ) from e


def _render_plain(subject: str, sections: tuple[NewsletterSection, ...]) -> str:
    lines = [subject, "=" * len(subject), ""]
    for s in sections:
        lines.append(s.title)
        lines.append("-" * len(s.title))
        lines.append(s.content)
        if s.source_urls:
            lines.append("")
            for url in s.source_urls:
                lines.append(f"  - {url}")
        lines.append("")
    return "\n".join(lines)


def _render_html(subject: str, sections: tuple[NewsletterSection, ...]) -> str:
    e = _html.escape
    parts = [
        "<!DOCTYPE html><html><body>",
        f"<h1>{e(subject)}</h1>",
    ]
    for s in sections:
        parts.append(f"<h2>{e(s.title)}</h2>")
        parts.append(f"<p>{e(s.content)}</p>")
        if s.source_urls:
            parts.append("<ul>")
            for url in s.source_urls:
                safe_url = e(url, quote=True)
                parts.append(f'<li><a href="{safe_url}">{e(url)}</a></li>')
            parts.append("</ul>")
    parts.append("</body></html>")
    return "".join(parts)
