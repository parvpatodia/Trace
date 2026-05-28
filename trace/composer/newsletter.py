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
# 5 full sections × ~1200 tokens each + overhead ≈ 6200 tokens needed.
# 4096 overflows by ~1900 tokens, producing truncated/unparseable JSON.
_DEFAULT_MAX_TOKENS = 8192

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an elite personal intelligence writer — think the best of Morning Brew, TLDR,
    and AlphaSignal, but written for ONE specific person based on their actual curiosity
    signals. You write with the voice of a brilliant friend who has read everything and
    knows exactly what you care about.

    Your writing philosophy:
    - SPECIFIC over generic. "You've been asking about attention mechanisms for 8 weeks" not
      "AI is trending."
    - INSIGHT-FIRST. Lead with the non-obvious take, not the summary.
    - PERSONAL. Mirror the reader's exact vocabulary from their search queries and ChatGPT
      questions. Never sound like a mass newsletter.
    - ACTIONABLE. Every section ends with something concrete to do, read, or try.
    - HONEST. If their curiosity is unresolved, say so. Don't paper over confusion.

    The payload top-level fields:
      * current_date — today's date (ISO 8601). Use it for temporal language:
        "last week", "this month", "6 weeks ago", etc.
      * debt_topic_ids — IDs of topics with unresolved curiosity debt.

    Each topic in the payload includes:
      * name, frequency, curiosity_type
      * span_days — total days this interest has been active in their history
      * days_since_last_seen — days since they last engaged with this topic
      * first_seen_days_ago — how long ago this interest first appeared
      * article_count — number of fresh articles available (may be 0)
      * sample_signals — their EXACT search queries or ChatGPT questions
      * debt_score > 0 — they keep returning to this without resolution (curiosity debt)
      * related_topics — other active topics sharing vocabulary with this one.
        Non-empty list = vocabulary bridge exists. Use this to write bridge_insight sections
        that connect the two topics with a non-obvious insight. If related_topics is empty,
        do NOT force a bridge — only write bridge_insight when the connection is real.

    ══════════════════════════════════════════════════════
    CURIOSITY FINGERPRINT (required, first section only)
    ══════════════════════════════════════════════════════
    The weekly_topics section MUST open with a 3-4 sentence curiosity fingerprint paragraph.
    This is the most important paragraph in the entire newsletter. It must:
      1. Name the dominant signal clusters this period using specific topic names
      2. Use temporal data: "X weeks active", "re-emerged after Y days", "new this week"
      3. Identify the reader's current MODE: DEEP (high frequency, unresolved),
         SURFACING (re-emerging), EXPLORING (new), or RESOLVING (frequency dropping)
      4. Name any curiosity debt explicitly — patterns that keep returning without closure

    Example fingerprint: "This week your signals cluster around transformer fine-tuning
    (8 weeks active, still unresolved — 42 searches, no resolution pattern yet) and
    Rust embedded systems (re-emerged 3 days ago after a 2-month gap). You're in DEEP
    mode on transformers — your question 'why does fine-tuning sometimes hurt base
    capabilities' is the thread that keeps pulling you back. The Rust thread suggests
    something in your project environment changed."

    ══════════════════════════════════════════════════════
    RICH SECTION SCHEMA
    ══════════════════════════════════════════════════════
    Return ONLY this JSON (no markdown fence, no preamble):
    {
      "subject_line": "<10-100 chars — specific, personal, names the dominant topic>",
      "sections": [
        {
          "title": "<punchy headline that names the specific topic and stakes>",
          "section_type": "<weekly_topics|curiosity_debt|rabbit_hole|emerging_spike|bridge_insight>",
          "content": "<≥50 chars — main narrative, 2-3 paragraphs, insight-first>",
          "source_urls": ["<url>", ...],
          "audit_reasoning": "<≥10 chars — why this section, frequency/span/recency/debt>",
          "tldr": ["<bullet 1, Morning Brew style — punchy, specific, 1 sentence>",
                   "<bullet 2>",
                   "<bullet 3>"],
          "deep_insight": "<2-3 paragraphs of genuine insight connecting topic to reader's curiosity pattern>",
          "why_this_matters": "<1-2 sentences: why THIS topic matters RIGHT NOW given their pattern>",
          "action_item": "<specific next step: 'Read this paper', 'Try this experiment', 'Search for X'>",
          "connection": "<how this connects to their other active interests>"
        }
      ]
    }

    SECTION TYPES:
    * weekly_topics — dominant interests this period (first section, always include)
    * curiosity_debt — recurring pattern without resolution (debt_score > 0)
    * rabbit_hole — deep dive on a single topic, educational if no articles available
    * emerging_spike — topic with sudden frequency increase this week
    * bridge_insight — surprising connection between two otherwise-separate interests

    WRITING RULES:
    * subject_line: SPECIFIC and personal. Name the dominant topic. NEVER: "Your Weekly
      Digest". PREFER: "The transformer fine-tuning question you keep reopening" or
      "Rust is back — 6 weeks of embedded silence just broke."
    * tldr: exactly 3 bullets. Morning Brew style — start with the key fact, be punchy,
      include a number or specific name when possible. E.g., "A new paper shows that
      LoRA fine-tuning on <100 examples consistently degrades reasoning on held-out tasks"
    * deep_insight: this is where you earn your keep. Connect the articles to the reader's
      specific confusion or curiosity pattern. Reference their sample_signals vocabulary.
      Explain what the field actually thinks, why common intuitions fail, what's unresolved.
    * why_this_matters: be blunt about timing. "You've been stuck on this for 6 weeks and
      a new paper just landed that speaks directly to your confusion."
    * action_item: specific and testable. NOT "explore more." YES: "Run the GLUE benchmark
      on your fine-tuned model before and after — the degradation pattern is diagnostic."
    * connection: only include if the connection is genuinely non-obvious. Skip if forced.
    * content: write like you're the smartest person in the room who also happens to care
      about THIS reader. Minimum 2 paragraphs. Use the reader's own vocabulary.
    * article_count == 0: write from training knowledge. section_type rabbit_hole.
      Do NOT invent source_urls — leave array empty.
    * source_urls: ONLY URLs present in the input. Never fabricate.
    * audit_reasoning: explain frequency, span, recency, debt — this powers the audit trail.
    * First section must be weekly_topics. curiosity_debt section for every debt_score > 0
      topic. Include at least one rabbit_hole for the deepest topic.
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

        # Report outcome to Kalibr router — feeds the Thompson Sampling bandit
        # so it learns which model/path produces valid newsletter JSON most reliably.
        try:
            from trace.agent.kalibr_guard import get_compose_router
            router = get_compose_router()
            if router:
                router.report(success=True)
        except Exception:
            pass

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
        "current_date": now.strftime("%Y-%m-%d"),
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
                "related_topics": ctx.related_topics.get(topic.id, []),
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
                tldr=s.get("tldr", []),
                deep_insight=s.get("deep_insight", ""),
                why_this_matters=s.get("why_this_matters", ""),
                action_item=s.get("action_item", ""),
                connection=s.get("connection", ""),
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
        if s.tldr:
            lines.append("TL;DR:")
            for b in s.tldr:
                lines.append(f"  • {b}")
            lines.append("")
        lines.append(s.content)
        if s.deep_insight:
            lines.append("")
            lines.append("Deep Insight:")
            lines.append(s.deep_insight)
        if s.why_this_matters:
            lines.append("")
            lines.append(f"Why now: {s.why_this_matters}")
        if s.action_item:
            lines.append(f"→ Try this: {s.action_item}")
        if s.connection:
            lines.append(f"Connection: {s.connection}")
        if s.source_urls:
            lines.append("")
            for url in s.source_urls:
                lines.append(f"  - {url}")
        lines.append("")
    return "\n".join(lines)


def _render_html(subject: str, sections: tuple[NewsletterSection, ...]) -> str:
    e = _html.escape

    def p(text: str) -> str:
        return "".join(f"<p>{e(para)}</p>" for para in text.split("\n\n") if para.strip())

    parts = [
        '<!DOCTYPE html><html><head><meta charset="UTF-8"></head>',
        '<body style="font-family:system-ui,sans-serif;max-width:680px;margin:0 auto;'
        'padding:24px;background:#0a0a14;color:#e0e0f0">',
        f'<h1 style="font-size:1.4rem;font-weight:700;margin-bottom:24px">{e(subject)}</h1>',
    ]
    for s in sections:
        parts.append('<div style="margin-bottom:2rem;padding-bottom:1.5rem;'
                     'border-bottom:1px solid rgba(148,163,255,0.1)">')
        parts.append(f'<h2 style="font-size:1.05rem;font-weight:700;margin-bottom:0.5rem">'
                     f'{e(s.title)}</h2>')
        if s.tldr:
            parts.append('<div style="background:rgba(99,102,241,0.08);border:1px solid '
                         'rgba(99,102,241,0.2);border-radius:6px;padding:12px 16px;'
                         'margin-bottom:12px">')
            parts.append('<div style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;'
                         'text-transform:uppercase;color:#a78bfa;margin-bottom:8px">TL;DR</div>')
            parts.append('<ul style="margin:0;padding:0;list-style:none">')
            for b in s.tldr:
                parts.append(f'<li style="font-size:0.88rem;padding:3px 0;color:#e0e0f0">'
                              f'<span style="color:#a78bfa">• </span>{e(b)}</li>')
            parts.append("</ul></div>")
        parts.append(f'<div style="color:#c0c8e0;line-height:1.75;font-size:0.9rem">'
                     f'{p(s.content)}</div>')
        if s.deep_insight:
            parts.append('<div style="border-left:3px solid #10b981;padding:10px 14px;'
                         'margin:12px 0;background:rgba(16,185,129,0.05)">')
            parts.append('<div style="font-size:0.6rem;font-weight:700;letter-spacing:0.12em;'
                         'text-transform:uppercase;color:#10b981;margin-bottom:6px">Deep Insight</div>')
            parts.append(f'<div style="font-size:0.88rem;color:#c8e6c9;line-height:1.7">'
                         f'{p(s.deep_insight)}</div></div>')
        if s.why_this_matters:
            parts.append('<div style="border-left:2px solid #f59e0b;padding:8px 12px;'
                         'margin:8px 0;background:rgba(245,158,11,0.05);font-size:0.85rem;'
                         'color:#d1b896">')
            parts.append(f'<strong style="color:#f59e0b">Why now:</strong> {e(s.why_this_matters)}'
                         f'</div>')
        if s.action_item:
            parts.append('<div style="border-left:2px solid #22d3ee;padding:8px 12px;'
                         'margin:8px 0;background:rgba(34,211,238,0.05);font-size:0.85rem;'
                         'color:#22d3ee">')
            parts.append(f'→ <strong>Try this:</strong> {e(s.action_item)}</div>')
        if s.connection:
            parts.append(f'<div style="font-size:0.82rem;color:#6b7280;font-style:italic;'
                         f'border-top:1px solid rgba(148,163,255,0.08);padding-top:10px;'
                         f'margin-top:10px"><strong style="font-style:normal;color:#9ca3af">'
                         f'Connection:</strong> {e(s.connection)}</div>')
        if s.source_urls:
            parts.append('<div style="margin-top:10px">')
            for url in s.source_urls:
                safe_url = e(url, quote=True)
                parts.append(f'<a href="{safe_url}" style="display:block;font-size:0.78rem;'
                              f'color:#818cf8;margin-bottom:3px">{e(url)}</a>')
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</body></html>")
    return "".join(parts)
