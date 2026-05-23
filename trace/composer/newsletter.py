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

    You will receive a JSON payload describing the reader's active curiosity topics, relevant
    articles scraped from ArXiv, Hacker News, and Reddit, and optional sample_signals: the
    reader's own search queries or questions that triggered each topic.

    Use sample_signals to:
    - Mirror the reader's vocabulary (if they searched "how does attention work", don't write
      "the self-attention mechanism is a well-known…" — connect to their framing instead).
    - Calibrate technical depth: raw Google searches suggest breadth interest; detailed ChatGPT
      questions suggest the reader is already deep and wants advanced material.
    - Personalise subject lines and section openers to feel tailored, not generic.

    Return ONLY a valid JSON object matching this exact schema (no markdown fence, no preamble):
    {
      "subject_line": "<10–100 chars, compelling and specific>",
      "sections": [
        {
          "title": "<section headline>",
          "section_type": "<weekly_topics | curiosity_debt | rabbit_hole>",
          "content": "<≥50 chars of substantive insight>",
          "source_urls": ["<url>", ...],
          "audit_reasoning": "<≥10 chars explaining why this topic was included>"
        }
      ]
    }

    Rules:
    - At least one section required.
    - section_type must be one of: weekly_topics, curiosity_debt, rabbit_hole.
    - content must be substantive (≥50 chars). Cite specific findings from the articles.
    - audit_reasoning must explain the signal pattern (frequency, recency, debt).
    - source_urls: include only URLs present in the input payload.
    - Do NOT hallucinate topic names, URLs, or article titles.
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
    payload: dict[str, Any] = {
        "topics": [],
        "debt_topic_ids": [t.id for t in ctx.debt_topics],
    }
    for topic in ctx.selected_topics:
        articles = ctx.articles_by_topic_id.get(topic.id, [])
        payload["topics"].append(
            {
                "id": topic.id,
                "name": topic.name,
                "frequency": topic.frequency,
                "recency_score": topic.recency_score,
                "debt_score": topic.debt_score,
                "curiosity_type": topic.curiosity_type.value,
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
