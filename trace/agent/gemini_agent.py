"""
GeminiContextAgent — the heart of the Personal Context API demo.

Given a user question, this agent issues two parallel Gemini calls:

  1. **Bare**: ``gemini-2.5-flash`` with no system instruction.
     Represents what a generic chatbot returns for the same query.

  2. **Contextual**: same model, but the top-K personal context items are
     formatted as a system instruction. This is what an agent backed by
     Trace would receive.

Both responses are returned in a single payload so the frontend can render
them side-by-side. The contrast is the demo — and the pitch.

Parallelism:
  The two calls are independent. We launch both with ``asyncio.gather`` so
  the wall-clock cost is one Gemini round-trip, not two. On Gemini 2.5 Flash
  this is typically 1–3 seconds.

What this is NOT:
  This is not a multi-turn ADK agent with tools. That's the v2 architecture
  (see Part 3.3 of the spec). For the hackathon, a stateless two-shot is
  sufficient to demonstrate the core value: personal context dramatically
  improves response relevance.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from trace.context.store import ContextItem

try:
    from google import genai  # type: ignore[import]
    from google.genai import types as genai_types  # type: ignore[import]
    _GENAI_AVAILABLE = True
except ModuleNotFoundError:
    _GENAI_AVAILABLE = False
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

_log = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-2.5-flash"
_MAX_CONTEXT_ITEMS_IN_PROMPT = 10
_MAX_OUTPUT_TOKENS = 1024

_SYSTEM_INSTRUCTION_TEMPLATE = """You are a personal AI assistant for ONE specific user.

Below is what we know about this user from their actual behavioural signals — what they read, search for, star on GitHub, and engage with in email. This is not a guess; it is grounded data.

PERSONAL CONTEXT
{context}

When you answer, use this context to make your response specific to THIS user — reference their actual interests, name the concrete topics they care about, and recommend things that match their demonstrated curiosity. Do not invent context that isn't listed. If the question is unrelated to anything in their context, say so and answer normally."""


class GeminiContextAgent:
    """Stateless two-shot agent: ``query()`` → {with_context, without_context}.

    Build one of these per request — they hold no per-user state and are
    cheap to construct.
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        max_output_tokens: int = _MAX_OUTPUT_TOKENS,
    ) -> None:
        if not _GENAI_AVAILABLE:
            raise RuntimeError(
                "google-genai is not installed — run "
                "`pip install google-genai`"
            )
        if not api_key:
            raise ValueError(
                "GeminiContextAgent: api_key is empty. "
                "Set GEMINI_API_KEY in your environment."
            )
        self._client = genai.Client(api_key=api_key)
        self._model_name = model
        self._bare_config = genai_types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=0.7,  # higher than topic extraction — we want prose
        )

    async def query(
        self,
        question: str,
        context_items: list[tuple[ContextItem, float]],
    ) -> dict[str, Any]:
        """Run the question against Gemini twice — bare and contextual.

        Returns:
            {
                "with_context": str,
                "without_context": str,
                "context_items_used": [...],
                "context_item_count": int,
                "model": str,
            }
        """
        bare_task = asyncio.to_thread(self._call_bare, question)
        ctx_task = asyncio.to_thread(self._call_with_context, question, context_items)

        bare_text, ctx_text = await asyncio.gather(
            bare_task, ctx_task, return_exceptions=False,
        )

        return {
            "with_context": ctx_text,
            "without_context": bare_text,
            "context_items_used": [
                {
                    "text": item.text,
                    "score": round(score, 4),
                    "source": item.source,
                    "topic": item.topic_name,
                }
                for item, score in context_items[:_MAX_CONTEXT_ITEMS_IN_PROMPT]
            ],
            "context_item_count": len(context_items),
            "model": self._model_name,
        }

    def _call_bare(self, question: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=question,
                config=self._bare_config,
            )
            return _safe_text(response)
        except Exception as exc:
            _log.warning("Bare Gemini call failed: %s", exc)
            return f"[Gemini error: {exc}]"

    def _call_with_context(
        self,
        question: str,
        context_items: list[tuple[ContextItem, float]],
    ) -> str:
        if not context_items:
            return self._call_bare(question) + "\n\n_(No personal context was found for this query.)_"

        context_block = _format_context_for_prompt(context_items)
        system_instruction = _SYSTEM_INSTRUCTION_TEMPLATE.format(context=context_block)
        ctx_config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=self._bare_config.max_output_tokens,
            temperature=self._bare_config.temperature,
        )
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=question,
                config=ctx_config,
            )
            return _safe_text(response)
        except Exception as exc:
            _log.warning("Contextual Gemini call failed: %s", exc)
            return f"[Gemini error: {exc}]"


def _format_context_for_prompt(items: list[tuple[ContextItem, float]]) -> str:
    """Render context items as a numbered list for the system prompt.

    Each line surfaces source + score so the model can weight items naturally.
    Capped at _MAX_CONTEXT_ITEMS_IN_PROMPT to keep the prompt small enough
    that the model treats every line as salient.
    """
    lines: list[str] = []
    for i, (item, score) in enumerate(items[:_MAX_CONTEXT_ITEMS_IN_PROMPT], 1):
        tag = item.source.upper()
        if item.topic_name and item.source == "signal":
            tag = f"SIGNAL · {item.topic_name}"
        lines.append(f"{i}. [{tag}] (relevance {score:.2f}) {item.text}")
    return "\n".join(lines)


def _safe_text(response: Any) -> str:
    """Extract text from a Gemini response, tolerating safety blocks."""
    try:
        text = response.text
        if text:
            return text
    except Exception:
        pass
    try:
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            parts = getattr(getattr(cand, "content", None), "parts", None) or []
            for part in parts:
                t = getattr(part, "text", None)
                if t:
                    return str(t)
    except Exception:
        pass
    return "[Gemini returned no text — possibly blocked by safety filters or rate-limited.]"
