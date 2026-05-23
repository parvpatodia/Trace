"""
TopicExtractor — calls Claude to identify curiosity topics from RawSignals.

Architecture:
  1. Batch signals (default 50 per call) to stay well under context limits.
  2. Per batch: serialize signal ids + content as JSON, call messages.create.
  3. Parse Claude's JSON response into list[RawTopicData].
  4. Merge topics with the same name across batches.

Prompt design:
  System prompt: cached via cache_control=ephemeral. Describes the task,
  output format, and rules for topic extraction. Cached because it is
  identical across all batch calls in a single collection run.

  User message: a JSON array of {id, source, content} objects, one per signal.
  Signal IDs are included so Claude can reference them in signal_ids arrays.

Output format (Claude responds with):
  [
    {
      "name": "transformer architecture",    <- lowercase, 2-5 words
      "aliases": ["attention mechanism"],    <- optional synonyms
      "signal_ids": ["uuid1", "uuid2"]       <- IDs from this batch only
    }
  ]

Merging strategy:
  Topics with the same normalized name (lowercased, stripped) are merged:
  - signal_ids: union (no duplicates)
  - aliases: union (no duplicates)
  This handles the common case where the same topic appears in multiple batches.

WHY DEPENDENCY INJECTION FOR anthropic.Anthropic:
  Same reasoning as RedditCollector — auth is the caller's responsibility.
  Keeps the extractor testable without env vars or network access.

WHY BATCH_SIZE = 50:
  At ~100 chars/signal average, 50 signals ≈ 5000 chars user message.
  Well within the 200k context window but large enough to give Claude
  cross-signal context for clustering. Configurable for tuning.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, TypedDict

import anthropic

from trace.models import RawSignal
from trace.utils import strip_markdown_fence

_log = logging.getLogger(__name__)


class RawTopicData(TypedDict):
    """Structured output from one round of topic extraction."""
    name: str
    aliases: list[str]
    signal_ids: list[str]


class TopicExtractionError(Exception):
    """Raised when Claude's response cannot be parsed or the API call fails."""


_DEFAULT_MODEL = "claude-sonnet-4-6-20251001"
_DEFAULT_BATCH_SIZE = 100

_SYSTEM_PROMPT = """You are a curiosity analyst. Your task is to identify topics of genuine intellectual curiosity from a person's digital signals.

A signal is a piece of text extracted from their browser history, saved Reddit posts, ChatGPT conversations, local documents, or similar sources.

Your job: given a batch of signals, identify the distinct topics the person is actively exploring and learning about.

Rules:
1. Extract 1-10 topics per batch. Do not over-fragment into tiny subtopics.
2. Each topic must have a canonical name: lowercase, 2-5 words, specific enough to be actionable.
3. Associate each signal_id with at most one topic (the most relevant one).
4. Ignore signals that don't represent genuine curiosity: pure navigation, e-commerce, social browsing, news headlines without depth.
5. Aliases are optional synonyms or related terms for the topic.

Respond ONLY with a valid JSON array in this exact format (no prose, no markdown, no code fences):
[
  {
    "name": "transformer architecture",
    "aliases": ["self-attention", "attention mechanism"],
    "signal_ids": ["<uuid>", "<uuid>"]
  }
]

If no topics are found, respond with an empty array: []"""


class TopicExtractor:
    """
    Extracts curiosity topics from a list of RawSignals using Claude.

    Parameters:
        client:     authenticated anthropic.Anthropic instance.
        model:      Claude model ID to use for extraction.
        batch_size: number of signals per API call (default 50).
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = _DEFAULT_MODEL,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        if client is None:
            raise ValueError(
                "TopicExtractor: client must not be None. "
                "Pass an authenticated anthropic.Anthropic instance."
            )
        if batch_size < 1:
            raise ValueError(
                f"TopicExtractor: batch_size must be >= 1, got {batch_size}"
            )
        self._client = client
        self._model = model
        self._batch_size = batch_size

    async def extract(self, signals: list[RawSignal]) -> list[RawTopicData]:
        if not signals:
            return []

        valid_ids: frozenset[str] = frozenset(s.id for s in signals)
        batches = [
            signals[i : i + self._batch_size]
            for i in range(0, len(signals), self._batch_size)
        ]
        _log.info("Extracting topics from %d signals in %d batch(es)", len(signals), len(batches))

        all_raw: list[RawTopicData] = []
        for i, batch in enumerate(batches):
            _log.debug("Processing batch %d/%d (%d signals)", i + 1, len(batches), len(batch))
            batch_topics = await asyncio.to_thread(self._call_api, batch, valid_ids)
            all_raw.extend(batch_topics)

        merged = self._merge_topics(all_raw)
        _log.info("Extracted %d unique topic(s)", len(merged))
        return merged

    def _call_api(
        self, batch: list[RawSignal], valid_ids: frozenset[str]
    ) -> list[RawTopicData]:
        payload = [
            {"id": s.id, "source": s.source.value, "content": s.content}
            for s in batch
        ]
        user_text = (
            "Here are the signals to analyze:\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_text}],
            )
        except anthropic.APIError as e:
            raise TopicExtractionError(f"Claude API error: {e}") from e

        if not response.content or not hasattr(response.content[0], "text"):
            raise TopicExtractionError(
                f"Unexpected Claude response format — empty or non-text content: "
                f"{response.content!r}"
            )
        raw_text: str = response.content[0].text
        return self._parse_response(raw_text, valid_ids)

    def _parse_response(
        self, text: str, valid_ids: frozenset[str]
    ) -> list[RawTopicData]:
        text = strip_markdown_fence(text)
        try:
            parsed: Any = json.loads(text)
        except json.JSONDecodeError as e:
            raise TopicExtractionError(
                f"Failed to parse Claude response as JSON: {e}\nResponse: {text[:500]}"
            ) from e

        if not isinstance(parsed, list):
            raise TopicExtractionError(
                f"Claude response must be a JSON array, got {type(parsed).__name__}"
            )

        topics: list[RawTopicData] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            signal_ids = item.get("signal_ids")
            if not name or not isinstance(signal_ids, list):
                continue

            # Filter out signal IDs Claude hallucinated (not in our batch)
            filtered_ids = [sid for sid in signal_ids if sid in valid_ids]
            if not filtered_ids:
                _log.warning(
                    "Discarding topic %r — all %d signal_id(s) not found in batch "
                    "(hallucinated by Claude)",
                    name,
                    len(signal_ids),
                )
                continue

            topics.append(
                RawTopicData(
                    name=str(name).strip().lower(),
                    aliases=[str(a) for a in item.get("aliases", []) if isinstance(a, str)],
                    signal_ids=filtered_ids,
                )
            )
        return topics

    def _merge_topics(self, topics: list[RawTopicData]) -> list[RawTopicData]:
        merged: dict[str, RawTopicData] = {}
        for topic in topics:
            name = topic["name"]
            if name not in merged:
                merged[name] = RawTopicData(
                    name=name,
                    aliases=list(topic["aliases"]),
                    signal_ids=list(topic["signal_ids"]),
                )
            else:
                existing = merged[name]
                # Deduplicated union for both signal_ids and aliases
                existing["signal_ids"] = list(
                    dict.fromkeys(existing["signal_ids"] + topic["signal_ids"])
                )
                existing["aliases"] = list(
                    dict.fromkeys(existing["aliases"] + topic["aliases"])
                )
        return list(merged.values())
