"""
GeminiTopicExtractor — drop-in replacement for TopicExtractor backed by Gemini.

Contract:
  ``async extract(signals) -> list[RawTopicData]`` — identical to TopicExtractor.
  CuriosityGraphBuilder accepts it via duck typing; no changes there.

What this DOES NOT do:
  - Anthropic-style prompt caching. Gemini has implicit caching for repeated
    prefixes on 2.5+ models and explicit caching via ``caching.CachedContent``.
    For a hackathon where each /ingest call has a different signal payload but
    the same system prompt, implicit caching is enough — Google's
    infrastructure auto-deducts the savings. Explicit caching would add code
    paths we don't need at this scale.

Prompt is shared with the Claude extractor (same task, same JSON output
format) so retrieval quality stays comparable across paths.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from trace.graph.extractor import RawTopicData, TopicExtractionError, _SYSTEM_PROMPT
from trace.models import RawSignal
from trace.utils import strip_markdown_fence

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
_DEFAULT_BATCH_SIZE = 100
_DEFAULT_MAX_OUTPUT_TOKENS = 4096


class GeminiTopicExtractor:
    """Extract curiosity topics from RawSignals using Google's Gemini API.

    Parameters mirror the Anthropic extractor so the two are interchangeable
    in the pipeline.
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_output_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        if not _GENAI_AVAILABLE:
            raise RuntimeError(
                "google-genai is not installed — run "
                "`pip install google-genai`"
            )
        if not api_key:
            raise ValueError(
                "GeminiTopicExtractor: api_key is empty. "
                "Set GEMINI_API_KEY in your environment."
            )
        if batch_size < 1:
            raise ValueError(
                f"GeminiTopicExtractor: batch_size must be >= 1, got {batch_size}"
            )
        self._client = genai.Client(api_key=api_key)
        self._model_name = model
        self._batch_size = batch_size
        self._generate_config = genai_types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            max_output_tokens=max_output_tokens,
            temperature=0.2,  # low — we want deterministic topic clustering
        )

    async def extract(self, signals: list[RawSignal]) -> list[RawTopicData]:
        if not signals:
            return []

        valid_ids: frozenset[str] = frozenset(s.id for s in signals)
        batches = [
            signals[i : i + self._batch_size]
            for i in range(0, len(signals), self._batch_size)
        ]
        _log.info(
            "GeminiTopicExtractor: extracting from %d signals in %d batch(es) using %s",
            len(signals), len(batches), self._model_name,
        )

        all_raw: list[RawTopicData] = []
        known_topics: list[str] = []
        for i, batch in enumerate(batches):
            _log.debug("Batch %d/%d (%d signals)", i + 1, len(batches), len(batch))
            batch_topics = await asyncio.to_thread(self._call_api, batch, valid_ids, known_topics)
            all_raw.extend(batch_topics)
            known_topics = list(
                dict.fromkeys(known_topics + [t["name"] for t in batch_topics])
            )

        merged = _merge_topics(all_raw)
        _log.info("GeminiTopicExtractor: %d unique topic(s)", len(merged))
        return merged

    def _call_api(
        self,
        batch: list[RawSignal],
        valid_ids: frozenset[str],
        known_topics: list[str],
    ) -> list[RawTopicData]:
        payload = [
            {"id": s.id, "source": s.source.value, "content": s.content}
            for s in batch
        ]
        preamble = ""
        if known_topics:
            names_json = json.dumps(known_topics, ensure_ascii=False)
            preamble = (
                f"Topics already identified from earlier batches: {names_json}\n"
                "Reuse these exact names when signals clearly belong to the same topic.\n\n"
            )
        user_text = (
            preamble
            + "Here are the signals to analyze:\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=user_text,
                config=self._generate_config,
            )
        except Exception as exc:  # google.genai raises a variety of errors
            raise TopicExtractionError(f"Gemini API error: {exc}") from exc

        raw_text = _extract_text(response)
        if not raw_text:
            raise TopicExtractionError(
                f"Gemini returned empty content. finish_reason="
                f"{getattr(response, 'candidates', [{}])[0] if getattr(response, 'candidates', None) else 'unknown'}"
            )
        return _parse_response(raw_text, valid_ids)


def _extract_text(response: Any) -> str:
    """Pull the text payload out of a Gemini response, tolerating shape drift.

    google.genai exposes ``response.text`` but raises if no usable text part
    exists (safety block, MAX_TOKENS, etc.). Walk parts manually so we can
    log the failure mode instead of crashing.
    """
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
    except Exception as exc:
        _log.warning("Could not walk Gemini response parts: %s", exc)
    return ""


def _parse_response(text: str, valid_ids: frozenset[str]) -> list[RawTopicData]:
    """Parse Gemini's JSON output into RawTopicData, defensively.

    Mirrors the Claude extractor's parser so behaviour is identical even
    if one model occasionally wraps JSON in code fences.
    """
    text = strip_markdown_fence(text)
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError as e:
        raise TopicExtractionError(
            f"Failed to parse Gemini response as JSON: {e}\nResponse: {text[:500]}"
        ) from e

    if not isinstance(parsed, list):
        raise TopicExtractionError(
            f"Gemini response must be a JSON array, got {type(parsed).__name__}"
        )

    topics: list[RawTopicData] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        signal_ids = item.get("signal_ids")
        if not name or not isinstance(signal_ids, list):
            continue

        filtered_ids = [sid for sid in signal_ids if sid in valid_ids]
        if not filtered_ids:
            _log.warning(
                "Discarding topic %r — all %d signal_id(s) hallucinated by Gemini",
                name, len(signal_ids),
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


def _merge_topics(topics: list[RawTopicData]) -> list[RawTopicData]:
    """Merge topics with the same canonical name across batches.

    Kept inline (rather than importing from the Claude extractor) so this
    module stays self-contained and the Claude path is fully optional.
    """
    import re

    def normalise(name: str) -> str:
        return re.sub(r"[\s\-_]+", "", name.lower())

    norm_to_canonical: dict[str, str] = {}
    merged: dict[str, RawTopicData] = {}

    for topic in topics:
        name = topic["name"]
        norm = normalise(name)
        if norm in norm_to_canonical:
            canonical = norm_to_canonical[norm]
            existing = merged[canonical]
            existing["signal_ids"] = list(
                dict.fromkeys(existing["signal_ids"] + topic["signal_ids"])
            )
            extra_aliases = ([name] if name != canonical else []) + topic["aliases"]
            existing["aliases"] = list(
                dict.fromkeys(existing["aliases"] + extra_aliases)
            )
        else:
            norm_to_canonical[norm] = name
            merged[name] = RawTopicData(
                name=name,
                aliases=list(topic["aliases"]),
                signal_ids=list(topic["signal_ids"]),
            )
    return list(merged.values())
