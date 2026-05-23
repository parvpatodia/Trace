"""
CuriosityGraphBuilder — converts RawSignals into a scored CuriosityGraph.

Two-pass algorithm:
  Pass 1 — Extract: call TopicExtractor to get raw topic assignments.
           Each RawTopicData maps a topic name to a list of signal IDs.

  Pass 2 — Score: for each raw topic, compute:
    frequency    = number of matched signals
    recency_score = exp(-λ * days_since_last_seen), λ = ln(2) / half_life_days
    debt_score   = (freq / threshold_freq) * (span_days / threshold_days)
                   when freq >= threshold AND span >= threshold; else 0.0
    curiosity_type = RECURRING if freq >= threshold AND span >= threshold_days
                     else SHALLOW

Scoring rationale:
  recency_score: exponential decay — a topic last seen 14 days ago (default
  half-life) scores 0.5; last seen today scores ~1.0; last seen 28 days ago
  scores ~0.25. This makes recently active topics float to the top.

  debt_score: captures topics that have appeared repeatedly over a long period
  without being "resolved" (e.g. appearing for 3 weeks but no deep dive).
  Weighted 2x in composite_score() because unresolved curiosity is the most
  actionable newsletter content.

  CuriosityType classification:
    SHALLOW   — only 1-2 occurrences, or all within a short span
    RECURRING — freq >= debt_threshold_occurrences AND span >= debt_threshold_days
    (DEEP and RESOLVED are set by downstream pipeline steps, not here)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import urlparse

from trace.graph.extractor import RawTopicData, TopicExtractor
from trace.models import CuriosityGraph, CuriosityType, RawSignal, SignalSource, Topic

# Signal source weights — higher = stronger evidence of genuine curiosity.
# Google search queries and ChatGPT questions are explicit intent; passive page
# visits are weaker signals. YouTube rewatches and Chrome revisits are in between.
_BASE_SIGNAL_WEIGHTS: dict[SignalSource, float] = {
    SignalSource.CHROME_HISTORY: 0.8,    # passive browsing (no search intent)
    SignalSource.GOOGLE_TAKEOUT: 1.0,    # upgraded per-signal in _signal_weight()
    SignalSource.CHATGPT_EXPORT: 1.5,    # explicit questions → strong intent
    SignalSource.YOUTUBE_TAKEOUT: 1.0,   # upgraded per-signal in _signal_weight()
    SignalSource.REDDIT_POST: 1.2,
    SignalSource.REDDIT_SAVED: 1.3,      # saved = higher intent than casual browsing
    SignalSource.GMAIL: 0.9,             # newsletter subject — moderate intent signal
    SignalSource.GMAIL_OPEN: 1.1,        # user actually opened it — stronger intent
    SignalSource.GMAIL_IGNORED: 0.6,     # subscription debt: ignored → weak curiosity signal
}

# Sources that carry explicit user intent — prioritised for signal_samples
_EXPLICIT_SOURCES = frozenset({SignalSource.CHATGPT_EXPORT, SignalSource.GOOGLE_TAKEOUT})

_log = logging.getLogger(__name__)

_DEFAULT_HALF_LIFE_DAYS: int = 14
_DEFAULT_DEBT_THRESHOLD_DAYS: int = 7
_DEFAULT_DEBT_THRESHOLD_OCCURRENCES: int = 3

# Cap on signals sent to Claude — beyond this, returns diminish while cost soars.
# 500 signals at batch_size=100 = 5 API calls instead of 1960.
_MAX_SIGNALS_FOR_EXTRACTION: int = 500
_MAX_SIGNALS_PER_DOMAIN: int = 3


class CuriosityGraphBuilder:
    """
    Builds a CuriosityGraph from a list of RawSignals.

    Parameters:
        extractor:                   TopicExtractor (or compatible duck type).
        half_life_days:              recency decay half-life in days (default 14).
        debt_threshold_days:         minimum span to qualify as RECURRING (default 7).
        debt_threshold_occurrences:  minimum frequency for RECURRING (default 3).
    """

    def __init__(
        self,
        extractor: TopicExtractor,
        half_life_days: int = _DEFAULT_HALF_LIFE_DAYS,
        debt_threshold_days: int = _DEFAULT_DEBT_THRESHOLD_DAYS,
        debt_threshold_occurrences: int = _DEFAULT_DEBT_THRESHOLD_OCCURRENCES,
    ) -> None:
        if extractor is None:
            raise ValueError(
                "CuriosityGraphBuilder: extractor must not be None."
            )
        if half_life_days <= 0:
            raise ValueError(
                f"CuriosityGraphBuilder: half_life_days must be > 0, got {half_life_days}"
            )
        self._extractor = extractor
        self._half_life_days = half_life_days
        self._debt_threshold_days = debt_threshold_days
        self._debt_threshold_occurrences = debt_threshold_occurrences
        # λ = ln(2) / half_life → at t=half_life, exp(-λt) = 0.5
        self._lambda = math.log(2) / half_life_days

    async def build(
        self,
        signals: list[RawSignal],
        *,
        now: datetime | None = None,
    ) -> CuriosityGraph:
        """Build a CuriosityGraph from the given signals.

        Args:
            signals: raw signals to analyse.
            now:     reference datetime for recency scoring. Defaults to
                     ``datetime.now(timezone.utc)``. Override in tests to
                     make recency scores deterministic.
        """
        _now = now or datetime.now(timezone.utc)
        signal_count = len(signals)
        source_breakdown = self._compute_source_breakdown(signals)

        if not signals:
            return CuriosityGraph(
                topics=(),
                signal_count=0,
                source_breakdown={},
            )

        sampled = self._sample_signals(signals)
        if len(sampled) < len(signals):
            _log.info(
                "Sampled %d signals from %d for topic extraction (domain-capped, recency-ranked)",
                len(sampled), len(signals),
            )
        raw_topics: list[RawTopicData] = await self._extractor.extract(sampled)

        signals_by_id: dict[str, RawSignal] = {s.id: s for s in signals}
        scored: list[Topic] = []
        for raw in raw_topics:
            topic = self._score_topic(raw, signals_by_id, _now)
            if topic is not None:
                scored.append(topic)
                _log.debug(
                    "Scored topic '%s': freq=%d recency=%.2f debt=%.2f type=%s",
                    topic.name, topic.frequency, topic.recency_score,
                    topic.debt_score, topic.curiosity_type.value,
                )

        _log.info("Graph built: %d topic(s) from %d signal(s)", len(scored), signal_count)
        return CuriosityGraph(
            topics=tuple(scored),
            signal_count=signal_count,
            source_breakdown=source_breakdown,
        )

    def _sample_signals(self, signals: list[RawSignal]) -> list[RawSignal]:
        """Return at most _MAX_SIGNALS_FOR_EXTRACTION signals.

        Strategy: sort by recency (newest first), then cap per domain so one
        noisy domain (e.g. GitHub, Google Docs) doesn't dominate the sample.
        """
        sorted_signals = sorted(signals, key=lambda s: s.timestamp, reverse=True)

        domain_counts: dict[str, int] = defaultdict(int)
        sampled: list[RawSignal] = []

        for sig in sorted_signals:
            if len(sampled) >= _MAX_SIGNALS_FOR_EXTRACTION:
                break
            domain = ""
            if sig.url:
                try:
                    domain = urlparse(sig.url).netloc
                except Exception:
                    pass
            if domain and domain_counts[domain] >= _MAX_SIGNALS_PER_DOMAIN:
                continue
            if domain:
                domain_counts[domain] += 1
            sampled.append(sig)

        return sampled

    @staticmethod
    def _signal_weight(signal: RawSignal) -> float:
        """Return a weight [1.0, 2.0] reflecting how strongly this signal
        indicates genuine curiosity. Explicit intent (search query, ChatGPT
        question, stuck YouTube video) scores higher than passive browsing.
        """
        meta = signal.metadata or {}
        base = _BASE_SIGNAL_WEIGHTS.get(signal.source, 1.0)
        if signal.source == SignalSource.GOOGLE_TAKEOUT:
            if meta.get("is_search_query"):
                return 2.0  # explicit search query = direct curiosity intent
            if meta.get("is_stuck") or meta.get("visit_count", 1) >= 3:
                return 1.5  # repeatedly visited = strong implicit interest
        if signal.source == SignalSource.YOUTUBE_TAKEOUT:
            if meta.get("is_stuck"):
                return 2.0  # stuck video = unresolved curiosity
            if meta.get("rewatch_count", 1) > 1:
                return 1.5  # rewatched = deeper interest
        return base

    def _score_topic(
        self,
        raw: RawTopicData,
        signals_by_id: dict[str, RawSignal],
        now: datetime,
    ) -> Topic | None:
        matched = [signals_by_id[sid] for sid in raw["signal_ids"] if sid in signals_by_id]
        if not matched:
            return None

        timestamps = [s.timestamp for s in matched]
        first_seen = min(timestamps)
        last_seen = max(timestamps)

        frequency = len(matched)
        span_days = max(0, (last_seen - first_seen).days)

        # Weighted frequency: explicit curiosity signals count more than passive ones.
        # Stored in depth_score (always 0.0 otherwise) for use in composite_score().
        weighted_frequency = sum(self._signal_weight(s) for s in matched)

        # Recency: exponential decay from last_seen to now
        days_since = max(0.0, (now - last_seen).total_seconds() / 86400)
        recency_score = min(1.0, math.exp(-self._lambda * days_since))

        # Debt and type — use raw frequency for threshold (not weighted)
        is_recurring = (
            frequency >= self._debt_threshold_occurrences
            and span_days >= self._debt_threshold_days
        )
        if is_recurring:
            debt_score = min(
                5.0,
                (frequency / self._debt_threshold_occurrences)
                * (span_days / self._debt_threshold_days),
            )
            curiosity_type = CuriosityType.RECURRING
        else:
            debt_score = 0.0
            curiosity_type = CuriosityType.SHALLOW

        source_types = frozenset(s.source for s in matched)

        # Representative signal contents: ChatGPT questions and search queries
        # first (most descriptive of intent), then by content length. Capped at
        # 2 × 200 chars so they fit in the newsletter prompt without bloat.
        sorted_for_samples = sorted(
            matched,
            key=lambda s: (0 if s.source in _EXPLICIT_SOURCES else 1, -len(s.content)),
        )
        signal_samples = tuple(s.content[:200] for s in sorted_for_samples[:2])

        return Topic(
            name=raw["name"],
            aliases=raw["aliases"],
            signal_ids=[s.id for s in matched],
            source_types=source_types,
            first_seen=first_seen,
            last_seen=last_seen,
            frequency=frequency,
            recency_score=recency_score,
            depth_score=weighted_frequency,  # repurposed: weighted signal count
            debt_score=debt_score,
            curiosity_type=curiosity_type,
            signal_samples=signal_samples,
        )

    def _compute_source_breakdown(
        self, signals: list[RawSignal]
    ) -> dict[str, int]:
        breakdown: dict[str, int] = {}
        for sig in signals:
            key = sig.source.value
            breakdown[key] = breakdown.get(key, 0) + 1
        return breakdown
