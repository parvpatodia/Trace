"""
Unit tests for trace/graph/builder.py — CuriosityGraphBuilder.

CuriosityGraphBuilder takes signals + a TopicExtractor, runs two passes:
  Pass 1: extract raw topics (name, aliases, signal_ids) via TopicExtractor
  Pass 2: score each topic → recency_score, frequency, debt_score,
          curiosity_type → build CuriosityGraph

A StubExtractor is used in all tests — no API calls.

Test categories:
  1. Constructor validation: None extractor raises
  2. Empty input: no signals → empty CuriosityGraph
  3. Topic scoring — frequency: correctly counts signal occurrences
  4. Topic scoring — recency: exponential decay by days since last_seen
  5. Topic scoring — debt: recurring topics with long span get debt_score > 0
  6. Curiosity type classification: SHALLOW / RECURRING rules
  7. CuriosityGraph structure: signal_count, source_breakdown, built_at
  8. first_seen / last_seen: min/max of signal timestamps
  9. Multiple topics: graph contains all scored topics
  10. Signals not matched to any topic: gracefully handled
  11. Composite score: recency * frequency + debt * 2.0
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from trace.graph.builder import CuriosityGraphBuilder
from trace.graph.extractor import RawTopicData, TopicExtractor
from trace.models import CuriosityGraph, CuriosityType, RawSignal, SignalSource, Topic


# ── Helpers ───────────────────────────────────────────────────────────────────

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def make_signal(
    content: str = "Explain transformer self-attention mechanisms in detail",
    source: SignalSource = SignalSource.GOOGLE_TAKEOUT,
    days_ago: float = 0.0,
    signal_id: str | None = None,
) -> RawSignal:
    ts = _NOW - timedelta(days=days_ago)
    sig = RawSignal(source=source, content=content, timestamp=ts)
    if signal_id is not None:
        sig = sig.model_copy(update={"id": signal_id})
    return sig


class StubExtractor:
    """Fake TopicExtractor — returns a pre-programmed list of RawTopicData."""

    def __init__(self, topics: list[RawTopicData]) -> None:
        self._topics = topics

    async def extract(self, signals: list[RawSignal]) -> list[RawTopicData]:
        return list(self._topics)


# ── Constructor validation ────────────────────────────────────────────────────

class TestConstructorValidation:
    def test_none_extractor_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="extractor"):
            CuriosityGraphBuilder(extractor=None)  # type: ignore[arg-type]

    def test_valid_extractor_accepted(self) -> None:
        extractor = StubExtractor([])
        builder = CuriosityGraphBuilder(extractor=extractor)
        assert builder is not None

    def test_custom_half_life_accepted(self) -> None:
        extractor = StubExtractor([])
        builder = CuriosityGraphBuilder(extractor=extractor, half_life_days=7)
        assert builder is not None

    def test_zero_half_life_raises_value_error(self) -> None:
        extractor = StubExtractor([])
        with pytest.raises(ValueError, match="half_life_days"):
            CuriosityGraphBuilder(extractor=extractor, half_life_days=0)

    def test_negative_half_life_raises_value_error(self) -> None:
        extractor = StubExtractor([])
        with pytest.raises(ValueError, match="half_life_days"):
            CuriosityGraphBuilder(extractor=extractor, half_life_days=-1)


# ── Empty input ───────────────────────────────────────────────────────────────

class TestEmptyInput:
    async def test_empty_signals_returns_empty_graph(self) -> None:
        builder = CuriosityGraphBuilder(extractor=StubExtractor([]))
        graph = await builder.build([], now=_NOW)
        assert isinstance(graph, CuriosityGraph)
        assert graph.is_empty()

    async def test_empty_graph_has_zero_signal_count(self) -> None:
        builder = CuriosityGraphBuilder(extractor=StubExtractor([]))
        graph = await builder.build([], now=_NOW)
        assert graph.signal_count == 0

    async def test_empty_graph_has_empty_source_breakdown(self) -> None:
        builder = CuriosityGraphBuilder(extractor=StubExtractor([]))
        graph = await builder.build([], now=_NOW)
        assert graph.source_breakdown == {}

    async def test_extractor_returning_no_topics_gives_empty_graph(self) -> None:
        sigs = [make_signal()]
        builder = CuriosityGraphBuilder(extractor=StubExtractor([]))
        graph = await builder.build(sigs, now=_NOW)
        assert graph.is_empty()


# ── Frequency scoring ─────────────────────────────────────────────────────────

class TestFrequencyScoring:
    async def test_single_signal_gives_frequency_one(self) -> None:
        sig = make_signal(signal_id="s1")
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ["s1"]}
        ])
        graph = await CuriosityGraphBuilder(extractor=extractor).build([sig], now=_NOW)
        assert graph.topics[0].frequency == 1

    async def test_three_signals_give_frequency_three(self) -> None:
        sigs = [make_signal(signal_id=f"s{i}") for i in range(3)]
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ["s0", "s1", "s2"]}
        ])
        graph = await CuriosityGraphBuilder(extractor=extractor).build(sigs, now=_NOW)
        assert graph.topics[0].frequency == 3

    async def test_frequency_counts_only_matched_signals(self) -> None:
        sigs = [make_signal(signal_id=f"s{i}") for i in range(5)]
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ["s0", "s2"]}
            # s1, s3, s4 not matched to any topic
        ])
        graph = await CuriosityGraphBuilder(extractor=extractor).build(sigs, now=_NOW)
        assert graph.topics[0].frequency == 2


# ── Recency scoring ───────────────────────────────────────────────────────────

class TestRecencyScoring:
    async def test_signal_from_today_has_high_recency(self) -> None:
        sig = make_signal(days_ago=0.0, signal_id="s1")
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ["s1"]}
        ])
        graph = await CuriosityGraphBuilder(extractor=extractor, half_life_days=14).build([sig], now=_NOW)
        # Signal from today → recency ≈ 1.0 (within a few hours, so > 0.99)
        assert graph.topics[0].recency_score > 0.99

    async def test_signal_at_one_half_life_has_recency_near_half(self) -> None:
        sig = make_signal(days_ago=14.0, signal_id="s1")
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ["s1"]}
        ])
        graph = await CuriosityGraphBuilder(extractor=extractor, half_life_days=14).build([sig], now=_NOW)
        # After exactly 1 half-life (14 days), score ≈ 0.5
        score = graph.topics[0].recency_score
        assert 0.45 < score < 0.55

    async def test_signal_at_two_half_lives_has_recency_near_quarter(self) -> None:
        sig = make_signal(days_ago=28.0, signal_id="s1")
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ["s1"]}
        ])
        graph = await CuriosityGraphBuilder(extractor=extractor, half_life_days=14).build([sig], now=_NOW)
        score = graph.topics[0].recency_score
        assert 0.20 < score < 0.30

    async def test_recency_score_is_bounded_0_to_1(self) -> None:
        sig = make_signal(days_ago=0.0, signal_id="s1")
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ["s1"]}
        ])
        graph = await CuriosityGraphBuilder(extractor=extractor).build([sig], now=_NOW)
        score = graph.topics[0].recency_score
        assert 0.0 <= score <= 1.0

    async def test_recency_uses_most_recent_signal(self) -> None:
        # Two signals: one old, one recent. Recency should use the recent one.
        sig_old = make_signal(days_ago=30.0, signal_id="s_old")
        sig_new = make_signal(days_ago=1.0, signal_id="s_new")
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ["s_old", "s_new"]}
        ])
        graph = await CuriosityGraphBuilder(extractor=extractor, half_life_days=14).build(
            [sig_old, sig_new], now=_NOW
        )
        # Most recent signal is 1 day ago → high recency
        assert graph.topics[0].recency_score > 0.9


# ── Debt scoring ──────────────────────────────────────────────────────────────

class TestDebtScoring:
    async def test_single_occurrence_has_no_debt(self) -> None:
        sig = make_signal(signal_id="s1")
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ["s1"]}
        ])
        graph = await CuriosityGraphBuilder(extractor=extractor).build([sig], now=_NOW)
        assert graph.topics[0].debt_score == 0.0

    async def test_recurring_topic_with_long_span_has_positive_debt(self) -> None:
        # 3 signals spread over 21 days → debt > 0
        sigs = [
            make_signal(days_ago=21.0, signal_id="s0"),
            make_signal(days_ago=10.0, signal_id="s1"),
            make_signal(days_ago=0.0, signal_id="s2"),
        ]
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ["s0", "s1", "s2"]}
        ])
        graph = await CuriosityGraphBuilder(
            extractor=extractor, debt_threshold_days=7, debt_threshold_occurrences=3
        ).build(sigs, now=_NOW)
        assert graph.topics[0].debt_score > 0.0

    async def test_high_frequency_low_span_has_no_debt(self) -> None:
        # 5 signals but all within 1 day → not a curiosity debt
        sigs = [make_signal(days_ago=0.1 * i, signal_id=f"s{i}") for i in range(5)]
        ids = [s.id for s in sigs]
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ids}
        ])
        graph = await CuriosityGraphBuilder(
            extractor=extractor, debt_threshold_days=7
        ).build(sigs, now=_NOW)
        assert graph.topics[0].debt_score == 0.0


# ── Curiosity type classification ─────────────────────────────────────────────

class TestCuriosityTypeClassification:
    async def test_single_occurrence_is_shallow(self) -> None:
        sig = make_signal(signal_id="s1")
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ["s1"]}
        ])
        graph = await CuriosityGraphBuilder(extractor=extractor).build([sig], now=_NOW)
        assert graph.topics[0].curiosity_type == CuriosityType.SHALLOW

    async def test_two_occurrences_same_day_is_shallow(self) -> None:
        sigs = [make_signal(days_ago=0.0, signal_id=f"s{i}") for i in range(2)]
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ["s0", "s1"]}
        ])
        graph = await CuriosityGraphBuilder(extractor=extractor).build(sigs, now=_NOW)
        assert graph.topics[0].curiosity_type == CuriosityType.SHALLOW

    async def test_threshold_occurrences_over_threshold_days_is_recurring(self) -> None:
        # 3 occurrences over 10 days → RECURRING
        sigs = [
            make_signal(days_ago=10.0, signal_id="s0"),
            make_signal(days_ago=5.0, signal_id="s1"),
            make_signal(days_ago=0.0, signal_id="s2"),
        ]
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ["s0", "s1", "s2"]}
        ])
        graph = await CuriosityGraphBuilder(
            extractor=extractor,
            debt_threshold_occurrences=3,
            debt_threshold_days=7,
        ).build(sigs, now=_NOW)
        assert graph.topics[0].curiosity_type == CuriosityType.RECURRING

    async def test_below_threshold_occurrences_is_not_recurring(self) -> None:
        # 2 occurrences over 30 days — span is long but count is below threshold=3
        sigs = [
            make_signal(days_ago=30.0, signal_id="s0"),
            make_signal(days_ago=0.0, signal_id="s1"),
        ]
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ["s0", "s1"]}
        ])
        graph = await CuriosityGraphBuilder(
            extractor=extractor,
            debt_threshold_occurrences=3,
        ).build(sigs, now=_NOW)
        assert graph.topics[0].curiosity_type == CuriosityType.SHALLOW


# ── first_seen / last_seen ────────────────────────────────────────────────────

class TestTemporalBounds:
    async def test_first_seen_is_earliest_signal_timestamp(self) -> None:
        sig_early = make_signal(days_ago=10.0, signal_id="early")
        sig_late = make_signal(days_ago=1.0, signal_id="late")
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ["early", "late"]}
        ])
        graph = await CuriosityGraphBuilder(extractor=extractor).build([sig_early, sig_late], now=_NOW)
        topic = graph.topics[0]
        assert topic.first_seen == sig_early.timestamp

    async def test_last_seen_is_latest_signal_timestamp(self) -> None:
        sig_early = make_signal(days_ago=10.0, signal_id="early")
        sig_late = make_signal(days_ago=1.0, signal_id="late")
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ["early", "late"]}
        ])
        graph = await CuriosityGraphBuilder(extractor=extractor).build([sig_early, sig_late], now=_NOW)
        topic = graph.topics[0]
        assert topic.last_seen == sig_late.timestamp

    async def test_single_signal_first_and_last_same(self) -> None:
        sig = make_signal(days_ago=5.0, signal_id="s1")
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ["s1"]}
        ])
        graph = await CuriosityGraphBuilder(extractor=extractor).build([sig], now=_NOW)
        topic = graph.topics[0]
        assert topic.first_seen == topic.last_seen == sig.timestamp


# ── CuriosityGraph structure ──────────────────────────────────────────────────

class TestCuriosityGraphStructure:
    async def test_graph_signal_count_matches_input(self) -> None:
        sigs = [make_signal(signal_id=f"s{i}") for i in range(5)]
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ["s0", "s1"]}
        ])
        graph = await CuriosityGraphBuilder(extractor=extractor).build(sigs, now=_NOW)
        assert graph.signal_count == 5

    async def test_source_breakdown_counts_per_source(self) -> None:
        sigs = [
            make_signal(source=SignalSource.GOOGLE_TAKEOUT, signal_id="g1"),
            make_signal(source=SignalSource.GOOGLE_TAKEOUT, signal_id="g2"),
            make_signal(source=SignalSource.CHATGPT_EXPORT, signal_id="c1"),
        ]
        extractor = StubExtractor([])
        graph = await CuriosityGraphBuilder(extractor=extractor).build(sigs, now=_NOW)
        assert graph.source_breakdown.get("google_takeout") == 2
        assert graph.source_breakdown.get("chatgpt_export") == 1

    async def test_graph_built_at_is_utc_aware(self) -> None:
        extractor = StubExtractor([])
        graph = await CuriosityGraphBuilder(extractor=extractor).build([], now=_NOW)
        assert graph.built_at.tzinfo is not None

    async def test_multiple_topics_all_present_in_graph(self) -> None:
        sigs = [make_signal(signal_id=f"s{i}") for i in range(4)]
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ["s0", "s1"]},
            {"name": "reinforcement learning", "aliases": [], "signal_ids": ["s2", "s3"]},
        ])
        graph = await CuriosityGraphBuilder(extractor=extractor).build(sigs, now=_NOW)
        assert len(graph.topics) == 2
        names = {t.name for t in graph.topics}
        assert "transformers" in names
        assert "reinforcement learning" in names

    async def test_source_types_populated_on_topic(self) -> None:
        sigs = [
            make_signal(source=SignalSource.GOOGLE_TAKEOUT, signal_id="g1"),
            make_signal(source=SignalSource.CHATGPT_EXPORT, signal_id="c1"),
        ]
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ["g1", "c1"]}
        ])
        graph = await CuriosityGraphBuilder(extractor=extractor).build(sigs, now=_NOW)
        topic = graph.topics[0]
        assert SignalSource.GOOGLE_TAKEOUT in topic.source_types
        assert SignalSource.CHATGPT_EXPORT in topic.source_types


# ── Composite score ───────────────────────────────────────────────────────────

class TestCompositeScore:
    async def test_composite_score_formula(self) -> None:
        # Verify: composite = recency_score * frequency + debt_score * 2.0
        sig = make_signal(days_ago=0.0, signal_id="s1")
        extractor = StubExtractor([
            {"name": "transformers", "aliases": [], "signal_ids": ["s1"]}
        ])
        graph = await CuriosityGraphBuilder(extractor=extractor).build([sig], now=_NOW)
        t = graph.topics[0]
        expected = t.recency_score * t.frequency + t.debt_score * 2.0
        assert abs(t.composite_score() - expected) < 1e-9

    async def test_top_n_returns_highest_scoring_topics(self) -> None:
        # Three signals: one very recent, one very old, one medium
        sigs = [
            make_signal(days_ago=0.0, signal_id="s_new"),
            make_signal(days_ago=60.0, signal_id="s_old"),
            make_signal(days_ago=7.0, signal_id="s_mid"),
        ]
        extractor = StubExtractor([
            {"name": "hot topic", "aliases": [], "signal_ids": ["s_new"]},
            {"name": "old topic", "aliases": [], "signal_ids": ["s_old"]},
            {"name": "medium topic", "aliases": [], "signal_ids": ["s_mid"]},
        ])
        graph = await CuriosityGraphBuilder(extractor=extractor, half_life_days=14).build(sigs, now=_NOW)
        top2 = graph.top_n(2)
        names = [t.name for t in top2]
        # hot topic and medium topic should rank above old topic
        assert "hot topic" in names
        assert "old topic" not in names
