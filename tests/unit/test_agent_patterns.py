"""Unit tests for trace/agent/patterns.py — pure logic, no external I/O."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trace.agent.patterns import (
    PatternEvent,
    detect_emerging_interests,
    detect_subscription_debt,
    detect_bridge_topic_shift,
    run_all_detectors,
)
from trace.models import CuriosityGraph, CuriosityType, SignalSource, Topic


def _make_topic(
    name: str,
    recency: float = 0.5,
    depth: float = 1.0,
    debt: float = 0.0,
    freq: int = 3,
    ctype: CuriosityType = CuriosityType.SHALLOW,
    sources: frozenset | None = None,
    span_days_val: int = 0,
) -> Topic:
    now = datetime.now(timezone.utc)
    first = now - timedelta(days=span_days_val) if span_days_val else now
    return Topic(
        name=name,
        frequency=freq,
        recency_score=recency,
        depth_score=depth,
        debt_score=debt,
        curiosity_type=ctype,
        source_types=sources or frozenset({SignalSource.CHROME_HISTORY}),
        first_seen=first,
        last_seen=now,
    )


def _make_graph(*topics: Topic, signal_count: int = 10) -> CuriosityGraph:
    return CuriosityGraph(topics=tuple(topics), signal_count=signal_count)


# ── PatternEvent ───────────────────────────────────────────────────────────────

class TestPatternEvent:
    def test_str_repr(self):
        ev = PatternEvent(pattern_type="emerging_interest", topic_name="ai", score=2.5)
        assert "emerging_interest" in str(ev)
        assert "ai" in str(ev)
        assert "2.500" in str(ev)

    def test_frozen_fields(self):
        ev = PatternEvent(pattern_type="x", topic_name="y", score=1.0)
        assert ev.pattern_type == "x"
        assert ev.score == 1.0

    def test_default_metadata_and_supporting(self):
        ev = PatternEvent(pattern_type="t", topic_name="n", score=0.5)
        assert ev.supporting_topics == []
        assert ev.metadata == {}


# ── detect_emerging_interests ─────────────────────────────────────────────────

class TestEmergingInterests:
    def test_new_topic_flagged_when_high_recency(self):
        t = _make_topic("ai safety", recency=0.9, depth=2.0)
        graph = _make_graph(t)
        events = detect_emerging_interests(graph, previous_graph=None)
        assert any(e.topic_name == "ai safety" for e in events)

    def test_low_recency_not_flagged(self):
        t = _make_topic("cooking", recency=0.2, depth=0.1)
        graph = _make_graph(t)
        events = detect_emerging_interests(graph, previous_graph=None)
        assert not any(e.topic_name == "cooking" for e in events)

    def test_score_surge_detected(self):
        old_topic = _make_topic("robotics", recency=0.2, depth=0.5)
        old_graph = _make_graph(old_topic)
        new_topic = _make_topic("robotics", recency=0.9, depth=5.0)
        new_graph = _make_graph(new_topic)
        events = detect_emerging_interests(new_graph, previous_graph=old_graph)
        assert any(e.topic_name == "robotics" and e.pattern_type == "emerging_interest" for e in events)

    def test_score_surge_below_threshold_not_flagged(self):
        old_topic = _make_topic("music", recency=0.5, depth=2.0)
        old_graph = _make_graph(old_topic)
        # New score only 1.5× old — below 2.0× threshold.
        new_topic = _make_topic("music", recency=0.6, depth=2.5)
        new_graph = _make_graph(new_topic)
        events = detect_emerging_interests(new_graph, old_graph)
        surge_events = [e for e in events if e.metadata.get("reason") == "score_surge"]
        assert not surge_events

    def test_brand_new_topic_flagged(self):
        old_graph = _make_graph(_make_topic("physics"))
        new_topic = _make_topic("diffusion policy", recency=0.8, depth=3.0)
        new_graph = _make_graph(_make_topic("physics"), new_topic)
        events = detect_emerging_interests(new_graph, old_graph)
        assert any(e.topic_name == "diffusion policy" for e in events)

    def test_resolved_topic_excluded(self):
        t = _make_topic("python", recency=0.95, depth=5.0, ctype=CuriosityType.RESOLVED)
        graph = _make_graph(t)
        events = detect_emerging_interests(graph, previous_graph=None)
        assert not any(e.topic_name == "python" for e in events)

    def test_min_absolute_score_filter(self):
        # Tiny scores should not trigger even on big ratio.
        old_topic = _make_topic("obscure", recency=0.01, depth=0.01)
        old_graph = _make_graph(old_topic)
        new_topic = _make_topic("obscure", recency=0.02, depth=0.02)
        new_graph = _make_graph(new_topic)
        events = detect_emerging_interests(new_graph, old_graph, min_absolute_score=0.3)
        assert not any(e.topic_name == "obscure" for e in events)


# ── detect_subscription_debt ──────────────────────────────────────────────────

class TestSubscriptionDebt:
    def test_high_debt_topic_flagged(self):
        t = _make_topic(
            "ml papers",
            debt=0.8,
            ctype=CuriosityType.RECURRING,
            span_days_val=14,
        )
        graph = _make_graph(t)
        events = detect_subscription_debt(graph)
        assert any(e.topic_name == "ml papers" for e in events)

    def test_low_debt_not_flagged(self):
        t = _make_topic("hiking", debt=0.2, ctype=CuriosityType.SHALLOW, span_days_val=3)
        graph = _make_graph(t)
        events = detect_subscription_debt(graph, debt_threshold=0.5)
        assert not any(e.topic_name == "hiking" for e in events)

    def test_short_span_not_flagged(self):
        # debt_score is high but span_days < 7 — should be skipped.
        t = _make_topic("news", debt=0.9, ctype=CuriosityType.RECURRING, span_days_val=3)
        graph = _make_graph(t)
        events = detect_subscription_debt(graph)
        assert not any(e.topic_name == "news" for e in events)

    def test_sorted_by_debt_desc(self):
        t1 = _make_topic("a", debt=0.9, ctype=CuriosityType.RECURRING, span_days_val=10)
        t2 = _make_topic("b", debt=0.6, ctype=CuriosityType.RECURRING, span_days_val=10)
        graph = _make_graph(t1, t2)
        events = detect_subscription_debt(graph)
        names = [e.topic_name for e in events]
        assert names.index("a") < names.index("b")

    def test_empty_graph(self):
        graph = CuriosityGraph()
        assert detect_subscription_debt(graph) == []


# ── detect_bridge_topic_shift ─────────────────────────────────────────────────

class TestBridgeTopicShift:
    def test_new_multi_domain_topic_flagged(self):
        t1 = _make_topic(
            "ai + music",
            sources=frozenset({SignalSource.CHROME_HISTORY, SignalSource.YOUTUBE_TAKEOUT}),
        )
        old_graph = _make_graph(
            _make_topic("ai"), _make_topic("music"), _make_topic("x"), _make_topic("y")
        )
        new_graph = _make_graph(
            _make_topic("ai"), _make_topic("music"), t1, _make_topic("x"), _make_topic("y")
        )
        events = detect_bridge_topic_shift(new_graph, old_graph)
        assert any(e.topic_name == "ai + music" for e in events)

    def test_existing_multi_domain_topic_not_reflagged(self):
        t1 = _make_topic(
            "ai + music",
            sources=frozenset({SignalSource.CHROME_HISTORY, SignalSource.YOUTUBE_TAKEOUT}),
        )
        old_graph = _make_graph(
            _make_topic("ai"), _make_topic("music"), t1, _make_topic("x")
        )
        new_graph = _make_graph(
            _make_topic("ai"), _make_topic("music"), t1, _make_topic("x")
        )
        events = detect_bridge_topic_shift(new_graph, old_graph)
        assert not any(e.topic_name == "ai + music" for e in events)

    def test_small_graph_skipped(self):
        graph = _make_graph(_make_topic("a"), _make_topic("b"))
        events = detect_bridge_topic_shift(graph, None, min_cluster_size=2)
        assert events == []


# ── run_all_detectors ─────────────────────────────────────────────────────────

class TestRunAllDetectors:
    def test_returns_list(self):
        graph = _make_graph(_make_topic("foo", recency=0.9, depth=3.0))
        result = run_all_detectors(graph, None)
        assert isinstance(result, list)

    def test_deduplication(self):
        # Same (pattern_type, topic_name) pair should appear once.
        t = _make_topic("ai", recency=0.95, depth=5.0)
        graph = _make_graph(t)
        events = run_all_detectors(graph, None)
        seen: set[tuple[str, str]] = set()
        for e in events:
            key = (e.pattern_type, e.topic_name)
            assert key not in seen, f"Duplicate pattern event: {key}"
            seen.add(key)

    def test_sorted_by_score_desc(self):
        t1 = _make_topic("a", recency=0.9, depth=4.0)
        t2 = _make_topic("b", recency=0.6, depth=1.0)
        graph = _make_graph(t1, t2)
        events = run_all_detectors(graph, None)
        scores = [e.score for e in events]
        assert scores == sorted(scores, reverse=True)

    def test_empty_graph_no_crash(self):
        graph = CuriosityGraph()
        events = run_all_detectors(graph, None)
        assert isinstance(events, list)
