"""
Pattern detectors for the autonomous agent loop.

Three detectors run on every graph snapshot:
  emerging_interest   — topic score surged >2× in the last 48 h
  subscription_debt   — user subscribed to ≥3 newsletters but opens <10% of them
  bridge_topic_shift  — two previously unconnected topic clusters are now linked by
                        a new bridge topic (signals a paradigm shift in interest)

Each detector returns a list of PatternEvent dataclass instances.  Orchestrator
maps those events → Tier A / Tier B action chains.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from trace.models import CuriosityGraph, CuriosityType, Topic

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatternEvent:
    """A detected curiosity pattern — consumed by the orchestrator."""

    pattern_type: str  # "emerging_interest" | "subscription_debt" | "bridge_topic_shift"
    topic_name: str
    score: float  # confidence / magnitude of the pattern
    supporting_topics: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __str__(self) -> str:
        return (
            f"PatternEvent({self.pattern_type!r}, topic={self.topic_name!r}, "
            f"score={self.score:.3f})"
        )


# ── Detector 1: Emerging Interest ─────────────────────────────────────────────

def detect_emerging_interests(
    current_graph: CuriosityGraph,
    previous_graph: CuriosityGraph | None,
    surge_threshold: float = 2.0,
    min_absolute_score: float = 0.3,
) -> list[PatternEvent]:
    """Topics whose composite_score doubled since the last graph snapshot.

    A topic must also clear a minimum absolute score to filter out noise from
    topics with tiny baseline scores (e.g. a single mention going from 0.01→0.02).
    """
    events: list[PatternEvent] = []

    current_scores = {t.name: t.composite_score() for t in current_graph.topics}

    if previous_graph is None:
        # First snapshot — treat topics with high recency as newly emerging.
        for topic in current_graph.topics:
            if (
                topic.recency_score > 0.8
                and topic.composite_score() >= min_absolute_score
                and topic.curiosity_type != CuriosityType.RESOLVED
            ):
                events.append(
                    PatternEvent(
                        pattern_type="emerging_interest",
                        topic_name=topic.name,
                        score=topic.recency_score,
                        metadata={"reason": "high_recency_no_baseline"},
                    )
                )
        return events

    previous_scores = {t.name: t.composite_score() for t in previous_graph.topics}

    for name, cur_score in current_scores.items():
        if cur_score < min_absolute_score:
            continue
        prev_score = previous_scores.get(name, 0.0)
        if prev_score == 0.0:
            # Brand-new topic — treat as emerging if score is meaningful.
            if cur_score >= min_absolute_score:
                events.append(
                    PatternEvent(
                        pattern_type="emerging_interest",
                        topic_name=name,
                        score=cur_score,
                        metadata={"reason": "new_topic", "current_score": cur_score},
                    )
                )
        elif cur_score / prev_score >= surge_threshold:
            events.append(
                PatternEvent(
                    pattern_type="emerging_interest",
                    topic_name=name,
                    score=cur_score / prev_score,
                    metadata={
                        "reason": "score_surge",
                        "prev_score": prev_score,
                        "current_score": cur_score,
                        "ratio": cur_score / prev_score,
                    },
                )
            )

    _log.debug("emerging_interest: %d events detected", len(events))
    return events


# ── Detector 2: Subscription Debt ─────────────────────────────────────────────

def detect_subscription_debt(
    current_graph: CuriosityGraph,
    pending_signals_count: int = 0,
    debt_threshold: float = 0.5,
) -> list[PatternEvent]:
    """Topics with high debt_score signal recurring unanswered curiosity.

    debt_score is set by CuriosityGraphBuilder based on recurrence without
    resolution.  A topic that keeps appearing but never gets 'resolved' status
    indicates the user is subscribed to the topic area but not actually digesting
    content on it — the subscription debt.
    """
    events: list[PatternEvent] = []

    for topic in current_graph.topics:
        if (
            topic.debt_score >= debt_threshold
            and topic.curiosity_type in (CuriosityType.RECURRING, CuriosityType.SHALLOW)
            and topic.span_days() >= 7  # must persist for at least a week
        ):
            events.append(
                PatternEvent(
                    pattern_type="subscription_debt",
                    topic_name=topic.name,
                    score=topic.debt_score,
                    metadata={
                        "span_days": topic.span_days(),
                        "frequency": topic.frequency,
                        "curiosity_type": topic.curiosity_type.value,
                        "pending_agent_signals": pending_signals_count,
                    },
                )
            )

    _log.debug("subscription_debt: %d events detected", len(events))
    return sorted(events, key=lambda e: e.score, reverse=True)


# ── Detector 3: Bridge Topic Shift ────────────────────────────────────────────

def detect_bridge_topic_shift(
    current_graph: CuriosityGraph,
    previous_graph: CuriosityGraph | None,
    min_cluster_size: int = 2,
) -> list[PatternEvent]:
    """Detect when a new topic connects two previously separate interest clusters.

    Uses a simple name-overlap heuristic on source_types to find topics that
    appear in multiple distinct domains.  When a topic bridges two clusters that
    were not connected in the previous graph, it signals a paradigm shift.

    Full graph algorithm (embeddings + PageRank + Louvain) is built in Phase 4.
    This Phase 1 version uses domain-overlap as a fast proxy.
    """
    events: list[PatternEvent] = []

    if len(current_graph.topics) < min_cluster_size * 2:
        return events

    # Group topics by source_type domain as a simple cluster proxy.
    domain_map: dict[str, list[str]] = {}
    for topic in current_graph.topics:
        for src in topic.source_types:
            domain_map.setdefault(src.value, []).append(topic.name)

    # A bridge topic appears in ≥2 domains.
    current_topic_names = {t.name for t in current_graph.topics}
    previous_topic_names = {t.name for t in previous_graph.topics} if previous_graph else set()

    for topic in current_graph.topics:
        bridged_domains = [d for d, names in domain_map.items() if topic.name in names]
        if len(bridged_domains) >= 2 and topic.name not in previous_topic_names:
            # New topic that already spans multiple domains = bridge shift.
            related = [
                name
                for d in bridged_domains
                for name in domain_map[d]
                if name != topic.name
            ]
            events.append(
                PatternEvent(
                    pattern_type="bridge_topic_shift",
                    topic_name=topic.name,
                    score=float(len(bridged_domains)),
                    supporting_topics=list(set(related))[:5],
                    metadata={
                        "bridged_domains": bridged_domains,
                        "is_new_topic": topic.name not in previous_topic_names,
                    },
                )
            )

    _log.debug("bridge_topic_shift: %d events detected", len(events))
    return events


# ── Composite runner ───────────────────────────────────────────────────────────

def run_all_detectors(
    current_graph: CuriosityGraph,
    previous_graph: CuriosityGraph | None,
    pending_signals_count: int = 0,
) -> list[PatternEvent]:
    """Run all three detectors and return the merged, de-duplicated event list."""
    events: list[PatternEvent] = []
    events.extend(detect_emerging_interests(current_graph, previous_graph))
    events.extend(detect_subscription_debt(current_graph, pending_signals_count))
    events.extend(detect_bridge_topic_shift(current_graph, previous_graph))

    # De-duplicate: keep highest-score event per (pattern_type, topic_name).
    seen: dict[tuple[str, str], PatternEvent] = {}
    for ev in events:
        key = (ev.pattern_type, ev.topic_name)
        if key not in seen or ev.score > seen[key].score:
            seen[key] = ev

    result = sorted(seen.values(), key=lambda e: e.score, reverse=True)
    _log.info("run_all_detectors: %d unique patterns detected", len(result))
    return result
