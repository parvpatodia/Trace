"""
Demo seed — injects a realistic curiosity profile for live hackathon demos.

Profile: ML researcher / robotics engineer
  - Deep interest in diffusion policy + robot learning (recurring, high debt)
  - Emerging interest in nuPlan (new, high recency)
  - Foundational interest in Karpathy / neural networks (deep, resolved)
  - Bridge topic: "embodied AI" (connects robotics + LLMs)
  - Subscription debt: ML papers newsletter (subscribed but not reading)

This seed populates _PROFILE_CACHE and _redis_store so that the MCP tools
and scheduler can demonstrate the full loop in < 5 minutes without real data.

Usage:
  POST /demo/seed          → inject the seed profile
  GET  /demo/seed/status   → check if seed is active
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from trace.models import (
    CuriosityGraph,
    CuriosityType,
    SignalSource,
    Topic,
)

_log = logging.getLogger(__name__)

_SEED_PROFILE_ID = "demo"


def _ago(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


def build_demo_graph() -> CuriosityGraph:
    """Build a realistic curiosity graph for the Parv/ML-robotics demo persona."""
    topics = (
        Topic(
            name="diffusion policy",
            frequency=42,
            recency_score=0.95,
            depth_score=15.0,
            debt_score=0.72,
            curiosity_type=CuriosityType.RECURRING,
            source_types=frozenset({
                SignalSource.CHROME_HISTORY,
                SignalSource.CHATGPT_EXPORT,
                SignalSource.YOUTUBE_TAKEOUT,
            }),
            first_seen=_ago(45),
            last_seen=_ago(1),
            signal_samples=(
                "how does diffusion policy work for robot manipulation?",
                "diffusion policy vs imitation learning comparison",
            ),
        ),
        Topic(
            name="robot learning",
            frequency=38,
            recency_score=0.88,
            depth_score=12.0,
            debt_score=0.65,
            curiosity_type=CuriosityType.RECURRING,
            source_types=frozenset({
                SignalSource.CHROME_HISTORY,
                SignalSource.CHATGPT_EXPORT,
            }),
            first_seen=_ago(60),
            last_seen=_ago(2),
            signal_samples=(
                "imitation learning for robot manipulation",
                "LeRobot huggingface tutorial",
            ),
        ),
        Topic(
            name="nuplan",
            frequency=12,
            recency_score=0.97,
            depth_score=4.0,
            debt_score=0.15,
            curiosity_type=CuriosityType.SHALLOW,
            source_types=frozenset({
                SignalSource.CHROME_HISTORY,
                SignalSource.CHATGPT_EXPORT,
            }),
            first_seen=_ago(8),
            last_seen=_ago(0),
            signal_samples=(
                "nuplan closed-loop evaluation metrics",
                "nuplan vs waymo open dataset comparison",
            ),
        ),
        Topic(
            name="karpathy neural networks",
            frequency=55,
            recency_score=0.4,
            depth_score=20.0,
            debt_score=0.05,
            curiosity_type=CuriosityType.DEEP,
            source_types=frozenset({
                SignalSource.YOUTUBE_TAKEOUT,
                SignalSource.CHROME_HISTORY,
            }),
            first_seen=_ago(180),
            last_seen=_ago(30),
            signal_samples=(
                "karpathy makemore series",
                "neural network backpropagation from scratch",
            ),
        ),
        Topic(
            name="embodied ai",
            frequency=8,
            recency_score=0.92,
            depth_score=3.0,
            debt_score=0.2,
            curiosity_type=CuriosityType.SHALLOW,
            source_types=frozenset({
                SignalSource.CHROME_HISTORY,
                SignalSource.CHATGPT_EXPORT,
                SignalSource.REDDIT_SAVED,
            }),
            first_seen=_ago(14),
            last_seen=_ago(1),
            signal_samples=(
                "LLM agents for robot control survey",
                "embodied AI benchmarks 2025",
            ),
        ),
        Topic(
            name="rust programming",
            frequency=15,
            recency_score=0.6,
            depth_score=5.0,
            debt_score=0.45,
            curiosity_type=CuriosityType.RECURRING,
            source_types=frozenset({
                SignalSource.CHROME_HISTORY,
                SignalSource.GMAIL,
            }),
            first_seen=_ago(90),
            last_seen=_ago(7),
            signal_samples=(
                "rust ownership model explained",
                "rust for systems programming 2025",
            ),
        ),
        Topic(
            name="pose estimation",
            frequency=20,
            recency_score=0.75,
            depth_score=8.0,
            debt_score=0.3,
            curiosity_type=CuriosityType.RECURRING,
            source_types=frozenset({
                SignalSource.CHROME_HISTORY,
                SignalSource.CHATGPT_EXPORT,
            }),
            first_seen=_ago(35),
            last_seen=_ago(5),
            signal_samples=(
                "6dof pose estimation transformer",
                "FoundPose paper summary",
            ),
        ),
        Topic(
            name="ai safety",
            frequency=9,
            recency_score=0.55,
            depth_score=3.0,
            debt_score=0.6,
            curiosity_type=CuriosityType.RECURRING,
            source_types=frozenset({
                SignalSource.CHROME_HISTORY,
                SignalSource.GMAIL,
                SignalSource.REDDIT_SAVED,
            }),
            first_seen=_ago(120),
            last_seen=_ago(10),
            signal_samples=(
                "RLHF alignment techniques survey",
                "AI safety weekly newsletter",
            ),
        ),
    )

    return CuriosityGraph(
        topics=topics,
        signal_count=245,
        source_breakdown={
            "chrome_history": 142,
            "chatgpt_export": 61,
            "youtube_takeout": 28,
            "gmail": 14,
        },
    )


async def inject_demo_seed() -> dict[str, Any]:
    """Inject the demo profile into Redis (if available) and in-memory cache.

    Returns a status dict confirming what was seeded.
    """
    graph = build_demo_graph()

    # Write to in-memory api.py cache.
    injected_memory = False
    try:
        from trace.delivery.api import _PROFILE_CACHE, _PROFILE_CACHE_MAX
        _PROFILE_CACHE[_SEED_PROFILE_ID] = graph
        if len(_PROFILE_CACHE) > _PROFILE_CACHE_MAX:
            _PROFILE_CACHE.popitem(last=False)
        injected_memory = True
    except Exception as exc:
        _log.warning("demo_seed: could not write to _PROFILE_CACHE: %s", exc)

    # Write to Redis.
    injected_redis = False
    try:
        from trace.mcp.redis_store import RedisCuriosityStore
        store = RedisCuriosityStore()
        if store.enabled:
            injected_redis = await store.store_profile(
                profile_id=_SEED_PROFILE_ID,
                graph=graph,
                user_id="demo-user",
            )
    except Exception as exc:
        _log.warning("demo_seed: Redis write failed: %s", exc)

    topic_names = [t.name for t in graph.topics]
    _log.info("demo_seed: injected profile=%s topics=%s", _SEED_PROFILE_ID, topic_names)

    return {
        "status": "ok",
        "profile_id": _SEED_PROFILE_ID,
        "topic_count": len(graph.topics),
        "signal_count": graph.signal_count,
        "topics": topic_names,
        "injected_memory": injected_memory,
        "injected_redis": injected_redis,
        "hint": (
            f"Use profile_id='{_SEED_PROFILE_ID}' in MCP tool calls to see the demo graph. "
            "Try: get_curiosity_topics, get_emerging_interests, get_topic_neighbors."
        ),
    }
