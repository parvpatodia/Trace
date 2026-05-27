"""
RedisCuriosityStore — fast persistent cache for CuriosityGraphs.

WHY REDIS FOR A CURIOSITY OS:
  The Trace MCP server is queried by AI agents in real time. When Claude calls
  get_curiosity_topics(), a >1s response breaks the agent loop. Redis lets us
  serve graphs in <50ms even after a server restart.

  Beyond latency, Redis enables:
  - Multi-process workers sharing a single graph store (horizontal scale)
  - TTL-based expiry of stale curiosity signals (interests decay over time)
  - Sorted-set indexes for "top topics by score" without deserializing the graph

DUAL-WRITE STRATEGY:
  Dual-write to both Redis and the in-memory OrderedDict cache in api.py.
  Redis is the source of truth; in-memory is a per-process L1.
  If Redis is unavailable (no config, network error), in-memory continues
  working — graceful degradation.

KEY SCHEMA:
  trace:profile:{profile_id}         → JSON-encoded CuriosityGraph
  trace:profile:{profile_id}:topics  → ZSET topic_name → composite_score
  trace:profile:{profile_id}:meta    → HASH (built_at, signal_count, user_id)
  trace:user:{user_id}:profiles      → ZSET profile_id → built_at timestamp
"""
from __future__ import annotations

import logging
from typing import Any

from trace.config import get_settings
from trace.models import CuriosityGraph

_log = logging.getLogger(__name__)
_PROFILE_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days

try:
    import redis.asyncio as aioredis  # type: ignore[import]
    _REDIS_AVAILABLE = True
except ModuleNotFoundError:
    _REDIS_AVAILABLE = False
    aioredis = None  # type: ignore[assignment]


class RedisCuriosityStore:
    """Async Redis-backed cache for CuriosityGraph instances.

    All methods are no-ops returning None/False when Redis is not configured,
    making it safe to call unconditionally from the pipeline.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        self._url = redis_url or get_settings().redis_url
        self._client: Any | None = None

    @property
    def enabled(self) -> bool:
        return _REDIS_AVAILABLE and bool(self._url)

    async def _ensure_client(self) -> Any | None:
        if not self.enabled:
            return None
        if self._client is None:
            try:
                self._client = aioredis.from_url(  # type: ignore[union-attr]
                    self._url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_timeout=2.0,
                    socket_connect_timeout=2.0,
                )
                await self._client.ping()
            except Exception as exc:
                _log.warning("Redis connect failed (%s) — falling back to in-memory", exc)
                self._client = None
        return self._client

    async def store_profile(
        self,
        profile_id: str,
        graph: CuriosityGraph,
        user_id: str = "",
    ) -> bool:
        client = await self._ensure_client()
        if client is None:
            return False
        try:
            pipe = client.pipeline(transaction=False)
            await pipe.set(
                f"trace:profile:{profile_id}",
                graph.model_dump_json(),
                ex=_PROFILE_TTL_SECONDS,
            )
            await pipe.hset(
                f"trace:profile:{profile_id}:meta",
                mapping={
                    "built_at": graph.built_at.isoformat(),
                    "signal_count": str(graph.signal_count),
                    "topic_count": str(len(graph.topics)),
                    "user_id": user_id,
                },
            )
            await pipe.expire(f"trace:profile:{profile_id}:meta", _PROFILE_TTL_SECONDS)
            topic_scores = {t.name: t.composite_score() for t in graph.topics}
            if topic_scores:
                await pipe.zadd(f"trace:profile:{profile_id}:topics", topic_scores)
                await pipe.expire(
                    f"trace:profile:{profile_id}:topics", _PROFILE_TTL_SECONDS
                )
            if user_id:
                await pipe.zadd(
                    f"trace:user:{user_id}:profiles",
                    {profile_id: graph.built_at.timestamp()},
                )
            await pipe.execute()
            return True
        except Exception as exc:
            _log.warning("Redis store_profile failed: %s", exc)
            return False

    async def load_profile(self, profile_id: str) -> CuriosityGraph | None:
        client = await self._ensure_client()
        if client is None:
            return None
        try:
            raw = await client.get(f"trace:profile:{profile_id}")
            if not raw:
                return None
            return CuriosityGraph.model_validate_json(raw)
        except Exception as exc:
            _log.warning("Redis load_profile failed: %s", exc)
            return None

    async def top_topics(
        self, profile_id: str, n: int = 10
    ) -> list[tuple[str, float]]:
        """Return top-N (topic_name, score) pairs without deserializing the full graph."""
        client = await self._ensure_client()
        if client is None:
            return []
        try:
            results = await client.zrevrange(
                f"trace:profile:{profile_id}:topics", 0, n - 1, withscores=True
            )
            return [(name, float(score)) for name, score in results]
        except Exception as exc:
            _log.warning("Redis top_topics failed: %s", exc)
            return []

    async def profile_meta(self, profile_id: str) -> dict[str, str]:
        client = await self._ensure_client()
        if client is None:
            return {}
        try:
            return await client.hgetall(f"trace:profile:{profile_id}:meta") or {}
        except Exception as exc:
            _log.warning("Redis profile_meta failed: %s", exc)
            return {}

    async def health(self) -> dict[str, Any]:
        """Return connection health for the health probe endpoints."""
        if not self.enabled:
            return {"status": "disabled", "enabled": False}
        client = await self._ensure_client()
        if client is None:
            return {"status": "unavailable", "enabled": True}
        return {"status": "ok", "enabled": True}

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None
