"""
ContextIndexer — flattens a CuriosityGraph + raw signals into a ContextStore.

What goes in:
  - One ContextItem per Topic in the graph (text = "<name>. <signal_samples>").
    Topics carry the highest signal-to-noise — they were already filtered by
    the curiosity heuristic during graph build.
  - One ContextItem per raw signal, capped at the N most recent. Raw signals
    add the texture an agent needs to give specific, evidence-grounded answers
    ("you starred bytecodealliance/wasmtime three days ago"), but they are
    noisier than topics, so we limit how many we keep.

What does NOT go in:
  - Scraped article summaries. Those belong to the legacy newsletter path.
    Keeping the context store focused on personal behaviour keeps the
    cosine-similarity scores meaningful.

Batching:
  All embedding is done in a single ``TopicEmbedder.encode()`` call. Encoding
  300 strings is ~3s on CPU; 300 separate calls would be ~30s.
"""

from __future__ import annotations

import logging

import numpy as np

from trace.context.store import ContextItem, ContextStore
from trace.graph.embedder import TopicEmbedder
from trace.models import CuriosityGraph, RawSignal

_log = logging.getLogger(__name__)

# How many of the most-recent raw signals to push into the context store.
# Sized for hackathon demo latency: 200 signals × ~5 ms/embed ≈ 1 s on CPU.
_RECENT_SIGNAL_CAP = 200


class ContextIndexer:
    """Build a ContextStore from a CuriosityGraph and the signals that produced it."""

    def __init__(
        self,
        embedder: TopicEmbedder,
        recent_signal_cap: int = _RECENT_SIGNAL_CAP,
    ) -> None:
        if embedder is None:
            raise ValueError("ContextIndexer: embedder must not be None")
        if recent_signal_cap < 0:
            raise ValueError(
                f"ContextIndexer: recent_signal_cap must be >= 0, got {recent_signal_cap}"
            )
        self._embedder = embedder
        self._recent_signal_cap = recent_signal_cap

    def index(
        self,
        store: ContextStore,
        graph: CuriosityGraph,
        signals: list[RawSignal],
    ) -> int:
        """Populate ``store`` with items derived from ``graph`` and ``signals``.

        Returns:
            The number of items added.
        """
        items: list[ContextItem] = []

        # 1. One item per topic. Composite description = name + up to 2 sample
        #    contents so the embedding captures both the label and the specific
        #    queries/pages that led to it.
        for topic in graph.topics:
            samples = " | ".join(topic.signal_samples[:2]) if topic.signal_samples else ""
            text = topic.name if not samples else f"{topic.name}. {samples}"
            items.append(
                ContextItem(
                    text=_truncate(text, 400),
                    source="topic",
                    topic_name=topic.name,
                    timestamp=topic.last_seen,
                    metadata={
                        "frequency": topic.frequency,
                        "recency_score": round(topic.recency_score, 3),
                        "composite_score": round(topic.composite_score(), 3),
                        "curiosity_type": topic.curiosity_type.value,
                    },
                )
            )

        # 2. Most-recent raw signals, capped.
        recent = sorted(signals, key=lambda s: s.timestamp, reverse=True)[: self._recent_signal_cap]
        for sig in recent:
            content = sig.content.strip()
            if not content:
                continue
            items.append(
                ContextItem(
                    text=_truncate(content, 400),
                    source="signal",
                    topic_name=None,
                    timestamp=sig.timestamp,
                    metadata={
                        "signal_source": sig.source.value,
                        "url": sig.url,
                    },
                )
            )

        if not items:
            _log.warning("ContextIndexer: no items to index (empty graph + no signals)")
            return 0

        texts = [it.text for it in items]
        emb_map = self._embedder.encode(texts)
        if not emb_map:
            _log.error(
                "ContextIndexer: embedder returned no vectors — "
                "is sentence-transformers installed?"
            )
            return 0

        # The embedder dedups by text key. If two items share the exact same
        # text, they will map to the same vector — which is fine for retrieval
        # quality but means we need to look each one up by text, not by index.
        # In practice texts are nearly always unique (timestamps differ inside
        # signal content), so this is a defensive path.
        vectors = np.stack([emb_map[t] for t in texts if t in emb_map])
        kept_items = [it for it in items if it.text in emb_map]

        if len(kept_items) != len(items):
            _log.warning(
                "ContextIndexer: %d/%d items had no embedding (likely model load failed)",
                len(items) - len(kept_items), len(items),
            )

        store.add(kept_items, vectors)
        _log.info(
            "ContextIndexer: indexed %d item(s) (%d topics + %d signals)",
            len(kept_items),
            len(graph.topics),
            len(kept_items) - len(graph.topics),
        )
        return len(kept_items)


def _truncate(text: str, max_len: int) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"
