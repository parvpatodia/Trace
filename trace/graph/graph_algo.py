"""
Graph algorithm layer for the Curiosity Graph.

PIPELINE:
  1. Encode topic names with TopicEmbedder (sentence-transformers/all-MiniLM-L6-v2)
  2. Build edges: cosine similarity > 0.40 between topic pairs
  3. Compute PageRank on the edge-weighted graph (networkx)
  4. Blend scores: 0.6 × PageRank + 0.4 × composite_score
  5. Detect communities with Louvain algorithm (python-louvain)
  6. Return enriched data for the CuriosityGraph

WHY PAGERANK FOR A CURIOSITY GRAPH:
  A topic well-connected to many other high-scoring topics is likely a
  "hub concept" — a foundational idea the user is building intuition around.
  PageRank captures this centrality in a way raw frequency can't.

BLEND RATIO 60/40:
  PageRank alone over-weights popular/general topics (e.g. "AI" links to
  everything). The 40% composite_score anchors the final rank to the user's
  actual behavioral signal (recency, frequency, debt).

LOUVAIN COMMUNITIES:
  Groups semantically related topics into clusters. Used by:
  - bridge_topic_shift detector (Phase 4+ version): bridge = topic in 2+ communities
  - D3 visualization: colour topics by community
  - MCP tool get_topic_neighbors: return neighbors in same community

GRACEFUL DEGRADATION:
  networkx or python-louvain not installed → returns empty edges/communities,
  scores fall back to composite_score only. Safe to call unconditionally.
"""
from __future__ import annotations

import logging
from typing import Any

from trace.models import CuriosityGraph, Topic

_log = logging.getLogger(__name__)

# Blend weights: PageRank score vs original composite_score.
_PAGERANK_WEIGHT = 0.6
_COMPOSITE_WEIGHT = 0.4

try:
    import networkx as nx  # type: ignore[import]
    _NX_AVAILABLE = True
except ModuleNotFoundError:
    _NX_AVAILABLE = False
    nx = None  # type: ignore[assignment]

try:
    import community as community_louvain  # type: ignore[import]
    _LOUVAIN_AVAILABLE = True
except ModuleNotFoundError:
    _LOUVAIN_AVAILABLE = False
    community_louvain = None  # type: ignore[assignment]


def _build_nx_graph(
    topics: list[Topic],
    edges: list[tuple[str, str, float]],
) -> "Any":
    """Build a weighted undirected NetworkX graph from topics + edges."""
    if not _NX_AVAILABLE:
        return None

    G = nx.Graph()
    for t in topics:
        G.add_node(t.name, composite_score=t.composite_score(), topic=t)

    for name_a, name_b, weight in edges:
        if G.has_node(name_a) and G.has_node(name_b):
            G.add_edge(name_a, name_b, weight=weight)

    return G


def compute_pagerank(
    topics: list[Topic],
    edges: list[tuple[str, str, float]],
    alpha: float = 0.85,
) -> dict[str, float]:
    """Return {topic_name: pagerank_score} normalised to [0, 1].

    Falls back to composite_score-based ranking if networkx is unavailable
    or the graph has no edges.
    """
    if not topics:
        return {}

    # Fallback: no networkx or no edges → all scores = 0 (will use composite only).
    if not _NX_AVAILABLE or not edges:
        return {t.name: 0.0 for t in topics}

    G = _build_nx_graph(topics, edges)
    if G is None or G.number_of_edges() == 0:
        return {t.name: 0.0 for t in topics}

    try:
        pr = nx.pagerank(G, alpha=alpha, weight="weight", max_iter=100)

        # Normalise to [0, 1].
        max_pr = max(pr.values()) if pr else 1.0
        if max_pr == 0:
            max_pr = 1.0
        return {name: score / max_pr for name, score in pr.items()}

    except Exception as exc:
        _log.warning("PageRank failed: %s — using zeros", exc)
        return {t.name: 0.0 for t in topics}


def detect_communities(
    topics: list[Topic],
    edges: list[tuple[str, str, float]],
) -> dict[str, int]:
    """Return {topic_name: community_id} via Louvain algorithm.

    Returns {} if python-louvain is unavailable or graph has < 2 nodes.
    """
    if not topics or not _NX_AVAILABLE or not _LOUVAIN_AVAILABLE:
        return {}
    if len(topics) < 2:
        return {topics[0].name: 0} if topics else {}

    G = _build_nx_graph(topics, edges)
    if G is None or G.number_of_nodes() < 2:
        return {}

    try:
        partition: dict[str, int] = community_louvain.best_partition(G, weight="weight")
        _log.debug(
            "Louvain: %d nodes → %d communities",
            G.number_of_nodes(), len(set(partition.values())),
        )
        return partition
    except Exception as exc:
        _log.warning("Louvain community detection failed: %s", exc)
        return {}


def blend_scores(
    topics: list[Topic],
    pagerank: dict[str, float],
) -> dict[str, float]:
    """Return {topic_name: blended_score} = 0.6×PageRank + 0.4×composite."""
    result: dict[str, float] = {}
    for t in topics:
        pr = pagerank.get(t.name, 0.0)
        composite = t.composite_score()
        # Normalise composite to [0, 1] for blending — divide by max across all topics.
        result[t.name] = _PAGERANK_WEIGHT * pr + _COMPOSITE_WEIGHT * composite
    # Normalise the composite part across all topics.
    max_composite = max((t.composite_score() for t in topics), default=1.0)
    if max_composite <= 0:
        max_composite = 1.0
    result = {
        name: _PAGERANK_WEIGHT * pagerank.get(name, 0.0)
               + _COMPOSITE_WEIGHT * (score / max_composite)
        for name, score in result.items()
    }
    return result


def enrich_graph(
    graph: CuriosityGraph,
    embedder: Any | None = None,
    cosine_threshold: float = 0.40,
) -> "GraphEnrichment":
    """Compute edges, PageRank, Louvain communities, and blended scores.

    Returns a GraphEnrichment dataclass (not modifying the immutable CuriosityGraph).
    The enrichment is stored in the Redis-cached JSON or attached to the MCP response.
    """
    from trace.graph.embedder import TopicEmbedder

    if embedder is None:
        embedder = TopicEmbedder()

    topics = list(graph.topics)
    if not topics:
        return GraphEnrichment(edges=[], communities={}, blended_scores={}, pagerank={})

    topic_names = [t.name for t in topics]

    # Step 1: build semantic edges.
    edges = embedder.build_edges(topic_names, threshold=cosine_threshold)

    # Step 2: PageRank on weighted edge graph.
    pagerank = compute_pagerank(topics, edges)

    # Step 3: Louvain community detection.
    communities = detect_communities(topics, edges)

    # Step 4: blend PageRank + composite.
    blended = blend_scores(topics, pagerank)

    _log.info(
        "enrich_graph: %d topics, %d edges, %d communities",
        len(topics), len(edges), len(set(communities.values())),
    )
    return GraphEnrichment(
        edges=edges,
        communities=communities,
        blended_scores=blended,
        pagerank=pagerank,
    )


class GraphEnrichment:
    """Result of the graph enrichment pipeline."""

    __slots__ = ("edges", "communities", "blended_scores", "pagerank")

    def __init__(
        self,
        edges: list[tuple[str, str, float]],
        communities: dict[str, int],
        blended_scores: dict[str, float],
        pagerank: dict[str, float],
    ) -> None:
        self.edges = edges
        self.communities = communities
        self.blended_scores = blended_scores
        self.pagerank = pagerank

    def top_topics(self, n: int = 10) -> list[tuple[str, float]]:
        """Return (name, blended_score) sorted descending."""
        return sorted(self.blended_scores.items(), key=lambda x: x[1], reverse=True)[:n]

    def neighbors(self, topic_name: str) -> list[tuple[str, float]]:
        """Return (neighbor_name, cosine_similarity) sorted descending."""
        result = [
            (b, w) for (a, b, w) in self.edges if a == topic_name
        ] + [
            (a, w) for (a, b, w) in self.edges if b == topic_name
        ]
        return sorted(result, key=lambda x: x[1], reverse=True)

    def community_members(self, community_id: int) -> list[str]:
        """Return all topic names in a given community."""
        return [name for name, cid in self.communities.items() if cid == community_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_count": len(self.edges),
            "community_count": len(set(self.communities.values())),
            "top_topics": self.top_topics(10),
            "communities": self.communities,
        }
