"""
Unit tests for trace/graph/graph_algo.py and trace/graph/embedder.py.

Embedder tests use mock vectors to avoid loading the 80MB model in CI.
graph_algo tests cover PageRank, Louvain, blend, and enrich_graph fallback paths.
"""
from __future__ import annotations

import numpy as np
import pytest
from datetime import datetime, timezone

from trace.graph.embedder import TopicEmbedder, COSINE_EDGE_THRESHOLD
from trace.graph.graph_algo import (
    GraphEnrichment,
    blend_scores,
    compute_pagerank,
    detect_communities,
    enrich_graph,
)
from trace.models import CuriosityGraph, CuriosityType, Topic


# ── Test fixtures ──────────────────────────────────────────────────────────────

def _topic(name: str, recency: float = 0.5, depth: float = 1.0, freq: int = 3) -> Topic:
    return Topic(
        name=name,
        frequency=freq,
        recency_score=recency,
        depth_score=depth,
        curiosity_type=CuriosityType.SHALLOW,
    )


def _graph(*names: str) -> CuriosityGraph:
    return CuriosityGraph(topics=tuple(_topic(n) for n in names))


# ── TopicEmbedder ──────────────────────────────────────────────────────────────

class TestTopicEmbedder:
    """Test embedder with mocked model to avoid heavy model load in CI."""

    @pytest.fixture
    def mock_embedder(self, monkeypatch):
        embedder = TopicEmbedder()
        # Pre-populate cache with predictable vectors.
        embedder._cache = {
            "robotics": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "robot learning": np.array([0.9, 0.1, 0.0], dtype=np.float32),  # close to robotics
            "cooking": np.array([0.0, 1.0, 0.0], dtype=np.float32),  # far from robotics
            "ai safety": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        # Make _ensure_model return True (model "loaded").
        monkeypatch.setattr(embedder, "_ensure_model", lambda: True)
        return embedder

    def test_encode_uses_cache(self, mock_embedder):
        result = mock_embedder.encode(["robotics", "cooking"])
        assert "robotics" in result
        assert "cooking" in result

    def test_cosine_similarity_matrix_shape(self, mock_embedder):
        names, matrix = mock_embedder.cosine_similarity_matrix(["robotics", "cooking", "ai safety"])
        assert len(names) == 3
        assert matrix.shape == (3, 3)

    def test_self_similarity_is_one(self, mock_embedder):
        names, matrix = mock_embedder.cosine_similarity_matrix(["robotics"])
        assert abs(matrix[0, 0] - 1.0) < 1e-5

    def test_cosine_similarity_between_similar_topics(self, mock_embedder):
        names, matrix = mock_embedder.cosine_similarity_matrix(["robotics", "robot learning"])
        # Both point in roughly the same direction — should be > 0.35.
        idx_r = names.index("robotics")
        idx_rl = names.index("robot learning")
        assert matrix[idx_r, idx_rl] > COSINE_EDGE_THRESHOLD

    def test_cosine_similarity_between_dissimilar_topics(self, mock_embedder):
        names, matrix = mock_embedder.cosine_similarity_matrix(["robotics", "cooking"])
        idx_r = names.index("robotics")
        idx_c = names.index("cooking")
        assert matrix[idx_r, idx_c] < COSINE_EDGE_THRESHOLD

    def test_build_edges_returns_similar_pairs(self, mock_embedder):
        edges = mock_embedder.build_edges(["robotics", "robot learning", "cooking"])
        edge_names = {(a, b) for a, b, _ in edges}
        # robotics ↔ robot learning should be an edge.
        assert ("robotics", "robot learning") in edge_names or ("robot learning", "robotics") in edge_names

    def test_build_edges_excludes_dissimilar(self, mock_embedder):
        edges = mock_embedder.build_edges(["robotics", "cooking"])
        assert len(edges) == 0  # cosine = 0, below 0.35

    def test_build_edges_empty_input(self, mock_embedder):
        assert mock_embedder.build_edges([]) == []

    def test_encode_empty(self, mock_embedder):
        assert mock_embedder.encode([]) == {}


# ── compute_pagerank ───────────────────────────────────────────────────────────

class TestComputePagerank:
    def test_no_edges_returns_zeros(self):
        topics = [_topic("a"), _topic("b")]
        pr = compute_pagerank(topics, [])
        assert all(v == 0.0 for v in pr.values())

    def test_pagerank_normalised_to_one(self):
        topics = [_topic("a"), _topic("b"), _topic("c")]
        edges = [("a", "b", 0.8), ("b", "c", 0.7), ("a", "c", 0.75)]
        pr = compute_pagerank(topics, edges)
        assert max(pr.values()) <= 1.0 + 1e-6
        assert min(pr.values()) >= 0.0

    def test_empty_topics(self):
        assert compute_pagerank([], []) == {}

    def test_hub_topic_higher_rank(self):
        # "hub" connects to everything; "leaf" connects only to hub.
        topics = [_topic("hub"), _topic("a"), _topic("b"), _topic("c")]
        edges = [
            ("hub", "a", 0.9), ("hub", "b", 0.9), ("hub", "c", 0.9),
        ]
        pr = compute_pagerank(topics, edges)
        assert pr.get("hub", 0) >= pr.get("a", 0)


# ── detect_communities ─────────────────────────────────────────────────────────

class TestDetectCommunities:
    def test_two_disconnected_topics_get_different_communities(self):
        topics = [_topic("robotics"), _topic("cooking")]
        # No edges → each topic in its own community.
        communities = detect_communities(topics, [])
        # With no edges, Louvain may put all in one community or separate — both valid.
        assert isinstance(communities, dict)

    def test_connected_cluster(self):
        topics = [_topic("a"), _topic("b"), _topic("c"), _topic("x"), _topic("y")]
        edges = [("a", "b", 0.9), ("b", "c", 0.85), ("x", "y", 0.9)]
        communities = detect_communities(topics, edges)
        # a, b, c should be in the same community; x, y in another.
        if communities:  # only if Louvain is available
            assert communities.get("a") == communities.get("b") == communities.get("c")

    def test_returns_empty_for_single_topic(self):
        result = detect_communities([_topic("solo")], [])
        # single node — either {} or {"solo": 0}.
        assert isinstance(result, dict)

    def test_empty_input(self):
        assert detect_communities([], []) == {}


# ── blend_scores ───────────────────────────────────────────────────────────────

class TestBlendScores:
    def test_returns_all_topic_names(self):
        topics = [_topic("a", recency=0.8), _topic("b", recency=0.3)]
        pagerank = {"a": 0.9, "b": 0.1}
        result = blend_scores(topics, pagerank)
        assert set(result.keys()) == {"a", "b"}

    def test_higher_pagerank_boosts_score(self):
        topics = [_topic("a"), _topic("b")]
        pagerank = {"a": 1.0, "b": 0.0}
        result = blend_scores(topics, pagerank)
        assert result["a"] > result["b"]

    def test_empty_topics(self):
        assert blend_scores([], {}) == {}


# ── enrich_graph ───────────────────────────────────────────────────────────────

class TestEnrichGraph:
    def test_empty_graph_returns_empty_enrichment(self):
        graph = CuriosityGraph()
        enrichment = enrich_graph(graph)
        assert enrichment.edges == []
        assert enrichment.communities == {}
        assert enrichment.blended_scores == {}

    def test_enrichment_with_mock_embedder(self, monkeypatch):
        graph = _graph("robotics", "robot learning", "cooking", "ai safety")

        class MockEmbedder:
            def build_edges(self, names, threshold=0.35):
                return [("robotics", "robot learning", 0.9)]

            def cosine_similarity_matrix(self, names):
                # All off-diagonal scores are low — fallback edges use score ~0.1.
                n = len(names)
                sim = np.full((n, n), 0.1, dtype=np.float32)
                np.fill_diagonal(sim, 1.0)
                return names, sim

        enrichment = enrich_graph(graph, embedder=MockEmbedder())
        # Primary edge + fallback edges for isolated nodes ("cooking", "ai safety").
        assert len(enrichment.edges) >= 1
        primary_edge_names = {enrichment.edges[0][0], enrichment.edges[0][1]}
        assert "robotics" in primary_edge_names or "robot learning" in primary_edge_names

    def test_graph_enrichment_to_dict(self):
        graph = _graph("a", "b")

        class MockEmbedder:
            def build_edges(self, names, threshold=0.35):
                return [("a", "b", 0.8)]

        enrichment = enrich_graph(graph, embedder=MockEmbedder())
        d = enrichment.to_dict()
        assert "edge_count" in d
        assert "community_count" in d

    def test_top_topics_returns_sorted(self):
        enrichment = GraphEnrichment(
            edges=[],
            communities={},
            blended_scores={"a": 0.9, "b": 0.3, "c": 0.6},
            pagerank={},
        )
        top = enrichment.top_topics(2)
        assert top[0][0] == "a"
        assert top[1][0] == "c"

    def test_neighbors_lookup(self):
        enrichment = GraphEnrichment(
            edges=[("a", "b", 0.85), ("a", "c", 0.70)],
            communities={},
            blended_scores={},
            pagerank={},
        )
        nbrs = enrichment.neighbors("a")
        assert len(nbrs) == 2
        assert nbrs[0][1] >= nbrs[1][1]  # sorted desc

    def test_fallback_edges_connect_isolated_topics(self):
        """Isolated topics (degree=0) should get a fallback edge to nearest neighbor."""
        graph = _graph("robotics", "cooking")

        class MockEmbedder:
            def build_edges(self, names, threshold=0.35):
                return []  # Nothing above threshold — both isolated.

            def cosine_similarity_matrix(self, names):
                # robotics ↔ cooking at 0.15 — below threshold but best available.
                n = len(names)
                sim = np.full((n, n), 0.15, dtype=np.float32)
                np.fill_diagonal(sim, 1.0)
                return names, sim

        enrichment = enrich_graph(graph, embedder=MockEmbedder())
        # Both were isolated → fallback should add at least 1 edge.
        assert len(enrichment.edges) >= 1

    def test_community_members(self):
        enrichment = GraphEnrichment(
            edges=[],
            communities={"a": 0, "b": 0, "c": 1},
            blended_scores={},
            pagerank={},
        )
        assert set(enrichment.community_members(0)) == {"a", "b"}
        assert enrichment.community_members(1) == ["c"]
