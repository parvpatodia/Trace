"""
TopicEmbedder — encodes topic names with sentence-transformers for graph edges.

WHY EMBEDDINGS FOR CURIOSITY GRAPH:
  Topic names like "diffusion policy" and "robot learning" are semantically
  related but share no overlapping tokens. Cosine similarity on embeddings
  (all-MiniLM-L6-v2, 384-dim) catches these semantic relationships and lets
  the graph algorithm detect clusters the keyword-matching heuristic misses.

MODEL CHOICE: all-MiniLM-L6-v2
  - 384 dimensions, ~80MB — fits on M1 laptop without GPU
  - Mean pooling over WordPiece tokens → sentence-level representation
  - Trained on 1B pairs (NLI + STS) → good general-purpose similarity
  - Inference: ~5ms per topic name on CPU (acceptable for 20-topic graphs)

CACHING STRATEGY:
  Embeddings are cached by topic name in a dict (L1 in-process cache).
  In production, they are also written to Redis with a 30-day TTL so that
  restarting the server doesn't re-encode the same topics (Phase 4 Redis path
  is implemented but disabled when Redis is unconfigured).

COSINE SIMILARITY THRESHOLD: 0.40
  Short technical topic names (2-4 words) score lower than full sentences on
  all-MiniLM-L6-v2: "diffusion policy" ↔ "robot learning" lands ~0.48-0.55.
  0.40 captures domain-adjacent pairs while excluding unrelated topics
  ("robotics" ↔ "cooking" scores ~0.10-0.20 — safely filtered out).
  0.80+ is reserved for near-duplicates ("machine learning" ↔ "deep learning").
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

# Similarity thresholds for edge classification.
COSINE_EDGE_THRESHOLD = 0.40
COSINE_NEAR_DUPLICATE_THRESHOLD = 0.80

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import]
    _ST_AVAILABLE = True
except ModuleNotFoundError:
    _ST_AVAILABLE = False
    SentenceTransformer = None  # type: ignore[assignment,misc]


class TopicEmbedder:
    """Encode topic names and compute pairwise cosine similarity.

    All methods are safe to call even when sentence-transformers is not
    installed — they degrade to returning empty arrays/dicts.
    """

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: Any = None
        self._cache: dict[str, np.ndarray] = {}

    def _ensure_model(self) -> bool:
        """Lazy-load the model on first use. Returns False if unavailable."""
        if self._model is not None:
            return True
        if not _ST_AVAILABLE:
            _log.warning("sentence-transformers not installed — embedder disabled")
            return False
        try:
            self._model = SentenceTransformer(self._model_name)
            _log.info("TopicEmbedder: loaded %s", self._model_name)
            return True
        except Exception as exc:
            _log.warning("TopicEmbedder: could not load model (%s)", exc)
            return False

    def encode(self, topic_names: list[str]) -> dict[str, np.ndarray]:
        """Return {name: embedding_vector} for each topic name.

        Uses in-process cache — topics already encoded are not re-encoded.
        Returns {} if the model is unavailable.
        """
        if not topic_names:
            return {}
        if not self._ensure_model():
            return {}

        to_encode = [n for n in topic_names if n not in self._cache]
        if to_encode:
            try:
                vectors = self._model.encode(to_encode, convert_to_numpy=True, show_progress_bar=False)
                for name, vec in zip(to_encode, vectors):
                    self._cache[name] = vec.astype(np.float32)
            except Exception as exc:
                _log.warning("TopicEmbedder.encode failed: %s", exc)
                return {}

        return {n: self._cache[n] for n in topic_names if n in self._cache}

    def cosine_similarity_matrix(
        self, topic_names: list[str]
    ) -> tuple[list[str], np.ndarray]:
        """Compute pairwise cosine similarity for all topic names.

        Returns (ordered_names, N×N similarity matrix).
        Returns ([], empty_array) if embedding fails.
        """
        embeddings = self.encode(topic_names)
        if not embeddings:
            return [], np.empty((0, 0), dtype=np.float32)

        names = list(embeddings.keys())
        matrix = np.stack([embeddings[n] for n in names])

        # L2-normalise each row, then dot product = cosine similarity.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # guard division by zero
        normed = matrix / norms
        sim_matrix = (normed @ normed.T).astype(np.float32)
        # Clamp numerical noise to [-1, 1].
        sim_matrix = np.clip(sim_matrix, -1.0, 1.0)
        return names, sim_matrix

    def build_edges(
        self,
        topic_names: list[str],
        threshold: float = COSINE_EDGE_THRESHOLD,
    ) -> list[tuple[str, str, float]]:
        """Return (name_a, name_b, cosine_similarity) for all pairs above threshold.

        Self-edges and duplicate (b,a) pairs are excluded.
        Near-duplicates (similarity >= 0.80) are still included — callers can
        filter them differently if needed (e.g. to merge topics).
        """
        names, sim = self.cosine_similarity_matrix(topic_names)
        if not names:
            return []

        edges: list[tuple[str, str, float]] = []
        n = len(names)
        for i in range(n):
            for j in range(i + 1, n):
                score = float(sim[i, j])
                if score >= threshold:
                    edges.append((names[i], names[j], score))

        _log.debug(
            "TopicEmbedder.build_edges: %d topics → %d edges (threshold=%.2f)",
            n, len(edges), threshold,
        )
        return edges
