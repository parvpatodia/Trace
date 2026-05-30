"""
ContextStore — in-memory vector store backing the Personal Context API.

Design choices and why:

  Numpy-backed dense index, no external vector DB.
    For a single-user personal context (typical 200–500 items) a flat float32
    matrix + dot product is faster than any ANN library, requires no
    daemon/process, and serialises trivially to disk. Adding Faiss/Pinecone
    would be premature optimisation.

  L2-normalised on insert.
    Cosine similarity = dot product when both vectors are unit-norm. Doing
    the normalisation once at insert avoids repeating it on every query.

  One ContextStore per profile.
    The store does not know about profiles — callers maintain a
    {profile_id: ContextStore} mapping. This keeps the data structure
    single-responsibility and easy to test.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextItem:
    """One unit of personal context the agent can retrieve.

    Attributes:
        text: The natural-language content used both for embedding and for
              injection into the agent prompt. Kept short (<= ~500 chars) so
              the prompt stays focused.
        source: Provenance tag — "topic", "signal", or "article". Surfaced in
                the API response so the demo can show *why* a result was
                returned.
        topic_name: Optional topic this item belongs to (None for raw signals).
        timestamp: When the underlying behavioural signal occurred. None for
                   synthesised items (e.g. topic summaries).
        metadata: Free-form provenance — URL, signal source enum, etc.
    """

    text: str
    source: str
    topic_name: str | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextStore:
    """Dense in-memory vector store for personal context items.

    Thread-safety:
        Not thread-safe. FastAPI's default ASGI loop is single-threaded;
        if you move to a multi-worker setup, wrap mutating methods in a lock.
    """

    def __init__(self) -> None:
        self._items: list[ContextItem] = []
        # Shape (N, D). Lazily initialised on first add() so we don't hardcode D.
        self._matrix: np.ndarray | None = None

    def add(self, items: list[ContextItem], embeddings: np.ndarray) -> None:
        """Insert a batch of items with their embeddings.

        Args:
            items: K context items.
            embeddings: float32 array of shape (K, D). Rows are L2-normalised
                        in place — caller's array is mutated. This is the
                        documented contract; callers pass a fresh array.
        """
        if len(items) == 0:
            return
        if embeddings.ndim != 2 or embeddings.shape[0] != len(items):
            raise ValueError(
                f"ContextStore.add: embeddings shape {embeddings.shape} "
                f"does not match {len(items)} items"
            )

        normed = _l2_normalise(embeddings.astype(np.float32, copy=False))

        if self._matrix is None:
            self._matrix = normed
        else:
            if normed.shape[1] != self._matrix.shape[1]:
                raise ValueError(
                    f"ContextStore.add: embedding dim {normed.shape[1]} "
                    f"does not match existing dim {self._matrix.shape[1]}"
                )
            self._matrix = np.vstack([self._matrix, normed])

        self._items.extend(items)
        _log.debug("ContextStore: added %d item(s), total=%d", len(items), len(self._items))

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[tuple[ContextItem, float]]:
        """Return the top-k items ranked by cosine similarity to the query.

        Args:
            query_embedding: float32 array of shape (D,). Need not be normalised.
            top_k: maximum number of results.

        Returns:
            List of (item, score) tuples ordered by score descending.
            Score is cosine similarity in [-1.0, 1.0]; in practice for
            sentence-transformers embeddings the useful band is roughly
            [0.2, 0.9].
        """
        if self._matrix is None or len(self._items) == 0:
            return []
        if top_k < 1:
            return []

        q = query_embedding.astype(np.float32, copy=False).reshape(-1)
        q_norm = np.linalg.norm(q)
        if q_norm == 0.0:
            return []
        q = q / q_norm

        # Dot product against pre-normalised rows = cosine similarity.
        scores = self._matrix @ q  # shape (N,)

        k = min(top_k, len(self._items))
        # argpartition is O(N), then we sort only the top-k slice.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [(self._items[int(i)], float(scores[int(i)])) for i in top_idx]

    def size(self) -> int:
        return len(self._items)

    def is_empty(self) -> bool:
        return len(self._items) == 0


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Return a copy of ``matrix`` with each row L2-normalised. Zero rows pass
    through unchanged (avoids NaNs from division-by-zero)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms
