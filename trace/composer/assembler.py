"""
ContextWindowAssembler — packs a CuriosityGraph + scraped articles into an
AssemblyContext that fits within a token budget.

Token estimation:
  Rough heuristic: 1 token ≈ 4 characters.
  Each topic contributes its name length.
  Each article contributes its title + summary lengths.
  Overhead constant accounts for structural text (labels, separators).

Topic selection order:
  Topics are sorted by composite_score() descending, then trimmed to
  max_topics.  RESOLVED topics are excluded entirely.

Token budget enforcement:
  After selecting topics and grouping articles, we compute a token estimate.
  If it exceeds the budget, we drop the lowest-scoring topics one by one
  until it fits.

Debt topics:
  From the SELECTED topics only, those with debt_score > 0.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from trace.models import CuriosityGraph, CuriosityType, ScrapedArticle, Topic

_CHARS_PER_TOKEN = 4
_OVERHEAD_PER_TOPIC = 50   # chars — labels, newlines, structural text
_OVERHEAD_PER_ARTICLE = 30


class AssemblyContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    selected_topics: tuple[Topic, ...]
    articles_by_topic_id: dict[str, list[ScrapedArticle]]
    debt_topics: tuple[Topic, ...]
    token_estimate: int
    # topic_id → up to 2 representative signal contents (search queries /
    # ChatGPT questions preferred).  Used by NewsletterComposer to write in the
    # reader's own vocabulary.
    signal_samples: dict[str, list[str]] = Field(default_factory=dict)
    # topic_id → list of other topic names sharing vocabulary — enables genuine
    # bridge_insight sections by telling Claude which topics are semantically linked.
    related_topics: dict[str, list[str]] = Field(default_factory=dict)
    assembled_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


def _estimate_chars(topics: list[Topic], articles_by_id: dict[str, list[ScrapedArticle]]) -> int:
    total = 0
    for t in topics:
        total += len(t.name) + _OVERHEAD_PER_TOPIC
        for a in articles_by_id.get(t.id, []):
            total += len(a.title) + len(a.summary) + _OVERHEAD_PER_ARTICLE
    return total


def _chars_to_tokens(chars: int) -> int:
    return max(0, (chars + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


class ContextWindowAssembler:
    """
    Assembles a token-bounded context window from a CuriosityGraph and a list
    of scraped articles.

    Parameters:
        token_budget: maximum number of tokens allowed in the assembled context.
        max_topics: hard cap on number of topics included.
        max_articles_per_topic: hard cap on articles per topic (highest relevance kept).
    """

    def __init__(
        self,
        token_budget: int = 8000,
        max_topics: int = 5,
        max_articles_per_topic: int = 3,
    ) -> None:
        if token_budget <= 0:
            raise ValueError("token_budget must be a positive integer")
        if max_topics <= 0:
            raise ValueError("max_topics must be a positive integer")
        if max_articles_per_topic <= 0:
            raise ValueError("max_articles_per_topic must be a positive integer")

        self._token_budget = token_budget
        self._max_topics = max_topics
        self._max_articles_per_topic = max_articles_per_topic

    def assemble(
        self,
        graph: CuriosityGraph,
        articles: list[ScrapedArticle],
    ) -> AssemblyContext:
        # 1. Filter and rank topics
        candidates = [
            t for t in graph.topics
            if t.curiosity_type != CuriosityType.RESOLVED
        ]
        candidates.sort(key=lambda t: t.composite_score(), reverse=True)
        selected = candidates[: self._max_topics]

        if not selected:
            return AssemblyContext(
                selected_topics=(),
                articles_by_topic_id={},
                debt_topics=(),
                token_estimate=0,
            )

        # 2. Group and cap articles per topic
        articles_index: dict[str, list[ScrapedArticle]] = {}
        for t in selected:
            matched = [a for a in articles if a.topic_id == t.id]
            matched.sort(key=lambda a: a.relevance_score, reverse=True)
            articles_index[t.id] = matched[: self._max_articles_per_topic]

        # 3. Enforce token budget — drop lowest-scored non-debt topics first.
        # Debt (RECURRING) topics are protected because curiosity debt sections
        # are the most actionable content — they exist precisely because the user
        # keeps returning without resolution.
        while len(selected) > 1:
            chars = _estimate_chars(selected, articles_index)
            if _chars_to_tokens(chars) <= self._token_budget:
                break
            non_debt = [t for t in selected if t.debt_score < 0.3]
            to_drop = non_debt[-1] if non_debt else selected[-1]
            selected.remove(to_drop)
            del articles_index[to_drop.id]

        # 4. Compute final token estimate
        token_estimate = _chars_to_tokens(
            _estimate_chars(selected, articles_index)
        )

        # 5. Debt topics: only from selected, only those with significant curiosity debt.
        # Threshold 0.3 prevents topics with minor debt (e.g. 0.05–0.15) from triggering
        # a full curiosity_debt section — those are covered in the weekly_topics section.
        debt_topics = tuple(t for t in selected if t.debt_score >= 0.3)

        # 6. Signal samples: read directly from Topic.signal_samples — they were
        #    computed by CuriosityGraphBuilder and travel with the graph through
        #    serialisation, so daily regeneration (run_from_graph) works without
        #    needing access to the original raw signals.
        signal_samples = {
            t.id: list(t.signal_samples)
            for t in selected
            if t.signal_samples
        }

        # 7. Related topics: word-overlap between topic name + signal samples.
        # Tells Claude which topics share vocabulary, enabling genuine bridge_insight
        # sections without requiring the full embedding pipeline in the hot path.
        topic_vocab: dict[str, set[str]] = {
            t.id: (
                set(t.name.lower().split())
                | {w for s in t.signal_samples for w in s.lower().split()[:12]}
            )
            for t in selected
        }
        related_topics: dict[str, list[str]] = {
            t.id: [
                other.name
                for other in selected
                if other.id != t.id and len(topic_vocab[t.id] & topic_vocab[other.id]) >= 2
            ]
            for t in selected
        }

        return AssemblyContext(
            selected_topics=tuple(selected),
            articles_by_topic_id=articles_index,
            debt_topics=debt_topics,
            token_estimate=token_estimate,
            signal_samples=signal_samples,
            related_topics={tid: names for tid, names in related_topics.items() if names},
        )
