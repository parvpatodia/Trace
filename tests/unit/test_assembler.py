"""
Unit tests for trace/composer/assembler.py — ContextWindowAssembler.

ContextWindowAssembler is pure data transformation — no I/O, no mocks needed.
It takes a CuriosityGraph + list[ScrapedArticle] and produces an AssemblyContext
that fits within a token budget.

Test categories:
  1. Constructor validation: invalid token_budget, max_topics, max_articles_per_topic
  2. Empty graph: returns empty AssemblyContext
  3. Topic selection: top_n topics by composite_score selected
  4. Article grouping: articles matched to topics by topic_id
  5. Article ordering: articles sorted by relevance_score descending per topic
  6. max_articles_per_topic: hard cap enforced
  7. Token budget: assembly stays within budget; lowest-scored topics dropped first
  8. Debt topics: topics with debt_score > 0 appear in debt_topics field
  9. Token estimation: non-zero, proportional to content size
  10. AssemblyContext structure: selected_topics, articles_by_topic_id,
      debt_topics, token_estimate, assembled_at
  11. Articles with no matching topic: not included in context
  12. Topics with no matching articles: included in context with empty article list
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trace.composer.assembler import AssemblyContext, ContextWindowAssembler
from trace.models import (
    ContentSource,
    CuriosityGraph,
    CuriosityType,
    ScrapedArticle,
    Topic,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

_NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


def make_topic(
    name: str = "transformer architecture",
    frequency: int = 3,
    recency_score: float = 0.8,
    debt_score: float = 0.0,
    curiosity_type: CuriosityType = CuriosityType.SHALLOW,
    topic_id: str | None = None,
) -> Topic:
    t = Topic(
        name=name,
        frequency=frequency,
        recency_score=recency_score,
        debt_score=debt_score,
        curiosity_type=curiosity_type,
    )
    if topic_id:
        t = t.model_copy(update={"id": topic_id})
    return t


def make_article(
    topic: Topic,
    title: str = "Attention Is All You Need",
    url: str = "https://arxiv.org/abs/1706.03762",
    summary: str = "A paper about transformer models and attention mechanisms.",
    relevance_score: float = 0.9,
) -> ScrapedArticle:
    return ScrapedArticle(
        topic_id=topic.id,
        topic_name=topic.name,
        source=ContentSource.ARXIV,
        title=title,
        url=url,
        summary=summary,
        relevance_score=relevance_score,
    )


def make_graph(*topics: Topic) -> CuriosityGraph:
    return CuriosityGraph(topics=tuple(topics), signal_count=len(topics) * 3)


# ── Constructor validation ────────────────────────────────────────────────────

class TestConstructorValidation:
    def test_zero_token_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="token_budget"):
            ContextWindowAssembler(token_budget=0)

    def test_negative_token_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="token_budget"):
            ContextWindowAssembler(token_budget=-1)

    def test_zero_max_topics_raises(self) -> None:
        with pytest.raises(ValueError, match="max_topics"):
            ContextWindowAssembler(max_topics=0)

    def test_zero_max_articles_per_topic_raises(self) -> None:
        with pytest.raises(ValueError, match="max_articles_per_topic"):
            ContextWindowAssembler(max_articles_per_topic=0)

    def test_valid_defaults_accepted(self) -> None:
        assembler = ContextWindowAssembler()
        assert assembler is not None

    def test_custom_values_accepted(self) -> None:
        assembler = ContextWindowAssembler(
            token_budget=4000, max_topics=3, max_articles_per_topic=2
        )
        assert assembler is not None


# ── Empty graph ───────────────────────────────────────────────────────────────

class TestEmptyGraph:
    def test_empty_graph_returns_empty_context(self) -> None:
        graph = make_graph()
        ctx = ContextWindowAssembler().assemble(graph, [])
        assert isinstance(ctx, AssemblyContext)
        assert len(ctx.selected_topics) == 0

    def test_empty_graph_has_zero_token_estimate(self) -> None:
        graph = make_graph()
        ctx = ContextWindowAssembler().assemble(graph, [])
        assert ctx.token_estimate == 0

    def test_empty_graph_has_no_debt_topics(self) -> None:
        graph = make_graph()
        ctx = ContextWindowAssembler().assemble(graph, [])
        assert len(ctx.debt_topics) == 0

    def test_articles_without_graph_produces_empty_context(self) -> None:
        graph = make_graph()
        topic = make_topic()
        articles = [make_article(topic)]
        ctx = ContextWindowAssembler().assemble(graph, articles)
        assert len(ctx.selected_topics) == 0


# ── Topic selection ───────────────────────────────────────────────────────────

class TestTopicSelection:
    def test_single_topic_selected(self) -> None:
        t = make_topic()
        graph = make_graph(t)
        ctx = ContextWindowAssembler().assemble(graph, [])
        assert len(ctx.selected_topics) == 1

    def test_max_topics_cap_enforced(self) -> None:
        topics = [make_topic(name=f"topic {i}", recency_score=0.9 - i * 0.1) for i in range(6)]
        graph = make_graph(*topics)
        ctx = ContextWindowAssembler(max_topics=3).assemble(graph, [])
        assert len(ctx.selected_topics) <= 3

    def test_topics_selected_by_composite_score_descending(self) -> None:
        # high recency + high frequency → higher composite score
        t_high = make_topic(name="hot topic", frequency=5, recency_score=0.95, topic_id="high")
        t_low = make_topic(name="cold topic", frequency=1, recency_score=0.1, topic_id="low")
        graph = make_graph(t_high, t_low)
        ctx = ContextWindowAssembler(max_topics=1).assemble(graph, [])
        assert ctx.selected_topics[0].id == t_high.id

    def test_resolved_topics_not_selected(self) -> None:
        t_resolved = make_topic(
            name="resolved topic",
            curiosity_type=CuriosityType.RESOLVED,
            recency_score=0.99,
        )
        t_active = make_topic(name="active topic", recency_score=0.5)
        graph = make_graph(t_resolved, t_active)
        ctx = ContextWindowAssembler().assemble(graph, [])
        names = [t.name for t in ctx.selected_topics]
        assert "resolved topic" not in names
        assert "active topic" in names


# ── Article grouping ──────────────────────────────────────────────────────────

class TestArticleGrouping:
    def test_articles_grouped_by_topic_id(self) -> None:
        t = make_topic()
        a1 = make_article(t, title="Paper One", url="https://arxiv.org/abs/1")
        a2 = make_article(t, title="Paper Two", url="https://arxiv.org/abs/2")
        graph = make_graph(t)
        ctx = ContextWindowAssembler().assemble(graph, [a1, a2])
        assert t.id in ctx.articles_by_topic_id
        assert len(ctx.articles_by_topic_id[t.id]) == 2

    def test_articles_for_unselected_topic_not_included(self) -> None:
        t_selected = make_topic(name="selected", recency_score=0.9)
        t_other = make_topic(name="other", recency_score=0.1)
        a_other = make_article(t_other, title="Article for other topic")
        graph = make_graph(t_selected, t_other)
        ctx = ContextWindowAssembler(max_topics=1).assemble(graph, [a_other])
        # Only t_selected is selected; its article list should be empty
        assert ctx.articles_by_topic_id.get(t_other.id) is None

    def test_topic_with_no_articles_has_empty_list(self) -> None:
        t = make_topic()
        graph = make_graph(t)
        ctx = ContextWindowAssembler().assemble(graph, [])
        assert t.id in ctx.articles_by_topic_id
        assert ctx.articles_by_topic_id[t.id] == []

    def test_articles_sorted_by_relevance_score_descending(self) -> None:
        t = make_topic()
        a_low = make_article(t, title="Low Relevance", url="https://example.com/1", relevance_score=0.3)
        a_high = make_article(t, title="High Relevance", url="https://example.com/2", relevance_score=0.95)
        graph = make_graph(t)
        ctx = ContextWindowAssembler().assemble(graph, [a_low, a_high])
        articles = ctx.articles_by_topic_id[t.id]
        assert articles[0].relevance_score >= articles[1].relevance_score

    def test_max_articles_per_topic_enforced(self) -> None:
        t = make_topic()
        articles = [
            make_article(t, title=f"Paper {i}", url=f"https://example.com/{i}",
                         relevance_score=1.0 - i * 0.1)
            for i in range(5)
        ]
        graph = make_graph(t)
        ctx = ContextWindowAssembler(max_articles_per_topic=2).assemble(graph, articles)
        assert len(ctx.articles_by_topic_id[t.id]) == 2

    def test_max_articles_keeps_highest_relevance(self) -> None:
        t = make_topic()
        articles = [
            make_article(t, title=f"Paper {i}", url=f"https://example.com/{i}",
                         relevance_score=float(i) / 10)
            for i in range(5)
        ]
        graph = make_graph(t)
        ctx = ContextWindowAssembler(max_articles_per_topic=2).assemble(graph, articles)
        kept = ctx.articles_by_topic_id[t.id]
        assert all(a.relevance_score >= 0.3 for a in kept)


# ── Debt topics ───────────────────────────────────────────────────────────────

class TestDebtTopics:
    def test_recurring_topic_with_debt_appears_in_debt_topics(self) -> None:
        t = make_topic(
            name="debt topic",
            debt_score=2.5,
            curiosity_type=CuriosityType.RECURRING,
        )
        graph = make_graph(t)
        ctx = ContextWindowAssembler().assemble(graph, [])
        assert any(dt.name == "debt topic" for dt in ctx.debt_topics)

    def test_shallow_topic_not_in_debt_topics(self) -> None:
        t = make_topic(name="shallow", debt_score=0.0, curiosity_type=CuriosityType.SHALLOW)
        graph = make_graph(t)
        ctx = ContextWindowAssembler().assemble(graph, [])
        assert len(ctx.debt_topics) == 0

    def test_debt_topics_only_from_selected_topics(self) -> None:
        # Debt topic that scores low enough to not be selected
        t_debt_low = make_topic(
            name="debt but low score",
            recency_score=0.01,
            frequency=1,
            debt_score=5.0,
            curiosity_type=CuriosityType.RECURRING,
        )
        t_high = make_topic(name="high score", recency_score=0.99, frequency=10)
        graph = make_graph(t_debt_low, t_high)
        ctx = ContextWindowAssembler(max_topics=1).assemble(graph, [])
        # t_debt_low was not selected (max_topics=1, high score wins)
        selected_ids = {t.id for t in ctx.selected_topics}
        for dt in ctx.debt_topics:
            assert dt.id in selected_ids


# ── Token budget ──────────────────────────────────────────────────────────────

class TestTokenBudget:
    def test_very_small_budget_drops_topics(self) -> None:
        topics = [
            make_topic(name=f"topic with a fairly long name {i}", recency_score=0.9 - i * 0.1)
            for i in range(5)
        ]
        graph = make_graph(*topics)
        ctx = ContextWindowAssembler(token_budget=10).assemble(graph, [])
        # Budget is tiny — should include very few (possibly zero) topics
        assert len(ctx.selected_topics) <= 2

    def test_token_estimate_fits_within_budget(self) -> None:
        topics = [make_topic(name=f"topic {i}", recency_score=0.9 - i * 0.05) for i in range(5)]
        articles = [
            make_article(t, title=f"Article {i}", url=f"https://example.com/{i}")
            for i, t in enumerate(topics)
        ]
        graph = make_graph(*topics)
        budget = 2000
        ctx = ContextWindowAssembler(token_budget=budget).assemble(graph, articles)
        assert ctx.token_estimate <= budget

    def test_large_budget_includes_all_topics(self) -> None:
        topics = [make_topic(name=f"t{i}") for i in range(3)]
        graph = make_graph(*topics)
        ctx = ContextWindowAssembler(token_budget=100_000).assemble(graph, [])
        assert len(ctx.selected_topics) == 3


# ── Token estimation ──────────────────────────────────────────────────────────

class TestTokenEstimation:
    def test_nonempty_context_has_positive_token_estimate(self) -> None:
        t = make_topic()
        graph = make_graph(t)
        ctx = ContextWindowAssembler().assemble(graph, [])
        assert ctx.token_estimate > 0

    def test_more_content_gives_higher_token_estimate(self) -> None:
        t = make_topic()
        a_short = make_article(t, summary="Short.", url="https://example.com/s")
        a_long = make_article(t, summary="x" * 1000, url="https://example.com/l")
        graph = make_graph(t)
        ctx_short = ContextWindowAssembler().assemble(graph, [a_short])
        ctx_long = ContextWindowAssembler().assemble(graph, [a_long])
        assert ctx_long.token_estimate > ctx_short.token_estimate


# ── AssemblyContext structure ─────────────────────────────────────────────────

class TestAssemblyContextStructure:
    def test_assembled_at_is_set(self) -> None:
        graph = make_graph()
        ctx = ContextWindowAssembler().assemble(graph, [])
        assert isinstance(ctx.assembled_at, datetime)
        assert ctx.assembled_at.tzinfo is not None

    def test_context_is_immutable_pydantic_model(self) -> None:
        graph = make_graph()
        ctx = ContextWindowAssembler().assemble(graph, [])
        with pytest.raises(Exception):
            ctx.token_estimate = 999  # type: ignore[misc]

    def test_multiple_topics_all_present_in_articles_dict(self) -> None:
        t1 = make_topic(name="topic one", topic_id="t1")
        t2 = make_topic(name="topic two", topic_id="t2")
        graph = make_graph(t1, t2)
        ctx = ContextWindowAssembler().assemble(graph, [])
        assert "t1" in ctx.articles_by_topic_id
        assert "t2" in ctx.articles_by_topic_id
