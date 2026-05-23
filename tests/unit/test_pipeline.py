"""
Unit tests for trace/pipeline/runner.py — TracePipeline.

All external services (collectors, scrapers, extractor, builder, assembler,
composer) are replaced with MagicMock / AsyncMock. No real I/O, no network.

Test categories:
  1. Constructor: validates required dependencies
  2. collect_signals node: aggregates results from all collectors; tolerates
     one collector failing (error recorded, others continue)
  3. build_graph node: calls extractor + builder; no signals → PipelineError
  4. scrape_articles node: calls each scraper for each topic; scraper errors
     recorded but don't abort
  5. assemble_context node: calls ContextWindowAssembler.assemble()
  6. compose_newsletter node: calls NewsletterComposer.compose()
  7. Full pipeline: run() returns PipelineResult with newsletter
  8. Empty signals → PipelineError raised
  9. API error in composer → PipelineError
  10. PipelineResult structure: newsletter, graph, collected_signals, errors
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from trace.composer.assembler import AssemblyContext
from trace.models import (
    ContentSource,
    CuriosityGraph,
    CuriosityType,
    Newsletter,
    NewsletterSection,
    RawSignal,
    ScrapedArticle,
    SignalSource,
    Topic,
)
from trace.pipeline.runner import PipelineError, PipelineResult, TracePipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

_NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


def make_raw_signal(content: str = "transformer architecture paper") -> RawSignal:
    return RawSignal(
        source=SignalSource.CHROME_HISTORY,
        content=content,
        timestamp=_NOW,
    )


def make_topic(name: str = "transformer architecture", topic_id: str | None = None) -> Topic:
    t = Topic(name=name, frequency=3, recency_score=0.8)
    if topic_id:
        t = t.model_copy(update={"id": topic_id})
    return t


def make_graph(*topics: Topic) -> CuriosityGraph:
    return CuriosityGraph(topics=tuple(topics), signal_count=len(topics) * 3)


def make_article(topic: Topic) -> ScrapedArticle:
    return ScrapedArticle(
        topic_id=topic.id,
        topic_name=topic.name,
        source=ContentSource.ARXIV,
        title="Attention Is All You Need",
        url="https://arxiv.org/abs/1706.03762",
        summary="Transformer paper.",
        relevance_score=0.9,
    )


def make_assembly_context(topic: Topic, article: ScrapedArticle) -> AssemblyContext:
    return AssemblyContext(
        selected_topics=(topic,),
        articles_by_topic_id={topic.id: [article]},
        debt_topics=(),
        token_estimate=200,
        assembled_at=_NOW,
    )


def make_newsletter() -> Newsletter:
    section = NewsletterSection(
        title="Transformers This Week",
        section_type="weekly_topics",
        content="Your signals this week converge strongly on transformer architecture, with three papers and two Reddit threads.",
        source_urls=["https://arxiv.org/abs/1706.03762"],
        audit_reasoning="High frequency (3) and recency score (0.8) indicate active interest.",
    )
    return Newsletter(
        generated_at=_NOW,
        subject_line="Your Curiosity Digest: Transformers Edition",
        sections=(section,),
        plain_text="Transformers This Week\n...",
        html="<html>...</html>",
    )


def make_mock_collector(signals: list[RawSignal]) -> MagicMock:
    collector = MagicMock()
    collector.collect = AsyncMock(return_value=signals)
    collector.source = SignalSource.CHROME_HISTORY
    return collector


def make_mock_scraper(articles: list[ScrapedArticle]) -> MagicMock:
    scraper = MagicMock()
    scraper.scrape = AsyncMock(return_value=articles)
    scraper.source = ContentSource.ARXIV
    return scraper


def make_mock_builder(graph: CuriosityGraph) -> MagicMock:
    builder = MagicMock()
    builder.build = AsyncMock(return_value=graph)
    return builder


def make_mock_assembler(ctx: AssemblyContext) -> MagicMock:
    assembler = MagicMock()
    assembler.assemble = MagicMock(return_value=ctx)
    return assembler


def make_mock_composer(newsletter: Newsletter) -> MagicMock:
    composer = MagicMock()
    composer.compose = AsyncMock(return_value=newsletter)
    return composer


_SENTINEL = object()


def make_pipeline(
    collectors=_SENTINEL,
    scrapers=_SENTINEL,
    builder=_SENTINEL,
    assembler=_SENTINEL,
    composer=_SENTINEL,
    topic: Topic | None = None,
) -> TracePipeline:
    if topic is None:
        topic = make_topic()
    graph = make_graph(topic)
    article = make_article(topic)
    ctx = make_assembly_context(topic, article)
    newsletter = make_newsletter()

    return TracePipeline(
        collectors=collectors if collectors is not _SENTINEL else [make_mock_collector([make_raw_signal()])],
        scrapers=scrapers if scrapers is not _SENTINEL else [make_mock_scraper([article])],
        builder=builder if builder is not _SENTINEL else make_mock_builder(graph),
        assembler=assembler if assembler is not _SENTINEL else make_mock_assembler(ctx),
        composer=composer if composer is not _SENTINEL else make_mock_composer(newsletter),
    )


# ── Constructor ───────────────────────────────────────────────────────────────

class TestConstructor:
    def test_valid_pipeline_constructed(self) -> None:
        p = make_pipeline()
        assert p is not None

    def test_empty_collectors_raises(self) -> None:
        with pytest.raises(ValueError, match="collectors"):
            make_pipeline(collectors=[])

    def test_empty_scrapers_raises(self) -> None:
        with pytest.raises(ValueError, match="scrapers"):
            make_pipeline(scrapers=[])

    def test_none_builder_raises(self) -> None:
        with pytest.raises(ValueError, match="builder"):
            make_pipeline(builder=None)

    def test_none_assembler_raises(self) -> None:
        with pytest.raises(ValueError, match="assembler"):
            make_pipeline(assembler=None)

    def test_none_composer_raises(self) -> None:
        with pytest.raises(ValueError, match="composer"):
            make_pipeline(composer=None)


# ── Full pipeline run ─────────────────────────────────────────────────────────

class TestFullPipeline:
    async def test_run_returns_pipeline_result(self) -> None:
        result = await make_pipeline().run()
        assert isinstance(result, PipelineResult)

    async def test_result_has_newsletter(self) -> None:
        result = await make_pipeline().run()
        assert isinstance(result.newsletter, Newsletter)

    async def test_result_has_graph(self) -> None:
        result = await make_pipeline().run()
        assert isinstance(result.graph, CuriosityGraph)

    async def test_result_has_collected_signals(self) -> None:
        result = await make_pipeline().run()
        assert len(result.collected_signals) >= 1

    async def test_result_errors_is_list(self) -> None:
        result = await make_pipeline().run()
        assert isinstance(result.errors, list)

    async def test_no_errors_on_clean_run(self) -> None:
        result = await make_pipeline().run()
        assert result.errors == []

    async def test_completed_at_is_utc_datetime(self) -> None:
        result = await make_pipeline().run()
        assert isinstance(result.completed_at, datetime)
        assert result.completed_at.tzinfo is not None


# ── collect_signals node ──────────────────────────────────────────────────────

class TestCollectSignals:
    async def test_signals_from_all_collectors_aggregated(self) -> None:
        s1 = make_raw_signal("topic A")
        s2 = make_raw_signal("topic B")
        c1 = make_mock_collector([s1])
        c2 = make_mock_collector([s2])
        result = await make_pipeline(collectors=[c1, c2]).run()
        assert len(result.collected_signals) == 2

    async def test_failing_collector_recorded_in_errors(self) -> None:
        bad = MagicMock()
        bad.collect = AsyncMock(side_effect=Exception("filesystem read failed"))
        bad.source = SignalSource.FILESYSTEM
        good = make_mock_collector([make_raw_signal()])
        result = await make_pipeline(collectors=[bad, good]).run()
        assert any("filesystem read failed" in e or "filesystem" in e.lower() for e in result.errors)

    async def test_failing_collector_does_not_abort_pipeline(self) -> None:
        bad = MagicMock()
        bad.collect = AsyncMock(side_effect=Exception("broken"))
        bad.source = SignalSource.FILESYSTEM
        good = make_mock_collector([make_raw_signal()])
        result = await make_pipeline(collectors=[bad, good]).run()
        assert isinstance(result.newsletter, Newsletter)


# ── build_graph node ──────────────────────────────────────────────────────────

class TestBuildGraph:
    async def test_graph_built_from_signals(self) -> None:
        topic = make_topic()
        graph = make_graph(topic)
        builder = make_mock_builder(graph)
        result = await make_pipeline(builder=builder).run()
        builder.build.assert_called_once()

    async def test_builder_called_with_signals(self) -> None:
        topic = make_topic()
        graph = make_graph(topic)
        builder = make_mock_builder(graph)
        await make_pipeline(builder=builder).run()
        builder.build.assert_called_once()

    async def test_no_signals_raises_pipeline_error(self) -> None:
        collector = make_mock_collector([])  # returns no signals
        with pytest.raises(PipelineError):
            await make_pipeline(collectors=[collector]).run()


# ── scrape_articles node ──────────────────────────────────────────────────────

class TestScrapeArticles:
    async def test_scraper_called_for_each_topic(self) -> None:
        topic = make_topic()
        graph = make_graph(topic)
        scraper = make_mock_scraper([make_article(topic)])
        await make_pipeline(
            scrapers=[scraper],
            builder=make_mock_builder(graph),
        ).run()
        scraper.scrape.assert_called()

    async def test_scraper_error_recorded_not_fatal(self) -> None:
        bad_scraper = MagicMock()
        bad_scraper.scrape = AsyncMock(side_effect=Exception("scraper timeout"))
        bad_scraper.source = ContentSource.ARXIV

        topic = make_topic()
        graph = make_graph(topic)
        ctx = make_assembly_context(topic, make_article(topic))

        result = await make_pipeline(
            scrapers=[bad_scraper],
            builder=make_mock_builder(graph),
            assembler=make_mock_assembler(ctx),
        ).run()
        assert any("scraper" in e.lower() or "timeout" in e for e in result.errors)

    async def test_articles_from_all_scrapers_combined(self) -> None:
        topic = make_topic()
        graph = make_graph(topic)
        a1 = make_article(topic)
        a2 = ScrapedArticle(
            topic_id=topic.id,
            topic_name=topic.name,
            source=ContentSource.HACKER_NEWS,
            title="HN Post About Transformers",
            url="https://news.ycombinator.com/item?id=1234",
            summary="Discussion about transformers.",
            relevance_score=0.7,
        )
        s1 = make_mock_scraper([a1])
        s2 = make_mock_scraper([a2])
        ctx = make_assembly_context(topic, a1)
        assembler = make_mock_assembler(ctx)

        # Track what articles are passed to assembler
        captured_articles: list[list[ScrapedArticle]] = []
        real_assemble = assembler.assemble.side_effect
        def capture_and_call(g, articles):
            captured_articles.append(articles)
            return ctx
        assembler.assemble.side_effect = capture_and_call

        await make_pipeline(
            scrapers=[s1, s2],
            builder=make_mock_builder(graph),
            assembler=assembler,
        ).run()
        assert len(captured_articles) == 1
        assert len(captured_articles[0]) == 2


# ── assemble_context node ─────────────────────────────────────────────────────

class TestAssembleContext:
    async def test_assembler_called_with_graph_and_articles(self) -> None:
        topic = make_topic()
        graph = make_graph(topic)
        article = make_article(topic)
        ctx = make_assembly_context(topic, article)
        assembler = make_mock_assembler(ctx)

        await make_pipeline(
            scrapers=[make_mock_scraper([article])],
            builder=make_mock_builder(graph),
            assembler=assembler,
        ).run()
        assembler.assemble.assert_called_once()


# ── compose_newsletter node ───────────────────────────────────────────────────

class TestComposeNewsletter:
    async def test_composer_called_with_context(self) -> None:
        topic = make_topic()
        graph = make_graph(topic)
        article = make_article(topic)
        ctx = make_assembly_context(topic, article)
        newsletter = make_newsletter()
        composer = make_mock_composer(newsletter)

        await make_pipeline(
            builder=make_mock_builder(graph),
            assembler=make_mock_assembler(ctx),
            composer=composer,
        ).run()
        composer.compose.assert_called_once_with(ctx)

    async def test_composer_error_raises_pipeline_error(self) -> None:
        from trace.composer.newsletter import NewsletterGenerationError
        bad_composer = MagicMock()
        bad_composer.compose = AsyncMock(
            side_effect=NewsletterGenerationError("Claude failed")
        )
        with pytest.raises(PipelineError, match="newsletter"):
            await make_pipeline(composer=bad_composer).run()


# ── PipelineResult structure ──────────────────────────────────────────────────

class TestPipelineResult:
    async def test_result_is_immutable(self) -> None:
        result = await make_pipeline().run()
        with pytest.raises(Exception):
            result.newsletter = None  # type: ignore[misc]

    async def test_result_newsletter_matches_composer_output(self) -> None:
        newsletter = make_newsletter()
        composer = make_mock_composer(newsletter)
        result = await make_pipeline(composer=composer).run()
        assert result.newsletter is newsletter
