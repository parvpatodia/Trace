"""
TracePipeline — orchestrates the full Trace pipeline as a LangGraph StateGraph.

Pipeline stages (sequential):
  1. collect_signals  — runs all SignalCollectors concurrently
  2. build_graph      — CuriosityGraphBuilder (which calls TopicExtractor internally)
  3. scrape_articles  — runs all ArticleScrapers for each topic concurrently
  4. assemble_context — ContextWindowAssembler packs content into token budget
  5. compose_newsletter — NewsletterComposer calls Claude

Error contract:
  - Collector failures: recorded in state.errors, pipeline continues
  - Scraper failures: recorded in state.errors, pipeline continues
  - No signals after collection: raises PipelineError (unrecoverable)
  - Graph build failure: raises PipelineError (unrecoverable)
  - NewsletterGenerationError: re-raised as PipelineError

WHY LangGraph StateGraph:
  Gives a typed, inspectable execution graph with built-in state passing.
  Each node receives the full state and returns a partial update dict.
  This makes individual nodes unit-testable without running the full graph.

WHY builder owns the extractor, not the pipeline:
  CuriosityGraphBuilder encapsulates the two-pass algorithm (extract → score).
  Exposing the extractor separately at the pipeline level would duplicate the
  extract call. The pipeline delegates entirely to builder.build().
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, TypedDict

_log = logging.getLogger(__name__)

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict

from trace.audit.writer import AuditWriter
from trace.composer.assembler import AssemblyContext, ContextWindowAssembler
from trace.composer.newsletter import NewsletterComposer, NewsletterGenerationError
from trace.graph.builder import CuriosityGraphBuilder
from trace.models import (
    AuditEntry,
    CuriosityGraph,
    Newsletter,
    RawSignal,
    ScrapedArticle,
    Topic,
)
from trace.scraper.base import ArticleScraper
from trace.signals.base import SignalCollector


class PipelineError(Exception):
    pass


class PipelineResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    newsletter: Newsletter
    graph: CuriosityGraph
    collected_signals: tuple[RawSignal, ...]
    errors: list[str]
    completed_at: datetime
    user_id: str = ""
    user_email: str = ""


class _State(TypedDict):
    signals: list[RawSignal]
    graph: CuriosityGraph | None
    articles: list[ScrapedArticle]
    context: AssemblyContext | None
    newsletter: Newsletter | None
    errors: list[str]


class TracePipeline:
    """
    Wires all Trace components into an executable LangGraph pipeline.

    Parameters:
        collectors: list of SignalCollector instances (≥1 required).
        scrapers: list of ArticleScraper instances (≥1 required).
        builder: CuriosityGraphBuilder — owns the TopicExtractor internally.
        assembler: ContextWindowAssembler for packing the context window.
        composer: NewsletterComposer for calling Claude.
    """

    def __init__(
        self,
        collectors: list[SignalCollector],
        scrapers: list[ArticleScraper],
        builder: CuriosityGraphBuilder,
        assembler: ContextWindowAssembler,
        composer: NewsletterComposer,
        max_concurrent_scrapers: int = 5,
        audit_writer: AuditWriter | None = None,
    ) -> None:
        if not collectors:
            raise ValueError("collectors must be a non-empty list")
        if not scrapers:
            raise ValueError("scrapers must be a non-empty list")
        if builder is None:
            raise ValueError("builder must not be None")
        if assembler is None:
            raise ValueError("assembler must not be None")
        if composer is None:
            raise ValueError("composer must not be None")
        if max_concurrent_scrapers < 1:
            raise ValueError("max_concurrent_scrapers must be >= 1")

        self._collectors = collectors
        self._scrapers = scrapers
        self._builder = builder
        self._assembler = assembler
        self._composer = composer
        self._max_concurrent_scrapers = max_concurrent_scrapers
        self._audit_writer = audit_writer
        self._graph = self._build_graph()

    # ── LangGraph construction ──────────────────────────────────────────────

    def _build_graph(self) -> Any:
        g: StateGraph = StateGraph(_State)
        g.add_node("collect_signals", self._node_collect_signals)
        g.add_node("build_graph", self._node_build_graph)
        g.add_node("scrape_articles", self._node_scrape_articles)
        g.add_node("assemble_context", self._node_assemble_context)
        g.add_node("compose_newsletter", self._node_compose_newsletter)

        g.set_entry_point("collect_signals")
        g.add_edge("collect_signals", "build_graph")
        g.add_edge("build_graph", "scrape_articles")
        g.add_edge("scrape_articles", "assemble_context")
        g.add_edge("assemble_context", "compose_newsletter")
        g.add_edge("compose_newsletter", END)

        return g.compile()

    # ── Node implementations ────────────────────────────────────────────────

    async def _node_collect_signals(self, state: _State) -> dict:
        _log.info("collect_signals: running %d collector(s)", len(self._collectors))
        tasks = [c.collect() for c in self._collectors]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals: list[RawSignal] = []
        errors: list[str] = list(state.get("errors", []))
        for collector, result in zip(self._collectors, results):
            if isinstance(result, Exception):
                errors.append(
                    f"[{collector.source.value}] collection failed: {result}"
                )
                _log.warning("Collector %s failed: %s", collector.source.value, result)
            else:
                signals.extend(result)
                _log.debug("Collector %s yielded %d signal(s)", collector.source.value, len(result))

        if not signals:
            raise PipelineError(
                "No signals collected from any source — cannot build graph"
            )

        _log.info("collect_signals: %d total signal(s) collected", len(signals))
        return {"signals": signals, "errors": errors}

    async def _node_build_graph(self, state: _State) -> dict:
        _log.info("build_graph: building curiosity graph from %d signal(s)", len(state["signals"]))
        try:
            graph = await self._builder.build(state["signals"])
        except Exception as e:
            raise PipelineError(f"Failed to build curiosity graph: {e}") from e
        _log.info("build_graph: %d topic(s) scored", len(graph.topics))
        return {"graph": graph}

    async def _node_scrape_articles(self, state: _State) -> dict:
        graph: CuriosityGraph = state["graph"]
        errors: list[str] = list(state.get("errors", []))

        if not graph.topics:
            _log.info("scrape_articles: no topics in graph — skipping")
            return {"articles": [], "errors": errors}

        topics = list(graph.top_n(len(graph.topics)))
        _log.info("scrape_articles: %d topic(s) × %d scraper(s)", len(topics), len(self._scrapers))
        semaphore = asyncio.Semaphore(self._max_concurrent_scrapers)

        async def _scrape_limited(scraper: ArticleScraper, topic: Topic) -> list[ScrapedArticle]:
            async with semaphore:
                return await scraper.scrape(topic)

        tasks = [
            _scrape_limited(scraper, topic)
            for topic in topics
            for scraper in self._scrapers
        ]
        scraper_sources = [
            (scraper.source, topic)
            for topic in topics
            for scraper in self._scrapers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        raw_articles: list[ScrapedArticle] = []
        for (source, topic), result in zip(scraper_sources, results):
            if isinstance(result, Exception):
                errors.append(
                    f"scraper [{source.value}] failed for topic '{topic.name}': {result}"
                )
                _log.warning("Scraper %s failed for topic '%s': %s", source.value, topic.name, result)
            else:
                raw_articles.extend(result)

        # Deduplicate by URL across all scrapers — keep the copy with the highest
        # relevance_score so the best-ranked match survives when e.g. ArXiv and
        # Apify both return the same paper.
        seen_urls: dict[str, ScrapedArticle] = {}
        for article in raw_articles:
            existing = seen_urls.get(article.url)
            if existing is None or article.relevance_score > existing.relevance_score:
                seen_urls[article.url] = article
        articles = list(seen_urls.values())

        _log.info(
            "scrape_articles: %d article(s) collected (%d deduped from %d)",
            len(articles), len(raw_articles) - len(articles), len(raw_articles),
        )
        return {"articles": articles, "errors": errors}

    async def _node_assemble_context(self, state: _State) -> dict:
        ctx = self._assembler.assemble(
            state["graph"],
            state["articles"],
            signals=state.get("signals") or [],
        )
        return {"context": ctx}

    async def _node_compose_newsletter(self, state: _State) -> dict:
        _log.info("compose_newsletter: calling Claude")
        try:
            newsletter = await self._composer.compose(state["context"])
        except NewsletterGenerationError as e:
            raise PipelineError(f"Failed to compose newsletter: {e}") from e
        _log.info("compose_newsletter: newsletter generated (%d section(s))", len(newsletter.sections))
        return {"newsletter": newsletter}

    # ── Public interface ────────────────────────────────────────────────────

    async def run(
        self,
        user_id: str = "",
        user_email: str = "",
    ) -> PipelineResult:
        initial: _State = {
            "signals": [],
            "graph": None,
            "articles": [],
            "context": None,
            "newsletter": None,
            "errors": [],
        }
        final = await self._graph.ainvoke(initial)
        result = PipelineResult(
            newsletter=final["newsletter"],
            graph=final["graph"],
            collected_signals=tuple(final["signals"]),
            errors=final["errors"],
            completed_at=datetime.now(timezone.utc),
            user_id=user_id,
            user_email=user_email,
        )
        if self._audit_writer is not None:
            await self._write_audit_entries(final, result, user_id, user_email)
        return result

    async def run_from_graph(
        self,
        graph: CuriosityGraph,
        user_id: str = "",
        user_email: str = "",
    ) -> PipelineResult:
        """Generate a fresh newsletter from a previously built CuriosityGraph.

        Skips signal collection and topic extraction (stages 1-2). Runs only
        scrape_articles → assemble_context → compose_newsletter (stages 3-5).
        This enables daily newsletter regeneration without re-uploading history:
        the user's curiosity interests are preserved from their initial upload,
        while the articles are freshly scraped from today's content.
        """
        scrape = await self._node_scrape_articles({"graph": graph, "errors": []})
        ctx = await self._node_assemble_context({
            "graph": graph,
            "articles": scrape["articles"],
        })
        nl = await self._node_compose_newsletter({"context": ctx["context"]})
        return PipelineResult(
            newsletter=nl["newsletter"],
            graph=graph,
            collected_signals=(),
            errors=scrape["errors"],
            completed_at=datetime.now(timezone.utc),
            user_id=user_id,
            user_email=user_email,
        )

    async def _write_audit_entries(
        self,
        final: dict,
        result: PipelineResult,
        user_id: str,
        user_email: str,
    ) -> None:
        """Write one AuditEntry per pipeline stage after a successful run."""
        on_behalf = user_email or user_id or "anonymous"
        signals: list[RawSignal] = final.get("signals", [])
        graph: CuriosityGraph = final["graph"]
        articles: list[ScrapedArticle] = final.get("articles", [])
        newsletter = result.newsletter

        # 1 — Signal collection
        sources: dict[str, int] = {}
        for sig in signals:
            sources[sig.source.value] = sources.get(sig.source.value, 0) + 1
        await self._audit_writer.record(  # type: ignore[union-attr]
            AuditEntry(
                pipeline_step="collect_signals",
                decision=f"Collected {len(signals)} signal(s) across {len(sources)} source(s)",
                reasoning=(
                    f"Running on behalf of {on_behalf}. "
                    "Signals represent the user's digital curiosity footprint."
                ),
                inputs_summary={"sources_active": list(sources.keys())},
                outputs_summary={"signal_count": len(signals), "by_source": sources},
            )
        )

        # 2 — Graph building
        topic_summaries = [
            {
                "name": t.name,
                "frequency": t.frequency,
                "recency_score": round(t.recency_score, 3),
                "type": t.curiosity_type.value,
            }
            for t in graph.topics
        ]
        await self._audit_writer.record(  # type: ignore[union-attr]
            AuditEntry(
                pipeline_step="build_graph",
                decision=f"Identified {len(graph.topics)} curiosity topic(s) from {len(signals)} signal(s)",
                reasoning="Claude extracted topics by clustering semantically related signals. "
                          "Topics with higher frequency and recency float to the top.",
                inputs_summary={"signal_count": len(signals)},
                outputs_summary={"topic_count": len(graph.topics), "topics": topic_summaries},
            )
        )

        # 3 — Article scraping
        await self._audit_writer.record(  # type: ignore[union-attr]
            AuditEntry(
                pipeline_step="scrape_articles",
                decision=f"Scraped {len(articles)} article(s) across {len(self._scrapers)} source(s)",
                reasoning="Articles provide fresh, relevant content for each curiosity topic.",
                inputs_summary={"topic_count": len(graph.topics), "scraper_count": len(self._scrapers)},
                outputs_summary={"article_count": len(articles), "errors": result.errors},
            )
        )

        # 4 — Newsletter composition
        section_summaries = [
            {
                "title": s.title,
                "type": s.section_type,
                "audit_reasoning": s.audit_reasoning,
            }
            for s in newsletter.sections
        ]
        await self._audit_writer.record(  # type: ignore[union-attr]
            AuditEntry(
                pipeline_step="compose_newsletter",
                decision=f"Generated newsletter '{newsletter.subject_line}' with {len(newsletter.sections)} section(s)",
                reasoning=(
                    f"Claude composed a personalized newsletter for {on_behalf} "
                    "based on their curiosity graph and scraped articles."
                ),
                inputs_summary={"topic_count": len(graph.topics), "article_count": len(articles)},
                outputs_summary={
                    "subject_line": newsletter.subject_line,
                    "sections": section_summaries,
                },
            )
        )
