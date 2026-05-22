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
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict

from trace.composer.assembler import AssemblyContext, ContextWindowAssembler
from trace.composer.newsletter import NewsletterComposer, NewsletterGenerationError
from trace.graph.builder import CuriosityGraphBuilder
from trace.models import (
    CuriosityGraph,
    Newsletter,
    RawSignal,
    ScrapedArticle,
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

        self._collectors = collectors
        self._scrapers = scrapers
        self._builder = builder
        self._assembler = assembler
        self._composer = composer
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
        tasks = [c.collect() for c in self._collectors]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals: list[RawSignal] = []
        errors: list[str] = list(state.get("errors", []))
        for collector, result in zip(self._collectors, results):
            if isinstance(result, Exception):
                errors.append(
                    f"[{collector.source.value}] collection failed: {result}"
                )
            else:
                signals.extend(result)

        if not signals:
            raise PipelineError(
                "No signals collected from any source — cannot build graph"
            )

        return {"signals": signals, "errors": errors}

    async def _node_build_graph(self, state: _State) -> dict:
        try:
            graph = await self._builder.build(state["signals"])
        except Exception as e:
            raise PipelineError(f"Failed to build curiosity graph: {e}") from e
        return {"graph": graph}

    async def _node_scrape_articles(self, state: _State) -> dict:
        graph: CuriosityGraph = state["graph"]
        errors: list[str] = list(state.get("errors", []))

        if not graph.topics:
            return {"articles": [], "errors": errors}

        topics = list(graph.top_n(len(graph.topics)))
        tasks = [
            scraper.scrape(topic)
            for topic in topics
            for scraper in self._scrapers
        ]
        scraper_sources = [
            (scraper.source, topic)
            for topic in topics
            for scraper in self._scrapers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        articles: list[ScrapedArticle] = []
        for (source, topic), result in zip(scraper_sources, results):
            if isinstance(result, Exception):
                errors.append(
                    f"scraper [{source.value}] failed for topic '{topic.name}': {result}"
                )
            else:
                articles.extend(result)

        return {"articles": articles, "errors": errors}

    async def _node_assemble_context(self, state: _State) -> dict:
        ctx = self._assembler.assemble(state["graph"], state["articles"])
        return {"context": ctx}

    async def _node_compose_newsletter(self, state: _State) -> dict:
        try:
            newsletter = await self._composer.compose(state["context"])
        except NewsletterGenerationError as e:
            raise PipelineError(f"Failed to compose newsletter: {e}") from e
        return {"newsletter": newsletter}

    # ── Public interface ────────────────────────────────────────────────────

    async def run(self) -> PipelineResult:
        initial: _State = {
            "signals": [],
            "graph": None,
            "articles": [],
            "context": None,
            "newsletter": None,
            "errors": [],
        }
        final = await self._graph.ainvoke(initial)
        return PipelineResult(
            newsletter=final["newsletter"],
            graph=final["graph"],
            collected_signals=tuple(final["signals"]),
            errors=final["errors"],
            completed_at=datetime.now(timezone.utc),
        )
