"""
Smoke test: full pipeline from signal collection through newsletter delivery.

All external I/O is mocked (no network, no files, no Claude API). The purpose
is to verify that every module in the stack wires together correctly — that
the data shapes flowing between layers are compatible.

This test is more coarse-grained than unit tests: it constructs real
instances of most components (except API clients) and runs the full
TracePipeline.run() call, then validates the PipelineResult shape.

Marked `smoke` — run with: pytest -m smoke
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trace.composer.assembler import ContextWindowAssembler
from trace.composer.newsletter import NewsletterComposer
from trace.graph.builder import CuriosityGraphBuilder
from trace.graph.extractor import TopicExtractor
from trace.models import (
    ContentSource,
    CuriosityGraph,
    Newsletter,
    NewsletterSection,
    RawSignal,
    ScrapedArticle,
    SignalSource,
    Topic,
)
from trace.pipeline.runner import PipelineResult, TracePipeline
from trace.scraper.arxiv import ArXivScraper
from trace.scraper.hackernews import HackerNewsScraper
from trace.signals.base import SignalCollector

pytestmark = pytest.mark.smoke

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


# ── Fake collector that returns canned signals ────────────────────────────────

class FakeSignalCollector(SignalCollector):
    source = SignalSource.CHROME_HISTORY

    def __init__(self, signals: list[RawSignal]) -> None:
        self._signals = signals

    async def collect(self) -> list[RawSignal]:
        return self._signals


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_signals() -> list[RawSignal]:
    texts = [
        "transformer architecture attention mechanisms deep learning paper",
        "vision transformer ViT image recognition BERT architecture",
        "self-attention mechanism scaled dot product query key value",
    ]
    return [
        RawSignal(
            source=SignalSource.CHROME_HISTORY,
            content=t,
            timestamp=_NOW,
        )
        for t in texts
    ]


def _make_extractor_response(signals: list[RawSignal]) -> list[dict]:
    return [
        {
            "name": "transformer architecture",
            "aliases": ["attention mechanism", "ViT"],
            "signal_ids": [s.id for s in signals],
        }
    ]


def _make_claude_message(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=content)]
    return msg


_NEWSLETTER_JSON = json.dumps({
    "subject_line": "Curiosity Digest: Transformer Architecture Deep Dive",
    "sections": [
        {
            "title": "Transformers: Your Persistent Curiosity",
            "section_type": "weekly_topics",
            "content": (
                "Three of your browsing sessions this week converged on transformer "
                "architecture — specifically the attention mechanism that powers modern "
                "language and vision models. The Attention Is All You Need paper remains "
                "the canonical reference, but your signals suggest you're moving beyond "
                "basics into implementation details."
            ),
            "source_urls": ["https://arxiv.org/abs/1706.03762"],
            "audit_reasoning": (
                "Signal frequency 3, recency score 0.8, spanning 3 sources. "
                "Pattern indicates active learning rather than casual browsing."
            ),
        }
    ],
})


@pytest.fixture
def arxiv_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        "<entry>"
        "<id>https://arxiv.org/abs/1706.03762</id>"
        "<title>Attention Is All You Need</title>"
        "<summary>The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer.</summary>"
        "<published>2017-06-12T17:00:00Z</published>"
        "</entry>"
        "</feed>"
    )


@pytest.fixture
def hn_json() -> dict:
    return {
        "hits": [
            {
                "objectID": "15310501",
                "title": "Attention Is All You Need (2017)",
                "url": "https://arxiv.org/abs/1706.03762",
                "story_text": None,
                "created_at": "2017-06-13T09:00:00.000Z",
                "points": 834,
                "_tags": ["story"],
            }
        ]
    }


# ── Full pipeline smoke test ──────────────────────────────────────────────────

def _make_clients(signals: list[RawSignal]) -> tuple[MagicMock, MagicMock]:
    """Return (extractor_client, composer_client) — separate mocks, each with
    a fixed return_value so asyncio.to_thread ordering is never an issue.
    The extractor JSON uses the real signal IDs so the builder can match them."""
    extractor_json = json.dumps([
        {
            "name": "transformer architecture",
            "aliases": ["attention mechanism", "self-attention"],
            "signal_ids": [s.id for s in signals],
        }
    ])
    extractor_client = MagicMock()
    extractor_client.messages.create.return_value = _make_claude_message(extractor_json)

    composer_client = MagicMock()
    composer_client.messages.create.return_value = _make_claude_message(_NEWSLETTER_JSON)
    return extractor_client, composer_client


@pytest.mark.smoke
async def test_full_pipeline_end_to_end(arxiv_xml: str, hn_json: dict) -> None:
    """
    Runs the complete TracePipeline with:
    - Real FakeSignalCollector (canned signals)
    - Real TopicExtractor backed by dedicated mock Anthropic client
    - Real CuriosityGraphBuilder
    - Real ArXivScraper + HackerNewsScraper backed by mock httpx responses
    - Real ContextWindowAssembler
    - Real NewsletterComposer backed by dedicated mock Anthropic client
    """
    signals = _make_signals()
    extractor_client, composer_client = _make_clients(signals)

    import httpx
    import respx

    with respx.mock:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=arxiv_xml)
        )
        respx.get("https://hn.algolia.com/api/v1/search").mock(
            return_value=httpx.Response(200, json=hn_json)
        )

        extractor = TopicExtractor(client=extractor_client, batch_size=50)
        builder = CuriosityGraphBuilder(extractor=extractor, half_life_days=14)
        assembler = ContextWindowAssembler(token_budget=8_000, max_topics=5)
        composer = NewsletterComposer(client=composer_client)

        pipeline = TracePipeline(
            collectors=[FakeSignalCollector(signals)],
            scrapers=[ArXivScraper(), HackerNewsScraper()],
            extractor=extractor,
            builder=builder,
            assembler=assembler,
            composer=composer,
        )

        result = await pipeline.run()

    assert isinstance(result, PipelineResult)
    assert isinstance(result.newsletter, Newsletter)
    assert len(result.newsletter.sections) >= 1
    assert len(result.newsletter.subject_line) >= 10
    assert isinstance(result.graph, CuriosityGraph)
    assert len(result.collected_signals) == 3
    assert isinstance(result.completed_at, datetime)
    assert result.completed_at.tzinfo is not None


@pytest.mark.smoke
async def test_pipeline_with_partial_scraper_failure(arxiv_xml: str) -> None:
    """
    ArXiv succeeds; HN scraper returns 503. Pipeline should still complete
    with the ArXiv articles and record the HN error.
    """
    signals = _make_signals()
    extractor_client, composer_client = _make_clients(signals)

    import httpx
    import respx

    with respx.mock:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=arxiv_xml)
        )
        respx.get("https://hn.algolia.com/api/v1/search").mock(
            return_value=httpx.Response(503)
        )

        extractor = TopicExtractor(client=extractor_client, batch_size=50)
        builder = CuriosityGraphBuilder(extractor=extractor)
        assembler = ContextWindowAssembler(token_budget=8_000)
        composer = NewsletterComposer(client=composer_client)

        pipeline = TracePipeline(
            collectors=[FakeSignalCollector(signals)],
            scrapers=[ArXivScraper(), HackerNewsScraper()],
            extractor=extractor,
            builder=builder,
            assembler=assembler,
            composer=composer,
        )

        result = await pipeline.run()

    assert isinstance(result.newsletter, Newsletter)
    assert any("503" in e or "scraper" in e.lower() for e in result.errors)


@pytest.mark.smoke
async def test_pipeline_result_newsletter_renders_html(arxiv_xml: str, hn_json: dict) -> None:
    """Newsletter HTML and plain_text are non-empty after a successful run."""
    signals = _make_signals()
    extractor_client, composer_client = _make_clients(signals)

    import httpx
    import respx

    with respx.mock:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=arxiv_xml)
        )
        respx.get("https://hn.algolia.com/api/v1/search").mock(
            return_value=httpx.Response(200, json=hn_json)
        )

        extractor = TopicExtractor(client=extractor_client, batch_size=50)
        builder = CuriosityGraphBuilder(extractor=extractor)
        assembler = ContextWindowAssembler(token_budget=8_000)
        composer = NewsletterComposer(client=composer_client)

        pipeline = TracePipeline(
            collectors=[FakeSignalCollector(signals)],
            scrapers=[ArXivScraper(), HackerNewsScraper()],
            extractor=extractor,
            builder=builder,
            assembler=assembler,
            composer=composer,
        )

        result = await pipeline.run()

    assert len(result.newsletter.html) > 0
    assert len(result.newsletter.plain_text) > 0
