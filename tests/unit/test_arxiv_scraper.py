"""
Unit tests for trace/scraper/arxiv.py — ArXivScraper.

All HTTP calls are intercepted with respx (no real network).
A minimal but structurally valid Atom XML fixture covers the happy path.

Test categories:
  1. Source attribute
  2. Happy path: valid XML → correct ScrapedArticle fields
  3. Empty feed: no entries → empty list
  4. Partial entries: missing id or title → skipped
  5. URL normalisation: http:// → https://
  6. max_results respected
  7. Relevance scoring: position-based, bounded [0, 1]
  8. Published_at parsing: ISO 8601 UTC → aware datetime
  9. HTTP errors → ScraperError
  10. Malformed XML → ScraperError
  11. ScrapedArticle fields: topic_id, topic_name, source, title, url, summary
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
import respx

from trace.models import ContentSource, Topic
from trace.scraper.arxiv import ArXivScraper
from trace.scraper.base import ScraperError


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_topic(name: str = "transformer architecture") -> Topic:
    return Topic(name=name, frequency=3, recency_score=0.8)


def _entry(
    arxiv_id: str = "http://arxiv.org/abs/1706.03762v5",
    title: str = "Attention Is All You Need",
    summary: str = "The dominant sequence transduction models are based on complex neural networks.",
    published: str = "2017-06-12T17:00:00Z",
) -> str:
    return (
        f"<entry>"
        f"<id>{arxiv_id}</id>"
        f"<title>{title}</title>"
        f"<summary>{summary}</summary>"
        f"<published>{published}</published>"
        f"</entry>"
    )


def _feed(entries: list[str]) -> str:
    body = "".join(entries)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        f"{body}"
        "</feed>"
    )


_ONE_RESULT_XML = _feed([_entry()])
_TWO_RESULTS_XML = _feed([
    _entry(arxiv_id="http://arxiv.org/abs/1111.0001", title="Paper One"),
    _entry(arxiv_id="http://arxiv.org/abs/2222.0002", title="Paper Two"),
])
_EMPTY_FEED_XML = _feed([])


# ── Source attribute ──────────────────────────────────────────────────────────

class TestSourceAttribute:
    def test_source_is_arxiv(self) -> None:
        assert ArXivScraper().source == ContentSource.ARXIV


# ── Happy path ────────────────────────────────────────────────────────────────

class TestHappyPath:
    @respx.mock
    async def test_one_entry_produces_one_article(self) -> None:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=_ONE_RESULT_XML)
        )
        topic = make_topic()
        articles = await ArXivScraper().scrape(topic, max_results=5)
        assert len(articles) == 1

    @respx.mock
    async def test_article_title_matches_entry_title(self) -> None:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=_ONE_RESULT_XML)
        )
        articles = await ArXivScraper().scrape(make_topic())
        assert articles[0].title == "Attention Is All You Need"

    @respx.mock
    async def test_article_url_uses_https(self) -> None:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=_ONE_RESULT_XML)
        )
        articles = await ArXivScraper().scrape(make_topic())
        assert articles[0].url.startswith("https://")

    @respx.mock
    async def test_article_url_points_to_arxiv_abs(self) -> None:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=_ONE_RESULT_XML)
        )
        articles = await ArXivScraper().scrape(make_topic())
        assert "arxiv.org/abs/" in articles[0].url

    @respx.mock
    async def test_article_summary_is_entry_abstract(self) -> None:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=_ONE_RESULT_XML)
        )
        articles = await ArXivScraper().scrape(make_topic())
        assert "sequence transduction" in articles[0].summary

    @respx.mock
    async def test_article_source_is_arxiv(self) -> None:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=_ONE_RESULT_XML)
        )
        articles = await ArXivScraper().scrape(make_topic())
        assert articles[0].source == ContentSource.ARXIV

    @respx.mock
    async def test_article_topic_id_matches_topic(self) -> None:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=_ONE_RESULT_XML)
        )
        topic = make_topic()
        articles = await ArXivScraper().scrape(topic)
        assert articles[0].topic_id == topic.id

    @respx.mock
    async def test_article_topic_name_matches_topic(self) -> None:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=_ONE_RESULT_XML)
        )
        topic = make_topic("self-attention mechanism")
        articles = await ArXivScraper().scrape(topic)
        assert articles[0].topic_name == "self-attention mechanism"


# ── Empty feed ────────────────────────────────────────────────────────────────

class TestEmptyFeed:
    @respx.mock
    async def test_no_entries_returns_empty_list(self) -> None:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=_EMPTY_FEED_XML)
        )
        articles = await ArXivScraper().scrape(make_topic())
        assert articles == []


# ── Partial / malformed entries ───────────────────────────────────────────────

class TestPartialEntries:
    @respx.mock
    async def test_entry_missing_title_skipped(self) -> None:
        xml = _feed(["<entry><id>http://arxiv.org/abs/1111.0001v1</id></entry>"])
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=xml)
        )
        articles = await ArXivScraper().scrape(make_topic())
        assert articles == []

    @respx.mock
    async def test_entry_missing_id_skipped(self) -> None:
        xml = _feed(["<entry><title>Some Paper Title</title></entry>"])
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=xml)
        )
        articles = await ArXivScraper().scrape(make_topic())
        assert articles == []

    @respx.mock
    async def test_valid_entry_next_to_invalid_partial_collected(self) -> None:
        xml = _feed([
            "<entry><title>No ID here</title></entry>",
            _entry(title="Valid Paper About Transformers"),
        ])
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=xml)
        )
        articles = await ArXivScraper().scrape(make_topic())
        assert len(articles) == 1
        assert articles[0].title == "Valid Paper About Transformers"


# ── URL normalisation ─────────────────────────────────────────────────────────

class TestURLNormalisation:
    @respx.mock
    async def test_http_id_normalised_to_https(self) -> None:
        xml = _feed([_entry(arxiv_id="http://arxiv.org/abs/1706.03762v5")])
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=xml)
        )
        articles = await ArXivScraper().scrape(make_topic())
        assert articles[0].url.startswith("https://")
        assert not articles[0].url.startswith("http://arxiv")


# ── max_results ───────────────────────────────────────────────────────────────

class TestMaxResults:
    @respx.mock
    async def test_max_results_one_returns_one_article(self) -> None:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=_TWO_RESULTS_XML)
        )
        articles = await ArXivScraper().scrape(make_topic(), max_results=1)
        assert len(articles) == 1

    @respx.mock
    async def test_max_results_larger_than_feed_returns_all(self) -> None:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=_TWO_RESULTS_XML)
        )
        articles = await ArXivScraper().scrape(make_topic(), max_results=10)
        assert len(articles) == 2


# ── Relevance scoring ─────────────────────────────────────────────────────────

class TestRelevanceScoring:
    @respx.mock
    async def test_first_result_has_highest_score(self) -> None:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=_TWO_RESULTS_XML)
        )
        articles = await ArXivScraper().scrape(make_topic(), max_results=5)
        assert articles[0].relevance_score >= articles[1].relevance_score

    @respx.mock
    async def test_relevance_score_bounded_0_to_1(self) -> None:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=_TWO_RESULTS_XML)
        )
        articles = await ArXivScraper().scrape(make_topic(), max_results=5)
        for a in articles:
            assert 0.0 <= a.relevance_score <= 1.0


# ── Published_at parsing ──────────────────────────────────────────────────────

class TestPublishedAt:
    @respx.mock
    async def test_published_at_parsed_to_utc_datetime(self) -> None:
        xml = _feed([_entry(published="2017-06-12T17:00:00Z")])
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=xml)
        )
        articles = await ArXivScraper().scrape(make_topic())
        assert articles[0].published_at == datetime(2017, 6, 12, 17, 0, 0, tzinfo=timezone.utc)

    @respx.mock
    async def test_missing_published_is_none(self) -> None:
        xml = _feed([
            "<entry><id>https://arxiv.org/abs/1706.03762v5</id><title>No Published Date</title></entry>"
        ])
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=xml)
        )
        articles = await ArXivScraper().scrape(make_topic())
        assert articles[0].published_at is None


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:
    @respx.mock
    async def test_http_404_raises_scraper_error(self) -> None:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(404)
        )
        with pytest.raises(ScraperError) as exc_info:
            await ArXivScraper().scrape(make_topic())
        assert exc_info.value.source == ContentSource.ARXIV
        assert "404" in str(exc_info.value)

    @respx.mock
    async def test_http_503_raises_scraper_error(self) -> None:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(503)
        )
        with pytest.raises(ScraperError):
            await ArXivScraper().scrape(make_topic())

    @respx.mock
    async def test_malformed_xml_raises_scraper_error(self) -> None:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text="<not valid xml <<>>")
        )
        with pytest.raises(ScraperError) as exc_info:
            await ArXivScraper().scrape(make_topic())
        assert "parse" in str(exc_info.value).lower() or "xml" in str(exc_info.value).lower()

    @respx.mock
    async def test_network_error_raises_scraper_error(self) -> None:
        respx.get("https://export.arxiv.org/api/query").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        with pytest.raises(ScraperError):
            await ArXivScraper().scrape(make_topic())
