"""
Unit tests for trace/scraper/hackernews.py — HackerNewsScraper.

All HTTP calls intercepted with respx.

Test categories:
  1. Source attribute
  2. Happy path: valid JSON → correct ScrapedArticle fields
  3. Empty hits: empty list returned
  4. URL resolution: external URL preferred, HN item URL as fallback
  5. Ask HN (story_text): summary populated from story_text
  6. max_results respected
  7. Relevance scoring: position-based, bounded [0, 1]
  8. published_at: created_at parsed to UTC datetime
  9. HTTP errors → ScraperError
  10. Malformed 'hits' type → ScraperError
  11. Non-dict hit skipped gracefully
  12. Missing title skipped
  13. metadata: objectID, points, rank present
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest
import respx

from trace.models import ContentSource, Topic
from trace.scraper.base import ScraperError
from trace.scraper.hackernews import HackerNewsScraper


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_topic(name: str = "transformer architecture") -> Topic:
    return Topic(name=name, frequency=2, recency_score=0.7)


def _hn_response(hits: list[dict]) -> httpx.Response:
    return httpx.Response(200, json={"hits": hits})


def _hit(
    object_id: str = "12345678",
    title: str = "Attention Is All You Need",
    url: str | None = "https://arxiv.org/abs/1706.03762",
    story_text: str | None = None,
    created_at: str = "2024-01-15T10:00:00.000Z",
    points: int = 500,
) -> dict:
    return {
        "objectID": object_id,
        "title": title,
        "url": url,
        "story_text": story_text,
        "created_at": created_at,
        "points": points,
        "_tags": ["story"],
    }


_SINGLE_HIT = _hn_response([_hit()])
_TWO_HITS = _hn_response([
    _hit(object_id="111", title="Hit One", url="https://example.com/one"),
    _hit(object_id="222", title="Hit Two", url="https://example.com/two"),
])
_EMPTY_HITS = _hn_response([])


# ── Source attribute ──────────────────────────────────────────────────────────

class TestSourceAttribute:
    def test_source_is_hacker_news(self) -> None:
        assert HackerNewsScraper().source == ContentSource.HACKER_NEWS


# ── Happy path ────────────────────────────────────────────────────────────────

class TestHappyPath:
    @respx.mock
    async def test_one_hit_produces_one_article(self) -> None:
        respx.get("https://hn.algolia.com/api/v1/search").mock(return_value=_SINGLE_HIT)
        articles = await HackerNewsScraper().scrape(make_topic())
        assert len(articles) == 1

    @respx.mock
    async def test_article_title_matches_hit_title(self) -> None:
        respx.get("https://hn.algolia.com/api/v1/search").mock(return_value=_SINGLE_HIT)
        articles = await HackerNewsScraper().scrape(make_topic())
        assert articles[0].title == "Attention Is All You Need"

    @respx.mock
    async def test_article_url_is_external_link(self) -> None:
        respx.get("https://hn.algolia.com/api/v1/search").mock(return_value=_SINGLE_HIT)
        articles = await HackerNewsScraper().scrape(make_topic())
        assert articles[0].url == "https://arxiv.org/abs/1706.03762"

    @respx.mock
    async def test_article_source_is_hacker_news(self) -> None:
        respx.get("https://hn.algolia.com/api/v1/search").mock(return_value=_SINGLE_HIT)
        articles = await HackerNewsScraper().scrape(make_topic())
        assert articles[0].source == ContentSource.HACKER_NEWS

    @respx.mock
    async def test_article_topic_id_matches(self) -> None:
        respx.get("https://hn.algolia.com/api/v1/search").mock(return_value=_SINGLE_HIT)
        topic = make_topic()
        articles = await HackerNewsScraper().scrape(topic)
        assert articles[0].topic_id == topic.id

    @respx.mock
    async def test_article_topic_name_matches(self) -> None:
        respx.get("https://hn.algolia.com/api/v1/search").mock(return_value=_SINGLE_HIT)
        topic = make_topic("vision transformers")
        articles = await HackerNewsScraper().scrape(topic)
        assert articles[0].topic_name == "vision transformers"


# ── Empty hits ────────────────────────────────────────────────────────────────

class TestEmptyHits:
    @respx.mock
    async def test_empty_hits_returns_empty_list(self) -> None:
        respx.get("https://hn.algolia.com/api/v1/search").mock(return_value=_EMPTY_HITS)
        articles = await HackerNewsScraper().scrape(make_topic())
        assert articles == []


# ── URL resolution ────────────────────────────────────────────────────────────

class TestURLResolution:
    @respx.mock
    async def test_external_url_used_when_present(self) -> None:
        hit = _hit(url="https://example.com/article", object_id="99")
        respx.get("https://hn.algolia.com/api/v1/search").mock(
            return_value=_hn_response([hit])
        )
        articles = await HackerNewsScraper().scrape(make_topic())
        assert articles[0].url == "https://example.com/article"

    @respx.mock
    async def test_hn_item_url_used_when_external_url_is_none(self) -> None:
        hit = _hit(url=None, object_id="42000000")
        respx.get("https://hn.algolia.com/api/v1/search").mock(
            return_value=_hn_response([hit])
        )
        articles = await HackerNewsScraper().scrape(make_topic())
        assert articles[0].url == "https://news.ycombinator.com/item?id=42000000"

    @respx.mock
    async def test_hit_with_no_url_and_no_objectid_skipped(self) -> None:
        hit = {"title": "Some Title", "url": None, "objectID": None,
               "story_text": None, "created_at": "2024-01-01T00:00:00.000Z", "points": 0}
        respx.get("https://hn.algolia.com/api/v1/search").mock(
            return_value=_hn_response([hit])
        )
        articles = await HackerNewsScraper().scrape(make_topic())
        assert articles == []


# ── Ask HN / story_text ───────────────────────────────────────────────────────

class TestStoryText:
    @respx.mock
    async def test_story_text_used_as_summary(self) -> None:
        hit = _hit(
            url=None,
            object_id="12345",
            story_text="Ask HN: What is the best way to learn transformers?",
        )
        respx.get("https://hn.algolia.com/api/v1/search").mock(
            return_value=_hn_response([hit])
        )
        articles = await HackerNewsScraper().scrape(make_topic())
        assert "Ask HN" in articles[0].summary

    @respx.mock
    async def test_null_story_text_gives_empty_summary(self) -> None:
        respx.get("https://hn.algolia.com/api/v1/search").mock(return_value=_SINGLE_HIT)
        articles = await HackerNewsScraper().scrape(make_topic())
        assert articles[0].summary == ""


# ── max_results ───────────────────────────────────────────────────────────────

class TestMaxResults:
    @respx.mock
    async def test_max_results_one_returns_one(self) -> None:
        respx.get("https://hn.algolia.com/api/v1/search").mock(return_value=_TWO_HITS)
        articles = await HackerNewsScraper().scrape(make_topic(), max_results=1)
        assert len(articles) == 1

    @respx.mock
    async def test_max_results_larger_than_hits_returns_all(self) -> None:
        respx.get("https://hn.algolia.com/api/v1/search").mock(return_value=_TWO_HITS)
        articles = await HackerNewsScraper().scrape(make_topic(), max_results=10)
        assert len(articles) == 2


# ── Relevance scoring ─────────────────────────────────────────────────────────

class TestRelevanceScoring:
    @respx.mock
    async def test_first_result_has_higher_score(self) -> None:
        respx.get("https://hn.algolia.com/api/v1/search").mock(return_value=_TWO_HITS)
        articles = await HackerNewsScraper().scrape(make_topic(), max_results=5)
        assert articles[0].relevance_score >= articles[1].relevance_score

    @respx.mock
    async def test_relevance_score_bounded_0_to_1(self) -> None:
        respx.get("https://hn.algolia.com/api/v1/search").mock(return_value=_TWO_HITS)
        articles = await HackerNewsScraper().scrape(make_topic(), max_results=5)
        for a in articles:
            assert 0.0 <= a.relevance_score <= 1.0


# ── published_at ──────────────────────────────────────────────────────────────

class TestPublishedAt:
    @respx.mock
    async def test_created_at_parsed_to_utc_datetime(self) -> None:
        hit = _hit(created_at="2024-01-15T10:00:00.000Z")
        respx.get("https://hn.algolia.com/api/v1/search").mock(
            return_value=_hn_response([hit])
        )
        articles = await HackerNewsScraper().scrape(make_topic())
        assert articles[0].published_at == datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

    @respx.mock
    async def test_missing_created_at_gives_none(self) -> None:
        hit = _hit(created_at=None)
        respx.get("https://hn.algolia.com/api/v1/search").mock(
            return_value=_hn_response([hit])
        )
        articles = await HackerNewsScraper().scrape(make_topic())
        assert articles[0].published_at is None


# ── Metadata ──────────────────────────────────────────────────────────────────

class TestMetadata:
    @respx.mock
    async def test_metadata_has_object_id(self) -> None:
        respx.get("https://hn.algolia.com/api/v1/search").mock(return_value=_SINGLE_HIT)
        articles = await HackerNewsScraper().scrape(make_topic())
        assert articles[0].metadata["objectID"] == "12345678"

    @respx.mock
    async def test_metadata_has_points(self) -> None:
        respx.get("https://hn.algolia.com/api/v1/search").mock(return_value=_SINGLE_HIT)
        articles = await HackerNewsScraper().scrape(make_topic())
        assert articles[0].metadata["points"] == 500

    @respx.mock
    async def test_metadata_has_rank(self) -> None:
        respx.get("https://hn.algolia.com/api/v1/search").mock(return_value=_SINGLE_HIT)
        articles = await HackerNewsScraper().scrape(make_topic())
        assert articles[0].metadata["rank"] == 0


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:
    @respx.mock
    async def test_http_500_raises_scraper_error(self) -> None:
        respx.get("https://hn.algolia.com/api/v1/search").mock(
            return_value=httpx.Response(500)
        )
        with pytest.raises(ScraperError) as exc_info:
            await HackerNewsScraper().scrape(make_topic())
        assert exc_info.value.source == ContentSource.HACKER_NEWS

    @respx.mock
    async def test_network_error_raises_scraper_error(self) -> None:
        respx.get("https://hn.algolia.com/api/v1/search").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        with pytest.raises(ScraperError):
            await HackerNewsScraper().scrape(make_topic())

    @respx.mock
    async def test_hits_not_a_list_raises_scraper_error(self) -> None:
        respx.get("https://hn.algolia.com/api/v1/search").mock(
            return_value=httpx.Response(200, json={"hits": "not a list"})
        )
        with pytest.raises(ScraperError):
            await HackerNewsScraper().scrape(make_topic())

    @respx.mock
    async def test_non_dict_hit_skipped_gracefully(self) -> None:
        respx.get("https://hn.algolia.com/api/v1/search").mock(
            return_value=_hn_response(["not a dict", _hit()])
        )
        articles = await HackerNewsScraper().scrape(make_topic())
        assert len(articles) == 1

    @respx.mock
    async def test_hit_with_empty_title_skipped(self) -> None:
        respx.get("https://hn.algolia.com/api/v1/search").mock(
            return_value=_hn_response([_hit(title="")])
        )
        articles = await HackerNewsScraper().scrape(make_topic())
        assert articles == []
