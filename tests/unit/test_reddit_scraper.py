"""
Unit tests for trace/scraper/reddit.py — RedditSearchScraper.

Uses MagicMock(spec=praw.models.Submission) — same pattern as test_reddit.py.

Test categories:
  1. Constructor: None client raises
  2. Source attribute
  3. Happy path: posts → ScrapedArticle fields
  4. URL construction: https://www.reddit.com + permalink
  5. Text posts: selftext → summary
  6. max_results respected
  7. Relevance scoring: position-based
  8. Short title filtered
  9. published_at from created_utc
  10. Metadata: subreddit, score, num_comments, rank
  11. PRAW exceptions → ScraperError
  12. Empty search results → empty list
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import praw.models
import pytest

from trace.models import ContentSource, Topic
from trace.scraper.base import ScraperError
from trace.scraper.reddit import RedditSearchScraper


# ── Helpers ───────────────────────────────────────────────────────────────────

_TS = 1_704_067_200.0  # 2024-01-01 00:00:00 UTC


def make_topic(name: str = "transformer architecture") -> Topic:
    return Topic(name=name, frequency=2, recency_score=0.7)


def make_post(
    title: str = "How does self-attention work in vision transformers?",
    permalink: str = "/r/MachineLearning/comments/abc123/how_does/",
    selftext: str = "",
    subreddit_name: str = "MachineLearning",
    score: int = 234,
    num_comments: int = 47,
    created_utc: float = _TS,
) -> MagicMock:
    post = MagicMock(spec=praw.models.Submission)
    post.title = title
    post.permalink = permalink
    post.selftext = selftext
    post.score = score
    post.num_comments = num_comments
    post.created_utc = created_utc
    post.subreddit = MagicMock(display_name=subreddit_name)
    return post


def make_reddit_client(search_results: list) -> MagicMock:
    client = MagicMock()
    sub_mock = MagicMock()
    sub_mock.search.return_value = iter(search_results)
    client.subreddit.return_value = sub_mock
    return client


# ── Constructor ───────────────────────────────────────────────────────────────

class TestConstructor:
    def test_none_client_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="reddit_client"):
            RedditSearchScraper(reddit_client=None)  # type: ignore[arg-type]

    def test_valid_client_accepted(self) -> None:
        scraper = RedditSearchScraper(make_reddit_client([]))
        assert scraper is not None

    def test_source_is_reddit(self) -> None:
        assert RedditSearchScraper(make_reddit_client([])).source == ContentSource.REDDIT


# ── Happy path ────────────────────────────────────────────────────────────────

class TestHappyPath:
    async def test_one_post_produces_one_article(self) -> None:
        client = make_reddit_client([make_post()])
        articles = await RedditSearchScraper(client).scrape(make_topic())
        assert len(articles) == 1

    async def test_article_title_matches_post_title(self) -> None:
        client = make_reddit_client([make_post(title="ViT beats CNNs on ImageNet")])
        articles = await RedditSearchScraper(client).scrape(make_topic())
        assert articles[0].title == "ViT beats CNNs on ImageNet"

    async def test_article_source_is_reddit(self) -> None:
        client = make_reddit_client([make_post()])
        articles = await RedditSearchScraper(client).scrape(make_topic())
        assert articles[0].source == ContentSource.REDDIT

    async def test_article_topic_id_matches(self) -> None:
        topic = make_topic()
        client = make_reddit_client([make_post()])
        articles = await RedditSearchScraper(client).scrape(topic)
        assert articles[0].topic_id == topic.id

    async def test_article_topic_name_matches(self) -> None:
        topic = make_topic("vision transformers")
        client = make_reddit_client([make_post()])
        articles = await RedditSearchScraper(client).scrape(topic)
        assert articles[0].topic_name == "vision transformers"


# ── URL construction ──────────────────────────────────────────────────────────

class TestURLConstruction:
    async def test_url_is_full_reddit_url(self) -> None:
        client = make_reddit_client([
            make_post(permalink="/r/MachineLearning/comments/abc123/")
        ])
        articles = await RedditSearchScraper(client).scrape(make_topic())
        assert articles[0].url == "https://www.reddit.com/r/MachineLearning/comments/abc123/"

    async def test_url_starts_with_https(self) -> None:
        client = make_reddit_client([make_post()])
        articles = await RedditSearchScraper(client).scrape(make_topic())
        assert articles[0].url.startswith("https://")


# ── Text post / summary ───────────────────────────────────────────────────────

class TestSummary:
    async def test_selftext_used_as_summary(self) -> None:
        client = make_reddit_client([
            make_post(selftext="I've been studying attention mechanisms and wondering...")
        ])
        articles = await RedditSearchScraper(client).scrape(make_topic())
        assert "attention mechanisms" in articles[0].summary

    async def test_empty_selftext_gives_empty_summary(self) -> None:
        client = make_reddit_client([make_post(selftext="")])
        articles = await RedditSearchScraper(client).scrape(make_topic())
        assert articles[0].summary == ""

    async def test_selftext_truncated_to_3000_chars(self) -> None:
        client = make_reddit_client([make_post(selftext="x" * 5000)])
        articles = await RedditSearchScraper(client).scrape(make_topic())
        assert len(articles[0].summary) == 3000


# ── max_results ───────────────────────────────────────────────────────────────

class TestMaxResults:
    async def test_max_results_one_returns_one_article(self) -> None:
        client = make_reddit_client([make_post(), make_post(title="Second Post About ML")])
        articles = await RedditSearchScraper(client).scrape(make_topic(), max_results=1)
        assert len(articles) == 1

    async def test_empty_search_returns_empty_list(self) -> None:
        client = make_reddit_client([])
        articles = await RedditSearchScraper(client).scrape(make_topic())
        assert articles == []


# ── Short title filtering ─────────────────────────────────────────────────────

class TestShortTitleFiltering:
    async def test_very_short_title_skipped(self) -> None:
        client = make_reddit_client([make_post(title="AI")])
        articles = await RedditSearchScraper(client).scrape(make_topic())
        assert articles == []

    async def test_title_exactly_3_chars_accepted(self) -> None:
        client = make_reddit_client([make_post(title="LLM")])
        articles = await RedditSearchScraper(client).scrape(make_topic())
        assert len(articles) == 1


# ── Relevance scoring ─────────────────────────────────────────────────────────

class TestRelevanceScoring:
    async def test_first_post_has_highest_score(self) -> None:
        client = make_reddit_client([
            make_post(title="First result about transformers"),
            make_post(title="Second result about attention mechanisms"),
        ])
        articles = await RedditSearchScraper(client).scrape(make_topic(), max_results=5)
        assert articles[0].relevance_score >= articles[1].relevance_score

    async def test_relevance_score_bounded_0_to_1(self) -> None:
        client = make_reddit_client([make_post(), make_post(title="Another Post About Neural Nets")])
        articles = await RedditSearchScraper(client).scrape(make_topic(), max_results=5)
        for a in articles:
            assert 0.0 <= a.relevance_score <= 1.0


# ── published_at ──────────────────────────────────────────────────────────────

class TestPublishedAt:
    async def test_created_utc_to_utc_datetime(self) -> None:
        client = make_reddit_client([make_post(created_utc=_TS)])
        articles = await RedditSearchScraper(client).scrape(make_topic())
        assert articles[0].published_at == datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    async def test_published_at_is_utc_aware(self) -> None:
        client = make_reddit_client([make_post()])
        articles = await RedditSearchScraper(client).scrape(make_topic())
        assert articles[0].published_at.tzinfo == timezone.utc


# ── Metadata ──────────────────────────────────────────────────────────────────

class TestMetadata:
    async def test_metadata_has_subreddit(self) -> None:
        client = make_reddit_client([make_post(subreddit_name="learnmachinelearning")])
        articles = await RedditSearchScraper(client).scrape(make_topic())
        assert articles[0].metadata["subreddit"] == "learnmachinelearning"

    async def test_metadata_has_score(self) -> None:
        client = make_reddit_client([make_post(score=999)])
        articles = await RedditSearchScraper(client).scrape(make_topic())
        assert articles[0].metadata["score"] == 999

    async def test_metadata_has_num_comments(self) -> None:
        client = make_reddit_client([make_post(num_comments=123)])
        articles = await RedditSearchScraper(client).scrape(make_topic())
        assert articles[0].metadata["num_comments"] == 123

    async def test_metadata_has_rank(self) -> None:
        client = make_reddit_client([make_post()])
        articles = await RedditSearchScraper(client).scrape(make_topic())
        assert articles[0].metadata["rank"] == 0


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:
    async def test_praw_exception_raises_scraper_error(self) -> None:
        import praw.exceptions
        client = MagicMock()
        client.subreddit.return_value.search.side_effect = praw.exceptions.PRAWException(
            "rate limited"
        )
        with pytest.raises(ScraperError) as exc_info:
            await RedditSearchScraper(client).scrape(make_topic())
        assert exc_info.value.source == ContentSource.REDDIT
        assert "rate limited" in str(exc_info.value)

    async def test_search_called_with_topic_name(self) -> None:
        client = make_reddit_client([])
        topic = make_topic("vision transformers")
        await RedditSearchScraper(client).scrape(topic, max_results=3)
        sub_mock = client.subreddit.return_value
        call_args = sub_mock.search.call_args
        assert call_args[0][0] == "vision transformers"
