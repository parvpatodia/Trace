"""
RedditSearchScraper — searches Reddit for posts relevant to a topic via PRAW.

Distinct from RedditCollector (trace/signals/reddit.py):
  - RedditCollector reads the user's *saved* items (inbound curiosity signals)
  - RedditSearchScraper searches Reddit for *new content* about a topic
    (outbound content retrieval for the newsletter)

Search strategy:
  subreddit("all").search(topic.name, sort="relevance", time_filter="year",
                          limit=max_results)

  Searching "all" gives the widest coverage. "year" time_filter avoids
  very stale content while still finding established discussions.

Signal extraction per submission:
  title      → ScrapedArticle.title
  permalink  → https://reddit.com{post.permalink}
  selftext   → ScrapedArticle.summary (text posts; truncated to 3000 chars)
  created_utc → ScrapedArticle.published_at

Relevance scoring: position-based (PRAW returns relevance-sorted results).

WHY NOT URL FOR LINK POSTS:
  Reddit link posts point to external content we don't control.
  For the newsletter, the *Reddit discussion thread* is the content
  (community reaction to the topic). The ScrapedArticle.url always points
  to the Reddit thread, not the linked article.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import praw
import praw.exceptions
import praw.models

from trace.models import ContentSource, ScrapedArticle, Topic
from trace.scraper.base import ArticleScraper, ScraperError

_REDDIT_BASE = "https://www.reddit.com"
_MIN_TITLE_LENGTH = 3


class RedditSearchScraper(ArticleScraper):
    """
    Searches Reddit for posts relevant to a curiosity topic.

    Parameters:
        reddit_client: authenticated praw.Reddit instance.
        time_filter:   PRAW time filter ("year", "month", "week", "all").
                       Default "year" — avoids stale content.
        subreddit:     subreddit to search (default "all").
    """

    source = ContentSource.REDDIT

    def __init__(
        self,
        reddit_client: praw.Reddit,
        time_filter: str = "year",
        subreddit: str = "all",
    ) -> None:
        if reddit_client is None:
            raise ValueError(
                "RedditSearchScraper: reddit_client must not be None."
            )
        self._client = reddit_client
        self._time_filter = time_filter
        self._subreddit = subreddit

    async def scrape(
        self, topic: Topic, max_results: int = 5
    ) -> list[ScrapedArticle]:
        try:
            posts = await asyncio.to_thread(
                self._search, topic.name, max_results
            )
        except praw.exceptions.PRAWException as e:
            raise ScraperError(
                self.source, f"Reddit search error: {e}"
            ) from e

        return self._parse(posts, topic, max_results)

    def _search(self, query: str, limit: int) -> list[praw.models.Submission]:
        sub = self._client.subreddit(self._subreddit)
        return list(
            sub.search(
                query,
                sort="relevance",
                time_filter=self._time_filter,
                limit=limit,
            )
        )

    def _parse(
        self,
        posts: list[praw.models.Submission],
        topic: Topic,
        max_results: int,
    ) -> list[ScrapedArticle]:
        articles: list[ScrapedArticle] = []
        for i, post in enumerate(posts[:max_results]):
            article = self._parse_post(post, topic, i, len(posts))
            if article is not None:
                articles.append(article)
        return articles

    def _parse_post(
        self,
        post: praw.models.Submission,
        topic: Topic,
        rank: int,
        total: int,
    ) -> ScrapedArticle | None:
        title: str = (post.title or "").strip()
        if len(title) < _MIN_TITLE_LENGTH:
            return None

        permalink: str = post.permalink or ""
        url = f"{_REDDIT_BASE}{permalink}"
        if not (url.startswith("http://") or url.startswith("https://")):
            return None

        summary: str = (post.selftext or "").strip()[:3000]
        published_at = datetime.fromtimestamp(
            post.created_utc, tz=timezone.utc
        )
        relevance_score = max(0.0, 1.0 - rank / max(1, total))

        return ScrapedArticle(
            topic_id=topic.id,
            topic_name=topic.name,
            source=self.source,
            title=title[:500],
            url=url,
            summary=summary,
            published_at=published_at,
            relevance_score=relevance_score,
            metadata={
                "subreddit": post.subreddit.display_name,
                "score": post.score,
                "num_comments": post.num_comments,
                "rank": rank,
            },
        )
