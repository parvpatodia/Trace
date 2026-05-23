"""
HackerNewsScraper — fetches stories from HN via the Algolia search API.

Algolia HN Search API (free, no key required):
  GET https://hn.algolia.com/api/v1/search
    ?query=<topic>
    &tags=story
    &hitsPerPage=<n>
    &numericFilters=created_at_i><30_days_ago_unix_timestamp>

Response format: JSON
  {
    "hits": [
      {
        "objectID":   "12345678",
        "title":      "Attention Is All You Need",
        "url":        "https://arxiv.org/abs/1706.03762",  <- None for Ask/Show HN
        "story_text": null,                                <- text for Ask/Show HN
        "created_at": "2024-01-15T10:00:00.000Z",
        "points":     500
      }
    ]
  }

Signal extraction per hit:
  title        → ScrapedArticle.title
  url          → external link; falls back to HN item URL when None
  story_text   → ScrapedArticle.summary (Ask/Show HN text posts)
  created_at   → ScrapedArticle.published_at
  points       → metadata["points"]

URL resolution:
  - External link posts: use `url` field directly (must be http/https)
  - Ask HN / Show HN (url=None): use https://news.ycombinator.com/item?id=<objectID>
  - Entries with neither url nor objectID: skipped

Relevance scoring: position-based (same as ArXiv).
  Algolia already sorts by a combination of relevance + recency.

WHY ALGOLIA NOT THE OFFICIAL HN FIREBASE API:
  The official HN API (hacker-news.firebaseio.com) provides items by ID only.
  Algolia's search endpoint is the canonical search interface for HN and is
  used by the official hn.algolia.com search UI.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

_log = logging.getLogger(__name__)

from trace.models import ContentSource, ScrapedArticle, Topic
from trace.scraper.base import ArticleScraper, ScraperError

_HN_SEARCH_API = "https://hn.algolia.com/api/v1/search"
_HN_ITEM_URL = "https://news.ycombinator.com/item?id={}"
_TIMEOUT = 10.0
_LOOKBACK_DAYS = 30


class HackerNewsScraper(ArticleScraper):
    """
    Fetches Hacker News stories relevant to a curiosity topic.

    Parameters:
        client: optional httpx.AsyncClient for dependency injection / testing.
        timeout: HTTP request timeout in seconds (default 10s).
    """

    source = ContentSource.HACKER_NEWS

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        timeout: float = _TIMEOUT,
    ) -> None:
        self._client = client
        self._timeout = timeout

    async def scrape(
        self, topic: Topic, max_results: int = 5
    ) -> list[ScrapedArticle]:
        now = datetime.now(timezone.utc)
        # Try 30-day window first; fall back to 90 days for niche topics that
        # don't generate frequent HN coverage.
        for lookback_days in (_LOOKBACK_DAYS, _LOOKBACK_DAYS * 3):
            cutoff_ts = int((now - timedelta(days=lookback_days)).timestamp())
            params = {
                "query": topic.name,
                "tags": "story",
                "hitsPerPage": str(max_results),
                "numericFilters": f"created_at_i>{cutoff_ts}",
            }
            data = await self._fetch(params)
            articles = self._parse(data, topic, max_results)
            if articles:
                return articles
            _log.debug(
                "HN: 0 hits for '%s' with %d-day window, widening to %d days",
                topic.name, lookback_days, lookback_days * 3,
            )
        return []

    async def _fetch(self, params: dict[str, str]) -> dict[str, Any]:
        try:
            if self._client is not None:
                response = await self._client.get(_HN_SEARCH_API, params=params)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as c:
                    response = await c.get(_HN_SEARCH_API, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise ScraperError(
                self.source,
                f"HTTP {e.response.status_code} from HN Algolia API",
            ) from e
        except httpx.RequestError as e:
            raise ScraperError(
                self.source, f"Network error calling HN Algolia API: {e}"
            ) from e

    def _parse(
        self, data: dict[str, Any], topic: Topic, max_results: int
    ) -> list[ScrapedArticle]:
        hits: list[Any] = data.get("hits", [])
        if not isinstance(hits, list):
            raise ScraperError(
                self.source,
                f"Unexpected 'hits' type in HN response: {type(hits).__name__}",
            )

        articles: list[ScrapedArticle] = []
        for i, hit in enumerate(hits[:max_results]):
            article = self._parse_hit(hit, topic, i, len(hits))
            if article is not None:
                articles.append(article)
        return articles

    def _parse_hit(
        self,
        hit: Any,
        topic: Topic,
        rank: int,
        total: int,
    ) -> ScrapedArticle | None:
        if not isinstance(hit, dict):
            return None

        title: str = (hit.get("title") or "").strip()
        if not title:
            return None

        object_id: str = str(hit.get("objectID") or "").strip()
        raw_url: str | None = hit.get("url")

        # Resolve URL: prefer external link, fall back to HN item page
        if raw_url and (raw_url.startswith("http://") or raw_url.startswith("https://")):
            url = raw_url
        elif object_id:
            url = _HN_ITEM_URL.format(object_id)
        else:
            return None  # no navigable URL

        story_text: str = (hit.get("story_text") or "").strip()
        published_at = _parse_hn_datetime(hit.get("created_at"))
        points: int = hit.get("points") or 0
        relevance_score = max(0.0, 1.0 - rank / max(1, total))

        return ScrapedArticle(
            topic_id=topic.id,
            topic_name=topic.name,
            source=self.source,
            title=title[:500],
            url=url,
            summary=story_text[:3000],
            published_at=published_at,
            relevance_score=relevance_score,
            metadata={"objectID": object_id, "points": points, "rank": rank},
        )


def _parse_hn_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
