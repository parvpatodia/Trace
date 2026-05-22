"""
ApifyScraper — fetches web articles via the Apify Bing Search Scraper actor.

Uses the `apify/bing-search-scraper` actor on Apify's platform to search Bing
for articles about a curiosity topic.  Results are converted into
ScrapedArticle objects and merged with ArXiv, HN, and Reddit results.

Actor: apify/bing-search-scraper
  Input:
    queries:           str  — the search query (one per line for multi-query)
    maxResultsPerQuery: int — maximum number of results to return
  Output items (JSONL from the default dataset):
    title:       str  — page title
    url:         str  — canonical page URL
    description: str  — search-result snippet (used as summary)
    date:        str  — publication date ISO-8601 (may be absent)

WHY APIFY OVER DIRECT BING SEARCH API:
  Bing Search API v7 requires an Azure Cognitive Services subscription.
  Apify's actor wraps Bing's web UI, giving equivalent results without
  additional cloud accounts — only an Apify token is needed.

WHY asyncio.to_thread:
  The Apify Python client is synchronous.  Wrapping actor runs in
  asyncio.to_thread keeps the pipeline's async event loop unblocked.

WHY max_results CAPPED AT 10:
  Apify charges per actor compute unit.  For a newsletter, 5-10 high-
  quality web articles per topic is sufficient.  Going higher raises cost
  without proportional quality improvement.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from apify_client import ApifyClient

from trace.models import ContentSource, ScrapedArticle, Topic
from trace.scraper.base import ArticleScraper, ScraperError

_log = logging.getLogger(__name__)
_DEFAULT_ACTOR = "tri_angle/bing-search-scraper"


class ApifyScraper(ArticleScraper):
    """
    Fetches web articles about a topic using Apify's Bing Search Scraper.

    Parameters:
        api_token:  Apify API token (required).
        actor_id:   Apify actor to use (default: apify/bing-search-scraper).
    """

    source = ContentSource.WEB_SEARCH

    def __init__(
        self,
        api_token: str,
        actor_id: str = _DEFAULT_ACTOR,
    ) -> None:
        if not api_token:
            raise ValueError("ApifyScraper: api_token must not be empty.")
        self._client = ApifyClient(token=api_token)
        self._actor_id = actor_id

    async def scrape(
        self, topic: Topic, max_results: int = 5
    ) -> list[ScrapedArticle]:
        try:
            items = await asyncio.to_thread(
                self._run_actor, topic.name, min(max_results, 10)
            )
        except Exception as exc:
            raise ScraperError(
                self.source,
                f"Apify actor '{self._actor_id}' failed: {exc}",
            ) from exc
        return self._parse(items, topic, max_results)

    # ── Synchronous helpers (called via asyncio.to_thread) ────────────────────

    def _run_actor(self, query: str, limit: int) -> list[dict[str, Any]]:
        run = self._client.actor(self._actor_id).call(
            run_input={
                "queries": query,
                "maxResultsPerQuery": limit,
            }
        )
        if not run:
            _log.warning("Apify actor run returned None for query %r", query)
            return []
        dataset_id: str | None = run.get("defaultDatasetId")
        if not dataset_id:
            _log.warning("Apify run missing defaultDatasetId for query %r", query)
            return []
        return list(self._client.dataset(dataset_id).iterate_items())

    def _parse(
        self,
        items: list[dict[str, Any]],
        topic: Topic,
        max_results: int,
    ) -> list[ScrapedArticle]:
        articles: list[ScrapedArticle] = []
        for rank, item in enumerate(items[:max_results]):
            article = self._parse_item(item, topic, rank, len(items))
            if article is not None:
                articles.append(article)
        _log.debug("Apify: %d article(s) for topic '%s'", len(articles), topic.name)
        return articles

    def _parse_item(
        self,
        item: dict[str, Any],
        topic: Topic,
        rank: int,
        total: int,
    ) -> ScrapedArticle | None:
        if not isinstance(item, dict):
            return None

        title: str = (item.get("title") or "").strip()
        url: str = (item.get("url") or "").strip()

        if not title or not url:
            return None
        if not (url.startswith("http://") or url.startswith("https://")):
            return None

        description: str = (item.get("description") or "").strip()
        published_at = _parse_date(item.get("date"))
        relevance_score = max(0.0, 1.0 - rank / max(1, total))

        return ScrapedArticle(
            topic_id=topic.id,
            topic_name=topic.name,
            source=self.source,
            title=title[:500],
            url=url,
            summary=description[:3000],
            published_at=published_at,
            relevance_score=relevance_score,
            metadata={"rank": rank},
        )


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
