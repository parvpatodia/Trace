"""
ApifyScraper — fetches web articles via an Apify actor using the native async client.

WHY ApifyClientAsync INSTEAD OF ApifyClient + asyncio.to_thread:
  ApifyClientAsync._wait_for_finish uses await asyncio.sleep(0.25) between
  status polls (verified from SDK source). This means asyncio.wait_for cancels
  it correctly at the next await point — no zombie OS threads.
  ApifyClient (sync) uses time.sleep, so asyncio.wait_for cannot cancel it.

TWO TIMEOUT LAYERS:
  timeout_secs=45  — Apify platform kills the actor after 45s (server-side).
  wait_secs=45     — SDK stops waiting after 45s and returns what it has.
  asyncio.wait_for(timeout=60) — final asyncio-level safety net.
  The first two are sufficient; the third guards against any SDK edge cases.

WHY max_results CAPPED AT 10:
  Apify charges per actor compute unit. 5-10 results per topic is sufficient
  for a newsletter without proportional quality improvement from more.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from apify_client import ApifyClientAsync

from trace.models import ContentSource, ScrapedArticle, Topic
from trace.scraper.base import ArticleScraper, ScraperError

_log = logging.getLogger(__name__)
_DEFAULT_ACTOR = "tri_angle/bing-search-scraper"


class ApifyScraper(ArticleScraper):
    """
    Fetches web articles about a topic using an Apify actor.

    Parameters:
        api_token:  Apify API token (required).
        actor_id:   Apify actor ID (default: tri_angle/bing-search-scraper).
    """

    source = ContentSource.WEB_SEARCH

    def __init__(
        self,
        api_token: str,
        actor_id: str = _DEFAULT_ACTOR,
    ) -> None:
        if not api_token:
            raise ValueError("ApifyScraper: api_token must not be empty.")
        self._client = ApifyClientAsync(token=api_token)
        self._actor_id = actor_id

    async def scrape(
        self, topic: Topic, max_results: int = 5
    ) -> list[ScrapedArticle]:
        try:
            items = await asyncio.wait_for(
                self._run_actor(topic.name, min(max_results, 10)),
                timeout=60,
            )
        except asyncio.TimeoutError:
            _log.warning("Apify actor timed out for topic %r — skipping", topic.name)
            return []
        except Exception as exc:
            raise ScraperError(
                self.source,
                f"Apify actor '{self._actor_id}' failed: {exc}",
            ) from exc
        return self._parse(items, topic, max_results)

    async def _run_actor(self, query: str, limit: int) -> list[dict[str, Any]]:
        run = await self._client.actor(self._actor_id).call(
            run_input={
                "queries": query,
                "maxResultsPerQuery": limit,
            },
            timeout_secs=45,
            wait_secs=45,
        )
        if not run:
            _log.warning("Apify actor run returned None for query %r", query)
            return []
        dataset_id: str | None = run.get("defaultDatasetId")
        if not dataset_id:
            _log.warning("Apify run missing defaultDatasetId for query %r", query)
            return []
        items: list[dict[str, Any]] = []
        async for item in self._client.dataset(dataset_id).iterate_items():
            items.append(item)
        return items

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
