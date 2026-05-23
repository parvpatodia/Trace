"""
ApifyScraper — fetches web articles via an Apify actor using the native async client.

DEFAULT ACTOR: apify/google-search-scraper
  This is the official Apify-maintained Google Search scraper. It is used instead
  of the community `tri_angle/bing-search-scraper` because:
  1. Bing aggressively blocks datacenter IP ranges used by shared Apify actor pools.
     The Bing scraper returns 0 results after the 45s timeout on every run.
  2. `apify/google-search-scraper` is maintained by Apify engineering with built-in
     proxy rotation and anti-detection, making it significantly more reliable.
  3. Google Search results are higher quality and more semantically diverse for
     newsletter content than Bing results.

WHY ApifyClientAsync INSTEAD OF ApifyClient + asyncio.to_thread:
  ApifyClientAsync._wait_for_finish uses await asyncio.sleep(0.25) between
  status polls (verified from SDK source). This means asyncio.wait_for cancels
  it correctly at the next await point — no zombie OS threads.
  ApifyClient (sync) uses time.sleep, so asyncio.wait_for cannot cancel it.

TWO TIMEOUT LAYERS:
  timeout_secs=45  — Apify platform kills the actor after 45s (server-side).
  wait_secs=45     — SDK stops waiting after 45s and returns what it has.
  asyncio.wait_for(timeout=60) — final asyncio-level safety net.

WHY max_results CAPPED AT 10:
  Apify charges per actor compute unit. 5-10 results per topic is sufficient
  for a newsletter without proportional quality improvement from more.

WHY _flatten_items():
  apify/google-search-scraper returns one dataset item per search query, with
  organic results nested inside an "organicResults" array. The legacy Bing
  scraper returned one item per result (flat). _flatten_items() normalises
  both formats so _parse_item receives individual result dicts in both cases.
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
_DEFAULT_ACTOR = "apify/google-search-scraper"


class ApifyScraper(ArticleScraper):
    """
    Fetches web articles about a topic using an Apify actor.

    Parameters:
        api_token:  Apify API token (required).
        actor_id:   Apify actor ID (default: apify/google-search-scraper).
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
                # apify/google-search-scraper uses resultsPerPage + maxPagesPerQuery.
                # Legacy tri_angle/bing-search-scraper uses maxResultsPerQuery.
                # Both keys are passed; unknown keys are silently ignored by actors.
                "resultsPerPage": limit,
                "maxResultsPerQuery": limit,
                "maxPagesPerQuery": 1,
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
        raw_items: list[dict[str, Any]] = []
        async for item in self._client.dataset(dataset_id).iterate_items():
            raw_items.append(item)
        return self._flatten_items(raw_items)

    def _flatten_items(self, raw_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalise actor output: handle both flat items and nested result arrays.

        apify/google-search-scraper returns one item per query with results nested
        inside "organicResults". Flat-item actors (e.g. legacy Bing scraper) return
        one item per result with top-level "title"/"url" keys.
        """
        flat: list[dict[str, Any]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            # Nested format: apify/google-search-scraper
            if "organicResults" in item and isinstance(item["organicResults"], list):
                flat.extend(r for r in item["organicResults"] if isinstance(r, dict))
            # Nested format: some actors use "items" key
            elif "items" in item and isinstance(item["items"], list):
                flat.extend(r for r in item["items"] if isinstance(r, dict))
            # Flat format: item is already a result (legacy Bing scraper)
            elif "title" in item or "url" in item:
                flat.append(item)
        return flat

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
        url: str = (item.get("url") or item.get("link") or "").strip()

        if not title or not url:
            return None
        if not (url.startswith("http://") or url.startswith("https://")):
            return None

        # Different actors use different field names for the summary/snippet
        description: str = (
            item.get("description")
            or item.get("snippet")
            or item.get("text")
            or item.get("summary")
            or ""
        ).strip()

        # Different actors use different field names for publication date
        raw_date = (
            item.get("date")
            or item.get("pubDate")
            or item.get("publishedAt")
            or item.get("published_at")
        )
        published_at = _parse_date(raw_date)
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
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
