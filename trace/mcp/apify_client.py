"""
ApifyMCPScraper — ArticleScraper implementation using Apify MCP via Scalekit.

TWO EXECUTION PATHS:

  Path A — Scalekit Connect (double-points sponsor track):
    Apify token lives in Scalekit's Token Vault, not in our env vars.
    Calls client.connect.execute_tool("apifymcp_call_actor", ...) which routes
    through Scalekit's Agent Connect layer. Earns double points in hackathon.
    Falls back to Path B on any failure.

  Path B — Direct Apify MCP (Streamable HTTP):
    Calls mcp.apify.com/sse directly with the Apify token. Zero auth overhead.
    Used when Scalekit Connect is not configured, or as Path A fallback.

DYNAMIC ACTOR SELECTION:
  Each topic is mapped to the most appropriate Apify Actor via a hint table.
  "instagram trends" → apify/instagram-scraper
  "AI research paper" → apify/rag-web-browser
  "github project" → apify/website-content-crawler
  "coffee shops" → compass/crawler-google-places
  Default → apify/google-search-scraper (general-purpose)
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from trace.models import ContentSource, ScrapedArticle, Topic
from trace.scraper.base import ArticleScraper

_log = logging.getLogger(__name__)

# Hint table: keyword → Apify Actor ID.
_ACTOR_HINTS: list[tuple[list[str], str]] = [
    (["instagram", "tiktok", "twitter", "x.com", "social media"], "apify/instagram-scraper"),
    (["arxiv", "paper", "research", "preprint", "academic"], "apify/rag-web-browser"),
    (["github", "repo", "open source", "codebase"], "apify/website-content-crawler"),
    (["youtube", "video", "channel", "playlist"], "apify/youtube-scraper"),
    (["reddit", "subreddit", "r/"], "apify/reddit-scraper"),
    (["linkedin", "job", "career", "hiring"], "apify/linkedin-profile-scraper"),
    (["amazon", "product", "review", "shopping"], "apify/amazon-product-scraper"),
    (["coffee", "restaurant", "place", "map", "nearby"], "compass/crawler-google-places"),
    (["news", "headline", "article", "blog"], "apify/web-scraper"),
    (["hacker news", "hn ", "show hn", "ask hn"], "apify/hacker-news-scraper"),
    (["patent", "uspto", "intellectual property"], "apify/rag-web-browser"),
    (["financial", "stock", "earnings", "sec filing"], "apify/web-scraper"),
    (["podcast", "transcript", "episode"], "apify/rag-web-browser"),
]
_DEFAULT_ACTOR = "apify/google-search-scraper"


def _pick_actor_for_topic(topic_name: str) -> str:
    lower = topic_name.lower()
    for keywords, actor_id in _ACTOR_HINTS:
        if any(kw in lower for kw in keywords):
            return actor_id
    return _DEFAULT_ACTOR


class ApifyMCPScraper(ArticleScraper):
    """Scraper that calls Apify Actors via MCP, with optional Scalekit routing."""

    source = ContentSource.WEB_SEARCH

    def __init__(
        self,
        api_token: str,
        scalekit_connection_name: str | None = None,
        scalekit_identifier: str | None = None,
    ) -> None:
        if not api_token:
            raise ValueError("ApifyMCPScraper: api_token must not be empty.")
        self._api_token = api_token
        self._scalekit_connection_name = scalekit_connection_name
        self._scalekit_identifier = scalekit_identifier

    @property
    def _via_scalekit(self) -> bool:
        return bool(self._scalekit_connection_name and self._scalekit_identifier)

    async def scrape(self, topic: Topic, max_results: int = 5) -> list[ScrapedArticle]:
        actor_id = _pick_actor_for_topic(topic.name)
        _log.info(
            "Apify MCP: topic=%r → actor=%s (via_scalekit=%s)",
            topic.name, actor_id, self._via_scalekit,
        )
        try:
            return await asyncio.wait_for(
                self._scrape_inner(topic, actor_id, max_results),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            _log.warning("Apify MCP timed out for topic %r", topic.name)
            return []
        except Exception as exc:
            _log.warning("Apify MCP scrape failed for %r: %s", topic.name, exc)
            return []

    async def _scrape_inner(
        self, topic: Topic, actor_id: str, limit: int
    ) -> list[ScrapedArticle]:
        actor_input = self._build_actor_input(actor_id, topic.name, limit)

        # Path A: route through Scalekit Connect (double-points track).
        if self._via_scalekit:
            try:
                items = await self._invoke_via_scalekit(actor_id, actor_input)
                if items:
                    return self._parse_items(items, topic, limit)
                _log.info("Scalekit→Apify returned empty, falling back to direct MCP")
            except Exception as exc:
                _log.warning("Scalekit→Apify failed (%s), falling back to direct MCP", exc)

        # Path B: direct Apify MCP over Streamable HTTP.
        return await self._invoke_direct(actor_id, actor_input, topic, limit)

    def _build_actor_input(self, actor_id: str, query: str, limit: int) -> dict[str, Any]:
        """Build actor-specific input payload."""
        if actor_id == "apify/google-search-scraper":
            return {"queries": query, "maxPagesPerQuery": 1, "resultsPerPage": limit}
        if actor_id == "apify/rag-web-browser":
            return {"query": query, "maxResults": limit}
        if actor_id == "apify/website-content-crawler":
            return {"startUrls": [{"url": f"https://github.com/search?q={query}"}], "maxCrawlPages": limit}
        # Generic fallback input shape.
        return {"query": query, "maxResults": limit}

    async def _invoke_via_scalekit(
        self, actor_id: str, actor_input: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Run an Apify Actor through Scalekit Connect's execute_tool.

        The Apify token lives in Scalekit's Token Vault — not in our env vars.
        This earns the Scalekit→Apify double-points hackathon track.
        """
        from trace.auth.scalekit import connect_execute_tool

        result = await connect_execute_tool(
            tool_name="apifymcp_call_actor",
            tool_input={"actor_id": actor_id, "input": actor_input},
            identifier=self._scalekit_identifier or "default",
        )

        if not isinstance(result, dict):
            return []
        if "error" in result:
            raise RuntimeError(str(result["error"]))

        # Normalize common Apify-via-Scalekit response shapes.
        for key in ("items", "data", "results"):
            val = result.get(key)
            if isinstance(val, list):
                return [v for v in val if isinstance(v, dict)]
        nested = result.get("output") or result.get("response") or {}
        if isinstance(nested, dict):
            for key in ("items", "data", "results"):
                val = nested.get(key)
                if isinstance(val, list):
                    return [v for v in val if isinstance(v, dict)]
        return []

    async def _invoke_direct(
        self,
        actor_id: str,
        actor_input: dict[str, Any],
        topic: Topic,
        limit: int,
    ) -> list[ScrapedArticle]:
        """Call Apify REST API directly as fallback."""
        url = f"https://api.apify.com/v2/acts/{actor_id.replace('/', '~')}/run-sync-get-dataset-items"
        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
                resp = await client.post(
                    url,
                    params={"token": self._api_token, "limit": limit},
                    json=actor_input,
                )
                resp.raise_for_status()
                items = resp.json()
                if isinstance(items, list):
                    # Archive raw scrape results to Tigris Data for provenance.
                    try:
                        import uuid as _uuid
                        from trace.storage.tigris import get_tigris_store
                        tigris = get_tigris_store()
                        if tigris.enabled:
                            run_id = str(_uuid.uuid4())[:8]
                            tigris.store_apify_artifact(topic.name, run_id, items[:limit])
                    except Exception:
                        pass
                    return self._parse_items(items, topic, limit)
        except Exception as exc:
            _log.warning("Apify REST fallback also failed for %r: %s", topic.name, exc)
        return []

    def _parse_items(
        self, items: list[dict[str, Any]], topic: Topic, limit: int
    ) -> list[ScrapedArticle]:
        results: list[ScrapedArticle] = []
        for i, item in enumerate(items[:limit]):
            url = item.get("url") or item.get("link") or item.get("canonicalUrl", "")
            title = item.get("title") or item.get("name") or item.get("heading", "")
            summary = (
                item.get("text")
                or item.get("description")
                or item.get("snippet")
                or item.get("markdown", "")[:500]
                or ""
            )
            if not url or not title:
                continue
            results.append(
                ScrapedArticle(
                    topic_id=topic.id,
                    topic_name=topic.name,
                    source=ContentSource.WEB_SEARCH,
                    title=str(title)[:200],
                    url=str(url),
                    summary=str(summary)[:1000],
                    relevance_score=max(0.0, 1.0 - i / max(1, limit)),
                )
            )
        return results

    def _extract_items(self, raw: str) -> list[dict[str, Any]]:
        """Parse a JSON string into a list of dicts."""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [v for v in parsed if isinstance(v, dict)]
            if isinstance(parsed, dict):
                for key in ("items", "data", "results"):
                    val = parsed.get(key)
                    if isinstance(val, list):
                        return [v for v in val if isinstance(v, dict)]
        except (json.JSONDecodeError, TypeError):
            pass
        return []
