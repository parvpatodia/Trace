"""
ApifyScraper — fetches web articles via an Apify actor using the native async client.

DEFAULT ACTOR: apify/rag-web-browser
  Semantic search gives significantly higher quality results than google-search-scraper
  for technical/research topics which make up the majority of curiosity topics.

SEARCH ALGORITHM (v2 — quality-first, mirrors trace/mcp/apify_client.py):
  1. QUERY AUGMENTATION — recency/quality modifiers tailored to curiosity type
  2. CANDIDATE EXPANSION — fetch FETCH_MULTIPLIER × max_results candidates
  3. SOURCE QUALITY SCORING — Tier 1/2 domain bonuses, reject-list filtering
  4. CLICKBAIT DETECTION — title pattern penalty
  5. RANK AND RETURN TOP N — domain-diversity enforced

WHY ApifyClientAsync INSTEAD OF ApifyClient + asyncio.to_thread:
  ApifyClientAsync._wait_for_finish uses await asyncio.sleep(0.25) between
  status polls (verified from SDK source). asyncio.wait_for cancels it
  correctly at the next await point — no zombie OS threads.

TWO TIMEOUT LAYERS:
  timeout_secs=45  — Apify platform kills the actor after 45s (server-side).
  wait_secs=45     — SDK stops waiting after 45s and returns what it has.
  asyncio.wait_for(timeout=60) — final asyncio-level safety net.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from apify_client import ApifyClientAsync

from trace.models import ContentSource, CuriosityType, ScrapedArticle, Topic
from trace.scraper.base import ArticleScraper, ScraperError

_log = logging.getLogger(__name__)
_DEFAULT_ACTOR = "apify/rag-web-browser"

# ── Source quality tiers ──────────────────────────────────────────────────────
_TIER1_DOMAINS = frozenset([
    "arxiv.org", "nature.com", "science.org", "cell.com", "sciencedirect.com",
    "openai.com", "deepmind.google", "anthropic.com", "research.google",
    "huggingface.co", "github.com", "github.blog", "papers.nips.cc",
    "proceedings.mlr.press", "distill.pub", "ml.berkeley.edu",
    "ai.googleblog.com", "ai.meta.com", "blog.research.google",
    "pair.withgoogle.com", "jmlr.org", "iclr.cc", "neurips.cc",
])

_TIER2_DOMAINS = frozenset([
    "towardsdatascience.com", "thegradient.pub", "ruder.io",
    "lilianweng.github.io", "colah.github.io", "jalammar.github.io",
    "sebastianraschka.com", "karpathy.github.io", "simonwillison.net",
    "eugeneyan.com", "hamel.dev", "lastweekin.ai", "importai.net",
    "theverge.com", "arstechnica.com", "wired.com", "technologyreview.com",
    "spectrum.ieee.org", "hackaday.com", "news.ycombinator.com",
    "substack.com", "techcrunch.com", "venturebeat.com",
])

_REJECT_DOMAINS = frozenset([
    "pinterest.com", "instagram.com", "tiktok.com", "facebook.com",
    "amazon.com", "ebay.com", "etsy.com", "shopify.com",
    "yelp.com", "tripadvisor.com", "booking.com",
])

_CLICKBAIT_RE = re.compile(
    r"(\d+\s+(ways|things|reasons|tips|tricks|hacks|steps|facts|secrets))|"
    r"(you (won't|will never|need to|have to) believe)|"
    r"(this (is why|one trick|simple))|"
    r"(what happens (when|if|next))|"
    r"(here's (why|what|how) (you|this|it))|"
    r"(the (truth|secret|real reason) (about|behind))",
    re.IGNORECASE,
)

# Fetch this many × max_results candidates, then filter to the best N
_FETCH_MULTIPLIER = 4

# ── Actor selection ───────────────────────────────────────────────────────────
_ACTOR_HINTS: list[tuple[list[str], str]] = [
    (["youtube", "video", "channel", "playlist"], "apify/youtube-scraper"),
    (["reddit", "subreddit", "r/"], "apify/reddit-scraper"),
    (["hacker news", "hn "], "apify/hacker-news-scraper"),
    # Default for everything technical/research
    (["arxiv", "paper", "research", "ml ", "ai ", "llm", "machine learning",
      "deep learning", "neural", "robotics", "programming", "code"], "apify/rag-web-browser"),
]


def _pick_actor(topic_name: str) -> str:
    lower = topic_name.lower()
    for keywords, actor_id in _ACTOR_HINTS:
        if any(kw in lower for kw in keywords):
            return actor_id
    return _DEFAULT_ACTOR


def _build_query(topic_name: str, topic: Topic) -> str:
    quoted = f'"{topic_name}"' if " " in topic_name else topic_name
    ctype = topic.curiosity_type
    depth = getattr(topic, "depth_score", 0.0)

    if ctype == CuriosityType.DEEP or depth > 12:
        return (
            f"{quoted} "
            f"(research OR paper OR study OR implementation OR analysis OR architecture) "
            f"after:2024-01-01"
        )
    elif ctype == CuriosityType.RECURRING:
        return (
            f"{quoted} "
            f"(2024 OR 2025) "
            f"(advances OR updates OR guide OR tutorial OR explained OR deep dive)"
        )
    elif ctype == CuriosityType.SHALLOW:
        return (
            f"{quoted} "
            f"(explained OR introduction OR beginner OR overview OR getting started) "
            f"after:2023-01-01"
        )
    return f"{quoted} (analysis OR guide OR explained OR research) after:2024-01-01"


def _domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _quality_score(item: dict[str, Any], topic_name: str) -> float:
    """Score an Apify result item [0, 1] for source quality."""
    url = str(item.get("url") or item.get("link") or item.get("canonicalUrl", ""))
    title = str(item.get("title") or item.get("name") or "")
    summary = str(
        item.get("text") or item.get("description") or
        item.get("snippet") or item.get("markdown", "")[:300] or ""
    )

    if not url or not title:
        return 0.0

    domain = _domain_of(url)
    if any(bad in domain for bad in _REJECT_DOMAINS):
        return 0.0
    if url.endswith(".pdf") and not summary:
        return 0.1

    score = 0.5
    if any(t1 in domain for t1 in _TIER1_DOMAINS):
        score += 0.35
    elif any(t2 in domain for t2 in _TIER2_DOMAINS):
        score += 0.15

    if _CLICKBAIT_RE.search(title):
        score -= 0.3

    tlen = len(title)
    if 40 <= tlen <= 100:
        score += 0.05
    elif tlen < 15 or tlen > 200:
        score -= 0.1

    slen = len(summary)
    if slen > 400:
        score += 0.1
    elif slen > 150:
        score += 0.05
    elif 0 < slen < 30:
        score -= 0.1

    topic_words = set(topic_name.lower().split())
    combined = (title + " " + summary).lower()
    hits = sum(1 for w in topic_words if w in combined)
    score += min(0.15, hits * 0.05)

    return max(0.0, min(1.0, score))


class ApifyScraper(ArticleScraper):
    """
    Fetches web articles about a topic using an Apify actor.

    v2: quality-first — uses rag-web-browser with curiosity-type-aware query
    augmentation, candidate expansion, and domain-quality ranking.

    Parameters:
        api_token:  Apify API token (required).
        actor_id:   Apify actor ID override (default: auto-selected per topic).
    """

    source = ContentSource.WEB_SEARCH

    def __init__(
        self,
        api_token: str,
        actor_id: str | None = None,
    ) -> None:
        if not api_token:
            raise ValueError("ApifyScraper: api_token must not be empty.")
        self._client = ApifyClientAsync(token=api_token)
        self._actor_id_override = actor_id

    async def scrape(self, topic: Topic, max_results: int = 5) -> list[ScrapedArticle]:
        actor_id = self._actor_id_override or _pick_actor(topic.name)
        fetch_n = max_results * _FETCH_MULTIPLIER
        try:
            items = await asyncio.wait_for(
                self._run_actor(topic, actor_id, min(fetch_n, 40)),
                timeout=60,
            )
        except asyncio.TimeoutError:
            _log.warning("Apify actor timed out for topic %r — skipping", topic.name)
            return []
        except Exception as exc:
            raise ScraperError(
                self.source,
                f"Apify actor '{actor_id}' failed: {exc}",
            ) from exc
        return self._rank_and_parse(items, topic, max_results)

    async def _run_actor(
        self, topic: Topic, actor_id: str, limit: int
    ) -> list[dict[str, Any]]:
        query = _build_query(topic.name, topic)
        actor_input = self._build_actor_input(actor_id, query, limit)

        run = await self._client.actor(actor_id).call(
            run_input=actor_input,
            timeout_secs=45,
            wait_secs=45,
        )
        if not run:
            _log.warning("Apify actor run returned None for topic %r", topic.name)
            return []
        dataset_id: str | None = run.get("defaultDatasetId")
        if not dataset_id:
            _log.warning("Apify run missing defaultDatasetId for topic %r", topic.name)
            return []
        raw_items: list[dict[str, Any]] = []
        async for item in self._client.dataset(dataset_id).iterate_items():
            raw_items.append(item)
        return self._flatten_items(raw_items)

    def _build_actor_input(
        self, actor_id: str, query: str, limit: int
    ) -> dict[str, Any]:
        if actor_id == "apify/rag-web-browser":
            return {"query": query, "maxResults": limit, "outputFormats": ["markdown"]}
        if actor_id == "apify/google-search-scraper":
            return {
                "queries": query,
                "resultsPerPage": limit,
                "maxResultsPerQuery": limit,
                "maxPagesPerQuery": 1,
            }
        if actor_id == "apify/hacker-news-scraper":
            return {"mode": "search", "searchQuery": query, "maxItems": limit}
        if actor_id == "apify/reddit-scraper":
            return {"searches": [query], "maxPostCount": limit, "searchType": "posts"}
        if actor_id == "apify/youtube-scraper":
            return {"searchKeywords": query, "maxResults": limit, "type": "search"}
        return {"query": query, "maxResults": limit}

    def _flatten_items(self, raw_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        flat: list[dict[str, Any]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            if "organicResults" in item and isinstance(item["organicResults"], list):
                flat.extend(r for r in item["organicResults"] if isinstance(r, dict))
            elif "items" in item and isinstance(item["items"], list):
                flat.extend(r for r in item["items"] if isinstance(r, dict))
            elif "title" in item or "url" in item:
                flat.append(item)
        return flat

    def _rank_and_parse(
        self, items: list[dict[str, Any]], topic: Topic, limit: int
    ) -> list[ScrapedArticle]:
        if not items:
            return []

        scored = [
            (q, item)
            for item in items
            if (q := _quality_score(item, topic.name)) > 0.05
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[ScrapedArticle] = []
        seen_domains: set[str] = set()

        for q_score, item in scored:
            if len(results) >= limit:
                break
            url = str(item.get("url") or item.get("link") or item.get("canonicalUrl", ""))
            title = str(item.get("title") or item.get("name") or "")
            if not url or not title:
                continue
            if not url.startswith(("http://", "https://")):
                continue

            domain = _domain_of(url)
            if domain and domain in seen_domains:
                continue
            if domain:
                seen_domains.add(domain)

            summary = str(
                item.get("text") or item.get("description") or
                item.get("snippet") or item.get("markdown", "")[:800] or ""
            )
            raw_date = (
                item.get("date") or item.get("pubDate") or
                item.get("publishedAt") or item.get("published_at")
            )
            results.append(
                ScrapedArticle(
                    topic_id=topic.id,
                    topic_name=topic.name,
                    source=self.source,
                    title=title[:500],
                    url=url,
                    summary=summary[:3000],
                    published_at=_parse_date(raw_date),
                    relevance_score=round(q_score, 3),
                    metadata={"quality_score": round(q_score, 3)},
                )
            )

        _log.info(
            "Apify ranked: topic=%r candidates=%d → kept=%d",
            topic.name, len(scored), len(results),
        )
        return results


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
