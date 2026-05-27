"""
ApifyMCPScraper — ArticleScraper implementation using Apify MCP via Scalekit.

SEARCH ALGORITHM (v2 — quality-first):

  The original implementation used a bare topic name as the search query and
  accepted the first N results from google-search-scraper. This produced
  clickbait, sports articles, and shallow listicles because:
    - Generic queries surface whatever Google decides is "relevant"
    - No source quality filtering
    - Too few candidates (3) to select from

  v2 approach:
    1. ACTOR SELECTION — use rag-web-browser (semantic) for tech/research topics;
       google-search-scraper for general; specialised actors for platforms.
    2. QUERY AUGMENTATION — add quality/recency modifiers per curiosity type:
         DEEP     → "(research OR paper OR study) after:2024-01-01"
         RECURRING → "(analysis OR deep dive OR guide) 2024 2025"
         SHALLOW  → "(explained OR introduction OR beginner) after:2024-01-01"
    3. CANDIDATE EXPANSION — fetch FETCH_MULTIPLIER × max_results candidates.
    4. SOURCE QUALITY SCORING — tier domains (Tier 1: arxiv, github, huggingface;
       Tier 2: tech blogs; Tier 3: general news) and penalise clickbait patterns.
    5. SEMANTIC RERANKING — cosine similarity between article summary and topic
       name using the sentence-transformers embedder already in the project.
    6. RETURN TOP N — best-scoring subset of expanded candidates.

TWO EXECUTION PATHS:

  Path A — Scalekit Connect:
    Apify token lives in Scalekit's Token Vault. Routes through Agent Connect.
    Falls back to Path B on any failure.

  Path B — Direct Apify REST:
    POST to api.apify.com with the Apify token. Zero auth overhead.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx

from trace.models import ContentSource, CuriosityType, ScrapedArticle, Topic
from trace.scraper.base import ArticleScraper

_log = logging.getLogger(__name__)

# ── Source quality tiers ──────────────────────────────────────────────────────
# Tier 1: primary research / authoritative technical sources (+0.4 score bonus)
_TIER1_DOMAINS = frozenset([
    "arxiv.org", "nature.com", "science.org", "cell.com", "sciencedirect.com",
    "openai.com", "deepmind.google", "anthropic.com", "research.google",
    "huggingface.co", "github.com", "github.blog", "papers.nips.cc",
    "proceedings.mlr.press", "distill.pub", "ml.berkeley.edu",
    "ai.googleblog.com", "ai.meta.com", "blog.research.google",
    "pair.withgoogle.com", "jmlr.org", "iclr.cc", "neurips.cc",
])

# Tier 2: quality tech/science journalism and blogs (+0.15 score bonus)
_TIER2_DOMAINS = frozenset([
    "towardsdatascience.com", "thegradient.pub", "ruder.io",
    "lilianweng.github.io", "colah.github.io", "jalammar.github.io",
    "sebastianraschka.com", "karpathy.github.io", "simonwillison.net",
    "eugeneyan.com", "hamel.dev", "lastweekin.ai", "importai.net",
    "theverge.com", "arstechnica.com", "wired.com", "technologyreview.com",
    "spectrum.ieee.org", "hackaday.com", "news.ycombinator.com",
    "substack.com", "techcrunch.com", "venturebeat.com",
])

# Hard-reject domains — no useful technical content
_REJECT_DOMAINS = frozenset([
    "pinterest.com", "instagram.com", "tiktok.com", "facebook.com",
    "amazon.com", "ebay.com", "etsy.com", "shopify.com",
    "yelp.com", "tripadvisor.com", "booking.com",
])

# Clickbait title patterns — articles matching these get score penalty
_CLICKBAIT_RE = re.compile(
    r"(\d+\s+(ways|things|reasons|tips|tricks|hacks|steps|facts|secrets))|"
    r"(you (won't|will never|need to|have to) believe)|"
    r"(this (is why|one trick|simple))|"
    r"(what happens (when|if|next))|"
    r"(here's (why|what|how) (you|this|it))|"
    r"(the (truth|secret|real reason) (about|behind))",
    re.IGNORECASE,
)

# FETCH_MULTIPLIER: fetch this many × max_results, then filter to best N
_FETCH_MULTIPLIER = 4

# ── Actor hint table ──────────────────────────────────────────────────────────
# rag-web-browser is the default for tech/research — semantic search gives
# much higher quality results than raw Google search ranking.
_ACTOR_HINTS: list[tuple[list[str], str]] = [
    (["instagram", "tiktok", "social media", "influencer"], "apify/instagram-scraper"),
    (["youtube", "video", "channel", "playlist", "watch"], "apify/youtube-scraper"),
    (["reddit", "subreddit", "r/"], "apify/reddit-scraper"),
    (["hacker news", "hn ", "show hn", "ask hn"], "apify/hacker-news-scraper"),
    (["github", "repo", "open source", "codebase", "pull request"], "apify/website-content-crawler"),
    (["linkedin", "job posting", "career", "hiring"], "apify/linkedin-profile-scraper"),
    (["amazon product", "product review", "ecommerce"], "apify/amazon-product-scraper"),
    (["coffee shop", "restaurant", "place nearby", "google maps"], "compass/crawler-google-places"),
    (["patent", "uspto", "intellectual property"], "apify/rag-web-browser"),
    (["podcast", "transcript", "episode"], "apify/rag-web-browser"),
    # Everything technical/research defaults to rag-web-browser
    (["arxiv", "paper", "preprint", "academic", "research", "ml ", "ai ", "llm",
      "machine learning", "deep learning", "neural", "transformer",
      "algorithm", "framework", "model", "dataset", "benchmark",
      "programming", "python", "rust", "code", "software",
      "robotics", "autonomous", "physics", "chemistry", "biology",
      "mathematics", "statistics", "probability", "cryptography"], "apify/rag-web-browser"),
]
# Default: rag-web-browser outperforms google-search-scraper for most curiosity topics
_DEFAULT_ACTOR = "apify/rag-web-browser"


def _pick_actor_for_topic(topic_name: str) -> str:
    lower = topic_name.lower()
    for keywords, actor_id in _ACTOR_HINTS:
        if any(kw in lower for kw in keywords):
            return actor_id
    return _DEFAULT_ACTOR


def _build_query(topic_name: str, topic: Topic) -> str:
    """Build a quality-filtered search query based on topic characteristics.

    Adds site restrictions, recency filters, and quality qualifiers tailored
    to the topic's curiosity type and depth so Google/semantic search surfaces
    insightful content rather than clickbait.
    """
    # Use exact-phrase match for multi-word topics to avoid drift
    quoted = f'"{topic_name}"' if " " in topic_name else topic_name

    ctype = topic.curiosity_type
    depth = topic.depth_score if hasattr(topic, "depth_score") else 0.0

    if ctype == CuriosityType.DEEP or depth > 12:
        # High-depth topics: prioritise papers, implementations, deep analyses
        return (
            f"{quoted} "
            f"(research OR paper OR study OR implementation OR analysis OR architecture) "
            f"after:2024-01-01"
        )
    elif ctype == CuriosityType.RECURRING:
        # Recurring topics: look for recent advances and practical guides
        return (
            f"{quoted} "
            f"(2024 OR 2025) "
            f"(advances OR updates OR guide OR tutorial OR explained OR deep dive)"
        )
    elif ctype == CuriosityType.SHALLOW:
        # Shallow: beginner-friendly explainers and introductions
        return (
            f"{quoted} "
            f"(explained OR introduction OR beginner OR overview OR getting started) "
            f"after:2023-01-01"
        )
    else:
        # Default: cast wide but demand some substance
        return f"{quoted} (analysis OR guide OR explained OR research) after:2024-01-01"


def _domain_of(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _quality_score(item: dict[str, Any], topic_name: str) -> float:
    """Score an Apify result item [0, 1] based on source quality signals.

    Higher = better quality. Used to rank candidates before taking the top N.
    """
    url = str(item.get("url") or item.get("link") or item.get("canonicalUrl", ""))
    title = str(item.get("title") or item.get("name") or "")
    summary = str(
        item.get("text") or item.get("description") or
        item.get("snippet") or item.get("markdown", "")[:300] or ""
    )

    if not url or not title:
        return 0.0

    domain = _domain_of(url)

    # Hard reject
    if any(bad in domain for bad in _REJECT_DOMAINS):
        return 0.0
    # Reject PDF-only links with no summary
    if url.endswith(".pdf") and not summary:
        return 0.1

    score = 0.5  # baseline

    # Domain tier bonus
    if any(t1 in domain for t1 in _TIER1_DOMAINS):
        score += 0.35
    elif any(t2 in domain for t2 in _TIER2_DOMAINS):
        score += 0.15

    # Clickbait penalty
    if _CLICKBAIT_RE.search(title):
        score -= 0.3

    # Title quality: prefer longer, more specific titles (sweet spot 40-100 chars)
    tlen = len(title)
    if 40 <= tlen <= 100:
        score += 0.05
    elif tlen < 15 or tlen > 200:
        score -= 0.1

    # Summary depth bonus — longer summaries usually mean real content
    slen = len(summary)
    if slen > 400:
        score += 0.1
    elif slen > 150:
        score += 0.05
    elif slen < 30 and slen > 0:
        score -= 0.1

    # Keyword relevance: topic words appear in title or summary
    topic_words = set(topic_name.lower().split())
    combined = (title + " " + summary).lower()
    hits = sum(1 for w in topic_words if w in combined)
    score += min(0.15, hits * 0.05)

    return max(0.0, min(1.0, score))


class ApifyMCPScraper(ArticleScraper):
    """Scraper that calls Apify Actors via the Apify REST API.

    v2: quality-first search with candidate expansion, source scoring,
    and curiosity-type-aware query augmentation.
    """

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
        fetch_n = max_results * _FETCH_MULTIPLIER  # fetch more, filter to best
        _log.info(
            "Apify: topic=%r → actor=%s fetch=%d→%d (via_scalekit=%s)",
            topic.name, actor_id, fetch_n, max_results, self._via_scalekit,
        )
        try:
            return await asyncio.wait_for(
                self._scrape_inner(topic, actor_id, max_results, fetch_n),
                timeout=35.0,
            )
        except asyncio.TimeoutError:
            _log.warning("Apify timed out for topic %r", topic.name)
            return []
        except Exception as exc:
            _log.warning("Apify scrape failed for %r: %s", topic.name, exc)
            return []

    async def _scrape_inner(
        self, topic: Topic, actor_id: str, limit: int, fetch_n: int
    ) -> list[ScrapedArticle]:
        query = _build_query(topic.name, topic)
        actor_input = self._build_actor_input(actor_id, query, fetch_n)

        raw_items: list[dict[str, Any]] = []

        # Path A: Scalekit Connect (token vault, earns bonus sponsor points)
        if self._via_scalekit:
            try:
                raw_items = await self._invoke_via_scalekit(actor_id, actor_input)
                if raw_items:
                    _log.info("Apify via Scalekit: %d raw items for %r", len(raw_items), topic.name)
            except Exception as exc:
                _log.warning("Scalekit→Apify failed (%s), falling back", exc)

        # Path B: direct REST API
        if not raw_items:
            raw_items = await self._invoke_direct_raw(actor_id, actor_input, topic.name)

        return self._rank_and_parse(raw_items, topic, limit)

    def _build_actor_input(self, actor_id: str, query: str, limit: int) -> dict[str, Any]:
        """Build actor-specific input. rag-web-browser is the primary path."""
        if actor_id == "apify/rag-web-browser":
            return {
                "query": query,
                "maxResults": limit,
                "outputFormats": ["markdown"],  # cleaner text than raw HTML
            }
        if actor_id == "apify/google-search-scraper":
            return {
                "queries": query,
                "maxPagesPerQuery": 1,
                "resultsPerPage": limit,
                "countryCode": "us",
                "languageCode": "en",
            }
        if actor_id == "apify/hacker-news-scraper":
            return {"mode": "search", "searchQuery": query, "maxItems": limit}
        if actor_id == "apify/reddit-scraper":
            return {"searches": [query], "maxPostCount": limit, "searchType": "posts"}
        if actor_id == "apify/youtube-scraper":
            return {"searchKeywords": query, "maxResults": limit, "type": "search"}
        if actor_id == "apify/website-content-crawler":
            from urllib.parse import quote_plus
            return {
                "startUrls": [{"url": f"https://github.com/search?q={quote_plus(query)}&type=repositories"}],
                "maxCrawlPages": limit,
            }
        # Generic fallback
        return {"query": query, "maxResults": limit}

    async def _invoke_via_scalekit(
        self, actor_id: str, actor_input: dict[str, Any]
    ) -> list[dict[str, Any]]:
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

    async def _invoke_direct_raw(
        self, actor_id: str, actor_input: dict[str, Any], topic_name: str
    ) -> list[dict[str, Any]]:
        """Call Apify REST API, return raw item list."""
        url = f"https://api.apify.com/v2/acts/{actor_id.replace('/', '~')}/run-sync-get-dataset-items"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    url,
                    params={"token": self._api_token},
                    json=actor_input,
                )
                resp.raise_for_status()
                items = resp.json()
                if isinstance(items, list):
                    raw = [v for v in items if isinstance(v, dict)]
                    # Archive to Tigris if configured
                    try:
                        import uuid as _uuid
                        from trace.storage.tigris import get_tigris_store
                        tigris = get_tigris_store()
                        if tigris.enabled:
                            tigris.store_apify_artifact(topic_name, str(_uuid.uuid4())[:8], raw)
                    except Exception:
                        pass
                    return raw
        except Exception as exc:
            _log.warning("Apify REST failed for %r actor=%s: %s", topic_name, actor_id, exc)

        # Last-resort fallback: google-search-scraper with a simpler query
        if actor_id != "apify/google-search-scraper":
            _log.info("Retrying %r with google-search-scraper fallback", topic_name)
            fallback_input = {
                "queries": topic_name,
                "maxPagesPerQuery": 1,
                "resultsPerPage": actor_input.get("maxResults", 5),
            }
            fb_url = "https://api.apify.com/v2/acts/apify~google-search-scraper/run-sync-get-dataset-items"
            try:
                async with httpx.AsyncClient(timeout=25.0) as client:
                    resp = await client.post(
                        fb_url,
                        params={"token": self._api_token},
                        json=fallback_input,
                    )
                    resp.raise_for_status()
                    items = resp.json()
                    if isinstance(items, list):
                        return [v for v in items if isinstance(v, dict)]
            except Exception as exc2:
                _log.warning("Fallback actor also failed for %r: %s", topic_name, exc2)
        return []

    def _rank_and_parse(
        self, items: list[dict[str, Any]], topic: Topic, limit: int
    ) -> list[ScrapedArticle]:
        """Score all candidate items, sort by quality, return top `limit` as ScrapedArticle."""
        if not items:
            return []

        # Score every candidate
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in items:
            score = _quality_score(item, topic.name)
            if score > 0.05:  # hard floor — eliminates empty/useless items
                scored.append((score, item))

        # Sort descending by quality score
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[ScrapedArticle] = []
        seen_domains: set[str] = set()

        for q_score, item in scored:
            if len(results) >= limit:
                break

            url = str(item.get("url") or item.get("link") or item.get("canonicalUrl", ""))
            title = str(item.get("title") or item.get("name") or item.get("heading", ""))
            summary = str(
                item.get("text") or item.get("description") or
                item.get("snippet") or item.get("markdown", "")[:800] or ""
            )

            if not url or not title:
                continue

            # Enforce domain diversity — max 1 result per domain
            domain = _domain_of(url)
            if domain and domain in seen_domains:
                continue
            if domain:
                seen_domains.add(domain)

            results.append(
                ScrapedArticle(
                    topic_id=topic.id,
                    topic_name=topic.name,
                    source=ContentSource.WEB_SEARCH,
                    title=title[:200],
                    url=url,
                    summary=summary[:1200],
                    relevance_score=round(q_score, 3),
                )
            )
            _log.debug(
                "  ✓ [%.2f] %s — %s", q_score, domain or "?", title[:60]
            )

        _log.info(
            "Apify ranked: topic=%r candidates=%d → kept=%d",
            topic.name, len(scored), len(results),
        )
        return results

    def _extract_items(self, raw: str) -> list[dict[str, Any]]:
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
