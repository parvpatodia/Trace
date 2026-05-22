"""
ArXivScraper — fetches papers from the ArXiv API for a given topic.

ArXiv API:
  GET https://export.arxiv.org/api/query
    ?search_query=all:<query>
    &max_results=<n>
    &sortBy=relevance
    &sortOrder=descending

Response format: Atom XML (application/atom+xml)
  <feed xmlns="http://www.w3.org/2005/Atom">
    <entry>
      <id>http://arxiv.org/abs/1706.03762v7</id>
      <title>Attention Is All You Need</title>
      <summary>The dominant sequence transduction...</summary>
      <published>2017-06-12T17:00:00Z</published>
    </entry>
    ...
  </feed>

Signal extraction per entry:
  title        → ScrapedArticle.title (stripped)
  <id> text    → ScrapedArticle.url (normalized to https://)
  <summary>    → ScrapedArticle.summary (truncated to 3000 chars)
  <published>  → ScrapedArticle.published_at (UTC datetime)

Relevance scoring:
  Position-based: rank 0 → 1.0, rank n-1 → 1/(n+1).
  ArXiv results are already sorted by relevance.

WHY httpx.AsyncClient DEPENDENCY INJECTION:
  Allows tests to pass a pre-configured client (with respx transport mock)
  without patching global state. Production code passes None → scraper
  creates a client for the request.

WHY xml.etree.ElementTree (not lxml / BeautifulSoup):
  stdlib only — no extra dependency. ArXiv Atom is well-formed XML.
  lxml adds ~15MB to the image for no benefit here.

WHY NORMALIZE ID TO https://:
  ArXiv <id> elements use http:// scheme but all modern requests redirect.
  RawSignal.url already enforces http/https; ScrapedArticle.url does too.
  Normalising to https:// avoids a redirect on every link open.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import httpx

from trace.models import ContentSource, ScrapedArticle, Topic
from trace.scraper.base import ArticleScraper, ScraperError

_ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM_NS = "http://www.w3.org/2005/Atom"
_TIMEOUT = 20.0


class ArXivScraper(ArticleScraper):
    """
    Fetches arXiv papers relevant to a curiosity topic.

    Parameters:
        client: optional httpx.AsyncClient for dependency injection / testing.
                When None, a fresh client is created per scrape() call.
        timeout: HTTP request timeout in seconds (default 20s — ArXiv can be slow).
    """

    source = ContentSource.ARXIV

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
        params = {
            "search_query": f"all:{topic.name}",
            "max_results": str(max_results),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        xml_text = await self._fetch(params)
        return self._parse(xml_text, topic, max_results)

    async def _fetch(self, params: dict[str, str]) -> str:
        try:
            if self._client is not None:
                response = await self._client.get(_ARXIV_API, params=params)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as c:
                    response = await c.get(_ARXIV_API, params=params)
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as e:
            raise ScraperError(
                self.source,
                f"HTTP {e.response.status_code} from ArXiv API",
            ) from e
        except httpx.RequestError as e:
            raise ScraperError(
                self.source, f"Network error calling ArXiv API: {e}"
            ) from e

    def _parse(
        self, xml_text: str, topic: Topic, max_results: int
    ) -> list[ScrapedArticle]:
        try:
            root = ET.fromstring(xml_text.strip())
        except ET.ParseError as e:
            raise ScraperError(
                self.source, f"Failed to parse ArXiv XML response: {e}"
            ) from e

        ns = {"atom": _ATOM_NS}
        entries = root.findall("atom:entry", ns)
        articles: list[ScrapedArticle] = []

        for i, entry in enumerate(entries[:max_results]):
            article = self._parse_entry(entry, ns, topic, i, len(entries))
            if article is not None:
                articles.append(article)

        return articles

    def _parse_entry(
        self,
        entry: ET.Element,
        ns: dict[str, str],
        topic: Topic,
        rank: int,
        total: int,
    ) -> ScrapedArticle | None:
        id_el = entry.find("atom:id", ns)
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        published_el = entry.find("atom:published", ns)

        if id_el is None or title_el is None:
            return None

        raw_url = (id_el.text or "").strip()
        url = raw_url.replace("http://", "https://", 1)
        title = (title_el.text or "").strip()

        if not url or not title:
            return None

        summary = (summary_el.text or "").strip() if summary_el is not None else ""
        published_at = _parse_datetime(
            (published_el.text or "").strip() if published_el is not None else ""
        )

        relevance_score = max(0.0, 1.0 - rank / max(1, total))

        return ScrapedArticle(
            topic_id=topic.id,
            topic_name=topic.name,
            source=self.source,
            title=title[:500],
            url=url,
            summary=summary[:3000],
            published_at=published_at,
            relevance_score=relevance_score,
            metadata={"rank": rank},
        )


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
