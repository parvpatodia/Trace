"""
ArticleScraper abstract base class and ScraperError.

Design mirrors SignalCollector:
  - source class attribute (ContentSource) identifies the scraper
  - scrape(topic, max_results) is async: all scrapers do network I/O
  - ScraperError is the single domain exception for all scraper failures

WHY max_results IN scrape() NOT __init__():
  The same scraper instance may be reused across many topics in a pipeline
  run. Putting max_results in scrape() lets the caller vary the budget
  per topic (e.g. give more results to RECURRING topics) without
  re-instantiating scrapers.

WHY ScraperError MIRRORS SignalCollectionError:
  Pipeline nodes that call scrapers need to catch one exception type per
  source. Propagating raw httpx.HTTPStatusError, xml.ParseError, etc. would
  require every caller to import and handle multiple exception types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from trace.models import ContentSource, ScrapedArticle, Topic


class ScraperError(Exception):
    """
    Raised by any ArticleScraper when scraping fails at runtime.

    Covers: HTTP errors, timeouts, malformed responses, parse failures.
    NOT for programming errors (those remain as TypeError, etc.).

    source attribute: which scraper raised this, for structured logging.
    """

    def __init__(self, source: ContentSource, message: str) -> None:
        self.source = source
        super().__init__(f"[{source.value}] {message}")


class ArticleScraper(ABC):
    """
    Abstract base for all article scrapers.

    Subclass contract:
      1. Declare `source = ContentSource.<VALUE>` as a class attribute.
      2. Implement `async scrape(topic, max_results) -> list[ScrapedArticle]`.
         Return empty list if no articles found (not an error).
         Raise ScraperError on any runtime failure.

    Example:
        class MyScraper(ArticleScraper):
            source = ContentSource.ARXIV

            async def scrape(self, topic: Topic, max_results: int = 5) -> list[ScrapedArticle]:
                ...
    """

    @property
    @abstractmethod
    def source(self) -> ContentSource:
        """Identifies which content source this scraper reads from."""
        ...

    @abstractmethod
    async def scrape(
        self, topic: Topic, max_results: int = 5
    ) -> list[ScrapedArticle]:
        """
        Fetch articles relevant to the given topic.

        Args:
            topic:       the curiosity topic to search for.
            max_results: maximum number of articles to return.

        Returns:
            List of ScrapedArticle objects. May be empty; never None.

        Raises:
            ScraperError: if the source is unreachable or returns
                          malformed data.
        """
        ...
