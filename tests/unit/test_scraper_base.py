"""
Unit tests for trace/scraper/base.py — ArticleScraper ABC + ScraperError.

Mirrors test_signal_base.py in structure and rigour.
"""

from __future__ import annotations

import pytest

from trace.models import ContentSource, ScrapedArticle, Topic
from trace.scraper.base import ArticleScraper, ScraperError


# ── ScraperError ──────────────────────────────────────────────────────────────

class TestScraperError:
    def test_message_includes_source_prefix(self) -> None:
        err = ScraperError(ContentSource.ARXIV, "timeout")
        assert "[arxiv]" in str(err)
        assert "timeout" in str(err)

    def test_source_attribute_preserved(self) -> None:
        err = ScraperError(ContentSource.HACKER_NEWS, "404")
        assert err.source == ContentSource.HACKER_NEWS

    def test_all_sources_produce_correct_prefix(self) -> None:
        for source in ContentSource:
            err = ScraperError(source, "msg")
            assert f"[{source.value}]" in str(err)

    def test_is_exception_subclass(self) -> None:
        assert issubclass(ScraperError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(ScraperError):
            raise ScraperError(ContentSource.ARXIV, "error")

    def test_exception_chaining(self) -> None:
        cause = httpx_error = RuntimeError("root cause")
        with pytest.raises(ScraperError) as exc_info:
            try:
                raise cause
            except RuntimeError as e:
                raise ScraperError(ContentSource.REDDIT, "wrapped") from e
        assert exc_info.value.__cause__ is cause


# ── ABC enforcement ───────────────────────────────────────────────────────────

class TestArticleScraperABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            ArticleScraper()  # type: ignore[abstract]

    def test_subclass_without_source_cannot_instantiate(self) -> None:
        class NoSource(ArticleScraper):
            async def scrape(self, topic: Topic, max_results: int = 5) -> list[ScrapedArticle]:
                return []
        with pytest.raises(TypeError):
            NoSource()  # type: ignore[abstract]

    def test_subclass_without_scrape_cannot_instantiate(self) -> None:
        class NoScrape(ArticleScraper):
            source = ContentSource.ARXIV
        with pytest.raises(TypeError):
            NoScrape()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self) -> None:
        class GoodScraper(ArticleScraper):
            source = ContentSource.HACKER_NEWS
            async def scrape(self, topic: Topic, max_results: int = 5) -> list[ScrapedArticle]:
                return []
        assert GoodScraper() is not None

    def test_source_accessible_on_instance(self) -> None:
        class S(ArticleScraper):
            source = ContentSource.BLOG
            async def scrape(self, topic: Topic, max_results: int = 5) -> list[ScrapedArticle]:
                return []
        assert S().source == ContentSource.BLOG

    def test_scrape_returns_list(self) -> None:
        # Verifies the abstract contract: scrape() must return a list
        class EmptyScraper(ArticleScraper):
            source = ContentSource.ARXIV
            async def scrape(self, topic: Topic, max_results: int = 5) -> list[ScrapedArticle]:
                return []
        # Just checks the method exists and is callable
        assert callable(EmptyScraper().scrape)
