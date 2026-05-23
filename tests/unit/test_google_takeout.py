"""
Unit tests for trace/signals/google_takeout.py — GoogleTakeoutCollector.

Test categories:
  1. Constructor validation: missing file, naive since datetime
  2. Happy-path collection: valid entries parsed into RawSignals
  3. Noise filtering: chrome:// URLs, uninformative titles, missing fields
  4. Time filtering: `since` parameter correctly excludes old entries
  5. Timestamp conversion: time_usec (microseconds) → UTC datetime
  6. URL handling: http/https kept, ftp/other become None
  7. Error cases: malformed JSON, wrong top-level type

All tests use tmp_path to write real JSON fixture files.
No mocking of I/O — collector reads files synchronously via asyncio.to_thread.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trace.models import RawSignal, SignalSource
from trace.signals.base import SignalCollectionError
from trace.signals.google_takeout import GoogleTakeoutCollector


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_history(entries: list[dict]) -> dict:
    return {"Browser History": entries}


def write_history(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "BrowserHistory.json"
    path.write_text(json.dumps(make_history(entries)), encoding="utf-8")
    return path


def valid_entry(
    title: str = "Attention Is All You Need - arXiv",
    url: str = "https://arxiv.org/abs/1706.03762",
    time_usec: int = 1_704_067_200_000_000,  # 2024-01-01 00:00:00 UTC
    page_transition: str = "LINK",
) -> dict:
    return {
        "title": title,
        "url": url,
        "time_usec": time_usec,
        "page_transition": page_transition,
    }


# ── Constructor validation ────────────────────────────────────────────────────

class TestConstructorValidation:
    async def test_missing_file_raises_signal_collection_error(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.json"
        collector = GoogleTakeoutCollector(path)
        with pytest.raises(SignalCollectionError) as exc_info:
            await collector.collect()
        assert exc_info.value.source == SignalSource.GOOGLE_TAKEOUT
        assert "BrowserHistory.json not found" in str(exc_info.value)

    def test_naive_since_raises_value_error(self, browser_history_path: Path) -> None:
        naive = datetime(2024, 1, 1, 12, 0, 0)  # no tzinfo
        with pytest.raises(ValueError, match="timezone-aware"):
            GoogleTakeoutCollector(browser_history_path, since=naive)

    def test_aware_since_accepted(self, browser_history_path: Path) -> None:
        since = datetime(2024, 1, 1, tzinfo=timezone.utc)
        collector = GoogleTakeoutCollector(browser_history_path, since=since)
        assert collector is not None

    def test_no_since_accepted(self, browser_history_path: Path) -> None:
        collector = GoogleTakeoutCollector(browser_history_path)
        assert collector is not None

    def test_source_is_google_takeout(self, browser_history_path: Path) -> None:
        collector = GoogleTakeoutCollector(browser_history_path)
        assert collector.source == SignalSource.GOOGLE_TAKEOUT


# ── Happy-path collection ─────────────────────────────────────────────────────

class TestHappyPath:
    async def test_collect_returns_list_of_raw_signals(
        self, browser_history_path: Path
    ) -> None:
        signals = await GoogleTakeoutCollector(browser_history_path).collect()
        assert isinstance(signals, list)
        assert all(isinstance(s, RawSignal) for s in signals)

    async def test_collect_returns_correct_count(
        self, browser_history_path: Path
    ) -> None:
        # browser_history_path fixture has 2 valid entries
        signals = await GoogleTakeoutCollector(browser_history_path).collect()
        assert len(signals) == 2

    async def test_signal_source_is_google_takeout(
        self, browser_history_path: Path
    ) -> None:
        signals = await GoogleTakeoutCollector(browser_history_path).collect()
        for sig in signals:
            assert sig.source == SignalSource.GOOGLE_TAKEOUT

    async def test_signal_content_is_page_title(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [valid_entry(title="ViT: An Image is Worth 16x16 Words")])
        signals = await GoogleTakeoutCollector(path).collect()
        assert len(signals) == 1
        assert signals[0].content == "ViT: An Image is Worth 16x16 Words"

    async def test_signal_url_is_preserved(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [valid_entry(url="https://example.com/paper")])
        signals = await GoogleTakeoutCollector(path).collect()
        assert signals[0].url == "https://example.com/paper"

    async def test_metadata_contains_time_usec(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [valid_entry(time_usec=1_704_067_200_000_000)])
        signals = await GoogleTakeoutCollector(path).collect()
        assert signals[0].metadata["time_usec"] == 1_704_067_200_000_000

    async def test_metadata_contains_page_transition(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [valid_entry(page_transition="TYPED")])
        signals = await GoogleTakeoutCollector(path).collect()
        assert signals[0].metadata["page_transition"] == "TYPED"

    async def test_empty_history_returns_empty_list(self, tmp_path: Path) -> None:
        path = tmp_path / "BrowserHistory.json"
        path.write_text(json.dumps({"Browser History": []}), encoding="utf-8")
        signals = await GoogleTakeoutCollector(path).collect()
        assert signals == []


# ── Timestamp conversion ──────────────────────────────────────────────────────

class TestTimestampConversion:
    async def test_time_usec_converts_to_utc_datetime(self, tmp_path: Path) -> None:
        # 1_704_067_200_000_000 usec = 1_704_067_200 sec = 2024-01-01 00:00:00 UTC
        path = write_history(tmp_path, [valid_entry(time_usec=1_704_067_200_000_000)])
        signals = await GoogleTakeoutCollector(path).collect()
        expected = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert signals[0].timestamp == expected

    async def test_timestamp_is_utc_aware(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [valid_entry()])
        signals = await GoogleTakeoutCollector(path).collect()
        assert signals[0].timestamp.tzinfo is not None
        assert signals[0].timestamp.tzinfo == timezone.utc

    async def test_two_entries_have_distinct_timestamps(self, tmp_path: Path) -> None:
        entries = [
            valid_entry(time_usec=1_704_067_200_000_000),
            valid_entry(title="Second Page", url="https://example.com/page2", time_usec=1_704_153_600_000_000),
        ]
        path = write_history(tmp_path, entries)
        signals = await GoogleTakeoutCollector(path).collect()
        assert signals[0].timestamp != signals[1].timestamp


# ── Noise filtering: URL prefixes ─────────────────────────────────────────────

class TestUrlFiltering:
    @pytest.mark.parametrize("bad_url", [
        "chrome://newtab/",
        "chrome://settings/passwords",
        "chrome-extension://abcdefg/popup.html",
        "edge://settings/",
        "about:blank",
        "about:newtab",
        "data:text/html,<html></html>",
        "file:///home/user/index.html",
    ])
    async def test_internal_url_filtered_out(
        self, tmp_path: Path, bad_url: str
    ) -> None:
        path = write_history(tmp_path, [valid_entry(url=bad_url)])
        signals = await GoogleTakeoutCollector(path).collect()
        assert signals == [], f"Expected {bad_url!r} to be filtered"

    async def test_https_url_not_filtered(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [valid_entry(url="https://example.com")])
        signals = await GoogleTakeoutCollector(path).collect()
        assert len(signals) == 1

    async def test_http_url_not_filtered(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [valid_entry(url="http://example.com")])
        signals = await GoogleTakeoutCollector(path).collect()
        assert len(signals) == 1

    async def test_ftp_url_produces_signal_with_none_url(
        self, tmp_path: Path
    ) -> None:
        # ftp:// is not an internal URL, so not filtered by _should_skip_url.
        # But RawSignal.url must be http/https, so safe_url becomes None.
        path = write_history(tmp_path, [valid_entry(url="ftp://ftp.example.com/file")])
        signals = await GoogleTakeoutCollector(path).collect()
        assert len(signals) == 1
        assert signals[0].url is None


# ── Noise filtering: title quality ────────────────────────────────────────────

class TestTitleFiltering:
    async def test_title_shorter_than_3_chars_filtered(
        self, tmp_path: Path
    ) -> None:
        path = write_history(tmp_path, [valid_entry(title="Ab")])
        signals = await GoogleTakeoutCollector(path).collect()
        assert signals == []

    async def test_title_exactly_3_chars_accepted(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [valid_entry(title="LLM")])
        signals = await GoogleTakeoutCollector(path).collect()
        assert len(signals) == 1

    async def test_title_equal_to_url_filtered(self, tmp_path: Path) -> None:
        url = "https://example.com/long-path"
        path = write_history(tmp_path, [valid_entry(title=url, url=url)])
        signals = await GoogleTakeoutCollector(path).collect()
        assert signals == []

    async def test_title_different_from_url_accepted(self, tmp_path: Path) -> None:
        url = "https://example.com"
        path = write_history(tmp_path, [valid_entry(title="Example Domain", url=url)])
        signals = await GoogleTakeoutCollector(path).collect()
        assert len(signals) == 1

    async def test_title_with_leading_whitespace_stripped(
        self, tmp_path: Path
    ) -> None:
        path = write_history(tmp_path, [valid_entry(title="  ViT Paper  ")])
        signals = await GoogleTakeoutCollector(path).collect()
        assert len(signals) == 1
        assert signals[0].content == "ViT Paper"


# ── Noise filtering: missing required fields ──────────────────────────────────

class TestMissingFieldFiltering:
    async def test_missing_title_filtered(self, tmp_path: Path) -> None:
        entry = {"url": "https://example.com", "time_usec": 1_704_067_200_000_000}
        path = write_history(tmp_path, [entry])
        signals = await GoogleTakeoutCollector(path).collect()
        assert signals == []

    async def test_missing_url_filtered(self, tmp_path: Path) -> None:
        entry = {"title": "Some Page", "time_usec": 1_704_067_200_000_000}
        path = write_history(tmp_path, [entry])
        signals = await GoogleTakeoutCollector(path).collect()
        assert signals == []

    async def test_missing_time_usec_filtered(self, tmp_path: Path) -> None:
        entry = {"title": "Some Page", "url": "https://example.com"}
        path = write_history(tmp_path, [entry])
        signals = await GoogleTakeoutCollector(path).collect()
        assert signals == []

    async def test_empty_title_filtered(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [valid_entry(title="")])
        signals = await GoogleTakeoutCollector(path).collect()
        assert signals == []

    async def test_empty_url_filtered(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [valid_entry(url="")])
        signals = await GoogleTakeoutCollector(path).collect()
        assert signals == []

    async def test_valid_entry_alongside_invalid_partial_parse(
        self, tmp_path: Path
    ) -> None:
        entries = [
            valid_entry(),  # good
            {"title": "Missing url", "time_usec": 1_704_067_200_000_000},  # bad
        ]
        path = write_history(tmp_path, entries)
        signals = await GoogleTakeoutCollector(path).collect()
        assert len(signals) == 1


# ── Time filtering (since parameter) ─────────────────────────────────────────

class TestSinceFiltering:
    async def test_entry_after_since_included(self, tmp_path: Path) -> None:
        # time_usec = 2024-01-01 00:00:00 UTC
        since = datetime(2023, 12, 31, tzinfo=timezone.utc)
        path = write_history(tmp_path, [valid_entry(time_usec=1_704_067_200_000_000)])
        signals = await GoogleTakeoutCollector(path, since=since).collect()
        assert len(signals) == 1

    async def test_entry_before_since_excluded(self, tmp_path: Path) -> None:
        # time_usec = 2024-01-01 00:00:00 UTC
        since = datetime(2024, 6, 1, tzinfo=timezone.utc)
        path = write_history(tmp_path, [valid_entry(time_usec=1_704_067_200_000_000)])
        signals = await GoogleTakeoutCollector(path, since=since).collect()
        assert signals == []

    async def test_entry_exactly_at_since_included(self, tmp_path: Path) -> None:
        # ts == since → included (not strictly less than)
        since = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        path = write_history(tmp_path, [valid_entry(time_usec=1_704_067_200_000_000)])
        signals = await GoogleTakeoutCollector(path, since=since).collect()
        assert len(signals) == 1

    async def test_since_none_includes_all_entries(self, tmp_path: Path) -> None:
        entries = [
            valid_entry(time_usec=1_000_000_000_000_000),  # very old
            valid_entry(title="Second page", url="https://example.com/second", time_usec=1_704_067_200_000_000),
        ]
        path = write_history(tmp_path, entries)
        signals = await GoogleTakeoutCollector(path, since=None).collect()
        assert len(signals) == 2

    async def test_since_filters_some_but_not_all(self, tmp_path: Path) -> None:
        since = datetime(2024, 1, 2, tzinfo=timezone.utc)  # 2024-01-02 00:00:00
        entries = [
            valid_entry(time_usec=1_704_067_200_000_000),  # 2024-01-01 → excluded
            valid_entry(title="Newer page", url="https://example.com/newer", time_usec=1_704_153_600_000_000),  # 2024-01-02 → included
        ]
        path = write_history(tmp_path, entries)
        signals = await GoogleTakeoutCollector(path, since=since).collect()
        assert len(signals) == 1
        assert signals[0].content == "Newer page"


# ── Error cases ───────────────────────────────────────────────────────────────

class TestErrorCases:
    async def test_malformed_json_raises_signal_collection_error(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "BrowserHistory.json"
        path.write_text("{not valid json}", encoding="utf-8")
        with pytest.raises(SignalCollectionError) as exc_info:
            await GoogleTakeoutCollector(path).collect()
        assert exc_info.value.source == SignalSource.GOOGLE_TAKEOUT
        assert "Failed to parse" in str(exc_info.value)

    async def test_wrong_top_level_structure_raises_signal_collection_error(
        self, tmp_path: Path
    ) -> None:
        # Browser History must be a list, not a dict
        path = tmp_path / "BrowserHistory.json"
        path.write_text(
            json.dumps({"Browser History": {"key": "not a list"}}),
            encoding="utf-8",
        )
        with pytest.raises(SignalCollectionError) as exc_info:
            await GoogleTakeoutCollector(path).collect()
        assert "must be a list" in str(exc_info.value)

    async def test_missing_browser_history_key_returns_empty(
        self, tmp_path: Path
    ) -> None:
        # data.get("Browser History", []) → empty list → no signals
        path = tmp_path / "BrowserHistory.json"
        path.write_text(json.dumps({"Other Key": []}), encoding="utf-8")
        signals = await GoogleTakeoutCollector(path).collect()
        assert signals == []

    async def test_non_dict_entries_skipped_gracefully(
        self, tmp_path: Path
    ) -> None:
        # _parse_entry receives a non-dict; dict.get() on non-dict would fail
        # but collector handles this via .get() which returns "" or None
        path = tmp_path / "BrowserHistory.json"
        path.write_text(
            json.dumps({"Browser History": [valid_entry(), "not a dict", None]}),
            encoding="utf-8",
        )
        # Should not raise; non-dict entries yield None from _parse_entry
        signals = await GoogleTakeoutCollector(path).collect()
        assert len(signals) == 1


# ── File size guard ───────────────────────────────────────────────────────────

class TestFileSizeGuard:
    async def test_file_too_large_raises_signal_collection_error(
        self, tmp_path: Path
    ) -> None:
        from unittest.mock import MagicMock, patch

        path = tmp_path / "BrowserHistory.json"
        path.write_text('{"Browser History": []}', encoding="utf-8")
        collector = GoogleTakeoutCollector(path)
        mock_stat = MagicMock()
        mock_stat.st_size = 200 * 1024 * 1024  # 200 MB
        with patch("pathlib.Path.stat", return_value=mock_stat):
            with pytest.raises(SignalCollectionError, match="too large"):
                await collector.collect()

    async def test_file_at_limit_is_accepted(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        path = tmp_path / "BrowserHistory.json"
        path.write_text('{"Browser History": []}', encoding="utf-8")
        collector = GoogleTakeoutCollector(path)
        mock_stat = MagicMock()
        mock_stat.st_size = 100 * 1024 * 1024  # exactly 100 MB
        with patch("pathlib.Path.stat", return_value=mock_stat):
            # Should not raise — limit is strictly greater than 100 MB
            signals = await collector.collect()
        assert signals == []

    async def test_file_under_limit_proceeds_normally(self, tmp_path: Path) -> None:
        path = tmp_path / "BrowserHistory.json"
        path.write_text(
            json.dumps({"Browser History": [valid_entry()]}), encoding="utf-8"
        )
        collector = GoogleTakeoutCollector(path)
        signals = await collector.collect()
        assert len(signals) == 1
