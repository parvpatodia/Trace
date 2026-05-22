"""
Unit tests for trace/signals/youtube_takeout.py — YouTubeWatchHistoryCollector.

Test categories:
  1. Constructor validation: invalid since raises
  2. File not found: raises SignalCollectionError
  3. File too large: raises SignalCollectionError
  4. Malformed JSON: raises SignalCollectionError
  5. Non-array JSON: raises SignalCollectionError
  6. Happy path: valid watch-history.json → RawSignal list
  7. "Watched " prefix stripping
  8. Rewatch detection: same video 3+ times → is_stuck=True in metadata
  9. Rewatch content enrichment: content includes "(watched Nx)"
  10. Non-YouTube URLs excluded
  11. since filter: entries before since are excluded
  12. Deduplication: only one signal per unique video URL (most recent)
  13. File size guard: exactly 100MB is accepted
  14. Channel name in content
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trace.models import SignalSource
from trace.signals.base import SignalCollectionError
from trace.signals.youtube_takeout import YouTubeWatchHistoryCollector


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts(year: int, month: int, day: int, hour: int = 0) -> str:
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:00:00.000Z"


def make_entry(
    title: str = "Watched Diffusion Models from Scratch",
    url: str = "https://www.youtube.com/watch?v=abc1234EFGH",
    time: str = _ts(2024, 1, 10),
    channel: str = "Stanford Online",
    channel_url: str = "https://www.youtube.com/channel/UC12345",
) -> dict:
    return {
        "header": "YouTube",
        "title": title,
        "titleUrl": url,
        "subtitles": [{"name": channel, "url": channel_url}],
        "time": time,
        "details": [],
        "products": ["YouTube"],
    }


def write_history(tmp_path: Path, entries: list) -> Path:
    p = tmp_path / "watch-history.json"
    p.write_text(json.dumps(entries), encoding="utf-8")
    return p


# ── Constructor ───────────────────────────────────────────────────────────────

class TestConstructor:
    def test_naive_since_raises(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [])
        with pytest.raises(ValueError, match="UTC-aware"):
            YouTubeWatchHistoryCollector(
                history_path=path,
                since=datetime(2024, 1, 1),  # naive — no tzinfo
            )

    def test_aware_since_accepted(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [])
        c = YouTubeWatchHistoryCollector(
            history_path=path,
            since=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert c is not None


# ── File errors ───────────────────────────────────────────────────────────────

class TestFileErrors:
    async def test_missing_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "nonexistent.json"
        with pytest.raises(SignalCollectionError):
            await YouTubeWatchHistoryCollector(history_path=p).collect()

    async def test_file_too_large_raises(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [])
        mock_stat = MagicMock()
        mock_stat.st_size = 200 * 1024 * 1024
        with patch("pathlib.Path.stat", return_value=mock_stat):
            with pytest.raises(SignalCollectionError, match="too large"):
                await YouTubeWatchHistoryCollector(history_path=path).collect()

    async def test_file_at_limit_accepted(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [])
        mock_stat = MagicMock()
        mock_stat.st_size = 100 * 1024 * 1024  # exactly 100 MB — not over limit
        with patch("pathlib.Path.stat", return_value=mock_stat):
            signals = await YouTubeWatchHistoryCollector(history_path=path).collect()
        assert signals == []

    async def test_malformed_json_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "watch-history.json"
        p.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(SignalCollectionError, match="parse"):
            await YouTubeWatchHistoryCollector(history_path=p).collect()

    async def test_non_array_json_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "watch-history.json"
        p.write_text('{"entries": []}', encoding="utf-8")
        with pytest.raises(SignalCollectionError, match="array"):
            await YouTubeWatchHistoryCollector(history_path=p).collect()


# ── Happy path ────────────────────────────────────────────────────────────────

class TestHappyPath:
    async def test_returns_signals(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [make_entry()])
        signals = await YouTubeWatchHistoryCollector(history_path=path).collect()
        assert len(signals) == 1

    async def test_source_is_youtube_takeout(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [make_entry()])
        signals = await YouTubeWatchHistoryCollector(history_path=path).collect()
        assert signals[0].source == SignalSource.YOUTUBE_TAKEOUT

    async def test_url_is_set(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [make_entry()])
        signals = await YouTubeWatchHistoryCollector(history_path=path).collect()
        assert signals[0].url == "https://www.youtube.com/watch?v=abc1234EFGH"

    async def test_timestamp_is_utc_aware(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [make_entry()])
        signals = await YouTubeWatchHistoryCollector(history_path=path).collect()
        assert signals[0].timestamp.tzinfo is not None

    async def test_empty_array_returns_empty(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [])
        signals = await YouTubeWatchHistoryCollector(history_path=path).collect()
        assert signals == []


# ── Title processing ──────────────────────────────────────────────────────────

class TestTitleProcessing:
    async def test_watched_prefix_stripped(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [make_entry(title="Watched Diffusion Models")])
        signals = await YouTubeWatchHistoryCollector(history_path=path).collect()
        assert "Watched" not in signals[0].content

    async def test_title_without_prefix_kept(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [make_entry(title="Diffusion Models Explained")])
        signals = await YouTubeWatchHistoryCollector(history_path=path).collect()
        assert "Diffusion Models Explained" in signals[0].content

    async def test_very_short_title_skipped(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [make_entry(title="OK")])
        signals = await YouTubeWatchHistoryCollector(history_path=path).collect()
        assert signals == []

    async def test_channel_name_in_content(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [make_entry(channel="Andrej Karpathy")])
        signals = await YouTubeWatchHistoryCollector(history_path=path).collect()
        assert "Andrej Karpathy" in signals[0].content


# ── Rewatch detection ─────────────────────────────────────────────────────────

class TestRewatchDetection:
    async def test_single_view_not_stuck(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [make_entry()])
        signals = await YouTubeWatchHistoryCollector(history_path=path).collect()
        assert signals[0].metadata["rewatch_count"] == 1
        assert signals[0].metadata["is_stuck"] is False

    async def test_three_views_is_stuck(self, tmp_path: Path) -> None:
        entries = [
            make_entry(time=_ts(2024, 1, 10)),
            make_entry(time=_ts(2024, 1, 12)),
            make_entry(time=_ts(2024, 1, 14)),
        ]
        path = write_history(tmp_path, entries)
        signals = await YouTubeWatchHistoryCollector(history_path=path).collect()
        assert signals[0].metadata["rewatch_count"] == 3
        assert signals[0].metadata["is_stuck"] is True

    async def test_stuck_content_includes_rewatch_count(self, tmp_path: Path) -> None:
        entries = [make_entry(time=_ts(2024, 1, d)) for d in [10, 12, 14]]
        path = write_history(tmp_path, entries)
        signals = await YouTubeWatchHistoryCollector(history_path=path).collect()
        assert "3x" in signals[0].content

    async def test_two_views_not_stuck_by_default(self, tmp_path: Path) -> None:
        entries = [make_entry(time=_ts(2024, 1, 10)), make_entry(time=_ts(2024, 1, 11))]
        path = write_history(tmp_path, entries)
        signals = await YouTubeWatchHistoryCollector(history_path=path).collect()
        assert signals[0].metadata["is_stuck"] is False

    async def test_custom_stuck_threshold(self, tmp_path: Path) -> None:
        entries = [make_entry(time=_ts(2024, 1, d)) for d in [10, 11]]
        path = write_history(tmp_path, entries)
        # threshold=2 means 2 views = stuck
        signals = await YouTubeWatchHistoryCollector(
            history_path=path, stuck_threshold=2
        ).collect()
        assert signals[0].metadata["is_stuck"] is True


# ── URL filtering ─────────────────────────────────────────────────────────────

class TestURLFiltering:
    async def test_non_youtube_urls_excluded(self, tmp_path: Path) -> None:
        entries = [
            make_entry(url="https://www.youtube.com/channel/UC12345"),  # channel page, not video
            make_entry(url="https://youtu.be/abc1234EFGH"),  # short URL — not matching RE
            make_entry(),  # valid video
        ]
        path = write_history(tmp_path, entries)
        signals = await YouTubeWatchHistoryCollector(history_path=path).collect()
        # Only the valid /watch?v= URL should produce a signal
        assert len(signals) == 1

    async def test_empty_url_skipped(self, tmp_path: Path) -> None:
        entry = make_entry()
        entry["titleUrl"] = ""
        path = write_history(tmp_path, [entry])
        signals = await YouTubeWatchHistoryCollector(history_path=path).collect()
        assert signals == []

    async def test_non_dict_entries_skipped(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [make_entry(), "not a dict", None, 42])
        signals = await YouTubeWatchHistoryCollector(history_path=path).collect()
        assert len(signals) == 1


# ── Since filter ──────────────────────────────────────────────────────────────

class TestSinceFilter:
    async def test_entries_before_since_excluded(self, tmp_path: Path) -> None:
        since = datetime(2024, 1, 5, tzinfo=timezone.utc)
        entries = [
            make_entry(time=_ts(2024, 1, 3)),  # before — excluded
            make_entry(
                title="Watched Newer Video",
                url="https://www.youtube.com/watch?v=XYZ1234abcd",
                time=_ts(2024, 1, 7),  # after — included
            ),
        ]
        path = write_history(tmp_path, entries)
        signals = await YouTubeWatchHistoryCollector(history_path=path, since=since).collect()
        assert len(signals) == 1
        assert "Newer Video" in signals[0].content

    async def test_entries_on_since_boundary_included(self, tmp_path: Path) -> None:
        since = datetime(2024, 1, 5, tzinfo=timezone.utc)
        path = write_history(tmp_path, [make_entry(time=_ts(2024, 1, 5))])
        signals = await YouTubeWatchHistoryCollector(history_path=path, since=since).collect()
        assert len(signals) == 1


# ── Deduplication ─────────────────────────────────────────────────────────────

class TestDeduplication:
    async def test_same_video_twice_yields_one_signal(self, tmp_path: Path) -> None:
        entries = [
            make_entry(time=_ts(2024, 1, 10)),
            make_entry(time=_ts(2024, 1, 8)),  # older visit, same URL
        ]
        path = write_history(tmp_path, entries)
        signals = await YouTubeWatchHistoryCollector(history_path=path).collect()
        assert len(signals) == 1

    async def test_different_videos_yield_separate_signals(self, tmp_path: Path) -> None:
        entries = [
            make_entry(url="https://www.youtube.com/watch?v=aaaaaaBBBBBB"),
            make_entry(url="https://www.youtube.com/watch?v=ccccccDDDDDD"),
        ]
        path = write_history(tmp_path, entries)
        signals = await YouTubeWatchHistoryCollector(history_path=path).collect()
        assert len(signals) == 2

    async def test_video_id_in_metadata(self, tmp_path: Path) -> None:
        path = write_history(tmp_path, [make_entry(url="https://www.youtube.com/watch?v=abc1234EFGH")])
        signals = await YouTubeWatchHistoryCollector(history_path=path).collect()
        assert signals[0].metadata["video_id"] == "abc1234EFGH"
