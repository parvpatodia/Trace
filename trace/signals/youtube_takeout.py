"""
YouTubeWatchHistoryCollector — parses watch-history.json from a Google Takeout export.

This is a SEPARATE file from BrowserHistory.json. To obtain it:
  Google Account → Data & Privacy → Download your data
  → Deselect all → Select "YouTube and YouTube Music"
  → Export → Download ZIP
  → Extract: Takeout/YouTube and YouTube Music/history/watch-history.json

Google Takeout YouTube watch-history.json format:
[
  {
    "header": "YouTube",
    "title": "Watched Attention Is All You Need (Stanford CS224N)",
    "titleUrl": "https://www.youtube.com/watch?v=rBCqOTEfxvg",
    "subtitles": [
      {"name": "Stanford Online", "url": "https://www.youtube.com/channel/UCBa5G_ESCn8Yd4vw5U-gIcg"}
    ],
    "time": "2024-01-15T14:22:00.000Z",
    "details": [],
    "products": ["YouTube"]
  }
]

Signal extraction:
  content   = video title (stripped of "Watched " prefix)
  url       = YouTube video URL (titleUrl)
  timestamp = UTC datetime from "time" field
  metadata  = {video_id, channel_name, rewatch_count, is_stuck}

Rewatch detection:
  The file is an append log — each view event is a separate entry.
  If the same video URL appears N times, the user opened it N times.
  rewatch_count ≥ 3 within a 14-day window → "stuck" signal.
  This is the closest we can get to "stuck at 20 minutes" without
  a Chrome extension reading video.currentTime from the DOM.

WHY rewatch matters:
  A single visit = curiosity. Repeated visits to the same video = either
  genuine deep interest or a comprehension block. Both are more actionable
  than a one-off browse. Claude's topic extractor will see higher frequency
  for this topic, which raises its composite_score and debt_score.

WHY NOT parse watch position / currentTime from Takeout:
  Google Takeout does not include watch position data. The "time" field
  records when the video was opened, not how far the user got. Actual
  watch position is only accessible via a Chrome extension content script
  reading document.querySelector('video').currentTime from the DOM
  (see ROADMAP: Chrome Extension for Trace).

WHY "Watched " prefix is stripped:
  The title field in YouTube Takeout always starts with "Watched " —
  this prefix is a display artifact, not meaningful content for the
  topic extractor. "Watched Diffusion Models" adds noise; "Diffusion Models"
  is the signal.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trace.models import RawSignal, SignalSource
from trace.signals.base import SignalCollectionError, SignalCollector

_log = logging.getLogger(__name__)
_MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB
_MIN_TITLE_LENGTH: int = 5
_REWATCH_STUCK_THRESHOLD: int = 3  # ≥ 3 views of same video = "stuck" signal
_YOUTUBE_VIDEO_RE = re.compile(
    r"https?://(?:www\.)?youtube\.com/watch\?v=([A-Za-z0-9_-]{11})"
)


class YouTubeWatchHistoryCollector(SignalCollector):
    """
    Collects curiosity signals from a YouTube Takeout watch-history.json.

    Obtain via: Google Takeout → YouTube and YouTube Music
    → extract ZIP → Takeout/YouTube and YouTube Music/history/watch-history.json

    Parameters:
        history_path: path to watch-history.json.
        since: if provided, entries with timestamp < since are excluded.
               Must be UTC-aware.
        stuck_threshold: number of rewatches before a video is marked "stuck"
                         (default: 3). Increases signal weight for that topic.
    """

    source = SignalSource.YOUTUBE_TAKEOUT

    def __init__(
        self,
        history_path: Path,
        since: datetime | None = None,
        stuck_threshold: int = _REWATCH_STUCK_THRESHOLD,
    ) -> None:
        if since is not None and since.tzinfo is None:
            raise ValueError("YouTubeWatchHistoryCollector: since must be UTC-aware.")
        self._path = history_path
        self._since = since
        self._stuck_threshold = stuck_threshold

    async def collect(self) -> list[RawSignal]:
        if not self._path.exists():
            raise SignalCollectionError(
                self.source,
                f"watch-history.json not found: {self._path}",
            )

        size = self._path.stat().st_size
        if size > _MAX_FILE_BYTES:
            raise SignalCollectionError(
                self.source,
                f"watch-history.json is too large ({size / 1024 / 1024:.1f} MB). "
                f"Maximum supported size is 100 MB.",
            )
        _log.info("Loading watch-history.json (%.1f MB)", size / 1024 / 1024)

        try:
            raw = await asyncio.to_thread(self._path.read_text, encoding="utf-8")
        except OSError as exc:
            raise SignalCollectionError(self.source, f"Cannot read file: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SignalCollectionError(
                self.source, f"Failed to parse watch-history.json: {exc}"
            ) from exc

        if not isinstance(data, list):
            raise SignalCollectionError(
                self.source,
                "watch-history.json must be a JSON array (list of watch events).",
            )

        signals = await asyncio.to_thread(self._parse_entries, data)
        _log.info(
            "YouTubeWatchHistory: %d signal(s) from %d entries", len(signals), len(data)
        )
        return signals

    # ── Synchronous parsing (called via asyncio.to_thread) ────────────────────

    def _parse_entries(self, entries: list[Any]) -> list[RawSignal]:
        # First pass: count views per video URL (rewatch detection)
        url_counts: Counter[str] = Counter()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            url = (entry.get("titleUrl") or "").strip()
            if url and _YOUTUBE_VIDEO_RE.match(url):
                url_counts[url] += 1

        # Second pass: deduplicate — emit one signal per unique video,
        # using the MOST RECENT view event's timestamp and enriching metadata
        # with rewatch count.
        seen_urls: set[str] = set()
        signals: list[RawSignal] = []
        skipped = 0

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                signal = self._parse_entry(entry, url_counts, seen_urls)
                if signal is not None:
                    signals.append(signal)
            except Exception:
                skipped += 1
        if skipped:
            _log.debug("Skipped %d malformed/oversized entries", skipped)

        return signals

    def _parse_entry(
        self,
        entry: dict[str, Any],
        url_counts: Counter[str],
        seen_urls: set[str],
    ) -> RawSignal | None:
        raw_title: str = (entry.get("title") or "").strip()
        url: str = (entry.get("titleUrl") or "").strip()
        time_str: str = (entry.get("time") or "").strip()

        # Only YouTube video watch events (not search or channel pages)
        if not url or not _YOUTUBE_VIDEO_RE.match(url):
            return None

        # Deduplicate: only emit signal for first occurrence in the list
        # (which is the most recent — Takeout is reverse-chronological)
        if url in seen_urls:
            return None
        seen_urls.add(url)

        # Strip "Watched " prefix that YouTube Takeout adds to all titles
        title = raw_title.removeprefix("Watched ")
        if len(title) < _MIN_TITLE_LENGTH:
            return None

        timestamp = _parse_time(time_str)
        if timestamp is None:
            return None
        if self._since is not None and timestamp < self._since:
            return None

        # Extract video ID and channel info
        video_id_match = _YOUTUBE_VIDEO_RE.match(url)
        video_id = video_id_match.group(1) if video_id_match else ""

        subtitles = entry.get("subtitles") or []
        channel_name = ""
        if subtitles and isinstance(subtitles[0], dict):
            channel_name = (subtitles[0].get("name") or "").strip()

        rewatch_count = url_counts[url]
        is_stuck = rewatch_count >= self._stuck_threshold

        # Enrich content with rewatch signal so topic extractor weights it higher
        content = title
        if channel_name:
            content = f"{title} — {channel_name}"
        if rewatch_count > 1:
            content = f"{content} (watched {rewatch_count}x)"

        return RawSignal(
            source=self.source,
            content=content[:2000],
            url=url,
            timestamp=timestamp,
            metadata={
                "video_id": video_id,
                "channel_name": channel_name,
                "rewatch_count": rewatch_count,
                "is_stuck": is_stuck,
                "raw_title": raw_title,
            },
        )


def _parse_time(value: str) -> datetime | None:
    """Parse ISO-8601 timestamp from YouTube Takeout (e.g. '2024-01-15T14:22:00.000Z')."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None
