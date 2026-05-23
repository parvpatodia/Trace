"""
GoogleTakeoutCollector — parses BrowserHistory.json from a Google Takeout export.

Google Takeout format (Takeout/Chrome/BrowserHistory.json):
{
  "Browser History": [
    {
      "title": "Page Title",
      "url": "https://example.com",
      "time_usec": 1704067200000000,   <- microseconds since Unix epoch
      "page_transition": "LINK",
      "favicon_url": "https://..."
    }
  ]
}

Signal extraction:
  content   = Google search query (for search URLs) OR page title
  url       = page URL (http/https only; internal chrome:// URLs are None)
  timestamp = UTC datetime from time_usec
  metadata  = {time_usec, page_transition, is_search_query}

Google search query extraction:
  URLs matching *.google.*/search?q=... are the strongest curiosity signals in
  Chrome history — they represent explicit intent, not passive browsing. The q=
  parameter is extracted via urllib.parse.parse_qs and used as content instead
  of the page title ("Google Search" or similar uninformative strings).

  Example: https://www.google.com/search?q=diffusion+models+tutorial
    → content = "diffusion models tutorial"  (not "diffusion models tutorial - Google Search")

  This matters because the page title often just appends " - Google Search",
  while the raw query is cleaner and more semantically precise for topic extraction.

Noise filtering removes:
  - Chrome-internal URLs: chrome://, chrome-extension://, edge://, about:, data:, file://
  - Entries with missing url or time_usec
  - For search URLs: entries with no q= parameter or query shorter than _MIN_TITLE_LENGTH
  - For non-search URLs: entries with missing/short title or title identical to URL

WHY time_usec / 1_000_000 (not / 1000):
  Chrome stores timestamps in microseconds (1e-6 s), not milliseconds (1e-3 s).
  datetime.fromtimestamp() expects seconds. Division by 1_000_000.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from trace.models import RawSignal, SignalSource
from trace.signals.base import SignalCollectionError, SignalCollector

_log = logging.getLogger(__name__)
_MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB


class GoogleTakeoutCollector(SignalCollector):
    """
    Collects browse history signals from a Google Takeout BrowserHistory.json.

    Obtain via: Google Account → Data & Privacy → Download your data
    → Select "Chrome" → Download. Extract BrowserHistory.json from the zip.

    Parameters:
        history_path: path to BrowserHistory.json.
        since: if provided, signals with timestamp < since are excluded.
               Must be UTC-aware. Default: no time limit.
    """

    source = SignalSource.GOOGLE_TAKEOUT

    _SKIP_URL_PREFIXES: frozenset[str] = frozenset({
        "chrome://",
        "chrome-extension://",
        "edge://",
        "about:",
        "data:",
        "file://",
    })

    _MIN_TITLE_LENGTH: int = 3

    def __init__(
        self,
        history_path: Path,
        since: datetime | None = None,
    ) -> None:
        if since is not None and since.tzinfo is None:
            raise ValueError(
                "GoogleTakeoutCollector.since must be timezone-aware. "
                "Use datetime(..., tzinfo=timezone.utc)."
            )
        self._path = history_path
        self._since = since

    async def collect(self) -> list[RawSignal]:
        if not self._path.exists():
            raise SignalCollectionError(
                self.source,
                f"BrowserHistory.json not found: {self._path}",
            )
        size = self._path.stat().st_size
        if size > _MAX_FILE_BYTES:
            raise SignalCollectionError(
                self.source,
                f"BrowserHistory.json is too large ({size / 1024 / 1024:.1f} MB). "
                f"Maximum supported size is 100 MB. "
                f"Export a smaller date range from Google Takeout.",
            )
        _log.info("Loading BrowserHistory.json (%.1f MB)", size / 1024 / 1024)
        try:
            raw = await asyncio.to_thread(self._path.read_text, encoding="utf-8")
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as e:
            raise SignalCollectionError(
                self.source,
                f"Failed to parse {self._path.name}: {e}",
            ) from e

        entries = data.get("Browser History", [])
        if not isinstance(entries, list):
            raise SignalCollectionError(
                self.source,
                f"'Browser History' must be a list, got {type(entries).__name__}",
            )

        signals: list[RawSignal] = []
        skipped = 0
        for entry in entries:
            try:
                signal = self._parse_entry(entry)
                if signal is not None:
                    signals.append(signal)
            except Exception:
                skipped += 1
        if skipped:
            _log.debug("Skipped %d malformed/oversized entries", skipped)
        _log.info("Collected %d signal(s) from BrowserHistory.json (%d raw entries)", len(signals), len(entries))
        return signals

    def _parse_entry(self, entry: dict[str, Any]) -> RawSignal | None:
        if not isinstance(entry, dict):
            return None
        url: str = entry.get("url", "")
        title: str = entry.get("title", "").strip()
        time_usec: int | None = entry.get("time_usec")

        if not url or time_usec is None:
            return None

        if self._should_skip_url(url):
            return None

        ts = datetime.fromtimestamp(time_usec / 1_000_000, tz=timezone.utc)
        if self._since is not None and ts < self._since:
            return None

        safe_url: str | None = (
            url if (url.startswith("http://") or url.startswith("https://")) else None
        )

        # Google search: extract the raw query — strongest explicit curiosity signal.
        # "google.com/search" covers google.com, www.google.com, google.co.uk, etc.
        is_search_query = False
        content: str = ""
        if "google." in url and "/search" in url:
            try:
                qs = parse_qs(urlparse(url).query)
                q_parts = qs.get("q") or []
                query = q_parts[0].strip() if q_parts else ""
                if len(query) >= self._MIN_TITLE_LENGTH:
                    content = query[:2000]
                    is_search_query = True
            except Exception:
                pass

        if not is_search_query:
            if not title or len(title) < self._MIN_TITLE_LENGTH or title == url:
                return None
            content = title[:2000]

        return RawSignal(
            source=self.source,
            content=content,
            url=safe_url,
            timestamp=ts,
            metadata={
                "time_usec": time_usec,
                "page_transition": entry.get("page_transition", ""),
                "is_search_query": is_search_query,
            },
        )

    def _should_skip_url(self, url: str) -> bool:
        return any(url.startswith(prefix) for prefix in self._SKIP_URL_PREFIXES)
