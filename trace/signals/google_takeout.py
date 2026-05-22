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
  content   = page title (sent to topic extractor; captures search intent)
  url       = page URL (http/https only; internal chrome:// URLs are None)
  timestamp = UTC datetime from time_usec
  metadata  = {time_usec, page_transition}

Noise filtering removes:
  - Chrome-internal URLs: chrome://, chrome-extension://, edge://, about:, data:, file://
  - Entries with missing title, url, or time_usec
  - Titles shorter than 3 characters (tab labels like "—", "•")
  - Titles identical to the URL (no real page title was set)

WHY PAGE TITLE AS content, NOT URL:
  The topic extractor (Claude) needs semantic text to cluster topics from.
  "Attention Is All You Need - arXiv" is a signal.
  "https://arxiv.org/abs/1706.03762" is not — it's opaque to the LLM
  without additional context. Title is always more semantically rich.

WHY time_usec / 1_000_000 (not / 1000):
  Chrome stores timestamps in microseconds (1e-6 s), not milliseconds (1e-3 s).
  datetime.fromtimestamp() expects seconds. Division by 1_000_000.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trace.models import RawSignal, SignalSource
from trace.signals.base import SignalCollectionError, SignalCollector


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
        if not history_path.exists():
            raise SignalCollectionError(
                self.source,
                f"BrowserHistory.json not found: {history_path}",
            )
        if since is not None and since.tzinfo is None:
            raise ValueError(
                "GoogleTakeoutCollector.since must be timezone-aware. "
                "Use datetime(..., tzinfo=timezone.utc)."
            )
        self._path = history_path
        self._since = since

    async def collect(self) -> list[RawSignal]:
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
        for entry in entries:
            signal = self._parse_entry(entry)
            if signal is not None:
                signals.append(signal)
        return signals

    def _parse_entry(self, entry: dict[str, Any]) -> RawSignal | None:
        if not isinstance(entry, dict):
            return None
        url: str = entry.get("url", "")
        title: str = entry.get("title", "").strip()
        time_usec: int | None = entry.get("time_usec")

        # Structural validation — skip incomplete entries silently
        if not url or not title or time_usec is None:
            return None

        # Skip browser-internal URLs that carry no curiosity signal
        if self._should_skip_url(url):
            return None

        # Skip uninformative titles
        if len(title) < self._MIN_TITLE_LENGTH or title == url:
            return None

        ts = datetime.fromtimestamp(time_usec / 1_000_000, tz=timezone.utc)

        if self._since is not None and ts < self._since:
            return None

        # Only include URL on the signal if it's a navigable http/https link.
        # chrome:// etc. are already excluded above, but other schemes (ftp://)
        # are technically possible — RawSignal.validate_url_scheme enforces http/https.
        safe_url: str | None = (
            url if (url.startswith("http://") or url.startswith("https://")) else None
        )

        return RawSignal(
            source=self.source,
            content=title,
            url=safe_url,
            timestamp=ts,
            metadata={
                "time_usec": time_usec,
                "page_transition": entry.get("page_transition", ""),
            },
        )

    def _should_skip_url(self, url: str) -> bool:
        return any(url.startswith(prefix) for prefix in self._SKIP_URL_PREFIXES)
