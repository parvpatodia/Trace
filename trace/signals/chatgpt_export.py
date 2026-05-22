"""
ChatGPTExportCollector — parses conversations.json from a ChatGPT data export.

ChatGPT data export structure (Settings → Data Controls → Export Data):
[
  {
    "title": "Conversation Title",
    "create_time": 1700000000.0,
    "update_time": 1700000001.0,
    "mapping": {
      "<node-uuid>": {
        "message": {          <- null for system/root nodes
          "author": {"role": "user" | "assistant" | "system"},
          "content": {
            "content_type": "text",
            "parts": ["the message text as a string"]
          },
          "create_time": 1700000000.5   <- float, seconds since epoch
        }
      }
    }
  }
]

Signal extraction:
  Only user messages are extracted. Assistant responses are NOT signals.
  The question the user asked reveals their curiosity; the answer they
  received does not.

  content   = joined text parts, stripped, truncated to 2000 chars
  timestamp = UTC datetime from message.create_time
  metadata  = {conversation_title, conversation_create_time}

Noise filtering removes:
  - Non-text content_type (code blocks, images embedded in tool calls)
  - Messages shorter than MIN_CONTENT_LENGTH (10 chars)
    — eliminates "yes", "ok", "thanks", "go on", etc.
  - Messages with missing create_time
  - Messages where all parts are non-string (rare but possible in export edge cases)

WHY MIN_CONTENT_LENGTH = 10:
  Very short user turns ("Yes", "Continue", "Go on") are acknowledgements,
  not curiosity signals. 10 chars is the minimum for a meaningful question
  fragment. This is a tunable constant — expose in config post-hackathon.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trace.models import RawSignal, SignalSource
from trace.signals.base import SignalCollectionError, SignalCollector

_log = logging.getLogger(__name__)
_MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB

_MIN_CONTENT_LENGTH: int = 10


class ChatGPTExportCollector(SignalCollector):
    """
    Collects curiosity signals from a ChatGPT conversations.json export.

    Obtain via: ChatGPT → Settings → Data Controls → Export Data.
    The downloaded zip contains conversations.json.

    Parameters:
        export_path: path to conversations.json.
        since: if provided, messages with timestamp < since are excluded.
               Must be UTC-aware. Default: no time limit.
    """

    source = SignalSource.CHATGPT_EXPORT

    def __init__(
        self,
        export_path: Path,
        since: datetime | None = None,
    ) -> None:
        if since is not None and since.tzinfo is None:
            raise ValueError(
                "ChatGPTExportCollector.since must be timezone-aware. "
                "Use datetime(..., tzinfo=timezone.utc)."
            )
        self._path = export_path
        self._since = since

    async def collect(self) -> list[RawSignal]:
        if not self._path.exists():
            raise SignalCollectionError(
                self.source,
                f"conversations.json not found: {self._path}",
            )
        size = self._path.stat().st_size
        if size > _MAX_FILE_BYTES:
            raise SignalCollectionError(
                self.source,
                f"conversations.json is too large ({size / 1024 / 1024:.1f} MB). "
                f"Maximum supported size is 100 MB. "
                f"Export a smaller date range from ChatGPT.",
            )
        _log.info("Loading conversations.json (%.1f MB)", size / 1024 / 1024)
        try:
            raw = await asyncio.to_thread(self._path.read_text, encoding="utf-8")
            conversations: list[dict[str, Any]] = json.loads(raw)
        except json.JSONDecodeError as e:
            raise SignalCollectionError(
                self.source,
                f"Failed to parse {self._path.name}: {e}",
            ) from e

        if not isinstance(conversations, list):
            raise SignalCollectionError(
                self.source,
                f"conversations.json must be a JSON array, got {type(conversations).__name__}",
            )

        signals: list[RawSignal] = []
        for convo in conversations:
            if not isinstance(convo, dict):
                continue
            signals.extend(self._extract_from_conversation(convo))
        _log.info("Collected %d signal(s) from conversations.json (%d conversation(s))", len(signals), len(conversations))
        return signals

    def _extract_from_conversation(
        self, convo: dict[str, Any]
    ) -> list[RawSignal]:
        mapping: dict[str, Any] = convo.get("mapping") or {}
        title: str = convo.get("title") or ""
        convo_create_time: float | None = convo.get("create_time")

        signals: list[RawSignal] = []
        for node in mapping.values():
            if not isinstance(node, dict):
                continue
            message = node.get("message")
            if message is None or not isinstance(message, dict):
                continue
            if message.get("author", {}).get("role") != "user":
                continue
            signal = self._parse_message(message, title, convo_create_time)
            if signal is not None:
                signals.append(signal)
        return signals

    def _parse_message(
        self,
        message: dict[str, Any],
        conversation_title: str,
        convo_create_time: float | None,
    ) -> RawSignal | None:
        content_obj: dict[str, Any] = message.get("content") or {}

        # Only process text messages; skip code interpreter outputs, images, etc.
        if content_obj.get("content_type") != "text":
            return None

        parts: list[Any] = content_obj.get("parts") or []
        text = " ".join(p for p in parts if isinstance(p, str)).strip()

        if len(text) < _MIN_CONTENT_LENGTH:
            return None

        create_time: float | None = message.get("create_time")
        if create_time is None:
            return None

        ts = datetime.fromtimestamp(create_time, tz=timezone.utc)
        if self._since is not None and ts < self._since:
            return None

        return RawSignal(
            source=self.source,
            # Truncate to RawSignal.content max_length=2000
            content=text[:2000],
            timestamp=ts,
            metadata={
                "conversation_title": conversation_title,
                "conversation_create_time": convo_create_time,
            },
        )
