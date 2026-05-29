"""
GmailCollector — reads Gmail signals via direct Google OAuth.

Uses GMAIL_REFRESH_TOKEN + GMAIL_CLIENT_ID + GMAIL_CLIENT_SECRET from .env.
No Scalekit required. Token is refreshed automatically on each collect() call.

Get a refresh token once via OAuth Playground:
  developers.google.com/oauthplayground
  → Use your own OAuth credentials → authorize gmail.readonly → exchange code
  → copy Refresh token → add to .env as GMAIL_REFRESH_TOKEN

SIGNALS EXTRACTED:
  - Subject lines of newsletter emails (subscription-debt proxy)
  - Unread newsletters get higher weight (debt_score contribution)

DEMO_MODE:
  When TRACE_DEMO_MODE=true and no token is configured, returns synthetic
  signals so the scheduler can demonstrate the loop without real credentials.
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

from trace.models import RawSignal, SignalSource

_log = logging.getLogger(__name__)
_DEMO_MODE = os.getenv("TRACE_DEMO_MODE", "false").lower() in ("true", "1", "yes")

_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"

_NEWSLETTER_KEYWORDS = [
    "unsubscribe", "newsletter", "weekly digest", "roundup", "dispatch",
    "briefing", "update", "edition", "issue #", "vol.", "weekly",
]


def _is_newsletter(subject: str, snippet: str) -> bool:
    lower = (subject + " " + snippet).lower()
    return any(kw in lower for kw in _NEWSLETTER_KEYWORDS)


def _parse_subject_to_signal(
    subject: str,
    from_addr: str,
    msg_id: str,
    received_at: datetime,
    is_unread: bool,
) -> RawSignal | None:
    content = subject.strip()
    if not content:
        return None
    try:
        return RawSignal(
            id=str(uuid.uuid4()),
            source=SignalSource.GMAIL,
            content=content[:500],
            timestamp=received_at,
            metadata={
                "gmail_message_id": msg_id,
                "from": from_addr,
                "is_unread": is_unread,
                "is_newsletter": _is_newsletter(subject, ""),
            },
        )
    except Exception as exc:
        _log.debug("Could not create RawSignal for msg %s: %s", msg_id, exc)
        return None


async def _get_access_token(client_id: str, client_secret: str, refresh_token: str) -> str:
    """Exchange refresh token for a fresh access token."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(_TOKEN_URL, data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        })
        r.raise_for_status()
        return r.json()["access_token"]


class GmailCollector:
    """Collects Gmail signals via direct Google OAuth refresh token.

    Falls back to synthetic demo signals when credentials are not configured
    and DEMO_MODE is enabled.
    """

    def __init__(self, max_messages: int = 50) -> None:
        from trace.config import get_settings
        s = get_settings()
        self._refresh_token = s.gmail_refresh_token
        self._client_id = s.gmail_client_id
        self._client_secret = s.gmail_client_secret
        self._max_messages = max_messages

    @property
    def _configured(self) -> bool:
        return bool(self._refresh_token and self._client_id and self._client_secret)

    async def collect(self) -> list[RawSignal]:
        if not self._configured:
            if _DEMO_MODE:
                return self._demo_signals()
            _log.info("GmailCollector: credentials not configured — skipping")
            return []
        try:
            return await self._collect_direct()
        except Exception as exc:
            _log.warning("GmailCollector failed: %s", exc)
            if _DEMO_MODE:
                return self._demo_signals()
            return []

    async def _collect_direct(self) -> list[RawSignal]:
        access_token = await _get_access_token(
            self._client_id, self._client_secret, self._refresh_token
        )
        headers = {"Authorization": f"Bearer {access_token}"}

        async with httpx.AsyncClient(timeout=20) as c:
            # List recent inbox messages
            r = await c.get(
                f"{_GMAIL_API}/messages",
                headers=headers,
                params={"maxResults": self._max_messages, "labelIds": "INBOX", "q": "newer_than:7d"},
            )
            r.raise_for_status()
            message_stubs = r.json().get("messages", [])

        if not message_stubs:
            _log.info("GmailCollector: no messages in inbox (last 7 days)")
            return []

        signals: list[RawSignal] = []
        async with httpx.AsyncClient(timeout=20) as c:
            for stub in message_stubs[:self._max_messages]:
                msg_id = stub.get("id", "")
                if not msg_id:
                    continue
                try:
                    r = await c.get(
                        f"{_GMAIL_API}/messages/{msg_id}",
                        headers=headers,
                        params={"format": "metadata", "metadataHeaders": ["Subject", "From", "Date"]},
                    )
                    r.raise_for_status()
                    signal = self._parse_message(r.json())
                    if signal:
                        signals.append(signal)
                except Exception as exc:
                    _log.debug("GmailCollector: skipping msg %s: %s", msg_id, exc)

        _log.info("GmailCollector: extracted %d signals from %d messages", len(signals), len(message_stubs))
        return signals

    def _parse_message(self, msg: dict[str, Any]) -> RawSignal | None:
        msg_id = msg.get("id", "")
        label_ids = msg.get("labelIds", [])
        is_unread = "UNREAD" in label_ids

        headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
        subject = headers.get("subject", "")
        from_addr = headers.get("from", "")

        raw_ts = msg.get("internalDate")
        try:
            received_at = datetime.fromtimestamp(int(raw_ts) / 1000, tz=timezone.utc) if raw_ts else datetime.now(timezone.utc)
        except Exception:
            received_at = datetime.now(timezone.utc)

        return _parse_subject_to_signal(subject, from_addr, msg_id, received_at, is_unread)

    def _demo_signals(self) -> list[RawSignal]:
        demo_subjects = [
            ("The Batch: Deep Learning Weekly", "deeplearning_ai@newsletter.ai"),
            ("Diffusion Policy: new IROS paper roundup", "robotics_digest@example.com"),
            ("Rust in Production — issue #47", "rustlang_weekly@example.com"),
            ("nuPlan dataset update: new closed-loop metrics", "waymo_research@example.com"),
            ("Andrej Karpathy: AI education weekly briefing", "karpathy_newsletter@example.com"),
        ]
        signals = []
        for i, (subject, from_addr) in enumerate(demo_subjects):
            signal = _parse_subject_to_signal(
                subject=subject, from_addr=from_addr,
                msg_id=f"demo-msg-{i:03d}",
                received_at=datetime.now(timezone.utc),
                is_unread=(i % 2 == 0),
            )
            if signal:
                signals.append(signal)
        _log.info("GmailCollector (DEMO): returning %d synthetic signals", len(signals))
        return signals
