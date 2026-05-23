"""
GmailCollector — reads Gmail signals via Scalekit Token Vault.

WHY SCALEKIT INSTEAD OF DIRECT GMAIL OAUTH:
  The Gmail OAuth token (refresh token + access token) lives in Scalekit's
  encrypted Token Vault.  We never store it in env vars or application memory.
  Every call goes through connect_execute_tool(), which:
    1. Retrieves the token from the Vault securely.
    2. Refreshes it transparently when expired.
    3. Executes the Gmail API call on our behalf.
    4. Returns structured JSON results.

SIGNALS EXTRACTED:
  - Subscribed newsletter senders (From: header pattern matching)
  - Subject lines of unread/archived newsletters (curiosity debt proxy)
  - Topics from email subjects that match curiosity keywords

DEMO_MODE:
  When TRACE_DEMO_MODE=true and Scalekit is not configured, returns synthetic
  signals so the scheduler job can demonstrate the loop without real credentials.
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from trace.models import RawSignal, SignalSource

_log = logging.getLogger(__name__)
_DEMO_MODE = os.getenv("TRACE_DEMO_MODE", "false").lower() in ("true", "1", "yes")

# Tool name registered in Scalekit dashboard for the Gmail MCP connector.
_GMAIL_TOOL = "gmail_list_messages"
_GMAIL_LABELS_TOOL = "gmail_list_labels"

# Newsletter-sender heuristics — subjects from these senders are weighted as
# subscription-debt signals (user subscribed but may not be reading them).
_NEWSLETTER_KEYWORDS = [
    "unsubscribe", "newsletter", "weekly digest", "roundup", "dispatch",
    "briefing", "update", "edition", "issue #", "vol.", "weekly",
]


def _is_newsletter(subject: str, snippet: str) -> bool:
    """True if the email looks like a newsletter (not a personal email)."""
    lower = (subject + " " + snippet).lower()
    return any(kw in lower for kw in _NEWSLETTER_KEYWORDS)


def _parse_subject_to_signal(
    subject: str,
    from_addr: str,
    msg_id: str,
    received_at: datetime,
    is_unread: bool,
) -> RawSignal | None:
    """Convert a Gmail message header into a RawSignal.

    Unread newsletters get higher weight via metadata — CuriosityGraphBuilder
    uses this to compute subscription debt_score.
    """
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


class GmailCollector:
    """Collects Gmail signals via Scalekit Token Vault.

    Falls back to synthetic demo signals when Scalekit is not configured
    and DEMO_MODE is enabled.
    """

    def __init__(
        self,
        identifier: str | None = None,
        connection_name: str = "gmail",
        max_messages: int = 50,
    ) -> None:
        from trace.config import get_settings
        s = get_settings()
        self._identifier = identifier or s.scalekit_default_identifier
        self._connection_name = connection_name
        self._max_messages = max_messages

    async def collect(self) -> list[RawSignal]:
        """Collect recent newsletter/email signals from Gmail."""
        from trace.auth.scalekit import connect_execute_tool, get_scalekit_client

        if get_scalekit_client() is None:
            if _DEMO_MODE:
                return self._demo_signals()
            _log.info("GmailCollector: Scalekit not configured — skipping")
            return []

        try:
            return await self._collect_via_scalekit(connect_execute_tool)
        except Exception as exc:
            _log.warning("GmailCollector failed: %s", exc)
            if _DEMO_MODE:
                return self._demo_signals()
            return []

    async def _collect_via_scalekit(self, execute_tool: Any) -> list[RawSignal]:
        """Call Gmail via Scalekit Token Vault and parse the results."""
        # Fetch recent messages (label: INBOX, past 7 days).
        result = await execute_tool(
            tool_name=_GMAIL_TOOL,
            tool_input={
                "maxResults": self._max_messages,
                "labelIds": ["INBOX"],
                "q": "newer_than:7d",
            },
            identifier=self._identifier,
        )

        messages = self._extract_messages(result)
        if not messages:
            _log.info("GmailCollector: no messages returned from Scalekit")
            return []

        signals: list[RawSignal] = []
        for msg in messages:
            signal = self._message_to_signal(msg)
            if signal:
                signals.append(signal)

        _log.info("GmailCollector: extracted %d signals from %d messages", len(signals), len(messages))
        return signals

    def _extract_messages(self, result: Any) -> list[dict[str, Any]]:
        """Normalise varied response shapes from the Gmail MCP tool."""
        if isinstance(result, list):
            return [m for m in result if isinstance(m, dict)]
        if isinstance(result, dict):
            for key in ("messages", "items", "data", "results"):
                val = result.get(key)
                if isinstance(val, list):
                    return [m for m in val if isinstance(m, dict)]
        return []

    def _message_to_signal(self, msg: dict[str, Any]) -> RawSignal | None:
        """Convert a raw Gmail message dict to a RawSignal."""
        msg_id = msg.get("id", "")
        subject = msg.get("subject", "") or msg.get("snippet", "")[:100]
        from_addr = msg.get("from", "") or msg.get("sender", "")

        # Parse timestamp — Gmail returns epoch ms or ISO string.
        received_at: datetime
        raw_ts = msg.get("internalDate") or msg.get("date")
        try:
            if isinstance(raw_ts, (int, float)):
                received_at = datetime.fromtimestamp(raw_ts / 1000, tz=timezone.utc)
            elif isinstance(raw_ts, str) and raw_ts.isdigit():
                received_at = datetime.fromtimestamp(int(raw_ts) / 1000, tz=timezone.utc)
            else:
                received_at = datetime.now(timezone.utc)
        except Exception:
            received_at = datetime.now(timezone.utc)

        label_ids = msg.get("labelIds", [])
        is_unread = "UNREAD" in label_ids

        return _parse_subject_to_signal(subject, from_addr, msg_id, received_at, is_unread)

    def _demo_signals(self) -> list[RawSignal]:
        """Synthetic signals for demo mode — realistic newsletter subjects."""
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
                subject=subject,
                from_addr=from_addr,
                msg_id=f"demo-msg-{i:03d}",
                received_at=datetime.now(timezone.utc),
                is_unread=(i % 2 == 0),  # every other is unread
            )
            if signal:
                signals.append(signal)
        _log.info("GmailCollector (DEMO): returning %d synthetic signals", len(signals))
        return signals
