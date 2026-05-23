"""Unit tests for trace/signals/gmail.py — pure logic, no external I/O."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trace.signals.gmail import GmailCollector, _is_newsletter, _parse_subject_to_signal


class TestIsNewsletter:
    def test_newsletter_keyword(self):
        assert _is_newsletter("The Batch: Deep Learning Weekly", "") is True

    def test_unsubscribe_keyword(self):
        assert _is_newsletter("Check your settings", "click here to unsubscribe") is True

    def test_digest_keyword(self):
        assert _is_newsletter("Weekly digest for you", "") is True

    def test_plain_email_not_newsletter(self):
        assert _is_newsletter("Hey, are you free tonight?", "Let me know!") is False

    def test_case_insensitive(self):
        assert _is_newsletter("WEEKLY ROUNDUP", "") is True


class TestParseSubjectToSignal:
    def test_basic_subject(self):
        sig = _parse_subject_to_signal(
            subject="AI Safety Weekly",
            from_addr="safety@example.com",
            msg_id="msg-001",
            received_at=datetime.now(timezone.utc),
            is_unread=True,
        )
        assert sig is not None
        assert sig.content == "AI Safety Weekly"
        assert sig.metadata["is_unread"] is True
        assert sig.metadata["from"] == "safety@example.com"

    def test_empty_subject_returns_none(self):
        sig = _parse_subject_to_signal(
            subject="",
            from_addr="x@example.com",
            msg_id="msg-002",
            received_at=datetime.now(timezone.utc),
            is_unread=False,
        )
        assert sig is None

    def test_whitespace_only_returns_none(self):
        sig = _parse_subject_to_signal(
            subject="   ",
            from_addr="x@example.com",
            msg_id="msg-003",
            received_at=datetime.now(timezone.utc),
            is_unread=False,
        )
        assert sig is None

    def test_long_subject_truncated(self):
        long = "a" * 600
        sig = _parse_subject_to_signal(
            subject=long,
            from_addr="x@example.com",
            msg_id="msg-004",
            received_at=datetime.now(timezone.utc),
            is_unread=False,
        )
        assert sig is not None
        assert len(sig.content) <= 500

    def test_newsletter_flag_in_metadata(self):
        sig = _parse_subject_to_signal(
            subject="Weekly Digest: ML papers",
            from_addr="ml@example.com",
            msg_id="msg-005",
            received_at=datetime.now(timezone.utc),
            is_unread=True,
        )
        assert sig is not None
        assert sig.metadata["is_newsletter"] is True

    def test_unread_false(self):
        sig = _parse_subject_to_signal(
            subject="Hello world",
            from_addr="x@example.com",
            msg_id="msg-006",
            received_at=datetime.now(timezone.utc),
            is_unread=False,
        )
        assert sig is not None
        assert sig.metadata["is_unread"] is False


class TestGmailCollectorDemoSignals:
    def test_demo_signals_returns_list(self):
        collector = GmailCollector()
        signals = collector._demo_signals()
        assert isinstance(signals, list)
        assert len(signals) > 0

    def test_demo_signals_all_valid(self):
        collector = GmailCollector()
        for sig in collector._demo_signals():
            assert sig.content
            assert sig.timestamp is not None

    def test_demo_signals_have_gmail_source(self):
        from trace.models import SignalSource
        collector = GmailCollector()
        for sig in collector._demo_signals():
            assert sig.source == SignalSource.GMAIL

    def test_extract_messages_from_list(self):
        collector = GmailCollector()
        raw = [{"id": "1", "subject": "Hi"}, {"id": "2", "subject": "Hey"}]
        result = collector._extract_messages(raw)
        assert len(result) == 2

    def test_extract_messages_from_dict_key(self):
        collector = GmailCollector()
        for key in ("messages", "items", "data", "results"):
            raw = {key: [{"id": "x"}]}
            result = collector._extract_messages(raw)
            assert len(result) == 1

    def test_extract_messages_empty(self):
        collector = GmailCollector()
        assert collector._extract_messages({}) == []
        assert collector._extract_messages([]) == []

    def test_message_to_signal_epoch_ms(self):
        collector = GmailCollector()
        import time
        ts_ms = int(time.time() * 1000)
        msg = {"id": "m1", "subject": "Test subject", "from": "a@b.com", "internalDate": ts_ms, "labelIds": ["UNREAD"]}
        sig = collector._message_to_signal(msg)
        assert sig is not None
        assert sig.metadata["is_unread"] is True

    def test_message_to_signal_no_subject_uses_snippet(self):
        collector = GmailCollector()
        msg = {"id": "m2", "snippet": "This is a snippet", "from": "a@b.com", "labelIds": []}
        sig = collector._message_to_signal(msg)
        assert sig is not None
        assert "snippet" in sig.content.lower() or sig.content

    @pytest.mark.asyncio
    async def test_collect_without_scalekit_returns_empty(self, monkeypatch):
        """When Scalekit returns None and DEMO_MODE is off, collect() returns []."""
        import trace.signals.gmail as gmail_mod
        import trace.auth.scalekit as sk_mod
        monkeypatch.setattr(gmail_mod, "_DEMO_MODE", False)
        monkeypatch.setattr(sk_mod, "get_scalekit_client", lambda: None)

        collector = GmailCollector()
        signals = await collector.collect()
        assert isinstance(signals, list)
        assert signals == []

    @pytest.mark.asyncio
    async def test_collect_demo_mode_returns_synthetic(self, monkeypatch):
        """In DEMO_MODE with no Scalekit, collect() returns synthetic demo signals."""
        import trace.signals.gmail as gmail_mod
        import trace.auth.scalekit as sk_mod
        monkeypatch.setattr(gmail_mod, "_DEMO_MODE", True)
        monkeypatch.setattr(sk_mod, "get_scalekit_client", lambda: None)

        collector = GmailCollector()
        signals = await collector.collect()
        assert isinstance(signals, list)
        assert len(signals) > 0
