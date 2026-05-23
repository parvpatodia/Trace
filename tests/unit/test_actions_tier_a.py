"""
Unit tests for Tier A actions: notion.py, calendar.py, slack.py.

Tests cover the stub path (Scalekit not configured) and the error-parsing logic.
Real Scalekit calls are integration tests and require live credentials.
"""
from __future__ import annotations

import pytest


# ── Notion ─────────────────────────────────────────────────────────────────────

class TestNotionAction:
    @pytest.mark.asyncio
    async def test_stub_when_scalekit_not_configured(self, monkeypatch):
        import trace.auth.scalekit as sk_mod
        monkeypatch.setattr(sk_mod, "get_scalekit_client", lambda: None)

        from trace.actions.notion import create_topic_page
        result = await create_topic_page("diffusion policy", "briefing text")
        assert result["status"] == "stub"
        assert result["topic"] == "diffusion policy"
        assert "page_title" in result

    @pytest.mark.asyncio
    async def test_error_returned_on_execute_tool_failure(self, monkeypatch):
        import trace.auth.scalekit as sk_mod
        monkeypatch.setattr(sk_mod, "get_scalekit_client", lambda: object())

        async def _failing_tool(**_kwargs):
            raise RuntimeError("network error")

        monkeypatch.setattr(sk_mod, "connect_execute_tool", _failing_tool)

        from trace.actions.notion import create_topic_page
        result = await create_topic_page("robotics", "briefing")
        assert result["status"] == "error"
        assert "network error" in result["error"]

    @pytest.mark.asyncio
    async def test_error_dict_from_tool(self, monkeypatch):
        import trace.auth.scalekit as sk_mod
        monkeypatch.setattr(sk_mod, "get_scalekit_client", lambda: object())

        async def _error_tool(**_kwargs):
            return {"error": "something went wrong"}

        monkeypatch.setattr(sk_mod, "connect_execute_tool", _error_tool)

        from trace.actions.notion import create_topic_page
        result = await create_topic_page("ai", "briefing")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_success_path(self, monkeypatch):
        import trace.auth.scalekit as sk_mod
        monkeypatch.setattr(sk_mod, "get_scalekit_client", lambda: object())

        async def _success_tool(**_kwargs):
            return {"url": "https://notion.so/page123", "id": "page123"}

        monkeypatch.setattr(sk_mod, "connect_execute_tool", _success_tool)

        from trace.actions.notion import create_topic_page
        result = await create_topic_page("nuplan", "good briefing")
        assert result["status"] == "created"
        assert result["via_scalekit"] is True
        assert "page123" in result["page_url"]

    @pytest.mark.asyncio
    async def test_auth_required_response(self, monkeypatch):
        import trace.auth.scalekit as sk_mod
        monkeypatch.setattr(sk_mod, "get_scalekit_client", lambda: object())

        async def _auth_error(**_kwargs):
            return {"error": "not_authorized: please connect Notion"}

        async def _mock_auth_link(**_kwargs: object) -> str:
            return "https://auth.link"

        monkeypatch.setattr(sk_mod, "connect_execute_tool", _auth_error)
        monkeypatch.setattr(sk_mod, "connect_get_authorization_link", _mock_auth_link)

        from trace.actions.notion import create_topic_page
        result = await create_topic_page("foo", "bar")
        assert result["status"] == "auth_required"
        assert result["connection"] == "notion"


# ── Calendar ───────────────────────────────────────────────────────────────────

class TestCalendarAction:
    @pytest.mark.asyncio
    async def test_stub_when_scalekit_not_configured(self, monkeypatch):
        import trace.auth.scalekit as sk_mod
        monkeypatch.setattr(sk_mod, "get_scalekit_client", lambda: None)

        from trace.actions.calendar import schedule_deep_dive
        result = await schedule_deep_dive("rust programming", "briefing")
        assert result["status"] == "stub"
        assert "scheduled_for" in result

    @pytest.mark.asyncio
    async def test_success_path(self, monkeypatch):
        import trace.auth.scalekit as sk_mod
        monkeypatch.setattr(sk_mod, "get_scalekit_client", lambda: object())

        async def _success(**_kwargs):
            return {"id": "evt123", "htmlLink": "https://cal.google.com/evt123"}

        monkeypatch.setattr(sk_mod, "connect_execute_tool", _success)

        from trace.actions.calendar import schedule_deep_dive
        result = await schedule_deep_dive("diffusion models", "good briefing", profile_id="user1")
        assert result["status"] == "created"
        assert result["event_id"] == "evt123"
        assert result["via_scalekit"] is True

    @pytest.mark.asyncio
    async def test_error_path(self, monkeypatch):
        import trace.auth.scalekit as sk_mod
        monkeypatch.setattr(sk_mod, "get_scalekit_client", lambda: object())

        async def _fail(**_kwargs):
            raise ConnectionError("timeout")

        monkeypatch.setattr(sk_mod, "connect_execute_tool", _fail)

        from trace.actions.calendar import schedule_deep_dive
        result = await schedule_deep_dive("topic", "briefing")
        assert result["status"] == "error"

    def test_next_morning_slot_is_tomorrow(self):
        from trace.actions.calendar import _next_morning_slot
        from datetime import datetime, timezone
        start, end = _next_morning_slot()
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        now = datetime.now(timezone.utc)
        assert start_dt > now
        assert (end_dt - start_dt).seconds == 3600  # 60 minutes
        assert start_dt.hour == 9


# ── Slack ──────────────────────────────────────────────────────────────────────

class TestSlackAction:
    @pytest.mark.asyncio
    async def test_stub_when_scalekit_not_configured(self, monkeypatch):
        import trace.auth.scalekit as sk_mod
        monkeypatch.setattr(sk_mod, "get_scalekit_client", lambda: None)

        from trace.actions.slack import send_pattern_alert
        result = await send_pattern_alert("Hello from Trace")
        assert result["status"] == "stub"
        assert "message_preview" in result

    @pytest.mark.asyncio
    async def test_success_path(self, monkeypatch):
        import trace.auth.scalekit as sk_mod
        monkeypatch.setattr(sk_mod, "get_scalekit_client", lambda: object())

        async def _success(**_kwargs):
            return {"ok": True, "ts": "1234567890.123456"}

        monkeypatch.setattr(sk_mod, "connect_execute_tool", _success)

        from trace.actions.slack import send_pattern_alert
        result = await send_pattern_alert("🧠 Test message", profile_id="user1")
        assert result["status"] == "sent"
        assert result["via_scalekit"] is True
        assert result["ts"] == "1234567890.123456"

    @pytest.mark.asyncio
    async def test_channel_whitelist_enforced(self, monkeypatch):
        """Arbitrary channels are replaced with the demo channel."""
        import trace.auth.scalekit as sk_mod
        monkeypatch.setattr(sk_mod, "get_scalekit_client", lambda: object())

        captured: list[dict] = []

        async def _capture(**kwargs):
            captured.append(kwargs.get("tool_input", {}))
            return {"ok": True, "ts": "ts1"}

        monkeypatch.setattr(sk_mod, "connect_execute_tool", _capture)

        from trace.actions.slack import send_pattern_alert, _DEMO_CHANNEL
        await send_pattern_alert("test", channel="#malicious-channel")
        assert captured[0]["channel"] == _DEMO_CHANNEL

    @pytest.mark.asyncio
    async def test_error_path(self, monkeypatch):
        import trace.auth.scalekit as sk_mod
        monkeypatch.setattr(sk_mod, "get_scalekit_client", lambda: object())

        async def _fail(**_kwargs):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(sk_mod, "connect_execute_tool", _fail)

        from trace.actions.slack import send_pattern_alert
        result = await send_pattern_alert("test")
        assert result["status"] == "error"
