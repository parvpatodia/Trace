"""Unit tests for Tier B actions: gmail_draft.py and reddit_draft.py."""
from __future__ import annotations

import pytest


class TestGmailDraft:
    @pytest.mark.asyncio
    async def test_stub_when_scalekit_not_configured(self, monkeypatch):
        import trace.auth.scalekit as sk
        monkeypatch.setattr(sk, "get_scalekit_client", lambda: None)

        from trace.actions.gmail_draft import create_digest_draft
        result = await create_digest_draft("ai safety", "briefing here")
        assert result["status"] == "stub"
        assert "subject" in result
        assert "[Trace]" in result["subject"]

    @pytest.mark.asyncio
    async def test_success_creates_draft_not_send(self, monkeypatch):
        import trace.auth.scalekit as sk
        monkeypatch.setattr(sk, "get_scalekit_client", lambda: object())

        async def _success(**_kwargs):
            return {"id": "draft_abc123"}

        monkeypatch.setattr(sk, "connect_execute_tool", _success)

        from trace.actions.gmail_draft import create_digest_draft
        result = await create_digest_draft("rust", "interesting briefing")
        assert result["status"] == "created"
        assert result["draft_id"] == "draft_abc123"
        assert "not sent" in result["note"].lower()
        assert result["via_scalekit"] is True

    @pytest.mark.asyncio
    async def test_error_path(self, monkeypatch):
        import trace.auth.scalekit as sk
        monkeypatch.setattr(sk, "get_scalekit_client", lambda: object())

        async def _fail(**_kwargs):
            raise RuntimeError("gmail error")

        monkeypatch.setattr(sk, "connect_execute_tool", _fail)

        from trace.actions.gmail_draft import create_digest_draft
        result = await create_digest_draft("topic", "briefing")
        assert result["status"] == "error"

    def test_build_draft_body_contains_topic(self):
        from trace.actions.gmail_draft import _build_draft_body
        body = _build_draft_body("diffusion policy", "Great briefing.", [])
        assert "diffusion policy" in body.lower()
        assert "never" not in body.lower() or "not sent" in body.lower() or "trace" in body.lower()

    def test_build_draft_body_with_sources(self):
        from trace.actions.gmail_draft import _build_draft_body
        sources = ["https://arxiv.org/1", "https://github.com/2"]
        body = _build_draft_body("robotics", "Brief.", sources)
        assert "arxiv.org" in body
        assert "github.com" in body


class TestRedditDraft:
    def test_build_reddit_draft_structure(self):
        from trace.actions.reddit_draft import build_reddit_draft
        payload = build_reddit_draft("embodied ai", "Great insight.", ["robotics", "llm"])
        assert "title" in payload
        assert "body" in payload
        assert payload["subreddit"] == ""  # intentionally blank
        assert "embodied ai" in payload["title"].lower() or "bridge" in payload["title"].lower()

    def test_build_reddit_draft_title_truncation(self):
        from trace.actions.reddit_draft import build_reddit_draft
        long_topic = "a" * 200
        payload = build_reddit_draft(long_topic, "brief", ["b"] * 20)
        assert len(payload["title"]) <= 300

    def test_build_reddit_draft_no_supporting_topics(self):
        from trace.actions.reddit_draft import build_reddit_draft
        payload = build_reddit_draft("ai safety", "Briefing.", [])
        assert "title" in payload
        assert "body" in payload

    @pytest.mark.asyncio
    async def test_submit_approved_stub_without_scalekit(self, monkeypatch):
        import trace.auth.scalekit as sk
        monkeypatch.setattr(sk, "get_scalekit_client", lambda: None)

        from trace.actions.reddit_draft import build_reddit_draft, submit_approved_post
        payload = build_reddit_draft("topic", "brief")
        payload["subreddit"] = "r/MachineLearning"
        result = await submit_approved_post(payload)
        assert result["status"] == "stub"

    @pytest.mark.asyncio
    async def test_submit_requires_subreddit(self, monkeypatch):
        import trace.auth.scalekit as sk
        monkeypatch.setattr(sk, "get_scalekit_client", lambda: object())

        from trace.actions.reddit_draft import build_reddit_draft, submit_approved_post
        payload = build_reddit_draft("topic", "brief")
        # subreddit is blank by default.
        result = await submit_approved_post(payload)
        assert result["status"] == "error"
        assert "subreddit" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_submit_success(self, monkeypatch):
        import trace.auth.scalekit as sk
        monkeypatch.setattr(sk, "get_scalekit_client", lambda: object())

        async def _success(**_kwargs):
            return {"url": "https://reddit.com/r/ML/123456"}

        monkeypatch.setattr(sk, "connect_execute_tool", _success)

        from trace.actions.reddit_draft import build_reddit_draft, submit_approved_post
        payload = build_reddit_draft("embodied ai", "good brief", ["robotics"])
        payload["subreddit"] = "r/MachineLearning"
        result = await submit_approved_post(payload)
        assert result["status"] == "submitted"
        assert "reddit.com" in result["post_url"]


class TestDemoSeed:
    def test_build_demo_graph_returns_graph(self):
        from trace.agent.demo_seed import build_demo_graph
        graph = build_demo_graph()
        assert len(graph.topics) == 8
        assert graph.signal_count > 0

    def test_demo_graph_topic_names(self):
        from trace.agent.demo_seed import build_demo_graph
        graph = build_demo_graph()
        names = {t.name for t in graph.topics}
        assert "diffusion policy" in names
        assert "robot learning" in names
        assert "nuplan" in names

    def test_demo_graph_has_varied_curiosity_types(self):
        from trace.agent.demo_seed import build_demo_graph
        from trace.models import CuriosityType
        graph = build_demo_graph()
        types = {t.curiosity_type for t in graph.topics}
        assert CuriosityType.RECURRING in types
        assert CuriosityType.DEEP in types

    def test_demo_graph_temporal_consistency(self):
        from trace.agent.demo_seed import build_demo_graph
        graph = build_demo_graph()
        for t in graph.topics:
            if t.first_seen and t.last_seen:
                assert t.first_seen <= t.last_seen

    @pytest.mark.asyncio
    async def test_inject_demo_seed_returns_ok(self):
        from trace.agent.demo_seed import inject_demo_seed
        result = await inject_demo_seed()
        assert result["status"] == "ok"
        assert result["topic_count"] == 8
        assert result["profile_id"] == "demo"
        assert "hint" in result
