"""
Shared pytest fixtures for all test tiers (unit, integration, smoke).

Conventions:
  - Fixtures that create files use tmp_path (pytest built-in).
  - UTC-aware datetimes only — naive datetimes are a bug, not a test choice.
  - All async fixtures use pytest-asyncio's auto mode (asyncio_mode = "auto").
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trace.models import RawSignal, SignalSource


# ── Shared datetime helpers ───────────────────────────────────────────────────

@pytest.fixture()
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture()
def utc_ts_2024() -> datetime:
    """A fixed UTC datetime in 2024 for deterministic tests."""
    return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# ── Shared RawSignal factories ────────────────────────────────────────────────

@pytest.fixture()
def make_raw_signal(utc_now: datetime):
    """Factory that creates a valid RawSignal with overrideable fields."""
    def _make(
        source: SignalSource = SignalSource.GOOGLE_TAKEOUT,
        content: str = "machine learning transformer architecture",
        timestamp: datetime | None = None,
        url: str | None = None,
        metadata: dict | None = None,
    ) -> RawSignal:
        return RawSignal(
            source=source,
            content=content,
            timestamp=timestamp or utc_now,
            url=url,
            metadata=metadata or {},
        )
    return _make


# ── Google Takeout fixture files ──────────────────────────────────────────────

@pytest.fixture()
def browser_history_path(tmp_path: Path) -> Path:
    """Minimal valid BrowserHistory.json with two navigable entries."""
    data = {
        "Browser History": [
            {
                "title": "Attention Is All You Need - arXiv",
                "url": "https://arxiv.org/abs/1706.03762",
                "time_usec": 1704067200_000_000,  # 2024-01-01 00:00:00 UTC
                "page_transition": "LINK",
            },
            {
                "title": "Vision Transformer explained",
                "url": "https://example.com/vit",
                "time_usec": 1704153600_000_000,  # 2024-01-02 00:00:00 UTC
                "page_transition": "TYPED",
            },
        ]
    }
    path = tmp_path / "BrowserHistory.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ── ChatGPT Export fixture files ──────────────────────────────────────────────

@pytest.fixture()
def chatgpt_conversations_path(tmp_path: Path) -> Path:
    """Minimal valid conversations.json with one conversation, two user messages."""
    data = [
        {
            "title": "Transformer Architecture Deep Dive",
            "create_time": 1704067200.0,
            "update_time": 1704067300.0,
            "mapping": {
                "node-1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {
                            "content_type": "text",
                            "parts": ["Explain self-attention in transformers"],
                        },
                        "create_time": 1704067210.0,
                    }
                },
                "node-2": {
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {
                            "content_type": "text",
                            "parts": ["Self-attention allows each token to attend..."],
                        },
                        "create_time": 1704067220.0,
                    }
                },
                "node-3": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {
                            "content_type": "text",
                            "parts": ["How does multi-head attention differ from single-head?"],
                        },
                        "create_time": 1704067230.0,
                    }
                },
            },
        }
    ]
    path = tmp_path / "conversations.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path
