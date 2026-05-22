"""
Unit tests for trace/signals/chatgpt_export.py — ChatGPTExportCollector.

Test categories:
  1. Constructor validation: missing file, naive since datetime
  2. Happy-path collection: user messages parsed into RawSignals
  3. Role filtering: only author.role == "user" extracted, not assistant/system
  4. Content filtering: non-text content_type, short messages (<10 chars), missing create_time
  5. Time filtering: `since` parameter correctly excludes old messages
  6. Content truncation: messages longer than 2000 chars truncated
  7. Metadata: conversation_title and conversation_create_time present
  8. Multi-conversation: signals from multiple conversations aggregated
  9. Error cases: malformed JSON, wrong top-level type

All tests write real JSON fixture files using tmp_path.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trace.models import RawSignal, SignalSource
from trace.signals.base import SignalCollectionError
from trace.signals.chatgpt_export import ChatGPTExportCollector


# ── Helpers ───────────────────────────────────────────────────────────────────

_DEFAULT_TS: float = 1_704_067_200.0  # 2024-01-01 00:00:00 UTC


def make_user_node(
    text: str,
    create_time: float = _DEFAULT_TS,
    content_type: str = "text",
    node_id: str = "node-1",
) -> dict:
    """Build a mapping node with a user message."""
    return {
        node_id: {
            "message": {
                "author": {"role": "user"},
                "content": {
                    "content_type": content_type,
                    "parts": [text],
                },
                "create_time": create_time,
            }
        }
    }


def make_assistant_node(text: str, node_id: str = "node-2") -> dict:
    return {
        node_id: {
            "message": {
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": [text]},
                "create_time": _DEFAULT_TS + 10,
            }
        }
    }


def make_conversation(
    mapping: dict,
    title: str = "Test Conversation",
    create_time: float = _DEFAULT_TS,
) -> dict:
    return {
        "title": title,
        "create_time": create_time,
        "update_time": create_time + 100,
        "mapping": mapping,
    }


def write_conversations(tmp_path: Path, conversations: list[dict]) -> Path:
    path = tmp_path / "conversations.json"
    path.write_text(json.dumps(conversations), encoding="utf-8")
    return path


def single_user_message(
    tmp_path: Path,
    text: str = "Explain self-attention in transformers",
    create_time: float = _DEFAULT_TS,
    title: str = "Test Conversation",
) -> Path:
    mapping = make_user_node(text, create_time)
    convo = make_conversation(mapping, title=title, create_time=create_time)
    return write_conversations(tmp_path, [convo])


# ── Constructor validation ────────────────────────────────────────────────────

class TestConstructorValidation:
    async def test_missing_file_raises_signal_collection_error(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "nonexistent.json"
        collector = ChatGPTExportCollector(path)
        with pytest.raises(SignalCollectionError) as exc_info:
            await collector.collect()
        assert exc_info.value.source == SignalSource.CHATGPT_EXPORT
        assert "conversations.json not found" in str(exc_info.value)

    def test_naive_since_raises_value_error(
        self, chatgpt_conversations_path: Path
    ) -> None:
        naive = datetime(2024, 1, 1, 12, 0, 0)
        with pytest.raises(ValueError, match="timezone-aware"):
            ChatGPTExportCollector(chatgpt_conversations_path, since=naive)

    def test_aware_since_accepted(
        self, chatgpt_conversations_path: Path
    ) -> None:
        since = datetime(2024, 1, 1, tzinfo=timezone.utc)
        collector = ChatGPTExportCollector(chatgpt_conversations_path, since=since)
        assert collector is not None

    def test_no_since_accepted(self, chatgpt_conversations_path: Path) -> None:
        collector = ChatGPTExportCollector(chatgpt_conversations_path)
        assert collector is not None

    def test_source_is_chatgpt_export(
        self, chatgpt_conversations_path: Path
    ) -> None:
        assert ChatGPTExportCollector(chatgpt_conversations_path).source == SignalSource.CHATGPT_EXPORT


# ── Happy-path collection ─────────────────────────────────────────────────────

class TestHappyPath:
    async def test_collect_returns_list_of_raw_signals(
        self, chatgpt_conversations_path: Path
    ) -> None:
        signals = await ChatGPTExportCollector(chatgpt_conversations_path).collect()
        assert isinstance(signals, list)
        assert all(isinstance(s, RawSignal) for s in signals)

    async def test_collect_returns_only_user_messages_count(
        self, chatgpt_conversations_path: Path
    ) -> None:
        # fixture has 1 conversation with 2 user messages and 1 assistant message
        signals = await ChatGPTExportCollector(chatgpt_conversations_path).collect()
        assert len(signals) == 2

    async def test_signal_source_is_chatgpt_export(
        self, chatgpt_conversations_path: Path
    ) -> None:
        signals = await ChatGPTExportCollector(chatgpt_conversations_path).collect()
        for sig in signals:
            assert sig.source == SignalSource.CHATGPT_EXPORT

    async def test_signal_content_matches_user_message(
        self, tmp_path: Path
    ) -> None:
        text = "How does the transformer encoder differ from the decoder?"
        path = single_user_message(tmp_path, text=text)
        signals = await ChatGPTExportCollector(path).collect()
        assert len(signals) == 1
        assert signals[0].content == text

    async def test_empty_conversations_returns_empty_list(
        self, tmp_path: Path
    ) -> None:
        path = write_conversations(tmp_path, [])
        signals = await ChatGPTExportCollector(path).collect()
        assert signals == []

    async def test_conversation_with_no_user_messages_returns_empty(
        self, tmp_path: Path
    ) -> None:
        mapping = make_assistant_node("Some assistant reply")
        convo = make_conversation(mapping)
        path = write_conversations(tmp_path, [convo])
        signals = await ChatGPTExportCollector(path).collect()
        assert signals == []


# ── Role filtering ────────────────────────────────────────────────────────────

class TestRoleFiltering:
    async def test_assistant_messages_not_extracted(self, tmp_path: Path) -> None:
        mapping = {
            **make_user_node("What is RLHF?"),
            **make_assistant_node("RLHF stands for Reinforcement Learning from Human Feedback..."),
        }
        path = write_conversations(tmp_path, [make_conversation(mapping)])
        signals = await ChatGPTExportCollector(path).collect()
        # Only 1 user message; assistant message must not appear
        assert len(signals) == 1
        assert "RLHF stands for" not in signals[0].content

    async def test_system_messages_not_extracted(self, tmp_path: Path) -> None:
        mapping = {
            "system-node": {
                "message": {
                    "author": {"role": "system"},
                    "content": {"content_type": "text", "parts": ["You are a helpful assistant."]},
                    "create_time": _DEFAULT_TS,
                }
            },
            **make_user_node("What is fine-tuning?"),
        }
        path = write_conversations(tmp_path, [make_conversation(mapping)])
        signals = await ChatGPTExportCollector(path).collect()
        assert len(signals) == 1
        assert signals[0].content == "What is fine-tuning?"

    async def test_null_message_node_skipped(self, tmp_path: Path) -> None:
        mapping = {
            "root-node": {"message": None},
            **make_user_node("Valid user message here"),
        }
        path = write_conversations(tmp_path, [make_conversation(mapping)])
        signals = await ChatGPTExportCollector(path).collect()
        assert len(signals) == 1


# ── Content filtering ─────────────────────────────────────────────────────────

class TestContentFiltering:
    async def test_non_text_content_type_filtered(self, tmp_path: Path) -> None:
        mapping = make_user_node(
            "print('hello')", content_type="code"
        )
        path = write_conversations(tmp_path, [make_conversation(mapping)])
        signals = await ChatGPTExportCollector(path).collect()
        assert signals == []

    @pytest.mark.parametrize("short_text", [
        "yes",
        "ok",
        "go on",
        "thanks",
        "ok ok",
        "!",
        "sure",
    ])
    async def test_short_messages_filtered(
        self, tmp_path: Path, short_text: str
    ) -> None:
        # All these are < 10 chars
        assert len(short_text) < 10
        mapping = make_user_node(short_text)
        path = write_conversations(tmp_path, [make_conversation(mapping)])
        signals = await ChatGPTExportCollector(path).collect()
        assert signals == [], f"{short_text!r} (len={len(short_text)}) should be filtered"

    async def test_message_exactly_10_chars_accepted(
        self, tmp_path: Path
    ) -> None:
        text = "1234567890"  # exactly 10 chars
        assert len(text) == 10
        mapping = make_user_node(text)
        path = write_conversations(tmp_path, [make_conversation(mapping)])
        signals = await ChatGPTExportCollector(path).collect()
        assert len(signals) == 1

    async def test_message_9_chars_filtered(self, tmp_path: Path) -> None:
        text = "123456789"  # 9 chars, < MIN_CONTENT_LENGTH
        assert len(text) == 9
        mapping = make_user_node(text)
        path = write_conversations(tmp_path, [make_conversation(mapping)])
        signals = await ChatGPTExportCollector(path).collect()
        assert signals == []

    async def test_missing_create_time_filtered(self, tmp_path: Path) -> None:
        mapping = {
            "node-1": {
                "message": {
                    "author": {"role": "user"},
                    "content": {
                        "content_type": "text",
                        "parts": ["This message has no timestamp"],
                    },
                    # no create_time key
                }
            }
        }
        path = write_conversations(tmp_path, [make_conversation(mapping)])
        signals = await ChatGPTExportCollector(path).collect()
        assert signals == []

    async def test_null_create_time_filtered(self, tmp_path: Path) -> None:
        mapping = {
            "node-1": {
                "message": {
                    "author": {"role": "user"},
                    "content": {
                        "content_type": "text",
                        "parts": ["Message with null timestamp"],
                    },
                    "create_time": None,
                }
            }
        }
        path = write_conversations(tmp_path, [make_conversation(mapping)])
        signals = await ChatGPTExportCollector(path).collect()
        assert signals == []

    async def test_non_string_parts_skipped(self, tmp_path: Path) -> None:
        # parts may contain non-string objects in rare export edge cases
        mapping = {
            "node-1": {
                "message": {
                    "author": {"role": "user"},
                    "content": {
                        "content_type": "text",
                        "parts": [{"type": "image_url", "url": "..."}],
                    },
                    "create_time": _DEFAULT_TS,
                }
            }
        }
        path = write_conversations(tmp_path, [make_conversation(mapping)])
        # Non-string parts produce empty text, which is < 10 chars → filtered
        signals = await ChatGPTExportCollector(path).collect()
        assert signals == []

    async def test_mixed_string_and_non_string_parts_joins_strings(
        self, tmp_path: Path
    ) -> None:
        mapping = {
            "node-1": {
                "message": {
                    "author": {"role": "user"},
                    "content": {
                        "content_type": "text",
                        "parts": ["Explain gradient descent please", {"type": "image"}],
                    },
                    "create_time": _DEFAULT_TS,
                }
            }
        }
        path = write_conversations(tmp_path, [make_conversation(mapping)])
        signals = await ChatGPTExportCollector(path).collect()
        assert len(signals) == 1
        assert signals[0].content == "Explain gradient descent please"


# ── Timestamp conversion ──────────────────────────────────────────────────────

class TestTimestampConversion:
    async def test_create_time_converts_to_utc_datetime(
        self, tmp_path: Path
    ) -> None:
        # 1_704_067_200.0 sec = 2024-01-01 00:00:00 UTC
        path = single_user_message(tmp_path, create_time=1_704_067_200.0)
        signals = await ChatGPTExportCollector(path).collect()
        expected = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert signals[0].timestamp == expected

    async def test_timestamp_is_utc_aware(self, tmp_path: Path) -> None:
        path = single_user_message(tmp_path)
        signals = await ChatGPTExportCollector(path).collect()
        assert signals[0].timestamp.tzinfo is not None
        assert signals[0].timestamp.tzinfo == timezone.utc


# ── Content truncation ────────────────────────────────────────────────────────

class TestContentTruncation:
    async def test_message_over_2000_chars_truncated(
        self, tmp_path: Path
    ) -> None:
        long_text = "a" * 3000
        path = single_user_message(tmp_path, text=long_text)
        signals = await ChatGPTExportCollector(path).collect()
        assert len(signals) == 1
        assert len(signals[0].content) == 2000

    async def test_message_exactly_2000_chars_not_truncated(
        self, tmp_path: Path
    ) -> None:
        text = "x" * 2000
        path = single_user_message(tmp_path, text=text)
        signals = await ChatGPTExportCollector(path).collect()
        assert len(signals[0].content) == 2000

    async def test_message_under_2000_chars_not_truncated(
        self, tmp_path: Path
    ) -> None:
        text = "How does beam search work in sequence models?"
        path = single_user_message(tmp_path, text=text)
        signals = await ChatGPTExportCollector(path).collect()
        assert signals[0].content == text


# ── Metadata ──────────────────────────────────────────────────────────────────

class TestMetadata:
    async def test_metadata_contains_conversation_title(
        self, tmp_path: Path
    ) -> None:
        path = single_user_message(
            tmp_path,
            text="Tell me about reinforcement learning",
            title="RL Deep Dive",
        )
        signals = await ChatGPTExportCollector(path).collect()
        assert signals[0].metadata["conversation_title"] == "RL Deep Dive"

    async def test_metadata_contains_conversation_create_time(
        self, tmp_path: Path
    ) -> None:
        create_time = 1_704_067_200.0
        path = single_user_message(tmp_path, create_time=create_time)
        signals = await ChatGPTExportCollector(path).collect()
        assert signals[0].metadata["conversation_create_time"] == create_time

    async def test_metadata_conversation_title_empty_string_when_missing(
        self, tmp_path: Path
    ) -> None:
        mapping = make_user_node("What is knowledge distillation?")
        convo = {
            # no "title" key
            "create_time": _DEFAULT_TS,
            "mapping": mapping,
        }
        path = write_conversations(tmp_path, [convo])
        signals = await ChatGPTExportCollector(path).collect()
        assert signals[0].metadata["conversation_title"] == ""

    async def test_metadata_conversation_create_time_none_when_missing(
        self, tmp_path: Path
    ) -> None:
        mapping = make_user_node("Explain BERT pretraining objectives")
        convo = {
            "title": "BERT Session",
            # no "create_time" key
            "mapping": mapping,
        }
        path = write_conversations(tmp_path, [convo])
        signals = await ChatGPTExportCollector(path).collect()
        assert signals[0].metadata["conversation_create_time"] is None


# ── Multi-conversation aggregation ───────────────────────────────────────────

class TestMultiConversation:
    async def test_signals_from_multiple_conversations_aggregated(
        self, tmp_path: Path
    ) -> None:
        convo1 = make_conversation(
            make_user_node("What is attention?", node_id="n1"),
            title="Attention",
        )
        convo2 = make_conversation(
            make_user_node("What is backpropagation?", node_id="n2"),
            title="Backprop",
        )
        path = write_conversations(tmp_path, [convo1, convo2])
        signals = await ChatGPTExportCollector(path).collect()
        assert len(signals) == 2

    async def test_signals_from_different_conversations_have_correct_titles(
        self, tmp_path: Path
    ) -> None:
        convo1 = make_conversation(
            make_user_node("What is attention mechanism?", node_id="n1"),
            title="Attention Research",
        )
        convo2 = make_conversation(
            make_user_node("How does dropout prevent overfitting?", node_id="n2"),
            title="Regularization Techniques",
        )
        path = write_conversations(tmp_path, [convo1, convo2])
        signals = await ChatGPTExportCollector(path).collect()
        titles = {s.metadata["conversation_title"] for s in signals}
        assert titles == {"Attention Research", "Regularization Techniques"}

    async def test_non_dict_conversation_skipped(self, tmp_path: Path) -> None:
        convo = make_conversation(
            make_user_node("What is GELU activation?"),
            title="Valid Convo",
        )
        path = write_conversations(tmp_path, [convo, "not a dict", None])
        signals = await ChatGPTExportCollector(path).collect()
        assert len(signals) == 1


# ── Time filtering (since parameter) ─────────────────────────────────────────

class TestSinceFiltering:
    async def test_message_after_since_included(self, tmp_path: Path) -> None:
        since = datetime(2023, 12, 31, tzinfo=timezone.utc)
        path = single_user_message(tmp_path, create_time=_DEFAULT_TS)  # 2024-01-01
        signals = await ChatGPTExportCollector(path, since=since).collect()
        assert len(signals) == 1

    async def test_message_before_since_excluded(self, tmp_path: Path) -> None:
        since = datetime(2024, 6, 1, tzinfo=timezone.utc)
        path = single_user_message(tmp_path, create_time=_DEFAULT_TS)  # 2024-01-01
        signals = await ChatGPTExportCollector(path, since=since).collect()
        assert signals == []

    async def test_message_exactly_at_since_included(
        self, tmp_path: Path
    ) -> None:
        since = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        path = single_user_message(tmp_path, create_time=_DEFAULT_TS)
        signals = await ChatGPTExportCollector(path, since=since).collect()
        assert len(signals) == 1

    async def test_since_filters_some_but_not_all(self, tmp_path: Path) -> None:
        since = datetime(2024, 1, 2, tzinfo=timezone.utc)
        mapping = {
            **make_user_node(
                "Old question about transformers",
                create_time=1_704_067_200.0,  # 2024-01-01 → excluded
                node_id="n1",
            ),
            **make_user_node(
                "Newer question about diffusion models",
                create_time=1_704_153_600.0,  # 2024-01-02 → included
                node_id="n2",
            ),
        }
        path = write_conversations(tmp_path, [make_conversation(mapping)])
        signals = await ChatGPTExportCollector(path, since=since).collect()
        assert len(signals) == 1
        assert "diffusion" in signals[0].content


# ── Error cases ───────────────────────────────────────────────────────────────

class TestErrorCases:
    async def test_malformed_json_raises_signal_collection_error(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "conversations.json"
        path.write_text("{not valid json}", encoding="utf-8")
        with pytest.raises(SignalCollectionError) as exc_info:
            await ChatGPTExportCollector(path).collect()
        assert exc_info.value.source == SignalSource.CHATGPT_EXPORT
        assert "Failed to parse" in str(exc_info.value)

    async def test_json_dict_not_list_raises_signal_collection_error(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "conversations.json"
        path.write_text(json.dumps({"conversations": []}), encoding="utf-8")
        with pytest.raises(SignalCollectionError) as exc_info:
            await ChatGPTExportCollector(path).collect()
        assert "must be a JSON array" in str(exc_info.value)

    async def test_json_integer_not_list_raises_signal_collection_error(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "conversations.json"
        path.write_text("42", encoding="utf-8")
        with pytest.raises(SignalCollectionError) as exc_info:
            await ChatGPTExportCollector(path).collect()
        assert "must be a JSON array" in str(exc_info.value)


# ── File size guard ───────────────────────────────────────────────────────────

class TestFileSizeGuard:
    async def test_file_too_large_raises_signal_collection_error(
        self, tmp_path: Path
    ) -> None:
        from unittest.mock import MagicMock, patch

        path = tmp_path / "conversations.json"
        path.write_text("[]", encoding="utf-8")
        collector = ChatGPTExportCollector(path)
        mock_stat = MagicMock()
        mock_stat.st_size = 200 * 1024 * 1024  # 200 MB
        with patch("pathlib.Path.stat", return_value=mock_stat):
            with pytest.raises(SignalCollectionError, match="too large"):
                await collector.collect()

    async def test_file_at_limit_is_accepted(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        path = tmp_path / "conversations.json"
        path.write_text("[]", encoding="utf-8")
        collector = ChatGPTExportCollector(path)
        mock_stat = MagicMock()
        mock_stat.st_size = 100 * 1024 * 1024  # exactly 100 MB
        with patch("pathlib.Path.stat", return_value=mock_stat):
            signals = await collector.collect()
        assert signals == []

    async def test_file_under_limit_proceeds_normally(self, tmp_path: Path) -> None:
        path = tmp_path / "conversations.json"
        mapping = make_user_node("What is RLHF in machine learning?")
        path.write_text(
            json.dumps([make_conversation(mapping)]), encoding="utf-8"
        )
        collector = ChatGPTExportCollector(path)
        signals = await collector.collect()
        assert len(signals) >= 1
