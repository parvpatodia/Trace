"""
Unit tests for trace/graph/extractor.py — TopicExtractor.

TopicExtractor calls the Claude API with batches of RawSignal content and
returns a list of RawTopicData dicts: [{name, aliases, signal_ids}].

All tests mock the Anthropic client — no real API calls, no credentials needed.

Test categories:
  1. Constructor validation: None client raises, invalid batch_size raises
  2. Empty input: no signals → empty list, no API call made
  3. Single batch: <= batch_size signals → one messages.create call
  4. Multiple batches: > batch_size signals → ceil(n/batch_size) calls
  5. JSON parsing: valid response parsed into RawTopicData list
  6. Topic merging: same topic name across batches → combined signal_ids
  7. Noise filtering: items missing required keys skipped; extra keys ignored
  8. Signal ID filtering: IDs in Claude response not in our signals → dropped
  9. Prompt structure: system prompt cached, signal IDs included in user msg
  10. Error handling: malformed JSON → TopicExtractionError
  11. Error handling: AnthropicError → TopicExtractionError
  12. Return type contract: list[RawTopicData] with correct shape
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from trace.models import RawSignal, SignalSource
from trace.graph.extractor import RawTopicData, TopicExtractionError, TopicExtractor


# ── Helpers ───────────────────────────────────────────────────────────────────

_TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


def make_signal(
    content: str = "Explain transformer self-attention mechanisms",
    source: SignalSource = SignalSource.GOOGLE_TAKEOUT,
    signal_id: str | None = None,
) -> RawSignal:
    sig = RawSignal(source=source, content=content, timestamp=_TS)
    if signal_id is not None:
        # Pydantic frozen model — rebuild with explicit id
        sig = sig.model_copy(update={"id": signal_id})
    return sig


def make_signals(n: int, prefix: str = "signal") -> list[RawSignal]:
    return [
        make_signal(
            content=f"Content {i}: machine learning topic about neural network training",
            signal_id=f"{prefix}-{i}",
        )
        for i in range(n)
    ]


def mock_response(topics: list[dict]) -> MagicMock:
    """Build a mock Anthropic messages.create response."""
    resp = MagicMock()
    resp.content = [MagicMock(text=json.dumps(topics))]
    return resp


def make_client(response: MagicMock | None = None) -> MagicMock:
    """Build a mock anthropic.Anthropic client."""
    client = MagicMock()
    client.messages.create.return_value = response or mock_response([])
    return client


# ── Constructor validation ────────────────────────────────────────────────────

class TestConstructorValidation:
    def test_none_client_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="client"):
            TopicExtractor(client=None)  # type: ignore[arg-type]

    def test_batch_size_zero_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            TopicExtractor(client=make_client(), batch_size=0)

    def test_negative_batch_size_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            TopicExtractor(client=make_client(), batch_size=-1)

    def test_valid_client_accepted(self) -> None:
        extractor = TopicExtractor(client=make_client())
        assert extractor is not None

    def test_custom_batch_size_accepted(self) -> None:
        extractor = TopicExtractor(client=make_client(), batch_size=10)
        assert extractor is not None

    def test_custom_model_accepted(self) -> None:
        extractor = TopicExtractor(client=make_client(), model="claude-haiku-4-5-20251001")
        assert extractor is not None


# ── Empty input ───────────────────────────────────────────────────────────────

class TestEmptyInput:
    async def test_empty_signals_returns_empty_list(self) -> None:
        client = make_client()
        result = await TopicExtractor(client).extract([])
        assert result == []

    async def test_empty_signals_makes_no_api_call(self) -> None:
        client = make_client()
        await TopicExtractor(client).extract([])
        client.messages.create.assert_not_called()


# ── Batch counting ────────────────────────────────────────────────────────────

class TestBatchCounting:
    async def test_signals_equal_to_batch_size_makes_one_call(self) -> None:
        client = make_client(mock_response([]))
        signals = make_signals(10)
        await TopicExtractor(client, batch_size=10).extract(signals)
        assert client.messages.create.call_count == 1

    async def test_signals_under_batch_size_makes_one_call(self) -> None:
        client = make_client(mock_response([]))
        signals = make_signals(7)
        await TopicExtractor(client, batch_size=10).extract(signals)
        assert client.messages.create.call_count == 1

    async def test_signals_over_batch_size_makes_two_calls(self) -> None:
        client = make_client(mock_response([]))
        signals = make_signals(15)
        await TopicExtractor(client, batch_size=10).extract(signals)
        assert client.messages.create.call_count == 2

    async def test_exact_multiple_of_batch_size(self) -> None:
        client = make_client(mock_response([]))
        signals = make_signals(30)
        await TopicExtractor(client, batch_size=10).extract(signals)
        assert client.messages.create.call_count == 3

    async def test_one_over_multiple_adds_extra_call(self) -> None:
        client = make_client(mock_response([]))
        signals = make_signals(11)
        await TopicExtractor(client, batch_size=10).extract(signals)
        assert client.messages.create.call_count == 2


# ── JSON parsing ──────────────────────────────────────────────────────────────

class TestJSONParsing:
    async def test_valid_response_parsed_into_raw_topic_data(self) -> None:
        sig = make_signal(signal_id="sig-1")
        client = make_client(mock_response([
            {"name": "transformer architecture", "aliases": ["attention"], "signal_ids": ["sig-1"]}
        ]))
        result = await TopicExtractor(client, batch_size=50).extract([sig])
        assert len(result) == 1
        assert result[0]["name"] == "transformer architecture"

    async def test_aliases_preserved(self) -> None:
        sig = make_signal(signal_id="sig-1")
        client = make_client(mock_response([
            {"name": "transformers", "aliases": ["BERT", "GPT", "attention"], "signal_ids": ["sig-1"]}
        ]))
        result = await TopicExtractor(client, batch_size=50).extract([sig])
        assert result[0]["aliases"] == ["BERT", "GPT", "attention"]

    async def test_signal_ids_preserved(self) -> None:
        sigs = make_signals(3, prefix="s")
        ids = [s.id for s in sigs]
        client = make_client(mock_response([
            {"name": "deep learning", "aliases": [], "signal_ids": ids}
        ]))
        result = await TopicExtractor(client, batch_size=50).extract(sigs)
        assert set(result[0]["signal_ids"]) == set(ids)

    async def test_multiple_topics_in_single_batch(self) -> None:
        sigs = make_signals(4, prefix="s")
        client = make_client(mock_response([
            {"name": "transformers", "aliases": [], "signal_ids": [sigs[0].id, sigs[1].id]},
            {"name": "reinforcement learning", "aliases": ["RL"], "signal_ids": [sigs[2].id, sigs[3].id]},
        ]))
        result = await TopicExtractor(client, batch_size=50).extract(sigs)
        assert len(result) == 2
        names = {t["name"] for t in result}
        assert names == {"transformers", "reinforcement learning"}

    async def test_empty_topic_list_in_response(self) -> None:
        sig = make_signal(signal_id="sig-1")
        client = make_client(mock_response([]))
        result = await TopicExtractor(client, batch_size=50).extract([sig])
        assert result == []


# ── Topic merging across batches ──────────────────────────────────────────────

class TestTopicMerging:
    async def test_same_topic_name_across_batches_merged(self) -> None:
        # batch_size=1 forces 2 API calls, each returning the same topic name
        sigs = make_signals(2, prefix="s")
        client = MagicMock()
        client.messages.create.side_effect = [
            mock_response([{"name": "transformers", "aliases": ["attention"], "signal_ids": [sigs[0].id]}]),
            mock_response([{"name": "transformers", "aliases": ["BERT"], "signal_ids": [sigs[1].id]}]),
        ]
        result = await TopicExtractor(client, batch_size=1).extract(sigs)
        # Should be merged into a single topic
        assert len(result) == 1
        assert result[0]["name"] == "transformers"

    async def test_merged_topic_has_all_signal_ids(self) -> None:
        sigs = make_signals(2, prefix="s")
        client = MagicMock()
        client.messages.create.side_effect = [
            mock_response([{"name": "transformers", "aliases": [], "signal_ids": [sigs[0].id]}]),
            mock_response([{"name": "transformers", "aliases": [], "signal_ids": [sigs[1].id]}]),
        ]
        result = await TopicExtractor(client, batch_size=1).extract(sigs)
        assert set(result[0]["signal_ids"]) == {sigs[0].id, sigs[1].id}

    async def test_merged_topic_has_combined_aliases_deduplicated(self) -> None:
        sigs = make_signals(2, prefix="s")
        client = MagicMock()
        client.messages.create.side_effect = [
            mock_response([{"name": "transformers", "aliases": ["attention", "BERT"], "signal_ids": [sigs[0].id]}]),
            mock_response([{"name": "transformers", "aliases": ["BERT", "GPT"], "signal_ids": [sigs[1].id]}]),
        ]
        result = await TopicExtractor(client, batch_size=1).extract(sigs)
        aliases = result[0]["aliases"]
        # No duplicates
        assert len(aliases) == len(set(aliases))
        assert set(aliases) == {"attention", "BERT", "GPT"}

    async def test_different_topic_names_not_merged(self) -> None:
        sigs = make_signals(2, prefix="s")
        client = MagicMock()
        client.messages.create.side_effect = [
            mock_response([{"name": "transformers", "aliases": [], "signal_ids": [sigs[0].id]}]),
            mock_response([{"name": "reinforcement learning", "aliases": [], "signal_ids": [sigs[1].id]}]),
        ]
        result = await TopicExtractor(client, batch_size=1).extract(sigs)
        assert len(result) == 2


# ── Noise / malformed items in response ───────────────────────────────────────

class TestResponseNoise:
    async def test_item_missing_name_skipped(self) -> None:
        sig = make_signal(signal_id="s1")
        client = make_client(mock_response([
            {"aliases": [], "signal_ids": ["s1"]},  # missing "name"
        ]))
        result = await TopicExtractor(client, batch_size=50).extract([sig])
        assert result == []

    async def test_item_missing_signal_ids_skipped(self) -> None:
        sig = make_signal(signal_id="s1")
        client = make_client(mock_response([
            {"name": "transformers", "aliases": []},  # missing "signal_ids"
        ]))
        result = await TopicExtractor(client, batch_size=50).extract([sig])
        assert result == []

    async def test_signal_ids_not_in_our_signals_dropped(self) -> None:
        sig = make_signal(signal_id="real-id")
        client = make_client(mock_response([
            {"name": "transformers", "aliases": [], "signal_ids": ["fake-id-not-ours"]}
        ]))
        result = await TopicExtractor(client, batch_size=50).extract([sig])
        # Topic has no valid signal_ids after filtering → dropped
        assert result == []

    async def test_extra_keys_in_response_ignored(self) -> None:
        sig = make_signal(signal_id="s1")
        client = make_client(mock_response([
            {"name": "transformers", "aliases": [], "signal_ids": ["s1"], "extra_key": "ignored"}
        ]))
        result = await TopicExtractor(client, batch_size=50).extract([sig])
        assert len(result) == 1
        assert result[0]["name"] == "transformers"

    async def test_non_dict_item_in_response_skipped(self) -> None:
        sig = make_signal(signal_id="s1")
        client = make_client(mock_response([
            "not a dict",
            {"name": "transformers", "aliases": [], "signal_ids": ["s1"]},
        ]))
        result = await TopicExtractor(client, batch_size=50).extract([sig])
        assert len(result) == 1


# ── Prompt structure ──────────────────────────────────────────────────────────

class TestPromptStructure:
    async def test_signal_ids_appear_in_user_message(self) -> None:
        sig = make_signal(signal_id="test-id-abc")
        client = make_client(mock_response([
            {"name": "transformers", "aliases": [], "signal_ids": ["test-id-abc"]}
        ]))
        await TopicExtractor(client, batch_size=50).extract([sig])
        call_kwargs = client.messages.create.call_args
        # messages is always passed as a keyword argument
        messages = call_kwargs.kwargs["messages"]
        user_text = messages[0]["content"]
        assert "test-id-abc" in user_text

    async def test_system_prompt_provided(self) -> None:
        sig = make_signal(signal_id="s1")
        client = make_client(mock_response([
            {"name": "transformers", "aliases": [], "signal_ids": ["s1"]}
        ]))
        await TopicExtractor(client, batch_size=50).extract([sig])
        call_kwargs = client.messages.create.call_args
        # system can be a string or a list of content blocks
        system = call_kwargs.kwargs.get("system")
        assert system is not None
        assert len(str(system)) > 50  # non-trivial system prompt

    async def test_model_passed_to_api(self) -> None:
        sig = make_signal(signal_id="s1")
        model = "claude-haiku-4-5-20251001"
        client = make_client(mock_response([
            {"name": "transformers", "aliases": [], "signal_ids": ["s1"]}
        ]))
        await TopicExtractor(client, batch_size=50, model=model).extract([sig])
        call_kwargs = client.messages.create.call_args
        assert call_kwargs.kwargs.get("model") == model


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:
    async def test_malformed_json_raises_topic_extraction_error(self) -> None:
        sig = make_signal(signal_id="s1")
        bad_resp = MagicMock()
        bad_resp.content = [MagicMock(text="not valid json {{{{")]
        client = make_client(bad_resp)
        with pytest.raises(TopicExtractionError) as exc_info:
            await TopicExtractor(client, batch_size=50).extract([sig])
        assert "json" in str(exc_info.value).lower() or "parse" in str(exc_info.value).lower()

    async def test_json_object_not_list_raises_topic_extraction_error(self) -> None:
        sig = make_signal(signal_id="s1")
        bad_resp = MagicMock()
        bad_resp.content = [MagicMock(text='{"name": "transformers"}')]
        client = make_client(bad_resp)
        with pytest.raises(TopicExtractionError):
            await TopicExtractor(client, batch_size=50).extract([sig])

    async def test_anthropic_error_raises_topic_extraction_error(self) -> None:
        import anthropic

        sig = make_signal(signal_id="s1")
        client = MagicMock()
        client.messages.create.side_effect = anthropic.APIError(
            message="rate limit exceeded",
            request=MagicMock(),
            body=None,
        )
        with pytest.raises(TopicExtractionError) as exc_info:
            await TopicExtractor(client, batch_size=50).extract([sig])
        assert "rate limit" in str(exc_info.value).lower() or "api" in str(exc_info.value).lower()

    async def test_topic_extraction_error_is_exception_subclass(self) -> None:
        assert issubclass(TopicExtractionError, Exception)


# ── Return type contract ──────────────────────────────────────────────────────

class TestReturnTypeContract:
    async def test_returns_list(self) -> None:
        client = make_client(mock_response([]))
        result = await TopicExtractor(client).extract([])
        assert isinstance(result, list)

    async def test_each_item_has_name_key(self) -> None:
        sig = make_signal(signal_id="s1")
        client = make_client(mock_response([
            {"name": "transformers", "aliases": [], "signal_ids": ["s1"]}
        ]))
        result = await TopicExtractor(client, batch_size=50).extract([sig])
        for item in result:
            assert "name" in item

    async def test_each_item_has_aliases_key(self) -> None:
        sig = make_signal(signal_id="s1")
        client = make_client(mock_response([
            {"name": "transformers", "aliases": ["attention"], "signal_ids": ["s1"]}
        ]))
        result = await TopicExtractor(client, batch_size=50).extract([sig])
        for item in result:
            assert "aliases" in item

    async def test_each_item_has_signal_ids_key(self) -> None:
        sig = make_signal(signal_id="s1")
        client = make_client(mock_response([
            {"name": "transformers", "aliases": [], "signal_ids": ["s1"]}
        ]))
        result = await TopicExtractor(client, batch_size=50).extract([sig])
        for item in result:
            assert "signal_ids" in item


# ── Response content guard ────────────────────────────────────────────────────

class TestResponseContentGuard:
    async def test_empty_response_content_raises_extraction_error(self) -> None:
        client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = []
        client.messages.create.return_value = mock_resp

        signals = make_signals(2)
        with pytest.raises(TopicExtractionError, match="Unexpected Claude response format"):
            await TopicExtractor(client=client).extract(signals)

    async def test_non_text_response_block_raises_extraction_error(self) -> None:
        client = MagicMock()
        mock_resp = MagicMock()
        non_text_block = MagicMock(spec=[])  # spec=[] → no attributes → hasattr(block, "text") is False
        mock_resp.content = [non_text_block]
        client.messages.create.return_value = mock_resp

        signals = make_signals(2)
        with pytest.raises(TopicExtractionError, match="Unexpected Claude response format"):
            await TopicExtractor(client=client).extract(signals)
