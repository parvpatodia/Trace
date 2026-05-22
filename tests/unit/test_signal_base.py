"""
Unit tests for trace/signals/base.py — SignalCollector ABC + SignalCollectionError.

These tests verify the contract that ALL concrete collectors must satisfy.
They use minimal inline subclasses to test the ABC mechanism directly,
without involving any file I/O or external state.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trace.models import RawSignal, SignalSource
from trace.signals.base import SignalCollectionError, SignalCollector


# ── SignalCollectionError ─────────────────────────────────────────────────────

class TestSignalCollectionError:
    def test_message_includes_source_prefix(self) -> None:
        err = SignalCollectionError(SignalSource.GOOGLE_TAKEOUT, "file not found")
        assert "[google_takeout]" in str(err)
        assert "file not found" in str(err)

    def test_source_attribute_preserved(self) -> None:
        err = SignalCollectionError(SignalSource.GMAIL, "API error")
        assert err.source == SignalSource.GMAIL

    def test_all_sources_produce_correct_prefix(self) -> None:
        for source in SignalSource:
            err = SignalCollectionError(source, "msg")
            assert f"[{source.value}]" in str(err)

    def test_is_exception_subclass(self) -> None:
        assert issubclass(SignalCollectionError, Exception)

    def test_can_be_raised_and_caught_as_exception(self) -> None:
        with pytest.raises(Exception):
            raise SignalCollectionError(SignalSource.REDDIT_POST, "network timeout")

    def test_can_be_caught_specifically(self) -> None:
        with pytest.raises(SignalCollectionError) as exc_info:
            raise SignalCollectionError(SignalSource.REDDIT_POST, "network timeout")
        assert exc_info.value.source == SignalSource.REDDIT_POST

    def test_exception_chaining_via_from(self) -> None:
        original = ValueError("original cause")
        with pytest.raises(SignalCollectionError) as exc_info:
            try:
                raise original
            except ValueError as e:
                raise SignalCollectionError(SignalSource.FILESYSTEM, "wrapped") from e
        assert exc_info.value.__cause__ is original

    def test_different_messages_produce_different_str(self) -> None:
        e1 = SignalCollectionError(SignalSource.GMAIL, "timeout")
        e2 = SignalCollectionError(SignalSource.GMAIL, "parse error")
        assert str(e1) != str(e2)


# ── SignalCollector ABC instantiation rules ───────────────────────────────────

class TestSignalCollectorABCEnforcement:
    def test_cannot_instantiate_abc_directly(self) -> None:
        with pytest.raises(TypeError):
            SignalCollector()  # type: ignore[abstract]

    def test_subclass_without_source_cannot_instantiate(self) -> None:
        class NoSource(SignalCollector):
            async def collect(self) -> list[RawSignal]:
                return []

        with pytest.raises(TypeError, match="abstract"):
            NoSource()  # type: ignore[abstract]

    def test_subclass_without_collect_cannot_instantiate(self) -> None:
        class NoCollect(SignalCollector):
            source = SignalSource.GOOGLE_TAKEOUT

        with pytest.raises(TypeError, match="abstract"):
            NoCollect()  # type: ignore[abstract]

    def test_subclass_with_both_required_members_can_instantiate(self) -> None:
        class ValidCollector(SignalCollector):
            source = SignalSource.FILESYSTEM

            async def collect(self) -> list[RawSignal]:
                return []

        collector = ValidCollector()
        assert collector is not None

    def test_class_attribute_satisfies_abstract_property(self) -> None:
        # Python ABCMeta allows class attributes to satisfy abstract property
        # requirements. This is the canonical pattern for our collectors.
        class ConcreteSrc(SignalCollector):
            source = SignalSource.CHATGPT_EXPORT

            async def collect(self) -> list[RawSignal]:
                return []

        c = ConcreteSrc()
        assert c.source == SignalSource.CHATGPT_EXPORT

    def test_source_accessible_on_instance(self) -> None:
        class C(SignalCollector):
            source = SignalSource.GMAIL

            async def collect(self) -> list[RawSignal]:
                return []

        assert C().source == SignalSource.GMAIL

    def test_source_accessible_on_class(self) -> None:
        class C(SignalCollector):
            source = SignalSource.REDDIT_POST

            async def collect(self) -> list[RawSignal]:
                return []

        assert C.source == SignalSource.REDDIT_POST


# ── SignalCollector behavioural contract ──────────────────────────────────────

class TestSignalCollectorContract:
    async def test_collect_returns_empty_list_when_no_signals(self) -> None:
        class EmptyCollector(SignalCollector):
            source = SignalSource.FILESYSTEM

            async def collect(self) -> list[RawSignal]:
                return []

        result = await EmptyCollector().collect()
        assert result == []
        assert isinstance(result, list)

    async def test_collect_returns_raw_signals(self) -> None:
        ts = datetime.now(timezone.utc)

        class OneSignalCollector(SignalCollector):
            source = SignalSource.CHATGPT_EXPORT

            async def collect(self) -> list[RawSignal]:
                return [
                    RawSignal(
                        source=self.source,
                        content="explain RLHF fine-tuning",
                        timestamp=ts,
                    )
                ]

        signals = await OneSignalCollector().collect()
        assert len(signals) == 1
        assert isinstance(signals[0], RawSignal)

    async def test_signals_have_correct_source(self) -> None:
        ts = datetime.now(timezone.utc)

        class TaggedCollector(SignalCollector):
            source = SignalSource.GOOGLE_TAKEOUT

            async def collect(self) -> list[RawSignal]:
                return [
                    RawSignal(source=self.source, content="ViT architecture", timestamp=ts),
                    RawSignal(source=self.source, content="BERT pretraining", timestamp=ts),
                ]

        signals = await TaggedCollector().collect()
        for sig in signals:
            assert sig.source == SignalSource.GOOGLE_TAKEOUT

    async def test_collect_raises_signal_collection_error_on_failure(self) -> None:
        class FailingCollector(SignalCollector):
            source = SignalSource.REDDIT_POST

            async def collect(self) -> list[RawSignal]:
                raise SignalCollectionError(self.source, "subreddit not found")

        with pytest.raises(SignalCollectionError) as exc_info:
            await FailingCollector().collect()
        assert exc_info.value.source == SignalSource.REDDIT_POST

    async def test_multiple_collectors_independent(self) -> None:
        ts = datetime.now(timezone.utc)

        class C1(SignalCollector):
            source = SignalSource.GMAIL

            async def collect(self) -> list[RawSignal]:
                return [RawSignal(source=self.source, content="email content", timestamp=ts)]

        class C2(SignalCollector):
            source = SignalSource.CHATGPT_EXPORT

            async def collect(self) -> list[RawSignal]:
                return [RawSignal(source=self.source, content="chatgpt content", timestamp=ts)]

        s1 = await C1().collect()
        s2 = await C2().collect()
        assert s1[0].source == SignalSource.GMAIL
        assert s2[0].source == SignalSource.CHATGPT_EXPORT
