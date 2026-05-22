"""
Unit tests for trace/signals/reddit.py — RedditCollector.

RedditCollector reads saved posts/comments and upvoted posts from a Reddit
account via PRAW. Because PRAW makes live API calls, all tests use
unittest.mock to patch the PRAW Reddit client — no real credentials needed.

Design: the collector receives a pre-constructed `praw.Reddit` instance
(dependency injection). Tests construct a fake client via MagicMock. This
keeps the constructor simple and makes the unit tests hermetic.

Test categories:
  1. Constructor validation: None client raises, naive since raises
  2. Saved items: posts and comments extracted, unsupported types skipped
  3. Source tagging: posts → REDDIT_POST, comments → REDDIT_COMMENT
  4. Content extraction: post title+selftext, comment body
  5. URL handling: external links on posts preserved, comments get None
  6. `since` filtering: items older than since excluded
  7. Content truncation: over 2000 chars truncated
  8. Metadata: subreddit, score, item id in metadata
  9. Empty / error handling: empty saved list → empty result
  10. PRAW exceptions: PRAWException → SignalCollectionError
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock, create_autospec

import praw.models
import pytest

from trace.models import RawSignal, SignalSource
from trace.signals.base import SignalCollectionError
from trace.signals.reddit import RedditCollector


# ── Fake PRAW helpers ─────────────────────────────────────────────────────────

_TS_2024: float = 1_704_067_200.0  # 2024-01-01 00:00:00 UTC
_TS_2023: float = 1_672_531_200.0  # 2023-01-01 00:00:00 UTC


def make_submission(
    title: str = "How does self-attention work in transformers?",
    selftext: str = "I've been reading about attention mechanisms...",
    url: str = "https://www.reddit.com/r/MachineLearning/comments/abc/",
    subreddit_name: str = "MachineLearning",
    score: int = 42,
    created_utc: float = _TS_2024,
    fullname: str = "t3_abc123",
    is_self: bool = True,
) -> MagicMock:
    """Build a mock praw.models.Submission.

    Uses spec=praw.models.Submission so isinstance() checks pass in the
    collector's _parse_item dispatcher.
    """
    post = MagicMock(spec=praw.models.Submission)
    post.title = title
    post.selftext = selftext
    post.url = url
    post.score = score
    post.created_utc = created_utc
    post.is_self = is_self
    post.fullname = fullname
    post.subreddit = MagicMock(display_name=subreddit_name)
    return post


def make_comment(
    body: str = "Self-attention lets each token attend to all others in the sequence.",
    subreddit_name: str = "learnmachinelearning",
    score: int = 15,
    created_utc: float = _TS_2024,
    fullname: str = "t1_xyz789",
    permalink: str = "/r/learnmachinelearning/comments/def/title/xyz789/",
) -> MagicMock:
    """Build a mock praw.models.Comment.

    Uses spec=praw.models.Comment so isinstance() checks pass in the
    collector's _parse_item dispatcher.
    """
    comment = MagicMock(spec=praw.models.Comment)
    comment.body = body
    comment.score = score
    comment.created_utc = created_utc
    comment.permalink = permalink
    comment.fullname = fullname
    comment.subreddit = MagicMock(display_name=subreddit_name)
    return comment


def make_reddit_client(saved_items: list) -> MagicMock:
    """Build a mock praw.Reddit with a user whose saved() returns saved_items."""
    client = MagicMock()
    client.user.me.return_value = MagicMock(
        saved=MagicMock(return_value=iter(saved_items))
    )
    return client


# ── Constructor validation ────────────────────────────────────────────────────

class TestConstructorValidation:
    def test_none_client_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="reddit_client"):
            RedditCollector(reddit_client=None)  # type: ignore[arg-type]

    def test_naive_since_raises_value_error(self) -> None:
        client = make_reddit_client([])
        with pytest.raises(ValueError, match="timezone-aware"):
            RedditCollector(client, since=datetime(2024, 1, 1))

    def test_aware_since_accepted(self) -> None:
        client = make_reddit_client([])
        since = datetime(2024, 1, 1, tzinfo=timezone.utc)
        collector = RedditCollector(client, since=since)
        assert collector is not None

    def test_no_since_accepted(self) -> None:
        client = make_reddit_client([])
        collector = RedditCollector(client)
        assert collector is not None

    def test_source_includes_reddit(self) -> None:
        client = make_reddit_client([])
        collector = RedditCollector(client)
        # RedditCollector emits multiple source types; verify both are valid
        assert collector.source in (
            SignalSource.REDDIT_POST,
            SignalSource.REDDIT_COMMENT,
            SignalSource.REDDIT_SAVED,
        )


# ── Saved posts ───────────────────────────────────────────────────────────────

class TestSavedPosts:
    async def test_saved_post_produces_signal(self) -> None:
        client = make_reddit_client([make_submission()])
        signals = await RedditCollector(client).collect()
        assert len(signals) == 1

    async def test_saved_post_source_is_reddit_post(self) -> None:
        client = make_reddit_client([make_submission()])
        signals = await RedditCollector(client).collect()
        assert signals[0].source == SignalSource.REDDIT_POST

    async def test_saved_post_content_combines_title_and_selftext(
        self,
    ) -> None:
        client = make_reddit_client([
            make_submission(
                title="What is gradient checkpointing?",
                selftext="I'm training a large model and running out of memory...",
            )
        ])
        signals = await RedditCollector(client).collect()
        assert "What is gradient checkpointing?" in signals[0].content
        assert "running out of memory" in signals[0].content

    async def test_saved_post_with_empty_selftext_uses_title_only(
        self,
    ) -> None:
        client = make_reddit_client([
            make_submission(title="Attention is all you need — paper discussion", selftext="")
        ])
        signals = await RedditCollector(client).collect()
        assert "Attention is all you need" in signals[0].content

    async def test_saved_link_post_url_preserved(self) -> None:
        client = make_reddit_client([
            make_submission(
                url="https://arxiv.org/abs/1706.03762",
                is_self=False,
            )
        ])
        signals = await RedditCollector(client).collect()
        assert signals[0].url == "https://arxiv.org/abs/1706.03762"

    async def test_saved_self_post_url_is_none(self) -> None:
        # Self-posts have a reddit URL, not an external link — don't include it
        client = make_reddit_client([make_submission(is_self=True)])
        signals = await RedditCollector(client).collect()
        assert signals[0].url is None

    async def test_saved_post_metadata_has_subreddit(self) -> None:
        client = make_reddit_client([
            make_submission(subreddit_name="MachineLearning")
        ])
        signals = await RedditCollector(client).collect()
        assert signals[0].metadata["subreddit"] == "MachineLearning"

    async def test_saved_post_metadata_has_score(self) -> None:
        client = make_reddit_client([make_submission(score=137)])
        signals = await RedditCollector(client).collect()
        assert signals[0].metadata["score"] == 137

    async def test_saved_post_metadata_has_id(self) -> None:
        client = make_reddit_client([make_submission(fullname="t3_abc123")])
        signals = await RedditCollector(client).collect()
        assert signals[0].metadata["fullname"] == "t3_abc123"


# ── Saved comments ────────────────────────────────────────────────────────────

class TestSavedComments:
    async def test_saved_comment_produces_signal(self) -> None:
        client = make_reddit_client([make_comment()])
        signals = await RedditCollector(client).collect()
        assert len(signals) == 1

    async def test_saved_comment_source_is_reddit_comment(self) -> None:
        client = make_reddit_client([make_comment()])
        signals = await RedditCollector(client).collect()
        assert signals[0].source == SignalSource.REDDIT_COMMENT

    async def test_saved_comment_content_is_body(self) -> None:
        body = "Gradient descent updates weights in the direction of steepest loss descent."
        client = make_reddit_client([make_comment(body=body)])
        signals = await RedditCollector(client).collect()
        assert signals[0].content == body

    async def test_saved_comment_url_is_none(self) -> None:
        client = make_reddit_client([make_comment()])
        signals = await RedditCollector(client).collect()
        assert signals[0].url is None

    async def test_saved_comment_metadata_has_subreddit(self) -> None:
        client = make_reddit_client([make_comment(subreddit_name="learnprogramming")])
        signals = await RedditCollector(client).collect()
        assert signals[0].metadata["subreddit"] == "learnprogramming"

    async def test_saved_comment_metadata_has_score(self) -> None:
        client = make_reddit_client([make_comment(score=88)])
        signals = await RedditCollector(client).collect()
        assert signals[0].metadata["score"] == 88


# ── Mixed saved items ─────────────────────────────────────────────────────────

class TestMixedItems:
    async def test_post_and_comment_both_collected(self) -> None:
        client = make_reddit_client([make_submission(), make_comment()])
        signals = await RedditCollector(client).collect()
        assert len(signals) == 2

    async def test_sources_assigned_correctly_in_mixed_list(self) -> None:
        client = make_reddit_client([make_submission(), make_comment()])
        signals = await RedditCollector(client).collect()
        sources = {s.source for s in signals}
        assert SignalSource.REDDIT_POST in sources
        assert SignalSource.REDDIT_COMMENT in sources

    async def test_unknown_item_type_skipped(self) -> None:
        unknown = MagicMock()
        unknown.__class__.__name__ = "Award"
        client = make_reddit_client([make_submission(), unknown, make_comment()])
        signals = await RedditCollector(client).collect()
        assert len(signals) == 2

    async def test_empty_saved_list_returns_empty(self) -> None:
        client = make_reddit_client([])
        signals = await RedditCollector(client).collect()
        assert signals == []


# ── Content filtering ─────────────────────────────────────────────────────────

class TestContentFiltering:
    async def test_deleted_comment_body_skipped(self) -> None:
        # Reddit marks deleted comments with "[deleted]" or "[removed]"
        for deleted_body in ("[deleted]", "[removed]"):
            client = make_reddit_client([make_comment(body=deleted_body)])
            signals = await RedditCollector(client).collect()
            assert signals == [], f"{deleted_body!r} should be filtered"

    async def test_short_content_skipped(self) -> None:
        # body shorter than MIN_CONTENT_LENGTH (10)
        client = make_reddit_client([make_comment(body="ok")])
        signals = await RedditCollector(client).collect()
        assert signals == []

    async def test_post_with_no_title_and_no_selftext_skipped(self) -> None:
        client = make_reddit_client([make_submission(title="", selftext="")])
        signals = await RedditCollector(client).collect()
        assert signals == []


# ── Content truncation ────────────────────────────────────────────────────────

class TestContentTruncation:
    async def test_long_comment_truncated_to_2000(self) -> None:
        client = make_reddit_client([make_comment(body="x" * 5000)])
        signals = await RedditCollector(client).collect()
        assert len(signals[0].content) == 2000

    async def test_long_post_selftext_truncated(self) -> None:
        client = make_reddit_client([
            make_submission(
                title="Short title",
                selftext="y" * 5000,
            )
        ])
        signals = await RedditCollector(client).collect()
        assert len(signals[0].content) <= 2000


# ── Timestamp conversion ──────────────────────────────────────────────────────

class TestTimestampConversion:
    async def test_created_utc_converts_to_utc_datetime(self) -> None:
        client = make_reddit_client([make_submission(created_utc=_TS_2024)])
        signals = await RedditCollector(client).collect()
        expected = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert signals[0].timestamp == expected

    async def test_timestamp_is_utc_aware(self) -> None:
        client = make_reddit_client([make_submission()])
        signals = await RedditCollector(client).collect()
        assert signals[0].timestamp.tzinfo == timezone.utc


# ── Since filtering ───────────────────────────────────────────────────────────

class TestSinceFiltering:
    async def test_item_after_since_included(self) -> None:
        since = datetime(2023, 6, 1, tzinfo=timezone.utc)
        client = make_reddit_client([make_submission(created_utc=_TS_2024)])
        signals = await RedditCollector(client, since=since).collect()
        assert len(signals) == 1

    async def test_item_before_since_excluded(self) -> None:
        since = datetime(2024, 6, 1, tzinfo=timezone.utc)
        client = make_reddit_client([make_submission(created_utc=_TS_2024)])
        signals = await RedditCollector(client, since=since).collect()
        assert signals == []

    async def test_item_exactly_at_since_included(self) -> None:
        since = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        client = make_reddit_client([make_submission(created_utc=_TS_2024)])
        signals = await RedditCollector(client, since=since).collect()
        assert len(signals) == 1

    async def test_since_filters_old_keeps_new(self) -> None:
        since = datetime(2024, 1, 1, tzinfo=timezone.utc)
        client = make_reddit_client([
            make_submission(created_utc=_TS_2023, fullname="t3_old"),  # excluded
            make_submission(created_utc=_TS_2024, fullname="t3_new", title="New post about LLMs and prompting"),  # included
        ])
        signals = await RedditCollector(client, since=since).collect()
        assert len(signals) == 1
        assert signals[0].metadata["fullname"] == "t3_new"


# ── PRAW error handling ───────────────────────────────────────────────────────

class TestPRAWErrorHandling:
    async def test_praw_exception_raises_signal_collection_error(self) -> None:
        import praw.exceptions

        client = MagicMock()
        client.user.me.side_effect = praw.exceptions.PRAWException("auth failed")
        with pytest.raises(SignalCollectionError) as exc_info:
            await RedditCollector(client).collect()
        assert exc_info.value.source in (
            SignalSource.REDDIT_POST,
            SignalSource.REDDIT_COMMENT,
            SignalSource.REDDIT_SAVED,
        )

    async def test_praw_exception_message_preserved(self) -> None:
        import praw.exceptions

        client = MagicMock()
        client.user.me.side_effect = praw.exceptions.PRAWException("rate limited")
        with pytest.raises(SignalCollectionError) as exc_info:
            await RedditCollector(client).collect()
        assert "rate limited" in str(exc_info.value)
