"""
RedditCollector — reads saved posts and comments from a Reddit account via PRAW.

Design: accepts a pre-constructed `praw.Reddit` instance (dependency injection).
The caller builds the client with credentials; the collector only uses it.
This makes unit testing hermetic (mock the client) and keeps auth concerns out
of the collector.

Signal extraction:
  Submissions (posts):
    source    = SignalSource.REDDIT_POST
    content   = "{title}\\n\\n{selftext}".strip(), truncated to 2000 chars
    url       = external link URL for link posts; None for self-posts
    timestamp = created_utc → UTC datetime

  Comments:
    source    = SignalSource.REDDIT_COMMENT
    content   = body text, truncated to 2000 chars
    url       = None (comment permalinks are relative, not curiosity signals)
    timestamp = created_utc → UTC datetime

Noise filtering removes:
  - Deleted/removed comments: body == "[deleted]" or "[removed]"
  - Content shorter than MIN_CONTENT_LENGTH (10 chars)
  - Unknown item types (praw.models.Award, etc.) — skipped silently
  - Items with created_utc < since (when since is set)

WHY DEPENDENCY INJECTION FOR THE PRAW CLIENT:
  praw.Reddit() reads credentials from environment or .env at construction time.
  Injecting the client lets the caller control auth (env vars, .env, OAuth2)
  and lets tests pass a mock without any credentials at all.

WHY SAVED ITEMS (NOT UPVOTES):
  Saves are intentional bookmarks — the user decided this was worth returning
  to. Upvotes are a weaker signal (agreement, not curiosity). Saved items are
  the strongest Reddit curiosity signal available without scraping post history.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import praw
import praw.exceptions
import praw.models

from trace.models import RawSignal, SignalSource
from trace.signals.base import SignalCollectionError, SignalCollector

_MIN_CONTENT_LENGTH: int = 10
_DELETED_BODIES: frozenset[str] = frozenset({"[deleted]", "[removed]"})


class RedditCollector(SignalCollector):
    """
    Collects curiosity signals from Reddit saved posts and comments.

    Obtain a praw.Reddit client via:
        import praw
        reddit = praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
            username=...,
            password=...,
        )

    Parameters:
        reddit_client: authenticated praw.Reddit instance.
        since: if provided, items with created_utc < since are excluded.
               Must be UTC-aware.
    """

    source = SignalSource.REDDIT_SAVED

    def __init__(
        self,
        reddit_client: praw.Reddit,
        since: datetime | None = None,
    ) -> None:
        if reddit_client is None:
            raise ValueError(
                "RedditCollector: reddit_client must not be None. "
                "Construct a praw.Reddit instance and pass it in."
            )
        if since is not None and since.tzinfo is None:
            raise ValueError(
                "RedditCollector.since must be timezone-aware. "
                "Use datetime(..., tzinfo=timezone.utc)."
            )
        self._client = reddit_client
        self._since = since

    async def collect(self) -> list[RawSignal]:
        try:
            items = await asyncio.to_thread(self._fetch_saved)
        except praw.exceptions.PRAWException as e:
            raise SignalCollectionError(
                self.source, f"Reddit API error: {e}"
            ) from e

        signals: list[RawSignal] = []
        for item in items:
            signal = self._parse_item(item)
            if signal is not None:
                signals.append(signal)
        return signals

    def _fetch_saved(self) -> list[Any]:
        me = self._client.user.me()
        return list(me.saved(limit=None))

    def _parse_item(self, item: Any) -> RawSignal | None:
        if isinstance(item, praw.models.Submission):
            return self._parse_submission(item)
        if isinstance(item, praw.models.Comment):
            return self._parse_comment(item)
        return None

    def _parse_submission(self, post: praw.models.Submission) -> RawSignal | None:
        parts = [post.title or ""]
        if post.selftext:
            parts.append(post.selftext)
        content = "\n\n".join(p for p in parts if p).strip()

        if len(content) < _MIN_CONTENT_LENGTH:
            return None

        ts = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
        if self._since is not None and ts < self._since:
            return None

        # Only attach URL for link posts (external content).
        # Self-post URLs point back to Reddit itself — not a curiosity signal.
        url: str | None = None
        if not post.is_self and post.url.startswith(("http://", "https://")):
            url = post.url

        return RawSignal(
            source=SignalSource.REDDIT_POST,
            content=content[:2000],
            url=url,
            timestamp=ts,
            metadata={
                "subreddit": post.subreddit.display_name,
                "score": post.score,
                "fullname": post.fullname,
            },
        )

    def _parse_comment(self, comment: praw.models.Comment) -> RawSignal | None:
        body = (comment.body or "").strip()

        if body in _DELETED_BODIES:
            return None

        if len(body) < _MIN_CONTENT_LENGTH:
            return None

        ts = datetime.fromtimestamp(comment.created_utc, tz=timezone.utc)
        if self._since is not None and ts < self._since:
            return None

        return RawSignal(
            source=SignalSource.REDDIT_COMMENT,
            content=body[:2000],
            url=None,
            timestamp=ts,
            metadata={
                "subreddit": comment.subreddit.display_name,
                "score": comment.score,
                "fullname": comment.fullname,
            },
        )
