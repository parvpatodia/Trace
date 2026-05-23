"""
Pending-approvals queue for Tier B actions (Gmail drafts, Reddit posts).

Tier A actions (Notion, Calendar, Slack) execute automatically.
Tier B actions REQUIRE user approval before execution.  This module
maintains an in-process queue of pending items with approve / reject endpoints
wired into the FastAPI app.

Thread-safety: queue operations use asyncio.Lock — safe for a single async
worker (the APScheduler job runs in the same event loop via asyncio executor).
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_log = logging.getLogger(__name__)

# Maximum pending items to hold in memory (oldest dropped when exceeded).
_MAX_QUEUE_SIZE = 100


@dataclass
class PendingAction:
    """A Tier B action awaiting user approval."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_type: str = ""           # "gmail_draft" | "reddit_post"
    profile_id: str = "default"
    title: str = ""                 # Human-readable description
    preview: str = ""               # Content preview (first 500 chars)
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    pattern_event_type: str = ""    # Which pattern triggered this
    status: str = "pending"         # "pending" | "approved" | "rejected"
    resolved_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "action_type": self.action_type,
            "profile_id": self.profile_id,
            "title": self.title,
            "preview": self.preview,
            "pattern_event_type": self.pattern_event_type,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


class ApprovalsQueue:
    """Thread-safe async queue for pending Tier B actions."""

    def __init__(self) -> None:
        self._queue: list[PendingAction] = []
        self._lock = asyncio.Lock()

    async def enqueue(self, action: PendingAction) -> PendingAction:
        """Add a pending action. Drops the oldest item if queue is full."""
        async with self._lock:
            if len(self._queue) >= _MAX_QUEUE_SIZE:
                dropped = self._queue.pop(0)
                _log.warning("Approvals queue full — dropped oldest: %s", dropped.id)
            self._queue.append(action)
            _log.info(
                "Enqueued Tier B action %s: %s (profile=%s)",
                action.action_type, action.title[:60], action.profile_id,
            )
        return action

    async def list_pending(self, profile_id: str | None = None) -> list[PendingAction]:
        """Return all pending (unresolved) actions, optionally filtered by profile."""
        async with self._lock:
            items = [a for a in self._queue if a.status == "pending"]
            if profile_id:
                items = [a for a in items if a.profile_id == profile_id]
            return list(items)

    async def list_all(self, profile_id: str | None = None) -> list[PendingAction]:
        """Return ALL actions (including resolved), optionally filtered."""
        async with self._lock:
            items = list(self._queue)
            if profile_id:
                items = [a for a in items if a.profile_id == profile_id]
            return items

    async def approve(self, action_id: str) -> PendingAction | None:
        """Mark an action as approved, returning it for execution."""
        async with self._lock:
            for action in self._queue:
                if action.id == action_id and action.status == "pending":
                    action.status = "approved"
                    action.resolved_at = datetime.now(timezone.utc)
                    _log.info("Action approved: %s (%s)", action_id, action.action_type)
                    return action
        _log.warning("Approve: action %s not found or already resolved", action_id)
        return None

    async def reject(self, action_id: str) -> PendingAction | None:
        """Mark an action as rejected."""
        async with self._lock:
            for action in self._queue:
                if action.id == action_id and action.status == "pending":
                    action.status = "rejected"
                    action.resolved_at = datetime.now(timezone.utc)
                    _log.info("Action rejected: %s (%s)", action_id, action.action_type)
                    return action
        _log.warning("Reject: action %s not found or already resolved", action_id)
        return None

    async def get(self, action_id: str) -> PendingAction | None:
        async with self._lock:
            for action in self._queue:
                if action.id == action_id:
                    return action
        return None

    def __len__(self) -> int:
        return len(self._queue)


# ── Module-level singleton — shared across api.py and orchestrator.py ─────────
_approvals_queue: ApprovalsQueue | None = None


def get_approvals_queue() -> ApprovalsQueue:
    global _approvals_queue
    if _approvals_queue is None:
        _approvals_queue = ApprovalsQueue()
    return _approvals_queue
