"""Unit tests for trace/agent/approvals.py — async queue, no external I/O."""
from __future__ import annotations

import pytest

from trace.agent.approvals import ApprovalsQueue, PendingAction, get_approvals_queue


def _make_action(**kwargs) -> PendingAction:
    defaults = {
        "action_type": "gmail_draft",
        "profile_id": "default",
        "title": "Test action",
        "preview": "Preview text",
        "pattern_event_type": "subscription_debt",
    }
    defaults.update(kwargs)
    return PendingAction(**defaults)


class TestPendingAction:
    def test_to_dict_has_required_keys(self):
        a = _make_action()
        d = a.to_dict()
        for key in ("id", "action_type", "profile_id", "title", "status", "created_at"):
            assert key in d

    def test_default_status_pending(self):
        a = _make_action()
        assert a.status == "pending"

    def test_resolved_at_none_by_default(self):
        a = _make_action()
        assert a.resolved_at is None


class TestApprovalsQueue:
    @pytest.fixture
    def queue(self):
        return ApprovalsQueue()

    @pytest.mark.asyncio
    async def test_enqueue_and_list_pending(self, queue):
        a = _make_action(title="action 1")
        await queue.enqueue(a)
        pending = await queue.list_pending()
        assert len(pending) == 1
        assert pending[0].title == "action 1"

    @pytest.mark.asyncio
    async def test_approve_marks_approved(self, queue):
        a = _make_action()
        await queue.enqueue(a)
        result = await queue.approve(a.id)
        assert result is not None
        assert result.status == "approved"
        assert result.resolved_at is not None

    @pytest.mark.asyncio
    async def test_reject_marks_rejected(self, queue):
        a = _make_action()
        await queue.enqueue(a)
        result = await queue.reject(a.id)
        assert result is not None
        assert result.status == "rejected"

    @pytest.mark.asyncio
    async def test_approved_action_not_in_pending(self, queue):
        a = _make_action()
        await queue.enqueue(a)
        await queue.approve(a.id)
        pending = await queue.list_pending()
        assert not any(x.id == a.id for x in pending)

    @pytest.mark.asyncio
    async def test_approve_nonexistent_returns_none(self, queue):
        result = await queue.approve("does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_reject_nonexistent_returns_none(self, queue):
        result = await queue.reject("does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_pending_filters_by_profile(self, queue):
        a1 = _make_action(profile_id="user1")
        a2 = _make_action(profile_id="user2")
        await queue.enqueue(a1)
        await queue.enqueue(a2)
        user1_pending = await queue.list_pending(profile_id="user1")
        assert all(a.profile_id == "user1" for a in user1_pending)

    @pytest.mark.asyncio
    async def test_queue_max_size_drops_oldest(self, queue):
        from trace.agent import approvals as _approvals_mod
        original_max = _approvals_mod._MAX_QUEUE_SIZE
        _approvals_mod._MAX_QUEUE_SIZE = 3
        try:
            ids = []
            for i in range(4):
                a = _make_action(title=f"action {i}")
                await queue.enqueue(a)
                ids.append(a.id)
            all_items = await queue.list_all()
            all_ids = {x.id for x in all_items}
            # Oldest (ids[0]) should have been dropped.
            assert ids[0] not in all_ids
            assert ids[3] in all_ids
        finally:
            _approvals_mod._MAX_QUEUE_SIZE = original_max

    @pytest.mark.asyncio
    async def test_get_by_id(self, queue):
        a = _make_action()
        await queue.enqueue(a)
        found = await queue.get(a.id)
        assert found is not None
        assert found.id == a.id

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, queue):
        result = await queue.get("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_double_approve_noop(self, queue):
        a = _make_action()
        await queue.enqueue(a)
        r1 = await queue.approve(a.id)
        r2 = await queue.approve(a.id)
        assert r1 is not None
        assert r2 is None  # Already resolved

    def test_len(self):
        q = ApprovalsQueue()
        assert len(q) == 0


class TestGetApprovalsQueue:
    def test_singleton(self):
        q1 = get_approvals_queue()
        q2 = get_approvals_queue()
        assert q1 is q2
