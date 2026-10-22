# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

import asyncio

from common.streaming import (
    compute_activity_budget,
    current_activity_queue,
    reset_current_activity_queue,
    set_current_activity_queue,
)


def test_compute_activity_budget_doubles_base():
    """Verify that activity budget is correctly computed by doubling the base limit."""
    assert compute_activity_budget(5) == 10
    assert compute_activity_budget(1) == 2
    assert compute_activity_budget(100) == 200


def test_current_activity_queue_default_is_none():
    assert current_activity_queue.get(None) is None


def test_set_and_reset_current_activity_queue():
    queue: asyncio.Queue[str] = asyncio.Queue()
    token = set_current_activity_queue(queue)
    assert current_activity_queue.get(None) is queue
    reset_current_activity_queue(token)
    assert current_activity_queue.get(None) is None


def test_set_current_activity_queue_is_task_local():
    """Two independent contexts have isolated activity queue bindings."""
    import contextvars

    queue_a: asyncio.Queue[str] = asyncio.Queue()
    queue_b: asyncio.Queue[str] = asyncio.Queue()

    ctx_a = contextvars.copy_context()
    ctx_b = contextvars.copy_context()

    ctx_a.run(set_current_activity_queue, queue_a)
    ctx_b.run(set_current_activity_queue, queue_b)

    assert ctx_a.run(current_activity_queue.get, None) is queue_a
    assert ctx_b.run(current_activity_queue.get, None) is queue_b
    # The parent context is unaffected
    assert current_activity_queue.get(None) is None
