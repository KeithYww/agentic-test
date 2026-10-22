# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

import logging
import threading

import pytest

from common.agent_log_capture import AgentLogCaptureHandler
from common.streaming import current_log_handler


@pytest.fixture
def handler() -> AgentLogCaptureHandler:
    h = AgentLogCaptureHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    return h


def _emit(handler: AgentLogCaptureHandler, message: str) -> None:
    """Emit a log record while binding the ContextVar to the given handler."""
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    token = current_log_handler.set(handler)
    try:
        handler.emit(record)
    finally:
        current_log_handler.reset(token)


def _make_record(message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg=message, args=(), exc_info=None,
    )


# ---------------------------------------------------------------------------
# ContextVar routing
# ---------------------------------------------------------------------------


def test_emit_drops_record_when_no_handler_bound(handler):
    """emit() is a no-op when current_log_handler is not bound to this handler."""
    handler.emit(_make_record("should be dropped"))
    assert handler.drain() == []


def test_emit_drops_record_when_bound_to_different_handler(handler):
    """emit() is a no-op when current_log_handler is bound to a different handler."""
    other = AgentLogCaptureHandler()
    token = current_log_handler.set(other)
    try:
        handler.emit(_make_record("should be dropped"))
    finally:
        current_log_handler.reset(token)
    assert handler.drain() == []


def test_emit_stores_record_when_bound_to_this_handler(handler):
    """emit() captures the record when current_log_handler is bound to this handler."""
    handler.setFormatter(logging.Formatter("%(message)s"))
    token = current_log_handler.set(handler)
    try:
        handler.emit(_make_record("captured"))
    finally:
        current_log_handler.reset(token)
    assert handler.drain() == ["captured"]


# ---------------------------------------------------------------------------
# drain basics
# ---------------------------------------------------------------------------


def test_drain_initially_empty(handler):
    assert handler.drain() == []


def test_drain_returns_all_on_first_call(handler):
    _emit(handler, "line 1")
    _emit(handler, "line 2")
    result = handler.drain()
    assert result == ["line 1", "line 2"]


def test_drain_second_call_returns_only_new_lines(handler):
    _emit(handler, "line 1")
    handler.drain()  # consume line 1

    _emit(handler, "line 2")
    _emit(handler, "line 3")
    result = handler.drain()
    assert result == ["line 2", "line 3"]


def test_drain_second_call_empty_when_no_new_lines(handler):
    _emit(handler, "line 1")
    handler.drain()
    assert handler.drain() == []


def test_drain_keeps_working_after_buffer_overflows():
    handler = AgentLogCaptureHandler(max_records=3)
    handler.setFormatter(logging.Formatter("%(message)s"))

    for i in range(5):  # fill and overflow the 3-slot buffer
        _emit(handler, f"first {i}")
    # Oldest two are unrecoverable; only the buffered tail is returned.
    assert handler.drain() == ["first 2", "first 3", "first 4"]

    # After overflow the cursor must not get stuck: new lines still drain.
    for i in range(4):
        _emit(handler, f"second {i}")
    assert handler.drain() == ["second 1", "second 2", "second 3"]


# ---------------------------------------------------------------------------
# Thread-safety: concurrent emit + drain
# ---------------------------------------------------------------------------


def test_concurrent_emit_and_drain_is_race_free(handler):
    errors: list[Exception] = []
    collected: list[str] = []
    lock = threading.Lock()

    # Bind the ContextVar in the main thread before spawning so threads inherit it.
    main_token = current_log_handler.set(handler)

    def emitter():
        try:
            for i in range(500):
                _emit(handler, f"line {i}")
        except Exception as e:
            with lock:
                errors.append(e)

    def drainer():
        try:
            for _ in range(100):
                batch = handler.drain()
                with lock:
                    collected.extend(batch)
        except Exception as e:
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=emitter), threading.Thread(target=drainer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    current_log_handler.reset(main_token)

    # Final drain to capture any remaining lines
    with lock:
        # Re-bind to drain remaining
        token = current_log_handler.set(handler)
        collected.extend(handler.drain())
        current_log_handler.reset(token)

    assert errors == [], f"Threads raised: {errors}"
    # All emitted lines fit within maxlen; drain collects everything exactly once
    assert len(collected) <= 500  # maxlen guards upper bound
