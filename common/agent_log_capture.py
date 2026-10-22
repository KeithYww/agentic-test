# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
In-memory log handler for capturing agent execution logs.

This module provides a log handler that captures log records during agent
execution and can export them as a string for inclusion in task artifacts.
"""

import logging
import threading
from collections import deque

from common.streaming import current_log_handler


class AgentLogCaptureHandler(logging.Handler):
    """
    A logging handler that captures log records in memory during agent execution.

    This handler is designed to be attached temporarily to a logger during agent
    task execution, then detached and its logs extracted to be returned as artifacts.
    """

    def __init__(self, max_records: int = 10000):
        super().__init__()
        self._buffer: deque[str] = deque(maxlen=max_records)
        self._lock = threading.Lock()
        self._drain_cursor: int = 0
        self._emitted_total: int = 0
        self.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        """Store the formatted log record in the buffer.

        Only captures records when the current asyncio Task context is bound to
        this handler via the current_log_handler ContextVar. This ensures that
        concurrent tasks with separate handlers do not cross-contaminate each
        other's log buffers.

        Note: this routing relies on asyncio Task context inheritance. Log calls
        from threading.Thread will see current_log_handler=None and be silently
        dropped — this is an intentional trade-off documented here to prevent
        future callers from being surprised.
        """
        if current_log_handler.get(None) is not self:
            return
        try:
            log_entry = self.format(record)
            with self._lock:
                self._buffer.append(log_entry)
                self._emitted_total += 1
        except Exception:
            self.handleError(record)

    def drain(self) -> list[str]:
        """Return lines appended since the last drain() and advance the cursor.

        Thread-safe. Tracks emitted records by a monotonic total so draining keeps
        working after the bounded buffer overflows: when more lines were emitted than
        the buffer can hold, the oldest are unrecoverable and only the buffered tail
        is returned.
        """
        with self._lock:
            new_count = self._emitted_total - self._drain_cursor
            self._drain_cursor = self._emitted_total
            if new_count <= 0:
                return []
            buf_len = len(self._buffer)
            k = min(new_count, buf_len)
            return [self._buffer[buf_len - i] for i in range(k, 0, -1)]
