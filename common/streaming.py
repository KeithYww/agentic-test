# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Streaming primitives for SSE event payloads and log handler ContextVar.
"""

from contextvars import ContextVar, Token
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    import asyncio

    from common.agent_log_capture import AgentLogCaptureHandler

# ---------------------------------------------------------------------------
# ContextVars
# ---------------------------------------------------------------------------

current_log_handler: ContextVar["AgentLogCaptureHandler | None"] = ContextVar("current_log_handler", default=None)


def set_current_log_handler(handler: "AgentLogCaptureHandler") -> Token:
    """Bind log handler to the current context; returns a reset token."""
    return current_log_handler.set(handler)


def reset_current_log_handler(token: Token) -> None:
    """Reset the log handler binding using the token returned by set_current_log_handler."""
    current_log_handler.reset(token)


# Per-request activity queue: executor binds a fresh queue before each agent.run(); the
# agent's report_activity reads from this ContextVar so concurrent tasks route their
# activity updates to the correct queue.
# Routing relies on asyncio Task context inheritance. threading.Thread callers will see
# None here and report_activity will fall back to the agent's instance queue.
current_activity_queue: ContextVar["asyncio.Queue[str] | None"] = ContextVar(
    "current_activity_queue", default=None
)


def set_current_activity_queue(queue: "asyncio.Queue[str]") -> Token:
    """Bind per-request activity queue to the current context; returns a reset token."""
    return current_activity_queue.set(queue)


def reset_current_activity_queue(token: Token) -> None:
    """Reset the activity queue binding using the token returned by set_current_activity_queue."""
    current_activity_queue.reset(token)


# Per-request task ID: set by executor so any logger in the call chain can emit it as a
# structured field without threading it through the call stack.
current_task_id: ContextVar["str | None"] = ContextVar("current_task_id", default=None)


def set_current_task_id(task_id: str) -> Token:
    """Bind task_id to the current context; returns a reset token."""
    return current_task_id.set(task_id)


def reset_current_task_id(token: Token) -> None:
    """Reset the task_id binding."""
    current_task_id.reset(token)


# ---------------------------------------------------------------------------
# Budget helper
# ---------------------------------------------------------------------------


def compute_activity_budget(base_limit: int) -> int:
    """Return the tool-calls limit that accounts for report_activity calls.

    UsageLimits offers no per-tool exclusion, so report_activity calls count against the
    same tool_calls_limit as real tool calls. Doubling base_limit assumes a roughly 1:1
    report_activity-to-real-tool pairing; for agents that rarely call report_activity this
    weakens the effective cap (it approaches 2x the intended real-tool budget).
    """
    return base_limit * 2


# ---------------------------------------------------------------------------
# SSE payload event models
# ---------------------------------------------------------------------------


class AgentActivityEvent(BaseModel):
    version: int = 1
    type: str = "agent_activity"
    task_id: str
    agent_id: str
    text: str


class LogBatchEvent(BaseModel):
    version: int = 1
    type: str = "log_batch"
    task_id: str
    lines: list[str]


class TaskDoneEvent(BaseModel):
    version: int = 1
    type: str = "task_done"
    task_id: str
    agent_id: str
    status: str
    error_message: str | None = None


class GapEvent(BaseModel):
    version: int = 1
    type: str = "gap"
    reason: str
    since: int
    until: int


class AgentSnapshot(BaseModel):
    id: str
    name: str
    status: str
    current_task_id: str | None = None


class RunningTaskSnapshot(BaseModel):
    task_id: str
    agent_id: str
    description: str
    current_activity: str | None = None


class SnapshotEvent(BaseModel):
    version: int = 1
    agents: list[AgentSnapshot] = []
    running_tasks: list[RunningTaskSnapshot] = []


class AuthErrorEvent(BaseModel):
    version: int = 1
    type: str = "auth_error"
