# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Redacts external-source content from a pydantic-ai message history before it is
persisted as a debug trace.

Source documents (Feishu/Jira content, PRD text) reach the LLM either as a
``UserPromptPart``/``ToolReturnPart`` string, a ``BinaryContent`` attachment, or a
``ToolCallPart`` argument (when an agent forwards received content to one of its own
tools). None of that is safe to store verbatim in the orchestrator's in-memory task
history behind only a dashboard JWT, so it is replaced with a size-only placeholder.

The model's own authored output (``TextPart``/``ThinkingPart`` in a ``ModelResponse``)
is never redacted — it is the reason the trace exists.
"""

import dataclasses
from typing import Any

from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

# Hardcoded, not configurable: long enough for short ids/messages, far too small for
# any real source document.
_MAX_TEXT_LENGTH = 200


def _redact_text(value: str) -> str:
    return value if len(value) <= _MAX_TEXT_LENGTH else f"<redacted: {len(value)} chars>"


def _redact_user_content_item(item: Any) -> Any:
    if isinstance(item, BinaryContent):
        return f"<redacted: {len(item.data)} bytes, {item.media_type}>"
    if isinstance(item, str):
        return _redact_text(item)
    return item


def _redact_tool_args(args: Any) -> Any:
    if isinstance(args, str):
        return _redact_text(args)
    if isinstance(args, dict):
        return {key: (_redact_text(value) if isinstance(value, str) else value) for key, value in args.items()}
    return args


def redact_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Return a copy of *messages* with external-source content replaced by placeholders."""
    redacted: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            new_parts = []
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    content = (
                        _redact_text(part.content)
                        if isinstance(part.content, str)
                        else [_redact_user_content_item(item) for item in part.content]
                    )
                    new_parts.append(dataclasses.replace(part, content=content))
                elif isinstance(part, ToolReturnPart) and isinstance(part.content, str):
                    new_parts.append(dataclasses.replace(part, content=_redact_text(part.content)))
                else:
                    # SystemPromptPart / RetryPromptPart carry our own authored content, not
                    # external source material — keep as-is.
                    new_parts.append(part)
            redacted.append(dataclasses.replace(message, parts=new_parts))
        elif isinstance(message, ModelResponse):
            new_parts = [
                dataclasses.replace(part, args=_redact_tool_args(part.args)) if isinstance(part, ToolCallPart) else part
                for part in message.parts
            ]
            redacted.append(dataclasses.replace(message, parts=new_parts))
        else:
            redacted.append(message)
    return redacted


def build_redacted_trace_json(messages: list[ModelMessage]) -> bytes:
    """Serialize *messages* to JSON with external-source content redacted."""
    return ModelMessagesTypeAdapter.dump_json(redact_messages(messages))
