# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Conservative, protocol-preserving history preparation for long QA tasks."""

import dataclasses
import hashlib

from fastapi import HTTPException
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

import config
from common.design_runtime import current_stage
from common.prompt_injection.guard import GuardPrompt, PromptGuardFactory


def validate_external_content(messages: list[ModelMessage]) -> None:
    """Check every external part before it can be removed by compaction."""
    if not config.PROMPT_INJECTION_CHECK_ENABLED:
        return
    context = current_stage.get()
    guard = PromptGuardFactory.get_prompt_guard(config.PROMPT_GUARD_PROVIDER)
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if not isinstance(part, UserPromptPart | ToolReturnPart):
                continue
            content = part.content if isinstance(part.content, str) else str(part.content)
            digest = hashlib.sha256(content.encode()).hexdigest()
            if context and digest in context.checked_content:
                continue
            if guard.is_injection(
                GuardPrompt(prompt_description="External QA input", prompt=content), config.PROMPT_INJECTION_MIN_SCORE
            ):
                raise HTTPException(status_code=400, detail="Prompt injection attack detected.")
            if context:
                context.checked_content.add(digest)


def estimated_tokens(messages: list[ModelMessage]) -> int:
    """Use a conservative UTF-8 upper estimate without guessing provider tokenizers."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    return len(ModelMessagesTypeAdapter.dump_json(messages))


def prepare_history(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Clear old tool payloads, then remove only complete tool exchanges."""
    validate_external_content(messages)
    if current_stage.get() is None or config.DesignConfig.CONTEXT_WINDOW <= 0:
        return messages
    target = int(config.DesignConfig.CONTEXT_WINDOW * 0.9)
    if estimated_tokens(messages) <= target:
        return messages
    calls = [
        p.tool_call_id for m in messages if isinstance(m, ModelResponse) for p in m.parts if isinstance(p, ToolCallPart)
    ]
    keep = set(calls[-6:])
    result = []
    for message in messages:
        if isinstance(message, ModelRequest):
            parts = [
                dataclasses.replace(p, content="[Earlier tool result cleared; use current stage state]")
                if isinstance(p, ToolReturnPart) and isinstance(p.content, str) and p.tool_call_id not in keep
                else p
                for p in message.parts
            ]
            result.append(dataclasses.replace(message, parts=parts))
        else:
            result.append(message)
    if estimated_tokens(result) <= target:
        return result
    # Delete only adjacent, fully closed exchanges. User/system instructions are never removed.
    i = 0
    while i < len(result) - 1 and estimated_tokens(result) > target:
        response, request = result[i : i + 2]
        if isinstance(response, ModelResponse) and isinstance(request, ModelRequest):
            call_ids = {p.tool_call_id for p in response.parts if isinstance(p, ToolCallPart)}
            return_ids = {p.tool_call_id for p in request.parts if isinstance(p, ToolReturnPart | RetryPromptPart)}
            protected = any(isinstance(p, UserPromptPart | SystemPromptPart) for p in request.parts)
            if call_ids and call_ids == return_ids and not protected and not call_ids & keep:
                del result[i : i + 2]
                continue
        i += 1
    if estimated_tokens(result) > target:
        raise ValueError("QA input exceeds the configured context window; split the PRD instead of truncating it")
    return result
