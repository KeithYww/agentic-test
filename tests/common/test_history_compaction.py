# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Compaction never hides unchecked input or splits parallel tool exchanges."""

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from common.design_runtime import StageContext, current_stage
from common.history_compaction import estimated_tokens, prepare_history


def test_oversized_user_input_fails_without_truncation():
    messages = [ModelRequest(parts=[UserPromptPart(content="需求" * 1000)])]
    token = current_stage.set(StageContext(100000, time.monotonic() + 5))
    try:
        with (
            patch("config.PROMPT_INJECTION_CHECK_ENABLED", False),
            patch("config.DesignConfig.CONTEXT_WINDOW", 100),
            pytest.raises(ValueError, match="split the PRD"),
        ):
            prepare_history(messages)
        assert messages[0].parts[0].content == "需求" * 1000
    finally:
        current_stage.reset(token)


def test_old_tool_output_is_checked_before_it_is_cleared():
    messages = [ModelRequest(parts=[ToolReturnPart(tool_name="read", tool_call_id="x", content="unsafe")])]
    guard = MagicMock()
    guard.is_injection.return_value = True
    with (
        patch("config.PROMPT_INJECTION_CHECK_ENABLED", True),
        patch("common.history_compaction.PromptGuardFactory.get_prompt_guard", return_value=guard),
        pytest.raises(HTTPException),
    ):
        prepare_history(messages)
    guard.is_injection.assert_called_once()


def test_parallel_tool_ids_survive_compaction_as_complete_pairs():
    messages = [ModelRequest(parts=[UserPromptPart(content="需求")])]
    for i in range(5):
        ids = [f"{i}a", f"{i}b"]
        messages.append(ModelResponse(parts=[ToolCallPart(tool_name="read", args={}, tool_call_id=v) for v in ids]))
        messages.append(
            ModelRequest(parts=[ToolReturnPart(tool_name="read", tool_call_id=v, content="x" * 500) for v in ids])
        )
    token = current_stage.set(StageContext(100000, time.monotonic() + 5))
    try:
        target = estimated_tokens(messages) - 1500
        with patch("config.PROMPT_INJECTION_CHECK_ENABLED", False), patch("config.DesignConfig.CONTEXT_WINDOW", target):
            compacted = prepare_history(messages)
        calls = [
            p.tool_call_id
            for m in compacted
            if isinstance(m, ModelResponse)
            for p in m.parts
            if isinstance(p, ToolCallPart)
        ]
        returns = [
            p.tool_call_id
            for m in compacted
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart)
        ]
        assert calls == returns
        assert len(calls) >= 6
        assert messages[2].parts[0].content == "x" * 500
    finally:
        current_stage.reset(token)
