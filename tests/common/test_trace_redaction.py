# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

import json

from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from common.trace_redaction import build_redacted_trace_json

_LONG_TEXT = "X" * 500
_SHORT_TEXT = "short value"


def test_long_user_prompt_string_is_redacted():
    messages = [ModelRequest(parts=[UserPromptPart(content=_LONG_TEXT)])]
    out = json.loads(build_redacted_trace_json(messages))
    content = out[0]["parts"][0]["content"]
    assert _LONG_TEXT not in content
    assert "500 chars" in content


def test_short_user_prompt_string_is_kept():
    messages = [ModelRequest(parts=[UserPromptPart(content=_SHORT_TEXT)])]
    out = json.loads(build_redacted_trace_json(messages))
    assert out[0]["parts"][0]["content"] == _SHORT_TEXT


def test_binary_content_is_redacted_not_base64_inlined():
    raw = b"\x89PNG" * 50
    messages = [ModelRequest(parts=[UserPromptPart(content=["intro", BinaryContent(data=raw, media_type="image/png")])])]
    serialized = build_redacted_trace_json(messages)
    assert raw not in serialized
    out = json.loads(serialized)
    assert "200 bytes, image/png" in out[0]["parts"][0]["content"][1]


def test_long_tool_return_content_is_redacted():
    messages = [
        ModelRequest(parts=[ToolReturnPart(tool_name="feishu_get_doc_content", content=_LONG_TEXT, tool_call_id="tc1")])
    ]
    out = json.loads(build_redacted_trace_json(messages))
    assert _LONG_TEXT not in out[0]["parts"][0]["content"]


def test_long_tool_call_arg_is_redacted():
    messages = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="_generate_test_cases",
                    args={"requirement_doc_content": _LONG_TEXT, "attachment_paths": []},
                    tool_call_id="tc2",
                )
            ]
        )
    ]
    out = json.loads(build_redacted_trace_json(messages))
    args = out[0]["parts"][0]["args"]
    assert _LONG_TEXT not in json.dumps(args)
    assert args["attachment_paths"] == []


def test_model_generated_text_is_never_redacted():
    """The whole point of the trace is to see what the LLM produced — never redact it,
    regardless of length."""
    long_model_output = "Y" * 500
    messages = [ModelResponse(parts=[TextPart(content=long_model_output)])]
    out = json.loads(build_redacted_trace_json(messages))
    assert out[0]["parts"][0]["content"] == long_model_output


def test_system_prompt_is_never_redacted():
    """System prompts are our own authored instructions, not external source content."""
    long_system_prompt = "Z" * 500
    messages = [ModelRequest(parts=[SystemPromptPart(content=long_system_prompt)])]
    out = json.loads(build_redacted_trace_json(messages))
    assert out[0]["parts"][0]["content"] == long_system_prompt
