# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Model routing honors explicit fixed-model and existing fallback policies."""

import json

import httpx
import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.settings import ThinkingLevel

from common.llm_provider import get_model


@pytest.mark.asyncio
async def test_bounded_sdk_error_has_one_http_attempt_and_no_fallback(monkeypatch):
    import time

    from common.design_runtime import StageContext, current_stage

    calls = []

    def respond(request):
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "test"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        provider = AnthropicProvider(api_key="test", base_url="https://relay.invalid", http_client=http)
        monkeypatch.setattr("common.llm_provider.AnthropicProvider", lambda **kwargs: provider)
        model = get_model("gpt-5.6-terra", "deepseek-v4-flash", bounded_requests=True)
        token = current_stage.set(StageContext(10000, time.monotonic() + 5))
        try:
            with pytest.raises(ModelHTTPError):
                await model.request([ModelRequest(parts=[UserPromptPart("test")])], {"max_tokens": 100}, ModelRequestParameters())
        finally:
            current_stage.reset(token)
    assert calls == ["gpt-5.6-terra"]


def test_single_model_has_no_fallback() -> None:
    model = get_model("deepseek-v4-flash")
    assert isinstance(model, AnthropicModel)
    assert model.model_name == "deepseek-v4-flash"


def test_two_model_policy_keeps_fallback() -> None:
    model = get_model("gpt-5.6-terra", "deepseek-v4-flash")
    assert isinstance(model, FallbackModel)
    assert model.model_name == "fallback:gpt-5.6-terra,deepseek-v4-flash"


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", ["low", "high"])
@pytest.mark.parametrize("model_name", ["deepseek-v4-flash", "deepseek-v4.1-flash"])
async def test_deepseek_sends_requested_effort_on_wire(
    monkeypatch: pytest.MonkeyPatch, thinking: ThinkingLevel, model_name: str,
) -> None:
    """The relay must receive effort, rather than only an ignored thinking budget."""
    requests: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "test-response", "type": "message", "role": "assistant",
            "model": model_name, "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        provider = AnthropicProvider(api_key="test", base_url="https://relay.invalid", http_client=http)
        monkeypatch.setattr("common.llm_provider.AnthropicProvider", lambda **kwargs: provider)
        model = get_model(model_name)
        await model.request(
            [ModelRequest(parts=[UserPromptPart("Return ok")])],
            {"thinking": thinking, "max_tokens": 65536, "timeout": 300}, ModelRequestParameters(),
        )

    assert len(requests) == 1
    assert requests[0]["model"] == model_name
    assert requests[0]["output_config"]["effort"] == thinking
    assert requests[0]["max_tokens"] == 65536
