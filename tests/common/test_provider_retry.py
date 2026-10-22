# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Provider HTTP retries preserve agent progress and unknown-usage protection."""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.providers.anthropic import AnthropicProvider

import config
from common import design_runtime, design_store
from common.custom_llm_wrapper import CustomLlmWrapper
from common.design_runtime import StageContext, current_stage
from common.llm_provider import get_model
from common.provider_errors import ProviderRetryError
from common.provider_policy import ProviderRetryPolicy


def model_response(content: list[dict], stop_reason: str = "end_turn") -> dict:
    """Create an Anthropic response with measurable usage."""
    return {
        "id": "reply", "type": "message", "role": "assistant", "model": "deepseek-v4-flash",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }


@pytest.fixture
def retry_enabled(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Opt into the explicit test relay contract without real waiting."""
    monkeypatch.setattr(ProviderRetryPolicy, "PHOTON_OVERLOAD_RETRY_ENABLED", True, raising=False)
    monkeypatch.setattr(ProviderRetryPolicy, "PHOTON_REJECTION_CONTRACT", "photon-preflight-v1", raising=False)
    sleep = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep)
    monkeypatch.setattr(design_runtime, "_model_semaphore", None)
    return sleep


@asynccontextmanager
async def relay(monkeypatch, respond):
    """Use the real SDK/adapter with only the external HTTP boundary replaced."""
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        provider = AnthropicProvider(api_key="test", base_url="https://relay.invalid", http_client=http)
        monkeypatch.setattr("common.llm_provider.AnthropicProvider", lambda **kwargs: provider)
        yield CustomLlmWrapper(get_model("deepseek-v4-flash", "qwen3.8-flash"), max_tokens=100)


async def request(wrapper):
    return await wrapper.request([ModelRequest(parts=[UserPromptPart("test")])], None, ModelRequestParameters())


@pytest.mark.asyncio
async def test_overload_after_tool_retries_only_the_failed_http_request(monkeypatch, retry_enabled):
    calls = []
    tools = []

    def mark() -> str:
        """Record a completed tool action."""
        tools.append("completed")
        return "saved"

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(200, json=model_response(
                [{"type": "tool_use", "id": "call-1", "name": "mark", "input": {}}], "tool_use",
            ))
        if len(calls) == 2:
            return httpx.Response(503, json={"error": {"type": "system_cpu_overloaded", "message": "busy"}})
        return httpx.Response(200, json=model_response([{"type": "text", "text": "done"}]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        provider = AnthropicProvider(api_key="test", base_url="https://relay.invalid", http_client=http)
        monkeypatch.setattr("common.llm_provider.AnthropicProvider", lambda **kwargs: provider)
        model = get_model("deepseek-v4-flash", "qwen3.8-flash", bounded_requests=True)
        agent = Agent(model=CustomLlmWrapper(model), tools=[mark])
        result = await agent.run("Save once, then confirm.")

    assert result.output == "done"
    assert tools == ["completed"]
    assert len(calls) == 3
    assert calls[1] == calls[2]
    assert [call["model"] for call in calls] == ["deepseek-v4-flash"] * 3
    retry_enabled.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,kind,contract,enabled,count,code", [
    (503, "system_cpu_overloaded", "photon-preflight-v1", True, 3, "provider_busy_exhausted"),
    (503, "system_cpu_overloaded", "photon-preflight-v1", False, 1, "provider_busy_exhausted"),
    (503, "system_cpu_overloaded", "", True, 1, "provider_usage_unknown"),
    (503, "new_api_error", "photon-preflight-v1", True, 1, "provider_usage_unknown"),
    (429, "system_cpu_overloaded", "photon-preflight-v1", True, 1, "provider_usage_unknown"),
    (401, "authentication_error", "photon-preflight-v1", True, 1, "provider_usage_unknown"),
])
async def test_actual_http_attempts_are_bounded_and_unknown_never_falls_back(
    monkeypatch, retry_enabled, status, kind, contract, enabled, count, code,
):
    monkeypatch.setattr(ProviderRetryPolicy, "PHOTON_REJECTION_CONTRACT", contract)
    monkeypatch.setattr(ProviderRetryPolicy, "PHOTON_OVERLOAD_RETRY_ENABLED", enabled)
    calls = []

    def respond(req):
        calls.append(json.loads(req.content)["model"])
        return httpx.Response(status, json={"error": {"type": kind, "message": "system cpu overloaded"}})

    async with relay(monkeypatch, respond) as wrapper:
        with pytest.raises(ProviderRetryError) as error:
            await request(wrapper)
    assert error.value.code == code
    assert calls == ["deepseek-v4-flash"] * count
    assert retry_enabled.await_count == count - 1


@pytest.mark.asyncio
@pytest.mark.parametrize("header,code", [
    ("120", "provider_busy_exhausted"),
    ("Fri, 31 Dec 2099 23:59:59 GMT", "provider_retry_deadline"),
])
async def test_retry_after_beyond_budget_stops_without_early_send(monkeypatch, retry_enabled, header, code):
    calls = []

    def respond(req):
        calls.append(req)
        return httpx.Response(503, headers={"retry-after": header}, json={"error": {"type": "system_cpu_overloaded"}})

    async with relay(monkeypatch, respond) as wrapper:
        with pytest.raises(ProviderRetryError, match=code):
            await request(wrapper)
    assert len(calls) == 1
    retry_enabled.assert_not_awaited()


@pytest.mark.asyncio
async def test_valid_retry_after_and_jitter_are_respected(monkeypatch, retry_enabled):
    monkeypatch.setattr("common.provider_retry.random.uniform", lambda start, end: 1)
    calls = []

    def respond(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(503, headers={"retry-after": "12"}, json={"error": {"type": "system_cpu_overloaded"}})
        return httpx.Response(200, json=model_response([{"type": "text", "text": "ok"}]))

    async with relay(monkeypatch, respond) as wrapper:
        await request(wrapper)
    retry_enabled.assert_awaited_once_with(13)


@pytest.mark.asyncio
async def test_persistent_rejection_then_success_replays_without_another_paid_call(tmp_path, monkeypatch, retry_enabled):
    owner = design_store.DesignStore(str(tmp_path))
    await owner.call("open")
    run, _ = await owner.call("create_run", "retry-story", {}, 10000, 100)
    monkeypatch.setattr(design_store, "store", owner)
    monkeypatch.setattr("common.provider_retry.retry_delay", lambda ordinal, failure: 0)
    context = StageContext(10000, time.monotonic() + 5, run_id=run["id"], generation=1,
                           task_key="classify-0", budget_ceiling=10000)
    token = current_stage.set(context)
    calls = []

    def respond(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(503, json={"error": {"type": "system_cpu_overloaded"}})
        return httpx.Response(200, json=model_response([{"type": "text", "text": "ok"}]))

    try:
        async with relay(monkeypatch, respond) as wrapper:
            first = await request(wrapper)
            replay = await request(wrapper)
        assert replay.parts == first.parts
        assert len(calls) == 2
        assert context.usage.requests == 2 and context.usage.total_tokens == 12
        assert context.usage.complete and context.reserved == 0
        actual = await owner.call("run", run["id"])
        assert actual["used"] == 12 and actual["reserved"] == 0 and not actual["usage_unknown"]
        attempts = await owner.call("provider_attempts", run["id"])
        assert [attempt["state"] for attempt in attempts] == ["rejected", "settled"]
        assert [attempt["used_tokens"] for attempt in attempts] == [0, 12]
    finally:
        current_stage.reset(token)
        await owner.close()


@pytest.mark.asyncio
async def test_generic_overload_retains_unknown_reservation(tmp_path, monkeypatch, retry_enabled):
    owner = design_store.DesignStore(str(tmp_path))
    await owner.call("open")
    run, _ = await owner.call("create_run", "unknown-story", {}, 10000, 100)
    monkeypatch.setattr(design_store, "store", owner)
    context = StageContext(10000, time.monotonic() + 5, run_id=run["id"], generation=1,
                           task_key="classify-0", budget_ceiling=10000)
    token = current_stage.set(context)
    calls = []

    def respond(req):
        calls.append(req)
        return httpx.Response(503, json={"error": {"type": "new_api_error", "message": "system cpu overloaded"}})

    try:
        async with relay(monkeypatch, respond) as wrapper:
            with pytest.raises(ProviderRetryError, match="provider_usage_unknown"):
                await request(wrapper)
            with pytest.raises(ProviderRetryError, match="provider_usage_unknown"):
                await request(wrapper)
        assert len(calls) == 1
        assert not context.usage.complete
        actual = await owner.call("run", run["id"])
        assert actual["usage_unknown"] and actual["reserved"] > 0
    finally:
        current_stage.reset(token)
        await owner.close()


@pytest.mark.asyncio
async def test_retry_wait_releases_slots_and_cancel_does_not_send_again(monkeypatch, retry_enabled):
    waiting = asyncio.Event()
    release = asyncio.Event()

    async def sleep(delay):
        waiting.set()
        await release.wait()

    monkeypatch.setattr("common.provider_retry.asyncio.sleep", sleep)
    monkeypatch.setattr(design_store, "store", None)
    context = StageContext(10000, time.monotonic() + 30)
    context.semaphore = asyncio.Semaphore(1)
    token = current_stage.set(context)
    calls = []

    def respond(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(503, json={"error": {"type": "system_cpu_overloaded"}})
        return httpx.Response(200, json=model_response([{"type": "text", "text": "ok"}]))

    try:
        async with relay(monkeypatch, respond) as wrapper:
            first = asyncio.create_task(request(wrapper))
            await waiting.wait()
            assert context.reserved == 0 and not context.semaphore.locked()
            await asyncio.wait_for(request(wrapper), timeout=1)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        assert len(calls) == 2 and context.usage.complete
    finally:
        current_stage.reset(token)


@pytest.mark.asyncio
async def test_cancel_during_reservation_releases_unsent_attempt(tmp_path, monkeypatch, retry_enabled):
    owner = design_store.DesignStore(str(tmp_path))
    await owner.call("open")
    run, _ = await owner.call("create_run", "unsent-story", {}, 10000, 100)
    monkeypatch.setattr(design_store, "store", owner)
    original = owner.call
    reserved, release = asyncio.Event(), asyncio.Event()

    async def held_call(action, *args):
        result = await original(action, *args)
        if action == "provider_reserve" and not reserved.is_set():
            reserved.set()
            await release.wait()
        return result

    monkeypatch.setattr(owner, "call", held_call)
    context = StageContext(10000, time.monotonic() + 5, run_id=run["id"], generation=1,
                           task_key="classify-0", budget_ceiling=10000)
    token = current_stage.set(context)
    calls = []

    def respond(req):
        calls.append(req)
        return httpx.Response(200, json=model_response([{"type": "text", "text": "ok"}]))

    try:
        async with relay(monkeypatch, respond) as wrapper:
            task = asyncio.create_task(request(wrapper))
            await reserved.wait()
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not calls and context.usage.complete
            actual = await owner.call("run", run["id"])
            assert actual["reserved"] == 0 and not actual["usage_unknown"]
            monkeypatch.setattr(ProviderRetryPolicy, "PHOTON_OVERLOAD_RETRY_ENABLED", False)
            await request(wrapper)
        assert len(calls) == 1
        attempts = await owner.call("provider_attempts", run["id"])
        assert [item["state"] for item in attempts] == ["not_sent", "settled"]
    finally:
        current_stage.reset(token)
        await owner.close()


@pytest.mark.asyncio
async def test_cancel_during_settlement_confirms_commit_before_propagating(tmp_path, monkeypatch, retry_enabled):
    owner = design_store.DesignStore(str(tmp_path))
    await owner.call("open")
    run, _ = await owner.call("create_run", "cancel-story", {}, 10000, 100)
    monkeypatch.setattr(design_store, "store", owner)
    original_call = owner.call
    settling, release = asyncio.Event(), asyncio.Event()

    async def held_call(action, *args):
        result = await original_call(action, *args)
        if action == "provider_finish" and args[-1]["state"] == "settled":
            settling.set()
            await release.wait()
        return result

    monkeypatch.setattr(owner, "call", held_call)
    context = StageContext(10000, time.monotonic() + 5, run_id=run["id"], generation=1,
                           task_key="classify-0", budget_ceiling=10000)
    token = current_stage.set(context)
    try:
        async with relay(monkeypatch, lambda req: httpx.Response(
            200, json=model_response([{"type": "text", "text": "ok"}]),
        )) as wrapper:
            task = asyncio.create_task(request(wrapper))
            await asyncio.wait_for(settling.wait(), 2)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        actual = await original_call("run", run["id"])
        assert actual["used"] == 12 and not actual["usage_unknown"]
        assert context.usage.total_tokens == 12 and context.usage.complete
        assert context.reserved == 0
        assert (await original_call("provider_attempts", run["id"]))[0]["state"] == "settled"
    finally:
        current_stage.reset(token)
        await owner.close()


def sse(complete: bool) -> httpx.Response:
    """Build a streamed reply; a truncated one ends after its thinking block, as the relay was observed to do."""
    events = [
        {"type": "message_start", "message": {
            "id": "msg_cut" if not complete else "msg_ok", "type": "message", "role": "assistant",
            "model": "claude-sonnet-5", "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 2}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_stop", "index": 0},
    ]
    if complete:
        events += [
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "ok"}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 2}},
            {"type": "message_stop"},
        ]
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


@asynccontextmanager
async def streaming_relay(monkeypatch, replies):
    calls = []

    def respond(req):
        calls.append(json.loads(req.content))
        return sse(replies[len(calls) - 1])

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        provider = AnthropicProvider(api_key="test", base_url="https://relay.invalid", http_client=http)
        monkeypatch.setattr("common.llm_provider.AnthropicProvider", lambda **kwargs: provider)
        # Above the SDK's non-streaming ceiling, so the request is streamed like the review stage.
        yield CustomLlmWrapper(get_model("claude-sonnet-5", bounded_requests=True), max_tokens=32000), calls


async def durable_stage(tmp_path, monkeypatch):
    owner = design_store.DesignStore(str(tmp_path))
    await owner.call("open")
    run, _ = await owner.call("create_run", "truncated-story", {}, 100000, 100)
    monkeypatch.setattr(design_store, "store", owner)
    context = StageContext(100000, time.monotonic() + 5, run_id=run["id"], generation=1,
                           task_key="review-0", budget_ceiling=100000)
    return owner, run, context


@pytest.mark.asyncio
async def test_truncated_stream_is_resent_once_and_charged_at_its_upper_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(design_runtime, "_model_semaphore", None)
    owner, run, context = await durable_stage(tmp_path, monkeypatch)
    token = current_stage.set(context)
    try:
        async with streaming_relay(monkeypatch, [False, True]) as (wrapper, calls):
            response = await request(wrapper)
        assert [part.content for part in response.parts if hasattr(part, "content")][-1] == "ok"
        assert len(calls) == 2 and calls[0] == calls[1] and calls[0]["stream"]
        assert context.usage.complete and context.reserved == 0
        assert context.usage.total_tokens == (10 + 32000) + 12
        actual = await owner.call("run", run["id"])
        assert (actual["used"], actual["reserved"], actual["usage_unknown"]) == (32022, 0, False)
        attempts = await owner.call("provider_attempts", run["id"])
        assert [(a["state"], a["used_tokens"], a["provider_request_id"]) for a in attempts] == [
            ("incomplete", 32010, "msg_cut"), ("settled", 12, None),
        ]
    finally:
        current_stage.reset(token)
        await owner.close()


@pytest.mark.asyncio
async def test_repeated_truncation_stops_with_known_usage(tmp_path, monkeypatch):
    monkeypatch.setattr(design_runtime, "_model_semaphore", None)
    owner, run, context = await durable_stage(tmp_path, monkeypatch)
    token = current_stage.set(context)
    try:
        async with streaming_relay(monkeypatch, [False, False, True]) as (wrapper, calls):
            with pytest.raises(ProviderRetryError, match="provider_stream_incomplete"):
                await request(wrapper)
        assert len(calls) == 2
        assert context.usage.complete and context.reserved == 0
        actual = await owner.call("run", run["id"])
        assert (actual["used"], actual["reserved"], actual["usage_unknown"]) == (64020, 0, False)
    finally:
        current_stage.reset(token)
        await owner.close()


@pytest.mark.asyncio
async def test_truncated_stream_outside_a_stage_is_resent_once(monkeypatch):
    monkeypatch.setattr(design_runtime, "_model_semaphore", None)
    async with streaming_relay(monkeypatch, [False, True]) as (wrapper, calls):
        response = await request(wrapper)
    assert len(calls) == 2
    assert [part.content for part in response.parts if hasattr(part, "content")][-1] == "ok"
    async with streaming_relay(monkeypatch, [False, False]) as (wrapper, calls):
        with pytest.raises(ProviderRetryError, match="provider_stream_incomplete"):
            await request(wrapper)
    assert len(calls) == 2
