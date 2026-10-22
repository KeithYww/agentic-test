# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Request isolation, replay and provider accounting safeguards."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.usage import RequestUsage

from common.custom_llm_wrapper import CustomLlmWrapper
from common.design_models import StageRequest, StageResult
from common.design_runtime import StageContext, StageRunner, current_stage


@pytest.mark.asyncio
async def test_unknown_usage_stops_waiting_request_before_provider():
    entered, release = asyncio.Event(), asyncio.Event()
    model = MagicMock(spec=Model)

    async def provider(*args):
        entered.set()
        await release.wait()
        raise ConnectionError("lost response")

    model.request = AsyncMock(side_effect=provider)
    context = StageContext(10000, time.monotonic() + 5)
    context.semaphore = asyncio.Semaphore(1)
    token = current_stage.set(context)
    try:
        with patch("config.PROMPT_INJECTION_CHECK_ENABLED", False):
            wrapper = CustomLlmWrapper(model, max_tokens=100)
            messages = [ModelRequest(parts=[UserPromptPart("test")])]
            first = asyncio.create_task(wrapper.request(messages, None, None))
            await entered.wait()
            second = asyncio.create_task(wrapper.request(messages, None, None))
            release.set()
            results = await asyncio.gather(first, second, return_exceptions=True)
            assert isinstance(results[0], ConnectionError)
            assert isinstance(results[1], RuntimeError)
            assert "unknown" in str(results[1])
            model.request.assert_awaited_once()
    finally:
        current_stage.reset(token)


@pytest.mark.asyncio
async def test_known_length_response_keeps_metered_usage_complete():
    from common.design_runtime import ModelOutputFailure
    model = MagicMock(spec=Model)
    model.request = AsyncMock(return_value=ModelResponse(
        parts=[], usage=RequestUsage(input_tokens=10, output_tokens=100), finish_reason="length",
    ))
    context = StageContext(10000, time.monotonic() + 5)
    token = current_stage.set(context)
    try:
        with patch("config.PROMPT_INJECTION_CHECK_ENABLED", False), pytest.raises(ModelOutputFailure):
            await CustomLlmWrapper(model, max_tokens=100).request(
                [ModelRequest(parts=[UserPromptPart("test")])], None, None,
            )
        assert context.usage.complete and context.usage.total_tokens == 110
    finally:
        current_stage.reset(token)


@pytest.mark.asyncio
async def test_durable_structured_output_replays_without_a_second_paid_call(tmp_path, monkeypatch):
    from pydantic_ai.messages import ToolCallPart

    from common import design_store

    owner = design_store.DesignStore(str(tmp_path))
    await owner.call("open")
    run, _ = await owner.call("create_run", "story", {}, 10000, 3600)
    monkeypatch.setattr(design_store, "store", owner)
    model = MagicMock(spec=Model)
    model.model_name = "test-model"
    model.request = AsyncMock(return_value=ModelResponse(
        parts=[ToolCallPart("final_result", {"ok": True}, "call-1")],
        usage=RequestUsage(input_tokens=30, output_tokens=10), finish_reason="tool_call",
    ))
    context = StageContext(10000, time.monotonic() + 5, run_id=run["id"], generation=1,
                           task_key="generate-0", budget_ceiling=10000)
    token = current_stage.set(context)
    try:
        wrapper = CustomLlmWrapper(model, max_tokens=100)
        messages = [ModelRequest(parts=[UserPromptPart("test")])]
        with patch("config.PROMPT_INJECTION_CHECK_ENABLED", False):
            first = await wrapper.request(messages, None, None)
            replay = await wrapper.request(messages, None, None)
        assert replay.parts == first.parts
        model.request.assert_awaited_once()
        actual = await owner.call("run", run["id"])
        assert actual["used"] == 40 and actual["reserved"] == 0
        assert context.usage.complete and context.usage.total_tokens == 40
    finally:
        current_stage.reset(token)
        await owner.close()


@pytest.mark.asyncio
async def test_persistent_budget_refusal_is_a_known_presend_failure(tmp_path, monkeypatch):
    from common import design_store
    from common.design_runtime import BudgetExceeded

    owner = design_store.DesignStore(str(tmp_path))
    await owner.call("open")
    run, _ = await owner.call("create_run", "story", {}, 1, 3600)
    monkeypatch.setattr(design_store, "store", owner)
    model = MagicMock(spec=Model)
    model.model_name = "test-model"
    model.request = AsyncMock()
    context = StageContext(10000, time.monotonic() + 5, run_id=run["id"], generation=1,
                           task_key="fix-0", budget_ceiling=10000)
    token = current_stage.set(context)
    try:
        with patch("config.PROMPT_INJECTION_CHECK_ENABLED", False), pytest.raises(BudgetExceeded) as failure:
            await CustomLlmWrapper(model, max_tokens=100).request(
                [ModelRequest(parts=[UserPromptPart("test")])], None, None,
            )
        assert failure.value.code == "budget_preflight_exceeded"
        assert failure.value.details["completed_requests"] == 0 and context.usage.complete
        model.request.assert_not_awaited()
        assert (await owner.call("run", run["id"]))["reserved"] == 0
    finally:
        current_stage.reset(token)
        await owner.close()


class Output(BaseModel):
    ok: bool = True


def test_old_stage_result_without_budget_fields_remains_readable():
    result = StageResult.model_validate({
        "protocol": "qa-design/1", "run_id": "old", "stage_id": "fix-1",
        "input_revision": 1, "execution_id": "old-execution", "status": "completed",
        "usage": {"input_tokens": 100, "output_tokens": 10, "requests": 1},
    })
    assert result.error_code is None
    assert result.error_details == {}


def request(**kwargs):
    return StageRequest(
        run_id="r1",
        stage_id="s1",
        operation="generate",
        required_skill_id="generate-test-cases",
        token_limit=10000,
        timeout_seconds=2,
        payload=kwargs,
    )


@pytest.mark.asyncio
async def test_duplicate_waiter_cancel_does_not_cancel_other_waiter():
    runner = StageRunner()
    ready = asyncio.Event()
    handler = AsyncMock(side_effect=lambda _: ready.wait())

    async def execute(_):
        await ready.wait()
        return Output()

    handler.side_effect = execute
    first = asyncio.create_task(runner.run(request(), handler))
    second = asyncio.create_task(runner.run(request(), handler))
    await asyncio.sleep(0.01)
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    ready.set()
    result = await second
    replay = await runner.run(request(), handler)
    assert result.status == "completed"
    assert replay.execution_id == result.execution_id
    handler.assert_awaited_once()
    with pytest.raises(ValueError, match="different input"):
        await runner.run(request(changed=True), handler)


@pytest.mark.asyncio
async def test_timeout_receipt_retains_usage_and_run_binding():
    async def handler(_):
        current_stage.get().usage.input_tokens = 123
        await asyncio.sleep(1)
        return Output()

    result = await StageRunner().run(request().model_copy(update={"timeout_seconds": 0.01}), handler)
    assert result.status == "timed_out"
    assert result.usage.input_tokens == 123
    assert result.run_id == "r1"
    assert current_stage.get() is None


@pytest.mark.asyncio
async def test_budget_refusal_happens_before_provider_call():
    model = MagicMock(spec=Model)
    model.request = AsyncMock(
        return_value=ModelResponse(parts=[], usage=RequestUsage(input_tokens=10, output_tokens=5))
    )
    wrapper = CustomLlmWrapper(model, max_tokens=100)
    messages = [ModelRequest(parts=[UserPromptPart(content="hello")])]
    context = StageContext(token_limit=1, deadline=time.monotonic() + 5)
    token = current_stage.set(context)
    try:
        with patch("config.PROMPT_INJECTION_CHECK_ENABLED", False), pytest.raises(RuntimeError, match="budget"):
            await wrapper.request(messages, None, None)
        model.request.assert_not_awaited()
        assert context.reserved == 0
    finally:
        current_stage.reset(token)


@pytest.mark.asyncio
async def test_provider_error_marks_usage_incomplete_and_releases_reservation():
    model = MagicMock(spec=Model)
    model.request = AsyncMock(side_effect=ConnectionError("lost response"))
    token = current_stage.set(context := StageContext(10000, time.monotonic() + 5))
    try:
        with patch("config.PROMPT_INJECTION_CHECK_ENABLED", False), pytest.raises(ConnectionError):
            await CustomLlmWrapper(model, max_tokens=100).request(
                [ModelRequest(parts=[UserPromptPart(content="hello")])], None, None
            )
        assert not context.usage.complete
        assert context.usage.requests == 1
        assert context.reserved == 0
    finally:
        current_stage.reset(token)


@pytest.mark.asyncio
async def test_cancel_last_waiter_returns_canceled_usage_receipt():
    entered = asyncio.Event()
    async def handler(_):
        current_stage.get().usage.input_tokens = 71
        entered.set()
        await asyncio.sleep(10)
        return Output()
    task = asyncio.create_task(StageRunner().run(request(), handler))
    await entered.wait()
    task.cancel()
    result = await task
    assert result.status == "canceled"
    assert result.usage.input_tokens == 71


@pytest.mark.asyncio
async def test_inflight_reservation_blocks_a_second_concurrent_request():
    from common.history_compaction import estimated_tokens
    model = MagicMock(spec=Model)
    entered, release = asyncio.Event(), asyncio.Event()
    async def provider(*args):
        entered.set()
        await release.wait()
        return ModelResponse(parts=[], usage=RequestUsage(input_tokens=10, output_tokens=5))
    model.request = AsyncMock(side_effect=provider)
    wrapper = CustomLlmWrapper(model, max_tokens=100)
    messages = [ModelRequest(parts=[UserPromptPart(content="hello")])]
    estimate = estimated_tokens(messages)+100
    context = StageContext(estimate+50, time.monotonic()+5)
    token = current_stage.set(context)
    try:
        with patch("config.PROMPT_INJECTION_CHECK_ENABLED", False):
            first = asyncio.create_task(wrapper.request(messages, None, None))
            await entered.wait()
            assert context.reserved == estimate
            with pytest.raises(RuntimeError, match="budget"):
                await wrapper.request(messages, None, None)
            release.set()
            await first
        model.request.assert_awaited_once()
        assert context.reserved == 0
        assert context.usage.total_tokens == 15
    finally:
        release.set()
        current_stage.reset(token)


@pytest.mark.asyncio
async def test_missing_provider_usage_stops_instead_of_resetting_budget():
    model = MagicMock(spec=Model)
    model.request = AsyncMock(return_value=ModelResponse(parts=[], usage=RequestUsage()))
    token = current_stage.set(context := StageContext(10000, time.monotonic()+5))
    try:
        with patch("config.PROMPT_INJECTION_CHECK_ENABLED", False), pytest.raises(RuntimeError, match="no measurable usage"):
            await CustomLlmWrapper(model, max_tokens=100).request(
                [ModelRequest(parts=[UserPromptPart(content="hello")])], None, None)
        assert not context.usage.complete
        assert context.usage.requests == 1
        assert context.reserved == 0
    finally:
        current_stage.reset(token)


@pytest.mark.asyncio
async def test_budget_failure_receipt_is_safe_specific_and_replays():
    model = MagicMock(spec=Model)
    model.request = AsyncMock()
    wrapper = CustomLlmWrapper(model, max_tokens=8192)
    runner = StageRunner()

    async def handler(req):
        await wrapper.request([ModelRequest(parts=[UserPromptPart(content="private document")])], None, None)
        return Output()

    req = request().model_copy(update={"token_limit": 100})
    with patch("config.PROMPT_INJECTION_CHECK_ENABLED", False):
        result = await runner.run(req, handler)
        replay = await runner.run(req, handler)
    assert result.error_code == "budget_preflight_exceeded"
    assert result.error_details["remaining_tokens"] == 100
    assert result.error_details["max_output_tokens"] == 8192
    assert "private document" not in result.model_dump_json()
    assert result.usage.complete and result.usage.requests == 0
    assert replay.execution_id == result.execution_id
    model.request.assert_not_awaited()
