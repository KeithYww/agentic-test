# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Bounded relay retries with one auditable attempt per candidate request."""

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from opentelemetry import trace
from pydantic import TypeAdapter
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelResponse, ThinkingPart
from pydantic_ai.models import Model, ModelRequestParameters, ModelSettings

import config
from common import design_store, utils
from common.design_runtime import BudgetExceeded, ModelOutputFailure, current_stage, current_unit, model_semaphore
from common.history_compaction import estimated_tokens
from common.llm_provider import photon_candidates
from common.provider_errors import ProviderRetryError
from common.provider_policy import ProviderRetryPolicy

logger = utils.get_logger("provider_retry")
current_deadline: ContextVar[float | None] = ContextVar("provider_deadline", default=None)
CONTRACT = "photon-preflight-v1"


@dataclass(slots=True)
class ProviderFailure:
    """Only safe metadata crosses the provider exception boundary."""

    known_rejection: bool
    status: int | None = None
    retry_after: float = 0
    request_id: str | None = None


@dataclass(slots=True)
class _RequestState:
    unknown: bool = False


def classify_failure(error: BaseException) -> ProviderFailure:
    """Require a declared preflight contract and a structured rejection code."""
    if not isinstance(error, ModelHTTPError):
        return ProviderFailure(False)
    body = error.body if isinstance(error.body, dict) else {}
    detail = body.get("error", {})
    known = (
        ProviderRetryPolicy.PHOTON_REJECTION_CONTRACT == CONTRACT
        and error.status_code == 503
        and isinstance(detail, dict)
        and (detail.get("type") == "system_cpu_overloaded" or detail.get("code") == "system_cpu_overloaded")
    )
    response = getattr(error.__cause__, "response", None)
    headers = response.headers if response is not None else {}
    retry_after = 0
    value = headers.get("retry-after")
    if value is not None:
        try:
            if value.strip().isascii() and value.strip().isdigit():
                retry_after = float(value)
            else:
                date = parsedate_to_datetime(value)
                if date.tzinfo is not None:
                    retry_after = max(0, (date - datetime.now(UTC)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            retry_after = 0
    return ProviderFailure(known, error.status_code, retry_after, headers.get("request-id") or headers.get("x-request-id"))


def retry_delay(ordinal: int, failure: ProviderFailure) -> float:
    """Never send earlier than the provider's valid Retry-After header."""
    base = 5 * (2 ** (ordinal - 1))
    return max(base, failure.retry_after) + random.uniform(0, base * 0.4)


async def _complete(owner: design_store.DesignStore, action: str, *args: Any) -> tuple[Any, bool]:
    """Confirm the database outcome even when the waiting coroutine is canceled."""
    task = asyncio.create_task(owner.call(action, *args))
    canceled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            canceled = True
    return task.result(), canceled


async def _write(owner: design_store.DesignStore, action: str, *args: Any) -> bool:
    try:
        _, canceled = await _complete(owner, action, *args)
        return canceled
    except Exception:
        context = current_stage.get()
        if context:
            context.usage.complete = False
        raise


@asynccontextmanager
async def _slots() -> AsyncIterator[None]:
    context = current_stage.get()
    async with model_semaphore():
        if context:
            async with context.semaphore:
                yield
        else:
            yield


def _fingerprint(messages: list[ModelMessage], settings: ModelSettings, parameters: ModelRequestParameters, model: Model) -> tuple[str, int]:
    def stable(value: object) -> object:
        if isinstance(value, dict):
            return {key: stable(item) for key, item in value.items() if key != "timestamp"}
        if isinstance(value, list):
            return [stable(item) for item in value]
        return value

    params = TypeAdapter(ModelRequestParameters).dump_python(parameters, mode="json") if parameters else None
    fingerprint = design_store.digest({
        "messages": stable(ModelMessagesTypeAdapter.dump_python(messages, mode="json")),
        "parameters": params, "settings": settings, "model": model.model_name,
    })
    estimate = estimated_tokens(messages) + settings.get("max_tokens", 4096) + len(json.dumps(params, ensure_ascii=False).encode())
    return fingerprint, estimate


async def request_with_retry(
    model: Model, messages: list[ModelMessage], settings: ModelSettings, parameters: ModelRequestParameters,
) -> ModelResponse:
    """Continue the current agent conversation; never replay tools or unknown calls."""
    context = current_stage.get()
    outcome = _RequestState()
    deadline = context.deadline if context else current_deadline.get() or time.monotonic() + config.OrchestratorConfig.TASK_EXECUTION_TIMEOUT
    try:
        async with asyncio.timeout_at(deadline):
            return await _request(model, messages, settings, parameters, deadline, outcome)
    except TimeoutError as exc:
        code = "provider_usage_unknown" if outcome.unknown else "provider_retry_deadline"
        raise ProviderRetryError(code) from exc


async def _request(
    model: Model, messages: list[ModelMessage], settings: ModelSettings,
    parameters: ModelRequestParameters, deadline: float,
    outcome: _RequestState,
) -> ModelResponse:
    context = current_stage.get()
    owner = design_store.store if context else None
    candidate = photon_candidates(model)[0]
    # The sole safe rejection contract is gateway-scoped. Switching aliases cannot bypass it.
    fingerprint, estimate = _fingerprint(messages, settings, parameters, model)
    ordinal, waited, truncated = 0, 0.0, 0
    while True:
        delay = None
        permit = None
        reserved = False
        state = "not_sent"
        async with _slots():
            if context:
                async with context.lock:
                    if not context.usage.complete:
                        raise ProviderRetryError("provider_usage_unknown")
                    if context.usage.total_tokens + context.reserved + estimate > context.token_limit:
                        raise BudgetExceeded("budget_preflight_exceeded", required_reservation=estimate)
                    if owner:
                        try:
                            permit, canceled = await _complete(owner,
                                "provider_reserve", context.run_id, context.generation,
                                current_unit.get() or context.task_key, fingerprint, estimate,
                                context.budget_ceiling, candidate.model_name,
                                ProviderRetryPolicy.PHOTON_OVERLOAD_RETRY_ENABLED,
                            )
                            if canceled:
                                if permit["state"] == "ready":
                                    await _write(owner, "provider_finish", context.run_id, context.generation, permit, {
                                        "state": "not_sent", "failure_kind": "canceled_before_send",
                                    })
                                raise asyncio.CancelledError
                        except design_store.RecoveryBlocked as exc:
                            if str(exc) == "persistent_budget_preflight":
                                raise BudgetExceeded("budget_preflight_exceeded", required_reservation=estimate) from exc
                            raise
                        match permit["state"]:
                            case "cached":
                                response = TypeAdapter(ModelResponse).validate_python(permit["response"])
                                if response.finish_reason == "length":
                                    raise ModelOutputFailure()
                                return response
                            case "waiting":
                                delay = max(0, permit["next_attempt_at"] - time.time())
                            case "rejected":
                                raise ProviderRetryError(permit["code"] or "provider_busy_exhausted")
                            case "ready":
                                ordinal, waited = permit["ordinal"] - 1, permit["waited"]
                    if delay is None:
                        context.reserved += estimate
                        reserved = True
            if delay is None:
                ordinal += 1
                state = "in_flight"
                if context:
                    context.usage.requests += 1
                try:
                    span = trace.get_current_span()
                    span.set_attribute("gen_ai.request.model", candidate.model_name)
                    span.set_attribute("quaia.provider.attempt", ordinal)
                    response = await candidate.request(messages, settings, parameters)
                    usage = response.usage
                    if context and usage.input_tokens + usage.output_tokens <= 0:
                        raise ProviderRetryError("provider_usage_unknown")
                    canceled = False
                    if not (response.provider_details or {}).get("finish_reason"):
                        # The stream closed before message_delta: the reply is partial yet billed, and its real
                        # output usage is lost. Charge the output ceiling and resend once instead of using it.
                        output_ceiling = settings.get("max_tokens", 4096)
                        if owner:
                            canceled = await _write(owner, "provider_finish", context.run_id, context.generation, permit, {
                                "state": "incomplete", "used": usage.input_tokens + output_ceiling,
                                "failure_kind": "stream_incomplete", "request_id": response.provider_response_id,
                            })
                        state = "incomplete"
                        truncated += 1
                        if context:
                            context.usage.input_tokens += usage.input_tokens
                            context.usage.output_tokens += output_ceiling
                        logger.warning("模型响应流提前结束，已按输出上限计费；第 %s 次", truncated,
                                       extra={"model": candidate.model_name, "provider_response_id": response.provider_response_id})
                        if canceled:
                            raise asyncio.CancelledError
                        if context and context.usage.total_tokens > context.token_limit:
                            raise BudgetExceeded("budget_actual_exceeded", used_tokens=context.usage.total_tokens)
                        if truncated >= 2:
                            raise ProviderRetryError("provider_stream_incomplete", attempts=ordinal)
                        continue
                    if owner:
                        safe = replace(response, parts=[part for part in response.parts if not isinstance(part, ThinkingPart)])
                        reusable = response.finish_reason == "length" or not any(isinstance(part, ThinkingPart) for part in response.parts)
                        canceled = await _write(owner, "provider_finish", context.run_id, context.generation, permit, {
                            "state": "settled", "used": usage.input_tokens + usage.output_tokens,
                            "response": TypeAdapter(ModelResponse).dump_python(safe, mode="json"), "reusable": reusable,
                        })
                    state = "settled"
                    if context:
                        context.usage.input_tokens += usage.input_tokens
                        context.usage.output_tokens += usage.output_tokens
                    if canceled:
                        raise asyncio.CancelledError
                    if context and context.usage.total_tokens > context.token_limit:
                        raise BudgetExceeded("budget_actual_exceeded", used_tokens=context.usage.total_tokens)
                    if context and response.finish_reason == "length":
                        raise ModelOutputFailure()
                    return response
                except BaseException as exc:
                    if state in {"settled", "incomplete"}:
                        raise
                    failure = classify_failure(exc)
                    if not failure.known_rejection:
                        outcome.unknown = True
                        if context:
                            context.usage.complete = False
                        if owner:
                            await _write(owner, "provider_finish", context.run_id, context.generation, permit, {
                                "state": "unknown", "status": failure.status, "failure_kind": "unknown_outcome",
                                "request_id": failure.request_id,
                            })
                        if isinstance(exc, asyncio.CancelledError):
                            raise
                        raise ProviderRetryError("provider_usage_unknown", attempts=ordinal) from exc
                    code = "provider_busy_exhausted"
                    if ProviderRetryPolicy.PHOTON_OVERLOAD_RETRY_ENABLED and ordinal < 3:
                        proposed = retry_delay(ordinal, failure)
                        if waited + proposed <= 30 and time.monotonic() + proposed < deadline:
                            delay = proposed
                        elif time.monotonic() + proposed >= deadline:
                            code = "provider_retry_deadline"
                    canceled = False
                    if owner:
                        canceled = await _write(owner, "provider_finish", context.run_id, context.generation, permit, {
                            "state": "rejected", "status": failure.status, "failure_kind": "known_rejection",
                            "request_id": failure.request_id, "contract": CONTRACT, "delay": delay, "code": code,
                        })
                    state = "rejected"
                    if canceled:
                        raise asyncio.CancelledError
                    if delay is None:
                        raise ProviderRetryError(code, attempts=ordinal) from exc
                finally:
                    if reserved:
                        context.reserved -= estimate
        waited += delay
        logger.warning("模型服务繁忙，等待 %.1f 秒后重试；已发送 %s/3 次", delay, ordinal,
                       extra={"model": candidate.model_name, "attempt": ordinal, "wait_seconds": delay})
        await asyncio.sleep(delay)
