# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

import asyncio
import json
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime

from opentelemetry import trace
from pydantic import TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import (
    Model,
    ModelRequestParameters,
    ModelSettings,
    StreamedResponse,
)
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ThinkingLevel

import config
from common import design_store, utils
from common.design_runtime import BudgetExceeded, ModelOutputFailure, current_stage, current_unit, model_semaphore
from common.history_compaction import estimated_tokens, prepare_history, validate_external_content
from common.llm_provider import photon_candidates
from common.models import JsonSerializableModel
from common.prompt_injection.guard import GuardPrompt
from common.provider_retry import request_with_retry

LOG_SEPARATOR = "-" * 80

logger = utils.get_logger("llm_wrapper")


class CustomLlmWrapper(WrapperModel):
    def __init__(self, model_name: str | Model, thinking_level: ThinkingLevel | None = None, max_tokens: int | None = None):
        super().__init__(model_name)
        candidates = photon_candidates(self.wrapped)
        self.provider_retry_managed = bool(candidates)
        for candidate in candidates:
            candidate.client.max_retries = 0
        self.wrapped_model_name: str = model_name.model_name if isinstance(model_name, Model) else model_name
        self.latest_instructions: str | None = None
        self.thinking_level = thinking_level
        self.max_tokens = max_tokens

    @classmethod
    def create_agent(
        cls,
        model_name: str | Model,
        output_type: type,
        instructions: str | None = None,
        system_prompt: str | None = None,
        name: str = "",
        thinking_level: ThinkingLevel | None = None,
        max_tokens: int | None = None,
        tools: Sequence = (),
        toolsets: Sequence = (),
        deps_type: type | None = None,
        retries: int = 3,
        output_retries: int = 3,
    ) -> Agent:
        """Creates a pydantic_ai Agent backed by a CustomLlmWrapper model."""
        return Agent(
            model=cls(model_name=model_name, thinking_level=thinking_level, max_tokens=max_tokens),
            output_type=output_type,
            instructions=instructions,
            system_prompt=system_prompt or (),
            name=name,
            tools=list(tools),
            toolsets=list(toolsets),
            deps_type=deps_type,
            retries=retries,
            output_retries=output_retries,
            history_processors=[prepare_history],
        )

    def _get_model_settings(self, provided_settings: ModelSettings | None) -> ModelSettings:
        extra = {"max_tokens": self.max_tokens} if self.max_tokens is not None else {}
        if self.thinking_level is None:
            defaults = ModelSettings(top_p=config.TOP_P, temperature=config.TEMPERATURE, **extra)
        else:
            # Anthropic models require temperature=1 and no top_p when thinking is enabled.
            defaults = ModelSettings(temperature=1, thinking=self.thinking_level, **extra)
        return {**defaults, **(provided_settings or {})}

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        if config.PROMPT_INJECTION_CHECK_ENABLED:
            self._validate_for_prompt_injection(messages)

        if current_stage.get() is None and messages and isinstance(messages[-1], ModelRequest):
            self._log_model_request(messages[-1])

        actual_settings = self._get_model_settings(model_settings)
        start_time = time.monotonic()
        context = current_stage.get()
        if self.provider_retry_managed:
            if context:
                actual_settings.setdefault("max_tokens", 4096)
            response = await request_with_retry(self.wrapped, messages, actual_settings, model_request_parameters)
            if context:
                from common.trace_redaction import build_redacted_trace_json

                trace_data = build_redacted_trace_json([*messages[-1:], response])
                if context.trace_bytes + len(trace_data) <= 262144:
                    context.trace.extend(json.loads(trace_data))
                    context.trace_bytes += len(trace_data)
        elif context is None:
            response = await self.wrapped.request(messages, actual_settings, model_request_parameters)
        else:
            actual_settings.setdefault("max_tokens", 4096)
            estimate = estimated_tokens(messages) + actual_settings["max_tokens"]
            permit = None
            owner = design_store.store
            async with context.semaphore, model_semaphore():
                if not context.usage.complete:
                    raise RuntimeError("QA usage is unknown; refusing further calls")
                if owner:
                    # Timestamps are incidental; model messages and output schemas are business inputs.
                    def stable(value):
                        if isinstance(value, dict):
                            return {k: stable(v) for k, v in value.items() if k != "timestamp"}
                        if isinstance(value, list):
                            return [stable(v) for v in value]
                        return value
                    params = TypeAdapter(ModelRequestParameters).dump_python(model_request_parameters, mode="json") if model_request_parameters else None
                    fingerprint = design_store.digest({
                        "messages": stable(ModelMessagesTypeAdapter.dump_python(messages, mode="json")),
                        "parameters": params, "settings": actual_settings, "model": self.wrapped_model_name,
                    })
                    estimate += len(json.dumps(params, ensure_ascii=False).encode())
                    try:
                        permit = await owner.call("reserve", context.run_id, context.generation,
                                                  current_unit.get() or context.task_key, fingerprint,
                                                  estimate, context.budget_ceiling)
                    except design_store.RecoveryBlocked as exc:
                        if str(exc) != "persistent_budget_preflight":
                            raise
                        raise BudgetExceeded(
                            "budget_preflight_exceeded", required_reservation=estimate,
                            completed_requests=context.usage.requests,
                        ) from exc
                    if permit["response"] is not None:
                        response = TypeAdapter(ModelResponse).validate_python(permit["response"])
                        if response.finish_reason == "length":
                            raise ModelOutputFailure()
                        return response
                async with context.lock:
                    if context.usage.total_tokens + context.reserved + estimate > context.token_limit:
                        if owner and permit:
                            # No provider request was sent; retry may reserve this identity again.
                            await owner.call("not_sent", context.run_id, context.generation, permit["id"])
                        raise BudgetExceeded(
                            "budget_preflight_exceeded",
                            remaining_tokens=context.token_limit - context.usage.total_tokens - context.reserved,
                            estimated_input_bytes=estimate - actual_settings["max_tokens"],
                            max_output_tokens=actual_settings["max_tokens"],
                            required_reservation=estimate,
                            completed_requests=context.usage.requests,
                        )
                    context.reserved += estimate
                usage_known = False
                try:
                    async with asyncio.timeout_at(context.deadline):
                        context.usage.requests += 1
                        response = await self.wrapped.request(messages, actual_settings, model_request_parameters)
                    usage = response.usage
                    if usage.input_tokens + usage.output_tokens <= 0:
                        context.usage.complete = False
                        raise RuntimeError("QA model returned no measurable usage; refusing further calls")
                    context.usage.input_tokens += usage.input_tokens
                    context.usage.output_tokens += usage.output_tokens
                    if owner and permit:
                        safe_response = replace(response, parts=[part for part in response.parts if not isinstance(part, ThinkingPart)])
                        reusable = response.finish_reason == "length" or not any(isinstance(part, ThinkingPart) for part in response.parts)
                        await owner.call("settle", context.run_id, context.generation, permit["id"],
                                         usage.input_tokens + usage.output_tokens,
                                         TypeAdapter(ModelResponse).dump_python(safe_response, mode="json"), reusable)
                    usage_known = True
                    from common.trace_redaction import build_redacted_trace_json
                    trace_data = build_redacted_trace_json([*messages[-1:], response])
                    if context.trace_bytes + len(trace_data) <= 262144:
                        context.trace.extend(json.loads(trace_data))
                        context.trace_bytes += len(trace_data)
                except BaseException:
                    if not usage_known:
                        context.usage.complete = False
                    if owner and permit and not usage_known:
                        await asyncio.shield(owner.call("unknown", context.run_id, context.generation, permit["id"]))
                    raise
                finally:
                    async with context.lock:
                        context.reserved -= estimate
                if context.usage.total_tokens > context.token_limit:
                    raise BudgetExceeded("budget_actual_exceeded", used_tokens=context.usage.total_tokens,
                                         token_limit=context.token_limit)
                if response.finish_reason == "length":
                    # This response was metered; it is not an unknown provider failure.
                    raise ModelOutputFailure()
        duration = time.monotonic() - start_time

        # Emit structured LLM call log — task_id comes from ContextVar so it
        # propagates without threading through the call stack.
        from common.streaming import current_task_id as _current_task_id
        task_id = _current_task_id.get(None)
        if task_id is not None:
            trace.get_current_span().set_attribute("quaia.task_id", task_id)
        try:
            usage = response.usage
            extra: dict = {
                "task_id":       task_id,
                "model":         self.wrapped_model_name,
                "input_tokens":  usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "duration_ms":   int(duration * 1000),
            }
        except Exception:
            extra = {"task_id": task_id, "model": self.wrapped_model_name,
                     "duration_ms": int(duration * 1000)}
        logger.info(f"LLM request to '{self.wrapped_model_name}' completed in {duration:.3f}s", extra=extra)
        if context is None:
            self._log_model_response(response)
        return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context,
    ) -> AsyncIterator[StreamedResponse]:
        if config.PROMPT_INJECTION_CHECK_ENABLED:
            self._validate_for_prompt_injection(messages)

        if current_stage.get() is not None:
            raise RuntimeError("QA stages require non-streaming model calls for bounded usage accounting")
        actual_settings = self._get_model_settings(model_settings)
        start_time = time.monotonic()
        async with self.wrapped.request_stream(
            messages, actual_settings, model_request_parameters, run_context
        ) as response_stream:
            yield response_stream
        duration = time.monotonic() - start_time
        logger.info(f"LLM streaming request to '{self.wrapped_model_name}' completed in {duration:.3f}s")

    @staticmethod
    def _get_prompt_from_messages(messages: list[ModelMessage]) -> GuardPrompt | None:
        for message in reversed(messages):
            if not isinstance(message, ModelRequest):
                continue
            for part in message.parts:
                if isinstance(part, ToolReturnPart):
                    if isinstance(part.content, str):
                        prompt_content = part.content
                    else:
                        prompt_content = CustomLlmWrapper._serialize_content(part.content)
                    return GuardPrompt(
                        prompt_description=f"Result of the execution of the tool '{part.tool_name}' : ",
                        prompt=prompt_content,
                    )
                if isinstance(part, UserPromptPart):
                    prompt_content = ""
                    if isinstance(part.content, str):
                        prompt_content = part.content
                    elif isinstance(part.content, Sequence):
                        prompt_content = "".join([c for c in part.content if isinstance(c, str)])
                    return GuardPrompt(prompt_description="", prompt=prompt_content)
        return None

    def _validate_for_prompt_injection(self, messages):
        validate_external_content(messages)

    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for objects not handled by default json encoder."""
        if isinstance(obj, JsonSerializableModel):
            return obj.model_dump()
        if isinstance(obj, BinaryContent):
            return (
                f"<BinaryContent: media_type={obj.media_type}, identifier={obj.identifier}, size={len(obj.data)} bytes>"
            )
        if isinstance(obj, bytes):
            return f"<bytes: {len(obj)}>"
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    @staticmethod
    def _serialize_content(content) -> str:
        """Serialize content to JSON, handling JsonSerializableModel and BinaryContent instances."""
        return json.dumps(content, indent=2, default=CustomLlmWrapper._json_serializer)

    def _log_model_request(self, message):
        if message.instructions and self.latest_instructions != message.instructions:
            self.latest_instructions = message.instructions
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.debug(
                f"[{timestamp}] Agent is using following instructions: "
                f"\n{LOG_SEPARATOR}\n{self.latest_instructions}\n{LOG_SEPARATOR}"
            )
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                payload = self._serialize_content(part.content)
                logger.debug(
                    f"[{timestamp}] Agent is responding with the execution result of tool: "
                    f"'{part.tool_name}' with result: \n{LOG_SEPARATOR}\n{payload}\n{LOG_SEPARATOR}"
                )
            elif isinstance(part, UserPromptPart):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if isinstance(part.content, str):
                    content_to_log = part.content
                elif isinstance(part.content, Sequence):
                    contents = []
                    for c in part.content:
                        if isinstance(c, str):
                            contents.append(c)
                        else:
                            contents.append(f"<{type(c).__name__}>")
                    content_to_log = "\n".join(contents)
                else:
                    content_to_log = f"<{type(part.content).__name__}>"
                logger.debug(
                    f"[{timestamp}] Agent is prompting the model with user input: "
                    f"\n{LOG_SEPARATOR}\n{content_to_log}\n{LOG_SEPARATOR}"
                )
            elif isinstance(part, SystemPromptPart):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.debug(
                    f"[{timestamp}] Agent is using system prompt: \n{LOG_SEPARATOR}\n{part.content}\n{LOG_SEPARATOR}"
                )
            elif isinstance(part, RetryPromptPart):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.debug(f"[{timestamp}] Agent is retrying prompting the model, the root cause: {part.content}")

    @staticmethod
    def _log_model_response(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        separator = "-" * 80
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                logger.debug(
                    f"[{timestamp}] Model is calling the tool: '{part.tool_name}' with arguments: "
                    f"\n{separator}\n{json.dumps(part.args, indent=2)}\n{separator}"
                )
            elif isinstance(part, ThinkingPart) and part.content:
                logger.debug(
                    f"[{timestamp}] Model is thinking the following:\n{separator}\n{part.content}\n{separator}"
                )
            elif isinstance(part, TextPart) and part.content:
                logger.debug(
                    f"[{timestamp}] Model is responding with the plain text:\n{separator}\n{part.content}\n{separator}"
                )
