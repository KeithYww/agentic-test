# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

import httpx
import uvicorn
from a2a.helpers import get_message_text, new_text_message
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Message
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_ai import Agent, Tool
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.mcp import MCPServerSSE
from pydantic_ai.messages import BinaryContent, UserContent
from pydantic_ai.settings import ThinkingLevel
from pydantic_ai.tools import AgentDepsT, ToolFuncEither
from pydantic_ai.usage import UsageLimits

import config
from common import observability, utils
from common.agent_executor import DefaultAgentExecutor
from common.custom_llm_wrapper import CustomLlmWrapper
from common.llm_provider import get_model
from common.models import AgentExecutionError, AgentRuntimeError, JsonSerializableModel
from common.provider_errors import ProviderRetryError
from common.provider_retry import current_deadline
from common.services.vector_db_service import VectorDbService
from common.streaming import compute_activity_budget
from common.token_usage import TokenUsage
from common.trace_redaction import build_redacted_trace_json

REGISTRATION_PATH = f"{config.ORCHESTRATOR_URL}/register"
MCP_SERVER_ATTACHMENTS_FOLDER_PATH = config.MCP_SERVER_ATTACHMENTS_FOLDER_PATH
ATTACHMENTS_LOCAL_DESTINATION_FOLDER_PATH = config.ATTACHMENTS_LOCAL_DESTINATION_FOLDER_PATH

logger = utils.get_logger("agent_base")

# Bound the activity queue so report_activity calls cannot accumulate without a
# consumer (e.g. standalone runs, where no executor drains the queue).
_ACTIVITY_QUEUE_MAXSIZE = 1000


class AgentBase(ABC):
    def __init__(
        self,
        agent_name: str,
        base_url: str,
        protocol: str,
        port: int,
        external_port: int,
        model_name: str,
        output_type: type[BaseModel],
        instructions: str,
        mcp_servers: list[MCPServerSSE],
        deps_type: type[BaseModel] | None = None,
        description: str = "",
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        vector_db_collection_name: str | None = None,
        fallback_model_name: str | None = None,
    ):
        """Initialise the agent and its underlying A2A server.

        Note for prompt-template authors: the ``report_activity`` tool and a one-line
        instruction snippet are appended to *instructions* automatically here.
        Do **not** include them in your system-prompt template files.
        """
        self.agent_name = agent_name
        self.base_url = base_url
        self.port = port
        self.external_port = external_port
        self.protocol = protocol
        self.url = f"{self.base_url}:{self.external_port}"
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.output_type = output_type
        self.instructions = instructions
        self.deps_type = deps_type
        self.description = description
        self.mcp_servers = mcp_servers or []
        self._activity_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=_ACTIVITY_QUEUE_MAXSIZE)
        self.tools = [*tools, self.report_activity]
        self.instructions = (
            self.instructions
            + "\nA `report_activity` tool is available — call it before any other tool call or reasoning phase."
        )
        self.agent = self._create_agent()
        self._mcp_managed_agents: list[Agent] = [self.agent]
        self.a2a_server = self._get_server()

        self.vector_db_service = None
        if vector_db_collection_name:
            self.vector_db_service = VectorDbService(vector_db_collection_name)
        self.latest_received_message: Message | None = None
        # Token usage of the most recent run; reset per task by the executor and read back
        # by it to emit the usage artifact. None until a run completes.
        self.latest_token_usage: TokenUsage | None = None
        # Redacted trace (full message history, minus external-source content) of the most
        # recent run; reset per task by the executor and read back to emit the trace artifact.
        self.latest_trace: bytes | None = None

    @property
    def activity_queue(self) -> asyncio.Queue[str]:
        """Queue of pending activity descriptions reported via report_activity.

        Exposed so the executor can drain reported activities without reaching across
        the privacy boundary into the internal queue.
        """
        return self._activity_queue

    @abstractmethod
    def get_thinking_level(self) -> ThinkingLevel:
        pass

    @abstractmethod
    def get_max_requests_per_task(self) -> int:
        pass

    @abstractmethod
    def get_skills(self) -> list[AgentSkill]:
        """Return the A2A skills this agent exposes in its Agent Card.

        Each skill's ``description`` is also read by the orchestrator's task-routing
        discovery prompt (see ``_get_agents_info`` in orchestrator/main.py), so it must
        be a real, capability-level summary — not a placeholder.
        """

    def get_max_tokens(self) -> int | None:
        return None

    def get_total_tokens_limit(self) -> int:
        return config.BudgetConfig.TOTAL_TOKENS_LIMIT_PER_TASK

    async def report_activity(self, description: str) -> None:
        """Report your current activity to the dashboard.

        Call this with one short sentence (≤ 120 chars) describing what you are
        about to do, whenever you start a new reasoning phase OR before invoking
        any other tool. You may call it in parallel with other tool calls.
        Examples: "Fetching Jira issue PROJ-123", "Generating test steps for AC-2".
        """
        from common.streaming import current_activity_queue

        q = current_activity_queue.get(None) or self._activity_queue
        try:
            q.put_nowait(description)
        except asyncio.QueueFull:
            # No active consumer (e.g. standalone run) or the consumer fell behind:
            # drop the update rather than letting the queue grow without bound.
            logger.debug("Activity queue full; dropping activity update: %s", description)

    def _create_agent(self) -> Agent:
        logger.info(f"""Creating agent '{self.agent_name}' with the following configuration:
        - Model: {self.model_name}
        - Output Type: {self.output_type.__name__}
        - MCP Servers: {[server.url for server in self.mcp_servers]}
        - Tools: {[getattr(tool, "__name__", None) or tool.name for tool in self.tools]}""")

        model = get_model(self.model_name, self.fallback_model_name) if self.fallback_model_name else self.model_name
        return CustomLlmWrapper.create_agent(
            model_name=model,
            output_type=self.output_type,
            instructions=self.instructions,
            name=self.agent_name,
            thinking_level=self.get_thinking_level(),
            max_tokens=self.get_max_tokens(),
            toolsets=self.mcp_servers,
            tools=self.tools,
            deps_type=self.deps_type,
            retries=config.RetryConfig.MAX_RETRIES,
            output_retries=config.RetryConfig.MAX_RETRIES,
        )

    async def _get_agent_execution_result(
        self, received_request: list[UserContent], deps: Any = None
    ) -> AgentRunResult[Any] | None:
        usage_limits = UsageLimits(
            tool_calls_limit=compute_activity_budget(self.get_max_requests_per_task()),
            total_tokens_limit=self.get_total_tokens_limit(),
        )
        deadline_token = current_deadline.set(
            current_deadline.get() or time.monotonic() + config.OrchestratorConfig.TASK_EXECUTION_TIMEOUT
        )
        try:
            for attempt in range(config.RetryConfig.MAX_RETRIES):
                try:
                    logger.info(f"Starting agent run (attempt {attempt + 1}/{config.RetryConfig.MAX_RETRIES})...")
                    try:
                        return await self.agent.run(received_request, deps=deps, usage_limits=usage_limits)
                    except ExceptionGroup as eg:
                        if any(isinstance(exc, httpx.ConnectError) for exc in eg.exceptions) and self.mcp_servers:
                            mcp_urls = [server.url for server in self.mcp_servers]
                            raise ConnectionError(
                                f"MCP connection failed: could not connect to MCP server(s) {mcp_urls}. "
                                "Ensure the MCP server(s) are running and accessible."
                            ) from eg
                        raise
                except (ModelHTTPError, httpx.TransportError) as e:
                    if getattr(self.agent.model, "provider_retry_managed", False) is True:
                        raise
                    is_retryable = isinstance(e, httpx.TransportError) or (
                        isinstance(e, ModelHTTPError) and e.status_code in config.RetryConfig.RETRYABLE_STATUS_CODES
                    )
                    if is_retryable and attempt < config.RetryConfig.MAX_RETRIES - 1:
                        delay = config.RetryConfig.RETRY_BASE_DELAY_SECONDS * (2**attempt)
                        logger.warning(
                            f"LLM provider request failed: {e} "
                            f"(attempt {attempt + 1}/{config.RetryConfig.MAX_RETRIES}), retrying in {delay:.0f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise
            return None
        finally:
            current_deadline.reset(deadline_token)

    async def run(self, received_message: Message, deps: Any = None) -> Message:
        text = get_message_text(received_message)
        from common.design_models import is_stage_message

        if is_stage_message(text):
            from common.design_models import StageRequest
            from common.design_runtime import StageRunner

            request = StageRequest.model_validate_json(text)
            if request.required_skill_id not in {s.id for s in self.get_skills()}:
                raise ValueError("QA operation sent to an incompatible agent")
            if not hasattr(self, "_stage_runner"):
                self._stage_runner = StageRunner()
            result = await self._stage_runner.run(request, self.execute_stage)
            return new_text_message(
                result.model_dump_json(), context_id=received_message.context_id, task_id=received_message.task_id
            )
        self.latest_received_message = received_message
        received_request = self._get_all_received_contents(received_message)

        try:
            result = await self._get_agent_execution_result(received_request, deps=deps)
            self._capture_token_usage(result)
            self._capture_trace(result)
            self._log_llm_comments_if_result_incomplete(result.output)
            return self._get_text_message_from_results(result)
        except Exception as e:
            logger.exception("Error during agent execution.")
            code = e.code if isinstance(e, ProviderRetryError) else None
            details = e.details if isinstance(e, ProviderRetryError) else {}
            error_model = AgentExecutionError(
                error_message=f"Agent execution failed with error: {e}", error_code=code, error_details=details,
            )
            context_id = getattr(received_message, "context_id", None)
            task_id = getattr(received_message, "task_id", None)
            error_message = new_text_message(
                text=error_model.model_dump_json(),
                context_id=context_id,
                task_id=task_id,
            )
            error_text = (
                "; ".join(f"{type(sub).__name__}: {sub}" for sub in e.exceptions)
                if isinstance(e, ExceptionGroup)
                else str(e)
            )
            raise AgentRuntimeError(list(error_message.parts), error_text, code, details) from e

    async def execute_stage(self, request: Any) -> BaseModel:
        """Execute a versioned QA operation; subclasses explicitly opt in."""
        raise ValueError("This agent does not implement the QA design protocol")

    def _capture_token_usage(self, result: AgentRunResult[Any] | None) -> None:
        """Record and log the token usage and estimated cost of a completed run."""
        if result is None:
            return
        self.latest_token_usage = TokenUsage.from_run_usage(result.usage(), self.model_name)
        logger.info(self.latest_token_usage.summary_line())

    def _capture_trace(self, result: AgentRunResult[Any] | None) -> None:
        """Record a redacted trace of the completed run's full message history."""
        if result is None:
            return
        self.latest_trace = build_redacted_trace_json(result.all_messages())

    def _log_llm_comments_if_result_incomplete(self, output: BaseModel | None | str) -> None:
        """Logs LLM comments if the agent result appears empty or incomplete.

        Args:
            output: The output from the agent execution.
        """
        if output is None:
            logger.warning("Agent returned None result.")
            return

        # Check if the output has the llm_comments attribute (from BaseAgentResult)
        llm_comments = getattr(output, "llm_comments", None)
        if not llm_comments:
            return

        # Check if the result appears to be empty or incomplete
        is_incomplete = self._check_if_result_incomplete(output)
        if is_incomplete:
            logger.warning(f"Agent returned incomplete result. LLM comments: {llm_comments}")

    @staticmethod
    def _check_if_result_incomplete(output: BaseModel) -> bool:
        """Checks if the agent result appears to be empty or incomplete.

        Args:
            output: The output model from the agent execution.

        Returns:
            True if the result appears incomplete, False otherwise.
        """
        if output is None:
            return True

        # Get all field names, excluding llm_comments which is metadata
        field_names = [name for name in output.model_fields if name != "llm_comments"]

        if not field_names:
            return False

        for field_name in field_names:
            value = getattr(output, field_name, None)
            if value is None:
                continue
            # Check if it's a non-empty collection
            if isinstance(value, list | dict | set):
                if len(value) > 0:
                    return False
            # Check if it's a non-empty string
            elif isinstance(value, str):
                if value.strip():
                    return False
            # Any other truthy value means the result is not incomplete
            elif value:
                return False

        return True

    # noinspection PyUnusedLocal
    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        from contextlib import AsyncExitStack

        logger.info(f"{self.agent_name} started.")
        observability.init_tracing(self.agent_name)
        logger.info(f"Using following MCP server URLs: {[server.url for server in self.mcp_servers]}")
        # Enter the agent context once at startup so MCP connections are established
        # in this (lifespan) task and held for the server's lifetime. Concurrent
        # agent.run() calls then share a single live MCP session without re-entering
        # the context, which would cause anyio cancel-scope cross-task errors.
        # Retry to tolerate brief MCP startup delays in container environments.
        stack = AsyncExitStack()
        for attempt in range(config.RetryConfig.MAX_RETRIES):
            try:
                for managed_agent in self._mcp_managed_agents:
                    await stack.enter_async_context(managed_agent)
                break
            except Exception as exc:
                await stack.aclose()
                stack = AsyncExitStack()
                if attempt < config.RetryConfig.MAX_RETRIES - 1:
                    delay = config.RetryConfig.RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
                    logger.warning(
                        f"MCP startup failed (attempt {attempt + 1}/{config.RetryConfig.MAX_RETRIES}): {exc}; "
                        f"retrying in {delay:.0f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise RuntimeError(
                        f"Agent '{self.agent_name}' failed to connect to MCP after "
                        f"{config.RetryConfig.MAX_RETRIES} attempts"
                    ) from exc
        try:
            yield
        finally:
            await stack.aclose()
            if self.vector_db_service:
                await self.vector_db_service.close()
            observability.shutdown_tracing()
            logger.info("Shutting down.")

    @staticmethod
    def _fetch_attachments(attachment_paths: list[str]) -> dict[str, BinaryContent]:
        """Fetches all attachments, returning them as binary content for multimodal processing.

        Args:
            attachment_paths: List of file paths to the downloaded attachments.

        Returns:
            Dictionary mapping filename to BinaryContent for valid, supported attachments.
        """
        from common.attachment_handler import fetch_all_attachments

        return fetch_all_attachments(attachment_paths)

    def _get_server(self) -> FastAPI:
        skills = self.get_skills()
        if not skills or any(not skill.description for skill in skills):
            raise ValueError(
                f"'{self.agent_name}' must declare at least one skill with a non-empty description via "
                "get_skills() — the orchestrator's task-routing discovery prompt reads it."
            )
        agent_card = AgentCard(
            name=self.agent_name,
            description=f"Model: {self.model_name}",
            version="1.0.0",
            default_input_modes=["text"],
            default_output_modes=["text", "image"],
            capabilities=AgentCapabilities(streaming=True),
            skills=skills,
            supported_interfaces=[
                AgentInterface(
                    protocol_binding="JSONRPC",
                    url=self.url,
                )
            ],
        )
        request_handler = DefaultRequestHandler(
            agent_executor=DefaultAgentExecutor(self),
            task_store=InMemoryTaskStore(),
            agent_card=agent_card,
        )
        routes = [
            *create_agent_card_routes(agent_card),
            *create_jsonrpc_routes(request_handler, "/"),
        ]
        a2a_app = FastAPI(routes=routes, lifespan=self._lifespan)
        agent_name = self.agent_name

        @a2a_app.get("/source")
        async def _source_offer():
            # AGPL-3.0 §13: offer the Corresponding Source to users interacting remotely.
            return {
                "name": agent_name,
                "copyright": "Copyright (C) 2025-2026 Taras Paruta",
                "license": "AGPL-3.0-only",
                "license_url": "https://www.gnu.org/licenses/agpl-3.0.html",
                "source_url": "https://github.com/KeithYww/agentic-test",
            }

        return a2a_app

    def start_as_server(self):
        parsed_url = urlparse(self.base_url)
        host = parsed_url.hostname
        uvicorn.run(self.a2a_server, host=host, port=self.port)

    @staticmethod
    def _get_all_received_contents(received_message) -> list[UserContent]:
        text_content: str = get_message_text(received_message)
        files_content: list[BinaryContent] = []
        for part in received_message.parts:
            if part.HasField("raw"):
                files_content.append(BinaryContent(data=part.raw, media_type=part.media_type))
        if files_content:
            logger.info(f"Passing {len(files_content)} file(s) to LLM context.")
        all_contents: list[UserContent] = [text_content, *files_content]
        return all_contents

    @staticmethod
    def _get_text_message_from_results(
        result: AgentRunResult, context_id: str | None = None, task_id: str | None = None
    ) -> Message:
        output = result.output
        if isinstance(output, JsonSerializableModel):
            return new_text_message(text=output.model_dump_json(), context_id=context_id, task_id=task_id)
        if isinstance(output, dict):
            text_parts = []
            for part in output.get("parts", []):
                if part.get("type", "") == "text":
                    text_parts.append(part.get("text", ""))
            return new_text_message(text="\n".join(text_parts), context_id=context_id, task_id=task_id)
        else:
            return new_text_message(text=str(output), context_id=context_id, task_id=task_id)
