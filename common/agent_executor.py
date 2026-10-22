# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Agent executor for pydantic-ai agents over the A2A protocol.
"""

import asyncio
import json
import logging
from uuid import uuid4

from a2a.helpers import get_message_text, new_task_from_user_message, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TaskState

import config
from common import utils
from common.a2a_contract import ArtifactName
from common.agent_log_capture import AgentLogCaptureHandler
from common.design_models import StageResult, is_stage_message
from common.models import AgentRuntimeError
from common.streaming import (
    reset_current_activity_queue,
    reset_current_log_handler,
    reset_current_task_id,
    set_current_activity_queue,
    set_current_log_handler,
    set_current_task_id,
)
from common.token_usage import TokenUsage

logger = utils.get_logger("agent_executor")

_LOG_FLUSH_INTERVAL_SECONDS = 2


class DefaultAgentExecutor(AgentExecutor):
    """
    Executes tasks by invoking the pydantic-ai agent.
    """

    def __init__(self, agent):
        self.agent = agent
        self._active_runs: dict[str, asyncio.Task] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        await self._execute_task(context, event_queue)

    async def _execute_task(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        logger.info(f"Executing task {task_id}")

        updater = TaskUpdater(event_queue, task_id, context.context_id)

        log_handler = AgentLogCaptureHandler()
        log_handler.setLevel(config.LOG_LEVEL)
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)
        handler_token = set_current_log_handler(log_handler)

        # Per-request activity queue and task_id: both bound via ContextVar so that
        # any logger in the call chain (including custom_llm_wrapper) can emit task_id
        # as a structured field without threading it through the call stack.
        per_request_activity_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)
        activity_token = set_current_activity_queue(per_request_activity_queue)
        task_id_token = set_current_task_id(task_id)

        logs_artifact_id = str(uuid4())  # stable id correlating every log chunk for this task
        sent_any_logs = False
        handler_detached = False

        # Reset usage/trace so a failed run cannot emit the previous task's artifacts.
        self.agent.latest_token_usage = None
        self.agent.latest_trace = None

        async def flush_activity_loop() -> None:
            # Forward each reported activity immediately as a WORKING status message.
            # A transient update_status failure is logged and the loop keeps streaming.
            while True:
                try:
                    description = await per_request_activity_queue.get()
                    await updater.update_status(TaskState.TASK_STATE_WORKING, message=new_text_message(description))
                except asyncio.CancelledError:
                    return
                except Exception:
                    logger.exception("Failed to forward reported activity; continuing to stream.")

        async def flush_logs_loop() -> None:
            nonlocal sent_any_logs
            while True:
                try:
                    await asyncio.sleep(_LOG_FLUSH_INTERVAL_SECONDS)
                    batch = log_handler.drain()
                    if batch:
                        await updater.add_artifact(
                            parts=[Part(raw="\n".join(batch).encode("utf-8"), media_type="text/plain")],
                            artifact_id=logs_artifact_id,
                            name=ArtifactName.LOGS,
                            append=sent_any_logs,
                            last_chunk=False,
                        )
                        sent_any_logs = True
                except asyncio.CancelledError:
                    return
                except Exception:
                    logger.exception("Failed to flush log batch; continuing to stream.")

        try:
            received_message = context.message
            if not received_message:
                raise ValueError("No message found in the request message.")

            task = context.current_task or new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)  # TaskUpdater does not emit the Task object
            await updater.start_work()

            activity_task = asyncio.create_task(flush_activity_loop())
            logs_task = asyncio.create_task(flush_logs_loop())

            run_task = asyncio.create_task(self.agent.run(received_message))
            self._active_runs[task_id] = run_task
            try:
                result = await run_task
                # Capture per-request artifacts immediately before any subsequent await
                # could allow another concurrent task to overwrite these instance attributes.
                stage_result = None
                if is_stage_message(get_message_text(received_message)):
                    stage_result = StageResult.model_validate_json(get_message_text(result))
                    usage = stage_result.usage
                    token_usage = TokenUsage(
                        model_name="qa-stage", input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                        total_tokens=usage.total_tokens, requests=usage.requests, cache_read_tokens=0,
                        tool_calls=0, cost_usd=None, complete=usage.complete,
                    )
                    trace = json.dumps(stage_result.trace).encode()
                else:
                    token_usage = self.agent.latest_token_usage
                    trace = self.agent.latest_trace
            finally:
                self._active_runs.pop(task_id, None)

                # 1. Stop loops BEFORE any terminal status (update_status latches terminal state).
                activity_task.cancel()
                logs_task.cancel()
                results = await asyncio.gather(activity_task, logs_task, return_exceptions=True)
                for res in results:
                    if isinstance(res, Exception) and not isinstance(res, asyncio.CancelledError):
                        logger.error(f"Error in background task during shutdown: {res}", exc_info=res)

                # 2. Final log chunk with last_chunk=True (must precede the terminal status).
                #    Sent even when empty (if anything was streamed) so the consumer can finalize.
                remaining = log_handler.drain()
                if remaining or sent_any_logs:
                    parts = (
                        [Part(raw="\n".join(remaining).encode("utf-8"), media_type="text/plain")] if remaining else []
                    )
                    await updater.add_artifact(
                        parts=parts,
                        artifact_id=logs_artifact_id,
                        name=ArtifactName.LOGS,
                        append=sent_any_logs,
                        last_chunk=True,
                    )

                # 3. Detach the handler and per-request queue ContextVar.
                reset_current_log_handler(handler_token)
                root_logger.removeHandler(log_handler)
                reset_current_activity_queue(activity_token)
                reset_current_task_id(task_id_token)
                handler_detached = True

            # 4. Execution-result artifact (auto-generated unique artifact_id avoids id collisions).
            await updater.add_artifact(parts=list(result.parts), name=ArtifactName.EXECUTION_RESULT)

            # 4b. Token-usage artifact so the orchestrator can record consumption and cost.
            if token_usage is not None:
                await updater.add_artifact(
                    parts=[
                        Part(
                            raw=token_usage.model_dump_json().encode("utf-8"),
                            media_type="application/json",
                            filename="usage.json",
                        )
                    ],
                    name=ArtifactName.USAGE,
                )

            # 4c. Redacted trace artifact so the dashboard can show a debug view of the run.
            if trace is not None:
                await updater.add_artifact(
                    parts=[Part(raw=trace, media_type="application/json", filename="trace.json")],
                    name=ArtifactName.TRACE,
                )

            # 5. Terminal status — always last.
            if stage_result is not None and stage_result.status != "completed":
                if stage_result.status == "canceled":
                    await updater.cancel()
                else:
                    await updater.failed(message=new_text_message(stage_result.error or stage_result.status))
                return
            await updater.complete()
            logger.info(f"Task {task_id} completed successfully.")

        except asyncio.CancelledError:
            # cancel() requested cancellation of run_task. The ordered shutdown (stop
            # loops → final log chunk → detach handler) already ran in the finally above.
            # Emit CANCELED here so the terminal status is always last, and swallow the
            # error so it does not propagate out of execute().
            logger.info(f"Task {task_id} was canceled.")
            await updater.cancel()

        except AgentRuntimeError as e:
            logger.error(f"Agent execution failed for task {task_id}: {e}")
            await updater.add_artifact(parts=list(e.parts), name=ArtifactName.EXECUTION_RESULT)
            await updater.failed(message=new_text_message(str(e)))

        except Exception as e:
            logger.exception(f"Error executing task {task_id}: {e}", extra={"task_id": task_id})
            await updater.failed(message=new_text_message(f"An error occurred: {e!s}"))
        finally:
            if not handler_detached:
                reset_current_log_handler(handler_token)
                root_logger.removeHandler(log_handler)
                reset_current_activity_queue(activity_token)
                reset_current_task_id(task_id_token)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        run_task = self._active_runs.get(task_id)
        if run_task is not None:
            # execute() owns the ordered shutdown and emits the CANCELED terminal status
            # on its own queue; here we only request cancellation of the active run.
            run_task.cancel()
        else:
            logger.warning(f"No active run found for task {task_id}; emitting CANCELED on cancel-call's queue")
            await TaskUpdater(event_queue, task_id, context.context_id).cancel()
