# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Request-local model accounting and bounded execution for QA stages."""

import asyncio
import hashlib
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from uuid import uuid4

from pydantic import BaseModel

import config
from common import design_store
from common.design_models import StageRequest, StageResult, StageUsage
from common.design_store import digest
from common.provider_errors import ProviderRetryError


class BudgetExceeded(RuntimeError):
    """Safe budget diagnostics that never contain model inputs or responses."""

    def __init__(self, code: str, **details: int | str) -> None:
        self.code = code
        self.details = details
        super().__init__(f"{code}: " + "; ".join(f"{key}={value}" for key, value in details.items()))


class ModelOutputFailure(RuntimeError):
    """A metered response hit its output cap; no raw model content is exposed."""

    code = "output_limit"

    def __init__(self) -> None:
        super().__init__(self.code)


@dataclass(slots=True)
class StageContext:
    """Shared by concurrent model calls in one stage, never across stories."""

    token_limit: int
    deadline: float
    usage: StageUsage = field(default_factory=StageUsage)
    reserved: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    semaphore: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(2))
    trace: list[dict] = field(default_factory=list)
    trace_bytes: int = 0
    checked_content: set[str] = field(default_factory=set)
    run_id: str = ""
    generation: int = 0
    task_key: str = ""
    budget_ceiling: int = 0


current_stage: ContextVar[StageContext | None] = ContextVar("qa_stage", default=None)
current_unit: ContextVar[str] = ContextVar("qa_unit", default="")
_model_semaphore: asyncio.Semaphore | None = None


def model_semaphore() -> asyncio.Semaphore:
    """Return the all-in-one process model-request limit."""
    global _model_semaphore
    if _model_semaphore is None:
        _model_semaphore = asyncio.Semaphore(config.DesignConfig.MODEL_CONCURRENCY)
    return _model_semaphore


async def checkpoint_unit(key: str, inputs: object, output_type: type[BaseModel], handler: Callable) -> BaseModel:
    """Reuse only a fully validated business unit from the same stage inputs."""
    context = current_stage.get()
    owner = design_store.store
    fingerprint = digest(inputs)
    unit_key = f"{context.task_key}/{key}" if context else key
    if owner and context:
        previous = await owner.call("get", context.run_id, "unit", unit_key, fingerprint)
        if previous:
            return output_type.model_validate(previous["value"])
    token = current_unit.set(unit_key)
    try:
        value = await handler()
        if owner and context:
            await owner.call("put", context.run_id, context.generation, "unit", unit_key, fingerprint,
                             value.model_dump(mode="json"))
        return value
    finally:
        current_unit.reset(token)


async def checkpoint_agent(key: str, agent: object, prompt: str, output_type: type[BaseModel], deps: object = None) -> BaseModel:
    """Checkpoint the validated output of an existing Pydantic AI agent."""
    async def invoke():
        return (await agent.run(prompt, deps=deps)).output

    return await checkpoint_unit(key, prompt, output_type, invoke)


class StageRunner:
    """Coalesce duplicate stages within a bounded, process-local replay window."""

    def __init__(self) -> None:
        self._waiters: dict[tuple, int] = {}
        self._runs: OrderedDict[tuple, tuple[str, asyncio.Task, float]] = OrderedDict()

    async def run(self, request: StageRequest, handler: Callable[[StageRequest], Awaitable[BaseModel]]) -> StageResult:
        """Execute a stage at most once in this runner's live replay window."""
        key = (request.run_id, request.stage_id, request.input_revision, request.operation)
        digest = hashlib.sha256(request.model_dump_json(exclude={"timeout_seconds"}).encode()).hexdigest()
        now = time.monotonic()
        for old_key, (_, task, expiry) in list(self._runs.items()):
            if task.done() and expiry < now:
                del self._runs[old_key]
        if key in self._runs:
            previous_digest, task, _ = self._runs[key]
            if previous_digest != digest:
                raise ValueError("A stage identity cannot be reused with different input")
        else:
            if len(self._runs) >= config.DesignConfig.STAGE_CACHE_SIZE:
                raise RuntimeError("Stage replay cache full; refusing unsafe eviction")
            task = asyncio.create_task(self._execute(request, handler))
            self._runs[key] = (digest, task, now + config.DesignConfig.TIMEOUT_SECONDS)
        self._waiters[key] = self._waiters.get(key, 0) + 1
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if self._waiters[key] != 1:
                raise
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return task.result()
        finally:
            self._waiters[key] -= 1
            if not self._waiters[key]:
                del self._waiters[key]
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    async def _execute(
        self, request: StageRequest, handler: Callable[[StageRequest], Awaitable[BaseModel]]
    ) -> StageResult:
        if config.DesignConfig.STATE_DIR and design_store.store is None:
            raise design_store.RecoveryBlocked("single_process_state_owner_required")
        context = StageContext(
            min(request.token_limit, config.DesignConfig.TOKEN_LIMIT),
            time.monotonic() + min(request.timeout_seconds, config.DesignConfig.TIMEOUT_SECONDS),
        )
        if design_store.store:
            run = await design_store.store.call("run", request.run_id)
            if run["state"] != "running":
                raise design_store.RecoveryBlocked("run_not_running")
            if not request.stage_id.endswith(f"-g{run['generation']}"):
                raise design_store.RecoveryBlocked("stale_stage_generation")
            context.run_id, context.generation = request.run_id, run["generation"]
            context.task_key = f"{request.operation}-{request.input_revision}"
            context.budget_ceiling = run["used"] + request.token_limit
            await design_store.store.call("snapshot", request.run_id, run["generation"], {"phase": request.operation})
        token = current_stage.set(context)
        status = "completed"
        error = None
        error_code = None
        error_details = {}
        payload = {}
        try:
            async with asyncio.timeout(min(request.timeout_seconds, config.DesignConfig.TIMEOUT_SECONDS)):
                payload = (await handler(request)).model_dump(mode="json")
        except BudgetExceeded as exc:
            status, error = "failed", str(exc)
            error_code, error_details = exc.code, exc.details
        except ModelOutputFailure as exc:
            status, error, error_code = "failed", str(exc), exc.code
        except ProviderRetryError as exc:
            status, error = "failed", str(exc)
            error_code, error_details = exc.code, exc.details
        except TimeoutError:
            status, error = "timed_out", "QA stage exceeded its deadline"
        except asyncio.CancelledError:
            status, error = "canceled", "QA stage canceled"
        except Exception as exc:
            # Do not put PRD/tool contents or provider response bodies into stored errors.
            status, error = "failed", f"QA stage failed ({type(exc).__name__})"
        finally:
            current_stage.reset(token)
        return StageResult(
            run_id=request.run_id,
            stage_id=request.stage_id,
            input_revision=request.input_revision,
            execution_id=str(uuid4()),
            status=status,
            payload=payload,
            error=error,
            error_code=error_code,
            error_details=error_details,
            usage=context.usage,
            trace=context.trace,
        )
