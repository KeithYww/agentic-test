# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

import asyncio
import json
from dataclasses import dataclass, field
from itertools import batched

from a2a.helpers import get_message_text, new_text_message
from a2a.types import AgentSkill, Message
from pydantic_ai import RunContext
from pydantic_ai.settings import ThinkingLevel

import config
from agents.test_case_classification.prompt import TestCaseClassificationBatchPrompt, TestCaseClassificationSystemPrompt
from common import utils
from common.agent_base import AgentBase
from common.custom_llm_wrapper import CustomLlmWrapper
from common.design_models import StageRequest, is_stage_message
from common.llm_provider import get_model
from common.models import ClassifiedTestCases, TestCase
from common.provider_errors import ProviderRetryError

logger = utils.get_logger("test_case_classification_agent")


@dataclass(slots=True)
class _ClassificationRunState:
    """Per-request state container for TestCaseClassificationAgent, threaded via pydantic-ai deps."""

    test_cases: list[TestCase] = field(default_factory=list)
    result: ClassifiedTestCases | None = None


def _parse_test_cases_from_text(text: str) -> list[TestCase]:
    """Extracts the test cases list from the orchestrator's "Test cases:\\n<json array>" message."""
    marker = "Test cases:\n"
    idx = text.find(marker)
    if idx == -1:
        return []
    try:
        raw = json.loads(text[idx + len(marker):])
    except json.JSONDecodeError:
        return []
    return [TestCase.model_validate(item) for item in raw]


class TestCaseClassificationAgent(AgentBase):
    __test__ = False

    def __init__(self):
        # Sub-agent that classifies one batch of test cases per call, keeping each call's
        # context (and token usage) bounded instead of sending the whole set in one shot.
        self.classification_agent = CustomLlmWrapper.create_agent(
            model_name=get_model(
                config.TestCaseClassificationAgentConfig.MODEL_NAME,
                config.TestCaseClassificationAgentConfig.FALLBACK_MODEL_NAME,
                bounded_requests=True,
            ),
            output_type=ClassifiedTestCases,
            system_prompt=TestCaseClassificationBatchPrompt().get_prompt(),
            name="classify_test_cases_batch",
            thinking_level=config.TestCaseClassificationAgentConfig.THINKING_LEVEL,
            max_tokens=config.TestCaseClassificationAgentConfig.MAX_TOKENS,
        )

        instruction_prompt = TestCaseClassificationSystemPrompt()
        super().__init__(
            agent_name=config.TestCaseClassificationAgentConfig.OWN_NAME,
            base_url=config.AGENT_BASE_URL,
            port=config.TestCaseClassificationAgentConfig.PORT,
            external_port=config.TestCaseClassificationAgentConfig.EXTERNAL_PORT,
            protocol=config.TestCaseClassificationAgentConfig.PROTOCOL,
            model_name=config.TestCaseClassificationAgentConfig.MODEL_NAME,
            fallback_model_name=config.TestCaseClassificationAgentConfig.FALLBACK_MODEL_NAME,
            deps_type=_ClassificationRunState,
            output_type=ClassifiedTestCases,
            instructions=instruction_prompt.get_prompt(),
            mcp_servers=[],
            description="Agent which classifies test cases based on their content",
            tools=[self._classify_test_cases],
        )

    def get_thinking_level(self) -> ThinkingLevel:
        return config.TestCaseClassificationAgentConfig.THINKING_LEVEL

    def get_max_requests_per_task(self) -> int:
        return config.TestCaseClassificationAgentConfig.MAX_REQUESTS_PER_TASK

    def get_total_tokens_limit(self) -> int:
        return config.TestCaseClassificationAgentConfig.TOTAL_TOKENS_LIMIT_PER_TASK

    def get_max_tokens(self) -> int | None:
        # The main orchestrating LLM only emits a short summary; the full result travels
        # via deps, so a small limit is sufficient and leaves more room for input.
        return 4096

    def get_skills(self) -> list[AgentSkill]:
        return [
            AgentSkill(
                id="classify-test-cases",
                name="Classify test cases",
                description="Classifies test cases based on their content",
                tags=["qa", "test-case-classification"],
            )
        ]

    async def run(self, received_message: Message) -> Message:
        text = get_message_text(received_message)
        if is_stage_message(text):
            return await super().run(received_message)
        test_cases = _parse_test_cases_from_text(text)
        context_id = getattr(received_message, "context_id", None)
        task_id = getattr(received_message, "task_id", None)
        state = _ClassificationRunState()
        if test_cases:
            state.test_cases = test_cases
            # Pass only a minimal instruction to the main LLM — the test cases travel via
            # deps, not through the main agent's own context window.
            received_message = new_text_message(
                text="Classify the test cases.", context_id=context_id, task_id=task_id
            )
        else:
            logger.warning("Could not extract test cases from classification request message.")
        msg = await super().run(received_message, deps=state)
        if state.result is None:
            return msg
        return new_text_message(text=state.result.model_dump_json(), context_id=context_id, task_id=task_id)

    async def execute_stage(self, request: StageRequest) -> ClassifiedTestCases:
        """Classify only the final active cases supplied by the design coordinator."""
        if request.operation != "classify":
            raise ValueError("Classification agent only supports classify")
        cases = [TestCase.model_validate(c) for c in request.payload["test_cases"]]
        context = _ClassificationRunState(test_cases=cases)
        # Reuse the existing batching implementation, without a routing-model round trip.
        from types import SimpleNamespace

        await self._classify_test_cases(SimpleNamespace(deps=context))
        return context.result

    async def _classify_test_cases(self, ctx: RunContext[_ClassificationRunState]) -> str:
        """
        Classifies all test cases from the request in batches.

        Reads test cases from ctx.deps; stores the result back in ctx.deps.result.

        Returns:
            A compact completion summary.
        """
        test_cases = ctx.deps.test_cases
        if not test_cases:
            raise RuntimeError("No test cases available for classification.")
        batch_size = config.TestCaseClassificationAgentConfig.BATCH_SIZE
        batches = list(batched(test_cases, batch_size, strict=False))
        logger.info(f"Classifying {len(test_cases)} test cases in {len(batches)} batch(es) of up to {batch_size}.")
        tasks = [self._classify_batch(batch, i + 1, len(batches)) for i, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_classified: list = []
        llm_comments = []
        for i, result in enumerate(results):
            if isinstance(result, ProviderRetryError):
                raise result
            if isinstance(result, BaseException):
                raise RuntimeError(f"Classification failed for batch {i + 1}: {result}") from result
            all_classified.extend(result.test_cases)
            if result.llm_comments:
                llm_comments.append(result.llm_comments)
        ctx.deps.result = ClassifiedTestCases(test_cases=all_classified, llm_comments="\n".join(llm_comments) or None)
        return f"Classification completed. Classified {len(all_classified)} test cases."

    async def _classify_batch(self, batch: list[TestCase], batch_number: int, total_batches: int) -> ClassifiedTestCases:
        logger.info(f"Starting classification of batch {batch_number}/{total_batches}.")
        from common.design_runtime import checkpoint_agent
        classified = await checkpoint_agent(
            f"classification:{batch_number}", self.classification_agent,
            f"Test cases:\n{json.dumps([tc.model_dump() for tc in batch], ensure_ascii=False)}", ClassifiedTestCases,
        )
        logger.info(f"Classified batch {batch_number}/{total_batches}: {len(classified.test_cases)} test cases.")
        return classified


agent = TestCaseClassificationAgent()
app = agent.a2a_server

if __name__ == "__main__":
    agent.start_as_server()
