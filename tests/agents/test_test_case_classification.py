# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from a2a.helpers import new_text_message
from a2a.types import Role

from agents.test_case_classification.main import TestCaseClassificationAgent, _parse_test_cases_from_text
from common.models import ClassifiedTestCase, ClassifiedTestCases, TestCase, TestStep


@pytest.fixture
def mock_config():
    with patch("agents.test_case_classification.main.config") as mock_conf:
        mock_conf.TestCaseClassificationAgentConfig.OWN_NAME = "classification_agent"
        mock_conf.AGENT_BASE_URL = "http://localhost"
        mock_conf.TestCaseClassificationAgentConfig.PORT = 8001
        mock_conf.TestCaseClassificationAgentConfig.EXTERNAL_PORT = 8001
        mock_conf.TestCaseClassificationAgentConfig.PROTOCOL = "http"
        mock_conf.TestCaseClassificationAgentConfig.MODEL_NAME = "test"
        mock_conf.TestCaseClassificationAgentConfig.FALLBACK_MODEL_NAME = "test"
        mock_conf.TestCaseClassificationAgentConfig.THINKING_LEVEL = "LOW"
        mock_conf.TestCaseClassificationAgentConfig.MAX_REQUESTS_PER_TASK = 5
        mock_conf.TestCaseClassificationAgentConfig.BATCH_SIZE = 2
        yield mock_conf


@pytest.fixture
def agent(mock_config):
    # Patch PromptBase.get_prompt to avoid file reading issues, and get_model to avoid
    # constructing a real Anthropic provider/client during tests.
    with (
        patch(
            "agents.test_case_classification.prompt.TestCaseClassificationSystemPrompt.get_prompt",
            return_value="Prompt",
        ),
        patch(
            "agents.test_case_classification.prompt.TestCaseClassificationBatchPrompt.get_prompt",
            return_value="Batch prompt",
        ),
        patch("agents.test_case_classification.main.get_model", return_value="test"),
        patch("common.agent_base.get_model", return_value="test"),
    ):
        return TestCaseClassificationAgent()


def test_agent_init(agent, mock_config):
    assert agent.agent_name == "classification_agent"
    assert agent.get_thinking_level() == "LOW"
    assert agent.get_max_requests_per_task() == 5


def _make_tc(name: str = "TC-1") -> TestCase:
    return TestCase(
        key=None,
        name=name,
        summary="s",
        comment="",
        preconditions=None,
        parent_issue_key=None,
        steps=[TestStep(action="a", expected_results="r", test_data=[])],
        labels=[],
    )


def _classified(issue_key: str) -> ClassifiedTestCase:
    return ClassifiedTestCase(
        issue_key=issue_key,
        name=issue_key,
        test_type="UI",
        automation_capability="automated",
        labels=["ui", "automated"],
        tool_use_comment="",
    )


def test_parse_test_cases_plain():
    text = f"Test cases:\n{json.dumps([_make_tc().model_dump()])}"
    result = _parse_test_cases_from_text(text)
    assert len(result) == 1
    assert result[0].name == "TC-1"


def test_parse_test_cases_returns_empty_when_no_marker():
    assert _parse_test_cases_from_text("no marker here") == []


def test_parse_test_cases_returns_empty_on_invalid_json():
    assert _parse_test_cases_from_text("Test cases:\nnot json") == []


@pytest.mark.asyncio
async def test_run_returns_result_set_by_tool_in_separate_task(agent):
    """The tool writes ctx.deps.result; run() reads it back even when the tool runs in a
    separate asyncio.Task (as pydantic-ai does), because deps is a mutable object shared
    by reference — not a ContextVar copy."""
    expected = ClassifiedTestCases(test_cases=[_classified("TC-1")])

    async def fake_super_run(received_message, deps=None):
        async def _tool():
            deps.result = expected

        task = asyncio.create_task(_tool())
        await task
        return new_text_message(text="fallback", role=Role.ROLE_AGENT)

    with (
        patch("agents.test_case_classification.main.AgentBase.run", side_effect=fake_super_run),
        patch("agents.test_case_classification.main._parse_test_cases_from_text", return_value=[_make_tc()]),
    ):
        received_message = new_text_message(text="classify these test cases", role=Role.ROLE_USER)
        result = await agent.run(received_message)

    assert ClassifiedTestCases.model_validate_json(result.parts[0].text) == expected


@pytest.mark.asyncio
async def test_run_falls_back_when_tool_never_sets_result(agent):
    """When the tool is never invoked (no test cases parsed), state.result stays None and
    run() returns the raw LLM message rather than attempting to parse classifications."""
    fallback_message = new_text_message(text="fallback", role=Role.ROLE_AGENT)

    async def fake_super_run(received_message, deps=None):
        return fallback_message

    with (
        patch("agents.test_case_classification.main.AgentBase.run", side_effect=fake_super_run),
        patch("agents.test_case_classification.main._parse_test_cases_from_text", return_value=[]),
    ):
        received_message = new_text_message(text="no test cases here", role=Role.ROLE_USER)
        result = await agent.run(received_message)

    assert result is fallback_message


@pytest.mark.asyncio
async def test_classify_test_cases_splits_into_batches_and_aggregates(agent):
    """Classification must split the full test case set into BATCH_SIZE-sized batches (so a
    large set never travels in a single LLM call) and aggregate every batch's results."""
    test_cases = [_make_tc(f"TC-{i}") for i in range(1, 6)]  # 5 cases, BATCH_SIZE=2 -> 3 batches
    ctx = type("Ctx", (), {"deps": type("Deps", (), {"test_cases": test_cases, "result": None})()})()

    batch_calls: list[int] = []

    async def fake_classify_batch(batch, batch_number, total_batches):
        batch_calls.append(len(batch))
        return ClassifiedTestCases(test_cases=[_classified(tc.name) for tc in batch])

    with patch.object(agent, "_classify_batch", side_effect=fake_classify_batch):
        summary = await agent._classify_test_cases(ctx)

    assert sorted(batch_calls) == [1, 2, 2]
    assert ctx.deps.result is not None
    assert len(ctx.deps.result.test_cases) == 5
    assert "5" in summary
