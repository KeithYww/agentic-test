# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.helpers import new_text_message
from a2a.types import Role

from agents.test_case_review.main import TestCaseReviewAgent, _parse_test_cases_from_text, _ReviewRunState
from common.design_models import StageRequest
from common.models import TestCase, TestCaseReviewFeedback, TestCaseReviewFeedbacks, TestStep


def test_review_model_routing_is_separate_from_generation_and_repair() -> None:
    import config

    with (
        patch("agents.test_case_review.main.get_model", return_value="test") as models,
        patch("common.agent_base.get_model", return_value="test"),
        patch("common.custom_llm_wrapper.Agent"),
    ):
        TestCaseReviewAgent()
    assert models.call_args.args == ("claude-sonnet-5", "gpt-5.6-sol")
    generation = config.TestCaseGenerationAgentConfig
    review = config.TestCaseReviewAgentConfig
    assert {review.MODEL_NAME, review.FALLBACK_MODEL_NAME}.isdisjoint({
        generation.MODEL_NAME, generation.FALLBACK_MODEL_NAME, generation.REPAIR_MODEL_NAME,
    })


@pytest.mark.asyncio
@pytest.mark.parametrize("always_invalid", [False, True])
async def test_stage_retries_unknown_requirement_ids(agent, always_invalid):
    from pydantic_ai import Agent
    from pydantic_ai.exceptions import UnexpectedModelBehavior
    from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    calls = {"_CaseReviewBatch": 0, "SuiteAssessment": 0}
    retries = []

    def create_agent(**kwargs):
        output_type = kwargs["output_type"]
        name = output_type.__name__

        def respond(messages, info):
            calls[name] += 1
            retries.extend(part.content for message in messages for part in message.parts
                           if isinstance(part, RetryPromptPart))
            finding = {
                "severity": "medium", "action": "modify", "owner_case_key": "TC-1",
                "related_case_keys": [], "requirement_ids": ["3.3" if always_invalid or calls[name] == 1 else "R1"],
                "category": "coverage", "reason": "缺少边界", "expected_change": "补充边界",
            }
            output = {"findings": [finding]}
            output.update({"case_key": "TC-1", "case_version": 1} if name == "_CaseReviewBatch"
                          else {"completed": True})
            if name == "_CaseReviewBatch":
                output = {"assessments": [output]}
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, json.dumps(output))])

        return Agent(FunctionModel(respond), output_type=output_type, output_retries=1)

    request = StageRequest(
        run_id="review-test", stage_id="review-1", operation="review", required_skill_id="review",
        token_limit=10000, timeout_seconds=30,
        payload={"document": "需求", "requirements": {"items": [{"id": "R1", "text": "边界"}]},
                 "knowledge": {}, "test_cases": [{"key": "TC-1", "version": 1, "covered_requirement_ids": ["R1"]}]},
    )
    with patch("agents.test_case_review.main.CustomLlmWrapper.create_agent", side_effect=create_agent):
        if always_invalid:
            with pytest.raises(UnexpectedModelBehavior):
                await agent.execute_stage(request)
            assert calls == {"_CaseReviewBatch": 2, "SuiteAssessment": 0}
        else:
            result = await agent.execute_stage(request)
            assert all(f.requirement_ids == ["R1"] for f in result.findings())
            assert calls == {"_CaseReviewBatch": 2, "SuiteAssessment": 2}
    assert any("R1" in str(prompt) for prompt in retries)



@pytest.mark.asyncio
async def test_suite_review_must_address_every_uncovered_requirement(agent):
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart, UserPromptPart
    from pydantic_ai.models.function import FunctionModel

    suite_calls, prompts, retries = [], [], []

    def create_agent(**kwargs):
        output_type = kwargs["output_type"]

        def respond(messages, info):
            for message in messages:
                for part in message.parts:
                    if isinstance(part, UserPromptPart):
                        prompts.append(str(part.content))
                    if isinstance(part, RetryPromptPart):
                        retries.append(str(part.content))
            if output_type.__name__ == "_CaseReviewBatch":
                output = {"assessments": [{"case_key": "TC-1", "case_version": 1, "findings": []}]}
            else:
                suite_calls.append(1)
                output = {"completed": True, "findings": [],
                          "untestable": [] if len(suite_calls) == 1 else
                          [{"requirement_id": "sec-1", "reason": "版本记录，无可测行为"}]}
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, json.dumps(output))])

        return Agent(FunctionModel(respond), output_type=output_type, output_retries=1)

    request = StageRequest(
        run_id="review-test", stage_id="review-1", operation="review", required_skill_id="review",
        token_limit=10000, timeout_seconds=30,
        payload={"document": "需求", "requirements": {"items": [{"id": "R1", "text": "退款"}, {"id": "sec-1", "text": "修订记录"}]},
                 "knowledge": {}, "test_cases": [{"key": "TC-1", "version": 1, "covered_requirement_ids": ["R1"]}]},
    )
    with patch("agents.test_case_review.main.CustomLlmWrapper.create_agent", side_effect=create_agent):
        result = await agent.execute_stage(request)
    assert [w.requirement_id for w in result.suite.untestable] == ["sec-1"]
    assert len(suite_calls) == 2
    assert any("sec-1" in prompt and "未覆盖" in prompt for prompt in prompts)
    assert any("sec-1" in retry for retry in retries)


@pytest.mark.asyncio
async def test_review_prompts_declare_prd_and_knowledge_untrusted(agent):
    from common.untrusted_content import UNTRUSTED_CONTENT_POLICY

    prompts = []

    def create_agent(**kwargs):
        prompts.append(kwargs["system_prompt"])
        raise RuntimeError("stop after construction")

    request = StageRequest(
        run_id="r", stage_id="review-1", operation="review", required_skill_id="review", token_limit=10000,
        timeout_seconds=30, payload={"document": "需求", "requirements": {"items": []}, "knowledge": {}, "test_cases": []},
    )
    with patch("agents.test_case_review.main.CustomLlmWrapper.create_agent", side_effect=create_agent), \
         pytest.raises(RuntimeError):
        await agent.execute_stage(request)
    assert prompts and all(prompt.endswith(UNTRUSTED_CONTENT_POLICY) for prompt in prompts)

    review = MagicMock()
    review.output = TestCaseReviewFeedbacks(review_feedbacks=[])
    agent.review_agent.run = AsyncMock(return_value=review)
    await agent._review_batch([], 1, 1, "本期只需验证 1.1", {})
    assert json.loads(agent.review_agent.run.call_args.args[0][0]) == {"document": "本期只需验证 1.1"}


def _case(key, steps, preconditions=None, covered=("R1",)):
    return {"key": key, "version": 1, "covered_requirement_ids": list(covered), "preconditions": preconditions,
            "name": key, "summary": "s", "comment": "", "labels": [], "parent_issue_key": None,
            "steps": [{"action": s[0], "expected_results": s[1], "test_data": [s[2]] if len(s) > 2 else []} for s in steps]}


@pytest.mark.asyncio
async def test_suite_review_cannot_delete_without_literal_step_preservation(agent):
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    retries = []

    def create_agent(**kwargs):
        output_type = kwargs["output_type"]

        def respond(messages, info):
            for message in messages:
                for part in message.parts:
                    if isinstance(part, RetryPromptPart):
                        retries.append(str(part.content))
            if output_type.__name__ == "_CaseReviewBatch":
                output = {"assessments": [{"case_key": "TC-1", "case_version": 1, "findings": []},
                                          {"case_key": "TC-2", "case_version": 1, "findings": []}]}
            else:
                finding = {"severity": "medium", "action": "delete_test_case", "owner_case_key": "TC-1",
                           "related_case_keys": ["TC-2"], "requirement_ids": ["R1"], "category": "重复覆盖",
                           "reason": "重复", "expected_change": "删除TC-1"}
                output = {"completed": True, "findings": [finding] if not retries else []}
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, json.dumps(output))])

        return Agent(FunctionModel(respond), output_type=output_type, output_retries=1)

    payload = {
        "document": "需求", "requirements": {"items": [{"id": "R1", "text": "规则"}]}, "knowledge": {},
        "test_cases": [
            _case("TC-1", [("提交交易", "不发返现", "金额100")], preconditions="同一前置"),
            _case("TC-2", [("提交交易", "不发返现", "金额200")], preconditions="同一前置"),
        ],
    }
    request = StageRequest(run_id="r", stage_id="review-1", operation="review", required_skill_id="review",
                           token_limit=100000, timeout_seconds=30, payload=payload)
    with patch("agents.test_case_review.main.CustomLlmWrapper.create_agent", side_effect=create_agent):
        result = await agent.execute_stage(request)
    assert not any(f.action == "delete_test_case" for f in result.suite.findings)
    assert any("逐字保留" in retry or "modify" in retry for retry in retries)


@pytest.mark.asyncio
async def test_suite_review_allows_delete_with_literal_step_preservation(agent):
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    def create_agent(**kwargs):
        output_type = kwargs["output_type"]

        def respond(messages, info):
            if output_type.__name__ == "_CaseReviewBatch":
                output = {"assessments": [{"case_key": "TC-1", "case_version": 1, "findings": []},
                                          {"case_key": "TC-2", "case_version": 1, "findings": []}]}
            else:
                finding = {"severity": "medium", "action": "delete_test_case", "owner_case_key": "TC-1",
                           "related_case_keys": ["TC-2"], "requirement_ids": ["R1"], "category": "重复覆盖",
                           "reason": "完全重复", "expected_change": "删除TC-1"}
                output = {"completed": True, "findings": [finding]}
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, json.dumps(output))])

        return Agent(FunctionModel(respond), output_type=output_type, output_retries=1)

    step = ("提交交易", "不发返现")
    payload = {
        "document": "需求", "requirements": {"items": [{"id": "R1", "text": "规则"}]}, "knowledge": {},
        "test_cases": [
            _case("TC-1", [step], preconditions="同一前置"),
            _case("TC-2", [step], preconditions="同一前置"),
        ],
    }
    request = StageRequest(run_id="r", stage_id="review-1", operation="review", required_skill_id="review",
                           token_limit=100000, timeout_seconds=30, payload=payload)
    with patch("agents.test_case_review.main.CustomLlmWrapper.create_agent", side_effect=create_agent):
        result = await agent.execute_stage(request)
    assert [f.action for f in result.suite.findings] == ["delete_test_case"]


@pytest.mark.asyncio
async def test_suite_review_is_split_by_case_and_requirement_ownership(agent):
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    suite_calls = []

    def create_agent(**kwargs):
        output_type = kwargs["output_type"]

        def respond(messages, info):
            if output_type.__name__ == "_CaseReviewBatch":
                keys = {c["key"]: c["version"] for c in payload["test_cases"]}
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, json.dumps(
                    {"assessments": [{"case_key": k, "case_version": v, "findings": []} for k, v in keys.items()]}
                ))])
            prompt = next(str(p.content) for m in messages for p in m.parts if hasattr(p, "content"))
            owned = json.loads(prompt[prompt.index("{"):])
            suite_calls.append(owned)
            gaps = [{"severity": "medium", "action": "add_test_case", "owner_case_key": None, "related_case_keys": [],
                    "requirement_ids": [rid], "category": "coverage", "reason": "缺口", "expected_change": "补充"}
                   for rid in owned["owned_requirement_ids"]]
            output = {"completed": True, "findings": gaps, "untestable": []}
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, json.dumps(output))])

        return Agent(FunctionModel(respond), output_type=output_type, output_retries=1)

    payload = {
        "document": "需求", "requirements": {"items": [{"id": f"R{i}", "text": "规则"} for i in range(1, 7)]},
        "knowledge": {}, "test_cases": [_case(f"TC-{i}", [(f"步骤{i}", f"结果{i}")], covered=(f"R{i}",)) for i in range(1, 5)],
    }
    request = StageRequest(run_id="r", stage_id="review-1", operation="review", required_skill_id="review",
                           token_limit=200000, timeout_seconds=30, payload=payload)
    with patch("agents.test_case_review.main._SUITE_REVIEW_BATCH_SIZE", 2), \
         patch("agents.test_case_review.main.CustomLlmWrapper.create_agent", side_effect=create_agent):
        result = await agent.execute_stage(request)

    assert len(suite_calls) == 2
    assert suite_calls[0]["owned_case_keys"] == ["TC-1", "TC-2"]
    assert suite_calls[0]["owned_requirement_ids"] == ["R5", "R6"]
    assert suite_calls[1]["owned_case_keys"] == ["TC-3", "TC-4"]
    assert suite_calls[1]["owned_requirement_ids"] == []
    assert all(len(c["test_cases"]) == 4 for c in suite_calls), "every batch must see the full case list"
    assert {ref for f in result.suite.findings for ref in f.requirement_ids} == {"R5", "R6"}
    assert result.suite.completed


@pytest.mark.asyncio
async def test_suite_review_batch_cannot_act_outside_its_ownership(agent):
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    retries, suite_call_count = [], 0

    def create_agent(**kwargs):
        output_type = kwargs["output_type"]

        def respond(messages, info):
            nonlocal suite_call_count
            for message in messages:
                for part in message.parts:
                    if isinstance(part, RetryPromptPart):
                        retries.append(str(part.content))
            if output_type.__name__ == "_CaseReviewBatch":
                keys = {c["key"]: c["version"] for c in payload["test_cases"]}
                output = {"assessments": [{"case_key": k, "case_version": v, "findings": []} for k, v in keys.items()]}
            else:
                suite_call_count += 1
                # Batch 1 (owns only TC-1) wrongly proposes a finding owned by TC-2, assigned to batch 2.
                finding = {"severity": "low", "action": "modify", "owner_case_key": "TC-2", "related_case_keys": [],
                          "requirement_ids": ["R2"], "category": "coverage", "reason": "越权", "expected_change": "x"}
                misbehaving_first_call = suite_call_count == 1
                output = {"completed": True, "findings": [finding] if misbehaving_first_call else [], "untestable": []}
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, json.dumps(output))])

        return Agent(FunctionModel(respond), output_type=output_type, output_retries=1)

    payload = {
        "document": "需求", "requirements": {"items": [{"id": "R1", "text": "r"}, {"id": "R2", "text": "r"}]},
        "knowledge": {}, "test_cases": [_case("TC-1", [("步骤1", "结果1")], covered=("R1",)),
                                       _case("TC-2", [("步骤2", "结果2")], covered=("R2",))],
    }
    request = StageRequest(run_id="r", stage_id="review-1", operation="review", required_skill_id="review",
                           token_limit=200000, timeout_seconds=30, payload=payload)
    with patch("agents.test_case_review.main._SUITE_REVIEW_BATCH_SIZE", 1), \
         patch("agents.test_case_review.main.CustomLlmWrapper.create_agent", side_effect=create_agent):
        result = await agent.execute_stage(request)
    assert retries and "TC-2" in retries[0]
    assert result.suite.findings == []

@pytest.fixture
def mock_config():
    with patch("agents.test_case_review.main.config") as mock_conf:
        mock_conf.TestCaseReviewAgentConfig.OWN_NAME = "review_agent"
        mock_conf.AGENT_BASE_URL = "http://localhost"
        mock_conf.TestCaseReviewAgentConfig.PORT = 8003
        mock_conf.TestCaseReviewAgentConfig.EXTERNAL_PORT = 8003
        mock_conf.TestCaseReviewAgentConfig.PROTOCOL = "http"
        mock_conf.TestCaseReviewAgentConfig.MODEL_NAME = "test"
        mock_conf.TestCaseReviewAgentConfig.FALLBACK_MODEL_NAME = "test"
        mock_conf.TestCaseReviewAgentConfig.THINKING_LEVEL = "LOW"
        mock_conf.TestCaseReviewAgentConfig.MAX_REQUESTS_PER_TASK = 8
        mock_conf.TestCaseReviewAgentConfig.MAX_TOKENS = 32000
        yield mock_conf


@pytest.fixture
def agent(mock_config):
    # Patch PromptBase.get_prompt to avoid file reading issues, and get_model to avoid
    # constructing a real Anthropic provider/client during tests.
    with (
        patch("agents.test_case_review.prompt.TestCaseReviewSystemPrompt.get_prompt", return_value="Prompt"),
        patch("agents.test_case_review.main.get_model", return_value="test"),
        patch("common.agent_base.get_model", return_value="test"),
    ):
        return TestCaseReviewAgent()


def test_agent_init(agent, mock_config):
    assert agent.agent_name == "review_agent"
    assert agent.get_thinking_level() == "LOW"
    assert agent.get_max_requests_per_task() == 8

@pytest.mark.asyncio
async def test_run_returns_result_set_by_tool_in_separate_task(agent):
    """The tool writes ctx.deps.result; run() reads it back even when the tool runs in a
    separate asyncio.Task (as pydantic-ai does), because deps is a mutable object shared
    by reference — not a ContextVar copy."""
    expected_feedbacks = TestCaseReviewFeedbacks(
        review_feedbacks=[TestCaseReviewFeedback(test_case_id="TC-1", review_feedback=["Looks good"])]
    )

    async def fake_super_run(received_message, deps=None):
        # Simulate pydantic-ai running the tool in a child task.
        # The child inherits the deps reference (mutable object), so mutations
        # in the child are visible in the parent context without any ContextVar tricks.
        async def _tool():
            deps.result = expected_feedbacks

        task = asyncio.create_task(_tool())
        await task
        return new_text_message(text="fallback", role=Role.ROLE_AGENT)

    with (
        patch("agents.test_case_review.main.AgentBase.run", side_effect=fake_super_run),
        patch("agents.test_case_review.main._parse_test_cases_from_text", return_value=["TC-1"]),
        patch.object(agent, "_fetch_attachments", return_value={}),
        patch.object(agent, "_review_batch", new=AsyncMock(return_value=expected_feedbacks)),
    ):
        received_message = new_text_message(text="review these test cases", role=Role.ROLE_USER)
        result = await agent.run(received_message)

    assert TestCaseReviewFeedbacks.model_validate_json(result.parts[0].text) == expected_feedbacks


@pytest.mark.asyncio
async def test_run_falls_back_when_tool_never_sets_result(agent):
    """When the tool is never invoked (no test cases parsed), state.result stays None and
    run() returns the raw LLM message rather than attempting to parse feedbacks."""
    fallback_message = new_text_message(text="fallback", role=Role.ROLE_AGENT)

    async def fake_super_run(received_message, deps=None):
        # Tool is never called; deps.result stays None.
        return fallback_message

    with (
        patch("agents.test_case_review.main.AgentBase.run", side_effect=fake_super_run),
        patch("agents.test_case_review.main._parse_test_cases_from_text", return_value=[]),
    ):
        received_message = new_text_message(text="no test cases here", role=Role.ROLE_USER)
        result = await agent.run(received_message)

    assert result is fallback_message


def _make_tc_json(name: str = "TC-1") -> str:
    tc = TestCase(
        key=None,
        name=name,
        summary="s",
        comment="",
        preconditions=None,
        parent_issue_key=None,
        steps=[TestStep(action="a", expected_results="r", test_data=[])],
        labels=[],
    )
    import json
    return json.dumps({"test_cases": [tc.model_dump()]})


def test_parse_test_cases_plain():
    """Finds test_cases when the JSON is the only content."""
    result = _parse_test_cases_from_text(_make_tc_json())
    assert len(result) == 1


def test_parse_test_cases_preceded_by_json_without_key():
    """Skips a preceding JSON object that lacks 'test_cases' and finds the real one."""
    noise = '{"foo": "bar", "nested": {"x": 1}}'
    text = f"Some doc content with JSON: {noise}\n\nTest cases:\n{_make_tc_json()}"
    result = _parse_test_cases_from_text(text)
    assert len(result) == 1


def test_parse_test_cases_doc_content_with_braces():
    """Handles requirement doc content that contains multiple { } blocks before the test cases."""
    doc = "Function call: getUser({id: 1}) and another obj: {key: value}"
    text = f"Requirement document content:\n{doc}\n\nTest cases:\n{_make_tc_json('TC-2')}"
    result = _parse_test_cases_from_text(text)
    assert len(result) == 1
    assert result[0].name == "TC-2"


def test_parse_test_cases_returns_empty_when_no_json():
    assert _parse_test_cases_from_text("no json here at all") == []


def test_parse_test_cases_returns_empty_when_no_test_cases_key():
    assert _parse_test_cases_from_text('{"foo": "bar"}') == []


@pytest.mark.asyncio
async def test_case_review_retains_cross_requirement_evidence_and_only_own_history(agent):
    from agents.test_case_review.main import _CaseReviewBatch
    from common.design_models import CaseAssessment, DesignReview, SuiteAssessment

    previous = {"assessments": [{"findings": [
        {"owner_case_key": "TC-1", "reason": "own-history"},
        {"owner_case_key": "TC-2", "reason": "unrelated-history"},
    ]}]}
    case_agent = MagicMock()
    suite_agent = MagicMock()
    case_agent.run = AsyncMock(return_value=MagicMock(output=_CaseReviewBatch(assessments=[CaseAssessment(case_key="TC-1", case_version=1, findings=[])])))
    suite_agent.run = AsyncMock(return_value=MagicMock(output=SuiteAssessment(completed=True, findings=[])))
    payload = {"document": "完整PRD", "requirements": {"items": [{"id": "R1", "text": "R1完整"},
                                                                       {"id": "R2", "text": "R2跨需求约束"}]},
               "knowledge": {"content": "全局知识"},
               "test_cases": [{"key": "TC-1", "version": 1, "covered_requirement_ids": ["R1"]}],
               "previous_review": previous}
    request = StageRequest(run_id="r", stage_id="review", operation="review", required_skill_id="review-test-cases",
                           token_limit=100000, timeout_seconds=30, payload=payload)
    with patch("agents.test_case_review.main.CustomLlmWrapper.create_agent", side_effect=[case_agent, suite_agent]):
        result = await agent.execute_stage(request)
    prompt = case_agent.run.await_args.args[0]
    assert "R2跨需求约束" in prompt and "完整PRD" in prompt and "全局知识" in prompt
    assert "own-history" in prompt and "unrelated-history" not in prompt
    assert "R2跨需求约束" in suite_agent.run.await_args.args[0]
    assert isinstance(result, DesignReview) and result.suite.completed


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["missing", "duplicate", "version", "owner", "action"])
async def test_batch_review_retries_incomplete_or_misowned_receipts(agent, corruption):
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    calls = 0

    def create_agent(**kwargs):
        output_type = kwargs["output_type"]

        def respond(messages, info):
            nonlocal calls
            if output_type.__name__ == "SuiteAssessment":
                output = {"completed": True, "findings": []}
            else:
                calls += 1
                receipts = [{"case_key": f"TC-{i}", "case_version": 1, "findings": []} for i in range(1, 5)]
                if calls == 1:
                    if corruption == "missing":
                        receipts.pop()
                    elif corruption == "duplicate":
                        receipts[-1] = receipts[0]
                    elif corruption == "version":
                        receipts[0]["case_version"] = 2
                    else:
                        receipts[0]["findings"] = [{
                            "severity": "high", "action": "modify" if corruption == "owner" else "delete_test_case",
                            "owner_case_key": "TC-2" if corruption == "owner" else "TC-1",
                            "related_case_keys": ["TC-2"], "requirement_ids": ["R1"],
                            "category": "coverage", "reason": "问题", "expected_change": "补充",
                        }]
                output = {"assessments": receipts}
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, json.dumps(output))])

        return Agent(FunctionModel(respond), output_type=output_type, deps_type=kwargs.get("deps_type"), output_retries=1)

    payload = {"document": "完整PRD", "requirements": {"items": [{"id": "R1", "text": "规则"}]},
               "knowledge": {}, "test_cases": [{"key": f"TC-{i}", "version": 1, "covered_requirement_ids": ["R1"]} for i in range(1, 5)]}
    request = StageRequest(run_id="batch", stage_id="review", operation="review", required_skill_id="review",
                           token_limit=100000, timeout_seconds=30, payload=payload)
    with patch("agents.test_case_review.main.CustomLlmWrapper.create_agent", side_effect=create_agent):
        result = await agent.execute_stage(request)
    assert calls == 2
    assert {(r.case_key, r.case_version) for r in result.assessments} == {(f"TC-{i}", 1) for i in range(1, 5)}


@pytest.mark.asyncio
@pytest.mark.parametrize("capped_phase", ["case", "suite"])
@pytest.mark.parametrize("configured_limit", [8000, 24000, 32000])
async def test_design_review_uses_configured_output_limit(agent, mock_config, capped_phase, configured_limit):
    """Both design reviewers must honor the cap used for thinking and JSON output."""
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.usage import RequestUsage

    from common.design_runtime import StageRunner

    mock_config.TestCaseReviewAgentConfig.MAX_TOKENS = configured_limit
    caps = {}

    def respond(messages, info):
        tool = info.output_tools[0]
        phase = "case" if "assessments" in tool.parameters_json_schema["properties"] else "suite"
        cap = info.model_settings["max_tokens"]
        caps[phase] = cap
        required = (18000 if phase == "case" else 10000) if phase == capped_phase else 100
        if cap < required:
            return ModelResponse(parts=[], finish_reason="length", usage=RequestUsage(input_tokens=10, output_tokens=cap))
        output = ({"assessments": [{"case_key": "TC-1", "case_version": 1, "findings": []}]}
                  if phase == "case" else {"completed": True, "findings": []})
        return ModelResponse(parts=[ToolCallPart(tool.name, output)], finish_reason="tool_call",
                             usage=RequestUsage(input_tokens=10, output_tokens=required))

    request = StageRequest(
        run_id="review-cap", stage_id="review", operation="review", required_skill_id="review",
        token_limit=200000, timeout_seconds=30,
        payload={"document": "需求", "requirements": {"items": [{"id": "R1", "text": "规则"}]},
                 "knowledge": {}, "test_cases": [{"key": "TC-1", "version": 1, "covered_requirement_ids": ["R1"]}]},
    )
    with patch("agents.test_case_review.main.get_model", return_value=FunctionModel(respond)):
        result = await StageRunner().run(request, agent.execute_stage)

    if configured_limit == 8000:
        assert result.status == "failed" and result.error_code == "output_limit"
        assert result.payload == {} and result.usage.complete
        assert result.usage.requests == (1 if capped_phase == "case" else 2)
        return
    assert result.status == "completed", result.error
    assert result.payload["suite"]["completed"]
    assert caps == {"case": configured_limit, "suite": configured_limit}
    assert result.usage.complete and result.usage.requests == 2


@pytest.mark.asyncio
async def test_23_case_review_keeps_all_receipts_and_complete_suite(agent):
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
    from pydantic_ai.models.function import FunctionModel

    batches = []
    suites = []

    def create_agent(**kwargs):
        output_type = kwargs["output_type"]
        assert kwargs["max_tokens"] == 32000

        def respond(messages, info):
            prompt = next(p.content for message in reversed(messages) for p in message.parts
                          if isinstance(p, UserPromptPart))
            data = json.loads(prompt.split("\n", 1)[1])
            assert data["document"] == "完整PRD" and data["knowledge"] == {"content": "完整知识"}
            assert len(data["requirements"]["items"]) == 2
            if output_type.__name__ == "SuiteAssessment":
                suites.append(data)
                output = {"completed": True, "findings": []}
            else:
                batches.append(data)
                output = {"assessments": [{"case_key": c["key"], "case_version": c["version"], "findings": []}
                                           for c in data["test_cases"]]}
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, json.dumps(output))])

        return Agent(FunctionModel(respond), output_type=output_type, deps_type=kwargs.get("deps_type"))

    payload = {"document": "完整PRD", "requirements": {"items": [{"id": "R1", "text": "规则一"},
                                                                 {"id": "R2", "text": "跨例规则"}]},
               "knowledge": {"content": "完整知识"},
               "test_cases": [{"key": f"TC-{i}", "version": i, "covered_requirement_ids": ["R1", "R2"]} for i in range(1, 24)]}
    request = StageRequest(run_id="all-cases", stage_id="review", operation="review", required_skill_id="review",
                           token_limit=100000, timeout_seconds=30, payload=payload)
    with patch("agents.test_case_review.main.CustomLlmWrapper.create_agent", side_effect=create_agent):
        result = await agent.execute_stage(request)
    assert [len(b["test_cases"]) for b in batches] == [4, 4, 4, 4, 4, 3]
    # _SUITE_REVIEW_BATCH_SIZE=10 over 23 cases -> 3 disjoint, complete ownership batches, each seeing every case.
    assert [len(s["owned_case_keys"]) for s in suites] == [10, 10, 3]
    owned_union = {key for s in suites for key in s["owned_case_keys"]}
    assert owned_union == {c["key"] for c in payload["test_cases"]}
    assert all(suite["test_cases"] == payload["test_cases"] for suite in suites)
    assert len(result.assessments) == 23
    assert {(r.case_key, r.case_version) for r in result.assessments} == {(f"TC-{i}", i) for i in range(1, 24)}
