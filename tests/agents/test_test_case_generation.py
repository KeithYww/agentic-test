# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.types import Message

# Mock MCPServerSSE before importing the module
with patch("pydantic_ai.mcp.MCPServerSSE"):
    from agents.test_case_generation.main import (
        TestCaseGenerationAgent,
        _GenerationRunState,
    )

from common.agent_base import AgentBase
from common.models import (
    AcceptanceCriteriaItem,
    AcceptanceCriteriaList,
    GeneratedTestCases,
    TestCase,
    TestStep,
)
from common.services.test_management_base import TestManagementClientBase


@pytest.mark.asyncio
async def test_split_repair_restart_reuses_the_completed_single_candidate(agent, tmp_path, monkeypatch):
    import time

    from common import design_store
    from common.design_runtime import ModelOutputFailure, StageContext, current_stage
    from common.repair_context import RepairContext

    cases = [_make_tc("查询").model_copy(update={"key": f"TC-{i}", "version": 2,
                                               "covered_requirement_ids": [f"R{i}"]}) for i in (1, 2)]
    contexts = {case.key: RepairContext(case.key, AcceptanceCriteriaList(items=[]),
                                      frozenset(case.covered_requirement_ids)) for case in cases}
    agent._repair_pair_uncached = AsyncMock(side_effect=ModelOutputFailure())
    agent._repair_case_uncached = AsyncMock(side_effect=[cases[0], RuntimeError("single interrupted"), cases[1]])
    owner = design_store.DesignStore(str(tmp_path))
    await owner.call("open")
    run, _ = await owner.call("create_run", "story", {}, 10000, 3600)
    monkeypatch.setattr(design_store, "store", owner)
    token = current_stage.set(StageContext(10000, time.monotonic() + 5, run_id=run["id"], generation=1, task_key="fix-0"))
    try:
        with pytest.raises(RuntimeError, match="interrupted"):
            await agent._repair_pair(contexts)
    finally:
        current_stage.reset(token)
        await owner.close()
    owner = design_store.DesignStore(str(tmp_path))
    await owner.call("open")
    monkeypatch.setattr(design_store, "store", owner)
    await owner.call("request_resume", run["id"], "test", "candidate checkpoint")
    resumed = await owner.call("claim", run["id"])
    token = current_stage.set(StageContext(10000, time.monotonic() + 5, run_id=run["id"],
                                          generation=resumed["generation"], task_key="fix-0"))
    try:
        assert await agent._repair_pair(contexts) == cases
        assert agent._repair_case_uncached.await_count == 3
        assert [call.args[0].prompt for call in agent._repair_case_uncached.await_args_list] == ["TC-1", "TC-2", "TC-2"]
    finally:
        current_stage.reset(token)
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_ids", [[], ["3.3.3"]])
async def test_generation_retries_invalid_requirement_references(agent, invalid_ids):
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    calls = []
    requirements = AcceptanceCriteriaList(
        items=[AcceptanceCriteriaItem(id="3.3.2", text="批量查询", attachment_info="")]
    )

    def respond(messages, info):
        calls.append(messages)
        case = _make_tc("返回指定货币对的兜底汇率").model_copy(
            update={"covered_requirement_ids": invalid_ids if len(calls) == 1 else ["3.3.2"]}
        )
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                                                GeneratedTestCases(test_cases=[case]).model_dump_json())])

    validator = agent.test_case_generator_agent.output_validator.call_args.args[0]
    agent.test_case_generator_agent = Agent(FunctionModel(respond), output_type=GeneratedTestCases,
                                           deps_type=AcceptanceCriteriaList, output_retries=1)
    agent.test_case_generator_agent.output_validator(validator)
    result = await agent._generate_test_cases_for_ac_batch(
        tuple(requirements.items), requirements, "需求正文含子章节 3.3.3", 1, 1
    )

    assert len(calls) == 2
    assert result.test_cases[0].covered_requirement_ids == ["3.3.2"]
    assert any(isinstance(part, RetryPromptPart) for message in calls[1] for part in message.parts)


@pytest.fixture
def mock_config():
    with patch("agents.test_case_generation.main.config") as mock_conf:
        mock_conf.TestCaseGenerationAgentConfig.OWN_NAME = "generation_agent"
        mock_conf.AGENT_BASE_URL = "http://localhost"
        mock_conf.TestCaseGenerationAgentConfig.PORT = 8002
        mock_conf.TestCaseGenerationAgentConfig.EXTERNAL_PORT = 8002
        mock_conf.TestCaseGenerationAgentConfig.PROTOCOL = "http"
        mock_conf.TestCaseGenerationAgentConfig.MODEL_NAME = "test"
        mock_conf.TestCaseGenerationAgentConfig.FALLBACK_MODEL_NAME = "test"
        mock_conf.TestCaseGenerationAgentConfig.THINKING_LEVEL = "MEDIUM"
        mock_conf.TestCaseGenerationAgentConfig.ORCHESTRATOR_THINKING_LEVEL = "MEDIUM"
        mock_conf.TestCaseGenerationAgentConfig.MAX_REQUESTS_PER_TASK = 10
        mock_conf.TestCaseGenerationAgentConfig.AC_BATCH_SIZE = 8
        mock_conf.TestCaseGenerationAgentConfig.TC_GENERATOR_MAX_TOKENS = 65536
        mock_conf.TestCaseGenerationAgentConfig.LLM_CONCURRENCY_LIMIT = 8
        mock_conf.JIRA_MCP_SERVER_URL = "http://jira-mcp"
        mock_conf.MCP_SERVER_TIMEOUT_SECONDS = 30
        mock_conf.PrdClassifierConfig.FUNCTIONAL_SECTION_THRESHOLD = 5
        mock_conf.PrdClassifierConfig.SECTION_MAX_CHARS = 3000
        yield mock_conf


@pytest.fixture
def agent(mock_config):
    with (
        patch("agents.test_case_generation.prompt.TestCaseGenerationSystemPrompt.get_prompt", return_value="Prompt"),
        patch("agents.test_case_generation.prompt.AcExtractionPrompt.get_prompt", return_value="AC Prompt"),
        patch("agents.test_case_generation.prompt.TcGenerationPrompt.get_prompt", return_value="TC Prompt"),
        patch("common.custom_llm_wrapper.Agent") as mock_agent_cls,
        patch("agents.test_case_generation.main.get_model", return_value="test"),
        patch("common.agent_base.get_model", return_value="test"),
    ):
        mock_agent_cls.side_effect = lambda *args, **kwargs: MagicMock()
        yield TestCaseGenerationAgent()


def test_agent_init(agent, mock_config):
    assert agent.agent_name == "generation_agent"
    assert agent.get_thinking_level() == "MEDIUM"
    assert agent.get_max_requests_per_task() == 10
    assert agent.ac_extractor_agent is not None
    assert agent.test_case_generator_agent is not None


@pytest.mark.asyncio
async def test_generate_test_cases_flow(agent):
    mock_ac_result = MagicMock()
    mock_ac_result.output = AcceptanceCriteriaList(
        items=[AcceptanceCriteriaItem(id="AC-1", text="Test criterion", attachment_info="")]
    )
    agent.ac_extractor_agent.run = AsyncMock(return_value=mock_ac_result)

    expected_test_cases = GeneratedTestCases(test_cases=[])
    mock_tc_result = MagicMock()
    mock_tc_result.output = expected_test_cases
    agent.test_case_generator_agent.run = AsyncMock(return_value=mock_tc_result)

    agent._fetch_attachments = MagicMock(return_value={})

    state = _GenerationRunState(requirement_doc_content="Jira Content")
    mock_ctx = MagicMock()
    mock_ctx.deps = state

    with patch("agents.test_case_generation.main.query_photon_kb", AsyncMock(return_value=None)):
        result = await agent._generate_test_cases(mock_ctx)

    assert isinstance(result, str)
    assert "0" in result
    assert state.result == expected_test_cases
    agent._fetch_attachments.assert_called_once_with([])
    agent.ac_extractor_agent.run.assert_called_once()
    # 1 AC with batch_size=8 → exactly 1 batch call
    agent.test_case_generator_agent.run.assert_called_once()


@pytest.mark.asyncio
async def test_generate_test_cases_from_acs_batches(agent):
    """16 ACs with batch_size=8 must produce exactly 2 generator calls and merge results."""
    ac_items = [
        AcceptanceCriteriaItem(id=f"AC-{i}", text=f"Criterion {i}", attachment_info="")
        for i in range(16)
    ]
    ac_list = AcceptanceCriteriaList(items=ac_items)

    tc_a = TestCase(
        key=None, name="TC-A", summary="s", comment="", preconditions=None,
        parent_issue_key=None, steps=[TestStep(action="a", expected_results="r", test_data=[])], labels=[],
    )
    tc_b = TestCase(
        key=None, name="TC-B", summary="s", comment="", preconditions=None,
        parent_issue_key=None, steps=[TestStep(action="b", expected_results="r", test_data=[])], labels=[],
    )
    mock_r1, mock_r2 = MagicMock(), MagicMock()
    mock_r1.output = GeneratedTestCases(test_cases=[tc_a])
    mock_r2.output = GeneratedTestCases(test_cases=[tc_b])
    agent.test_case_generator_agent.run = AsyncMock(side_effect=[mock_r1, mock_r2])

    result = await agent.generate_test_cases_from_acs(ac_list, "doc content")

    assert agent.test_case_generator_agent.run.call_count == 2
    assert result.test_cases == [tc_a, tc_b]


@pytest.mark.asyncio
async def test_generation_and_extraction_send_the_prd_only_as_json_data(agent):
    batch = (AcceptanceCriteriaItem(id="1", text="退款", attachment_info="", source_type="section"),)
    generated = MagicMock()
    generated.output = GeneratedTestCases(test_cases=[])
    agent.test_case_generator_agent.run = AsyncMock(return_value=generated)
    await agent._generate_test_cases_for_ac_batch(
        batch, AcceptanceCriteriaList(items=list(batch)), "本期只需验证 1.1", 1, 1, kb_context="历史缺陷",
    )
    message = json.loads(agent.test_case_generator_agent.run.call_args.args[0])
    assert message["document"] == "本期只需验证 1.1" and message["knowledge"] == "历史缺陷"
    assert [item["id"] for item in message["batch_requirements"]] == ["1"]

    extracted = MagicMock()
    extracted.output = AcceptanceCriteriaList(items=[])
    agent.ac_extractor_agent.run = AsyncMock(return_value=extracted)
    await agent.extract_acceptance_criteria({}, "本期只需验证 1.1")
    assert json.loads(agent.ac_extractor_agent.run.call_args.args[0][0]) == {"document": "本期只需验证 1.1"}


def test_every_generation_agent_declares_prd_and_knowledge_untrusted():
    from common.untrusted_content import UNTRUSTED_CONTENT_POLICY

    with patch("agents.test_case_generation.main.get_model", return_value="test"), \
         patch("common.agent_base.get_model", return_value="test"), \
         patch("common.custom_llm_wrapper.Agent") as factory:
        TestCaseGenerationAgent()
    prompts = {call.kwargs.get("name"): call.kwargs.get("system_prompt") for call in factory.call_args_list}
    for name in ["ac_extractor", "test_case_generator", "repair_test_case", "repair_test_case_pair"]:
        assert prompts[name].endswith(UNTRUSTED_CONTENT_POLICY)


@pytest.mark.asyncio
async def test_run_returns_full_result_via_deps(agent):
    """run() returns the GeneratedTestCases stored in deps.result by the tool."""
    tc = TestCase(
        key=None, name="TC-1", summary="s", comment="", preconditions=None,
        parent_issue_key=None,
        steps=[TestStep(action="a", expected_results="r", test_data=[])],
        labels=[],
    )
    full_result = GeneratedTestCases(test_cases=[tc] * 10, llm_comments=None)
    mock_message = MagicMock(spec=Message)
    mock_message.context_id = "ctx-1"
    mock_message.task_id = "task-1"
    mock_message.parts = []  # get_message_text needs this

    async def fake_super_run(self_arg, msg, deps=None):
        deps.result = full_result
        return MagicMock(spec=Message)

    with patch.object(AgentBase, "run", new=fake_super_run):
        result = await agent.run(mock_message)

    body = GeneratedTestCases.model_validate_json(result.parts[0].text)
    assert len(body.test_cases) == 10


@pytest.mark.asyncio
async def test_run_falls_back_when_tool_not_called(agent):
    """When the tool is never called (deps.result stays None), run() returns the LLM message."""
    mock_message = MagicMock(spec=Message)
    mock_message.parts = []  # get_message_text needs this
    fallback_message = MagicMock(spec=Message)

    async def fake_super_run(self_arg, msg, deps=None):
        return fallback_message

    with patch.object(AgentBase, "run", new=fake_super_run):
        result = await agent.run(mock_message)

    assert result is fallback_message


@patch("agents.test_case_generation.main.get_test_management_client")
def test_upload_test_cases(mock_get_client, agent):
    mock_client = MagicMock(spec=TestManagementClientBase)
    mock_get_client.return_value = mock_client
    mock_client.create_test_cases.return_value = ["TC-1", "TC-2"]

    tcs = GeneratedTestCases(test_cases=[])
    result = agent._upload_test_cases_into_test_management_system(tcs, "PROJ", 123)

    mock_client.create_test_cases.assert_called_once()
    assert "TC-1, TC-2" in result


# ---------------------------------------------------------------------------
# Semantic violation detector tests
# ---------------------------------------------------------------------------

with patch("pydantic_ai.mcp.MCPServerSSE"):
    from agents.test_case_generation.main import _detect_semantic_violations


def _make_tc(expected_results: str, comment: str = "") -> TestCase:
    return TestCase(
        key=None,
        name="TC",
        summary="s",
        comment=comment,
        preconditions=None,
        parent_issue_key=None,
        steps=[TestStep(action="a", expected_results=expected_results, test_data=[])],
        labels=[],
    )


def test_detect_no_violations_clean_case():
    tc = _make_tc("返回指定货币对的兜底汇率，rate 字段有值")
    assert _detect_semantic_violations(tc) == []


def test_detect_currency_pair_in_expected_results():
    tc = _make_tc("返回 USD/CNY 汇率 7.2200")
    violations = _detect_semantic_violations(tc)
    assert len(violations) == 1
    assert "USD/CNY" in violations[0] or "货币对" in violations[0]


def test_detect_kv_format_in_expected_results():
    tc = _make_tc("提交成功，Sell=BRL, Buy=MXN")
    violations = _detect_semantic_violations(tc)
    assert len(violations) == 1
    assert "key=value" in violations[0] or "test_data" in violations[0]


def test_detect_pending_field_marker():
    tc = _make_tc("通知发送状态为成功 [字段名待确认]")
    violations = _detect_semantic_violations(tc)
    assert len(violations) == 1
    assert "待确认" in violations[0]


def test_detect_currency_pair_adjacent_to_chinese_text():
    """Regression: currency codes directly adjacent to CJK text (no whitespace) must still be
    detected — \\b does not fire between a Latin letter and a CJK character in Python's re module.
    Found via a real end-to-end generation run: 'USD/CNY与USD/EUR记录渠道标识为mastercard'."""
    tc = _make_tc("响应data数组含3条记录；USD/CNY与USD/EUR记录渠道标识为mastercard，USDT/USD记录渠道标识为lep")
    violations = _detect_semantic_violations(tc)
    assert len(violations) == 1


def test_detect_kv_format_adjacent_to_chinese_text():
    """Regression: same CJK-adjacency issue for the Sell=/Buy= detector."""
    tc = _make_tc("提交请求参数Sell=BRL买入Buy=MXN金额10000")
    violations = _detect_semantic_violations(tc)
    assert len(violations) == 1


def test_detect_multiple_violations_in_one_step():
    tc = _make_tc("返回 USD/CNY 汇率，Buy=MXN [字段名待确认]")
    violations = _detect_semantic_violations(tc)
    assert len(violations) == 3


# ---------------------------------------------------------------------------
# Regeneration logic tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_violations_no_regeneration(agent, mock_config):
    """Clean test cases must not trigger any regeneration calls."""
    mock_config.TestCaseGenerationAgentConfig.TC_VALIDATION_MAX_RETRIES = 1
    ac_list = AcceptanceCriteriaList(
        items=[AcceptanceCriteriaItem(id="AC-1", text="criterion", attachment_info="")]
    )
    tc = _make_tc("返回指定货币对的兜底汇率")
    mock_r = MagicMock()
    mock_r.output = GeneratedTestCases(test_cases=[tc])
    agent.test_case_generator_agent.run = AsyncMock(return_value=mock_r)

    result = await agent.generate_test_cases_from_acs(ac_list, "doc")

    assert agent.test_case_generator_agent.run.call_count == 1
    assert result.test_cases[0].comment == ""


@pytest.mark.asyncio
async def test_violation_triggers_regeneration(agent, mock_config):
    """A violating case must trigger one extra generator call and be replaced by the clean result."""
    mock_config.TestCaseGenerationAgentConfig.TC_VALIDATION_MAX_RETRIES = 1
    ac_list = AcceptanceCriteriaList(
        items=[AcceptanceCriteriaItem(id="AC-1", text="criterion", attachment_info="")]
    )
    violating_tc = _make_tc("返回 USD/CNY 汇率")
    clean_tc = _make_tc("返回指定货币对的汇率")

    r_batch = MagicMock()
    r_batch.output = GeneratedTestCases(test_cases=[violating_tc])
    r_regen = MagicMock()
    r_regen.output = GeneratedTestCases(test_cases=[clean_tc])
    agent.test_case_generator_agent.run = AsyncMock(side_effect=[r_batch, r_regen])

    result = await agent.generate_test_cases_from_acs(ac_list, "doc")

    assert agent.test_case_generator_agent.run.call_count == 2
    assert result.test_cases[0] is clean_tc


@pytest.mark.asyncio
async def test_regeneration_max_retries_adds_comment(agent, mock_config):
    """When all retries are exhausted the original case is kept with a human-review marker."""
    mock_config.TestCaseGenerationAgentConfig.TC_VALIDATION_MAX_RETRIES = 1
    ac_list = AcceptanceCriteriaList(
        items=[AcceptanceCriteriaItem(id="AC-1", text="criterion", attachment_info="")]
    )
    violating_tc = _make_tc("返回 USD/CNY 汇率")

    mock_r = MagicMock()
    mock_r.output = GeneratedTestCases(test_cases=[violating_tc])
    agent.test_case_generator_agent.run = AsyncMock(return_value=mock_r)

    result = await agent.generate_test_cases_from_acs(ac_list, "doc")

    # 1 batch call + 1 regen attempt = 2 total
    assert agent.test_case_generator_agent.run.call_count == 2
    final_tc = result.test_cases[0]
    assert "需人工复核" in final_tc.comment


@pytest.mark.asyncio
async def test_repair_local_scope_isolated_during_output_retries(agent):
    import asyncio
    import json

    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
    from pydantic_ai.models.function import FunctionModel

    from agents.test_case_generation.main import _validate_repair
    from common.repair_context import RepairContext

    calls = {"R1": 0, "R2": 0}

    async def respond(messages, info):
        prompt = next(p.content for m in messages for p in m.parts if isinstance(p, UserPromptPart))
        required = json.loads(prompt)["required"]
        calls[required] += 1
        await asyncio.sleep(0)
        refs = ["R2" if required == "R1" else "R1"] if calls[required] == 1 else [required]
        case = _make_tc("返回指定汇率").model_copy(update={"covered_requirement_ids": refs})
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                                                GeneratedTestCases(test_cases=[case]).model_dump_json())])

    repair = Agent(FunctionModel(respond), output_type=GeneratedTestCases, deps_type=RepairContext, output_retries=1)
    repair.output_validator(_validate_repair)
    contexts = [RepairContext(json.dumps({"required": ref}), AcceptanceCriteriaList(items=[
        AcceptanceCriteriaItem(id=ref, text=ref + "完整需求", attachment_info="")]), frozenset([ref]))
        for ref in calls]
    results = await asyncio.gather(*(repair.run(ctx.prompt, deps=ctx) for ctx in contexts))
    assert calls == {"R1": 2, "R2": 2}
    assert [r.output.test_cases[0].covered_requirement_ids for r in results] == [["R1"], ["R2"]]


@pytest.mark.asyncio
async def test_design_fix_semantic_retry_and_gap_use_repair_agent_only(agent, mock_config):
    from common.design_models import StageRequest

    mock_config.TestCaseGenerationAgentConfig.TC_VALIDATION_MAX_RETRIES = 1
    mock_config.DesignConfig.MAX_NEW_CASES_PER_ROUND = 10
    original = _make_tc("返回指定货币对汇率").model_copy(update={"key": "TC-1", "version": 1,
                                                                       "covered_requirement_ids": ["R1"]})
    unrelated = original.model_copy(update={"key": "TC-2", "covered_requirement_ids": ["R2"],
                                            "steps": [TestStep(action="unrelated-body", expected_results="结果", test_data=[])]})
    finding = {"severity": "high", "action": "modify", "owner_case_key": "TC-1",
               "related_case_keys": [], "requirement_ids": ["R1"], "category": "coverage",
               "reason": "需要修复", "expected_change": "增加边界"}
    payload = {"document": "完整PRD", "knowledge": {"status": "ok", "content": "完整KB"},
               "requirements": AcceptanceCriteriaList(items=[AcceptanceCriteriaItem(id=r, text=r+"完整条目", attachment_info="")
                                                              for r in ["R1", "R2"]]).model_dump(),
               "test_cases": [original.model_dump(), unrelated.model_dump()],
               "findings": [finding, {**finding, "action": "add_test_case", "owner_case_key": None}]}
    outputs = [original.model_copy(update={"steps": [TestStep(action="查询", expected_results="返回 USD/CNY 汇率", test_data=[])]}),
               original, original]
    agent.repair_agent.run = AsyncMock(side_effect=[MagicMock(output=GeneratedTestCases(test_cases=[c])) for c in outputs])
    request = StageRequest(run_id="fix", stage_id="fix-0", operation="fix", required_skill_id="fix-test-cases",
                           token_limit=100000, timeout_seconds=30, payload=payload)
    result = await agent.execute_stage(request)
    assert len(result.replacements) == len(result.additions) == 1
    assert result.replacements[0].version == 2
    assert agent.repair_agent.run.await_count == 3
    agent.test_case_generator_agent.run.assert_not_called()
    for call in agent.repair_agent.run.await_args_list:
        prompt = call.args[0]
        assert "完整PRD" in prompt and "完整KB" in prompt
        assert "R1完整条目" in prompt and "R2完整条目" not in prompt
        assert "unrelated-body" not in prompt
        assert {r.id for r in call.kwargs["deps"].requirements.items} == {"R1"}


def test_repair_agent_has_independent_output_ceiling():
    import config
    with patch("agents.test_case_generation.main.get_model", return_value="test"), \
         patch("common.agent_base.get_model", return_value="test"), \
         patch("common.custom_llm_wrapper.Agent") as factory:
        TestCaseGenerationAgent()
    calls = {call.kwargs.get("name"): call.kwargs for call in factory.call_args_list}
    assert calls["test_case_generator"]["model"].max_tokens == config.TestCaseGenerationAgentConfig.TC_GENERATOR_MAX_TOKENS
    assert calls["repair_test_case"]["model"].max_tokens == 32768
    assert calls["repair_test_case_pair"]["model"].max_tokens == 65536
    assert calls["repair_test_case"]["model"].thinking_level == "high"
    assert calls["repair_test_case_pair"]["model"].thinking_level == "high"


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["missing", "duplicate", "extra", "version", "sibling_id", "lost_coverage"])
async def test_pair_repair_retries_identity_and_per_case_scope(corruption):
    import json

    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from agents.test_case_generation.main import _validate_repair_pair
    from common.repair_context import RepairContext

    originals = [_make_tc("查询").model_copy(update={"key": f"TC-{i}", "version": 1,
                                                        "covered_requirement_ids": [f"R{i}"]}) for i in [1, 2]]
    scopes = {c.key: RepairContext("prompt", AcceptanceCriteriaList(items=[
        AcceptanceCriteriaItem(id=c.covered_requirement_ids[0], text="完整需求", attachment_info="")]),
        frozenset(c.covered_requirement_ids), c) for c in originals}
    calls = 0

    def respond(messages, info):
        nonlocal calls
        calls += 1
        candidates = [c.model_copy(update={"version": 2}) for c in originals]
        if calls == 1:
            if corruption == "missing":
                candidates.pop()
            elif corruption == "duplicate":
                candidates[-1] = candidates[0]
            elif corruption == "extra":
                candidates[-1] = candidates[-1].model_copy(update={"key": "TC-99"})
            elif corruption == "version":
                candidates[0] = candidates[0].model_copy(update={"version": 1})
            else:
                ids = ["R1", "R2"] if corruption == "sibling_id" else []
                candidates[0] = candidates[0].model_copy(update={"covered_requirement_ids": ids})
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
            GeneratedTestCases(test_cases=list(reversed(candidates))).model_dump_json())])

    repair = Agent(FunctionModel(respond), output_type=GeneratedTestCases,
                   deps_type=dict[str, RepairContext], output_retries=1)
    repair.output_validator(_validate_repair_pair)
    result = await repair.run("修复两例", deps=scopes)
    assert calls == 2
    assert [(c.key, c.version, c.covered_requirement_ids) for c in result.output.test_cases] == [
        ("TC-2", 2, ["R2"]), ("TC-1", 2, ["R1"])]


@pytest.mark.asyncio
async def test_pair_shares_sources_but_only_bad_case_uses_single_semantic_retry(agent, mock_config):
    from common.design_models import StageRequest

    mock_config.TestCaseGenerationAgentConfig.TC_VALIDATION_MAX_RETRIES = 1
    mock_config.DesignConfig.MAX_NEW_CASES_PER_ROUND = 10
    originals = [_make_tc("查询").model_copy(update={"key": f"TC-{i}", "version": 1,
                                                        "covered_requirement_ids": [f"R{i}"]}) for i in [1, 2, 3]]
    originals[2].steps[0].action = "unrelated-body"
    finding = {"severity": "high", "action": "modify", "related_case_keys": [], "category": "coverage",
               "reason": "需要修复", "expected_change": "补充边界"}
    payload = {"document": "完整原始PRD", "knowledge": {"status": "ok", "content": "完整原始KB"},
               "requirements": AcceptanceCriteriaList(items=[
                   AcceptanceCriteriaItem(id=f"R{i}", text=f"需求{i}全文", attachment_info="") for i in [1, 2, 3]]).model_dump(),
               "test_cases": [c.model_dump() for c in originals],
               "findings": [{**finding, "owner_case_key": f"TC-{i}", "requirement_ids": [f"R{i}"]} for i in [1, 2]]}
    bad = originals[0].model_copy(update={"version": 2, "steps": [
        TestStep(action="查询", expected_results="返回 USD/CNY 汇率", test_data=[])]})
    good = originals[1].model_copy(update={"version": 2})
    agent.pair_repair_agent.run = AsyncMock(return_value=MagicMock(output=GeneratedTestCases(test_cases=[good, bad])))
    agent.repair_agent.run = AsyncMock(return_value=MagicMock(output=GeneratedTestCases(test_cases=[originals[0]])))
    request = StageRequest(run_id="pair", stage_id="fix", operation="fix", required_skill_id="fix-test-cases",
                           token_limit=200000, timeout_seconds=30, payload=payload)
    result = await agent.execute_stage(request)
    assert [(c.key, c.version, c.covered_requirement_ids) for c in result.replacements] == [
        ("TC-1", 2, ["R1"]), ("TC-2", 2, ["R2"])]
    assert result.replacements[1] == good
    assert agent.repair_agent.run.await_count == 1
    assert agent.repair_agent.run.await_args.kwargs["deps"].owner.key == "TC-1"
    prompt = agent.pair_repair_agent.run.await_args.args[0]
    assert prompt.count("完整原始PRD") == prompt.count("完整原始KB") == 1
    assert "unrelated-body" not in prompt and "需求3全文" not in prompt
    scopes = agent.pair_repair_agent.run.await_args.kwargs["deps"]
    assert [{r.id for r in scope.requirements.items} for scope in scopes.values()] == [{"R1"}, {"R2"}]
    agent.test_case_generator_agent.run.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("max_retries", [0, 2])
async def test_pair_semantic_failure_respects_remaining_single_retry_bound(agent, mock_config, max_retries):
    from common.repair_context import RepairContext

    mock_config.TestCaseGenerationAgentConfig.TC_VALIDATION_MAX_RETRIES = max_retries
    original = _make_tc("查询").model_copy(update={"key": "TC-1", "covered_requirement_ids": ["R1"]})
    bad = original.model_copy(update={"steps": [TestStep(action="查询", expected_results="返回 USD/CNY", test_data=[])]})
    context = RepairContext("原修复依据", AcceptanceCriteriaList(items=[
        AcceptanceCriteriaItem(id="R1", text="完整需求", attachment_info="")]), frozenset(["R1"]), original)
    agent.repair_agent.run = AsyncMock(return_value=MagicMock(output=GeneratedTestCases(test_cases=[bad])))
    with pytest.raises(ValueError, match="still violates"):
        await agent._repair_case(context, bad)
    assert agent.repair_agent.run.await_count == max_retries
    agent.pair_repair_agent.run.assert_not_called()


def test_generation_and_all_repair_calls_use_separate_models():
    from pydantic_ai.models.test import TestModel

    import config

    generation_model, repair_model = TestModel(), TestModel()
    with patch("agents.test_case_generation.main.get_model", side_effect=[generation_model, repair_model]) as models, \
         patch("common.agent_base.get_model", return_value="test"), \
         patch("common.custom_llm_wrapper.Agent") as factory:
        TestCaseGenerationAgent()
    assert models.call_args_list[0].args == ("gpt-5.6-terra", "deepseek-v4.1-flash")
    assert models.call_args_list[1].args == ("deepseek-v4-flash",)
    calls = {call.kwargs.get("name"): call.kwargs for call in factory.call_args_list}
    for name in ["ac_extractor", "test_case_generator"]:
        assert calls[name]["model"].wrapped is generation_model
    for name in ["repair_test_case", "repair_test_case_pair"]:
        assert calls[name]["model"].wrapped is repair_model
    assert config.TestCaseGenerationAgentConfig.REPAIR_MODEL_NAME == "deepseek-v4-flash"
