# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

import asyncio
import json
from dataclasses import dataclass, field
from itertools import batched, zip_longest
from typing import TYPE_CHECKING

from a2a.helpers import get_message_text, new_text_message
from a2a.types import AgentSkill, Message
from pydantic import BaseModel
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.settings import ThinkingLevel

import config
from agents.test_case_review.prompt import TestCaseReviewSystemPrompt, TestCaseReviewWithAttachmentsPrompt
from common import observability, utils
from common.agent_base import MCP_SERVER_ATTACHMENTS_FOLDER_PATH, AgentBase
from common.custom_llm_wrapper import CustomLlmWrapper
from common.design_models import CaseAssessment, DesignReview, StageRequest, SuiteAssessment, is_stage_message
from common.llm_provider import get_model
from common.models import TestCaseReviewFeedbacks, TestCaseReviewRequest, TestStep
from common.untrusted_content import UNTRUSTED_CONTENT_POLICY, untrusted_input

if TYPE_CHECKING:
    from pydantic_ai.messages import BinaryContent

logger = utils.get_logger("test_case_review_agent")
_DESIGN_REVIEW_BATCH_SIZE = 4
# Bounds one suite-review call's output; a single unbatched call over 40+ cases hit the token cap.
_SUITE_REVIEW_BATCH_SIZE = 10


class _CaseReviewBatch(BaseModel):
    """Independent version receipts returned by one bounded review call."""

    assessments: list[CaseAssessment]


@dataclass(slots=True)
class _ReviewRunState:
    """Per-request state container for TestCaseReviewAgent, threaded via pydantic-ai deps."""

    test_cases: list = field(default_factory=list)
    doc_content: str = ""
    result: TestCaseReviewFeedbacks | None = None


def _extract_doc_content_from_text(text: str) -> str:
    """Extracts the requirement document content from the orchestrator message."""
    marker = "Requirement document content:\n"
    idx = text.find(marker)
    if idx == -1:
        return text
    content = text[idx + len(marker):]
    # Strip the test cases block that may follow
    tc_marker = "\n\nTest cases:\n"
    tc_idx = content.find(tc_marker)
    return content[:tc_idx] if tc_idx != -1 else content


def _parse_test_cases_from_text(text: str) -> list:
    """Extracts test cases from the first JSON object that contains a 'test_cases' key."""
    pos = 0
    while pos < len(text):
        start = text.find("{", pos)
        if start == -1:
            break
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[start : i + 1])
                        if "test_cases" in data:
                            return TestCaseReviewRequest.model_validate(data).test_cases
                    except (json.JSONDecodeError, ValueError):
                        pass
                    pos = i + 1
                    break
        else:
            break
    return []


class TestCaseReviewAgent(AgentBase):
    __test__ = False

    def __init__(self):
        # Create a sub-agent for reviewing with attachments
        self.review_agent = CustomLlmWrapper.create_agent(
            model_name=get_model(
                config.TestCaseReviewAgentConfig.MODEL_NAME, config.TestCaseReviewAgentConfig.FALLBACK_MODEL_NAME,
                bounded_requests=True,
            ),
            output_type=TestCaseReviewFeedbacks,
            system_prompt=TestCaseReviewWithAttachmentsPrompt().get_prompt() + "\n\n" + UNTRUSTED_CONTENT_POLICY,
            name="review_test_cases_with_attachments",
            thinking_level=config.TestCaseReviewAgentConfig.THINKING_LEVEL,
            max_tokens=config.TestCaseReviewAgentConfig.MAX_TOKENS,
        )

        instruction_prompt = TestCaseReviewSystemPrompt(
            attachments_remote_folder_path=MCP_SERVER_ATTACHMENTS_FOLDER_PATH,
            batch_size=config.TestCaseReviewAgentConfig.TEST_CASE_REVIEW_BATCH_SIZE,
        )
        super().__init__(
            agent_name=config.TestCaseReviewAgentConfig.OWN_NAME,
            base_url=config.AGENT_BASE_URL,
            port=config.TestCaseReviewAgentConfig.PORT,
            external_port=config.TestCaseReviewAgentConfig.EXTERNAL_PORT,
            protocol=config.TestCaseReviewAgentConfig.PROTOCOL,
            model_name=config.TestCaseReviewAgentConfig.MODEL_NAME,
            fallback_model_name=config.TestCaseReviewAgentConfig.FALLBACK_MODEL_NAME,
            deps_type=_ReviewRunState,
            output_type=TestCaseReviewFeedbacks,
            instructions=instruction_prompt.get_prompt(),
            mcp_servers=[],
            description="Agent which reviews generated test cases for coherence, redundancy, and effectiveness.",
            tools=[self._review_test_cases_with_attachments],
        )

    def get_thinking_level(self) -> ThinkingLevel:
        return config.TestCaseReviewAgentConfig.THINKING_LEVEL

    def get_max_requests_per_task(self) -> int:
        return config.TestCaseReviewAgentConfig.MAX_REQUESTS_PER_TASK

    def get_total_tokens_limit(self) -> int:
        return config.TestCaseReviewAgentConfig.TOTAL_TOKENS_LIMIT_PER_TASK

    def get_max_tokens(self) -> int | None:
        # The main orchestrating LLM only emits a short summary; the full result travels
        # via deps, so a small limit is sufficient and leaves more room for input.
        return 4096

    def get_skills(self) -> list[AgentSkill]:
        return [
            AgentSkill(
                id="review-test-cases",
                name="Review test cases",
                description="Reviews generated test cases for coherence, redundancy, and effectiveness.",
                tags=["qa", "test-case-review"],
            )
        ]

    async def run(self, received_message: Message) -> Message:
        text = get_message_text(received_message)
        if is_stage_message(text):
            return await super().run(received_message)
        test_cases = _parse_test_cases_from_text(text)
        context_id = getattr(received_message, "context_id", None)
        task_id = getattr(received_message, "task_id", None)
        state = _ReviewRunState()
        if test_cases:
            state.test_cases = test_cases
            state.doc_content = _extract_doc_content_from_text(text)
            # Pass only a minimal instruction to the main LLM — both test cases and
            # doc content travel via deps, not through the LLM context window.
            received_message = new_text_message(
                text="Review the test cases using the requirement document.",
                context_id=context_id,
                task_id=task_id,
            )
        else:
            logger.warning("Could not extract test cases from review request message.")
        msg = await super().run(received_message, deps=state)
        if state.result is None:
            return msg
        return new_text_message(text=state.result.model_dump_json(), context_id=context_id, task_id=task_id)

    async def execute_stage(self, request: StageRequest) -> DesignReview:
        """Review each case version and then the complete suite for missing coverage."""
        if request.operation != "review":
            raise ValueError("Review agent only supports review")
        payload = request.payload
        from common.design_runtime import checkpoint_agent
        requirement_ids = {item["id"] for item in payload["requirements"]["items"]}
        model = get_model(config.TestCaseReviewAgentConfig.MODEL_NAME,
                          config.TestCaseReviewAgentConfig.FALLBACK_MODEL_NAME, bounded_requests=True)
        rubric = (
            "你是测试质量评审专家，输出简体中文。只指出有需求依据的问题。"
            "severity: critical=核心业务结果错误或重大资金/安全风险；high=关键流程不可执行或遗漏；"
            "medium=场景/边界覆盖缺失或重复覆盖；low=不影响正确性的表达改进。"
            "每项问题使用合法需求ID。修改需明确owner。只在其他存活用例完整承接所有独有场景时建议删除，"
            "related_case_keys列出承接者，reason说明承接证据；部分重叠应合并修改。"
            "自动delete仅在前置条件完全相同且源用例每个原始步骤/数据都在承接者中原样保留时允许，"
            "语义相似但文字不同应提出modify合并，不能直接delete。"
            "遗漏是新流程才add_test_case；可在现有流程补充的用modify。"
            "若缺口由已删除用例覆盖，add_test_case中填写restore_case_key恢复原用例。"
            "untestable只用于没有可测试行为的说明性需求项（如概述、目标、修订记录），并写明理由；"
            "含规则、数值、状态、流程或界面行为的需求项必须测试，不得列入untestable，也不能同时引用于finding。"
            f"合法需求ID仅为：{sorted(requirement_ids)}。引用需求项ID，不得自行编造章节编号。"
            "\n\n" + UNTRUSTED_CONTENT_POLICY
        )
        covered = {ref for case in payload["test_cases"] for ref in case.get("covered_requirement_ids", [])}
        uncovered = sorted(requirement_ids - covered)
        cases_by_key = {case["key"]: case for case in payload["test_cases"]}

        def retained_steps(case: dict) -> set[str]:
            return {TestStep.model_validate(step).model_dump_json() for step in case["steps"]}

        def validate_deletes(findings: list) -> None:
            for finding in findings:
                if finding.action != "delete_test_case":
                    continue
                owner = cases_by_key[finding.owner_case_key]
                survivors = {step for key in finding.related_case_keys
                            if cases_by_key[key]["preconditions"] == owner["preconditions"]
                            for step in retained_steps(cases_by_key[key])}
                if not retained_steps(owner) <= survivors:
                    raise ModelRetry(
                        f"{finding.owner_case_key}的删除未证明承接者逐字保留其每一步骤/数据；"
                        "语义相似但文字不同应提出modify合并，不能直接delete。"
                    )

        def validate_requirements(output: CaseAssessment | SuiteAssessment) -> CaseAssessment | SuiteAssessment:
            unknown = {ref for finding in output.findings for ref in finding.requirement_ids} - requirement_ids
            if unknown:
                raise ModelRetry(f"未知需求ID：{sorted(unknown)}；合法需求ID仅为：{sorted(requirement_ids)}")
            return output

        def validate_batch(ctx: RunContext[dict[str, int]], output: _CaseReviewBatch) -> _CaseReviewBatch:
            actual = [(receipt.case_key, receipt.case_version) for receipt in output.assessments]
            if len(actual) != len(ctx.deps) or set(actual) != set(ctx.deps.items()):
                raise ModelRetry("必须为本批每个key/version恰好返回一份回执；不能遗漏、重复或新增。")
            for receipt in output.assessments:
                validate_requirements(receipt)
                if any(f.owner_case_key != receipt.case_key or f.action != "modify" for f in receipt.findings):
                    raise ModelRetry("每份回执只能modify所属用例；跨例、新增、删除交给整套评审。")
            return output

        reviewer = CustomLlmWrapper.create_agent(
            model_name=model, output_type=_CaseReviewBatch, system_prompt=rubric,
            name="design_case_review", thinking_level=self.get_thinking_level(),
            max_tokens=config.TestCaseReviewAgentConfig.MAX_TOKENS, deps_type=dict[str, int],
        )
        # The suite review's own output (not its input) is what overflows at scale: it must list every
        # gap/duplicate it finds in one response. Split responsibility for cases and uncovered requirements
        # into disjoint batches so no two calls can emit conflicting findings for the same case or gap,
        # while every call still sees every case — duplicate/survivor evidence can be anywhere.
        ownership = {"cases": frozenset(), "requirements": frozenset()}

        def validate_suite(output: SuiteAssessment) -> SuiteAssessment:
            validate_requirements(output)
            validate_deletes(output.findings)
            owned_cases, owned_requirements = ownership["cases"], ownership["requirements"]
            misowned = [f for f in output.findings if f.action != "add_test_case" and f.owner_case_key not in owned_cases]
            if misowned:
                raise ModelRetry(f"本次仅负责用例{sorted(owned_cases)}；以下发现的所属用例不在其中："
                                 f"{sorted(f.owner_case_key for f in misowned)}。")
            unowned_gaps = [f for f in output.findings
                           if f.action == "add_test_case" and not set(f.requirement_ids) & owned_requirements]
            if unowned_gaps:
                raise ModelRetry(f"本次仅负责需求缺口{sorted(owned_requirements)}；新增用例须至少引用其中一个需求ID。")
            waived = {w.requirement_id for w in output.untestable}
            if waived - owned_requirements:
                raise ModelRetry(f"本次仅能豁免{sorted(owned_requirements)}范围内的需求：越权豁免{sorted(waived - owned_requirements)}。")
            gaps = {ref for f in output.findings if f.action == "add_test_case" for ref in f.requirement_ids}
            if unaddressed := sorted(owned_requirements - gaps - waived):
                raise ModelRetry(f"未覆盖需求项{unaddressed}必须提出add_test_case，或列入untestable并写明理由。")
            return output

        suite_reviewer = CustomLlmWrapper.create_agent(
            model_name=model, output_type=SuiteAssessment, system_prompt=rubric,
            name="design_suite_review", thinking_level=self.get_thinking_level(),
            max_tokens=config.TestCaseReviewAgentConfig.MAX_TOKENS,
        )

        reviewer.output_validator(validate_batch)
        suite_reviewer.output_validator(validate_suite)
        suite_case_batches = list(batched(sorted(cases_by_key), _SUITE_REVIEW_BATCH_SIZE, strict=False))
        suite_requirement_batches = list(batched(uncovered, _SUITE_REVIEW_BATCH_SIZE, strict=False))
        receipts = []
        for batch in batched(payload["test_cases"], _DESIGN_REVIEW_BATCH_SIZE, strict=False):
            expected = {case["key"]: case["version"] for case in batch}
            previous = payload.get("previous_review", {})
            previous_findings = [
                finding
                for receipt in [*previous.get("assessments", []), previous.get("suite", {})]
                for finding in receipt.get("findings", [])
                if finding.get("owner_case_key") in expected or set(expected) & set(finding.get("related_case_keys", []))
            ]
            result = await checkpoint_agent(
                "case-review:" + ",".join(expected), reviewer,
                "逐例独立评估本批每个用例声明场景的质量，每例恰好返回一份case_key/case_version回执；"
                "通过时findings为空。本次finding只能modify其所属case；不要要求每个用例覆盖整份需求。"
                "新增、删除、跨用例重复及整套遗漏由后续整套评审处理。\n"
                + json.dumps({
                    "test_cases": list(batch), "document": payload["document"],
                    "requirements": payload["requirements"], "knowledge": payload["knowledge"],
                    "previous_findings": previous_findings,
                }, ensure_ascii=False, separators=(",", ":")), _CaseReviewBatch, deps=expected,
            )
            receipts.extend(result.assessments)
        suite_findings, suite_untestable, suite_completed = [], [], True
        for owned_cases, owned_requirements in zip_longest(suite_case_batches, suite_requirement_batches, fillvalue=()):
            ownership["cases"] = frozenset(owned_cases)
            ownership["requirements"] = frozenset(owned_requirements)
            result = await checkpoint_agent(
                "suite-review:" + ",".join(owned_cases) + "|" + ",".join(owned_requirements), suite_reviewer,
                "审查整个用例集合的覆盖缺口和重复覆盖；完成后completed必须为true，不能省略评审。"
                f"本次仅负责判断用例{list(owned_cases)}是否存在重复覆盖（delete_test_case/modify/"
                f"remove_duplicate_steps，owner须在其中）；当前未覆盖需求项（本次负责部分）：{list(owned_requirements)}，"
                "每项须提出add_test_case或列入untestable并写明理由（可为空表示本次无缺口待填）。"
                "其余用例/需求项由其他批次负责，但可将任意存活用例作为重复覆盖的承接者(related_case_keys)。\n"
                + json.dumps({**{key: payload.get(key, {}) for key in
                                ("document", "requirements", "knowledge", "test_cases", "deleted")},
                             "owned_case_keys": list(owned_cases), "owned_requirement_ids": list(owned_requirements)},
                             ensure_ascii=False, separators=(",", ":")), SuiteAssessment,
            )
            suite_findings.extend(result.findings)
            suite_untestable.extend(result.untestable)
            suite_completed = suite_completed and result.completed
        return DesignReview(assessments=receipts,
                            suite=SuiteAssessment(completed=suite_completed, findings=suite_findings, untestable=suite_untestable))

    async def _review_test_cases_with_attachments(self, ctx: RunContext[_ReviewRunState]) -> str:
        """
        Reviews all test cases from the request in batches, using the requirement document.

        Reads test cases and doc content from ctx.deps; stores the result back in ctx.deps.result.

        Returns:
            A compact completion summary.
        """
        test_cases = ctx.deps.test_cases
        if not test_cases:
            raise RuntimeError("No test cases available for review.")
        with observability.tracer.start_as_current_span("test_case_review.quality_gate") as span:
            requirement_doc_content = ctx.deps.doc_content
            attachments_content = {}
            batch_size = config.TestCaseReviewAgentConfig.TEST_CASE_REVIEW_BATCH_SIZE
            batches = list(batched(test_cases, batch_size, strict=False))
            logger.info(f"Reviewing {len(test_cases)} test cases in {len(batches)} batch(es) of up to {batch_size}.")
            tasks = [
                self._review_batch(batch, i + 1, len(batches), requirement_doc_content, attachments_content)
                for i, batch in enumerate(batches)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            all_feedbacks = []
            llm_comments = []
            for i, result in enumerate(results):
                if isinstance(result, BaseException):
                    raise RuntimeError(f"Review failed for batch {i + 1}: {result}") from result
                all_feedbacks.extend(result.review_feedbacks)
                if result.llm_comments:
                    llm_comments.append(result.llm_comments)
            ctx.deps.result = TestCaseReviewFeedbacks(
                review_feedbacks=all_feedbacks,
                llm_comments="\n".join(llm_comments) or None,
            )
            issues_count = sum(1 for f in all_feedbacks if f.review_feedback)
            span.set_attribute("review.total_cases", len(all_feedbacks))
            span.set_attribute("review.issues_count", issues_count)
            span.set_attribute("review.issues_ratio", issues_count / len(all_feedbacks) if all_feedbacks else 0.0)
        return f"Review completed. Reviewed {len(all_feedbacks)} test cases."

    async def _review_batch(
        self,
        batch,
        batch_number: int,
        total_batches: int,
        requirement_doc_content: str,
        attachments_content: dict,
    ) -> TestCaseReviewFeedbacks:
        user_message_parts: list[str | BinaryContent] = [
            untrusted_input(document=requirement_doc_content),
            f"Test Cases to Review:\n{json.dumps([tc.model_dump() for tc in batch], ensure_ascii=False)}",
        ]
        if attachments_content:
            for filename, binary_content in attachments_content.items():
                user_message_parts.append(f"Attachment: {filename}")
                user_message_parts.append(binary_content)
        logger.info(f"Starting review of batch {batch_number}/{total_batches}.")
        result = await self.review_agent.run(user_message_parts)
        feedbacks: TestCaseReviewFeedbacks = result.output
        logger.info(
            f"Reviewed batch {batch_number}/{total_batches}: {len(feedbacks.review_feedbacks)} feedbacks."
        )
        return feedbacks

agent = TestCaseReviewAgent()
app = agent.a2a_server

if __name__ == "__main__":
    agent.start_as_server()
