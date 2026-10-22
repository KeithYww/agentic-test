# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

import asyncio
import html
import json
import re
from dataclasses import dataclass
from itertools import batched
from typing import Literal

from a2a.helpers import get_message_text, new_text_message
from a2a.types import AgentSkill, Message
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.mcp import MCPServerSSE
from pydantic_ai.messages import BinaryContent
from pydantic_ai.settings import ThinkingLevel

import config
from agents.test_case_generation.prompt import (
    AcExtractionPrompt,
    TcGenerationPrompt,
    TestCaseGenerationSystemPrompt,
)
from common import observability, utils
from common.agent_base import AgentBase
from common.custom_llm_wrapper import CustomLlmWrapper
from common.design_models import (
    DesignDraft,
    FixResult,
    KnowledgeSnapshot,
    ReviewFinding,
    StageRequest,
    is_stage_message,
)
from common.llm_provider import get_model
from common.models import (
    AcceptanceCriteriaItem,
    AcceptanceCriteriaList,
    GeneratedTestCases,
    TestCase,
)
from common.repair_context import RepairContext, build_repair_context
from common.services.kb_client import query_knowledge_snapshot, query_photon_kb
from common.services.test_management_system_client_provider import get_test_management_client
from common.untrusted_content import UNTRUSTED_CONTENT_POLICY, untrusted_input

logger = utils.get_logger("test_case_generation_agent")
feishu_mcp_server = MCPServerSSE(url=config.FEISHU_MCP_SERVER_URL, timeout=config.MCP_SERVER_TIMEOUT_SECONDS)

_SECTION_NUMBER_RE = re.compile(r"(?<!\d)\d+\.\d+(?:\.\d+)?(?!\d)")
_HTML_HEADING_RE = re.compile(r"<h([23])[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)
_NUMBERED_TITLE_RE = re.compile(r"^[ \t]*(\d+(?:\.\d+)+)[ \t]+(.+)$", re.MULTILINE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Semantic violation detectors — high-precision only, prefer false-negatives over false-positives
# NOTE: \b is unreliable here because Python's re module treats CJK characters as \w under
# Unicode mode, so a word boundary never fires between a Latin letter and an adjacent Chinese
# character (e.g. "USD/CNY与USD/EUR" has no \b before "与"). Use explicit ASCII-letter lookaround
# instead so these patterns still match when directly adjacent to Chinese text.
_CURRENCY_PAIR_RE = re.compile(r"(?<![A-Za-z])[A-Z]{3}/[A-Z]{3}(?![A-Za-z])")  # e.g. USD/CNY, BRL/MXN
_KV_TEST_DATA_RE = re.compile(r"(?<![A-Za-z])(?:Sell|Buy)=\S+")  # e.g. Sell=BRL, Buy=MXN
_PENDING_FIELD_RE = re.compile(r"\[字段名待确认\]")


@dataclass(slots=True)
class _Section:
    """Internal representation of a PRD section before it becomes a chunk."""

    section_id: str
    title: str
    content: str
    level: int = 1

    @property
    def char_count(self) -> int:
        return len(self.content)


def _classify_prd(content: str) -> Literal["functional", "narrative"]:
    """Select the chunking strategy from the document's structure only.

    The PRD is untrusted test basis: its wording (e.g. an acceptance-criteria list) must not
    choose the strategy or narrow the scope.

    Returns:
        "functional" - PRD is structured with numbered sections.
        "narrative"  - No detectable structure; fall back to LLM-based requirement derivation.
    """
    if len(_SECTION_NUMBER_RE.findall(content)) >= config.PrdClassifierConfig.FUNCTIONAL_SECTION_THRESHOLD:
        return "functional"
    return "narrative"


def _strip_html(text: str) -> str:
    text = re.sub(r"</(?:h[1-6]|p|tr|table|div|li)>|<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    return html.unescape(_HTML_TAG_RE.sub(" ", text)).strip()


def _parse_sections_from_html(html: str) -> list[_Section]:
    """Extract sections from lark-cli HTML output using heading tags."""
    sections: list[_Section] = []
    parts = _HTML_HEADING_RE.split(html)
    # parts layout: [text_before_h1, heading_level, heading_text, body_after_heading, ...]
    for i in range(1, len(parts) - 2, 3):
        raw_title = _strip_html(parts[i + 1])
        # Derive section id from leading number pattern in the title
        m = re.match(r"(\d+(?:\.\d+)*)", raw_title)
        section_id = m.group(1) if m else f"sec-{len(sections) + 1}"
        sections.append(_Section(section_id, raw_title, _strip_html(parts[i + 2]), int(parts[i])))
    return sections


def _parse_sections_from_text(text: str) -> list[_Section]:
    """Fallback: extract numbered sections from plain text."""
    matches = list(_NUMBERED_TITLE_RE.finditer(text))
    sections: list[_Section] = []
    for idx, match in enumerate(matches):
        section_id = match.group(1)
        title = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections.append(_Section(section_id, title, text[start:end].strip(), section_id.count(".")))
    return sections


def _fold_heading_only_parents(sections: list[_Section]) -> list[_Section]:
    """Keep every section as its own requirement; a body-less heading followed by deeper
    headings only contributes its title as a path prefix of those descendants."""
    result: list[_Section] = []
    parents: list[_Section] = []
    for index, sec in enumerate(sections):
        while parents and parents[-1].level >= sec.level:
            parents.pop()
        following = sections[index + 1] if index + 1 < len(sections) else None
        if not sec.content.strip() and following and following.level > sec.level:
            parents.append(sec)
            continue
        title = " / ".join([*(parent.title for parent in parents), sec.title])
        result.append(_Section(sec.section_id, title, sec.content, sec.level))
    return result


def _unique_ids(sections: list[_Section]) -> list[_Section]:
    seen: dict[str, int] = {}
    for sec in sections:
        seen[sec.section_id] = seen.get(sec.section_id, 0) + 1
        if seen[sec.section_id] > 1:
            sec.section_id = f"{sec.section_id}-{seen[sec.section_id]}"
    return sections


def _split_long_sections(sections: list[_Section], max_chars: int) -> list[_Section]:
    """Split sections above max_chars on sub-heading boundaries."""
    result: list[_Section] = []
    for sec in sections:
        if sec.char_count <= max_chars:
            result.append(sec)
            continue
        # Split on sub-numbered lines within the body
        sub_matches = list(_NUMBERED_TITLE_RE.finditer(sec.content))
        if not sub_matches:
            result.append(sec)
            continue
        # First chunk: content before first sub-heading
        preamble = sec.content[: sub_matches[0].start()].strip()
        if preamble:
            result.append(_Section(section_id=sec.section_id, title=sec.title, content=preamble))
        for i, m in enumerate(sub_matches):
            sub_id = m.group(1)
            sub_title = m.group(2).strip()
            sub_start = m.end()
            sub_end = sub_matches[i + 1].start() if i + 1 < len(sub_matches) else len(sec.content)
            result.append(
                _Section(
                    section_id=sub_id,
                    title=sub_title,
                    content=sec.content[sub_start:sub_end].strip() or sub_title,
                )
            )
    return result


def _chunk_by_sections(content: str) -> AcceptanceCriteriaList:
    """Convert a functional PRD into requirement chunks keyed by section number."""
    sections = _parse_sections_from_html(content)
    if not sections:
        logger.info("HTML section parsing yielded no sections; falling back to text parsing.")
        sections = _parse_sections_from_text(content)

    max_chars = config.PrdClassifierConfig.SECTION_MAX_CHARS
    sections = _unique_ids(_split_long_sections(_fold_heading_only_parents(sections), max_chars))

    logger.info(f"Section chunker produced {len(sections)} chunks (max_chars={max_chars}).")
    return AcceptanceCriteriaList(
        items=[
            AcceptanceCriteriaItem(
                id=s.section_id,
                # A heading may carry the whole rule (observed in Feishu exports), so it is never dropped.
                text=f"{s.title}\n\n{s.content}".strip(),
                attachment_info="",
                source_type="section",
            )
            for s in sections
        ]
    )


def _detect_semantic_violations(tc: TestCase) -> list[str]:
    """Return human-readable violation messages for a test case's expected_results fields."""
    violations: list[str] = []
    for i, step in enumerate(tc.steps):
        er = step.expected_results
        if _CURRENCY_PAIR_RE.search(er):
            violations.append(
                f"步骤 {i + 1}：expected_results 包含货币对代码（如 USD/CNY），"
                "应将其移入 test_data 并在 expected_results 中以标签引用。"
            )
        if _KV_TEST_DATA_RE.search(er):
            violations.append(
                f"步骤 {i + 1}：expected_results 包含 key=value 格式的测试数据，"
                "应将其移入 test_data。"
            )
        if _PENDING_FIELD_RE.search(er):
            violations.append(
                f"步骤 {i + 1}：expected_results 包含 [字段名待确认] 标记，"
                "应改用可观测行为描述，将字段名移至 TestCase.comment。"
            )
    return violations


@dataclass(slots=True)
class _GenerationRunState:

    requirement_doc_content: str = ""
    result: GeneratedTestCases | None = None


def _validate_generated_requirements(
    ctx: RunContext[AcceptanceCriteriaList], output: GeneratedTestCases
) -> GeneratedTestCases:
    valid_ids = {item.id for item in ctx.deps.items}
    if any(not case.covered_requirement_ids or set(case.covered_requirement_ids) - valid_ids
           for case in output.test_cases):
        raise ModelRetry(f"每条用例的 covered_requirement_ids 必须非空且只引用合法需求项 ID：{sorted(valid_ids)}；"
                         "文档子章节若归属于合并需求项，请引用该需求项 ID。保留用例场景，仅纠正引用。")
    return output


def _validate_repair(ctx: RunContext[RepairContext], output: GeneratedTestCases) -> GeneratedTestCases:
    """Reject unsupported references and lost coverage inside bounded output retries."""
    if len(output.test_cases) != 1:
        raise ModelRetry("修复必须且只能返回一个完整用例。")
    _validate_repair_case(ctx.deps, output.test_cases[0])
    return output


def _validate_repair_case(context: RepairContext, candidate: TestCase) -> None:
    allowed = {item.id for item in context.requirements.items}
    references = set(candidate.covered_requirement_ids)
    if not references or not references <= allowed or not context.required_ids <= references:
        raise ModelRetry(f"只能引用本例已提供全文的需求ID {sorted(allowed)}，且必须保留 {sorted(context.required_ids)}。")


def _validate_repair_pair(
    ctx: RunContext[dict[str, RepairContext]], output: GeneratedTestCases,
) -> GeneratedTestCases:
    keys = [case.key for case in output.test_cases]
    if len(keys) != len(ctx.deps) or set(keys) != set(ctx.deps):
        raise ModelRetry("必须为本批每个key恰好返回一个完整用例；不能遗漏、重复或新增。")
    for candidate in output.test_cases:
        context = ctx.deps[candidate.key]
        if context.owner is None:
            raise ValueError("Pair repair requires an existing owner")
        if candidate.version != context.owner.version + 1:
            raise ModelRetry("修复版本必须为该用例原version加一。")
        _validate_repair_case(context, candidate)
    return output


def _repair_retry_prompt(context: RepairContext, candidate: TestCase) -> str:
    return (context.prompt + "\n以下候选存在违规，请仅修正违规，仍遵守原覆盖及局部ID限制：\n"
            + candidate.model_dump_json() + "\n" + "\n".join(_detect_semantic_violations(candidate)))


class TestCaseGenerationAgent(AgentBase):
    __test__ = False

    def __init__(self):
        self.ac_extraction_prompt = AcExtractionPrompt()
        self.tc_generation_prompt = TcGenerationPrompt()

        model_name = get_model(
            config.TestCaseGenerationAgentConfig.MODEL_NAME, config.TestCaseGenerationAgentConfig.FALLBACK_MODEL_NAME,
            bounded_requests=True,
        )

        self.ac_extractor_agent = CustomLlmWrapper.create_agent(
            model_name=model_name,
            output_type=AcceptanceCriteriaList,
            system_prompt=self.ac_extraction_prompt.get_prompt() + "\n\n" + UNTRUSTED_CONTENT_POLICY,
            toolsets=[feishu_mcp_server],
            name="ac_extractor",
            thinking_level=config.TestCaseGenerationAgentConfig.AC_EXTRACTOR_THINKING_LEVEL,
            max_tokens=config.TestCaseGenerationAgentConfig.AC_EXTRACTOR_MAX_TOKENS,
        )

        self.test_case_generator_agent = CustomLlmWrapper.create_agent(
            model_name=model_name,
            output_type=GeneratedTestCases,
            system_prompt=self.tc_generation_prompt.get_prompt() + "\n\n" + UNTRUSTED_CONTENT_POLICY,
            name="test_case_generator",
            deps_type=AcceptanceCriteriaList,
            thinking_level=config.TestCaseGenerationAgentConfig.TC_GENERATOR_THINKING_LEVEL,
            max_tokens=config.TestCaseGenerationAgentConfig.TC_GENERATOR_MAX_TOKENS,
        )
        self.test_case_generator_agent.output_validator(_validate_generated_requirements)
        repair_model = get_model(config.TestCaseGenerationAgentConfig.REPAIR_MODEL_NAME, bounded_requests=True)
        self.repair_agent = CustomLlmWrapper.create_agent(
            model_name=repair_model, output_type=GeneratedTestCases,
            system_prompt=self.tc_generation_prompt.get_prompt() + "\n\n" + UNTRUSTED_CONTENT_POLICY, name="repair_test_case",
            deps_type=RepairContext,
            thinking_level=config.TestCaseGenerationAgentConfig.TC_REPAIR_THINKING_LEVEL,
            max_tokens=config.TestCaseGenerationAgentConfig.TC_REPAIR_MAX_TOKENS,
        )
        self.repair_agent.output_validator(_validate_repair)
        self.pair_repair_agent = CustomLlmWrapper.create_agent(
            model_name=repair_model, output_type=GeneratedTestCases,
            system_prompt=self.tc_generation_prompt.get_prompt() + "\n\n" + UNTRUSTED_CONTENT_POLICY, name="repair_test_case_pair",
            deps_type=dict[str, RepairContext],
            thinking_level=config.TestCaseGenerationAgentConfig.TC_REPAIR_THINKING_LEVEL,
            max_tokens=2 * config.TestCaseGenerationAgentConfig.TC_REPAIR_MAX_TOKENS,
        )
        self.pair_repair_agent.output_validator(_validate_repair_pair)


        instruction_prompt = TestCaseGenerationSystemPrompt()
        super().__init__(
            agent_name=config.TestCaseGenerationAgentConfig.OWN_NAME,
            base_url=config.AGENT_BASE_URL,
            port=config.TestCaseGenerationAgentConfig.PORT,
            external_port=config.TestCaseGenerationAgentConfig.EXTERNAL_PORT,
            protocol=config.TestCaseGenerationAgentConfig.PROTOCOL,
            model_name=config.TestCaseGenerationAgentConfig.MODEL_NAME,
            fallback_model_name=config.TestCaseGenerationAgentConfig.FALLBACK_MODEL_NAME,
            output_type=str,
            instructions=instruction_prompt.get_prompt(),
            mcp_servers=[],
            deps_type=_GenerationRunState,
            description="Agent which generates test cases based on requirement documents.",
            tools=[self._generate_test_cases],
        )
        # ac_extractor_agent holds its own MCP toolset (feishu_mcp_server), separate from
        # the outer agent's (empty) mcp_servers, so it must be registered here too to get
        # the lifespan-held connection protection from AgentBase._lifespan.
        self._mcp_managed_agents.append(self.ac_extractor_agent)

    def get_thinking_level(self) -> ThinkingLevel:
        return config.TestCaseGenerationAgentConfig.ORCHESTRATOR_THINKING_LEVEL

    def get_max_requests_per_task(self) -> int:
        return config.TestCaseGenerationAgentConfig.MAX_REQUESTS_PER_TASK

    def get_skills(self) -> list[AgentSkill]:
        return [
            AgentSkill(
                id="generate-test-cases",
                name="Generate test cases",
                description="Generates test cases based on requirement documents.",
                tags=["qa", "test-case-generation"],
            )
            ,AgentSkill(
                id="fix-test-cases", name="Fix test cases",
                description="Applies structured review findings to a versioned design draft.",
                tags=["qa", "test-case-fix"],
            )
        ]

    async def _generate_test_cases(self, ctx: RunContext[_GenerationRunState]) -> str:
        """
        Generates test cases from the requirement document provided via deps.

        Classifies the PRD type first, then routes to the appropriate chunking
        strategy before TC generation.  Stores the result in ctx.deps.result.

        Returns:
            Summary string.
        """
        content = ctx.deps.requirement_doc_content
        doc_type = _classify_prd(content)
        logger.info(f"PRD classified as: '{doc_type}'")

        match doc_type:
            case "functional":
                chunks = _chunk_by_sections(content)
            case _:
                attachments_content = self._fetch_attachments([])
                chunks = await self.extract_acceptance_criteria(attachments_content, content)

        kb_context = await query_photon_kb(
            f"生成测试用例前，请提供以下需求相关的历史缺陷、风险点、模块规则、注意事项：\n{_strip_html(content)[:500]}"
        )

        generated_test_cases = await self.generate_test_cases_from_acs(chunks, content, kb_context)
        ctx.deps.result = generated_test_cases
        return f"Successfully generated {len(generated_test_cases.test_cases)} test cases."

    async def run(self, received_message: Message) -> Message:
        if is_stage_message(get_message_text(received_message)):
            return await super().run(received_message)
        doc_content = get_message_text(received_message)
        state = _GenerationRunState(requirement_doc_content=doc_content)
        result_message = await super().run(received_message, deps=state)
        if state.result is None:
            return result_message
        context_id = getattr(received_message, "context_id", None)
        task_id = getattr(received_message, "task_id", None)
        return new_text_message(
            text=state.result.model_dump_json(),
            context_id=context_id,
            task_id=task_id,
        )

    async def execute_stage(self, request: StageRequest) -> DesignDraft | FixResult:
        """Generate a grounded draft or propose only the requested case changes."""
        payload = request.payload
        content = payload["document"]
        if request.operation == "generate":
            from common.design_runtime import checkpoint_unit
            async def fetch_knowledge():
                snapshot = await query_knowledge_snapshot(
                    f"提供以下需求相关的历史缺陷、风险点与模块规则：\n{_strip_html(content)[:500]}"
                )
                if snapshot.status not in {"ok", "no_match"}:
                    raise RuntimeError("Required internal knowledge base is unavailable")
                return snapshot
            knowledge = await checkpoint_unit("knowledge", content, KnowledgeSnapshot, fetch_knowledge)
            if knowledge.status not in {"ok", "no_match"}:
                raise RuntimeError("Required internal knowledge base is unavailable")
            # The stage owns its immutable document; do not let an extraction tool reread it.
            with self.ac_extractor_agent.override(toolsets=[]):
                async def extract():
                    return (_chunk_by_sections(content) if _classify_prd(content) == "functional"
                            else await self.extract_acceptance_criteria({}, content))
                requirements = await checkpoint_unit("requirements", content, AcceptanceCriteriaList, extract)
            if not requirements.items:
                raise ValueError("No requirements extracted")
            generated = await self.generate_test_cases_from_acs(requirements, content, knowledge.content)
            valid_ids = {r.id for r in requirements.items}
            for case in generated.test_cases:
                if _detect_semantic_violations(case):
                    raise ValueError("Generated case still violates generation constraints")
                if not case.covered_requirement_ids or not set(case.covered_requirement_ids) <= valid_ids:
                    raise ValueError("Generated case does not reference known requirement IDs")
            if not generated.test_cases:
                raise ValueError("Generation returned no cases")
            return DesignDraft(requirements=requirements, knowledge=knowledge, test_cases=generated.test_cases)
        if request.operation != "fix":
            raise ValueError("Generation agent supports only generate and fix")
        knowledge = KnowledgeSnapshot.model_validate(payload["knowledge"])
        if knowledge.status not in {"ok", "no_match"}:
            raise ValueError("Fix requires the original knowledge snapshot")
        cases = {c.key: c for c in (TestCase.model_validate(v) for v in payload["test_cases"])}
        findings = [ReviewFinding.model_validate(f) for f in payload["findings"]]
        replacements = []
        additions = []
        by_owner: dict[str, list[ReviewFinding]] = {}
        for finding in findings:
            if finding.action in {"modify", "remove_duplicate_steps"}:
                by_owner.setdefault(finding.owner_case_key, []).append(finding)
        for group in batched(by_owner, 2, strict=False):
            contexts = {key: build_repair_context(payload, by_owner[key], cases[key]) for key in group}
            candidates = ([await self._repair_case(contexts[group[0]])] if len(group) == 1
                          else await self._repair_pair(contexts))
            for key, candidate in zip(group, candidates, strict=True):
                replacements.append(candidate.model_copy(update={"key": key, "version": cases[key].version + 1}))
        gaps = [f for f in findings if f.action == "add_test_case" and not f.restore_case_key]
        if len(gaps) > config.DesignConfig.MAX_NEW_CASES_PER_ROUND:
            raise ValueError("Too many coverage gaps for one bounded repair")
        for gap in gaps:
            additions.append(await self._repair_case(build_repair_context(payload, [gap])))
        return FixResult(replacements=replacements, additions=additions)

    async def _repair_pair(self, contexts: dict[str, RepairContext]) -> list[TestCase]:
        """Persist a complete pair candidate; split a metered length failure once."""
        from common.design_runtime import ModelOutputFailure, checkpoint_unit

        async def repair():
            try:
                candidates = await self._repair_pair_uncached(contexts)
            except ModelOutputFailure:
                candidates = [await self._repair_case(context) for context in contexts.values()]
            return GeneratedTestCases(test_cases=candidates)

        result = await checkpoint_unit(
            "pair:" + ",".join(contexts), {key: context.prompt for key, context in contexts.items()},
            GeneratedTestCases, repair,
        )
        return result.test_cases

    async def _repair_pair_uncached(self, contexts: dict[str, RepairContext]) -> list[TestCase]:
        """Share complete sources while keeping each replacement's evidence and scope."""
        views = {key: json.loads(context.prompt) for key, context in contexts.items()}
        shared = next(iter(views.values()))
        requirements = {item.id: item for context in contexts.values() for item in context.requirements.items}
        prompt = {
            "instruction": "逐例独立修复本批全部用例，返回原key及原version+1；严格保留各自原覆盖，不能交换场景或覆盖。",
            **{key: shared[key] for key in ("document", "knowledge", "case_index")},
            "requirements": {"items": [item.model_dump(mode="json") for item in requirements.values()]},
            "repairs": [{**{field: view[field] for field in ("owner", "findings", "related_cases")},
                         "allowed_requirement_ids": [item.id for item in contexts[key].requirements.items]}
                        for key, view in views.items()],
        }
        result = await self.pair_repair_agent.run(
            json.dumps(prompt, ensure_ascii=False, separators=(",", ":")), deps=contexts,
        )
        candidates = {case.key: case for case in result.output.test_cases}
        repaired = []
        for key, context in contexts.items():
            candidate = candidates[key]
            if _detect_semantic_violations(candidate):
                candidate = await self._repair_case(context, candidate)
            repaired.append(candidate)
        return repaired

    async def _repair_case(self, context: RepairContext, previous_candidate: TestCase | None = None) -> TestCase:
        """Reuse a successful owner candidate without committing a partial revision."""
        from common.design_runtime import checkpoint_unit

        async def repair():
            return await self._repair_case_uncached(context, previous_candidate)

        inputs = {"prompt": context.prompt, "previous": previous_candidate.model_dump(mode="json") if previous_candidate else None}
        from common.design_store import digest
        return await checkpoint_unit("single:" + digest(inputs),
                                     inputs, TestCase, repair)

    async def _repair_case_uncached(self, context: RepairContext, previous_candidate: TestCase | None = None) -> TestCase:
        """Use the same local scope and output cap for initial and semantic repair."""
        prompt = _repair_retry_prompt(context, previous_candidate) if previous_candidate else context.prompt
        attempts = config.TestCaseGenerationAgentConfig.TC_VALIDATION_MAX_RETRIES
        if previous_candidate is None:
            attempts += 1
        for _ in range(attempts):
            result = await self.repair_agent.run(prompt, deps=context)
            candidate = result.output.test_cases[0]
            violations = _detect_semantic_violations(candidate)
            if not violations:
                return candidate
            prompt = _repair_retry_prompt(context, candidate)
        raise ValueError("Fixed case still violates generation constraints")

    async def _generate_test_cases_for_ac_batch(
        self,
        batch: tuple[AcceptanceCriteriaItem, ...],
        all_acs: AcceptanceCriteriaList,
        requirement_doc_content: str,
        batch_number: int,
        total_batches: int,
        kb_context: str | None = None,
    ) -> GeneratedTestCases:
        """Generates test cases for a batch of ACs in a single LLM call."""
        user_message = untrusted_input(
            document=requirement_doc_content,
            knowledge=kb_context or "",
            all_requirements=all_acs.model_dump(mode="json")["items"],
            batch_requirements=[item.model_dump(mode="json") for item in batch],
        )
        from common.design_runtime import checkpoint_unit
        async def generate():
            return (await self.test_case_generator_agent.run(user_message, deps=all_acs)).output
        batch_result = await checkpoint_unit(f"raw-generation:{batch_number}", user_message, GeneratedTestCases, generate)
        logger.info(
            f"Generated {len(batch_result.test_cases)} test cases in batch {batch_number}/{total_batches}."
        )
        return batch_result

    async def _regenerate_single_case(
        self,
        violating_case: TestCase,
        violations: list[str],
        requirement_doc_content: str,
        all_acs: AcceptanceCriteriaList,
        kb_context: str | None = None,
    ) -> TestCase:
        """
        Regenerate a single violating test case with targeted violation feedback.

        Returns the fixed case if violations are resolved, otherwise the original with a
        human-review warning appended to its comment field.
        """
        max_retries = config.TestCaseGenerationAgentConfig.TC_VALIDATION_MAX_RETRIES
        violation_text = "\n".join(f"- {v}" for v in violations)
        user_message = (
            f"以下测试用例存在规则违规，请仅修正违规问题后重新输出该测试用例（不要生成其他用例）：\n\n"
            f"{violating_case.model_dump_json()}\n\n"
            f"违规问题：\n{violation_text}\n\n"
            + untrusted_input(document=requirement_doc_content, knowledge=kb_context or "",
                              all_requirements=all_acs.model_dump(mode="json")["items"])
        )
        for attempt in range(max_retries):
            result = await self.test_case_generator_agent.run(user_message, deps=all_acs)
            regenerated: GeneratedTestCases = result.output
            if regenerated.test_cases:
                candidate = regenerated.test_cases[0]
                remaining = _detect_semantic_violations(candidate)
                if not remaining:
                    logger.info(
                        f"Regeneration succeeded for case '{violating_case.name}' on attempt {attempt + 1}."
                    )
                    return candidate
                logger.warning(
                    f"Regeneration attempt {attempt + 1}/{max_retries} for case "
                    f"'{violating_case.name}' still has {len(remaining)} violation(s)."
                )
        warning = "⚠️ 需人工复核：自动修正未能消除规则违规"
        updated_comment = f"{violating_case.comment}\n{warning}".strip() if violating_case.comment else warning
        logger.error(
            f"Case '{violating_case.name}' could not be auto-corrected after {max_retries} attempt(s)."
        )
        return violating_case.model_copy(update={"comment": updated_comment})

    async def generate_test_cases_from_acs(
        self,
        extracted_acceptance_criteria: AcceptanceCriteriaList,
        requirement_doc_content: str,
        kb_context: str | None = None,
    ) -> GeneratedTestCases:
        """Generates all test cases from ACs using batched LLM calls."""
        batch_size = config.TestCaseGenerationAgentConfig.AC_BATCH_SIZE
        batches = list(batched(extracted_acceptance_criteria.items, batch_size, strict=False))
        logger.info(
            f"Generating test cases for {len(extracted_acceptance_criteria.items)} ACs "
            f"in {len(batches)} batch(es) of up to {batch_size}."
        )
        sem = asyncio.Semaphore(config.TestCaseGenerationAgentConfig.LLM_CONCURRENCY_LIMIT)

        async def _run_batch(batch: tuple[AcceptanceCriteriaItem, ...], idx: int) -> GeneratedTestCases:
            async with sem:
                return await self._generate_test_cases_for_ac_batch(
                    batch, extracted_acceptance_criteria, requirement_doc_content, idx + 1, len(batches), kb_context
                )

        tasks = [_run_batch(batch, i) for i, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        test_cases, llm_comments = [], []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                raise RuntimeError(f"TC generation failed for batch {i + 1}: {result}") from result
            test_cases.extend(result.test_cases)
            if result.llm_comments:
                llm_comments.append(result.llm_comments)
        logger.info(f"Generated {len(test_cases)} test cases in total.")

        with observability.tracer.start_as_current_span("test_case_generation.semantic_violation_gate") as span:
            violating = [(i, v) for i, tc in enumerate(test_cases) if (v := _detect_semantic_violations(tc))]
            if violating:
                logger.info(f"Found {len(violating)} case(s) with semantic violations; regenerating...")
                sem_regen = asyncio.Semaphore(config.TestCaseGenerationAgentConfig.LLM_CONCURRENCY_LIMIT)

                async def _fix(idx: int, violations: list[str]) -> tuple[int, TestCase]:
                    async with sem_regen:
                        return idx, await self._regenerate_single_case(
                            test_cases[idx], violations, requirement_doc_content, extracted_acceptance_criteria, kb_context
                        )

                repairs = await asyncio.gather(*[_fix(i, v) for i, v in violating], return_exceptions=True)
                for repair in repairs:
                    if isinstance(repair, BaseException):
                        raise RuntimeError("Semantic regeneration failed") from repair
                    idx, fixed_case = repair
                    test_cases[idx] = fixed_case
                logger.info("Post-processing complete.")
            span.set_attribute("generation.total_cases", len(test_cases))
            span.set_attribute("generation.semantic_violations_count", len(violating))

        return GeneratedTestCases(test_cases=test_cases, llm_comments="\n".join(llm_comments) or None)

    async def extract_acceptance_criteria(
        self, attachments_content: dict[str, BinaryContent], requirement_doc_content: str
    ) -> AcceptanceCriteriaList:
        user_message_parts: list[str | BinaryContent] = [untrusted_input(document=requirement_doc_content)]
        if attachments_content:
            for filename, binary_content in attachments_content.items():
                user_message_parts.append(f"Attachment: {filename}")
                user_message_parts.append(binary_content)

        logger.info("Starting AC extraction with %d attachments", len(attachments_content))
        result = await self.ac_extractor_agent.run(user_message_parts)
        extracted_acceptance_criteria: AcceptanceCriteriaList = result.output
        logger.info(f"Extracted {len(extracted_acceptance_criteria.items)} ACs")
        return extracted_acceptance_criteria

    @staticmethod
    def _upload_test_cases_into_test_management_system(
        test_cases: GeneratedTestCases, project_key: str, user_story_id: int
    ) -> str:
        """
        Uploads the provided test cases in the configured test management system.

        Args:
            test_cases: The list of test cases to be created.
            project_key: The key of the Jira project to which the Jira issue belongs.
            user_story_id: ID of the Jira user story (not its key), e.g. 120.

        Returns:
            A confirmation message with the keys (IDs) of the created test cases.
        """
        client = get_test_management_client()
        created_test_case_ids = client.create_test_cases(test_cases.test_cases, project_key, user_story_id)
        return f"Successfully created test cases with following keys (IDs): {', '.join(created_test_case_ids)}"


agent = TestCaseGenerationAgent()
app = agent.a2a_server

if __name__ == "__main__":
    agent.start_as_server()
