# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Safety properties of bounded review/repair and publishing."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.design_models import (
    CaseAssessment,
    DesignDraft,
    DesignReview,
    FixResult,
    KnowledgeSnapshot,
    ReviewFinding,
    StageResult,
    StageUsage,
    SuiteAssessment,
    UntestableRequirement,
)
from common.models import AcceptanceCriteriaList, ClassifiedTestCases, TestCase, TestStep
from common.services.feishu_bitable_service import BitablePublishResult
from orchestrator.test_case_design import DesignCoordinator, DesignSession, commit_repair, validate_review


def case(key="TC-1", version=1, ids=None):
    return TestCase(
        key=key,
        version=version,
        covered_requirement_ids=ids or ["AC-1"],
        name="验证支付结果",
        summary="支付成功",
        comment="",
        labels=[],
        preconditions=None,
        parent_issue_key="S1",
        steps=[TestStep(action="支付", expected_results="成功", test_data=[])],
    )


def draft():
    return DesignDraft(
        requirements=AcceptanceCriteriaList(
            items=[
                {
                    "id": "AC-1",
                    "text": "支付成功",
                    "attachment_info": "",
                }
            ]
        ),
        knowledge=KnowledgeSnapshot(status="no_match"),
        test_cases=[case()],
    )


def finding(action="modify", owner="TC-1", **kwargs):
    return ReviewFinding(
        severity="high",
        action=action,
        owner_case_key=owner,
        related_case_keys=kwargs.pop("related_case_keys", []),
        requirement_ids=["AC-1"],
        category="coverage",
        reason="缺少断言",
        expected_change="补充结果断言",
        **kwargs,
    )


def review(cases, findings=None):
    return DesignReview(
        assessments=[
            CaseAssessment(case_key=c.key, case_version=c.version, findings=(findings or []) if c.key == "TC-1" else [])
            for c in cases
        ],
        suite=SuiteAssessment(completed=True, findings=[]),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("patch_upgrade", [False, True])
async def test_durable_coordinator_restart_reuses_complete_stages_without_rebilling(tmp_path, monkeypatch, patch_upgrade):
    from common import design_store

    paid = []
    fail_classification = True
    owner = design_store.DesignStore(str(tmp_path))
    await owner.call("open")
    previous = {"workflow": {"sources": {"orchestrator/test_case_design.py": "before"},
                             "policy": {}, "targets": {}, "provider": "same"},
                "config_sha256": "same", "environment_sha256": "same", "runtime_sources": {}}
    run, _ = await owner.call("create_run", "story", {"workflow": design_store.digest(previous["workflow"])}, 6000000, 3600)
    monkeypatch.setattr(design_store, "store", owner)

    async def send(request):
        nonlocal fail_classification
        if request.operation == "classify" and fail_classification:
            fail_classification = False
            raise RuntimeError("safe failure before request")
        if request.operation == "generate":
            output = draft().model_dump(mode="json")
        elif request.operation == "review":
            output = review([case()]).model_dump(mode="json")
        else:
            output = {"test_cases": [{"issue_key": "TC-1", "name": "case", "test_type": "API",
                                      "automation_capability": "manual", "labels": [], "tool_use_comment": ""}]}
        active = await owner.call("run", run["id"])
        permit = await owner.call("reserve", run["id"], active["generation"], request.operation, request.operation, 1000, 6000000)
        await owner.call("settle", run["id"], active["generation"], permit["id"], 100, {}, False)
        paid.append(request.operation)
        return StageResult(run_id=request.run_id, stage_id=request.stage_id,
                           input_revision=request.input_revision, execution_id=request.stage_id,
                           status="completed", usage=StageUsage(input_tokens=100, requests=1), payload=output)

    try:
        with pytest.raises(RuntimeError, match="before request"):
            await DesignCoordinator(send, run).design("prd", "S1")
    finally:
        await owner.close()
    owner = design_store.DesignStore(str(tmp_path))
    await owner.call("open")
    monkeypatch.setattr(design_store, "store", owner)
    try:
        if patch_upgrade:
            current = {**previous, "workflow": {**previous["workflow"],
                                               "sources": {"orchestrator/test_case_design.py": "after"}}}
            await owner.call("upgrade", run["id"], previous, current, ["orchestrator/test_case_design.py"],
                             "test", "reviewed patch; reuse paid stages")
        else:
            await owner.call("request_resume", run["id"], "test", "known pre-send interruption")
        resumed = await owner.call("claim", run["id"])
        coordinator = DesignCoordinator(send, resumed)
        final, _ = await coordinator.design("prd", "S1")
        assert paid == ["generate", "review", "classify"]
        assert coordinator.used_tokens == 300
        assert final.test_cases[0].version == 1 and coordinator.session.stop_reason == "converged"
        assert (await owner.call("run", run["id"]))["snapshot"]["revision"] == 0
    finally:
        await owner.close()


def test_stale_or_missing_receipts_rejected():
    session = DesignSession(cases=[case(version=2)])
    with pytest.raises(ValueError, match="stale"):
        validate_review(review([case()]), session, draft())


@pytest.mark.asyncio
@pytest.mark.parametrize("waived", [False, True])
async def test_every_requirement_must_be_covered_or_waived_with_a_reason(waived):
    data = draft()
    data.requirements.items.append(data.requirements.items[0].model_copy(update={
        "id": "sec-1", "text": "修订记录\n版本 v0.1", "source_type": "section",
    }))
    operations = []

    async def send(request):
        operations.append(request.operation)
        if request.operation == "generate":
            payload = data.model_dump(mode="json")
        elif request.operation == "review":
            result = review([case()])
            if waived:
                result.suite.untestable = [UntestableRequirement(requirement_id="sec-1", reason="版本记录，无可测行为")]
            payload = result.model_dump(mode="json")
        else:
            payload = {"test_cases": [{"issue_key": "TC-1", "name": "case", "test_type": "API",
                                       "automation_capability": "manual", "labels": [], "tool_use_comment": ""}]}
        return StageResult(run_id=request.run_id, stage_id=request.stage_id,
                           input_revision=request.input_revision, execution_id=request.stage_id,
                           status="completed", payload=payload, usage=StageUsage())

    coordinator = DesignCoordinator(send)
    if not waived:
        with pytest.raises(ValueError, match="uncovered requirements"):
            await coordinator.design("PRD", "S1")
        assert operations == ["generate", "review"]
    else:
        final, _ = await coordinator.design("PRD", "S1")
        assert operations == ["generate", "review", "classify"]
        assert coordinator.session.stop_reason == "converged"
        assert coordinator.session.reviews[-1].suite.untestable[0].requirement_id == "sec-1"
        assert len(final.test_cases) == 1


@pytest.mark.parametrize(("requirement_id", "with_finding", "error"), [
    ("sec-9", False, "unknown requirement"),
    ("AC-1", True, "raised a finding"),
])
def test_review_cannot_waive_unknown_or_testable_requirements(requirement_id, with_finding, error):
    result = review([case()], [finding()] if with_finding else [])
    result.suite.untestable = [UntestableRequirement(requirement_id=requirement_id, reason="说明性内容")]
    with pytest.raises(ValueError, match=error):
        validate_review(result, DesignSession(cases=[case()]), draft())


def test_waiver_requires_a_reason():
    with pytest.raises(ValueError):
        UntestableRequirement(requirement_id="sec-1", reason="  ")


def test_invalid_repair_does_not_mutate_session():
    session = DesignSession(cases=[case()], next_key=2)
    before = session.cases[0].model_dump()
    with pytest.raises(ValueError, match="stale"):
        commit_repair(session, draft(), [finding()], FixResult(replacements=[case()], additions=[]), "S1")
    assert session.cases[0].model_dump() == before
    assert session.revision == 0


def test_delete_restore_keeps_identity_and_increases_version():
    session = DesignSession(cases=[case(), case("TC-2")], next_key=3)
    deletion = finding("delete_test_case", related_case_keys=["TC-2"])
    commit_repair(session, draft(), [deletion], FixResult(replacements=[], additions=[]), "S1")
    assert [c.key for c in session.cases] == ["TC-2"]
    restoration = finding("add_test_case", None, restore_case_key="TC-1")
    commit_repair(session, draft(), [restoration], FixResult(replacements=[], additions=[]), "S1")
    assert next(c.version for c in session.cases if c.key == "TC-1") == 2
    assert session.next_key == 3
    assert not session.deleted


def test_new_case_must_cover_its_assigned_gap():
    data = draft()
    data.requirements.items.append(data.requirements.items[0].model_copy(update={"id": "AC-2"}))
    gap = finding("add_test_case", None).model_copy(update={"requirement_ids": ["AC-2"]})
    session = DesignSession(cases=[case()], next_key=2)
    with pytest.raises(ValueError, match="assigned"):
        commit_repair(session, data, [gap], FixResult(replacements=[], additions=[case()]), "S1")
    assert session.next_key == 2


@pytest.mark.asyncio
async def test_no_semantic_progress_stops_before_extra_fix():
    operations = []

    async def send(request):
        operations.append(request.operation)
        if request.operation == "generate":
            payload = draft().model_dump(mode="json")
        elif request.operation == "review":
            cases = [TestCase.model_validate(c) for c in request.payload["test_cases"]]
            payload = review(cases, [finding()]).model_dump(mode="json")
        elif request.operation == "fix":
            payload = FixResult(replacements=[case(version=2)], additions=[]).model_dump(mode="json")
        else:
            payload = {
                "test_cases": [
                    {
                        "issue_key": "TC-1",
                        "name": "case",
                        "test_type": "API",
                        "automation_capability": "manual",
                        "labels": [],
                        "tool_use_comment": "",
                    }
                ]
            }
        return StageResult(
            run_id=request.run_id,
            stage_id=request.stage_id,
            input_revision=request.input_revision,
            execution_id=str(len(operations)),
            status="completed",
            payload=payload,
            usage=StageUsage(input_tokens=10),
        )

    coordinator = DesignCoordinator(send)
    await coordinator.design("PRD", "S1")
    assert coordinator.session.stop_reason == "no_progress"
    assert operations == ["generate", "review", "fix", "review", "classify"]
    assert coordinator.used_tokens == 50


@pytest.mark.asyncio
async def test_failed_stage_records_known_usage_and_stops():
    async def send(request):
        return StageResult(
            run_id=request.run_id,
            stage_id=request.stage_id,
            input_revision=0,
            execution_id="e",
            status="failed",
            error="failure",
            usage=StageUsage(input_tokens=42),
        )

    coordinator = DesignCoordinator(send)
    with pytest.raises(RuntimeError, match="failure"):
        await coordinator.stage("generate", {})
    assert coordinator.used_tokens == 42


@pytest.mark.asyncio
async def test_repeatedly_truncated_stage_stops_as_a_provider_error_with_known_usage():
    from common.provider_errors import ProviderRetryError

    async def send(request):
        return StageResult(
            run_id=request.run_id, stage_id=request.stage_id, input_revision=0, execution_id="e",
            status="failed", error="provider_stream_incomplete", error_code="provider_stream_incomplete",
            usage=StageUsage(input_tokens=64020),
        )

    coordinator = DesignCoordinator(send)
    with pytest.raises(ProviderRetryError, match="provider_stream_incomplete"):
        await coordinator.stage("review", {})
    assert coordinator.used_tokens == 64020


@pytest.mark.asyncio
@pytest.mark.parametrize("archive_enabled", [False, True])
@pytest.mark.parametrize("failure", ["archive", "publish", "unconverged", "no_progress", "confirm", "success"])
async def test_pipeline_confirm_gate_and_preserved_url(failure, archive_enabled):
    import orchestrator.main as module

    archive_failed = archive_enabled and failure == "archive"
    succeeded = failure in {"success", "unconverged", "no_progress"} or (failure == "archive" and not archive_enabled)
    design = draft()
    coordinator = MagicMock()
    coordinator.design = AsyncMock(return_value=(design, ClassifiedTestCases(test_cases=[])))
    coordinator.session = DesignSession(
        cases=design.test_cases,
        reviews=[review(design.test_cases)],
        stop_reason={"unconverged": "iteration_limit", "no_progress": "no_progress"}.get(failure, "converged"),
    )
    coordinator.used_tokens = 10
    published = BitablePublishResult(
        url="https://example/base/1", expected=1, written=1, comments_complete=failure != "publish", accessible=True
    )
    with (
        patch.object(module, "feishu_get_doc_content", AsyncMock(return_value="PRD")),
        patch.object(module, "DesignCoordinator", return_value=coordinator),
        patch.object(module, "_resolve_story_business_line", AsyncMock(return_value=None)),
        patch(
            "common.services.oss_archive.archive_design",
            AsyncMock(side_effect=RuntimeError("archive_failed") if failure == "archive" else None),
        ) as archive,
        patch.object(module, "publish_bitable_report", return_value=published) as publish,
        patch.object(
            module,
            "_write_bitable_url_to_story",
            AsyncMock(side_effect=RuntimeError("confirm_unknown") if failure == "confirm" else None),
        ) as confirm,
        patch.object(module, "notify_pipeline_failure", AsyncMock()) as notify_failure,
        patch.object(module, "notify_pipeline_success", AsyncMock()) as notify_success,
        patch.object(module, "_record_error"),
        patch("config.FEISHU_BITABLE_SYNC_ENABLED", True),
        patch("config.DesignConfig.OSS_ARCHIVE_ENABLED", archive_enabled),
    ):
        await module._run_pipeline("S1", "P1", "https://example/doc")
    assert archive.await_count == int(archive_enabled)
    assert publish.call_count == int(not archive_failed)
    assert confirm.await_count == int(succeeded or failure == "confirm")
    assert notify_success.await_count == int(succeeded)
    if not succeeded:
        assert notify_failure.call_args.kwargs["bitable_url"] == (None if archive_failed else published.url)


@pytest.mark.asyncio
@pytest.mark.parametrize("matching_agents", [1, 2])
async def test_protocol_route_uses_exact_skill_without_model_selection(matching_agents):
    from types import SimpleNamespace

    import orchestrator.main as module
    registry = MagicMock()
    registry.is_empty = AsyncMock(return_value=False)
    cards = {str(i): SimpleNamespace(skills=[SimpleNamespace(id="review-test-cases")]) for i in range(matching_agents)}
    registry.get_all_cards = AsyncMock(return_value=cards)
    registry.get_available_agents = AsyncMock(return_value=list(cards))
    registry.update_status = AsyncMock()
    with patch.object(module, "agent_registry", registry), patch.object(module, "_select_agent", AsyncMock()) as select:
        if matching_agents == 1:
            selected, card = await module.reserve_agent_waiting_if_needed("review", required_skill_id="review-test-cases")
            assert selected == "0" and card is cards["0"]
            registry.update_status.assert_awaited_once()
        else:
            from fastapi import HTTPException
            with pytest.raises(HTTPException, match="Expected one agent"):
                await module.reserve_agent_waiting_if_needed("review", required_skill_id="review-test-cases")
            registry.update_status.assert_not_awaited()
    select.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow", ["_run_pipeline", "_run_requirement_review"])
async def test_prd_permission_failure_is_visible_without_dispatch(workflow):
    """Document failures appear in dashboard history before any agent call."""
    import orchestrator.main as module
    from orchestrator.models import TaskHistory, TaskStatus

    history = TaskHistory()
    with (
        patch("config.DesignConfig.REQUIREMENT_REVIEW_ENABLED", True),
        patch.object(module, "task_history", history),
        patch.object(module, "feishu_get_doc_content", AsyncMock(side_effect=RuntimeError("99991672: sheets:spreadsheet:read"))),
        patch.object(module, "DesignCoordinator") as coordinator,
        patch.object(module, "notify_pipeline_failure", AsyncMock()),
        patch.object(module, "_record_error"),
    ):
        await getattr(module, workflow)("S1", "P1", "https://example/doc")
    task, = await history.get_all()
    assert task.status == TaskStatus.FAILED
    assert "99991672" in task.error_message
    assert task.current_activity is None
    assert task.end_time is not None
    coordinator.assert_not_called()


@pytest.mark.asyncio
async def test_disabled_requirement_review_has_no_side_effects():
    import orchestrator.main as module

    with (
        patch("config.DesignConfig.REQUIREMENT_REVIEW_ENABLED", False),
        patch.object(module, "task_history") as history,
        patch.object(module, "feishu_get_doc_content", AsyncMock()) as fetch,
        patch.object(module, "DesignCoordinator") as coordinator,
        patch.object(module, "MeegoClient") as client,
    ):
        await module._run_requirement_review("S1", "P1", "https://example/doc")
    history.add.assert_not_called()
    fetch.assert_not_awaited()
    coordinator.assert_not_called()
    client.assert_not_called()


def test_repair_cannot_drop_original_coverage_even_if_ids_are_known():
    data = draft()
    data.requirements.items.append(data.requirements.items[0].model_copy(update={"id": "AC-2"}))
    original = case(ids=["AC-1", "AC-2"])
    session = DesignSession(cases=[original], next_key=2)
    with pytest.raises(ValueError, match="coverage"):
        commit_repair(session, data, [finding()],
                      FixResult(replacements=[case(version=2)], additions=[]), "S1")
    assert session.cases == [original]
    assert session.revision == 0


@pytest.mark.asyncio
async def test_fix_cannot_spend_final_review_and_classification_reserve():
    sent = []

    async def send(request):
        sent.append(request)
        return StageResult(run_id=request.run_id, stage_id=request.stage_id,
                           input_revision=request.input_revision, execution_id=str(len(sent)),
                           status="completed", payload={}, usage=StageUsage(input_tokens=10))

    coordinator = DesignCoordinator(send)
    coordinator.review_reserve = 200
    coordinator.classification_reserve = 100
    with patch("config.DesignConfig.TOKEN_LIMIT", 1000):
        await coordinator.stage("fix", {})
        assert sent[-1].token_limit == 700
        await coordinator.stage("review", {})
        assert sent[-1].token_limit == 890
        await coordinator.stage("classify", {})
        assert sent[-1].token_limit == 980
        coordinator.used_tokens = 750
        with pytest.raises(RuntimeError, match="budget_closure_reserved"):
            await coordinator.stage("fix", {})
    assert len(sent) == 3


@pytest.mark.asyncio
async def test_three_reviews_two_repairs_keep_budget_and_version_receipts():
    requests = []
    review_count = 0

    async def send(request):
        nonlocal review_count
        requests.append(request)
        if request.operation == "generate":
            output = draft().model_copy(update={"test_cases": [case(f"TC-{i}") for i in range(1, 24)]})
        elif request.operation == "review":
            review_count += 1
            active = [TestCase.model_validate(c) for c in request.payload["test_cases"]]
            changes = [finding().model_copy(update={"reason": f"round {review_count}"})] if review_count < 3 else []
            output = review(active, changes)
        elif request.operation == "fix":
            original = TestCase.model_validate(request.payload["test_cases"][0])
            output = FixResult(replacements=[original.model_copy(update={"version": original.version + 1,
                                                                         "summary": f"round {review_count}"})], additions=[])
            assert request.token_limit < 1000000 - coordinator.used_tokens
        else:
            output = ClassifiedTestCases.model_validate({"test_cases": [
                {"issue_key": c["key"], "name": "case", "test_type": "API", "automation_capability": "manual",
                 "labels": [], "tool_use_comment": ""} for c in request.payload["test_cases"]]})
        return StageResult(run_id=request.run_id, stage_id=request.stage_id, input_revision=request.input_revision,
                           execution_id=str(len(requests)), status="completed", payload=output.model_dump(mode="json"),
                           usage=StageUsage(input_tokens=50000, output_tokens=1000))

    coordinator = DesignCoordinator(send)
    result, _ = await coordinator.design("Synthetic PRD", "S1")
    assert [r.operation for r in requests] == ["generate", "review", "fix", "review", "fix", "review", "classify"]
    assert coordinator.used_tokens == 7 * 51000
    assert coordinator.session.stop_reason == "converged"
    assert len(result.test_cases) == 23
    assert coordinator.session.reviews[-1].assessments[0].case_version == 3


@pytest.mark.asyncio
async def test_expensive_initial_review_stops_before_unfunded_repair():
    operations = []

    async def send(request):
        operations.append(request.operation)
        if request.operation == "generate":
            output, spent = draft(), 100000
        elif request.operation == "review":
            output, spent = review([case()], [finding()]), 500000
        else:
            assert request.operation == "classify"
            output = ClassifiedTestCases.model_validate({"test_cases": [{
                "issue_key": "TC-1", "name": "case", "test_type": "API", "automation_capability": "manual",
                "labels": [], "tool_use_comment": "",
            }]})
            spent = 1000
        return StageResult(
            run_id=request.run_id, stage_id=request.stage_id, input_revision=request.input_revision,
            execution_id=str(len(operations)), status="completed", payload=output.model_dump(mode="json"),
            usage=StageUsage(input_tokens=spent),
        )

    coordinator = DesignCoordinator(send)
    with patch("config.DesignConfig.TOKEN_LIMIT", 1000000):
        await coordinator.design("Synthetic PRD", "S1")
    assert operations == ["generate", "review", "classify"]
    assert coordinator.used_tokens == 601000
    assert coordinator.session.stop_reason == "budget_handoff"
    assert coordinator.session.cases == [case()]
    assert coordinator.session.revision == 0


async def _published_feedback(design: DesignDraft, final_review: DesignReview, stop_reason: str) -> dict[str, list[str]]:
    """Run the pipeline with a fixed final review and return the per-case comments sent to the Bitable."""
    import orchestrator.main as module

    coordinator = MagicMock()
    coordinator.design = AsyncMock(return_value=(design, ClassifiedTestCases(test_cases=[])))
    old_finding = finding(finding_id="old").model_copy(update={"reason": "已经修好的历史问题"})
    coordinator.session = DesignSession(
        cases=design.test_cases, reviews=[review([case()], [old_finding]), final_review], stop_reason=stop_reason,
    )
    with (
        patch.object(module, "feishu_get_doc_content", AsyncMock(return_value="PRD")),
        patch.object(module, "DesignCoordinator", return_value=coordinator),
        patch.object(module, "_resolve_story_business_line", AsyncMock(return_value=None)),
        patch.object(module, "publish_bitable_report", return_value=BitablePublishResult(
            url="https://example/base/final", expected=2, written=2, accessible=True,
        )) as publish,
        patch.object(module, "_write_bitable_url_to_story", AsyncMock()) as confirm,
        patch.object(module, "notify_pipeline_success", AsyncMock()) as success,
        patch.object(module, "notify_pipeline_failure", AsyncMock()) as failure,
        patch("config.FEISHU_BITABLE_SYNC_ENABLED", True),
        patch("config.DesignConfig.OSS_ARCHIVE_ENABLED", False),
    ):
        await module._run_pipeline("S1", "P1", "https://example/doc")
    confirm.assert_awaited_once_with("https://example/base/final", "S1", "P1")
    assert success.await_count == 1
    failure.assert_not_awaited()
    return publish.call_args.args[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["resolved", "case", "suite_owner", "low"])
async def test_pipeline_comments_only_final_unresolved_findings(scope):
    design = draft()
    design.test_cases = [case(version=3), case(key="TC-2", version=2)]
    new_finding = finding(finding_id="new").model_copy(update={"reason": "最终仍需补充的断言"})
    remaining = scope != "resolved"
    final_review = review(design.test_cases)
    if scope == "case":
        final_review.assessments[0].findings = [new_finding]
    elif scope == "low":
        final_review.assessments[0].findings = [new_finding.model_copy(update={"severity": "low"})]
    elif scope == "suite_owner":
        final_review.suite.findings = [new_finding]
    feedback = await _published_feedback(
        design, final_review, "iteration_limit" if scope not in {"resolved", "low"} else "converged",
    )
    assert not feedback["TC-2"]
    assert "已经修好的历史问题" not in str(feedback)
    if remaining:
        assert len(feedback["TC-1"]) == 1
        assert "待人工评审" in feedback["TC-1"][0]
        assert "v3" in feedback["TC-1"][0]
        assert "最终仍需补充的断言" in feedback["TC-1"][0]
    else:
        assert not any(feedback.values())


@pytest.mark.asyncio
@pytest.mark.parametrize(("related", "requirements", "expected_owner"), [
    (["TC-3"], ["AC-1"], "TC-3"),
    ([], ["AC-2"], "TC-2"),
    ([], ["AC-9"], "TC-1"),
])
async def test_suite_gap_comment_is_placed_only_on_relevant_cases(related, requirements, expected_owner):
    design = draft()
    design.test_cases = [case(ids=["AC-1"]), case(key="TC-2", ids=["AC-2"]), case(key="TC-3", ids=["AC-3"])]
    final_review = review(design.test_cases)
    final_review.suite.findings = [finding(
        "add_test_case", owner=None, related_case_keys=related,
    ).model_copy(update={"requirement_ids": requirements, "reason": "缺少新流程用例"})]

    feedback = await _published_feedback(design, final_review, "iteration_limit")

    assert {key: len(lines) for key, lines in feedback.items() if lines} == {expected_owner: 1}
    line = feedback[expected_owner][0]
    assert line.startswith(f"整套评审（{', '.join(requirements)}）：")
    assert "缺少新流程用例" in line
    assert " v1" not in line
