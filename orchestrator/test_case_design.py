# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Deterministic, request-local generation/review/fix coordination."""

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from math import ceil
from uuid import uuid4

import config
from common import design_store
from common.design_models import (
    SEVERITY,
    DesignDraft,
    DesignReview,
    FixResult,
    ReviewFinding,
    StageRequest,
    StageResult,
)
from common.design_runtime import BudgetExceeded
from common.models import ClassifiedTestCases, TestCase
from common.provider_errors import ProviderRetryError

SKILLS = {
    "generate": "generate-test-cases",
    "review": "review-test-cases",
    "fix": "fix-test-cases",
    "classify": "classify-test-cases",
    "requirements_review": "review-requirements",
}


@dataclass(slots=True)
class DesignSession:
    """A story's mutable state; only complete repair rounds are committed."""

    run_id: str = field(default_factory=lambda: str(uuid4()))
    revision: int = 0
    next_key: int = 1
    cases: list[TestCase] = field(default_factory=list)
    deleted: dict[str, TestCase] = field(default_factory=dict)
    reviews: list[DesignReview] = field(default_factory=list)
    stop_reason: str = ""

    def assign(self, case: TestCase, story_id: str) -> TestCase:
        """Allocate a run-local key once, without ever reusing deleted keys."""
        key = f"TC-{self.next_key}"
        self.next_key += 1
        return case.model_copy(update={"key": key, "parent_issue_key": story_id, "version": 1})


def validate_cases(cases: list[TestCase], draft: DesignDraft) -> None:
    """Reject empty sets, duplicate identities and fabricated requirement references."""
    keys = [c.key for c in cases]
    requirements = {r.id for r in draft.requirements.items}
    if not cases or None in keys or len(keys) != len(set(keys)):
        raise ValueError("The active design must contain uniquely identified cases")
    for case in cases:
        if not case.steps or any(not s.action.strip() or not s.expected_results.strip() for s in case.steps):
            raise ValueError("A case must contain executable steps and observable expected results")
        if not case.covered_requirement_ids or not set(case.covered_requirement_ids) <= requirements:
            raise ValueError("A case references missing or unknown requirements")


def validate_review(review: DesignReview, session: DesignSession, draft: DesignDraft) -> None:
    """Require complete receipts and a coherent, non-destructive repair plan."""
    expected = {(c.key, c.version) for c in session.cases}
    actual = [(a.case_key, a.case_version) for a in review.assessments]
    if not review.suite.completed or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Review omitted, duplicated, or used a stale case version")
    keys = {c.key for c in session.cases}
    requirements = {r.id for r in draft.requirements.items}
    for receipt in review.assessments:
        if any(f.owner_case_key != receipt.case_key for f in receipt.findings):
            raise ValueError("A case assessment must own its findings")
    findings = review.findings()
    deleted = {f.owner_case_key for f in findings if f.action == "delete_test_case"}
    if keys and keys <= deleted:
        raise ValueError("Review would delete every case")
    modifying = {f.owner_case_key for f in findings if f.action in {"modify", "remove_duplicate_steps"}}
    if deleted & modifying:
        raise ValueError("A case cannot be modified and deleted in the same review")
    if len(deleted) != sum(f.action == "delete_test_case" for f in findings):
        raise ValueError("Duplicate deletion findings")
    restores = [f.restore_case_key for f in findings if f.restore_case_key]
    if len(restores) != len(set(restores)):
        raise ValueError("Duplicate restoration findings")
    removals = {f.owner_case_key: set(f.related_case_keys) for f in findings if f.action == "remove_duplicate_steps"}
    for finding in findings:
        if not set(finding.requirement_ids) <= requirements:
            raise ValueError("Review references an unknown requirement")
        if finding.owner_case_key and finding.owner_case_key not in keys:
            raise ValueError("Review references an unknown owner")
        if not set(finding.related_case_keys) <= keys:
            raise ValueError("Review references an unknown related case")
        if finding.action == "delete_test_case" and (
            set(finding.related_case_keys) & deleted or finding.owner_case_key in finding.related_case_keys
        ):
            raise ValueError("A deleted case cannot be the surviving coverage owner")
        if finding.action == "remove_duplicate_steps" and (
            finding.owner_case_key in deleted
            or any(finding.owner_case_key in removals.get(k, set()) for k in finding.related_case_keys)
        ):
            raise ValueError("Review would remove shared steps from both sides")
        if finding.restore_case_key and finding.restore_case_key not in session.deleted:
            raise ValueError("Review requests restoration of an unknown deleted case")
        if finding.action == "add_test_case" and not finding.requirement_ids:
            raise ValueError("A coverage gap must identify its requirements")
    waived = {w.requirement_id for w in review.suite.untestable}
    if not waived <= requirements:
        raise ValueError("Review waived an unknown requirement")
    if waived & {ref for finding in findings for ref in finding.requirement_ids}:
        raise ValueError("Review waived a requirement it raised a finding on")


def commit_repair(
    session: DesignSession,
    draft: DesignDraft,
    findings: list[ReviewFinding],
    fixed: FixResult,
    story_id: str,
) -> None:
    """Validate a complete candidate before changing any live session state."""
    current = {c.key: c for c in session.cases}
    required = {f.owner_case_key for f in findings if f.action in {"modify", "remove_duplicate_steps"}}
    replacements = {c.key: c for c in fixed.replacements}
    if len(replacements) != len(fixed.replacements) or set(replacements) != required:
        raise ValueError("Repair did not return exactly the requested replacements")
    for key, candidate in replacements.items():
        if candidate.version != current[key].version + 1:
            raise ValueError("Repair returned a stale case version")
        if not set(current[key].covered_requirement_ids) <= set(candidate.covered_requirement_ids):
            raise ValueError("Repair loses original requirement coverage")
        current[key] = candidate.model_copy(update={"parent_issue_key": story_id})
    deleted = dict(session.deleted)
    for finding in findings:
        if finding.restore_case_key:
            case = deleted.pop(finding.restore_case_key)
            if not set(finding.requirement_ids) <= set(case.covered_requirement_ids):
                raise ValueError("Restored case does not cover its assigned finding requirements")
            current[case.key] = case.model_copy(update={"version": case.version + 1})
    deleting = {f.owner_case_key for f in findings if f.action == "delete_test_case"}
    for finding in findings:
        if finding.action == "delete_test_case":
            owner = current[finding.owner_case_key]
            survivors = [current[k] for k in finding.related_case_keys if k not in deleting]
            covered = {r for c in survivors for r in c.covered_requirement_ids}
            if not set(owner.covered_requirement_ids) <= covered:
                raise ValueError("Deletion loses requirement coverage")
            # Requirement IDs alone cannot prove boundary/error scenarios survive.
            retained_steps = {
                step.model_dump_json() for c in survivors if c.preconditions == owner.preconditions for step in c.steps
            }
            if not {step.model_dump_json() for step in owner.steps} <= retained_steps:
                raise ValueError("Deletion cannot prove preservation of every observable scenario")
            deleted[owner.key] = current.pop(owner.key)
    expected_additions = sum(f.action == "add_test_case" and not f.restore_case_key for f in findings)
    if len(fixed.additions) != expected_additions or expected_additions > config.DesignConfig.MAX_NEW_CASES_PER_ROUND:
        raise ValueError("Repair returned an unexpected number of new cases")
    next_key = session.next_key
    gaps = [f for f in findings if f.action == "add_test_case" and not f.restore_case_key]
    for finding, candidate in zip(gaps, fixed.additions, strict=True):
        if not set(finding.requirement_ids) <= set(candidate.covered_requirement_ids):
            raise ValueError("A new case does not cover its assigned finding requirements")
        key = f"TC-{next_key}"
        next_key += 1
        current[key] = candidate.model_copy(update={"key": key, "version": 1, "parent_issue_key": story_id})
    validate_cases(list(current.values()), draft)
    session.cases = list(current.values())
    session.deleted = deleted
    session.next_key = next_key
    session.revision += 1


class DesignCoordinator:
    """Keep total time, usage and stop decisions outside model control."""

    def __init__(self, send: Callable[[StageRequest], Awaitable[StageResult]], run: dict | None = None) -> None:
        self.send = send
        self.run = run
        self.deadline = time.monotonic() + (run["deadline"] - time.time() if run else config.DesignConfig.TIMEOUT_SECONDS)
        self.used_tokens = 0
        self.executions: set[str] = set()
        self.session = DesignSession()
        if run:
            self.session.run_id = run["id"]
            self.used_tokens = run["used"]
        self.review_reserve = 0
        self.classification_reserve = 0
        self.last_review_tokens = 0

    async def stage(self, operation: str, payload: dict) -> dict:
        """Dispatch one bound stage and account for its server-collected usage."""
        key = f"{operation}-{self.session.revision}"
        fingerprint = design_store.digest(payload)
        owner = design_store.store if self.run else None
        if owner:
            saved = await owner.call("get", self.run["id"], "stage", key, fingerprint)
            if saved:
                if operation == "review":
                    self.last_review_tokens = await owner.call("operation_usage", self.run["id"], key)
                return saved["value"]
            actual = await owner.call("run", self.run["id"], self.run["generation"])
            self.used_tokens = actual["used"]
            if actual["usage_unknown"]:
                raise design_store.RecoveryBlocked("usage_unknown")
        remaining = self.deadline - time.monotonic()
        tokens = (self.run["token_limit"] if self.run else config.DesignConfig.TOKEN_LIMIT) - self.used_tokens
        if remaining <= 0 or tokens <= 0:
            raise RuntimeError("Design budget or deadline exhausted")
        reserve = 0
        if operation == "fix":
            reserve = self.review_reserve + self.classification_reserve
        elif operation == "review":
            reserve = self.classification_reserve
        if tokens <= reserve:
            raise BudgetExceeded("budget_closure_reserved", remaining_tokens=tokens,
                                 closure_reserve=reserve, stage=operation)
        request = StageRequest(
            run_id=self.session.run_id,
            stage_id=key + (f"-g{self.run['generation']}" if self.run else ""),
            input_revision=self.session.revision,
            operation=operation,
            required_skill_id=SKILLS[operation],
            token_limit=tokens - reserve,
            timeout_seconds=remaining,
            payload=payload,
        )
        async with asyncio.timeout(remaining):
            result = await self.send(request)
        if (result.run_id, result.stage_id, result.input_revision) != (
            request.run_id,
            request.stage_id,
            request.input_revision,
        ):
            raise ValueError("QA stage result does not match the current request")
        if owner:
            self.used_tokens = (await owner.call("run", self.run["id"]))["used"]
        elif result.execution_id not in self.executions:
            self.executions.add(result.execution_id)
            self.used_tokens += result.usage.total_tokens
        if result.error_code in {"provider_busy_exhausted", "provider_retry_deadline", "provider_usage_unknown",
                                 "provider_stream_incomplete"}:
            raise ProviderRetryError(result.error_code, **result.error_details)
        if not result.usage.complete or self.used_tokens > (self.run["token_limit"] if self.run else config.DesignConfig.TOKEN_LIMIT):
            raise RuntimeError("QA usage is incomplete or exhausted; refusing to continue")
        if operation == "review":
            self.last_review_tokens = (await owner.call("operation_usage", self.run["id"], key)
                                       if owner else result.usage.total_tokens)
        if result.status != "completed":
            if result.error_code:
                raise BudgetExceeded(result.error_code, **result.error_details)
            raise RuntimeError(result.error or f"QA stage {result.status}")
        if owner:
            await owner.call("put", self.run["id"], self.run["generation"], "stage", key, fingerprint, result.payload)
        return result.payload

    async def _save_revision(self) -> None:
        if not self.run:
            return
        snapshot = {
            "cases": [case.model_dump(mode="json") for case in self.session.cases],
            "deleted": {key: case.model_dump(mode="json") for key, case in self.session.deleted.items()},
            "next_key": self.session.next_key,
        }
        await design_store.store.call("put", self.run["id"], self.run["generation"], "revision",
                                      str(self.session.revision), design_store.digest(snapshot), snapshot)

    async def design(self, document: str, story_id: str) -> tuple[DesignDraft, ClassifiedTestCases]:
        """Generate, review and repair, always ending the loop with a review."""
        if config.DesignConfig.MAX_REVIEWS < 1 or config.DesignConfig.FIX_MIN_SEVERITY not in SEVERITY:
            raise ValueError("Invalid design iteration/severity settings")
        draft = DesignDraft.model_validate(await self.stage("generate", {"document": document}))
        if draft.knowledge.status not in {"ok", "no_match"}:
            raise ValueError("Design is missing its required knowledge snapshot")
        if len({r.id for r in draft.requirements.items}) != len(draft.requirements.items):
            raise ValueError("Duplicate requirement IDs")
        self.session.cases = [self.session.assign(c, story_id) for c in draft.test_cases]
        validate_cases(self.session.cases, draft)
        await self._save_revision()
        previous_signature = None
        for iteration in range(config.DesignConfig.MAX_REVIEWS):
            payload = self._payload(document, draft)
            # Classification is batched; retain input bytes and each batch's output ceiling.
            cases_json = json.dumps(payload["test_cases"], ensure_ascii=False, separators=(",", ":"))
            batches = (len(self.session.cases) + config.TestCaseClassificationAgentConfig.BATCH_SIZE - 1) // config.TestCaseClassificationAgentConfig.BATCH_SIZE
            self.classification_reserve = len(cases_json.encode()) + batches * (
                config.TestCaseClassificationAgentConfig.MAX_TOKENS + 4096
            )
            review = DesignReview.model_validate(await self.stage("review", payload))
            for index, finding in enumerate(review.findings(), 1):
                finding.finding_id = f"R{self.session.revision}-F{index}"
            validate_review(review, self.session, draft)
            self.session.reviews.append(review)
            blocking = [
                f for f in review.findings() if SEVERITY[f.severity] >= SEVERITY[config.DesignConfig.FIX_MIN_SEVERITY]
            ]
            coverage = {r for c in self.session.cases for r in c.covered_requirement_ids}
            # Every requirement item is QA scope; only a justified suite-review waiver may leave one uncovered.
            waived = {w.requirement_id for w in review.suite.untestable}
            missing = {r.id for r in draft.requirements.items} - coverage
            if missing and not missing <= {r for f in blocking for r in f.requirement_ids} | waived:
                raise ValueError("Suite review silently omitted uncovered requirements")
            signature = (
                tuple(sorted(f.fingerprint() for f in blocking)),
                tuple(sorted(waived)),
                hashlib.sha256(
                    json.dumps([c.model_dump(exclude={"version"}) for c in self.session.cases], sort_keys=True).encode()
                ).hexdigest(),
            )
            if not blocking and not missing - waived:
                self.session.stop_reason = "converged"
                break
            if iteration + 1 >= config.DesignConfig.MAX_REVIEWS:
                self.session.stop_reason = "iteration_limit"
                break
            if signature == previous_signature:
                self.session.stop_reason = "no_progress"
                break
            previous_signature = signature
            payload["findings"] = [f.model_dump(mode="json") for f in blocking]
            # Last full review plus 20% headroom is a planning reserve, not a usage guarantee.
            additions = sum(f.action == "add_test_case" for f in blocking)
            future_count = len(self.session.cases) + additions
            self.review_reserve = ceil(
                self.last_review_tokens * 1.2 * (future_count + 1) / (len(self.session.cases) + 1)
            )
            future_batches = ceil(future_count / config.TestCaseClassificationAgentConfig.BATCH_SIZE)
            self.classification_reserve = ceil(len(cases_json.encode()) * future_count / len(self.session.cases)) + future_batches * (
                config.TestCaseClassificationAgentConfig.MAX_TOKENS + 4096
            )
            try:
                fixed = FixResult.model_validate(await self.stage("fix", payload))
            except BudgetExceeded as exc:
                if exc.code not in {"budget_closure_reserved", "budget_preflight_exceeded"}:
                    raise
                # Only refusal before any fix call may become a normal handoff.
                if exc.details.get("completed_requests", 0):
                    raise
                self.session.stop_reason = "budget_handoff"
                break
            candidate_session = DesignSession(
                run_id=self.session.run_id, revision=self.session.revision, next_key=self.session.next_key,
                cases=list(self.session.cases), deleted=dict(self.session.deleted), reviews=list(self.session.reviews),
            )
            commit_repair(candidate_session, draft, blocking, fixed, story_id)
            original = self.session
            self.session = candidate_session
            try:
                await self._save_revision()
            except BaseException:
                self.session = original
                raise
        classified = ClassifiedTestCases.model_validate(
            await self.stage("classify", {"test_cases": [c.model_dump(mode="json") for c in self.session.cases]})
        )
        keys = [c.issue_key for c in classified.test_cases]
        if len(keys) != len(self.session.cases) or set(keys) != {c.key for c in self.session.cases}:
            raise ValueError("Classification does not match the final case set")
        draft.test_cases = self.session.cases
        return draft, classified

    def _payload(self, document: str, draft: DesignDraft) -> dict:
        return {
            "document": document,
            "requirements": draft.requirements.model_dump(mode="json"),
            "knowledge": draft.knowledge.model_dump(mode="json"),
            "test_cases": [c.model_dump(mode="json") for c in self.session.cases],
            "deleted": {k: c.model_dump(mode="json") for k, c in self.session.deleted.items()},
            "previous_review": self.session.reviews[-1].model_dump(mode="json") if self.session.reviews else {},
        }
