# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Versioned contracts for deterministic QA design stages."""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from common.models import AcceptanceCriteriaList, TestCase

PROTOCOL = "qa-design/1"
SEVERITY = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def is_stage_message(text: str) -> bool:
    """Recognize an envelope without interpreting PRD prose as a protocol request."""
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(value, dict) and str(value.get("protocol", "")).startswith("qa-design/")


class KnowledgeSnapshot(BaseModel):
    """One knowledge-base response reused throughout a design."""

    status: Literal["ok", "no_match", "unavailable", "disabled"]
    content: str = ""


class DesignDraft(BaseModel):
    """Generated cases and their immutable grounding."""

    requirements: AcceptanceCriteriaList
    knowledge: KnowledgeSnapshot
    test_cases: list[TestCase]


class ReviewFinding(BaseModel):
    """A validated action proposed by a reviewer, never an executable tool."""

    model_config = ConfigDict(extra="forbid")
    finding_id: str = ""
    severity: Literal["low", "medium", "high", "critical"]
    action: Literal["modify", "add_test_case", "delete_test_case", "remove_duplicate_steps"]
    owner_case_key: str | None
    related_case_keys: list[str]
    requirement_ids: list[str]
    category: str
    reason: str
    expected_change: str
    restore_case_key: str | None = None

    @model_validator(mode="after")
    def _validate_owner(self) -> "ReviewFinding":
        if self.action != "add_test_case" and not self.owner_case_key:
            raise ValueError("An existing-case action requires an owner")
        if self.action == "add_test_case" and self.owner_case_key is not None:
            raise ValueError("A new-case finding must not have an owner")
        if self.restore_case_key and self.action != "add_test_case":
            raise ValueError("Only a coverage gap can restore a deleted case")
        if self.action == "delete_test_case" and not self.related_case_keys:
            raise ValueError("Deletion requires surviving cases that cover every unique scenario")
        return self

    def fingerprint(self) -> str:
        """Return a stable comparison key independent of generated finding IDs."""
        return "|".join(
            (
                self.action,
                self.owner_case_key or "",
                self.category,
                ",".join(sorted(self.requirement_ids)),
                ",".join(sorted(self.related_case_keys)),
                " ".join(self.reason.split()),
                " ".join(self.expected_change.split()),
            )
        )


class CaseAssessment(BaseModel):
    """Receipt proving that a particular case version was reviewed."""

    case_key: str
    case_version: int = Field(ge=1)
    findings: list[ReviewFinding]


class UntestableRequirement(BaseModel):
    """A reviewer's justified decision that a requirement item has no testable behaviour."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    requirement_id: str
    reason: str = Field(min_length=1)


class SuiteAssessment(BaseModel):
    """Explicit completion receipt for coverage and duplicate-coverage review."""

    completed: bool
    findings: list[ReviewFinding]
    untestable: list[UntestableRequirement] = Field(default_factory=list)


class DesignReview(BaseModel):
    """Per-case receipts plus a whole-suite assessment."""

    assessments: list[CaseAssessment]
    suite: SuiteAssessment

    def findings(self) -> list[ReviewFinding]:
        """Return all findings without dropping whole-suite gaps."""
        return [f for a in self.assessments for f in a.findings] + self.suite.findings


class FixResult(BaseModel):
    """Candidate changes; only the orchestrator can commit or delete cases."""

    replacements: list[TestCase]
    additions: list[TestCase]


class StageRequest(BaseModel):
    """Internal JSON envelope, validated before any model call."""

    model_config = ConfigDict(extra="forbid")
    protocol: Literal["qa-design/1"] = PROTOCOL
    run_id: str = Field(min_length=1, max_length=100)
    stage_id: str = Field(min_length=1, max_length=100)
    input_revision: int = Field(default=0, ge=0)
    operation: Literal["generate", "review", "fix", "classify", "requirements_review"]
    required_skill_id: str
    token_limit: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    payload: dict


class StageUsage(BaseModel):
    """Usage observed at the model boundary, including failed stages."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    requests: int = Field(default=0, ge=0)
    complete: bool = True

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class StageResult(BaseModel):
    """A bound result and usage receipt independent of successful model output."""

    protocol: Literal["qa-design/1"] = PROTOCOL
    run_id: str
    stage_id: str
    input_revision: int
    execution_id: str
    status: Literal["completed", "failed", "timed_out", "canceled"]
    payload: dict = Field(default_factory=dict)
    error: str | None = None
    usage: StageUsage
    error_code: str | None = None
    error_details: dict[str, int | str] = Field(default_factory=dict)
    trace: list[dict] = Field(default_factory=list)


class FocusReview(BaseModel):
    """An individual requirements-review perspective."""

    focus: str
    feedback: str


class RequirementsAssessment(BaseModel):
    """Final merged feedback and any perspectives that could not be assessed."""

    suggested_improvements: str
    completed_focuses: list[str]
    failed_focuses: list[str]
