# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Build a bounded repair view without discarding authoritative PRD or KB evidence."""

import json
from dataclasses import dataclass

from common.design_models import ReviewFinding
from common.models import AcceptanceCriteriaList, TestCase


@dataclass(slots=True)
class RepairContext:
    """Evidence and validation scope belonging to exactly one model call."""

    prompt: str
    requirements: AcceptanceCriteriaList
    required_ids: frozenset[str]
    owner: TestCase | None = None


def build_repair_context(
    payload: dict, findings: list[ReviewFinding], owner: TestCase | None = None,
) -> RepairContext:
    """Keep complete sources and only the case bodies needed by this repair."""
    cases = {case["key"]: case for case in payload["test_cases"]}
    related_keys = {key for finding in findings for key in finding.related_case_keys}
    if owner:
        related_keys.discard(owner.key)
    related = [cases[key] for key in sorted(related_keys)]
    required = set(owner.covered_requirement_ids if owner else [])
    if owner is None:
        required.update(ref for finding in findings for ref in finding.requirement_ids)
    selected_ids = required | {ref for finding in findings for ref in finding.requirement_ids}
    selected_ids.update(ref for case in related for ref in case["covered_requirement_ids"])
    requirements = AcceptanceCriteriaList.model_validate(payload["requirements"])
    known = {item.id for item in requirements.items}
    if not selected_ids or not selected_ids <= known:
        raise ValueError("Repair evidence references missing requirements")
    selected = AcceptanceCriteriaList(items=[item for item in requirements.items if item.id in selected_ids])
    content = {
        "instruction": "只返回一个完整用例；修正给定问题，保留原场景与需求覆盖。索引仅用于导航，不是需求依据。",
        "owner": owner.model_dump(mode="json") if owner else None,
        "findings": [finding.model_dump(mode="json") for finding in findings],
        "related_cases": related,
        "case_index": [
            {key: case[key] for key in ("key", "version", "name", "covered_requirement_ids")}
            for case in cases.values()
        ],
        "document": payload["document"],
        "requirements": selected.model_dump(mode="json"),
        "knowledge": payload["knowledge"],
    }
    return RepairContext(
        prompt=json.dumps(content, ensure_ascii=False, separators=(",", ":")),
        requirements=selected,
        required_ids=frozenset(required),
        owner=owner,
    )
