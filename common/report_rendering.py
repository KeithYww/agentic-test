# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Merges generation/classification/review results into one report, keyed by test case ID."""

from common.models import ClassifiedTestCases, GeneratedTestCases, TestCaseReviewFeedbacks


def merge_test_case_results(
    generated: GeneratedTestCases, classified: ClassifiedTestCases, reviewed: TestCaseReviewFeedbacks
) -> list[dict]:
    """Merges generated test cases with their classification and review feedback by test case key."""
    classified_by_key = {tc.issue_key: tc for tc in classified.test_cases}
    review_by_key = {fb.test_case_id: fb for fb in reviewed.review_feedbacks}

    merged = []
    for test_case in generated.test_cases:
        classification = classified_by_key.get(test_case.key)
        review = review_by_key.get(test_case.key)
        merged.append(
            {
                "test_case": test_case.model_dump(),
                "classification": classification.model_dump() if classification else None,
                "review_feedback": review.review_feedback if review else [],
            }
        )
    return merged


def render_markdown_report(merged: list[dict], title: str = "Test Case Pipeline Results") -> str:
    """Renders merged test case results as a Markdown report."""
    lines = [f"# {title}", ""]
    for entry in merged:
        test_case = entry["test_case"]
        classification = entry["classification"]
        lines.append(f"## {test_case['name']} ({test_case['key']})")
        lines.append("")
        lines.append(f"**Summary:** {test_case['summary']}")
        if test_case.get("preconditions"):
            lines.append("")
            lines.append(f"**Preconditions:** {test_case['preconditions']}")
        lines.append("")
        lines.append("**Steps:**")
        for i, step in enumerate(test_case["steps"], start=1):
            lines.append(f"{i}. {step['action']} -> {step['expected_results']}")
            if step["test_data"]:
                lines.append(f"   - Test data: {', '.join(step['test_data'])}")
        lines.append("")
        if classification:
            lines.append(
                f"**Classification:** type={classification['test_type']}, "
                f"automation={classification['automation_capability']}, labels={', '.join(classification['labels'])}"
            )
            lines.append("")
        if entry["review_feedback"]:
            lines.append("**Review feedback:**")
            for feedback in entry["review_feedback"]:
                lines.append(f"- {feedback}")
            lines.append("")
    return "\n".join(lines)


# Feishu DocxXML colors are limited to a fixed set of hues; these are the three legal values of
# ClassifiedTestCase.automation_capability (common/models.py), used to color-code that column/span.
_AUTOMATION_COLOR = {"automated": "green", "semi-automated": "yellow", "manual": "gray"}


def _escape_xml_text(text: str) -> str:
    """Escapes free text for inclusion in Feishu DocxXML tag content (see
    lark-doc-xml.md's escaping rules) — unlike Markdown, invalid XML fails the whole
    docs +create call, so every piece of LLM-generated text must go through this first.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")


def _render_steps_table_xml(steps: list[dict]) -> str:
    header = (
        "<table><thead><tr>"
        '<th background-color="light-gray"><p>#</p></th>'
        '<th background-color="light-gray"><p>操作</p></th>'
        '<th background-color="light-gray"><p>测试数据</p></th>'
        '<th background-color="light-gray"><p>预期结果</p></th>'
        "</tr></thead><tbody>"
    )
    rows = []
    for i, step in enumerate(steps, start=1):
        test_data = ", ".join(step["test_data"]) if step["test_data"] else "-"
        rows.append(
            "<tr>"
            f"<td><p>{i}</p></td>"
            f"<td><p>{_escape_xml_text(step['action'])}</p></td>"
            f"<td><p>{_escape_xml_text(test_data)}</p></td>"
            f"<td><p>{_escape_xml_text(step['expected_results'])}</p></td>"
            "</tr>"
        )
    return header + "".join(rows) + "</tbody></table>"


def _render_overview_table_xml(merged: list[dict]) -> str:
    header = (
        "<table><thead><tr>"
        '<th background-color="light-gray"><p>用例</p></th>'
        '<th background-color="light-gray"><p>类型</p></th>'
        '<th background-color="light-gray"><p>自动化</p></th>'
        '<th background-color="light-gray"><p>标签</p></th>'
        '<th background-color="light-gray"><p>Review 条数</p></th>'
        "</tr></thead><tbody>"
    )
    rows = []
    for entry in merged:
        test_case = entry["test_case"]
        classification = entry["classification"]
        name_cell = f"{_escape_xml_text(test_case['name'])} ({_escape_xml_text(test_case['key'] or '')})"
        if classification:
            test_type = _escape_xml_text(classification["test_type"])
            automation = classification["automation_capability"]
            color = _AUTOMATION_COLOR.get(automation, "gray")
            automation_cell = f'<td background-color="light-{color}"><p>{_escape_xml_text(automation)}</p></td>'
            labels_cell = _escape_xml_text(", ".join(classification["labels"])) or "-"
        else:
            test_type = "-"
            automation_cell = "<td><p>-</p></td>"
            labels_cell = "-"
        rows.append(
            "<tr>"
            f"<td><p>{name_cell}</p></td>"
            f"<td><p>{test_type}</p></td>"
            f"{automation_cell}"
            f"<td><p>{labels_cell}</p></td>"
            f"<td><p>{len(entry['review_feedback'])}</p></td>"
            "</tr>"
        )
    return header + "".join(rows) + "</tbody></table>"


def _render_test_case_section_xml(entry: dict) -> str:
    test_case = entry["test_case"]
    classification = entry["classification"]
    review_feedback = entry["review_feedback"]

    name_and_key = f"{_escape_xml_text(test_case['name'])} ({_escape_xml_text(test_case['key'] or '')})"
    parts = [f'<h2 seq="auto">{name_and_key}</h2>', f"<p><b>Summary:</b> {_escape_xml_text(test_case['summary'])}</p>"]
    if test_case.get("preconditions"):
        parts.append(f"<p><b>Preconditions:</b> {_escape_xml_text(test_case['preconditions'])}</p>")
    parts.append(_render_steps_table_xml(test_case["steps"]))

    if classification:
        color = _AUTOMATION_COLOR.get(classification["automation_capability"], "gray")
        labels = _escape_xml_text(", ".join(classification["labels"]))
        parts.append(
            "<p><b>分类:</b> "
            f"{_escape_xml_text(classification['test_type'])} · "
            f'<span background-color="light-{color}">{_escape_xml_text(classification["automation_capability"])}</span>'
            f" · 标签: {labels}</p>"
        )

    if review_feedback:
        items = "".join(f"<li>{_escape_xml_text(feedback)}</li>" for feedback in review_feedback)
        parts.append(f'<callout emoji="📝" background-color="light-orange"><ul>{items}</ul></callout>')

    parts.append("<hr/>")
    return "".join(parts)


def render_report_xml(merged: list[dict]) -> str:
    """Renders merged test case results as Feishu DocxXML: an overview table followed by one
    section per test case (steps table, colored classification, review feedback callout).

    Takes no title: feishu_doc_service.create_report_doc's --title flag already has the CLI
    prepend a <title> tag; a second one in the content would conflict with it.
    """
    parts = ["<h1>用例总览</h1>", _render_overview_table_xml(merged), "<hr/>", "<h1>用例详情</h1>"]
    parts.extend(_render_test_case_section_xml(entry) for entry in merged)
    return "".join(parts)
