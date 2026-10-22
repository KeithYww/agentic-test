# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

from common.models import (
    ClassifiedTestCase,
    ClassifiedTestCases,
    GeneratedTestCases,
    TestCase,
    TestCaseReviewFeedback,
    TestCaseReviewFeedbacks,
    TestStep,
)
from common.report_rendering import merge_test_case_results, render_markdown_report, render_report_xml


def _make_test_case(key: str) -> TestCase:
    return TestCase(
        key=key,
        labels=[],
        name=f"Test case {key}",
        summary="A summary",
        comment="",
        preconditions="Preconditions text",
        steps=[TestStep(action="Do something", expected_results="Something happens", test_data=[])],
        parent_issue_key=None,
    )


def test_merge_test_case_results_aligns_by_key():
    generated = GeneratedTestCases(test_cases=[_make_test_case("TC-1"), _make_test_case("TC-2")])
    classified = ClassifiedTestCases(
        test_cases=[
            ClassifiedTestCase(
                issue_key="TC-1", name="Test case TC-1", test_type="UI",
                automation_capability="automated", labels=["ui"], tool_use_comment="",
            )
        ]
    )
    reviewed = TestCaseReviewFeedbacks(
        review_feedbacks=[TestCaseReviewFeedback(test_case_id="TC-2", review_feedback=["Add negative case"])]
    )

    merged = merge_test_case_results(generated, classified, reviewed)

    assert len(merged) == 2
    assert merged[0]["classification"]["test_type"] == "UI"
    assert merged[0]["review_feedback"] == []
    assert merged[1]["classification"] is None
    assert merged[1]["review_feedback"] == ["Add negative case"]


def test_render_markdown_report_includes_all_sections():
    generated = GeneratedTestCases(test_cases=[_make_test_case("TC-1")])
    classified = ClassifiedTestCases(
        test_cases=[
            ClassifiedTestCase(
                issue_key="TC-1", name="Test case TC-1", test_type="API",
                automation_capability="manual", labels=["api"], tool_use_comment="",
            )
        ]
    )
    reviewed = TestCaseReviewFeedbacks(
        review_feedbacks=[TestCaseReviewFeedback(test_case_id="TC-1", review_feedback=["Looks good"])]
    )
    merged = merge_test_case_results(generated, classified, reviewed)

    report = render_markdown_report(merged, title="My Report")

    assert report.startswith("# My Report")
    assert "Test case TC-1" in report
    assert "type=API" in report
    assert "Looks good" in report


def test_render_report_xml_includes_table_span_and_callout():
    generated = GeneratedTestCases(test_cases=[_make_test_case("TC-1")])
    classified = ClassifiedTestCases(
        test_cases=[
            ClassifiedTestCase(
                issue_key="TC-1", name="Test case TC-1", test_type="API",
                automation_capability="automated", labels=["api"], tool_use_comment="",
            )
        ]
    )
    reviewed = TestCaseReviewFeedbacks(
        review_feedbacks=[TestCaseReviewFeedback(test_case_id="TC-1", review_feedback=["Looks good"])]
    )
    merged = merge_test_case_results(generated, classified, reviewed)

    report = render_report_xml(merged)

    assert "<table>" in report
    assert "Test case TC-1" in report
    assert 'background-color="light-green"' in report  # automated -> green
    assert "<callout" in report
    assert "Looks good" in report


def test_render_report_xml_handles_missing_classification_and_review():
    generated = GeneratedTestCases(test_cases=[_make_test_case("TC-1")])
    classified = ClassifiedTestCases(test_cases=[])
    reviewed = TestCaseReviewFeedbacks(review_feedbacks=[])
    merged = merge_test_case_results(generated, classified, reviewed)

    report = render_report_xml(merged)

    assert "<callout" not in report
    assert "Test case TC-1" in report


def test_render_report_xml_escapes_special_characters():
    test_case = _make_test_case("TC-1")
    test_case.summary = "A <script> & \"quote\"\nnew line"
    generated = GeneratedTestCases(test_cases=[test_case])
    classified = ClassifiedTestCases(test_cases=[])
    reviewed = TestCaseReviewFeedbacks(review_feedbacks=[])
    merged = merge_test_case_results(generated, classified, reviewed)

    report = render_report_xml(merged)

    assert "&lt;script&gt;" in report
    assert "&amp;" in report
    assert "<br/>" in report
    assert "<script>" not in report
