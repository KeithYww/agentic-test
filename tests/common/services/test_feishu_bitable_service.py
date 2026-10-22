# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

import json
from unittest.mock import MagicMock, patch

from common.models import TestCase, TestStep
from common.services.feishu_bitable_service import create_bitable_report

_BASE_CREATE_RESPONSE = json.dumps({
    "ok": True,
    "data": {
        "base": {"base_token": "BT1", "url": "https://example.feishu.cn/base/BT1"},
        "table": {"id": "tbl1", "views": [{"id": "vew1"}]},
    },
})


def _make_test_case(name: str) -> TestCase:
    return TestCase(
        key="TC-1",
        labels=[],
        name=name,
        summary="s",
        comment="",
        preconditions=None,
        steps=[TestStep(action="do it", expected_results="done", test_data=[])],
        parent_issue_key=None,
    )


@patch("subprocess.run")
def test_batch_write_passes_json_via_file_not_argv(mock_run):
    """The batch-create call must reference the payload via an @-file, not inline JSON — inline
    JSON for a full batch can exceed the OS argv length limit (Errno 7)."""
    mock_run.side_effect = [
        MagicMock(returncode=0, stdout=_BASE_CREATE_RESPONSE, stderr=""),
        MagicMock(
            returncode=0,
            stdout=json.dumps({"ok": True, "data": {"record_id_list": ["rec1"]}}),
            stderr="",
        ),
        MagicMock(returncode=0, stdout=json.dumps({"ok": True}), stderr=""),  # tenant_editable
    ]

    create_bitable_report("story-1", "Story Title", None, [_make_test_case("TC1")], {}, {})

    batch_create_call = mock_run.call_args_list[1]
    args = batch_create_call.args[0]
    json_value = args[args.index("--json") + 1]
    assert json_value.startswith("@./"), f"expected an @-file reference, got: {json_value}"
    assert batch_create_call.kwargs.get("cwd"), "lark-cli must be run with a cwd so the @-file path resolves"


@patch("subprocess.run")
def test_zero_records_written_is_treated_as_failure(mock_run):
    """If every batch write fails, the Bitable is empty — this must not be reported as success."""
    mock_run.side_effect = [
        MagicMock(returncode=0, stdout=_BASE_CREATE_RESPONSE, stderr=""),
        MagicMock(returncode=0, stdout=json.dumps({"ok": False, "error": "boom"}), stderr=""),
    ]

    result = create_bitable_report("story-1", "Story Title", None, [_make_test_case("TC1")], {}, {})

    assert result is None


@patch("subprocess.run")
def test_successful_write_returns_bitable_url(mock_run):
    mock_run.side_effect = [
        MagicMock(returncode=0, stdout=_BASE_CREATE_RESPONSE, stderr=""),
        MagicMock(
            returncode=0,
            stdout=json.dumps({"ok": True, "data": {"record_id_list": ["rec1"]}}),
            stderr="",
        ),
        MagicMock(returncode=0, stdout=json.dumps({"ok": True}), stderr=""),  # tenant_editable
    ]

    result = create_bitable_report("story-1", "Story Title", None, [_make_test_case("TC1")], {}, {})

    assert result == "https://example.feishu.cn/base/BT1"


@patch("subprocess.run")
def test_duplicate_record_ids_cannot_satisfy_publish_gate(mock_run):
    from common.services.feishu_bitable_service import publish_bitable_report
    mock_run.side_effect = [
        MagicMock(returncode=0, stdout=_BASE_CREATE_RESPONSE, stderr=""),
        MagicMock(returncode=0, stdout=json.dumps({"ok": True, "data": {"record_id_list": ["same", "same"]}}), stderr=""),
        MagicMock(returncode=0, stdout=json.dumps({"ok": True}), stderr=""),
    ]
    result = publish_bitable_report("S", "title", None, [_make_test_case("one"), _make_test_case("two")], {}, {})
    assert result.url
    assert not result.records_complete
    assert not result.complete


@patch("subprocess.run")
def test_review_comment_failure_preserves_url_but_blocks_completion(mock_run):
    from common.services.feishu_bitable_service import publish_bitable_report
    mock_run.side_effect = [
        MagicMock(returncode=0, stdout=_BASE_CREATE_RESPONSE, stderr=""),
        MagicMock(returncode=0, stdout=json.dumps({"ok": True, "data": {"record_id_list": ["rec1"]}}), stderr=""),
        MagicMock(returncode=1, stdout="", stderr="comment denied"),
        MagicMock(returncode=0, stdout=json.dumps({"ok": True}), stderr=""),
    ]
    result = publish_bitable_report("S", "title", None, [_make_test_case("one")], {}, {"TC-1": ["review"]})
    assert result.url
    assert not result.comments_complete
    assert not result.complete


@patch("subprocess.run")
def test_access_permission_failure_blocks_completion(mock_run):
    from common.services.feishu_bitable_service import publish_bitable_report
    mock_run.side_effect = [
        MagicMock(returncode=0, stdout=_BASE_CREATE_RESPONSE, stderr=""),
        MagicMock(returncode=0, stdout=json.dumps({"ok": True, "data": {"record_id_list": ["rec1"]}}), stderr=""),
        MagicMock(returncode=1, stdout="", stderr="permission denied"),
    ]
    result = publish_bitable_report("S", "title", None, [_make_test_case("one")], {}, {})
    assert result.url
    assert not result.accessible
    assert not result.complete


@patch("subprocess.run")
def test_final_review_status_and_comments_follow_each_case(mock_run):
    from pathlib import Path

    from common.services.feishu_bitable_service import publish_bitable_report

    rows = []
    comments = []

    def run_cli(args, **kwargs):
        if args[1:3] == ["base", "+base-create"]:
            return MagicMock(returncode=0, stdout=_BASE_CREATE_RESPONSE, stderr="")
        if args[1:3] == ["base", "+record-batch-create"]:
            path = args[args.index("--json") + 1][1:]
            payload = json.loads((Path(kwargs["cwd"]) / path).read_text())
            rows.extend(payload["create_records"])
            return MagicMock(returncode=0, stdout=json.dumps({"ok": True, "data": {
                "record_id_list": ["rec1", "rec2", "rec3"],
            }}), stderr="")
        if args[1:3] == ["drive", "+add-comment"]:
            comments.append(args)
        return MagicMock(returncode=0, stdout=json.dumps({"ok": True}), stderr="")

    mock_run.side_effect = run_cli
    cases = [_make_test_case(name).model_copy(update={"key": key, "version": version})
             for name, key, version in [("pending", "TC-1", 3), ("fixed", "TC-2", 2), ("clean", "TC-3", 1)]]
    result = publish_bitable_report(
        "S", "[待人工评审] story", None, cases, {},
        {"TC-1": ["TC-1 v3：[待人工评审] 最终意见"], "TC-2": [], "TC-3": []},
    )
    assert result.complete
    assert [row["评审状态"] for row in rows] == [
        "待人工评审（有遗留意见）", "自动评审无遗留意见", "自动评审无遗留意见",
    ]
    assert len(comments) == 1
    assert comments[0][comments[0].index("--block-id") + 1].split("!")[1] == "rec1"
