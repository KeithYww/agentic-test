# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

import json
from unittest.mock import MagicMock, patch

import pytest

import config
from common.models import TestCase
from common.services.meego_client import MeegoClient


@pytest.fixture
def meego_client():
    with (
        patch("config.MEEGO_BASE_URL", "http://meego"),
        patch("config.MEEGO_PLUGIN_ID", "plugin_id"),
        patch("config.MEEGO_PLUGIN_SECRET", "plugin_secret"),
        patch("config.MEEGO_USER_KEY", None),
    ):
        client = MeegoClient()
    with patch.object(MeegoClient, "_get_token", return_value="token"):
        yield client


@patch("httpx.Client.post")
def test_create_test_cases_links_test_case_to_story(mock_post, meego_client):
    mock_post.return_value = MagicMock(
        status_code=200, json=lambda: {"code": 0, "data": {"work_item_id": "111"}}
    )

    test_case = TestCase(
        key=None,
        labels=[],
        name="TC1",
        summary="Summary",
        comment="",
        preconditions=None,
        steps=[],
        parent_issue_key=None,
    )

    created_ids = meego_client.create_test_cases([test_case], "example-space", "1000000018")

    assert created_ids == ["111"]
    sent_fields = mock_post.call_args.kwargs["json"]["fields"]
    related_story_field = next(f for f in sent_fields if f["field_key"] == "field_related_story")
    assert related_story_field["field_value"] == "[1000000018]"


@patch("httpx.Client.post")
def test_add_labels_to_test_case_posts_to_typed_work_item_path(mock_post, meego_client):
    mock_post.return_value = MagicMock(status_code=200, json=lambda: {"code": 0})

    meego_client.add_labels_to_test_case("MEEGO-1", ["ui", "automated"])

    sent_url = mock_post.call_args.args[0]
    assert sent_url.endswith("/open_api/example-space/work_item/test_cases/MEEGO-1")
    sent_fields = mock_post.call_args.kwargs["json"]["fields"]
    labels_field = next(f for f in sent_fields if f["field_key"] == "field_labels")
    assert labels_field["field_value"] == ["ui", "automated"]


@patch("httpx.Client.post")
def test_work_item_comment_uses_typed_work_item_path(mock_post, meego_client):
    mock_post.return_value = MagicMock(status_code=200, json=lambda: {"err_code": 0})

    meego_client.add_work_item_comment("example-space", "story", "1000000019", "评审结果")

    assert mock_post.call_args.args[0].endswith("/open_api/example-space/work_item/story/1000000019/comment/create")
    assert mock_post.call_args.kwargs["json"] == {"content": "评审结果"}


@patch("httpx.Client.post")
def test_update_story_test_case_bitable_writes_ai_node_field_and_confirms(mock_post, meego_client):
    mock_post.return_value = MagicMock(status_code=200, json=lambda: {"err_code": 0})

    meego_client.update_story_test_case_bitable(
        "example-space", "1000000020", "https://example.feishu.cn/base/abc123"
    )

    calls = mock_post.call_args_list
    assert [c.args[0].rsplit("/", 3)[-3:] for c in calls] == [
        ["open_api", "ai_node", "edit"],
        ["open_api", "ai_node", "edit"],
        ["open_api", "ai_node", "confirm"],
    ]
    for call in calls:
        body = call.kwargs["json"]
        assert body["project_key"] == "example-project-key"
        assert body["work_item_id"] == "1000000020"
        assert body["node_uuid"] == f"node_{config.MEEGO_AI_PREGENERATION_NODE_KEY}_1000000020"

    write_body, done_body, confirm_body = (c.kwargs["json"] for c in calls)
    conform_field = write_body["conform_fields"][0]
    assert conform_field["field_key"] == "field_test_cases"
    value = json.loads(conform_field["value"])
    assert value["doc_text"] == "多维表格：https://example.feishu.cn/base/abc123"
    assert write_body["status"] == "running"
    assert done_body["status"] == "done"
    assert "conform_fields" not in confirm_body
