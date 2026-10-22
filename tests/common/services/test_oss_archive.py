# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""OSS boundary and unsafe node-transition regressions."""

import hashlib
from unittest.mock import MagicMock, patch

import httpx
import pytest

from common.services.meego_client import MeegoClient
from common.services.oss_archive import archive_design


@pytest.mark.asyncio
async def test_missing_oss_configuration_blocks_without_upload(monkeypatch):
    monkeypatch.delenv("OSS_BUCKET", raising=False)
    with (
        patch("common.services.oss_archive.oss2.Bucket") as bucket,
        pytest.raises(RuntimeError, match="archive_failed"),
    ):
        await archive_design("P", "S", "R", {})
    bucket.assert_not_called()


@pytest.mark.asyncio
async def test_oss_signed_upload_checksum_and_no_overwrite(monkeypatch):
    for key, value in {
        "OSS_ENDPOINT": "https://oss.example",
        "OSS_BUCKET": "qa",
        "OSS_REGION": "cn-hangzhou",
        "OSS_ACCESS_KEY_ID": "test",
        "OSS_ACCESS_KEY_SECRET": "test",
    }.items():
        monkeypatch.setenv(key, value)
    with patch("common.services.oss_archive.oss2.Bucket") as bucket:
        key = await archive_design("P/1", "S", "R", {"reviewed": True})
    args = bucket.return_value.put_object.call_args
    assert key == "runs/P%2F1/S/R/final.json"
    assert args.args[0] == key
    assert args.kwargs["headers"]["x-oss-forbid-overwrite"] == "true"
    assert args.kwargs["headers"]["x-oss-meta-sha256"] == hashlib.sha256(args.args[1]).hexdigest()


@pytest.mark.parametrize("failed_edit", [1, 2])
def test_meego_edit_failure_never_confirms(failed_edit):
    client = MeegoClient("P")
    with (
        patch.object(client, "_post", side_effect=[{}] * (failed_edit - 1) + [RuntimeError("edit failed")]) as post,
        pytest.raises(RuntimeError, match="edit failed"),
    ):
        client.update_story_test_case_bitable("P", "S", "https://example/base")
    assert post.call_count == failed_edit
    assert all(c.args[0] != "/open_api/ai_node/confirm" for c in post.call_args_list)


def test_meego_lost_confirm_response_is_unknown_and_not_retried():
    client = MeegoClient("P")
    with (
        patch.object(client, "_post", side_effect=httpx.ReadTimeout("lost response")) as post,
        pytest.raises(RuntimeError, match="confirm_unknown"),
    ):
        client.confirm_story_pregeneration_node("P", "S")
    post.assert_called_once()
