# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Pipeline completion is a handoff, not automatic quality approval."""

from unittest.mock import AsyncMock, patch

import pytest

from common.services.feishu_bot_notifier import notify_pipeline_failure, notify_pipeline_success


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [0, 3])
async def test_success_card_identifies_human_review(pending: int) -> None:
    with patch("common.services.feishu_bot_notifier._send", AsyncMock()) as send:
        await notify_pipeline_success("S1", 20, "https://example/base/1", pending_findings=pending)
    card = send.call_args.args[0]["card"]
    assert card["header"]["title"]["content"] == "用例生成完成，待人工评审"
    assert card["header"]["template"] == ("yellow" if pending else "green")
    if pending:
        assert "自动评审仍有3条意见" in str(card["elements"])



@pytest.mark.asyncio
async def test_success_card_lists_waived_requirements_for_human_review() -> None:
    with patch("common.services.feishu_bot_notifier._send", AsyncMock()) as send:
        await notify_pipeline_success("S1", 20, "https://example/base/1", untestable=["修订记录：版本信息，无可测行为"])
    assert "自动评审判定无需测试的需求项" in str(send.call_args.args[0]["card"]["elements"])
    assert "修订记录：版本信息，无可测行为" in str(send.call_args.args[0]["card"]["elements"])

@pytest.mark.asyncio
@pytest.mark.parametrize("public_url,run_id,expected", [
    ("", "run-1", False),
    ("https://qa.example.com", None, False),
    ("https://qa.example.com", "run-1", True),
])
async def test_failure_retry_button_only_links_configured_persistent_task(monkeypatch, public_url, run_id, expected):
    monkeypatch.setattr("config.DASHBOARD_PUBLIC_URL", public_url)
    with patch("common.services.feishu_bot_notifier._send", AsyncMock()) as send:
        await notify_pipeline_failure("S1", RuntimeError("failure"), project_key="P", run_id=run_id)
    card = send.call_args.args[0]["card"]
    buttons = [button for element in card["elements"] if element["tag"] == "action" for button in element["actions"]]
    retry = [b for b in buttons if "Retry" in b["text"]["content"]]
    assert len(retry) == int(expected)
    if expected:
        assert retry[0]["url"] == "https://qa.example.com/?run_id=run-1"
        assert "value" not in retry[0]  # URL navigation, not a Feishu callback permission.
