# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

import subprocess
from unittest.mock import MagicMock, patch

from common.services.feishu_doc_service import create_report_doc


@patch("subprocess.run")
def test_create_report_doc_returns_url_on_success(mock_run):
    mock_run.return_value = MagicMock(
        stdout='{"ok": true, "data": {"url": "https://example.feishu.cn/docx/abc123"}}'
    )

    doc_url = create_report_doc("My Report", "<h1>Content</h1>")

    assert doc_url == "https://example.feishu.cn/docx/abc123"
    assert mock_run.call_args.kwargs["input"] == "<h1>Content</h1>"
    assert "--doc-format" in mock_run.call_args.args[0]
    doc_format_index = mock_run.call_args.args[0].index("--doc-format") + 1
    assert mock_run.call_args.args[0][doc_format_index] == "xml"


@patch("subprocess.run")
def test_create_report_doc_returns_none_on_cli_error(mock_run):
    mock_run.return_value = MagicMock(stdout='{"ok": false, "error": {"message": "not authorized"}}')

    assert create_report_doc("My Report", "<h1>Content</h1>") is None


@patch("subprocess.run")
def test_create_report_doc_returns_none_on_non_json_output(mock_run):
    mock_run.return_value = MagicMock(stdout="not json")

    assert create_report_doc("My Report", "<h1>Content</h1>") is None


@patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="lark-cli", timeout=60))
def test_create_report_doc_returns_none_on_timeout(mock_run):
    assert create_report_doc("My Report", "<h1>Content</h1>") is None
