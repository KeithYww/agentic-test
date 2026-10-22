# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.feishu_mcp_server import _resolve_document_id


@pytest.mark.asyncio
async def test_resolve_document_id_docx_url():
    document_id = await _resolve_document_id(
        "https://example.feishu.cn/docx/ExampleDocumentToken12", "fake-token"
    )
    assert document_id == "ExampleDocumentToken12"


@pytest.mark.asyncio
async def test_resolve_document_id_wiki_url():
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_response = MagicMock()
        mock_response.json.return_value = {"code": 0, "data": {"node": {"obj_type": "docx", "obj_token": "resolved-token"}}}
        mock_client.get.return_value = mock_response

        document_id = await _resolve_document_id("https://example.feishu.cn/wiki/ExampleDocumentToken13", "fake-token")

    assert document_id == "resolved-token"


@pytest.mark.asyncio
async def test_resolve_document_id_rejects_meego_story_link():
    """PRDs commonly link back to their own Feishu Project (Meego) story for cross-reference.
    That link shares the feishu.cn domain but is a work item page, not a document — it must be
    rejected immediately with an actionable message instead of silently passed through to a
    guaranteed-to-fail raw_content request."""
    with pytest.raises(ValueError, match=r"not a Feishu wiki .* or docx .* document URL"):
        await _resolve_document_id("https://project.feishu.cn/example-space/story/detail/1000000014", "fake-token")


@pytest.mark.asyncio
async def test_resolve_document_id_rejects_other_domain_url():
    with pytest.raises(ValueError, match=r"not a Feishu wiki .* or docx .* document URL"):
        await _resolve_document_id("https://internal.example.com/some/page", "fake-token")


@pytest.mark.asyncio
async def test_resolve_document_id_passes_through_bare_token_when_wiki_lookup_fails():
    """A bare alphanumeric token that isn't a valid wiki node (and isn't a URL) is passed
    through unchanged, since it might still be a valid docx token."""
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_client.get.return_value = mock_response

        document_id = await _resolve_document_id("ExampleDocumentToken12", "fake-token")

    assert document_id == "ExampleDocumentToken12"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "permission", "truncated"])
async def test_document_read_requires_complete_embedded_sheets(failure):
    """A PRD must contain its referenced table or fail before model execution."""
    import json
    import subprocess

    from scripts.feishu_mcp_server import feishu_get_doc_content

    document = '<p>PRD</p><sheet sheet-id="abc" token="book"></sheet>'
    responses = [
        {"ok": True, "data": {"document": {"content": document}}},
        {"ok": True, "data": {"sheets": [{"sheet_id": "abc", "row_count": 2, "column_count": 2}]}},
        {"ok": True, "data": {"has_more": failure == "truncated", "ranges": [{"cells": [[{"value": "field"}, {"value": "a < b"}]]}]}},
    ]
    if failure == "permission":
        responses[1] = {"ok": False, "error": {"code": 99991672, "message": "missing sheets:spreadsheet:read"}}

    def run_cli(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=json.dumps(responses.pop(0)), stderr="")

    with (
        patch("scripts.feishu_mcp_server._get_tenant_access_token", AsyncMock(return_value="token")),
        patch("scripts.feishu_mcp_server._resolve_document_id", AsyncMock(return_value="document")),
        patch("subprocess.run", side_effect=run_cli),
        patch("httpx.AsyncClient") as http,
    ):
        response = MagicMock()
        response.json.return_value = {"code": 0, "data": {"content": "PRD without embedded sheet"}}
        http.return_value.__aenter__.return_value.get = AsyncMock(return_value=response)
        if failure:
            with pytest.raises(RuntimeError, match="sheets:spreadsheet:read" if failure == "permission" else "truncated"):
                await feishu_get_doc_content("document")
        else:
            content = await feishu_get_doc_content("document")
            assert "<sheet " not in content
            assert "<td>field</td>" in content
            assert "a &lt; b" in content
