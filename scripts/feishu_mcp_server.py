# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""MCP server exposing Feishu (Lark) document content as a tool.

Serves as the content source for agents that need to read a PRD maintained in
Feishu instead of Jira. Authenticates against the Feishu Open API using an
internal app's tenant access token, which it caches until shortly before
expiry.

The MCP SSE transport is served under ``/sse`` (+ ``/messages/``).
"""

import asyncio
import html
import json
import re
import subprocess
import time

import httpx
from mcp.server.fastmcp import FastMCP

import config
from common import utils

logger = utils.get_logger("feishu_mcp_server")

FEISHU_API_BASE_URL = config.FEISHU_API_BASE_URL
_DOC_TOKEN_URL_PATTERN = re.compile(r"/(?:docx|docs)/([A-Za-z0-9]+)")
_WIKI_TOKEN_URL_PATTERN = re.compile(r"/wiki/([A-Za-z0-9]+)")

mcp = FastMCP("feishu-mcp")

_cached_tenant_access_token: str | None = None
_cached_tenant_access_token_expires_at: float = 0.0
_token_refresh_lock = asyncio.Lock()


async def _get_tenant_access_token() -> str:
    global _cached_tenant_access_token, _cached_tenant_access_token_expires_at

    # Fast path: no lock needed when token is still valid.
    if _cached_tenant_access_token and time.monotonic() < _cached_tenant_access_token_expires_at:
        return _cached_tenant_access_token

    async with _token_refresh_lock:
        # Re-check inside the lock: another coroutine may have refreshed while we waited.
        if _cached_tenant_access_token and time.monotonic() < _cached_tenant_access_token_expires_at:
            return _cached_tenant_access_token

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{FEISHU_API_BASE_URL}/auth/v3/tenant_access_token/internal",
                json={"app_id": config.FEISHU_APP_ID, "app_secret": config.FEISHU_APP_SECRET},
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != 0:
                raise RuntimeError(f"Failed to obtain Feishu tenant access token: {payload}")

        token: str = payload["tenant_access_token"]
        _cached_tenant_access_token = token
        # Refresh a bit before the token actually expires to avoid races with in-flight requests.
        _cached_tenant_access_token_expires_at = time.monotonic() + payload["expire"] - 60
        return token


async def _resolve_document_id(doc_token_or_url: str, tenant_access_token: str) -> str:
    """Resolves a doc token, docx/docs URL or wiki URL to a docx document ID.

    Wiki URLs (the org's primary way of sharing PRDs) don't carry the docx document ID
    directly - the wiki node token must be exchanged for the underlying document's obj_token.
    """
    if wiki_match := _WIKI_TOKEN_URL_PATTERN.search(doc_token_or_url):
        node_token = wiki_match.group(1)
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{FEISHU_API_BASE_URL}/wiki/v2/spaces/get_node",
                params={"token": node_token},
                headers={"Authorization": f"Bearer {tenant_access_token}"},
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != 0:
                raise RuntimeError(f"Failed to resolve Feishu wiki node '{node_token}': {payload}")
        node = payload["data"]["node"]
        if node["obj_type"] != "docx":
            raise RuntimeError(f"Feishu wiki node '{node_token}' is a '{node['obj_type']}', not a docx document")
        return node["obj_token"]

    if docx_match := _DOC_TOKEN_URL_PATTERN.search(doc_token_or_url):
        return docx_match.group(1)

    # Bare alphanumeric token with no URL context — try wiki API first (wiki node tokens
    # look identical to docx IDs; wiki resolution avoids a guaranteed 400 from the docx API).
    if re.fullmatch(r"[A-Za-z0-9]{10,}", doc_token_or_url):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{FEISHU_API_BASE_URL}/wiki/v2/spaces/get_node",
                    params={"token": doc_token_or_url},
                    headers={"Authorization": f"Bearer {tenant_access_token}"},
                )
            if response.status_code == 200:
                payload = response.json()
                if payload.get("code") == 0 and payload["data"]["node"]["obj_type"] == "docx":
                    logger.info("Resolved bare token '%s' via wiki API → %s", doc_token_or_url, payload["data"]["node"]["obj_token"])
                    return payload["data"]["node"]["obj_token"]
        except Exception:
            pass

    if "://" in doc_token_or_url:
        # A URL that isn't a /wiki/ or /docx/ link — e.g. a Feishu Project (Meego) story link
        # such as project.feishu.cn/.../story/detail/... PRDs commonly link back to their own
        # story for cross-reference. It shares the feishu.cn domain but isn't a document, so
        # silently passing it through would build a guaranteed-to-fail raw_content request.
        # Raising here instead gives the calling agent (which gets one retry) a clear, actionable
        # reason to skip the link rather than repeat the same doomed call.
        raise ValueError(
            f"'{doc_token_or_url}' is not a Feishu wiki (/wiki/) or docx (/docx/) document URL — "
            "it cannot be fetched as a document. Do not retry this URL; skip it and continue."
        )

    return doc_token_or_url


def _read_lark(args: list[str]) -> dict:
    result = subprocess.run(["lark-cli", *args, "--as", "bot"], capture_output=True, text=True, timeout=60)
    try:
        payload = json.loads(result.stdout or result.stderr)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Feishu document read returned invalid JSON") from exc
    if result.returncode or not payload.get("ok"):
        error = payload.get("error") or {}
        raise RuntimeError(f"Feishu read failed ({error.get('code', result.returncode)}): {error.get('message', 'unknown error')}")
    return payload["data"]


def _fetch_complete_document(document_id: str) -> str:
    content = _read_lark(["docs", "+fetch", "--doc", document_id])["document"]["content"]
    workbooks: dict[str, dict] = {}

    def expand_sheet(match: re.Match[str]) -> str:
        sheet_id, token = match.groups()
        if token not in workbooks:
            data = _read_lark(["sheets", "+workbook-info", "--spreadsheet-token", token])
            workbooks[token] = {sheet["sheet_id"]: sheet for sheet in data["sheets"]}
        sheet = workbooks[token][sheet_id]
        column = ""
        count = sheet["column_count"]
        while count:
            count, remainder = divmod(count - 1, 26)
            column = chr(65 + remainder) + column
        data = _read_lark([
            "sheets", "+cells-get", "--spreadsheet-token", token, "--sheet-id", sheet_id,
            "--range", f"A1:{column}{sheet['row_count']}", "--include", "value",
        ])
        if data.get("has_more") or data.get("truncated"):
            raise RuntimeError(f"Feishu embedded sheet {sheet_id} was truncated")
        rows = [row for region in data["ranges"] for row in region["cells"]]
        return "<table>" + "".join(
            "<tr>" + "".join("<td>" + html.escape(str(cell.get("value", ""))) + "</td>" for cell in row) + "</tr>"
            for row in rows
        ) + "</table>"

    content = re.sub(r'<sheet sheet-id="([^"]+)" token="([^"]+)"></sheet>', expand_sheet, content)
    if re.search(r"<sheet[ >]", content):
        raise RuntimeError("Feishu PRD contains an unresolved embedded sheet")
    return content


@mcp.tool()
async def feishu_get_doc_content(doc_token_or_url: str) -> str:
    """
    Fetches Feishu document content with complete embedded spreadsheet tables.

    IMPORTANT: Only call this tool for Feishu/Lark documents. Valid inputs are:
    - A Feishu wiki URL: https://xxx.feishu.cn/wiki/...  or  https://xxx.larksuite.com/wiki/...
    - A Feishu docx URL: https://xxx.feishu.cn/docx/...  or  https://xxx.larksuite.com/docx/...
    - A bare Feishu document token (alphanumeric string, e.g. "ExampleDocumentToken12")

    Do NOT call this tool with URLs from other domains (e.g. internal systems, Jira, SIT
    environments, or any non-feishu.cn / non-larksuite.com URL). Those are not Feishu documents
    and will always fail.

    Do NOT call this tool with a Feishu Project (Meego) work item link either, even though its
    domain is feishu.cn — e.g. https://project.feishu.cn/<space>/story/detail/<id>. PRDs commonly
    link back to their own story for cross-reference; that is a work item page, not a document,
    and fetching it will always fail. Only /wiki/ and /docx/ paths are documents.

    If the fetched content contains any Feishu document URLs (patterns: /wiki/ or /docx/ under
    feishu.cn or larksuite.com), call this tool again for each of those URLs. Fetch only one
    level deep — do not follow links found inside the linked documents themselves.

    Args:
        doc_token_or_url: A Feishu wiki/docx URL or a bare Feishu document token.

    Returns:
        The plain-text content of the Feishu document.
    """
    tenant_access_token = await _get_tenant_access_token()
    document_id = await _resolve_document_id(doc_token_or_url, tenant_access_token)

    content = await asyncio.to_thread(_fetch_complete_document, document_id)

    logger.info(f"Fetched Feishu document '{document_id}'")
    return content


app = mcp.sse_app()
