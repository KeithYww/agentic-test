# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Recording SSE MCP mock standing in for ``scripts/feishu_mcp_server.py``.

Advertises the ``feishu_get_doc_content`` tool used by the test-case generation
and review agents, returns a canned PRD, and records every call for the smoke
assertions.

The MCP SSE transport is served under ``/sse`` (+ ``/messages/``); a plain
``GET /__recorded`` HTTP route is mounted alongside it.
"""

import json

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

SEEDED_DOC_CONTENT = (
    "Title: Smoke password-reset PRD\n\n"
    "As a registered user I want to reset my password via an email link "
    "so that I can regain access when I forget it.\n\n"
    "Acceptance Criteria:\n"
    "1. A 'Forgot password' link on the login page opens a form for an email address.\n"
    "2. Submitting a registered email queues a password-reset link expiring after 60 minutes.\n"
    "3. Submitting an unregistered email shows the same confirmation message.\n"
    "4. Following a valid link lets the user set a new password meeting the complexity policy.\n"
    "5. An expired or already-used link shows an error and offers to request a new one.\n"
)

_recorded: dict[str, list] = {"fetched_docs": []}

mcp = FastMCP(
    "feishu-mcp-mock",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@mcp.tool()
async def feishu_get_doc_content(doc_token_or_url: str) -> str:
    """Fetches the plain-text content of a Feishu (Lark) document.

    Args:
        doc_token_or_url: The Feishu document token, or the full URL to the document.

    Returns:
        The plain-text content of the document.
    """
    _recorded["fetched_docs"].append(doc_token_or_url)
    return SEEDED_DOC_CONTENT


async def _recorded_endpoint(_request: Request) -> JSONResponse:
    return JSONResponse(_recorded)


app = Starlette(
    routes=[
        Route("/__recorded", _recorded_endpoint),
        Mount("/", app=mcp.sse_app()),
    ]
)
