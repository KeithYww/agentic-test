# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Creates a Feishu document from Feishu DocxXML content via the lark-cli binary.

Requires the `lark-cli` binary on PATH, configured with a bot identity that has been
granted docx write scopes (docx:document, docx:document:create).
"""

import json
import subprocess

import config
from common import utils

logger = utils.get_logger(__name__)

_LARK_CLI_TIMEOUT_SECONDS = 60


def _grant_doc_edit_permission(doc_token: str, user_ids: list[str]) -> None:
    """Proactively grant edit access to each user on a bot-owned document.

    Uses drive permission.members create instead of the apply→approve flow,
    which is dead-on-arrival for bot-owned docs (no one clicks the approval card).
    Non-fatal: failures are logged as warnings and do not abort the pipeline.
    """
    for uid in user_ids:
        try:
            result = subprocess.run(
                [
                    "lark-cli", "drive", "permission.members", "create",
                    "--token", doc_token,
                    "--type", "docx",
                    "--as", "bot",
                    "--data", json.dumps({
                        "member_type": "openid",
                        "member_id": uid,
                        "perm": "edit",
                        "type": "user",
                    }),
                    "--yes",
                ],
                capture_output=True,
                text=True,
                timeout=_LARK_CLI_TIMEOUT_SECONDS,
            )
            resp = json.loads(result.stdout) if result.stdout else {}
            if resp.get("ok"):
                logger.info(f"Granted edit access on doc {doc_token} to {uid}")
            else:
                logger.warning(f"Could not grant access on doc {doc_token} to {uid}: {resp.get('error')}")
        except Exception as exc:
            logger.warning(f"Exception granting edit access on doc {doc_token} to {uid}: {exc}")


def create_report_doc(title: str, content: str) -> str | None:
    """Creates a Feishu document from Feishu DocxXML content and returns its URL, or None on
    failure. Automatically grants edit access to FEISHU_DOC_COLLABORATOR_IDS after creation."""
    try:
        result = subprocess.run(
            ["lark-cli", "docs", "+create", "--content", "-", "--doc-format", "xml", "--as", "bot",
             "--title", title],
            input=content,
            capture_output=True,
            text=True,
            timeout=_LARK_CLI_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"Failed to run lark-cli to create report doc '{title}': {exc}")
        return None

    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.warning(f"lark-cli returned non-JSON output for report doc '{title}': {result.stdout!r}")
        return None

    if not response.get("ok"):
        logger.warning(f"lark-cli failed to create report doc '{title}': {response.get('error')}")
        return None

    # Confirmed real response shape: {"data": {"document": {"url": ..., "document_id": ...}}}.
    # data.get("url") kept as a defensive fallback in case the shape ever varies.
    data = response.get("data", {})
    doc_url = data.get("url") or (data.get("document") or {}).get("url")
    doc_token = (data.get("document") or {}).get("document_id")
    if not doc_url:
        logger.warning(f"lark-cli response for report doc '{title}' had no recognizable URL field: {data}")
        return None

    if doc_token and config.FEISHU_DOC_COLLABORATOR_IDS:
        _grant_doc_edit_permission(doc_token, config.FEISHU_DOC_COLLABORATOR_IDS)

    return doc_url
