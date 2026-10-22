# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Temporary integration with photon_kb (testplatform's internal knowledge base).

Queries the same external HTTP API and shared default key that photon_kb's own MCP bridge
uses (photon_kb/views/mcp.py in the testplatform repo). This is a stopgap for test-case
generation context — not the project's long-term RAG knowledge-base architecture.
"""

import asyncio

import httpx

import config
from common import utils
from common.design_models import KnowledgeSnapshot

logger = utils.get_logger(__name__)

# Retry request failures only; a successful empty answer is normal, even with
# confidence="high", and must not delay or block test-case generation.
_MAX_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2.0


async def query_knowledge_snapshot(question: str) -> KnowledgeSnapshot:
    """Distinguish a successful empty search from an unavailable required upstream."""
    if not config.KB_QUERY_ENABLED or not config.KB_API_KEY:
        return KnowledgeSnapshot(status="disabled")
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        data = await _post_query(question, "qa", attempt)
        if data is None:
            continue
        answer = data.get("answer") or ""
        if data.get("confidence") == "fallback" or not answer.strip():
            return KnowledgeSnapshot(status="no_match")
        return KnowledgeSnapshot(status="ok", content=answer)
    return KnowledgeSnapshot(status="unavailable")


async def query_photon_kb(question: str, role: str = "qa") -> str | None:
    """Queries photon_kb's external API and returns the generated answer text, or None if the
    feature is disabled or every attempt fails (non-fatal — callers must treat this as optional
    context, never block generation on it)."""
    if not config.KB_QUERY_ENABLED or not config.KB_API_KEY:
        return None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        data = await _post_query(question, role, attempt)
        if data is None:
            continue

        answer = data.get("answer")
        confidence = data.get("confidence")
        if confidence == "fallback":
            logger.info("photon_kb returned a fallback-confidence answer; skipping injection.")
            return None
        if answer and answer.strip():
            logger.info(f"photon_kb returned an answer (confidence={confidence}, hits={len(data.get('hits', []))}).")
            return answer

        logger.info("photon_kb returned an empty answer; continuing without supplementary knowledge.")
        return None

    logger.warning(f"photon_kb gave no usable answer after {_MAX_ATTEMPTS} attempts; skipping injection.")
    return None


async def _post_query(question: str, role: str, attempt: int) -> dict | None:
    """Makes one attempt at the request; returns the parsed JSON body, or None on failure."""
    try:
        async with httpx.AsyncClient(timeout=config.KB_QUERY_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{config.KB_BASE_URL}/api/photon-kb/external/query/",
                headers={"X-KB-Api-Key": config.KB_API_KEY},
                json={"question": question, "role": role},
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.warning(f"photon_kb query failed on attempt {attempt}/{_MAX_ATTEMPTS}: {e}")
        if attempt < _MAX_ATTEMPTS:
            await asyncio.sleep(_RETRY_DELAY_SECONDS)
        return None
