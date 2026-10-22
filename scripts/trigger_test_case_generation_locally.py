# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Manual local trigger for the ``test_case_generation`` agent.

Bypasses the orchestrator and its Jira webhook entirely: sends a Feishu
document token/URL directly to a locally running ``test_case_generation``
agent over A2A, and prints the generated test cases.

Usage:
    uv run python scripts/trigger_test_case_generation_locally.py <feishu_doc_token_or_url>
"""

import asyncio
import sys

import httpx
from a2a.client import ClientConfig, create_client
from a2a.client.card_resolver import parse_agent_card
from a2a.helpers import get_message_text, new_text_message
from a2a.types import Message, Role, SendMessageRequest, TaskState

import config

AGENT_BASE_URL = f"http://localhost:{config.TestCaseGenerationAgentConfig.PORT}"
TIMEOUT_SECONDS = 300


async def _fetch_agent_card(agent_base_url: str):
    agent_card_url = f"{agent_base_url}/.well-known/agent-card.json"
    async with httpx.AsyncClient() as client:
        response = await client.get(agent_card_url, timeout=30)
        response.raise_for_status()
        return parse_agent_card(response.json())


async def trigger(doc_token_or_url: str) -> None:
    agent_card = await _fetch_agent_card(AGENT_BASE_URL)
    message: Message = new_text_message(doc_token_or_url, role=Role.ROLE_USER)

    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as httpx_client:
        a2a_client = await create_client(agent_card, client_config=ClientConfig(httpx_client=httpx_client))
        response_iterator = a2a_client.send_message(SendMessageRequest(message=message))
        async for chunk in response_iterator:
            if chunk.HasField("status_update"):
                status = chunk.status_update.status
                if status.state == TaskState.TASK_STATE_FAILED:
                    error_msg = get_message_text(status.message) if status.message else "unknown error"
                    print(f"Task failed: {error_msg}", file=sys.stderr)
                    return
                if status.state == TaskState.TASK_STATE_REJECTED:
                    print("Task rejected by agent.", file=sys.stderr)
                    return
                if status.state == TaskState.TASK_STATE_WORKING and status.message:
                    print(f"[working] {get_message_text(status.message)}")
            elif chunk.HasField("artifact_update"):
                for part in chunk.artifact_update.artifact.parts:
                    if part.HasField("text"):
                        print(part.text)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <feishu_doc_token_or_url>")
        sys.exit(1)
    asyncio.run(trigger(sys.argv[1]))
