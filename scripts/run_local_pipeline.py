# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Local end-to-end pipeline: generation -> classification -> review.

Bypasses the orchestrator entirely: sends a Feishu document token/URL directly
to the three locally running agents over A2A, merges their results by test
case key, and writes the combined result to local ``output/*.json`` and
``output/*.md`` files instead of uploading to Zephyr/Xray.

Usage:
    uv run python scripts/run_local_pipeline.py <feishu_doc_token_or_url>
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx
from a2a.client import ClientConfig, create_client
from a2a.client.card_resolver import parse_agent_card
from a2a.helpers import get_message_text, new_text_message
from a2a.types import Message, Role, SendMessageRequest, TaskState

import config
from common.models import (
    AgentExecutionError,
    ClassifiedTestCases,
    GeneratedTestCases,
    JsonSerializableModel,
    TestCaseReviewFeedbacks,
)
from common.report_rendering import merge_test_case_results, render_markdown_report

TIMEOUT_SECONDS = 3600
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


async def _fetch_agent_card(agent_base_url: str):
    agent_card_url = f"{agent_base_url}/.well-known/agent-card.json"
    async with httpx.AsyncClient() as client:
        response = await client.get(agent_card_url, timeout=30)
        response.raise_for_status()
        return parse_agent_card(response.json())


async def _send_message_and_get_text[T: JsonSerializableModel](
    agent_base_url: str, text: str, model_type: type[T]
) -> T:
    """Sends a text message to a locally running agent over A2A and parses its response.

    Args:
        agent_base_url: Base URL of the agent to send the message to.
        text: Text content of the message.
        model_type: The expected model type to parse the agent's final text artifact as.

    Returns:
        The parsed model of type ``model_type``.

    Raises:
        RuntimeError: If the task fails, is rejected, returns no text artifact, or the
            agent returned an ``AgentExecutionError``.
    """
    agent_card = await _fetch_agent_card(agent_base_url)
    message: Message = new_text_message(text, role=Role.ROLE_USER)

    text_parts: list[str] = []
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as httpx_client:
        a2a_client = await create_client(agent_card, client_config=ClientConfig(httpx_client=httpx_client))
        response_iterator = a2a_client.send_message(SendMessageRequest(message=message))
        async for chunk in response_iterator:
            if chunk.HasField("status_update"):
                status = chunk.status_update.status
                if status.state == TaskState.TASK_STATE_FAILED:
                    error_msg = get_message_text(status.message) if status.message else "unknown error"
                    raise RuntimeError(f"Task failed: {error_msg}")
                if status.state == TaskState.TASK_STATE_REJECTED:
                    raise RuntimeError("Task rejected by agent.")
                if status.state == TaskState.TASK_STATE_WORKING and status.message:
                    print(f"[working] {get_message_text(status.message)}")
            elif chunk.HasField("artifact_update"):
                for part in chunk.artifact_update.artifact.parts:
                    if part.HasField("text") and part.text:
                        text_parts.append(part.text)

    if len(text_parts) != 1:
        raise RuntimeError(f"Expected exactly one text artifact from the agent, but received {len(text_parts)}.")

    text_content = text_parts[0]
    try:
        error = AgentExecutionError.model_validate_json(text_content)
        raise RuntimeError(f"Agent returned an execution error: {error.error_message}")
    except ValueError:
        pass

    return model_type.model_validate_json(text_content)


def _write_json(merged: list[dict], output_path: Path) -> None:
    output_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_markdown(merged: list[dict], output_path: Path) -> None:
    output_path.write_text(render_markdown_report(merged), encoding="utf-8")


async def run_pipeline(doc_token_or_url: str) -> None:
    generation_url = f"http://localhost:{config.TestCaseGenerationAgentConfig.PORT}"
    classification_url = f"http://localhost:{config.TestCaseClassificationAgentConfig.PORT}"
    review_url = f"http://localhost:{config.TestCaseReviewAgentConfig.PORT}"

    print("Requesting test case generation...")
    generated = await _send_message_and_get_text(generation_url, doc_token_or_url, GeneratedTestCases)
    print(f"Generated {len(generated.test_cases)} test cases.")

    # Without a real Zephyr/Xray backend, generated test cases have no key. Assign a local
    # sequential one here so classification and review have a stable identifier to echo back.
    for i, test_case in enumerate(generated.test_cases, start=1):
        test_case.key = f"TC-{i}"

    print("Requesting test case classification...")
    classified = await _send_message_and_get_text(
        classification_url, f"Test cases:\n{generated.test_cases}", ClassifiedTestCases
    )
    print(f"Classified {len(classified.test_cases)} test cases.")

    print("Requesting test case review...")
    reviewed = await _send_message_and_get_text(
        review_url,
        f"Test cases:\n{json.dumps({'test_cases': [tc.model_dump() for tc in generated.test_cases]})}\nFeishu document: {doc_token_or_url}",
        TestCaseReviewFeedbacks,
    )
    print(f"Received review feedback for {len(reviewed.review_feedbacks)} test cases.")

    merged = merge_test_case_results(generated, classified, reviewed)

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"{timestamp}.json"
    md_path = OUTPUT_DIR / f"{timestamp}.md"
    _write_json(merged, json_path)
    _write_markdown(merged, md_path)

    print(f"Wrote results to {json_path} and {md_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <feishu_doc_token_or_url>")
        sys.exit(1)
    asyncio.run(run_pipeline(sys.argv[1]))
