# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Hermetic end-to-end smoke checks for the core QuAIA flows.

All of our own code runs for real (orchestrator + agents + the real Photon API relay); only
the external boundaries are mocked. Each test asserts on what reached a mocked
boundary, read back from its ``/__recorded`` endpoint:

* Requirements review   -> a non-empty comment reached the reviewed work item in Feishu Project.
* Requirements review   -> the agent first fetched the requirement document via the Feishu MCP.
* Test-case generation, classification, and review -> all three agents run for real, and their
                           combined output reaches the report doc handed to lark-cli (recorded via
                           the lark-cli shim) — no per-test-case work items are created in Feishu
                           Project anymore.
* Test-case pipeline    -> the consolidated report doc's URL is written back to the seeded
                           story's "ai用例预生成" node in Feishu Project.
* Test execution        -> a failed automated test drove a real Bug issue into the seeded project.
* Test execution        -> a failed execution for the seeded case was reported to Zephyr,
                           and the created bug was linked to that execution.
* Test execution        -> the incident-creation flow consulted the vector DB for duplicates.
* RAG DB update         -> the sync pushed the seeded story into the vector DB.
* Observability         -> orchestrator and agents export OpenTelemetry traces to the OTLP
                           collector (self-hosted Langfuse in real deployments).
* Negative paths        -> the three authenticated webhooks reject a bad API key (401); the
                           requirement-review webhook is unauthenticated by design and instead
                           silently drops a malformed payload or a missing PRD link (204); the
                           required-field webhooks reject a missing required field (400), the
                           project-key webhooks reject a missing project_key (422), and the
                           dashboard API rejects a missing token (401).
"""

import os
import re
import time
from collections.abc import Callable

import httpx
import pytest

import config
from tests.smoke.conftest import (
    FEISHU_MCP_RECORDED_URL,
    JIRA_MCP_RECORDED_URL,
    MEEGO_RECORDED_URL,
    ORCHESTRATOR_URL,
    OTEL_COLLECTOR_RECORDED_URL,
    QDRANT_RECORDED_URL,
    SEEDED_EXECUTABLE_TC_KEY,
    SEEDED_ISSUE_KEY,
    SEEDED_PROJECT_KEY,
    SEEDED_REVIEW_WORK_ITEM_ID,
    SEEDED_STORY_ID,
    SMOKE_REQUIREMENT_REVIEW_DOC_URL,
    TICKETS_COLLECTION_NAME,
    ZEPHYR_RECORDED_URL,
)
from tests.smoke.mocks.feishu_api_mock import SEEDED_REQUIREMENT_IDS

pytestmark = pytest.mark.smoke

# The webhook returns 202 immediately; the pipeline runs asynchronously. Each stage
# (generation, classification, review) involves one or more LLM calls, so the full
# pipeline for a small story can take several minutes. Allow enough time per test.
# Requirements review and test-case generation now share the same orchestrator queue/consumer
# (see orchestrator/main.py's _pipeline_queue), so when both fire around the same time one can
# queue behind the other's full run — the timeout must cover that added wait, not just one
# flow's own LLM latency.
RECORD_POLL_TIMEOUT = 480.0
RECORD_POLL_INTERVAL = 5.0


def _wait_for_recorded(
    http_client: httpx.Client, url: str, predicate: Callable[[dict], bool]
) -> dict:
    """Poll a mock's /__recorded endpoint until the predicate holds or time runs out."""
    deadline = time.monotonic() + RECORD_POLL_TIMEOUT
    data: dict = {}
    while time.monotonic() < deadline:
        response = http_client.get(url)
        if response.status_code == 200:
            data = response.json()
            if predicate(data):
                return data
        time.sleep(RECORD_POLL_INTERVAL)
    return data


# --- Requirements review flow ----------------------------------------------------------


def test_requirements_review_webhook_accepted(requirements_review_response: httpx.Response) -> None:
    expected = 202 if config.DesignConfig.REQUIREMENT_REVIEW_ENABLED else 204
    assert requirements_review_response.status_code == expected, (
        f"Requirements-review webhook failed: "
        f"{requirements_review_response.status_code} {requirements_review_response.text}"
    )


def test_agent_fetched_requirement_doc_for_review(
    requirements_review_response: httpx.Response, http_client: httpx.Client
) -> None:
    """The review must be grounded in the real requirement doc: the agent must fetch it via the Feishu MCP first."""
    if not config.DesignConfig.REQUIREMENT_REVIEW_ENABLED:
        data = http_client.get(FEISHU_MCP_RECORDED_URL).json()
        assert SMOKE_REQUIREMENT_REVIEW_DOC_URL.rsplit("/", 1)[-1] not in data.get("fetched_docs", [])
        return
    data = _wait_for_recorded(
        http_client, FEISHU_MCP_RECORDED_URL, lambda d: SMOKE_REQUIREMENT_REVIEW_DOC_URL.rsplit("/", 1)[-1] in d.get("fetched_docs", [])
    )
    assert SMOKE_REQUIREMENT_REVIEW_DOC_URL.rsplit("/", 1)[-1] in data.get("fetched_docs", []), (
        f"Requirements review agent never fetched the requirement doc via Feishu MCP. Recorded: {data}"
    )


def test_otel_traces_exported(test_case_flow_response: httpx.Response, http_client: httpx.Client) -> None:
    """Orchestrator and agents must export OpenTelemetry traces to the OTLP collector."""
    data = _wait_for_recorded(http_client, OTEL_COLLECTOR_RECORDED_URL, lambda d: d.get("traces_received", 0) > 0)
    assert data.get("traces_received", 0) > 0, (
        f"No OpenTelemetry traces reached the OTLP collector mock. Recorded: {data}"
    )


def test_review_comment_reached_feishu_project(
    requirements_review_response: httpx.Response, http_client: httpx.Client
) -> None:
    """The agent must post a non-empty review comment to the reviewed work item in Feishu Project."""
    if not config.DesignConfig.REQUIREMENT_REVIEW_ENABLED:
        data = http_client.get(MEEGO_RECORDED_URL).json()
        assert not any(
            wi.get("work_item_id") == SEEDED_REVIEW_WORK_ITEM_ID and wi.get("comments")
            for wi in data.get("test_cases", [])
        )
        return
    data = _wait_for_recorded(
        http_client,
        MEEGO_RECORDED_URL,
        lambda d: any(
            wi.get("work_item_id") == SEEDED_REVIEW_WORK_ITEM_ID and any(c.strip() for c in wi.get("comments", []))
            for wi in d.get("test_cases", [])
        ),
    )
    reviewed = [
        wi
        for wi in data.get("test_cases", [])
        if wi.get("work_item_id") == SEEDED_REVIEW_WORK_ITEM_ID and any(c.strip() for c in wi.get("comments", []))
    ]
    assert reviewed, f"No non-empty review comment reached Feishu Project work item {SEEDED_REVIEW_WORK_ITEM_ID}. Recorded: {data}"


# --- Test-case generation / classification / review flow -------------------------------


def test_test_case_flow_webhook_accepted(test_case_flow_response: httpx.Response) -> None:
    assert test_case_flow_response.status_code == 202, (
        f"Test-case flow webhook failed: "
        f"{test_case_flow_response.status_code} {test_case_flow_response.text}"
    )


def test_agent_fetched_feishu_doc(
    test_case_flow_response: httpx.Response, http_client: httpx.Client
) -> None:
    """The generation agent must fetch the Feishu document before generating test cases."""
    data = _wait_for_recorded(
        http_client,
        FEISHU_MCP_RECORDED_URL,
        lambda d: bool(d.get("fetched_docs")),
    )
    assert data.get("fetched_docs"), f"Generation agent never called feishu_get_doc_content. Recorded: {data}"
    assert data.get("fetched_sheets"), "Embedded PRD sheets were never read"


def test_real_test_cases_generated_and_reviewed_in_report_doc(
    test_case_flow_response: httpx.Response, http_client: httpx.Client
) -> None:
    """Publishing works without OSS; enabled archiving precedes confirmation."""
    data = _wait_for_recorded(http_client, MEEGO_RECORDED_URL,
                              lambda d: bool(d.get("confirmations")))
    archive = http_client.get("http://localhost:8096/__recorded").json()
    assert archive["kb_queries"]
    assert archive["kb_responses"]
    assert all(r["confidence"] == "high" and r["answer"] == "" for r in archive["kb_responses"])
    records = data["lark_cli_bitable_records"]
    assert records
    assert len({record["用例ID"] for record in records}) == len(records)
    assert all(record["版本"] and record["需求项"] for record in records)
    cited = {ref.strip() for record in records for ref in record["需求项"].split(",")}
    # Requirements come from every PRD section, so the in-document "only verify 1.1" cannot narrow them.
    assert cited <= SEEDED_REQUIREMENT_IDS and "1" in cited
    assert len(data["bitable_comments"]) <= len(data["lark_cli_bitable_records"])
    for comment in data["bitable_comments"]:
        text = "\n".join(part["text"] for part in comment)
        assert "待人工评审" in text
        assert "建议：" in text
        assert "评审轮数：" not in text
        assert "停止原因：" not in text
        assert not re.search(r"TC-\d+ v\d+ 整套评审", text), "Suite findings must not pose as a case's own"
    comment_texts = ["\n".join(part["text"] for part in comment) for comment in data["bitable_comments"]]
    if not config.DesignConfig.OSS_ARCHIVE_ENABLED:
        assert not archive["archives"]
        assert data["lark_cli_bitable_records"]
        return
    saved = archive["archives"][-1]
    final = saved["payload"]
    assert {r["id"] for r in final["requirements"]["items"]} == SEEDED_REQUIREMENT_IDS
    assert all(r["source_type"] == "section" for r in final["requirements"]["items"])
    covered = {ref for c in final["test_cases"] for ref in c["covered_requirement_ids"]}
    waivers = final["reviews"][-1]["suite"]["untestable"]
    assert all(w["reason"].strip() for w in waivers)
    assert SEEDED_REQUIREMENT_IDS - covered <= {w["requirement_id"] for w in waivers}
    assert archive["kb_queries"]
    assert saved["signed"] and saved["checksum"]
    assert final["stop_reason"] in {"converged", "iteration_limit", "no_progress", "budget_handoff"}
    assert final["reviews"][-1]["suite"]["completed"]
    expected = {(c["key"], c["version"]) for c in final["test_cases"]}
    assessed = {(a["case_key"], a["case_version"]) for a in final["reviews"][-1]["assessments"]}
    assert assessed == expected
    assert len(data["lark_cli_bitable_records"]) == len(expected)
    for gap in (f for f in final["reviews"][-1]["suite"]["findings"] if not f["owner_case_key"]):
        relevant = ({k for k in gap["related_case_keys"] if any(c["key"] == k for c in final["test_cases"])}
                    or {c["key"] for c in final["test_cases"] if set(c["covered_requirement_ids"]) & set(gap["requirement_ids"])}
                    or {final["test_cases"][0]["key"]})
        assert sum(gap["reason"] in text for text in comment_texts) == len(relevant), (
            "A suite gap must be attached only to its related cases"
        )
    assert saved["at"] <= data["confirmations"][-1]["at"]


def test_generated_cases_handed_off_to_human_review(
    test_case_flow_response: httpx.Response, http_client: httpx.Client, auth_headers: dict[str, str],
) -> None:
    """A completed automatic loop hands off to humans even with remaining findings."""
    deadline = time.monotonic() + RECORD_POLL_TIMEOUT
    while time.monotonic() < deadline:
        response = http_client.get(f"{ORCHESTRATOR_URL}/api/dashboard/tasks", headers=auth_headers)
        response.raise_for_status()
        tasks = [task for task in response.json() if task["description"].startswith("用例生成全流程：")]
        completed = [task for task in tasks if task["status"] == "COMPLETED"]
        if completed:
            if os.environ.get("SMOKE_DURABLE") == "true":
                recovery = completed[-1]["recovery"]
                assert recovery["state"] == "completed"
                assert 0 < recovery["used_tokens"] <= recovery["token_limit"]
                assert recovery["reserved_tokens"] == 0 and not recovery["usage_unknown"]
                assert recovery["url"]
            assert "已回写并流转至人工评审" in completed[-1]["result_summary"]
            assert "待确认意见：" in completed[-1]["result_summary"]
            return
        time.sleep(RECORD_POLL_INTERVAL)
    pytest.fail("No completed generation handoff was recorded")


def test_consolidated_report_doc_written_to_story(
    test_case_flow_response: httpx.Response, http_client: httpx.Client
) -> None:
    """The pipeline must create a consolidated report doc and write its URL onto the story."""
    data = _wait_for_recorded(
        http_client,
        MEEGO_RECORDED_URL,
        lambda d: any(
            wi.get("work_item_id") == SEEDED_STORY_ID and wi.get("test_case_doc", "").strip()
            for wi in d.get("test_cases", [])
        ),
    )
    story_entry = next(
        (wi for wi in data.get("test_cases", []) if wi.get("work_item_id") == SEEDED_STORY_ID), None
    )
    assert story_entry, f"No field write reached Feishu Project for story {SEEDED_STORY_ID}. Recorded: {data}"
    assert story_entry.get("test_case_doc", "").strip(), (
        f"Story {SEEDED_STORY_ID} had no report doc URL written. Recorded: {story_entry}"
    )



# --- Test execution / incident-creation flow -------------------------------------------


def test_failed_execution_creates_bug_in_jira(
    execute_tests_response: httpx.Response, http_client: httpx.Client
) -> None:
    """A failed automated test must drive incident creation: a real Bug reaches the seeded project."""
    data = _wait_for_recorded(
        http_client,
        JIRA_MCP_RECORDED_URL,
        lambda d: any(
            i.get("summary", "").strip() and i.get("description", "").strip() for i in d.get("created_issues", [])
        ),
    )
    bugs = [
        i
        for i in data.get("created_issues", [])
        if i.get("summary", "").strip()
        and i.get("description", "").strip()
        and i.get("issue_type") == "Bug"
        and i.get("project_key") == SEEDED_PROJECT_KEY
    ]
    assert bugs, (
        f"No Bug issue for project {SEEDED_PROJECT_KEY} reached Jira from the incident-creation flow. "
        f"Recorded: {data}"
    )

    # Regression guard: the redacted debug-trace artifact (ArtifactName.TRACE) must be excluded
    # from the file parts forwarded to the incident-creation agent (see
    # _get_file_contents_from_artifacts in orchestrator/main.py). If that exclusion were ever
    # dropped, the raw pydantic-ai message dump — identifiable by its own serialization field
    # names, which would never otherwise appear in an LLM-authored bug report — could leak into
    # the created issue.
    trace_markers = ("part_kind", "tool_call_id")
    for issue in bugs:
        description = issue.get("description", "")
        assert not any(marker in description for marker in trace_markers), (
            f"Bug description appears to contain a leaked execution trace: {description!r}"
        )


def test_failed_execution_reported_to_zephyr(
    execute_tests_response: httpx.Response, http_client: httpx.Client
) -> None:
    """The reporting half of /execute-tests: a failed execution of the seeded case must reach
    Zephyr, inside a test cycle created for the seeded project."""
    data = _wait_for_recorded(
        http_client,
        ZEPHYR_RECORDED_URL,
        lambda d: any(e.get("testCaseKey") == SEEDED_EXECUTABLE_TC_KEY for e in d.get("test_executions", [])),
    )
    executions = [e for e in data.get("test_executions", []) if e.get("testCaseKey") == SEEDED_EXECUTABLE_TC_KEY]
    assert executions, f"No test execution for {SEEDED_EXECUTABLE_TC_KEY} reached Zephyr. Recorded: {data}"
    failed = [e for e in executions if e.get("statusName") == "Fail"]
    assert failed, f"The execution of {SEEDED_EXECUTABLE_TC_KEY} was not reported as failed. Recorded: {executions}"
    cycle_keys = {cycle.get("key") for cycle in data.get("test_cycles", []) if cycle.get("projectKey") == SEEDED_PROJECT_KEY}
    assert any(e.get("testCycleKey") in cycle_keys for e in failed), (
        f"No failed execution belongs to a test cycle of project {SEEDED_PROJECT_KEY}. "
        f"Executions: {failed}, cycles: {data.get('test_cycles', [])}"
    )


def test_created_bug_linked_to_test_execution(
    execute_tests_response: httpx.Response, http_client: httpx.Client
) -> None:
    """The bug created for the failed execution must be linked back to the Zephyr execution."""
    zephyr = _wait_for_recorded(http_client, ZEPHYR_RECORDED_URL, lambda d: bool(d.get("execution_issue_links")))
    links = zephyr.get("execution_issue_links", [])
    assert links, f"No issue was linked to any test execution in Zephyr. Recorded: {zephyr}"
    mcp = _wait_for_recorded(http_client, JIRA_MCP_RECORDED_URL, lambda d: bool(d.get("created_issues")))
    created_issue_ids = {str(issue.get("id")) for issue in mcp.get("created_issues", [])}
    assert any(str(link.get("issue_id")) in created_issue_ids for link in links), (
        f"No created bug ({created_issue_ids}) was linked to a test execution. Links: {links}"
    )


def test_incident_creation_consulted_vector_db(
    execute_tests_response: httpx.Response, http_client: httpx.Client
) -> None:
    """The duplicate search must consult the vector DB (at least the collection-list probe)."""
    data = _wait_for_recorded(http_client, QDRANT_RECORDED_URL, lambda d: d.get("collections_probes", 0) > 0)
    assert data.get("collections_probes", 0) > 0, f"The vector DB was never consulted. Recorded: {data}"


# --- RAG vector DB update flow ----------------------------------------------------------


def test_update_rag_db_webhook_accepted(update_rag_db_response: httpx.Response) -> None:
    assert update_rag_db_response.status_code == 200, (
        f"RAG-update webhook failed: {update_rag_db_response.status_code} {update_rag_db_response.text}"
    )
    details = update_rag_db_response.json().get("details", {})
    assert details.get("processed_count", 0) >= 1, f"The RAG sync processed no issues: {details}"


def test_rag_sync_upserted_seeded_story_into_vector_db(
    update_rag_db_response: httpx.Response, http_client: httpx.Client
) -> None:
    """The sync must push the seeded story into the tickets collection of the vector DB."""
    data = _wait_for_recorded(
        http_client,
        QDRANT_RECORDED_URL,
        lambda d: any(
            p.get("collection") == TICKETS_COLLECTION_NAME and p.get("payload", {}).get("key") == SEEDED_ISSUE_KEY
            for p in d.get("upserted_points", [])
        ),
    )
    upserts = [
        p
        for p in data.get("upserted_points", [])
        if p.get("collection") == TICKETS_COLLECTION_NAME and p.get("payload", {}).get("key") == SEEDED_ISSUE_KEY
    ]
    assert upserts, f"The seeded story {SEEDED_ISSUE_KEY} never reached the vector DB. Recorded: {data}"
    payload = upserts[0]["payload"]
    assert payload.get("project_key") == SEEDED_PROJECT_KEY, f"Wrong project on the upserted story: {payload}"
    assert payload.get("summary", "").strip(), f"The upserted story has no summary: {payload}"
    assert TICKETS_COLLECTION_NAME in data.get("created_collections", []), (
        f"The tickets collection was never created. Recorded: {data}"
    )


# --- Negative paths (auth + validation; reach the orchestrator only, no LLM) ------------

REQUIRED_FIELD_WEBHOOK_PATHS = ["/story-ready-for-test-case-generation"]
PROJECT_KEY_WEBHOOK_PATHS = ["/execute-tests", "/update-rag-db"]
AUTHENTICATED_WEBHOOKS = [
    ("/story-ready-for-test-case-generation", {"story_id": SEEDED_STORY_ID, "feishu_doc": "https://example.feishu.cn/wiki/x"}),
    ("/execute-tests", {"project_key": SEEDED_PROJECT_KEY}),
    ("/update-rag-db", {"project_key": SEEDED_PROJECT_KEY}),
]


@pytest.mark.parametrize(("path", "payload"), AUTHENTICATED_WEBHOOKS)
def test_webhook_rejects_invalid_api_key(http_client: httpx.Client, path: str, payload: dict[str, str]) -> None:
    """A wrong orchestrator API key must be rejected with 401 before any work starts."""
    response = http_client.post(f"{ORCHESTRATOR_URL}{path}", headers={"X-API-Key": "wrong-key"}, json=payload)
    assert response.status_code == 401, f"{path} accepted an invalid API key: {response.status_code} {response.text}"


@pytest.mark.parametrize("path", REQUIRED_FIELD_WEBHOOK_PATHS)
def test_webhook_rejects_missing_required_fields(
    http_client: httpx.Client, webhook_headers: dict[str, str], path: str
) -> None:
    """A valid key but empty payload must fail validation with 400, without dispatching to an agent."""
    response = http_client.post(f"{ORCHESTRATOR_URL}{path}", headers=webhook_headers, json={})
    assert response.status_code == 400, (
        f"{path} did not reject a missing required field with 400: {response.status_code} {response.text}"
    )


@pytest.mark.parametrize("path", PROJECT_KEY_WEBHOOK_PATHS)
def test_webhook_rejects_missing_project_key(
    http_client: httpx.Client, webhook_headers: dict[str, str], path: str
) -> None:
    """A valid key but no project_key must fail request-model validation with 422."""
    response = http_client.post(f"{ORCHESTRATOR_URL}{path}", headers=webhook_headers, json={})
    assert response.status_code == 422, (
        f"{path} did not reject a missing project_key with 422: {response.status_code} {response.text}"
    )


def test_requirement_review_webhook_rejects_malformed_payload(http_client: httpx.Client) -> None:
    """Unauthenticated by design; a malformed (non-object) payload is silently dropped (204), not an error."""
    response = http_client.post(f"{ORCHESTRATOR_URL}/requirement-ready-for-review", json=["not", "an", "object"])
    assert response.status_code == 204, (
        f"Malformed payload was not silently dropped: {response.status_code} {response.text}"
    )


def test_requirement_review_webhook_rejects_missing_prd_link(http_client: httpx.Client) -> None:
    """A work item whose PRD-link field isn't filled in yet is silently dropped (204), not enqueued."""
    payload = {
        "payload": {
            "id": "SMOKE-REVIEW-NO-LINK",
            "project_simple_name": SEEDED_PROJECT_KEY,
            "nodes": [{"node_form": []}],
        }
    }
    response = http_client.post(f"{ORCHESTRATOR_URL}/requirement-ready-for-review", json=payload)
    assert response.status_code == 204, (
        f"Missing PRD link was not silently dropped: {response.status_code} {response.text}"
    )


@pytest.mark.parametrize("path", [
    "/api/dashboard/agents", "/api/dashboard/design-runs/nonexistent",
    "/api/dashboard/design-runs/nonexistent/resume",
])
def test_dashboard_api_rejects_missing_token(http_client: httpx.Client, path: str) -> None:
    """The dashboard API must reject requests without a bearer token with 401."""
    if path.endswith("/resume"):
        response = http_client.post(f"{ORCHESTRATOR_URL}{path}", json={"generation": 1})
    else:
        response = http_client.get(f"{ORCHESTRATOR_URL}{path}")
    assert response.status_code == 401, (
        f"The dashboard API accepted a request without a token: {response.status_code} {response.text}"
    )
