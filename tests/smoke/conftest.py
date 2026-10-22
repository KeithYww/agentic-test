# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Fixtures for the hermetic smoke suite.

The suite runs against the ``docker-compose.smoke.yml`` topology: a real
orchestrator and the agents (driven by the real Photon API relay), with the external
boundaries (Jira MCP, Jira REST, Zephyr, Qdrant + embedding) replaced by
recording mocks. The fixtures wait for all agents to register, then fire the
four webhooks once, concurrently, for correctness independence between flows —
though requirements review and the test-case flow share the orchestrator's single
pipeline queue, so one can still delay the other's start (see the RECORD_POLL_TIMEOUT
note in test_smoke.py). The test functions read the mocks' ``/__recorded`` endpoints
and assert on what reached each boundary.

URLs default to the published compose ports and are overridable via ``SMOKE_*``
env vars. The dashboard/API credentials are the fixed throwaway values baked into
``docker-compose.smoke.yml``.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

import config

ORCHESTRATOR_URL = os.environ.get("SMOKE_ORCHESTRATOR_URL", "http://localhost:8000").rstrip("/")
JIRA_MCP_RECORDED_URL = os.environ.get("SMOKE_JIRA_MCP_RECORDED_URL", "http://localhost:9000/__recorded")
ZEPHYR_RECORDED_URL = os.environ.get("SMOKE_ZEPHYR_RECORDED_URL", "http://localhost:8090/__recorded")
QDRANT_RECORDED_URL = os.environ.get("SMOKE_QDRANT_RECORDED_URL", "http://localhost:6333/__recorded")
MEEGO_RECORDED_URL = os.environ.get("SMOKE_MEEGO_RECORDED_URL", "http://localhost:8095/__recorded")
FEISHU_MCP_RECORDED_URL = os.environ.get("SMOKE_FEISHU_MCP_RECORDED_URL", "http://localhost:8092/__recorded")
OTEL_COLLECTOR_RECORDED_URL = os.environ.get("SMOKE_OTEL_COLLECTOR_RECORDED_URL", "http://localhost:9020/__recorded")

# Fixed test credentials, matching docker-compose.smoke.yml.
ORCHESTRATOR_API_KEY = "smoke-api-key"
DASHBOARD_USERNAME = "smoke"
DASHBOARD_PASSWORD = "smoke-pass"

# The story seeded by the Jira MCP mock (jira_mcp_mock.SEEDED_ISSUE_KEY) and its numeric id.
SEEDED_ISSUE_KEY = "SMOKE-1"
SEEDED_ISSUE_ID = 10001
# The project the executable test case is seeded under (zephyr_mock + jira_mcp_mock).
SEEDED_PROJECT_KEY = "SMOKE"
# Feishu Project story reference used by the /story-ready-for-test-case-generation flow.
SEEDED_STORY_ID = "SMOKE-STORY-1"
# Feishu Project work item ID used by the /requirement-ready-for-review flow.
SEEDED_REVIEW_WORK_ITEM_ID = "1000000011"
# Feishu document URL passed to the generation agent; the feishu_mcp_mock returns canned content for any URL.
SMOKE_FEISHU_DOC_URL = os.environ.get("SMOKE_FEISHU_DOC_URL", "https://example.feishu.cn/wiki/SMOKE-DOC-1")
# Feishu document URL passed to the requirements-review agent; kept distinct from SMOKE_FEISHU_DOC_URL so the
# feishu_mcp_mock's recorded "fetched_docs" can attribute a fetch to the review flow specifically.
SMOKE_REQUIREMENT_REVIEW_DOC_URL = os.environ.get(
    "SMOKE_REQUIREMENT_REVIEW_DOC_URL", "https://example.feishu.cn/wiki/SMOKE-REVIEW-DOC-1"
)
# The ready-for-execution test case seeded by the Zephyr mock (zephyr_mock._EXECUTABLE_TC_KEY).
SEEDED_EXECUTABLE_TC_KEY = "SMOKE-T100"
# The collection the RAG sync stores Jira issues in; tracks config as the source of truth.
TICKETS_COLLECTION_NAME = config.QdrantConfig.TICKETS_COLLECTION_NAME
# Name the mock executor registers under; must match mocks/execution_agent.EXECUTION_AGENT_NAME.
EXECUTION_AGENT_NAME = "Smoke API Test Executor"

# Canonical agent names the four agents register under; tracks config as the source of truth.
EXPECTED_AGENT_NAMES: set[str] = {
    config.RequirementsReviewAgentConfig.OWN_NAME,
    config.TestCaseGenerationAgentConfig.OWN_NAME,
    config.TestCaseClassificationAgentConfig.OWN_NAME,
    config.TestCaseReviewAgentConfig.OWN_NAME,
}
HEALTHY_AGENT_STATUSES = {"AVAILABLE", "BUSY"}

# Agents the /execute-tests + incident-creation flow needs (beyond the core four).
EXECUTION_FLOW_AGENT_NAMES: set[str] = {EXECUTION_AGENT_NAME, config.IncidentCreationAgentConfig.OWN_NAME}

# The orchestrator startup + initial agent discovery can take a while to settle.
ORCHESTRATOR_READY_TIMEOUT = 120.0
AGENT_READY_TIMEOUT = 240.0
# A single webhook drives real LLM routing plus one or more full agent runs.
WEBHOOK_TIMEOUT = httpx.Timeout(1200.0)
POLL_INTERVAL = 5.0


@pytest.fixture(scope="session")
def http_client() -> httpx.Client:
    with httpx.Client(timeout=httpx.Timeout(60.0), follow_redirects=True) as client:
        yield client


@pytest.fixture(scope="session")
def auth_headers(http_client: httpx.Client) -> dict[str, str]:
    """Log in to the dashboard, retrying while the orchestrator is still starting."""
    deadline = time.monotonic() + ORCHESTRATOR_READY_TIMEOUT
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            response = http_client.post(
                f"{ORCHESTRATOR_URL}/api/auth/login",
                json={"username": DASHBOARD_USERNAME, "password": DASHBOARD_PASSWORD},
            )
            if response.status_code == 200:
                return {"Authorization": f"Bearer {response.json()['access_token']}"}
            last_error = f"{response.status_code} {response.text}"
        except httpx.TransportError as exc:
            last_error = str(exc)
        time.sleep(POLL_INTERVAL)
    pytest.fail(f"Orchestrator dashboard login did not succeed within {ORCHESTRATOR_READY_TIMEOUT}s: {last_error}")


@pytest.fixture(scope="session")
def webhook_headers() -> dict[str, str]:
    return {"X-API-Key": ORCHESTRATOR_API_KEY}


def _wait_for_agents_healthy(
    http_client: httpx.Client, auth_headers: dict[str, str], expected_names: set[str]
) -> None:
    """Wait until every expected agent is registered and healthy.

    Triggers a fresh discovery each cycle so the wait does not depend on the
    orchestrator's periodic discovery interval.
    """
    deadline = time.monotonic() + AGENT_READY_TIMEOUT
    registered: dict[str, str] = {}
    while time.monotonic() < deadline:
        http_client.post(f"{ORCHESTRATOR_URL}/api/dashboard/discovery", headers=auth_headers)
        response = http_client.get(f"{ORCHESTRATOR_URL}/api/dashboard/agents", headers=auth_headers)
        if response.status_code == 200:
            registered = {agent["name"]: agent["status"] for agent in response.json()}
            healthy = {name for name in expected_names if registered.get(name) in HEALTHY_AGENT_STATUSES}
            if healthy == expected_names:
                return
        time.sleep(POLL_INTERVAL)
    missing = expected_names - {n for n, s in registered.items() if s in HEALTHY_AGENT_STATUSES}
    pytest.fail(f"Agents not registered/healthy within {AGENT_READY_TIMEOUT}s. Missing: {missing}. Seen: {registered}")


@pytest.fixture(scope="session")
def all_agents_ready(http_client: httpx.Client, auth_headers: dict[str, str]) -> None:
    """Wait once until every agent the four flows need is registered and healthy."""
    _wait_for_agents_healthy(http_client, auth_headers, EXPECTED_AGENT_NAMES | EXECUTION_FLOW_AGENT_NAMES)


def _post_webhook(path: str, headers: dict[str, str], payload: dict[str, object]) -> httpx.Response:
    with httpx.Client(timeout=WEBHOOK_TIMEOUT, follow_redirects=True) as client:
        return client.post(f"{ORCHESTRATOR_URL}{path}", headers=headers, json=payload)


def _requirement_review_event(work_item_id: str, project_key: str, feishu_doc: str) -> dict:
    """Build a Feishu Project native WorkFlowNodeStatusEvent payload, matching what the
    "send HTTP request" automation action actually sends (see orchestrator/main.py's
    _extract_requirement_review_ref)."""
    return {
        "payload": {
            "id": work_item_id,
            "project_simple_name": project_key,
            "nodes": [{"node_form": [{"field_type_key": "link", "field_value": feishu_doc}]}],
        }
    }


# The four flows are correctness-independent: requirements review writes a comment to the reviewed
# work item in Feishu Project; the test-case flow's cases end at "Review Complete" and never become
# executable; /execute-tests selects only the seeded Approved + "automated" case; the RAG sync
# involves no agent at all. So they can safely run concurrently without interfering with each
# other's results. They are NOT fully wall-clock independent, though: requirements review and the
# test-case flow both run through the orchestrator's single shared _pipeline_queue/_pipeline_consumer,
# so whichever is enqueued first can delay the other's start by its own full run time.
# Each entry is (path, payload, needs_auth) — the requirement-review webhook is unauthenticated by
# design (Feishu's automation action cannot attach custom headers), unlike the other three.
_WEBHOOKS: dict[str, tuple[str, dict[str, object], bool]] = {
    "requirements_review": (
        "/requirement-ready-for-review",
        _requirement_review_event(SEEDED_REVIEW_WORK_ITEM_ID, SEEDED_PROJECT_KEY, SMOKE_REQUIREMENT_REVIEW_DOC_URL),
        False,
    ),
    "test_case_flow": (
        "/story-ready-for-test-case-generation",
        {"story_id": SEEDED_STORY_ID, "project_key": SEEDED_PROJECT_KEY, "feishu_doc": SMOKE_FEISHU_DOC_URL},
        True,
    ),
    "execute_tests": ("/execute-tests", {"project_key": SEEDED_PROJECT_KEY}, True),
    "update_rag_db": ("/update-rag-db", {"project_key": SEEDED_PROJECT_KEY}, True),
}


@pytest.fixture(scope="session")
def webhook_responses(all_agents_ready: None, webhook_headers: dict[str, str]) -> dict[str, httpx.Response]:
    """Fire all four webhooks once, concurrently, and share the responses.

    The test-case workflow and the requirement-review webhook both return after being accepted
    for background processing; the other webhooks return after their flows complete. Posting them
    from a thread pool still minimizes the suite's wall time.
    """
    with ThreadPoolExecutor(max_workers=len(_WEBHOOKS)) as pool:
        futures = {
            name: pool.submit(_post_webhook, path, webhook_headers if needs_auth else {}, payload)
            for name, (path, payload, needs_auth) in _WEBHOOKS.items()
        }
        return {name: future.result() for name, future in futures.items()}


@pytest.fixture(scope="session")
def requirements_review_response(webhook_responses: dict[str, httpx.Response]) -> httpx.Response:
    return webhook_responses["requirements_review"]


@pytest.fixture(scope="session")
def test_case_flow_response(webhook_responses: dict[str, httpx.Response]) -> httpx.Response:
    return webhook_responses["test_case_flow"]


@pytest.fixture(scope="session")
def execute_tests_response(webhook_responses: dict[str, httpx.Response]) -> httpx.Response:
    return webhook_responses["execute_tests"]


@pytest.fixture(scope="session")
def update_rag_db_response(webhook_responses: dict[str, httpx.Response]) -> httpx.Response:
    return webhook_responses["update_rag_db"]
