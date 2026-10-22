# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

import asyncio
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.types import Artifact, Part, Task, TaskState, TaskStatus
from fastapi.testclient import TestClient

import config
import orchestrator.main as main_module
from common.streaming import SnapshotEvent
from orchestrator.auth import auth_service
from orchestrator.main import (
    _build_snapshot,
    _mint_stream_token,
    _sse_hub_events,
    _validate_api_key,
    _validate_stream_token,
    orchestrator_app,
)

client = TestClient(orchestrator_app)


# Override auth dependency
def mock_validate_api_key():
    pass


orchestrator_app.dependency_overrides[_validate_api_key] = mock_validate_api_key


@pytest.fixture
def mock_task_completed():
    task = MagicMock(spec=Task)
    task.status = TaskStatus(state=TaskState.TASK_STATE_COMPLETED)
    task.artifacts = [Artifact(name="art-1", parts=[Part(text='{"message": "success"}')])]
    return task


@pytest.mark.asyncio
@pytest.mark.parametrize("requirement_review_enabled", [False, True])
async def test_trigger_test_case_generation_workflow(requirement_review_enabled):
    mock_queue = MagicMock()
    mock_queue.put = AsyncMock()
    mock_queue.qsize.return_value = 1
    with (
        patch("config.DesignConfig.REQUIREMENT_REVIEW_ENABLED", requirement_review_enabled),
        patch("orchestrator.main._tc_gen_webhook_guard.check", AsyncMock(return_value=None)),
        patch("orchestrator.main._pipeline_queue", mock_queue),
    ):
        response = client.post(
            "/story-ready-for-test-case-generation",
            json=_requirement_review_event(
                work_item_id="1000000017",
                project_key="example-space",
                feishu_link="https://example.feishu.cn/wiki/ExampleDocumentToken16",
            ),
        )

    assert response.status_code == 202
    mock_queue.put.assert_awaited_once()
    (enqueued_job,) = mock_queue.put.await_args.args
    assert enqueued_job.func is main_module._run_pipeline
    assert enqueued_job.args[0] == "1000000017"
    assert enqueued_job.args[2] == "https://example.feishu.cn/wiki/ExampleDocumentToken16"


def _requirement_review_event(
    work_item_id: str = "WI-1", project_key: str = "PROJ", feishu_link: str | None = "https://example.feishu.cn/docx/x"
) -> dict:
    """Build a minimal Feishu Project native WorkFlowNodeStatusEvent payload."""
    node_form = [{"field_type_key": "link", "field_value": feishu_link}] if feishu_link else []
    return {
        "payload": {
            "id": work_item_id,
            "project_simple_name": project_key,
            "nodes": [{"node_form": node_form}],
        }
    }


@pytest.mark.asyncio
async def test_requirement_review_webhook_enqueues_on_valid_payload():
    mock_queue = MagicMock()
    mock_queue.put = AsyncMock()
    mock_queue.qsize.return_value = 1
    mock_guard = MagicMock()
    mock_guard.check = AsyncMock(return_value=None)
    with (
        patch("config.DesignConfig.REQUIREMENT_REVIEW_ENABLED", True),
        patch("orchestrator.main._pipeline_queue", mock_queue),
        patch("orchestrator.main._requirement_review_guard", mock_guard),
    ):
        response = client.post("/requirement-ready-for-review", json=_requirement_review_event())

    assert response.status_code == 202
    mock_queue.put.assert_awaited_once()
    (enqueued_job,) = mock_queue.put.await_args.args
    assert enqueued_job.func is main_module._run_requirement_review
    assert enqueued_job.args == ("WI-1", "PROJ", "https://example.feishu.cn/docx/x")


@pytest.mark.asyncio
async def test_requirement_review_webhook_drops_non_dict_body():
    mock_queue = MagicMock()
    mock_queue.put = AsyncMock()
    with (
        patch("config.DesignConfig.REQUIREMENT_REVIEW_ENABLED", True),
        patch("orchestrator.main._pipeline_queue", mock_queue),
    ):
        response = client.post("/requirement-ready-for-review", json=["not", "a", "dict"])

    assert response.status_code == 204
    mock_queue.put.assert_not_awaited()


@pytest.mark.asyncio
async def test_requirement_review_webhook_drops_missing_work_item_id():
    mock_queue = MagicMock()
    mock_queue.put = AsyncMock()
    payload = _requirement_review_event()
    payload["payload"]["id"] = ""
    with (
        patch("config.DesignConfig.REQUIREMENT_REVIEW_ENABLED", True),
        patch("orchestrator.main._pipeline_queue", mock_queue),
    ):
        response = client.post("/requirement-ready-for-review", json=payload)

    assert response.status_code == 204
    mock_queue.put.assert_not_awaited()


@pytest.mark.asyncio
async def test_requirement_review_webhook_drops_missing_prd_link():
    mock_queue = MagicMock()
    mock_queue.put = AsyncMock()
    with (
        patch("config.DesignConfig.REQUIREMENT_REVIEW_ENABLED", True),
        patch("orchestrator.main._pipeline_queue", mock_queue),
    ):
        response = client.post("/requirement-ready-for-review", json=_requirement_review_event(feishu_link=None))

    assert response.status_code == 204
    mock_queue.put.assert_not_awaited()


@pytest.mark.asyncio
async def test_requirement_review_webhook_drops_duplicate():
    mock_queue = MagicMock()
    mock_queue.put = AsyncMock()
    mock_guard = MagicMock()
    mock_guard.check = AsyncMock(return_value="duplicate")
    with (
        patch("config.DesignConfig.REQUIREMENT_REVIEW_ENABLED", True),
        patch("orchestrator.main._pipeline_queue", mock_queue),
        patch("orchestrator.main._requirement_review_guard", mock_guard),
    ):
        response = client.post("/requirement-ready-for-review", json=_requirement_review_event())

    assert response.status_code == 204
    mock_queue.put.assert_not_awaited()


@pytest.mark.asyncio
async def test_requirement_review_webhook_drops_when_rate_limited():
    mock_queue = MagicMock()
    mock_queue.put = AsyncMock()
    mock_guard = MagicMock()
    mock_guard.check = AsyncMock(return_value="rate_limited")
    with (
        patch("config.DesignConfig.REQUIREMENT_REVIEW_ENABLED", True),
        patch("orchestrator.main._pipeline_queue", mock_queue),
        patch("orchestrator.main._requirement_review_guard", mock_guard),
    ):
        response = client.post("/requirement-ready-for-review", json=_requirement_review_event())

    assert response.status_code == 204
    mock_queue.put.assert_not_awaited()


@pytest.mark.asyncio
async def test_requirement_review_webhook_requires_no_api_key():
    """The endpoint must not depend on _validate_api_key at all (Feishu can't send custom headers)."""
    mock_guard = MagicMock()
    mock_guard.check = AsyncMock(return_value="duplicate")  # short-circuit before any real dispatch
    with (
        patch("config.DesignConfig.REQUIREMENT_REVIEW_ENABLED", True),
        patch.dict(orchestrator_app.dependency_overrides, {}, clear=True),
        patch("orchestrator.main._requirement_review_guard", mock_guard),
    ):
        response = client.post("/requirement-ready-for-review", json=_requirement_review_event())

    assert response.status_code == 204


@pytest.mark.asyncio
async def test_execute_tests_endpoint():
    # This is complex. It fetches TCs, groups them, executes them, generates report.
    # We'll mock the high level functions.

    with (
        patch("orchestrator.main.get_test_management_client") as mock_get_client,
        patch("orchestrator.main._group_test_cases_by_labels", new_callable=AsyncMock) as mock_group,
        patch("orchestrator.main._request_all_test_cases_execution", new_callable=AsyncMock) as mock_exec,
        patch("orchestrator.main._generate_test_report", new_callable=AsyncMock) as mock_report,
    ):
        mock_tm_client = MagicMock()
        mock_get_client.return_value = mock_tm_client
        mock_tm_client.fetch_ready_for_execution_test_cases_by_labels.return_value = {
            config.OrchestratorConfig.AUTOMATED_TC_LABEL: [MagicMock()]
        }

        mock_group.return_value = {"UI": [MagicMock()]}
        mock_exec.return_value = [MagicMock()]  # results

        response = client.post("/execute-tests", json={"project_key": "PROJ"})

        assert response.status_code == 200
        mock_exec.assert_called_once()
        mock_report.assert_called_once()


# =============================================================================
# POST /api/dashboard/stream-token
# =============================================================================


def _valid_bearer_header() -> dict:
    token = auth_service.create_token("test-user").access_token
    return {"Authorization": f"Bearer {token}"}


def test_stream_token_without_jwt_returns_401():
    response = client.post("/api/dashboard/stream-token")
    assert response.status_code == 401


def test_stream_token_with_valid_jwt_returns_token_and_expiry():
    response = client.post("/api/dashboard/stream-token", headers=_valid_bearer_header())
    assert response.status_code == 200
    body = response.json()
    assert "stream_token" in body
    assert "expires_at" in body
    assert len(body["stream_token"]) > 10


# =============================================================================
# GET /api/dashboard/stream — auth guard
# =============================================================================


def test_global_sse_stream_without_token_returns_422():
    # stream_token query param is required; missing → 422 Unprocessable Entity
    response = client.get("/api/dashboard/stream")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_global_sse_stream_with_invalid_token_returns_401():
    response = client.get("/api/dashboard/stream", params={"stream_token": "bogus-token"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_global_sse_stream_with_expired_token_returns_401():
    token, _ = await _mint_stream_token("test-user")
    # Forcibly expire the token by backdating it in the store
    from orchestrator.main import _stream_token_store

    _stream_token_store[token] = ("test-user", datetime.now(UTC) - timedelta(seconds=1))

    response = client.get("/api/dashboard/stream", params={"stream_token": token})
    assert response.status_code == 401


# =============================================================================
# _build_snapshot — schema check
# =============================================================================


@pytest.mark.asyncio
async def test_build_snapshot_returns_valid_snapshot_event():
    with (
        patch("orchestrator.main.agent_registry") as mock_registry,
        patch("orchestrator.main.task_history") as mock_history,
    ):
        mock_registry.get_all_cards = AsyncMock(return_value={})
        mock_history.get_all = AsyncMock(return_value=[])

        snapshot = await _build_snapshot()

    assert isinstance(snapshot, SnapshotEvent)
    assert snapshot.version == 1
    assert isinstance(snapshot.agents, list)
    assert isinstance(snapshot.running_tasks, list)


# =============================================================================
# _sse_hub_events — live event forwarding and auth-error frame
# =============================================================================


class _FakeSubscriber:
    """Minimal stand-in for streaming_hub._Subscriber exposing async get()."""

    def __init__(self, events: list[dict]) -> None:
        self._events = list(events)

    async def get(self) -> dict:
        if self._events:
            return self._events.pop(0)
        await asyncio.sleep(100)  # block once exhausted


@pytest.mark.asyncio
async def test_sse_hub_events_forwards_live_event():
    live = {"type": "agent_activity", "task_id": "t1", "agent_id": "a1", "text": "working", "version": 1}

    expires = datetime.now(UTC) + timedelta(minutes=5)
    events = []

    async for sse in _sse_hub_events(_FakeSubscriber([live]), expires):
        events.append(sse)
        break  # close generator after first frame

    assert len(events) == 1
    assert events[0].event == "agent_activity"
    data = json.loads(events[0].data)
    assert data["task_id"] == "t1"
    assert data["text"] == "working"


@pytest.mark.asyncio
async def test_sse_hub_events_emits_auth_error_when_token_expired():
    expired = datetime.now(UTC) - timedelta(seconds=1)
    events = []

    async for sse in _sse_hub_events(_FakeSubscriber([]), expired):
        events.append(sse)

    assert len(events) == 1
    assert events[0].event == "auth_error"


@pytest.mark.asyncio
async def test_sse_hub_events_emits_auth_error_even_with_incoming_events():
    expired = datetime.now(UTC) - timedelta(seconds=1)
    live = {"type": "agent_activity", "task_id": "t1", "agent_id": "a1", "text": "working", "version": 1}
    events = []

    async for sse in _sse_hub_events(_FakeSubscriber([live]), expired):
        events.append(sse)

    assert len(events) == 1
    assert events[0].event == "auth_error"


def _dashboard_auth_header(monkeypatch) -> dict[str, str]:
    monkeypatch.setattr(config.DashboardAuthConfig, "USERNAME", "admin")
    monkeypatch.setattr(config.DashboardAuthConfig, "PASSWORD", "s3cret")
    monkeypatch.setattr(config.DashboardAuthConfig, "JWT_SECRET", "a-secret")
    token = auth_service.create_token("test-user").access_token
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_get_task_trace_returns_stored_trace(monkeypatch):
    headers = _dashboard_auth_header(monkeypatch)
    with patch(
        "orchestrator.dashboard_service.dashboard_service.get_task_trace", new_callable=AsyncMock
    ) as mock_get_trace:
        mock_get_trace.return_value = '[{"parts": []}]'

        response = client.get("/api/dashboard/tasks/some-task-id/trace", headers=headers)

        assert response.status_code == 200
        assert response.json() == [{"parts": []}]
        mock_get_trace.assert_called_once_with("some-task-id")


@pytest.mark.asyncio
async def test_get_task_trace_404_when_not_found(monkeypatch):
    headers = _dashboard_auth_header(monkeypatch)
    with patch(
        "orchestrator.dashboard_service.dashboard_service.get_task_trace", new_callable=AsyncMock
    ) as mock_get_trace:
        mock_get_trace.return_value = None

        response = client.get("/api/dashboard/tasks/unknown-task/trace", headers=headers)

        assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_task_trace_requires_auth():
    response = client.get("/api/dashboard/tasks/some-task-id/trace")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_disabled_requirement_review_does_not_queue_or_consume_guard():
    with (
        patch("config.DesignConfig.REQUIREMENT_REVIEW_ENABLED", False),
        patch("orchestrator.main._pipeline_queue") as queue,
        patch("orchestrator.main._requirement_review_guard") as guard,
    ):
        response = client.post("/requirement-ready-for-review", json=_requirement_review_event())
    assert response.status_code == 204
    queue.put.assert_not_called()
    guard.check.assert_not_called()
