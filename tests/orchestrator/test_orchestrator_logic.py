# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Artifact, Part, TaskArtifactUpdateEvent

import config
from common.a2a_contract import ArtifactName
from common.streaming import AgentActivityEvent, LogBatchEvent, TaskDoneEvent
from orchestrator.main import (
    AgentStatus,
    BrokenReason,
    _discover_agents,
    _discover_agents_with_startup_retry,
    _fetch_agent_card,
    _finalize_task,
    _get_agents_info,
    _get_file_contents_from_artifacts,
    _handle_stream_chunk,
    _health_check_agents,
    _LogStreamState,
    _save_agent_trace_from_task,
    _select_agent,
    agent_registry,
    cancellation_queue,
    discovery_agent,
)
from orchestrator.models import TaskRecord, TaskStatus, task_history


def _drain_queue(queue):
    while not queue.empty():
        queue.get_nowait()


@pytest.fixture
async def clear_registry():
    # Clear registry before/after test
    agent_registry._cards.clear()
    agent_registry._statuses.clear()
    agent_registry._broken_reasons.clear()
    agent_registry._stuck_task_ids.clear()
    _drain_queue(cancellation_queue)
    yield
    agent_registry._cards.clear()
    agent_registry._statuses.clear()
    agent_registry._broken_reasons.clear()
    agent_registry._stuck_task_ids.clear()
    _drain_queue(cancellation_queue)


@pytest.fixture
def mock_agent_card():
    return AgentCard(
        name="Discovered Agent",
        description="Desc",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        skills=[],
        default_input_modes=["text"],
        default_output_modes=["text"],
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url="http://localhost:8001")],
    )


@pytest.mark.asyncio
async def test_fetch_agent_card_success(mock_agent_card):
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        from google.protobuf.json_format import MessageToDict

        mock_response.json.return_value = MessageToDict(mock_agent_card, preserving_proto_field_name=True)
        mock_client.get.return_value = mock_response

        card = await _fetch_agent_card("http://localhost:8001")
        assert card.name == "Discovered Agent"


@pytest.mark.asyncio
async def test_fetch_agent_card_failure():
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        mock_client.get.side_effect = Exception("Connection error")

        card = await _fetch_agent_card("http://bad-url")
        assert card is None


@pytest.mark.asyncio
async def test_discover_agents_success(clear_registry, mock_agent_card):
    with (
        patch("config.OrchestratorConfig.REMOTE_EXECUTION_AGENT_HOSTS", "http://localhost"),
        patch("config.OrchestratorConfig.AGENT_DISCOVERY_PORTS", "8001-8001"),
        patch("orchestrator.main._fetch_agent_card", return_value=mock_agent_card),
    ):
        await _discover_agents()

        assert not await agent_registry.is_empty()
        cards = await agent_registry.get_all_cards()
        assert len(cards) == 1
        assert next(iter(cards.values())).name == "Discovered Agent"


@pytest.mark.asyncio
async def test_discover_agents_with_startup_retry_stops_once_converged(clear_registry, mock_agent_card):
    """A late-binding peer registers on attempt 2; attempt 3 finds nothing new and stops
    early instead of using the full attempt budget."""
    call_count = 0

    async def fake_discover():
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            await agent_registry.register("late-agent", mock_agent_card)

    with (
        patch("orchestrator.main._discover_agents", side_effect=fake_discover),
        patch("config.OrchestratorConfig.INITIAL_DISCOVERY_MAX_ATTEMPTS", 5),
        patch("config.OrchestratorConfig.INITIAL_DISCOVERY_RETRY_DELAY_SECONDS", 0),
    ):
        await _discover_agents_with_startup_retry()

    assert call_count == 3  # attempt 1: 0 -> attempt 2: 1 (new) -> attempt 3: 1 (converged, stop)
    assert len(await agent_registry.get_all_cards()) == 1


@pytest.mark.asyncio
async def test_discover_agents_with_startup_retry_gives_up_fast_when_empty(clear_registry):
    """Nothing is ever discoverable: two consecutive empty scans should stop early rather
    than burning the full attempt budget."""
    with (
        patch("orchestrator.main._discover_agents", new_callable=AsyncMock) as mock_discover,
        patch("config.OrchestratorConfig.INITIAL_DISCOVERY_MAX_ATTEMPTS", 5),
        patch("config.OrchestratorConfig.INITIAL_DISCOVERY_RETRY_DELAY_SECONDS", 0),
    ):
        await _discover_agents_with_startup_retry()

    assert mock_discover.call_count == 2


@pytest.mark.asyncio
async def test_select_agent(clear_registry, mock_agent_card):
    # Register an agent first
    await agent_registry.register("test-id", mock_agent_card)

    # Mock LLM response
    mock_result = MagicMock()
    mock_result.output.id = "test-id"

    # Mock discovery agent run
    with patch.object(discovery_agent, "run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result

        agent_id = await _select_agent("some task", ["test-id"])
        assert agent_id == "test-id"


@pytest.mark.asyncio
async def test_select_agent_none_found(clear_registry):
    agent_id = await _select_agent("some task", [])
    assert agent_id is None


@pytest.mark.asyncio
async def test_get_agents_info_includes_real_skill_descriptions(clear_registry):
    """The discovery-LLM prompt must carry each agent's real skill description, not a
    generic placeholder — task routing quality depends on this text (see AgentBase.get_skills)."""
    card = AgentCard(
        name="Some Agent",
        description="Desc",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        skills=[AgentSkill(id="s1", name="Skill", description="Does the important thing.", tags=["qa"])],
        default_input_modes=["text"],
        default_output_modes=["text"],
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url="http://localhost:8001")],
    )
    await agent_registry.register("test-id", card)

    info = await _get_agents_info(["test-id"])

    assert "Some Agent" in info
    assert "Does the important thing." in info


@pytest.mark.asyncio
async def test_discover_agents_skips_existing(clear_registry, mock_agent_card):
    # Pre-register the agent
    await agent_registry.register("existing-id", mock_agent_card)

    with (
        patch("config.OrchestratorConfig.REMOTE_EXECUTION_AGENT_HOSTS", "http://localhost"),
        patch("config.OrchestratorConfig.AGENT_DISCOVERY_PORTS", "8001-8001"),
        patch("orchestrator.main._fetch_agent_card", return_value=mock_agent_card) as mock_fetch,
        patch("orchestrator.main._check_agent_reachability", return_value=True) as mock_check,
    ):
        await _discover_agents()

        # A known URL is skipped: discovery neither fetches the card nor probes liveness.
        mock_fetch.assert_not_called()
        mock_check.assert_not_called()

        # Verify agent is still there
        assert await agent_registry.contains("existing-id")


@pytest.mark.asyncio
async def test_health_check_unreachable_marks_broken(clear_registry, mock_agent_card):
    await agent_registry.register("existing-id", mock_agent_card)

    with patch("orchestrator.main._check_agent_reachability", return_value=False) as mock_check:
        await _health_check_agents()

        mock_check.assert_called_once_with("http://localhost:8001")

        # Agent is marked BROKEN (OFFLINE), NOT removed.
        assert await agent_registry.contains("existing-id")
        assert await agent_registry.get_status("existing-id") == AgentStatus.BROKEN
        broken_reason, _ = await agent_registry.get_broken_context("existing-id")
        assert broken_reason == BrokenReason.OFFLINE

    # Agent was queued for the recovery worker.
    queued_agent_id, _ = cancellation_queue.get_nowait()
    assert queued_agent_id == "existing-id"


@pytest.mark.asyncio
async def test_health_check_reachable_leaves_available(clear_registry, mock_agent_card):
    await agent_registry.register("existing-id", mock_agent_card)

    with patch("orchestrator.main._check_agent_reachability", return_value=True):
        await _health_check_agents()

        assert await agent_registry.get_status("existing-id") == AgentStatus.AVAILABLE

    assert cancellation_queue.empty()


@pytest.mark.asyncio
async def test_health_check_skips_non_available(clear_registry, mock_agent_card):
    await agent_registry.register("existing-id", mock_agent_card)
    await agent_registry.update_status("existing-id", AgentStatus.BUSY)

    with patch("orchestrator.main._check_agent_reachability", return_value=False) as mock_check:
        await _health_check_agents()

        # BUSY agents are not probed by the health check.
        mock_check.assert_not_called()
        assert await agent_registry.get_status("existing-id") == AgentStatus.BUSY


@pytest.mark.asyncio
async def test_discover_agents_fetches_new(clear_registry, mock_agent_card):
    # Registry empty

    with (
        patch("config.OrchestratorConfig.REMOTE_EXECUTION_AGENT_HOSTS", "http://localhost"),
        patch("config.OrchestratorConfig.AGENT_DISCOVERY_PORTS", "8001-8001"),
        patch("orchestrator.main._fetch_agent_card", return_value=mock_agent_card) as mock_fetch,
    ):
        await _discover_agents()

        # Verify _fetch_agent_card WAS called
        mock_fetch.assert_called_once_with("http://localhost:8001")


# =============================================================================
# _handle_stream_chunk
# =============================================================================


def _text_artifact(name: str, text: str) -> Artifact:
    return Artifact(name=name, parts=[Part(text=text)])


def _raw_artifact(name: str, raw_data: bytes, artifact_id: str = "logs-123") -> Artifact:
    return Artifact(name=name, artifact_id=artifact_id, parts=[Part(raw=raw_data, media_type="text/plain")])


@pytest.mark.asyncio
async def test_handle_stream_chunk_log_batch_updates_history_and_publishes_to_agent():
    artifact = _raw_artifact("logs", b"line1\nline2", "logs-123")
    collected: list[Artifact] = []
    log_state = _LogStreamState()
    event = TaskArtifactUpdateEvent(
        context_id="ctx-1",
        task_id="task-1",
        artifact=artifact,
        last_chunk=False,
    )

    with (
        patch("orchestrator.main.task_history") as mock_history,
        patch("orchestrator.main.streaming_hub") as mock_hub,
    ):
        mock_history.append_log_batch = AsyncMock()
        mock_hub.publish_agent = AsyncMock()

        await _handle_stream_chunk(event, "task-1", "agent-1", collected, log_state)

    mock_history.append_log_batch.assert_called_once_with("task-1", ["line1", "line2"])
    mock_hub.publish_agent.assert_called_once()
    published = mock_hub.publish_agent.call_args[0][1]
    assert published["type"] == "log_batch"
    assert "line1" in published["lines"]
    assert log_state.artifact_id == "logs-123"
    assert log_state.lines == ["line1", "line2"]
    assert collected == []


@pytest.mark.asyncio
async def test_handle_stream_chunk_log_last_chunk_consolidates():
    collected: list[Artifact] = []
    log_state = _LogStreamState()

    # Chunk 1
    artifact_1 = _raw_artifact("logs", b"line1\n", "logs-123")
    event_1 = TaskArtifactUpdateEvent(
        context_id="ctx-1",
        task_id="task-1",
        artifact=artifact_1,
        last_chunk=False,
    )
    with (
        patch("orchestrator.main.task_history") as mock_history,
        patch("orchestrator.main.streaming_hub") as mock_hub,
    ):
        mock_history.append_log_batch = AsyncMock()
        mock_hub.publish_agent = AsyncMock()
        await _handle_stream_chunk(event_1, "task-1", "agent-1", collected, log_state)

    # Chunk 2 (last)
    artifact_2 = _raw_artifact("logs", b"line2", "logs-123")
    event_2 = TaskArtifactUpdateEvent(
        context_id="ctx-1",
        task_id="task-1",
        artifact=artifact_2,
        last_chunk=True,
    )
    with (
        patch("orchestrator.main.task_history") as mock_history,
        patch("orchestrator.main.streaming_hub") as mock_hub,
    ):
        mock_history.append_log_batch = AsyncMock()
        mock_hub.publish_agent = AsyncMock()
        await _handle_stream_chunk(event_2, "task-1", "agent-1", collected, log_state)

    # Shoud have created a consolidated logs artifact
    assert len(collected) == 1
    consolidated = collected[0]
    assert consolidated.name == "logs"
    assert len(consolidated.parts) == 1
    assert consolidated.parts[0].filename == "execution_logs.txt"
    assert consolidated.parts[0].raw == b"line1\nline2"


@pytest.mark.asyncio
async def test_handle_stream_chunk_other_artifact_appended_to_collected():
    artifact = _text_artifact("agent_execution_result", '{"result": "ok"}')
    collected: list[Artifact] = []
    log_state = _LogStreamState()
    event = TaskArtifactUpdateEvent(
        context_id="ctx-1",
        task_id="task-1",
        artifact=artifact,
        last_chunk=False,
    )

    with (
        patch("orchestrator.main.task_history"),
        patch("orchestrator.main.streaming_hub"),
    ):
        await _handle_stream_chunk(event, "task-1", "agent-1", collected, log_state)

    assert len(collected) == 1
    assert collected[0] == artifact


# =============================================================================
# _finalize_task
# =============================================================================


@pytest.mark.asyncio
async def test_finalize_task_updates_history_and_publishes_task_done():
    with (
        patch("orchestrator.main.task_history") as mock_history,
        patch("orchestrator.main.streaming_hub") as mock_hub,
    ):
        mock_history.update = AsyncMock()
        mock_history.clear_current_activity = AsyncMock()
        mock_hub.publish_global = AsyncMock()
        mock_hub.publish_agent = AsyncMock()

        await _finalize_task("task-1", "agent-1", TaskStatus.COMPLETED)

    mock_history.update.assert_called_once()
    update_args = mock_history.update.call_args[0]
    assert update_args[0] == "task-1"
    assert update_args[1] == TaskStatus.COMPLETED

    mock_history.clear_current_activity.assert_called_once_with("task-1")

    mock_hub.publish_global.assert_called_once()
    global_event = mock_hub.publish_global.call_args[0][0]
    assert global_event["type"] == "task_done"
    assert global_event["task_id"] == "task-1"
    assert global_event["agent_id"] == "agent-1"
    assert global_event["status"] == TaskStatus.COMPLETED.value

    mock_hub.publish_agent.assert_called_once()
    agent_event = mock_hub.publish_agent.call_args[0][1]
    assert agent_event["type"] == "task_done"


@pytest.mark.asyncio
async def test_finalize_task_includes_error_message_when_provided():
    with (
        patch("orchestrator.main.task_history") as mock_history,
        patch("orchestrator.main.streaming_hub") as mock_hub,
    ):
        mock_history.update = AsyncMock()
        mock_history.clear_current_activity = AsyncMock()
        mock_hub.publish_global = AsyncMock()
        mock_hub.publish_agent = AsyncMock()

        await _finalize_task("task-1", "agent-1", TaskStatus.FAILED, "something broke")

    global_event = mock_hub.publish_global.call_args[0][0]
    assert global_event["error_message"] == "something broke"


# ---------------------------------------------------------------------------
# Trace artifact: exclusion from file forwarding + orchestrator-side storage
# ---------------------------------------------------------------------------


def test_get_file_contents_from_artifacts_excludes_usage_and_trace():
    """Bookkeeping artifacts (USAGE, TRACE) must never be forwarded as file parts —
    e.g. into the incident-creation agent's context on the test-execution path."""
    usage_artifact = Artifact(
        name=ArtifactName.USAGE, parts=[Part(raw=b"{}", media_type="application/json", filename="usage.json")]
    )
    trace_artifact = Artifact(
        name=ArtifactName.TRACE, parts=[Part(raw=b"[]", media_type="application/json", filename="trace.json")]
    )
    payload_artifact = Artifact(
        name=ArtifactName.EXECUTION_RESULT,
        parts=[Part(raw=b"payload", media_type="text/plain", filename="result.txt")],
    )

    files = _get_file_contents_from_artifacts([usage_artifact, trace_artifact, payload_artifact])

    assert len(files) == 1
    assert files[0].name == "result.txt"


@pytest.mark.asyncio
async def test_save_agent_trace_from_task_stores_trace():
    record = TaskRecord(
        task_id="trace-task-1",
        agent_id="agent-1",
        agent_name="Test Agent",
        description="desc",
        status=TaskStatus.RUNNING,
        start_time=datetime.now(),
    )
    await task_history.add(record)

    trace_artifact = Artifact(
        name=ArtifactName.TRACE,
        parts=[Part(raw=b'[{"parts": []}]', media_type="application/json", filename="trace.json")],
    )
    task = MagicMock(artifacts=[trace_artifact])

    await _save_agent_trace_from_task(task, "trace-task-1")

    stored = await task_history.get_by_id("trace-task-1")
    assert stored.trace_json == '[{"parts": []}]'
