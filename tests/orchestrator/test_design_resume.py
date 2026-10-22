# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Authenticated web recovery uses real SQLite checkpoints and queues only one generation."""

import asyncio

import httpx
import pytest
import pytest_asyncio

import config
import orchestrator.main as main
from common.design_policy import workflow_fingerprint
from common.design_store import DesignStore
from orchestrator.auth import auth_service


@pytest_asyncio.fixture
async def recovery(tmp_path, monkeypatch):
    owner = DesignStore(str(tmp_path))
    await owner.call("open")
    monkeypatch.setattr(main.design_store, "store", owner)
    monkeypatch.setattr(config, "DASHBOARD_PUBLIC_URL", "https://qa.example.com")
    queue = asyncio.Queue()
    monkeypatch.setattr(main, "_pipeline_queue", queue)
    run, _ = await owner.call("create_run", "P:S", {
        "story_id": "S", "project_key": "P", "story_title": "恢复测试", "feishu_doc": "https://example/doc",
        "workflow": workflow_fingerprint(),
    }, 6000000, 3600)
    await owner.call("snapshot", run["id"], 1, {"document": "private PRD must not appear in API", "published_url": "https://example/base"})
    permit = await owner.call("reserve", run["id"], 1, "generate", "input", 1000, 6000000)
    await owner.call("settle", run["id"], 1, permit["id"], 100, {"paid_result": True}, True)
    await owner.call("put", run["id"], 1, "unit", "generate", "input", {"paid_result": True})
    await owner.call("state", run["id"], 1, "paused", "ProviderBusy")
    token = auth_service.create_token(config.DashboardAuthConfig.USERNAME).access_token
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.orchestrator_app), base_url="http://test") as client:
        client.headers["Authorization"] = f"Bearer {token}"
        yield owner, queue, run, client
    await owner.close()


@pytest.mark.asyncio
async def test_confirm_resume_preserves_paid_results_and_concurrent_clicks_queue_once(recovery):
    owner, queue, run, client = recovery
    path = f"/api/dashboard/design-runs/{run['id']}"
    detail = await client.get(path)
    assert detail.status_code == 200
    assert detail.json()["can_resume"] is True
    assert detail.json()["completed_units"] == 1
    assert "private PRD" not in detail.text
    assert queue.empty()  # Merely opening or previewing a card cannot execute a task.
    responses = await asyncio.gather(*(client.post(path + "/resume", json={"generation": 1}) for _ in range(2)))
    assert sorted(r.status_code for r in responses) == [202, 409]
    assert queue.qsize() == 1
    job = queue.get_nowait()
    assert job.func is main._run_pipeline
    resumed = job.args[-1]
    assert resumed["generation"] == 2 and resumed["used"] == 100
    assert resumed["deadline"] == run["deadline"] and resumed["token_limit"] == run["token_limit"]
    assert resumed["snapshot"]["document"] == "private PRD must not appear in API"
    assert (await owner.call("get", run["id"], "unit", "generate", "input"))["value"] == {"paid_result": True}
    replay = await owner.call("reserve", run["id"], 2, "generate", "input", 1000, 6000000)
    assert replay["response"] == {"paid_result": True}
    # A delayed duplicate must not restart a later paused generation.
    await owner.call("state", run["id"], 2, "paused", "later_failure")
    assert (await client.post(path + "/resume", json={"generation": 1})).status_code == 409
    assert queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked", ["disabled", "model_changed", "unknown", "publish", "expired", "budget", "completed"])
async def test_blocked_resume_does_not_mutate_or_enqueue(recovery, monkeypatch, blocked):
    owner, queue, run, client = recovery
    if blocked == "disabled":
        monkeypatch.setattr(config, "DASHBOARD_PUBLIC_URL", "")
    elif blocked == "model_changed":
        monkeypatch.setattr(config.TestCaseGenerationAgentConfig, "REPAIR_MODEL_NAME", "changed-model")
    elif blocked in {"unknown", "publish", "completed"}:
        # Recreate an interrupted external operation before the pause.
        await owner.call("request_resume", run["id"], "test", "setup")
        active = await owner.call("claim", run["id"])
        if blocked == "unknown":
            permit = await owner.call("reserve", run["id"], 2, "repair", "input", 1000, 6000000)
            await owner.call("unknown", run["id"], 2, permit["id"])
        elif blocked == "publish":
            await owner.call("publish_begin", run["id"], 2, "create-base", {"title": "test"})
        await owner.call("state", run["id"], active["generation"], "completed" if blocked == "completed" else "paused")
    elif blocked == "expired":
        monkeypatch.setattr(main.time, "time", lambda: run["deadline"] + 1)
    elif blocked == "budget":
        # Spend the remaining budget without altering or resetting the ledger.
        await owner.call("request_resume", run["id"], "test", "setup")
        await owner.call("claim", run["id"])
        permit = await owner.call("reserve", run["id"], 2, "repair", "input", 5999900, 6000000)
        await owner.call("settle", run["id"], 2, permit["id"], 5999900, {}, False)
        await owner.call("state", run["id"], 2, "paused")
    before = await owner.call("run", run["id"])
    path = f"/api/dashboard/design-runs/{run['id']}"
    detail = (await client.get(path)).json()
    assert detail["can_resume"] is False and detail["blocked_reason"]
    response = await client.post(path + "/resume", json={"generation": before["generation"]})
    assert response.status_code == 409
    after = await owner.call("run", run["id"])
    assert after == before and queue.empty()


@pytest.mark.asyncio
async def test_resume_requires_dashboard_login_and_missing_run_is_404(recovery):
    _, queue, run, client = recovery
    path = f"/api/dashboard/design-runs/{run['id']}"
    token = client.headers.pop("Authorization")
    assert (await client.get(path)).status_code == 401
    assert (await client.post(path + "/resume", json={"generation": 1})).status_code == 401
    client.headers["Authorization"] = token
    assert (await client.get("/api/dashboard/design-runs/missing")).status_code == 404
    assert (await client.post(path + "/resume", json={"generation": 0})).status_code == 422
    assert queue.empty()
