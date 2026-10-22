# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Controlled alarm-to-confirmation flow; only the outgoing bot delivery is recorded."""

import asyncio
from urllib.parse import urlsplit

import httpx
import pytest

import config
import orchestrator.main as main
from common.design_policy import workflow_fingerprint
from common.design_store import DesignStore
from common.services import feishu_bot_notifier


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_alarm_link_requires_login_and_confirm_before_checkpoint_resume(tmp_path, monkeypatch):
    owner = DesignStore(str(tmp_path))
    await owner.call("open")
    cards = []

    async def record(payload):
        cards.append(payload["card"])

    monkeypatch.setattr(main.design_store, "store", owner)
    monkeypatch.setattr(config, "DASHBOARD_PUBLIC_URL", "https://qa.example.com")
    monkeypatch.setattr(feishu_bot_notifier, "_send", record)
    queue = asyncio.Queue()
    monkeypatch.setattr(main, "_pipeline_queue", queue)
    try:
        run, _ = await owner.call("create_run", "smoke:resume", {
            "story_id": "SMOKE-RESUME", "project_key": "SMOKE", "feishu_doc": "https://example/doc",
            "story_title": "受控续跑验证", "workflow": workflow_fingerprint(),
        }, 6000000, 3600)
        await owner.call("put", run["id"], 1, "unit", "generate", "saved", {"cases": ["TC1"]})
        await owner.call("state", run["id"], 1, "paused", "ProviderBusy")
        await feishu_bot_notifier.notify_pipeline_failure("SMOKE-RESUME", RuntimeError("ProviderBusy"), run_id=run["id"])
        link = next(button["url"] for element in cards[0]["elements"] if element["tag"] == "action"
                    for button in element["actions"] if "Retry" in button["text"]["content"])
        parsed = urlsplit(link)
        assert parsed.netloc == "qa.example.com" and parsed.query == f"run_id={run['id']}"
        path = f"/api/dashboard/design-runs/{run['id']}"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.orchestrator_app), base_url=config.DASHBOARD_PUBLIC_URL) as client:
            assert (await client.get(path)).status_code == 401
            login = await client.post("/api/auth/login", json={
                "username": config.DashboardAuthConfig.USERNAME, "password": config.DashboardAuthConfig.PASSWORD,
            })
            assert login.status_code == 200
            client.headers["Authorization"] = "Bearer " + login.json()["access_token"]
            detail = await client.get(path)
            assert detail.json()["can_resume"] and detail.json()["completed_units"] == 1
            assert queue.empty()
            assert (await client.post(path + "/resume", json={"generation": 1})).status_code == 202
            assert (await client.post(path + "/resume", json={"generation": 1})).status_code == 409
        assert queue.qsize() == 1
        resumed = queue.get_nowait().args[-1]
        assert resumed["id"] == run["id"] and resumed["generation"] == 2
        assert resumed["deadline"] == run["deadline"] and resumed["token_limit"] == run["token_limit"]
        assert (await owner.call("get", run["id"], "unit", "generate", "saved"))["value"] == {"cases": ["TC1"]}
    finally:
        await owner.close()
