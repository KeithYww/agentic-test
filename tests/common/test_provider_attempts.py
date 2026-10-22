# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""SQLite provider-attempt migration, recovery, and transaction boundaries."""

import asyncio
import sqlite3

import pytest

from common.design_store import DesignStore, RecoveryBlocked


def rejection(delay: float | None = 0) -> dict:
    """Return an explicit, verified preflight receipt."""
    return {"state": "rejected", "status": 503, "contract": "photon-preflight-v1",
            "delay": delay, "code": "provider_busy_exhausted", "failure_kind": "known_rejection"}


async def reserve(owner: DesignStore, run: dict) -> dict:
    return await owner.call("provider_reserve", run["id"], run["generation"], "batch-1", "same-input", 100, 1000, "deepseek-v4-flash")


@pytest.mark.asyncio
async def test_v1_migration_and_reopen_do_not_reset_schema_or_paid_work(tmp_path):
    with sqlite3.connect(tmp_path / "qa_design.sqlite") as connection:
        connection.execute("""CREATE TABLE attempts (
            id TEXT PRIMARY KEY, run_id TEXT NOT NULL, generation INTEGER NOT NULL,
            task_key TEXT NOT NULL, digest TEXT NOT NULL, state TEXT NOT NULL,
            reserved INTEGER NOT NULL, used INTEGER NOT NULL DEFAULT 0,
            response TEXT, reusable INTEGER NOT NULL DEFAULT 0, UNIQUE(run_id,task_key,digest))""")
        connection.execute("PRAGMA user_version=1")
    owner = DesignStore(str(tmp_path))
    await owner.call("open")
    run, _ = await owner.call("create_run", "migration", {}, 1000, 100)
    permit = await owner.call("reserve", run["id"], 1, "old-unit", "old-input", 100, 1000)
    await owner.call("settle", run["id"], 1, permit["id"], 12, {"result": "old"}, True)
    await owner.close()
    owner = DesignStore(str(tmp_path))
    await owner.call("open")
    try:
        with sqlite3.connect(tmp_path / "qa_design.sqlite") as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        actual = await owner.call("run", run["id"])
        assert actual["used"] == 12
        assert await owner.call("provider_attempts", run["id"]) == []
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_retry_wait_generation_is_taken_over_without_resetting_send_count(tmp_path):
    owner = DesignStore(str(tmp_path))
    await owner.call("open")
    run, _ = await owner.call("create_run", "recover", {}, 1000, 100)
    first = await reserve(owner, run)
    await owner.call("provider_finish", run["id"], 1, first, rejection())
    await owner.close()
    owner = DesignStore(str(tmp_path))
    await owner.call("open")
    try:
        await owner.call("request_resume", run["id"], "test", "reviewed safe rejection")
        resumed = await owner.call("claim", run["id"])
        assert resumed["generation"] == 2
        second = await reserve(owner, resumed)
        assert second["ordinal"] == 2 and second["id"] == first["id"]
        with pytest.raises(RecoveryBlocked, match="stale_run_generation"):
            await owner.call("provider_finish", run["id"], 1, first, {"state": "unknown"})
        await owner.call("provider_finish", run["id"], 2, second, {
            "state": "settled", "used": 12, "response": {"result": "ok"}, "reusable": True,
        })
        replay = await reserve(owner, resumed)
        assert replay == {"state": "cached", "response": {"result": "ok"}}
        assert [item["generation"] for item in await owner.call("provider_attempts", run["id"])] == [1, 2]
        assert (await owner.call("run", run["id"]))["used"] == 12
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_concurrent_retry_claim_has_only_one_sender(tmp_path):
    owner = DesignStore(str(tmp_path))
    await owner.call("open")
    try:
        run, _ = await owner.call("create_run", "concurrent", {}, 1000, 100)
        first = await reserve(owner, run)
        await owner.call("provider_finish", run["id"], 1, first, rejection())
        results = await asyncio.gather(reserve(owner, run), reserve(owner, run), return_exceptions=True)
        assert sum(isinstance(result, dict) and result["state"] == "ready" for result in results) == 1
        assert sum(isinstance(result, RecoveryBlocked) for result in results) == 1
        assert len(await owner.call("provider_attempts", run["id"])) == 2
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_restart_marks_in_flight_parent_and_child_unknown(tmp_path):
    owner = DesignStore(str(tmp_path))
    await owner.call("open")
    run, _ = await owner.call("create_run", "interrupted", {}, 1000, 100)
    await reserve(owner, run)
    await owner.close()
    owner = DesignStore(str(tmp_path))
    await owner.call("open")
    try:
        actual = await owner.call("run", run["id"])
        assert actual["usage_unknown"] and actual["reserved"] == 100
        assert (await owner.call("provider_attempts", run["id"]))[0]["state"] == "unknown"
        with pytest.raises(RecoveryBlocked):
            await owner.call("request_resume", run["id"], "test", "must remain blocked")
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_wait_deadline_and_wait_budget_survive_resume(tmp_path, monkeypatch):
    import time

    now = time.time()
    monkeypatch.setattr("common.design_store.time.time", lambda: now)
    owner = DesignStore(str(tmp_path))
    await owner.call("open")
    try:
        run, _ = await owner.call("create_run", "waiting", {}, 1000, 100)
        first = await reserve(owner, run)
        await owner.call("provider_finish", run["id"], 1, first, rejection(12))
        waiting = await reserve(owner, run)
        assert waiting == {"state": "waiting", "next_attempt_at": now + 12}
        await owner.call("state", run["id"], 1, "paused")
        await owner.call("request_resume", run["id"], "test", "safe rejection")
        resumed = await owner.call("claim", run["id"])
        assert await reserve(owner, resumed) == waiting
        now += 13
        second = await reserve(owner, resumed)
        assert second["waited"] == 12 and second["ordinal"] == 2
        await owner.call("provider_finish", run["id"], 2, second, rejection(None))
        assert (await reserve(owner, resumed))["state"] == "rejected"
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_disabled_retry_cannot_resume_a_pending_send(tmp_path):
    owner = DesignStore(str(tmp_path))
    await owner.call("open")
    try:
        run, _ = await owner.call("create_run", "disabled", {}, 1000, 100)
        first = await reserve(owner, run)
        await owner.call("provider_finish", run["id"], 1, first, rejection())
        result = await owner.call("provider_reserve", run["id"], 1, "batch-1", "same-input", 100, 1000, "deepseek-v4-flash", False)
        assert result["state"] == "rejected"
        assert len(await owner.call("provider_attempts", run["id"])) == 1
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_incomplete_stream_accumulates_usage_and_resends_once_per_generation(tmp_path):
    owner = DesignStore(str(tmp_path))
    await owner.call("open")
    run, _ = await owner.call("create_run", "truncated", {}, 1000, 100)
    incomplete = {"state": "incomplete", "used": 110, "failure_kind": "stream_incomplete", "request_id": "msg_cut"}
    first = await reserve(owner, run)
    await owner.call("provider_finish", run["id"], 1, first, incomplete)
    state = await owner.call("run", run["id"])
    assert (state["used"], state["reserved"], state["usage_unknown"]) == (110, 0, False)
    second = await reserve(owner, run)
    assert second["ordinal"] == 2 and second["id"] == first["id"]
    await owner.call("provider_finish", run["id"], 1, second, incomplete)
    assert (await owner.call("run", run["id"]))["used"] == 220
    assert await reserve(owner, run) == {"state": "rejected", "code": "provider_stream_incomplete"}
    attempts = await owner.call("provider_attempts", run["id"])
    assert [(a["state"], a["used_tokens"], a["provider_request_id"]) for a in attempts] == [
        ("incomplete", 110, "msg_cut"), ("incomplete", 110, "msg_cut"),
    ]
    await owner.close()
    owner = DesignStore(str(tmp_path))
    await owner.call("open")
    try:
        await owner.call("request_resume", run["id"], "test", "operator accepted another paid send")
        resumed = await owner.call("claim", run["id"])
        third = await reserve(owner, resumed)
        assert third["ordinal"] == 3
        await owner.call("provider_finish", run["id"], 2, third, {
            "state": "settled", "used": 12, "response": {"result": "ok"}, "reusable": True,
        })
        assert (await owner.call("run", run["id"]))["used"] == 232
        assert await reserve(owner, resumed) == {"state": "cached", "response": {"result": "ok"}}
    finally:
        await owner.close()
