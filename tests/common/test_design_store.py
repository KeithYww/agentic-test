# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Real SQLite recovery, budget and external-effect safety boundaries."""

import pytest

from common.design_store import DesignStore, RecoveryBlocked, digest


def patch_manifest(source="before"):
    return {"workflow": {"sources": {"orchestrator/test_case_design.py": source},
                         "policy": {"model": "original"}, "targets": {"node": "state_48"},
                         "provider": "original"}, "config_sha256": "same-config", "environment_sha256": "same-env",
                "runtime_sources": {"scripts/start_all.py": "same-script"}}


@pytest.mark.asyncio
async def test_patch_upgrade_preserves_paid_work_budget_and_deadline(tmp_path):
    owner = DesignStore(str(tmp_path))
    await owner.call("open")
    try:
        before, after = patch_manifest(), patch_manifest("after")
        run, _ = await owner.call("create_run", "upgrade", {"workflow": digest(before["workflow"])}, 6000000, 7200)
        permit = await owner.call("reserve", run["id"], 1, "generate", "input", 1000, 6000000)
        await owner.call("settle", run["id"], 1, permit["id"], 100, {}, False)
        await owner.call("put", run["id"], 1, "unit", "candidate", "input", {"version": 1})
        await owner.call("state", run["id"], 1, "paused", "ValueError")
        await owner.call("upgrade", run["id"], before, after, ["orchestrator/test_case_design.py"],
                         "operator", "Reviewed coverage bugfix; same config and targets")
        upgraded = await owner.call("run", run["id"])
        assert upgraded["state"] == "resume_requested"
        assert upgraded["generation"] == run["generation"]
        assert upgraded["deadline"] == run["deadline"] and upgraded["token_limit"] == 6000000
        assert upgraded["used"] == 100 and upgraded["reserved"] == 0
        assert upgraded["spec"]["workflow"] == digest(after["workflow"])
        assert upgraded["spec"]["upgrade_manifest"] == digest(after)
        assert (await owner.call("get", run["id"], "unit", "candidate", "input"))["value"] == {"version": 1}
        with pytest.raises(RecoveryBlocked):
            await owner.call("upgrade", run["id"], before, after, ["orchestrator/test_case_design.py"], "operator", "repeat")
        resumed = await owner.call("claim", run["id"])
        assert resumed["generation"] == 2 and resumed["used"] == 100
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["policy", "targets", "provider", "config", "environment", "unreviewed", "wrong_previous", "unknown", "runtime_script"])
async def test_patch_upgrade_rejects_unsafe_or_unverified_changes(tmp_path, change):
    owner = DesignStore(str(tmp_path))
    await owner.call("open")
    try:
        before, after = patch_manifest(), patch_manifest("after")
        run, _ = await owner.call("create_run", "upgrade", {"workflow": digest(before["workflow"])}, 6000000, 7200)
        if change in {"policy", "targets", "provider"}:
            after["workflow"][change] = "changed"
        elif change == "config":
            after["config_sha256"] = "changed"
        elif change == "environment":
            after["environment_sha256"] = "changed-kb-url"
        elif change == "unreviewed":
            after["workflow"]["sources"]["agents/test_case_generation/main.py"] = "changed"
        elif change == "runtime_script":
            after["runtime_sources"]["scripts/start_all.py"] = "unreviewed-script"
        elif change == "wrong_previous":
            before["workflow"]["sources"]["orchestrator/test_case_design.py"] = "unverified"
        elif change == "unknown":
            permit = await owner.call("reserve", run["id"], 1, "generate", "input", 1000, 6000000)
            await owner.call("unknown", run["id"], 1, permit["id"])
        await owner.call("state", run["id"], 1, "paused", "ValueError")
        with pytest.raises(RecoveryBlocked):
            await owner.call("upgrade", run["id"], before, after, ["orchestrator/test_case_design.py"], "operator", "reviewed")
        assert (await owner.call("run", run["id"]))["spec"] == run["spec"]
        assert (await owner.call("run", run["id"]))["state"] == "paused"
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_hard_exit_retains_committed_candidate_and_unknown_reservation(tmp_path):
    import asyncio
    import sys

    script = '''
import asyncio, os, sys
from common.design_store import DesignStore
async def run():
    owner = DesignStore(sys.argv[1])
    await owner.call("open")
    run, _ = await owner.call("create_run", "crash", {}, 1000, 3600)
    await owner.call("put", run["id"], 1, "unit", "pair-a", "hash", {"cases": [1, 2]})
    await owner.call("reserve", run["id"], 1, "pair-b", "request", 800, 1000)
    print(run["id"], flush=True)
    os._exit(17)
asyncio.run(run())
'''
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script, str(tmp_path), stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()
    assert process.returncode == 17
    run_id = stdout.decode().strip()
    owner = DesignStore(str(tmp_path))
    await owner.call("open")
    try:
        run = await owner.call("run", run_id)
        assert run["state"] == "paused" and run["usage_unknown"]
        assert run["reserved"] == 800
        assert (await owner.call("get", run_id, "unit", "pair-a", "hash"))["value"] == {"cases": [1, 2]}
        with pytest.raises(RecoveryBlocked):
            await owner.call("request_resume", run_id, "test", "crash")
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_snapshot_cannot_roll_back_revision_pointer(state):
    owner, run = state
    await owner.call("put", run["id"], 1, "revision", "0", "v0", {"cases": [1]})
    await owner.call("snapshot", run["id"], 1, {"published_url": "test"})
    with pytest.raises(RecoveryBlocked, match="atomic"):
        await owner.call("snapshot", run["id"], 1, {"revision": -1})
    assert (await owner.call("run", run["id"]))["snapshot"]["revision"] == 0


@pytest.mark.asyncio
async def test_new_execution_requires_completed_parent_and_is_idempotent(state):
    owner, _ = state
    parent, _ = await owner.call("create_run", "project:story", {
        "project_key": "project", "story_id": "story", "timeout_seconds": 3600,
    }, 1000, 3600)
    with pytest.raises(RecoveryBlocked):
        await owner.call("new_execution", parent["id"], "visit-2", "operator", "new workflow visit")
    await owner.call("state", parent["id"], 1, "completed")
    child = await owner.call("new_execution", parent["id"], "visit-2", "operator", "new workflow visit")
    assert child["state"] == "resume_requested" and child["used"] == 0
    assert child["spec"]["parent_run_id"] == parent["id"]
    again = await owner.call("new_execution", parent["id"], "visit-2", "operator", "duplicate")
    assert again["id"] == child["id"]
    with pytest.raises(RecoveryBlocked, match="unfinished"):
        await owner.call("new_execution", parent["id"], "visit-3", "operator", "cannot abandon child")


@pytest.mark.asyncio
async def test_publication_resume_reuses_known_step_and_blocks_uncertain_step(state):
    import asyncio

    from common.publication_journal import PublicationJournal

    owner, run = state
    calls = []
    journal = PublicationJournal(owner, run["id"], 1)

    def create():
        calls.append("base")
        return {"id": "base"}

    def write():
        calls.append("records")
        raise ConnectionError("response lost")

    await asyncio.to_thread(journal.perform, "create-base", {}, create)
    with pytest.raises(ConnectionError):
        await asyncio.to_thread(journal.perform, "records:0", {}, write)
    assert await asyncio.to_thread(journal.perform, "create-base", {}, create) == {"id": "base"}
    with pytest.raises(RecoveryBlocked, match="unknown"):
        await asyncio.to_thread(journal.perform, "records:0", {}, write)
    assert calls == ["base", "records"]


@pytest.fixture
async def state(tmp_path):
    owner = DesignStore(str(tmp_path))
    await owner.call("open")
    run, _ = await owner.call("create_run", "story", {"source": "prd"}, 1000, 3600)
    yield owner, run
    await owner.close()


@pytest.mark.asyncio
async def test_second_owner_cannot_open_same_volume(state, tmp_path):
    other = DesignStore(str(tmp_path))
    try:
        with pytest.raises(BlockingIOError):
            await other.call("open")
    finally:
        await other.close()


@pytest.mark.asyncio
async def test_run_and_candidate_survive_restart_and_old_generation_is_rejected(tmp_path):
    first = DesignStore(str(tmp_path))
    await first.call("open")
    run, _ = await first.call("create_run", "story", {}, 1000, 3600)
    await first.call("put", run["id"], 1, "unit", "pair-a", "hash", {"cases": [1, 2]})
    await first.close()
    second = DesignStore(str(tmp_path))
    await second.call("open")
    try:
        assert (await second.call("run", run["id"]))["state"] == "paused"
        await second.call("request_resume", run["id"], "operator", "known progress")
        resumed = await second.call("claim", run["id"])
        assert resumed["generation"] == 2
        assert (await second.call("get", run["id"], "unit", "pair-a", "hash"))["value"]["cases"] == [1, 2]
        with pytest.raises(RecoveryBlocked, match="stale"):
            await second.call("put", run["id"], 1, "unit", "bad", "hash", {})
        with pytest.raises(RecoveryBlocked, match="changed"):
            await second.call("get", run["id"], "unit", "pair-a", "other")
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_unknown_attempt_keeps_reservation_and_blocks_resume(state):
    owner, run = state
    permit = await owner.call("reserve", run["id"], 1, "pair", "request", 800, 1000)
    with pytest.raises(RecoveryBlocked, match="budget"):
        await owner.call("reserve", run["id"], 1, "other", "request2", 300, 1000)
    await owner.call("unknown", run["id"], 1, permit["id"])
    actual = await owner.call("run", run["id"])
    assert actual["reserved"] == 800 and actual["usage_unknown"]
    await owner.call("state", run["id"], 1, "paused", "transport_error")
    with pytest.raises(RecoveryBlocked):
        await owner.call("request_resume", run["id"], "operator", "retry")


@pytest.mark.asyncio
async def test_usage_response_atomic_settlement_is_reused_without_double_charge(state):
    owner, run = state
    permit = await owner.call("reserve", run["id"], 1, "pair", "request", 800, 1000)
    await owner.call("settle", run["id"], 1, permit["id"], 123, {"parts": []}, True)
    replay = await owner.call("reserve", run["id"], 1, "pair", "request", 800, 1000)
    assert replay["id"] == permit["id"] and replay["response"] == {"parts": []}
    assert (await owner.call("run", run["id"]))["used"] == 123
    with pytest.raises(RecoveryBlocked):
        await owner.call("settle", run["id"], 1, permit["id"], 123, {}, True)


@pytest.mark.asyncio
async def test_remote_intent_is_not_blindly_replayed(state):
    owner, run = state
    intent = {"base": "test"}
    assert await owner.call("publish_begin", run["id"], 1, "create-base", intent) is None
    with pytest.raises(RecoveryBlocked, match="unknown"):
        await owner.call("publish_begin", run["id"], 1, "create-base", intent)
    await owner.call("put", run["id"], 1, "publish", "create-base", digest(intent), {"id": "existing"})
    assert (await owner.call("publish_begin", run["id"], 1, "create-base", intent))["id"] == "existing"


@pytest.mark.asyncio
async def test_manual_resolution_requires_full_receipt_and_audits_before_resume(state):
    owner, run = state
    intent = {"records": [{"key": "TC-1"}, {"key": "TC-2"}]}
    await owner.call("publish_begin", run["id"], 1, "records:0", intent)
    await owner.call("state", run["id"], 1, "paused", "remote_unknown")
    with pytest.raises(RecoveryBlocked):
        await owner.call("request_resume", run["id"], "operator", "not resolved")
    with pytest.raises(RecoveryBlocked, match="incomplete"):
        await owner.call("resolve_publish", run["id"], "records:0", "adopt_existing",
                         {"data": {"record_id_list": ["r1"]}}, "operator", "partial receipt")
    receipt = {"data": {"record_id_list": ["r1", "r2"]}}
    await owner.call("resolve_publish", run["id"], "records:0", "adopt_existing", receipt,
                     "operator", "verified complete remote rows in original order")
    await owner.call("request_resume", run["id"], "operator", "resolved")
    resumed = await owner.call("claim", run["id"])
    assert await owner.call("publish_begin", run["id"], resumed["generation"], "records:0", intent) == receipt


@pytest.mark.asyncio
async def test_explicit_verified_absence_requires_operator_evidence(state):
    owner, run = state
    await owner.call("publish_begin", run["id"], 1, "node:confirm", {"node": "existing"})
    await owner.call("state", run["id"], 1, "paused", "remote_unknown")
    with pytest.raises(RecoveryBlocked, match="evidence"):
        await owner.call("resolve_publish", run["id"], "node:confirm", "verified_absent", None, "operator", "")
    await owner.call("resolve_publish", run["id"], "node:confirm", "leave_blocked", None,
                     "operator", "unable to exclude late completion")
    with pytest.raises(RecoveryBlocked):
        await owner.call("request_resume", run["id"], "operator", "still uncertain")
