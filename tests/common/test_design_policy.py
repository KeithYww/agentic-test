# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Patch approval remains valid only for the reviewed startup environment."""

import json
from pathlib import Path

import pytest

import config
from common import design_policy
from common.design_store import DesignStore, RecoveryBlocked, digest
from common.provider_policy import ProviderRetryPolicy


def test_upgrade_manifest_covers_runtime_scripts() -> None:
    sources = design_policy.upgrade_manifest()["runtime_sources"]
    assert {"scripts/start_all.py", "scripts/feishu_mcp_server.py", "scripts/design_recovery.py"} <= sources.keys()


@pytest.mark.parametrize("change", [None, "environment_sha256", "config_sha256", "runtime_sources"])
def test_resume_rechecks_patch_approval_before_claim(monkeypatch, change: str | None) -> None:
    approved = {"environment_sha256": "original", "config_sha256": "same", "runtime_sources": {}}
    current = {**approved}
    if change is not None:
        current[change] = "changed-after-approval"
    monkeypatch.setattr(design_policy, "workflow_fingerprint", lambda: "workflow")
    monkeypatch.setattr(design_policy, "upgrade_manifest", lambda: current)
    spec = {"workflow": "workflow", "upgrade_manifest": digest(approved)}
    if change is None:
        design_policy.validate_resume(spec)
    else:
        with pytest.raises(RecoveryBlocked, match="upgrade_manifest_changed"):
            design_policy.validate_resume(spec)


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", ["unset", "explicit_default", "enabled", "contract"])
async def test_pre_retry_manifest_upgrade_preserves_paid_checkpoints(tmp_path, monkeypatch, setting):
    previous = json.loads((Path(__file__).parents[1] / "fixtures/provider_retry_previous_manifest.json").read_text())
    # Replay the fixed 523c15a policy under this test process's deployment environment.
    for name, values in previous["workflow"]["policy"].items():
        for key, value in values.items():
            monkeypatch.setattr(getattr(config, name), key, value)
    monkeypatch.setattr(config, "PHOTON_API_BASE_URL", previous["workflow"]["provider"])
    for key, value in previous["workflow"]["targets"].items():
        monkeypatch.setattr(config, key, value)
    monkeypatch.setattr(ProviderRetryPolicy, "PHOTON_OVERLOAD_RETRY_ENABLED", False)
    monkeypatch.setattr(ProviderRetryPolicy, "PHOTON_REJECTION_CONTRACT", "")
    monkeypatch.delenv("PHOTON_OVERLOAD_RETRY_ENABLED", raising=False)
    monkeypatch.delenv("PHOTON_REJECTION_CONTRACT", raising=False)
    baseline = design_policy.upgrade_manifest()
    previous["environment_sha256"] = baseline["environment_sha256"]
    # The historical policy above is replayed using today's config source file.
    previous["config_sha256"] = baseline["config_sha256"]
    if setting == "explicit_default":
        monkeypatch.setenv("PHOTON_OVERLOAD_RETRY_ENABLED", "false")
        monkeypatch.setenv("PHOTON_REJECTION_CONTRACT", "")
    elif setting == "enabled":
        monkeypatch.setattr(ProviderRetryPolicy, "PHOTON_OVERLOAD_RETRY_ENABLED", True)
        monkeypatch.setenv("PHOTON_OVERLOAD_RETRY_ENABLED", "true")
    elif setting == "contract":
        monkeypatch.setattr(ProviderRetryPolicy, "PHOTON_REJECTION_CONTRACT", "photon-preflight-v1")
        monkeypatch.setenv("PHOTON_REJECTION_CONTRACT", "photon-preflight-v1")
    current = design_policy.upgrade_manifest()
    assert previous["config_sha256"] == current["config_sha256"]
    old_sources = {**previous["workflow"]["sources"], **previous["runtime_sources"]}
    new_sources = {**current["workflow"]["sources"], **current["runtime_sources"]}
    changes = [path for path in old_sources.keys() | new_sources.keys() if old_sources.get(path) != new_sources.get(path)]
    assert "common/provider_policy.py" in changes
    owner = DesignStore(str(tmp_path))
    await owner.call("open")
    try:
        run, _ = await owner.call("create_run", "old-version", {"workflow": digest(previous["workflow"])}, 6000000, 3600)
        permit = await owner.call("reserve", run["id"], 1, "generate", "input", 1000, 6000000)
        await owner.call("settle", run["id"], 1, permit["id"], 100, {"paid": True}, True)
        await owner.call("put", run["id"], 1, "stage", "generate-0", "input", {"paid": True})
        await owner.call("state", run["id"], 1, "paused", "interrupted")
        if setting in {"enabled", "contract"}:
            with pytest.raises(RecoveryBlocked, match="upgrade_changes_policy_or_targets"):
                await owner.call("upgrade", run["id"], previous, current, changes, "operator", "reviewed retry fix")
            return
        assert previous["workflow"]["policy"] == current["workflow"]["policy"]
        assert previous["environment_sha256"] == current["environment_sha256"]
        await owner.call("upgrade", run["id"], previous, current, changes, "operator", "reviewed retry fix")
        resumed = await owner.call("claim", run["id"])
        design_policy.validate_resume(resumed["spec"])
        assert resumed["generation"] == 2 and resumed["used"] == 100 and resumed["reserved"] == 0
        assert resumed["deadline"] == run["deadline"] and resumed["token_limit"] == run["token_limit"]
        checkpoint = await owner.call("get", run["id"], "stage", "generate-0", "input")
        assert checkpoint["value"] == {"paid": True}
        replay = await owner.call("reserve", run["id"], 2, "generate", "input", 1000, 6000000)
        assert replay["response"] == {"paid": True}
    finally:
        await owner.close()


def test_model_upgrade_blocks_resume_with_previous_model_policy(monkeypatch) -> None:
    current = design_policy.workflow_fingerprint()
    with monkeypatch.context() as old_policy:
        old_policy.setattr(config.TestCaseGenerationAgentConfig, "REPAIR_MODEL_NAME", "deepseek-v4.1-flash")
        previous = design_policy.workflow_fingerprint()
    assert current != previous
    with pytest.raises(RecoveryBlocked, match="workflow_changed"):
        design_policy.validate_resume({"workflow": previous})
