# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Immutable workflow compatibility shared by the application and offline recovery."""

import hashlib
import os
from pathlib import Path

import config
from common.design_store import RecoveryBlocked, digest
from common.provider_policy import explicit_policy


def workflow_manifest() -> dict:
    """Describe the code, prompts, policy and targets covered by the fingerprint."""
    root = Path(__file__).resolve().parent.parent
    sources = {}
    for folder in ("common", "agents", "orchestrator"):
        for path in sorted((root / folder).rglob("*")):
            if path.suffix in {".py", ".txt"}:
                sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    policy = {name: {key: value for key, value in vars(getattr(config, name)).items()
                     if key.isupper() and isinstance(value, (str, int, float, bool, type(None)))}
              for name in ("DesignConfig", "TestCaseGenerationAgentConfig", "TestCaseReviewAgentConfig", "TestCaseClassificationAgentConfig")}
    if retry_policy := explicit_policy():
        policy["ProviderRetryPolicy"] = retry_policy
    targets = {key: getattr(config, key) for key in (
        "MEEGO_BASE_URL", "MEEGO_PROJECT_KEY", "MEEGO_PLUGIN_ID", "MEEGO_AI_PREGENERATION_NODE_KEY",
        "FEISHU_APP_ID", "FEISHU_API_BASE_URL", "FEISHU_DOC_COLLABORATOR_IDS", "FEISHU_BITABLE_SYNC_ENABLED",
    )}
    return {"sources": sources, "policy": policy, "targets": targets, "provider": config.PHOTON_API_BASE_URL}


def workflow_fingerprint() -> str:
    """Refuse replay across changes to code, prompts or QA model policy."""
    return digest(workflow_manifest())


def upgrade_manifest() -> dict:
    """Include config source when verifying an explicitly reviewed patch upgrade."""
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"HOSTNAME", "PWD", "SHLVL", "_"}}
    if environment.get("PHOTON_OVERLOAD_RETRY_ENABLED", "false").lower() != "true":
        environment.pop("PHOTON_OVERLOAD_RETRY_ENABLED", None)
    if not environment.get("PHOTON_REJECTION_CONTRACT"):
        environment.pop("PHOTON_REJECTION_CONTRACT", None)
    return {"workflow": workflow_manifest(),
            "config_sha256": hashlib.sha256(Path(config.__file__).read_bytes()).hexdigest(),
            "environment_sha256": digest(environment),
            "runtime_sources": {str(path.relative_to(Path(config.__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted((Path(config.__file__).parent / "scripts").rglob("*.py"))}}


def validate_resume(spec: dict) -> None:
    """Recheck approved code, configuration and environment before claiming a run."""
    if spec.get("workflow") != workflow_fingerprint():
        raise RecoveryBlocked("workflow_changed")
    approved = spec.get("upgrade_manifest")
    if approved is not None and approved != digest(upgrade_manifest()):
        raise RecoveryBlocked("upgrade_manifest_changed")
