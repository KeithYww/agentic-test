# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Verify optional agents are absent from the single-container runtime."""

import runpy
import sys
from types import ModuleType

import pytest

import config


@pytest.mark.parametrize("enabled", [False, True])
def test_requirements_agent_only_loaded_when_enabled(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    """Disabling review removes its listener without removing the test-case agents."""
    monkeypatch.setattr(config.DesignConfig, "REQUIREMENT_REVIEW_ENABLED", enabled)
    modules = {
        "agents.test_case_classification.main": "app",
        "agents.test_case_generation.main": "app",
        "agents.test_case_review.main": "app",
        "orchestrator.main": "orchestrator_app",
        "scripts.feishu_mcp_server": "app",
    }
    for name, attribute in modules.items():
        module = ModuleType(name)
        setattr(module, attribute, object())
        monkeypatch.setitem(sys.modules, name, module)
    if enabled:
        module = ModuleType("agents.requirements_review.main")
        module.app = object()
        monkeypatch.setitem(sys.modules, "agents.requirements_review.main", module)
    else:
        monkeypatch.setitem(sys.modules, "agents.requirements_review.main", None)

    namespace = runpy.run_path("scripts/start_all.py")
    services = {name: port for name, _, port in namespace["_SERVICES"]}

    assert ("requirements-review" in services) is enabled
    assert services | {"requirements-review": 8001} == {
        "orchestrator": 8000,
        "requirements-review": 8001,
        "tc-generation": 8002,
        "tc-classification": 8003,
        "tc-review": 8004,
        "feishu-mcp": 9010,
    }
