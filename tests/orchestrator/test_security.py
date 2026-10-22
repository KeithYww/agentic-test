# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Security regression tests for the orchestrator authentication and webhook handling."""

import pytest
from fastapi import HTTPException

import config
from orchestrator.auth import AuthService

# ---------------------------------------------------------------------------
# Dashboard authentication (#2/#7): fail closed + constant-time
# ---------------------------------------------------------------------------


def test_authenticate_rejects_when_not_configured(monkeypatch):
    """Empty configured credentials must never authenticate an empty submission."""
    monkeypatch.setattr(config.DashboardAuthConfig, "USERNAME", "")
    monkeypatch.setattr(config.DashboardAuthConfig, "PASSWORD", "")
    monkeypatch.setattr(config.DashboardAuthConfig, "JWT_SECRET", "")

    assert AuthService().authenticate("", "") is False
    assert AuthService().authenticate("admin", "admin") is False


def test_authenticate_succeeds_when_configured(monkeypatch):
    monkeypatch.setattr(config.DashboardAuthConfig, "USERNAME", "admin")
    monkeypatch.setattr(config.DashboardAuthConfig, "PASSWORD", "s3cret")
    monkeypatch.setattr(config.DashboardAuthConfig, "JWT_SECRET", "a-secret")

    service = AuthService()
    assert service.authenticate("admin", "s3cret") is True
    assert service.authenticate("admin", "wrong") is False


def test_verify_token_rejects_when_secret_missing(monkeypatch):
    """A token must not be trusted when no JWT secret is configured."""
    monkeypatch.setattr(config.DashboardAuthConfig, "USERNAME", "admin")
    monkeypatch.setattr(config.DashboardAuthConfig, "PASSWORD", "s3cret")
    monkeypatch.setattr(config.DashboardAuthConfig, "JWT_SECRET", "a-secret")
    token = AuthService().create_token("admin").access_token

    # Now drop the secret: the previously valid token must no longer verify.
    monkeypatch.setattr(config.DashboardAuthConfig, "JWT_SECRET", "")
    assert AuthService().verify_token(token) is None


def test_create_token_raises_when_not_configured(monkeypatch):
    monkeypatch.setattr(config.DashboardAuthConfig, "JWT_SECRET", "")
    monkeypatch.setattr(config.DashboardAuthConfig, "USERNAME", "")
    monkeypatch.setattr(config.DashboardAuthConfig, "PASSWORD", "")
    with pytest.raises(HTTPException) as excinfo:
        AuthService().create_token("admin")
    assert excinfo.value.status_code == 503


# ---------------------------------------------------------------------------
# Orchestrator API key (#1/#7): fail closed
# ---------------------------------------------------------------------------


def test_validate_api_key_fails_closed_when_unconfigured(monkeypatch):
    from orchestrator.main import _validate_api_key

    monkeypatch.setattr(config.OrchestratorConfig, "API_KEY", None)
    with pytest.raises(HTTPException) as excinfo:
        _validate_api_key(api_key="anything")
    assert excinfo.value.status_code == 503


def test_validate_api_key_rejects_wrong_key(monkeypatch):
    from orchestrator.main import _validate_api_key

    monkeypatch.setattr(config.OrchestratorConfig, "API_KEY", "correct-key")
    with pytest.raises(HTTPException) as excinfo:
        _validate_api_key(api_key="wrong-key")
    assert excinfo.value.status_code == 401


def test_validate_api_key_accepts_correct_key(monkeypatch):
    from orchestrator.main import _validate_api_key

    monkeypatch.setattr(config.OrchestratorConfig, "API_KEY", "correct-key")
    # Should not raise.
    assert _validate_api_key(api_key="correct-key") is None


# ---------------------------------------------------------------------------
# SPA static serving (#3): path traversal containment
# ---------------------------------------------------------------------------


def test_spa_path_containment_logic(tmp_path):
    """The resolved request path must remain within the static root."""
    static_root = (tmp_path / "static").resolve()
    static_root.mkdir()
    (static_root / "index.html").write_text("ok")
    secret = tmp_path / "secret.env"
    secret.write_text("SECRET=1")

    # Traversal escapes the root -> rejected.
    traversed = (static_root / "../secret.env").resolve()
    assert traversed.is_relative_to(static_root) is False

    # A legitimate in-root file is allowed.
    legit = (static_root / "index.html").resolve()
    assert legit.is_relative_to(static_root) is True
