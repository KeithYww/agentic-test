# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.services.kb_client import query_photon_kb


def _mock_client(mock_client_cls, response_json: dict, raise_for_status_exc: Exception | None = None):
    mock_client = AsyncMock()
    mock_client_cls.return_value.__aenter__.return_value = mock_client
    mock_response = MagicMock()
    if raise_for_status_exc:
        mock_response.raise_for_status.side_effect = raise_for_status_exc
    mock_response.json.return_value = response_json
    mock_client.post.return_value = mock_response
    return mock_client


@pytest.mark.asyncio
async def test_query_photon_kb_returns_answer_on_high_confidence():
    with patch("httpx.AsyncClient") as mock_client_cls:
        _mock_client(mock_client_cls, {"answer": "some knowledge", "confidence": "high"})
        with patch("config.KB_QUERY_ENABLED", True), patch("config.KB_API_KEY", "kb_test_key"):
            result = await query_photon_kb("some question")

    assert result == "some knowledge"


@pytest.mark.asyncio
async def test_query_photon_kb_discards_fallback_confidence():
    with patch("httpx.AsyncClient") as mock_client_cls:
        _mock_client(mock_client_cls, {"answer": "low quality guess", "confidence": "fallback"})
        with patch("config.KB_QUERY_ENABLED", True), patch("config.KB_API_KEY", "kb_test_key"):
            result = await query_photon_kb("some question")

    assert result is None


@pytest.mark.asyncio
async def test_query_photon_kb_returns_none_when_answer_missing():
    with patch("httpx.AsyncClient") as mock_client_cls:
        _mock_client(mock_client_cls, {"confidence": "high"})
        with patch("config.KB_QUERY_ENABLED", True), patch("config.KB_API_KEY", "kb_test_key"):
            result = await query_photon_kb("some question")

    assert result is None


@pytest.mark.asyncio
async def test_query_photon_kb_returns_none_when_disabled():
    with patch("config.KB_QUERY_ENABLED", False), patch("config.KB_API_KEY", "kb_test_key"):
        result = await query_photon_kb("some question")

    assert result is None


@pytest.mark.asyncio
async def test_query_photon_kb_returns_none_when_no_api_key():
    with patch("config.KB_QUERY_ENABLED", True), patch("config.KB_API_KEY", ""):
        result = await query_photon_kb("some question")

    assert result is None


@pytest.mark.asyncio
async def test_query_photon_kb_non_fatal_on_request_failure():
    with patch("httpx.AsyncClient") as mock_client_cls:
        _mock_client(mock_client_cls, {}, raise_for_status_exc=RuntimeError("connection reset"))
        with patch("config.KB_QUERY_ENABLED", True), patch("config.KB_API_KEY", "kb_test_key"):
            result = await query_photon_kb("some question")

    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize("response,expected", [
    ({"confidence": "fallback", "answer": ""}, "no_match"),
    ({"confidence": "high", "answer": ""}, "no_match"),
    ({"confidence": "high", "answer": "  \n"}, "no_match"),
    ({"confidence": "high", "answer": None}, "no_match"),
    ({"confidence": "high"}, "no_match"),
    ({"confidence": "high", "answer": "known rule"}, "ok"),
    (None, "unavailable"),
])
async def test_snapshot_distinguishes_missing_knowledge_from_outage(response, expected):
    from common.services.kb_client import query_knowledge_snapshot
    with patch("config.KB_QUERY_ENABLED", True), patch("config.KB_API_KEY", "test"), patch(
        "common.services.kb_client._post_query", new_callable=AsyncMock, return_value=response
    ) as query:
        result = await query_knowledge_snapshot("question")
        assert result.status == expected
        assert result.content == ("known rule" if expected == "ok" else "")
        assert query.await_count == (3 if response is None else 1)


@pytest.mark.asyncio
async def test_legacy_query_accepts_empty_success_without_retry():
    with patch("httpx.AsyncClient") as mock_client_cls:
        client = _mock_client(mock_client_cls, {"confidence": "high", "answer": ""})
        with patch("config.KB_QUERY_ENABLED", True), patch("config.KB_API_KEY", "test"):
            assert await query_photon_kb("question") is None
        client.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_disabled_required_knowledge_is_explicit():
    from common.services.kb_client import query_knowledge_snapshot
    with patch("config.KB_QUERY_ENABLED", False):
        assert (await query_knowledge_snapshot("question")).status == "disabled"
