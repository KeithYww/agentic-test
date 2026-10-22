# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Controlled model-boundary failure coverage, independent of the live relay stack."""

import pytest

from tests.smoke.provider_retry_probe import run_probe


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_classification_recovers_only_rejected_batch_and_propagates_unknown():
    result = await run_probe()
    assert result["http_requests"] == 8
    assert result["truncated_stream_resent_once"] and result["repeated_truncation_stopped_with_known_usage"]
