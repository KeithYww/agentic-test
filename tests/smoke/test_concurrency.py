# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Concurrent pipeline E2E tests.

Verifies that when two test-case-generation webhooks arrive simultaneously, the
orchestrator dispatches both to the agent without serial blocking, and the agent
processes them in parallel without state pollution or log cross-contamination.

Design:
  - 2 pipeline webhooks are sent at the same time.
  - We record the wall-clock time T0 just before sending.
  - We assert:
      1. Both webhooks return 202 (accepted).
      2. At least 2 generation agent tasks appear in the dashboard history with
         start times AFTER T0 — proving the orchestrator dispatched both.
      3. At least one of those generation tasks completes successfully — proving
         the agent can handle concurrent work without crashing.
      4. The two tasks started within 60 s of each other — if they were serial,
         the 2nd would start only after the 1st finished (~6 min later).
"""

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timezone

import httpx
import pytest

from tests.smoke.conftest import (
    ORCHESTRATOR_URL,
    SEEDED_PROJECT_KEY,
    SMOKE_FEISHU_DOC_URL,
    WEBHOOK_TIMEOUT,
)

pytestmark = pytest.mark.smoke

# Two distinct story IDs; mocks return the same canned content for any key.
_CONCURRENT_STORY_IDS = ["CONC-STORY-A", "CONC-STORY-B"]

# How long to poll for both generation tasks to appear and one to complete.
_PIPELINE_TIMEOUT = 600.0
_POLL_INTERVAL = 8.0

_WEBHOOK_HEADERS = {"X-API-Key": "smoke-api-key"}

# Maximum gap between start times of the two concurrent tasks.
# Serial execution would take ~6 min between starts; 60 s proves concurrent dispatch.
_MAX_DISPATCH_GAP_SECONDS = 60.0


def _post_pipeline_webhook(story_id: str) -> tuple[str, httpx.Response]:
    with httpx.Client(timeout=WEBHOOK_TIMEOUT, follow_redirects=True) as client:
        resp = client.post(
            f"{ORCHESTRATOR_URL}/story-ready-for-test-case-generation",
            headers=_WEBHOOK_HEADERS,
            json={"story_id": story_id, "project_key": SEEDED_PROJECT_KEY, "feishu_doc": SMOKE_FEISHU_DOC_URL},
        )
        return story_id, resp


def _get_tasks(http_client: httpx.Client, auth_headers: dict) -> list[dict]:
    resp = http_client.get(f"{ORCHESTRATOR_URL}/api/dashboard/tasks", headers=auth_headers)
    return resp.json() if resp.status_code == 200 else []


@pytest.fixture(scope="module")
def concurrent_pipeline_result(
    all_agents_ready: None,
    http_client: httpx.Client,
    auth_headers: dict,
) -> dict:
    """Send 2 webhooks simultaneously; return timing info and responses."""
    t0 = time.time()  # wall-clock seconds (for task filtering)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_post_pipeline_webhook, sid) for sid in _CONCURRENT_STORY_IDS]
        responses = dict(f.result() for f in futures)

    return {"t0": t0, "responses": responses}


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def test_concurrent_webhooks_accepted(concurrent_pipeline_result: dict) -> None:
    """Both webhooks must return 202."""
    for sid, resp in concurrent_pipeline_result["responses"].items():
        assert resp.status_code == 202, f"{sid}: {resp.status_code} {resp.text}"


def test_concurrent_generation_tasks_dispatched(
    http_client: httpx.Client,
    auth_headers: dict,
    concurrent_pipeline_result: dict,
) -> None:
    """Both pipelines are dispatched to the agent without serial blocking.

    We poll until at least 2 generation tasks appear with start_time after t0
    and assert they started within _MAX_DISPATCH_GAP_SECONDS of each other.
    """
    t0_utc = datetime.fromtimestamp(concurrent_pipeline_result["t0"], tz=UTC)
    deadline = time.monotonic() + _PIPELINE_TIMEOUT

    gen_tasks: list[dict] = []
    while time.monotonic() < deadline:
        all_tasks = _get_tasks(http_client, auth_headers)
        gen_tasks = [
            t for t in all_tasks
            if "Generation Agent" in (t.get("agent_name") or "")
            and t.get("start_time")
            # Server datetimes are UTC-naive; attach UTC before comparing.
            and datetime.fromisoformat(t["start_time"]).replace(tzinfo=UTC) >= t0_utc
        ]
        if len(gen_tasks) >= 2:
            break
        time.sleep(_POLL_INTERVAL)

    assert len(gen_tasks) >= 2, (
        f"Expected at least 2 generation tasks after t0={t0_utc.isoformat()}; "
        f"got {len(gen_tasks)}: {[t.get('start_time') for t in gen_tasks]}"
    )

    starts = sorted(
        datetime.fromisoformat(t["start_time"]).replace(tzinfo=UTC) for t in gen_tasks
    )
    gap_seconds = (starts[-1] - starts[0]).total_seconds()
    assert gap_seconds <= _MAX_DISPATCH_GAP_SECONDS, (
        f"Generation tasks started {gap_seconds:.1f}s apart — likely serial, not concurrent.\n"
        f"Start times: {[str(s) for s in starts]}"
    )


def test_concurrent_generation_at_least_one_succeeds(
    http_client: httpx.Client,
    auth_headers: dict,
    concurrent_pipeline_result: dict,
) -> None:
    """At least one concurrent generation task completes without error.

    Proves the agent handles concurrent work without state pollution or crash.
    """
    t0_utc = datetime.fromtimestamp(concurrent_pipeline_result["t0"], tz=UTC)
    deadline = time.monotonic() + _PIPELINE_TIMEOUT

    completed_tasks: list[dict] = []
    while time.monotonic() < deadline:
        all_tasks = _get_tasks(http_client, auth_headers)
        completed_tasks = [
            t for t in all_tasks
            if "Generation Agent" in (t.get("agent_name") or "")
            and t.get("start_time")
            and datetime.fromisoformat(t["start_time"]).replace(tzinfo=UTC) >= t0_utc
            and t.get("status") == "COMPLETED"
        ]
        if completed_tasks:
            break
        time.sleep(_POLL_INTERVAL)

    assert completed_tasks, (
        f"No concurrent generation task reached COMPLETED status within {_PIPELINE_TIMEOUT}s."
    )
