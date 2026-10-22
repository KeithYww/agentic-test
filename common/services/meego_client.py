# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

import json
import os
import time

import httpx

import config
from common import utils
from common.models import TestCase, TestExecutionResult
from common.publication_journal import PublicationJournal
from common.services.test_management_base import TestManagementClientBase

logger = utils.get_logger(__name__)

# Feishu Project (Meego) work item type key for test cases in the target space.
_TEST_CASE_TYPE_KEY = "test_cases"

# Feishu Project field keys for the test_cases work item type.
_FIELD_STEPS = os.environ.get("MEEGO_FIELD_STEPS", "field_steps")       # 执行步骤 (multi-text)
_FIELD_EXPECTED = os.environ.get("MEEGO_FIELD_EXPECTED", "field_expected")    # 预期结果 (multi-text)
_FIELD_LABELS = os.environ.get("MEEGO_FIELD_LABELS", "field_labels")      # 标签 (multi-select)
_FIELD_RELATED_STORY = os.environ.get("MEEGO_FIELD_RELATED_STORY", "field_related_story")  # 关联需求 (workitem_related_multi_select)

# The "AI用例预生成" AI node name on a story's workflow (case-insensitive match), and its
# "测试用例" field, where the generated Bitable test-case table URL is written back.
# The node also carries other fields depending on the workflow template; only the
# "测试用例" field (field_test_cases) is written by this pipeline.
# Example field key for a custom template (story 1000000021): field_test_cases | 测试用例 | multi-text
_AI_PREGENERATION_NODE_NAME = "ai用例预生成"
_FIELD_TEST_CASES = os.environ.get("MEEGO_FIELD_TEST_CASES", "field_test_cases")  # 测试用例 (multi-text) — Bitable URL is written here

# The Feishu Project (Meego) Open API is a distinct product from Feishu itself, with its own
# project_key namespace: a space's URL simple_name (e.g. "example-space") is accepted in the
# work_item/* URL path, but the /ai_node/* endpoints require the real project_key. There's no
# plugin_token-authenticated endpoint to resolve one from the other, so — matching how other
# Feishu Project integrations in this org handle it — the mapping is a static lookup.
_REAL_PROJECT_KEY_BY_SIMPLE_NAME = json.loads(os.environ.get(
    "MEEGO_PROJECT_KEY_MAP_JSON", '{"example-space": "example-project-key"}'
))

# Status option IDs for the test_cases work item type.
_STATUS_OPTIONS: dict[str, str] = {
    "not_reviewed": "not_reviewed",
    "review_passed": "review_passed",
    "review_failed": "review_failed",
    "systemended": "systemEnded",
}

# The story's "业务线" (business line) custom field key. Its selected value is an option_id
# (e.g. "example-business-line-4"), not a display name — resolve it via workitem meta-fields.
# The lookup endpoint (get_story_business_line_option_id) is confirmed correct against a real
# story, but currently 401s with err_code 10301 because this plugin has not been granted the
# "获取指定的工作项列表" API permission in the Feishu Project plugin console — an admin/plugin-console
# grant, not a code defect. The caller treats any failure here as non-fatal.
_FIELD_BUSINESS_LINE = "business"


class MeegoClient(TestManagementClientBase):
    """Client for Feishu Project (Meego) Open API."""

    def __init__(self, project_key: str | None = None) -> None:
        self._base_url = (config.MEEGO_BASE_URL or "").rstrip("/")
        if not self._base_url:
            raise ValueError("MEEGO_BASE_URL is not configured.")
        self._plugin_id = config.MEEGO_PLUGIN_ID
        self._plugin_secret = config.MEEGO_PLUGIN_SECRET
        self._default_project_key = project_key or config.MEEGO_PROJECT_KEY
        self._user_key = config.MEEGO_USER_KEY
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        """Return a valid plugin_token, refreshing if within 60 s of expiry."""
        if self._token and time.monotonic() < self._token_expires_at - 60:
            return self._token
        if not self._plugin_id or not self._plugin_secret:
            raise ValueError("MEEGO_PLUGIN_ID and MEEGO_PLUGIN_SECRET must be set.")
        with httpx.Client() as client:
            response = client.post(
                f"{self._base_url}/open_api/authen/plugin_token",
                # type: 1 was confirmed by direct API probing to make Feishu Project reject every
                # subsequent call for this plugin with "plugin owner has no project admin
                # permission" (401) — including plain reads. type: 0 passes the same permission
                # check without error; this is not a scope/permission gap on the plugin itself.
                json={"plugin_id": self._plugin_id, "plugin_secret": self._plugin_secret, "type": 0},
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
        self._token = data["data"]["token"]
        # Feishu Project tokens expire in ~7200 s; fall back to 7200 if field absent.
        ttl = data["data"].get("expire_time", 7200)
        self._token_expires_at = time.monotonic() + ttl
        return self._token

    def _headers(self) -> dict[str, str]:
        headers = {"X-PLUGIN-TOKEN": self._get_token(), "Content-Type": "application/json"}
        if self._user_key:
            headers["X-USER-KEY"] = self._user_key
        return headers

    @staticmethod
    def _validate_response(response: httpx.Response) -> dict:
        response.raise_for_status()
        data = response.json()
        # Most endpoints report errors via "code"; /ai_node/* uses "err_code" instead.
        if data.get("code", 0) != 0 or data.get("err_code", 0) != 0:
            raise RuntimeError(f"Meego API request failed: {data}")
        return data

    def _post(self, path: str, payload: dict) -> dict:
        with httpx.Client() as client:
            response = client.post(
                f"{self._base_url}{path}",
                headers=self._headers(),
                json=payload,
                timeout=30,
            )
            return self._validate_response(response)

    def get_story_business_line_option_id(self, project_key: str, story_id: str) -> str | None:
        """Reads the story's "业务线" custom field and returns its selected option_id, or None
        if the field is absent/unset. The field is not carried in workflow-event webhook payloads
        (those only expose fields explicitly added to the triggering node's form), so callers
        needing it must fetch it directly via this method.

        Uses POST /open_api/:project_key/work_item/filter (confirmed via the Open API error-code
        reference — GET-by-id on /work_item/:id does not exist and 404s). Requires the plugin to
        have been granted the "获取指定的工作项列表" API permission; without it the call 401s with
        err_code 10301 ("Check Token Perm Failed") — this is an admin/plugin-console grant, not a
        code defect, so callers must treat any failure here as non-fatal.
        """
        data = self._post(
            f"/open_api/{project_key}/work_item/filter",
            {
                "work_item_ids": [int(story_id)],
                "work_item_type_keys": ["story"],
                "fields": [_FIELD_BUSINESS_LINE],
            },
        )
        for work_item in data.get("data", {}).get("work_items", []):
            for field in work_item.get("fields", []):
                if field.get("field_key") == _FIELD_BUSINESS_LINE:
                    value = field.get("field_value")
                    if isinstance(value, list):
                        value = value[0] if value else None
                    return str(value) if value else None
        return None

    def get_test_case_writing_node_emails(self, project_key: str, story_id: str) -> list[str]:
        """Return email addresses of owners assigned to the 测试用例编写 workflow node.

        Used to auto-grant Bitable edit access to the QA engineers owning this node.
        Requires the meegle CLI to be authenticated in the execution environment; returns an
        empty list on any failure (including missing CLI auth) so callers never block.
        """
        import subprocess as _sp

        node_name = config.MEEGO_TC_WRITING_NODE_NAME
        try:
            node_result = _sp.run(
                [
                    "meegle", "workflow", "get-node",
                    "--project-key", project_key,
                    "--work-item-id", str(story_id),
                    "--node-id-list", '["_all"]',
                    "--format", "json",
                ],
                capture_output=True, text=True, timeout=30,
            )
            nodes = json.loads(node_result.stdout).get("list", [])
            matched = [n for n in nodes if n.get("basic", {}).get("name") == node_name]
            if not matched:
                return []
            user_keys = [
                owner["user_key"]
                for node in matched
                for owner in (node.get("assignees") or {}).get("owners", [])
                if owner.get("user_key")
            ]
            if not user_keys:
                return []
            user_result = _sp.run(
                [
                    "meegle", "user", "search",
                    "--project-key", project_key,
                    "--user-keys", json.dumps(user_keys),
                    "--format", "json",
                ],
                capture_output=True, text=True, timeout=30,
            )
            users = json.loads(user_result.stdout)
            if isinstance(users, list):
                return [u["email"] for u in users if u.get("email")]
        except Exception as exc:
            logger.warning(f"Could not resolve '{node_name}' node owners for story {story_id}: {exc}")
        return []

    # ------------------------------------------------------------------
    # TestManagementClientBase implementation
    # ------------------------------------------------------------------

    def create_test_cases(self, test_cases: list[TestCase], project_key: str, user_story_id: str) -> list[str]:
        """Create test case work items in Feishu Project and return their IDs."""
        created_ids: list[str] = []
        for tc in test_cases:
            steps_text = "\n".join(
                f"{i}. {step.action}" + (f"\n   Data: {', '.join(step.test_data)}" if step.test_data else "")
                for i, step in enumerate(tc.steps, start=1)
            )
            expected_text = "\n".join(step.expected_results for step in tc.steps if step.expected_results)
            fields = [
                {"field_key": "name", "field_value": tc.name},
                {"field_key": "description", "field_value": tc.preconditions or ""},
                {"field_key": _FIELD_STEPS, "field_value": steps_text},
                {"field_key": _FIELD_EXPECTED, "field_value": expected_text},
                {"field_key": _FIELD_RELATED_STORY, "field_value": f"[{user_story_id}]"},
            ]
            payload = {
                "work_item_type_key": _TEST_CASE_TYPE_KEY,
                "fields": fields,
            }
            logger.info(f"Creating test case '{tc.name}' in Feishu Project space '{project_key}'")
            data = self._post(f"/open_api/{project_key}/work_item/create", payload)
            work_item_id = str(data.get("data", {}).get("work_item_id", ""))
            if work_item_id:
                logger.info(f"Created test case work item {work_item_id} for '{tc.name}'")
                created_ids.append(work_item_id)
            else:
                logger.warning(f"No work_item_id in response for test case '{tc.name}': {data}")
        return created_ids

    def add_labels_to_test_case(self, test_case_key: str, labels: list[str]) -> None:
        """Add classification labels to a test case work item.

        POST /work_item/{type}/{id}, not PATCH /work_item/{id} — confirmed by probing the real
        API directly: the bare PATCH path 404s (no such route), while POST with the type segment
        reaches a real permission check (401 "plugin owner has no project admin permission")
        instead of a 404, for the same plugin/story combination.
        """
        payload = {"fields": [{"field_key": _FIELD_LABELS, "field_value": labels}]}
        self._post(f"/open_api/{self._default_project_key}/work_item/{_TEST_CASE_TYPE_KEY}/{test_case_key}", payload)
        logger.info(f"Added labels {labels} to test case {test_case_key}")

    def add_test_case_review_comment(self, test_case_key: str, comment: str) -> None:
        """Add a review comment to a test case work item."""
        payload = {"content": comment, "type": 1}
        self._post(f"/open_api/{self._default_project_key}/work_item/{test_case_key}/comment/create", payload)
        logger.info(f"Added review comment to test case {test_case_key}")

    def add_work_item_comment(self, project_key: str, work_item_type_key: str, work_item_id: str, content: str) -> None:
        """Add a comment using the work item's type and ID in the API path."""
        self._post(
            f"/open_api/{project_key}/work_item/{work_item_type_key}/{work_item_id}/comment/create",
            {"content": content},
        )
        logger.info(f"Added comment to work item {work_item_id}")

    def update_story_test_case_bitable(self, project_key: str, story_id: str, bitable_url: str, journal: PublicationJournal | None = None) -> None:
        """Writes the per-run Bitable test-case table URL to the story's "AI用例预生成" AI node,
        in its "测试用例" field, then confirms the node to advance the workflow.

        Uses the dedicated /ai_node/edit and /ai_node/confirm endpoints (not the generic
        work-item field-update endpoint): this AI node's fields can only be written while its
        ai_info.status is "running", which the story's own workflow sets when it reaches this
        node; setting status to "done" only marks the AI step itself finished — /ai_node/confirm
        is a separate call required to actually advance the underlying workflow node.

        The node key is read from MEEGO_AI_PREGENERATION_NODE_KEY (default: state_48). Override
        this env var when a different workflow template uses a different internal key.
        """
        node_key = config.MEEGO_AI_PREGENERATION_NODE_KEY
        real_project_key = _REAL_PROJECT_KEY_BY_SIMPLE_NAME.get(project_key, project_key)
        node_uuid = f"node_{node_key}_{story_id}"
        doc_text = f"多维表格：{bitable_url}"
        value = json.dumps(
            {
                "doc": bitable_url,
                "doc_text": doc_text,
                "doc_img": None,
                "doc_html": "",
                "is_empty": False,
                "notify_user_list": [],
                "notify_user_type": "",
                "external_storage_key": "",
                "is_trunc": False,
            },
            ensure_ascii=False,
        )
        base_payload = {"project_key": real_project_key, "work_item_id": story_id, "node_uuid": node_uuid}
        running_payload = {
                **base_payload,
                "status": "running",
                "conform_fields": [{"field_key": _FIELD_TEST_CASES, "value": value}],
            }
        if journal:
            journal.perform("node:write-url", running_payload,
                            lambda: self._post("/open_api/ai_node/edit", running_payload), lambda value: isinstance(value, dict))
            done_payload = {**base_payload, "status": "done"}
            journal.perform("node:done", done_payload,
                            lambda: self._post("/open_api/ai_node/edit", done_payload), lambda value: isinstance(value, dict))
            def confirm():
                self.confirm_story_pregeneration_node(project_key, story_id)
                return {"confirmed": True}
            journal.perform("node:confirm", base_payload, confirm)
        else:
            self._post("/open_api/ai_node/edit", running_payload)
            self._post("/open_api/ai_node/edit", {**base_payload, "status": "done"})
            self.confirm_story_pregeneration_node(project_key, story_id)
        logger.info(f"Wrote Bitable URL to story {story_id}'s AI用例预生成 node and confirmed it.")

    def confirm_story_pregeneration_node(self, project_key: str, story_id: str) -> None:
        """Confirm only after the caller has successfully published all required outputs."""
        try:
            self._post("/open_api/ai_node/confirm", {
                "project_key": _REAL_PROJECT_KEY_BY_SIMPLE_NAME.get(project_key, project_key),
                "work_item_id": story_id,
                "node_uuid": f"node_{config.MEEGO_AI_PREGENERATION_NODE_KEY}_{story_id}",
            })
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RuntimeError("confirm_unknown: inspect the workflow node before retrying") from exc

    def change_test_case_status(self, project_key: str, test_case_key: str, new_status: str) -> None:
        """Transition a test case work item to the given status."""
        status_id = _STATUS_OPTIONS.get(new_status.lower().replace(" ", "_"))
        if not status_id:
            # Fall back to case-insensitive match of the raw value.
            status_id = next(
                (v for k, v in _STATUS_OPTIONS.items() if k.replace("_", " ") == new_status.lower()),
                new_status,
            )
        payload = {"transition_status_id": status_id}
        self._post(f"/open_api/{project_key}/work_item/{test_case_key}/transition_state", payload)
        logger.info(f"Changed status of {test_case_key} to '{new_status}'")

    # ------------------------------------------------------------------
    # Methods used only by /execute-tests flow (not in scope)
    # ------------------------------------------------------------------

    def fetch_ready_for_execution_test_cases_by_labels(
        self, project_key: str, target_labels: list[str], max_results: int = 100
    ) -> dict[str, list[TestCase]]:
        raise NotImplementedError("fetch_ready_for_execution_test_cases_by_labels not supported by MeegoClient")

    def create_test_execution(
        self, test_execution_results: list[TestExecutionResult], project_key: str, version_id: str
    ) -> None:
        raise NotImplementedError("create_test_execution not supported by MeegoClient")

    def create_test_plan(self, project_key: str, name: str, description: str | None = None) -> str:
        raise NotImplementedError("create_test_plan not supported by MeegoClient")

    def fetch_test_case_by_key(self, test_case_key: str) -> TestCase:
        raise NotImplementedError("fetch_test_case_by_key not supported by MeegoClient")

    def fetch_linked_issues(self, test_case_key: str) -> list[dict]:
        raise NotImplementedError("fetch_linked_issues not supported by MeegoClient")

    def link_issue_to_test_case(self, test_case_key: str, issue_id: int, link_type: str) -> None:
        raise NotImplementedError("link_issue_to_test_case not supported by MeegoClient")
