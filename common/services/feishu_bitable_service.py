# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Creates a brand-new, per-run Feishu Bitable populated with generated test cases.

Every pipeline run creates its own Bitable (never writes into an existing/shared one). The
field schema mirrors the team's reference template by name, but with freshly-defined, clean
select options — the reference template's historical option lists (which include garbage
values from past manual data-entry mistakes) are never carried over.

Requires the `lark-cli` binary on PATH, configured with a bot identity that has base/bitable
write scopes and drive comment scopes.
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass

import config
from common import utils
from common.models import ClassifiedTestCase, TestCase
from common.publication_journal import PublicationJournal

logger = utils.get_logger(__name__)

_LARK_CLI_TIMEOUT_SECONDS = 60
_RECORD_BATCH_SIZE = 200


@dataclass(slots=True)
class BitablePublishResult:
    """A URL is useful even when publishing the report was incomplete."""

    url: str | None = None
    expected: int = 0
    written: int = 0
    records_complete: bool = True
    comments_complete: bool = True
    accessible: bool = False

    @property
    def complete(self) -> bool:
        return bool(self.url) and self.expected == self.written and self.records_complete and self.comments_complete and self.accessible

# Fixed, deterministic mappings — no LLM judgment involved.
_PRIORITY_BY_SCENARIO = {"happy_path": "高", "boundary": "中", "exception": "中"}
_SMOKE_BY_SCENARIO = {"happy_path": "是", "boundary": "否", "exception": "否"}
_AUTOMATION_LABEL_MAP = {"automated": "是", "manual": "否", "semi-automated": "部分"}

# Field schema mirrors the reference template's field names/types (see field-list probe against
# example-base-token/example-table-id). Execution-phase fields (执行结果/执行人/备注2 etc.)
# and the two formula fields (季度/业务线) are intentionally NOT recreated here: formula fields
# can't be usefully created from scratch without their source base, and execution-phase fields
# have no meaning until a human runs the test case — they stay reserved for manual use in the UI.
_TABLE_FIELDS: list[dict] = [
    {"name": "用例名称", "type": "text"},
    {"name": "前置步骤", "type": "text"},
    {"name": "步骤描述", "type": "text"},
    {"name": "预期结果", "type": "text"},
    {"name": "所属模块", "type": "text"},
    {"name": "子模块", "type": "text"},
    {"name": "需求名称", "type": "select", "options": []},
    {"name": "优先级", "type": "select", "options": [{"name": "高"}, {"name": "中"}, {"name": "低"}]},
    {"name": "是否作为冒烟", "type": "select", "options": [{"name": "是"}, {"name": "否"}]},
    {"name": "是否可用于自动化", "type": "select", "options": [{"name": "是"}, {"name": "否"}, {"name": "部分"}]},
    {"name": "是否可生产验证", "type": "select", "options": [{"name": "是"}, {"name": "否"}]},
    {"name": "备注1", "type": "text"},
]


def _run_lark_cli(args: list[str], input_text: str | None = None, cwd: str | None = None) -> dict | None:
    """Runs a lark-cli subcommand and returns its parsed JSON response, or None on any failure
    (non-fatal by design — callers log context and continue)."""
    try:
        result = subprocess.run(
            ["lark-cli", *args],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=_LARK_CLI_TIMEOUT_SECONDS,
            cwd=cwd,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"lark-cli call failed ({args[:2]}): {exc}")
        return None
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.warning(
            f"lark-cli returned non-JSON output for {args[:2]}: stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        return None
    if not response.get("ok"):
        logger.warning(f"lark-cli call failed for {args[:2]}: {response.get('error')}")
        return None
    return response


def resolve_business_line_display_name(option_id: str | None) -> str | None:
    """Resolves a story's "业务线" option_id to a display name for labelling the generated
    Bitable's name. Purely informational — does not affect the schema or where records go.
    """
    if not option_id:
        return None
    try:
        mapping = json.loads(config.FEISHU_BUSINESS_LINE_OPTION_NAMES_JSON)
    except json.JSONDecodeError:
        logger.warning("FEISHU_BUSINESS_LINE_OPTION_NAMES_JSON is not valid JSON; cannot resolve business line.")
        return None
    display_name = mapping.get(option_id)
    if display_name is None:
        logger.warning(
            f"业务线 option_id '{option_id}' has no entry in FEISHU_BUSINESS_LINE_OPTION_NAMES_JSON "
            "(distinct from the story having no business line set at all) — the generated "
            "Bitable's name will omit the business line label."
        )
    return display_name


def _flatten_steps(steps: list) -> tuple[str, str]:
    """Renders steps as two parallel numbered text blocks (action / expected results). Since
    the template has no dedicated test-data column, each step's test_data is appended as a
    labelled line under its action rather than woven back into the action's wording — this
    keeps the transform a simple, deterministic serialization with no room to misrepresent the
    generation agent's carefully-separated action/test_data content.
    """
    action_lines, expected_lines = [], []
    for i, step in enumerate(steps, start=1):
        action = step.action
        if step.test_data:
            action = f"{action}\nTest Data: {', '.join(step.test_data)}"
        action_lines.append(f"{i}. {action}")
        expected_lines.append(f"{i}. {step.expected_results}")
    return "\n".join(action_lines), "\n".join(expected_lines)


def _attach_comment(base_token: str, table_id: str, view_id: str, record_id: str, content: str,
                    journal: PublicationJournal | None = None) -> bool:
    args = [
        "drive", "+add-comment",
        "--doc", base_token,
        "--type", "bitable",
        "--block-id", f"{table_id}!{record_id}!{view_id}",
        "--content", json.dumps([{"type": "text", "text": content}], ensure_ascii=False),
        "--as", "bot",
    ]
    resp = (journal.perform(f"comment:{record_id}", args, lambda: _run_lark_cli(args))
            if journal else _run_lark_cli(args))
    if not resp:
        logger.warning(f"Failed to attach a comment to Bitable record {record_id} in table {table_id}.")
    return bool(resp)


def publish_bitable_report(
    story_id: str,
    story_title: str,
    business_line_display_name: str | None,
    test_cases: list[TestCase],
    classified_by_key: dict[str, ClassifiedTestCase],
    review_feedback_by_key: dict[str, list[str]],
    journal: PublicationJournal | None = None,
) -> BitablePublishResult:
    """Publish final cases and report each required boundary's completion."""
    result = BitablePublishResult(expected=len(test_cases))
    base_name = f"用例生成-{story_title}"[:100]
    if business_line_display_name:
        base_name = f"{base_name}（{business_line_display_name}）"

    fields = [dict(f) for f in _TABLE_FIELDS]
    fields.extend({"name": name, "type": "text"} for name in ("用例ID", "需求项", "版本", "评审状态"))
    for field in fields:
        if field["name"] == "需求名称":
            field["options"] = [{"name": story_title[:100]}]

    create_args = [
        "base", "+base-create",
        "--name", base_name,
        "--table-name", "测试用例",
        "--fields", json.dumps(fields, ensure_ascii=False),
        "--as", "bot",
    ]
    def valid_base(response):
        data = (response or {}).get("data", {})
        return bool(data.get("base", {}).get("base_token") and data.get("base", {}).get("url")
                    and data.get("table", {}).get("id"))
    create_resp = (journal.perform("create-base", create_args, lambda: _run_lark_cli(create_args), valid_base)
                   if journal else _run_lark_cli(create_args))
    if not create_resp:
        logger.warning(f"Failed to create a Bitable for story {story_id}.")
        return result

    data = create_resp.get("data", {})
    base_token = data.get("base", {}).get("base_token")
    table_id = data.get("table", {}).get("id")
    base_url = data.get("base", {}).get("url")
    result.url = base_url
    views = data.get("table", {}).get("views") or []
    view_id = views[0]["id"] if views else None
    if not base_token or not table_id or not base_url:
        logger.warning(f"Bitable creation for story {story_id} returned no usable base/table id: {data}")
        return result

    create_records = []
    for tc in test_cases:
        steps_text, expected_text = _flatten_steps(tc.steps)
        record: dict = {
            "用例ID": tc.key or "",
            "需求项": ", ".join(tc.covered_requirement_ids),
            "版本": str(tc.version),
            "评审状态": ("待人工评审（有遗留意见）" if review_feedback_by_key.get(tc.key or "") else
                         "自动评审无遗留意见"),
            "用例名称": tc.name,
            "前置步骤": tc.preconditions or "",
            "步骤描述": steps_text,
            "预期结果": expected_text,
            "需求名称": [story_title[:100]],
            "备注1": "[AI生成]",
        }
        if tc.scenario_type:
            record["优先级"] = [_PRIORITY_BY_SCENARIO.get(tc.scenario_type, "中")]
            record["是否作为冒烟"] = [_SMOKE_BY_SCENARIO.get(tc.scenario_type, "否")]
        classification = classified_by_key.get(tc.key or "")
        if classification:
            automation_value = _AUTOMATION_LABEL_MAP.get(classification.automation_capability)
            if automation_value:
                record["是否可用于自动化"] = [automation_value]
        create_records.append(record)

    written_pairs: list[tuple[str, TestCase]] = []
    for batch_start in range(0, len(create_records), _RECORD_BATCH_SIZE):
        batch = create_records[batch_start:batch_start + _RECORD_BATCH_SIZE]
        batch_test_cases = test_cases[batch_start:batch_start + _RECORD_BATCH_SIZE]
        # A batch's JSON (free-text Chinese steps/expected results, up to 200 records) can run
        # well past the OS's command-line argument length limit if passed inline via --json,
        # failing with OSError: [Errno 7] Argument list too long. lark-cli's --json flag accepts
        # a '@'-prefixed file path (relative to its cwd) instead, so write the payload to a temp
        # file and pass that.
        tmpdir = tempfile.mkdtemp()
        try:
            payload_path = os.path.join(tmpdir, "batch.json")
            with open(payload_path, "w", encoding="utf-8") as f:
                json.dump({"create_records": batch}, f, ensure_ascii=False)
            write_args = [
                    "base", "+record-batch-create",
                    "--base-token", base_token,
                    "--table-id", table_id,
                    "--json", "@./batch.json",
                    "--as", "bot",
                ]
            def write(args: list[str] = write_args, directory: str = tmpdir) -> dict | None:
                return _run_lark_cli(args, cwd=directory)
            def valid_records(response: dict | None, expected: int = len(batch)) -> bool:
                ids = (response or {}).get("data", {}).get("record_id_list", [])
                return len(ids) == expected and len(set(ids)) == len(ids) and all(ids)
            write_resp = (journal.perform(f"records:{batch_start}", {"base": base_token, "table": table_id, "records": batch},
                                           write, valid_records) if journal else write())
        finally:
            try:
                os.remove(payload_path)
                os.rmdir(tmpdir)
            except OSError as e:
                logger.warning(f"Failed to clean up temp dir {tmpdir}: {e}")
        if not write_resp:
            logger.warning(
                f"Failed to write a batch of {len(batch)} test cases for story {story_id} to Bitable {base_token}."
            )
            continue
        batch_record_ids = write_resp.get("data", {}).get("record_id_list", [])
        existing_ids = {record_id for record_id, _ in written_pairs}
        if len(batch_record_ids) != len(batch) or len(set(batch_record_ids)) != len(batch_record_ids) or existing_ids.intersection(batch_record_ids):
            result.records_complete = False
        # Pair each returned record_id with the test case at the same position within this
        # batch only — never with a flat, cross-batch index — so a failed earlier batch can't
        # shift a later batch's record_ids out of alignment with their test cases.
        written_pairs.extend(zip(batch_record_ids, batch_test_cases, strict=False))

    if test_cases and not written_pairs:
        logger.error(
            f"Bitable {base_token} for story {story_id} was created but every batch write failed "
            f"— 0 of {len(test_cases)} test cases were written. Treating as a failed report."
        )
        return result

    if view_id:
        for record_id, tc in written_pairs:
            feedback = review_feedback_by_key.get(tc.key or "")
            if feedback:
                attached = (_attach_comment(base_token, table_id, view_id, record_id, "\n".join(feedback), journal)
                            if journal else _attach_comment(base_token, table_id, view_id, record_id, "\n".join(feedback)))
                if not attached:
                    result.comments_complete = False
    elif any(review_feedback_by_key.values()):
        result.comments_complete = False

    logger.info(f"Created Bitable report for story {story_id}: {base_url} ({len(written_pairs)} test cases written).")

    # Open the Bitable to anyone in the org with the link — avoids per-user permission grants
    # and lets QA engineers edit directly without needing to be individually added as collaborators.
    result.written = len(written_pairs)
    result.accessible = (journal.perform("tenant-permission", {"base": base_token, "permission": "tenant_editable"},
                                         lambda: _set_tenant_editable(base_token))
                         if journal else _set_tenant_editable(base_token))

    if config.FEISHU_DOC_COLLABORATOR_IDS:
        _grant_bitable_edit_permission(base_token, config.FEISHU_DOC_COLLABORATOR_IDS)

    return result


def create_bitable_report(
    story_id: str, story_title: str, business_line_display_name: str | None,
    test_cases: list[TestCase], classified_by_key: dict[str, ClassifiedTestCase],
    review_feedback_by_key: dict[str, list[str]],
) -> str | None:
    """Compatibility entry point; incomplete reports are no longer reported as success."""
    result = publish_bitable_report(story_id, story_title, business_line_display_name,
                                    test_cases, classified_by_key, review_feedback_by_key)
    return result.url if result.complete else None


def _set_tenant_editable(base_token: str) -> bool:
    """Set the Bitable's link-share permission to 'tenant_editable' (anyone in the org with
    the link can edit). Must be called as bot since the Bitable is bot-owned. Non-fatal."""
    try:
        result = subprocess.run(
            [
                "lark-cli", "drive", "permission.public", "patch",
                "--token", base_token,
                "--type", "bitable",
                "--data", '{"link_share_entity": "tenant_editable"}',
                "--as", "bot",
                "--yes",
                "--format", "json",
            ],
            capture_output=True, text=True, timeout=30,
        )
        resp = json.loads(result.stdout)
        if resp.get("ok"):
            logger.info(f"Set Bitable {base_token} to tenant_editable.")
            return True
        else:
            logger.warning(f"Could not set tenant_editable on Bitable {base_token}: {resp.get('error')}")
    except Exception as e:
        logger.warning(f"Could not set tenant_editable on Bitable {base_token}: {e}")
    return False


def _grant_bitable_edit_permission(base_token: str, user_ids: list[str]) -> None:
    """Proactively grant edit access to each user on a bot-owned Bitable.

    Uses drive permission.members create (--type bitable) instead of the apply→approve flow,
    which is dead-on-arrival for bot-owned Bitables. Non-fatal: failures log as warnings only.
    """
    for uid in user_ids:
        try:
            result = subprocess.run(
                [
                    "lark-cli", "drive", "permission.members", "create",
                    "--token", base_token,
                    "--type", "bitable",
                    "--as", "bot",
                    "--data", json.dumps({
                        "member_type": "openid",
                        "member_id": uid,
                        "perm": "edit",
                        "type": "user",
                    }),
                    "--yes",
                ],
                capture_output=True,
                text=True,
                timeout=_LARK_CLI_TIMEOUT_SECONDS,
            )
            resp = json.loads(result.stdout) if result.stdout else {}
            if resp.get("ok"):
                logger.info(f"Granted edit access on Bitable {base_token} to {uid}")
            else:
                logger.warning(f"Could not grant access on Bitable {base_token} to {uid}: {resp.get('error')}")
        except Exception as exc:
            logger.warning(f"Exception granting edit access on Bitable {base_token} to {uid}: {exc}")
