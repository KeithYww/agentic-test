# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Stateful recording mock for the Feishu Project (Meego) Open API surface.

Serves the endpoints ``MeegoClient`` exercises: plugin-token auth, work-item
create, comment create (including the typed work-item path), status
transition, field update (labels), and the AI-node edit/confirm endpoints used
to write the consolidated report doc URL onto the "ai用例预生成" node. Records
every creation/mutation in memory and exposes the aggregate state at
``GET /__recorded`` for the smoke assertions.
"""

import json
import time

from fastapi import FastAPI, Request

app = FastAPI()

# Keyed by work_item_id; each value holds the merged state for that work item.
_work_items: dict[str, dict] = {}
_counter = 0
_lark_cli_content = ""
_lark_cli_bitable_records: list[dict] = []
_bitable_comments: list = []
_confirmations: list = []


def _get_or_create(work_item_id: str) -> dict:
    if work_item_id not in _work_items:
        _work_items[work_item_id] = {
            "work_item_id": work_item_id,
            "name": "",
            "steps": "",
            "expected": "",
            "related_story": "",
            "labels": [],
            "comments": [],
            "transitions": [],
            "test_case_doc": "",
        }
    return _work_items[work_item_id]


@app.post("/open_api/authen/plugin_token")
async def get_plugin_token() -> dict:
    return {"code": 0, "data": {"token": "smoke-plugin-token", "expire_time": 7200}}


@app.post("/open_api/{project_key}/work_item/create")
async def create_work_item(project_key: str, request: Request) -> dict:
    global _counter
    payload = await request.json()
    _counter += 1
    work_item_id = f"MEEGO-{_counter}"
    entry = _get_or_create(work_item_id)
    entry["project_key"] = project_key
    for field in payload.get("fields", []):
        key = field.get("field_key", "")
        value = field.get("field_value", "")
        if key == "name":
            entry["name"] = value
        elif key == "field_steps":
            entry["steps"] = value
        elif key == "field_expected":
            entry["expected"] = value
        elif key == "description":
            entry["description"] = value
        elif key == "field_related_story":
            entry["related_story"] = value
    return {"code": 0, "data": {"work_item_id": work_item_id}}


@app.post("/open_api/{project_key}/work_item/{work_item_id}/comment/create")
async def add_comment(project_key: str, work_item_id: str, request: Request) -> dict:
    payload = await request.json()
    entry = _get_or_create(work_item_id)
    entry["comments"].append(payload.get("content", ""))
    return {"code": 0}


@app.post("/open_api/{project_key}/work_item/{work_item_type_key}/{work_item_id}/comment/create")
async def add_typed_comment(project_key: str, work_item_type_key: str, work_item_id: str, request: Request) -> dict:
    """Record a comment addressed by work-item type and ID in the path."""
    payload = await request.json()
    entry = _get_or_create(work_item_id)
    entry["project_key"] = project_key
    entry["comments"].append(payload.get("content", ""))
    return {"code": 0}


@app.post("/open_api/{project_key}/work_item/{work_item_id}/transition_state")
async def transition_state(project_key: str, work_item_id: str, request: Request) -> dict:
    payload = await request.json()
    entry = _get_or_create(work_item_id)
    entry["transitions"].append(payload.get("transition_status_id", ""))
    return {"code": 0}


@app.post("/open_api/{project_key}/work_item/{work_item_type_key}/{work_item_id}")
async def update_work_item(project_key: str, work_item_type_key: str, work_item_id: str, request: Request) -> dict:
    payload = await request.json()
    entry = _get_or_create(work_item_id)
    for field in payload.get("fields", []):
        if field.get("field_key") == "field_labels":
            entry["labels"] = field.get("field_value", [])
    return {"code": 0}


@app.post("/open_api/ai_node/edit")
async def edit_ai_node(request: Request) -> dict:
    """Serves MeegoClient.update_story_test_case_doc's write to the "ai用例预生成" node."""
    payload = await request.json()
    entry = _get_or_create(str(payload.get("work_item_id", "")))
    for field in payload.get("conform_fields", []):
        if field.get("field_key") == "field_review_comments":
            entry["test_case_doc"] = json.loads(field["value"])["doc_text"]
    return {"err": {}, "err_code": 0, "err_msg": "", "data": {}}


@app.post("/open_api/ai_node/confirm")
async def confirm_ai_node(request: Request) -> dict:
    _confirmations.append({"payload": await request.json(), "at": time.time()})
    return {"err": {}, "err_code": 0, "err_msg": "", "data": {}}


@app.post("/__lark_cli_content")
async def record_lark_cli_content(request: Request) -> dict:
    """Records the Markdown the lark-cli shim received via stdin (see lark_cli_shim.sh), so the
    smoke suite can verify classification/review output actually reached the final report."""
    global _lark_cli_content
    _lark_cli_content = (await request.body()).decode("utf-8")
    return {"ok": True}


@app.post("/__lark_cli_bitable_records")
async def record_lark_cli_bitable_records(request: Request) -> dict:
    """Records the batch-create payload the lark-cli shim read from its @-file (see
    lark_cli_shim.sh), so the smoke suite can verify the Bitable actually received records."""
    global _lark_cli_bitable_records
    payload = await request.json()
    _lark_cli_bitable_records.extend(payload.get("create_records", []))
    return {"ok": True}


@app.post("/__bitable_comment")
async def record_bitable_comment(request: Request) -> dict:
    _bitable_comments.append(json.loads(await request.body()))
    return {"ok": True}


@app.get("/__recorded")
async def recorded() -> dict:
    return {
        "test_cases": list(_work_items.values()),
        "lark_cli_content": _lark_cli_content,
        "bitable_comments": _bitable_comments,
        "confirmations": _confirmations,
        "lark_cli_bitable_records": _lark_cli_bitable_records,
    }
