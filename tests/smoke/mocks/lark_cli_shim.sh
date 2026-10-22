#!/bin/sh
# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

# Fake lark-cli for the hermetic smoke stack: the real binary would call the real Feishu
# Open API over the network, which the smoke topology must not do. Mounted over the real
# lark-cli install (see docker-compose.smoke.yml) for the orchestrator service only.
#
# Supports just the subcommands common/services/feishu_doc_service.py,
# common/services/feishu_bitable_service.py and orchestrator/main.py's _init_lark_cli
# actually invoke.

case "$1 $2" in
  "config init")
    echo '{"ok": true}'
    ;;
  "docs +fetch")
    DOC_ID=""
    while [ "$#" -gt 0 ]; do
      if [ "$1" = "--doc" ]; then DOC_ID="$2"; fi
      shift
    done
    curl -fsS --get --data-urlencode "doc_id=$DOC_ID" "${FEISHU_API_BASE_URL}/__docxml"
    ;;
  "sheets +workbook-info")
    curl -fsS "${FEISHU_API_BASE_URL}/__workbook"
    ;;
  "sheets +cells-get")
    curl -fsS "${FEISHU_API_BASE_URL}/__cells"
    ;;
  "docs +create")
    # Drain stdin (the real CLI reads the Markdown content from it via --content -), forwarding
    # it to meego_mock so the smoke suite can assert classification/review content made it into
    # the final report (see MEEGO_BASE_URL in docker-compose.smoke.yml for the host:port).
    curl -sS -X POST http://meego_mock:8095/__lark_cli_content --data-binary @- >/dev/null
    echo '{"ok": true, "data": {"url": "https://smoke.feishu.cn/docx/SMOKE-REPORT-DOC"}}'
    ;;
  "base +base-create")
    echo '{"ok": true, "data": {"base": {"base_token": "SMOKE-BT1", "url": "https://smoke.feishu.cn/base/SMOKE-BT1"}, "table": {"id": "tblSMOKE1", "views": [{"id": "vewSMOKE1"}]}}}'
    ;;
  "base +record-batch-create")
    JSON_ARG=""
    while [ "$#" -gt 0 ]; do
      if [ "$1" = "--json" ]; then
        JSON_ARG="$2"
      fi
      shift
    done
    case "$JSON_ARG" in
      @./*)
        # Regression guard: the real bug (Errno 7: Argument list too long) was caused by
        # passing a full batch's JSON inline via --json instead of via an @-file reference.
        FILE_PATH="${JSON_ARG#@}"
        curl -sS -X POST http://meego_mock:8095/__lark_cli_bitable_records --data-binary "@$FILE_PATH" >/dev/null
        COUNT=$(python3 -c "import json; print(len(json.load(open('$FILE_PATH'))['create_records']))")
        python3 -c "print('{\"ok\": true, \"data\": {\"record_id_list\": [' + ','.join(f'\"SMOKE-REC-{i}\"' for i in range(1, $COUNT + 1)) + ']}}')"
        ;;
      *)
        echo '{"ok": false, "error": {"message": "shim: --json must use an @./file reference, not inline JSON"}}' >&2
        exit 1
        ;;
    esac
    ;;
  "drive +add-comment")
    CONTENT=""
    while [ "$#" -gt 0 ]; do
      if [ "$1" = "--content" ]; then CONTENT="$2"; fi
      shift
    done
    curl -fsS -X POST http://meego_mock:8095/__bitable_comment --data-binary "$CONTENT" >/dev/null || exit 1
    echo '{"ok": true}'
    ;;
  "drive permission.public")
    echo '{"ok": true}'
    ;;
  *)
    echo "{\"ok\": false, \"error\": {\"message\": \"lark-cli shim: unsupported subcommand: $1 $2\"}}" >&2
    exit 1
    ;;
esac
