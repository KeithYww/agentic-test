# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Offline, exclusively locked operator tooling for existing durable QA runs."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from common.design_store import DesignStore, RecoveryBlocked


async def main() -> None:
    """Inspect progress or request recovery after stopping the application owner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("action", choices=["list", "show", "resume", "resolve", "new-execution", "upgrade"])
    parser.add_argument("--run-id")
    parser.add_argument("--operation")
    parser.add_argument("--execution-id")
    parser.add_argument("--disposition", choices=["adopt_existing", "verified_absent", "leave_blocked"])
    parser.add_argument("--receipt-file")
    parser.add_argument("--actor")
    parser.add_argument("--evidence")
    parser.add_argument("--previous-manifest")
    parser.add_argument("--changed-file", action="append", default=[])
    args = parser.parse_args()
    if args.action != "list" and not args.run_id:
        parser.error("--run-id is required")
    if args.action in {"resume", "resolve", "new-execution", "upgrade"} and not (args.actor and args.evidence):
        parser.error("--actor and --evidence are required")
    if args.action == "resolve" and not (args.operation and args.disposition):
        parser.error("--operation and --disposition are required")
    if args.action == "new-execution" and not args.execution_id:
        parser.error("--execution-id is required")
    if args.action == "upgrade" and not (args.previous_manifest and args.changed_file):
        parser.error("--previous-manifest and --changed-file are required")
    owner = DesignStore(args.state_dir)
    try:
        await owner.call("open")
        if args.action == "list":
            runs = await owner.call("list")
            print(json.dumps([{key: run[key] for key in ("id", "state", "generation", "used", "reserved", "usage_unknown", "deadline", "token_limit", "error")}
                              for run in runs], ensure_ascii=False, indent=2))
        elif args.action == "show":
            run = await owner.call("run", args.run_id)
            print(json.dumps({"run_id": run["id"], "state": run["state"], "used": run["used"], "reserved": run["reserved"],
                              "operations": [{key: op[key] for key in ("key", "state", "digest")}
                                             for op in await owner.call("operations", args.run_id)]}, indent=2))
        elif args.action == "resume":
            await owner.call("request_resume", args.run_id, args.actor, args.evidence)
            print("已登记续跑申请；使用原状态卷启动应用后接续。")
        elif args.action == "upgrade":
            from common.design_policy import upgrade_manifest

            await owner.call("upgrade", args.run_id, json.loads(Path(args.previous_manifest).read_text()),
                             upgrade_manifest(), args.changed_file, args.actor, args.evidence)
            print("补丁版本已核验并登记续跑；原预算、费用、截止时间和已完成结果保持不变。")
        elif args.action == "new-execution":
            from common.design_policy import workflow_fingerprint

            run = await owner.call("new_execution", args.run_id, args.execution_id, args.actor, args.evidence, workflow_fingerprint())
            print(json.dumps({"run_id": run["id"], "state": run["state"]}))
        else:
            receipt = json.loads(Path(args.receipt_file).read_text()) if args.receipt_file else None
            await owner.call("resolve_publish", args.run_id, args.operation, args.disposition, receipt, args.actor, args.evidence)
            print("已记录人工核实结果；尚未执行任何远端写入。")
    finally:
        await owner.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (RecoveryBlocked, BlockingIOError) as exc:
        print(f"恢复操作未执行：{type(exc).__name__}（{exc}）。请先停止应用 owner 并核实未决结果。", file=sys.stderr)
        sys.exit(1)
