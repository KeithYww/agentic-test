# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Single-owner SQLite state for QA runs, paid attempts and remote write intents."""

import asyncio
import fcntl
import hashlib
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4


def digest(value: Any) -> str:
    """Fingerprint JSON business inputs without storing them in error messages."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


class RecoveryBlocked(RuntimeError):
    """An ambiguous or incompatible operation requires evidence before recovery."""


class DesignStore:
    """Serialize short transactions on one connection in a dedicated thread."""

    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qa-state")
        self._connection: sqlite3.Connection | None = None
        self._lock_file = None

    async def call(self, action: str, *args: Any) -> Any:
        """Execute one store operation without blocking the application event loop."""
        return await asyncio.get_running_loop().run_in_executor(self._executor, self._dispatch, action, args)

    def call_sync(self, action: str, *args: Any) -> Any:
        """Use the same owner from an existing blocking CLI/HTTP service thread."""
        return self._executor.submit(self._dispatch, action, args).result()

    def _dispatch(self, action: str, args: tuple) -> Any:
        return getattr(self, f"_{action}")(*args)

    def _open(self) -> None:
        if not self.directory.is_dir():
            raise RecoveryBlocked("state_directory_missing")
        self._lock_file = (self.directory / "qa_design.lock").open("a")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._connection = sqlite3.connect(self.directory / "qa_design.sqlite", timeout=5)
            self._connection.row_factory = sqlite3.Row
            version = self._connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2):
                raise RecoveryBlocked("state_schema_incompatible")
            self._connection.executescript("""
                PRAGMA journal_mode=DELETE;
                PRAGMA synchronous=FULL;
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, business_key TEXT NOT NULL UNIQUE,
                    state TEXT NOT NULL, generation INTEGER NOT NULL, spec TEXT NOT NULL,
                    snapshot TEXT NOT NULL DEFAULT '{}', deadline REAL NOT NULL,
                    token_limit INTEGER NOT NULL, error TEXT
                );
                CREATE TABLE IF NOT EXISTS entries (
                    run_id TEXT NOT NULL REFERENCES runs(id), namespace TEXT NOT NULL,
                    key TEXT NOT NULL, digest TEXT NOT NULL, state TEXT NOT NULL,
                    value TEXT, PRIMARY KEY(run_id, namespace, key)
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
                    generation INTEGER NOT NULL, task_key TEXT NOT NULL, digest TEXT NOT NULL,
                    state TEXT NOT NULL, reserved INTEGER NOT NULL, used INTEGER NOT NULL DEFAULT 0,
                    response TEXT, reusable INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(run_id, task_key, digest)
                );
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, event TEXT NOT NULL,
                    actor TEXT NOT NULL, evidence TEXT NOT NULL, created REAL NOT NULL
                );
            """)
            if version < 2:
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    for definition in (
                        "send_count INTEGER NOT NULL DEFAULT 0",
                        "next_attempt_at REAL NOT NULL DEFAULT 0",
                        "wait_budget_used_seconds REAL NOT NULL DEFAULT 0",
                        "policy_version INTEGER NOT NULL DEFAULT 1",
                        "retry_error TEXT",
                    ):
                        self._connection.execute(f"ALTER TABLE attempts ADD COLUMN {definition}")
                    self._connection.execute("""
                        CREATE TABLE provider_attempts (
                            id TEXT PRIMARY KEY, logical_attempt_id TEXT NOT NULL REFERENCES attempts(id),
                            ordinal INTEGER NOT NULL, generation INTEGER NOT NULL, model_name TEXT NOT NULL,
                            state TEXT NOT NULL, started_at REAL NOT NULL, finished_at REAL,
                            http_status INTEGER, failure_kind TEXT, provider_request_id TEXT,
                            rejection_contract_version TEXT, used_tokens INTEGER,
                            UNIQUE(logical_attempt_id, ordinal)
                        )
                    """)
                    self._connection.execute("PRAGMA user_version=2")
                    self._connection.commit()
                except BaseException:
                    self._connection.rollback()
                    raise
            os.chmod(self.directory / "qa_design.sqlite", 0o600)
            with self._connection:
                self._connection.execute("UPDATE runs SET state='paused', error='owner_interrupted' WHERE state='running'")
                self._connection.execute("UPDATE attempts SET state='unknown' WHERE state='in_flight'")
                self._connection.execute("UPDATE provider_attempts SET state='unknown' WHERE state='in_flight'")
                self._connection.execute("UPDATE entries SET state='unknown' WHERE namespace='publish' AND state='in_flight'")
        except BaseException:
            self._close()
            raise

    def _close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._lock_file is not None:
            self._lock_file.close()
            self._lock_file = None

    async def close(self) -> None:
        """Drain writes and release the owner lock before shutting down the executor."""
        await self.call("close")
        self._executor.shutdown(wait=True)

    def _run(self, run_id: str, generation: int | None = None) -> dict:
        row = self._connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise RecoveryBlocked("run_not_found")
        run = dict(row)
        if generation is not None and (run["generation"] != generation or run["state"] != "running"):
            raise RecoveryBlocked("stale_run_generation")
        run["spec"] = json.loads(run["spec"])
        run["snapshot"] = json.loads(run["snapshot"])
        usage = self._connection.execute(
            "SELECT COALESCE(SUM(used),0), COALESCE(SUM(CASE WHEN state IN ('in_flight','unknown') THEN reserved ELSE 0 END),0), SUM(state='unknown') FROM attempts WHERE run_id=?",
            (run_id,),
        ).fetchone()
        run.update(used=usage[0], reserved=usage[1], usage_unknown=bool(usage[2]))
        return run

    def _create_run(self, business_key: str, spec: dict, limit: int, timeout: float) -> tuple[dict, bool]:
        existing = self._connection.execute("SELECT id FROM runs WHERE business_key=?", (business_key,)).fetchone()
        if existing:
            run = self._run(existing[0])
            if run["spec"] != spec:
                raise RecoveryBlocked("run_input_changed")
            return run, False
        run_id = str(uuid4())
        with self._connection:
            self._connection.execute(
                "INSERT INTO runs(id,business_key,state,generation,spec,deadline,token_limit) VALUES(?,?,'running',1,?,?,?)",
                (run_id, business_key, json.dumps(spec), time.time() + timeout, limit),
            )
        return self._run(run_id), True

    def _snapshot(self, run_id: str, generation: int, snapshot: dict) -> None:
        if "revision" in snapshot:
            raise RecoveryBlocked("revision_requires_atomic_commit")
        current = self._run(run_id, generation)["snapshot"]
        current.update(snapshot)
        with self._connection:
            self._connection.execute("UPDATE runs SET snapshot=? WHERE id=?", (json.dumps(current, ensure_ascii=False), run_id))

    def _state(self, run_id: str, generation: int, state: str, error: str | None = None) -> None:
        self._run(run_id, generation)
        if state not in {"paused", "completed"}:
            raise ValueError("Invalid terminal run state")
        with self._connection:
            self._connection.execute("UPDATE runs SET state=?,error=? WHERE id=?", (state, error, run_id))

    def _get(self, run_id: str, namespace: str, key: str, fingerprint: str) -> dict | None:
        row = self._connection.execute(
            "SELECT * FROM entries WHERE run_id=? AND namespace=? AND key=?", (run_id, namespace, key),
        ).fetchone()
        if row is None:
            return None
        if row["digest"] != fingerprint:
            raise RecoveryBlocked("unit_input_changed")
        return {"state": row["state"], "value": json.loads(row["value"]) if row["value"] is not None else None}

    def _put(self, run_id: str, generation: int, namespace: str, key: str, fingerprint: str, value: Any) -> None:
        run = self._run(run_id, generation)
        existing = self._get(run_id, namespace, key, fingerprint)
        if existing and existing["state"] == "completed":
            if existing["value"] != value:
                raise RecoveryBlocked("completed_unit_changed")
            return
        with self._connection:
            if namespace == "revision":
                if int(key) != run["snapshot"].get("revision", -1) + 1:
                    raise RecoveryBlocked("revision_compare_and_swap_failed")
                run["snapshot"]["revision"] = int(key)
                self._connection.execute("UPDATE runs SET snapshot=? WHERE id=?", (json.dumps(run["snapshot"]), run_id))
            self._connection.execute(
                "INSERT INTO entries VALUES(?,?,?,?,'completed',?) ON CONFLICT(run_id,namespace,key) DO UPDATE SET state='completed',value=excluded.value",
                (run_id, namespace, key, fingerprint, json.dumps(value, ensure_ascii=False)),
            )

    def _reserve(self, run_id: str, generation: int, task_key: str, fingerprint: str, reservation: int, ceiling: int) -> dict:
        run = self._run(run_id, generation)
        if run["usage_unknown"] or time.time() >= run["deadline"]:
            raise RecoveryBlocked("usage_unknown_or_deadline")
        row = self._connection.execute(
            "SELECT * FROM attempts WHERE run_id=? AND task_key=? AND digest=?", (run_id, task_key, fingerprint),
        ).fetchone()
        if row:
            if row["state"] != "settled" or not row["reusable"]:
                raise RecoveryBlocked("model_result_not_replayable")
            return {"id": row["id"], "response": json.loads(row["response"])}
        if run["used"] + run["reserved"] + reservation > min(run["token_limit"], ceiling):
            raise RecoveryBlocked("persistent_budget_preflight")
        attempt_id = str(uuid4())
        with self._connection:
            self._connection.execute(
                "INSERT INTO attempts(id,run_id,generation,task_key,digest,state,reserved) VALUES(?,?,?,?,?,'in_flight',?)",
                (attempt_id, run_id, generation, task_key, fingerprint, reservation),
            )
        return {"id": attempt_id, "response": None}

    def _settle(self, run_id: str, generation: int, attempt_id: str, used: int, response: dict, reusable: bool) -> None:
        self._run(run_id, generation)
        with self._connection:
            changed = self._connection.execute(
                "UPDATE attempts SET used=?,response=?,reusable=?,state='settled' WHERE id=? AND run_id=? AND generation=? AND state='in_flight'",
                (used, json.dumps(response, ensure_ascii=False), reusable, attempt_id, run_id, generation),
            ).rowcount
            if changed != 1:
                raise RecoveryBlocked("attempt_already_settled_or_stale")

    def _not_sent(self, run_id: str, generation: int, attempt_id: str) -> None:
        self._run(run_id, generation)
        with self._connection:
            self._connection.execute(
                "DELETE FROM attempts WHERE id=? AND run_id=? AND generation=? AND state='in_flight' AND used=0",
                (attempt_id, run_id, generation),
            )

    def _unknown(self, run_id: str, generation: int, attempt_id: str) -> None:
        self._run(run_id, generation)
        with self._connection:
            self._connection.execute("UPDATE attempts SET state='unknown' WHERE id=? AND state='in_flight'", (attempt_id,))

    def _provider_reserve(
        self, run_id: str, generation: int, task_key: str, fingerprint: str,
        reservation: int, ceiling: int, model_name: str,
        allow_retry: bool = True,
    ) -> dict:
        run = self._run(run_id, generation)
        if run["usage_unknown"] or time.time() >= run["deadline"]:
            raise RecoveryBlocked("usage_unknown_or_deadline")
        row = self._connection.execute(
            "SELECT * FROM attempts WHERE run_id=? AND task_key=? AND digest=?",
            (run_id, task_key, fingerprint),
        ).fetchone()
        if row:
            if row["state"] == "settled" and row["reusable"]:
                return {"state": "cached", "response": json.loads(row["response"])}
            if row["state"] == "rejected":
                return {"state": "rejected", "code": row["retry_error"]}
            if row["state"] == "incomplete":
                # A truncated stream is already paid for: resend it at most once per confirmed generation.
                truncated = self._connection.execute(
                    "SELECT COUNT(*) FROM provider_attempts WHERE logical_attempt_id=? AND generation=? AND state='incomplete'",
                    (row["id"], generation),
                ).fetchone()[0]
                if truncated >= 2:
                    return {"state": "rejected", "code": "provider_stream_incomplete"}
            elif row["state"] not in {"retry_wait", "not_sent"}:
                raise RecoveryBlocked("model_result_not_replayable")
            elif not allow_retry and row["state"] == "retry_wait":
                with self._connection:
                    self._connection.execute(
                        "UPDATE attempts SET state='rejected',retry_error='provider_busy_exhausted' WHERE id=?",
                        (row["id"],),
                    )
                return {"state": "rejected", "code": "provider_busy_exhausted"}
            elif row["policy_version"] != 1 or row["send_count"] >= 3:
                raise RecoveryBlocked("provider_policy_exhausted")
            elif row["next_attempt_at"] > time.time():
                return {"state": "waiting", "next_attempt_at": row["next_attempt_at"]}
        if run["used"] + run["reserved"] + reservation > min(run["token_limit"], ceiling):
            raise RecoveryBlocked("persistent_budget_preflight")
        attempt_id = row["id"] if row else str(uuid4())
        ordinal = row["send_count"] + 1 if row else 1
        child_id = str(uuid4())
        with self._connection:
            if row:
                changed = self._connection.execute(
                    "UPDATE attempts SET state='in_flight',generation=?,reserved=?,send_count=send_count+1 "
                    "WHERE id=? AND state IN ('retry_wait','not_sent','incomplete') AND send_count=? AND next_attempt_at<=? "
                    "AND NOT EXISTS (SELECT 1 FROM provider_attempts WHERE logical_attempt_id=? "
                    "AND state IN ('in_flight','unknown'))",
                    (generation, reservation, attempt_id, row["send_count"], time.time(), attempt_id),
                ).rowcount
                if changed != 1:
                    raise RecoveryBlocked("provider_attempt_already_claimed")
            else:
                self._connection.execute(
                    "INSERT INTO attempts(id,run_id,generation,task_key,digest,state,reserved,send_count) "
                    "VALUES(?,?,?,?,?,'in_flight',?,1)",
                    (attempt_id, run_id, generation, task_key, fingerprint, reservation),
                )
            self._connection.execute(
                "INSERT INTO provider_attempts(id,logical_attempt_id,ordinal,generation,model_name,state,started_at) "
                "VALUES(?,?,?,?,?,'in_flight',?)",
                (child_id, attempt_id, ordinal, generation, model_name, time.time()),
            )
        return {
            "state": "ready", "id": attempt_id, "child_id": child_id, "ordinal": ordinal,
            "waited": row["wait_budget_used_seconds"] if row else 0,
        }

    def _provider_finish(self, run_id: str, generation: int, permit: dict, receipt: dict) -> None:
        self._run(run_id, generation)
        row = self._connection.execute(
            "SELECT * FROM attempts WHERE id=? AND run_id=? AND generation=?",
            (permit["id"], run_id, generation),
        ).fetchone()
        child = self._connection.execute(
            "SELECT * FROM provider_attempts WHERE id=? AND logical_attempt_id=? AND generation=?",
            (permit["child_id"], permit["id"], generation),
        ).fetchone()
        if row is None or child is None:
            raise RecoveryBlocked("provider_attempt_stale")
        if row["state"] == "settled" and receipt["state"] == "unknown":
            return
        if row["state"] != "in_flight" or child["state"] != "in_flight":
            raise RecoveryBlocked("provider_attempt_already_finished")
        state = receipt["state"]
        if state not in {"settled", "rejected", "unknown", "not_sent", "incomplete"}:
            raise ValueError("Invalid provider receipt state")
        if state == "rejected" and (
            receipt.get("contract") != "photon-preflight-v1" or receipt.get("status") != 503
        ):
            raise ValueError("Unverified provider rejection receipt")
        delay = receipt.get("delay")
        if state == "rejected" and delay is not None and (
            delay < 0 or row["send_count"] >= 3 or row["wait_budget_used_seconds"] + delay > 30
        ):
            raise RecoveryBlocked("provider_retry_budget_exceeded")
        parent_state = "retry_wait" if state == "rejected" and delay is not None else state
        with self._connection:
            self._connection.execute(
                "UPDATE provider_attempts SET state=?,finished_at=?,http_status=?,failure_kind=?,"
                "provider_request_id=?,rejection_contract_version=?,used_tokens=? WHERE id=?",
                (state, time.time(), receipt.get("status"), receipt.get("failure_kind"), receipt.get("request_id"),
                 receipt.get("contract"), receipt.get("used", 0 if state in {"rejected", "not_sent"} else None), permit["child_id"]),
            )
            # Sends of one logical attempt are each billed; a resend must not overwrite earlier usage.
            self._connection.execute(
                "UPDATE attempts SET state=?,reserved=?,used=used+?,response=?,reusable=?,next_attempt_at=?,"
                "wait_budget_used_seconds=wait_budget_used_seconds+?,retry_error=? WHERE id=?",
                (parent_state, row["reserved"] if state == "unknown" else 0, receipt.get("used", 0),
                 json.dumps(receipt["response"]) if "response" in receipt else None,
                 receipt.get("reusable", False), time.time() + delay if delay is not None else 0,
                 delay or 0, receipt.get("code"), permit["id"]),
            )

    def _provider_attempts(self, run_id: str) -> list[dict]:
        return [dict(row) for row in self._connection.execute(
            "SELECT p.* FROM provider_attempts p JOIN attempts a ON a.id=p.logical_attempt_id "
            "WHERE a.run_id=? ORDER BY p.started_at,p.ordinal", (run_id,),
        )]

    def _publish_begin(self, run_id: str, generation: int, key: str, payload: Any) -> dict | None:
        self._run(run_id, generation)
        fingerprint = digest(payload)
        previous = self._get(run_id, "publish", key, fingerprint)
        if previous:
            if previous["state"] == "verified_absent":
                with self._connection:
                    self._connection.execute("UPDATE entries SET state='in_flight' WHERE run_id=? AND namespace='publish' AND key=?", (run_id, key))
                return None
            if previous["state"] != "completed":
                raise RecoveryBlocked(f"remote_outcome_unknown:{key}")
            return previous["value"]
        with self._connection:
            self._connection.execute("INSERT INTO entries VALUES(?,'publish',?,?,'in_flight',?)",
                                     (run_id, key, fingerprint, json.dumps({"intent": payload}, ensure_ascii=False)))
        return None

    def _list(self) -> list[dict]:
        return [self._run(row[0]) for row in self._connection.execute("SELECT id FROM runs").fetchall()]

    def _operation_usage(self, run_id: str, task_key: str) -> int:
        return self._connection.execute(
            "SELECT COALESCE(SUM(used),0) FROM attempts WHERE run_id=? AND (task_key=? OR task_key LIKE ?)",
            (run_id, task_key, task_key + "/%"),
        ).fetchone()[0]

    def _request_resume(self, run_id: str, actor: str, evidence: str) -> None:
        run = self._run(run_id)
        if run["state"] == "resume_requested":
            return
        self._check_resume(run)
        with self._connection:
            self._connection.execute("UPDATE runs SET state='resume_requested' WHERE id=?", (run_id,))
            self._connection.execute("INSERT INTO audit(run_id,event,actor,evidence,created) VALUES(?,'resume',?,?,?)",
                                     (run_id, actor, evidence, time.time()))

    def _check_resume(self, run: dict) -> None:
        unresolved = self._connection.execute(
            "SELECT 1 FROM entries WHERE run_id=? AND namespace='publish' AND state NOT IN ('completed','verified_absent')", (run["id"],),
        ).fetchone()
        if run["state"] != "paused" or run["usage_unknown"] or run["reserved"] or unresolved or time.time() >= run["deadline"] or run["used"] >= run["token_limit"]:
            raise RecoveryBlocked("resume_preconditions_not_met")

    def _resume_claim(self, run_id: str, generation: int, actor: str) -> dict:
        """Claim one confirmed web request atomically on the owner thread."""
        run = self._run(run_id)
        if run["generation"] != generation:
            raise RecoveryBlocked("stale_run_generation")
        self._check_resume(run)
        self._request_resume(run_id, actor, "Dashboard user confirmed checkpoint resume")
        return self._claim(run_id)

    def _upgrade(self, run_id: str, previous: dict, current: dict, changed_files: list[str],
                 actor: str, evidence: str) -> None:
        run = self._run(run_id)
        self._check_resume(run)
        if not actor.strip() or not evidence.strip():
            raise RecoveryBlocked("operator_evidence_required")
        old, new = previous["workflow"], current["workflow"]
        if digest(old) != run["spec"].get("workflow"):
            raise RecoveryBlocked("previous_manifest_mismatch")
        if (previous["config_sha256"] != current["config_sha256"]
                or previous["environment_sha256"] != current["environment_sha256"]
                or any(old[key] != new[key] for key in ("policy", "targets", "provider"))):
            raise RecoveryBlocked("upgrade_changes_policy_or_targets")
        old_sources = {**old["sources"], **previous["runtime_sources"]}
        new_sources = {**new["sources"], **current["runtime_sources"]}
        changed = {path for path in old_sources.keys() | new_sources.keys()
                   if old_sources.get(path) != new_sources.get(path)}
        if not changed or changed != set(changed_files):
            raise RecoveryBlocked("upgrade_changes_not_reviewed")
        spec = {**run["spec"], "workflow": digest(new), "upgrade_manifest": digest(current)}
        audit = json.dumps({"reason": evidence, "previous": digest(old), "current": digest(new),
                            "changed_files": sorted(changed), "upgrade_manifest": digest(current)}, ensure_ascii=False)
        with self._connection:
            self._connection.execute("UPDATE runs SET spec=?,state='resume_requested' WHERE id=?",
                                     (json.dumps(spec), run_id))
            self._connection.execute("INSERT INTO audit(run_id,event,actor,evidence,created) VALUES(?,'upgrade',?,?,?)",
                                     (run_id, actor, audit, time.time()))

    def _claim(self, run_id: str) -> dict:
        run = self._run(run_id)
        unresolved = self._connection.execute(
            "SELECT 1 FROM entries WHERE run_id=? AND namespace='publish' AND state NOT IN ('completed','verified_absent')", (run_id,),
        ).fetchone()
        if run["state"] != "resume_requested" or run["usage_unknown"] or run["reserved"] or unresolved or time.time() >= run["deadline"]:
            raise RecoveryBlocked("resume_preconditions_not_met")
        with self._connection:
            self._connection.execute("UPDATE runs SET state='running',generation=generation+1,error=NULL WHERE id=?", (run_id,))
        return self._run(run_id)

    def _reject_resume(self, run_id: str, reason: str) -> None:
        run = self._run(run_id)
        if run["state"] != "resume_requested":
            raise RecoveryBlocked("resume_not_requested")
        with self._connection:
            self._connection.execute("UPDATE runs SET state='paused',error=? WHERE id=?", (reason, run_id))

    def _new_execution(self, run_id: str, execution_id: str, actor: str, evidence: str, workflow: str | None = None) -> dict:
        parent = self._run(run_id)
        if parent["state"] != "completed" or not execution_id.strip() or not actor.strip() or not evidence.strip():
            raise RecoveryBlocked("new_execution_requires_completed_parent_and_evidence")
        spec = {**parent["spec"], "parent_run_id": run_id, "execution_id": execution_id}
        if workflow is not None:
            spec["workflow"] = workflow
        business_key = f"{spec['project_key']}:{spec['story_id']}:execution:{execution_id}"
        existing = self._connection.execute("SELECT id FROM runs WHERE business_key=?", (business_key,)).fetchone()
        if existing:
            child = self._run(existing[0])
            if child["spec"] != spec:
                raise RecoveryBlocked("execution_identity_conflict")
            return child
        for other in self._list():
            if (other["state"] != "completed" and other["spec"].get("story_id") == spec["story_id"]
                    and other["spec"].get("project_key") == spec["project_key"]):
                raise RecoveryBlocked("previous_execution_unfinished")
        timeout = parent["spec"].get("timeout_seconds")
        if timeout is None:
            raise RecoveryBlocked("parent_policy_missing")
        child_id = str(uuid4())
        with self._connection:
            self._connection.execute(
                "INSERT INTO runs(id,business_key,state,generation,spec,deadline,token_limit) VALUES(?,?,'resume_requested',1,?,?,?)",
                (child_id, business_key, json.dumps(spec), time.time() + timeout, parent["token_limit"]),
            )
            self._connection.execute("INSERT INTO audit(run_id,event,actor,evidence,created) VALUES(?,'new_execution',?,?,?)",
                                     (child_id, actor, evidence, time.time()))
        return self._run(child_id)

    def _operations(self, run_id: str) -> list[dict]:
        self._run(run_id)
        return [dict(row) for row in self._connection.execute(
            "SELECT key,state,digest,value FROM entries WHERE run_id=? AND namespace='publish'", (run_id,),
        )]

    def _progress(self, run_id: str) -> dict:
        run = self._run(run_id)
        count = self._connection.execute(
            "SELECT COUNT(*) FROM entries WHERE run_id=? AND namespace='unit' AND state='completed'", (run_id,),
        ).fetchone()[0]
        url = run["snapshot"].get("published_url")
        row = self._connection.execute(
            "SELECT value FROM entries WHERE run_id=? AND namespace='publish' AND key='create-base' AND state='completed'", (run_id,),
        ).fetchone()
        if not url and row:
            url = json.loads(row[0]).get("data", {}).get("base", {}).get("url")
        return {"completed_units": count, "phase": run["snapshot"].get("phase"), "url": url}

    def _resolve_publish(self, run_id: str, key: str, disposition: str, receipt: Any, actor: str, evidence: str) -> None:
        run = self._run(run_id)
        row = self._connection.execute("SELECT * FROM entries WHERE run_id=? AND namespace='publish' AND key=?", (run_id, key)).fetchone()
        if run["state"] != "paused" or row is None or row["state"] not in {"in_flight", "unknown"}:
            raise RecoveryBlocked("operation_not_awaiting_resolution")
        if not actor.strip() or not evidence.strip():
            raise RecoveryBlocked("operator_evidence_required")
        if disposition == "adopt_existing":
            if key == "create-base":
                data = receipt.get("data", {}) if isinstance(receipt, dict) else {}
                if not (data.get("base", {}).get("base_token") and data.get("base", {}).get("url") and data.get("table", {}).get("id")):
                    raise RecoveryBlocked("base_receipt_incomplete")
            elif key.startswith("records:"):
                ids = receipt.get("data", {}).get("record_id_list", []) if isinstance(receipt, dict) else []
                intent = json.loads(row["value"])["intent"]
                if len(ids) != len(intent["records"]) or not all(ids) or len(set(ids)) != len(ids):
                    raise RecoveryBlocked("record_receipt_incomplete")
            elif key.startswith("comment:"):
                if not isinstance(receipt, dict) or receipt.get("ok") is not True or not receipt.get("data", {}).get("comment_id"):
                    raise RecoveryBlocked("comment_receipt_incomplete")
            elif key == "tenant-permission":
                if receipt is not True:
                    raise RecoveryBlocked("permission_receipt_incomplete")
            elif key.startswith("node:"):
                if not isinstance(receipt, dict) or receipt.get("confirmed" if key == "node:confirm" else "code") != (True if key == "node:confirm" else 0):
                    raise RecoveryBlocked("node_receipt_incomplete")
            else:
                raise RecoveryBlocked("unsupported_operation")
            state, value = "completed", json.dumps(receipt, ensure_ascii=False)
        elif disposition == "verified_absent":
            state, value = "verified_absent", row["value"]
        elif disposition == "leave_blocked":
            state, value = "unknown", row["value"]
        else:
            raise RecoveryBlocked("unsupported_resolution")
        with self._connection:
            self._connection.execute("UPDATE entries SET state=?,value=? WHERE run_id=? AND namespace='publish' AND key=?", (state, value, run_id, key))
            self._connection.execute("INSERT INTO audit(run_id,event,actor,evidence,created) VALUES(?,?,?,?,?)",
                                     (run_id, f"{disposition}:{key}", actor, evidence, time.time()))


store: DesignStore | None = None
