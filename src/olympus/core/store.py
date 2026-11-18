"""SQLite WAL-mode authoritative state and append-only event store."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from olympus.core.contracts import (
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    BudgetLedger,
    BudgetPolicy,
    DesignSpec,
    GoalSpec,
    RunEvent,
    RunMode,
    RunStatus,
    TaskNode,
    TaskPlan,
    TaskStatus,
    new_id,
    utc_now,
)


class StoreError(RuntimeError):
    """Base persistence error."""


class ConcurrencyConflictError(StoreError):
    """Raised on optimistic state_revision mismatch."""


class NotFoundError(StoreError):
    """Raised when a run or task does not exist."""


class RunStateSnapshot(dict[str, Any]):
    """Typed convenience alias returned by get_run."""


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    state_revision INTEGER NOT NULL,
    budget_policy_json TEXT NOT NULL,
    budget_ledger_json TEXT NOT NULL,
    spec_json TEXT,
    plan_json TEXT,
    design_json TEXT,
    workflow_name TEXT NOT NULL DEFAULT 'repo_fast_patch',
    provider_config_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL,
    event_seq INTEGER NOT NULL,
    event_id TEXT NOT NULL UNIQUE,
    stage TEXT NOT NULL,
    event_type TEXT NOT NULL,
    actor TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    state_revision INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, event_seq)
);

CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    target TEXT NOT NULL,
    target_id TEXT NOT NULL,
    target_hash TEXT NOT NULL,
    commit_sha TEXT,
    approver TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_approvals_lookup
ON approvals (run_id, target, target_id, target_hash);

CREATE TABLE IF NOT EXISTS commands (
    operation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class RunStore:
    """Durable SQLite persistence for Olympus4SWE runs, events, approvals, and leases."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA_SQL)
            self._migrate_runs_columns(conn)
        finally:
            conn.close()

    def _migrate_runs_columns(self, conn: sqlite3.Connection) -> None:
        """Idempotently add columns introduced after the initial schema shipped."""
        existing = {
            str(row["name"]) for row in conn.execute("PRAGMA table_info(runs)").fetchall()
        }
        additions: dict[str, str] = {
            "design_json": "design_json TEXT",
            "workflow_name": "workflow_name TEXT NOT NULL DEFAULT 'repo_fast_patch'",
            "provider_config_json": "provider_config_json TEXT",
        }
        for column, ddl in additions.items():
            if column in existing:
                continue
            conn.execute(f"ALTER TABLE runs ADD COLUMN {ddl}")

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            yield conn
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def _next_event_seq(self, conn: sqlite3.Connection, run_id: str) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(event_seq), 0) AS max_seq FROM events WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return int(row["max_seq"]) + 1

    def _insert_event(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        stage: str,
        event_type: str,
        actor: str,
        payload: dict[str, Any],
        state_revision: int,
    ) -> RunEvent:
        evt = RunEvent(
            event_seq=self._next_event_seq(conn, run_id),
            run_id=run_id,
            stage=stage,
            event_type=event_type,
            actor=actor,
            payload=payload,
            state_revision=state_revision,
        )
        conn.execute(
            """
            INSERT INTO events (
                run_id, event_seq, event_id, stage, event_type, actor,
                payload_json, state_revision, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evt.run_id,
                evt.event_seq,
                evt.event_id,
                evt.stage,
                evt.event_type,
                evt.actor,
                json.dumps(evt.payload, ensure_ascii=False, sort_keys=True),
                evt.state_revision,
                evt.created_at.isoformat(),
            ),
        )
        return evt

    def append_event_in_tx(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        stage: str,
        event_type: str,
        actor: str,
        payload: dict[str, Any],
        state_revision: int,
    ) -> RunEvent:
        """Append an event inside an active transaction."""
        return self._insert_event(
            conn,
            run_id=run_id,
            stage=stage,
            event_type=event_type,
            actor=actor,
            payload=payload,
            state_revision=state_revision,
        )

    def update_plan_in_tx(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        cur_rev: int,
        mutator: Callable[[TaskPlan], TaskPlan],
        stage: str,
        event_type: str,
        actor: str,
        payload: dict[str, Any],
    ) -> int:
        """Transform stored TaskPlan and append an audit event inside an active tx."""
        row = self._get_run_row(conn, run_id)
        if not row["plan_json"]:
            raise NotFoundError(f"Run {run_id} has no TaskPlan")
        plan = TaskPlan.model_validate_json(row["plan_json"])
        updated = mutator(plan)
        new_rev = self._bump_run_field(
            conn,
            run_id=run_id,
            column="plan_json",
            value=updated.model_dump_json(),
            cur_rev=cur_rev,
        )
        self._insert_event(
            conn,
            run_id=run_id,
            stage=stage,
            event_type=event_type,
            actor=actor,
            payload=payload,
            state_revision=new_rev,
        )
        return new_rev

    def append_event(
        self,
        *,
        run_id: str,
        stage: str,
        event_type: str,
        actor: str,
        payload: dict[str, Any] | None = None,
    ) -> RunEvent:
        """Atomically record a non-mutating audit/trace event for a run."""
        with self._tx() as conn:
            row = self._get_run_row(conn, run_id)
            cur_rev = int(row["state_revision"])
            return self._insert_event(
                conn,
                run_id=run_id,
                stage=stage,
                event_type=event_type,
                actor=actor,
                payload=payload or {},
                state_revision=cur_rev,
            )

    def _get_run_row(self, conn: sqlite3.Connection, run_id: str) -> sqlite3.Row:
        row = cast(
            sqlite3.Row | None,
            conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone(),
        )
        if row is None:
            raise NotFoundError(f"Unknown run_id: {run_id}")
        return row

    def create_run(
        self,
        *,
        title: str,
        mode: RunMode = RunMode.EXECUTION,
        budget_policy: BudgetPolicy | None = None,
        run_id: str | None = None,
        actor: str = "system",
        workflow_name: str = "product_blueprint",
        provider_config: dict[str, Any] | None = None,
    ) -> str:
        rid = run_id or new_id("run")
        policy = budget_policy or BudgetPolicy()
        ledger = BudgetLedger()
        now = utc_now().isoformat()
        provider_json = (
            json.dumps(provider_config, ensure_ascii=False, sort_keys=True)
            if provider_config is not None
            else None
        )
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, title, mode, status, state_revision,
                    budget_policy_json, budget_ledger_json,
                    spec_json, plan_json, design_json,
                    workflow_name, provider_config_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?, ?, ?)
                """,
                (
                    rid,
                    title,
                    mode.value,
                    RunStatus.CREATED.value,
                    1,
                    policy.model_dump_json(),
                    ledger.model_dump_json(),
                    workflow_name,
                    provider_json,
                    now,
                    now,
                ),
            )
            self._insert_event(
                conn,
                run_id=rid,
                stage="system",
                event_type="run.created",
                actor=actor,
                payload={
                    "title": title,
                    "mode": mode.value,
                    "workflow_name": workflow_name,
                },
                state_revision=1,
            )
        return rid

    def _bump_run_field(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        column: str,
        value: str,
        cur_rev: int,
    ) -> int:
        allowed = {"status", "spec_json", "plan_json", "design_json", "budget_ledger_json"}
        if column not in allowed:
            raise ValueError(f"Unsupported run column update: {column}")
        new_rev = cur_rev + 1
        conn.execute(
            f"UPDATE runs SET {column} = ?, state_revision = ?, updated_at = ? WHERE run_id = ?",
            (value, new_rev, utc_now().isoformat(), run_id),
        )
        return new_rev

    def get_run(self, run_id: str) -> dict[str, Any]:
        conn = self._connect()
        try:
            row = self._get_run_row(conn, run_id)
            spec = GoalSpec.model_validate_json(row["spec_json"]) if row["spec_json"] else None
            plan = TaskPlan.model_validate_json(row["plan_json"]) if row["plan_json"] else None
            design = (
                DesignSpec.model_validate_json(row["design_json"])
                if row["design_json"]
                else None
            )
            return {
                "run_id": row["run_id"],
                "title": row["title"],
                "mode": RunMode(row["mode"]),
                "status": RunStatus(row["status"]),
                "state_revision": int(row["state_revision"]),
                "budget_policy": BudgetPolicy.model_validate_json(row["budget_policy_json"]),
                "budget_ledger": BudgetLedger.model_validate_json(row["budget_ledger_json"]),
                "spec": spec,
                "design": design,
                "plan": plan,
                "workflow_name": row["workflow_name"],
                "provider_config": (
                    json.loads(row["provider_config_json"])
                    if row["provider_config_json"]
                    else None
                ),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        finally:
            conn.close()

    def set_run_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        expected_revision: int | None = None,
        actor: str = "chronos",
        stage: str = "system",
        detail: dict[str, Any] | None = None,
    ) -> int:
        with self._tx() as conn:
            row = self._get_run_row(conn, run_id)
            cur_rev = int(row["state_revision"])
            if expected_revision is not None and cur_rev != expected_revision:
                raise ConcurrencyConflictError(
                    f"Revision mismatch: expected {expected_revision}, got {cur_rev}"
                )
            new_rev = self._bump_run_field(
                conn,
                run_id=run_id,
                column="status",
                value=status.value,
                cur_rev=cur_rev,
            )
            self._insert_event(
                conn,
                run_id=run_id,
                stage=stage,
                event_type="run.status_changed",
                actor=actor,
                payload={"from": row["status"], "to": status.value, **(detail or {})},
                state_revision=new_rev,
            )
            return new_rev

    def save_spec(self, run_id: str, spec: GoalSpec, *, actor: str = "align") -> int:
        with self._tx() as conn:
            row = self._get_run_row(conn, run_id)
            new_rev = self._bump_run_field(
                conn,
                run_id=run_id,
                column="spec_json",
                value=spec.model_dump_json(),
                cur_rev=int(row["state_revision"]),
            )
            self._insert_event(
                conn,
                run_id=run_id,
                stage="align",
                event_type="spec.saved",
                actor=actor,
                payload={
                    "spec_id": spec.spec_id,
                    "version": spec.version,
                    "content_hash": spec.content_hash,
                    "status": spec.status.value,
                },
                state_revision=new_rev,
            )
            return new_rev

    def save_design(self, run_id: str, design: DesignSpec, *, actor: str = "design") -> int:
        with self._tx() as conn:
            row = self._get_run_row(conn, run_id)
            new_rev = self._bump_run_field(
                conn,
                run_id=run_id,
                column="design_json",
                value=design.model_dump_json(),
                cur_rev=int(row["state_revision"]),
            )
            self._insert_event(
                conn,
                run_id=run_id,
                stage="design",
                event_type="design.saved",
                actor=actor,
                payload={
                    "design_id": design.design_id,
                    "version": design.version,
                    "content_hash": design.content_hash,
                    "module_count": len(design.modules),
                    "status": design.status.value,
                },
                state_revision=new_rev,
            )
            return new_rev

    def save_plan(self, run_id: str, plan: TaskPlan, *, actor: str = "orchestrate") -> int:
        with self._tx() as conn:
            row = self._get_run_row(conn, run_id)
            new_rev = self._bump_run_field(
                conn,
                run_id=run_id,
                column="plan_json",
                value=plan.model_dump_json(),
                cur_rev=int(row["state_revision"]),
            )
            self._insert_event(
                conn,
                run_id=run_id,
                stage="orchestrate",
                event_type="plan.saved",
                actor=actor,
                payload={
                    "plan_id": plan.plan_id,
                    "version": plan.version,
                    "content_hash": plan.content_hash,
                    "task_count": len(plan.tasks),
                },
                state_revision=new_rev,
            )
            return new_rev

    def reserve_llm_call(self, run_id: str, *, actor: str, purpose: str) -> BudgetLedger:
        with self._tx() as conn:
            row = self._get_run_row(conn, run_id)
            policy = BudgetPolicy.model_validate_json(row["budget_policy_json"])
            ledger = BudgetLedger.model_validate_json(row["budget_ledger_json"])
            updated = ledger.reserve(policy)
            new_rev = self._bump_run_field(
                conn,
                run_id=run_id,
                column="budget_ledger_json",
                value=updated.model_dump_json(),
                cur_rev=int(row["state_revision"]),
            )
            self._insert_event(
                conn,
                run_id=run_id,
                stage="budget",
                event_type="budget.reserved",
                actor=actor,
                payload={"purpose": purpose, "reserved": updated.llm_calls_reserved},
                state_revision=new_rev,
            )
            return updated

    def settle_llm_call(
        self,
        run_id: str,
        *,
        actor: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        unknown: bool = False,
    ) -> BudgetLedger:
        with self._tx() as conn:
            row = self._get_run_row(conn, run_id)
            ledger = BudgetLedger.model_validate_json(row["budget_ledger_json"])
            updated = ledger.settle(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                unknown=unknown,
            )
            new_rev = self._bump_run_field(
                conn,
                run_id=run_id,
                column="budget_ledger_json",
                value=updated.model_dump_json(),
                cur_rev=int(row["state_revision"]),
            )
            self._insert_event(
                conn,
                run_id=run_id,
                stage="budget",
                event_type="budget.settled",
                actor=actor,
                payload={
                    "used": updated.llm_calls_used,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "unknown": unknown,
                },
                state_revision=new_rev,
            )
            return updated

    def acquire_task_lease(self, run_id: str, task_id: str, *, actor: str = "chronos") -> TaskNode:
        """Atomically transition a task to RUNNING with an incremented lease_generation."""
        with self._tx() as conn:
            row = self._get_run_row(conn, run_id)
            if not row["plan_json"]:
                raise NotFoundError(f"Run {run_id} has no TaskPlan")
            plan = TaskPlan.model_validate_json(row["plan_json"])
            updated_tasks: list[TaskNode] = []
            leased_task: TaskNode | None = None
            for task in plan.tasks:
                if task.task_id == task_id:
                    if task.status not in (TaskStatus.WAITING, TaskStatus.READY):
                        raise ConcurrencyConflictError(
                            f"Task {task_id} cannot acquire lease from status {task.status.value}"
                        )
                    leased_task = task.model_copy(
                        update={
                            "status": TaskStatus.RUNNING,
                            "attempt": task.attempt + 1,
                            "lease_generation": task.lease_generation + 1,
                        }
                    )
                    updated_tasks.append(leased_task)
                else:
                    updated_tasks.append(task)
            if leased_task is None:
                raise NotFoundError(f"Task {task_id} not found in run {run_id}")
            new_plan = plan.model_copy(update={"tasks": updated_tasks})
            new_rev = self._bump_run_field(
                conn,
                run_id=run_id,
                column="plan_json",
                value=new_plan.model_dump_json(),
                cur_rev=int(row["state_revision"]),
            )
            self._insert_event(
                conn,
                run_id=run_id,
                stage="orchestrate",
                event_type="task.leased",
                actor=actor,
                payload={
                    "task_id": task_id,
                    "attempt": leased_task.attempt,
                    "lease_generation": leased_task.lease_generation,
                },
                state_revision=new_rev,
            )
            return leased_task

    def update_task_with_lease(
        self,
        run_id: str,
        task_id: str,
        *,
        expected_lease_generation: int,
        mutator: Callable[[TaskNode], TaskNode],
        actor: str,
        event_type: str,
    ) -> TaskNode:
        """Update task state only if the caller still holds the active lease_generation."""
        with self._tx() as conn:
            row = self._get_run_row(conn, run_id)
            if not row["plan_json"]:
                raise NotFoundError(f"Run {run_id} has no TaskPlan")
            plan = TaskPlan.model_validate_json(row["plan_json"])
            updated_tasks: list[TaskNode] = []
            target_after: TaskNode | None = None
            for task in plan.tasks:
                if task.task_id == task_id:
                    if task.lease_generation != expected_lease_generation:
                        raise ConcurrencyConflictError(
                            f"Stale lease for {task_id}: expected "
                            f"{expected_lease_generation}, active {task.lease_generation}"
                        )
                    target_after = mutator(task)
                    updated_tasks.append(target_after)
                else:
                    updated_tasks.append(task)
            if target_after is None:
                raise NotFoundError(f"Task {task_id} not found in run {run_id}")
            new_plan = plan.model_copy(update={"tasks": updated_tasks})
            new_rev = self._bump_run_field(
                conn,
                run_id=run_id,
                column="plan_json",
                value=new_plan.model_dump_json(),
                cur_rev=int(row["state_revision"]),
            )
            self._insert_event(
                conn,
                run_id=run_id,
                stage="orchestrate",
                event_type=event_type,
                actor=actor,
                payload={
                    "task_id": task_id,
                    "status": target_after.status.value,
                    "lease_generation": target_after.lease_generation,
                },
                state_revision=new_rev,
            )
            return target_after

    def record_approval(self, record: ApprovalRecord) -> int:
        with self._tx() as conn:
            row = self._get_run_row(conn, record.run_id)
            new_rev = int(row["state_revision"]) + 1
            conn.execute(
                """
                INSERT INTO approvals (
                    approval_id, run_id, target, target_id, target_hash,
                    commit_sha, approver, decision, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.approval_id,
                    record.run_id,
                    record.target.value,
                    record.target_id,
                    record.target_hash,
                    record.commit_sha,
                    record.approver,
                    record.decision.value,
                    record.reason,
                    record.created_at.isoformat(),
                ),
            )
            conn.execute(
                "UPDATE runs SET state_revision = ?, updated_at = ? WHERE run_id = ?",
                (new_rev, utc_now().isoformat(), record.run_id),
            )
            self._insert_event(
                conn,
                run_id=record.run_id,
                stage="gate",
                event_type="approval.recorded",
                actor=record.approver,
                payload={
                    "approval_id": record.approval_id,
                    "target": record.target.value,
                    "target_id": record.target_id,
                    "target_hash": record.target_hash,
                    "commit_sha": record.commit_sha,
                    "decision": record.decision.value,
                },
                state_revision=new_rev,
            )
            return new_rev

    def is_approved(
        self,
        *,
        run_id: str,
        target: ApprovalTarget,
        target_id: str,
        target_hash: str,
        commit_sha: str | None = None,
    ) -> bool:
        """Return True if the latest matching approval for the hash/commit is approved."""
        conn = self._connect()
        try:
            if commit_sha is None:
                row = conn.execute(
                    """
                    SELECT decision FROM approvals
                    WHERE run_id = ? AND target = ? AND target_id = ? AND target_hash = ?
                    ORDER BY rowid DESC LIMIT 1
                    """,
                    (run_id, target.value, target_id, target_hash),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT decision FROM approvals
                    WHERE run_id = ? AND target = ? AND target_id = ?
                      AND target_hash = ? AND commit_sha = ?
                    ORDER BY rowid DESC LIMIT 1
                    """,
                    (run_id, target.value, target_id, target_hash, commit_sha),
                ).fetchone()
            return row is not None and row["decision"] in (
                ApprovalDecision.APPROVED.value,
                ApprovalDecision.ACCEPT_RISK.value,
            )
        finally:
            conn.close()

    def list_runs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT run_id, title, mode, status, state_revision, workflow_name,
                       created_at, updated_at
                FROM runs ORDER BY rowid DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def apply_idempotent_command(
        self,
        *,
        operation_id: str,
        run_id: str,
        actor: str,
        action: str,
        expected_state_revision: int,
        payload: dict[str, Any],
        handler: Callable[[sqlite3.Connection, int], dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute a console/CLI command once per operation_id and enforce state_revision."""
        with self._tx() as conn:
            existing = conn.execute(
                "SELECT result_json FROM commands WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if existing is not None:
                result: dict[str, Any] = json.loads(existing["result_json"])
                return {**result, "deduplicated": True}

            row = self._get_run_row(conn, run_id)
            cur_rev = int(row["state_revision"])
            if cur_rev != expected_state_revision:
                raise ConcurrencyConflictError(
                    f"Command {action} rejected: expected revision "
                    f"{expected_state_revision}, got {cur_rev}"
                )

            result = handler(conn, cur_rev)
            conn.execute(
                """
                INSERT INTO commands (
                    operation_id, run_id, actor, action, payload_json, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    operation_id,
                    run_id,
                    actor,
                    action,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    json.dumps(result, ensure_ascii=False, sort_keys=True),
                    utc_now().isoformat(),
                ),
            )
            return {**result, "deduplicated": False}

    def list_events(self, run_id: str, *, after_seq: int = 0) -> list[RunEvent]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT * FROM events
                WHERE run_id = ? AND event_seq > ?
                ORDER BY event_seq ASC
                """,
                (run_id, after_seq),
            ).fetchall()
            return [
                RunEvent(
                    event_seq=int(r["event_seq"]),
                    event_id=r["event_id"],
                    run_id=r["run_id"],
                    stage=r["stage"],
                    event_type=r["event_type"],
                    actor=r["actor"],
                    payload=json.loads(r["payload_json"]),
                    state_revision=int(r["state_revision"]),
                    created_at=r["created_at"],
                )
                for r in rows
            ]
        finally:
            conn.close()
