"""SQLite WAL-mode authoritative state and append-only event store."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

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


ApprovalArtifact = GoalSpec | DesignSpec | TaskPlan
_APPROVAL_ARTIFACTS: dict[
    ApprovalTarget, tuple[str, type[ApprovalArtifact], str]
] = {
    ApprovalTarget.GOAL_SPEC: ("spec_json", GoalSpec, "spec_id"),
    ApprovalTarget.DESIGN_SPEC: ("design_json", DesignSpec, "design_id"),
    ApprovalTarget.TASK_PLAN: ("plan_json", TaskPlan, "plan_id"),
}
_ARTIFACT_TARGETS = {fields[0]: target for target, fields in _APPROVAL_ARTIFACTS.items()}
_APPROVED_DECISIONS = (ApprovalDecision.APPROVED, ApprovalDecision.ACCEPT_RISK)


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
    execution_drivers_json TEXT,
    workspace_root TEXT,
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

CREATE TABLE IF NOT EXISTS approval_artifacts (
    approval_id TEXT PRIMARY KEY REFERENCES approvals (approval_id),
    artifact_json TEXT NOT NULL
);

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
            "execution_drivers_json": "execution_drivers_json TEXT",
            "workspace_root": "workspace_root TEXT",
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

    def promote_workflow_in_tx(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        cur_rev: int,
        workflow_name: str,
        status: RunStatus,
        budget_policy: BudgetPolicy,
        actor: str,
        payload: dict[str, Any],
    ) -> int:
        """Promote a run's bound workflow and budget policy inside a transaction."""
        new_rev = cur_rev + 1
        conn.execute(
            """
            UPDATE runs
            SET workflow_name = ?, status = ?, budget_policy_json = ?,
                state_revision = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (
                workflow_name,
                status.value,
                budget_policy.model_dump_json(),
                new_rev,
                utc_now().isoformat(),
                run_id,
            ),
        )
        self.append_event_in_tx(
            conn,
            run_id=run_id,
            stage="graph",
            event_type="run.workflow_promoted",
            actor=actor,
            payload=payload,
            state_revision=new_rev,
        )
        return new_rev

    def update_status_in_tx(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        cur_rev: int,
        status: RunStatus,
        stage: str,
        event_type: str,
        actor: str,
        payload: dict[str, Any],
    ) -> int:
        """Set runs.status and append the status event inside an active transaction."""
        new_rev = cur_rev + 1
        conn.execute(
            """
            UPDATE runs
            SET status = ?, state_revision = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (status.value, new_rev, utc_now().isoformat(), run_id),
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
        expected_state_revision: int | None = None,
    ) -> RunEvent:
        """Atomically record a non-mutating audit/trace event for a run."""
        with self._tx() as conn:
            row = self._get_run_row(conn, run_id)
            cur_rev = int(row["state_revision"])
            if expected_state_revision is not None and cur_rev != expected_state_revision:
                raise ConcurrencyConflictError(
                    f"Run {run_id} changed while recording {event_type}"
                )
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
        execution_drivers: dict[str, Any] | None = None,
        workspace_root: Path | str | None = None,
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
        ws_root = str(Path(workspace_root).resolve()) if workspace_root is not None else None
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, title, mode, status, state_revision,
                    budget_policy_json, budget_ledger_json,
                    spec_json, plan_json, design_json,
                    workflow_name, provider_config_json, execution_drivers_json, workspace_root,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?, ?, ?, ?, ?)
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
                    json.dumps(execution_drivers) if execution_drivers is not None else None,
                    ws_root,
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
                    "workspace_root": ws_root,
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
        if column in _ARTIFACT_TARGETS:
            self._preserve_legacy_approval(conn, run_id, _ARTIFACT_TARGETS[column])
        new_rev = cur_rev + 1
        conn.execute(
            f"UPDATE runs SET {column} = ?, state_revision = ?, updated_at = ? WHERE run_id = ?",
            (value, new_rev, utc_now().isoformat(), run_id),
        )
        return new_rev

    @staticmethod
    def _check_revision(current: int, expected: int | None) -> None:
        if expected is not None and current != expected:
            raise ConcurrencyConflictError(
                f"State revision mismatch: expected {expected}, got {current}"
            )

    def get_run(
        self, run_id: str, *, expected_revision: int | None = None,
        include_approval_baselines: bool = False,
    ) -> dict[str, Any]:
        conn = self._connect()
        try:
            if include_approval_baselines:
                conn.execute("BEGIN")
            row = self._get_run_row(conn, run_id)
            self._check_revision(int(row["state_revision"]), expected_revision)
            snapshot = self._snapshot_from_row(row)
            if include_approval_baselines:
                baselines: dict[ApprovalTarget, ApprovalArtifact | None] = {}
                approved = {}
                for target, (column, _, id_field) in _APPROVAL_ARTIFACTS.items():
                    artifact = snapshot[column.removesuffix("_json")]
                    approved[target] = artifact is not None and self._is_approved(
                        conn, run_id, target, getattr(artifact, id_field), artifact.content_hash
                    )
                    approval = self._latest_artifact_approval(conn, run_id, target)
                    if approval is not None:
                        baselines[target] = (
                            self._matching_artifact(target, approval, approval["artifact_json"])
                            or self._matching_artifact(target, approval, row[column])
                            or self._matching_archived_artifact(conn, run_id, target, approval)
                        )
                snapshot["approval_baselines"] = baselines
                snapshot["artifact_approved"] = approved
            return snapshot
        finally:
            conn.close()

    def get_run_in_tx(self, conn: sqlite3.Connection, run_id: str) -> dict[str, Any]:
        """Return the run snapshot as read inside an already open transaction."""
        return self._snapshot_from_row(self._get_run_row(conn, run_id))

    @staticmethod
    def _snapshot_from_row(row: sqlite3.Row) -> dict[str, Any]:
        spec = GoalSpec.model_validate_json(row["spec_json"]) if row["spec_json"] else None
        plan = TaskPlan.model_validate_json(row["plan_json"]) if row["plan_json"] else None
        design = (
            DesignSpec.model_validate_json(row["design_json"]) if row["design_json"] else None
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
                json.loads(row["provider_config_json"]) if row["provider_config_json"] else None
            ),
            "execution_drivers": (
                json.loads(row["execution_drivers_json"]) if row["execution_drivers_json"] else None
            ),
            "workspace_root": row["workspace_root"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

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

    def save_spec(
        self,
        run_id: str,
        spec: GoalSpec,
        *,
        actor: str = "align",
        expected_revision: int | None = None,
        approval: ApprovalRecord | None = None,
        extra_events: Sequence[tuple[str, str, dict[str, Any]]] = (),
    ) -> int:
        with self._tx() as conn:
            row = self._get_run_row(conn, run_id)
            cur_rev = int(row["state_revision"])
            self._check_revision(cur_rev, expected_revision)
            return self.save_spec_in_tx(
                conn,
                run_id=run_id,
                spec=spec,
                actor=actor,
                cur_rev=cur_rev,
                approval=approval,
                extra_events=extra_events,
            )

    def save_spec_in_tx(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        spec: GoalSpec,
        actor: str,
        cur_rev: int,
        approval: ApprovalRecord | None = None,
        extra_events: Sequence[tuple[str, str, dict[str, Any]]] = (),
    ) -> int:
        """Persist a GoalSpec, its audit events, and any approval in one transaction."""
        new_rev = self._bump_run_field(
            conn,
            run_id=run_id,
            column="spec_json",
            value=spec.model_dump_json(),
            cur_rev=cur_rev,
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
        for stage, event_type, payload in extra_events:
            self._insert_event(
                conn,
                run_id=run_id,
                stage=stage,
                event_type=event_type,
                actor=actor,
                payload=payload,
                state_revision=new_rev,
            )
        if approval is not None:
            return self._record_approval(conn, approval, new_rev)
        return new_rev

    def save_design(
        self,
        run_id: str,
        design: DesignSpec,
        *,
        actor: str = "design",
        expected_revision: int | None = None,
        approval: ApprovalRecord | None = None,
    ) -> int:
        with self._tx() as conn:
            row = self._get_run_row(conn, run_id)
            cur_rev = int(row["state_revision"])
            self._check_revision(cur_rev, expected_revision)
            return self.save_design_in_tx(
                conn,
                run_id=run_id,
                design=design,
                actor=actor,
                cur_rev=cur_rev,
                approval=approval,
            )

    def save_design_in_tx(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        design: DesignSpec,
        actor: str,
        cur_rev: int,
        approval: ApprovalRecord | None = None,
    ) -> int:
        """Persist a DesignSpec, its audit event, and any approval in one transaction."""
        new_rev = self._bump_run_field(
            conn,
            run_id=run_id,
            column="design_json",
            value=design.model_dump_json(),
            cur_rev=cur_rev,
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
        if approval is not None:
            return self._record_approval(conn, approval, new_rev)
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
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        unknown: bool = False,
        cancelled: bool = False,
    ) -> BudgetLedger:
        with self._tx() as conn:
            row = self._get_run_row(conn, run_id)
            ledger = BudgetLedger.model_validate_json(row["budget_ledger_json"])
            if cancelled:
                if ledger.llm_calls_reserved == 0:
                    raise ValueError("Cannot release budget without an active reservation")
                updated = ledger.model_copy(
                    update={"llm_calls_reserved": ledger.llm_calls_reserved - 1}
                )
            else:
                updated = ledger.settle(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
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
                event_type="budget.released" if cancelled else "budget.settled",
                actor=actor,
                payload={
                    "used": updated.llm_calls_used,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_tokens": cache_read_tokens,
                    "cache_write_tokens": cache_write_tokens,
                    "unknown": unknown,
                },
                state_revision=new_rev,
            )
            return updated

    def record_external_llm_usage(
        self,
        run_id: str,
        *,
        actor: str,
        purpose: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> bool:
        """Record external-harness LLM usage monotonically; return True if budget was exceeded."""
        with self._tx() as conn:
            row = self._get_run_row(conn, run_id)
            policy = BudgetPolicy.model_validate_json(row["budget_policy_json"])
            ledger = BudgetLedger.model_validate_json(row["budget_ledger_json"])
            over_budget = not ledger.can_reserve(policy)
            updated = ledger.model_copy(
                update={
                    "llm_calls_used": ledger.llm_calls_used + 1,
                    "input_tokens": ledger.input_tokens + max(0, input_tokens),
                    "output_tokens": ledger.output_tokens + max(0, output_tokens),
                    "cache_read_tokens": ledger.cache_read_tokens + max(0, cache_read_tokens),
                    "cache_write_tokens": ledger.cache_write_tokens + max(0, cache_write_tokens),
                }
            )
            if (
                policy.max_total_tokens is not None
                and updated.total_tokens > policy.max_total_tokens
            ):
                over_budget = True
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
                    "purpose": purpose,
                    "used": updated.llm_calls_used,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_tokens": cache_read_tokens,
                    "cache_write_tokens": cache_write_tokens,
                    "over_budget": over_budget,
                },
                state_revision=new_rev,
            )
            return over_budget

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
        reject_if_run_inactive: bool = False,
    ) -> TaskNode:
        """Update task state only if the caller still holds the active lease_generation."""
        with self._tx() as conn:
            row = self._get_run_row(conn, run_id)
            if reject_if_run_inactive and row["status"] in (
                RunStatus.PAUSED.value,
                RunStatus.CANCELLED.value,
            ):
                raise ConcurrencyConflictError(
                    f"Run {run_id} is {row['status']}; refusing {event_type} for {task_id}"
                )
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

    def record_approval(
        self, record: ApprovalRecord, *, expected_revision: int | None = None
    ) -> int:
        with self._tx() as conn:
            row = self._get_run_row(conn, record.run_id)
            cur_rev = int(row["state_revision"])
            self._check_revision(cur_rev, expected_revision)
            return self._record_approval(conn, record, cur_rev)

    def record_approval_in_tx(
        self, conn: sqlite3.Connection, record: ApprovalRecord, *, cur_rev: int
    ) -> int:
        """Record an approval and bump the run revision inside an open transaction."""
        return self._record_approval(conn, record, cur_rev)

    @staticmethod
    def _latest_artifact_approval(
        conn: sqlite3.Connection, run_id: str, target: ApprovalTarget
    ) -> sqlite3.Row | None:
        return cast(sqlite3.Row | None, conn.execute(
            """
            SELECT a.approval_id, a.target_id, a.target_hash, s.artifact_json
            FROM approvals a LEFT JOIN approval_artifacts s USING (approval_id)
            WHERE a.run_id = ? AND a.target = ? AND a.decision IN (?, ?)
            ORDER BY a.rowid DESC LIMIT 1
            """,
            (run_id, target, *_APPROVED_DECISIONS),
        ).fetchone())

    @staticmethod
    def _matching_artifact(
        target: ApprovalTarget, approval: sqlite3.Row | ApprovalRecord, raw: str | None
    ) -> ApprovalArtifact | None:
        if raw is None:
            return None
        _, model, id_field = _APPROVAL_ARTIFACTS[target]
        try:
            artifact = model.model_validate_json(raw)
        except ValidationError:
            return None
        target_id, target_hash = (
            (approval.target_id, approval.target_hash) if isinstance(approval, ApprovalRecord)
            else (approval["target_id"], approval["target_hash"])
        )
        if getattr(artifact, id_field) == target_id and artifact.content_hash == target_hash:
            return artifact
        return None

    def _matching_archived_artifact(
        self, conn: sqlite3.Connection, run_id: str, target: ApprovalTarget, approval: sqlite3.Row
    ) -> ApprovalArtifact | None:
        rows = conn.execute(
            """
            SELECT s.artifact_json FROM approvals a JOIN approval_artifacts s USING (approval_id)
            WHERE a.run_id = ? AND a.target = ? AND a.target_id = ? AND a.target_hash = ?
              AND a.approval_id != ?
            ORDER BY a.rowid DESC
            """,
            (
                run_id, target, approval["target_id"], approval["target_hash"],
                approval["approval_id"],
            ),
        )
        for row in rows:
            artifact = self._matching_artifact(target, approval, row["artifact_json"])
            if artifact is not None:
                return artifact
        return None

    def _preserve_legacy_approval(
        self, conn: sqlite3.Connection, run_id: str, target: ApprovalTarget
    ) -> None:
        approval = self._latest_artifact_approval(conn, run_id, target)
        if approval is None or approval["artifact_json"] is not None:
            return
        column, _, _ = _APPROVAL_ARTIFACTS[target]
        raw = self._get_run_row(conn, run_id)[column]
        if self._matching_artifact(target, approval, raw) is not None:
            conn.execute(
                "INSERT INTO approval_artifacts (approval_id, artifact_json) VALUES (?, ?)",
                (approval["approval_id"], raw),
            )

    def _record_approval(
        self, conn: sqlite3.Connection, record: ApprovalRecord, cur_rev: int
    ) -> int:
        new_rev = cur_rev + 1
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
        if record.target in _APPROVAL_ARTIFACTS and record.decision in _APPROVED_DECISIONS:
            column, _, _ = _APPROVAL_ARTIFACTS[record.target]
            raw = self._get_run_row(conn, record.run_id)[column]
            if self._matching_artifact(record.target, record, raw) is not None:
                conn.execute(
                    "INSERT INTO approval_artifacts (approval_id, artifact_json) VALUES (?, ?)",
                    (record.approval_id, raw),
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
            return self._is_approved(conn, run_id, target, target_id, target_hash, commit_sha)
        finally:
            conn.close()

    def latest_approval_rowid(
        self,
        *,
        run_id: str,
        target: ApprovalTarget,
        target_id: str,
        target_hash: str,
    ) -> int | None:
        """Return the latest approved rowid for the exact artifact hash, or None."""
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT rowid, decision FROM approvals
                WHERE run_id = ? AND target = ? AND target_id = ? AND target_hash = ?
                ORDER BY rowid DESC LIMIT 1
                """,
                (run_id, target, target_id, target_hash),
            ).fetchone()
            if row is None or row["decision"] not in _APPROVED_DECISIONS:
                return None
            return int(row["rowid"])
        finally:
            conn.close()

    @staticmethod
    def _is_approved(
        conn: sqlite3.Connection, run_id: str, target: ApprovalTarget,
        target_id: str, target_hash: str, commit_sha: str | None = None,
    ) -> bool:
        row = conn.execute(
            """
            SELECT decision FROM approvals
            WHERE run_id = ? AND target = ? AND target_id = ? AND target_hash = ?
              AND (? IS NULL OR commit_sha = ?)
            ORDER BY rowid DESC LIMIT 1
            """,
            (run_id, target, target_id, target_hash, commit_sha, commit_sha),
        ).fetchone()
        return bool(row is not None and row["decision"] in _APPROVED_DECISIONS)

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

    def lookup_command(
        self,
        *,
        operation_id: str,
        run_id: str,
        action: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return a previously recorded command result, enforcing operation identity.

        Reusing an ``operation_id`` for a different run, action, or payload is a
        conflict rather than a silent replay of unrelated work.
        """
        conn = self._connect()
        try:
            existing = conn.execute(
                """
                SELECT run_id, action, payload_json, result_json
                FROM commands WHERE operation_id = ?
                """,
                (operation_id,),
            ).fetchone()
        finally:
            conn.close()
        if existing is None:
            return None
        self._assert_command_identity(
            existing,
            operation_id=operation_id,
            run_id=run_id,
            action=action,
            payload=payload,
        )
        result: dict[str, Any] = json.loads(existing["result_json"])
        return {**result, "deduplicated": True}

    @staticmethod
    def _assert_command_identity(
        existing: sqlite3.Row,
        *,
        operation_id: str,
        run_id: str,
        action: str,
        payload: dict[str, Any],
    ) -> None:
        canonical_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        recorded = (
            str(existing["run_id"]),
            str(existing["action"]),
            str(existing["payload_json"]),
        )
        if recorded != (run_id, action, canonical_payload):
            raise ConcurrencyConflictError(
                f"operation_id '{operation_id}' was already used for "
                f"action '{existing['action']}' on run '{existing['run_id']}' "
                "with a different payload"
            )

    def apply_idempotent_command(
        self,
        *,
        operation_id: str,
        run_id: str,
        actor: str,
        action: str,
        expected_state_revision: int | None,
        payload: dict[str, Any],
        handler: Callable[[sqlite3.Connection, int], dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute a console/CLI command once per operation_id and enforce state_revision."""
        canonical_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._tx() as conn:
            existing = conn.execute(
                """
                SELECT run_id, action, payload_json, result_json
                FROM commands WHERE operation_id = ?
                """,
                (operation_id,),
            ).fetchone()
            if existing is not None:
                self._assert_command_identity(
                    existing,
                    operation_id=operation_id,
                    run_id=run_id,
                    action=action,
                    payload=payload,
                )
                result: dict[str, Any] = json.loads(existing["result_json"])
                return {**result, "deduplicated": True}

            row = self._get_run_row(conn, run_id)
            cur_rev = int(row["state_revision"])
            if expected_state_revision is not None and cur_rev != expected_state_revision:
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
                    canonical_payload,
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
