"""Provider-free deterministic HITL control plane built on RunStore.

Every human-in-the-loop mutation — pausing, clarifying, gating artifacts, and
task-level interventions — lives here so the web console, CLI, stage wrappers,
and workflow graph share one auditable implementation without touching a model
provider or importing higher layers.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
from typing import Any, overload

from olympus.core.contracts import (
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    BudgetLedger,
    BudgetPolicy,
    DesignSpec,
    GoalSpec,
    RunStatus,
    SpecStatus,
    TaskNode,
    TaskPlan,
    TaskStatus,
)
from olympus.core.dag import validate_goal_spec
from olympus.core.store import ConcurrencyConflictError, RunStore

_TARGET_HASH_MISMATCH = "Approval target hash no longer matches"
_ACTIVE_STATUSES = (
    RunStatus.RUNNING,
    RunStatus.ALIGNING,
    RunStatus.DESIGNING,
    RunStatus.PLANNING,
)
_DEFAULT_HUMAN_ASSIGNMENT_NOTE = "已改派给人工工程师处理（提交后仍须经独立验收）"


def is_design_bound_to_spec(
    store: RunStore, run_id: str, snap: Mapping[str, Any]
) -> bool:
    """Return True when the current DesignSpec is approved and not superseded by GoalSpec."""
    design: DesignSpec | None = snap.get("design")
    if design is None:
        return False
    design_rowid = store.latest_approval_rowid(
        run_id=run_id,
        target=ApprovalTarget.DESIGN_SPEC,
        target_id=design.design_id,
        target_hash=design.content_hash,
    )
    if design_rowid is None:
        return False
    spec: GoalSpec | None = snap.get("spec")
    if spec is None:
        return True
    spec_rowid = store.latest_approval_rowid(
        run_id=run_id,
        target=ApprovalTarget.GOAL_SPEC,
        target_id=spec.spec_id,
        target_hash=spec.content_hash,
    )
    if spec_rowid is None:
        # When GoalSpec exists and is not yet approved (e.g. after reopen or clarify),
        # any DesignSpec bound to a GoalSpec is stale until the current GoalSpec is
        # approved; only a spec-independent design (no spec_id/spec_hash) can proceed.
        return design.spec_id is None and design.spec_hash is None
    hash_mismatch = design.spec_hash is not None and design.spec_hash != spec.content_hash
    id_mismatch = design.spec_id is not None and design.spec_id != spec.spec_id
    if hash_mismatch or id_mismatch:
        # Stale when GoalSpec was approved after DesignSpec; still valid if the human
        # explicitly approved DesignSpec after reviewing the current GoalSpec.
        return design_rowid > spec_rowid
    return True


def is_plan_bound_to_upstream(
    store: RunStore, run_id: str, snap: Mapping[str, Any]
) -> bool:
    """Return True only when the current plan binds to an approved, non-stale upstream chain.

    A plan whose ``spec_hash`` / ``design_hash`` no longer matches the current
    approved GoalSpec / DesignSpec — or whose DesignSpec references an older
    GoalSpec than the currently approved one — is stale and must not unlock execution.
    """
    plan: TaskPlan | None = snap.get("plan")
    if plan is None:
        return False
    spec: GoalSpec | None = snap.get("spec")
    design: DesignSpec | None = snap.get("design")
    spec_approved = bool(
        spec is not None
        and store.is_approved(
            run_id=run_id,
            target=ApprovalTarget.GOAL_SPEC,
            target_id=spec.spec_id,
            target_hash=spec.content_hash,
        )
    )
    if spec is not None and plan.spec_hash is not None:
        if not spec_approved:
            return False
        if plan.spec_hash != spec.content_hash:
            return False
        if plan.spec_id is not None and plan.spec_id != spec.spec_id:
            return False

    spec_bound = False
    if spec is not None and (plan.spec_hash is not None or design is None):
        spec_bound = bool(
            spec_approved
            and plan.spec_hash == spec.content_hash
            and (plan.spec_id is None or plan.spec_id == spec.spec_id)
        )
    design_ok = design is not None and is_design_bound_to_spec(store, run_id, snap)
    design_bound = False
    if design is not None and (plan.design_hash is not None or design_ok):
        design_bound = bool(
            design_ok
            and plan.design_hash == design.content_hash
            and (plan.design_id is None or plan.design_id == design.design_id)
        )
    if design is not None and plan.design_hash is not None and not design_bound:
        return False
    return spec_bound or design_bound


class RunControl:
    """Deterministic mutations for human-in-the-loop gates over one RunStore."""

    def __init__(self, store: RunStore) -> None:
        self.store = store

    # ------------------------------------------------------------------ run status

    def set_paused(
        self,
        run_id: str,
        *,
        paused: bool,
        operation_id: str,
        expected_state_revision: int,
        actor: str = "zeus",
    ) -> dict[str, Any]:
        """Idempotently pause or resume a run at an exact state revision."""
        action = "pause" if paused else "resume"
        target_status = RunStatus.PAUSED if paused else RunStatus.RUNNING
        event_type = "run.paused" if paused else "run.resumed"

        def handler(conn: sqlite3.Connection, cur_rev: int) -> dict[str, Any]:
            new_rev = self.store.update_status_in_tx(
                conn,
                run_id=run_id,
                cur_rev=cur_rev,
                status=target_status,
                stage="orchestrate",
                event_type=event_type,
                actor=actor,
                payload={"status": target_status.value},
            )
            return {
                "run_id": run_id,
                "status": target_status.value,
                "state_revision": new_rev,
            }

        return self.store.apply_idempotent_command(
            operation_id=operation_id,
            run_id=run_id,
            actor=actor,
            action=action,
            expected_state_revision=expected_state_revision,
            payload={"action": action},
            handler=handler,
        )

    def prepare_workflow_retry(
        self,
        run_id: str,
        *,
        expected_revision: int,
        actor: str = "zeus",
        operation_id: str | None = None,
        command_meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Move a failed run back to RUNNING so the supervisor can resume its graph."""
        payload: dict[str, Any] = {"action": "retry_workflow", **(command_meta or {})}
        if operation_id is not None:

            def handler(conn: sqlite3.Connection, cur_rev: int) -> dict[str, Any]:
                snap = self.store.get_run_in_tx(conn, run_id)
                status: RunStatus = snap["status"]
                new_rev = cur_rev
                if status == RunStatus.FAILED:
                    new_rev = self.store.update_status_in_tx(
                        conn,
                        run_id=run_id,
                        cur_rev=cur_rev,
                        status=RunStatus.RUNNING,
                        stage="graph",
                        event_type="run.status_changed",
                        actor=actor,
                        payload={
                            "from": status.value,
                            "to": RunStatus.RUNNING.value,
                            "action": "retry_workflow",
                        },
                    )
                    status = RunStatus.RUNNING
                return {
                    "run_id": run_id,
                    "status": status.value,
                    "state_revision": new_rev,
                }

            return self.store.apply_idempotent_command(
                operation_id=operation_id,
                run_id=run_id,
                actor=actor,
                action="retry_workflow",
                expected_state_revision=expected_revision,
                payload=payload,
                handler=handler,
            )

        snap = self.store.get_run(run_id, expected_revision=expected_revision)
        status = snap["status"]
        if status == RunStatus.FAILED:
            new_rev = self.store.set_run_status(
                run_id,
                RunStatus.RUNNING,
                expected_revision=expected_revision,
                actor=actor,
                stage="graph",
                detail={"action": "retry_workflow"},
            )
            return {"run_id": run_id, "status": RunStatus.RUNNING.value, "state_revision": new_rev}
        return {"run_id": run_id, "status": status.value, "state_revision": snap["state_revision"]}

    def complete_workflow_retry(
        self,
        run_id: str,
        *,
        operation_id: str,
        workflow: dict[str, Any],
        pre_dispatch_revision: int,
        actor: str = "zeus",
        sync: bool = False,
    ) -> dict[str, Any]:
        """Record a successfully dispatched workflow retry without clobbering newer run state."""
        payload: dict[str, Any] = {"action": "retry_workflow"}
        if sync:
            payload["sync"] = True

        def handler(conn: sqlite3.Connection, cur_rev: int) -> dict[str, Any]:
            snap = self.store.get_run_in_tx(conn, run_id)
            status: RunStatus = snap["status"]
            new_rev = cur_rev
            # Only clear FAILED -> RUNNING if nothing has transitioned the run since dispatch
            # (e.g., an async worker that already wrote graph.failed must NOT be overwritten).
            if (
                status == RunStatus.FAILED
                and cur_rev == pre_dispatch_revision
                and workflow.get("status") != RunStatus.FAILED.value
            ):
                new_rev = self.store.update_status_in_tx(
                    conn,
                    run_id=run_id,
                    cur_rev=cur_rev,
                    status=RunStatus.RUNNING,
                    stage="graph",
                    event_type="run.status_changed",
                    actor=actor,
                    payload={
                        "from": status.value,
                        "to": RunStatus.RUNNING.value,
                        "action": "retry_workflow",
                    },
                )
                status = RunStatus.RUNNING
            return {
                "run_id": run_id,
                "status": status.value,
                "state_revision": new_rev,
                "workflow": workflow,
            }

        return self.store.apply_idempotent_command(
            operation_id=operation_id,
            run_id=run_id,
            actor=actor,
            action="retry_workflow",
            expected_state_revision=None,
            payload=payload,
            handler=handler,
        )

    # ---------------------------------------------------------------- spec gates

    @overload
    def clarify_spec(
        self,
        run_id: str,
        *,
        answers: dict[str, str],
        resolutions: dict[str, str] | None = None,
        extra_out_of_scope: list[str] | None = None,
        actor: str = "zeus",
        expected_revision: int | None = None,
        operation_id: None = None,
        command_meta: Mapping[str, Any] | None = None,
    ) -> GoalSpec: ...

    @overload
    def clarify_spec(
        self,
        run_id: str,
        *,
        answers: dict[str, str],
        resolutions: dict[str, str] | None = None,
        extra_out_of_scope: list[str] | None = None,
        actor: str = "zeus",
        expected_revision: int | None = None,
        operation_id: str,
        command_meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def clarify_spec(
        self,
        run_id: str,
        *,
        answers: dict[str, str],
        resolutions: dict[str, str] | None = None,
        extra_out_of_scope: list[str] | None = None,
        actor: str = "zeus",
        expected_revision: int | None = None,
        operation_id: str | None = None,
        command_meta: Mapping[str, Any] | None = None,
    ) -> GoalSpec | dict[str, Any]:
        """Apply human answers/resolutions, producing a new spec version (new hash).

        Passing ``operation_id`` records the command result inside one transaction so
        concurrent or retried HTTP requests replay cleanly without revision races.
        """
        res_map = resolutions or {}

        def _build_next(
            snap: Mapping[str, Any],
        ) -> tuple[GoalSpec, tuple[str, str, dict[str, Any]]]:
            spec: GoalSpec | None = snap["spec"]
            if spec is None:
                raise ValueError(f"Run {run_id} has no GoalSpec to clarify")
            updated_ambiguities = [
                amb.model_copy(update={"answer": answers[amb.id]}) if amb.id in answers else amb
                for amb in spec.open_ambiguities
            ]
            updated_notes = [
                note.model_copy(update={"resolved": True, "resolution_note": res_map[note.id]})
                if note.id in res_map
                else note
                for note in spec.minority_notes
            ]
            merged_oos = list(dict.fromkeys([*spec.out_of_scope, *(extra_out_of_scope or [])]))
            next_spec = spec.model_copy(
                update={
                    "version": spec.version + 1,
                    "open_ambiguities": updated_ambiguities,
                    "minority_notes": updated_notes,
                    "out_of_scope": merged_oos,
                    "status": SpecStatus.AWAITING_HUMAN,
                }
            )
            clarified_event = (
                "align",
                "spec.clarified",
                {
                    "version": next_spec.version,
                    "spec_hash": next_spec.content_hash,
                    "remaining_blockers": next_spec.unresolved_blockers(),
                },
            )
            return next_spec, clarified_event

        if operation_id is None:
            snap = self.store.get_run(run_id, expected_revision=expected_revision)
            cas_revision = (
                expected_revision if expected_revision is not None else int(snap["state_revision"])
            )
            next_spec, clarified_event = _build_next(snap)
            self.store.save_spec(
                run_id,
                next_spec,
                actor=actor,
                expected_revision=cas_revision,
                extra_events=[clarified_event],
            )
            return next_spec

        payload: dict[str, Any] = {
            "answers": dict(answers),
            "resolutions": dict(res_map),
            "extra_out_of_scope": list(extra_out_of_scope or []),
            **(command_meta or {}),
        }

        def handler(conn: sqlite3.Connection, cur_rev: int) -> dict[str, Any]:
            snap = self.store.get_run_in_tx(conn, run_id)
            next_spec, clarified_event = _build_next(snap)
            self.store.save_spec_in_tx(
                conn,
                run_id=run_id,
                spec=next_spec,
                actor=actor,
                cur_rev=cur_rev,
                extra_events=[clarified_event],
            )
            return {
                "run_id": run_id,
                "spec_id": next_spec.spec_id,
                "version": next_spec.version,
                "spec_hash": next_spec.content_hash,
                "remaining_blockers": next_spec.unresolved_blockers(),
            }

        return self.store.apply_idempotent_command(
            operation_id=operation_id,
            run_id=run_id,
            actor=actor,
            action="clarify",
            expected_state_revision=expected_revision,
            payload=payload,
            handler=handler,
        )

    @overload
    def confirm_spec(
        self,
        run_id: str,
        *,
        approver: str = "zeus",
        accept_risk: bool = False,
        reason: str = "",
        commit_sha: str | None = None,
        expected_revision: int | None = None,
        expected_target_hash: str | None = None,
        operation_id: None = None,
        command_meta: Mapping[str, Any] | None = None,
    ) -> GoalSpec: ...

    @overload
    def confirm_spec(
        self,
        run_id: str,
        *,
        approver: str = "zeus",
        accept_risk: bool = False,
        reason: str = "",
        commit_sha: str | None = None,
        expected_revision: int | None = None,
        expected_target_hash: str | None = None,
        operation_id: str,
        command_meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def confirm_spec(
        self,
        run_id: str,
        *,
        approver: str = "zeus",
        accept_risk: bool = False,
        reason: str = "",
        commit_sha: str | None = None,
        expected_revision: int | None = None,
        expected_target_hash: str | None = None,
        operation_id: str | None = None,
        command_meta: Mapping[str, Any] | None = None,
    ) -> GoalSpec | dict[str, Any]:
        """Freeze and approve the current GoalSpec once blocking ambiguities are resolved."""

        def _strict_validation_failed(
            spec: GoalSpec, conn: sqlite3.Connection | None
        ) -> bool:
            if conn is not None:
                rows = conn.execute(
                    """
                    SELECT payload_json FROM events
                    WHERE run_id = ? AND event_type = 'spec.validation_failed'
                    ORDER BY event_seq DESC
                    """,
                    (run_id,),
                ).fetchall()
                payloads = [json.loads(row["payload_json"]) for row in rows]
            else:
                payloads = [
                    event.payload
                    for event in reversed(self.store.list_events(run_id))
                    if event.event_type == "spec.validation_failed"
                ]
            for payload in payloads:
                recorded_hash = payload.get("spec_hash")
                if recorded_hash is None or recorded_hash == spec.content_hash:
                    return True
            return False

        def _prepare(
            snap: Mapping[str, Any], conn: sqlite3.Connection | None = None
        ) -> tuple[GoalSpec, ApprovalRecord]:
            spec: GoalSpec | None = snap["spec"]
            if expected_target_hash is not None and (
                spec is None or spec.content_hash != expected_target_hash
            ):
                raise ConcurrencyConflictError(_TARGET_HASH_MISMATCH)
            if spec is None:
                raise ValueError(f"Run {run_id} has no GoalSpec to confirm")
            blockers = spec.unresolved_blockers()
            if blockers and not accept_risk:
                raise ValueError(
                    f"Cannot confirm spec with unresolved blockers: {', '.join(blockers)}. "
                    "Resolve them via clarify or explicitly pass accept_risk=True with a reason."
                )
            if not accept_risk and _strict_validation_failed(spec, conn):
                strict_violations = validate_goal_spec(spec)
                if strict_violations:
                    raise ValueError(
                        "Cannot confirm GoalSpec with unresolved strict validation violations: "
                        f"{'; '.join(strict_violations)}. "
                        "Repair the spec or explicitly pass accept_risk=True with a reason."
                    )
            decision = ApprovalDecision.ACCEPT_RISK if accept_risk else ApprovalDecision.APPROVED
            confirmed = spec.model_copy(update={"status": SpecStatus.CONFIRMED})
            record = ApprovalRecord(
                run_id=run_id,
                target=ApprovalTarget.GOAL_SPEC,
                target_id=confirmed.spec_id,
                target_hash=confirmed.content_hash,
                commit_sha=commit_sha,
                approver=approver,
                decision=decision,
                reason=reason,
            )
            return confirmed, record

        if operation_id is None:
            snap = self.store.get_run(run_id, expected_revision=expected_revision)
            cas_revision = (
                expected_revision if expected_revision is not None else int(snap["state_revision"])
            )
            confirmed, record = _prepare(snap)
            self.store.save_spec(
                run_id,
                confirmed,
                actor=approver,
                expected_revision=cas_revision,
                approval=record,
            )
            return confirmed

        payload: dict[str, Any] = {
            "accept_risk": accept_risk,
            "reason": reason,
            "commit_sha": commit_sha,
            "expected_target_hash": expected_target_hash,
            **(command_meta or {}),
        }

        def handler(conn: sqlite3.Connection, cur_rev: int) -> dict[str, Any]:
            snap = self.store.get_run_in_tx(conn, run_id)
            confirmed, record = _prepare(snap, conn)
            self.store.save_spec_in_tx(
                conn,
                run_id=run_id,
                spec=confirmed,
                actor=approver,
                cur_rev=cur_rev,
                approval=record,
            )
            return {
                "run_id": run_id,
                "spec_id": confirmed.spec_id,
                "spec_hash": confirmed.content_hash,
                "status": confirmed.status.value,
                "approval_id": record.approval_id,
            }

        return self.store.apply_idempotent_command(
            operation_id=operation_id,
            run_id=run_id,
            actor=approver,
            action="approve_spec",
            expected_state_revision=expected_revision,
            payload=payload,
            handler=handler,
        )

    @overload
    def approve_design(
        self,
        run_id: str,
        *,
        approver: str = "zeus",
        reason: str = "",
        expected_revision: int | None = None,
        expected_target_hash: str | None = None,
        operation_id: None = None,
        command_meta: Mapping[str, Any] | None = None,
    ) -> ApprovalRecord: ...

    @overload
    def approve_design(
        self,
        run_id: str,
        *,
        approver: str = "zeus",
        reason: str = "",
        expected_revision: int | None = None,
        expected_target_hash: str | None = None,
        operation_id: str,
        command_meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def approve_design(
        self,
        run_id: str,
        *,
        approver: str = "zeus",
        reason: str = "",
        expected_revision: int | None = None,
        expected_target_hash: str | None = None,
        operation_id: str | None = None,
        command_meta: Mapping[str, Any] | None = None,
    ) -> ApprovalRecord | dict[str, Any]:
        """Freeze the current DesignSpec and record a hash-bound approval atomically."""

        def _prepare(snap: Mapping[str, Any]) -> tuple[DesignSpec, ApprovalRecord]:
            design: DesignSpec | None = snap["design"]
            if expected_target_hash is not None and (
                design is None or design.content_hash != expected_target_hash
            ):
                raise ConcurrencyConflictError(_TARGET_HASH_MISMATCH)
            if design is None:
                raise ValueError(f"Run {run_id} has no DesignSpec to approve.")
            confirmed = design.model_copy(update={"status": SpecStatus.CONFIRMED})
            record = ApprovalRecord(
                run_id=run_id,
                target=ApprovalTarget.DESIGN_SPEC,
                target_id=confirmed.design_id,
                target_hash=confirmed.content_hash,
                approver=approver,
                decision=ApprovalDecision.APPROVED,
                reason=reason,
            )
            return confirmed, record

        if operation_id is None:
            snap = self.store.get_run(run_id, expected_revision=expected_revision)
            cas_revision = (
                expected_revision if expected_revision is not None else int(snap["state_revision"])
            )
            confirmed, record = _prepare(snap)
            self.store.save_design(
                run_id,
                confirmed,
                actor=approver,
                expected_revision=cas_revision,
                approval=record,
            )
            return record

        payload: dict[str, Any] = {
            "reason": reason,
            "expected_target_hash": expected_target_hash,
            **(command_meta or {}),
        }

        def handler(conn: sqlite3.Connection, cur_rev: int) -> dict[str, Any]:
            snap = self.store.get_run_in_tx(conn, run_id)
            confirmed, record = _prepare(snap)
            self.store.save_design_in_tx(
                conn,
                run_id=run_id,
                design=confirmed,
                actor=approver,
                cur_rev=cur_rev,
                approval=record,
            )
            return record.model_dump(mode="json")

        return self.store.apply_idempotent_command(
            operation_id=operation_id,
            run_id=run_id,
            actor=approver,
            action="approve_design",
            expected_state_revision=expected_revision,
            payload=payload,
            handler=handler,
        )

    @overload
    def approve_plan(
        self,
        run_id: str,
        *,
        approver: str = "zeus",
        reason: str = "",
        expected_revision: int | None = None,
        expected_target_hash: str | None = None,
        operation_id: None = None,
        command_meta: Mapping[str, Any] | None = None,
    ) -> ApprovalRecord: ...

    @overload
    def approve_plan(
        self,
        run_id: str,
        *,
        approver: str = "zeus",
        reason: str = "",
        expected_revision: int | None = None,
        expected_target_hash: str | None = None,
        operation_id: str,
        command_meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def approve_plan(
        self,
        run_id: str,
        *,
        approver: str = "zeus",
        reason: str = "",
        expected_revision: int | None = None,
        expected_target_hash: str | None = None,
        operation_id: str | None = None,
        command_meta: Mapping[str, Any] | None = None,
    ) -> ApprovalRecord | dict[str, Any]:
        """Record human approval bound to the current TaskPlan content_hash."""

        def _prepare(snap: Mapping[str, Any]) -> ApprovalRecord:
            plan: TaskPlan | None = snap["plan"]
            if expected_target_hash is not None and (
                plan is None or plan.content_hash != expected_target_hash
            ):
                raise ConcurrencyConflictError(_TARGET_HASH_MISMATCH)
            if plan is None:
                raise ValueError(f"Run {run_id} has no TaskPlan to approve.")
            spec: GoalSpec | None = snap["spec"]
            design: DesignSpec | None = snap["design"]
            if not is_plan_bound_to_upstream(self.store, run_id, snap):
                if spec is None and design is None:
                    raise ValueError(
                        f"Run {run_id} has no GoalSpec or DesignSpec; complete Stage 1 first."
                    )
                raise ValueError(
                    "TaskPlan does not match any approved upstream artifact "
                    f"(spec_hash={plan.spec_hash}, design_hash={plan.design_hash}); "
                    "regenerate the plan first."
                )
            return ApprovalRecord(
                run_id=run_id,
                target=ApprovalTarget.TASK_PLAN,
                target_id=plan.plan_id,
                target_hash=plan.content_hash,
                approver=approver,
                decision=ApprovalDecision.APPROVED,
                reason=reason,
            )

        if operation_id is None:
            snap = self.store.get_run(run_id, expected_revision=expected_revision)
            cas_revision = (
                expected_revision if expected_revision is not None else int(snap["state_revision"])
            )
            record = _prepare(snap)
            self.store.record_approval(record, expected_revision=cas_revision)
            return record

        payload: dict[str, Any] = {
            "reason": reason,
            "expected_target_hash": expected_target_hash,
            **(command_meta or {}),
        }

        def handler(conn: sqlite3.Connection, cur_rev: int) -> dict[str, Any]:
            snap = self.store.get_run_in_tx(conn, run_id)
            record = _prepare(snap)
            self.store.record_approval_in_tx(conn, record, cur_rev=cur_rev)
            return record.model_dump(mode="json")

        return self.store.apply_idempotent_command(
            operation_id=operation_id,
            run_id=run_id,
            actor=approver,
            action="approve_plan",
            expected_state_revision=expected_revision,
            payload=payload,
            handler=handler,
        )

    # ---------------------------------------------------------------- task control

    def retry_task(
        self,
        run_id: str,
        task_id: str,
        *,
        operation_id: str,
        expected_state_revision: int,
        hint: str = "",
        actor: str = "zeus",
        command_meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Idempotently reset a BLOCKED/FAILED task back to READY for another cycle.

        ``attempt`` is preserved so the next lease acquisition stays monotonic and
        ``OrchestrateStage`` reopens the retained sandbox instead of a clean checkout.
        """

        def mutate_retry(task: TaskNode) -> TaskNode:
            if task.status not in (TaskStatus.BLOCKED, TaskStatus.FAILED):
                raise ValueError(
                    f"Task {task_id} status is {task.status.value}, not blocked/failed"
                )
            new_error = (
                f"{task.last_error or ''}\n[Human hint]: {hint}".strip()
                if hint
                else task.last_error
            )
            return task.model_copy(
                update={
                    "status": TaskStatus.READY,
                    "attempt": task.attempt,
                    "review_cycle": 0,
                    "last_error": new_error,
                    "in_place_recoveries": task.in_place_recoveries + 1,
                }
            )

        return self._apply_task_command(
            run_id=run_id,
            task_id=task_id,
            operation_id=operation_id,
            expected_state_revision=expected_state_revision,
            action="retry_task",
            event_type="task.retried_by_human",
            actor=actor,
            payload={"task_id": task_id, "hint": hint},
            task_mutator=mutate_retry,
            result_status=TaskStatus.READY.value,
            command_meta=command_meta,
        )

    def accept_task_risk(
        self,
        run_id: str,
        task_id: str,
        *,
        operation_id: str,
        expected_state_revision: int,
        reason: str,
        actor: str = "zeus",
    ) -> dict[str, Any]:
        """Explicitly mark a blocked task accepted-with-risk (never marked verified)."""
        clean_reason = reason.strip()
        if not clean_reason:
            raise ValueError("accept_task_risk requires a non-empty reason")

        return self._apply_task_command(
            run_id=run_id,
            task_id=task_id,
            operation_id=operation_id,
            expected_state_revision=expected_state_revision,
            action="accept_risk",
            event_type="task.accepted_with_risk",
            actor=actor,
            payload={"task_id": task_id, "reason": clean_reason},
            task_mutator=lambda task: task.model_copy(
                update={"status": TaskStatus.ACCEPTED, "accepted_risk_reason": clean_reason}
            ),
            result_status="accepted_with_risk",
        )

    def reassign_task_human(
        self,
        run_id: str,
        task_id: str,
        *,
        operation_id: str,
        expected_state_revision: int,
        note: str = "",
        actor: str = "zeus",
    ) -> dict[str, Any]:
        """Hand a task over to a human engineer, keeping independent verification mandatory."""
        clean_note = note.strip() or _DEFAULT_HUMAN_ASSIGNMENT_NOTE

        def mutate_reassign(task: TaskNode) -> TaskNode:
            if task.status not in (TaskStatus.READY, TaskStatus.BLOCKED, TaskStatus.FAILED):
                raise ValueError(
                    f"Task {task.task_id} in state {task.status.value} cannot be reassigned"
                )
            return task.model_copy(
                update={
                    "status": TaskStatus.READY,
                    "last_error": f"[HUMAN_ASSIGNED] {clean_note}",
                }
            )

        return self._apply_task_command(
            run_id=run_id,
            task_id=task_id,
            operation_id=operation_id,
            expected_state_revision=expected_state_revision,
            action="reassign_human",
            event_type="task.reassigned_human",
            actor=actor,
            payload={"task_id": task_id, "note": clean_note},
            task_mutator=mutate_reassign,
            result_status=TaskStatus.READY.value,
        )

    def _apply_task_command(
        self,
        *,
        run_id: str,
        task_id: str,
        operation_id: str,
        expected_state_revision: int,
        action: str,
        event_type: str,
        actor: str,
        payload: dict[str, Any],
        task_mutator: Callable[[TaskNode], TaskNode],
        result_status: str,
        command_meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Shared idempotent task-mutation command helper."""
        payload = {**(command_meta or {}), **payload}

        def handler(conn: sqlite3.Connection, cur_rev: int) -> dict[str, Any]:
            def mutate_plan(plan: TaskPlan) -> TaskPlan:
                updated_tasks: list[TaskNode] = []
                found = False
                for task in plan.tasks:
                    if task.task_id == task_id:
                        found = True
                        updated_tasks.append(task_mutator(task))
                    else:
                        updated_tasks.append(task)
                if not found:
                    raise ValueError(f"Task {task_id} not found")
                return plan.model_copy(update={"tasks": updated_tasks})

            new_rev = self.store.update_plan_in_tx(
                conn,
                run_id=run_id,
                cur_rev=cur_rev,
                mutator=mutate_plan,
                stage="control",
                event_type=event_type,
                actor=actor,
                payload=payload,
            )
            return {"task_id": task_id, "status": result_status, "state_revision": new_rev}

        return self.store.apply_idempotent_command(
            operation_id=operation_id,
            run_id=run_id,
            actor=actor,
            action=action,
            expected_state_revision=expected_state_revision,
            payload=payload,
            handler=handler,
        )

    # ------------------------------------------------------------- workflow promotion

    def request_revision(
        self,
        run_id: str,
        *,
        feedback: str,
        actor: str = "human",
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        """Record a human revision request and reopen the workflow for re-alignment."""
        clean = feedback.strip()
        if not clean:
            raise ValueError("request_revision requires non-empty feedback")
        snap = self.store.get_run(run_id, expected_revision=expected_revision)
        status: RunStatus = snap["status"]
        if status in _ACTIVE_STATUSES:
            raise ValueError(f"Run {run_id} is currently active ({status.value})")
        if status not in (
            RunStatus.AWAITING_HUMAN,
            RunStatus.PAUSED,
            RunStatus.FAILED,
            RunStatus.COMPLETED,
        ):
            raise ValueError(f"Run {run_id} cannot be revised from status {status.value}")
        cas_revision = int(snap["state_revision"])
        new_rev = self.store.set_run_status(
            run_id,
            RunStatus.RUNNING,
            expected_revision=cas_revision,
            actor=actor,
            stage="align",
            detail={"action": "revision_requested", "feedback": clean},
        )
        self.store.append_event(
            run_id=run_id,
            stage="align",
            event_type="run.revision_requested",
            actor=actor,
            payload={"feedback": clean, "from_status": status.value},
        )
        return {
            "run_id": run_id,
            "status": RunStatus.RUNNING.value,
            "state_revision": new_rev,
            "feedback": clean,
        }

    def reopen_run(
        self,
        run_id: str,
        *,
        feedback: str,
        actor: str = "human",
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        """Record post-completion follow-up work and re-enter the workflow incrementally."""
        clean = feedback.strip()
        if not clean:
            raise ValueError("reopen_run requires non-empty feedback")
        snap = self.store.get_run(run_id, expected_revision=expected_revision)
        if snap["status"] != RunStatus.COMPLETED:
            raise ValueError(
                f"Run {run_id} is {snap['status'].value}; incremental reopen requires completed"
            )
        cas_revision = int(snap["state_revision"])
        new_rev = self.store.set_run_status(
            run_id,
            RunStatus.RUNNING,
            expected_revision=cas_revision,
            actor=actor,
            stage="align",
            detail={"action": "reopened", "feedback": clean},
        )
        self.store.append_event(
            run_id=run_id,
            stage="align",
            event_type="run.reopened",
            actor=actor,
            payload={"feedback": clean},
        )
        return {
            "run_id": run_id,
            "status": RunStatus.RUNNING.value,
            "state_revision": new_rev,
            "feedback": clean,
        }

    def promote_to_execute(
        self,
        run_id: str,
        *,
        target_workflow: str,
        current_stop_after: str,
        operation_id: str,
        expected_state_revision: int,
        max_llm_calls: int | None = None,
        auto_advance: bool = True,
        sync: bool = False,
        actor: str = "zeus",
    ) -> dict[str, Any]:
        """Promote an approved blueprint run into its execution workflow with budget uplift."""
        # Keep default payload byte-compatible with legacy records (`{"workflow_name": ...}`)
        # while still detecting conflicting non-default options on operation_id replay.
        payload: dict[str, Any] = {"workflow_name": target_workflow}
        if max_llm_calls is not None:
            payload["max_llm_calls"] = max_llm_calls
        if not auto_advance:
            payload["auto_advance"] = False
        if sync:
            payload["sync"] = True

        def handler(conn: sqlite3.Connection, cur_rev: int) -> dict[str, Any]:
            # Every guard reads the transaction snapshot so a replayed operation_id
            # returns the recorded result before these state checks are re-evaluated.
            snap = self.store.get_run_in_tx(conn, run_id)
            if current_stop_after == "execute" and snap["status"] == RunStatus.COMPLETED:
                raise ValueError(f"Run {run_id} already completed execution")
            if snap["status"] in _ACTIVE_STATUSES:
                raise ValueError(
                    f"Run {run_id} is currently active ({snap['status'].value})"
                )

            plan: TaskPlan | None = snap["plan"]
            if plan is None:
                raise ValueError(f"Run {run_id} has no TaskPlan to execute")
            if not self.store.is_approved(
                run_id=run_id,
                target=ApprovalTarget.TASK_PLAN,
                target_id=plan.plan_id,
                target_hash=plan.content_hash,
            ):
                raise ValueError(
                    f"TaskPlan {plan.plan_id} must be approved before execution"
                )
            if not is_plan_bound_to_upstream(self.store, run_id, snap):
                raise ValueError(
                    f"TaskPlan {plan.plan_id} is stale: its spec_hash/design_hash no longer "
                    "match the current approved upstream artifacts; regenerate the plan first."
                )

            policy: BudgetPolicy = snap["budget_policy"]
            ledger: BudgetLedger = snap["budget_ledger"]
            if policy.max_llm_calls is None and max_llm_calls is None:
                new_max_calls: int | None = None
            else:
                min_calls = ledger.llm_calls_used + max(60, len(plan.tasks) * 25)
                new_max_calls = max(policy.max_llm_calls or 0, min_calls, max_llm_calls or 0)
            if policy.max_total_tokens is None:
                new_max_tokens: int | None = None
            else:
                min_tokens = ledger.total_tokens + max(500_000, len(plan.tasks) * 250_000)
                new_max_tokens = max(policy.max_total_tokens, min_tokens)
            updated_policy = policy.model_copy(
                update={
                    "max_llm_calls": new_max_calls,
                    "max_total_tokens": new_max_tokens,
                }
            )
            target_status = RunStatus.RUNNING if auto_advance else RunStatus.AWAITING_HUMAN
            from_workflow = snap["workflow_name"]

            new_rev = self.store.promote_workflow_in_tx(
                conn,
                run_id=run_id,
                cur_rev=cur_rev,
                workflow_name=target_workflow,
                status=target_status,
                budget_policy=updated_policy,
                actor=actor,
                payload={
                    "from_workflow": from_workflow,
                    "to_workflow": target_workflow,
                    "budget_max_calls": updated_policy.max_llm_calls,
                    "budget_max_tokens": updated_policy.max_total_tokens,
                },
            )
            return {
                "run_id": run_id,
                "from_workflow": from_workflow,
                "workflow": target_workflow,
                "status": target_status.value,
                "budget_max_calls": updated_policy.max_llm_calls,
                "state_revision": new_rev,
            }

        return self.store.apply_idempotent_command(
            operation_id=operation_id,
            run_id=run_id,
            actor=actor,
            action="promote_to_execute",
            expected_state_revision=expected_state_revision,
            payload=payload,
            handler=handler,
        )


__all__ = ["RunControl", "is_design_bound_to_spec", "is_plan_bound_to_upstream"]
