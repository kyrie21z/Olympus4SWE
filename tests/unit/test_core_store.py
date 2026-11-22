"""Unit tests for SQLite RunStore invariants: events, leases, approvals, idempotency."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from olympus.core.contracts import (
    AcceptanceCheck,
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    BudgetPolicy,
    CheckKind,
    GoalSpec,
    RequirementItem,
    RunMode,
    RunStatus,
    TaskNode,
    TaskPlan,
    TaskStatus,
)
from olympus.core.store import ConcurrencyConflictError, RunStore


def _make_plan(spec: GoalSpec) -> TaskPlan:
    return TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1",
                title="Implement",
                objective="Write module",
                requirement_refs=["R1"],
                write_scope=["src/mod.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="C1",
                        kind=CheckKind.COMMAND,
                        description="pytest",
                        command="pytest -q",
                    )
                ],
            )
        ],
    )


def test_store_events_optimistic_revision_and_replay(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "olympus.db")
    rid = store.create_run(title="Demo", mode=RunMode.SIMULATION)

    rev = store.set_run_status(rid, RunStatus.ALIGNING, expected_revision=1)
    assert rev == 2

    with pytest.raises(ConcurrencyConflictError):
        store.set_run_status(rid, RunStatus.RUNNING, expected_revision=1)

    events = store.list_events(rid)
    assert [e.event_seq for e in events] == [1, 2]
    assert [e.event_type for e in events] == ["run.created", "run.status_changed"]
    assert [e.event_seq for e in store.list_events(rid, after_seq=1)] == [2]


def test_store_lease_generation_rejects_stale_updates(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "olympus.db")
    rid = store.create_run(title="Lease test")
    spec = GoalSpec(
        goal="Goal",
        requirements=[RequirementItem(id="R1", description="Do X", acceptance_criteria=["OK"])],
    )
    store.save_spec(rid, spec)
    plan = _make_plan(spec)
    store.save_plan(rid, plan)

    leased1 = store.acquire_task_lease(rid, "T1")
    assert leased1.lease_generation == 1
    assert leased1.status == TaskStatus.RUNNING

    # Simulate reset to READY and re-lease by another attempt
    store.update_task_with_lease(
        rid,
        "T1",
        expected_lease_generation=1,
        mutator=lambda t: t.model_copy(update={"status": TaskStatus.READY}),
        actor="chronos",
        event_type="task.requeued",
    )
    leased2 = store.acquire_task_lease(rid, "T1")
    assert leased2.lease_generation == 2

    # Stale worker with lease_generation=1 must be rejected
    with pytest.raises(ConcurrencyConflictError):
        store.update_task_with_lease(
            rid,
            "T1",
            expected_lease_generation=1,
            mutator=lambda t: t.model_copy(update={"status": TaskStatus.ACCEPTED}),
            actor="stale-worker",
            event_type="task.accepted",
        )


def test_store_approval_invalidated_on_hash_or_commit_change(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "olympus.db")
    rid = store.create_run(title="Approval test")
    spec_v1 = GoalSpec(
        spec_id="spec-a",
        version=1,
        goal="Goal v1",
        requirements=[RequirementItem(id="R1", description="Req", acceptance_criteria=["AC"])],
    )
    store.save_spec(rid, spec_v1)

    store.record_approval(
        ApprovalRecord(
            run_id=rid,
            target=ApprovalTarget.GOAL_SPEC,
            target_id=spec_v1.spec_id,
            target_hash=spec_v1.content_hash,
            commit_sha="deadbeef",
            approver="zeus",
            decision=ApprovalDecision.APPROVED,
        )
    )
    assert store.is_approved(
        run_id=rid,
        target=ApprovalTarget.GOAL_SPEC,
        target_id=spec_v1.spec_id,
        target_hash=spec_v1.content_hash,
        commit_sha="deadbeef",
    )
    # New commit invalidates commit-bound approval
    assert not store.is_approved(
        run_id=rid,
        target=ApprovalTarget.GOAL_SPEC,
        target_id=spec_v1.spec_id,
        target_hash=spec_v1.content_hash,
        commit_sha="cafebabe",
    )
    # Content change invalidates spec approval
    spec_v2 = spec_v1.model_copy(update={"version": 2, "out_of_scope": ["extra"]})
    assert spec_v2.content_hash != spec_v1.content_hash
    assert not store.is_approved(
        run_id=rid,
        target=ApprovalTarget.GOAL_SPEC,
        target_id=spec_v2.spec_id,
        target_hash=spec_v2.content_hash,
    )


def test_store_idempotent_commands_and_monotonic_budget(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "olympus.db")
    rid = store.create_run(title="Cmd test", budget_policy=BudgetPolicy(max_llm_calls=1))

    calls = 0

    def handler(conn, cur_rev):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        new_rev = cur_rev + 1
        conn.execute(
            "UPDATE runs SET status = ?, state_revision = ? WHERE run_id = ?",
            (RunStatus.PAUSED.value, new_rev, rid),
        )
        store.append_event_in_tx(
            conn,
            run_id=rid,
            stage="control",
            event_type="run.paused",
            actor="zeus",
            payload={},
            state_revision=new_rev,
        )
        return {"status": "paused"}

    res1 = store.apply_idempotent_command(
        operation_id="op-1",
        run_id=rid,
        actor="zeus",
        action="pause",
        expected_state_revision=1,
        payload={},
        handler=handler,
    )
    res2 = store.apply_idempotent_command(
        operation_id="op-1",
        run_id=rid,
        actor="zeus",
        action="pause",
        expected_state_revision=1,
        payload={},
        handler=handler,
    )
    assert calls == 1
    assert res1["deduplicated"] is False
    assert res2["deduplicated"] is True
    assert res2["status"] == "paused"

    # Budget exhaustion across reload
    store.reserve_llm_call(rid, actor="align", purpose="probe")
    store.settle_llm_call(rid, actor="align", input_tokens=10, output_tokens=5)
    with pytest.raises(ValueError):
        store.reserve_llm_call(rid, actor="align", purpose="exceed")


def test_store_accounts_cache_tokens_and_reads_legacy_ledger_json(tmp_path: Path) -> None:
    db_path = tmp_path / "olympus.db"
    store = RunStore(db_path)
    rid = store.create_run(
        title="Cache ledger", budget_policy=BudgetPolicy(max_total_tokens=500)
    )
    legacy_ledger = {
        "llm_calls_used": 0,
        "llm_calls_reserved": 0,
        "input_tokens": 10,
        "output_tokens": 5,
        "unknown_outcome_calls": 0,
    }
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE runs SET budget_ledger_json = ? WHERE run_id = ?",
            (json.dumps(legacy_ledger), rid),
        )

    assert store.get_run(rid)["budget_ledger"].total_tokens == 15

    store.reserve_llm_call(rid, actor="align", purpose="cached call")
    ledger = store.settle_llm_call(
        rid,
        actor="align",
        input_tokens=20,
        output_tokens=10,
        cache_read_tokens=300,
        cache_write_tokens=100,
    )
    assert (ledger.cache_read_tokens, ledger.cache_write_tokens) == (300, 100)
    assert ledger.total_tokens == 445
    assert ledger.can_reserve(BudgetPolicy(max_total_tokens=445)) is False

    over_budget = store.record_external_llm_usage(
        rid,
        actor="pi",
        purpose="pi_step_1",
        input_tokens=30,
        output_tokens=10,
        cache_read_tokens=60,
        cache_write_tokens=10,
    )
    assert over_budget is True
    ledger = store.get_run(rid)["budget_ledger"]
    assert ledger.llm_calls_used == 2
    assert (ledger.cache_read_tokens, ledger.cache_write_tokens) == (360, 110)
    assert ledger.total_tokens == 555
