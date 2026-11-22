"""Regression tests for Issue #54: budget recovery, retry auto-advance, and resume coalescing."""

from __future__ import annotations

import json
import re
import subprocess
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from olympus.core.contracts import (
    AcceptanceCheck,
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    BudgetLedger,
    BudgetPolicy,
    CheckKind,
    GoalSpec,
    RunMode,
    RunStatus,
    TaskNode,
    TaskPlan,
    TaskStatus,
)
from olympus.core.control import RunControl
from olympus.core.store import ConcurrencyConflictError, RunStore
from olympus.providers.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    MockLLMProvider,
    ToolCall,
    ToolSpec,
    UsageStats,
)
from olympus.stages.orchestrate import OrchestrateStage
from olympus.web.projection import build_run_projection
from olympus.web.server import AriadneApp
from olympus.web.supervisor import RunSupervisor

Responder = Callable[[Sequence[ChatMessage], Sequence[ToolSpec]], LLMResponse]

_WEB_SRC = Path(__file__).resolve().parents[2] / "web" / "src"


def _init_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# Sample\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Dev"], cwd=root, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)


def _spec() -> GoalSpec:
    return GoalSpec(
        spec_id="spec-54",
        goal="Add a feature flag module",
        requirements=[
            {
                "id": "R1",
                "description": "Provide feature.py with FLAG = True",
                "acceptance_criteria": ["feature.FLAG is True"],
            }
        ],
    )


def _plan(spec: GoalSpec, *, task_status: TaskStatus, file_name: str = "feature.py") -> TaskPlan:
    return TaskPlan.model_validate(
        {
            "plan_id": "plan-54",
            "spec_id": spec.spec_id,
            "spec_hash": spec.content_hash,
            "version": 1,
            "tasks": [
                {
                    "task_id": "T1",
                    "title": f"Implement {file_name}",
                    "objective": f"Create {file_name} inside the sandbox",
                    "requirement_refs": ["R1"],
                    "depends_on": [],
                    "write_scope": [file_name],
                    "max_cycles": 3,
                    "status": task_status.value,
                    "acceptance_checks": [
                        {
                            "id": "CHK-1",
                            "kind": "command",
                            "description": "Verify the flag module imports",
                            "command": "python3 -c 'import feature; assert feature.FLAG'",
                        }
                    ],
                }
            ],
        }
    )


def _approve(
    store: RunStore, run_id: str, target: ApprovalTarget, target_id: str, hash_: str
) -> None:
    store.record_approval(
        ApprovalRecord(
            run_id=run_id,
            target=target,
            target_id=target_id,
            target_hash=hash_,
            approver="zeus",
            decision=ApprovalDecision.APPROVED,
        )
    )


def _seed_exec_run(
    tmp_path: Path,
    *,
    budget_policy: BudgetPolicy | None = None,
    task_status: TaskStatus = TaskStatus.READY,
    mode: RunMode = RunMode.SIMULATION,
    supervised: bool = True,
    extra_tasks: list[TaskNode] | None = None,
) -> tuple[RunStore, str, GoalSpec, TaskPlan]:
    store = RunStore(tmp_path / ".olympus" / "olympus.db")
    run_id = store.create_run(
        title="Issue 54",
        workflow_name="full_sdlc_tdd",
        mode=mode,
        budget_policy=budget_policy,
        provider_config={"provider": "mock"},
    )
    spec = _spec()
    store.save_spec(run_id, spec, actor="align")
    _approve(store, run_id, ApprovalTarget.GOAL_SPEC, spec.spec_id, spec.content_hash)
    plan = _plan(spec, task_status=task_status)
    if extra_tasks:
        plan = plan.model_copy(update={"tasks": [*plan.tasks, *extra_tasks]})
    store.save_plan(run_id, plan, actor="orchestrate")
    _approve(store, run_id, ApprovalTarget.TASK_PLAN, plan.plan_id, plan.content_hash)
    store.set_run_status(run_id, RunStatus.AWAITING_HUMAN, actor="zeus", stage="control")
    if supervised:
        store.append_event(
            run_id=run_id,
            stage="graph",
            event_type="graph.node_finished",
            actor="supervisor",
            payload={"node": "gate_plan"},
        )
    return store, run_id, spec, plan


class _OneCallExhaustProvider(LLMProvider):
    """Issues a read tool call once to consume a single-call budget."""

    model_name = "exhaust-mock"

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
        temperature: float = 0.0,
    ) -> LLMResponse:
        del messages, tools, temperature
        self.calls += 1
        return LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc-1", name="list_directory", arguments={"path": "."})],
            usage=UsageStats(input_tokens=10, output_tokens=5),
        )


def _budget_responder(*, file_name: str = "feature.py") -> Responder:
    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        system = messages[0].content
        user = messages[1].content if len(messages) > 1 else ""
        if "Execution Engineer" in system:
            match = re.search(r"Sandbox workspace root: (.+)", user)
            assert match is not None
            sandbox = Path(match.group(1).strip())
            if not any(msg.role == "tool" for msg in messages):
                return LLMResponse(
                    content=f"writing {file_name}",
                    tool_calls=(
                        ToolCall(
                            id="w1",
                            name="write_file",
                            arguments={
                                "path": str(sandbox / file_name),
                                "content": "FLAG = True\n",
                            },
                        ),
                    ),
                )
            return LLMResponse(content=f"implemented {file_name}")
        if "Independent Verifier" in system:
            return LLMResponse(
                content=json.dumps(
                    {
                        "task_id": "T1",
                        "attempt": 1,
                        "passed": True,
                        "summary": "All checks verified.",
                        "findings": [],
                    }
                )
            )
        raise AssertionError(f"Unexpected system prompt: {system[:80]}")

    return responder


# --------------------------------------------------------------------------- 1. unlimited default


def test_budget_policy_defaults_are_unlimited() -> None:
    policy = BudgetPolicy()
    assert policy.max_llm_calls is None
    assert policy.max_total_tokens is None
    assert policy.max_concurrent_tasks == 3


def test_can_reserve_unlimited_dimensions_ignore_usage() -> None:
    policy = BudgetPolicy()
    ledger = BudgetLedger(llm_calls_used=10_000, llm_calls_reserved=5, input_tokens=9_000_000)
    assert ledger.can_reserve(policy) is True
    # An explicit cap on one dimension still binds even when the other is unlimited.
    capped_calls = BudgetPolicy(max_llm_calls=10_005)
    assert ledger.can_reserve(capped_calls) is False
    capped_tokens = BudgetPolicy(max_total_tokens=9_000_000)
    assert ledger.can_reserve(capped_tokens) is False


def test_can_reserve_enforces_explicit_caps_strictly() -> None:
    ledger = BudgetLedger(llm_calls_used=4, llm_calls_reserved=1)
    assert ledger.can_reserve(BudgetPolicy(max_llm_calls=6)) is True
    assert ledger.can_reserve(BudgetPolicy(max_llm_calls=5)) is False
    tokens = BudgetLedger(input_tokens=300, output_tokens=200)
    assert tokens.can_reserve(BudgetPolicy(max_total_tokens=501)) is True
    assert tokens.can_reserve(BudgetPolicy(max_total_tokens=500)) is False


def test_promote_keeps_unlimited_budget_when_no_explicit_cap(tmp_path: Path) -> None:
    store, run_id, _, _ = _seed_exec_run(tmp_path, task_status=TaskStatus.WAITING)
    control = RunControl(store)
    promoted = control.promote_to_execute(
        run_id,
        target_workflow="full_sdlc_tdd",
        current_stop_after="plan",
        operation_id="op-promote-unlimited",
        expected_state_revision=store.get_run(run_id)["state_revision"],
        auto_advance=False,
    )
    assert promoted["budget_max_calls"] is None
    policy = store.get_run(run_id)["budget_policy"]
    assert policy.max_llm_calls is None
    assert policy.max_total_tokens is None


def test_promote_uplifts_when_caller_supplies_cap_on_unlimited_policy(tmp_path: Path) -> None:
    store, run_id, _, plan = _seed_exec_run(tmp_path, task_status=TaskStatus.WAITING)
    control = RunControl(store)
    promoted = control.promote_to_execute(
        run_id,
        target_workflow="full_sdlc_tdd",
        current_stop_after="plan",
        operation_id="op-promote-caller-cap",
        expected_state_revision=store.get_run(run_id)["state_revision"],
        max_llm_calls=250,
        auto_advance=False,
    )
    min_calls = max(60, len(plan.tasks) * 25)
    assert promoted["budget_max_calls"] == max(0, min_calls, 250)
    policy = store.get_run(run_id)["budget_policy"]
    assert policy.max_llm_calls == max(0, min_calls, 250)
    # Token dimension stays unlimited because the policy never set an explicit token cap.
    assert policy.max_total_tokens is None


# --------------------------------------------------------------------------- 2. no retry spiral


def _run_budget_execution(tmp_path: Path, *, max_llm_calls: int) -> tuple[RunStore, str, Any]:
    _init_repo(tmp_path)
    store, run_id, _, _ = _seed_exec_run(
        tmp_path,
        budget_policy=BudgetPolicy(max_llm_calls=max_llm_calls),
        task_status=TaskStatus.READY,
        supervised=False,
    )
    orch = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=_budget_responder()),
        workspace_root=tmp_path,
    )
    snap = orch.run_execution(run_id=run_id)
    return store, run_id, snap


def test_worker_budget_exhaustion_pauses_without_retry_spiral(tmp_path: Path) -> None:
    # max_llm_calls=1: the worker's second provider call cannot reserve, so it stops
    # with budget_exhausted before completing. The task must pause, not burn a cycle.
    store, run_id, snap = _run_budget_execution(tmp_path, max_llm_calls=1)
    assert snap["status"] == RunStatus.AWAITING_HUMAN
    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.READY
    assert task.review_cycle == 0
    assert task.worktree_path is None
    assert task.last_error is not None
    assert task.last_error.startswith("Paused due to budget limit:")
    assert snap["budget_ledger"].llm_calls_used == 1

    events = store.list_events(run_id)
    assert any(e.event_type == "task.budget_paused" for e in events)
    assert not any(e.event_type == "task.blocked" for e in events)
    assert not any(e.event_type == "task.requeued" for e in events)
    status_events = [e for e in events if e.event_type == "run.status_changed"]
    assert status_events[-1].payload["to"] == RunStatus.AWAITING_HUMAN.value
    assert status_events[-1].payload["reason"] == "budget_exhausted"

    banner = build_run_projection(store, tmp_path, run_id)["hitl_banner"]
    assert banner["kind"] == "budget_exhausted"
    assert banner["actions"] == []


def test_review_budget_exhaustion_pauses_without_retry_spiral(tmp_path: Path) -> None:
    # max_llm_calls=2: the worker completes cleanly (2 calls), then the independent
    # review cannot reserve a third call and raises RuntimeError(budget_exhausted).
    store, run_id, snap = _run_budget_execution(tmp_path, max_llm_calls=2)
    assert snap["status"] == RunStatus.AWAITING_HUMAN
    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.READY
    assert task.review_cycle == 0
    assert task.worktree_path is None
    assert task.last_error is not None
    assert task.last_error.startswith("Paused due to budget limit:")
    assert "budget_exhausted" in task.last_error
    assert snap["budget_ledger"].llm_calls_used == 2

    events = store.list_events(run_id)
    assert any(e.event_type == "task.budget_paused" for e in events)
    assert not any(e.event_type == "task.blocked" for e in events)
    banner = build_run_projection(store, tmp_path, run_id)["hitl_banner"]
    assert banner["kind"] == "budget_exhausted"


# --------------------------------------------------------------------------- 3. retry + banners


def test_projection_ready_to_resume_banner(tmp_path: Path) -> None:
    store, run_id, _, _ = _seed_exec_run(tmp_path, task_status=TaskStatus.READY)
    banner = build_run_projection(store, tmp_path, run_id)["hitl_banner"]
    assert banner["kind"] == "ready_to_resume"
    assert banner["tone"] == "info"
    assert banner["task_id"] is None
    assert banner["actions"] == ["retry_workflow"]


def test_projection_budget_exhausted_banner(tmp_path: Path) -> None:
    store, run_id, _, _ = _seed_exec_run(
        tmp_path, budget_policy=BudgetPolicy(max_llm_calls=1), task_status=TaskStatus.READY
    )
    store.reserve_llm_call(run_id, actor="test", purpose="seed")
    store.settle_llm_call(run_id, actor="test")
    banner = build_run_projection(store, tmp_path, run_id)["hitl_banner"]
    assert banner["kind"] == "budget_exhausted"
    assert banner["tone"] == "warn"
    assert banner["task_id"] is None
    assert banner["actions"] == []


def test_retry_task_auto_advances_supervised_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id, _, _ = _seed_exec_run(tmp_path, task_status=TaskStatus.BLOCKED)
    supervisor = RunSupervisor(
        store=store,
        workspace_root=tmp_path,
        provider_builder=lambda **kwargs: MockLLMProvider(),
    )
    calls: list[bool] = []

    def spy_resume(rid: str, *, sync: bool = False) -> dict[str, Any]:
        calls.append(sync)
        return {"run_id": rid, "status": "queued"}

    monkeypatch.setattr(supervisor, "resume_run", spy_resume)
    app = AriadneApp(store=store, workspace_root=tmp_path, supervisor=supervisor)
    try:
        result = app.execute_command(
            run_id, {"action": "retry_task", "task_id": "T1", "hint": "try again"}
        )
        assert calls == [False]
        assert result["command_result"]["workflow"]["status"] == "queued"
        assert store.get_run(run_id)["plan"].tasks[0].status == TaskStatus.READY
    finally:
        supervisor.shutdown()


def test_retry_task_auto_advance_false_does_not_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id, _, _ = _seed_exec_run(tmp_path, task_status=TaskStatus.BLOCKED)
    supervisor = RunSupervisor(
        store=store,
        workspace_root=tmp_path,
        provider_builder=lambda **kwargs: MockLLMProvider(),
    )
    calls: list[bool] = []

    def spy_resume(rid: str, *, sync: bool = False) -> dict[str, Any]:
        calls.append(sync)
        return {"run_id": rid, "status": "queued"}

    monkeypatch.setattr(supervisor, "resume_run", spy_resume)
    app = AriadneApp(store=store, workspace_root=tmp_path, supervisor=supervisor)
    try:
        result = app.execute_command(
            run_id,
            {
                "action": "retry_task",
                "task_id": "T1",
                "hint": "again",
                "auto_advance": False,
                "operation_id": "op-retry-no-advance",
            },
        )
        assert calls == []
        assert "workflow" not in result["command_result"]
        assert store.get_run(run_id)["plan"].tasks[0].status == TaskStatus.READY
    finally:
        supervisor.shutdown()


def test_retry_task_default_payload_byte_compatible_and_meta_conflicts(tmp_path: Path) -> None:
    store, run_id, _, _ = _seed_exec_run(tmp_path, task_status=TaskStatus.BLOCKED)
    control = RunControl(store)
    rev = store.get_run(run_id)["state_revision"]
    first = control.retry_task(
        run_id,
        "T1",
        operation_id="op-retry-meta",
        expected_state_revision=rev,
        hint="h",
    )
    assert first["deduplicated"] is False
    # Default payload must remain exactly {"task_id", "hint"} (legacy byte-compatibility).
    replay = store.lookup_command(
        operation_id="op-retry-meta",
        run_id=run_id,
        action="retry_task",
        payload={"task_id": "T1", "hint": "h"},
    )
    assert replay is not None
    # Replaying the same operation with a non-default sync flag changes the payload => conflict.
    with pytest.raises(ConcurrencyConflictError, match="operation_id"):
        control.retry_task(
            run_id,
            "T1",
            operation_id="op-retry-meta",
            expected_state_revision=rev,
            hint="h",
            command_meta={"sync": True},
        )


# --------------------------------------------------------------------------- 4. resume coalescing


def test_resume_run_coalesces_concurrent_async_advances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_repo(tmp_path)
    store, run_id, _, _ = _seed_exec_run(tmp_path, task_status=TaskStatus.READY)
    supervisor = RunSupervisor(
        store=store,
        workspace_root=tmp_path,
        provider_builder=lambda **kwargs: MockLLMProvider(),
    )
    entered = threading.Event()
    release = threading.Event()
    starts: list[str | None] = []

    def fake_advance(
        rid: str,
        *,
        raw_input: str,
        start_from: str | None,
        auto_approve: bool,
        recompute_entry: bool = False,
    ) -> dict[str, Any]:
        del raw_input, auto_approve, recompute_entry
        starts.append(start_from)
        snap_at_entry = store.get_run(rid)
        rev = snap_at_entry["state_revision"]
        if len(starts) == 1:
            entered.set()
            assert release.wait(timeout=5), "first advance was never released"
        return {
            "run_id": rid,
            "workflow": snap_at_entry["workflow_name"],
            "status": snap_at_entry["status"].value,
            "state_revision": rev,
            "paused": None,
        }

    monkeypatch.setattr(supervisor, "_advance", fake_advance)
    try:
        first = supervisor.resume_run(run_id, sync=False)
        assert first["status"] == "queued"
        assert "coalesced" not in first
        assert entered.wait(timeout=5), "first advance never started"

        # A second async resume while the first is in flight must coalesce, not spawn a thread.
        second = supervisor.resume_run(run_id, sync=False)
        assert second["coalesced"] is True
        assert second["status"] == "queued"
        assert second["paused"] is None
        assert len(starts) == 1

        # Simulate genuine progress (a revision-bumping command) before a second
        # coalesced resume arrives, and return the *final* revision from _advance
        # just like real RunSupervisor._advance does.
        store.set_run_status(run_id, RunStatus.AWAITING_HUMAN, actor="test", stage="control")
        third = supervisor.resume_run(run_id, sync=False)
        assert third["coalesced"] is True
        release.set()
        supervisor.wait(run_id, timeout=5)
        assert len(starts) == 2
        assert starts[1] == "execute"
        with supervisor._guard:
            assert run_id not in supervisor._active_tokens
            assert run_id not in supervisor._pending_resume
    finally:
        release.set()
        supervisor.shutdown()


def test_resume_run_coalesced_midflight_mutation_runs_even_when_advance_returns_final_rev(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mid-flight command + coalesced resume must not be dropped when _advance returns final rev."""
    _init_repo(tmp_path)
    store, run_id, _, _ = _seed_exec_run(tmp_path, task_status=TaskStatus.READY)
    supervisor = RunSupervisor(
        store=store,
        workspace_root=tmp_path,
        provider_builder=lambda **kwargs: MockLLMProvider(),
    )
    entered = threading.Event()
    release = threading.Event()
    starts: list[str | None] = []

    def fake_advance(
        rid: str,
        *,
        raw_input: str,
        start_from: str | None,
        auto_approve: bool,
        recompute_entry: bool = False,
    ) -> dict[str, Any]:
        del raw_input, auto_approve, recompute_entry
        starts.append(start_from)
        if len(starts) == 1:
            entered.set()
            assert release.wait(timeout=5)
            # Real _advance also writes its own status at exit and returns final rev.
            store.set_run_status(rid, RunStatus.AWAITING_HUMAN, actor="supervisor", stage="graph")
        final = store.get_run(rid)
        return {
            "run_id": rid,
            "workflow": final["workflow_name"],
            "status": final["status"].value,
            "state_revision": final["state_revision"],
            "paused": None,
        }

    monkeypatch.setattr(supervisor, "_advance", fake_advance)
    try:
        supervisor.resume_run(run_id, sync=False)
        assert entered.wait(timeout=5)
        # External HITL command bumps state_revision while pass 1 is active.
        store.set_run_status(run_id, RunStatus.AWAITING_HUMAN, actor="zeus", stage="control")
        res1 = supervisor.resume_run(run_id, sync=False)
        res2 = supervisor.resume_run(run_id, sync=False)
        assert res1["coalesced"] is True
        assert res2["coalesced"] is True
        release.set()
        supervisor.wait(run_id, timeout=5)
        # Both mid-flight resumes coalesce into one follow-up pass (never dropped, never 2x).
        assert starts == ["execute", "execute"]
    finally:
        release.set()
        supervisor.shutdown()


def test_resume_run_coalesced_pass_skips_when_no_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_repo(tmp_path)
    store, run_id, _, _ = _seed_exec_run(tmp_path, task_status=TaskStatus.READY)
    supervisor = RunSupervisor(
        store=store,
        workspace_root=tmp_path,
        provider_builder=lambda **kwargs: MockLLMProvider(),
    )
    entered = threading.Event()
    release = threading.Event()
    starts: list[str | None] = []

    def fake_advance(
        rid: str,
        *,
        raw_input: str,
        start_from: str | None,
        auto_approve: bool,
        recompute_entry: bool = False,
    ) -> dict[str, Any]:
        del raw_input, auto_approve, recompute_entry
        starts.append(start_from)
        snap_at_entry = store.get_run(rid)
        rev = snap_at_entry["state_revision"]
        if len(starts) == 1:
            entered.set()
            assert release.wait(timeout=5)
        return {
            "run_id": rid,
            "workflow": snap_at_entry["workflow_name"],
            "status": snap_at_entry["status"].value,
            "state_revision": rev,
            "paused": None,
        }

    monkeypatch.setattr(supervisor, "_advance", fake_advance)
    try:
        supervisor.resume_run(run_id, sync=False)
        assert entered.wait(timeout=5)
        second = supervisor.resume_run(run_id, sync=False)
        assert second["coalesced"] is True
        # No state progress: the coalesced pass must NOT re-run (avoids re-pausing forever).
        release.set()
        supervisor.wait(run_id, timeout=5)
        assert len(starts) == 1
        with supervisor._guard:
            assert run_id not in supervisor._active_tokens
            assert run_id not in supervisor._pending_resume
    finally:
        release.set()
        supervisor.shutdown()


def test_sync_resume_not_coalesced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_repo(tmp_path)
    store, run_id, _, _ = _seed_exec_run(tmp_path, task_status=TaskStatus.READY)
    supervisor = RunSupervisor(
        store=store,
        workspace_root=tmp_path,
        provider_builder=lambda **kwargs: MockLLMProvider(),
    )
    starts: list[str | None] = []

    def fake_advance(
        rid: str,
        *,
        raw_input: str,
        start_from: str | None,
        auto_approve: bool,
        recompute_entry: bool = False,
    ) -> dict[str, Any]:
        del raw_input, auto_approve, recompute_entry
        starts.append(start_from)
        snap = store.get_run(rid)
        return {
            "run_id": rid,
            "workflow": snap["workflow_name"],
            "status": snap["status"].value,
            "state_revision": snap["state_revision"],
            "paused": None,
        }

    monkeypatch.setattr(supervisor, "_advance", fake_advance)
    try:
        result = supervisor.resume_run(run_id, sync=True)
        assert "coalesced" not in result
        assert starts == ["execute"]
        with supervisor._guard:
            assert run_id not in supervisor._active_tokens
    finally:
        supervisor.shutdown()


# --------------------------------------------------------------------------- 6. Codex hardening


def test_queued_resume_recomputes_entry_inside_lock_after_clarify(tmp_path: Path) -> None:
    """Clarify landing after async execute queued must pause at gate_spec, not fail."""
    _init_repo(tmp_path)
    store, run_id, _, _ = _seed_exec_run(tmp_path, task_status=TaskStatus.READY)
    supervisor = RunSupervisor(
        store=store,
        workspace_root=tmp_path,
        provider_builder=lambda **kwargs: MockLLMProvider(),
    )
    lock = supervisor._lock_for(run_id)
    try:
        # Hold run lock so the queued async execute thread waits before entering _advance.
        lock.acquire()
        queued = supervisor.resume_run(run_id, sync=False)
        assert queued["status"] == "queued"

        # Clarify produces unapproved GoalSpec v2 before the queued worker acquires the lock.
        control = RunControl(store)
        rev = store.get_run(run_id)["state_revision"]
        control.clarify_spec(run_id, answers={}, actor="zeus", expected_revision=rev)
        lock.release()

        result = supervisor.wait(run_id, timeout=10)
        assert result is not None
        assert result["status"] == RunStatus.AWAITING_HUMAN.value
        assert result["paused"] is not None
        assert result["paused"]["node"] == "gate_spec"
        assert store.get_run(run_id)["status"] == RunStatus.AWAITING_HUMAN
    finally:
        if lock.locked():
            lock.release()
        supervisor.shutdown()


def test_sync_resume_coalesces_with_in_flight_async_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sync=True arriving while async execute is in flight must not launch a second execute."""
    _init_repo(tmp_path)
    store, run_id, _, _ = _seed_exec_run(tmp_path, task_status=TaskStatus.READY)
    supervisor = RunSupervisor(
        store=store,
        workspace_root=tmp_path,
        provider_builder=lambda **kwargs: MockLLMProvider(),
    )
    entered = threading.Event()
    release = threading.Event()
    starts: list[str | None] = []

    def fake_advance(
        rid: str,
        *,
        raw_input: str,
        start_from: str | None,
        auto_approve: bool,
        recompute_entry: bool = False,
    ) -> dict[str, Any]:
        del raw_input, auto_approve, recompute_entry
        starts.append(start_from)
        entered.set()
        assert release.wait(timeout=5)
        snap = store.get_run(rid)
        return {
            "run_id": rid,
            "workflow": snap["workflow_name"],
            "status": RunStatus.COMPLETED.value,
            "state_revision": snap["state_revision"],
            "paused": None,
        }

    monkeypatch.setattr(supervisor, "_advance", fake_advance)
    sync_results: list[dict[str, Any]] = []
    try:
        supervisor.resume_run(run_id, sync=False)
        assert entered.wait(timeout=5)
        t = threading.Thread(
            target=lambda: sync_results.append(supervisor.resume_run(run_id, sync=True))
        )
        t.start()
        # Wait until the sync thread has coalesced onto the in-flight active token.
        deadline = threading.Event()
        for _ in range(200):
            with supervisor._guard:
                if run_id in supervisor._pending_resume:
                    break
            deadline.wait(0.01)
        release.set()
        t.join(timeout=5)
        supervisor.wait(run_id, timeout=5)
        assert len(starts) == 1
        assert sync_results and sync_results[0]["status"] == RunStatus.COMPLETED.value
    finally:
        release.set()
        supervisor.shutdown()


def test_stale_token_cannot_release_newer_active_token(tmp_path: Path) -> None:
    """_release_token must ignore stale tokens owned by an older worker pass."""
    store, run_id, _, _ = _seed_exec_run(tmp_path, task_status=TaskStatus.READY)
    supervisor = RunSupervisor(
        store=store,
        workspace_root=tmp_path,
        provider_builder=lambda **kwargs: MockLLMProvider(),
    )
    try:
        with supervisor._guard:
            supervisor._active_tokens[run_id] = 2
        supervisor._release_token(run_id, token=1, clear_pending=True)
        with supervisor._guard:
            assert supervisor._active_tokens.get(run_id) == 2
        supervisor._release_token(run_id, token=2)
        with supervisor._guard:
            assert run_id not in supervisor._active_tokens
    finally:
        supervisor.shutdown()


def test_upstream_provider_error_mentioning_budget_exhausted_not_misclassified(
    tmp_path: Path,
) -> None:
    """Upstream provider errors containing 'budget_exhausted' must NOT pause as local budget."""
    _init_repo(tmp_path)
    (tmp_path / "feature.py").write_text("FLAG = True\n", encoding="utf-8")
    subprocess.run(["git", "add", "feature.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "add feature"], cwd=tmp_path, check=True, capture_output=True
    )
    store, run_id, _, _ = _seed_exec_run(tmp_path, task_status=TaskStatus.READY)

    class UpstreamErrorProvider(LLMProvider):
        model_name = "upstream-err"

        def __init__(self) -> None:
            self.calls = 0

        def complete(
            self,
            messages: Sequence[ChatMessage],
            tools: Sequence[ToolSpec] = (),
            temperature: float = 0.0,
        ) -> LLMResponse:
            del messages, tools, temperature
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    content="Done editing.",
                    tool_calls=[],
                    usage=UsageStats(input_tokens=10, output_tokens=5),
                )
            raise RuntimeError("upstream gateway error: budget_exhausted on remote tenant")

    provider = UpstreamErrorProvider()
    orch = OrchestrateStage(
        store=store,
        provider=provider,
        workspace_root=tmp_path,
        artifacts_root=tmp_path / "artifacts",
    )
    with pytest.raises(RuntimeError, match="provider_error"):
        orch.run_execution(run_id=run_id)
    assert provider.calls == 2


def test_run_execution_does_not_overwrite_concurrent_human_pause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PAUSED set before entry or during budget check must not be overwritten."""
    _init_repo(tmp_path)
    store, run_id, _, _ = _seed_exec_run(
        tmp_path,
        budget_policy=BudgetPolicy(max_llm_calls=1),
        task_status=TaskStatus.READY,
    )
    store.set_run_status(run_id, RunStatus.PAUSED, actor="zeus", stage="control")
    provider = _OneCallExhaustProvider()
    orch = OrchestrateStage(
        store=store,
        provider=provider,
        workspace_root=tmp_path,
        artifacts_root=tmp_path / "artifacts",
    )
    snap = orch.run_execution(run_id=run_id)
    assert snap["status"] == RunStatus.PAUSED
    assert provider.calls == 0

    # Also test PAUSED landing right before budget_exhausted status write.
    store.set_run_status(run_id, RunStatus.AWAITING_HUMAN, actor="zeus", stage="control")
    store.reserve_llm_call(run_id, actor="test", purpose="fill")
    store.settle_llm_call(run_id, actor="test", input_tokens=1, output_tokens=1)
    orig_set_status = store.set_run_status
    injected = False

    def racing_set_status(rid: str, status: RunStatus, **kwargs: Any) -> int:
        nonlocal injected
        if status == RunStatus.AWAITING_HUMAN and not injected:
            injected = True
            orig_set_status(rid, RunStatus.PAUSED, actor="zeus", stage="control")
        return orig_set_status(rid, status, **kwargs)

    monkeypatch.setattr(store, "set_run_status", racing_set_status)
    snap2 = orch.run_execution(run_id=run_id)
    assert snap2["status"] == RunStatus.PAUSED


def test_pause_for_budget_preserves_existing_human_hint(tmp_path: Path) -> None:
    """Zero-call budget pause must not overwrite an unconsumed [Human hint] on the task."""
    _init_repo(tmp_path)
    store, run_id, _, _ = _seed_exec_run(
        tmp_path,
        budget_policy=BudgetPolicy(max_llm_calls=1),
        task_status=TaskStatus.READY,
        extra_tasks=[
            TaskNode(
                task_id="T2",
                title="Second task with hint",
                objective="Apply hint",
                requirement_refs=["R1"],
                write_scope=["b.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="c2", kind=CheckKind.COMMAND, description="ok", command="true"
                    )
                ],
                status=TaskStatus.READY,
                last_error="[Human hint]: keep atomic lock order",
                max_cycles=3,
            )
        ],
    )
    orch = OrchestrateStage(
        store=store,
        provider=_OneCallExhaustProvider(),
        workspace_root=tmp_path,
        artifacts_root=tmp_path / "artifacts",
    )
    snap = orch.run_execution(run_id=run_id)
    by_id = {t.task_id: t for t in snap["plan"].tasks}
    assert by_id["T2"].status == TaskStatus.READY
    assert by_id["T2"].last_error == "[Human hint]: keep atomic lock order"


def test_ready_task_blocked_by_cancelled_dependency_has_no_ready_banner(tmp_path: Path) -> None:
    """READY task whose depends_on is CANCELLED is not dispatchable => no ready_to_resume."""
    store, run_id, _, _ = _seed_exec_run(
        tmp_path,
        task_status=TaskStatus.CANCELLED,
        extra_tasks=[
            TaskNode(
                task_id="T2",
                title="Dependent task",
                objective="Depends on cancelled T1",
                requirement_refs=["R1"],
                depends_on=["T1"],
                write_scope=["b.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="c2", kind=CheckKind.COMMAND, description="ok", command="true"
                    )
                ],
                status=TaskStatus.READY,
                max_cycles=3,
            )
        ],
    )
    proj = build_run_projection(store, tmp_path, run_id)
    assert proj["hitl_banner"] is None


def test_retry_task_command_meta_cannot_overwrite_core_payload_fields(tmp_path: Path) -> None:
    """command_meta must never spoof task_id or hint in idempotency/audit payloads."""
    store, run_id, _, _ = _seed_exec_run(tmp_path, task_status=TaskStatus.BLOCKED)
    control = RunControl(store)
    rev = store.get_run(run_id)["state_revision"]
    control.retry_task(
        run_id,
        "T1",
        operation_id="op-meta-guard",
        expected_state_revision=rev,
        hint="real hint",
        command_meta={"task_id": "T2", "hint": "spoofed", "sync": True},
    )
    events = [e for e in store.list_events(run_id) if e.event_type == "task.retried_by_human"]
    assert events[-1].payload["task_id"] == "T1"
    assert events[-1].payload["hint"] == "real hint"
    assert events[-1].payload["sync"] is True


@pytest.mark.parametrize("human_status", [RunStatus.PAUSED, RunStatus.CANCELLED])
def test_advance_guarded_exception_preserves_paused_and_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, human_status: RunStatus
) -> None:
    """Async failure in _advance_guarded must never overwrite human PAUSED or CANCELLED."""
    _init_repo(tmp_path)
    store, run_id, _, _ = _seed_exec_run(tmp_path, task_status=TaskStatus.READY)
    supervisor = RunSupervisor(
        store=store,
        workspace_root=tmp_path,
        provider_builder=lambda **kwargs: MockLLMProvider(),
    )

    def failing_advance(
        rid: str,
        *,
        raw_input: str,
        start_from: str | None,
        auto_approve: bool,
        recompute_entry: bool = False,
    ) -> dict[str, Any]:
        del raw_input, start_from, auto_approve, recompute_entry
        store.set_run_status(rid, human_status, actor="zeus", stage="control")
        raise RuntimeError("upstream error: budget_exhausted on remote tenant")

    monkeypatch.setattr(supervisor, "_advance", failing_advance)
    try:
        supervisor.resume_run(run_id, sync=False)
        res = supervisor.wait(run_id, timeout=5)
        assert res is not None
        assert res["status"] == human_status.value
        assert store.get_run(run_id)["status"] == human_status
    finally:
        supervisor.shutdown()


def test_completed_run_reopened_by_clarify_routes_to_gate_spec(tmp_path: Path) -> None:
    """Clarifying a COMPLETED run produces unapproved GoalSpec v2 and must reopen to gate_spec."""
    _init_repo(tmp_path)
    store, run_id, _, _ = _seed_exec_run(tmp_path, task_status=TaskStatus.ACCEPTED)
    store.set_run_status(run_id, RunStatus.COMPLETED, actor="supervisor", stage="graph")
    supervisor = RunSupervisor(
        store=store,
        workspace_root=tmp_path,
        provider_builder=lambda **kwargs: MockLLMProvider(),
    )
    app = AriadneApp(store=store, workspace_root=tmp_path, supervisor=supervisor)
    try:
        cur = store.get_run(run_id)
        out = app.execute_command(
            run_id,
            {
                "action": "clarify",
                "answers": {},
                "expected_state_revision": cur["state_revision"],
                "auto_advance": True,
                "sync": True,
            },
        )
        wf = out["command_result"]["workflow"]
        assert wf["status"] == RunStatus.AWAITING_HUMAN.value
        assert wf["paused"] is not None
        assert wf["paused"]["node"] == "gate_spec"
        assert store.get_run(run_id)["status"] == RunStatus.AWAITING_HUMAN
    finally:
        supervisor.shutdown()


def test_web_spa_renders_unlimited_llm_calls() -> None:
    text = (_WEB_SRC / "components" / "TopologyView.tsx").read_text(encoding="utf-8")
    assert "budget.max_calls == null" in text
    assert "`${budget.calls_used} 次`" in text
    assert "`${budget.calls_used} / ${budget.max_calls}`" in text


def test_web_spa_gate_actions_show_retry_for_banner_actions() -> None:
    text = (_WEB_SRC / "components" / "HitlBanner.tsx").read_text(encoding="utf-8")
    assert 'gateBanner.actions.includes("retry_workflow")' in text
    assert "inspectFailure()" in text


def test_web_spa_has_no_hardcoded_waitlist_defaults() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted([*_WEB_SRC.rglob("*.ts"), *_WEB_SRC.rglob("*.tsx")])
    )
    for stale in (
        "SELECT WAITLISTED",
        "由值班后端工程师接手修复事务锁边界",
        "首期社团活动仅单管理员人工触发取消",
    ):
        assert stale not in combined
    for placeholder in (
        "输入要注入的人工提示",
        "输入改派备注",
        "输入接受风险的明确理由",
    ):
        assert placeholder in combined