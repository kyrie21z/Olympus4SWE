"""Seam tests for anomaly-driven Watchdog triggers and RunStore budget accounting."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from olympus.core.contracts import (
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    BudgetPolicy,
    CheckKind,
    GoalSpec,
    RunMode,
    RunStatus,
    TaskPlan,
    TaskStatus,
)
from olympus.core.store import RunStore
from olympus.harness.driver import (
    DriverRegistry,
    DriverTurnRequest,
    ExecutionDriversConfig,
    TurnOutcome,
)
from olympus.harness.watchdog import Watchdog, WatchdogConfig
from olympus.harness.worktree import SandboxHandle
from olympus.providers.base import (
    ChatMessage,
    LLMResponse,
    MockLLMProvider,
    ToolSpec,
    UsageStats,
)
from olympus.stages.align import AlignStage
from olympus.stages.orchestrate import OrchestrateStage


def _init_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# Sample\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Dev"], cwd=root, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)


def _confirmed_run(
    tmp_path: Path, *, budget_policy: BudgetPolicy | None = None
) -> tuple[RunStore, str]:
    store = RunStore(tmp_path / ".olympus" / "olympus.db")
    run_id = store.create_run(
        title="Watchdog run", mode=RunMode.SIMULATION, budget_policy=budget_policy
    )
    spec = GoalSpec(
        spec_id="spec-watchdog",
        goal="Add a feature flag module",
        requirements=[
            {
                "id": "R1",
                "description": "Provide feature.py with FLAG = True",
                "acceptance_criteria": ["feature.FLAG is True"],
            }
        ],
    )
    store.save_spec(run_id, spec)
    AlignStage(store=store, provider=MockLLMProvider(), workspace_root=tmp_path).confirm_spec(
        run_id=run_id, approver="zeus"
    )
    plan = TaskPlan.model_validate(
        {
            "plan_id": "plan-watchdog",
            "spec_id": spec.spec_id,
            "spec_hash": spec.content_hash,
            "tasks": [
                {
                    "task_id": "T1",
                    "title": "Implement the flag module",
                    "objective": "Implement the assigned flag module",
                    "requirement_refs": ["R1"],
                    "write_scope": ["feature.py"],
                    "acceptance_checks": [
                        {
                            "id": "CHK-1",
                            "kind": CheckKind.COMMAND.value,
                            "description": "Write a marker proving acceptance checks ran",
                            "command": (
                                "python3 -c 'from pathlib import Path; "
                                'Path("check-ran.txt").write_text("ran")\''
                            ),
                        }
                    ],
                }
            ],
        }
    )
    store.save_plan(run_id, plan)
    store.record_approval(
        ApprovalRecord(
            run_id=run_id,
            target=ApprovalTarget.TASK_PLAN,
            target_id=plan.plan_id,
            target_hash=plan.content_hash,
            approver="zeus",
            decision=ApprovalDecision.APPROVED,
        )
    )
    return store, run_id


class _ReviewDriver:
    """Deterministic cold-start reviewer that consumes no LLM budget."""

    name = "reviewer"
    supports_steer = False

    def steer(self, guidance: str, *, source: str) -> bool:
        del guidance, source
        return False

    def cancel(self, *, reason: str) -> None:
        del reason

    def run_turn(
        self,
        request: DriverTurnRequest,
        *,
        on_event: Any = None,
        on_handle: Any = None,
    ) -> TurnOutcome:
        del request, on_event
        if on_handle is not None:
            on_handle(self)
        return TurnOutcome(
            completed=True,
            stop_reason="final_response",
            final_content=json.dumps(
                {
                    "task_id": "T1",
                    "attempt": 1,
                    "passed": True,
                    "summary": "Verified.",
                    "findings": [],
                }
            ),
            steps_taken=1,
            driver_name="reviewer",
        )


def _stage(
    store: RunStore,
    tmp_path: Path,
    driver: Any,
    monitor: MockLLMProvider,
    *,
    config: WatchdogConfig | None = None,
    on_driver_event: Any = None,
) -> OrchestrateStage:
    return OrchestrateStage(
        store=store,
        provider=MockLLMProvider(),
        workspace_root=tmp_path,
        monitor_provider=monitor,
        watchdog_config=config,
        execution_drivers=ExecutionDriversConfig(
            worker_driver="seam", reviewer_driver="reviewer"
        ),
        driver_registry=DriverRegistry({"seam": driver, "reviewer": _ReviewDriver()}),
        on_driver_event=on_driver_event,
    )


def _write_feature(request: DriverTurnRequest) -> None:
    (request.workspace_path / "feature.py").write_text("FLAG = True\n", encoding="utf-8")


def _verdict(**fields: Any) -> LLMResponse:
    return LLMResponse(
        content=json.dumps({"verdict": "HEALTHY", "summary": "Progress is steady.", **fields}),
        usage=UsageStats(input_tokens=11, output_tokens=7),
    )


def test_default_config_healthy_turn_skips_l2_entirely(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)

    class HealthyDriver:
        name = "seam"
        supports_steer = False

        def steer(self, guidance: str, *, source: str) -> bool:
            del guidance, source
            return False

        def cancel(self, *, reason: str) -> None:
            del reason

        def run_turn(
            self,
            request: DriverTurnRequest,
            *,
            on_event: Any = None,
            on_handle: Any = None,
        ) -> TurnOutcome:
            assert on_event is not None
            assert on_handle is not None
            on_handle(self)
            for step, tool in enumerate(("write_file", "edit_file", "bash", "read_file"), start=1):
                on_event(
                    "agent.tool_finished",
                    {"tool": tool, "step": step, "status": "ok", "error_signature": ""},
                )
            for _ in range(2):
                on_event("agent.compaction", {"phase": "start"})
                on_event("agent.compaction", {"phase": "end"})
            _write_feature(request)
            return TurnOutcome(
                completed=True,
                stop_reason="final_response",
                final_content="done",
                steps_taken=4,
                driver_name="seam",
            )

    monitor = MockLLMProvider()
    out = _stage(store, tmp_path, HealthyDriver(), monitor).run_execution(run_id=run_id)

    assert out["status"] == RunStatus.COMPLETED
    assert out["plan"].tasks[0].status == TaskStatus.ACCEPTED
    assert monitor.recorded_requests == []
    assert not [e for e in store.list_events(run_id) if e.event_type.startswith("watchdog.")]


def test_compaction_threshold_pair_counts_as_single_thrash_trigger(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    checked = Event()

    class CompactingDriver:
        name = "seam"
        supports_steer = False

        def steer(self, guidance: str, *, source: str) -> bool:
            del guidance, source
            return False

        def cancel(self, *, reason: str) -> None:
            del reason

        def run_turn(
            self,
            request: DriverTurnRequest,
            *,
            on_event: Any = None,
            on_handle: Any = None,
        ) -> TurnOutcome:
            assert on_event is not None
            assert on_handle is not None
            on_handle(self)
            for _ in range(2):
                on_event("agent.compaction", {"phase": "start"})
                on_event("agent.compaction", {"phase": "end"})
            assert not checked.wait(0.3), "Two compactions must stay below the threshold"
            on_event("agent.compaction", {"phase": "start"})
            on_event("agent.compaction", {"phase": "end"})
            assert checked.wait(3), "Third completed compaction must trigger L2"
            _write_feature(request)
            return TurnOutcome(
                completed=True,
                stop_reason="final_response",
                final_content="done",
                steps_taken=3,
                driver_name="seam",
            )

    def observe(kind: str, payload: dict[str, Any]) -> None:
        del payload
        if kind == "watchdog.checked":
            checked.set()

    monitor = MockLLMProvider(responses=[_verdict()])
    out = _stage(
        store, tmp_path, CompactingDriver(), monitor, on_driver_event=observe
    ).run_execution(run_id=run_id)

    assert out["status"] == RunStatus.COMPLETED
    assert len(monitor.recorded_requests) == 1
    checked_events = [
        e for e in store.list_events(run_id) if e.event_type == "watchdog.checked"
    ]
    assert [e.payload["trigger"] for e in checked_events] == ["compaction_thrash"]
    ledger = store.get_run(run_id)["budget_ledger"]
    assert ledger.llm_calls_used == 1
    assert (ledger.input_tokens, ledger.output_tokens) == (11, 7)
    budget_actors = {
        e.actor for e in store.list_events(run_id) if e.event_type.startswith("budget.")
    }
    assert budget_actors == {"watchdog"}


def test_scope_violation_mid_turn_steers_cleanup_with_violating_paths(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    steered = Event()
    guidance: list[tuple[str, str]] = []
    snapshots: list[dict[str, Any]] = []

    class StrayFileDriver:
        name = "seam"
        supports_steer = True

        def steer(self, text: str, *, source: str) -> bool:
            guidance.append((text, source))
            steered.set()
            return True

        def cancel(self, *, reason: str) -> None:
            pytest.fail(f"Unexpected cancellation: {reason}")

        def run_turn(
            self,
            request: DriverTurnRequest,
            *,
            on_event: Any = None,
            on_handle: Any = None,
        ) -> TurnOutcome:
            assert on_event is not None
            assert on_handle is not None
            on_handle(self)
            _write_feature(request)
            (request.workspace_path / "stray.db").write_text("junk\n", encoding="utf-8")
            on_event(
                "agent.tool_finished",
                {"tool": "bash", "step": 1, "status": "ok", "error_signature": ""},
            )
            assert steered.wait(3), "Out-of-scope file must trigger a steering nudge"
            (request.workspace_path / "stray.db").unlink()
            on_event(
                "agent.tool_finished",
                {"tool": "edit_file", "step": 2, "status": "ok", "error_signature": ""},
            )
            return TurnOutcome(
                completed=True,
                stop_reason="final_response",
                final_content="done",
                steps_taken=2,
                driver_name="seam",
            )

    def monitor_response(
        messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        del tools
        snapshots.append(json.loads(messages[-1].content))
        return LLMResponse(
            content=json.dumps(
                {
                    "verdict": "SCOPE_DRIFT",
                    "summary": "A stray database file is outside the write scope.",
                    "suggested_steer": "Delete stray.db before finishing.",
                }
            ),
            usage=UsageStats(input_tokens=13, output_tokens=9),
        )

    monitor = MockLLMProvider(responder=monitor_response)
    out = _stage(store, tmp_path, StrayFileDriver(), monitor).run_execution(run_id=run_id)

    assert out["status"] == RunStatus.COMPLETED
    assert out["plan"].tasks[0].status == TaskStatus.ACCEPTED
    assert not [
        e for e in store.list_events(run_id) if e.event_type == "task.diff_gate_rejected"
    ]
    assert snapshots and snapshots[0]["violating_paths"] == ["stray.db"]
    assert guidance == [("Delete stray.db before finishing.", "watchdog")]
    checked_events = [
        e for e in store.list_events(run_id) if e.event_type == "watchdog.checked"
    ]
    assert [e.payload["trigger"] for e in checked_events] == ["scope_violation"]
    ledger = store.get_run(run_id)["budget_ledger"]
    assert ledger.llm_calls_used == 1
    assert (ledger.input_tokens, ledger.output_tokens) == (13, 9)


def test_budget_exhaustion_degrades_without_aborting_worker(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(
        tmp_path, budget_policy=BudgetPolicy(max_llm_calls=1)
    )
    checked = Event()
    errored = Event()
    errors: list[dict[str, Any]] = []

    class RepeatThenFinishDriver:
        name = "seam"
        supports_steer = False

        def steer(self, guidance: str, *, source: str) -> bool:
            del guidance, source
            return False

        def cancel(self, *, reason: str) -> None:
            pytest.fail(f"Budget exhaustion must not cancel the worker: {reason}")

        def run_turn(
            self,
            request: DriverTurnRequest,
            *,
            on_event: Any = None,
            on_handle: Any = None,
        ) -> TurnOutcome:
            assert on_event is not None
            assert on_handle is not None
            on_handle(self)
            failed = {
                "tool": "write_file", "step": 1, "status": "error",
                "error_signature": "repeat-me",
            }
            on_event("agent.tool_finished", failed)
            assert checked.wait(3)
            on_event("agent.tool_finished", failed)
            assert errored.wait(3), "Exhausted budget must surface a watchdog.error"
            _write_feature(request)
            return TurnOutcome(
                completed=True,
                stop_reason="final_response",
                final_content="done",
                steps_taken=2,
                driver_name="seam",
            )

    def observe(kind: str, payload: dict[str, Any]) -> None:
        if kind == "watchdog.checked":
            checked.set()
        if kind == "watchdog.error":
            errors.append(payload)
            errored.set()

    monitor = MockLLMProvider(responses=[_verdict()])
    out = _stage(
        store,
        tmp_path,
        RepeatThenFinishDriver(),
        monitor,
        config=WatchdogConfig(repeat_error_threshold=1),
        on_driver_event=observe,
    ).run_execution(run_id=run_id)

    assert out["status"] == RunStatus.COMPLETED
    assert out["plan"].tasks[0].status == TaskStatus.ACCEPTED
    assert len(monitor.recorded_requests) == 1
    assert errors and errors[0]["error"] == "BudgetExhausted"
    assert errors[0]["trigger"] == "repeat_error"
    ledger = store.get_run(run_id)["budget_ledger"]
    assert ledger.llm_calls_used == 1
    assert ledger.llm_calls_reserved == 0


class _Handle:
    supports_steer = False

    def __init__(self) -> None:
        self.cancellations: list[str] = []

    def steer(self, guidance: str, *, source: str) -> bool:
        del guidance, source
        return False

    def cancel(self, *, reason: str) -> None:
        self.cancellations.append(reason)


def test_malformed_l2_responses_back_off_exponentially(tmp_path: Path) -> None:
    clock = {"now": 0.0}
    errors: list[float] = []
    reached = Event()

    def wait(condition: Any, timeout: float) -> None:
        del condition
        clock["now"] += max(0.0, timeout)

    def emit(kind: str, payload: dict[str, Any]) -> None:
        del payload
        if kind == "watchdog.error" and len(errors) < 3:
            errors.append(clock["now"])
            if len(errors) == 3:
                reached.set()

    watchdog = Watchdog(
        config=WatchdogConfig(
            silence_seconds=1,
            error_backoff_seconds=10,
            max_error_backoff_seconds=40,
            clock=lambda: clock["now"],
            wait=wait,
        ),
        provider_factory=lambda: MockLLMProvider(
            responder=lambda messages, tools: LLMResponse(content="not json")
        ),
        sandbox=SandboxHandle(tmp_path, tmp_path, None, False),
        objective="Keep going",
        write_scope=["feature.py"],
        emit=emit,
    )
    handle = _Handle()
    watchdog.start(handle)
    try:
        assert reached.wait(5), "Malformed responses must keep retrying with backoff"
    finally:
        watchdog.stop()

    assert errors == [1.0, 11.0, 31.0]
    assert handle.cancellations == []


def test_default_config_disables_periodic_polling_and_validates_fields() -> None:
    config = WatchdogConfig()
    assert config.interval_seconds is None
    assert config.compaction_threshold == 3
    with pytest.raises(ValueError):
        WatchdogConfig(interval_seconds=0)
    with pytest.raises(ValueError):
        WatchdogConfig(compaction_threshold=0)
    with pytest.raises(ValueError):
        WatchdogConfig(error_backoff_seconds=10, max_error_backoff_seconds=5)


def test_scope_violation_snapshot_prioritizes_newly_added_paths(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    checked = [Event(), Event()]
    snapshots: list[list[str]] = []

    class _ManyViolationsDriver:
        name = "many-violations"
        supports_steer = False

        def steer(self, guidance: str, *, source: str) -> bool:
            del guidance, source
            return False

        def cancel(self, *, reason: str) -> None:
            raise AssertionError(reason)

        def run_turn(
            self,
            request: DriverTurnRequest,
            *,
            on_event: Any = None,
            on_handle: Any = None,
        ) -> TurnOutcome:
            on_handle(self)
            _write_feature(request)
            ws = request.workspace_path
            files = [ws / f"a{idx:02d}.db" for idx in range(24)]
            for path in files:
                path.touch()
            on_event(
                "agent.tool_finished",
                {"tool": "bash", "step": 1, "status": "ok", "error_signature": ""},
            )
            assert checked[0].wait(3)
            new_file = ws / "z-new.db"
            new_file.touch()
            on_event(
                "agent.tool_finished",
                {"tool": "bash", "step": 2, "status": "ok", "error_signature": ""},
            )
            assert checked[1].wait(3)
            for path in [*files, new_file]:
                path.unlink()
            return TurnOutcome(True, "final_response", "done", 2)

    def responder(messages: Any, tools: Any) -> LLMResponse:
        del tools
        snapshots.append(json.loads(messages[-1].content)["violating_paths"])
        return LLMResponse(content='{"verdict":"HEALTHY","summary":"Inspecting."}')

    stage = _stage(store, tmp_path, _ManyViolationsDriver(), MockLLMProvider(responder=responder))
    stage.on_driver_event = lambda kind, _payload: (
        checked[0 if not checked[0].is_set() else 1].set() if kind == "watchdog.checked" else None
    )
    end = stage.run_execution(run_id=run_id)
    assert end["status"] == RunStatus.COMPLETED
    assert len(snapshots) == 2
    assert "z-new.db" in snapshots[1]


def test_budget_exhausted_watchdog_emits_single_error_without_spam(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run.db")
    run_id = store.create_run(
        title="wd-budget",
        mode=RunMode.SIMULATION,
        budget_policy=BudgetPolicy(max_llm_calls=1),
    )
    store.reserve_llm_call(run_id, actor="worker", purpose="exhaust")
    store.settle_llm_call(run_id, actor="worker", input_tokens=1, output_tokens=1)

    clock = {"now": 0.0}
    errors: list[str] = []
    first_error = Event()
    ticks = 0
    enough_ticks = Event()

    def wait(condition: Any, timeout: float) -> None:
        nonlocal ticks
        del condition
        clock["now"] += max(0.0, timeout)
        ticks += 1
        if ticks >= 4:
            enough_ticks.set()

    def emit(kind: str, payload: dict[str, Any]) -> None:
        if kind == "watchdog.error":
            errors.append(str(payload.get("error")))
            first_error.set()

    watchdog = Watchdog(
        config=WatchdogConfig(
            silence_seconds=1,
            error_backoff_seconds=1,
            max_error_backoff_seconds=2,
            clock=lambda: clock["now"],
            wait=wait,
        ),
        provider_factory=lambda: MockLLMProvider(),
        sandbox=SandboxHandle(tmp_path, tmp_path, None, False),
        objective="Keep going",
        write_scope=["feature.py"],
        emit=emit,
        store=store,
        run_id=run_id,
    )
    watchdog.start(_Handle())
    try:
        assert first_error.wait(3)
        assert enough_ticks.wait(3)
    finally:
        watchdog.stop()
    assert errors == ["BudgetExhausted"]
