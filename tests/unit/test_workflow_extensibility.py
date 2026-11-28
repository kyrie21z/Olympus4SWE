"""Tests for Issue #188: pluggable workflow manifests, explicit resume cursors,
declarative god/phase projection, role-scoped drivers, and headless context parity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from olympus.core.contracts import (
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    GoalSpec,
    RequirementItem,
    RunMode,
    RunStatus,
    SpecStatus,
)
from olympus.core.store import RunStore
from olympus.graph import (
    END,
    START,
    WorkflowDefinition,
    build_workflow_view,
    describe_workflow,
    extract_paused_cursor,
    load_workflow,
    resolve_resume_node,
)
from olympus.graph.loader import WorkflowEdgeDef, WorkflowNodeDef
from olympus.graph.progress import classify_run_board
from olympus.harness.driver import ExecutionDriversConfig
from olympus.providers.base import MockLLMProvider
from olympus.session.runner import SessionConfig, SessionRunner
from olympus.web.schemas import WorkflowSummary, WorkflowView


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "olympus.sqlite")


def _custom_pstack_definition() -> WorkflowDefinition:
    """Custom non-SDLC workflow using only registered node types for engine tests."""
    return WorkflowDefinition(
        name="pstack_demo",
        description="Evidence-first custom workflow with declarative metadata",
        family="pstack",
        autonomy="autonomous_hard_stops",
        applicable_scenarios=["brownfield multi-file feature", "cross-boundary refactor"],
        anti_patterns=["vague greenfield product idea"],
        stop_after="verify",
        topology="evidence_first",
        nodes=[
            WorkflowNodeDef(
                name="explore",
                type="intake_triage",
                god="prometheus",
                phase="align",
                role="explorer",
                skill_refs=["pstack/fix-root-causes"],
            ),
            WorkflowNodeDef(
                name="checkpoint",
                type="human_gate",
                params={"gate": "spec"},
                god="athena",
                phase="design",
                role="architect",
            ),
            WorkflowNodeDef(
                name="interrogate",
                type="human_gate",
                params={"gate": "spec"},
                god="momus",
                phase="review",
                role="interrogator",
                skill_refs=["pstack/interrogate", "pstack/thermo-review"],
                max_visits=3,
            ),
            WorkflowNodeDef(
                name="verify_surface",
                type="human_gate",
                params={"gate": "spec"},
                god="argus",
                phase="verify",
                role="verifier",
            ),
        ],
        edges=[
            WorkflowEdgeDef(source=START, target="explore"),
            WorkflowEdgeDef(source="explore", target="checkpoint"),
            WorkflowEdgeDef(source="checkpoint", target="interrogate"),
            WorkflowEdgeDef(source="interrogate", target="verify_surface"),
            WorkflowEdgeDef(source="verify_surface", target=END),
        ],
    )


_LEGACY_SDLC_WORKFLOWS = ("full_sdlc_tdd", "product_blueprint", "repo_fast_patch")


def test_builtin_workflows_keep_backward_compatible_manifests() -> None:
    for name in _LEGACY_SDLC_WORKFLOWS:
        definition = load_workflow(name)
        assert definition.family == "sdlc"
        assert definition.autonomy == "gated"
        summary = WorkflowSummary.model_validate(describe_workflow(definition))
        assert summary.name == name
        assert summary.family == "sdlc"
        assert summary.autonomy == "gated"
        assert summary.applicable_scenarios == []
        assert summary.anti_patterns == []
        assert all(node.god is None and node.phase is None for node in summary.graph.nodes)


def test_custom_workflow_manifest_and_node_metadata_round_trip() -> None:
    definition = _custom_pstack_definition()
    described = describe_workflow(definition)
    summary = WorkflowSummary.model_validate(described)

    assert summary.family == "pstack"
    assert summary.autonomy == "autonomous_hard_stops"
    assert summary.stop_after == "verify"
    assert summary.applicable_scenarios == [
        "brownfield multi-file feature",
        "cross-boundary refactor",
    ]
    assert summary.anti_patterns == ["vague greenfield product idea"]

    by_name = {node.name: node for node in summary.graph.nodes}
    assert by_name["explore"].god == "prometheus"
    assert by_name["explore"].phase == "align"
    assert by_name["explore"].role == "explorer"
    assert by_name["explore"].skill_refs == ["pstack/fix-root-causes"]
    assert by_name["interrogate"].god == "momus"
    assert by_name["interrogate"].phase == "review"
    assert by_name["interrogate"].role == "interrogator"
    assert by_name["interrogate"].max_visits == 3
    assert by_name["verify_surface"].god == "argus"
    assert by_name["verify_surface"].phase == "verify"


def test_workflow_definition_rejects_invalid_manifest_values() -> None:
    with pytest.raises(ValidationError):
        WorkflowNodeDef(name="x", type="human_gate", god="zeus")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        WorkflowDefinition(
            name="bad",
            autonomy="yolo",  # type: ignore[arg-type]
            nodes=[WorkflowNodeDef(name="a", type="human_gate")],
        )

    with pytest.raises(ValidationError):
        WorkflowDefinition(
            name="bad_stop",
            stop_after="ship",  # type: ignore[arg-type]
            nodes=[WorkflowNodeDef(name="a", type="human_gate")],
        )


def test_explicit_pause_cursor_resumes_custom_workflow_without_sdlc_artifacts(
    store: RunStore,
) -> None:
    definition = _custom_pstack_definition()
    run_id = store.create_run(
        title="Custom pstack run",
        mode=RunMode.EXECUTION,
        workflow_name=definition.name,
    )

    # Fresh run without SDLC nodes falls back to the declared START edge target.
    assert resolve_resume_node(definition, store.get_run(run_id), store=store) == "explore"

    # Graph pause persists cursor; resume enters the exact paused node without GoalSpec/TaskPlan.
    store.append_event(
        run_id=run_id,
        stage="graph",
        event_type="graph.paused",
        actor="graph",
        payload={"node": "interrogate", "reason": "independent review blocked"},
    )
    assert extract_paused_cursor(store.list_events(run_id)) == "interrogate"
    assert resolve_resume_node(definition, store.get_run(run_id), store=store) == "interrogate"

    # Headless session pause updates cursor to a different node.
    store.append_event(
        run_id=run_id,
        stage="session",
        event_type="session.paused",
        actor="session",
        payload={"node": "checkpoint", "pause_kind": "clarify", "reason": "need decision"},
    )
    assert extract_paused_cursor(store.list_events(run_id)) == "checkpoint"
    assert resolve_resume_node(definition, store.get_run(run_id), store=store) == "checkpoint"

    # Completing that paused node clears the active cursor.
    store.append_event(
        run_id=run_id,
        stage="graph",
        event_type="graph.node_finished",
        actor="graph",
        payload={"node": "checkpoint", "updated_keys": []},
    )
    assert extract_paused_cursor(store.list_events(run_id)) is None


def test_explicit_cursor_preserves_legacy_sdlc_gate_progression(store: RunStore) -> None:
    definition = load_workflow("full_sdlc_tdd")
    run_id = store.create_run(
        title="Legacy SDLC run",
        mode=RunMode.EXECUTION,
        workflow_name=definition.name,
    )
    spec = GoalSpec(
        spec_id="spec_1",
        status=SpecStatus.DRAFT,
        goal="Ship feature",
        requirements=[
            RequirementItem(
                id="R1",
                description="Feature works",
                acceptance_criteria=["Tests pass"],
            )
        ],
        out_of_scope=["Unrelated modules"],
    )
    store.save_spec(run_id, spec)
    store.append_event(
        run_id=run_id,
        stage="graph",
        event_type="graph.paused",
        actor="graph",
        payload={"node": "gate_spec", "reason": "awaiting spec approval"},
    )

    # Still unapproved -> stays at gate_spec.
    assert resolve_resume_node(definition, store.get_run(run_id), store=store) == "gate_spec"

    # Once spec is approved, legacy gate_spec cursor is superseded and advances to design.
    store.record_approval(
        ApprovalRecord(
            run_id=run_id,
            target=ApprovalTarget.GOAL_SPEC,
            target_id=spec.spec_id,
            target_hash=spec.content_hash,
            approver="zeus",
            decision=ApprovalDecision.APPROVED,
        )
    )
    assert resolve_resume_node(definition, store.get_run(run_id), store=store) == "design"


def test_declarative_god_metadata_drives_board_classification_and_workflow_view(
    store: RunStore,
) -> None:
    definition = _custom_pstack_definition()
    run_id = store.create_run(
        title="Momus & Argus attribution",
        mode=RunMode.EXECUTION,
        workflow_name=definition.name,
    )
    store.append_event(
        run_id=run_id,
        stage="graph",
        event_type="graph.node_started",
        actor="graph",
        payload={"node": "interrogate", "kind": "human_gate"},
    )
    events = store.list_events(run_id)
    view = build_workflow_view(
        definition.name,
        events,
        RunStatus.RUNNING,
        definition=definition,
    )
    validated = WorkflowView.model_validate(view)
    assert validated.family == "pstack"
    assert validated.autonomy == "autonomous_hard_stops"

    board = classify_run_board(
        store.get_run(run_id),
        events,
        store=store,
        workflow_view=view,
    )
    assert board.active_god == "momus"

    # Move to verify_surface and complete -> Argus attribution and non-blueprint delivery.
    store.append_event(
        run_id=run_id,
        stage="graph",
        event_type="graph.node_finished",
        actor="graph",
        payload={"node": "interrogate", "updated_keys": []},
    )
    store.append_event(
        run_id=run_id,
        stage="graph",
        event_type="graph.node_started",
        actor="graph",
        payload={"node": "verify_surface", "kind": "human_gate"},
    )
    store.append_event(
        run_id=run_id,
        stage="graph",
        event_type="graph.node_finished",
        actor="graph",
        payload={"node": "verify_surface", "updated_keys": []},
    )
    store.append_event(
        run_id=run_id,
        stage="graph",
        event_type="graph.finished",
        actor="graph",
        payload={"graph": definition.name, "steps": 2},
    )
    snap = store.get_run(run_id)
    store.set_run_status(
        run_id,
        RunStatus.COMPLETED,
        expected_revision=int(snap["state_revision"]),
    )
    done_events = store.list_events(run_id)
    done_view = build_workflow_view(
        definition.name,
        done_events,
        RunStatus.COMPLETED,
        definition=definition,
    )
    done_board = classify_run_board(
        store.get_run(run_id),
        done_events,
        store=store,
        workflow_view=done_view,
    )
    assert done_board.active_god == "argus"
    assert done_board.status_badge == "delivered"


def test_role_drivers_single_model_fallback_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_cfg = ExecutionDriversConfig.resolve(
        stage_driver="native@stage-m",
        worker_driver="native@worker-m",
        reviewer_driver="native@review-m",
    )
    assert default_cfg.resolve_driver_for_role("explorer") == "native@stage-m"
    assert default_cfg.resolve_driver_for_role("architect") == "native@stage-m"
    assert default_cfg.resolve_driver_for_role("builder") == "native@worker-m"
    assert default_cfg.resolve_driver_for_role("hygiene") == "native@worker-m"
    assert default_cfg.resolve_driver_for_role("interrogator") == "native@review-m"
    assert default_cfg.resolve_driver_for_role("verifier") == "native@review-m"
    assert default_cfg.resolve_driver_for_role("arena_judge") == "native@review-m"
    assert default_cfg.resolve_driver_for_role(None) == "native@worker-m"
    assert default_cfg.resolve_driver_for_role("custom_unknown") == "native@worker-m"

    monkeypatch.setenv(
        "OLYMPUS_ROLE_DRIVERS",
        '{"Interrogator": "opencode@gpt-6.2-sol", "arena_judge": "pi@claude-opus"}',
    )
    overridden = ExecutionDriversConfig.resolve(
        worker_driver="native@default-m",
        role_drivers={"interrogator": "opencode@override-m"},
    )
    assert overridden.resolve_driver_for_role("interrogator") == "opencode@override-m"
    assert overridden.resolve_driver_for_role("arena_judge") == "pi@claude-opus"
    assert overridden.resolve_driver_for_role("builder") == "native@default-m"

    monkeypatch.setenv("OLYMPUS_ROLE_DRIVERS", "not-json")
    with pytest.raises(ValueError, match="OLYMPUS_ROLE_DRIVERS"):
        ExecutionDriversConfig.resolve()

    monkeypatch.delenv("OLYMPUS_ROLE_DRIVERS", raising=False)
    with pytest.raises(ValidationError):
        ExecutionDriversConfig(role_drivers={" ": "native"})


def test_headless_session_runner_propagates_workspace_context_and_role_drivers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    state_dir = tmp_path / "state"
    captured_states: list[dict[str, Any]] = []

    def _fake_run_graph(
        graph: Any,
        state: dict[str, Any],
        *,
        observer: Any = None,
        max_steps: int = 25,
        start_from: str | None = None,
    ) -> dict[str, Any]:
        del graph, observer, max_steps, start_from
        captured_states.append(dict(state))
        return {**state, "_paused_node": "gate_spec", "_pause_reason": "test stop"}

    monkeypatch.setattr("olympus.session.runner.run_graph", _fake_run_graph)

    runner = SessionRunner(
        SessionConfig(
            workspace_root=workspace,
            state_dir=state_dir,
            workflow="product_blueprint",
            provider="mock",
            workspace_id="ws_demo",
            workspace_mode="greenfield",
            task_type="refactor",
            base_ref="origin/main",
            scope_paths=["src/app.py", "tests/test_app.py"],
            role_drivers={"interrogator": "native@review-override"},
            provider_builder=lambda **_kwargs: MockLLMProvider(
                ['{"action": "approve", "feedback": "", "answers": {}}']
            ),
        )
    )
    envelope = runner.start(goal="Refactor application module")
    assert envelope.error is None
    assert envelope.run_id is not None
    assert len(captured_states) == 1
    state = captured_states[0]
    assert state["workspace_id"] == "ws_demo"
    assert state["workspace_mode"] == "greenfield"
    assert state["task_type"] == "refactor"
    assert state["base_ref"] == "origin/main"
    assert state["scope_paths"] == ["src/app.py", "tests/test_app.py"]

    snap = runner.store.get_run(envelope.run_id)
    assert snap["workspace_mode"] == "greenfield"
    assert snap["task_type"] == "refactor"
    assert snap["base_ref"] == "origin/main"
    assert snap["scope_paths"] == ["src/app.py", "tests/test_app.py"]
    assert snap["execution_drivers"]["role_drivers"] == {
        "interrogator": "native@review-override"
    }


def test_builtin_workflow_model_dump_preserves_legacy_workflow_hash() -> None:
    """New default manifest/node fields must not change legacy content_hash on upgrade."""
    for name in _LEGACY_SDLC_WORKFLOWS:
        dumped = load_workflow(name).model_dump(mode="json")
        for new_wf_key in ("family", "autonomy", "applicable_scenarios", "anti_patterns"):
            assert new_wf_key not in dumped
        for node in dumped["nodes"]:
            for new_node_key in ("god", "phase", "role", "skill_refs"):
                assert new_node_key not in node


def test_custom_human_gate_pauses_under_own_node_name_and_non_sdlc_gate_spec_resumes(
    store: RunStore,
    tmp_path: Path,
) -> None:
    from olympus.graph import WorkflowRuntime, build_graph, run_graph

    definition = _custom_pstack_definition()
    run_id = store.create_run(
        title="Custom gate run",
        mode=RunMode.EXECUTION,
        workflow_name=definition.name,
    )
    runtime = WorkflowRuntime(
        store=store,
        workspace_root=tmp_path,
        artifacts_root=tmp_path / "artifacts",
        provider=MockLLMProvider([]),
    )
    recorded_events: list[tuple[str, dict[str, Any]]] = []
    result = run_graph(
        build_graph(definition),
        {"run_id": run_id, "raw_input": "x", "_runtime": runtime},
        start_from="interrogate",
        observer=lambda name, payload: recorded_events.append((name, payload)),
    )
    assert result["_paused_node"] == "interrogate"
    assert any(
        ev == "graph_paused" and payload.get("node") == "interrogate"
        for ev, payload in recorded_events
    )

    # Non-SDLC workflow with a node named "gate_spec" must not be filtered out by SDLC rules.
    non_sdlc_gate = WorkflowDefinition(
        name="pstack_with_gate_spec",
        family="pstack",
        autonomy="autonomous_hard_stops",
        nodes=[
            WorkflowNodeDef(name="gate_spec", type="human_gate", params={"gate": "spec"}),
            WorkflowNodeDef(name="review", type="human_gate", params={"gate": "spec"}),
        ],
        edges=[
            WorkflowEdgeDef(source=START, target="gate_spec"),
            WorkflowEdgeDef(source="gate_spec", target="review"),
            WorkflowEdgeDef(source="review", target=END),
        ],
    )
    run_id2 = store.create_run(
        title="Non-SDLC gate_spec",
        mode=RunMode.EXECUTION,
        workflow_name=non_sdlc_gate.name,
    )
    spec = GoalSpec(
        spec_id="spec_non_sdlc",
        status=SpecStatus.DRAFT,
        goal="Check non-SDLC resume",
        requirements=[
            RequirementItem(id="R1", description="Works", acceptance_criteria=["Passes"])
        ],
        out_of_scope=["Other"],
    )
    store.save_spec(run_id2, spec)
    store.record_approval(
        ApprovalRecord(
            run_id=run_id2,
            target=ApprovalTarget.GOAL_SPEC,
            target_id=spec.spec_id,
            target_hash=spec.content_hash,
            approver="zeus",
            decision=ApprovalDecision.APPROVED,
        )
    )
    store.append_event(
        run_id=run_id2,
        stage="graph",
        event_type="graph.paused",
        actor="graph",
        payload={"node": "gate_spec", "reason": "awaiting approval"},
    )
    assert resolve_resume_node(non_sdlc_gate, store.get_run(run_id2), store=store) == "gate_spec"


def test_repo_fast_patch_failed_validate_plan_resumes_at_plan(store: RunStore) -> None:
    from olympus.core.contracts import AcceptanceCheck, CheckKind, TaskNode, TaskPlan

    definition = load_workflow("repo_fast_patch")
    run_id = store.create_run(
        title="Fast patch plan validation failure",
        mode=RunMode.EXECUTION,
        workflow_name=definition.name,
    )
    spec = GoalSpec(
        spec_id="spec_fp",
        status=SpecStatus.DRAFT,
        goal="Fix small bug",
        requirements=[
            RequirementItem(id="R1", description="Bug fixed", acceptance_criteria=["Tests pass"])
        ],
        out_of_scope=["Other"],
    )
    store.save_spec(run_id, spec)
    store.record_approval(
        ApprovalRecord(
            run_id=run_id,
            target=ApprovalTarget.GOAL_SPEC,
            target_id=spec.spec_id,
            target_hash=spec.content_hash,
            approver="zeus",
            decision=ApprovalDecision.APPROVED,
        )
    )
    plan = TaskPlan(
        plan_id="plan_fp",
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1",
                title="Patch",
                objective="Apply patch",
                write_scope=["src/a.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="C1",
                        description="Run unit tests",
                        kind=CheckKind.COMMAND,
                        command="pytest -q",
                    )
                ],
            )
        ],
    )
    store.save_plan(run_id, plan)
    store.append_event(
        run_id=run_id,
        stage="graph",
        event_type="graph.paused",
        actor="graph",
        payload={"node": "validate_plan", "reason": "plan validation failed"},
    )
    assert resolve_resume_node(definition, store.get_run(run_id), store=store) == "plan"


def test_build_workflow_view_loads_custom_yaml_from_recorded_workflow_path(
    store: RunStore,
    tmp_path: Path,
) -> None:
    import yaml

    custom_yaml = tmp_path / "custom_pstack.yaml"
    custom_yaml.write_text(
        yaml.safe_dump(_custom_pstack_definition().model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    run_id = store.create_run(
        title="Custom YAML projection",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_demo",
    )
    store.append_event(
        run_id=run_id,
        stage="session",
        event_type="session.started",
        actor="session",
        payload={"workflow": "pstack_demo", "workflow_path": str(custom_yaml)},
    )
    store.append_event(
        run_id=run_id,
        stage="graph",
        event_type="graph.node_started",
        actor="graph",
        payload={"node": "interrogate", "kind": "human_gate"},
    )
    view = build_workflow_view(
        "pstack_demo",
        store.list_events(run_id),
        RunStatus.RUNNING,
    )
    assert view["family"] == "pstack"
    assert view["autonomy"] == "autonomous_hard_stops"
    assert view["graph"] is not None
    board = classify_run_board(
        store.get_run(run_id),
        store.list_events(run_id),
        store=store,
        workflow_view=view,
    )
    assert board.active_god == "momus"


def test_role_drivers_explicit_wins_over_mixed_case_env_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "OLYMPUS_ROLE_DRIVERS",
        '{"reviewer": "env-lower", "Reviewer": "env-mixed"}',
    )
    cfg = ExecutionDriversConfig.resolve(role_drivers={"reviewer": "explicit-driver"})
    assert cfg.resolve_driver_for_role("reviewer") == "explicit-driver"


def test_session_runner_recognizes_custom_human_gate_approval_and_blocked_verify_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import yaml

    from olympus.core.contracts import AcceptanceCheck, CheckKind, TaskNode, TaskPlan, TaskStatus
    from olympus.session.contracts import PauseKind, SessionStatus

    workspace = tmp_path / "repo"
    workspace.mkdir()
    state_dir = tmp_path / "state"

    custom_wf = WorkflowDefinition(
        name="pstack_gate_test",
        family="pstack",
        autonomy="autonomous_hard_stops",
        stop_after="verify",
        nodes=[
            WorkflowNodeDef(
                name="goal_review",
                type="human_gate",
                params={"gate": "spec"},
                god="prometheus",
                phase="align",
            ),
        ],
        edges=[
            WorkflowEdgeDef(source=START, target="goal_review"),
            WorkflowEdgeDef(source="goal_review", target=END),
        ],
    )
    wf_file = tmp_path / "pstack_gate_test.yaml"
    wf_file.write_text(
        yaml.safe_dump(custom_wf.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    runner = SessionRunner(
        SessionConfig(
            workspace_root=workspace,
            state_dir=state_dir,
            workflow=str(wf_file),
            provider="mock",
            provider_builder=lambda **_kwargs: MockLLMProvider(
                ['{"action": "approve", "feedback": "", "answers": {}}']
            ),
        )
    )

    # 1. Custom human_gate ("goal_review") pausing for spec approval must emit APPROVAL + gate=spec
    spec = GoalSpec(
        spec_id="spec_custom_gate",
        status=SpecStatus.DRAFT,
        goal="Ship custom gate",
        requirements=[
            RequirementItem(id="R1", description="Works", acceptance_criteria=["Passes"])
        ],
        out_of_scope=["Other"],
    )

    def _pause_custom_gate(graph: Any, state: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        del graph
        runner.store.save_spec(state["run_id"], spec)
        return {
            **state,
            "_paused_node": "goal_review",
            "_pause_reason": "GoalSpec awaiting approval",
        }

    monkeypatch.setattr("olympus.session.runner.run_graph", _pause_custom_gate)
    env = runner.start(goal="Run custom gate")
    assert env.status == SessionStatus.NEEDS_INPUT
    assert env.pause_kind == PauseKind.APPROVAL
    assert env.run_id is not None
    pause_events = [
        ev for ev in runner.store.list_events(env.run_id) if ev.event_type == "session.paused"
    ]
    assert pause_events[-1].payload["node"] == "goal_review"
    assert pause_events[-1].payload["gate"] == "spec"

    # 1b. Concurrent conflict during approve must preserve gate="spec" so retry still approves.
    from olympus.core.store import ConcurrencyConflictError

    approve_calls: list[str] = []
    real_approve = runner._approve_gate

    def _flaky_approve(rid: str, gate_name: str, *, expected_revision: int | None) -> None:
        approve_calls.append(gate_name)
        if len(approve_calls) == 1:
            raise ConcurrencyConflictError("transient revision conflict")
        real_approve(rid, gate_name, expected_revision=expected_revision)

    monkeypatch.setattr(runner, "_approve_gate", _flaky_approve)
    conflict_env = runner.reply(run_id=env.run_id, message="approve")
    assert conflict_env.status == SessionStatus.NEEDS_INPUT
    assert conflict_env.pause_kind == PauseKind.APPROVAL
    pause_after_conflict = [
        ev for ev in runner.store.list_events(env.run_id) if ev.event_type == "session.paused"
    ][-1]
    assert pause_after_conflict.payload["node"] == "goal_review"
    assert pause_after_conflict.payload["gate"] == "spec"

    monkeypatch.setattr("olympus.session.runner.run_graph", lambda _g, st, **_kw: dict(st))
    retry_env = runner.reply(run_id=env.run_id, message="approve")
    assert retry_env.status == SessionStatus.COMPLETED
    assert approve_calls == ["spec", "spec"]

    # 2. stop_after="verify" with a BLOCKED task must stay BLOCKED, never COMPLETED.
    def _finish_with_blocked_task(
        graph: Any, state: dict[str, Any], **_kw: Any
    ) -> dict[str, Any]:
        del graph
        rid = state["run_id"]
        runner.store.save_spec(rid, spec)
        runner.store.record_approval(
            ApprovalRecord(
                run_id=rid,
                target=ApprovalTarget.GOAL_SPEC,
                target_id=spec.spec_id,
                target_hash=spec.content_hash,
                approver="zeus",
                decision=ApprovalDecision.APPROVED,
            )
        )
        blocked_plan = TaskPlan(
            plan_id="plan_blocked",
            spec_id=spec.spec_id,
            spec_hash=spec.content_hash,
            tasks=[
                TaskNode(
                    task_id="T1",
                    title="Blocked task",
                    objective="Fail verification",
                    status=TaskStatus.BLOCKED,
                    write_scope=["src/a.py"],
                    acceptance_checks=[
                        AcceptanceCheck(
                            id="C1",
                            description="Run tests",
                            kind=CheckKind.COMMAND,
                            command="pytest -q",
                        )
                    ],
                )
            ],
        )
        runner.store.save_plan(rid, blocked_plan)
        runner.store.record_approval(
            ApprovalRecord(
                run_id=rid,
                target=ApprovalTarget.TASK_PLAN,
                target_id=blocked_plan.plan_id,
                target_hash=blocked_plan.content_hash,
                approver="zeus",
                decision=ApprovalDecision.APPROVED,
            )
        )
        return dict(state)

    monkeypatch.setattr("olympus.session.runner.run_graph", _finish_with_blocked_task)
    env2 = runner.start(goal="Run blocked verify")
    assert env2.status == SessionStatus.NEEDS_INPUT
    assert env2.pause_kind == PauseKind.BLOCKED
    assert env2.run_id is not None
    assert runner.store.get_run(env2.run_id)["status"] == RunStatus.AWAITING_HUMAN


def test_extract_paused_cursor_reads_top_level_status_changed_paused_node(
    store: RunStore,
) -> None:
    definition = _custom_pstack_definition()
    run_id = store.create_run(
        title="Status-changed top-level cursor",
        mode=RunMode.EXECUTION,
        workflow_name=definition.name,
    )
    snap = store.get_run(run_id)
    store.set_run_status(
        run_id,
        RunStatus.AWAITING_HUMAN,
        expected_revision=int(snap["state_revision"]),
        actor="session",
        stage="session",
        detail={"paused_node": "checkpoint", "pause_reason": "interrupted before session.paused"},
    )
    assert extract_paused_cursor(store.list_events(run_id)) == "checkpoint"
    assert resolve_resume_node(definition, store.get_run(run_id), store=store) == "checkpoint"


def test_build_workflow_view_prefers_recorded_custom_yaml_even_with_builtin_name(
    store: RunStore,
    tmp_path: Path,
) -> None:
    import yaml

    custom_override = WorkflowDefinition(
        name="product_blueprint",
        family="pstack",
        autonomy="autonomous_hard_stops",
        nodes=[
            WorkflowNodeDef(
                name="goal_review",
                type="human_gate",
                params={"gate": "spec"},
                god="momus",
                phase="review",
            )
        ],
        edges=[
            WorkflowEdgeDef(source=START, target="goal_review"),
            WorkflowEdgeDef(source="goal_review", target=END),
        ],
    )
    custom_yaml = tmp_path / "override_blueprint.yaml"
    custom_yaml.write_text(
        yaml.safe_dump(custom_override.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    run_id = store.create_run(
        title="Override builtin name",
        mode=RunMode.EXECUTION,
        workflow_name="product_blueprint",
    )
    store.append_event(
        run_id=run_id,
        stage="session",
        event_type="session.started",
        actor="session",
        payload={"workflow": "product_blueprint", "workflow_path": str(custom_yaml)},
    )
    view = build_workflow_view(
        "product_blueprint",
        store.list_events(run_id),
        RunStatus.RUNNING,
    )
    assert view["family"] == "pstack"
    assert [n["name"] for n in view["graph"]["nodes"]] == ["goal_review"]


