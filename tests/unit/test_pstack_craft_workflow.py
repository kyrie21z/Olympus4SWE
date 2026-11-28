"""Offline contract, isolation, DiffGate, back-edge, and E2E tests for Issue #189
(`pstack_craft` native workflow and six orthogonal craft primitives).
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from olympus.core.contracts import (
    ArchitectSketch,
    CandidateShape,
    EvidenceBrief,
    EvidenceFact,
    InterrogateFinding,
    InterrogateReport,
    RunMode,
    SurfaceCheckResult,
    SurfaceVerificationReport,
)
from olympus.core.store import RunStore
from olympus.graph import (
    WorkflowRuntime,
    build_graph,
    describe_workflow,
    list_builtin_workflows,
    load_workflow,
    resolve_resume_node,
    run_graph,
)
from olympus.harness.driver import (
    ActiveTurnHandle,
    DriverEvent,
    DriverRegistry,
    DriverTurnRequest,
    ExecutionDriversConfig,
    TurnOutcome,
)
from olympus.providers.base import MockLLMProvider
from olympus.providers.types import LLMResponse
from olympus.stages.craft import (
    CraftPause,
    CraftStage,
    _probe_bwrap,
    load_skill_pack,
)
from olympus.web.schemas import WorkflowSummary


@pytest.fixture(autouse=True)
def _ci_bwrap_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep local Linux tests running under real bwrap; fail loudly when CI requires isolation."""
    if _probe_bwrap() is None:
        if os.environ.get("OLYMPUS_REQUIRE_ISOLATION_TESTS") == "1":
            pytest.fail(
                "OLYMPUS_REQUIRE_ISOLATION_TESTS=1 is set, but bwrap isolation is unavailable"
            )
        monkeypatch.setenv("OLYMPUS_ALLOW_UNSANDBOXED_REPRO", "1")


def _mock_provider(payloads: list[str]) -> MockLLMProvider:
    return MockLLMProvider([LLMResponse(content=item) for item in payloads])


class _ScriptedDriver:
    """Deterministic test AgentDriver executing scripted filesystem/JSON actions."""

    name = "scripted"

    def __init__(self, handler: Callable[[DriverTurnRequest], str]) -> None:
        self._handler = handler
        self.requests: list[DriverTurnRequest] = []

    def run_turn(
        self,
        request: DriverTurnRequest,
        *,
        on_event: DriverEvent | None = None,
        on_handle: Callable[[ActiveTurnHandle], None] | None = None,
    ) -> TurnOutcome:
        del on_event, on_handle
        self.requests.append(request)
        content = self._handler(request)
        return TurnOutcome(
            completed=True,
            stop_reason="completed",
            final_content=content,
            steps_taken=1,
            driver_name=self.name,
        )


def _init_demo_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "app.py").write_text("def add(a: int, b: int) -> int:\n    return a - b\n", "utf-8")
    (repo / "test_app.py").write_text(
        "import unittest\n"
        "import app\n\n"
        "class TestApp(unittest.TestCase):\n"
        "    def test_add(self) -> None:\n"
        "        self.assertEqual(app.add(2, 3), 5)\n",
        "utf-8",
    )
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)


def _sample_evidence(*, unresolved: list[str] | None = None) -> str:
    return EvidenceBrief(
        summary="app.add subtracts instead of adding",
        facts=[EvidenceFact(path="app.py", line=2, observation="returns a - b")],
        candidate_files=["app.py", "test_app.py"],
        unresolved_questions=list(unresolved or []),
    ).model_dump_json()


def _sample_sketch(*, verify_cmds: list[str] | None = None) -> str:
    cmds = verify_cmds if verify_cmds is not None else [f"{sys.executable} -m unittest test_app.py"]
    return ArchitectSketch(
        candidate_shapes=[
            CandidateShape(
                name="direct_fix",
                data_flow="Return a + b directly in app.add",
                tradeoffs="Minimal diff, zero new indirection",
            ),
            CandidateShape(
                name="operator_table",
                data_flow="Dispatch arithmetic through a lookup table",
                tradeoffs="Unnecessary abstraction for a single helper",
            ),
        ],
        chosen_shape="direct_fix",
        rationale="Direct fix satisfies subtract-before-add with minimal blast radius",
        state_owners={"arithmetic": "app.py"},
        write_scope=["app.py"],
        forbidden_globs=["test_app.py"],
        blast_radius="Only callers of app.add",
        verification_plan=cmds,
    ).model_dump_json()


def test_pstack_craft_manifest_and_skill_packs_validate() -> None:
    assert "pstack_craft" in list_builtin_workflows()
    definition = load_workflow("pstack_craft")
    summary = WorkflowSummary.model_validate(describe_workflow(definition))

    assert summary.family == "pstack"
    assert summary.autonomy == "autonomous_hard_stops"
    assert summary.stop_after == "verify"
    assert summary.applicable_scenarios
    assert summary.anti_patterns

    by_name = {node.name: node for node in summary.graph.nodes}
    assert by_name["explore"].god == "prometheus"
    assert by_name["architect"].god == "athena"
    assert by_name["build"].god == "hephaestus"
    assert by_name["hygiene"].god == "hephaestus"
    assert by_name["interrogate"].god == "momus"
    assert by_name["verify_surface"].god == "argus"

    for node in definition.nodes:
        pack = load_skill_pack(node.skill_refs)
        assert pack.strip()

    with pytest.raises(ValueError, match="Invalid skill_ref"):
        load_skill_pack(["../outside"])
    with pytest.raises(FileNotFoundError):
        load_skill_pack(["pstack/does-not-exist"])


def test_craft_contract_invariants_reject_invalid_shapes_and_inconclusive_pass() -> None:
    with pytest.raises(ValidationError):
        ArchitectSketch(
            candidate_shapes=[
                CandidateShape(name="only_one", data_flow="x", tradeoffs="y"),
            ],
            chosen_shape="only_one",
            rationale="not enough shapes",
            write_scope=["app.py"],
            verification_plan=["pytest -q"],
        )

    with pytest.raises(ValidationError, match="chosen_shape"):
        ArchitectSketch(
            candidate_shapes=[
                CandidateShape(name="a", data_flow="x", tradeoffs="y"),
                CandidateShape(name="b", data_flow="z", tradeoffs="w"),
            ],
            chosen_shape="missing",
            rationale="mismatch",
            write_scope=["app.py"],
            verification_plan=["pytest -q"],
        )

    with pytest.raises(ValidationError, match="non-passing"):
        SurfaceVerificationReport(
            passed=True,
            checks=[
                SurfaceCheckResult(
                    surface="check_1",
                    command="pytest -q",
                    status="inconclusive",
                    evidence="timed out",
                )
            ],
        )

    with pytest.raises(ValidationError, match="zero checks"):
        SurfaceVerificationReport(passed=True, checks=[])


def test_pstack_craft_autonomous_happy_path_filters_unproven_findings_and_promotes(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Fix app.add",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=repo,
    )

    def _builder_action(req: DriverTurnRequest) -> str:
        if req.actor == "hephaestus:build":
            (req.workspace_path / "app.py").write_text(
                "# obvious comment\ndef add(a: int, b: int) -> int:\n    return a + b\n",
                "utf-8",
            )
            # Ephemeral untracked junk outside write_scope should be cleaned before DiffGate.
            (req.workspace_path / "scratch.tmp").write_text("temp", "utf-8")
            return "Implemented add fix."
        if req.actor == "hephaestus:hygiene":
            (req.workspace_path / "app.py").write_text(
                "def add(a: int, b: int) -> int:\n    return a + b\n",
                "utf-8",
            )
            return "Removed redundant comment."
        raise AssertionError(f"Unexpected worker actor: {req.actor}")

    builder_driver = _ScriptedDriver(_builder_action)

    # Interrogate reports two speculative/passing-repro claims -> both must be filtered out!
    unproven_review = InterrogateReport(
        verdict="findings",
        findings=[
            InterrogateFinding(
                severity="important",
                file_path="app.py",
                claim="Speculative claim with no repro command",
                repro_command=None,
                proven=True,  # LLM tries to claim proven=True without running anything!
            ),
            InterrogateFinding(
                severity="important",
                file_path="app.py",
                claim="Claim whose repro command actually passes (exit 0)",
                repro_command=f"{sys.executable} -m unittest test_app.py",
                proven=True,
            ),
        ],
        summary="Trying to block without real proof",
    ).model_dump_json()

    provider = _mock_provider([_sample_evidence(), _sample_sketch(), unproven_review])
    runtime = WorkflowRuntime(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=provider,
        apply_to_workspace=True,
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="scripted",
            reviewer_driver="native",
        ),
        driver_registry=DriverRegistry({"scripted": builder_driver}),
    )

    definition = load_workflow("pstack_craft")
    result = run_graph(
        build_graph(definition),
        {"run_id": run_id, "raw_input": "Fix app.add to sum integers", "_runtime": runtime},
    )

    assert result.get("_paused_node") is None
    assert result["interrogate_report"]["verdict"] == "pass"
    assert result["interrogate_report"]["cold_start"] is True
    assert len(result["interrogate_report"]["unproven_notes"]) == 2
    assert all(not f["proven"] for f in result["interrogate_report"]["findings"])
    assert result["surface_verification"]["passed"] is True
    assert result["applied_paths"] == ["app.py"]

    # Real artifact promoted to host repo passes real unittest cleanly!
    assert "return a + b" in (repo / "app.py").read_text("utf-8")
    assert "# obvious comment" not in (repo / "app.py").read_text("utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "test_app.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    # Verify cold-start isolation: Momus prompt did not receive builder trace/history.
    events = store.list_events(run_id)
    event_types = [ev.event_type for ev in events]
    assert "craft.evidence_recorded" in event_types
    assert "craft.architect_recorded" in event_types
    assert "craft.build_completed" in event_types
    assert "craft.hygiene_completed" in event_types
    assert "craft.interrogate_recorded" in event_types
    assert "craft.verify_recorded" in event_types
    interrogate_ev = next(ev for ev in events if ev.event_type == "craft.interrogate_recorded")
    assert interrogate_ev.payload["cold_start"] is True
    assert interrogate_ev.payload["filtered_unproven_count"] == 2
    assert interrogate_ev.payload["proven_blocking_count"] == 0


def test_pstack_craft_pauses_on_intent_ambiguity_and_resumes_at_explore(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Ambiguous request",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=repo,
    )
    definition = load_workflow("pstack_craft")
    provider = _mock_provider(
        [
            _sample_evidence(unresolved=["Should add() accept floats or ints only?"]),
            _sample_evidence(unresolved=[]),
            _sample_sketch(),
        ]
    )
    runtime = WorkflowRuntime(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=provider,
    )

    def _record(kind: str, payload: dict[str, Any]) -> None:
        if kind == "graph_paused":
            store.append_event(
                run_id=run_id,
                stage="graph",
                event_type="graph.paused",
                actor="graph",
                payload=payload,
            )

    paused = run_graph(
        build_graph(definition),
        {"run_id": run_id, "raw_input": "Update add", "_runtime": runtime},
        observer=_record,
    )
    assert paused["_paused_node"] == "explore"
    assert paused["pause_kind"] == "clarify"
    assert resolve_resume_node(definition, store.get_run(run_id), store=store) == "explore"


def test_build_direct_retries_diff_gate_violation_and_blocks_if_exhausted(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="DiffGate test",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=repo,
    )

    attempt_num = 0

    def _violate_then_fix(req: DriverTurnRequest) -> str:
        nonlocal attempt_num
        attempt_num += 1
        if attempt_num == 1:
            (req.workspace_path / "outside.py").write_text("oops = 1\n", "utf-8")
            (req.workspace_path / "app.py").write_text(
                "def add(a: int, b: int) -> int:\n    return a + b\n", "utf-8"
            )
            return "First attempt"
        assert "[DIFF GATE REJECTED]" in req.user_prompt
        (req.workspace_path / "outside.py").unlink(missing_ok=True)
        return "Cleaned out-of-scope file"

    driver = _ScriptedDriver(_violate_then_fix)
    stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=MockLLMProvider([]),
        execution_drivers=ExecutionDriversConfig.resolve(worker_driver="scripted"),
        driver_registry=DriverRegistry({"scripted": driver}),
    )
    sketch = json.loads(_sample_sketch())
    result = stage.build_direct(
        run_id=run_id,
        raw_input="Fix add",
        state={"architect_sketch": sketch},
        max_diff_gate_retries=2,
    )
    assert result["route"] == "built"
    assert result["craft_changed_paths"] == ["app.py"]
    assert attempt_num == 2
    assert any(ev.event_type == "craft.diff_gate_rejected" for ev in store.list_events(run_id))

    # Persistent forbidden edit exhausts retries and raises CraftPause.
    run_id2 = store.create_run(
        title="DiffGate block",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=repo,
    )
    bad_driver = _ScriptedDriver(
        lambda req: (
            (req.workspace_path / "test_app.py").write_text("# tampered\n", "utf-8"),
            "tampered",
        )[1]
    )
    bad_stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=MockLLMProvider([]),
        execution_drivers=ExecutionDriversConfig.resolve(worker_driver="scripted"),
        driver_registry=DriverRegistry({"scripted": bad_driver}),
    )
    with pytest.raises(CraftPause) as exc_info:
        bad_stage.build_direct(
            run_id=run_id2,
            raw_input="Fix add",
            state={"architect_sketch": sketch},
            max_diff_gate_retries=2,
        )
    assert exc_info.value.node_name == "build"
    assert exc_info.value.state_updates["pause_kind"] == "blocked"


def test_interrogate_proven_finding_loops_back_to_build_and_caps_at_max_visits(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Interrogate back-edge",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=repo,
    )

    build_calls = 0

    def _build_then_repair(req: DriverTurnRequest) -> str:
        nonlocal build_calls
        if req.actor == "hephaestus:build":
            build_calls += 1
            if build_calls == 1:
                # Introduce a subtle bug on cycle 1 (off-by-one)
                (req.workspace_path / "app.py").write_text(
                    "def add(a: int, b: int) -> int:\n    return a + b + 1\n", "utf-8"
                )
            else:
                assert "off-by-one" in req.user_prompt
                (req.workspace_path / "app.py").write_text(
                    "def add(a: int, b: int) -> int:\n    return a + b\n", "utf-8"
                )
        return "done"

    failing_repro = f"{sys.executable} -m unittest test_app.py"
    cycle1_review = InterrogateReport(
        verdict="findings",
        findings=[
            InterrogateFinding(
                severity="important",
                file_path="app.py",
                claim="add() has an off-by-one bug",
                repro_command=failing_repro,
            )
        ],
    ).model_dump_json()
    cycle2_review = InterrogateReport(verdict="pass", findings=[]).model_dump_json()

    provider = _mock_provider([_sample_evidence(), _sample_sketch(), cycle1_review, cycle2_review])
    runtime = WorkflowRuntime(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=provider,
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="scripted",
            reviewer_driver="native",
        ),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_build_then_repair)}),
    )

    result = run_graph(
        build_graph(load_workflow("pstack_craft")),
        {"run_id": run_id, "raw_input": "Fix app.add", "_runtime": runtime},
    )
    assert build_calls == 2
    assert result["craft_review_cycle"] == 2
    assert result["interrogate_report"]["verdict"] == "pass"
    assert result["surface_verification"]["passed"] is True

    # Convergence cap: 3 consecutive proven failures pause cleanly at interrogate
    # (never raising an uncaught GraphVisitLimitError).
    run_id_cap = store.create_run(
        title="Interrogate cap",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=repo,
    )
    stubborn_builder = _ScriptedDriver(
        lambda req: (
            (req.workspace_path / "app.py").write_text(
                "def add(a: int, b: int) -> int:\n    return a + b + 99\n", "utf-8"
            ),
            "still broken",
        )[1]
    )
    cap_provider = _mock_provider(
        [_sample_evidence(), _sample_sketch(), cycle1_review, cycle1_review, cycle1_review]
    )
    cap_runtime = WorkflowRuntime(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=cap_provider,
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="scripted",
            reviewer_driver="native",
        ),
        driver_registry=DriverRegistry({"scripted": stubborn_builder}),
    )
    capped = run_graph(
        build_graph(load_workflow("pstack_craft")),
        {"run_id": run_id_cap, "raw_input": "Fix app.add", "_runtime": cap_runtime},
    )
    assert capped["_paused_node"] == "interrogate"
    assert capped["pause_kind"] == "blocked"
    assert capped["interrogate_report"]["verdict"] == "blocked"
    assert capped["craft_review_cycle"] == 3


def test_verify_surface_rejects_inconclusive_and_thermo_dual_lane_dedupes(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Inconclusive verify",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=repo,
    )

    stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=MockLLMProvider([]),
        command_runner=lambda _cmd, **_kw: (None, "timed out"),
    )
    with pytest.raises(CraftPause) as exc_info:
        stage.verify_surface(
            run_id=run_id,
            state={"architect_sketch": json.loads(_sample_sketch())},
            apply_to_workspace=True,
        )
    assert exc_info.value.node_name == "verify_surface"
    assert exc_info.value.state_updates["surface_verification"]["passed"] is False
    assert (
        exc_info.value.state_updates["surface_verification"]["checks"][0]["status"]
        == "inconclusive"
    )
    # Host workspace must remain untouched when verification did not pass.
    assert "return a - b" in (repo / "app.py").read_text("utf-8")

    # Thermo dual-lane deduplicates findings and tags code-judo lane.
    run_id2 = store.create_run(
        title="Thermo dual lane",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=repo,
    )
    lane1 = InterrogateReport(
        verdict="findings",
        findings=[
            InterrogateFinding(
                severity="nit",
                file_path="app.py",
                claim="Duplicate note",
            )
        ],
    ).model_dump_json()
    lane2 = InterrogateReport(
        verdict="findings",
        findings=[
            InterrogateFinding(severity="nit", file_path="app.py", claim="Duplicate note"),
            InterrogateFinding(
                severity="nit",
                file_path="app.py",
                claim="Simplify wrapper",
            ),
        ],
    ).model_dump_json()
    thermo_stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([lane1, lane2]),
    )
    out = thermo_stage.interrogate_review(
        run_id=run_id2,
        raw_input="Check thermo",
        state={"architect_sketch": json.loads(_sample_sketch())},
        enable_thermo=True,
    )
    assert out["route"] == "pass"
    claims = [(f["claim"], f["lane"]) for f in out["interrogate_report"]["findings"]]
    assert claims == [("Duplicate note", "interrogate"), ("Simplify wrapper", "code_judo")]


def test_vacuous_shell_commands_rejected_in_interrogate_and_verify_and_build_incomplete(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Vacuous command guard",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=repo,
    )

    # 1. Interrogate must filter out vacuous `false` / `exit 1` and exit 127 typos as unproven.
    vacuous_review = InterrogateReport(
        verdict="findings",
        findings=[
            InterrogateFinding(
                severity="important",
                file_path="app.py",
                claim="Fake finding using false",
                repro_command="false",
            ),
            InterrogateFinding(
                severity="important",
                file_path="app.py",
                claim="Fake finding using exit 1",
                repro_command="exit 1",
            ),
            InterrogateFinding(
                severity="important",
                file_path="app.py",
                claim="Typo command exiting 127",
                repro_command="nonexistent_cmd_xyz_189_test",
            ),
        ],
    ).model_dump_json()
    stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([vacuous_review]),
    )
    out = stage.interrogate_review(
        run_id=run_id,
        raw_input="Check vacuous repro",
        state={"architect_sketch": json.loads(_sample_sketch())},
    )
    assert out["route"] == "pass"
    assert out["interrogate_report"]["verdict"] == "pass"
    assert all(not f["proven"] for f in out["interrogate_report"]["findings"])
    assert len(out["interrogate_report"]["unproven_notes"]) == 3

    # 1b. Allowlisted runner exiting 127 (command not found) is also filtered as unproven.
    missing_runner_review = InterrogateReport(
        verdict="findings",
        findings=[
            InterrogateFinding(
                severity="important",
                file_path="app.py",
                claim="Missing runner exiting 127",
                repro_command="pytest -q",
            )
        ],
    ).model_dump_json()
    stage_127 = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([missing_runner_review]),
        command_runner=lambda _cmd, *, cwd, timeout_s: (127, "Command not found: pytest"),
    )
    out_127 = stage_127.interrogate_review(
        run_id=run_id,
        raw_input="Check exit 127",
        state={"architect_sketch": json.loads(_sample_sketch())},
    )
    assert out_127["route"] == "pass"
    assert out_127["interrogate_report"]["verdict"] == "pass"
    assert not out_127["interrogate_report"]["findings"][0]["proven"]
    assert "exit=127" in out_127["interrogate_report"]["unproven_notes"][0]

    # 1c. pytest exit=5 (no tests collected) must not prove either interrogate or explore repro.
    pytest_exit5_review = InterrogateReport(
        verdict="findings",
        findings=[
            InterrogateFinding(
                severity="important",
                file_path="app.py",
                claim="No tests collected",
                repro_command="pytest -q -k nonexistent_selector",
            )
        ],
    ).model_dump_json()
    stage_exit5 = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([pytest_exit5_review]),
        command_runner=lambda _cmd, *, cwd, timeout_s: (5, "no tests ran"),
    )
    out_exit5 = stage_exit5.interrogate_review(
        run_id=run_id,
        raw_input="Check pytest exit 5",
        state={"architect_sketch": json.loads(_sample_sketch())},
    )
    assert out_exit5["route"] == "pass"
    assert not out_exit5["interrogate_report"]["findings"][0]["proven"]
    assert "exit=5" in out_exit5["interrogate_report"]["unproven_notes"][0]

    explore_exit5 = EvidenceBrief(
        summary="Pytest empty collection",
        repro_command="pytest -q -k nonexistent_selector",
    ).model_dump_json()
    stage_explore5 = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([explore_exit5]),
        command_runner=lambda _cmd, *, cwd, timeout_s: (5, "no tests ran"),
    )
    with pytest.raises(CraftPause, match="exit=5"):
        stage_explore5.explore_reproduce(
            run_id=run_id,
            raw_input="Repro with no collected tests",
            require_repro=True,
        )

    # 1d. Signal termination (-15 / SIGTERM) must not prove failure either.
    stage_sigterm = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([pytest_exit5_review]),
        command_runner=lambda _cmd, *, cwd, timeout_s: (-15, "Terminated"),
    )
    out_sigterm = stage_sigterm.interrogate_review(
        run_id=run_id,
        raw_input="Check signal termination",
        state={"architect_sketch": json.loads(_sample_sketch())},
    )
    assert out_sigterm["route"] == "pass"
    assert not out_sigterm["interrogate_report"]["findings"][0]["proven"]
    assert "signal" in out_sigterm["interrogate_report"]["unproven_notes"][0]

    # 2. verify_surface must mark vacuous `true` / `:` as inconclusive, never pass.
    with pytest.raises(CraftPause) as exc_info:
        stage.verify_surface(
            run_id=run_id,
            state={"architect_sketch": json.loads(_sample_sketch(verify_cmds=["true", ":"]))},
        )
    statuses = [c["status"] for c in exc_info.value.state_updates["surface_verification"]["checks"]]
    assert statuses == ["inconclusive", "inconclusive"]

    # 3. build_direct failing (completed=False) with zero changed files must pause blocked.
    class _IncompleteDriver:
        name = "incomplete"

        def run_turn(
            self,
            request: DriverTurnRequest,
            *,
            on_event: DriverEvent | None = None,
            on_handle: Callable[[ActiveTurnHandle], None] | None = None,
        ) -> TurnOutcome:
            del request, on_event, on_handle
            return TurnOutcome(
                completed=False,
                stop_reason="max_steps_exceeded",
                final_content="",
                steps_taken=1,
                driver_name=self.name,
            )

    inc_stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=MockLLMProvider([]),
        execution_drivers=ExecutionDriversConfig.resolve(worker_driver="incomplete"),
        driver_registry=DriverRegistry({"incomplete": _IncompleteDriver()}),
    )
    with pytest.raises(CraftPause) as build_exc:
        inc_stage.build_direct(
            run_id=run_id,
            raw_input="Fix add",
            state={"architect_sketch": json.loads(_sample_sketch())},
        )
    assert build_exc.value.state_updates["pause_kind"] == "blocked"
    assert "max_steps_exceeded" in build_exc.value.reason


def test_repro_isolation_and_promotion_symlink_and_scope_security(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("secret", "utf-8")
    # Existing external symlink inside write_scope that the worker will delete.
    (repo / "old_link.txt").symlink_to(outside)
    subprocess.run(["git", "add", "old_link.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "add link"], cwd=repo, check=True, capture_output=True)

    store = RunStore(tmp_path / "olympus.sqlite")

    # 1. explore_reproduce(require_repro=True) runs allowlisted test repro inside throwaway sandbox!
    (repo / "test_repro_side_effect.py").write_text(
        "import unittest\n"
        "from pathlib import Path\n\n"
        "class SideEffectRepro(unittest.TestCase):\n"
        "    def test_repro(self) -> None:\n"
        "        Path('host_polluted.txt').write_text('sandbox_only', 'utf-8')\n"
        "        self.fail('reproduced bug')\n",
        "utf-8",
    )
    subprocess.run(["git", "add", "test_repro_side_effect.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "add repro test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    run_id_repro = store.create_run(
        title="Repro isolation",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=repo,
    )
    mutating_repro = EvidenceBrief(
        summary="Sandboxed allowlisted repro check",
        repro_command=f"{sys.executable} -m unittest test_repro_side_effect.py",
    ).model_dump_json()
    repro_stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([mutating_repro]),
    )
    out = repro_stage.explore_reproduce(
        run_id=run_id_repro,
        raw_input="Investigate",
        require_repro=True,
    )
    assert out["evidence_brief"]["repro_verified"] is True
    assert not (repo / "host_polluted.txt").exists()

    # 1b. Host-escaping, spoofed-basename, compileall, or non-allowlisted repro is rejected.
    evil_dir = tmp_path / "evil_bin"
    evil_dir.mkdir()
    fake_pytest = evil_dir / "pytest"
    fake_pytest.write_text("#!/bin/sh\necho pwned\nexit 1\n", "utf-8")
    fake_pytest.chmod(0o755)
    obfuscated_py = (
        f'{sys.executable} -c "import os; '
        "open(os.path.join(chr(47), 'tmp', 'pwn'), 'w').write('x')\""
    )
    for unsafe_cmd in (
        f"echo pwn > {outside}",
        "rm -rf app.py",
        f"{sys.executable} -c \"open('{outside}', 'w').write('pwn')\"",
        obfuscated_py,
        f"{sys.executable} -m compileall .",
        f"{fake_pytest} -q",
        "./pytest -q",
        f"pytest --basetemp={outside}",
        "pytest -pno:cacheprovider",
        "pytest -ofile=bad",
        "pytest -qo",
        "ruff check --fix .",
        "ruff check -w .",
    ):
        bad_repro = EvidenceBrief(summary="Unsafe", repro_command=unsafe_cmd).model_dump_json()
        unsafe_stage = CraftStage(
            store=store,
            workspace_root=repo,
            artifacts_root=tmp_path / "artifacts",
            sandbox_root=tmp_path / "sandboxes",
            provider=_mock_provider([bad_repro]),
        )
        with pytest.raises(CraftPause, match="Unsafe repro_command rejected"):
            unsafe_stage.explore_reproduce(
                run_id=run_id_repro,
                raw_input="Investigate",
                require_repro=True,
            )
    assert outside.read_text("utf-8") == "secret"

    # 1c. Bare command resolution ignores untrusted PATH entries and fails closed without bwrap.
    from olympus.stages import craft as craft_mod

    monkeypatch.setenv("PATH", f"{evil_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    resolved_argv, resolve_err = craft_mod._parse_sandboxed_repro_argv(
        "pytest -q",
        host_root=repo,
        sandbox_root=tmp_path / "sandboxes",
    )
    assert resolve_err is None and resolved_argv is not None
    assert str(evil_dir) not in resolved_argv[0]

    monkeypatch.setattr(craft_mod, "_probe_bwrap", lambda: None)
    fail_closed_stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([mutating_repro]),
        allow_unsandboxed_repro=False,
    )
    with pytest.raises(CraftPause, match="bwrap"):
        fail_closed_stage.explore_reproduce(
            run_id=run_id_repro,
            raw_input="Fail closed without bwrap",
            require_repro=True,
        )
    monkeypatch.undo()

    # 2. Promotion is atomic: a valid file edit paired with an escaping symlink copies NOTHING!
    run_id_prom = store.create_run(
        title="Promotion security",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=repo,
    )
    sketch_escape = ArchitectSketch(
        candidate_shapes=[
            CandidateShape(name="a", data_flow="x", tradeoffs="y"),
            CandidateShape(name="b", data_flow="z", tradeoffs="w"),
        ],
        chosen_shape="a",
        rationale="test",
        write_scope=["app.py", "old_link.txt", "escape_link.txt"],
        verification_plan=[f"{sys.executable} -c \"print('ok')\""],
    )
    prom_stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=MockLLMProvider([]),
    )
    handle, baseline = prom_stage._ensure_sandbox(run_id_prom)
    # Modify app.py (alphabetically BEFORE escape_link.txt) AND add an escaping symlink!
    (handle.workspace_path / "app.py").write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n", "utf-8"
    )
    (handle.workspace_path / "escape_link.txt").symlink_to(outside)
    with pytest.raises(CraftPause, match="symlink escaping workspace root"):
        prom_stage._promote_sandbox_to_workspace(handle, baseline, sketch_escape)
    assert not (repo / "escape_link.txt").exists()
    # Atomic pre-validation guarantee: app.py on host must NOT have been partially copied!
    assert "return a - b" in (repo / "app.py").read_text("utf-8")

    # Replacing a tracked file in scope with a directory must also reject atomically!
    (handle.workspace_path / "escape_link.txt").unlink()
    (handle.workspace_path / "old_link.txt").unlink()
    (handle.workspace_path / "old_link.txt").mkdir()
    (handle.workspace_path / "old_link.txt" / "nested.txt").write_text("dir-file", "utf-8")
    sketch_dir_replace = ArchitectSketch(
        candidate_shapes=[
            CandidateShape(name="a", data_flow="x", tradeoffs="y"),
            CandidateShape(name="b", data_flow="z", tradeoffs="w"),
        ],
        chosen_shape="a",
        rationale="test",
        write_scope=["app.py", "old_link.txt", "old_link.txt/**"],
        verification_plan=[f"{sys.executable} -m unittest test_app.py"],
    )
    with pytest.raises(CraftPause, match="non-directory|non-regular"):
        prom_stage._promote_sandbox_to_workspace(handle, baseline, sketch_dir_replace)
    assert "return a - b" in (repo / "app.py").read_text("utf-8")
    shutil.rmtree(handle.workspace_path / "old_link.txt")
    (handle.workspace_path / "old_link.txt").symlink_to(outside)

    # Add out-of-scope file -> strict DiffGate must reject!
    (handle.workspace_path / "out_of_scope.py").write_text("x = 1\n", "utf-8")
    with pytest.raises(CraftPause, match="Diff Gate rejected"):
        prom_stage._promote_sandbox_to_workspace(handle, baseline, sketch_escape)
    assert not (repo / "out_of_scope.py").exists()

    # Remove out-of-scope file, delete old_link.txt (external symlink) -> must unlink on host!
    (handle.workspace_path / "out_of_scope.py").unlink()
    (handle.workspace_path / "old_link.txt").unlink()
    applied = prom_stage._promote_sandbox_to_workspace(handle, baseline, sketch_escape)
    assert "old_link.txt" in applied
    assert not (repo / "old_link.txt").exists()
    assert not (repo / "old_link.txt").is_symlink()
    assert outside.exists()

    # 3. Rollback preserves executable mode bits (0o755) and survives a secondary PermissionError.
    rb_repo = tmp_path / "rb_repo"
    _init_demo_repo(rb_repo)
    exec_script = rb_repo / "run.sh"
    exec_script.write_text("#!/bin/sh\necho v1\n", "utf-8")
    exec_script.chmod(0o755)
    gone_file = rb_repo / "keep_me.txt"
    gone_file.write_text("important-data", "utf-8")
    second_file = rb_repo / "second.py"
    second_file.write_text("x = 1\n", "utf-8")
    subprocess.run(["git", "add", "."], cwd=rb_repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "add rb fixtures"],
        cwd=rb_repo,
        check=True,
        capture_output=True,
    )

    rb_run = store.create_run(
        title="Rollback mode and resilience",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=rb_repo,
    )
    rb_stage = CraftStage(
        store=store,
        workspace_root=rb_repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=MockLLMProvider([]),
    )
    rb_handle, rb_base = rb_stage._ensure_sandbox(rb_run)
    (rb_handle.workspace_path / "keep_me.txt").unlink()
    (rb_handle.workspace_path / "run.sh").write_text("#!/bin/sh\necho v2\n", "utf-8")
    (rb_handle.workspace_path / "second.py").write_text("x = 2\n", "utf-8")
    rb_sketch = ArchitectSketch(
        candidate_shapes=[
            CandidateShape(name="a", data_flow="x", tradeoffs="y"),
            CandidateShape(name="b", data_flow="z", tradeoffs="w"),
        ],
        chosen_shape="a",
        rationale="test rollback",
        write_scope=["keep_me.txt", "run.sh", "second.py"],
        verification_plan=[f"{sys.executable} -m unittest test_app.py"],
    )
    real_copy2 = shutil.copy2
    real_write_bytes = Path.write_bytes

    def _fail_on_second(src: Path, dst: Path) -> Any:
        if Path(dst).name == "second.py":
            raise PermissionError("simulated write failure on second.py")
        return real_copy2(src, dst)

    def _fail_rb_on_second(self_path: Path, data: bytes) -> int:
        if self_path.name == "second.py":
            raise PermissionError("simulated rollback failure on second.py")
        return real_write_bytes(self_path, data)

    monkeypatch.setattr(shutil, "copy2", _fail_on_second)
    monkeypatch.setattr(Path, "write_bytes", _fail_rb_on_second)
    with pytest.raises(CraftPause, match="rolled back"):
        rb_stage._promote_sandbox_to_workspace(rb_handle, rb_base, rb_sketch)
    monkeypatch.undo()
    # Previously deleted file must still be restored, untouched second.py preserved, and 0o755 kept!
    assert gone_file.read_text("utf-8") == "important-data"
    assert second_file.read_text("utf-8") == "x = 1\n"
    assert "echo v1" in exec_script.read_text("utf-8")
    assert stat.S_IMODE(exec_script.stat().st_mode) == 0o755

    # 4. Replacing an existing regular file with an in-workspace symlink promotes cleanly!
    (rb_handle.workspace_path / "second.py").unlink()
    (rb_handle.workspace_path / "second.py").symlink_to("app.py")
    (rb_handle.workspace_path / "keep_me.txt").write_text("important-data", "utf-8")
    (rb_handle.workspace_path / "run.sh").write_text("#!/bin/sh\necho v1\n", "utf-8")
    applied_sym = rb_stage._promote_sandbox_to_workspace(rb_handle, rb_base, rb_sketch)
    assert "second.py" in applied_sym
    assert (rb_repo / "second.py").is_symlink()
    assert os.readlink(rb_repo / "second.py") == "app.py"


def test_pstack_craft_cli_web_clarify_plan_only_and_blocked_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from olympus.session.contracts import PauseKind, SessionStatus, StepMode
    from olympus.session.runner import SessionConfig, SessionRunner, plan_only_definition
    from olympus.web.server import AriadneApp

    # 1. plan_only_definition strips all mutating craft nodes.
    plan_only = plan_only_definition(load_workflow("pstack_craft"))
    assert [n.name for n in plan_only.nodes] == ["explore", "architect"]

    # 2. CLI clarify on non-GoalSpec pstack_craft records clarification and advances on reply.
    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    state_dir = tmp_path / "state"

    def _capture_driver(req: DriverTurnRequest) -> str:
        if req.actor == "hephaestus:build":
            (req.workspace_path / "app.py").write_text(
                "def add(a: int, b: int) -> int:\n    return a + b\n", "utf-8"
            )
            return "built"
        if req.actor == "hephaestus:hygiene":
            return "clean"
        raise AssertionError(req.actor)

    responses = [
        _sample_evidence(unresolved=["Support ints only?"]),
        # reply() router classification -> clarify
        '{"action": "clarify", "answers": {}, "resolutions": {}, "feedback": "Ints only"}',
        # explore re-run sees recorded clarification -> ready
        _sample_evidence(unresolved=[]),
        _sample_sketch(),
        InterrogateReport(verdict="pass", findings=[]).model_dump_json(),
    ]

    shared_provider = _mock_provider(responses)
    scripted = _ScriptedDriver(_capture_driver)
    monkeypatch.setattr(
        "olympus.session.runner.DriverRegistry",
        lambda *a, **kw: DriverRegistry({"scripted": scripted}),
    )
    runner = SessionRunner(
        SessionConfig(
            workspace_root=repo,
            state_dir=state_dir,
            workflow="pstack_craft",
            step_mode=StepMode.EXECUTE,
            provider="mock",
            stage_driver="native",
            worker_driver="scripted",
            reviewer_driver="native",
            provider_builder=lambda **_kw: shared_provider,
        )
    )

    start_env = runner.start(goal="Fix app.add")
    assert start_env.status == SessionStatus.NEEDS_INPUT, start_env.error
    assert start_env.pause_kind == PauseKind.CLARIFY
    assert start_env.run_id is not None

    reply_env = runner.reply(run_id=start_env.run_id, message="Ints only, please")
    assert reply_env.status == SessionStatus.COMPLETED, reply_env.error
    explore_prompts = [
        "\n".join(str(m.content) for m in msgs)
        for msgs, _tools in shared_provider.recorded_requests
        if any("Prometheus (Evidence-First Explorer)" in str(m.content) for m in msgs)
    ]
    assert len(explore_prompts) == 2
    assert "Ints only" in explore_prompts[1]

    # 3. Web clarify without GoalSpec enforces revision CAS and operation_id idempotency.
    from olympus.core.store import ConcurrencyConflictError

    web_store = RunStore(tmp_path / "web.sqlite")
    web_run_id = web_store.create_run(
        title="Web craft clarify",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=repo,
    )
    app = AriadneApp(store=web_store, workspace_root=repo)
    initial_rev = int(web_store.get_run(web_run_id)["state_revision"])

    with pytest.raises(ConcurrencyConflictError):
        app.execute_command(
            web_run_id,
            {
                "action": "clarify",
                "clarification_note": "Stale revision",
                "expected_state_revision": initial_rev + 99,
                "auto_advance": False,
            },
        )

    cmd_res = app.execute_command(
        web_run_id,
        {
            "action": "clarify",
            "clarification_note": "Keep integer signature only",
            "expected_state_revision": initial_rev,
            "operation_id": "op-craft-clarify-1",
            "auto_advance": False,
        },
    )
    assert cmd_res["ok"] is True
    # Replaying the exact same operation_id must deduplicate without writing a second event.
    replay_res = app.execute_command(
        web_run_id,
        {
            "action": "clarify",
            "clarification_note": "Keep integer signature only",
            "expected_state_revision": initial_rev,
            "operation_id": "op-craft-clarify-1",
            "auto_advance": False,
        },
    )
    assert replay_res["ok"] is True
    assert replay_res["command_result"].get("deduplicated") is True
    clarify_events = [
        ev
        for ev in web_store.list_events(web_run_id)
        if ev.event_type == "craft.clarification_recorded"
    ]
    assert len(clarify_events) == 1

    # 4. Blocked craft recovery resumes at build (not explore) and resets exhausted review_cycle.
    rec_store = runner.store
    rec_run = rec_store.create_run(
        title="Blocked craft recovery",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=repo,
    )
    rec_store.append_event(
        run_id=rec_run,
        stage="craft",
        event_type="craft.architect_recorded",
        actor="athena",
        payload={"sketch_id": "arch_1"},
    )
    rec_store.append_event(
        run_id=rec_run,
        stage="craft",
        event_type="craft.interrogate_recorded",
        actor="momus",
        payload={"cycle": 3, "verdict": "blocked"},
    )
    rec_store.append_event(
        run_id=rec_run,
        stage="session",
        event_type="session.paused",
        actor="session",
        payload={"pause_kind": "blocked", "node": "interrogate", "reason": "exhausted cycles"},
    )
    started_entries: list[str | None] = []
    orig_drive = runner._drive

    def _spy_drive(*, start_from: str | None, **kwargs: Any) -> Any:
        started_entries.append(start_from)
        return orig_drive(start_from=start_from, **kwargs)

    # Verify _recover_blocked_craft emits recovery_reset and routes to "build", not "explore".
    monkeypatch_stage = CraftStage(
        store=rec_store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        provider=MockLLMProvider([]),
    )
    assert monkeypatch_stage._active_review_cycle(rec_run, {}) == 3

    runner._drive = lambda **kw: (  # type: ignore[method-assign]
        started_entries.append(kw.get("start_from")),
        runner._completed_envelope(rec_run, rec_store.get_run(rec_run)),
    )[1]
    rec_env = runner._recover_blocked(
        rec_run, load_workflow("pstack_craft"), rec_store.get_run(rec_run), "Fix off-by-one"
    )
    assert rec_env.status == SessionStatus.COMPLETED
    assert started_entries == ["build"]
    assert monkeypatch_stage._active_review_cycle(rec_run, {}) == 0


def test_craft_assumptions_non_blocking_and_architect_verification_plan_repair(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Assumptions & plan repair",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=repo,
    )

    # 1. EvidenceBrief.assumptions must NOT pause explore_reproduce (unlike unresolved_questions).
    brief_with_assumptions = EvidenceBrief(
        summary="Ready with sensible defaults",
        assumptions=["Preserve existing KeyError on missing key", "Preserve insertion order"],
        unresolved_questions=[],
        candidate_files=["app.py"],
    ).model_dump_json()
    stage_explore = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([brief_with_assumptions]),
    )
    explore_out = stage_explore.explore_reproduce(run_id=run_id, raw_input="Fix app.py")
    assert explore_out["route"] == "ready"
    assert len(explore_out["evidence_brief"]["assumptions"]) == 2

    # 2. architect_sketch repairs invalid cd/python -c/grep verification_plan before build starts,
    #    preserving original valid commands and keeping write_scope/forbidden_globs immutable.
    bad_sketch = ArchitectSketch(
        candidate_shapes=[
            CandidateShape(name="A", data_flow="direct", tradeoffs="simple"),
            CandidateShape(name="B", data_flow="wrapper", tradeoffs="indirection"),
        ],
        chosen_shape="A",
        rationale="Direct is cleaner",
        write_scope=["app.py"],
        forbidden_globs=["secrets/*"],
        verification_plan=[
            f"{sys.executable} -m unittest test_app.py",
            f"cd {repo} && python3 -m unittest test_app.py",
            "python3 -c 'import app; assert app.add(1, 2) == 3'",
            "grep -n 'def add' app.py",
            "pytest --version",
        ],
    ).model_dump_json()
    repaired_sketch = ArchitectSketch(
        candidate_shapes=[
            CandidateShape(name="A", data_flow="direct", tradeoffs="simple"),
            CandidateShape(name="B", data_flow="wrapper", tradeoffs="indirection"),
        ],
        chosen_shape="B",
        rationale="Attempted scope expansion",
        write_scope=["app.py", "other.py"],
        forbidden_globs=[],
        verification_plan=[f"{sys.executable} -m unittest -v test_app.py"],
    ).model_dump_json()
    stage_arch = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([bad_sketch, repaired_sketch]),
    )
    arch_out = stage_arch.architect_sketch(run_id=run_id, raw_input="Fix app.py", state={})
    assert arch_out["route"] == "ready"
    assert arch_out["architect_sketch"]["chosen_shape"] == "A"
    assert arch_out["architect_sketch"]["write_scope"] == ["app.py"]
    assert arch_out["architect_sketch"]["forbidden_globs"] == ["secrets/*"]
    assert arch_out["architect_sketch"]["verification_plan"] == [
        f"{sys.executable} -m unittest test_app.py",
        f"{sys.executable} -m unittest -v test_app.py",
    ]
    rehydrated_sketch = ArchitectSketch.model_validate(arch_out["architect_sketch"])
    assert rehydrated_sketch.content_hash == arch_out["architect_sketch"]["content_hash"]

    # 3. If even repair yields zero allowlisted commands, architect_sketch pauses immediately.
    all_bad_sketch = ArchitectSketch(
        candidate_shapes=[
            CandidateShape(name="A", data_flow="direct", tradeoffs="simple"),
            CandidateShape(name="B", data_flow="wrapper", tradeoffs="indirection"),
        ],
        chosen_shape="A",
        rationale="Direct is cleaner",
        write_scope=["app.py"],
        verification_plan=[
            "pytest --version",
            "pytest -V",
            "pytest --funcargs",
            "pytest --cache-show",
            "pytest --collect-only",
            f"{sys.executable} -m unittest --help",
        ],
    ).model_dump_json()
    stage_arch_bad = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([all_bad_sketch, all_bad_sketch]),
    )
    with pytest.raises(CraftPause, match="no allowlisted read-only commands"):
        stage_arch_bad.architect_sketch(run_id=run_id, raw_input="Fix app.py", state={})

    # pytest --setup-show executes tests and must NOT be blocked by the allowlist.
    valid_setup_show, rejected_setup_show = stage_arch._partition_verification_plan(
        ["pytest --setup-show"]
    )
    assert valid_setup_show == ["pytest --setup-show"]
    assert rejected_setup_show == []

    # 4. Empty EvidenceBrief.assumptions preserves legacy content_hash (Pydantic 2.9+ safe).
    b_empty = EvidenceBrief(summary="Same", unresolved_questions=[], candidate_files=["app.py"])
    assert "assumptions" not in b_empty._hash_payload()
    b_with = EvidenceBrief(
        summary="Same",
        assumptions=["default"],
        unresolved_questions=[],
        candidate_files=["app.py"],
    )
    assert b_with.content_hash != b_empty.content_hash

    # 5. External driver respects max_llm_calls=1 and never executes an over-budget repair turn.
    from olympus.core.contracts import BudgetPolicy
    from olympus.harness.driver import (
        AgentDriver,
        DriverRegistry,
        ExecutionDriversConfig,
        TurnOutcome,
    )

    class _CountingDriver(AgentDriver):
        name = "counting_ext"

        def __init__(self, payloads: list[str]) -> None:
            self.payloads = list(payloads)
            self.calls = 0

        def run_turn(self, request: Any, on_event: Any = None) -> TurnOutcome:
            self.calls += 1
            text = self.payloads.pop(0)
            return TurnOutcome(
                completed=True,
                stop_reason="completed",
                final_content=text,
                steps_taken=1,
                total_input_tokens=10,
                total_output_tokens=10,
            )

    capped_run = store.create_run(
        title="Budget capped external repair",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=repo,
        budget_policy=BudgetPolicy(max_llm_calls=1),
    )
    ext_driver = _CountingDriver([bad_sketch, repaired_sketch])
    registry = DriverRegistry({"counting_ext": ext_driver})
    stage_capped = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([]),
        driver_registry=registry,
        execution_drivers=ExecutionDriversConfig(role_drivers={"architect": "counting_ext"}),
    )
    capped_out = stage_capped.architect_sketch(run_id=capped_run, raw_input="Fix app.py", state={})
    assert capped_out["route"] == "ready"
    assert ext_driver.calls == 1
    assert store.get_run(capped_run)["budget_ledger"].llm_calls_used == 1
    assert capped_out["architect_sketch"]["verification_plan"] == [
        f"{sys.executable} -m unittest test_app.py"
    ]

    # 5b. External driver repair parse-retry must respect max_llm_calls=2 (never calls 3rd time).
    capped_retry_run = store.create_run(
        title="Budget capped external repair retry",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=repo,
        budget_policy=BudgetPolicy(max_llm_calls=2),
    )
    retry_driver = _CountingDriver([bad_sketch, "not valid json", repaired_sketch])
    stage_retry_capped = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([]),
        driver_registry=DriverRegistry({"counting_ext": retry_driver}),
        execution_drivers=ExecutionDriversConfig(role_drivers={"architect": "counting_ext"}),
    )
    retry_out = stage_retry_capped.architect_sketch(
        run_id=capped_retry_run, raw_input="Fix app.py", state={}
    )
    assert retry_out["route"] == "ready"
    assert retry_driver.calls == 2
    assert store.get_run(capped_retry_run)["budget_ledger"].llm_calls_used == 2

    # 5b2. Drivers with records_usage_internally=True (like PiDriver) also obey pre-turn budget.
    class _SelfAccountingDriver(AgentDriver):
        name = "self_accounting_ext"
        records_usage_internally = True

        def __init__(self, target_run_id: str, payloads: list[str]) -> None:
            self.target_run_id = target_run_id
            self.payloads = list(payloads)
            self.calls = 0

        def run_turn(self, request: Any, on_event: Any = None) -> TurnOutcome:
            self.calls += 1
            store.record_external_llm_usage(
                self.target_run_id,
                actor="self_accounting",
                purpose=f"turn_{self.calls}",
                input_tokens=15,
                output_tokens=15,
            )
            text = self.payloads.pop(0)
            return TurnOutcome(
                completed=True,
                stop_reason="completed",
                final_content=text,
                steps_taken=1,
                total_input_tokens=15,
                total_output_tokens=15,
            )

    pi_like_run = store.create_run(
        title="Budget capped self-accounting repair retry",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=repo,
        budget_policy=BudgetPolicy(max_llm_calls=2, max_total_tokens=60),
    )
    pi_like_driver = _SelfAccountingDriver(
        pi_like_run, [bad_sketch, "not valid json", repaired_sketch]
    )
    stage_pi_capped = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([]),
        driver_registry=DriverRegistry({"self_accounting_ext": pi_like_driver}),
        execution_drivers=ExecutionDriversConfig(role_drivers={"architect": "self_accounting_ext"}),
    )
    pi_out = stage_pi_capped.architect_sketch(
        run_id=pi_like_run, raw_input="Fix app.py", state={}
    )
    assert pi_out["route"] == "ready"
    assert pi_like_driver.calls == 2
    assert store.get_run(pi_like_run)["budget_ledger"].llm_calls_used == 2
    assert store.get_run(pi_like_run)["budget_ledger"].total_tokens == 60

    # 5c. Loader step budget aligns SessionRunner default worker_max_steps=40 to build_direct=24.
    from olympus.graph.loader import WorkflowRuntime, _craft_step_budget

    session_like_rt = WorkflowRuntime(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        provider=_mock_provider([]),
        worker_max_steps=40,
    )
    assert _craft_step_budget({}, session_like_rt, 24, use_worker_steps=True) == 24

    # 6. Zero-test unittest execution (exit 0 + "Ran 0 tests") must fail verify_surface.
    zero_run = store.create_run(
        title="Zero-test unittest fail",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=repo,
    )
    zero_output = "-----------------------------------\nRan 0 tests in 0.000s\n\nOK\n"
    stage_zero = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([]),
        command_runner=lambda cmd, cwd, timeout_s: (0, zero_output),
    )
    zero_sketch = ArchitectSketch(
        candidate_shapes=[
            CandidateShape(name="A", data_flow="direct", tradeoffs="simple"),
            CandidateShape(name="B", data_flow="wrapper", tradeoffs="indirection"),
        ],
        chosen_shape="A",
        rationale="Direct",
        write_scope=["app.py"],
        verification_plan=[f"{sys.executable} -m unittest test_app.py"],
    )
    with pytest.raises(CraftPause, match="check_1.*fail"):
        stage_zero.verify_surface(
            run_id=zero_run,
            state={"architect_sketch": zero_sketch.model_dump(mode="json")},
        )
