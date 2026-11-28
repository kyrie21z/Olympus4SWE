"""Offline contract, repro-gate, failing-test-first, read-only forensics, and manifest tests
for Issue #190 (`pstack_bugfix`, `investigate`, and non-overlapping builtin manifests).
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from olympus.core.contracts import (
    ArchitectSketch,
    CandidateShape,
    EvidenceBrief,
    EvidenceFact,
    InterrogateReport,
    RunMode,
    RunStatus,
)
from olympus.core.store import RunStore
from olympus.graph import (
    WorkflowRuntime,
    build_graph,
    describe_workflow,
    list_builtin_workflows,
    load_workflow,
    run_graph,
)
from olympus.graph.progress import classify_run_board
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
from olympus.session.runner import SessionConfig, SessionRunner
from olympus.stages.craft import (
    CraftPause,
    CraftStage,
    _is_proven_failure_exit,
    _is_proven_green_exit,
    _probe_bwrap,
    load_skill_pack,
)
from olympus.web.schemas import WorkflowSummary


@pytest.fixture(autouse=True)
def _ci_bwrap_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep local Linux tests running under real bwrap; opt in only when CI lacks bwrap."""
    if _probe_bwrap() is None:
        monkeypatch.setenv("OLYMPUS_ALLOW_UNSANDBOXED_REPRO", "1")


def _mock_provider(payloads: list[str]) -> MockLLMProvider:
    return MockLLMProvider([LLMResponse(content=item) for item in payloads])


class _ScriptedDriver:
    """Deterministic test AgentDriver executing scripted filesystem actions."""

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


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _init_buggy_repo(repo: Path, *, with_failing_test: bool = True) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "pricing.py").write_text(
        "def apply_discount(price: int, percent: int) -> int:\n"
        "    # Off-by-one bug: adds discount instead of subtracting it.\n"
        "    return price + (price * percent // 100)\n",
        "utf-8",
    )
    if with_failing_test:
        test_body = (
            "import unittest\n"
            "import pricing\n\n"
            "class TestPricing(unittest.TestCase):\n"
            "    def test_discount_reduces_price(self) -> None:\n"
            "        self.assertEqual(pricing.apply_discount(100, 20), 80)\n"
        )
    else:
        test_body = (
            "import unittest\n"
            "import pricing\n\n"
            "class TestPricing(unittest.TestCase):\n"
            "    def test_zero_discount(self) -> None:\n"
            "        self.assertEqual(pricing.apply_discount(100, 0), 100)\n"
        )
    (repo / "test_pricing.py").write_text(test_body, "utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")


def _sample_bug_sketch(
    *,
    write_scope: list[str] | None = None,
    verification_plan: list[str] | None = None,
) -> str:
    return ArchitectSketch(
        candidate_shapes=[
            CandidateShape(
                name="direct_subtraction",
                data_flow="pricing.apply_discount subtracts floor(price * percent / 100)",
                tradeoffs="Minimal, exact fix at root cause.",
            ),
            CandidateShape(
                name="post_clamp_wrapper",
                data_flow="Wrap apply_discount with a secondary clamping helper",
                tradeoffs="Masks symptom and adds unnecessary indirection.",
            ),
        ],
        chosen_shape="direct_subtraction",
        rationale="Fixes the sign error directly at pricing.py:3 with regression coverage.",
        state_owners={"pricing.apply_discount": "pricing.py"},
        write_scope=write_scope or ["pricing.py", "test_pricing.py"],
        verification_plan=verification_plan
        or [f"{sys.executable} -m unittest test_pricing.py"],
    ).model_dump_json()


def test_pstack_bugfix_and_investigate_manifests_and_skills_validate() -> None:
    builtin = set(list_builtin_workflows())
    assert {"pstack_bugfix", "investigate"} <= builtin

    bugfix_def = load_workflow("pstack_bugfix")
    bugfix_summary = WorkflowSummary.model_validate(describe_workflow(bugfix_def))
    assert bugfix_summary.family == "pstack"
    assert bugfix_summary.autonomy == "autonomous_hard_stops"
    assert bugfix_summary.stop_after == "verify"
    assert bugfix_summary.applicable_scenarios
    assert bugfix_summary.anti_patterns

    by_name = {node.name: node for node in bugfix_summary.graph.nodes}
    assert by_name["explore"].god == "prometheus"
    assert by_name["architect"].god == "athena"
    assert by_name["build"].god == "hephaestus"
    assert by_name["hygiene"].god == "hephaestus"
    assert by_name["interrogate"].god == "momus"
    assert by_name["verify_surface"].god == "argus"
    assert bugfix_def.node("explore") is not None
    assert bugfix_def.node("explore").params["require_repro"] is True  # type: ignore[union-attr]
    assert bugfix_def.node("build") is not None
    assert bugfix_def.node("build").params["failing_test_first"] is True  # type: ignore[union-attr]
    assert bugfix_def.node("interrogate") is not None
    assert bugfix_def.node("interrogate").max_visits == 3  # type: ignore[union-attr]
    for node in bugfix_def.nodes:
        assert load_skill_pack(node.skill_refs).strip()

    inv_def = load_workflow("investigate")
    inv_summary = WorkflowSummary.model_validate(describe_workflow(inv_def))
    assert inv_summary.family == "pstack"
    assert inv_summary.autonomy == "autonomous_hard_stops"
    assert inv_summary.stop_after == "report"
    assert [node.name for node in inv_def.nodes] == ["explore"]
    assert inv_def.nodes[0].god == "prometheus"
    assert inv_def.nodes[0].params["require_repro"] is False
    assert inv_def.nodes[0].params["read_only_report"] is True
    assert load_skill_pack(inv_def.nodes[0].skill_refs).strip()


def test_all_six_builtin_workflows_have_complete_non_overlapping_manifests() -> None:
    expected = {
        "full_sdlc_tdd": ("sdlc", "gated", "execute"),
        "product_blueprint": ("sdlc", "gated", "plan"),
        "repo_fast_patch": ("minimal", "autonomous_hard_stops", "execute"),
        "pstack_craft": ("pstack", "autonomous_hard_stops", "verify"),
        "pstack_arena": ("pstack", "autonomous_hard_stops", "verify"),
        "pstack_bugfix": ("pstack", "autonomous_hard_stops", "verify"),
        "investigate": ("pstack", "autonomous_hard_stops", "report"),
        "pstack_hillclimb": ("pstack", "autonomous_hard_stops", "verify"),
    }
    assert set(list_builtin_workflows()) == set(expected)

    seen_scenarios: dict[str, str] = {}
    for name, (family, autonomy, stop_after) in expected.items():
        definition = load_workflow(name)
        summary = WorkflowSummary.model_validate(describe_workflow(definition))
        assert summary.family == family
        assert summary.autonomy == autonomy
        assert summary.stop_after == stop_after
        assert summary.applicable_scenarios, f"{name} missing applicable_scenarios"
        assert summary.anti_patterns, f"{name} missing anti_patterns"
        for scenario in summary.applicable_scenarios:
            normalized = " ".join(scenario.lower().split())
            assert normalized not in seen_scenarios, (
                f"Duplicate applicable_scenario between {seen_scenarios[normalized]} and {name}: "
                f"{scenario!r}"
            )
            seen_scenarios[normalized] = name

    fast_patch = load_workflow("repo_fast_patch")
    assert any("known-cause" in s.lower() for s in fast_patch.applicable_scenarios)
    assert any("unknown-cause" in s.lower() for s in fast_patch.anti_patterns)

    bugfix = load_workflow("pstack_bugfix")
    assert any("unknown-cause" in s.lower() for s in bugfix.applicable_scenarios)
    assert any("known-cause" in s.lower() for s in bugfix.anti_patterns)


def test_pstack_bugfix_blocks_when_repro_does_not_fail_on_base_ref(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    # Base repo test passes (exit 0), so repro_command does not prove a failure on base_ref.
    _init_buggy_repo(repo, with_failing_test=False)
    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Unreproduced bug",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_bugfix",
        workspace_root=repo,
    )

    builder_called = False

    def _never_called(_req: DriverTurnRequest) -> str:
        nonlocal builder_called
        builder_called = True
        return "should not run"

    passing_brief = EvidenceBrief(
        summary="Claimed bug without failing reproduction",
        facts=[EvidenceFact(path="pricing.py", line=3, observation="adds discount")],
        repro_command=f"{sys.executable} -m unittest test_pricing.py",
        repro_verified=True,  # LLM attempt to spoof verification must be ignored.
        candidate_files=["pricing.py"],
    ).model_dump_json()

    runtime = WorkflowRuntime(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([passing_brief]),
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="scripted",
            reviewer_driver="native",
        ),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_never_called)}),
    )

    result = run_graph(
        build_graph(load_workflow("pstack_bugfix")),
        {
            "run_id": run_id,
            "raw_input": "Fix pricing discount bug",
            "base_ref": "HEAD",
            "_runtime": runtime,
        },
    )

    assert result["_paused_node"] == "explore"
    assert result["pause_kind"] == "blocked"
    assert "did not prove failure" in result["_pause_reason"]
    assert result["evidence_brief"]["repro_verified"] is False
    assert result["evidence_brief"]["repro_exit_code"] == 0
    assert result["repro_receipt"]["verified"] is False
    assert not builder_called
    assert "architect_sketch" not in result


def test_pstack_bugfix_blocks_unsafe_repro_command_preserving_allowlist(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_buggy_repo(repo, with_failing_test=True)
    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Unsafe repro bugfix",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_bugfix",
        workspace_root=repo,
    )

    unsafe_brief = EvidenceBrief(
        summary="Tries shell metacharacter injection in repro_command",
        facts=[EvidenceFact(path="pricing.py", line=3, observation="adds discount")],
        repro_command="pytest test_pricing.py && touch pwned.txt",
        candidate_files=["pricing.py"],
    ).model_dump_json()

    runtime = WorkflowRuntime(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([unsafe_brief]),
    )

    result = run_graph(
        build_graph(load_workflow("pstack_bugfix")),
        {"run_id": run_id, "raw_input": "Fix pricing bug", "_runtime": runtime},
    )

    assert result["_paused_node"] == "explore"
    assert result["pause_kind"] == "blocked"
    assert "Unsafe repro_command rejected" in result["_pause_reason"]
    assert not (repo / "pwned.txt").exists()


def test_pstack_bugfix_repro_checks_explicit_base_ref(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_buggy_repo(repo, with_failing_test=True)
    buggy_sha = _git(repo, "rev-parse", "HEAD")

    # Commit a fix on HEAD so HEAD passes while buggy_sha still fails!
    (repo / "pricing.py").write_text(
        "def apply_discount(price: int, percent: int) -> int:\n"
        "    return price - (price * percent // 100)\n",
        "utf-8",
    )
    _git(repo, "add", "pricing.py")
    _git(repo, "commit", "-m", "fix on HEAD")

    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Explicit base_ref repro",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_bugfix",
        workspace_root=repo,
    )
    brief_json = EvidenceBrief(
        summary="Bug fails at historical base_ref",
        facts=[EvidenceFact(path="pricing.py", line=3, observation="adds instead of subtracts")],
        repro_command=f"{sys.executable} -m unittest test_pricing.py",
        candidate_files=["pricing.py"],
    ).model_dump_json()

    stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([brief_json]),
    )
    out = stage.explore_reproduce(
        run_id=run_id,
        raw_input="Reproduce bug at base_ref",
        base_ref=buggy_sha,
        require_repro=True,
    )
    assert out["evidence_brief"]["repro_verified"] is True
    assert out["repro_receipt"]["verified"] is True
    assert out["repro_receipt"]["base_ref"] == buggy_sha


def test_pstack_bugfix_e2e_red_test_first_to_green_fix_and_promotion(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_buggy_repo(repo, with_failing_test=True)
    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Fix pricing discount sign",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_bugfix",
        workspace_root=repo,
    )

    build_attempts = 0

    def _builder_action(req: DriverTurnRequest) -> str:
        nonlocal build_attempts
        if req.actor == "hephaestus:build":
            build_attempts += 1
            if build_attempts == 1:
                # First attempt still fails -> build_direct(failing_test_first=True) must retry!
                (req.workspace_path / "pricing.py").write_text(
                    "def apply_discount(price: int, percent: int) -> int:\n"
                    "    return price + 1\n",
                    "utf-8",
                )
                return "Attempt 1 (incomplete fix)"
            (req.workspace_path / "pricing.py").write_text(
                "# obvious comment\n"
                "def apply_discount(price: int, percent: int) -> int:\n"
                "    return price - (price * percent // 100)\n",
                "utf-8",
            )
            return "Fixed sign error at root cause."
        if req.actor == "hephaestus:hygiene":
            (req.workspace_path / "pricing.py").write_text(
                "def apply_discount(price: int, percent: int) -> int:\n"
                "    return price - (price * percent // 100)\n",
                "utf-8",
            )
            return "Stripped redundant comment."
        raise AssertionError(f"Unexpected actor: {req.actor}")

    driver = _ScriptedDriver(_builder_action)
    evidence_json = EvidenceBrief(
        summary="pricing.apply_discount adds instead of subtracts discount amount",
        facts=[EvidenceFact(path="pricing.py", line=3, observation="uses + instead of -")],
        repro_command=f"{sys.executable} -m unittest test_pricing.py",
        candidate_files=["pricing.py", "test_pricing.py"],
    ).model_dump_json()
    clean_review = InterrogateReport(
        verdict="pass",
        findings=[],
        summary="Verified root-cause fix cleanly turns test green.",
    ).model_dump_json()

    runtime = WorkflowRuntime(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([evidence_json, _sample_bug_sketch(), clean_review]),
        apply_to_workspace=True,
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="scripted",
            reviewer_driver="native",
        ),
        driver_registry=DriverRegistry({"scripted": driver}),
    )

    result = run_graph(
        build_graph(load_workflow("pstack_bugfix")),
        {"run_id": run_id, "raw_input": "Fix discount sign error", "_runtime": runtime},
    )

    assert result.get("_paused_node") is None
    assert build_attempts == 2
    assert result["evidence_brief"]["repro_verified"] is True
    assert result["failing_test_receipt"]["same_surface"] is True
    assert result["failing_test_receipt"]["red"]["verified"] is True
    assert result["failing_test_receipt"]["red"]["exit_code"] != 0
    assert result["failing_test_receipt"]["green"]["verified"] is True
    assert result["failing_test_receipt"]["green"]["exit_code"] == 0
    assert result["surface_verification"]["passed"] is True
    assert result["applied_paths"] == ["pricing.py"]

    events = [ev.event_type for ev in store.list_events(run_id)]
    assert "craft.red_test_recorded" in events
    assert "craft.failing_test_still_red" in events
    assert "craft.failing_test_verified" in events

    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "test_pricing.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_build_direct_failing_test_first_two_phase_red_test_creation(tmp_path: Path) -> None:
    """When base test suite passes, failing_test_first enforces a red-test turn before fixing."""
    repo = tmp_path / "repo"
    _init_buggy_repo(repo, with_failing_test=False)
    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Add regression test before fix",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_bugfix",
        workspace_root=repo,
    )

    actors_seen: list[str] = []

    def _two_phase_worker(req: DriverTurnRequest) -> str:
        actors_seen.append(req.actor)
        if req.actor == "hephaestus:red_test":
            (req.workspace_path / "test_pricing.py").write_text(
                "import unittest\n"
                "import pricing\n\n"
                "class TestPricing(unittest.TestCase):\n"
                "    def test_zero_discount(self) -> None:\n"
                "        self.assertEqual(pricing.apply_discount(100, 0), 100)\n\n"
                "    def test_discount_reduces_price(self) -> None:\n"
                "        self.assertEqual(pricing.apply_discount(100, 20), 80)\n",
                "utf-8",
            )
            return "Added failing regression test."
        if req.actor == "hephaestus:build":
            (req.workspace_path / "pricing.py").write_text(
                "def apply_discount(price: int, percent: int) -> int:\n"
                "    return price - (price * percent // 100)\n",
                "utf-8",
            )
            return "Applied root-cause fix."
        raise AssertionError(f"Unexpected actor: {req.actor}")

    stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([]),
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="scripted",
            reviewer_driver="native",
        ),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_two_phase_worker)}),
    )

    sketch = ArchitectSketch.model_validate_json(_sample_bug_sketch())
    out = stage.build_direct(
        run_id=run_id,
        raw_input="Fix discount bug with regression test first",
        state={"architect_sketch": sketch.model_dump(mode="json")},
        failing_test_first=True,
    )

    assert actors_seen == ["hephaestus:red_test", "hephaestus:build"]
    receipt = out["failing_test_receipt"]
    assert receipt["same_surface"] is True
    assert receipt["red"]["source"] == "red_test_turn"
    assert receipt["red"]["exit_code"] != 0
    assert receipt["green"]["exit_code"] == 0
    assert set(out["craft_changed_paths"]) == {"pricing.py", "test_pricing.py"}


def test_build_direct_failing_test_first_blocks_if_red_test_turn_does_not_fail(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_buggy_repo(repo, with_failing_test=False)
    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Missing red test",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_bugfix",
        workspace_root=repo,
    )

    def _noop_red_worker(_req: DriverTurnRequest) -> str:
        return "Did not write a failing test."

    stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([]),
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="scripted",
            reviewer_driver="native",
        ),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_noop_red_worker)}),
    )
    sketch = ArchitectSketch.model_validate_json(_sample_bug_sketch())
    with pytest.raises(CraftPause, match="requires a proven failing test"):
        stage.build_direct(
            run_id=run_id,
            raw_input="Fix discount bug",
            state={"architect_sketch": sketch.model_dump(mode="json")},
            failing_test_first=True,
        )


def test_investigate_workflow_zero_workspace_mutation_and_structured_report(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_buggy_repo(repo, with_failing_test=True)
    before_head = _git(repo, "rev-parse", "HEAD")
    before_files = sorted(
        str(p.relative_to(repo)) for p in repo.rglob("*") if ".git" not in p.parts
    )
    assert _git(repo, "status", "--porcelain") == ""

    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Why does apply_discount overcharge?",
        mode=RunMode.EXECUTION,
        workflow_name="investigate",
        workspace_root=repo,
    )

    forensics_brief = EvidenceBrief(
        summary=(
            "pricing.apply_discount at pricing.py:3 uses '+' instead of '-', adding the "
            "computed discount amount to price rather than subtracting it."
        ),
        facts=[
            EvidenceFact(
                path="pricing.py",
                line=3,
                observation="returns `price + (price * percent // 100)` instead of subtracting",
            ),
            EvidenceFact(
                path="test_pricing.py",
                line=6,
                observation="asserts apply_discount(100, 20) == 80, which fails with 120",
            ),
        ],
        repro_command=f"{sys.executable} -m unittest test_pricing.py",
        candidate_files=["pricing.py", "test_pricing.py"],
        unresolved_questions=["Should negative percent values raise ValueError?"],
    ).model_dump_json()

    artifacts_root = tmp_path / "artifacts"
    runtime = WorkflowRuntime(
        store=store,
        workspace_root=repo,
        artifacts_root=artifacts_root,
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([forensics_brief]),
        apply_to_workspace=True,  # Must still never mutate workspace in investigate!
    )

    result = run_graph(
        build_graph(load_workflow("investigate")),
        {
            "run_id": run_id,
            "raw_input": "Why does apply_discount return 120 instead of 80?",
            "_runtime": runtime,
        },
    )

    assert result.get("_paused_node") is None
    assert result["route"] == "report"
    assert result["read_only"] is True
    assert result["evidence_brief"]["summary"].startswith("pricing.apply_discount")
    assert len(result["evidence_brief"]["facts"]) == 2
    assert "# Investigation Report" in result["investigation_report"]
    assert "`pricing.py:3`" in result["investigation_report"]
    assert "## Open Questions" in result["investigation_report"]

    report_file = artifacts_root / run_id / "craft" / "investigation_report.md"
    assert report_file.is_file()
    assert report_file.read_text(encoding="utf-8") == result["investigation_report"]

    events = [ev.event_type for ev in store.list_events(run_id)]
    assert "craft.evidence_recorded" in events
    assert "craft.investigation_report_recorded" in events

    # Strict zero workspace mutation invariant.
    assert _git(repo, "status", "--porcelain") == ""
    assert _git(repo, "rev-parse", "HEAD") == before_head
    after_files = sorted(
        str(p.relative_to(repo)) for p in repo.rglob("*") if ".git" not in p.parts
    )
    assert after_files == before_files
    assert not (tmp_path / "sandboxes").exists()

    # Board classification handles stop_after="report" cleanly.
    store.set_run_status(
        run_id,
        RunStatus.COMPLETED,
        expected_revision=int(store.get_run(run_id)["state_revision"]),
    )
    board = classify_run_board(
        store.get_run(run_id),
        store.list_events(run_id),
        store=store,
    )
    assert board.ball_holder == "done"


def test_architect_sketch_normalizes_in_workspace_absolute_write_scope(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_buggy_repo(repo, with_failing_test=True)
    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Normalize write_scope",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_bugfix",
        workspace_root=repo,
    )
    abs_sketch = _sample_bug_sketch(
        write_scope=[str((repo / "pricing.py").resolve()), "test_pricing.py"]
    )
    stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([abs_sketch]),
    )
    out = stage.architect_sketch(
        run_id=run_id,
        raw_input="Fix pricing bug",
        state={},
    )
    assert out["architect_sketch"]["write_scope"] == ["pricing.py", "test_pricing.py"]


def test_legacy_sdlc_session_resume_matches_origin_main_workflow_hashes(tmp_path: Path) -> None:
    """Resuming legacy runs started with origin/main workflow_hashes must succeed."""
    origin_main_hashes = {
        "full_sdlc_tdd": (
            "sha256:291faa5f5f3fb4d33efe1c28a579e04c4bc3ae0e085f0bce5c8110bf06a3cd7d"
        ),
        "product_blueprint": (
            "sha256:26841aed6e876cf8b7773727cc96e867053412b50c6851710ee1598987b2fae7"
        ),
        "repo_fast_patch": (
            "sha256:87aa62e27d58fb5fdea94754dace076bffbac8e458228d25b02f9f03d0b424a5"
        ),
    }
    workspace = tmp_path / "repo"
    workspace.mkdir()
    runner = SessionRunner(
        SessionConfig(
            workspace_root=workspace,
            state_dir=tmp_path / "state",
            workflow_explicit=False,
            provider_builder=lambda **_kwargs: _mock_provider([]),
        )
    )
    store = runner.store
    for wf_name, legacy_hash in origin_main_hashes.items():
        assert load_workflow(wf_name).execution_hash() == legacy_hash
        run_id = store.create_run(
            title=f"Legacy {wf_name}",
            mode=RunMode.EXECUTION,
            workflow_name=wf_name,
            workspace_root=workspace,
        )
        store.append_event(
            run_id=run_id,
            stage="session",
            event_type="session.started",
            actor="session",
            payload={"workflow": wf_name, "workflow_hash": legacy_hash},
        )
        reloaded = runner._workflow_for_run(run_id, store.get_run(run_id))
        assert reloaded.name == wf_name


def test_failing_test_first_restricts_red_phase_to_test_files_only(tmp_path: Path) -> None:
    """Red-test phase must reject edits to production files via test-only write_scope."""
    repo = tmp_path / "repo"
    _init_buggy_repo(repo, with_failing_test=False)
    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Red phase production mutation blocked",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_bugfix",
        workspace_root=repo,
    )

    seen_red_scope: list[str] = []

    def _cheating_red_worker(req: DriverTurnRequest) -> str:
        seen_red_scope.extend(req.write_scope)
        # Attempt to mutate production code during red_test phase!
        (req.workspace_path / "pricing.py").write_text(
            "def apply_discount(price: int, percent: int) -> int:\n"
            "    return price - (price * percent // 100)\n",
            "utf-8",
        )
        return "Mutated production file during red phase."

    stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([]),
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="scripted",
            reviewer_driver="native",
        ),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_cheating_red_worker)}),
    )
    sketch = ArchitectSketch.model_validate_json(_sample_bug_sketch())
    with pytest.raises(CraftPause, match="pricing.py"):
        stage.build_direct(
            run_id=run_id,
            raw_input="Fix discount bug",
            state={"architect_sketch": sketch.model_dump(mode="json")},
            failing_test_first=True,
        )
    assert seen_red_scope == ["test_pricing.py"]


def test_failing_test_first_rejects_deleted_red_test_and_vacuous_zero_test_green(
    tmp_path: Path,
) -> None:
    """Green check must reject deleting/reverting red tests or running 0 tests with exit 0."""
    repo = tmp_path / "repo"
    _init_buggy_repo(repo, with_failing_test=False)
    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Cheating builder deletes red test",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_bugfix",
        workspace_root=repo,
    )

    original_test = (repo / "test_pricing.py").read_text("utf-8")

    def _revert_test_worker(req: DriverTurnRequest) -> str:
        if req.actor == "hephaestus:red_test":
            (req.workspace_path / "test_pricing.py").write_text(
                original_test
                + "\n    def test_discount_reduces_price(self) -> None:\n"
                "        self.assertEqual(pricing.apply_discount(100, 20), 80)\n",
                "utf-8",
            )
            return "Added failing regression test."
        # Cheating build turn reverts the regression test instead of fixing pricing.py!
        (req.workspace_path / "test_pricing.py").write_text(original_test, "utf-8")
        return "Reverted test so unittest exits 0."

    stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([]),
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="scripted",
            reviewer_driver="native",
        ),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_revert_test_worker)}),
    )
    sketch = ArchitectSketch.model_validate_json(_sample_bug_sketch())
    with pytest.raises(CraftPause, match=r"(?:SHA-256 mismatch|reverted to pre-red baseline)"):
        stage.build_direct(
            run_id=run_id,
            raw_input="Fix discount bug",
            state={"architect_sketch": sketch.model_dump(mode="json")},
            failing_test_first=True,
            max_diff_gate_retries=1,
        )

    # Verify vacuous zero-test output (exit 0 with Ran 0 tests / collected 0 items) is rejected.
    repo2 = tmp_path / "repo2"
    _init_buggy_repo(repo2, with_failing_test=True)
    run_id2 = store.create_run(
        title="Vacuous zero-test green",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_bugfix",
        workspace_root=repo2,
    )

    def _emptying_test_worker(req: DriverTurnRequest) -> str:
        # Replaces test file with empty test class so unittest prints "Ran 0 tests" and exits 0!
        (req.workspace_path / "test_pricing.py").write_text(
            "import unittest\n\nclass TestPricing(unittest.TestCase):\n    pass\n",
            "utf-8",
        )
        return "Emptied test class."

    stage2 = CraftStage(
        store=store,
        workspace_root=repo2,
        artifacts_root=tmp_path / "artifacts2",
        sandbox_root=tmp_path / "sandboxes2",
        provider=_mock_provider([]),
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="scripted",
            reviewer_driver="native",
        ),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_emptying_test_worker)}),
    )
    with pytest.raises(CraftPause, match=r"(?:SHA-256 mismatch|Ran 0 tests|vacuous zero-test run)"):
        stage2.build_direct(
            run_id=run_id2,
            raw_input="Fix discount bug",
            state={"architect_sketch": sketch.model_dump(mode="json")},
            failing_test_first=True,
            max_diff_gate_retries=1,
        )

    # Verify exit_code=0 with "Ran 0 tests" (older unittest / custom runner) is blocked
    # even when the test file itself is untouched.
    run_id3 = store.create_run(
        title="Vacuous exit-0 zero-test green",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_bugfix",
        workspace_root=repo2,
    )
    call_count = 0

    def _vacuous_runner(_cmd: str, *, cwd: Path, timeout_s: int = 60) -> tuple[int | None, str]:
        nonlocal call_count
        del cwd, timeout_s
        call_count += 1
        if call_count == 1:
            return 1, "FAIL: test_discount_reduces_price\nRan 1 test in 0.001s\nFAILED (failures=1)"
        return 0, "Ran 0 tests in 0.000s\n\nOK"

    def _fix_only_prod_worker(req: DriverTurnRequest) -> str:
        (req.workspace_path / "pricing.py").write_text(
            "def apply_discount(price: int, percent: int) -> int:\n"
            "    return price - (price * percent // 100)\n",
            "utf-8",
        )
        return "Fixed pricing.py only."

    stage3 = CraftStage(
        store=store,
        workspace_root=repo2,
        artifacts_root=tmp_path / "artifacts3",
        sandbox_root=tmp_path / "sandboxes3",
        provider=_mock_provider([]),
        command_runner=_vacuous_runner,
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="scripted",
            reviewer_driver="native",
        ),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_fix_only_prod_worker)}),
    )
    with pytest.raises(CraftPause, match="vacuous zero-test run"):
        stage3.build_direct(
            run_id=run_id3,
            raw_input="Fix discount bug",
            state={"architect_sketch": sketch.model_dump(mode="json")},
            failing_test_first=True,
            max_diff_gate_retries=1,
        )


def test_failing_test_first_ignores_stale_historical_base_ref_repro_receipt(tmp_path: Path) -> None:
    """When explore ran repro on an older base_ref while HEAD passes, build must not reuse it."""
    repo = tmp_path / "repo"
    _init_buggy_repo(repo, with_failing_test=True)
    buggy_sha = _git(repo, "rev-parse", "HEAD")

    # Fix on HEAD so current sandbox passes without a new red test.
    (repo / "pricing.py").write_text(
        "def apply_discount(price: int, percent: int) -> int:\n"
        "    return price - (price * percent // 100)\n",
        "utf-8",
    )
    _git(repo, "add", "pricing.py")
    _git(repo, "commit", "-m", "already fixed on HEAD")

    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Stale historical repro",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_bugfix",
        workspace_root=repo,
    )
    brief_json = EvidenceBrief(
        summary="Historical bug on buggy_sha",
        facts=[EvidenceFact(path="pricing.py", line=3, observation="sign error on buggy_sha")],
        repro_command=f"{sys.executable} -m unittest test_pricing.py",
        candidate_files=["pricing.py", "test_pricing.py"],
    ).model_dump_json()

    actors: list[str] = []

    def _noop_worker(req: DriverTurnRequest) -> str:
        actors.append(req.actor)
        return "noop"

    stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([brief_json]),
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="scripted",
            reviewer_driver="native",
        ),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_noop_worker)}),
    )
    explore_out = stage.explore_reproduce(
        run_id=run_id,
        raw_input="Repro on older base_ref",
        base_ref=buggy_sha,
        require_repro=True,
    )
    assert explore_out["repro_receipt"]["verified"] is True

    sketch = ArchitectSketch.model_validate_json(_sample_bug_sketch())
    with pytest.raises(CraftPause, match="requires a proven failing test"):
        stage.build_direct(
            run_id=run_id,
            raw_input="Fix bug",
            state={
                "evidence_brief": explore_out["evidence_brief"],
                "architect_sketch": sketch.model_dump(mode="json"),
            },
            failing_test_first=True,
        )
    # Must have entered red_test phase instead of blindly trusting buggy_sha's stale receipt.
    assert actors == ["hephaestus:red_test"]


def test_repro_explicit_base_ref_fails_closed_on_non_git_copy_sandbox_and_zero_tests(
    tmp_path: Path,
) -> None:
    # Non-git directory -> copy sandbox cannot verify explicit base_ref.
    non_git_dir = tmp_path / "plain_dir"
    non_git_dir.mkdir()
    (non_git_dir / "pricing.py").write_text("x = 1\n", "utf-8")
    (non_git_dir / "test_pricing.py").write_text(
        "import unittest\nclass T(unittest.TestCase):\n    def test_x(self): self.fail('boom')\n",
        "utf-8",
    )
    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Copy sandbox base_ref",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_bugfix",
        workspace_root=non_git_dir,
    )
    brief_json = EvidenceBrief(
        summary="Defect in non-git workspace",
        facts=[EvidenceFact(path="pricing.py", line=1, observation="x = 1")],
        repro_command=f"{sys.executable} -m unittest test_pricing.py",
        candidate_files=["pricing.py"],
    ).model_dump_json()

    stage = CraftStage(
        store=store,
        workspace_root=non_git_dir,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([brief_json]),
    )
    with pytest.raises(CraftPause, match="non-git copy sandbox"):
        stage.explore_reproduce(
            run_id=run_id,
            raw_input="Repro on base_ref",
            base_ref="v1.0",
            require_repro=True,
        )

    # Zero-test output patterns rejected in both failure and green helpers.
    for sample_out in (
        "Ran 0 tests in 0.000s\n\nOK",
        "collected 0 items\nno tests ran in 0.01s",
        "running 0 tests\ntest result: ok. 0 passed",
        "?   example.com/pkg [no test files]",
    ):
        ok_fail, detail_fail = _is_proven_failure_exit(["pytest"], 1, output=sample_out)
        assert not ok_fail and "zero tests" in detail_fail
        ok_green, detail_green = _is_proven_green_exit(["pytest"], 0, output=sample_out)
        assert not ok_green and "zero-test" in detail_green


def test_investigate_redirects_artifacts_when_artifacts_root_inside_workspace(
    tmp_path: Path,
) -> None:
    """Even if artifacts_root points inside workspace_root, investigate must never mutate repo."""
    repo = tmp_path / "repo"
    _init_buggy_repo(repo, with_failing_test=True)
    before_files = sorted(
        str(p.relative_to(repo)) for p in repo.rglob("*") if ".git" not in p.parts
    )
    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Misconfigured in-repo artifacts root",
        mode=RunMode.EXECUTION,
        workflow_name="investigate",
        workspace_root=repo,
    )
    forensics_brief = EvidenceBrief(
        summary="pricing.apply_discount adds instead of subtracts at pricing.py:3",
        facts=[EvidenceFact(path="pricing.py", line=3, observation="adds discount")],
        candidate_files=["pricing.py"],
    ).model_dump_json()

    in_repo_artifacts = repo / ".olympus" / "artifacts"
    runtime = WorkflowRuntime(
        store=store,
        workspace_root=repo,
        artifacts_root=in_repo_artifacts,
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([forensics_brief]),
    )
    result = run_graph(
        build_graph(load_workflow("investigate")),
        {"run_id": run_id, "raw_input": "Why does apply_discount fail?", "_runtime": runtime},
    )

    assert result["route"] == "report"
    assert result["read_only"] is True
    assert not in_repo_artifacts.exists()
    assert not (repo / ".olympus").exists()
    assert _git(repo, "status", "--porcelain") == ""
    after_files = sorted(
        str(p.relative_to(repo)) for p in repo.rglob("*") if ".git" not in p.parts
    )
    assert after_files == before_files
    recorded = [
        ev
        for ev in store.list_events(run_id)
        if ev.event_type == "craft.investigation_report_recorded"
    ]
    assert len(recorded) == 1
    report_path = Path(recorded[0].payload["report_path"])
    assert report_path.is_file()
    assert not report_path.resolve().is_relative_to(repo.resolve())


def test_green_phase_blocks_skip_and_weakened_assertions_via_sha256_lock(tmp_path: Path) -> None:
    """Green phase must lock red test files in forbidden_globs and verify exact SHA-256."""
    repo = tmp_path / "repo"
    _init_buggy_repo(repo, with_failing_test=True)
    store = RunStore(tmp_path / "olympus.sqlite")
    sketch = ArchitectSketch.model_validate_json(_sample_bug_sketch())

    # 1) Attempting to weaken assertion (`assertTrue(True)`) must be blocked by SHA-256 lock.
    run_id = store.create_run(
        title="Weaken assertion bypass",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_bugfix",
        workspace_root=repo,
    )
    seen_forbidden: list[list[str]] = []

    def _weaken_assertion_worker(req: DriverTurnRequest) -> str:
        seen_forbidden.append(list(req.forbidden_globs))
        (req.workspace_path / "test_pricing.py").write_text(
            "import unittest\n\n"
            "class TestPricing(unittest.TestCase):\n"
            "    def test_discount_reduces_price(self) -> None:\n"
            "        self.assertTrue(True)\n",
            "utf-8",
        )
        return "Weakened assertion."

    stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([]),
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="scripted",
            reviewer_driver="native",
        ),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_weaken_assertion_worker)}),
    )
    with pytest.raises(CraftPause, match="SHA-256 mismatch"):
        stage.build_direct(
            run_id=run_id,
            raw_input="Fix discount bug",
            state={"architect_sketch": sketch.model_dump(mode="json")},
            failing_test_first=True,
            max_diff_gate_retries=1,
        )
    assert seen_forbidden and "test_pricing.py" in seen_forbidden[0]

    # 2) All-skipped test runs rejected by _is_proven_green_exit while partial skips still pass.
    for skipped_out in (
        "s\n----------------------------------------------------------------------\n"
        "Ran 1 test in 0.000s\n\nOK (skipped=1)",
        "1 skipped in 0.01s",
        "test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured",
    ):
        ok, detail = _is_proven_green_exit(["pytest"], 0, output=skipped_out)
        assert not ok and "skipped" in detail

    ok_partial_ut, _ = _is_proven_green_exit(
        ["unittest"],
        0,
        output=".s\n----------------------------------------------------------------------\n"
        "Ran 2 tests in 0.001s\n\nOK (skipped=1)",
    )
    assert ok_partial_ut
    ok_partial_pt, _ = _is_proven_green_exit(["pytest"], 0, output="2 passed, 1 skipped in 0.02s")
    assert ok_partial_pt


def test_stale_red_test_receipt_invalidated_when_sandbox_baseline_changes(tmp_path: Path) -> None:
    """Persisted red_test_receipt bound to an old baseline must not be reused on a new baseline."""
    repo = tmp_path / "repo"
    _init_buggy_repo(repo, with_failing_test=False)
    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Stale baseline receipt",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_bugfix",
        workspace_root=repo,
    )
    actors: list[str] = []

    def _noop_worker(req: DriverTurnRequest) -> str:
        actors.append(req.actor)
        return "noop"

    stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([]),
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="scripted",
            reviewer_driver="native",
        ),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_noop_worker)}),
    )
    # Seed a stale red_test_receipt bound to an obsolete baseline fingerprint.
    stage._save_persisted(
        run_id,
        {
            "red_test_receipt": {
                "command": f"{sys.executable} -m unittest test_pricing.py",
                "exit_code": 1,
                "excerpt": "old failure",
                "verified": True,
                "source": "sandbox_pre_fix",
                "baseline_fingerprint": "sha256:stale-old-baseline-fingerprint",
                "red_test_file_hashes": {"test_pricing.py": "deadbeef"},
            }
        },
    )
    sketch = ArchitectSketch.model_validate_json(_sample_bug_sketch())
    with pytest.raises(CraftPause, match="requires a proven failing test"):
        stage.build_direct(
            run_id=run_id,
            raw_input="Fix bug",
            state={"architect_sketch": sketch.model_dump(mode="json")},
            failing_test_first=True,
        )
    # Must have invalidated the stale receipt and entered red_test phase!
    assert actors == ["hephaestus:red_test"]


def test_long_leading_logs_before_zero_or_skipped_tail_are_never_missed(tmp_path: Path) -> None:
    """Zero-test and all-skipped checks must inspect untrimmed output past _MAX_EXCERPT_CHARS."""
    repo = tmp_path / "repo"
    _init_buggy_repo(repo, with_failing_test=True)
    # Write a test file whose import emits >2500 chars of stdout before unittest reports skipped=1.
    (repo / "test_verbose_skip.py").write_text(
        "import unittest\n"
        "print('NOISE_' * 600)\n\n"
        "class TestVerboseSkip(unittest.TestCase):\n"
        "    @unittest.skip('all skipped')\n"
        "    def test_skipped(self) -> None:\n"
        "        self.fail('unreachable')\n",
        "utf-8",
    )
    _git(repo, "add", "test_verbose_skip.py")
    _git(repo, "commit", "-m", "add verbose skipped test")

    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Verbose tail detection",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_bugfix",
        workspace_root=repo,
    )
    stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([]),
    )
    sketch = ArchitectSketch.model_validate_json(
        _sample_bug_sketch(
            verification_plan=[f"{sys.executable} -m unittest test_verbose_skip.py"]
        )
    )
    with pytest.raises(CraftPause, match="Surface verification did not pass"):
        stage.verify_surface(
            run_id=run_id,
            state={"architect_sketch": sketch.model_dump(mode="json")},
        )
    persisted_verify = stage._load_persisted(run_id)["surface_verification"]
    assert persisted_verify["passed"] is False
    assert persisted_verify["checks"][0]["status"] == "inconclusive"


def test_pstack_craft_legacy_hash_resume_and_family_decoupled_resume_node(
    tmp_path: Path,
) -> None:
    """Legacy pstack_craft sessions resume cleanly and resume routing ignores descriptive family."""
    from olympus.core.contracts import (
        ApprovalDecision,
        ApprovalRecord,
        ApprovalTarget,
        GoalSpec,
        RequirementItem,
        SpecStatus,
    )
    from olympus.graph import END, START, WorkflowDefinition, resolve_resume_node
    from olympus.graph.loader import WorkflowEdgeDef, WorkflowNodeDef

    workspace = tmp_path / "repo"
    workspace.mkdir()
    runner = SessionRunner(
        SessionConfig(
            workspace_root=workspace,
            state_dir=tmp_path / "state",
            workflow_explicit=False,
            provider_builder=lambda **_kwargs: _mock_provider([]),
        )
    )
    store = runner.store
    craft_def = load_workflow("pstack_craft")
    run_id = store.create_run(
        title="Legacy pstack_craft session",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=workspace,
    )
    # Historical origin/main pstack_craft stored content_hash(model_dump(mode="json")).
    store.append_event(
        run_id=run_id,
        stage="session",
        event_type="session.started",
        actor="session",
        payload={"workflow": "pstack_craft", "workflow_hash": craft_def.content_hash},
    )
    reloaded = runner._workflow_for_run(run_id, store.get_run(run_id))
    assert reloaded.name == "pstack_craft"

    # Changing descriptive `family` on an SDLC-node graph must NOT alter resolve_resume_node.
    sdlc_custom_family = WorkflowDefinition(
        name="custom_sdlc_renamed_family",
        family="experimental_label",
        autonomy="autonomous_hard_stops",
        nodes=[
            WorkflowNodeDef(name="align", type="align_goal"),
            WorkflowNodeDef(name="gate_spec", type="human_gate", params={"gate": "spec"}),
            WorkflowNodeDef(name="plan", type="pm_decompose"),
        ],
        edges=[
            WorkflowEdgeDef(source=START, target="align"),
            WorkflowEdgeDef(source="align", target="gate_spec"),
            WorkflowEdgeDef(source="gate_spec", target="plan"),
            WorkflowEdgeDef(source="plan", target=END),
        ],
    )
    run_id2 = store.create_run(
        title="Custom family SDLC resume",
        mode=RunMode.EXECUTION,
        workflow_name=sdlc_custom_family.name,
        workspace_root=workspace,
    )
    spec = GoalSpec(
        spec_id="spec_1",
        status=SpecStatus.DRAFT,
        goal="Check SDLC resume",
        requirements=[
            RequirementItem(id="R1", description="Works", acceptance_criteria=["Passes"])
        ],
        out_of_scope=["Other"],
    )
    store.save_spec(run_id2, spec)
    store.append_event(
        run_id=run_id2,
        stage="graph",
        event_type="graph.paused",
        actor="gate_spec",
        payload={"node": "gate_spec", "reason": "awaiting approval"},
    )
    store.record_approval(
        ApprovalRecord(
            run_id=run_id2,
            target=ApprovalTarget.GOAL_SPEC,
            target_id=spec.spec_id,
            target_hash=spec.content_hash,
            decision=ApprovalDecision.APPROVED,
            approver="zeus",
        )
    )
    # Because gate_spec is superseded by the approved spec, SDLC node graph must advance to plan.
    assert resolve_resume_node(sdlc_custom_family, store.get_run(run_id2), store=store) == "plan"


def test_investigate_fails_closed_when_runstore_db_lives_inside_workspace(tmp_path: Path) -> None:
    """Read-only investigate must refuse to run if RunStore SQLite DB is inside workspace_root."""
    repo = tmp_path / "repo"
    _init_buggy_repo(repo, with_failing_test=True)
    in_repo_db = repo / ".olympus" / "olympus.sqlite"
    store = RunStore(in_repo_db)
    run_id = store.create_run(
        title="In-repo RunStore blocked",
        mode=RunMode.EXECUTION,
        workflow_name="investigate",
        workspace_root=repo,
    )
    events_before = len(store.list_events(run_id))

    runtime = WorkflowRuntime(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([]),
    )
    result = run_graph(
        build_graph(load_workflow("investigate")),
        {"run_id": run_id, "raw_input": "Why does apply_discount fail?", "_runtime": runtime},
    )
    assert result["_paused_node"] == "explore"
    assert result["pause_kind"] == "blocked"
    assert "RunStore inside workspace_root" in result["_pause_reason"]
    # Zero craft events written to the in-workspace DB!
    assert len(store.list_events(run_id)) == events_before


def test_pytest_qq_all_skipped_output_blocked_in_green_and_verify_surface(tmp_path: Path) -> None:
    """pytest -qq all-skipped compact output (`s [100%]`, `s`, `x [100%]`) must never pass."""
    for compact_skip in (
        "s                                                                        [100%]",
        "s [100%]",
        "sss [100%]",
        "x [100%]",
        "s",
        "setup noise\ns [100%]",
    ):
        ok, detail = _is_proven_green_exit(["pytest", "-qq"], 0, output=compact_skip)
        assert not ok and "skipped" in detail

    for compact_pass in (".s [100%]", ".. [100%]", "."):
        ok, _ = _is_proven_green_exit(["pytest", "-qq"], 0, output=compact_pass)
        assert ok

    # Real `pytest -qq` run against an all-skipped test file must be rejected by verify_surface.
    repo = tmp_path / "repo"
    _init_buggy_repo(repo, with_failing_test=True)
    (repo / "test_all_skipped_pytest.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.skip(reason='skipped')\n"
        "def test_never_runs() -> None:\n"
        "    assert False\n",
        "utf-8",
    )
    _git(repo, "add", "test_all_skipped_pytest.py")
    _git(repo, "commit", "-m", "add all-skipped pytest file")

    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="pytest -qq all skipped",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_bugfix",
        workspace_root=repo,
    )
    stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([]),
    )
    sketch = ArchitectSketch.model_validate_json(
        _sample_bug_sketch(
            verification_plan=[f"{sys.executable} -m pytest -qq test_all_skipped_pytest.py"]
        )
    )
    with pytest.raises(CraftPause, match="Surface verification did not pass"):
        stage.verify_surface(
            run_id=run_id,
            state={"architect_sketch": sketch.model_dump(mode="json")},
        )
    persisted_verify = stage._load_persisted(run_id)["surface_verification"]
    assert persisted_verify["passed"] is False
    assert persisted_verify["checks"][0]["status"] == "inconclusive"


def test_recreating_sandbox_after_new_commit_refreshes_baseline_and_invalidates_red_receipt(
    tmp_path: Path,
) -> None:
    """Rebuilding a removed sandbox after a new commit must refresh baseline & drop old red."""
    import shutil

    repo = tmp_path / "repo"
    _init_buggy_repo(repo, with_failing_test=True)
    buggy_sha = _git(repo, "rev-parse", "HEAD")

    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Sandbox recreate invalidates stale red receipt",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_bugfix",
        workspace_root=repo,
    )
    actors: list[str] = []

    def _noop_worker(req: DriverTurnRequest) -> str:
        actors.append(req.actor)
        return "noop"

    stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([]),
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="scripted",
            reviewer_driver="native",
        ),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_noop_worker)}),
    )
    sketch = ArchitectSketch.model_validate_json(_sample_bug_sketch())
    # Create initial sandbox at buggy commit and record its red_test_receipt.
    handle1, baseline1 = stage._ensure_sandbox(run_id)
    assert baseline1.base_commit == buggy_sha
    receipt1 = stage._ensure_red_test_receipt(
        run_id=run_id,
        raw_input="Fix pricing bug",
        state={},
        brief=None,
        sketch=sketch,
        handle=handle1,
        baseline=baseline1,
        node_name="build",
        role="builder",
        system_prompt="builder",
        strict_scope=True,
        max_steps=8,
    )
    assert receipt1["verified"] is True

    # Remove the sandbox directory while leaving stale baseline.json on disk, then commit a fix!
    shutil.rmtree(handle1.workspace_path)
    _git(repo, "worktree", "prune")
    assert handle1.baseline_path.is_file()

    (repo / "pricing.py").write_text(
        "def apply_discount(price: int, percent: int) -> int:\n"
        "    return price - (price * percent // 100)\n",
        "utf-8",
    )
    _git(repo, "add", "pricing.py")
    _git(repo, "commit", "-m", "fix pricing on main")
    fixed_sha = _git(repo, "rev-parse", "HEAD")
    assert fixed_sha != buggy_sha

    # Re-entering build_direct recreates the sandbox at fixed_sha, refreshes baseline.json,
    # invalidates the buggy_sha red_test_receipt, and enters hephaestus:red_test!
    with pytest.raises(CraftPause, match="requires a proven failing test"):
        stage.build_direct(
            run_id=run_id,
            raw_input="Fix pricing bug",
            state={"architect_sketch": sketch.model_dump(mode="json")},
            failing_test_first=True,
        )
    assert actors == ["hephaestus:red_test"]
    _handle2, baseline2 = stage._ensure_sandbox(run_id)
    assert baseline2.base_commit == fixed_sha


def test_pytest_qq_color_yes_ansi_skipped_output_blocked(tmp_path: Path) -> None:
    """ANSI-colored `pytest -qq --color=yes` all-skipped output must be stripped and blocked."""
    ansi_skip = "\x1b[33ms\x1b[0m\x1b[33m                                   [100%]\x1b[0m"
    ok, detail = _is_proven_green_exit(["pytest", "-qq", "--color=yes"], 0, output=ansi_skip)
    assert not ok and "skipped" in detail

    repo = tmp_path / "repo"
    _init_buggy_repo(repo, with_failing_test=True)
    (repo / "test_color_skip.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.skip(reason='ansi skipped')\n"
        "def test_skipped_ansi() -> None:\n"
        "    assert False\n",
        "utf-8",
    )
    _git(repo, "add", "test_color_skip.py")
    _git(repo, "commit", "-m", "add ansi skipped test")

    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="pytest -qq --color=yes all skipped",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_bugfix",
        workspace_root=repo,
    )
    stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([]),
    )
    sketch = ArchitectSketch.model_validate_json(
        _sample_bug_sketch(
            verification_plan=[
                f"{sys.executable} -m pytest -qq --color=yes test_color_skip.py"
            ]
        )
    )
    with pytest.raises(CraftPause, match="Surface verification did not pass"):
        stage.verify_surface(
            run_id=run_id,
            state={"architect_sketch": sketch.model_dump(mode="json")},
        )
    persisted_verify = stage._load_persisted(run_id)["surface_verification"]
    assert persisted_verify["passed"] is False
    assert persisted_verify["checks"][0]["status"] == "inconclusive"


def test_retained_sandbox_empty_commit_and_same_sha_rebuild_invalidate_red_receipt(
    tmp_path: Path,
) -> None:
    """Empty commit in retained sandbox AND same-SHA sandbox rebuild must clear red receipt."""
    import shutil

    repo = tmp_path / "repo"
    _init_buggy_repo(repo, with_failing_test=True)
    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Empty commit and same-SHA recreate",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_bugfix",
        workspace_root=repo,
    )
    sketch = ArchitectSketch.model_validate_json(_sample_bug_sketch())
    stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([]),
    )

    # 1) Initial sandbox records red_test_receipt.
    handle1, baseline1 = stage._ensure_sandbox(run_id)
    stage._ensure_red_test_receipt(
        run_id=run_id,
        raw_input="Fix pricing bug",
        state={},
        brief=None,
        sketch=sketch,
        handle=handle1,
        baseline=baseline1,
        node_name="build",
        role="builder",
        system_prompt="builder",
        strict_scope=True,
        max_steps=8,
    )
    assert stage._load_persisted(run_id).get("red_test_receipt") is not None

    # 2) Empty commit inside retained sandbox changes HEAD -> must refresh baseline & clear receipt.
    _git(handle1.workspace_path, "commit", "--allow-empty", "-m", "empty commit in sandbox")
    new_sandbox_head = _git(handle1.workspace_path, "rev-parse", "HEAD")
    assert new_sandbox_head != baseline1.base_commit

    handle2, baseline2 = stage._ensure_sandbox(run_id)
    assert handle2.reused is True
    assert baseline2.base_commit == new_sandbox_head
    assert stage._load_persisted(run_id).get("red_test_receipt") is None

    # 3) Re-record receipt, then remove sandbox and recreate on the SAME repo SHA -> must clear!
    stage._ensure_red_test_receipt(
        run_id=run_id,
        raw_input="Fix pricing bug",
        state={},
        brief=None,
        sketch=sketch,
        handle=handle2,
        baseline=baseline2,
        node_name="build",
        role="builder",
        system_prompt="builder",
        strict_scope=True,
        max_steps=8,
    )
    assert stage._load_persisted(run_id).get("red_test_receipt") is not None

    shutil.rmtree(handle2.workspace_path)
    _git(repo, "worktree", "prune")
    handle3, baseline3 = stage._ensure_sandbox(run_id)
    assert handle3.reused is False
    assert baseline3.base_commit == baseline1.base_commit
    assert stage._load_persisted(run_id).get("red_test_receipt") is None
