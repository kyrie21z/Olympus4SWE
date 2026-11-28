"""Unit and end-to-end tests for Issue #191 Part B (`pstack_hillclimb` workflow)."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import olympus.stages.hillclimb as hillclimb_mod
from olympus.core.contracts import (
    EvidenceBrief,
    EvidenceFact,
    HillclimbConfig,
    HillclimbIterationRow,
    HillclimbReport,
    RunMode,
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
from olympus.session.contracts import StepMode
from olympus.session.runner import SessionConfig, SessionRunner
from olympus.stages.craft import load_skill_pack
from olympus.stages.hillclimb import (
    HillclimbPause,
    HillclimbStage,
    _parse_safe_argv,
    _probe_bwrap,
    _probe_bwrap_unshare_net,
    _probe_bwrap_unshare_pid,
    _run_argv_command,
    hillclimb_loop,
    parse_metric_value,
    snapshot_frozen_harness,
)


@pytest.fixture(autouse=True)
def _ci_bwrap_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use real bwrap locally when available; opt in only when CI lacks userns bwrap."""
    if _probe_bwrap() is None:
        monkeypatch.setenv("OLYMPUS_ALLOW_UNSANDBOXED_REPRO", "1")


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _init_opt_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@olympus.local")
    _git(repo, "config", "user.name", "Olympus Test")
    (repo / "solver.py").write_text(
        "def score_value() -> int:\n"
        "    return 100\n"
        "\n"
        "def has_duplicate(nums: list[int]) -> bool:\n"
        "    for i in range(len(nums)):\n"
        "        for j in range(i + 1, len(nums)):\n"
        "            if nums[i] == nums[j]:\n"
        "                return True\n"
        "    return False\n",
        encoding="utf-8",
    )
    (repo / "bench.py").write_text(
        "import solver\n"
        "assert solver.has_duplicate([1, 2, 3, 2]) is True\n"
        "assert solver.has_duplicate([1, 2, 3, 4]) is False\n"
        "print(f'SCORE={solver.score_value()}')\n",
        encoding="utf-8",
    )
    (repo / "test_solver.py").write_text(
        "import unittest\n"
        "import solver\n"
        "\n"
        "class TestSolver(unittest.TestCase):\n"
        "    def test_duplicates(self) -> None:\n"
        "        self.assertTrue(solver.has_duplicate([3, 1, 3]))\n"
        "        self.assertFalse(solver.has_duplicate([1, 2, 3]))\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mock_provider(texts: list[str]) -> MockLLMProvider:
    return MockLLMProvider([LLMResponse(content=text) for text in texts])


class _ScriptedDriver:
    name = "scripted"

    def __init__(self, handler: Callable[[DriverTurnRequest], str]) -> None:
        self._handler = handler
        self.calls: list[DriverTurnRequest] = []

    def run_turn(
        self,
        request: DriverTurnRequest,
        *,
        on_event: DriverEvent | None = None,
        on_handle: Callable[[ActiveTurnHandle], None] | None = None,
    ) -> TurnOutcome:
        del on_event, on_handle
        self.calls.append(request)
        text = self._handler(request)
        return TurnOutcome(
            completed=True,
            stop_reason="end_turn",
            final_content=text,
            steps_taken=1,
            total_input_tokens=12,
            total_output_tokens=12,
            driver_name="scripted",
        )


def _sample_evidence() -> str:
    return EvidenceBrief(
        summary="solver.score_value returns 100 and has_duplicate uses O(n^2) loops.",
        facts=[EvidenceFact(path="solver.py", line=1, observation="baseline score_value=100")],
        candidate_files=["solver.py"],
    ).model_dump_json()


def _base_config(
    *,
    direction: str = "minimize",
    target_value: float | None = None,
    max_iterations: int = 5,
    plateau_patience: int = 3,
) -> HillclimbConfig:
    return HillclimbConfig(
        frozen_harness_paths=["bench.py"],
        metric_command=f"{sys.executable} bench.py",
        metric_regex=r"SCORE=([0-9]+(?:\.[0-9]+)?)",
        direction=direction,  # type: ignore[arg-type]
        target_value=target_value,
        max_iterations=max_iterations,
        plateau_patience=plateau_patience,
        write_scope=["solver.py"],
    )


def test_pstack_hillclimb_manifest_and_skill_pack_validate() -> None:
    assert "pstack_hillclimb" in list_builtin_workflows()
    definition = load_workflow("pstack_hillclimb")
    assert definition.family == "pstack"
    assert definition.autonomy == "autonomous_hard_stops"
    assert definition.stop_after == "verify"
    assert definition.applicable_scenarios
    assert definition.anti_patterns

    by_name = {node.name: node for node in definition.nodes}
    assert by_name["explore"].type == "explore_reproduce"
    assert by_name["explore"].god == "prometheus"
    assert by_name["hillclimb"].type == "hillclimb_loop"
    assert by_name["hillclimb"].god == "hephaestus"
    assert by_name["hillclimb"].skill_refs == ["pstack/hillclimb"]
    assert by_name["verify_surface"].type == "verify_surface"
    assert by_name["verify_surface"].god == "argus"

    described = describe_workflow(definition)
    assert described["family"] == "pstack"
    assert described["stop_after"] == "verify"

    pack = load_skill_pack(["pstack/hillclimb"])
    assert "frozen_harness_paths" in pack
    assert "HYPOTHESIS:" in pack


def test_hillclimb_contracts_enforce_strict_invariants_and_fingerprints() -> None:
    cfg1 = _base_config(direction="minimize", target_value=10.0)
    cfg2 = _base_config(direction="minimize", target_value=10.0)
    assert cfg1.content_hash
    assert cfg1.content_hash == cfg2.content_hash

    with pytest.raises(ValidationError):
        HillclimbConfig(
            frozen_harness_paths=["bench.py"],
            metric_command="python bench.py",
            metric_regex="[invalid(",
            direction="minimize",
            write_scope=["solver.py"],
        )

    with pytest.raises(ValidationError):
        HillclimbConfig(
            frozen_harness_paths=["bench.py"],
            metric_command="python bench.py",
            metric_regex=r"SCORE=(\d+)",
            direction="minimize",
            write_scope=["bench.py"],
        )

    with pytest.raises(ValidationError):
        HillclimbConfig(
            frozen_harness_paths=["../escape.py"],
            metric_command="python bench.py",
            metric_regex=r"SCORE=(\d+)",
            direction="minimize",
            write_scope=["solver.py"],
        )

    with pytest.raises(ValidationError):
        HillclimbIterationRow(
            iteration=1,
            hypothesis="Claim keep without commit",
            metric_before=100.0,
            metric_after=50.0,
            decision="keep",
            commit_sha=None,
        )

    with pytest.raises(ValidationError):
        HillclimbIterationRow(
            iteration=1,
            hypothesis="Revert cannot retain commit",
            metric_before=100.0,
            metric_after=120.0,
            decision="revert",
            commit_sha="deadbeef",
        )

    kept_row = HillclimbIterationRow(
        iteration=1,
        hypothesis="Use hash set",
        metric_before=100.0,
        metric_after=40.0,
        decision="keep",
        commit_sha="abc1234",
        reason="Improved",
        elapsed_s=0.2,
    )
    assert kept_row.content_hash

    with pytest.raises(ValidationError):
        HillclimbReport(
            baseline_metric=100.0,
            best_metric=40.0,
            kept_iterations=2,
            reverted_iterations=0,
            stopped_reason="max_iterations",
            rows=[kept_row],
        )


def test_anti_tamper_guard_rejects_and_reverts_frozen_harness_edits(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_opt_repo(repo)
    orig_bench_hash = _sha256(repo / "bench.py")

    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Hillclimb anti-tamper",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_hillclimb",
        workspace_root=repo,
    )

    sandbox_seen: list[Path] = []

    def _worker(req: DriverTurnRequest) -> str:
        sandbox_seen.append(req.workspace_path)
        if req.actor.endswith("_1"):
            # Attempt direct harness tamper to fake SCORE=1 alongside a solver edit.
            (req.workspace_path / "bench.py").write_text("print('SCORE=1')\n", encoding="utf-8")
            (req.workspace_path / "solver.py").write_text(
                "def score_value() -> int:\n    return 999\n"
                "def has_duplicate(nums: list[int]) -> bool:\n"
                "    return len(set(nums)) < len(nums)\n",
                encoding="utf-8",
            )
            (req.workspace_path / "untracked_junk.txt").write_text("junk", encoding="utf-8")
            return "HYPOTHESIS: Fake benchmark score by editing bench.py"
        if req.actor.endswith("_2"):
            # Attempt runtime tamper: solver.py mutates bench.py when imported by bench.py!
            (req.workspace_path / "solver.py").write_text(
                "from pathlib import Path\n"
                "def score_value() -> int:\n"
                "    Path('bench.py').write_text('print(\"SCORE=0\")\\n', encoding='utf-8')\n"
                "    return 5\n"
                "def has_duplicate(nums: list[int]) -> bool:\n"
                "    return len(set(nums)) < len(nums)\n",
                encoding="utf-8",
            )
            return "HYPOTHESIS: Mutate frozen bench.py at runtime"
        if req.actor.endswith("_3"):
            # Legitimate improvement strictly inside write_scope.
            (req.workspace_path / "solver.py").write_text(
                "def score_value() -> int:\n    return 25\n"
                "def has_duplicate(nums: list[int]) -> bool:\n"
                "    return len(set(nums)) < len(nums)\n",
                encoding="utf-8",
            )
            return "HYPOTHESIS: Optimize solver legitimately to score 25"
        raise AssertionError(f"Unexpected actor: {req.actor}")

    driver = _ScriptedDriver(_worker)
    result = hillclimb_loop(
        store=store,
        workspace_root=repo,
        provider=_mock_provider([]),
        run_id=run_id,
        raw_input="Minimize score without touching bench.py",
        config=_base_config(direction="minimize", max_iterations=3, plateau_patience=5),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        execution_drivers=ExecutionDriversConfig.resolve(worker_driver="scripted"),
        driver_registry=DriverRegistry({"scripted": driver}),
        apply_to_workspace=False,
    )

    report = HillclimbReport.model_validate(result["hillclimb_report"])
    assert report.baseline_metric == 100.0
    assert report.best_metric == 25.0
    assert report.kept_iterations == 1
    assert report.reverted_iterations == 2
    assert report.rows[0].decision == "rejected_tamper"
    # Under bwrap --ro-bind, runtime write fails with EROFS (error) before bench.py can mutate;
    # without bwrap, post-metric SHA-256 catches it as rejected_tamper.
    if _probe_bwrap() is not None:
        assert report.rows[1].decision == "error"
        assert "Read-only file system" in report.rows[1].reason or "PermissionError" in report.rows[
            1
        ].reason
    else:
        assert report.rows[1].decision == "rejected_tamper"
    assert report.rows[2].decision == "keep"
    assert report.rows[0].commit_sha is None
    assert report.rows[1].commit_sha is None
    assert report.rows[2].commit_sha is not None

    sandbox_ws = sandbox_seen[-1]
    assert _sha256(sandbox_ws / "bench.py") == orig_bench_hash
    assert not (sandbox_ws / "untracked_junk.txt").exists()
    assert "return 25" in (sandbox_ws / "solver.py").read_text(encoding="utf-8")


def test_keep_vs_revert_minimize_and_maximize_with_sqlite_and_trail(tmp_path: Path) -> None:
    # --- Case 1: minimize (improve -> regress -> equal -> improve) ---
    repo_min = tmp_path / "repo_min"
    _init_opt_repo(repo_min)
    store = RunStore(tmp_path / "olympus.sqlite")
    run_min = store.create_run(
        title="Minimize loop",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_hillclimb",
        workspace_root=repo_min,
    )

    min_scores = {1: 60, 2: 90, 3: 60, 4: 20}
    min_sandboxes: list[Path] = []

    def _min_worker(req: DriverTurnRequest) -> str:
        min_sandboxes.append(req.workspace_path)
        it = int(req.actor.rsplit("_", 1)[-1])
        val = min_scores[it]
        (req.workspace_path / "solver.py").write_text(
            f"def score_value() -> int:\n    return {val}\n"
            "def has_duplicate(nums: list[int]) -> bool:\n"
            "    return len(set(nums)) < len(nums)\n",
            encoding="utf-8",
        )
        if it == 2:
            # Verify git clean -fd wipes untracked in-scope files on revert.
            (req.workspace_path / "helper_reverted.py").write_text("X = 1\n", encoding="utf-8")
        return f"HYPOTHESIS: Set score_value to {val}"

    min_cfg = HillclimbConfig(
        frozen_harness_paths=["bench.py"],
        metric_command=f"{sys.executable} bench.py",
        metric_regex=r"SCORE=([0-9]+(?:\.[0-9]+)?)",
        direction="minimize",
        max_iterations=4,
        plateau_patience=4,
        write_scope=["solver.py", "helper_*.py"],
    )
    res_min = hillclimb_loop(
        store=store,
        workspace_root=repo_min,
        provider=_mock_provider([]),
        run_id=run_min,
        raw_input="Minimize score",
        config=min_cfg,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes_min",
        execution_drivers=ExecutionDriversConfig.resolve(worker_driver="scripted"),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_min_worker)}),
    )

    rep_min = HillclimbReport.model_validate(res_min["hillclimb_report"])
    assert rep_min.baseline_metric == 100.0
    assert rep_min.best_metric == 20.0
    assert rep_min.kept_iterations == 2
    assert rep_min.reverted_iterations == 2
    assert rep_min.stopped_reason == "max_iterations"
    assert [r.decision for r in rep_min.rows] == ["keep", "revert", "revert", "keep"]
    assert rep_min.rows[0].commit_sha is not None
    assert rep_min.rows[1].commit_sha is None
    assert rep_min.rows[2].commit_sha is None
    assert rep_min.rows[3].commit_sha is not None
    assert rep_min.rows[3].commit_sha != rep_min.rows[0].commit_sha

    # Verify git history inside worktree kept ONLY the 2 improving commits + initial commit.
    ws_min = min_sandboxes[-1]
    assert not (ws_min / "helper_reverted.py").exists()
    log_lines = _git(ws_min, "log", "--oneline").splitlines()
    assert len(log_lines) == 3  # initial + iter 1 + iter 4
    assert "return 20" in (ws_min / "solver.py").read_text(encoding="utf-8")

    # Verify SQLite persistence and decision_trail.json.
    events = store.list_events(run_min)
    iter_events = [ev for ev in events if ev.event_type == "hillclimb.iteration_recorded"]
    assert len(iter_events) == 4
    assert [ev.payload["decision"] for ev in iter_events] == ["keep", "revert", "revert", "keep"]
    completed_events = [ev for ev in events if ev.event_type == "hillclimb.completed"]
    assert len(completed_events) == 1
    assert completed_events[0].payload["best_metric"] == 20.0

    trail_path = Path(res_min["hillclimb_decision_trail_path"])
    assert trail_path.is_file()
    trail_data = json.loads(trail_path.read_text(encoding="utf-8"))
    assert len(trail_data["rows"]) == 4
    assert trail_data["best_metric"] == 20.0
    assert trail_data["stopped_reason"] == "max_iterations"

    # --- Case 2: maximize (regress -> improve -> target reached) ---
    repo_max = tmp_path / "repo_max"
    _init_opt_repo(repo_max)
    run_max = store.create_run(
        title="Maximize loop",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_hillclimb",
        workspace_root=repo_max,
    )
    max_scores = {1: 80, 2: 150, 3: 220}

    def _max_worker(req: DriverTurnRequest) -> str:
        it = int(req.actor.rsplit("_", 1)[-1])
        val = max_scores[it]
        (req.workspace_path / "solver.py").write_text(
            f"def score_value() -> int:\n    return {val}\n"
            "def has_duplicate(nums: list[int]) -> bool:\n"
            "    return len(set(nums)) < len(nums)\n",
            encoding="utf-8",
        )
        return f"HYPOTHESIS: Raise score_value to {val}"

    res_max = hillclimb_loop(
        store=store,
        workspace_root=repo_max,
        provider=_mock_provider([]),
        run_id=run_max,
        raw_input="Maximize score to >= 200",
        config=_base_config(
            direction="maximize",
            target_value=200.0,
            max_iterations=5,
            plateau_patience=3,
        ),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes_max",
        execution_drivers=ExecutionDriversConfig.resolve(worker_driver="scripted"),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_max_worker)}),
        apply_to_workspace=True,
    )

    rep_max = HillclimbReport.model_validate(res_max["hillclimb_report"])
    assert rep_max.baseline_metric == 100.0
    assert rep_max.best_metric == 220.0
    assert rep_max.stopped_reason == "target_reached"
    assert [r.decision for r in rep_max.rows] == ["revert", "keep", "keep"]
    assert res_max["applied_paths"] == ["solver.py"]
    assert "return 220" in (repo_max / "solver.py").read_text(encoding="utf-8")


def test_stop_predicates_target_plateau_max_iterations_and_fail_closed_baseline(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "olympus.sqlite")

    # 1. Plateau stop after 2 consecutive non-improving iterations (max_iterations=6).
    repo_plateau = tmp_path / "repo_plateau"
    _init_opt_repo(repo_plateau)
    run_plateau = store.create_run(
        title="Plateau stop",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_hillclimb",
        workspace_root=repo_plateau,
    )

    def _plateau_worker(req: DriverTurnRequest) -> str:
        it = int(req.actor.rsplit("_", 1)[-1])
        val = 50 if it == 1 else (60 if it == 2 else 70)
        (req.workspace_path / "solver.py").write_text(
            f"def score_value() -> int:\n    return {val}\n"
            "def has_duplicate(nums: list[int]) -> bool:\n"
            "    return len(set(nums)) < len(nums)\n",
            encoding="utf-8",
        )
        return f"HYPOTHESIS: Try value {val}"

    driver_plateau = _ScriptedDriver(_plateau_worker)
    res_plateau = hillclimb_loop(
        store=store,
        workspace_root=repo_plateau,
        provider=_mock_provider([]),
        run_id=run_plateau,
        raw_input="Minimize score",
        config=_base_config(direction="minimize", max_iterations=6, plateau_patience=2),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes_plateau",
        execution_drivers=ExecutionDriversConfig.resolve(worker_driver="scripted"),
        driver_registry=DriverRegistry({"scripted": driver_plateau}),
    )
    rep_plateau = HillclimbReport.model_validate(res_plateau["hillclimb_report"])
    assert rep_plateau.stopped_reason == "plateau"
    assert len(rep_plateau.rows) == 3  # iter 1 keep, iter 2 revert, iter 3 revert -> plateau=2
    assert len(driver_plateau.calls) == 3
    assert rep_plateau.best_metric == 50.0

    # 2. Target stop immediately after iter 1 hits target (max_iterations=5).
    repo_target = tmp_path / "repo_target"
    _init_opt_repo(repo_target)
    run_target = store.create_run(
        title="Target stop",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_hillclimb",
        workspace_root=repo_target,
    )

    def _target_worker(req: DriverTurnRequest) -> str:
        (req.workspace_path / "solver.py").write_text(
            "def score_value() -> int:\n    return 10\n"
            "def has_duplicate(nums: list[int]) -> bool:\n"
            "    return len(set(nums)) < len(nums)\n",
            encoding="utf-8",
        )
        return "HYPOTHESIS: Hit target directly"

    driver_target = _ScriptedDriver(_target_worker)
    res_target = hillclimb_loop(
        store=store,
        workspace_root=repo_target,
        provider=_mock_provider([]),
        run_id=run_target,
        raw_input="Minimize score to <= 15",
        config=_base_config(
            direction="minimize", target_value=15.0, max_iterations=5, plateau_patience=3
        ),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes_target",
        execution_drivers=ExecutionDriversConfig.resolve(worker_driver="scripted"),
        driver_registry=DriverRegistry({"scripted": driver_target}),
    )
    rep_target = HillclimbReport.model_validate(res_target["hillclimb_report"])
    assert rep_target.stopped_reason == "target_reached"
    assert len(rep_target.rows) == 1
    assert len(driver_target.calls) == 1

    # 3. Fail-closed baseline when metric regex fails to match.
    repo_bad = tmp_path / "repo_bad"
    _init_opt_repo(repo_bad)
    run_bad = store.create_run(
        title="Bad baseline",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_hillclimb",
        workspace_root=repo_bad,
    )
    bad_cfg = HillclimbConfig(
        frozen_harness_paths=["bench.py"],
        metric_command=f"{sys.executable} bench.py",
        metric_regex=r"NONEXISTENT_METRIC=([0-9]+)",
        direction="minimize",
        max_iterations=2,
        plateau_patience=2,
        write_scope=["solver.py"],
    )
    stage = HillclimbStage(
        store=store,
        workspace_root=repo_bad,
        provider=_mock_provider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes_bad",
    )
    with pytest.raises(HillclimbPause, match="Baseline metric measurement failed"):
        stage.hillclimb_loop(run_id=run_bad, raw_input="Fail closed", config=bad_cfg)


def test_pstack_hillclimb_e2e_graph_verify_surface_and_host_promotion(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_opt_repo(repo)
    orig_bench_hash = _sha256(repo / "bench.py")

    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Optimize solver end-to-end",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_hillclimb",
        workspace_root=repo,
    )

    def _worker(req: DriverTurnRequest) -> str:
        it = int(req.actor.rsplit("_", 1)[-1])
        if it == 1:
            # Attempt tamper -> rejected and reverted.
            (req.workspace_path / "bench.py").write_text("print('SCORE=0')\n", encoding="utf-8")
            return "HYPOTHESIS: Tamper bench.py"
        if it == 2:
            # Regression -> reverted.
            (req.workspace_path / "solver.py").write_text(
                "def score_value() -> int:\n    return 150\n"
                "def has_duplicate(nums: list[int]) -> bool:\n"
                "    return len(set(nums)) < len(nums)\n",
                encoding="utf-8",
            )
            return "HYPOTHESIS: Regress score to 150"
        if it == 3:
            # Real improvement meeting target_value=15 -> kept!
            (req.workspace_path / "solver.py").write_text(
                "def score_value() -> int:\n    return 12\n"
                "def has_duplicate(nums: list[int]) -> bool:\n"
                "    return len(set(nums)) < len(nums)\n",
                encoding="utf-8",
            )
            return "HYPOTHESIS: Replace O(n^2) loop with hash set and reduce score to 12"
        raise AssertionError(f"Unexpected iteration: {it}")

    runtime = WorkflowRuntime(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=_mock_provider([_sample_evidence()]),
        apply_to_workspace=True,
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="scripted",
            reviewer_driver="native",
        ),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_worker)}),
    )

    definition = load_workflow("pstack_hillclimb")
    initial_state: dict[str, Any] = {
        "run_id": run_id,
        "raw_input": "Optimize solver.py to minimize SCORE below 15 while keeping tests green",
        "hillclimb_config": _base_config(
            direction="minimize", target_value=15.0, max_iterations=5, plateau_patience=3
        ).model_dump(mode="json"),
        "verification_commands": [f"{sys.executable} -m unittest test_solver.py"],
        "_runtime": runtime,
    }
    result = run_graph(build_graph(definition), initial_state)

    assert result.get("_paused_node") is None
    report = HillclimbReport.model_validate(result["hillclimb_report"])
    assert report.baseline_metric == 100.0
    assert report.best_metric == 12.0
    assert report.kept_iterations == 1
    assert report.reverted_iterations == 2
    assert report.stopped_reason == "target_reached"
    assert [row.decision for row in report.rows] == ["rejected_tamper", "revert", "keep"]

    assert result["surface_verification"]["passed"] is True
    surfaces = {chk["surface"]: chk["status"] for chk in result["surface_verification"]["checks"]}
    assert surfaces["frozen_harness_integrity"] == "pass"
    assert surfaces["hillclimb_metric"] == "pass"
    assert surfaces["check_1"] == "pass"
    assert result["applied_paths"] == ["solver.py"]

    # Host workspace has promoted solver.py and untouched frozen bench.py!
    assert _sha256(repo / "bench.py") == orig_bench_hash
    assert snapshot_frozen_harness(repo, ["bench.py"]) == {"bench.py": orig_bench_hash}
    assert "return 12" in (repo / "solver.py").read_text(encoding="utf-8")

    bench_proc = subprocess.run(
        [sys.executable, "bench.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bench_proc.returncode == 0
    assert "SCORE=12" in bench_proc.stdout

    test_proc = subprocess.run(
        [sys.executable, "-m", "unittest", "test_solver.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert test_proc.returncode == 0

    event_types = [ev.event_type for ev in store.list_events(run_id)]
    assert "craft.evidence_recorded" in event_types
    assert "hillclimb.baseline_recorded" in event_types
    assert "hillclimb.iteration_recorded" in event_types
    assert "hillclimb.completed" in event_types
    assert "craft.verify_recorded" in event_types


def test_command_isolation_blocks_forbidden_flags_and_secret_env_leaks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_opt_repo(repo)
    host_secret_file = tmp_path / "host_secret_outside_sandbox.txt"
    host_secret_file.write_text("TOP_SECRET_HOST_DATA", encoding="utf-8")
    (repo / "bench.py").write_text(
        "import os, pathlib, solver\n"
        "leaked = (\n"
        "    os.environ.get('OLYMPUS_TEST_SECRET_TOKEN')\n"
        "    or os.environ.get('OLYMPUS_TEST_PRIVATE_KEY')\n"
        ")\n"
        "assert leaked is None, f'Secret env leaked: {leaked!r}'\n"
        f"secret_path = pathlib.Path({str(host_secret_file)!r})\n"
        "if os.environ.get('CHECK_HOST_FS_HIDDEN') != '0':\n"
        "    pass\n"
        "print(f'HOST_SECRET_EXISTS={secret_path.exists()}')\n"
        "print(f'SCORE={solver.score_value()}')\n",
        encoding="utf-8",
    )
    _git(repo, "add", "bench.py")
    _git(repo, "commit", "-m", "check env and fs isolation")

    # Forbidden flags must fail closed in _parse_safe_argv.
    for bad_cmd in (
        f"{sys.executable} -c 'print(1)'",
        f"{sys.executable} -e bench.py",
        "pytest --basetemp=/tmp/evil",
        "ruff check --fix .",
        f"{sys.executable} -oc bench.py",
    ):
        argv, issue = _parse_safe_argv(bad_cmd, host_root=repo, sandbox_root=repo)
        assert argv is None and issue is not None, f"Expected rejection for {bad_cmd!r}"

    # Host secrets must NEVER leak into benchmark subprocess env.
    monkeypatch.setenv("OLYMPUS_TEST_SECRET_TOKEN", "secret-token-must-not-leak")
    monkeypatch.setenv("OLYMPUS_TEST_PRIVATE_KEY", "private-key-must-not-leak")
    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Env isolation",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_hillclimb",
        workspace_root=repo,
    )
    stage = HillclimbStage(
        store=store,
        workspace_root=repo,
        provider=_mock_provider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
    )
    cfg = _base_config(direction="minimize", target_value=100.0, max_iterations=1)
    res = stage.hillclimb_loop(run_id=run_id, raw_input="Verify env isolation", config=cfg)
    assert HillclimbReport.model_validate(res["hillclimb_report"]).baseline_metric == 100.0

    # Verify bwrap uses minimal ro mounts (NO --ro-bind / /), hides host files, uses --unshare-net
    # and --ro-bind on frozen_harness_paths, and fails closed if --unshare-net is unavailable.
    if _probe_bwrap() is not None and _probe_bwrap_unshare_net():
        captured_cmds: list[list[str]] = []
        real_run = subprocess.run

        def _spy_run(cmd: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(cmd, list) and cmd and Path(cmd[0]).name == "bwrap":
                captured_cmds.append(list(cmd))
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(hillclimb_mod.subprocess, "run", _spy_run)
        val, err, excerpt = stage._measure_metric(
            sandbox_path=repo,
            config=cfg,
            timeout_s=30,
            node_name="hillclimb",
        )
        assert err is None and val == 100.0
        assert "HOST_SECRET_EXISTS=False" in excerpt
        assert captured_cmds
        bwrap_cmd = captured_cmds[-1]
        assert "--unshare-net" in bwrap_cmd
        assert "--unshare-pid" in bwrap_cmd
        assert "--tmpfs" in bwrap_cmd
        bench_resolved = str((repo / "bench.py").resolve())
        ro_pairs = [
            (bwrap_cmd[i + 1], bwrap_cmd[i + 2])
            for i in range(len(bwrap_cmd) - 2)
            if bwrap_cmd[i] == "--ro-bind"
        ]
        assert ("/", "/") not in ro_pairs
        assert (bench_resolved, bench_resolved) in ro_pairs

        # Fail closed when --unshare-net is unavailable under require_fs_isolation=True.
        monkeypatch.setattr(hillclimb_mod, "_probe_bwrap_unshare_net", lambda: False)
        strict_stage = HillclimbStage(
            store=store,
            workspace_root=repo,
            provider=_mock_provider([]),
            artifacts_root=tmp_path / "artifacts",
            sandbox_root=tmp_path / "sandboxes",
            allow_unsandboxed_metric=False,
        )
        with pytest.raises(HillclimbPause, match="unshare-net"):
            strict_stage._measure_metric(
                sandbox_path=repo,
                config=cfg,
                timeout_s=30,
                node_name="hillclimb",
            )

        # Fail closed when --unshare-pid is unavailable under require_fs_isolation=True.
        monkeypatch.setattr(hillclimb_mod, "_probe_bwrap_unshare_net", lambda: True)
        monkeypatch.setattr(hillclimb_mod, "_probe_bwrap_unshare_pid", lambda: False)
        with pytest.raises(HillclimbPause, match="unshare-pid"):
            strict_stage._measure_metric(
                sandbox_path=repo,
                config=cfg,
                timeout_s=30,
                node_name="hillclimb",
            )


def test_bwrap_mounts_multi_hop_symlinked_interpreter_prefixes_and_hides_host_pids(
    tmp_path: Path,
) -> None:
    if _probe_bwrap() is None or not _probe_bwrap_unshare_net() or not _probe_bwrap_unshare_pid():
        pytest.skip("bwrap with --unshare-net and --unshare-pid unavailable")

    repo = tmp_path / "repo"
    _init_opt_repo(repo)
    host_secret = tmp_path / "unrelated_secret.txt"
    host_secret.write_text("SECRET", encoding="utf-8")
    (repo / "bench.py").write_text(
        "import os, pathlib, solver\n"
        f"print('HOST_SECRET=' + str(pathlib.Path({str(host_secret)!r}).exists()))\n"
        "pids = [p for p in os.listdir('/proc') if p.isdigit()]\n"
        "print(f'PROC_COUNT={len(pids)}')\n"
        "print(f'SCORE={solver.score_value()}')\n",
        encoding="utf-8",
    )

    # Simulate a venv python symlink traversing a symlinked directory into another prefix
    # (mirroring GitHub Actions uv/toolcache layouts).
    real_py_prefix = tmp_path / "ext_real_py"
    (real_py_prefix / "bin").mkdir(parents=True)
    os.symlink(sys.executable, real_py_prefix / "bin" / "python3")
    alias_py_prefix = tmp_path / "ext_alias_py"
    os.symlink(real_py_prefix, alias_py_prefix)
    venv_prefix = tmp_path / "ext_venv"
    (venv_prefix / "bin").mkdir(parents=True)
    venv_python = venv_prefix / "bin" / "python"
    os.symlink(alias_py_prefix / "bin" / "python3", venv_python)

    code, output = _run_argv_command(
        [str(venv_python), "bench.py"],
        cwd=repo,
        timeout_s=30,
        host_root=repo,
        require_fs_isolation=True,
        read_only_files=[repo / "bench.py"],
    )
    assert code == 0, output
    assert "SCORE=100" in output
    assert "HOST_SECRET=False" in output
    # Under --unshare-pid, /proc only exposes the tiny sandbox PID namespace (<= 5 PIDs).
    proc_line = next(line for line in output.splitlines() if line.startswith("PROC_COUNT="))
    assert int(proc_line.split("=", 1)[1]) <= 5


def test_pre_execution_tamper_short_circuits_without_running_metric_and_unsandboxed_runtime_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_opt_repo(repo)
    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Short-circuit tamper",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_hillclimb",
        workspace_root=repo,
    )

    metric_invocations: list[str] = []

    def _spy_runner(cmd: str, *, cwd: Path, timeout_s: int = 60) -> tuple[int | None, str]:
        del timeout_s
        metric_invocations.append(cmd)
        proc = subprocess.run(
            [sys.executable, "bench.py"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode, proc.stdout + proc.stderr

    sandbox_paths: list[Path] = []

    def _worker(req: DriverTurnRequest) -> str:
        sandbox_paths.append(req.workspace_path)
        if req.actor.endswith("_1"):
            (req.workspace_path / "bench.py").write_text("print('SCORE=1')\n", encoding="utf-8")
            return "HYPOTHESIS: Tamper bench before metric run"
        if req.actor.endswith("_2"):
            (req.workspace_path / "solver.py").write_text(
                "def score_value() -> int:\n    return 40\n"
                "def has_duplicate(nums: list[int]) -> bool:\n"
                "    return len(set(nums)) < len(nums)\n",
                encoding="utf-8",
            )
            return "HYPOTHESIS: Valid improvement"
        raise AssertionError(req.actor)

    stage = HillclimbStage(
        store=store,
        workspace_root=repo,
        provider=_mock_provider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        execution_drivers=ExecutionDriversConfig.resolve(worker_driver="scripted"),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_worker)}),
        command_runner=_spy_runner,
    )
    res = stage.hillclimb_loop(
        run_id=run_id,
        raw_input="Test tamper short-circuit",
        config=_base_config(direction="minimize", max_iterations=2, plateau_patience=3),
    )
    # Only baseline (1) + iter 2 (1) ran metric_command; tampered iter 1 NEVER executed it!
    assert len(metric_invocations) == 2
    assert [r["decision"] for r in res["hillclimb_report"]["rows"]] == ["rejected_tamper", "keep"]

    # Now tamper bench.py before verify_surface: metric_command MUST NOT execute!
    metric_invocations.clear()
    (sandbox_paths[-1] / "bench.py").write_text("print('SCORE=0')\n", encoding="utf-8")
    with pytest.raises(HillclimbPause, match="frozen_harness_integrity=fail"):
        stage.verify_surface(run_id=run_id, state=res, apply_to_workspace=True)
    assert metric_invocations == []
    assert "return 100" in (repo / "solver.py").read_text(encoding="utf-8")

    # Without bwrap (opt-in unsandboxed), runtime tamper is still caught as rejected_tamper.
    monkeypatch.setattr(hillclimb_mod, "_probe_bwrap", lambda: None)
    run_rt = store.create_run(
        title="Unsandboxed runtime tamper",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_hillclimb",
        workspace_root=repo,
    )

    def _rt_worker(req: DriverTurnRequest) -> str:
        (req.workspace_path / "solver.py").write_text(
            "from pathlib import Path\n"
            "def score_value() -> int:\n"
            "    Path('bench.py').write_text('print(\"SCORE=0\")\\n', encoding='utf-8')\n"
            "    return 5\n"
            "def has_duplicate(nums: list[int]) -> bool:\n"
            "    return len(set(nums)) < len(nums)\n",
            encoding="utf-8",
        )
        return "HYPOTHESIS: Mutate bench.py during runtime"

    rt_res = hillclimb_loop(
        store=store,
        workspace_root=repo,
        provider=_mock_provider([]),
        run_id=run_rt,
        raw_input="Runtime tamper fallback check",
        config=_base_config(direction="minimize", max_iterations=1, plateau_patience=2),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes_rt",
        execution_drivers=ExecutionDriversConfig.resolve(worker_driver="scripted"),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_rt_worker)}),
        allow_unsandboxed_metric=True,
    )
    assert rt_res["hillclimb_report"]["rows"][0]["decision"] == "rejected_tamper"


def test_verify_surface_enforces_winning_commit_best_metric_and_clean_sandbox(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_opt_repo(repo)
    (repo / "mutating_check.py").write_text(
        "from pathlib import Path\n"
        "Path('verify_side_effect.txt').write_text('mutated', encoding='utf-8')\n",
        encoding="utf-8",
    )
    _git(repo, "add", "mutating_check.py")
    _git(repo, "commit", "-m", "add mutating check script")

    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Verify binding and cleanliness",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_hillclimb",
        workspace_root=repo,
    )
    sandboxes: list[Path] = []

    def _worker(req: DriverTurnRequest) -> str:
        sandboxes.append(req.workspace_path)
        (req.workspace_path / "solver.py").write_text(
            "def score_value() -> int:\n    return 30\n"
            "def has_duplicate(nums: list[int]) -> bool:\n"
            "    return len(set(nums)) < len(nums)\n",
            encoding="utf-8",
        )
        return "HYPOTHESIS: Lower score to 30"

    stage = HillclimbStage(
        store=store,
        workspace_root=repo,
        provider=_mock_provider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        execution_drivers=ExecutionDriversConfig.resolve(worker_driver="scripted"),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_worker)}),
    )
    res = stage.hillclimb_loop(
        run_id=run_id,
        raw_input="Optimize to 30",
        config=_base_config(direction="minimize", max_iterations=1),
    )
    ws = sandboxes[-1]
    winning_sha = res["hillclimb_winning_commit_sha"]

    # 0. Direct promote_winner MUST fail closed before Argus verify_surface has passed!
    with pytest.raises(HillclimbPause, match="missing passing Argus verify_surface"):
        stage.promote_winner(run_id=run_id, state=res)
    assert "return 100" in (repo / "solver.py").read_text(encoding="utf-8")

    # 1. Extra verification command mutating sandbox (including .gitignore'd files!) fails.
    with pytest.raises(HillclimbPause, match="sandbox_cleanliness=fail"):
        stage.verify_surface(
            run_id=run_id,
            state=res,
            commands=[f"{sys.executable} mutating_check.py"],
            apply_to_workspace=True,
        )
    assert "return 100" in (repo / "solver.py").read_text(encoding="utf-8")
    (ws / "verify_side_effect.txt").unlink()

    # 2a. Forged in-memory HillclimbReport disagreeing with SQLite fails closed immediately!
    fake_worse_state = dict(res)
    fake_worse_report = dict(res["hillclimb_report"])
    fake_worse_report["best_metric"] = 10.0
    fake_worse_state["hillclimb_report"] = fake_worse_report
    with pytest.raises(HillclimbPause, match="disagrees with authoritative SQLite"):
        stage.verify_surface(run_id=run_id, state=fake_worse_state, apply_to_workspace=True)
    assert "return 100" in (repo / "solver.py").read_text(encoding="utf-8")

    # 2b. Altered HillclimbConfig (config_hash mismatch vs SQLite) fails closed!
    bad_cfg_state = dict(res)
    bad_cfg_state["hillclimb_config"] = _base_config(
        direction="maximize", max_iterations=1
    ).model_dump(mode="json")
    with pytest.raises(HillclimbPause, match="does not match SQLite recorded baseline config_hash"):
        stage.verify_surface(run_id=run_id, state=bad_cfg_state, apply_to_workspace=True)

    # 2c. Real metric regression worse than SQLite best_metric (50 vs 30, still < 100 baseline)!
    regressing_stage = HillclimbStage(
        store=store,
        workspace_root=repo,
        provider=_mock_provider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        command_runner=lambda _cmd, **_kw: (0, "SCORE=50\n"),
    )
    with pytest.raises(HillclimbPause, match="hillclimb_metric=fail"):
        regressing_stage.verify_surface(run_id=run_id, state=res, apply_to_workspace=True)
    assert "return 100" in (repo / "solver.py").read_text(encoding="utf-8")

    # 3. Drifted sandbox HEAD (extra commit after winning commit) fails winning_commit_binding!
    (ws / "solver.py").write_text(
        "def score_value() -> int:\n    return 5\n"
        "def has_duplicate(nums: list[int]) -> bool:\n"
        "    return len(set(nums)) < len(nums)\n",
        encoding="utf-8",
    )
    _git(ws, "add", "solver.py")
    _git(ws, "commit", "-m", "unauthorized post-loop commit")
    with pytest.raises(HillclimbPause, match="winning_commit_binding=fail"):
        stage.verify_surface(run_id=run_id, state=res, apply_to_workspace=True)
    assert "return 100" in (repo / "solver.py").read_text(encoding="utf-8")

    # Restoring exact winning_sha passes verification and promotes cleanly.
    _git(ws, "reset", "--hard", winning_sha)
    ok = stage.verify_surface(run_id=run_id, state=res, apply_to_workspace=True)
    assert ok["surface_verification"]["passed"] is True
    assert ok["applied_paths"] == ["solver.py"]
    assert "return 30" in (repo / "solver.py").read_text(encoding="utf-8")


def test_atomic_promotion_symlink_delete_hardlink_isolation_and_dir_mode_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --- A. Hard-link alias must NOT be mutated in place on file overwrite ---
    repo_hl = tmp_path / "repo_hl"
    _init_opt_repo(repo_hl)
    alias = tmp_path / "solver_hardlink_alias.py"
    os.link(repo_hl / "solver.py", alias)
    orig_alias_content = alias.read_text(encoding="utf-8")

    store = RunStore(tmp_path / "olympus.sqlite")
    run_hl = store.create_run(
        title="Hardlink isolation",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_hillclimb",
        workspace_root=repo_hl,
    )

    def _hl_worker(req: DriverTurnRequest) -> str:
        (req.workspace_path / "solver.py").write_text(
            "def score_value() -> int:\n    return 20\n"
            "def has_duplicate(nums: list[int]) -> bool:\n"
            "    return len(set(nums)) < len(nums)\n",
            encoding="utf-8",
        )
        return "HYPOTHESIS: Improve to 20"

    hillclimb_loop(
        store=store,
        workspace_root=repo_hl,
        provider=_mock_provider([]),
        run_id=run_hl,
        raw_input="Test hardlink preservation",
        config=_base_config(direction="minimize", max_iterations=1),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes_hl",
        execution_drivers=ExecutionDriversConfig.resolve(worker_driver="scripted"),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_hl_worker)}),
        apply_to_workspace=True,
    )
    assert "return 20" in (repo_hl / "solver.py").read_text(encoding="utf-8")
    assert alias.read_text(encoding="utf-8") == orig_alias_content

    # --- B. Deleting through an in-repo symlinked directory MUST be rejected ---
    repo_sym = tmp_path / "repo_sym"
    _init_opt_repo(repo_sym)
    (repo_sym / "real_dir").mkdir()
    (repo_sym / "real_dir" / "victim.txt").write_text("keep-me", encoding="utf-8")
    (repo_sym / "sub").mkdir()
    (repo_sym / "sub" / "victim.txt").write_text("tracked", encoding="utf-8")
    _git(repo_sym, "add", ".")
    _git(repo_sym, "commit", "-m", "add real_dir and sub")

    run_sym = store.create_run(
        title="Symlink delete guard",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_hillclimb",
        workspace_root=repo_sym,
    )
    stage_sym = HillclimbStage(
        store=store,
        workspace_root=repo_sym,
        provider=_mock_provider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes_sym",
    )
    handle_sym, baseline_sym = stage_sym._ensure_sandbox(run_sym)
    # Delete sub/victim.txt in sandbox and replace host `sub` with an in-repo symlink to `real_dir`.
    (handle_sym.workspace_path / "sub" / "victim.txt").unlink()
    (repo_sym / "sub" / "victim.txt").unlink()
    (repo_sym / "sub").rmdir()
    os.symlink("real_dir", repo_sym / "sub")

    sym_cfg = HillclimbConfig(
        frozen_harness_paths=["bench.py"],
        metric_command=f"{sys.executable} bench.py",
        metric_regex=r"SCORE=([0-9]+)",
        direction="minimize",
        write_scope=["sub/**"],
    )
    frozen_sym = snapshot_frozen_harness(handle_sym.workspace_path, ["bench.py"])
    with pytest.raises(HillclimbPause, match="symlinked ancestor directory"):
        stage_sym._promote_sandbox_to_workspace(
            handle_sym, baseline_sym, sym_cfg, frozen_hashes=frozen_sym
        )
    assert (repo_sym / "real_dir" / "victim.txt").read_text(encoding="utf-8") == "keep-me"

    # --- C. Rollback restores removed empty parent directory mode bits ---
    repo_rb = tmp_path / "repo_rb"
    _init_opt_repo(repo_rb)
    nested_dir = repo_rb / "nested_mod"
    nested_dir.mkdir(mode=0o750)
    os.chmod(nested_dir, 0o750)
    (nested_dir / "obsolete.py").write_text("OLD = 1\n", encoding="utf-8")
    _git(repo_rb, "add", ".")
    _git(repo_rb, "commit", "-m", "add nested_mod")
    os.chmod(nested_dir, 0o750)

    run_rb = store.create_run(
        title="Dir mode rollback",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_hillclimb",
        workspace_root=repo_rb,
    )
    stage_rb = HillclimbStage(
        store=store,
        workspace_root=repo_rb,
        provider=_mock_provider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes_rb",
    )
    handle_rb, baseline_rb = stage_rb._ensure_sandbox(run_rb)
    # Sandbox deletes nested_mod/obsolete.py (causing host rmdir) and modifies solver.py.
    (handle_rb.workspace_path / "nested_mod" / "obsolete.py").unlink()
    (handle_rb.workspace_path / "solver.py").write_text(
        "def score_value() -> int:\n    return 10\n"
        "def has_duplicate(nums: list[int]) -> bool:\n"
        "    return len(set(nums)) < len(nums)\n",
        encoding="utf-8",
    )
    rb_cfg = HillclimbConfig(
        frozen_harness_paths=["bench.py"],
        metric_command=f"{sys.executable} bench.py",
        metric_regex=r"SCORE=([0-9]+)",
        direction="minimize",
        write_scope=["solver.py", "nested_mod/**"],
    )
    frozen_rb = snapshot_frozen_harness(handle_rb.workspace_path, ["bench.py"])

    def _boom_replace(_src: Path, _dst: Path) -> None:
        raise OSError("Simulated disk full during copy_plan")

    monkeypatch.setattr(hillclimb_mod, "_atomic_replace_from_src", _boom_replace)
    with pytest.raises(HillclimbPause, match="rolled back"):
        stage_rb._promote_sandbox_to_workspace(
            handle_rb, baseline_rb, rb_cfg, frozen_hashes=frozen_rb
        )
    assert nested_dir.is_dir()
    assert stat.S_IMODE(nested_dir.stat().st_mode) == 0o750
    assert (nested_dir / "obsolete.py").read_text(encoding="utf-8") == "OLD = 1\n"


def test_revert_cleans_gitignored_artifacts_and_resume_preserves_sqlite_trail(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_opt_repo(repo)
    (repo / ".gitignore").write_text("*.ignored_cache\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "add gitignore")

    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="Git clean -fdx and resume trail",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_hillclimb",
        workspace_root=repo,
    )
    seen_ignored_on_iter2: list[bool] = []
    seen_actors: list[str] = []

    def _worker_part1(req: DriverTurnRequest) -> str:
        seen_actors.append(req.actor)
        it = int(req.actor.rsplit("_", 1)[-1])
        if it == 1:
            # Regress and drop a .gitignore'd file; git clean -fdx must purge it!
            (req.workspace_path / "leak.ignored_cache").write_text("stale", encoding="utf-8")
            (req.workspace_path / "solver.py").write_text(
                "def score_value() -> int:\n    return 200\n"
                "def has_duplicate(nums: list[int]) -> bool:\n"
                "    return len(set(nums)) < len(nums)\n",
                encoding="utf-8",
            )
            return "HYPOTHESIS: Regress with gitignored artifact"
        if it == 2:
            seen_ignored_on_iter2.append((req.workspace_path / "leak.ignored_cache").exists())
            (req.workspace_path / "solver.py").write_text(
                "def score_value() -> int:\n    return 50\n"
                "def has_duplicate(nums: list[int]) -> bool:\n"
                "    return len(set(nums)) < len(nums)\n",
                encoding="utf-8",
            )
            return "HYPOTHESIS: Improve to 50"
        if it == 3:
            # Simulate an unexpected mid-loop worker crash after iter 1 & 2 are persisted.
            raise RuntimeError("transient provider outage on iter 3")
        raise AssertionError(req.actor)

    shared_cfg = _base_config(direction="minimize", max_iterations=3, plateau_patience=3)
    with pytest.raises(RuntimeError, match="transient provider outage"):
        hillclimb_loop(
            store=store,
            workspace_root=repo,
            provider=_mock_provider([]),
            run_id=run_id,
            raw_input="Run 3 iterations with crash on iter 3",
            config=shared_cfg,
            artifacts_root=tmp_path / "artifacts",
            sandbox_root=tmp_path / "sandboxes",
            execution_drivers=ExecutionDriversConfig.resolve(worker_driver="scripted"),
            driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_worker_part1)}),
        )
    assert seen_ignored_on_iter2 == [False]

    # Resuming with a DIFFERENT config_hash MUST fail closed!
    drifted_cfg = _base_config(direction="maximize", max_iterations=3, plateau_patience=3)
    with pytest.raises(HillclimbPause, match="config_hash"):
        hillclimb_loop(
            store=store,
            workspace_root=repo,
            provider=_mock_provider([]),
            run_id=run_id,
            raw_input="Attempt resume with mutated config",
            config=drifted_cfg,
            artifacts_root=tmp_path / "artifacts",
            sandbox_root=tmp_path / "sandboxes",
            execution_drivers=ExecutionDriversConfig.resolve(worker_driver="scripted"),
            driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_worker_part1)}),
        )

    # Resume the SAME run_id with the SAME config_hash: continues at iteration 3 (no duplicates!).
    def _worker_part2(req: DriverTurnRequest) -> str:
        seen_actors.append(req.actor)
        assert req.actor == "hephaestus:hillclimb_iter_3"
        (req.workspace_path / "solver.py").write_text(
            "def score_value() -> int:\n    return 15\n"
            "def has_duplicate(nums: list[int]) -> bool:\n"
            "    return len(set(nums)) < len(nums)\n",
            encoding="utf-8",
        )
        return "HYPOTHESIS: Resumed iter 3 improves to 15"

    res2 = hillclimb_loop(
        store=store,
        workspace_root=repo,
        provider=_mock_provider([]),
        run_id=run_id,
        raw_input="Resume to 3 iterations",
        config=shared_cfg,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        execution_drivers=ExecutionDriversConfig.resolve(worker_driver="scripted"),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_worker_part2)}),
    )
    rep2 = HillclimbReport.model_validate(res2["hillclimb_report"])
    assert [r.iteration for r in rep2.rows] == [1, 2, 3]
    assert [r.decision for r in rep2.rows] == ["revert", "keep", "keep"]
    assert rep2.best_metric == 15.0
    assert seen_actors == [
        "hephaestus:hillclimb_iter_1",
        "hephaestus:hillclimb_iter_2",
        "hephaestus:hillclimb_iter_3",
        "hephaestus:hillclimb_iter_3",
    ]

    events = store.list_events(run_id)
    base_events = [ev for ev in events if ev.event_type == "hillclimb.baseline_recorded"]
    assert len(base_events) == 1
    assert base_events[0].payload["config_hash"] == shared_cfg.content_hash
    assert base_events[0].payload["metric_command"] == shared_cfg.metric_command
    assert base_events[0].payload["metric_regex"] == shared_cfg.metric_regex
    assert base_events[0].payload["frozen_harness_hashes"]
    iter_events = [ev for ev in events if ev.event_type == "hillclimb.iteration_recorded"]
    assert [ev.payload["iteration"] for ev in iter_events] == [1, 2, 3]
    assert iter_events[-1].payload["best_metric"] == 15.0
    assert iter_events[-1].payload["config_hash"] == shared_cfg.content_hash

    trail = json.loads(Path(res2["hillclimb_decision_trail_path"]).read_text(encoding="utf-8"))
    assert [r["iteration"] for r in trail["rows"]] == [1, 2, 3]


def test_metric_parsing_uses_untruncated_output_and_rejects_ambiguous_multi_match(
    tmp_path: Path,
) -> None:
    # 1. Conflicting multi-match output fails closed; identical repeated matches succeed.
    val_amb, err_amb = parse_metric_value("SCORE=100\nSCORE=42\n", r"SCORE=([0-9]+)")
    assert val_amb is None and err_amb is not None and "ambiguous" in err_amb

    val_same, err_same = parse_metric_value("SCORE=42\nSCORE=42\n", r"SCORE=([0-9]+)")
    assert err_same is None and val_same == 42.0

    # 2. Metric emitted AFTER >2400 chars of stdout noise is still parsed cleanly.
    repo = tmp_path / "repo"
    _init_opt_repo(repo)
    (repo / "bench.py").write_text(
        "import solver\n"
        "print('X' * 5000)\n"
        "print(f'SCORE={solver.score_value()}')\n",
        encoding="utf-8",
    )
    _git(repo, "add", "bench.py")
    _git(repo, "commit", "-m", "noisy stdout before metric")

    store = RunStore(tmp_path / "olympus.sqlite")
    stage = HillclimbStage(
        store=store,
        workspace_root=repo,
        provider=_mock_provider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
    )
    val, err, excerpt = stage._measure_metric(
        sandbox_path=repo,
        config=_base_config(direction="minimize", max_iterations=1),
        timeout_s=30,
        node_name="hillclimb",
    )
    assert err is None
    assert val == 100.0
    assert excerpt.endswith("...[truncated]")


def test_host_drift_detection_and_hardlink_inode_rollback_and_ignored_verify_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path / "olympus.sqlite")

    # 1. Host drift protection: local edit on host during hillclimb blocks promotion!
    repo_drift = tmp_path / "repo_drift"
    _init_opt_repo(repo_drift)
    run_drift = store.create_run(
        title="Host drift",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_hillclimb",
        workspace_root=repo_drift,
    )

    def _drift_worker(req: DriverTurnRequest) -> str:
        (req.workspace_path / "solver.py").write_text(
            "def score_value() -> int:\n    return 18\n"
            "def has_duplicate(nums: list[int]) -> bool:\n"
            "    return len(set(nums)) < len(nums)\n",
            encoding="utf-8",
        )
        return "HYPOTHESIS: Improve to 18"

    stage_drift = HillclimbStage(
        store=store,
        workspace_root=repo_drift,
        provider=_mock_provider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes_drift",
        execution_drivers=ExecutionDriversConfig.resolve(worker_driver="scripted"),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_drift_worker)}),
    )
    res_drift = stage_drift.hillclimb_loop(
        run_id=run_drift,
        raw_input="Optimize solver",
        config=_base_config(direction="minimize", max_iterations=1),
    )
    # Human modifies solver.py locally on host before verify_surface promotes.
    (repo_drift / "solver.py").write_text("# local human edit\n", encoding="utf-8")
    with pytest.raises(HillclimbPause, match="host workspace drifted"):
        stage_drift.verify_surface(run_id=run_drift, state=res_drift, apply_to_workspace=True)
    assert (repo_drift / "solver.py").read_text(encoding="utf-8") == "# local human edit\n"

    # 2. Hard-link inode preservation on rollback!
    repo_hl = tmp_path / "repo_hl_rb"
    _init_opt_repo(repo_hl)
    (repo_hl / "second.py").write_text("VAL = 1\n", encoding="utf-8")
    _git(repo_hl, "add", "second.py")
    _git(repo_hl, "commit", "-m", "add second.py")
    alias = tmp_path / "solver_rb_alias.py"
    os.link(repo_hl / "solver.py", alias)
    orig_ino = alias.stat().st_ino

    run_hl = store.create_run(
        title="Hardlink rollback inode",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_hillclimb",
        workspace_root=repo_hl,
    )
    stage_hl = HillclimbStage(
        store=store,
        workspace_root=repo_hl,
        provider=_mock_provider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes_hl_rb",
    )
    handle_hl, baseline_hl = stage_hl._ensure_sandbox(run_hl)
    (handle_hl.workspace_path / "solver.py").write_text("def score_value(): return 5\n", "utf-8")
    (handle_hl.workspace_path / "second.py").write_text("VAL = 2\n", "utf-8")
    hl_cfg = HillclimbConfig(
        frozen_harness_paths=["bench.py"],
        metric_command=f"{sys.executable} bench.py",
        metric_regex=r"SCORE=([0-9]+)",
        direction="minimize",
        write_scope=["solver.py", "second.py"],
    )
    frozen_hl = snapshot_frozen_harness(handle_hl.workspace_path, ["bench.py"])

    real_replace = hillclimb_mod._atomic_replace_from_src
    call_count = 0

    def _fail_on_second(src: Path, dst: Path) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("Simulated failure on second file")
        real_replace(src, dst)

    monkeypatch.setattr(hillclimb_mod, "_atomic_replace_from_src", _fail_on_second)
    with pytest.raises(HillclimbPause, match="rolled back"):
        stage_hl._promote_sandbox_to_workspace(
            handle_hl, baseline_hl, hl_cfg, frozen_hashes=frozen_hl
        )
    # Rollback MUST restore the EXACT original inode so solver.py and alias remain hard-linked!
    assert (repo_hl / "solver.py").stat().st_ino == orig_ino
    assert alias.stat().st_ino == orig_ino
    assert not list(repo_hl.glob(".olympus-hlbak-*"))

    # 3. Extra verification command writing a .gitignore'd file fails sandbox_cleanliness!
    repo_ign = tmp_path / "repo_ign"
    _init_opt_repo(repo_ign)
    (repo_ign / ".gitignore").write_text("*.verify_ignored\n", encoding="utf-8")
    (repo_ign / "write_ignored.py").write_text(
        "from pathlib import Path\n"
        "Path('sneaky.verify_ignored').write_text('x', encoding='utf-8')\n",
        encoding="utf-8",
    )
    _git(repo_ign, "add", ".")
    _git(repo_ign, "commit", "-m", "add gitignore and write_ignored")
    run_ign = store.create_run(
        title="Ignored verify mutation",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_hillclimb",
        workspace_root=repo_ign,
    )
    stage_ign = HillclimbStage(
        store=store,
        workspace_root=repo_ign,
        provider=_mock_provider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes_ign",
        execution_drivers=ExecutionDriversConfig.resolve(worker_driver="scripted"),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_drift_worker)}),
    )
    res_ign = stage_ign.hillclimb_loop(
        run_id=run_ign,
        raw_input="Optimize",
        config=_base_config(direction="minimize", max_iterations=1),
    )
    with pytest.raises(HillclimbPause, match="sandbox_cleanliness=fail"):
        stage_ign.verify_surface(
            run_id=run_ign,
            state=res_ign,
            commands=[f"{sys.executable} write_ignored.py"],
            apply_to_workspace=True,
        )


def test_session_runner_starts_and_resumes_pstack_hillclimb_from_verify_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_opt_repo(repo)
    # Add a toggleable external test gate so verify_surface pauses on run 1 and passes on resume!
    gate_file = tmp_path / "allow_verify_pass.flag"
    (repo / "check_gate.py").write_text(
        "import pathlib, sys\n"
        f"sys.exit(0 if pathlib.Path({str(gate_file)!r}).exists() else 1)\n",
        encoding="utf-8",
    )
    _git(repo, "add", "check_gate.py")
    _git(repo, "commit", "-m", "add toggleable verify gate")

    # Note: under minimal bwrap, tmp_path outside sandbox is hidden, so use allow_unsandboxed_metric
    # for this toggle-file gate check while still exercising full SessionRunner + loader dispatch.
    monkeypatch.setenv("OLYMPUS_ALLOW_UNSANDBOXED_REPRO", "1")
    monkeypatch.setattr(hillclimb_mod, "_probe_bwrap", lambda: None)

    def _worker(req: DriverTurnRequest) -> str:
        (req.workspace_path / "solver.py").write_text(
            "def score_value() -> int:\n    return 14\n"
            "def has_duplicate(nums: list[int]) -> bool:\n"
            "    return len(set(nums)) < len(nums)\n",
            encoding="utf-8",
        )
        return "HYPOTHESIS: Optimize score to 14"

    scripted = _ScriptedDriver(_worker)
    orig_factories = hillclimb_mod.default_driver_factories

    def _factories_with_scripted(*args: Any, **kwargs: Any) -> dict[str, Any]:
        factories = dict(orig_factories(*args, **kwargs))
        factories["scripted"] = lambda: scripted
        return factories

    monkeypatch.setattr(hillclimb_mod, "default_driver_factories", _factories_with_scripted)

    hc_cfg = _base_config(direction="minimize", target_value=15.0, max_iterations=2)
    state_dir = tmp_path / "session_state"
    runner = SessionRunner(
        SessionConfig(
            workspace_root=repo,
            state_dir=state_dir,
            workflow="pstack_hillclimb",
            step_mode=StepMode.EXECUTE,
            stage_driver="native",
            worker_driver="scripted",
            reviewer_driver="native",
            hillclimb_config=hc_cfg.model_dump(mode="json"),
            metadata={"verification_commands": [f"{sys.executable} check_gate.py"]},
            provider_builder=lambda **_kw: _mock_provider([_sample_evidence()]),
        )
    )

    env1 = runner.start(goal="Optimize solver.py via SessionRunner")
    assert env1.status == "needs_input"
    assert env1.pause_kind == "blocked"
    paused_events = [
        ev for ev in runner.store.list_events(env1.run_id) if ev.event_type == "session.paused"
    ]
    assert paused_events and paused_events[-1].payload.get("node") == "verify_surface"
    assert "return 100" in (repo / "solver.py").read_text(encoding="utf-8")

    # Now enable the external check gate and resume using a brand-new SessionRunner instance
    # (no in-memory hillclimb_report in state!).
    gate_file.write_text("ok", encoding="utf-8")
    resumed_runner = SessionRunner(
        SessionConfig(
            workspace_root=repo,
            state_dir=state_dir,
            workflow="pstack_hillclimb",
            step_mode=StepMode.EXECUTE,
            stage_driver="native",
            worker_driver="scripted",
            reviewer_driver="native",
            provider_builder=lambda **_kw: _mock_provider([]),
        )
    )
    env2 = resumed_runner.reply(run_id=env1.run_id, message="retry")
    assert env2.status == "completed", env2
    assert "return 14" in (repo / "solver.py").read_text(encoding="utf-8")
    # Worker was NOT re-invoked on resume; loader routed straight to HillclimbStage.verify_surface!
    assert len(scripted.calls) == 1


def test_latest_argus_gate_host_mode_drift_and_session_start_explicit_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path / "olympus.sqlite")

    # 1. Promotion gate MUST honor ONLY the LATEST actor=='argus' verify_recorded event.
    repo_gate = tmp_path / "repo_gate"
    _init_opt_repo(repo_gate)
    run_gate = store.create_run(
        title="Latest argus gate",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_hillclimb",
        workspace_root=repo_gate,
    )

    def _ok_worker(req: DriverTurnRequest) -> str:
        (req.workspace_path / "solver.py").write_text(
            "def score_value() -> int:\n    return 22\n"
            "def has_duplicate(nums: list[int]) -> bool:\n"
            "    return len(set(nums)) < len(nums)\n",
            encoding="utf-8",
        )
        return "HYPOTHESIS: Improve to 22"

    stage_gate = HillclimbStage(
        store=store,
        workspace_root=repo_gate,
        provider=_mock_provider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes_gate",
        execution_drivers=ExecutionDriversConfig.resolve(worker_driver="scripted"),
        driver_registry=DriverRegistry({"scripted": _ScriptedDriver(_ok_worker)}),
    )
    cfg_gate = _base_config(direction="minimize", max_iterations=1)
    res_gate = stage_gate.hillclimb_loop(run_id=run_gate, raw_input="Opt", config=cfg_gate)
    win_sha = res_gate["hillclimb_winning_commit_sha"]

    # Record a passing Argus verification, then a LATER failing Argus verification, then a
    # non-argus fake pass — promote_winner MUST reject!
    stage_gate.verify_surface(run_id=run_gate, state=res_gate, apply_to_workspace=False)
    store.append_event(
        run_id=run_gate,
        stage="hillclimb",
        event_type="craft.verify_recorded",
        actor="argus",
        payload={
            "passed": False,
            "winning_commit_sha": win_sha,
            "config_hash": cfg_gate.content_hash,
            "summary": "Later argus verification failed",
        },
    )
    store.append_event(
        run_id=run_gate,
        stage="hillclimb",
        event_type="craft.verify_recorded",
        actor="hephaestus",
        payload={
            "passed": True,
            "winning_commit_sha": win_sha,
            "config_hash": cfg_gate.content_hash,
            "summary": "Spoofed non-argus pass",
        },
    )
    with pytest.raises(HillclimbPause, match="missing passing Argus verify_surface"):
        stage_gate.promote_winner(run_id=run_gate, state=res_gate)
    assert "return 100" in (repo_gate / "solver.py").read_text(encoding="utf-8")

    # 2. Host permission mode drift (chmod 0o600 on host solver.py) MUST block promotion!
    os.chmod(repo_gate / "solver.py", 0o600)
    with pytest.raises(HillclimbPause, match="host workspace drifted"):
        stage_gate.verify_surface(run_id=run_gate, state=res_gate, apply_to_workspace=True)
    assert stat.S_IMODE((repo_gate / "solver.py").stat().st_mode) == 0o600
    assert "return 100" in (repo_gate / "solver.py").read_text(encoding="utf-8")

    # 3. SessionRunner.start(..., hillclimb_config=explicit_cfg) overrides SessionConfig default
    # consistently for both the initial run and a resumed SessionRunner!
    repo_sess = tmp_path / "repo_sess"
    _init_opt_repo(repo_sess)
    gate_file = tmp_path / "override_gate.flag"
    (repo_sess / "check_gate.py").write_text(
        "import pathlib, sys\n"
        f"sys.exit(0 if pathlib.Path({str(gate_file)!r}).exists() else 1)\n",
        encoding="utf-8",
    )
    _git(repo_sess, "add", "check_gate.py")
    _git(repo_sess, "commit", "-m", "add override gate")

    monkeypatch.setenv("OLYMPUS_ALLOW_UNSANDBOXED_REPRO", "1")
    monkeypatch.setattr(hillclimb_mod, "_probe_bwrap", lambda: None)

    scripted_sess = _ScriptedDriver(_ok_worker)
    orig_factories = hillclimb_mod.default_driver_factories

    def _factories_override(*args: Any, **kwargs: Any) -> dict[str, Any]:
        factories = dict(orig_factories(*args, **kwargs))
        factories["scripted"] = lambda: scripted_sess
        return factories

    monkeypatch.setattr(hillclimb_mod, "default_driver_factories", _factories_override)

    default_cfg = _base_config(direction="minimize", target_value=None, max_iterations=5)
    override_cfg = _base_config(direction="minimize", target_value=25.0, max_iterations=2)
    assert default_cfg.content_hash != override_cfg.content_hash

    sess_dir = tmp_path / "sess_override_state"
    runner = SessionRunner(
        SessionConfig(
            workspace_root=repo_sess,
            state_dir=sess_dir,
            workflow="pstack_hillclimb",
            step_mode=StepMode.EXECUTE,
            stage_driver="native",
            worker_driver="scripted",
            reviewer_driver="native",
            hillclimb_config=default_cfg.model_dump(mode="json"),
            metadata={"verification_commands": [f"{sys.executable} check_gate.py"]},
            provider_builder=lambda **_kw: _mock_provider([_sample_evidence()]),
        )
    )
    env1 = runner.start(
        goal="Optimize with explicit override config",
        hillclimb_config=override_cfg.model_dump(mode="json"),
    )
    assert env1.status == "needs_input"
    base_evs = [
        ev
        for ev in runner.store.list_events(env1.run_id)
        if ev.event_type == "hillclimb.baseline_recorded"
    ]
    assert base_evs and base_evs[0].payload["config_hash"] == override_cfg.content_hash

    # Resume with a brand-new SessionRunner that STILL carries default_cfg in SessionConfig!
    gate_file.write_text("go", encoding="utf-8")
    resumed = SessionRunner(
        SessionConfig(
            workspace_root=repo_sess,
            state_dir=sess_dir,
            workflow="pstack_hillclimb",
            step_mode=StepMode.EXECUTE,
            stage_driver="native",
            worker_driver="scripted",
            reviewer_driver="native",
            hillclimb_config=default_cfg.model_dump(mode="json"),
            provider_builder=lambda **_kw: _mock_provider([]),
        )
    )
    env2 = resumed.reply(run_id=env1.run_id, message="retry")
    assert env2.status == "completed", env2
    assert "return 22" in (repo_sess / "solver.py").read_text(encoding="utf-8")
