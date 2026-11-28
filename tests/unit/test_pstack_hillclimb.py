"""Unit and end-to-end tests for Issue #191 Part B (`pstack_hillclimb` workflow)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

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
from olympus.stages.craft import load_skill_pack
from olympus.stages.hillclimb import (
    HillclimbPause,
    HillclimbStage,
    _probe_bwrap,
    hillclimb_loop,
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
    assert [row.decision for row in report.rows] == [
        "rejected_tamper",
        "rejected_tamper",
        "keep",
    ]
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
