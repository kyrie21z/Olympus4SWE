"""Offline contract, concurrency, judging, grafting, cleanup, and E2E tests
for Issue #191 Part A (``pstack_arena`` multi-worktree race + Rubric Judge + graft).
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from olympus.core.contracts import (
    ArenaCandidateResult,
    ArenaGraftItem,
    ArenaRubric,
    ArenaRubricCriterion,
    ArenaVerdict,
    EvidenceBrief,
    EvidenceFact,
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
from olympus.stages.arena import ArenaStage
from olympus.stages.craft import CraftPause, _probe_bwrap, load_skill_pack
from olympus.web.schemas import WorkflowSummary


@pytest.fixture(autouse=True)
def _ci_bwrap_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep local Linux tests running under real bwrap; opt in only when CI lacks bwrap."""
    if _probe_bwrap() is None:
        monkeypatch.setenv("OLYMPUS_ALLOW_UNSANDBOXED_REPRO", "1")


class _ScriptedDriver:
    """Thread-safe deterministic AgentDriver executing scripted filesystem/JSON actions."""

    def __init__(
        self,
        name: str,
        handler: Callable[[DriverTurnRequest], str],
    ) -> None:
        self.name = name
        self._handler = handler
        self._lock = threading.Lock()
        self.requests: list[DriverTurnRequest] = []

    def run_turn(
        self,
        request: DriverTurnRequest,
        *,
        on_event: DriverEvent | None = None,
        on_handle: Callable[[ActiveTurnHandle], None] | None = None,
    ) -> TurnOutcome:
        del on_event, on_handle
        with self._lock:
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
    (repo / "calc.py").write_text(
        "def compute(x: int, y: int) -> int:\n    return x - y\n",
        encoding="utf-8",
    )
    (repo / "README.md").write_text("# Demo Repo\n", encoding="utf-8")
    (repo / "test_calc.py").write_text(
        "import unittest\n"
        "import calc\n\n"
        "class TestCalc(unittest.TestCase):\n"
        "    def test_add(self) -> None:\n"
        "        self.assertEqual(calc.compute(2, 3), 5)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)


def _git_worktree_paths(repo: Path) -> list[Path]:
    proc = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    paths: list[Path] = []
    for line in proc.stdout.splitlines():
        if line.startswith("worktree "):
            paths.append(Path(line.removeprefix("worktree ").strip()).resolve())
    return paths


def _sample_rubric() -> ArenaRubric:
    return ArenaRubric(
        criteria=[
            ArenaRubricCriterion(
                criterion_id="correctness",
                description="Passes unit tests and edge cases.",
                weight=2.0,
            ),
            ArenaRubricCriterion(
                criterion_id="simplicity",
                description="Minimal, clear code without unnecessary wrappers.",
                weight=1.5,
            ),
            ArenaRubricCriterion(
                criterion_id="clarity",
                description="Readable function signature and error handling.",
                weight=1.0,
            ),
        ],
        write_scope=["calc.py"],
        forbidden_globs=["README.md"],
        verification_plan=[f"{sys.executable} -m unittest test_calc.py"],
        summary="Compare candidate implementations of calc.compute.",
    )


def _sample_evidence() -> EvidenceBrief:
    return EvidenceBrief(
        summary="calc.compute currently subtracts instead of adding.",
        facts=[EvidenceFact(path="calc.py", line=2, observation="returns x - y")],
        candidate_files=["calc.py", "test_calc.py"],
    )


# ---------------------------------------------------------------------- 1. Contract validation


def test_arena_contracts_validation_and_fingerprints() -> None:
    rubric = _sample_rubric()
    assert rubric.content_hash.startswith("sha256:")
    assert len(rubric.criteria) == 3
    assert all(c.content_hash.startswith("sha256:") for c in rubric.criteria)

    # Fewer than 3 criteria rejected.
    with pytest.raises(ValidationError):
        ArenaRubric(
            criteria=rubric.criteria[:2],
            write_scope=["calc.py"],
            verification_plan=["pytest"],
        )

    # More than 6 criteria rejected.
    with pytest.raises(ValidationError):
        ArenaRubric(
            criteria=[
                ArenaRubricCriterion(criterion_id=f"c{i}", description=f"criterion {i}")
                for i in range(7)
            ],
            write_scope=["calc.py"],
            verification_plan=["pytest"],
        )

    # Duplicate criterion_id rejected.
    with pytest.raises(ValidationError):
        ArenaRubric(
            criteria=[
                ArenaRubricCriterion(criterion_id="dup", description="one"),
                ArenaRubricCriterion(criterion_id="dup", description="two"),
                ArenaRubricCriterion(criterion_id="other", description="three"),
            ],
            write_scope=["calc.py"],
            verification_plan=["pytest"],
        )

    cand = ArenaCandidateResult(
        candidate_id="cand_1",
        role="arena_runner_1",
        driver_spec="native",
        diff_summary="1 file changed",
        diff_gate_passed=True,
        verification_passed=True,
        verification_exit_code=0,
        status="completed",
        worktree_status="cleaned",
    )
    assert cand.content_hash.startswith("sha256:")

    # Valid verdict with graft item auto-aligns convergence_mode to winner_plus_graft.
    verdict = ArenaVerdict(
        winner_id="cand_1",
        scores={
            "cand_1": {"correctness": 9.5, "simplicity": 9.0, "clarity": 8.0},
            "cand_2": {"correctness": 8.0, "simplicity": 7.5, "clarity": 9.5},
        },
        graft_items=[
            ArenaGraftItem(
                source_candidate_id="cand_2",
                idea="Keep explicit integer type validation from cand_2.",
                target_files=["calc.py"],
            )
        ],
        convergence_mode="single_winner",
        rationale="cand_1 is cleaner and passes cleanly; graft cand_2's validation.",
    )
    assert verdict.convergence_mode == "winner_plus_graft"
    assert verdict.content_hash.startswith("sha256:")
    totals = verdict.weighted_totals(rubric)
    assert totals["cand_1"] > totals["cand_2"]
    verdict.validate_against(rubric, ["cand_1", "cand_2"], eligible_winner_ids=["cand_1"])

    # winner_id must match a scored candidate.
    with pytest.raises(ValidationError):
        ArenaVerdict(
            winner_id="cand_99",
            scores={
                "cand_1": {"correctness": 9.0, "simplicity": 8.0, "clarity": 8.0},
                "cand_2": {"correctness": 7.0, "simplicity": 7.0, "clarity": 7.0},
            },
            rationale="invalid winner",
        )

    # Graft source cannot reference winner itself.
    with pytest.raises(ValidationError):
        ArenaVerdict(
            winner_id="cand_1",
            scores={
                "cand_1": {"correctness": 9.0, "simplicity": 8.0, "clarity": 8.0},
                "cand_2": {"correctness": 7.0, "simplicity": 7.0, "clarity": 7.0},
            },
            graft_items=[ArenaGraftItem(source_candidate_id="cand_1", idea="self graft")],
            rationale="bad self graft",
        )

    # winner_plus_graft without graft_items rejected.
    with pytest.raises(ValidationError):
        ArenaVerdict(
            winner_id="cand_1",
            scores={
                "cand_1": {"correctness": 9.0, "simplicity": 8.0, "clarity": 8.0},
                "cand_2": {"correctness": 7.0, "simplicity": 7.0, "clarity": 7.0},
            },
            convergence_mode="winner_plus_graft",
            rationale="missing graft items",
        )

    # validate_against rejects ineligible (DiffGate-failed) winner.
    with pytest.raises(ValueError, match="not an eligible"):
        verdict.validate_against(rubric, ["cand_1", "cand_2"], eligible_winner_ids=["cand_2"])


# ---------------------------------------------------------------------- 2. Manifest & skill


def test_pstack_arena_manifest_and_skill_pack() -> None:
    assert "pstack_arena" in list_builtin_workflows()
    definition = load_workflow("pstack_arena")
    summary = WorkflowSummary.model_validate(describe_workflow(definition))
    assert summary.family == "pstack"
    assert summary.autonomy == "autonomous_hard_stops"
    assert summary.stop_after == "verify"
    assert summary.applicable_scenarios
    assert summary.anti_patterns
    by_name = {node.name: node for node in summary.graph.nodes}
    assert list(by_name) == ["explore", "arena_race", "hygiene", "verify_surface"]
    assert by_name["explore"].god == "prometheus"
    assert by_name["arena_race"].god == "athena"
    assert by_name["hygiene"].god == "hephaestus"
    assert by_name["verify_surface"].god == "argus"
    skill_text = load_skill_pack(["pstack/arena"])
    assert "Rubric" in skill_text or "rubric" in skill_text


# ---------------------------------------------------------------------- 3. Concurrent race + judge


def test_arena_race_runs_concurrent_worktrees_judges_and_cleans_up(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    store = RunStore(tmp_path / "olympus.db")
    run_id = store.create_run(
        title="Arena concurrent race",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_arena",
    )
    sandbox_root = tmp_path / "sandboxes"
    artifacts_root = tmp_path / "artifacts"

    # Barrier proves cand_1 and cand_2 execute concurrently in distinct isolated worktrees.
    barrier = threading.Barrier(2, timeout=10.0)
    seen_workspaces: dict[str, Path] = {}
    ws_lock = threading.Lock()

    def run_cand_1(req: DriverTurnRequest) -> str:
        barrier.wait()
        with ws_lock:
            seen_workspaces["cand_1"] = req.workspace_path.resolve()
        (req.workspace_path / "calc.py").write_text(
            "def compute(x: int, y: int) -> int:\n    return x + y\n",
            encoding="utf-8",
        )
        return "cand_1 implemented minimal sum"

    def run_cand_2(req: DriverTurnRequest) -> str:
        barrier.wait()
        with ws_lock:
            seen_workspaces["cand_2"] = req.workspace_path.resolve()
        (req.workspace_path / "calc.py").write_text(
            "def compute(x: int, y: int) -> int:\n"
            "    total = x + y\n"
            "    return total\n",
            encoding="utf-8",
        )
        return "cand_2 implemented verbose sum"

    def run_judge(req: DriverTurnRequest) -> str:
        assert "Candidate cand_1" in req.user_prompt
        assert "Candidate cand_2" in req.user_prompt
        return ArenaVerdict(
            winner_id="cand_1",
            scores={
                "cand_1": {"correctness": 10.0, "simplicity": 9.5, "clarity": 9.0},
                "cand_2": {"correctness": 10.0, "simplicity": 7.5, "clarity": 8.0},
            },
            convergence_mode="single_winner",
            rationale="cand_1 is minimal and passes all checks.",
        ).model_dump_json()

    d1 = _ScriptedDriver("runner1", run_cand_1)
    d2 = _ScriptedDriver("runner2", run_cand_2)
    judge = _ScriptedDriver("judge", run_judge)

    registry = DriverRegistry({"runner1": d1, "runner2": d2, "judge": judge})

    drivers_cfg = ExecutionDriversConfig.resolve(
        stage_driver="native",
        worker_driver="native",
        reviewer_driver="judge",
        role_drivers={
            "arena_runner_1": "runner1@model-a",
            "arena_runner_2": "runner2@model-b",
            "arena_judge": "judge@model-j",
        },
    )
    stage = ArenaStage(
        store=store,
        workspace_root=repo,
        provider=MockLLMProvider([]),
        artifacts_root=artifacts_root,
        sandbox_root=sandbox_root,
        execution_drivers=drivers_cfg,
        driver_registry=registry,
    )

    out = stage.arena_race(
        run_id=run_id,
        raw_input="Fix calc.compute to add x and y.",
        state={
            "evidence_brief": _sample_evidence().model_dump(mode="json"),
            "arena_rubric": _sample_rubric().model_dump(mode="json"),
        },
    )

    assert out["route"] == "pass"
    assert out["arena_verdict"]["winner_id"] == "cand_1"
    assert out["arena_verdict"]["judge_driver"] == "judge@model-j"
    assert out["arena_verdict"]["cold_start"] is True
    assert seen_workspaces["cand_1"] != seen_workspaces["cand_2"]

    # Winning diff checked out into run's craft sandbox.
    craft_ws = sandbox_root / f"{run_id}-craft"
    assert (craft_ws / "calc.py").read_text(encoding="utf-8") == (
        "def compute(x: int, y: int) -> int:\n    return x + y\n"
    )

    # All temporary candidate worktrees cleaned up on disk and in git worktree list.
    assert not seen_workspaces["cand_1"].exists()
    assert not seen_workspaces["cand_2"].exists()
    wt_paths = _git_worktree_paths(repo)
    assert all("arena-cand_" not in str(p) for p in wt_paths)
    assert all(cand["worktree_status"] == "cleaned" for cand in out["arena_candidates"])

    # Auditable SQLite events & arena artifacts persisted.
    event_types = [ev.event_type for ev in store.list_events(run_id)]
    assert "arena.started" in event_types
    assert event_types.count("arena.candidate_completed") == 2
    assert "arena.judged" in event_types
    assert "arena.grafted" in event_types
    arena_dir = artifacts_root / run_id / "arena"
    assert (arena_dir / "rubric.json").is_file()
    assert (arena_dir / "cand_1.patch").is_file()
    assert (arena_dir / "cand_2.patch").is_file()
    assert (arena_dir / "verdict.json").is_file()
    assert (arena_dir / "summary.json").is_file()


# ---------------------------------------------------------------------- 4. Graft + DiffGate


def test_arena_race_grafts_non_winner_idea_and_enforces_diff_gate(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    store = RunStore(tmp_path / "olympus.db")
    sandbox_root = tmp_path / "sandboxes"
    artifacts_root = tmp_path / "artifacts"

    # Case A: clean graft onto winner passes DiffGate.
    run_ok = store.create_run(
        title="Arena graft ok", mode=RunMode.EXECUTION, workflow_name="pstack_arena"
    )

    def build_graft_case(req: DriverTurnRequest) -> str:
        if req.actor == "hephaestus:cand_1":
            (req.workspace_path / "calc.py").write_text(
                "def compute(x: int, y: int) -> int:\n    return x + y\n",
                encoding="utf-8",
            )
            return "cand_1 done"
        if req.actor == "hephaestus:cand_2":
            (req.workspace_path / "calc.py").write_text(
                "def compute(x: int, y: int) -> int:\n"
                "    if not isinstance(x, int) or not isinstance(y, int):\n"
                "        raise TypeError('ints required')\n"
                "    return x + y\n",
                encoding="utf-8",
            )
            return "cand_2 done"
        if req.actor == "hephaestus:arena_graft":
            assert "Source candidate cand_2 diff" in req.user_prompt
            (req.workspace_path / "calc.py").write_text(
                "def compute(x: int, y: int) -> int:\n"
                "    if not isinstance(x, int) or not isinstance(y, int):\n"
                "        raise TypeError('ints required')\n"
                "    return x + y\n",
                encoding="utf-8",
            )
            return "graft applied"
        raise AssertionError(f"Unexpected actor: {req.actor}")

    def judge_graft_case(req: DriverTurnRequest) -> str:
        del req
        return ArenaVerdict(
            winner_id="cand_1",
            scores={
                "cand_1": {"correctness": 9.0, "simplicity": 9.5, "clarity": 8.5},
                "cand_2": {"correctness": 9.5, "simplicity": 7.5, "clarity": 8.0},
            },
            graft_items=[
                ArenaGraftItem(
                    source_candidate_id="cand_2",
                    idea="Transplant integer TypeError guard onto cand_1.",
                    target_files=["calc.py"],
                )
            ],
            convergence_mode="winner_plus_graft",
            rationale="cand_1 has cleaner baseline; transplant cand_2's TypeError check.",
        ).model_dump_json()

    reg = DriverRegistry(
        {
            "worker": _ScriptedDriver("worker", build_graft_case),
            "judge": _ScriptedDriver("judge", judge_graft_case),
        }
    )
    cfg = ExecutionDriversConfig.resolve(
        stage_driver="native",
        worker_driver="worker",
        reviewer_driver="judge",
    )
    stage = ArenaStage(
        store=store,
        workspace_root=repo,
        provider=MockLLMProvider([]),
        artifacts_root=artifacts_root,
        sandbox_root=sandbox_root,
        execution_drivers=cfg,
        driver_registry=reg,
    )
    out = stage.arena_race(
        run_id=run_ok,
        raw_input="Fix calc.compute with type validation.",
        state={"arena_rubric": _sample_rubric().model_dump(mode="json")},
    )
    assert out["arena_verdict"]["convergence_mode"] == "winner_plus_graft"
    craft_calc = (sandbox_root / f"{run_ok}-craft" / "calc.py").read_text(encoding="utf-8")
    assert "raise TypeError('ints required')" in craft_calc
    assert (artifacts_root / run_ok / "arena" / "grafted.patch").is_file()
    graft_events = [
        ev for ev in store.list_events(run_ok) if ev.event_type == "arena.grafted"
    ]
    assert len(graft_events) == 1
    assert graft_events[0].payload["skipped"] is False

    # Case B: graft mutating outside write_scope is blocked by DiffGate & worktrees still cleaned.
    run_bad = store.create_run(
        title="Arena graft violation", mode=RunMode.EXECUTION, workflow_name="pstack_arena"
    )

    def build_bad_graft(req: DriverTurnRequest) -> str:
        if req.actor == "hephaestus:arena_graft":
            (req.workspace_path / "README.md").write_text("unauthorized edit\n", encoding="utf-8")
            return "bad graft"
        (req.workspace_path / "calc.py").write_text(
            "def compute(x: int, y: int) -> int:\n    return x + y\n",
            encoding="utf-8",
        )
        return "cand ok"

    reg_bad = DriverRegistry(
        {
            "worker": _ScriptedDriver("worker", build_bad_graft),
            "judge": _ScriptedDriver("judge", judge_graft_case),
        }
    )
    stage_bad = ArenaStage(
        store=store,
        workspace_root=repo,
        provider=MockLLMProvider([]),
        artifacts_root=artifacts_root,
        sandbox_root=sandbox_root,
        execution_drivers=cfg,
        driver_registry=reg_bad,
    )
    with pytest.raises(CraftPause, match="Diff Gate rejected"):
        stage_bad.arena_race(
            run_id=run_bad,
            raw_input="Fix calc.compute.",
            state={"arena_rubric": _sample_rubric().model_dump(mode="json")},
        )
    assert all("arena-cand_" not in str(p) for p in _git_worktree_paths(repo))
    assert not (sandbox_root / f"{run_bad}-arena-cand_1").exists()
    assert not (sandbox_root / f"{run_bad}-arena-cand_2").exists()


# ---------------------------------------------------------------------- 5. Failure cleanup


def test_arena_race_guarantees_worktree_cleanup_on_injected_failure(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    store = RunStore(tmp_path / "olympus.db")
    run_id = store.create_run(
        title="Arena injected failure",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_arena",
    )
    sandbox_root = tmp_path / "sandboxes"
    created_paths: list[Path] = []
    lock = threading.Lock()

    def build_worker(req: DriverTurnRequest) -> str:
        with lock:
            created_paths.append(req.workspace_path.resolve())
        (req.workspace_path / "calc.py").write_text(
            "def compute(x: int, y: int) -> int:\n    return x + y\n",
            encoding="utf-8",
        )
        return "built"

    def exploding_judge(req: DriverTurnRequest) -> str:
        del req
        raise RuntimeError("injected judge crash")

    reg = DriverRegistry(
        {
            "worker": _ScriptedDriver("worker", build_worker),
            "judge": _ScriptedDriver("judge", exploding_judge),
        }
    )
    stage = ArenaStage(
        store=store,
        workspace_root=repo,
        provider=MockLLMProvider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=sandbox_root,
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="worker",
            reviewer_driver="judge",
        ),
        driver_registry=reg,
    )

    with pytest.raises(RuntimeError, match="injected judge crash"):
        stage.arena_race(
            run_id=run_id,
            raw_input="Fix calc.compute.",
            state={"arena_rubric": _sample_rubric().model_dump(mode="json")},
        )

    assert len(created_paths) == 2
    for path in created_paths:
        assert not path.exists(), f"Leaked candidate worktree on failure: {path}"
    assert _git_worktree_paths(repo) == [repo.resolve()]


def test_arena_race_rejects_judge_picking_diff_gate_rejected_candidate(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    store = RunStore(tmp_path / "olympus.db")
    run_id = store.create_run(
        title="Arena ineligible winner",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_arena",
    )
    sandbox_root = tmp_path / "sandboxes"

    def build_worker(req: DriverTurnRequest) -> str:
        (req.workspace_path / "calc.py").write_text(
            "def compute(x: int, y: int) -> int:\n    return x + y\n",
            encoding="utf-8",
        )
        if req.actor == "hephaestus:cand_2":
            (req.workspace_path / "README.md").write_text("out of scope\n", encoding="utf-8")
        return "done"

    def biased_judge(req: DriverTurnRequest) -> str:
        del req
        return ArenaVerdict(
            winner_id="cand_2",
            scores={
                "cand_1": {"correctness": 9.0, "simplicity": 8.0, "clarity": 8.0},
                "cand_2": {"correctness": 9.5, "simplicity": 9.0, "clarity": 9.0},
            },
            rationale="Attempting to select rejected candidate.",
        ).model_dump_json()

    reg = DriverRegistry(
        {
            "worker": _ScriptedDriver("worker", build_worker),
            "judge": _ScriptedDriver("judge", biased_judge),
        }
    )
    stage = ArenaStage(
        store=store,
        workspace_root=repo,
        provider=MockLLMProvider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=sandbox_root,
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="worker",
            reviewer_driver="judge",
        ),
        driver_registry=reg,
    )

    with pytest.raises(CraftPause, match="failed deterministic validation"):
        stage.arena_race(
            run_id=run_id,
            raw_input="Fix calc.compute.",
            state={"arena_rubric": _sample_rubric().model_dump(mode="json")},
        )
    assert _git_worktree_paths(repo) == [repo.resolve()]


# ---------------------------------------------------------------------- 6. End-to-end graph


def test_pstack_arena_end_to_end_graph_verify_and_promote(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    store = RunStore(tmp_path / "olympus.db")
    run_id = store.create_run(
        title="E2E pstack_arena run",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_arena",
    )
    sandbox_root = tmp_path / "sandboxes"
    artifacts_root = tmp_path / "artifacts"

    # Structured stage turns via MockLLMProvider:
    # 1. explore (EvidenceBrief)
    # 2. arena rubric synthesis (ArenaRubric)
    provider = MockLLMProvider(
        [
            LLMResponse(content=_sample_evidence().model_dump_json()),
            LLMResponse(content=_sample_rubric().model_dump_json()),
        ]
    )

    def worker_handler(req: DriverTurnRequest) -> str:
        if req.actor == "hephaestus:cand_1":
            (req.workspace_path / "calc.py").write_text(
                "def compute(x: int, y: int) -> int:\n    # slop comment\n    return x + y\n",
                encoding="utf-8",
            )
            return "cand_1 ok"
        if req.actor == "hephaestus:cand_2":
            (req.workspace_path / "calc.py").write_text(
                "def compute(x: int, y: int) -> int:\n"
                "    return int(x) + int(y)\n",
                encoding="utf-8",
            )
            return "cand_2 ok"
        if req.actor == "hephaestus:arena_graft":
            (req.workspace_path / "calc.py").write_text(
                "def compute(x: int, y: int) -> int:\n"
                "    # slop comment\n"
                "    return int(x) + int(y)\n",
                encoding="utf-8",
            )
            return "graft ok"
        if req.actor == "hephaestus:hygiene":
            (req.workspace_path / "calc.py").write_text(
                "def compute(x: int, y: int) -> int:\n    return int(x) + int(y)\n",
                encoding="utf-8",
            )
            return "hygiene ok"
        raise AssertionError(f"Unexpected worker actor: {req.actor}")

    def judge_handler(req: DriverTurnRequest) -> str:
        del req
        return ArenaVerdict(
            winner_id="cand_1",
            scores={
                "cand_1": {"correctness": 9.5, "simplicity": 9.0, "clarity": 8.5},
                "cand_2": {"correctness": 9.5, "simplicity": 8.0, "clarity": 9.0},
            },
            graft_items=[
                ArenaGraftItem(
                    source_candidate_id="cand_2",
                    idea="Coerce inputs with int() as in cand_2.",
                    target_files=["calc.py"],
                )
            ],
            convergence_mode="winner_plus_graft",
            rationale="cand_1 wins overall; graft cand_2's int() coercion.",
        ).model_dump_json()

    registry = DriverRegistry(
        {
            "worker": _ScriptedDriver("worker", worker_handler),
            "judge": _ScriptedDriver("judge", judge_handler),
        }
    )

    runtime = WorkflowRuntime(
        store=store,
        workspace_root=repo,
        artifacts_root=artifacts_root,
        sandbox_root=sandbox_root,
        provider=provider,
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="worker",
            reviewer_driver="judge",
        ),
        driver_registry=registry,
        apply_to_workspace=True,
    )
    graph = build_graph(load_workflow("pstack_arena"))
    final_state = run_graph(
        graph,
        {
            "run_id": run_id,
            "raw_input": "Fix calc.compute so it returns the integer sum of x and y.",
            "_runtime": runtime,
        },
    )

    assert final_state["route"] == "pass"
    assert final_state["arena_verdict"]["winner_id"] == "cand_1"
    assert final_state["surface_verification"]["passed"] is True
    assert final_state["applied_paths"] == ["calc.py"]

    # Host workspace promoted cleanly and passes unittest.
    assert (repo / "calc.py").read_text(encoding="utf-8") == (
        "def compute(x: int, y: int) -> int:\n    return int(x) + int(y)\n"
    )
    verify_proc = subprocess.run(
        [sys.executable, "-m", "unittest", "test_calc.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert verify_proc.returncode == 0, verify_proc.stderr

    # Zero temporary worktrees (neither arena candidates nor craft sandbox) remain.
    assert _git_worktree_paths(repo) == [repo.resolve()]
    assert not (sandbox_root / f"{run_id}-arena-cand_1").exists()
    assert not (sandbox_root / f"{run_id}-arena-cand_2").exists()
    assert not (sandbox_root / f"{run_id}-craft").exists()
    summary_data = json.loads(
        (artifacts_root / run_id / "arena" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary_data["verdict"]["winner_id"] == "cand_1"


def test_arena_normalizes_absolute_write_scope_under_workspace_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    store = RunStore(tmp_path / "olympus.db")
    run_id = store.create_run(
        title="Arena abs scope normalization",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_arena",
    )

    def build_worker(req: DriverTurnRequest) -> str:
        (req.workspace_path / "calc.py").write_text(
            "def compute(x: int, y: int) -> int:\n    return x + y\n",
            encoding="utf-8",
        )
        return "done"

    def judge_worker(req: DriverTurnRequest) -> str:
        del req
        return ArenaVerdict(
            winner_id="cand_1",
            scores={
                "cand_1": {"correctness": 10.0, "simplicity": 9.0, "clarity": 9.0},
                "cand_2": {"correctness": 10.0, "simplicity": 8.5, "clarity": 8.5},
            },
            rationale="cand_1 passes cleanly.",
        ).model_dump_json()

    abs_rubric = _sample_rubric().model_copy(
        update={"write_scope": [str((repo / "calc.py").resolve())]}
    )
    stage = ArenaStage(
        store=store,
        workspace_root=repo,
        provider=MockLLMProvider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="worker",
            reviewer_driver="judge",
        ),
        driver_registry=DriverRegistry(
            {
                "worker": _ScriptedDriver("worker", build_worker),
                "judge": _ScriptedDriver("judge", judge_worker),
            }
        ),
    )
    out = stage.arena_race(
        run_id=run_id,
        raw_input="Fix calc.compute.",
        state={"arena_rubric": abs_rubric.model_dump(mode="json")},
    )
    assert out["arena_rubric"]["write_scope"] == ["calc.py"]
    assert all(c["diff_gate_passed"] for c in out["arena_candidates"])
    assert all("arena-cand_" not in str(p) for p in _git_worktree_paths(repo))


def test_arena_verification_plan_sanitizer_preserves_fail_closed(tmp_path: Path) -> None:
    from olympus.stages.arena import _filter_allowed_verification_plan

    valid_cmd = f"{sys.executable} -m unittest test_calc.py"
    mixed = [valid_cmd, "git status --porcelain", f"{sys.executable} -c 'print(1)'"]
    assert _filter_allowed_verification_plan(mixed, workspace_root=tmp_path) == [valid_cmd]

    # When zero valid commands exist, fail closed by preserving originals for rejection.
    all_disallowed = ["git status --porcelain", f"{sys.executable} -c 'print(1)'"]
    assert (
        _filter_allowed_verification_plan(all_disallowed, workspace_root=tmp_path)
        == all_disallowed
    )


# ---------------------------------------------------------------------- 7. Codex review hardening


def test_arena_verification_failure_and_cancellation_cannot_win(tmp_path: Path) -> None:
    """Finding 1: failed verification and cancelled turns are never eligible winners."""
    from olympus.harness.loop import TurnCancelledError

    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    store = RunStore(tmp_path / "olympus.db")
    sandbox_root = tmp_path / "sandboxes"

    # Case A: cand_1 passes verification, cand_2 passes DiffGate but fails unittest (exit=1).
    # Judge attempting to pick cand_2 must be deterministically rejected.
    run_a = store.create_run(
        title="Arena verify fail winner rejected",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_arena",
    )

    def worker_a(req: DriverTurnRequest) -> str:
        if req.actor == "hephaestus:cand_1":
            (req.workspace_path / "calc.py").write_text(
                "def compute(x: int, y: int) -> int:\n    return x + y\n",
                encoding="utf-8",
            )
        else:
            (req.workspace_path / "calc.py").write_text(
                "def compute(x: int, y: int) -> int:\n    return x * y\n",
                encoding="utf-8",
            )
        return "done"

    def judge_picks_failing_cand_2(req: DriverTurnRequest) -> str:
        del req
        return ArenaVerdict(
            winner_id="cand_2",
            scores={
                "cand_1": {"correctness": 9.0, "simplicity": 8.0, "clarity": 8.0},
                "cand_2": {"correctness": 9.5, "simplicity": 9.0, "clarity": 9.0},
            },
            rationale="Improperly picking failing cand_2.",
        ).model_dump_json()

    stage_a = ArenaStage(
        store=store,
        workspace_root=repo,
        provider=MockLLMProvider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=sandbox_root,
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="worker",
            reviewer_driver="judge",
        ),
        driver_registry=DriverRegistry(
            {
                "worker": _ScriptedDriver("worker", worker_a),
                "judge": _ScriptedDriver("judge", judge_picks_failing_cand_2),
            }
        ),
    )
    with pytest.raises(CraftPause, match="failed deterministic validation"):
        stage_a.arena_race(
            run_id=run_a,
            raw_input="Fix calc.compute.",
            state={"arena_rubric": _sample_rubric().model_dump(mode="json")},
        )
    assert _git_worktree_paths(repo) == [repo.resolve()]

    # Case B: both candidates fail verification -> pauses without invoking judge.
    run_b = store.create_run(
        title="Arena all fail verification",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_arena",
    )
    judge_called = False

    def should_not_run_judge(req: DriverTurnRequest) -> str:
        nonlocal judge_called
        judge_called = True
        del req
        raise AssertionError("Judge must not be invoked when zero candidates pass verification")

    def worker_all_fail(req: DriverTurnRequest) -> str:
        (req.workspace_path / "calc.py").write_text(
            "def compute(x: int, y: int) -> int:\n    return x * y\n",
            encoding="utf-8",
        )
        return "broken"

    stage_b = ArenaStage(
        store=store,
        workspace_root=repo,
        provider=MockLLMProvider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=sandbox_root,
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="worker",
            reviewer_driver="judge",
        ),
        driver_registry=DriverRegistry(
            {
                "worker": _ScriptedDriver("worker", worker_all_fail),
                "judge": _ScriptedDriver("judge", should_not_run_judge),
            }
        ),
    )
    with pytest.raises(CraftPause, match="No eligible arena candidates passed DiffGate and verif"):
        stage_b.arena_race(
            run_id=run_b,
            raw_input="Fix calc.compute.",
            state={"arena_rubric": _sample_rubric().model_dump(mode="json")},
        )
    assert judge_called is False
    assert _git_worktree_paths(repo) == [repo.resolve()]

    # Case C: cancelled candidate (via TurnOutcome stop_reason='cancelled' OR TurnCancelledError)
    # is marked status='cancelled', never 'completed', and cannot win.
    run_c = store.create_run(
        title="Arena cancelled candidate",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_arena",
    )

    class _CancellingDriver:
        name = "cancelling"

        def run_turn(
            self,
            request: DriverTurnRequest,
            *,
            on_event: DriverEvent | None = None,
            on_handle: Callable[[ActiveTurnHandle], None] | None = None,
        ) -> TurnOutcome:
            del on_event, on_handle
            if request.actor == "hephaestus:cand_1":
                (request.workspace_path / "calc.py").write_text(
                    "def compute(x: int, y: int) -> int:\n    return x + y\n",
                    encoding="utf-8",
                )
                return TurnOutcome(
                    completed=True,
                    stop_reason="completed",
                    final_content="ok",
                    steps_taken=1,
                    driver_name=self.name,
                )
            if request.actor == "hephaestus:cand_2":
                # Even if a partial edit happened before cancel, status MUST be cancelled.
                (request.workspace_path / "calc.py").write_text(
                    "def compute(x: int, y: int) -> int:\n    return x + y\n",
                    encoding="utf-8",
                )
                return TurnOutcome(
                    completed=False,
                    stop_reason="cancelled",
                    final_content="",
                    steps_taken=1,
                    driver_name=self.name,
                )
            raise TurnCancelledError("cancelled")

    def judge_picks_cand_1(req: DriverTurnRequest) -> str:
        del req
        return ArenaVerdict(
            winner_id="cand_1",
            scores={
                "cand_1": {"correctness": 10.0, "simplicity": 9.0, "clarity": 9.0},
                "cand_2": {"correctness": 0.0, "simplicity": 0.0, "clarity": 0.0},
                "cand_3": {"correctness": 0.0, "simplicity": 0.0, "clarity": 0.0},
            },
            rationale="Only cand_1 completed and verified cleanly.",
        ).model_dump_json()

    stage_c = ArenaStage(
        store=store,
        workspace_root=repo,
        provider=MockLLMProvider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=sandbox_root,
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="cancelling",
            reviewer_driver="judge",
        ),
        driver_registry=DriverRegistry(
            {
                "cancelling": _CancellingDriver(),
                "judge": _ScriptedDriver("judge", judge_picks_cand_1),
            }
        ),
    )
    out_c = stage_c.arena_race(
        run_id=run_c,
        raw_input="Fix calc.compute.",
        state={"arena_rubric": _sample_rubric().model_dump(mode="json")},
        num_candidates=3,
    )
    by_id = {c["candidate_id"]: c for c in out_c["arena_candidates"]}
    assert by_id["cand_1"]["status"] == "completed"
    assert by_id["cand_2"]["status"] == "cancelled"
    assert by_id["cand_2"]["verification_passed"] is False
    assert by_id["cand_3"]["status"] == "cancelled"
    assert by_id["cand_3"]["verification_passed"] is False


def test_arena_graft_failure_preserves_winner_craft_sandbox(tmp_path: Path) -> None:
    """Finding 2: graft exception, DiffGate violation, or verification failure never corrupts
    the winning candidate's craft sandbox.
    """
    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    store = RunStore(tmp_path / "olympus.db")
    sandbox_root = tmp_path / "sandboxes"
    winner_code = "def compute(x: int, y: int) -> int:\n    return x + y\n"

    def judge_with_graft(req: DriverTurnRequest) -> str:
        del req
        return ArenaVerdict(
            winner_id="cand_1",
            scores={
                "cand_1": {"correctness": 9.5, "simplicity": 9.5, "clarity": 9.0},
                "cand_2": {"correctness": 9.0, "simplicity": 8.0, "clarity": 8.5},
            },
            graft_items=[
                ArenaGraftItem(
                    source_candidate_id="cand_2",
                    idea="Add helper check from cand_2.",
                    target_files=["calc.py"],
                )
            ],
            convergence_mode="winner_plus_graft",
            rationale="cand_1 wins; attempt graft from cand_2.",
        ).model_dump_json()

    # Sub-case 1: graft breaks calc.py and fails verification -> craft sandbox stays pristine.
    run_verif_fail = store.create_run(
        title="Arena graft verification rollback",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_arena",
    )

    def worker_graft_breaks_tests(req: DriverTurnRequest) -> str:
        if req.actor == "hephaestus:arena_graft":
            (req.workspace_path / "calc.py").write_text(
                "def compute(x: int, y: int) -> int:\n    return x - y - 999\n",
                encoding="utf-8",
            )
            return "regressed graft"
        (req.workspace_path / "calc.py").write_text(winner_code, encoding="utf-8")
        return "ok"

    stage_1 = ArenaStage(
        store=store,
        workspace_root=repo,
        provider=MockLLMProvider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=sandbox_root,
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="worker",
            reviewer_driver="judge",
        ),
        driver_registry=DriverRegistry(
            {
                "worker": _ScriptedDriver("worker", worker_graft_breaks_tests),
                "judge": _ScriptedDriver("judge", judge_with_graft),
            }
        ),
    )
    with pytest.raises(CraftPause, match="Arena graft failed verification"):
        stage_1.arena_race(
            run_id=run_verif_fail,
            raw_input="Fix calc.compute.",
            state={"arena_rubric": _sample_rubric().model_dump(mode="json")},
        )
    assert (sandbox_root / f"{run_verif_fail}-craft" / "calc.py").read_text(
        encoding="utf-8"
    ) == winner_code
    assert not (sandbox_root / f"{run_verif_fail}-arena-graft").exists()

    # Sub-case 2: graft raises mid-edit -> craft sandbox stays pristine.
    run_exc = store.create_run(
        title="Arena graft exception rollback",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_arena",
    )

    def worker_graft_explodes(req: DriverTurnRequest) -> str:
        if req.actor == "hephaestus:arena_graft":
            (req.workspace_path / "calc.py").write_text("CORRUPT HALF-WRITE\n", encoding="utf-8")
            raise RuntimeError("graft crashed mid-edit")
        (req.workspace_path / "calc.py").write_text(winner_code, encoding="utf-8")
        return "ok"

    stage_2 = ArenaStage(
        store=store,
        workspace_root=repo,
        provider=MockLLMProvider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=sandbox_root,
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="worker",
            reviewer_driver="judge",
        ),
        driver_registry=DriverRegistry(
            {
                "worker": _ScriptedDriver("worker", worker_graft_explodes),
                "judge": _ScriptedDriver("judge", judge_with_graft),
            }
        ),
    )
    with pytest.raises(CraftPause, match="graft crashed mid-edit"):
        stage_2.arena_race(
            run_id=run_exc,
            raw_input="Fix calc.compute.",
            state={"arena_rubric": _sample_rubric().model_dump(mode="json")},
        )
    assert (sandbox_root / f"{run_exc}-craft" / "calc.py").read_text(
        encoding="utf-8"
    ) == winner_code
    assert not (sandbox_root / f"{run_exc}-arena-graft").exists()


def test_arena_strict_cleanup_on_partial_create_and_removal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 3: partial create cleans up immediately; failed removal is never reported cleaned."""
    from olympus.harness import worktree as wt_mod
    from olympus.stages import arena as arena_mod

    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    store = RunStore(tmp_path / "olympus.db")
    sandbox_root = tmp_path / "sandboxes"

    # Sub-case A: cand_2 creation leaves a partial directory and raises -> both cleaned up.
    run_partial = store.create_run(
        title="Arena partial create failure",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_arena",
    )
    real_create = arena_mod.create_sandbox
    call_count = 0

    def flaky_create(repo_root: Path | str, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            partial_dir = Path(kwargs["sandbox_root"]) / kwargs["label"]
            partial_dir.mkdir(parents=True, exist_ok=True)
            (partial_dir / "partial.tmp").write_text("leaked", encoding="utf-8")
            raise RuntimeError("injected create_sandbox crash on cand_2")
        return real_create(repo_root, **kwargs)

    monkeypatch.setattr(arena_mod, "create_sandbox", flaky_create)
    stage_a = ArenaStage(
        store=store,
        workspace_root=repo,
        provider=MockLLMProvider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=sandbox_root,
    )
    with pytest.raises(RuntimeError, match="injected create_sandbox crash on cand_2"):
        stage_a.arena_race(
            run_id=run_partial,
            raw_input="Fix calc.compute.",
            state={"arena_rubric": _sample_rubric().model_dump(mode="json")},
        )
    assert not (sandbox_root / f"{run_partial}-arena-cand_1").exists()
    assert not (sandbox_root / f"{run_partial}-arena-cand_2").exists()
    assert _git_worktree_paths(repo) == [repo.resolve()]
    monkeypatch.setattr(arena_mod, "create_sandbox", real_create)

    # Sub-case B: strict remove_sandbox fails on cand_2 -> NOT reported cleaned & raises CraftPause.
    run_fail_rm = store.create_run(
        title="Arena cleanup failure reported honestly",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_arena",
    )
    real_remove = arena_mod.remove_sandbox

    def failing_remove(handle: Any, *, strict: bool = False) -> None:
        if handle.workspace_path.name.endswith("cand_2"):
            raise RuntimeError(f"locked directory: {handle.workspace_path}")
        real_remove(handle, strict=strict)

    def ok_worker(req: DriverTurnRequest) -> str:
        (req.workspace_path / "calc.py").write_text(
            "def compute(x: int, y: int) -> int:\n    return x + y\n",
            encoding="utf-8",
        )
        return "ok"

    def ok_judge(req: DriverTurnRequest) -> str:
        del req
        return ArenaVerdict(
            winner_id="cand_1",
            scores={
                "cand_1": {"correctness": 10.0, "simplicity": 9.0, "clarity": 9.0},
                "cand_2": {"correctness": 10.0, "simplicity": 8.0, "clarity": 8.0},
            },
            rationale="cand_1 wins.",
        ).model_dump_json()

    monkeypatch.setattr(arena_mod, "remove_sandbox", failing_remove)
    stage_b = ArenaStage(
        store=store,
        workspace_root=repo,
        provider=MockLLMProvider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=sandbox_root,
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="worker",
            reviewer_driver="judge",
        ),
        driver_registry=DriverRegistry(
            {
                "worker": _ScriptedDriver("worker", ok_worker),
                "judge": _ScriptedDriver("judge", ok_judge),
            }
        ),
    )
    with pytest.raises(CraftPause, match="Failed to clean up temporary arena worktree"):
        stage_b.arena_race(
            run_id=run_fail_rm,
            raw_input="Fix calc.compute.",
            state={"arena_rubric": _sample_rubric().model_dump(mode="json")},
        )
    monkeypatch.setattr(arena_mod, "remove_sandbox", real_remove)
    # Verify honest worktree_status persisted and cleanup_failed event recorded.
    persisted = json.loads(
        (tmp_path / "artifacts" / run_fail_rm / "arena" / "candidates.json").read_text(
            encoding="utf-8"
        )
    )
    by_id = {item["candidate_id"]: item for item in persisted}
    assert by_id["cand_1"]["worktree_status"] == "cleaned"
    assert by_id["cand_2"]["worktree_status"] == "cleanup_failed"
    assert any(ev.event_type == "arena.cleanup_failed" for ev in store.list_events(run_fail_rm))
    # Clean up the simulated locked cand_2 directory now.
    real_remove(
        wt_mod.SandboxHandle(repo, sandbox_root / f"{run_fail_rm}-arena-cand_2", None, True),
        strict=True,
    )

    # Sub-case C: worktree.remove_sandbox(strict=True) itself raises if directory persists.
    stuck_dir = tmp_path / "stuck_sandbox"
    stuck_dir.mkdir()
    monkeypatch.setattr(wt_mod.shutil, "rmtree", lambda *_a, **_kw: None)
    with pytest.raises(RuntimeError, match="directory still exists"):
        wt_mod.remove_sandbox(wt_mod.SandboxHandle(repo, stuck_dir, None, False), strict=True)


def test_pooled_provider_concurrent_429_does_not_skip_healthy_endpoint() -> None:
    """Finding 4: concurrent 429 on endpoint A advances once to B and never skips to C."""
    from concurrent.futures import ThreadPoolExecutor

    from olympus.providers.auth import CredentialPool, EndpointTarget, PooledProvider
    from olympus.providers.types import ChatMessage, HttpStatusError

    endpoints = [
        EndpointTarget(base_url="https://ep-a.test", api_key="key-a"),
        EndpointTarget(base_url="https://ep-b.test", api_key="key-b"),
        EndpointTarget(base_url="https://ep-c.test", api_key="key-c"),
    ]
    pool = CredentialPool(endpoints)
    barrier = threading.Barrier(2, timeout=5.0)
    calls: list[str] = []
    calls_lock = threading.Lock()

    class _Concurrent429Dialect:
        wire_api = "openai-completions"

        def send(self, endpoint: EndpointTarget, **_kwargs: Any) -> LLMResponse:
            with calls_lock:
                calls.append(endpoint.base_url)
            if endpoint.base_url == "https://ep-a.test":
                barrier.wait()
                raise HttpStatusError(
                    status_code=429,
                    url=endpoint.base_url,
                    body_preview="rate limited",
                )
            if endpoint.base_url == "https://ep-b.test":
                return LLMResponse(content="served-by-b")
            raise AssertionError(f"Endpoint C must never be hit; got {endpoint.base_url}")

    provider = PooledProvider(
        model_name="test-model",
        dialect=_Concurrent429Dialect(),  # type: ignore[arg-type]
        pool=pool,
    )
    msg = [ChatMessage(role="user", content="hello")]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(provider.complete, messages=msg) for _ in range(2)]
        results = [f.result() for f in futures]

    assert [r.content for r in results] == ["served-by-b", "served-by-b"]
    assert pool.active_index == 1
    assert "https://ep-c.test" not in calls
    assert calls.count("https://ep-b.test") == 2


def test_arena_locks_base_ref_uniformly_and_rejects_explicit_ref_in_non_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 5: base_ref (including HEAD) locks to one SHA across all sandboxes; non-git
    explicit base_ref fails closed.
    """
    from olympus.stages import arena as arena_mod

    # Sub-case A: non-git directory with explicit base_ref must fail closed.
    nongit = tmp_path / "nongit"
    nongit.mkdir()
    (nongit / "calc.py").write_text("def compute(x, y): return x - y\n", encoding="utf-8")
    store = RunStore(tmp_path / "olympus.db")
    run_ng = store.create_run(
        title="Arena non-git explicit base_ref",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_arena",
    )
    stage_ng = ArenaStage(
        store=store,
        workspace_root=nongit,
        provider=MockLLMProvider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
    )
    with pytest.raises(CraftPause, match="not a git repository"):
        stage_ng.arena_race(
            run_id=run_ng,
            raw_input="Fix calc.compute.",
            state={"arena_rubric": _sample_rubric().model_dump(mode="json")},
            base_ref="main",
        )

    # Sub-case B: HEAD advances mid-creation -> cand_1, cand_2, and craft all stay pinned
    # to the single initial locked SHA.
    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    initial_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    real_create = arena_mod.create_sandbox
    recorded_base_commits: list[str | None] = []
    moved_head = False

    def moving_head_create(repo_root: Path | str, **kwargs: Any) -> Any:
        nonlocal moved_head
        handle = real_create(repo_root, **kwargs)
        recorded_base_commits.append(handle.base_commit)
        if not moved_head:
            moved_head = True
            (repo / "README.md").write_text("# Moved HEAD\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
            subprocess.run(
                ["git", "commit", "-m", "advance HEAD mid-race"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
        return handle

    monkeypatch.setattr(arena_mod, "create_sandbox", moving_head_create)
    run_locked = store.create_run(
        title="Arena locked base_ref SHA",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_arena",
    )

    def ok_worker(req: DriverTurnRequest) -> str:
        (req.workspace_path / "calc.py").write_text(
            "def compute(x: int, y: int) -> int:\n    return x + y\n",
            encoding="utf-8",
        )
        return "ok"

    def ok_judge(req: DriverTurnRequest) -> str:
        del req
        return ArenaVerdict(
            winner_id="cand_1",
            scores={
                "cand_1": {"correctness": 10.0, "simplicity": 9.0, "clarity": 9.0},
                "cand_2": {"correctness": 10.0, "simplicity": 8.0, "clarity": 8.0},
            },
            rationale="cand_1 wins.",
        ).model_dump_json()

    stage_locked = ArenaStage(
        store=store,
        workspace_root=repo,
        provider=MockLLMProvider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="worker",
            reviewer_driver="judge",
        ),
        driver_registry=DriverRegistry(
            {
                "worker": _ScriptedDriver("worker", ok_worker),
                "judge": _ScriptedDriver("judge", ok_judge),
            }
        ),
    )
    stage_locked.arena_race(
        run_id=run_locked,
        raw_input="Fix calc.compute.",
        state={"arena_rubric": _sample_rubric().model_dump(mode="json")},
    )
    assert len(recorded_base_commits) == 3  # cand_1, cand_2, craft
    assert all(sha == initial_sha for sha in recorded_base_commits)


def test_arena_verdict_rejects_duplicate_stripped_criteria_and_excess_grafts(
    tmp_path: Path,
) -> None:
    """Finding 6: duplicate normalized criterion keys raise ValidationError; excess graft_items
    are rejected deterministically before persisting verdict.
    """
    # Duplicate criterion key after .strip() must raise ValidationError (never overwrite).
    with pytest.raises(ValidationError, match="duplicate criterion_id"):
        ArenaVerdict(
            winner_id="cand_1",
            scores={
                "cand_1": {
                    "correctness": 9.0,
                    "correctness ": 2.0,
                    "simplicity": 8.0,
                    "clarity": 8.0,
                },
                "cand_2": {"correctness": 8.0, "simplicity": 8.0, "clarity": 8.0},
            },
            rationale="duplicate criterion key after stripping",
        )

    # Excess graft_items (> max_graft_items) rejected in judge validation before persistence.
    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    store = RunStore(tmp_path / "olympus.db")
    run_id = store.create_run(
        title="Arena excess graft_items rejected",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_arena",
    )

    def ok_worker(req: DriverTurnRequest) -> str:
        (req.workspace_path / "calc.py").write_text(
            "def compute(x: int, y: int) -> int:\n    return x + y\n",
            encoding="utf-8",
        )
        return "ok"

    def judge_too_many_grafts(req: DriverTurnRequest) -> str:
        del req
        return ArenaVerdict(
            winner_id="cand_1",
            scores={
                "cand_1": {"correctness": 10.0, "simplicity": 9.0, "clarity": 9.0},
                "cand_2": {"correctness": 9.0, "simplicity": 8.0, "clarity": 8.5},
            },
            graft_items=[
                ArenaGraftItem(source_candidate_id="cand_2", idea="graft 1"),
                ArenaGraftItem(source_candidate_id="cand_2", idea="graft 2"),
                ArenaGraftItem(source_candidate_id="cand_2", idea="graft 3"),
            ],
            convergence_mode="winner_plus_graft",
            rationale="Too many grafts.",
        ).model_dump_json()

    artifacts_root = tmp_path / "artifacts"
    stage = ArenaStage(
        store=store,
        workspace_root=repo,
        provider=MockLLMProvider([]),
        artifacts_root=artifacts_root,
        sandbox_root=tmp_path / "sandboxes",
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="worker",
            reviewer_driver="judge",
        ),
        driver_registry=DriverRegistry(
            {
                "worker": _ScriptedDriver("worker", ok_worker),
                "judge": _ScriptedDriver("judge", judge_too_many_grafts),
            }
        ),
    )
    with pytest.raises(CraftPause, match="exceeding max_graft_items=2"):
        stage.arena_race(
            run_id=run_id,
            raw_input="Fix calc.compute.",
            state={"arena_rubric": _sample_rubric().model_dump(mode="json")},
            max_graft_items=2,
        )
    assert not (artifacts_root / run_id / "arena" / "verdict.json").exists()
    assert not any(ev.event_type == "arena.judged" for ev in store.list_events(run_id))
    assert _git_worktree_paths(repo) == [repo.resolve()]


# ---------------------------------------------------------------------- 8. Round-2 edge fixes


def test_arena_graft_patch_write_failure_and_dir_mode_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-2 #1: grafted.patch write failure leaves craft untouched; directory mode restored."""
    import os
    import stat

    from olympus.harness.worktree import SandboxHandle, snapshot_sandbox_state
    from olympus.stages import arena as arena_mod

    # Sub-case A: grafted.patch write fails -> craft sandbox keeps pristine winner code.
    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    store = RunStore(tmp_path / "olympus.db")
    run_id = store.create_run(
        title="Arena grafted.patch write failure",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_arena",
    )
    sandbox_root = tmp_path / "sandboxes"
    artifacts_root = tmp_path / "artifacts"
    winner_code = "def compute(x: int, y: int) -> int:\n    return x + y\n"
    graft_code = "def compute(x: int, y: int) -> int:\n    return int(x) + int(y)\n"

    # Pre-create grafted.patch as a directory so write_text raises IsADirectoryError (OSError).
    (artifacts_root / run_id / "arena" / "grafted.patch").mkdir(parents=True, exist_ok=True)

    def worker(req: DriverTurnRequest) -> str:
        if req.actor == "hephaestus:arena_graft":
            (req.workspace_path / "calc.py").write_text(graft_code, encoding="utf-8")
            return "grafted"
        (req.workspace_path / "calc.py").write_text(winner_code, encoding="utf-8")
        return "built"

    def judge(req: DriverTurnRequest) -> str:
        del req
        return ArenaVerdict(
            winner_id="cand_1",
            scores={
                "cand_1": {"correctness": 10.0, "simplicity": 9.5, "clarity": 9.0},
                "cand_2": {"correctness": 10.0, "simplicity": 8.0, "clarity": 9.5},
            },
            graft_items=[
                ArenaGraftItem(
                    source_candidate_id="cand_2",
                    idea="Coerce inputs with int().",
                    target_files=["calc.py"],
                )
            ],
            convergence_mode="winner_plus_graft",
            rationale="cand_1 wins with int() graft.",
        ).model_dump_json()

    stage = ArenaStage(
        store=store,
        workspace_root=repo,
        provider=MockLLMProvider([]),
        artifacts_root=artifacts_root,
        sandbox_root=sandbox_root,
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="worker",
            reviewer_driver="judge",
        ),
        driver_registry=DriverRegistry(
            {
                "worker": _ScriptedDriver("worker", worker),
                "judge": _ScriptedDriver("judge", judge),
            }
        ),
    )
    with pytest.raises(CraftPause, match="Failed to persist grafted.patch"):
        stage.arena_race(
            run_id=run_id,
            raw_input="Fix calc.compute.",
            state={"arena_rubric": _sample_rubric().model_dump(mode="json")},
        )
    assert (sandbox_root / f"{run_id}-craft" / "calc.py").read_text(
        encoding="utf-8"
    ) == winner_code
    assert not (sandbox_root / f"{run_id}-arena-graft").exists()

    # Sub-case B: _mirror_sandbox_changes deletes a file inside a 0o700 dir, then hits OSError
    # on a later copy -> removed 0o700 dir and deleted file are restored with exact mode 0o700.
    base_dir = tmp_path / "base_ws"
    src_dir = tmp_path / "src_ws"
    dst_dir = tmp_path / "dst_ws"
    for d in (base_dir, src_dir, dst_dir):
        d.mkdir()
        (d / "keep.py").write_text("v0\n", encoding="utf-8")
    base_handle = SandboxHandle(base_dir, base_dir, None, False)
    base_baseline = snapshot_sandbox_state(base_handle)

    secret_dir = dst_dir / "private_dir"
    secret_dir.mkdir(mode=0o700)
    os.chmod(secret_dir, 0o700)
    old_file = secret_dir / "old.py"
    old_file.write_text("winner-only\n", encoding="utf-8")
    os.chmod(old_file, 0o600)

    # Source removes private_dir/old.py and updates keep.py; inject OSError on copy2(keep.py).
    (src_dir / "keep.py").write_text("v1-graft\n", encoding="utf-8")

    def boom_copy2(_src: Any, _dst: Any) -> None:
        raise OSError("disk full during mirror")

    monkeypatch.setattr(arena_mod.shutil, "copy2", boom_copy2)
    with pytest.raises(CraftPause, match="rolled back"):
        arena_mod._mirror_sandbox_changes(
            source_handle=SandboxHandle(base_dir, src_dir, None, False),
            target_handle=SandboxHandle(base_dir, dst_dir, None, False),
            base_baseline=base_baseline,
            allowed_write_scope=["private_dir/old.py", "keep.py"],
            forbidden_globs=[],
            node_name="arena_race",
        )
    assert secret_dir.is_dir()
    assert stat.S_IMODE(secret_dir.stat().st_mode) == 0o700
    assert old_file.read_text(encoding="utf-8") == "winner-only\n"
    assert stat.S_IMODE(old_file.stat().st_mode) == 0o600
    assert (dst_dir / "keep.py").read_text(encoding="utf-8") == "v0\n"


def test_strict_remove_sandbox_fails_when_git_worktree_list_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-2 #2: non-zero exit from git worktree list --porcelain must fail strict cleanup."""
    from olympus.harness import worktree as wt_mod

    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    handle = wt_mod.create_sandbox(
        repo,
        label="strict-verify-test",
        reuse=False,
        sandbox_root=tmp_path / "sandboxes",
    )
    real_run_git = wt_mod._run_git

    def failing_list_git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        if args[:2] == ("worktree", "list"):
            return subprocess.CompletedProcess(
                ["git", *args],
                returncode=128,
                stdout="",
                stderr="fatal: unable to read worktree registry",
            )
        return real_run_git(cwd, *args)

    monkeypatch.setattr(wt_mod, "_run_git", failing_list_git)
    with pytest.raises(RuntimeError, match="git worktree list --porcelain failed \\(exit=128\\)"):
        wt_mod.remove_sandbox(handle, strict=True)

    # Also verify locked worktree is cleaned cleanly via double --force.
    monkeypatch.setattr(wt_mod, "_run_git", real_run_git)
    locked_handle = wt_mod.create_sandbox(
        repo,
        label="locked-wt-test",
        reuse=False,
        sandbox_root=tmp_path / "sandboxes",
    )
    subprocess.run(
        ["git", "worktree", "lock", str(locked_handle.workspace_path)],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    wt_mod.remove_sandbox(locked_handle, strict=True)
    assert not locked_handle.workspace_path.exists()
    assert _git_worktree_paths(repo) == [repo.resolve()]


def test_arena_craft_sandbox_init_failure_cleans_up_craft_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-2 #3: failure during -craft sandbox init immediately cleans up -craft worktree."""
    from olympus.stages import arena as arena_mod

    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    store = RunStore(tmp_path / "olympus.db")
    run_id = store.create_run(
        title="Arena craft init failure cleanup",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_arena",
    )
    sandbox_root = tmp_path / "sandboxes"

    real_save_baseline = arena_mod.save_sandbox_baseline

    def fail_on_craft_baseline(handle: Any, baseline: Any) -> None:
        if handle.workspace_path.name.endswith("-craft"):
            raise OSError("injected craft baseline save failure")
        real_save_baseline(handle, baseline)

    monkeypatch.setattr(arena_mod, "save_sandbox_baseline", fail_on_craft_baseline)

    def ok_worker(req: DriverTurnRequest) -> str:
        (req.workspace_path / "calc.py").write_text(
            "def compute(x: int, y: int) -> int:\n    return x + y\n",
            encoding="utf-8",
        )
        return "ok"

    def ok_judge(req: DriverTurnRequest) -> str:
        del req
        return ArenaVerdict(
            winner_id="cand_1",
            scores={
                "cand_1": {"correctness": 10.0, "simplicity": 9.0, "clarity": 9.0},
                "cand_2": {"correctness": 10.0, "simplicity": 8.0, "clarity": 8.0},
            },
            rationale="cand_1 wins.",
        ).model_dump_json()

    stage = ArenaStage(
        store=store,
        workspace_root=repo,
        provider=MockLLMProvider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=sandbox_root,
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="worker",
            reviewer_driver="judge",
        ),
        driver_registry=DriverRegistry(
            {
                "worker": _ScriptedDriver("worker", ok_worker),
                "judge": _ScriptedDriver("judge", ok_judge),
            }
        ),
    )
    with pytest.raises(OSError, match="injected craft baseline save failure"):
        stage.arena_race(
            run_id=run_id,
            raw_input="Fix calc.compute.",
            state={"arena_rubric": _sample_rubric().model_dump(mode="json")},
        )
    assert not (sandbox_root / f"{run_id}-craft").exists()
    assert not (sandbox_root / f"{run_id}-arena-cand_1").exists()
    assert not (sandbox_root / f"{run_id}-arena-cand_2").exists()
    assert _git_worktree_paths(repo) == [repo.resolve()]


def test_arena_unborn_head_git_repo_locks_single_frozen_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-2 #4: unborn-HEAD git repo snapshots one immutable baseline before candidate
    allocation so a concurrent initial commit never skews baselines across candidates/craft.
    """
    from olympus.stages import arena as arena_mod

    repo = tmp_path / "unborn_repo"
    repo.mkdir()
    initial_readme = "# Initial Unborn Baseline\n"
    (repo / "calc.py").write_text(
        "def compute(x: int, y: int) -> int:\n    return x - y\n",
        encoding="utf-8",
    )
    (repo / "README.md").write_text(initial_readme, encoding="utf-8")
    (repo / "test_calc.py").write_text(
        "import unittest\n"
        "import calc\n\n"
        "class TestCalc(unittest.TestCase):\n"
        "    def test_add(self) -> None:\n"
        "        self.assertEqual(calc.compute(2, 3), 5)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    # Intentionally NO git commit yet -> HEAD is unborn.

    store = RunStore(tmp_path / "olympus.db")
    run_id = store.create_run(
        title="Arena unborn HEAD frozen baseline",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_arena",
    )
    sandbox_root = tmp_path / "sandboxes"

    real_create = arena_mod.create_sandbox
    created_readmes: list[str] = []
    committed_first = False

    def create_with_concurrent_first_commit(repo_root: Path | str, **kwargs: Any) -> Any:
        nonlocal committed_first
        handle = real_create(repo_root, **kwargs)
        created_readmes.append((handle.workspace_path / "README.md").read_text(encoding="utf-8"))
        if not committed_first:
            committed_first = True
            # Mutate host repo and create its very first commit right after cand_1 is spawned.
            (repo / "README.md").write_text("# Mutated After First Commit\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(
                ["git", "commit", "-m", "initial commit mid-race"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
        return handle

    monkeypatch.setattr(arena_mod, "create_sandbox", create_with_concurrent_first_commit)

    def ok_worker(req: DriverTurnRequest) -> str:
        (req.workspace_path / "calc.py").write_text(
            "def compute(x: int, y: int) -> int:\n    return x + y\n",
            encoding="utf-8",
        )
        return "ok"

    def ok_judge(req: DriverTurnRequest) -> str:
        del req
        return ArenaVerdict(
            winner_id="cand_1",
            scores={
                "cand_1": {"correctness": 10.0, "simplicity": 9.0, "clarity": 9.0},
                "cand_2": {"correctness": 10.0, "simplicity": 8.0, "clarity": 8.0},
            },
            rationale="cand_1 wins cleanly.",
        ).model_dump_json()

    stage = ArenaStage(
        store=store,
        workspace_root=repo,
        provider=MockLLMProvider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=sandbox_root,
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="worker",
            reviewer_driver="judge",
        ),
        driver_registry=DriverRegistry(
            {
                "worker": _ScriptedDriver("worker", ok_worker),
                "judge": _ScriptedDriver("judge", ok_judge),
            }
        ),
    )
    out = stage.arena_race(
        run_id=run_id,
        raw_input="Fix calc.compute.",
        state={"arena_rubric": _sample_rubric().model_dump(mode="json")},
    )
    assert out["route"] == "pass"
    # cand_1, cand_2, and -craft ALL saw the exact initial unborn baseline README.
    assert len(created_readmes) == 3
    assert all(content == initial_readme for content in created_readmes)
    # Temporary frozen baseline snapshot dir and candidate dirs are cleaned up.
    assert not (sandbox_root / f"{run_id}-arena-base").exists()
    assert not (sandbox_root / f"{run_id}-arena-cand_1").exists()
    assert not (sandbox_root / f"{run_id}-arena-cand_2").exists()
    assert (sandbox_root / f"{run_id}-craft" / "README.md").read_text(
        encoding="utf-8"
    ) == initial_readme


# ---------------------------------------------------------------------- 9. Round-3 counterexamples


def test_mirror_rollback_preserves_preexisting_empty_and_replaced_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-3 #1: rollback removes only newly created dirs and never deletes preexisting
    empty directories or directories emptied and reused during mirroring.
    """
    import os
    import stat

    from olympus.harness.worktree import SandboxHandle, snapshot_sandbox_state
    from olympus.stages import arena as arena_mod

    base_dir = tmp_path / "base_ws"
    src_dir = tmp_path / "src_ws"
    dst_dir = tmp_path / "dst_ws"
    for d in (base_dir, src_dir, dst_dir):
        d.mkdir()
        (d / "fail.py").write_text("v0\n", encoding="utf-8")
    base_baseline = snapshot_sandbox_state(SandboxHandle(base_dir, base_dir, None, False))

    # Preexisting empty 0o700 directory in dst_dir that graft writes a new file into.
    preexisting_empty = dst_dir / "preexisting_empty"
    preexisting_empty.mkdir(mode=0o700)
    os.chmod(preexisting_empty, 0o700)

    # Preexisting 0o700 dir with old.py that graft replaces with new.py.
    replaced_dir = dst_dir / "replaced_dir"
    replaced_dir.mkdir(mode=0o700)
    os.chmod(replaced_dir, 0o700)
    (replaced_dir / "old.py").write_text("old\n", encoding="utf-8")

    # Source writes into preexisting_empty, brand_new_dir, replaces replaced_dir/old.py -> new.py,
    # and modifies fail.py (which triggers OSError).
    (src_dir / "preexisting_empty").mkdir()
    (src_dir / "preexisting_empty" / "added.py").write_text("added\n", encoding="utf-8")
    (src_dir / "brand_new_dir").mkdir()
    (src_dir / "brand_new_dir" / "fresh.py").write_text("fresh\n", encoding="utf-8")
    (src_dir / "replaced_dir").mkdir()
    (src_dir / "replaced_dir" / "new.py").write_text("new\n", encoding="utf-8")
    (src_dir / "fail.py").write_text("v1\n", encoding="utf-8")

    real_copy2 = arena_mod.shutil.copy2

    def fail_on_fail_py(src: Path, dst: Path) -> Any:
        if Path(src).name == "fail.py":
            raise OSError("injected failure on fail.py")
        return real_copy2(src, dst)

    monkeypatch.setattr(arena_mod.shutil, "copy2", fail_on_fail_py)
    with pytest.raises(CraftPause, match="rolled back"):
        arena_mod._mirror_sandbox_changes(
            source_handle=SandboxHandle(base_dir, src_dir, None, False),
            target_handle=SandboxHandle(base_dir, dst_dir, None, False),
            base_baseline=base_baseline,
            allowed_write_scope=[
                "preexisting_empty/added.py",
                "brand_new_dir/fresh.py",
                "replaced_dir/old.py",
                "replaced_dir/new.py",
                "fail.py",
            ],
            forbidden_globs=[],
            node_name="arena_race",
        )

    # Preexisting empty 0o700 dir MUST still exist with mode 0o700 and added.py removed.
    assert preexisting_empty.is_dir()
    assert stat.S_IMODE(preexisting_empty.stat().st_mode) == 0o700
    assert not (preexisting_empty / "added.py").exists()

    # Preexisting replaced_dir MUST still exist with mode 0o700 and old.py restored (new.py gone).
    assert replaced_dir.is_dir()
    assert stat.S_IMODE(replaced_dir.stat().st_mode) == 0o700
    assert (replaced_dir / "old.py").read_text(encoding="utf-8") == "old\n"
    assert not (replaced_dir / "new.py").exists()

    # Truly newly created brand_new_dir MUST be cleaned up.
    assert not (dst_dir / "brand_new_dir").exists()
    assert (dst_dir / "fail.py").read_text(encoding="utf-8") == "v0\n"


def test_strict_remove_sandbox_fails_when_git_repo_probe_exits_128(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-3 #2: if git rev-parse probe returns 128 on a git worktree handle,
    remove_sandbox(strict=True) must raise RuntimeError instead of skipping cleanup.
    """
    from olympus.harness import worktree as wt_mod

    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    handle = wt_mod.create_sandbox(
        repo,
        label="probe-128-test",
        reuse=False,
        sandbox_root=tmp_path / "sandboxes",
    )
    real_run_git = wt_mod._run_git

    def probe_exits_128(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        if args[:2] == ("rev-parse", "--is-inside-work-tree"):
            return subprocess.CompletedProcess(
                ["git", *args],
                returncode=128,
                stdout="",
                stderr="fatal: dubious ownership in repository",
            )
        return real_run_git(cwd, *args)

    monkeypatch.setattr(wt_mod, "_run_git", probe_exits_128)
    with pytest.raises(
        RuntimeError, match="git rev-parse --is-inside-work-tree failed \\(exit=128\\)"
    ):
        wt_mod.remove_sandbox(handle, strict=True)

    # Restore git and clean up the worktree properly afterwards.
    monkeypatch.setattr(wt_mod, "_run_git", real_run_git)
    wt_mod.remove_sandbox(handle, strict=True)
    assert _git_worktree_paths(repo) == [repo.resolve()]


def test_create_sandbox_cleans_up_on_base_exception_after_git_worktree_add(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-3 #3: BaseException (e.g. KeyboardInterrupt) after git worktree add before
    return must immediately clean up the directory and git worktree registration.
    """
    from olympus.harness import worktree as wt_mod

    repo = tmp_path / "repo"
    _init_demo_repo(repo)
    sandbox_root = tmp_path / "sandboxes"
    real_handle_cls = wt_mod.SandboxHandle

    def interrupted_handle(*args: Any, **kwargs: Any) -> Any:
        # Simulate KeyboardInterrupt right after git worktree add succeeded.
        if kwargs.get("is_git_worktree") is True:
            raise KeyboardInterrupt("simulated interrupt after git worktree add")
        return real_handle_cls(*args, **kwargs)

    monkeypatch.setattr(wt_mod, "SandboxHandle", interrupted_handle)
    with pytest.raises(KeyboardInterrupt, match="simulated interrupt"):
        wt_mod.create_sandbox(
            repo,
            label="interrupt-after-add",
            reuse=False,
            sandbox_root=sandbox_root,
        )

    assert not (sandbox_root / "interrupt-after-add").exists()
    assert _git_worktree_paths(repo) == [repo.resolve()]


def test_arena_unborn_head_default_sandbox_root_none_locks_baseline_across_candidates_and_graft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-3 #4: with default sandbox_root=None inside an unborn-HEAD git repo, a mid-race
    first commit must NOT leak into cand_2, graft, or -craft sandboxes.
    """
    from olympus.stages import arena as arena_mod

    repo = tmp_path / "unborn_default_repo"
    repo.mkdir()
    initial_readme = "# Frozen Default-Sandbox Unborn Baseline\n"
    (repo / "calc.py").write_text(
        "def compute(x: int, y: int) -> int:\n    return x - y\n",
        encoding="utf-8",
    )
    (repo / "README.md").write_text(initial_readme, encoding="utf-8")
    (repo / "test_calc.py").write_text(
        "import unittest\n"
        "import calc\n\n"
        "class TestCalc(unittest.TestCase):\n"
        "    def test_add(self) -> None:\n"
        "        self.assertEqual(calc.compute(2, 3), 5)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)

    store = RunStore(tmp_path / "olympus.db")
    run_id = store.create_run(
        title="Arena unborn HEAD sandbox_root=None",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_arena",
    )

    real_create = arena_mod.create_sandbox
    seen_readmes: list[tuple[str, str]] = []
    committed_first = False

    def intercept_create(repo_root: Path | str, **kwargs: Any) -> Any:
        nonlocal committed_first
        handle = real_create(repo_root, **kwargs)
        seen_readmes.append(
            (kwargs["label"], (handle.workspace_path / "README.md").read_text(encoding="utf-8"))
        )
        if not committed_first:
            committed_first = True
            # Mutate host repo and create its very first commit right after cand_1 is created!
            (repo / "README.md").write_text("# Host First Commit Mid-Race\n", encoding="utf-8")
            subprocess.run(
                ["git", "add", "calc.py", "README.md", "test_calc.py"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "initial commit mid-race"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
        return handle

    monkeypatch.setattr(arena_mod, "create_sandbox", intercept_create)

    def worker(req: DriverTurnRequest) -> str:
        if req.actor == "hephaestus:arena_graft":
            (req.workspace_path / "calc.py").write_text(
                "def compute(x: int, y: int) -> int:\n    return int(x) + int(y)\n",
                encoding="utf-8",
            )
            return "grafted"
        (req.workspace_path / "calc.py").write_text(
            "def compute(x: int, y: int) -> int:\n    return x + y\n",
            encoding="utf-8",
        )
        return "ok"

    def judge(req: DriverTurnRequest) -> str:
        del req
        return ArenaVerdict(
            winner_id="cand_1",
            scores={
                "cand_1": {"correctness": 10.0, "simplicity": 9.5, "clarity": 9.0},
                "cand_2": {"correctness": 10.0, "simplicity": 8.0, "clarity": 9.5},
            },
            graft_items=[
                ArenaGraftItem(
                    source_candidate_id="cand_2",
                    idea="Coerce with int().",
                    target_files=["calc.py"],
                )
            ],
            convergence_mode="winner_plus_graft",
            rationale="cand_1 wins with int() graft.",
        ).model_dump_json()

    # sandbox_root is intentionally omitted (None -> repo/.olympus/sandboxes).
    stage = ArenaStage(
        store=store,
        workspace_root=repo,
        provider=MockLLMProvider([]),
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=None,
        execution_drivers=ExecutionDriversConfig.resolve(
            stage_driver="native",
            worker_driver="worker",
            reviewer_driver="judge",
        ),
        driver_registry=DriverRegistry(
            {
                "worker": _ScriptedDriver("worker", worker),
                "judge": _ScriptedDriver("judge", judge),
            }
        ),
    )
    out = stage.arena_race(
        run_id=run_id,
        raw_input="Fix calc.compute.",
        state={"arena_rubric": _sample_rubric().model_dump(mode="json")},
    )
    assert out["route"] == "pass"
    # cand_1, cand_2, -craft, and -arena-graft ALL saw the exact frozen initial unborn README!
    assert len(seen_readmes) == 4
    assert all(text == initial_readme for _label, text in seen_readmes), seen_readmes

    default_sandboxes = repo / ".olympus" / "sandboxes"
    assert not (default_sandboxes / f"{run_id}-arena-base").exists()
    assert not (default_sandboxes / f"{run_id}-arena-cand_1").exists()
    assert not (default_sandboxes / f"{run_id}-arena-cand_2").exists()
    assert not (default_sandboxes / f"{run_id}-arena-graft").exists()
    assert (default_sandboxes / f"{run_id}-craft" / "README.md").read_text(
        encoding="utf-8"
    ) == initial_readme
    assert _git_worktree_paths(repo) == [repo.resolve()]


# ---------------------------------------------------------------------- 10. Round-4 edge fixes


def test_unborn_head_default_sandbox_preserves_graft_through_hygiene_pass(
    tmp_path: Path,
) -> None:
    """Round-4 P1: unborn-HEAD git repo with sandbox_root=None runs arena_race (with graft)
    followed by CraftStage.hygiene_pass; hygiene reuses -craft and sees grafted code intact.
    """
    from olympus.stages.craft import CraftStage

    repo = tmp_path / "unborn_hygiene_repo"
    repo.mkdir()
    (repo / "calc.py").write_text(
        "def compute(x: int, y: int) -> int:\n    return x - y\n",
        encoding="utf-8",
    )
    (repo / "test_calc.py").write_text(
        "import unittest\n"
        "import calc\n\n"
        "class TestCalc(unittest.TestCase):\n"
        "    def test_add(self) -> None:\n"
        "        self.assertEqual(calc.compute(2, 3), 5)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    # Intentionally unborn HEAD (no commits).

    store = RunStore(tmp_path / "olympus.db")
    run_id = store.create_run(
        title="Arena unborn HEAD -> hygiene retention",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_arena",
    )
    artifacts_root = tmp_path / "artifacts"
    grafted_code = (
        "def compute(x: int, y: int) -> int:\n"
        "    # grafted helper comment\n"
        "    return int(x) + int(y)\n"
    )
    hygiene_seen_code: list[str] = []

    def worker(req: DriverTurnRequest) -> str:
        if req.actor == "hephaestus:arena_graft":
            (req.workspace_path / "calc.py").write_text(grafted_code, encoding="utf-8")
            return "grafted"
        if req.actor == "hephaestus:hygiene":
            hygiene_seen_code.append((req.workspace_path / "calc.py").read_text(encoding="utf-8"))
            (req.workspace_path / "calc.py").write_text(
                "def compute(x: int, y: int) -> int:\n    return int(x) + int(y)\n",
                encoding="utf-8",
            )
            return "hygiene ok"
        (req.workspace_path / "calc.py").write_text(
            "def compute(x: int, y: int) -> int:\n    return x + y\n",
            encoding="utf-8",
        )
        return "ok"

    def judge(req: DriverTurnRequest) -> str:
        del req
        return ArenaVerdict(
            winner_id="cand_1",
            scores={
                "cand_1": {"correctness": 10.0, "simplicity": 9.5, "clarity": 9.0},
                "cand_2": {"correctness": 10.0, "simplicity": 8.0, "clarity": 9.5},
            },
            graft_items=[
                ArenaGraftItem(
                    source_candidate_id="cand_2",
                    idea="Coerce inputs with int().",
                    target_files=["calc.py"],
                )
            ],
            convergence_mode="winner_plus_graft",
            rationale="cand_1 wins with int() graft.",
        ).model_dump_json()

    drivers_cfg = ExecutionDriversConfig.resolve(
        stage_driver="native",
        worker_driver="worker",
        reviewer_driver="judge",
    )
    registry = DriverRegistry(
        {
            "worker": _ScriptedDriver("worker", worker),
            "judge": _ScriptedDriver("judge", judge),
        }
    )
    arena_stage = ArenaStage(
        store=store,
        workspace_root=repo,
        provider=MockLLMProvider([]),
        artifacts_root=artifacts_root,
        sandbox_root=None,
        execution_drivers=drivers_cfg,
        driver_registry=registry,
    )
    arena_out = arena_stage.arena_race(
        run_id=run_id,
        raw_input="Fix calc.compute.",
        state={"arena_rubric": _sample_rubric().model_dump(mode="json")},
    )
    assert arena_out["route"] == "pass"

    craft_stage = CraftStage(
        store=store,
        workspace_root=repo,
        provider=MockLLMProvider([]),
        artifacts_root=artifacts_root,
        sandbox_root=None,
        execution_drivers=drivers_cfg,
        driver_registry=registry,
    )
    hygiene_out = craft_stage.hygiene_pass(run_id=run_id, state=arena_out)
    assert hygiene_out["route"] == "clean"
    assert hygiene_seen_code == [grafted_code]
    assert (repo / ".olympus" / "sandboxes" / f"{run_id}-craft" / "calc.py").read_text(
        encoding="utf-8"
    ) == "def compute(x: int, y: int) -> int:\n    return int(x) + int(y)\n"


def test_mirror_nested_mkdir_enospc_rolls_back_created_parent_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-4 P2: ENOSPC during nested mkdir rolls back already-created shallow parent dirs."""
    import errno

    from olympus.harness.worktree import SandboxHandle, snapshot_sandbox_state
    from olympus.stages import arena as arena_mod

    base_dir = tmp_path / "base_ws"
    src_dir = tmp_path / "src_ws"
    dst_dir = tmp_path / "dst_ws"
    for d in (base_dir, src_dir, dst_dir):
        d.mkdir()
        (d / "keep.py").write_text("v0\n", encoding="utf-8")
    base_baseline = snapshot_sandbox_state(SandboxHandle(base_dir, base_dir, None, False))

    (src_dir / "brand_new" / "deep_child").mkdir(parents=True)
    (src_dir / "brand_new" / "deep_child" / "mod.py").write_text("x = 1\n", encoding="utf-8")

    real_mkdir = Path.mkdir

    def enospc_on_deep_child(self: Path, *args: Any, **kwargs: Any) -> None:
        if self.name == "deep_child" and self.parent.name == "brand_new":
            raise OSError(errno.ENOSPC, "No space left on device")
        real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", enospc_on_deep_child)
    with pytest.raises(CraftPause, match="rolled back"):
        arena_mod._mirror_sandbox_changes(
            source_handle=SandboxHandle(base_dir, src_dir, None, False),
            target_handle=SandboxHandle(base_dir, dst_dir, None, False),
            base_baseline=base_baseline,
            allowed_write_scope=["brand_new/deep_child/mod.py"],
            forbidden_globs=[],
            node_name="arena_race",
        )

    assert not (dst_dir / "brand_new").exists()
    assert (dst_dir / "keep.py").read_text(encoding="utf-8") == "v0\n"


