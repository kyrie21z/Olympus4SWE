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


