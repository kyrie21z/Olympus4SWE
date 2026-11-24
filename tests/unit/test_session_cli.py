"""Seam 1: offline deterministic tests for ``olympus session start/reply``.

Every test drives the real CLI through Typer's ``CliRunner`` against a real
temporary git repository, with a scripted ``MockLLMProvider`` and an external
state directory, so the observable contract is exactly what an external
interactive harness sees.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from olympus.cli.app import app
from olympus.core.contracts import ApprovalTarget, GoalSpec, RunStatus, TaskStatus
from olympus.core.store import RunStore
from olympus.graph.loader import WORKFLOW_DIR
from olympus.providers.base import ChatMessage, LLMResponse, MockLLMProvider, ToolCall, ToolSpec
from olympus.session.contracts import PauseKind
from olympus.session.router import parse_reply
from olympus.session.runner import resolve_state_dir

RUNNER = CliRunner()

_CALC_CODE = (
    "def divide(a, b):\n"
    "    if b == 0:\n"
    "        raise ValueError('zero')\n"
    "    return a / b\n"
)
_CALC_CODE_V2 = _CALC_CODE + "\n\ndef subtract(a, b):\n    return a - b\n"

_DIVIDE_CHECK = (
    "python3 -c \"import calculator; assert calculator.divide(6, 2) == 3\""
)
_SUBTRACT_CHECK = (
    "python3 -c \"import calculator; assert calculator.subtract(5, 2) == 3\""
)
_FAILING_CHECK = 'python3 -c "raise SystemExit(1)"'


def _flag_check(flag: Path) -> str:
    return (
        'python3 -c "import calculator, pathlib; '
        "assert calculator.divide(6, 2) == 3; "
        f"assert pathlib.Path('{flag}').is_file()\""
    )


# --------------------------------------------------------------------------- fixtures


def _init_git_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# Calculator\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Dev"], cwd=root, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True
    )


def _git_status(root: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


class SessionScript:
    """Deterministic provider script keyed by stage system-prompt markers."""

    def __init__(self) -> None:
        self.align: list[str] = []
        self.design: list[str] = []
        self.plan: list[str] = []
        self.router: list[str] = []
        self.workers: list[tuple[str, str] | None] = []
        self.worker_rounds = 0
        self.planner_prompts: list[str] = []
        self.worker_prompts: list[str] = []

    def provider(self) -> MockLLMProvider:
        return MockLLMProvider(responder=self._respond)

    def _respond(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        del tools
        system = messages[0].content if messages else ""
        user = messages[1].content if len(messages) > 1 else ""
        if "Session Reply Router" in system:
            assert self.router, "router response queue exhausted"
            return LLMResponse(content=self.router.pop(0))
        if "Requirements Alignment Architect" in system:
            assert self.align, "align response queue exhausted"
            return LLMResponse(content=self.align.pop(0))
        if "White-Box Solution Architect" in system:
            assert self.design, "design response queue exhausted"
            return LLMResponse(content=self.design.pop(0))
        if "Task Planner" in system:
            self.planner_prompts.append(user)
            assert self.plan, "plan response queue exhausted"
            return LLMResponse(content=self.plan.pop(0))
        if "Independent Verifier" in system:
            return LLMResponse(
                content=json.dumps(
                    {
                        "task_id": "T1",
                        "attempt": 1,
                        "passed": True,
                        "summary": "Implementation matches the frozen contract.",
                        "findings": [],
                    }
                )
            )
        if "Execution Engineer" in system:
            if any(message.role == "tool" for message in messages):
                self.worker_rounds += 1
                return LLMResponse(content="Implementation finished.")
            assert self.workers, "worker response queue exhausted"
            self.worker_prompts.append(user)
            entry = self.workers.pop(0)
            if entry is None:
                return LLMResponse(content="Implementation finished.")
            relative, content = entry
            match = re.search(r"Sandbox workspace root: (.+)", user)
            assert match is not None
            sandbox = Path(match.group(1).strip())
            return LLMResponse(
                content="Writing the calculator implementation.",
                tool_calls=(
                    ToolCall(
                        id=f"write-{self.worker_rounds}",
                        name="write_file",
                        arguments={"path": str(sandbox / relative), "content": content},
                    ),
                ),
            )
        raise AssertionError(f"Unexpected system prompt: {system[:120]}")


def _patch_provider(monkeypatch: pytest.MonkeyPatch, script: SessionScript) -> None:
    instance = script.provider()

    def _builder(
        *, provider: str | None = None, model: str | None = None, base_url: str | None = None
    ) -> MockLLMProvider:
        del provider, model, base_url
        return instance

    monkeypatch.setattr("olympus.cli.session_cmd.build_session_provider", _builder)


def _goal_spec(
    *,
    ambiguity: bool = False,
    two_ambiguities: bool = False,
    include_subtract: bool = False,
    version: int = 1,
) -> dict[str, object]:
    requirements: list[dict[str, object]] = [
        {
            "id": "R1",
            "description": (
                "Expose divide(a, b) returning a / b and rejecting a zero divisor"
            ),
            "actor": "developer",
            "ears_criteria": [
                {
                    "id": "R1-C1",
                    "pattern": "ubiquitous",
                    "outcome": "return a / b for any non-zero divisor",
                },
                {
                    "id": "R1-C2",
                    "pattern": "unwanted_behavior",
                    "condition": "b equals zero",
                    "outcome": "raise ValueError with message 'zero'",
                },
            ],
        }
    ]
    if include_subtract:
        requirements.append(
            {
                "id": "R2",
                "description": "Expose subtract(a, b) returning a - b",
                "actor": "developer",
                "ears_criteria": [
                    {
                        "id": "R2-C1",
                        "pattern": "ubiquitous",
                        "outcome": "return a - b for integer inputs",
                    }
                ],
            }
        )
    spec: dict[str, object] = {
        "spec_id": "spec-calc",
        "version": version,
        "goal": "Build a small Python calculator library for developers",
        "deliverable_kind": "library",
        "observable_surfaces": ["calculator module"],
        "actors": ["developer"],
        "test_seams": ["calculator.divide(a, b)"],
        "requirements": requirements,
        "out_of_scope": ["graphical interface"],
        "constraints": [],
    }
    if ambiguity:
        spec["open_ambiguities"] = [
            {
                "id": "AMB-1",
                "question": "What behavior should divide(1, 0) have?",
                "impact": "Error contract",
                "options": ["raise ValueError('zero')", "return None"],
                "default_assumption": "raise ValueError('zero')",
                "blocking": True,
            }
        ]
        if two_ambiguities:
            spec["open_ambiguities"].append(
                {
                    "id": "AMB-2",
                    "question": "Which numeric types must be accepted?",
                    "impact": "Input contract",
                    "options": ["integers only", "int and float"],
                    "default_assumption": "int and float",
                    "blocking": True,
                }
            )
    return spec


def _design(*, requirements: tuple[str, ...] = ("R1",)) -> dict[str, object]:
    return {
        "architecture_summary": "A single calculator module with an explicit error contract",
        "spec_id": "spec-calc",
        "spec_hash": "sha256:pending",
        "shared_invariants": [
            {
                "id": "INV-ERR",
                "category": "error_envelope",
                "rule": "Raise ValueError('zero') when the divisor is zero",
                "enforcement_hint": "guard clause before division",
                "applies_to": ["M1"],
            }
        ],
        "shared_contract_paths": [],
        "modules": [
            {
                "id": "M1",
                "name": "Calculator",
                "responsibility": "Provide arithmetic helpers with explicit errors",
                "requirement_refs": list(requirements),
                "target_paths": ["calculator.py"],
            }
        ],
        "data_models": [],
        "interfaces": [],
        "execution_strategy": "serial_handoff",
        "strategy_rationale": (
            "One module and one task keep the change atomic and easy to verify."
        ),
    }


def _plan(
    *,
    requirements: tuple[str, ...] = ("R1",),
    objective: str = "Implement divide(a, b) in calculator.py with a zero guard",
    command: str = _DIVIDE_CHECK,
    version: int = 1,
) -> dict[str, object]:
    return {
        "plan_id": "plan-calc",
        "spec_id": "spec-calc",
        "spec_hash": "sha256:pending",
        "version": version,
        "tasks": [
            {
                "task_id": "T1",
                "title": "Implement calculator helpers",
                "objective": objective,
                "requirement_refs": list(requirements),
                "design_refs": ["M1"],
                "write_scope": ["calculator.py"],
                "acceptance_checks": [
                    {
                        "id": "C1",
                        "kind": "command",
                        "description": "Verify the calculator contract",
                        "command": command,
                    }
                ],
            }
        ],
    }


def _invalid_overlapping_plan() -> dict[str, object]:
    return {
        "plan_id": "plan-bad",
        "spec_id": "spec-calc",
        "spec_hash": "sha256:pending",
        "version": 1,
        "tasks": [
            {
                "task_id": "T1",
                "title": "Write calculator",
                "objective": "Implement the calculator module",
                "requirement_refs": ["R1"],
                "design_refs": ["M1"],
                "write_scope": ["calculator.py"],
                "acceptance_checks": [
                    {
                        "id": "C1",
                        "kind": "command",
                        "description": "Check",
                        "command": _DIVIDE_CHECK,
                    }
                ],
            },
            {
                "task_id": "T2",
                "title": "Write calculator again",
                "objective": "Duplicate the same module concurrently",
                "requirement_refs": ["R1"],
                "design_refs": ["M1"],
                "write_scope": ["calculator.py"],
                "acceptance_checks": [
                    {
                        "id": "C2",
                        "kind": "command",
                        "description": "Check",
                        "command": _DIVIDE_CHECK,
                    }
                ],
            },
        ],
    }


def _invoke(args: list[str]) -> tuple[int, dict[str, object]]:
    result = RUNNER.invoke(app, ["session", *args])
    assert result.stdout.strip(), f"no JSON envelope on stdout (stderr={result.stderr})"
    return result.exit_code, json.loads(result.stdout)


def _base(repo: Path, state: Path) -> list[str]:
    return [
        "--workspace",
        str(repo),
        "--state-dir",
        str(state),
        "--provider",
        "mock",
    ]


# --------------------------------------------------------------------------- tests


def test_start_clarify_reply_then_execute_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec(ambiguity=True, two_ambiguities=True)))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.router.append(
        json.dumps(
            {
                "action": "clarify",
                "answers": {
                    "AMB-1": "raise ValueError('zero')",
                    "AMB-2": "int and float",
                },
            }
        )
    )
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    code, start = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0
    assert start["status"] == "needs_input"
    assert start["pause_kind"] == "clarify"
    assert start["unresolved_blockers"] == ["ambiguity:AMB-1", "ambiguity:AMB-2"]
    assert start["step_mode"] == "execute"
    assert start["workflow"] == "full_sdlc_tdd"
    assert start["changed_files"] == []
    assert start["error"] is None
    assert "AMB-1" in str(start["message"])
    assert "AMB-2" in str(start["message"])
    assert _git_status(repo) == ""
    assert not (repo / ".olympus").exists()

    code, reply = _invoke(
        [
            "reply",
            str(start["run_id"]),
            "Make divide(1, 0) raise ValueError('zero'), and accept int and float inputs.",
            *_base(repo, state),
        ]
    )
    assert code == 0, json.dumps(reply, indent=2)
    assert reply["status"] == "completed"
    assert reply["pause_kind"] is None
    assert reply["changed_files"] == ["calculator.py"]
    assert (repo / "calculator.py").read_text(encoding="utf-8") == _CALC_CODE
    assert not (repo / ".olympus").exists()
    assert (state / "olympus.db").is_file()

    store = RunStore(state / "olympus.db")
    snap = store.get_run(str(start["run_id"]))
    assert snap["status"].value == "completed"
    assert snap["design"] is not None and snap["plan"] is not None
    assert store.is_approved(
        run_id=str(start["run_id"]),
        target=ApprovalTarget.DESIGN_SPEC,
        target_id=snap["design"].design_id,
        target_hash=snap["design"].content_hash,
    )
    assert store.is_approved(
        run_id=str(start["run_id"]),
        target=ApprovalTarget.TASK_PLAN,
        target_id=snap["plan"].plan_id,
        target_hash=snap["plan"].content_hash,
    )


def test_plan_only_handoff_mutates_nothing_then_execute_applies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.router.append(json.dumps({"action": "approve", "message": "go ahead"}))
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    code, started = _invoke(
        [
            "start",
            "Build a calculator library",
            "--step-mode",
            "plan_only",
            *_base(repo, state),
        ]
    )
    assert code == 0
    assert started["status"] == "needs_input"
    assert started["pause_kind"] == "plan_handoff"
    assert started["step_mode"] == "plan_only"
    assert started["changed_files"] == []
    assert _git_status(repo) == ""
    assert not (repo / ".olympus").exists()
    assert (state / "olympus.db").is_file()

    code, executed = _invoke(
        [
            "reply",
            str(started["run_id"]),
            "Plan looks good, implement it.",
            "--step-mode",
            "execute",
            *_base(repo, state),
        ]
    )
    assert code == 0
    assert executed["status"] == "completed"
    assert executed["changed_files"] == ["calculator.py"]
    assert (repo / "calculator.py").is_file()
    assert not (repo / ".olympus").exists()


def test_default_gate_policy_pauses_for_every_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.router.extend(json.dumps({"action": "approve"}) for _ in range(3))
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    code, first = _invoke(
        ["start", "Build a calculator library", "--gate-policy", "default", *_base(repo, state)]
    )
    assert code == 0
    assert first["status"] == "needs_input"
    assert first["pause_kind"] == "approval"
    run_id = str(first["run_id"])

    code, second = _invoke(["reply", run_id, "Approve the spec.", *_base(repo, state)])
    assert code == 0
    assert second["pause_kind"] == "approval", json.dumps(second, indent=2)
    assert "DesignSpec" in str(second["message"])

    code, third = _invoke(["reply", run_id, "Approve the design.", *_base(repo, state)])
    assert code == 0
    assert third["pause_kind"] == "approval"
    assert "TaskPlan" in str(third["message"])

    code, final = _invoke(["reply", run_id, "Approve the plan.", *_base(repo, state)])
    assert code == 0
    assert final["status"] == "completed"
    assert (repo / "calculator.py").is_file()


def test_free_form_revision_restarts_alignment_with_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.extend(
        [
            json.dumps(_goal_spec(ambiguity=True)),
            json.dumps(_goal_spec(version=2)),
        ]
    )
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.router.append(
        json.dumps(
            {
                "action": "revise",
                "feedback": "Do not ask about zero division; always raise ValueError instead.",
            }
        )
    )
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    code, started = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0
    assert started["pause_kind"] == "clarify"
    run_id = str(started["run_id"])

    code, revised = _invoke(
        [
            "reply",
            run_id,
            "Instead of asking, always raise ValueError for zero division.",
            *_base(repo, state),
        ]
    )
    assert code == 0, json.dumps(revised, indent=2)
    assert revised["status"] == "completed"
    assert (repo / "calculator.py").is_file()

    store = RunStore(state / "olympus.db")
    snap = store.get_run(run_id)
    assert snap["spec"].version == 2
    event_types = [event.event_type for event in store.list_events(run_id)]
    assert "run.revision_requested" in event_types
    assert "session.continued" in event_types


def test_blueprint_workflow_completes_at_approved_plan_without_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    _patch_provider(monkeypatch, script)

    code, envelope = _invoke(
        [
            "start",
            "Build a calculator library",
            "--workflow",
            "product_blueprint",
            *_base(repo, state),
        ]
    )
    assert code == 0
    assert envelope["status"] == "completed"
    assert envelope["workflow"] == "product_blueprint"
    assert envelope["changed_files"] == []
    assert _git_status(repo) == ""


def test_malformed_reply_preserves_state_and_asks_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec(ambiguity=True)))
    script.router.append("this is definitely not json {{{")
    _patch_provider(monkeypatch, script)

    code, started = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0
    assert started["pause_kind"] == "clarify"
    run_id = str(started["run_id"])
    before = RunStore(state / "olympus.db").get_run(run_id)
    before_spec = before["spec"]

    code, after = _invoke(
        ["reply", run_id, "hmm, I am not sure what you mean", *_base(repo, state)]
    )
    assert code == 0
    assert after["status"] == "needs_input"
    assert after["pause_kind"] == "clarify"
    assert after["unresolved_blockers"] == ["ambiguity:AMB-1"]
    assert "AMB-1" in str(after["message"])
    current = RunStore(state / "olympus.db").get_run(run_id)
    assert current["spec"].version == before_spec.version
    assert current["spec"].content_hash == before_spec.content_hash
    assert current["status"].value == "awaiting_human"


def test_deterministic_validation_failure_pauses_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.extend(json.dumps(_invalid_overlapping_plan()) for _ in range(3))
    _patch_provider(monkeypatch, script)

    code, envelope = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0
    assert envelope["status"] == "needs_input"
    assert envelope["pause_kind"] == "blocked"
    assert "overlapping write_scope" in str(envelope["message"])
    assert envelope["changed_files"] == []
    assert _git_status(repo) == ""


def test_blocked_pause_hint_restarts_from_plan_and_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    hint = (
        "The host has no tests directory; verify calculator.py with a "
        "python3 -c import check instead."
    )
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.extend(json.dumps(_invalid_overlapping_plan()) for _ in range(3))
    script.plan.append(json.dumps(_plan()))
    script.router.append("invalid JSON")
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    code, blocked = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0, json.dumps(blocked, indent=2)
    assert blocked["pause_kind"] == "blocked"
    assert "overlapping write_scope" in str(blocked["message"])
    run_id = str(blocked["run_id"])

    code, recovered = _invoke(["reply", run_id, hint, *_base(repo, state)])
    assert code == 0, json.dumps(recovered, indent=2)
    assert recovered["status"] == "completed"
    assert recovered["changed_files"] == ["calculator.py"]
    assert (repo / "calculator.py").read_text(encoding="utf-8") == _CALC_CODE

    store = RunStore(state / "olympus.db")
    continued = [
        event.payload
        for event in store.list_events(run_id)
        if event.event_type == "session.continued"
    ]
    assert continued[-1]["recovery_node"] == "plan"
    assert hint in continued[-1]["raw_input"]
    assert hint in script.planner_prompts[-1]
    assert store.get_run(run_id)["status"].value == "completed"


def test_blocked_pause_hint_without_design_restarts_from_align(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.extend(
        [json.dumps(_goal_spec()), json.dumps(_goal_spec(version=2))]
    )
    script.design.extend(["not a DesignSpec"] * 3)
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.router.append("invalid JSON")
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    code, blocked = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0, json.dumps(blocked, indent=2)
    assert blocked["pause_kind"] == "blocked"

    code, recovered = _invoke(
        [
            "reply",
            str(blocked["run_id"]),
            "Keep the design to calculator.py only; do not add new modules.",
            *_base(repo, state),
        ]
    )
    assert code == 0, json.dumps(recovered, indent=2)
    assert recovered["status"] == "completed"
    assert (repo / "calculator.py").is_file()

    events = RunStore(state / "olympus.db").list_events(str(blocked["run_id"]))
    recovery = [
        event.payload
        for event in events
        if event.event_type == "session.continued"
    ][-1]
    assert recovery["recovery_node"] == "align"


def test_execution_blocked_hint_recovers_in_place_on_retained_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    unblock_flag = tmp_path / "human-unblock.flag"
    hint = "The flag was missing; keep calculator.py and continue without rewriting it."
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan(command=_flag_check(unblock_flag))))
    script.workers.extend([("calculator.py", _CALC_CODE)] * 3)
    script.workers.append(None)
    script.router.append(json.dumps({"action": "revise", "feedback": hint}))
    _patch_provider(monkeypatch, script)

    code, blocked = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0, json.dumps(blocked, indent=2)
    assert blocked["pause_kind"] == "blocked"
    run_id = str(blocked["run_id"])
    store = RunStore(state / "olympus.db")
    plan_before = store.get_run(run_id)["plan"]
    assert plan_before is not None
    blocked_task = plan_before.tasks[0]
    assert blocked_task.task_id == "T1"
    assert blocked_task.status == TaskStatus.BLOCKED
    assert blocked_task.attempt == 3
    assert blocked_task.review_cycle == 3
    assert blocked_task.in_place_recoveries == 0
    assert len(script.worker_prompts) == 3
    assert len(script.planner_prompts) == 1

    sandbox = state / "sandboxes" / f"{run_id}-T1"
    assert (sandbox / "calculator.py").read_text(encoding="utf-8") == _CALC_CODE
    unblock_flag.write_text("ready\n", encoding="utf-8")

    code, recovered = _invoke(["reply", run_id, hint, *_base(repo, state)])
    assert code == 0, json.dumps(recovered, indent=2)
    assert recovered["status"] == "completed"
    assert recovered["changed_files"] == ["calculator.py"]
    assert (repo / "calculator.py").read_text(encoding="utf-8") == _CALC_CODE

    plan_after = store.get_run(run_id)["plan"]
    assert plan_after is not None
    assert plan_after.plan_id == plan_before.plan_id
    task = plan_after.tasks[0]
    assert task.task_id == "T1"
    assert task.status == TaskStatus.ACCEPTED
    assert task.attempt == 4
    assert task.review_cycle == 0
    assert task.in_place_recoveries == 1
    assert len(script.planner_prompts) == 1
    assert script.plan == []
    assert script.align == []
    assert script.design == []

    assert len(script.worker_prompts) == 4
    repair_prompt = script.worker_prompts[-1]
    assert "[IN-PLACE REPAIR]" in repair_prompt
    assert f"Sandbox workspace root: {sandbox}" in repair_prompt
    assert f"[Human hint]: {hint}" in repair_prompt

    events = store.list_events(run_id)
    continued = [event.payload for event in events if event.event_type == "session.continued"]
    assert len(continued) == 1
    assert continued[0]["feedback"] == hint
    assert continued[0]["recovery_mode"] == "in_place"
    assert continued[0]["recovery_node"] == "execute"
    assert continued[0]["retried_task_ids"] == ["T1"]
    retried = [event for event in events if event.event_type == "task.retried_by_human"]
    assert len(retried) == 1
    assert retried[0].payload["hint"] == hint
    assert not any(event.event_type == "run.revision_requested" for event in events)


def test_execution_blocked_in_place_recovery_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    hints = [
        "Keep the current file and adjust only the zero guard.",
        "Inspect the retained file before editing it again.",
        "Try a smaller change inside calculator.py.",
    ]
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan(command=_FAILING_CHECK)))
    script.workers.extend([("calculator.py", _CALC_CODE)] * 9)
    script.router.extend(json.dumps({"action": "revise", "feedback": hint}) for hint in hints)
    _patch_provider(monkeypatch, script)

    code, blocked = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0, json.dumps(blocked, indent=2)
    assert blocked["pause_kind"] == "blocked"
    run_id = str(blocked["run_id"])
    store = RunStore(state / "olympus.db")

    for expected_in_place, expected_attempt in ((1, 6), (2, 9)):
        code, envelope = _invoke(
            ["reply", run_id, hints[expected_in_place - 1], *_base(repo, state)]
        )
        assert code == 0, json.dumps(envelope, indent=2)
        assert envelope["status"] == "needs_input"
        assert envelope["pause_kind"] == "blocked"
        task = store.get_run(run_id)["plan"].tasks[0]
        assert task.status == TaskStatus.BLOCKED
        assert task.in_place_recoveries == expected_in_place
        assert task.attempt == expected_attempt
        assert task.review_cycle == 3

    worker_calls_before = len(script.worker_prompts)
    planner_calls_before = len(script.planner_prompts)
    code, exhausted = _invoke(["reply", run_id, hints[2], *_base(repo, state)])
    assert code == 0, json.dumps(exhausted, indent=2)
    assert exhausted["status"] == "needs_input"
    assert exhausted["pause_kind"] == "blocked"
    assert "max_in_place_recoveries = 2" in str(exhausted["message"])
    task = store.get_run(run_id)["plan"].tasks[0]
    assert task.status == TaskStatus.BLOCKED
    assert task.in_place_recoveries == 2
    assert task.attempt == 9
    assert len(script.worker_prompts) == worker_calls_before
    assert len(script.planner_prompts) == planner_calls_before
    assert script.workers == []
    assert store.get_run(run_id)["status"] == RunStatus.AWAITING_HUMAN

    events = store.list_events(run_id)
    continued = [event for event in events if event.event_type == "session.continued"]
    assert len(continued) == 2
    assert all(event.payload["recovery_mode"] == "in_place" for event in continued)
    assert all(event.payload["recovery_node"] == "execute" for event in continued)
    assert not any(event.event_type == "run.revision_requested" for event in events)


def test_planning_blocked_recovery_is_bounded_at_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    hint = "Keep the write scopes disjoint and retry the plan."
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.extend(json.dumps(_invalid_overlapping_plan()) for _ in range(12))
    script.router.extend(json.dumps({"action": "revise", "feedback": hint}) for _ in range(4))
    _patch_provider(monkeypatch, script)

    code, blocked = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0, json.dumps(blocked, indent=2)
    assert blocked["pause_kind"] == "blocked"
    run_id = str(blocked["run_id"])
    store = RunStore(state / "olympus.db")

    for expected in (1, 2, 3):
        code, envelope = _invoke(["reply", run_id, hint, *_base(repo, state)])
        assert code == 0, json.dumps(envelope, indent=2)
        assert envelope["status"] == "needs_input"
        assert envelope["pause_kind"] == "blocked"
        continued = [
            event.payload
            for event in store.list_events(run_id)
            if event.event_type == "session.continued"
        ]
        assert len(continued) == expected
        assert continued[-1]["recovery_node"] == "plan"
        assert "recovery_mode" not in continued[-1]
        assert hint in continued[-1]["raw_input"]
    assert script.plan == []
    assert len(script.planner_prompts) == 3 * 4

    planner_calls_before = len(script.planner_prompts)
    code, exhausted = _invoke(["reply", run_id, hint, *_base(repo, state)])
    assert code == 0, json.dumps(exhausted, indent=2)
    assert exhausted["status"] == "needs_input"
    assert exhausted["pause_kind"] == "blocked"
    assert "max_plan_recoveries = 3" in str(exhausted["message"])
    assert len(script.planner_prompts) == planner_calls_before
    continued = [
        event for event in store.list_events(run_id) if event.event_type == "session.continued"
    ]
    assert len(continued) == 3
    assert not any(event.payload.get("recovery_mode") == "in_place" for event in continued)
    assert store.get_run(run_id)["status"] == RunStatus.AWAITING_HUMAN


def test_post_completion_reply_reopens_and_applies_follow_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.extend(
        [
            json.dumps(_goal_spec()),
            json.dumps(_goal_spec(include_subtract=True, version=2)),
        ]
    )
    script.design.extend(
        [json.dumps(_design()), json.dumps(_design(requirements=("R1", "R2")))]
    )
    script.plan.extend(
        [
            json.dumps(_plan()),
            json.dumps(
                _plan(
                    requirements=("R1", "R2"),
                    objective="Implement divide and subtract in calculator.py",
                    command=_SUBTRACT_CHECK,
                    version=2,
                )
            ),
        ]
    )
    script.router.append(
        json.dumps(
            {
                "action": "reopen",
                "feedback": "Also add a subtract(a, b) helper.",
            }
        )
    )
    script.workers.extend([("calculator.py", _CALC_CODE), ("calculator.py", _CALC_CODE_V2)])
    _patch_provider(monkeypatch, script)

    code, started = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0
    assert started["status"] == "completed"
    run_id = str(started["run_id"])

    code, reopened = _invoke(
        [
            "reply",
            run_id,
            "Please also add a subtract(a, b) helper.",
            *_base(repo, state),
        ]
    )
    assert code == 0
    assert reopened["status"] == "completed"
    assert reopened["changed_files"] == ["calculator.py"]
    content = (repo / "calculator.py").read_text(encoding="utf-8")
    assert "def subtract" in content
    assert "def divide" in content

    store = RunStore(state / "olympus.db")
    snap = store.get_run(run_id)
    assert snap["spec"].version == 2
    assert {req.id for req in snap["spec"].requirements} == {"R1", "R2"}
    event_types = {event.event_type for event in store.list_events(run_id)}
    assert "run.reopened" in event_types
    assert "session.continued" in event_types
    continued = [
        event.payload
        for event in store.list_events(run_id)
        if event.event_type == "session.continued"
    ]
    assert continued[-1]["reopened"] is True
    assert "recovery_mode" not in continued[-1]


def test_state_dir_inside_workspace_is_rejected_without_creating_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    script = SessionScript()
    _patch_provider(monkeypatch, script)

    code, envelope = _invoke(
        [
            "start",
            "Build a calculator library",
            "--workspace",
            str(repo),
            "--state-dir",
            str(repo / ".olympus"),
            "--provider",
            "mock",
        ]
    )
    assert code == 1
    assert envelope["status"] == "failed"
    assert "inside the target workspace" in str(envelope["error"])
    assert not (repo / ".olympus").exists()
    assert _git_status(repo) == ""


def test_default_state_dir_and_message_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    monkeypatch.setattr(
        "olympus.session.runner.tempfile.gettempdir", lambda: str(tmp_path / "tmproot")
    )
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    _patch_provider(monkeypatch, script)

    code, envelope = _invoke(
        [
            "start",
            "--message",
            "Build a calculator library",
            "--step-mode",
            "plan_only",
            "--workspace",
            str(repo),
            "--provider",
            "mock",
        ]
    )
    assert code == 0
    assert envelope["status"] == "needs_input"
    assert envelope["pause_kind"] == "plan_handoff"
    session_root = tmp_path / "tmproot" / "olympus-session"
    assert session_root.is_dir()
    assert any(session_root.iterdir())
    assert not (repo / ".olympus").exists()
    assert _git_status(repo) == ""


def test_resolve_state_dir_defaults_outside_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    resolved = resolve_state_dir(repo, None)
    assert not resolved.is_relative_to(repo)
    assert resolved.is_dir()


# --------------------------------------------------------------------------- review A1-A5


def test_reply_rejects_a_different_workspace_without_writing_anywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    other = tmp_path / "other"
    state = tmp_path / "state"
    _init_git_repo(repo)
    _init_git_repo(other)
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.router.append(json.dumps({"action": "approve"}))
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    code, started = _invoke(
        [
            "start",
            "Build a calculator library",
            "--gate-policy",
            "interactive_minimal",
            "--step-mode",
            "plan_only",
            *_base(repo, state),
        ]
    )
    assert code == 0
    assert started["pause_kind"] == "plan_handoff"
    run_id = str(started["run_id"])

    code, rebound = _invoke(
        [
            "reply",
            run_id,
            "approve",
            "--step-mode",
            "execute",
            "--workspace",
            str(other),
            "--state-dir",
            str(state),
            "--provider",
            "mock",
        ]
    )
    assert code == 1, json.dumps(rebound, indent=2)
    assert rebound["status"] == "failed"
    assert "does not match the run's recorded workspace" in str(rebound["error"])
    assert _git_status(repo) == ""
    assert _git_status(other) == ""
    assert not (repo / "calculator.py").exists()
    assert not (other / "calculator.py").exists()

    store = RunStore(state / "olympus.db")
    assert Path(str(store.get_run(run_id)["workspace_root"])) == repo
    assert store.get_run(run_id)["status"].value == "awaiting_human"


def test_reply_without_workspace_resumes_the_recorded_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    _init_git_repo(repo)
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.router.append(json.dumps({"action": "approve"}))
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    code, started = _invoke(
        [
            "start",
            "Build a calculator library",
            "--step-mode",
            "plan_only",
            *_base(repo, state),
        ]
    )
    assert code == 0
    assert started["pause_kind"] == "plan_handoff"

    # No --workspace: the recorded workspace must win over the invocation cwd.
    code, executed = _invoke(
        [
            "reply",
            str(started["run_id"]),
            "Plan looks good, implement it.",
            "--step-mode",
            "execute",
            "--state-dir",
            str(state),
            "--provider",
            "mock",
        ]
    )
    assert code == 0, json.dumps(executed, indent=2)
    assert executed["status"] == "completed"
    assert (repo / "calculator.py").read_text(encoding="utf-8") == _CALC_CODE
    store = RunStore(state / "olympus.db")
    assert Path(str(store.get_run(str(started["run_id"]))["workspace_root"])) == repo


def test_state_dir_inside_enclosing_git_repository_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    nested = repo / "nested"
    nested.mkdir()
    script = SessionScript()
    _patch_provider(monkeypatch, script)

    code, envelope = _invoke(
        [
            "start",
            "Build a calculator library",
            "--workspace",
            str(nested),
            "--state-dir",
            str(repo / "session-state"),
            "--provider",
            "mock",
        ]
    )
    assert code == 1
    assert envelope["status"] == "failed"
    assert "inside" in str(envelope["error"])
    assert "git repository" in str(envelope["error"])
    assert not (repo / "session-state").exists()
    assert _git_status(repo) == ""


@pytest.mark.parametrize(
    "message",
    [
        "do not proceed",
        "Please don't proceed.",
        "I do not approve this",
        "no, not this version",
        "reject",
        "cancel",
        "不要继续",
        "别继续了",
        "拒绝这个方案",
        "取消",
    ],
)
def test_negated_approval_never_routes_to_approve(message: str) -> None:
    provider = MockLLMProvider(responses=[LLMResponse(content="invalid JSON")])
    intent = parse_reply(
        provider=provider,
        message=message,
        snapshot={"status": RunStatus.AWAITING_HUMAN},
        pause_kind=PauseKind.APPROVAL,
        gate="spec",
    )
    assert intent.action != "approve"
    assert intent.answers == {}
    assert intent.resolutions == {}


@pytest.mark.parametrize("message", ["looks good, proceed", "特别好的方案，批准"])
def test_unambiguous_approval_fallback_still_approves(message: str) -> None:
    provider = MockLLMProvider(responses=[LLMResponse(content="invalid JSON")])
    intent = parse_reply(
        provider=provider,
        message=message,
        snapshot={"status": RunStatus.AWAITING_HUMAN},
        pause_kind=PauseKind.APPROVAL,
        gate="spec",
    )
    assert intent.action == "approve"


def test_negated_reply_never_approves_the_pending_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.router.append("invalid JSON")
    _patch_provider(monkeypatch, script)

    code, started = _invoke(
        ["start", "Build a calculator library", "--gate-policy", "default", *_base(repo, state)]
    )
    assert code == 0
    assert started["pause_kind"] == "approval"
    run_id = str(started["run_id"])
    store = RunStore(state / "olympus.db")
    before = store.get_run(run_id)
    assert not store.is_approved(
        run_id=run_id,
        target=ApprovalTarget.GOAL_SPEC,
        target_id=before["spec"].spec_id,
        target_hash=before["spec"].content_hash,
    )

    code, refused = _invoke(["reply", run_id, "Please do not proceed.", *_base(repo, state)])
    assert code == 0, json.dumps(refused, indent=2)
    assert refused["status"] == "needs_input"
    assert refused["pause_kind"] == "approval"
    after = store.get_run(run_id)
    assert after["spec"].version == before["spec"].version
    assert after["spec"].content_hash == before["spec"].content_hash
    assert after["status"].value == "awaiting_human"
    assert not store.is_approved(
        run_id=run_id,
        target=ApprovalTarget.GOAL_SPEC,
        target_id=after["spec"].spec_id,
        target_hash=after["spec"].content_hash,
    )

    script.router.append(json.dumps({"action": "approve"}))
    _invoke(["reply", run_id, "Approve the spec.", *_base(repo, state)])
    assert store.is_approved(
        run_id=run_id,
        target=ApprovalTarget.GOAL_SPEC,
        target_id=after["spec"].spec_id,
        target_hash=after["spec"].content_hash,
    )


def test_completed_run_ignores_unparseable_follow_up_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    code, started = _invoke(
        [
            "start",
            "Build a calculator library",
            "--gate-policy",
            "interactive_minimal",
            *_base(repo, state),
        ]
    )
    assert code == 0
    assert started["status"] == "completed"
    run_id = str(started["run_id"])
    store = RunStore(state / "olympus.db")
    before_events = [event.event_type for event in store.list_events(run_id)]

    script.router.append("invalid JSON")
    code, reply = _invoke(["reply", run_id, "???", *_base(repo, state)])
    assert code == 0, json.dumps(reply, indent=2)
    assert reply["status"] == "needs_input"
    assert "already completed" in str(reply["message"])

    after_events = [event.event_type for event in store.list_events(run_id)]
    assert "run.reopened" not in after_events
    assert "session.continued" not in after_events[len(before_events) :]
    assert store.get_run(run_id)["status"].value == "completed"
    assert script.align == []


def test_unavailable_provider_on_reply_emits_a_failure_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    with pytest.MonkeyPatch.context() as scoped:
        script = SessionScript()
        script.align.append(json.dumps(_goal_spec()))
        _patch_provider(scoped, script)
        code, started = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0

    code, reply = _invoke(
        [
            "reply",
            str(started["run_id"]),
            "approve",
            "--workspace",
            str(repo),
            "--state-dir",
            str(state),
            "--provider",
            "definitely-unsupported",
        ]
    )
    assert code == 1
    assert reply["status"] == "failed"
    assert "Unknown provider" in str(reply["error"])


def test_unavailable_provider_on_start_emits_a_failure_envelope(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"

    code, envelope = _invoke(
        [
            "start",
            "Build a calculator library",
            "--workspace",
            str(repo),
            "--state-dir",
            str(state),
            "--provider",
            "definitely-unsupported",
        ]
    )
    assert code == 1
    assert envelope["status"] == "failed"
    assert "Unknown provider" in str(envelope["error"])


def _custom_workflow(tmp_path: Path) -> Path:
    custom = tmp_path / "review_workflow.yaml"
    custom.write_text(
        (WORKFLOW_DIR / "full_sdlc_tdd.yaml")
        .read_text(encoding="utf-8")
        .replace("name: full_sdlc_tdd", "name: review_workflow"),
        encoding="utf-8",
    )
    return custom


def test_custom_yaml_workflow_resumes_on_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    _init_git_repo(repo)
    custom = _custom_workflow(tmp_path)
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.router.extend(json.dumps({"action": "approve"}) for _ in range(3))
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    common = [
        "--workspace",
        str(repo),
        "--state-dir",
        str(state),
        "--provider",
        "mock",
        "--gate-policy",
        "default",
    ]
    code, started = _invoke(
        ["start", "Build a calculator library", "--workflow", str(custom), *common]
    )
    assert code == 0, json.dumps(started, indent=2)
    assert started["pause_kind"] == "approval"
    assert started["workflow"] == "review_workflow"
    run_id = str(started["run_id"])

    payloads = [
        event.payload
        for event in RunStore(state / "olympus.db").list_events(run_id)
        if event.event_type == "session.started"
    ]
    assert Path(str(payloads[0]["workflow_path"])) == custom.resolve()

    # Explicit --workflow <path> on reply reloads the same custom YAML.
    code, second = _invoke(
        ["reply", run_id, "Approve the spec.", "--workflow", str(custom), *common]
    )
    assert code == 0, json.dumps(second, indent=2)
    assert second["pause_kind"] == "approval"
    assert second["workflow"] == "review_workflow"

    # Omitted --workflow on reply falls back to the persisted workflow path.
    code, third = _invoke(["reply", run_id, "Approve the design.", *common])
    assert code == 0, json.dumps(third, indent=2)
    assert third["pause_kind"] == "approval"
    assert third["workflow"] == "review_workflow"

    code, final = _invoke(["reply", run_id, "Approve the plan.", *common])
    assert code == 0, json.dumps(final, indent=2)
    assert final["status"] == "completed"
    assert (repo / "calculator.py").is_file()


def test_reply_rejects_a_workflow_that_does_not_match_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    _init_git_repo(repo)
    custom = _custom_workflow(tmp_path)
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    _patch_provider(monkeypatch, script)

    code, started = _invoke(
        [
            "start",
            "Build a calculator library",
            "--workflow",
            str(custom),
            "--gate-policy",
            "default",
            *_base(repo, state),
        ]
    )
    assert code == 0
    assert started["workflow"] == "review_workflow"

    code, mismatch = _invoke(
        [
            "reply",
            str(started["run_id"]),
            "Approve the spec.",
            "--workflow",
            "full_sdlc_tdd",
            *_base(repo, state),
        ]
    )
    assert code == 1, json.dumps(mismatch, indent=2)
    assert mismatch["status"] == "failed"
    assert "does not match the run's workflow" in str(mismatch["error"])



def test_reply_rejects_empty_db_inside_repo_without_writing_bytes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    nested = repo / "nested"
    nested.mkdir()
    state = repo / "session-state"
    state.mkdir()
    db = state / "olympus.db"
    db.write_bytes(b"")

    code, env = _invoke(["reply", "run-missing", "Approve.", *_base(nested, state)])
    assert code == 1
    assert env["status"] == "failed"
    assert db.stat().st_size == 0


def test_router_guards_valid_json_negations_gibberish_and_provider_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from olympus.providers.types import LLMResponse, MockLLMProvider, ProviderError
    from olympus.session.contracts import PauseKind
    from olympus.session.router import parse_reply

    snap = {"status": RunStatus.AWAITING_HUMAN, "spec": None, "plan": None}
    approve_llm = MockLLMProvider(
        responder=lambda _m, _t: LLMResponse(content='{"action": "approve"}')
    )
    bad_json_llm = MockLLMProvider(responder=lambda _m, _t: LLMResponse(content="not json"))
    reopen_llm = MockLLMProvider(
        responder=lambda _m, _t: LLMResponse(
            content='{"action": "reopen", "feedback": "invented task"}'
        )
    )

    assert (
        parse_reply(
            provider=bad_json_llm,
            message="I am not ready to proceed.",
            snapshot=snap,
            pause_kind=PauseKind.APPROVAL,
        ).action
        != "approve"
    )
    assert (
        parse_reply(
            provider=approve_llm,
            message="Please do not proceed.",
            snapshot=snap,
            pause_kind=PauseKind.APPROVAL,
        ).action
        != "approve"
    )
    assert (
        parse_reply(
            provider=reopen_llm,
            message="???",
            snapshot=snap,
            pause_kind=None,
            completed=True,
        ).action
        == "unknown"
    )

    def _fail(_m: object, _t: object) -> LLMResponse:
        raise ProviderError("boom")

    with pytest.raises(ProviderError, match="boom"):
        parse_reply(
            provider=MockLLMProvider(responder=_fail),
            message="yes",
            snapshot=snap,
            pause_kind=PauseKind.APPROVAL,
        )


def test_reply_rejects_same_name_tampered_yaml_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    first = _custom_workflow(tmp_path)
    tampered = tmp_path / "wf_b.yaml"
    tampered.write_text(
        first.read_text(encoding="utf-8").replace(
            "approve_design: true", "approve_design: false"
        )
    )
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    _patch_provider(monkeypatch, script)

    code, started = _invoke(
        [
            "start",
            "Add add(a, b)",
            "--workflow",
            str(first),
            "--gate-policy",
            "default",
            *_base(repo, state),
        ]
    )
    assert code == 0
    code, rejected = _invoke(
        [
            "reply",
            str(started["run_id"]),
            "Approve the spec.",
            "--workflow",
            str(tampered),
            *_base(repo, state),
        ]
    )
    assert code == 1
    assert rejected["status"] == "failed"
    assert "does not match the workflow that started this run" in str(rejected["error"])


def test_negated_approval_and_no_change_replies_never_mutate_or_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.router.append("invalid JSON")
    _patch_provider(monkeypatch, script)

    code, started = _invoke(
        ["start", "Build a calculator library", "--gate-policy", "default", *_base(repo, state)]
    )
    assert code == 0
    run_id = str(started["run_id"])
    store = RunStore(state / "olympus.db")
    snap = store.get_run(run_id)

    code, refused = _invoke(
        ["reply", run_id, "I am not going to approve this.", *_base(repo, state)]
    )
    assert code == 0
    assert refused["pause_kind"] == "approval"
    assert not store.is_approved(
        run_id=run_id,
        target=ApprovalTarget.GOAL_SPEC,
        target_id=snap["spec"].spec_id,
        target_hash=snap["spec"].content_hash,
    )

    repo2 = tmp_path / "repo2"
    _init_git_repo(repo2)
    state2 = tmp_path / "state2"
    script2 = SessionScript()
    script2.align.extend([json.dumps(_goal_spec()), json.dumps(_goal_spec(version=2))])
    script2.design.extend([json.dumps(_design())] * 2)
    script2.plan.extend([json.dumps(_plan())] * 2)
    script2.workers.extend([("calculator.py", _CALC_CODE)] * 2)
    script2.router.extend(["invalid JSON", json.dumps({"action": "reopen", "feedback": "edit"})])
    _patch_provider(monkeypatch, script2)

    code, done = _invoke(["start", "Build a calculator library", *_base(repo2, state2)])
    assert code == 0
    assert done["status"] == "completed"
    done_id = str(done["run_id"])

    for _ in range(2):
        code, stayed = _invoke(
            ["reply", done_id, "Do not change anything.", *_base(repo2, state2)]
        )
        assert code == 0
        assert stayed["status"] == "needs_input"

    store2 = RunStore(state2 / "olympus.db")
    assert store2.get_run(done_id)["status"] == RunStatus.COMPLETED
    assert not any(ev.event_type == "run.reopened" for ev in store2.list_events(done_id))


def test_explicit_ambiguity_answer_resolves_blocker_when_router_json_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec(ambiguity=True)))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.router.append("invalid JSON")
    _patch_provider(monkeypatch, script)

    code, started = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0
    assert started["pause_kind"] == "clarify"
    assert started["unresolved_blockers"]

    code, clarified = _invoke(
        [
            "reply",
            str(started["run_id"]),
            "AMB-1: Use your default assumptions and proceed.",
            *_base(repo, state),
        ]
    )
    assert code == 0, json.dumps(clarified, indent=2)
    assert not clarified["unresolved_blockers"]


def test_default_assumption_reply_resolves_all_blockers_under_malformed_router_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec(ambiguity=True, two_ambiguities=True)))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.router.append("this is definitely not json {{{")
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    code, started = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0
    assert started["pause_kind"] == "clarify"
    assert started["unresolved_blockers"] == ["ambiguity:AMB-1", "ambiguity:AMB-2"]
    run_id = str(started["run_id"])

    code, resumed = _invoke(
        ["reply", run_id, "Use your default assumptions and proceed.", *_base(repo, state)]
    )
    assert code == 0, json.dumps(resumed, indent=2)
    assert resumed["status"] == "completed"
    assert resumed["unresolved_blockers"] == []
    assert resumed["changed_files"] == ["calculator.py"]
    assert (repo / "calculator.py").read_text(encoding="utf-8") == _CALC_CODE

    snap = RunStore(state / "olympus.db").get_run(run_id)
    answers = {amb.id: amb.answer for amb in snap["spec"].open_ambiguities}
    assert answers == {
        "AMB-1": "raise ValueError('zero')",
        "AMB-2": "int and float",
    }


def test_align_repairs_blocking_ambiguity_without_default_assumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    missing_default = _goal_spec(ambiguity=True)
    del missing_default["open_ambiguities"][0]["default_assumption"]
    script = SessionScript()
    script.align.extend([json.dumps(missing_default), json.dumps(_goal_spec(ambiguity=True))])
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.router.append("invalid JSON")
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    code, started = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0, json.dumps(started, indent=2)
    assert started["pause_kind"] == "clarify"
    assert script.align == []

    snap = RunStore(state / "olympus.db").get_run(str(started["run_id"]))
    ambiguity = snap["spec"].open_ambiguities[0]
    assert ambiguity.default_assumption == "raise ValueError('zero')"

    code, resumed = _invoke(
        [
            "reply",
            str(started["run_id"]),
            "Use your default assumptions and proceed.",
            *_base(repo, state),
        ]
    )
    assert code == 0, json.dumps(resumed, indent=2)
    assert resumed["status"] == "completed"


@pytest.mark.parametrize(
    "message",
    [
        "Use your default assumptions and proceed.",
        "proceed with your recommendations",
        "go with defaults",
        "按默认方案继续",
        "按推荐方案继续",
    ],
)
@pytest.mark.parametrize(
    "router_output",
    ["invalid JSON", '{"action": "unknown"}', '{"action": "approve"}'],
)
def test_default_acceptance_resolves_pending_blockers_for_every_router_outcome(
    message: str, router_output: str
) -> None:
    spec = GoalSpec.model_validate(_goal_spec(ambiguity=True, two_ambiguities=True))
    intent = parse_reply(
        provider=MockLLMProvider(responses=[LLMResponse(content=router_output)]),
        message=message,
        snapshot={"status": RunStatus.AWAITING_HUMAN, "spec": spec, "plan": None},
        pause_kind=PauseKind.CLARIFY,
        gate="spec",
    )
    assert intent.action == "clarify"
    assert intent.answers == {
        "AMB-1": "raise ValueError('zero')",
        "AMB-2": "int and float",
    }


def test_default_acceptance_falls_back_to_first_option_then_user_message() -> None:
    no_default = _goal_spec(ambiguity=True)
    del no_default["open_ambiguities"][0]["default_assumption"]
    spec = GoalSpec.model_validate(no_default)
    intent = parse_reply(
        provider=MockLLMProvider(responses=[LLMResponse(content="invalid JSON")]),
        message="Use your default assumptions and proceed.",
        snapshot={"status": RunStatus.AWAITING_HUMAN, "spec": spec, "plan": None},
        pause_kind=PauseKind.CLARIFY,
        gate="spec",
    )
    assert intent.action == "clarify"
    assert intent.answers == {"AMB-1": "raise ValueError('zero')"}

    bare = _goal_spec(ambiguity=True)
    del bare["open_ambiguities"][0]["default_assumption"]
    bare["open_ambiguities"][0]["options"] = []
    intent = parse_reply(
        provider=MockLLMProvider(responses=[LLMResponse(content="invalid JSON")]),
        message="按默认方案继续",
        snapshot={
            "status": RunStatus.AWAITING_HUMAN,
            "spec": GoalSpec.model_validate(bare),
            "plan": None,
        },
        pause_kind=PauseKind.CLARIFY,
        gate="spec",
    )
    assert intent.action == "clarify"
    assert intent.answers == {"AMB-1": "按默认方案继续"}


def test_explicit_id_pointing_at_defaults_uses_the_recorded_default() -> None:
    spec = GoalSpec.model_validate(_goal_spec(ambiguity=True, two_ambiguities=True))
    intent = parse_reply(
        provider=MockLLMProvider(responses=[LLMResponse(content="invalid JSON")]),
        message="AMB-1: Use your default assumptions and proceed.",
        snapshot={"status": RunStatus.AWAITING_HUMAN, "spec": spec, "plan": None},
        pause_kind=PauseKind.CLARIFY,
        gate="spec",
    )
    assert intent.action == "clarify"
    assert intent.answers == {
        "AMB-1": "raise ValueError('zero')",
        "AMB-2": "int and float",
    }


def test_default_acceptance_resolves_blocking_critiques() -> None:
    data = _goal_spec()
    data["minority_notes"] = [
        {
            "id": "CRIT-1",
            "perspective": "qa",
            "severity": "blocking",
            "summary": "The error contract has no regression evidence.",
        }
    ]
    spec = GoalSpec.model_validate(data)
    intent = parse_reply(
        provider=MockLLMProvider(responses=[LLMResponse(content="invalid JSON")]),
        message="Use your default assumptions and proceed.",
        snapshot={"status": RunStatus.AWAITING_HUMAN, "spec": spec, "plan": None},
        pause_kind=PauseKind.CLARIFY,
        gate="spec",
    )
    assert intent.action == "clarify"
    assert intent.resolutions == {"CRIT-1": "Use your default assumptions and proceed."}


@pytest.mark.parametrize(
    "message",
    [
        "I am not going to approve this.",
        "Please do not proceed.",
        "Do not change anything.",
        "Do not use your default assumptions.",
    ],
)
@pytest.mark.parametrize("router_output", ["invalid JSON", '{"action": "approve"}'])
def test_negated_or_no_change_replies_never_approve_or_resolve_defaults(
    message: str, router_output: str
) -> None:
    spec = GoalSpec.model_validate(_goal_spec(ambiguity=True))
    intent = parse_reply(
        provider=MockLLMProvider(responses=[LLMResponse(content=router_output)]),
        message=message,
        snapshot={"status": RunStatus.AWAITING_HUMAN, "spec": spec, "plan": None},
        pause_kind=PauseKind.CLARIFY,
        gate="spec",
    )
    assert intent.action not in ("approve", "clarify", "reopen")
    assert intent.answers == {}
    assert intent.resolutions == {}


@pytest.mark.parametrize(
    "message",
    [
        "C:\\Users\\demo\\.local\\share\\opencode\\opencode.db exists on this computer",
        "What providers need _build_project_name?",
        "The failing check should import calculator from the repo root.",
    ],
)
@pytest.mark.parametrize(
    "router_output",
    [
        "invalid JSON",
        '{"action": "unknown"}',
        '{"action": "approve"}',
        '{"action": "reopen", "feedback": "invented"}',
    ],
)
def test_blocked_pause_actionable_hints_route_to_revise(
    message: str, router_output: str
) -> None:
    intent = parse_reply(
        provider=MockLLMProvider(responses=[LLMResponse(content=router_output)]),
        message=message,
        snapshot={"status": RunStatus.AWAITING_HUMAN, "spec": None, "plan": None},
        pause_kind=PauseKind.BLOCKED,
        gate=None,
    )
    assert intent.action == "revise"
    assert intent.feedback == message


@pytest.mark.parametrize(
    "message",
    ["Please do not proceed.", "Do not change anything.", "???", "no"],
)
def test_blocked_pause_negated_or_empty_hints_never_route_to_revise(message: str) -> None:
    intent = parse_reply(
        provider=MockLLMProvider(responses=[LLMResponse(content="invalid JSON")]),
        message=message,
        snapshot={"status": RunStatus.AWAITING_HUMAN, "spec": None, "plan": None},
        pause_kind=PauseKind.BLOCKED,
        gate=None,
    )
    assert intent.action != "revise"


def test_delivery_checks_and_symlinked_sandboxes_never_pollute_target_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from olympus.harness import delivery

    repo = tmp_path / "repo"
    state = tmp_path / "state"
    _init_git_repo(repo)
    plan = _plan()
    plan["delivery_checks"] = [
        {"requirement_id": "R1", "criterion_index": 1, "command": _DIVIDE_CHECK},
    ]
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(plan))
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    observed: list[tuple[Path, Path]] = []
    real_create = delivery.create_sandbox

    def _observe(*args: object, **kwargs: object) -> object:
        handle = real_create(*args, **kwargs)  # type: ignore[arg-type] -- forwarding mock wrapper
        observed.append((handle.workspace_path, handle.tmp_dir))
        return handle

    monkeypatch.setattr(delivery, "create_sandbox", _observe)
    code, result = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0, json.dumps(result, indent=2)
    assert result["status"] == "completed"
    assert observed
    for worktree, tmp_dir in observed:
        assert not worktree.resolve().is_relative_to(repo.resolve())
        assert not tmp_dir.resolve().is_relative_to(repo.resolve())
    assert not (repo / ".olympus").exists()

    repo2 = tmp_path / "repo2"
    state2 = tmp_path / "state2"
    _init_git_repo(repo2)
    state2.mkdir()
    (state2 / "sandboxes").symlink_to(repo2 / ".olympus" / "sandboxes", target_is_directory=True)
    code2, rejected = _invoke(["start", "Build a calculator library", *_base(repo2, state2)])
    assert code2 == 1
    assert rejected["status"] == "failed"
    assert "inside" in str(rejected["error"])
    assert not (repo2 / ".olympus").exists()


def test_session_start_records_stage_driver_option_and_defaults_to_auto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OLYMPUS_STAGE_DRIVER", raising=False)
    monkeypatch.delenv("OLYMPUS_WORKER_DRIVER", raising=False)
    monkeypatch.delenv("OLYMPUS_REVIEWER_DRIVER", raising=False)
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    _patch_provider(monkeypatch, script)
    code, start = _invoke(
        [
            "start",
            "Build a calculator library",
            "--gate-policy",
            "default",
            "--stage-driver",
            "native",
            *_base(repo, tmp_path / "state-explicit"),
        ]
    )
    assert code == 0, json.dumps(start, indent=2)
    explicit = RunStore(tmp_path / "state-explicit" / "olympus.db").get_run(
        str(start["run_id"])
    )["execution_drivers"]
    assert explicit["stage_driver"] == "native"
    assert explicit["worker_driver"] == "auto"
    assert explicit["reviewer_driver"] == "auto"

    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    _patch_provider(monkeypatch, script)
    code, default_start = _invoke(
        [
            "start",
            "Build a calculator library",
            "--gate-policy",
            "default",
            *_base(repo, tmp_path / "state-default"),
        ]
    )
    assert code == 0, json.dumps(default_start, indent=2)
    recorded = RunStore(tmp_path / "state-default" / "olympus.db").get_run(
        str(default_start["run_id"])
    )["execution_drivers"]
    assert recorded["stage_driver"] == "auto"


def test_session_start_records_pi_options_and_rejects_invalid_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OLYMPUS_PI_PROVIDER", raising=False)
    monkeypatch.delenv("OLYMPUS_THINKING_LEVEL", raising=False)
    monkeypatch.delenv("OLYMPUS_CONTEXT_WINDOW", raising=False)
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    _patch_provider(monkeypatch, script)

    code, start = _invoke(
        [
            "start",
            "Build a calculator library",
            "--gate-policy",
            "default",
            "--pi-provider",
            "gateway-x",
            "--thinking",
            "HIGH",
            "--context-window",
            "262144",
            *_base(repo, tmp_path / "state-ok"),
        ]
    )
    assert code == 0, json.dumps(start, indent=2)
    recorded = RunStore(tmp_path / "state-ok" / "olympus.db").get_run(
        str(start["run_id"])
    )["execution_drivers"]
    assert recorded["pi_provider"] == "gateway-x"
    assert recorded["thinking_level"] == "high"
    assert recorded["context_window"] == 262144

    code, failed = _invoke(
        [
            "start",
            "Build a calculator library",
            "--thinking",
            "extreme",
            *_base(repo, tmp_path / "state-bad-thinking"),
        ]
    )
    assert code == 1
    assert failed["status"] == "failed"
    assert "thinking_level" in failed["error"]

    invalid_window = RUNNER.invoke(
        app,
        [
            "session",
            "start",
            "Build a calculator library",
            "--context-window",
            "512",
            *_base(repo, tmp_path / "state-bad-window"),
        ],
    )
    # Typer rejects the below-minimum window before the command runs.
    assert invalid_window.exit_code == 2
    assert "1024" in invalid_window.stdout or "1024" in invalid_window.stderr

    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    _patch_provider(monkeypatch, script)
    code, env_start = _invoke(
        [
            "start",
            "Build a calculator library",
            "--gate-policy",
            "default",
            *_base(repo, tmp_path / "state-env"),
        ]
    )
    assert code == 0, json.dumps(env_start, indent=2)
    recorded = RunStore(tmp_path / "state-env" / "olympus.db").get_run(
        str(env_start["run_id"])
    )["execution_drivers"]
    assert recorded["pi_provider"] is None
    assert recorded["thinking_level"] is None
    assert recorded["context_window"] is None


def test_session_reply_applies_explicit_pi_overrides_onto_recorded_drivers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from olympus.harness.driver import (
        ExecutionDriversConfig,
    )
    from olympus.harness.driver import (
        default_driver_factories as real_factories,
    )

    captured: list[ExecutionDriversConfig] = []

    def spy(*, store, run_id, provider, execution_drivers=None):
        captured.append(execution_drivers)
        return real_factories(
            store=store, run_id=run_id, provider=provider, execution_drivers=execution_drivers
        )

    monkeypatch.setattr("olympus.session.runner.default_driver_factories", spy)
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec(ambiguity=True)))
    _patch_provider(monkeypatch, script)

    code, start = _invoke(
        [
            "start",
            "Build a calculator library",
            "--pi-provider",
            "gateway-one",
            "--thinking",
            "high",
            *_base(repo, state),
        ]
    )
    assert code == 0, json.dumps(start, indent=2)
    assert start["pause_kind"] == "clarify"
    assert captured[-1].pi_provider == "gateway-one"
    assert captured[-1].thinking_level == "high"

    script.router.append(
        json.dumps({"action": "clarify", "answers": {"AMB-1": "raise ValueError('zero')"}})
    )
    code, reply = _invoke(
        [
            "reply",
            str(start["run_id"]),
            "Make divide(1, 0) raise ValueError('zero').",
            "--pi-provider",
            "gateway-two",
            "--thinking",
            "low",
            "--context-window",
            "200000",
            *_base(repo, state),
        ]
    )
    assert code == 0, json.dumps(reply, indent=2)
    # The recorded snapshot keeps the start-time values; the live runtime honors
    # the explicit reply overrides on top of them.
    recorded = RunStore(state / "olympus.db").get_run(str(start["run_id"]))[
        "execution_drivers"
    ]
    assert recorded["pi_provider"] == "gateway-one"
    assert captured[-1].pi_provider == "gateway-two"
    assert captured[-1].thinking_level == "low"
    assert captured[-1].context_window == 200000
    assert captured[-1].stage_driver == "auto"


def test_session_runner_token_usage_aggregates_ledger_and_actors(tmp_path: Path) -> None:
    from olympus.core.contracts import BudgetPolicy, RunMode
    from olympus.session.contracts import TokenUsageSummary
    from olympus.session.runner import SessionConfig, SessionRunner

    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    runner = SessionRunner(SessionConfig(workspace_root=repo, state_dir=state))
    run_id = runner.store.create_run(
        title="usage", mode=RunMode.SIMULATION, budget_policy=BudgetPolicy(max_llm_calls=100)
    )
    runner.store.record_external_llm_usage(
        run_id,
        actor="align:proposer",
        purpose="align",
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=300,
        cache_write_tokens=25,
        reasoning_tokens=10,
    )
    runner.store.record_external_llm_usage(
        run_id,
        actor="hephaestus:T1",
        purpose="pi_step_1",
        input_tokens=20,
        output_tokens=10,
        reasoning_tokens=4,
    )

    usage = runner._token_usage(run_id)
    assert isinstance(usage, TokenUsageSummary)
    assert usage.llm_calls == 2
    assert usage.input_tokens == 120
    assert usage.output_tokens == 60
    assert usage.reasoning_tokens == 14
    assert usage.cache_read_tokens == 300
    assert usage.cache_write_tokens == 25
    # Reasoning is a subset of output tokens: never added into the total.
    assert usage.total_tokens == 120 + 60 + 300 + 25
    assert usage.cache_hit_rate == round(300 / (120 + 300), 4)

    align = usage.by_actor["align:proposer"]
    assert align.llm_calls == 1
    assert (align.input_tokens, align.output_tokens) == (100, 50)
    assert (align.cache_read_tokens, align.cache_write_tokens) == (300, 25)
    assert align.reasoning_tokens == 10
    assert align.total_tokens == 100 + 50 + 300 + 25
    worker = usage.by_actor["hephaestus:T1"]
    assert (worker.input_tokens, worker.output_tokens, worker.reasoning_tokens) == (20, 10, 4)
    assert worker.total_tokens == 30

    missing = runner._token_usage("run-does-not-exist")
    assert missing == TokenUsageSummary()


def test_session_envelope_reports_token_usage_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec(ambiguity=True)))
    _patch_provider(monkeypatch, script)

    code, start = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0, json.dumps(start, indent=2)
    usage = start["token_usage"]
    assert set(usage) == {
        "llm_calls",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "total_tokens",
        "cache_hit_rate",
        "by_actor",
    }
    assert usage["llm_calls"] >= 1
    assert usage["cache_hit_rate"] == 0.0
    assert usage["reasoning_tokens"] == 0
    assert usage["total_tokens"] == (
        usage["input_tokens"]
        + usage["output_tokens"]
        + usage["cache_read_tokens"]
        + usage["cache_write_tokens"]
    )
    assert "align:proposer" in usage["by_actor"]
    actor = usage["by_actor"]["align:proposer"]
    assert set(actor) == {
        "llm_calls",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "total_tokens",
    }

    # Failure envelopes without a stored run still carry an empty summary.
    code, failed = _invoke(["start", "   ", *_base(repo, state)])
    assert code == 1
    assert failed["token_usage"]["llm_calls"] == 0
    assert failed["token_usage"]["by_actor"] == {}


def test_router_blocks_negated_defaults_questions_and_unchanged_refusals() -> None:
    from olympus.core.contracts import GoalSpec, RunStatus
    from olympus.providers.base import MockLLMProvider
    from olympus.providers.types import LLMResponse
    from olympus.session.contracts import PauseKind
    from olympus.session.router import parse_reply

    spec = GoalSpec.model_validate(
        {
            **_goal_spec(),
            "open_ambiguities": [
                {
                    "id": "backend",
                    "question": "Choose backend?",
                    "impact": "Determines storage isolation",
                    "blocking": True,
                    "default_assumption": "Use isolated storage",
                }
            ],
        }
    )
    snap = {"status": RunStatus.AWAITING_HUMAN, "spec": spec, "design": None, "plan": None}

    for phrase in ("Do not use your default assumptions.", "Do not change anything."):
        provider = MockLLMProvider(
            responses=[
                LLMResponse(
                    content=json.dumps(
                        {"action": "clarify", "answers": {"backend": "Use isolated storage"}}
                    )
                )
            ]
        )
        intent = parse_reply(
            provider=provider,
            message=phrase,
            snapshot=snap,
            pause_kind=PauseKind.CLARIFY,
        )
        assert intent.action == "unknown", (phrase, intent)
        assert intent.answers == {}

    malformed = MockLLMProvider(responses=[LLMResponse(content="not json")])
    q_intent = parse_reply(
        provider=malformed,
        message="What are the default assumptions?",
        snapshot=snap,
        pause_kind=PauseKind.CLARIFY,
    )
    assert q_intent.action == "unknown"

    malformed2 = MockLLMProvider(responses=[LLMResponse(content="not json")])
    keep_intent = parse_reply(
        provider=malformed2,
        message="Keep everything unchanged.",
        snapshot=snap,
        pause_kind=PauseKind.BLOCKED,
    )
    assert keep_intent.action == "unknown"

    # Conditional warnings and questions about defaults must never auto-fill blockers.
    cond_intent = parse_reply(
        provider=MockLLMProvider(responses=[LLMResponse(content="not json")]),
        message="If we use your default assumptions, we might lose data.",
        snapshot=snap,
        pause_kind=PauseKind.CLARIFY,
    )
    assert cond_intent.action == "unknown"
    assert cond_intent.answers == {}

    # Rejecting defaults while supplying one explicit ID must keep only that explicit ID.
    spec2 = GoalSpec.model_validate(
        {
            **_goal_spec(),
            "open_ambiguities": [
                {
                    "id": "backend",
                    "question": "Choose backend?",
                    "impact": "Storage",
                    "blocking": True,
                    "default_assumption": "isolation",
                },
                {
                    "id": "limit",
                    "question": "Choose limit?",
                    "impact": "Quota",
                    "blocking": True,
                    "default_assumption": "20",
                },
            ],
        }
    )
    snap2 = {"status": RunStatus.AWAITING_HUMAN, "spec": spec2, "design": None, "plan": None}
    mixed_intent = parse_reply(
        provider=MockLLMProvider(
            responses=[
                LLMResponse(
                    content=json.dumps(
                        {"action": "clarify", "answers": {"backend": "guessed", "limit": "20"}}
                    )
                )
            ]
        ),
        message="Do not use your default assumptions.\nbackend: exact choice",
        snapshot=snap2,
        pause_kind=PauseKind.CLARIFY,
    )
    assert mixed_intent.action == "clarify"
    assert mixed_intent.answers == {"backend": "exact choice"}


def test_session_reply_preserves_builtin_workflow_and_explicit_stage_driver_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    _init_git_repo(repo)
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.plan.append("not-json-plan")
    script.plan.append("not-json-plan")
    script.plan.append("not-json-plan")
    _patch_provider(monkeypatch, script)

    code, start = _invoke(
        [
            "start",
            "Build a calculator library",
            "--workflow",
            "repo_fast_patch",
            "--stage-driver",
            "native",
            "--workspace",
            str(repo),
            "--state-dir",
            str(state),
            "--provider",
            "mock",
        ]
    )
    assert code == 0, json.dumps(start, indent=2)
    assert start["pause_kind"] == "blocked"
    run_id = str(start["run_id"])

    # Reply omitting --workflow must reload repo_fast_patch and recover at plan.
    script2 = SessionScript()
    script2.router.append(json.dumps({"action": "revise", "feedback": "Use calc.py"}))
    script2.plan.append(json.dumps(_plan()))
    _patch_provider(monkeypatch, script2)
    code2, resumed = _invoke(
        [
            "reply",
            run_id,
            "Use calc.py",
            "--step-mode",
            "plan_only",
            "--workspace",
            str(repo),
            "--state-dir",
            str(state),
            "--provider",
            "mock",
        ]
    )
    assert code2 == 0, json.dumps(resumed, indent=2)
    assert resumed["workflow"] == "repo_fast_patch"
    assert resumed["pause_kind"] == "plan_handoff"
    events = RunStore(state / "olympus.db").list_events(run_id)
    continued = [e for e in events if e.event_type == "session.continued"]
    assert continued[-1].payload["recovery_node"] == "plan"

    # Explicit --stage-driver pi on reply must not be ignored even when snapshot recorded native.
    monkeypatch.setenv("OLYMPUS_PI_BIN", str(tmp_path / "missing-pi"))
    from olympus.harness.pi import _reset_pi_availability_cache

    _reset_pi_availability_cache()
    script3 = SessionScript()
    script3.router.append(json.dumps({"action": "revise", "feedback": "Try with pi"}))
    _patch_provider(monkeypatch, script3)
    code3, failed = _invoke(
        [
            "reply",
            run_id,
            "Try with pi",
            "--stage-driver",
            "pi",
            "--workspace",
            str(repo),
            "--state-dir",
            str(state),
            "--provider",
            "mock",
        ]
    )
    assert code3 == 1
    assert failed["status"] == "failed"
    assert "PiDriver requires" in str(failed["error"])
    _reset_pi_availability_cache()


def test_session_reopen_completed_custom_align_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import re

    repo = tmp_path / "repo"
    state = tmp_path / "state"
    _init_git_repo(repo)
    custom = tmp_path / "custom.yaml"
    original = (WORKFLOW_DIR / "product_blueprint.yaml").read_text(encoding="utf-8")
    renamed = re.sub(r"(?m)(:\s*)align$", r"\1custom_align", original)
    custom.write_text(
        renamed.replace("name: product_blueprint", "name: custom_blueprint"),
        encoding="utf-8",
    )
    script = SessionScript()
    script.align.extend([json.dumps(_goal_spec()), json.dumps(_goal_spec(version=2))])
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.router.append(json.dumps({"action": "reopen", "feedback": "Add validation rule"}))
    _patch_provider(monkeypatch, script)

    code, started = _invoke(
        [
            "start",
            "Build a calculator library",
            "--workflow",
            str(custom),
            *_base(repo, state),
        ]
    )
    assert code == 0 and started["status"] == "completed", started
    code2, reopened = _invoke(
        ["reply", str(started["run_id"]), "Add validation rule", *_base(repo, state)]
    )
    assert code2 == 0, reopened
    assert reopened["workflow"] == "custom_blueprint"
    assert reopened["error"] is None


def test_explicit_revise_prefix_while_execution_blocked_triggers_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    _init_git_repo(repo)
    # Negation wording ("never") would be classified as unknown by the NL router if not
    # parsed deterministically before parse_reply; no script.router entry is queued.
    revision = "revise: never modify tests and remove division"
    script = SessionScript()
    script.align.extend([json.dumps(_goal_spec()), json.dumps(_goal_spec(version=2))])
    script.design.extend([json.dumps(_design()), json.dumps(_design())])
    script.plan.extend([json.dumps(_plan(command=_FAILING_CHECK)), json.dumps(_plan())])
    script.workers.extend([("calculator.py", _CALC_CODE)] * 6)
    _patch_provider(monkeypatch, script)

    code, blocked = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0 and blocked["pause_kind"] == "blocked", blocked
    run_id = str(blocked["run_id"])

    code2, reply = _invoke(["reply", run_id, revision, *_base(repo, state)])
    assert code2 == 0, reply
    store = RunStore(state / "olympus.db")
    events = store.list_events(run_id)
    assert any(e.event_type == "run.revision_requested" for e in events)
    continuations = [e.payload for e in events if e.event_type == "session.continued"]
    assert continuations and continuations[-1].get("recovery_mode") != "in_place"


def test_budget_paused_execution_resumes_in_place_without_replanning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from olympus.core.contracts import RunStatus, TaskStatus

    repo = tmp_path / "repo"
    state = tmp_path / "state"
    _init_git_repo(repo)
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.workers.extend([("calculator.py", _CALC_CODE)] * 4)
    script.router.append(json.dumps({"action": "revise", "feedback": "Continue execution"}))
    _patch_provider(monkeypatch, script)

    # Start in plan-only mode to get an approved bound plan, then simulate a budget pause
    # where the task returned to READY and the run paused in AWAITING_HUMAN / BLOCKED.
    code, handoff = _invoke(
        ["start", "Build a calculator library", "--step-mode", "plan_only", *_base(repo, state)]
    )
    assert code == 0 and handoff["pause_kind"] == "plan_handoff", handoff
    run_id = str(handoff["run_id"])
    store = RunStore(state / "olympus.db")
    snap = store.get_run(run_id)
    assert snap["plan"] is not None
    task = snap["plan"].tasks[0]
    store.save_plan(
        run_id,
        snap["plan"].model_copy(
            update={"tasks": [task.model_copy(update={"status": TaskStatus.READY})]}
        ),
        actor="chronos",
    )
    snap2 = store.get_run(run_id)
    store.set_run_status(
        run_id,
        RunStatus.AWAITING_HUMAN,
        expected_revision=int(snap2["state_revision"]),
        actor="chronos",
        stage="verify",
        detail={"reason": "Paused due to budget limit"},
    )
    store.append_event(
        run_id=run_id,
        stage="session",
        event_type="session.paused",
        actor="session",
        payload={"pause_kind": "blocked", "node": None, "reason": "Budget paused"},
    )

    code2, resumed = _invoke(
        ["reply", run_id, "Continue execution", "--step-mode", "execute", *_base(repo, state)]
    )
    assert code2 == 0 and resumed["status"] == "completed", resumed
    events = store.list_events(run_id)
    assert not any(e.event_type == "run.revision_requested" for e in events)
    continuations = [e.payload for e in events if e.event_type == "session.continued"]
    assert continuations and continuations[-1].get("recovery_mode") == "resume_execution"


def test_parse_reply_fast_paths_avoid_llm_and_accept_standalone_go_on(tmp_path: Path) -> None:
    from olympus.session.contracts import PauseKind
    from olympus.session.router import parse_reply
    from olympus.session.runner import SessionConfig

    def _forbid_llm(_m: object, _t: object) -> LLMResponse:
        raise AssertionError("fast-path should resolve without invoking LLM")

    forbid = MockLLMProvider(responder=_forbid_llm)
    spec = GoalSpec.model_validate(_goal_spec(ambiguity=True, two_ambiguities=True))
    snap = {"status": RunStatus.AWAITING_HUMAN, "spec": spec, "plan": None}

    # 1. Blocked recovery hint resolves deterministically with 0 LLM calls.
    blocked_intent = parse_reply(
        provider=forbid,
        message="Retry after transient network error.",
        snapshot=snap,
        pause_kind=PauseKind.BLOCKED,
    )
    assert blocked_intent.action == "revise"
    assert blocked_intent.feedback == "Retry after transient network error."

    # 2. Standalone continuation replies accept recorded defaults during clarify with 0 LLM calls.
    for phrase in ("Go on", "continue", "proceed.", "Please go ahead"):
        intent = parse_reply(
            provider=forbid,
            message=phrase,
            snapshot=snap,
            pause_kind=PauseKind.CLARIFY,
            gate="spec",
        )
        assert intent.action == "clarify"
        assert intent.answers == {
            "AMB-1": "raise ValueError('zero')",
            "AMB-2": "int and float",
        }

    # 3. Complete explicit ID answers resolve with 0 LLM calls (even when pause_kind=BLOCKED).
    for pause in (PauseKind.CLARIFY, PauseKind.BLOCKED):
        explicit = parse_reply(
            provider=forbid,
            message="AMB-1: return None\nAMB-2: floats only",
            snapshot=snap,
            pause_kind=pause,
            gate="spec" if pause == PauseKind.CLARIFY else None,
        )
        assert explicit.action == "clarify"
        assert explicit.answers == {"AMB-1": "return None", "AMB-2": "floats only"}

    # 4. Failed run recovery hint resolves deterministically with 0 LLM calls.
    failed_snap = {"status": RunStatus.FAILED, "spec": None, "plan": None}
    failed_intent = parse_reply(
        provider=forbid,
        message="Redo the interrupted step after the transient network error.",
        snapshot=failed_snap,
        pause_kind=None,
    )
    assert failed_intent.action == "revise"
    assert failed_intent.feedback == "Redo the interrupted step after the transient network error."

    # 5. Pure or negated proceed/approve questions must NEVER trigger approve/revise.
    blocked_no_spec = {"status": RunStatus.AWAITING_HUMAN, "spec": None, "plan": None}
    for test_snap, test_pause in (
        (blocked_no_spec, PauseKind.BLOCKED),
        (failed_snap, None),
    ):
        for test_msg in ("Should we proceed?", "Keep everything unchanged. Should we proceed?"):
            for provider_impl in (
                MockLLMProvider(responses=[LLMResponse(content="not json")]),
                MockLLMProvider(
                    responses=[
                        LLMResponse(content=json.dumps({"action": "approve", "message": "ok"}))
                    ]
                ),
                MockLLMProvider(
                    responses=[
                        LLMResponse(
                            content=json.dumps(
                                {"action": "revise", "feedback": "Proceed", "message": "ok"}
                            )
                        )
                    ]
                ),
            ):
                question_intent = parse_reply(
                    provider=provider_impl,
                    message=test_msg,
                    snapshot=test_snap,
                    pause_kind=test_pause,
                )
                assert question_intent.action == "unknown", (test_msg, test_pause, question_intent)

    # 6. Headless SessionConfig uses a bounded worker_max_steps default for convergence.
    assert SessionConfig(workspace_root=tmp_path).worker_max_steps == 40


def test_reply_requeues_stranded_running_or_verifying_task_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resuming a paused/recovered run with a stranded VERIFYING task must requeue and complete."""
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    _init_git_repo(repo)
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.workers.extend([("calculator.py", _CALC_CODE)] * 4)
    _patch_provider(monkeypatch, script)

    code, handoff = _invoke(
        ["start", "Build a calculator library", "--step-mode", "plan_only", *_base(repo, state)]
    )
    assert code == 0 and handoff["pause_kind"] == "plan_handoff", handoff
    run_id = str(handoff["run_id"])

    # Simulate an abrupt kill/timeout while T1 was mid-flight in VERIFYING status.
    store = RunStore(state / "olympus.db")
    leased = store.acquire_task_lease(run_id, "T1", actor="chronos")
    store.update_task_with_lease(
        run_id,
        "T1",
        expected_lease_generation=leased.lease_generation,
        mutator=lambda node: node.model_copy(update={"status": TaskStatus.VERIFYING}),
        actor="chronos",
        event_type="task.verifying",
    )
    snap2 = store.get_run(run_id)
    store.set_run_status(
        run_id,
        RunStatus.AWAITING_HUMAN,
        expected_revision=int(snap2["state_revision"]),
        actor="session",
        stage="session",
        detail={"reason": "Recovered stranded run"},
    )
    store.append_event(
        run_id=run_id,
        stage="session",
        event_type="session.paused",
        actor="session",
        payload={"pause_kind": "blocked", "node": None, "reason": "Recovered stranded run"},
    )

    code2, resumed = _invoke(
        ["reply", run_id, "Resume after timeout", "--step-mode", "execute", *_base(repo, state)]
    )
    assert code2 == 0 and resumed["status"] == "completed", resumed
    events = store.list_events(run_id)
    assert not any(e.event_type == "run.revision_requested" for e in events)
    requeues = [e for e in events if e.event_type == "task.requeued"]
    assert requeues and requeues[-1].payload["task_id"] == "T1"
    continuations = [e.payload for e in events if e.event_type == "session.continued"]
    assert continuations and continuations[-1].get("requeued_task_ids") == ["T1"]


def test_reply_requeues_stranded_task_alongside_blocked_sibling_on_in_place_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-place recovery of a BLOCKED task must also requeue any stranded RUNNING/VERIFYING peer."""
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    _init_git_repo(repo)
    plan_two = _plan()
    plan_two["tasks"].append(
        {
            "task_id": "T2",
            "title": "Helper module",
            "objective": "Add helper.py",
            "write_scope": ["helper.py"],
            "requirement_refs": ["R1"],
            "depends_on": [],
            "acceptance_checks": [
                {
                    "id": "C2",
                    "kind": "command",
                    "description": "Import helper",
                    "command": "python3 -c 'import helper'",
                }
            ],
        }
    )
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(plan_two))
    script.workers.extend(
        [
            ("calculator.py", _CALC_CODE),
            ("helper.py", "VALUE = 1\n"),
            ("calculator.py", _CALC_CODE),
            ("helper.py", "VALUE = 1\n"),
        ]
    )
    _patch_provider(monkeypatch, script)

    code, handoff = _invoke(
        ["start", "Build a calculator library", "--step-mode", "plan_only", *_base(repo, state)]
    )
    assert code == 0 and handoff["pause_kind"] == "plan_handoff", handoff
    run_id = str(handoff["run_id"])

    store = RunStore(state / "olympus.db")
    l1 = store.acquire_task_lease(run_id, "T1", actor="chronos")
    store.update_task_with_lease(
        run_id,
        "T1",
        expected_lease_generation=l1.lease_generation,
        mutator=lambda node: node.model_copy(
            update={"status": TaskStatus.BLOCKED, "last_error": "needs hint"}
        ),
        actor="chronos",
        event_type="task.blocked",
    )
    l2 = store.acquire_task_lease(run_id, "T2", actor="chronos")
    store.update_task_with_lease(
        run_id,
        "T2",
        expected_lease_generation=l2.lease_generation,
        mutator=lambda node: node.model_copy(update={"status": TaskStatus.VERIFYING}),
        actor="chronos",
        event_type="task.verifying",
    )
    snap2 = store.get_run(run_id)
    store.set_run_status(
        run_id,
        RunStatus.AWAITING_HUMAN,
        expected_revision=int(snap2["state_revision"]),
        actor="session",
        stage="session",
        detail={"reason": "Blocked with stranded peer"},
    )
    store.append_event(
        run_id=run_id,
        stage="session",
        event_type="session.paused",
        actor="session",
        payload={"pause_kind": "blocked", "node": None, "reason": "Blocked with stranded peer"},
    )

    code2, resumed = _invoke(
        ["reply", run_id, "Fix both tasks", "--step-mode", "execute", *_base(repo, state)]
    )
    assert code2 == 0 and resumed["status"] == "completed", resumed
    events = store.list_events(run_id)
    continuations = [e.payload for e in events if e.event_type == "session.continued"]
    assert continuations
    assert continuations[-1].get("retried_task_ids") == ["T1"]
    assert continuations[-1].get("requeued_task_ids") == ["T2"]


def test_router_disables_thinking_and_uses_streaming_short_timeout() -> None:
    from olympus.providers.base import LLMResponse, MockLLMProvider
    from olympus.session.router import parse_reply

    provider = MockLLMProvider(
        responses=[LLMResponse(content='{"action": "approve", "message": "looks good"}')]
    )
    intent = parse_reply(
        provider=provider,
        message="Looks reasonable, please proceed once ready.",
        snapshot={"spec": None, "status": RunStatus.AWAITING_HUMAN},
        pause_kind=PauseKind.APPROVAL,
        gate="plan",
    )
    assert intent.action == "approve"
    assert len(provider.recorded_options) == 1
    assert provider.recorded_options[0]["enable_thinking"] is False
    assert provider.recorded_options[0]["stream"] is True
    assert provider.recorded_options[0]["timeout"] == 30.0

    class _LegacyProvider:
        model_name = "legacy"

        def complete(
            self,
            *,
            messages: Sequence[ChatMessage],
            tools: Sequence[object] = (),
            temperature: float = 0.0,
        ) -> LLMResponse:
            del messages, tools, temperature
            return LLMResponse(content='{"action": "approve", "message": "ok"}')

    legacy_intent = parse_reply(
        provider=_LegacyProvider(),  # type: ignore[arg-type]
        message="Looks reasonable, please proceed once ready.",
        snapshot={"spec": None, "status": RunStatus.AWAITING_HUMAN},
        pause_kind=PauseKind.APPROVAL,
        gate="plan",
    )
    assert legacy_intent.action == "approve"
