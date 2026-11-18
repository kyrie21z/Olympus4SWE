"""End-to-end integration tests for Stage 1 (Align), Stage 2 (Orchestrate), and CLI."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from olympus.cli.app import app
from olympus.core.contracts import (
    ApprovalTarget,
    CheckKind,
    CritiquePerspective,
    CritiqueSeverity,
    GoalSpec,
    RunMode,
    RunStatus,
    TaskStatus,
)
from olympus.core.store import RunStore
from olympus.providers.base import ChatMessage, LLMResponse, MockLLMProvider, ToolCall, ToolSpec
from olympus.stages.align import AlignStage
from olympus.stages.common import parse_model_json
from olympus.stages.orchestrate import OrchestrateStage


def _init_git_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# Sample\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Dev"], cwd=root, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)


def test_stage1_align_critique_clarify_and_freeze(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    store = RunStore(tmp_path / ".olympus" / "olympus.db")
    run_id = store.create_run(title="Auth align", mode=RunMode.EXECUTION)

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        sys_msg = messages[0].content
        if "Requirements Alignment Architect" in sys_msg:
            return LLMResponse(
                content=json.dumps(
                    {
                        "spec_id": "spec-auth",
                        "version": 1,
                        "goal": "Implement token signer",
                        "requirements": [
                            {
                                "id": "R1",
                                "description": "Sign and verify HMAC tokens",
                                "acceptance_criteria": ["Expired token rejected"],
                            }
                        ],
                        "out_of_scope": ["OAuth"],
                        "open_ambiguities": [
                            {
                                "id": "AMB-1",
                                "question": "Default token TTL?",
                                "impact": "Expiration behavior",
                                "options": ["15m", "1h"],
                                "blocking": True,
                            }
                        ],
                    }
                )
            )
        if "product reviewer" in sys_msg:
            return LLMResponse(
                content=json.dumps(
                    {
                        "notes": [
                            {
                                "id": "CRIT-P1",
                                "perspective": CritiquePerspective.PRODUCT.value,
                                "severity": CritiqueSeverity.WARNING.value,
                                "summary": "Consider clock-skew documentation",
                            }
                        ]
                    }
                )
            )
        if "engineering reviewer" in sys_msg:
            return LLMResponse(content=json.dumps({"notes": []}))
        return LLMResponse(
            content=json.dumps(
                {
                    "notes": [
                        {
                            "id": "CRIT-QA1",
                            "perspective": CritiquePerspective.QA.value,
                            "severity": CritiqueSeverity.BLOCKING.value,
                            "summary": "Missing negative test criterion for tampered signature",
                        }
                    ]
                }
            )
        )

    align = AlignStage(
        store=store,
        provider=MockLLMProvider(responder=responder),
        workspace_root=tmp_path,
    )
    spec = align.propose_spec(run_id=run_id, raw_goal="Build token signer", enable_critique=True)

    assert sorted(spec.unresolved_blockers()) == ["ambiguity:AMB-1", "critique:CRIT-QA1"]
    # Cannot freeze while blockers remain
    with pytest.raises(ValueError, match="unresolved blockers"):
        align.confirm_spec(run_id=run_id)

    clarified = align.apply_clarifications(
        run_id=run_id,
        answers={"AMB-1": "1h"},
        resolutions={"CRIT-QA1": "Added tampered signature check to R1"},
        extra_out_of_scope=["Key rotation UI"],
    )
    assert clarified.version == 2
    assert clarified.content_hash != spec.content_hash
    assert clarified.unresolved_blockers() == []
    # Non-blocking minority note preserved
    assert any(n.id == "CRIT-P1" for n in clarified.minority_notes)

    confirmed = align.confirm_spec(run_id=run_id, approver="zeus")
    assert store.is_approved(
        run_id=run_id,
        target=ApprovalTarget.GOAL_SPEC,
        target_id=confirmed.spec_id,
        target_hash=confirmed.content_hash,
    )


def test_stage2_end_to_end_rework_cold_review_and_dag_unlock(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    store = RunStore(tmp_path / ".olympus" / "olympus.db")
    run_id = store.create_run(title="Math module", mode=RunMode.EXECUTION)

    spec = GoalSpec(
        spec_id="spec-math",
        goal="Create safe_div module and tests",
        requirements=[
            {
                "id": "R1",
                "description": "Implement divide(a, b) raising ValueError on zero",
                "acceptance_criteria": ["divide(6, 2) == 3", "divide(1, 0) raises ValueError"],
            },
            {
                "id": "R2",
                "description": "Add unit tests verifying divide",
                "acceptance_criteria": ["python3 -m unittest passes"],
            },
        ],
        out_of_scope=["CLI calculator"],
    )
    align = AlignStage(store=store, provider=MockLLMProvider(), workspace_root=tmp_path)
    store.save_spec(run_id, spec)
    confirmed_spec = align.confirm_spec(run_id=run_id, approver="zeus")

    # Track per-task worker & review calls dynamically
    t1_attempts = 0
    review_calls = 0

    def orchestrate_responder(
        messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        nonlocal t1_attempts, review_calls
        del tools
        sys_msg = messages[0].content
        user_msg = messages[1].content

        if "Task Planner" in sys_msg:
            return LLMResponse(
                content=json.dumps(
                    {
                        "plan_id": "plan-math",
                        "spec_id": confirmed_spec.spec_id,
                        "spec_hash": confirmed_spec.content_hash,
                        "version": 1,
                        "tasks": [
                            {
                                "task_id": "T1",
                                "title": "Implement math_lib",
                                "objective": "Write math_lib.py with divide(a, b)",
                                "requirement_refs": ["R1"],
                                "depends_on": [],
                                "write_scope": ["math_lib.py"],
                                "acceptance_checks": [
                                    {
                                        "id": "CHK-1",
                                        "kind": CheckKind.COMMAND.value,
                                        "description": "Verify divide(6, 2)",
                                        "command": (
                                            "python3 -c 'import math_lib; "
                                            "assert math_lib.divide(6, 2) == 3'"
                                        ),
                                    },
                                    {
                                        "id": "CHK-REV-1",
                                        "kind": CheckKind.INDEPENDENT_REVIEW.value,
                                        "description": "Independent cold-start review",
                                    },
                                ],
                            },
                            {
                                "task_id": "T2",
                                "title": "Add test_math_lib",
                                "objective": "Write test_math_lib.py testing divide(a, b)",
                                "requirement_refs": ["R2"],
                                "depends_on": ["T1"],
                                "write_scope": ["test_math_lib.py"],
                                "acceptance_checks": [
                                    {
                                        "id": "CHK-2",
                                        "kind": CheckKind.COMMAND.value,
                                        "description": "Run unittest",
                                        "command": "python3 -m unittest test_math_lib.py",
                                    }
                                ],
                            },
                        ],
                    }
                )
            )

        if "Execution Engineer" in sys_msg:
            m = re.search(r"Sandbox workspace root: (.+)", user_msg)
            assert m is not None
            ws = Path(m.group(1).strip())
            has_tool_result = any(msg.role == "tool" for msg in messages)

            if "Task T1:" in user_msg:
                if not has_tool_result:
                    t1_attempts += 1
                    # Attempt 1 intentionally omits zero-division ValueError guard!
                    code = (
                        "def divide(a: float, b: float) -> float:\n    return a / b\n"
                        if t1_attempts == 1
                        else (
                            "def divide(a: float, b: float) -> float:\n"
                            "    if b == 0:\n"
                            "        raise ValueError('zero')\n"
                            "    return a / b\n"
                        )
                    )
                    return LLMResponse(
                        content="Writing math_lib.py",
                        tool_calls=(
                            ToolCall(
                                id="w1",
                                name="write_file",
                                arguments={"path": str(ws / "math_lib.py"), "content": code},
                            ),
                        ),
                    )
                return LLMResponse(content="Implemented math_lib.py")

            if "Task T2:" in user_msg:
                if not has_tool_result:
                    test_code = (
                        "import unittest\n"
                        "import math_lib\n\n"
                        "class TestMath(unittest.TestCase):\n"
                        "    def test_div(self) -> None:\n"
                        "        self.assertEqual(math_lib.divide(6, 2), 3)\n"
                        "        with self.assertRaises(ValueError):\n"
                        "            math_lib.divide(1, 0)\n"
                    )
                    return LLMResponse(
                        content="Writing test_math_lib.py",
                        tool_calls=(
                            ToolCall(
                                id="w2",
                                name="write_file",
                                arguments={
                                    "path": str(ws / "test_math_lib.py"),
                                    "content": test_code,
                                },
                            ),
                        ),
                    )
                return LLMResponse(content="Added unit tests")

        # Cold-start Independent Reviewer (Argus/Momus)
        if "Independent Verifier" in sys_msg:
            review_calls += 1
            # Verify reviewer has ZERO prior implementer messages
            assert len(messages) == 2
            if '"task_id":"T1"' in user_msg and "if b == 0:" not in user_msg:
                return LLMResponse(
                    content=json.dumps(
                        {
                            "task_id": "T1",
                            "attempt": 1,
                            "passed": False,
                            "summary": "Missing ValueError on zero division",
                            "findings": [
                                {
                                    "id": "F-1",
                                    "severity": "important",
                                    "file_path": "math_lib.py",
                                    "line": 2,
                                    "message": "Raises ZeroDivisionError instead of ValueError",
                                }
                            ],
                        }
                    )
                )
            return LLMResponse(
                content=json.dumps(
                    {
                        "task_id": "T1" if "T1" in user_msg else "T2",
                        "attempt": 1,
                        "passed": True,
                        "summary": "All checks and requirements satisfied",
                        "findings": [],
                    }
                )
            )

        raise AssertionError(f"Unexpected prompt: {sys_msg}")

    orch = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=orchestrate_responder),
        workspace_root=tmp_path,
    )
    plan = orch.propose_plan(run_id=run_id)
    assert [t.task_id for t in plan.tasks] == ["T1", "T2"]

    # Refuses execution before plan approval
    with pytest.raises(ValueError, match="not human-approved"):
        orch.run_execution(run_id=run_id)

    orch.approve_plan(run_id=run_id, approver="zeus")
    final_snap = orch.run_execution(run_id=run_id, apply_to_workspace=True)

    assert final_snap["status"] == RunStatus.COMPLETED
    tasks_by_id = {t.task_id: t for t in final_snap["plan"].tasks}
    assert tasks_by_id["T1"].status == TaskStatus.ACCEPTED
    assert tasks_by_id["T1"].attempt == 2
    assert tasks_by_id["T1"].review_cycle == 1
    assert tasks_by_id["T2"].status == TaskStatus.ACCEPTED
    assert review_calls == 3  # T1 fail + T1 pass + T2 pass

    # Synced deliverables actually pass in main repo
    assert (tmp_path / "math_lib.py").is_file()
    assert (tmp_path / "test_math_lib.py").is_file()
    proc = subprocess.run(
        ["python3", "-m", "unittest", "test_math_lib.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0


def test_max_cycles_blocks_task_and_cli_can_retry_or_accept_risk(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    store = RunStore(tmp_path / ".olympus" / "olympus.db")
    run_id = store.create_run(title="Blocked test", mode=RunMode.SIMULATION)

    spec = GoalSpec(
        spec_id="spec-blk",
        goal="Impossible check",
        requirements=[
            {
                "id": "R1",
                "description": "Task fails command check",
                "acceptance_criteria": ["Pass"],
            }
        ],
    )
    align = AlignStage(store=store, provider=MockLLMProvider(), workspace_root=tmp_path)
    store.save_spec(run_id, spec)
    confirmed = align.confirm_spec(run_id=run_id, approver="zeus")

    def failing_responder(
        messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        del tools
        if "Task Planner" in messages[0].content:
            return LLMResponse(
                content=json.dumps(
                    {
                        "plan_id": "plan-blk",
                        "spec_id": confirmed.spec_id,
                        "spec_hash": confirmed.content_hash,
                        "version": 1,
                        "tasks": [
                            {
                                "task_id": "T1",
                                "title": "Fails",
                                "objective": "Do work",
                                "requirement_refs": ["R1"],
                                "write_scope": ["a.py"],
                                "max_cycles": 2,
                                "acceptance_checks": [
                                    {
                                        "id": "C1",
                                        "kind": CheckKind.COMMAND.value,
                                        "description": "Always fails",
                                        "command": "python3 -c 'import sys; sys.exit(2)'",
                                    }
                                ],
                            }
                        ],
                    }
                )
            )
        return LLMResponse(content="Finished attempt")

    orch = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=failing_responder),
        workspace_root=tmp_path,
    )
    orch.propose_plan(run_id=run_id, auto_approve=True)
    snap = orch.run_execution(run_id=run_id)
    assert snap["status"] == RunStatus.AWAITING_HUMAN
    assert snap["plan"].tasks[0].status == TaskStatus.BLOCKED
    assert snap["plan"].tasks[0].review_cycle == 2

    # Exercise CLI status, events, retry, and accept-risk
    runner = CliRunner()
    res_status = runner.invoke(app, ["status", run_id, "--workspace", str(tmp_path)])
    assert res_status.exit_code == 0
    rev = json.loads(res_status.stdout)["state_revision"]

    res_retry = runner.invoke(
        app,
        [
            "retry",
            run_id,
            "T1",
            "--revision",
            str(rev),
            "--hint",
            "Check exit code",
            "--workspace",
            str(tmp_path),
        ],
    )
    assert res_retry.exit_code == 0
    assert json.loads(res_retry.stdout)["status"] == "ready"

    rev2 = store.get_run(run_id)["state_revision"]
    res_risk = runner.invoke(
        app,
        [
            "accept-risk",
            run_id,
            "T1",
            "--revision",
            str(rev2),
            "--reason",
            "Verified manually by owner",
            "--workspace",
            str(tmp_path),
        ],
    )
    assert res_risk.exit_code == 0
    assert json.loads(res_risk.stdout)["status"] == "accepted_with_risk"
    task0 = store.get_run(run_id)["plan"].tasks[0]
    assert task0.accepted_risk_reason == "Verified manually by owner"

    res_list = runner.invoke(app, ["list", "--workspace", str(tmp_path)])
    assert res_list.exit_code == 0
    listed_runs = json.loads(res_list.stdout)
    assert any(item["run_id"] == run_id for item in listed_runs)

    res_events = runner.invoke(app, ["events", run_id, "--workspace", str(tmp_path)])
    assert res_events.exit_code == 0
    events = json.loads(res_events.stdout)
    assert len(events) >= 3
    assert [e["event_seq"] for e in events] == sorted(e["event_seq"] for e in events)
    assert events[-1]["event_type"] == "task.accepted_with_risk"

    last_seq = events[-1]["event_seq"]
    res_events_tail = runner.invoke(
        app,
        ["events", run_id, "--after-seq", str(last_seq - 1), "--workspace", str(tmp_path)],
    )
    assert res_events_tail.exit_code == 0
    tail_events = json.loads(res_events_tail.stdout)
    assert len(tail_events) == 1
    assert tail_events[0]["event_seq"] == last_seq


def test_parse_model_json_strips_meta_keys_and_preserves_extra_forbid() -> None:
    raw = json.dumps(
        {
            "$comment": "generated by model",
            "$schema": "https://example.invalid/schema.json",
            "goal": "Implement guard",
            "requirements": [
                {
                    "id": "R1",
                    "description": "Reject zero divisor",
                    "acceptance_criteria": ["pytest passes"],
                    "$comment": "nested comment",
                }
            ],
        }
    )
    spec = parse_model_json(raw, GoalSpec)
    assert spec.goal == "Implement guard"

    with pytest.raises(ValidationError):
        parse_model_json(
            json.dumps(
                {
                    "goal": "x",
                    "requirements": [
                        {"id": "R1", "description": "d", "acceptance_criteria": ["a"]}
                    ],
                    "hallucinated_field": 123,
                }
            ),
            GoalSpec,
        )


def test_parse_model_json_ignores_embedded_markdown_fences_inside_json_strings() -> None:
    from olympus.core.contracts import ReviewVerdict
    from olympus.stages.common import extract_json_payload

    payload = json.dumps(
        {
            "task_id": "T1",
            "attempt": 1,
            "passed": True,
            "summary": (
                "Handled exception cleanly:\n"
                "```python\nexcept ValueError:\n    raise\n```\n"
            ),
            "findings": [],
        }
    )
    assert parse_model_json(payload, ReviewVerdict).passed is True
    wrapped = f"Here is the verdict:\n```json\n{payload}\n```\nDone."
    assert parse_model_json(wrapped, ReviewVerdict).summary.startswith("Handled exception")

    # Broken JSON containing a ```python snippet must report the outer JSON error, not 'char 0'
    # from extracting the inner code block.
    broken = (
        '{\n"summary": "see:\n```python\nx = 1\n```\n"broken",\n"passed": true\n}'
    )
    extracted = extract_json_payload(broken)
    assert extracted.lstrip().startswith("{")
    with pytest.raises(json.JSONDecodeError) as exc:
        json.loads(extracted)
    assert exc.value.pos > 0


def test_stage1_propose_spec_injects_observed_workspace_inventory(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    store = RunStore(tmp_path / ".olympus" / "olympus.db")
    run_id = store.create_run(title="Inventory check", mode=RunMode.EXECUTION)
    captured_user_prompts: list[str] = []

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        captured_user_prompts.append(messages[1].content)
        return LLMResponse(
            content=json.dumps(
                {
                    "goal": "Update module",
                    "requirements": [
                        {
                            "id": "R1",
                            "description": "Update src/mod.py",
                            "acceptance_criteria": ["Compile cleanly"],
                        }
                    ],
                }
            )
        )

    align = AlignStage(
        store=store,
        provider=MockLLMProvider(responder=responder),
        workspace_root=tmp_path,
    )
    align.propose_spec(run_id=run_id, raw_goal="Update mod")
    assert len(captured_user_prompts) == 1
    assert "Observed workspace files (2/2): README.md, src/mod.py" in captured_user_prompts[0]
    assert ".git" not in captured_user_prompts[0]


def test_build_sandbox_env_isolates_venv_and_pythonpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from olympus.harness.worktree import build_sandbox_env

    host_root = tmp_path / "host"
    (host_root / ".venv" / "bin").mkdir(parents=True)
    sandbox_root = tmp_path / "sandbox"
    (sandbox_root / "src").mkdir(parents=True)

    monkeypatch.setenv("VIRTUAL_ENV", "/polluted/olympus/.venv")
    monkeypatch.setenv("PYTHONHOME", "/polluted/pythonhome")
    monkeypatch.setenv("PYTHONPATH", "/polluted/other")
    monkeypatch.setenv("PATH", "/polluted/olympus/.venv/bin:/usr/bin")

    env = build_sandbox_env(sandbox_root, host_root=host_root)
    assert env["VIRTUAL_ENV"] == str((host_root / ".venv").resolve())
    assert "PYTHONHOME" not in env
    assert env["PATH"].split(":")[0] == str((host_root / ".venv" / "bin").resolve())
    pythonpath_parts = env["PYTHONPATH"].split(":")
    assert pythonpath_parts[:2] == [
        str((sandbox_root / "src").resolve()),
        str(sandbox_root.resolve()),
    ]


def test_stage2_planner_inventory_and_clarification_propagation(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "app.py").write_text("def greet():\n    return ''\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "add app"], cwd=tmp_path, check=True, capture_output=True
    )

    store = RunStore(tmp_path / ".olympus" / "olympus.db")
    run_id = store.create_run(title="Propagate Q&A", mode=RunMode.EXECUTION)

    captured_planner_prompts: list[str] = []
    captured_worker_prompts: list[str] = []
    captured_reviewer_prompts: list[str] = []

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        sys_msg = messages[0].content
        user_msg = messages[1].content
        if "Requirements Alignment Architect" in sys_msg:
            return LLMResponse(
                content=json.dumps(
                    {
                        "spec_id": "spec-prop",
                        "version": 1,
                        "goal": "Implement transform()",
                        "constraints": ["Preserve insertion order"],
                        "requirements": [
                            {
                                "id": "R1",
                                "description": "Transform dict keys",
                                "acceptance_criteria": ["Handles empty dicts"],
                            }
                        ],
                        "open_ambiguities": [
                            {
                                "id": "AMB-1",
                                "question": "What should transform({}) return?",
                                "impact": "Empty dict edge case",
                                "blocking": True,
                            }
                        ],
                    }
                )
            )
        if "Task Planner" in sys_msg:
            captured_planner_prompts.append(user_msg)
            m = re.search(r"Set spec_id='([^']+)' and spec_hash='([^']+)'", user_msg)
            assert m is not None
            return LLMResponse(
                content=json.dumps(
                    {
                        "plan_id": "plan-prop",
                        "spec_id": m.group(1),
                        "spec_hash": m.group(2),
                        "tasks": [
                            {
                                "task_id": "T1",
                                "title": "Update app.py",
                                "objective": "Implement transform in app.py",
                                "requirement_refs": ["R1"],
                                "write_scope": ["app.py"],
                                "acceptance_checks": [
                                    {
                                        "id": "C1",
                                        "kind": CheckKind.COMMAND.value,
                                        "description": "Verify transform handles empty dict",
                                        "command": (
                                            "python3 -c 'import app; "
                                            "assert app.transform({}) == {}'"
                                        ),
                                    },
                                    {
                                        "id": "C2",
                                        "kind": CheckKind.INDEPENDENT_REVIEW.value,
                                        "description": "Verify constraints and Q&A adherence",
                                    },
                                ],
                            }
                        ],
                    }
                )
            )
        if "Execution Engineer" in sys_msg:
            if not any(msg.role == "tool" for msg in messages):
                captured_worker_prompts.append(user_msg)
                ws_match = re.search(r"Sandbox workspace root: (.+)", user_msg)
                assert ws_match is not None
                ws = Path(ws_match.group(1).strip())
                return LLMResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="call-write",
                            name="write_file",
                            arguments={
                                "path": str(ws / "app.py"),
                                "content": "def transform(d):\n    return dict(d)\n",
                            },
                        ),
                    ),
                )
            return LLMResponse(content="Updated app.py cleanly.")
        if "Independent Verifier" in sys_msg:
            captured_reviewer_prompts.append(user_msg)
            return LLMResponse(
                content=json.dumps(
                    {
                        "task_id": "T1",
                        "attempt": 1,
                        "passed": True,
                        "summary": "Matches constraint and clarified empty-dict behavior.",
                        "findings": [],
                    }
                )
            )
        raise AssertionError(f"Unexpected prompt: {sys_msg}")

    provider = MockLLMProvider(responder=responder)
    align = AlignStage(store=store, provider=provider, workspace_root=tmp_path)
    align.propose_spec(run_id=run_id, raw_goal="Implement transform()")
    align.apply_clarifications(
        run_id=run_id,
        answers={"AMB-1": "Return an empty dict {} rather than None"},
    )
    align.confirm_spec(run_id=run_id)

    orch = OrchestrateStage(
        store=store,
        provider=provider,
        workspace_root=tmp_path,
        artifacts_root=tmp_path / ".olympus" / "artifacts",
    )
    orch.propose_plan(run_id=run_id, auto_approve=True)
    final_snap = orch.run_execution(run_id=run_id, apply_to_workspace=True)
    tasks_by_id = {t.task_id: t for t in final_snap["plan"].tasks}
    assert tasks_by_id["T1"].status == TaskStatus.ACCEPTED
    assert (
        len(captured_planner_prompts) == 1
        and "Observed workspace files" in captured_planner_prompts[0]
        and "'command', 'independent_review', or 'human'" in captured_planner_prompts[0]
    )
    for prompt_list in (captured_worker_prompts, captured_reviewer_prompts):
        assert len(prompt_list) == 1
        assert "Preserve insertion order" in prompt_list[0]
        assert "Return an empty dict {} rather than None" in prompt_list[0]
