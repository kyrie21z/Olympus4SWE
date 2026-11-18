"""Stage 2 Orchestrate & Verify MANIFEST.md integrity artifact tests."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from olympus.core.contracts import (
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    CheckKind,
    GoalSpec,
    RunMode,
    RunStatus,
    TaskPlan,
    TaskStatus,
)
from olympus.core.store import RunStore
from olympus.providers.base import (
    ChatMessage,
    LLMResponse,
    MockLLMProvider,
    ToolCall,
    ToolSpec,
)
from olympus.stages.align import AlignStage
from olympus.stages.orchestrate import OrchestrateStage

Responder = Callable[[Sequence[ChatMessage], Sequence[ToolSpec]], LLMResponse]


def _init_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# Sample\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Dev"], cwd=root, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)


def _confirmed_run(tmp_path: Path) -> tuple[RunStore, str]:
    store = RunStore(tmp_path / ".olympus" / "olympus.db")
    run_id = store.create_run(title="Manifest run", mode=RunMode.SIMULATION)
    spec = GoalSpec(
        spec_id="spec-manifest",
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
    align = AlignStage(store=store, provider=MockLLMProvider(), workspace_root=tmp_path)
    align.confirm_spec(run_id=run_id, approver="zeus")
    return store, run_id


def _plan_payload(
    *,
    spec_id: str,
    spec_hash: str,
    file_name: str,
    checks: list[dict[str, Any]],
    max_cycles: int = 3,
) -> dict[str, Any]:
    return {
        "plan_id": "plan-manifest",
        "spec_id": spec_id,
        "spec_hash": spec_hash,
        "version": 1,
        "tasks": [
            {
                "task_id": "T1",
                "title": f"Implement {file_name}",
                "objective": f"Create {file_name} inside the sandbox",
                "requirement_refs": ["R1"],
                "depends_on": [],
                "write_scope": [file_name],
                "max_cycles": max_cycles,
                "acceptance_checks": checks,
            }
        ],
    }


def _stage_responder(
    *,
    plan: dict[str, Any],
    file_name: str,
    review_passes: list[bool],
) -> Responder:
    review_calls = 0

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        nonlocal review_calls
        del tools
        system_message = messages[0].content
        user_message = messages[1].content if len(messages) > 1 else ""

        if "Task Planner" in system_message:
            return LLMResponse(content=json.dumps(plan))

        if "Execution Engineer" in system_message:
            match = re.search(r"Sandbox workspace root: (.+)", user_message)
            assert match is not None
            sandbox = Path(match.group(1).strip())
            if not any(msg.role == "tool" for msg in messages):
                return LLMResponse(
                    content=f"writing {file_name}",
                    tool_calls=(
                        ToolCall(
                            id="w1",
                            name="write_file",
                            arguments={
                                "path": str(sandbox / file_name),
                                "content": "FLAG = True\n",
                            },
                        ),
                    ),
                )
            return LLMResponse(content=f"implemented {file_name}")

        if "Independent Verifier" in system_message:
            passed = review_passes[min(review_calls, len(review_passes) - 1)]
            review_calls += 1
            findings: list[dict[str, Any]] = []
            if not passed:
                findings.append(
                    {
                        "id": "F1",
                        "severity": "important",
                        "file_path": file_name,
                        "line": 1,
                        "message": "Missing overflow guard",
                    }
                )
            verdict = {
                "task_id": "T1",
                "attempt": 1,
                "passed": passed,
                "summary": "Looks good" if passed else "Missing overflow guard",
                "findings": findings,
            }
            return LLMResponse(content=json.dumps(verdict))

        raise AssertionError(f"Unexpected system prompt: {system_message[:80]}")

    return responder


def test_orchestrate_writes_manifest_for_rejected_then_accepted_attempts(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    plan = _plan_payload(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        file_name="feature.py",
        checks=[
            {
                "id": "CHK-1",
                "kind": CheckKind.COMMAND.value,
                "description": "Verify the flag module imports",
                "command": "python3 -c 'import feature; assert feature.FLAG'",
            },
            {
                "id": "CHK-REV-1",
                "kind": CheckKind.INDEPENDENT_REVIEW.value,
                "description": "Cold-start independent review",
            },
        ],
    )
    orch = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(
            responder=_stage_responder(
                plan=plan,
                file_name="feature.py",
                review_passes=[False, True],
            )
        ),
        workspace_root=tmp_path,
    )
    orch.propose_plan(run_id=run_id, auto_approve=True)
    snap = orch.run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.COMPLETED
    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.ACCEPTED
    assert task.attempt == 2

    attempts_dir = tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "T1"
    rejected_manifest = attempts_dir / "attempt-1" / "MANIFEST.md"
    accepted_manifest = attempts_dir / "attempt-2" / "MANIFEST.md"
    assert rejected_manifest.is_file()
    assert accepted_manifest.is_file()

    rejected_text = rejected_manifest.read_text(encoding="utf-8")
    assert "# MANIFEST — T1 (attempt 1)" in rejected_text
    assert "Outcome: rejected" in rejected_text
    assert "Checks: 1/2 passed" in rejected_text
    assert "Diff: +1/-0 lines" in rejected_text
    assert "Review verdict" in rejected_text
    assert "Missing overflow guard" in rejected_text
    assert "Turn: turn-" in rejected_text
    assert "hephaestus:T1 · 2 calls" in rejected_text

    accepted_text = accepted_manifest.read_text(encoding="utf-8")
    assert "# MANIFEST — T1 (attempt 2)" in accepted_text
    assert "Outcome: accepted" in accepted_text
    assert "Checks: 2/2 passed" in accepted_text
    assert "Looks good" in accepted_text

    # Existing artifacts remain untouched alongside the new manifest
    assert (attempts_dir / "attempt-1" / "diff.patch").is_file()
    assert (attempts_dir / "attempt-1" / "verdict-cycle-1.json").is_file()
    assert (attempts_dir / "attempt-2" / "diff.patch").is_file()
    assert (attempts_dir / "attempt-2" / "verdict-cycle-2.json").is_file()


def test_orchestrate_writes_manifest_when_command_checks_fail(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    plan = _plan_payload(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        file_name="gate.py",
        max_cycles=2,
        checks=[
            {
                "id": "CHK-1",
                "kind": CheckKind.COMMAND.value,
                "description": "Always fails",
                "command": "python3 -c 'import sys; sys.exit(1)'",
            }
        ],
    )
    orch = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(
            responder=_stage_responder(
                plan=plan,
                file_name="gate.py",
                review_passes=[True],
            )
        ),
        workspace_root=tmp_path,
    )
    orch.propose_plan(run_id=run_id, auto_approve=True)
    snap = orch.run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.AWAITING_HUMAN
    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.BLOCKED
    assert task.review_cycle == 2

    attempts_dir = tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "T1"
    for attempt in (1, 2):
        manifest = attempts_dir / f"attempt-{attempt}" / "MANIFEST.md"
        assert manifest.is_file()
        text = manifest.read_text(encoding="utf-8")
        assert f"# MANIFEST — T1 (attempt {attempt})" in text
        assert "Outcome: checks_failed" in text
        assert "Checks: 0/1 passed" in text
        assert "CHK-1 (command): not passed — Always fails" in text
        assert (attempts_dir / f"attempt-{attempt}" / "check-1-CHK-1.log").is_file()


def test_design_spec_v2_readonly_contract_hook_and_serial_handoff(tmp_path: Path) -> None:
    from olympus.core.contracts import (
        ApprovalDecision,
        ApprovalRecord,
        ApprovalTarget,
        DesignSpec,
        ExecutionStrategy,
        SpecStatus,
        TaskKind,
        TaskPlan,
    )
    from olympus.harness.hooks import default_write_scope_hook

    # 1. Wave 1 hook rejects edits to shared_contract_paths even if write_scope is broad.
    hook = default_write_scope_hook(
        ["src/"],
        forbidden_globs=["src/contracts/"],
        workspace_root=tmp_path,
    )
    allowed, _ = hook("write_file", {"path": str(tmp_path / "src" / "api" / "app.py")})
    assert allowed is True
    denied, reason = hook("write_file", {"path": str(tmp_path / "src" / "contracts" / "dto.py")})
    assert denied is False
    assert "read-only shared contract path" in reason

    # 2. ExecutionStrategy.SERIAL_HANDOFF forces sequential execution and injects invariants.
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    design = DesignSpec.model_validate(
        {
            "design_id": "design-serial-1",
            "spec_id": spec.spec_id,
            "spec_hash": spec.content_hash,
            "architecture_summary": "Tightly coupled flag engine sharing single config table.",
            "shared_invariants": [
                {
                    "id": "INV-FLAG-01",
                    "category": "naming_convention",
                    "rule": "Prefix every feature flag key with 'ff_'.",
                    "enforcement_hint": "FlagKey validator",
                }
            ],
            "shared_contract_paths": ["contracts/"],
            "execution_strategy": ExecutionStrategy.SERIAL_HANDOFF.value,
            "strategy_rationale": "Single config ledger mutates shared schema sequentially.",
            "modules": [
                {
                    "id": "M1",
                    "name": "Core",
                    "responsibility": "Flag core",
                    "requirement_refs": ["R1"],
                    "target_paths": ["src/flags/"],
                }
            ],
            "interfaces": [],
            "status": SpecStatus.CONFIRMED.value,
        }
    )
    store.save_design(run_id, design, actor="design")
    store.record_approval(
        ApprovalRecord(
            run_id=run_id,
            target=ApprovalTarget.DESIGN_SPEC,
            target_id=design.design_id,
            target_hash=design.content_hash,
            approver="zeus",
            decision=ApprovalDecision.APPROVED,
        )
    )
    plan = TaskPlan.model_validate(
        {
            "plan_id": "plan-serial-1",
            "spec_id": spec.spec_id,
            "spec_hash": spec.content_hash,
            "design_id": design.design_id,
            "design_hash": design.content_hash,
            "tasks": [
                {
                    "task_id": "T0",
                    "title": "Pin contracts",
                    "objective": "Create contract file",
                    "requirement_refs": ["R1"],
                    "design_refs": ["M1"],
                    "kind": TaskKind.CONTRACT.value,
                    "write_scope": ["contracts/"],
                    "acceptance_checks": [
                        {
                            "id": "T0-C1",
                            "kind": CheckKind.COMMAND.value,
                            "description": "ok",
                            "command": "python -c 'pass'",
                        }
                    ],
                },
                {
                    "task_id": "T1",
                    "title": "Implement core",
                    "objective": "Implement flags",
                    "requirement_refs": ["R1"],
                    "design_refs": ["M1"],
                    "depends_on": ["T0"],
                    "kind": TaskKind.IMPLEMENT.value,
                    "write_scope": ["src/flags/"],
                    "acceptance_checks": [
                        {
                            "id": "T1-C1",
                            "kind": CheckKind.COMMAND.value,
                            "description": "ok",
                            "command": "python -c 'pass'",
                        }
                    ],
                },
            ],
        }
    )
    store.save_plan(run_id, plan, actor="orchestrate")
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

    seen_prompts: list[str] = []

    def _capture_responder(
        messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        del tools
        system = messages[0].content
        user_msg = messages[-1].content
        seen_prompts.append(user_msg)
        if "Independent Verifier (Argus)" in system:
            return LLMResponse(
                content=json.dumps(
                    {
                        "task_id": "T0",
                        "attempt": 1,
                        "passed": True,
                        "summary": "Verified invariants and seams.",
                        "findings": [],
                    }
                )
            )
        return LLMResponse(content="Done.")

    orch = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=_capture_responder),
        workspace_root=tmp_path,
    )
    snap = orch.run_execution(run_id=run_id)
    assert snap["status"] == RunStatus.COMPLETED
    assert any("INV-FLAG-01" in p for p in seen_prompts)
    assert any("READ-ONLY shared contracts" in p for p in seen_prompts)
    assert any("Prior Accepted Handoff Summary (serial_handoff)" in p for p in seen_prompts)
    events = [
        e for e in store.list_events(run_id) if e.event_type == "run.status_changed"
    ]
    assert any(e.payload.get("serial_execution") is True for e in events)


def test_propose_plan_repairs_broken_inline_python_check_via_validation_loop(
    tmp_path: Path,
) -> None:
    store, run_id = _confirmed_run(tmp_path)
    snap = store.get_run(run_id)
    spec = snap["spec"]
    assert spec is not None

    bad_check = {
        "id": "CHK-BAD",
        "kind": CheckKind.COMMAND.value,
        "description": "Frozen check",
        "command": 'python -c "import feature; try: x = 1; except Exception: pass"',
    }
    good_check = {
        "id": "CHK-GOOD",
        "kind": CheckKind.COMMAND.value,
        "description": "Frozen check",
        "command": "python -m pytest -q",
    }
    calls: list[str] = []

    def _repair_responder(
        messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        del tools
        calls.append(messages[-1].content)
        payload = _plan_payload(
            spec_id=spec.spec_id,
            spec_hash=spec.content_hash,
            file_name="feature.py",
            checks=[bad_check if len(calls) == 1 else good_check],
        )
        return LLMResponse(content=json.dumps(payload))

    orch = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=_repair_responder),
        workspace_root=tmp_path,
    )
    plan = orch.propose_plan(run_id=run_id)
    assert len(calls) == 2
    assert "[PLAN VALIDATION FAILED]" in calls[1]
    assert "invalid Python syntax" in calls[1]
    assert plan.tasks[0].acceptance_checks[0].command == "python -m pytest -q"


def test_independent_review_submit_tool_prose_and_clean_overlay_diff(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    snap = store.get_run(run_id)
    spec = snap["spec"]
    assert spec is not None

    plan = TaskPlan.model_validate(
        {
            "plan_id": "plan-agent-native",
            "spec_id": spec.spec_id,
            "spec_hash": spec.content_hash,
            "tasks": [
                {
                    "task_id": "T0",
                    "title": "Pin contract",
                    "objective": "Create contract.py",
                    "requirement_refs": ["R1"],
                    "kind": "contract",
                    "write_scope": ["contract.py"],
                    "acceptance_checks": [
                        {
                            "id": "T0-c",
                            "kind": "command",
                            "description": "import contract",
                            "command": "python -c 'import contract'",
                        }
                    ],
                },
                {
                    "task_id": "T1",
                    "title": "Implement feature",
                    "objective": "Create feature.py",
                    "requirement_refs": ["R1"],
                    "depends_on": ["T0"],
                    "write_scope": ["feature.py"],
                    "acceptance_checks": [
                        {
                            "id": "T1-c",
                            "kind": "command",
                            "description": "import feature",
                            "command": "python -c 'import contract, feature; assert feature.FLAG'",
                        }
                    ],
                },
            ],
        }
    )
    store.save_plan(run_id, plan, actor="orchestrate")
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

    review_calls = 0

    def _responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        nonlocal review_calls
        sys_msg = messages[0].content
        user_msg = messages[1].content if len(messages) > 1 else ""
        if "Execution Engineer" in sys_msg:
            if not any(m.role == "tool" for m in messages):
                ws_match = re.search(r"Sandbox workspace root: (.+)", user_msg)
                assert ws_match is not None
                ws = Path(ws_match.group(1).strip())
                target = "contract.py" if "Task T0:" in user_msg else "feature.py"
                content = "CONST = 1\n" if target == "contract.py" else "FLAG = True\n"
                return LLMResponse(
                    content=f"writing {target}",
                    tool_calls=(
                        ToolCall(
                            id="w1",
                            name="write_file",
                            arguments={"path": str(ws / target), "content": content},
                        ),
                    ),
                )
            return LLMResponse(content="Implemented cleanly.")
        if "Independent Verifier (Argus)" in sys_msg:
            review_calls += 1
            tool_names = {t.name for t in tools}
            assert "submit_review" in tool_names
            if review_calls == 1:
                # T0: submit verdict directly via submit_review tool with embedded ```python!
                return LLMResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="sr-1",
                            name="submit_review",
                            arguments={
                                "passed": True,
                                "summary": "Contract OK:\n```python\nCONST = 1\n```",
                                "findings": [],
                            },
                        ),
                    ),
                )
            # T1: overlay contract.py must NOT pollute T1's git diff; also test freeform prose!
            assert "contract.py" not in user_msg.split("Git diff / changes:")[1]
            return LLMResponse(content="All checks passed. Looks good, no defects remain.")
        raise AssertionError(f"Unexpected prompt: {sys_msg[:100]}")

    orch = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=_responder),
        workspace_root=tmp_path,
    )
    final_snap = orch.run_execution(run_id=run_id)
    assert final_snap["status"] == RunStatus.COMPLETED
    assert review_calls == 2
    t1_diff = (
        tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "T1" / "attempt-1" / "diff.patch"
    ).read_text(encoding="utf-8")
    assert "feature.py" in t1_diff
    assert "contract.py" not in t1_diff
