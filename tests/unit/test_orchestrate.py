"""Stage 2 Orchestrate & Verify MANIFEST.md integrity artifact tests."""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import pytest
from fastapi.testclient import TestClient

import olympus.stages.orchestrate as orchestrate_module
from olympus.core.contracts import (
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    BudgetPolicy,
    CheckKind,
    DesignSpec,
    ExecutionStrategy,
    GoalSpec,
    RunMode,
    RunStatus,
    SpecStatus,
    TaskKind,
    TaskPlan,
    TaskStatus,
)
from olympus.core.control import RunControl
from olympus.core.dag import PlanValidationError, validate_plan
from olympus.core.store import RunStore
from olympus.harness.driver import (
    ActiveTurnHandle,
    DriverRegistry,
    DriverTurnRequest,
    ExecutionDriversConfig,
    TurnOutcome,
)
from olympus.providers.base import (
    ChatMessage,
    LLMResponse,
    MockLLMProvider,
    ToolCall,
    ToolSpec,
)
from olympus.stages.align import AlignStage
from olympus.stages.orchestrate import _WORKER_SYSTEM_PROMPT, OrchestrateStage
from olympus.tools.aci import build_sandbox_env
from olympus.web.projection import build_run_projection
from olympus.web.server import AriadneApp, create_fastapi_app

Responder = Callable[[Sequence[ChatMessage], Sequence[ToolSpec]], LLMResponse]


def _init_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# Sample\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Dev"], cwd=root, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)


def _confirmed_run(
    tmp_path: Path,
    *,
    criteria: list[str] | None = None,
    budget_policy: BudgetPolicy | None = None,
    mode: RunMode = RunMode.SIMULATION,
) -> tuple[RunStore, str]:
    store = RunStore(tmp_path / ".olympus" / "olympus.db")
    run_id = store.create_run(
        title="Manifest run", mode=mode, budget_policy=budget_policy
    )
    spec = GoalSpec(
        spec_id="spec-manifest",
        goal="Add a feature flag module",
        requirements=[
            {
                "id": "R1",
                "description": "Provide feature.py with FLAG = True",
                "acceptance_criteria": criteria or ["feature.FLAG is True"],
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
                content = "FLAG = True\n"
                if "[PRIOR REJECTION" in user_message:
                    content += "GUARD = True\n"
                return LLMResponse(
                    content=f"writing {file_name}",
                    tool_calls=(
                        ToolCall(
                            id="w1",
                            name="write_file",
                            arguments={
                                "path": str(sandbox / file_name),
                                "content": content,
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
                module = Path(file_name).stem
                findings.append(
                    {
                        "id": "F1",
                        "axis": "spec",
                        "severity": "important",
                        "file_path": file_name,
                        "line": 1,
                        "requirement_ref": "R1",
                        "repro_command": (
                            f"python3 -c 'import {module}; assert {module}.GUARD'"
                        ),
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


def test_delivery_acceptance_uses_executed_checks_on_the_composed_output(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(
        tmp_path, criteria=["feature.FLAG is True", "feature.AUDIT is True", "UI exposes FLAG"]
    )
    spec = store.get_run(run_id)["spec"]
    checks = [
        {
            "id": "CHK-1",
            "kind": "command",
            "description": "Verify the flag module imports",
            "command": "python3 -c 'import feature; assert feature.FLAG'",
        }
    ]
    plan = _plan_payload(
        spec_id=spec.spec_id, spec_hash=spec.content_hash, file_name="feature.py", checks=checks
    )
    plan["delivery_checks"] = [
        {
            "requirement_id": "R1",
            "criterion_index": 1,
            "command": "python3 -c 'import feature; assert feature.FLAG; print(\"tested FLAG\")'",
        },
        {
            "requirement_id": "R1",
            "criterion_index": 2,
            "command": "python3 -c 'import feature; assert feature.AUDIT'",
        },
    ]
    orch = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(
            responder=_stage_responder(plan=plan, file_name="feature.py", review_passes=[True])
        ),
        workspace_root=tmp_path,
    )
    orch.propose_plan(run_id=run_id, auto_approve=True)
    pending = build_run_projection(store, tmp_path, run_id)["delivery_acceptance"]
    assert pending["counts"] == {"passed": 0, "failed": 0, "unverified": 3}

    orch.run_execution(run_id=run_id)
    review = build_run_projection(store, tmp_path, run_id)["delivery_acceptance"]
    assert review["counts"] == {"passed": 1, "failed": 1, "unverified": 1}
    passed, failed, unverified = review["criteria"]
    assert [row["text"] for row in review["criteria"]] == [
        "feature.FLAG is True", "feature.AUDIT is True", "UI exposes FLAG"
    ]
    assert "tested FLAG" in passed["evidence"]["log"]
    assert failed["evidence"]["delivery_version"] == passed["evidence"]["delivery_version"]
    assert failed["reason"] and unverified["reason"]
    assert unverified["evidence"] is None
    assert not (tmp_path / "feature.py").exists()

    executed = next(
        event
        for event in store.list_events(run_id)
        if event.event_seq == passed["evidence"]["event_seq"]
    )
    wrong_version = {"delivery_version": "sha256:" + "0" * 64}
    store.append_event(
        run_id=run_id,
        stage=executed.stage,
        event_type=executed.event_type,
        actor=executed.actor,
        payload=executed.payload | wrong_version,
    )
    stale = build_run_projection(store, tmp_path, run_id)["delivery_acceptance"]
    assert stale["counts"] == {"passed": 0, "failed": 0, "unverified": 3}
    assert "交付版本" in stale["criteria"][0]["reason"]

    store.append_event(
        run_id=run_id,
        stage=executed.stage,
        event_type=executed.event_type,
        actor=executed.actor,
        payload=executed.payload,
    )
    artifact = tmp_path / ".olympus" / "runs" / run_id / "artifacts"
    check_log = artifact / passed["evidence"]["receipt_id"] / "check-1.log"
    check_log.write_text("different run output", encoding="utf-8")
    tampered = build_run_projection(store, tmp_path, run_id)["delivery_acceptance"]
    assert tampered["counts"] == {"passed": 0, "failed": 1, "unverified": 2}
    assert "指纹不一致" in tampered["criteria"][0]["reason"]

    store.save_plan(run_id, store.get_run(run_id)["plan"])
    replaced = build_run_projection(store, tmp_path, run_id)["delivery_acceptance"]
    assert replaced["counts"] == {"passed": 0, "failed": 0, "unverified": 3}

    revised = spec.model_copy(
        update={
            "version": spec.version + 1,
            "requirements": [
                spec.requirements[0].model_copy(update={"acceptance_criteria": ["FLAG is false"]})
            ],
        }
    )
    store.save_spec(run_id, revised)
    assert build_run_projection(store, tmp_path, run_id)["delivery_acceptance"] is None
    AlignStage(store=store, provider=MockLLMProvider(), workspace_root=tmp_path).confirm_spec(
        run_id=run_id
    )
    new_review = build_run_projection(store, tmp_path, run_id)["delivery_acceptance"]
    assert new_review["spec_hash"] == revised.content_hash
    assert new_review["counts"] == {"passed": 0, "failed": 0, "unverified": 1}
    assert new_review["criteria"][0]["text"] == "FLAG is false"


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


@pytest.mark.parametrize("topology", ["tdd_first", "vertical_slices"])
def test_propose_plan_injects_vertical_tracer_bullet_guidance(
    tmp_path: Path, topology: str
) -> None:
    store, run_id = _confirmed_run(tmp_path)
    snap = store.get_run(run_id)
    spec = snap["spec"]
    assert spec is not None
    prompts: list[str] = []

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        prompts.append(messages[0].content)
        return LLMResponse(
            content=json.dumps(
                _plan_payload(
                    spec_id=spec.spec_id,
                    spec_hash=spec.content_hash,
                    file_name="feature.py",
                    checks=[
                        {
                            "id": "CHK-SEAM",
                            "kind": CheckKind.COMMAND.value,
                            "description": "Exercise the seam",
                            "command": "python -m pytest -q",
                        }
                    ],
                )
            )
        )

    orch = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=responder),
        workspace_root=tmp_path,
    )
    plan = orch.propose_plan(run_id=run_id, topology=topology)
    assert len(prompts) == 1
    assert "Vertical tracer-bullet decomposition" in prompts[0]
    assert "tracer bullet" in prompts[0].lower()
    assert "depends_on" in prompts[0]
    assert "kind='command'" in prompts[0]
    assert plan.tasks[0].kind == TaskKind.IMPLEMENT


@pytest.mark.parametrize("topology", ["tdd_first", "vertical_slices"])
def test_propose_plan_vertical_topologies_reject_slices_without_command(
    tmp_path: Path, topology: str
) -> None:
    store, run_id = _confirmed_run(tmp_path)
    snap = store.get_run(run_id)
    spec = snap["spec"]
    assert spec is not None
    review_only = {
        "id": "CHK-REV",
        "kind": CheckKind.INDEPENDENT_REVIEW.value,
        "description": "Cold review only",
    }

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        return LLMResponse(
            content=json.dumps(
                _plan_payload(
                    spec_id=spec.spec_id,
                    spec_hash=spec.content_hash,
                    file_name="feature.py",
                    checks=[review_only],
                )
            )
        )

    orch = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=responder),
        workspace_root=tmp_path,
    )
    with pytest.raises(PlanValidationError, match="command"):
        orch.propose_plan(run_id=run_id, topology=topology)

    legacy = orch.propose_plan(run_id=run_id)
    assert legacy.tasks[0].acceptance_checks[0].kind == CheckKind.INDEPENDENT_REVIEW


def test_propose_plan_legacy_topologies_keep_their_guidance(tmp_path: Path) -> None:
    store, run_id = _confirmed_run(tmp_path)
    snap = store.get_run(run_id)
    spec = snap["spec"]
    assert spec is not None
    spec_only_prompts: list[str] = []
    modular_prompts: list[str] = []

    def _plan_for(*, with_design: bool) -> dict[str, Any]:
        task: dict[str, Any] = {
            "task_id": "T1",
            "title": "Implement the flag module",
            "objective": "Create feature.py",
            "requirement_refs": ["R1"],
            "write_scope": ["src/feature.py"],
            "acceptance_checks": [
                {
                    "id": "CHK-1",
                    "kind": CheckKind.COMMAND.value,
                    "description": "Smoke",
                    "command": "python -m pytest -q",
                }
            ],
        }
        payload: dict[str, Any] = {
            "plan_id": "plan-guidance",
            "spec_id": spec.spec_id,
            "spec_hash": spec.content_hash,
            "tasks": [task],
        }
        if with_design:
            task["design_refs"] = ["M1"]
        return payload

    def spec_only_responder(
        messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        del tools
        spec_only_prompts.append(messages[0].content)
        return LLMResponse(content=json.dumps(_plan_for(with_design=False)))

    orch = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=spec_only_responder),
        workspace_root=tmp_path,
    )
    orch.propose_plan(run_id=run_id, topology="minimal_patch")
    assert len(spec_only_prompts) == 1
    assert "Vertical tracer-bullet decomposition" not in spec_only_prompts[0]

    design = _contract_design(spec, shared_paths=[])
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

    def modular_responder(
        messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        del tools
        modular_prompts.append(messages[0].content)
        return LLMResponse(content=json.dumps(_plan_for(with_design=True)))

    modular_orch = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=modular_responder),
        workspace_root=tmp_path,
    )
    modular_orch.propose_plan(run_id=run_id, topology="modular_product")
    assert len(modular_prompts) == 1
    assert "Modular product decomposition" in modular_prompts[0]
    assert "Vertical tracer-bullet decomposition" not in modular_prompts[0]


def test_timeout_retry_is_visible_while_an_independent_task_completes(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    plan = _plan_payload(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        file_name="feature.py",
        checks=[
            {"id": "review", "kind": CheckKind.INDEPENDENT_REVIEW.value, "description": "Review"}
        ],
    )
    plan["tasks"].append({**plan["tasks"][0], "task_id": "T2", "write_scope": ["other.py"]})
    plan["tasks"].append(
        {**plan["tasks"][0], "task_id": "T3", "depends_on": ["T1"], "write_scope": ["child.py"]}
    )
    feature = _stage_responder(plan=plan, file_name="feature.py", review_passes=[True])
    other = _stage_responder(plan=plan, file_name="other.py", review_passes=[True])
    child = _stage_responder(plan=plan, file_name="child.py", review_passes=[True])
    timed_out = False
    during_retry = None
    after_recovery = None

    def respond(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        nonlocal timed_out, during_retry, after_recovery
        prompt = messages[1].content
        if "Task T1:" in prompt and not any(msg.role == "tool" for msg in messages):
            if not timed_out:
                timed_out = True
                raise TimeoutError("test call timed out")
            deadline = time.monotonic() + 5
            while not any(
                event.event_type == "task.accepted" and event.payload["task_id"] == "T2"
                for event in store.list_events(run_id)
            ):
                assert time.monotonic() < deadline, "Independent task did not finish during retry"
                time.sleep(0.01)
            during_retry = build_run_projection(store, tmp_path, run_id)
        elif "Task T1:" in prompt:
            after_recovery = build_run_projection(store, tmp_path, run_id)
        if "Task T2:" in prompt or '"task_id":"T2"' in prompt:
            return other(messages, tools)
        if "Task T3:" in prompt or '"task_id":"T3"' in prompt:
            return child(messages, tools)
        return feature(messages, tools)

    orch = OrchestrateStage(
        store=store, provider=MockLLMProvider(responder=respond), workspace_root=tmp_path
    )
    orch.propose_plan(run_id=run_id, auto_approve=True)
    assert orch.run_execution(run_id=run_id)["status"] == RunStatus.COMPLETED

    assert during_retry is not None
    active = {node["task_id"]: node for node in during_retry["dag"]["nodes"]}
    assert active["T1"]["status"] == "running"
    assert active["T1"]["provider_retry"]["phase"] == "started"
    assert active["T2"]["status"] == "accepted"
    assert active["T3"]["status"] == "waiting"
    assert active["T3"]["blocked_by"] == []
    assert during_retry["progress"]["accepted_tasks"] == 1
    assert during_retry["hitl_banner"] is None

    assert after_recovery is not None
    running = after_recovery["dag"]["nodes"][0]
    assert running["status"] == "running"
    assert running["provider_retry"] is None
    assert running["provider_retry_history"][-1]["phase"] == "recovered"

    final = build_run_projection(store, tmp_path, run_id)
    task = final["dag"]["nodes"][0]
    assert task["status"] == "accepted"
    assert task["attempt"] == 1
    assert task["provider_retry"] is None
    history = task["provider_retry_history"]
    assert [event["phase"] for event in history] == ["started", "recovered"]
    assert history[0]["retry_id"] == history[1]["retry_id"]
    assert history[0]["event_seq"] < history[1]["event_seq"]


def test_failed_timeout_retry_blocks_only_its_dependency_chain(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    plan = _plan_payload(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        file_name="feature.py",
        max_cycles=1,
        checks=[
            {"id": "review", "kind": CheckKind.INDEPENDENT_REVIEW.value, "description": "Review"}
        ],
    )
    root = plan["tasks"][0]
    root["deliverables"] = ["feature.py"]
    plan["tasks"].extend(
        {
            **root,
            "task_id": task_id,
            "depends_on": deps,
            "write_scope": [path],
            "deliverables": [path],
        }
        for task_id, deps, path in [
            ("T2", [], "other.py"),
            ("T3", ["T1", "T2"], "child.py"),
            ("T4", ["T3"], "release.py"),
        ]
    )
    other = _stage_responder(plan=plan, file_name="other.py", review_passes=[True])
    while_blocked = None

    def respond(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        nonlocal while_blocked
        prompt = messages[1].content
        if "Task T1:" in prompt:
            raise TimeoutError("flag service timed out")
        if "Task T2:" in prompt and not any(msg.role == "tool" for msg in messages):
            deadline = time.monotonic() + 5
            while not any(
                event.event_type == "task.blocked" for event in store.list_events(run_id)
            ):
                assert time.monotonic() < deadline, "Expected T1 to reach its task cycle limit"
                time.sleep(0.01)
            while_blocked = build_run_projection(store, tmp_path, run_id)
        return other(messages, tools)

    orch = OrchestrateStage(
        store=store, provider=MockLLMProvider(responder=respond), workspace_root=tmp_path
    )
    orch.propose_plan(run_id=run_id, auto_approve=True)
    snap = orch.run_execution(run_id=run_id)
    assert snap["status"] == RunStatus.AWAITING_HUMAN

    assert while_blocked is not None
    nodes = {node["task_id"]: node for node in while_blocked["dag"]["nodes"]}
    assert nodes["T1"]["status"] == "blocked"
    assert nodes["T1"]["provider_retry"] is None
    assert [row["phase"] for row in nodes["T1"]["provider_retry_history"]] == ["started", "failed"]
    assert nodes["T2"]["status"] == "running"
    assert nodes["T2"]["blocked_by"] == []
    assert nodes["T3"]["blocked_by"] == ["T1"]
    assert nodes["T4"]["blocked_by"] == ["T1"]
    banner = while_blocked["hitl_banner"]
    assert banner["task_id"] == "T1"
    assert banner["affected_task_ids"] == ["T1", "T3", "T4"]
    assert "flag service timed out" in banner["detail"]
    assert "retry" in banner["actions"]
    assert {s["id"]: s["status"] for s in while_blocked["stage_pipeline"]}["execute"] == "active"
    assert {(e["from"], e["to"]) for e in while_blocked["dag"]["edges"] if e["blocked_edge"]} == {
        ("T1", "T3"),
        ("T3", "T4"),
    }

    final = build_run_projection(store, tmp_path, run_id)
    assert final["progress"]["accepted_tasks"] == 1
    assert {s["id"]: s["status"] for s in final["stage_pipeline"]}["execute"] == "blocked"
    orch.retry_blocked_task(
        operation_id="retry-t1",
        run_id=run_id,
        task_id="T1",
        expected_state_revision=snap["state_revision"],
    )
    after_manual_retry = build_run_projection(store, tmp_path, run_id)
    nodes = {node["task_id"]: node for node in after_manual_retry["dag"]["nodes"]}
    assert nodes["T1"]["status"] == "ready"
    assert "flag service timed out" in nodes["T1"]["last_error"]
    assert nodes["T1"]["diagnostic_output"] == "就绪待调度"
    assert nodes["T1"]["provider_retry"] is None
    assert nodes["T3"]["blocked_by"] == []
    # A ready task on an awaiting run now surfaces the resume banner (Issue #54).
    assert after_manual_retry["hitl_banner"]["kind"] == "ready_to_resume"
    assert after_manual_retry["hitl_banner"]["actions"] == ["retry_workflow"]


def _approve_plan(
    store: RunStore,
    run_id: str,
    plan: TaskPlan,
    *,
    design: DesignSpec | None = None,
) -> None:
    """Persist and approve a hand-built plan (optionally pinning an approved design)."""
    if design is not None:
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


def _contract_design(spec: GoalSpec, *, shared_paths: list[str]) -> DesignSpec:
    return DesignSpec.model_validate(
        {
            "design_id": "design-diff-gate",
            "spec_id": spec.spec_id,
            "spec_hash": spec.content_hash,
            "architecture_summary": "Shared contracts pin cross-task boundaries.",
            "shared_contract_paths": shared_paths,
            "execution_strategy": ExecutionStrategy.CONTRACT_FIRST_PARALLEL.value,
            "strategy_rationale": "Wave 0 contract task locks shared paths before implement.",
            "modules": [
                {
                    "id": "M1",
                    "name": "Core",
                    "responsibility": "Feature flag core",
                    "requirement_refs": ["R1"],
                    "target_paths": ["src/"],
                }
            ],
            "status": SpecStatus.CONFIRMED.value,
        }
    )


def _single_task_plan(
    *,
    spec: GoalSpec,
    write_scope: list[str],
    checks: list[dict[str, Any]],
    max_cycles: int = 3,
    kind: TaskKind = TaskKind.IMPLEMENT,
) -> TaskPlan:
    return TaskPlan.model_validate(
        {
            "plan_id": "plan-diff-gate",
            "spec_id": spec.spec_id,
            "spec_hash": spec.content_hash,
            "tasks": [
                {
                    "task_id": "T1",
                    "title": "Implement the flag module",
                    "objective": "Implement the assigned flag module",
                    "requirement_refs": ["R1"],
                    "kind": kind.value,
                    "write_scope": write_scope,
                    "max_cycles": max_cycles,
                    "acceptance_checks": checks,
                }
            ],
        }
    )


def _marker_command_check() -> dict[str, Any]:
    return {
        "id": "CHK-1",
        "kind": CheckKind.COMMAND.value,
        "description": "Write a marker proving acceptance checks ran",
        "command": (
            "python3 -c 'from pathlib import Path; "
            'Path("check-ran.txt").write_text("ran")\''
        ),
    }


def _bash_action(command: str) -> Callable[[Path], ToolCall]:
    return lambda sandbox: ToolCall(  # noqa: ARG005 - action bound to one sandbox
        id="bash-1",
        name="run_bash",
        arguments={"command": command},
    )


def _write_action(file_name: str, content: str) -> Callable[[Path], ToolCall]:
    return lambda sandbox: ToolCall(
        id="write-1",
        name="write_file",
        arguments={"path": str(sandbox / file_name), "content": content},
    )


def _scope_gate_responder(
    actions: Sequence[Callable[[Path], ToolCall]],
) -> tuple[Responder, dict[str, int]]:
    """Replay one worker action per attempt; the last action repeats if needed.

    Review verdicts always pass so a requeue can only come from the Diff Gate.
    """
    state = {"attempts": 0, "reviews": 0}

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        system_message = messages[0].content
        user_message = messages[1].content if len(messages) > 1 else ""
        if "Independent Verifier" in system_message:
            state["reviews"] += 1
            return LLMResponse(
                content=json.dumps(
                    {
                        "task_id": "T1",
                        "attempt": 1,
                        "passed": True,
                        "summary": "Scope-clean implementation verified.",
                        "findings": [],
                    }
                )
            )
        if "Execution Engineer" not in system_message:
            raise AssertionError(f"Unexpected system prompt: {system_message[:80]}")
        match = re.search(r"Sandbox workspace root: (.+)", user_message)
        assert match is not None
        sandbox = Path(match.group(1).strip())
        if not any(msg.role == "tool" for msg in messages):
            action = actions[min(state["attempts"], len(actions) - 1)]
            state["attempts"] += 1
            return LLMResponse(content="acting", tool_calls=(action(sandbox),))
        return LLMResponse(content="finished")

    return responder, state


def _gate_rejections(store: RunStore, run_id: str) -> list[Any]:
    return [e for e in store.list_events(run_id) if e.event_type == "task.diff_gate_rejected"]


class _DirectWriteWorker:
    """Deterministic worker driver: writes prepared files per task and completes."""

    name = "direct"
    supports_steer = False

    def __init__(self, files: dict[str, dict[str, str]]) -> None:
        self._files = files
        self.prompts: dict[str, str] = {}

    def steer(self, guidance: str, *, source: str) -> bool:
        del guidance, source
        return False

    def cancel(self, *, reason: str) -> None:
        del reason

    def run_turn(
        self,
        request: DriverTurnRequest,
        *,
        on_event: Any = None,
        on_handle: Any = None,
    ) -> TurnOutcome:
        del on_event, on_handle
        if request.task_id is not None:
            self.prompts[request.task_id] = request.user_prompt
        for rel, content in self._files[request.task_id].items():
            target = request.workspace_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return TurnOutcome(
            completed=True,
            stop_reason="final_response",
            final_content="done",
            steps_taken=1,
            driver_name="direct",
        )


def _capturing_reviewer(prompts: dict[str, str]) -> MockLLMProvider:
    class CapturingReviewer(MockLLMProvider):
        def complete(self, *args: Any, **kwargs: Any) -> LLMResponse:
            del args
            user_prompt = kwargs["messages"][1].content
            task_id = re.search(r'"task_id":"(T\d+)"', user_prompt)
            assert task_id is not None
            prompts[task_id.group(1)] = user_prompt
            return LLMResponse(
                content=json.dumps(
                    {
                        "task_id": task_id.group(1),
                        "attempt": 1,
                        "passed": True,
                        "summary": "ok",
                        "findings": [],
                    }
                )
            )

    return CapturingReviewer()


def test_diff_gate_rejects_tracked_edit_outside_write_scope_before_checks(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    notes = tmp_path / "docs" / "notes.md"
    notes.parent.mkdir()
    notes.write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "docs/notes.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "docs"], cwd=tmp_path, check=True, capture_output=True)

    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    plan = _single_task_plan(
        spec=spec,
        write_scope=["feature.py"],
        max_cycles=2,
        checks=[
            _marker_command_check(),
            {
                "id": "CHK-REV-1",
                "kind": CheckKind.INDEPENDENT_REVIEW.value,
                "description": "Cold-start independent review",
            },
        ],
    )
    _approve_plan(store, run_id, plan)
    responder, state = _scope_gate_responder(
        [
            _bash_action("printf 'tampered\\n' >> docs/notes.md"),
            # The sandbox is retained, so the retry repairs the stray edit in place.
            _bash_action("git checkout -- docs/notes.md && printf 'FLAG = True\\n' > feature.py"),
        ]
    )
    orch = OrchestrateStage(
        store=store, provider=MockLLMProvider(responder=responder), workspace_root=tmp_path
    )
    snap = orch.run_execution(run_id=run_id, strict_scope=True)

    assert snap["status"] == RunStatus.COMPLETED
    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.ACCEPTED
    assert task.attempt == 2
    assert state["attempts"] == 2
    assert state["reviews"] == 1

    events = store.list_events(run_id)
    rejections = [e for e in events if e.event_type == "task.diff_gate_rejected"]
    assert len(rejections) == 1
    payload = rejections[0].payload
    assert payload["task_id"] == "T1"
    assert payload["attempt"] == 1
    assert payload["violating_paths"] == ["docs/notes.md"]
    assert payload["allowed_write_scope"] == ["feature.py"]
    assert payload["forbidden_globs"] == []

    # Gate rejection precedes every check and review event; the retry is the only review.
    check_events = [e for e in events if e.event_type == "check.command_completed"]
    assert check_events
    assert all(e.event_seq > rejections[0].event_seq for e in check_events)
    review_events = [e for e in events if e.event_type == "review.verdict_recorded"]
    assert [e.payload["attempt"] for e in review_events] == [2]
    assert len([e for e in events if e.event_type == "task.requeued"]) == 1

    attempt_dir = tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "T1" / "attempt-1"
    assert (attempt_dir / "diff.patch").is_file()
    assert (attempt_dir / "MANIFEST.md").is_file()
    assert not list(attempt_dir.glob("check-*.log"))
    assert not list(attempt_dir.glob("verdict-cycle-*.json"))

    overlay = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay"
    assert (overlay / "feature.py").is_file()
    assert not (overlay / "docs" / "notes.md").exists()
    assert notes.read_text(encoding="utf-8") == "original\n"


def test_diff_gate_rejects_untracked_file_outside_write_scope(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    plan = _single_task_plan(
        spec=spec,
        write_scope=["feature.py"],
        max_cycles=1,
        checks=[_marker_command_check()],
    )
    _approve_plan(store, run_id, plan)
    responder, state = _scope_gate_responder(
        [_bash_action("mkdir -p stray && printf 'x\\n' > stray/notes.txt")]
    )
    orch = OrchestrateStage(
        store=store, provider=MockLLMProvider(responder=responder), workspace_root=tmp_path
    )
    snap = orch.run_execution(run_id=run_id, strict_scope=True)

    assert snap["status"] == RunStatus.AWAITING_HUMAN
    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.BLOCKED
    assert task.review_cycle == 1
    assert task.last_error is not None
    assert "Diff Gate" in task.last_error
    assert "stray/notes.txt" in task.last_error
    assert "feature.py" in task.last_error
    assert state["reviews"] == 0

    rejections = _gate_rejections(store, run_id)
    assert len(rejections) == 1
    payload = rejections[0].payload
    assert payload["violating_paths"] == ["stray/notes.txt"]
    assert payload["allowed_write_scope"] == ["feature.py"]

    attempt_dir = tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "T1" / "attempt-1"
    assert not list(attempt_dir.glob("check-*.log"))
    overlay = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay"
    assert not overlay.exists() or not any(overlay.iterdir())


def test_diff_gate_rejects_forbidden_contract_path_even_when_write_scope_matches(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    design = _contract_design(spec, shared_paths=["src/contracts/"])
    plan = _single_task_plan(
        spec=spec,
        write_scope=["src/"],
        max_cycles=1,
        checks=[_marker_command_check()],
    )
    _approve_plan(store, run_id, plan, design=design)
    responder, state = _scope_gate_responder(
        [_bash_action("mkdir -p src/contracts && printf 'x\\n' > src/contracts/api.py")]
    )
    orch = OrchestrateStage(
        store=store, provider=MockLLMProvider(responder=responder), workspace_root=tmp_path
    )
    snap = orch.run_execution(run_id=run_id)

    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.BLOCKED
    assert task.last_error is not None
    assert "forbidden" in task.last_error
    assert "src/contracts/api.py" in task.last_error
    assert "src/contracts/" in task.last_error
    assert state["reviews"] == 0

    rejections = _gate_rejections(store, run_id)
    assert len(rejections) == 1
    payload = rejections[0].payload
    assert payload["violating_paths"] == ["src/contracts/api.py"]
    assert payload["allowed_write_scope"] == ["src/"]
    assert payload["forbidden_globs"] == ["src/contracts/"]
    assert payload["violations"][0]["reason"] == "forbidden"


def test_diff_gate_allows_contract_task_to_edit_shared_contract_paths(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    design = _contract_design(spec, shared_paths=["contracts/"])
    plan = _single_task_plan(
        spec=spec,
        write_scope=["contracts/"],
        kind=TaskKind.CONTRACT,
        checks=[
            {
                "id": "CHK-1",
                "kind": CheckKind.COMMAND.value,
                "description": "Contract file exists with pinned content",
                "command": (
                    "python3 -c 'from pathlib import Path; "
                    'assert Path("contracts/api.py").read_text() == "PINNED = 1\\n"\''
                ),
            }
        ],
    )
    _approve_plan(store, run_id, plan, design=design)
    responder, state = _scope_gate_responder([_write_action("contracts/api.py", "PINNED = 1\n")])
    orch = OrchestrateStage(
        store=store, provider=MockLLMProvider(responder=responder), workspace_root=tmp_path
    )
    snap = orch.run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.COMPLETED
    assert snap["plan"].tasks[0].status == TaskStatus.ACCEPTED
    assert state["reviews"] == 1
    assert _gate_rejections(store, run_id) == []
    overlay = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay"
    assert (overlay / "contracts" / "api.py").read_text(encoding="utf-8") == "PINNED = 1\n"


def test_diff_gate_passes_tracked_and_untracked_changes_inside_write_scope(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    app = tmp_path / "app.py"
    app.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "app"], cwd=tmp_path, check=True, capture_output=True)

    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    plan = _single_task_plan(
        spec=spec,
        write_scope=["app.py", "pkg/"],
        checks=[
            {
                "id": "CHK-1",
                "kind": CheckKind.COMMAND.value,
                "description": "Tracked edit and new module are both present",
                "command": (
                    "python3 -c 'from pathlib import Path; "
                    'assert "VALUE = 2" in Path("app.py").read_text(); '
                    'assert Path("pkg/util.py").is_file()\''
                ),
            }
        ],
    )
    _approve_plan(store, run_id, plan)
    responder, state = _scope_gate_responder(
        [
            _bash_action(
                "printf 'VALUE = 2\\n' > app.py && mkdir -p pkg && printf 'X = 1\\n' > pkg/util.py"
            )
        ]
    )
    orch = OrchestrateStage(
        store=store, provider=MockLLMProvider(responder=responder), workspace_root=tmp_path
    )
    snap = orch.run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.COMPLETED
    assert snap["plan"].tasks[0].status == TaskStatus.ACCEPTED
    assert state["reviews"] == 1
    assert _gate_rejections(store, run_id) == []
    overlay = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay"
    assert (overlay / "app.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (overlay / "pkg" / "util.py").read_text(encoding="utf-8") == "X = 1\n"
    # SIMULATION mode with apply_to_workspace=False never touches the host checkout.
    assert app.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_diff_gate_validates_non_git_sandbox_without_crashing(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Sample\n", encoding="utf-8")
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    plan = _single_task_plan(
        spec=spec,
        write_scope=["feature.py"],
        max_cycles=2,
        checks=[_marker_command_check()],
    )
    _approve_plan(store, run_id, plan)
    responder, state = _scope_gate_responder(
        [
            _bash_action("printf 'x\\n' > stray.py"),
            # The retained sandbox keeps stray.py on disk, so the retry removes it in place.
            _bash_action("rm stray.py && printf 'FLAG = True\\n' > feature.py"),
        ]
    )
    orch = OrchestrateStage(
        store=store, provider=MockLLMProvider(responder=responder), workspace_root=tmp_path
    )
    snap = orch.run_execution(run_id=run_id, strict_scope=True)

    assert snap["status"] == RunStatus.COMPLETED
    assert snap["plan"].tasks[0].status == TaskStatus.ACCEPTED
    assert snap["plan"].tasks[0].attempt == 2
    assert state["attempts"] == 2
    assert state["reviews"] == 1

    rejections = _gate_rejections(store, run_id)
    assert len(rejections) == 1
    assert rejections[0].payload["violating_paths"] == ["stray.py"]
    overlay = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay"
    assert (overlay / "feature.py").is_file()
    assert not (overlay / "stray.py").exists()


def test_diff_gate_rejects_deleted_and_renamed_paths_outside_write_scope(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("original\n", encoding="utf-8")
    (docs / "legacy.txt").write_text("legacy\n", encoding="utf-8")
    subprocess.run(["git", "add", "docs/"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "docs"], cwd=tmp_path, check=True, capture_output=True)

    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    plan = _single_task_plan(
        spec=spec,
        write_scope=["feature.py"],
        max_cycles=1,
        checks=[_marker_command_check()],
    )
    _approve_plan(store, run_id, plan)
    responder, state = _scope_gate_responder(
        [_bash_action("mv docs/notes.md docs/notes-renamed.md && rm docs/legacy.txt")]
    )
    orch = OrchestrateStage(
        store=store, provider=MockLLMProvider(responder=responder), workspace_root=tmp_path
    )
    snap = orch.run_execution(run_id=run_id, strict_scope=True)

    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.BLOCKED
    assert task.last_error is not None
    for path in ("docs/legacy.txt", "docs/notes-renamed.md", "docs/notes.md"):
        assert path in task.last_error
    assert state["reviews"] == 0

    rejections = _gate_rejections(store, run_id)
    assert len(rejections) == 1
    assert rejections[0].payload["violating_paths"] == [
        "docs/legacy.txt",
        "docs/notes-renamed.md",
        "docs/notes.md",
    ]
    attempt_dir = tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "T1" / "attempt-1"
    assert not list(attempt_dir.glob("check-*.log"))
    # The host checkout keeps both documents intact: nothing is auto-reverted or promoted.
    assert (docs / "notes.md").is_file()
    assert (docs / "legacy.txt").is_file()
    assert not (docs / "notes-renamed.md").exists()


def test_single_task_adjacent_edit_is_advisory_and_promoted(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    subprocess.run(["git", "add", "pkg/__init__.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "pkg"], cwd=tmp_path, check=True, capture_output=True)

    store, run_id = _confirmed_run(tmp_path, mode=RunMode.EXECUTION)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    plan = _single_task_plan(
        spec=spec,
        write_scope=["pkg/flag.py"],
        checks=[
            {
                "id": "CHK-1",
                "kind": CheckKind.COMMAND.value,
                "description": "Adjacent re-export makes the flag importable",
                "command": (
                    "python3 -c 'import pkg; assert pkg.FLAG is True; "
                    "import pkg.flag; assert pkg.flag.FLAG is True'"
                ),
            }
        ],
    )
    _approve_plan(store, run_id, plan)
    prompts: dict[str, str] = {}
    worker = _DirectWriteWorker(
        {
            "T1": {
                "pkg/flag.py": "FLAG = True\n",
                "pkg/__init__.py": "from pkg.flag import FLAG\n",
            }
        }
    )
    stage = OrchestrateStage(
        store=store,
        workspace_root=tmp_path,
        provider=_capturing_reviewer(prompts),
        execution_drivers=ExecutionDriversConfig(worker_driver="direct"),
        driver_registry=DriverRegistry({"direct": worker}),
    )
    out = stage.run_execution(run_id=run_id, apply_to_workspace=True)

    assert out["status"] == RunStatus.COMPLETED
    assert out["plan"].tasks[0].status == TaskStatus.ACCEPTED
    assert _gate_rejections(store, run_id) == []
    overlay = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay"
    assert (overlay / "pkg" / "flag.py").is_file()
    assert (overlay / "pkg" / "__init__.py").read_text(encoding="utf-8") == (
        "from pkg.flag import FLAG\n"
    )
    # The adjacent non-forbidden edit is promoted to the real workspace on acceptance.
    assert (pkg / "__init__.py").read_text(encoding="utf-8") == "from pkg.flag import FLAG\n"
    assert (pkg / "flag.py").is_file()
    assert "pkg/__init__.py" in prompts["T1"]
    assert "Authorized out-of-declared-scope adjacent edits" in prompts["T1"]
    assert "pkg/__init__.py" in prompts["T1"].split("Authorized out-of-declared-scope")[1]
    assert "no concurrent sibling task shares this run wave" in worker.prompts["T1"]


def test_concurrent_wave_enforces_strict_scope_and_sibling_forbidden_globs(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(
        tmp_path, budget_policy=BudgetPolicy(max_concurrent_tasks=2)
    )
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    plan = _parallel_plan(spec, scopes={"T1": ["t1.py"], "T2": ["t2.py"]})
    plan = plan.model_copy(
        update={"tasks": [t.model_copy(update={"max_cycles": 1}) for t in plan.tasks]}
    )
    _approve_plan(store, run_id, plan)
    worker = _DirectWriteWorker(
        {
            "T1": {"t1.py": "T1 = True\n", "pkg/__init__.py": "T1 = True\n", "t2.py": "CLOBBER\n"},
            "T2": {"t2.py": "T2 = True\n"},
        }
    )
    stage = OrchestrateStage(
        store=store,
        workspace_root=tmp_path,
        provider=MockLLMProvider(responder=_passing_reviewer),
        execution_drivers=ExecutionDriversConfig(worker_driver="direct"),
        driver_registry=DriverRegistry({"direct": worker}),
    )
    out = stage.run_execution(run_id=run_id)

    tasks = {task.task_id: task for task in out["plan"].tasks}
    assert tasks["T1"].status == TaskStatus.BLOCKED
    assert tasks["T2"].status == TaskStatus.ACCEPTED
    rejections = _gate_rejections(store, run_id)
    assert len(rejections) == 1
    payload = rejections[0].payload
    assert payload["task_id"] == "T1"
    assert payload["strict_scope"] is True
    assert payload["advisory_out_of_scope"] == []
    assert payload["forbidden_globs"] == ["t2.py"]
    assert payload["violating_paths"] == ["pkg/__init__.py", "t2.py"]
    assert {
        (violation["path"], violation["reason"]) for violation in payload["violations"]
    } == {
        ("pkg/__init__.py", "outside_write_scope"),
        ("t2.py", "forbidden"),
    }
    overlay = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay"
    assert (overlay / "t2.py").read_text(encoding="utf-8") == "T2 = True\n"
    assert not (overlay / "t1.py").exists()
    assert not (overlay / "pkg").exists()


def test_ephemeral_junk_cleaned_before_diff_gate_is_logged_and_not_promoted(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    plan = _single_task_plan(
        spec=spec,
        write_scope=["feature.py"],
        checks=[
            {
                "id": "CHK-1",
                "kind": CheckKind.COMMAND.value,
                "description": "Feature flag module imports",
                "command": "python3 -c 'import feature; assert feature.FLAG is True'",
            }
        ],
    )
    _approve_plan(store, run_id, plan)
    prompts: dict[str, str] = {}
    worker = _DirectWriteWorker(
        {
            "T1": {
                "feature.py": "FLAG = True\n",
                "feature.py.backup": "stale\n",
                "scratch.tmp": "stale\n",
                "pkg/__pycache__/mod.cpython-313.pyc": "bytecode",
            }
        }
    )
    stage = OrchestrateStage(
        store=store,
        workspace_root=tmp_path,
        provider=_capturing_reviewer(prompts),
        execution_drivers=ExecutionDriversConfig(worker_driver="direct"),
        driver_registry=DriverRegistry({"direct": worker}),
    )
    out = stage.run_execution(run_id=run_id)

    assert out["status"] == RunStatus.COMPLETED
    assert out["plan"].tasks[0].status == TaskStatus.ACCEPTED
    assert _gate_rejections(store, run_id) == []
    cleaned = [
        e
        for e in store.list_events(run_id)
        if e.event_type == "task.ephemeral_artifacts_cleaned"
    ]
    assert len(cleaned) == 1
    assert cleaned[0].payload["removed_paths"] == [
        "feature.py.backup",
        "pkg/__pycache__/mod.cpython-313.pyc",
        "scratch.tmp",
    ]
    overlay = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay"
    assert (overlay / "feature.py").is_file()
    assert not (overlay / "feature.py.backup").exists()
    assert not (overlay / "scratch.tmp").exists()
    assert not (overlay / "pkg").exists()
    prompt = prompts["T1"]
    assert "feature.py" in prompt
    assert "feature.py.backup" not in prompt
    assert "scratch.tmp" not in prompt


@pytest.mark.parametrize(
    ("serial_execution", "stray_source"), [(False, False), (True, False), (False, True)]
)
def test_worker_smoke_artifacts_are_cleaned_before_scope_checks_and_promotion(
    tmp_path: Path, serial_execution: bool, stray_source: bool
) -> None:
    _init_repo(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/existing.db").write_text("existing", encoding="utf-8")
    (tmp_path / "smoke.sh").write_text(
        "set -eu\n"
        "mkdir -p data/uploads\n"
        "test -f data/existing.db\n"
        "touch data/blacknote.db data/blacknote.db-wal data/uploads/sample.png\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "data/existing.db", "smoke.sh"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "smoke fixture"], cwd=tmp_path, check=True)
    store, run_id = _confirmed_run(
        tmp_path, budget_policy=BudgetPolicy(max_concurrent_tasks=2), mode=RunMode.EXECUTION
    )
    plan = _parallel_plan(
        store.get_run(run_id)["spec"], scopes={"T1": ["feature.db"], "T2": ["peer.py"]}
    )
    plan = plan.model_copy(
        update={"tasks": [task.model_copy(update={"max_cycles": 1}) for task in plan.tasks]}
    )
    _approve_plan(store, run_id, plan)

    class SmokeWorker(_DirectWriteWorker):
        def run_turn(self, request: DriverTurnRequest, **kwargs: Any) -> TurnOutcome:
            if request.task_id == "T1":
                assert request.strict_scope is (not serial_execution)
                subprocess.run(["sh", "smoke.sh"], cwd=request.workspace_path, check=True)
            return super().run_turn(request, **kwargs)

    files = {"T1": {"feature.db": "feature"}, "T2": {"peer.py": "PEER = True\n"}}
    if stray_source:
        files["T1"]["data/hack.py"] = "ROGUE = True\n"
    stage = OrchestrateStage(
        store=store,
        workspace_root=tmp_path,
        provider=_capturing_reviewer({}),
        execution_drivers=ExecutionDriversConfig(worker_driver="direct"),
        driver_registry=DriverRegistry({"direct": SmokeWorker(files)}),
    )
    out = stage.run_execution(
        run_id=run_id, serial_execution=serial_execution, apply_to_workspace=True
    )

    tasks = {task.task_id: task for task in out["plan"].tasks}
    assert tasks["T2"].status == TaskStatus.ACCEPTED
    cleaned = [
        event.payload["removed_paths"]
        for event in store.list_events(run_id)
        if event.event_type == "task.ephemeral_artifacts_cleaned"
    ]
    assert cleaned == [["data/blacknote.db", "data/blacknote.db-wal", "data/uploads/sample.png"]]
    assert (tmp_path / "data/existing.db").read_text(encoding="utf-8") == "existing"
    overlay = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay"
    assert (overlay / "peer.py").is_file()
    assert not (overlay / "data").exists()
    assert list((tmp_path / "data").iterdir()) == [tmp_path / "data/existing.db"]
    if stray_source:
        assert tasks["T1"].status == TaskStatus.BLOCKED
        rejection, = _gate_rejections(store, run_id)
        assert rejection.payload["strict_scope"] is True
        assert rejection.payload["violating_paths"] == ["data/hack.py"]
        assert not (overlay / "feature.db").exists()
    else:
        assert out["status"] == RunStatus.COMPLETED
        assert tasks["T1"].status == TaskStatus.ACCEPTED
        assert _gate_rejections(store, run_id) == []
        assert (overlay / "feature.db").read_text(encoding="utf-8") == "feature"
        assert (tmp_path / "feature.db").read_text(encoding="utf-8") == "feature"


def test_retained_sandbox_retry_still_cleans_prior_stopped_early_runtime_artifacts(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(
        tmp_path, budget_policy=BudgetPolicy(max_concurrent_tasks=2), mode=RunMode.EXECUTION
    )
    plan = _single_task_plan(
        spec=store.get_run(run_id)["spec"],
        write_scope=["feature.py"],
        checks=[{"id": "C1", "kind": "independent_review", "description": "Review"}],
        max_cycles=2,
    )
    _approve_plan(store, run_id, plan)
    attempts = 0

    class _PauseThenFinishWorker:
        name = "pause-then-finish"
        supports_steer = False

        def run_turn(self, request: DriverTurnRequest, **kwargs: Any) -> TurnOutcome:
            nonlocal attempts
            del kwargs
            attempts += 1
            ws = request.workspace_path
            if attempts == 1:
                (ws / "data").mkdir(parents=True, exist_ok=True)
                (ws / "data/blacknote.db").write_text("temp db", encoding="utf-8")
                return TurnOutcome(False, "budget_exhausted", "", 1)
            assert (ws / "data/blacknote.db").is_file()
            (ws / "feature.py").write_text("OK = True\n", encoding="utf-8")
            return TurnOutcome(True, "completed", "done", 1)

    stage = OrchestrateStage(
        store=store,
        workspace_root=tmp_path,
        provider=_capturing_reviewer({}),
        execution_drivers=ExecutionDriversConfig(worker_driver="pause-then-finish"),
        driver_registry=DriverRegistry({"pause-then-finish": _PauseThenFinishWorker()}),
    )
    first = stage.run_execution(run_id=run_id, strict_scope=True, apply_to_workspace=True)
    assert first["plan"].tasks[0].status == TaskStatus.ACCEPTED
    assert _gate_rejections(store, run_id) == []
    assert not (tmp_path / "data/blacknote.db").exists()
    assert (tmp_path / "feature.py").read_text(encoding="utf-8") == "OK = True\n"


AttemptAction = Callable[[int, Path, str], None]


class _AttemptWorker:
    """Deterministic driver replaying one scripted action per leased attempt."""

    name = "attempt-scripted"
    supports_steer = False

    def __init__(self, script: Sequence[AttemptAction]) -> None:
        self._script = list(script)
        self.sandboxes: list[Path] = []
        self.prompts: list[str] = []

    def steer(self, guidance: str, *, source: str) -> bool:
        del guidance, source
        return False

    def cancel(self, *, reason: str) -> None:
        del reason

    def run_turn(
        self,
        request: DriverTurnRequest,
        *,
        on_event: Any = None,
        on_handle: Any = None,
    ) -> TurnOutcome:
        del on_event, on_handle
        self.sandboxes.append(request.workspace_path)
        self.prompts.append(request.user_prompt)
        action = self._script[min(len(self.sandboxes) - 1, len(self._script) - 1)]
        action(len(self.sandboxes), request.workspace_path, request.user_prompt)
        return TurnOutcome(
            completed=True,
            stop_reason="final_response",
            final_content="done",
            steps_taken=1,
            driver_name=self.name,
        )


def _retained_stage(
    store: RunStore,
    tmp_path: Path,
    worker: _AttemptWorker,
    *,
    provider: MockLLMProvider | None = None,
) -> OrchestrateStage:
    return OrchestrateStage(
        store=store,
        provider=provider or _capturing_reviewer({}),
        workspace_root=tmp_path,
        execution_drivers=ExecutionDriversConfig(worker_driver=worker.name),
        driver_registry=DriverRegistry({worker.name: worker}),
    )


def _review_script(
    verdicts: Sequence[dict[str, Any]],
) -> tuple[MockLLMProvider, dict[str, Any]]:
    """Reviewer provider replaying scripted verdicts while counting LLM calls."""
    state: dict[str, Any] = {"calls": 0, "prompts": []}

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        assert "Independent Verifier" in messages[0].content
        state["calls"] += 1
        state["prompts"].append(messages[1].content if len(messages) > 1 else "")
        index = min(state["calls"] - 1, len(verdicts) - 1)
        return LLMResponse(content=json.dumps(verdicts[index]))

    return MockLLMProvider(responder=responder), state


def _empty_dir(path: Path) -> bool:
    return not path.exists() or not any(path.iterdir())


def test_retained_sandbox_enables_in_place_repair_on_diff_gate_retry(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["feature.py"],
            max_cycles=3,
            checks=[
                {
                    "id": "CHK-1",
                    "kind": CheckKind.COMMAND.value,
                    "description": "Feature module exposes the new flag",
                    "command": "python3 -c 'import feature; assert feature.FLAG == 2'",
                }
            ],
        ),
    )

    def repair(attempt: int, ws: Path, prompt: str) -> None:
        if attempt == 1:
            (ws / "feature.py").write_text("FLAG = 1\n", encoding="utf-8")
            (ws / "scratch.py").write_text("stray\n", encoding="utf-8")
            return
        assert (ws / "feature.py").read_text(encoding="utf-8") == "FLAG = 1\n"
        assert (ws / "scratch.py").is_file()
        assert "IN-PLACE REPAIR" in prompt
        (ws / "scratch.py").unlink()
        (ws / "feature.py").write_text("FLAG = 2\n", encoding="utf-8")

    worker = _AttemptWorker([repair])
    snap = _retained_stage(store, tmp_path, worker).run_execution(
        run_id=run_id, strict_scope=True
    )

    assert snap["status"] == RunStatus.COMPLETED
    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.ACCEPTED
    assert task.attempt == 2
    assert len(worker.sandboxes) == 2
    assert worker.sandboxes[0] == worker.sandboxes[1]
    rejections = _gate_rejections(store, run_id)
    assert [(e.payload["attempt"], e.payload["violating_paths"]) for e in rejections] == [
        (1, ["scratch.py"])
    ]
    assert _empty_dir(tmp_path / ".olympus" / "sandboxes")
    assert _empty_dir(tmp_path / ".olympus" / "sandbox_tmp")


def test_retained_sandbox_never_rebaselines_out_of_scope_violations(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["feature.py"],
            max_cycles=2,
            checks=[_marker_command_check()],
        ),
    )

    def stray(attempt: int, ws: Path, prompt: str) -> None:
        del attempt, prompt
        (ws / "feature.py").write_text("FLAG = True\n", encoding="utf-8")
        (ws / "scratch.py").write_text("stray\n", encoding="utf-8")

    def leave_stray(attempt: int, ws: Path, prompt: str) -> None:
        del attempt, prompt
        assert (ws / "scratch.py").is_file()

    worker = _AttemptWorker([stray, leave_stray])
    snap = _retained_stage(store, tmp_path, worker).run_execution(
        run_id=run_id, strict_scope=True
    )

    assert snap["status"] == RunStatus.AWAITING_HUMAN
    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.BLOCKED
    assert task.review_cycle == 2
    assert worker.sandboxes[0] == worker.sandboxes[1]
    assert [(e.payload["attempt"], e.payload["violating_paths"])
            for e in _gate_rejections(store, run_id)] == [
        (1, ["scratch.py"]),
        (2, ["scratch.py"]),
    ]
    assert worker.sandboxes[0].is_dir()


def test_command_check_failure_reuses_retained_sandbox_for_repair(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["feature.py"],
            max_cycles=3,
            checks=[
                {
                    "id": "CHK-1",
                    "kind": CheckKind.COMMAND.value,
                    "description": "Feature flag must be 2",
                    "command": "python3 -c 'import feature; assert feature.FLAG == 2'",
                }
            ],
        ),
    )

    def repair(attempt: int, ws: Path, prompt: str) -> None:
        del prompt
        if attempt == 1:
            (ws / "feature.py").write_text("FLAG = 1\n", encoding="utf-8")
            return
        assert (ws / "feature.py").read_text(encoding="utf-8") == "FLAG = 1\n"
        (ws / "feature.py").write_text("FLAG = 2  # repaired\n", encoding="utf-8")

    worker = _AttemptWorker([repair])
    snap = _retained_stage(store, tmp_path, worker).run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.COMPLETED
    assert snap["plan"].tasks[0].status == TaskStatus.ACCEPTED
    assert snap["plan"].tasks[0].attempt == 2
    assert worker.sandboxes[0] == worker.sandboxes[1]
    check_events = [
        e for e in store.list_events(run_id) if e.event_type == "check.command_completed"
    ]
    assert [e.payload["passed"] for e in check_events] == [False, True]


def test_argus_rejection_reuses_retained_sandbox_for_targeted_repair(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["feature.py"],
            max_cycles=3,
            checks=[
                {
                    "id": "CHK-REV",
                    "kind": CheckKind.INDEPENDENT_REVIEW.value,
                    "description": "Audit the flag implementation",
                }
            ],
        ),
    )
    review_passes = [False, True]

    def responder(
        messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        del tools
        assert "Independent Verifier" in messages[0].content
        passed = review_passes.pop(0)
        findings = [] if passed else [{
            "id": "F1",
            "axis": "spec",
            "severity": "important",
            "file_path": "feature.py",
            "line": 1,
            "requirement_ref": "R1",
            "repro_command": "python3 -c 'import feature; assert feature.AUDIT'",
            "message": "Missing audit flag",
        }]
        return LLMResponse(content=json.dumps({
            "task_id": "T1",
            "attempt": 1,
            "passed": passed,
            "summary": "ok" if passed else "Missing audit flag",
            "findings": findings,
        }))

    def repair(attempt: int, ws: Path, prompt: str) -> None:
        del prompt
        if attempt == 1:
            (ws / "feature.py").write_text("FLAG = True\n", encoding="utf-8")
            return
        assert (ws / "feature.py").read_text(encoding="utf-8") == "FLAG = True\n"
        (ws / "feature.py").write_text(
            "FLAG = True\nAUDIT = True\n", encoding="utf-8"
        )

    worker = _AttemptWorker([repair])
    snap = _retained_stage(
        store, tmp_path, worker, provider=MockLLMProvider(responder=responder)
    ).run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.COMPLETED
    assert snap["plan"].tasks[0].status == TaskStatus.ACCEPTED
    assert snap["plan"].tasks[0].attempt == 2
    assert worker.sandboxes[0] == worker.sandboxes[1]
    reviews = [e for e in store.list_events(run_id) if e.event_type == "review.verdict_recorded"]
    assert [e.payload["attempt"] for e in reviews] == [1, 2]


def test_unverified_exit_zero_and_standards_findings_never_block_acceptance(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["feature.py"],
            checks=[
                {
                    "id": "CHK-REV",
                    "kind": CheckKind.INDEPENDENT_REVIEW.value,
                    "description": "Audit the flag implementation",
                }
            ],
        ),
    )
    provider, state = _review_script(
        [
            {
                "task_id": "T1",
                "attempt": 1,
                "passed": False,
                "summary": "Three unproven claims",
                "findings": [
                    {
                        "id": "F-PROSE",
                        "axis": "spec",
                        "severity": "important",
                        "file_path": "feature.py",
                        "line": 1,
                        "message": "No overflow guard",
                    },
                    {
                        "id": "F-EXIT0",
                        "axis": "spec",
                        "severity": "important",
                        "file_path": "feature.py",
                        "line": 1,
                        "requirement_ref": "R1",
                        "repro_command": "python3 -c 'pass'",
                        "message": "Guard is missing",
                    },
                    {
                        "id": "F-STYLE",
                        "axis": "standards",
                        "severity": "important",
                        "file_path": "feature.py",
                        "line": 1,
                        "message": "Prefer double quotes",
                    },
                ],
            }
        ]
    )

    def write_flag(attempt: int, ws: Path, prompt: str) -> None:
        del attempt, prompt
        (ws / "feature.py").write_text("FLAG = True\n", encoding="utf-8")

    snap = _retained_stage(
        store, tmp_path, _AttemptWorker([write_flag]), provider=provider
    ).run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.COMPLETED
    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.ACCEPTED
    assert task.attempt == 1
    assert task.review_cycle == 0
    assert state["calls"] == 1
    events = store.list_events(run_id)
    assert not [e for e in events if e.event_type == "task.requeued"]

    verdict = json.loads(
        (
            tmp_path
            / ".olympus"
            / "runs"
            / run_id
            / "artifacts"
            / "T1"
            / "attempt-1"
            / "verdict-cycle-1.json"
        ).read_text(encoding="utf-8")
    )
    assert verdict["passed"] is True
    assert verdict["dropped_unverified_count"] == 2
    assert {finding["severity"] for finding in verdict["findings"]} == {"nit"}
    verified_events = [e for e in events if e.event_type == "review.verdict_verified"]
    assert len(verified_events) == 1
    payload = verified_events[0].payload
    assert payload["passed"] is True
    assert payload["verified_important"] == 0
    assert payload["dropped_unverified"] == 2
    assert payload["standards_count"] == 1


@pytest.mark.parametrize(
    ("command", "timeout_seconds"),
    [
        ("python3 -c 'def ('", None),
        ("olympus-missing-repro-binary-122", None),
        ("python3 -c 'import time; time.sleep(30)'", 0.3),
    ],
    ids=["syntax-error", "exit-127", "timeout"],
)
def test_broken_reviewer_probes_never_block_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    timeout_seconds: float | None,
) -> None:
    if timeout_seconds is not None:
        monkeypatch.setattr(
            orchestrate_module, "_REPRO_TIMEOUT_SECONDS", timeout_seconds
        )
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["feature.py"],
            checks=[
                {
                    "id": "CHK-REV",
                    "kind": CheckKind.INDEPENDENT_REVIEW.value,
                    "description": "Audit the flag implementation",
                }
            ],
        ),
    )
    provider, state = _review_script(
        [
            {
                "task_id": "T1",
                "attempt": 1,
                "passed": False,
                "summary": "Broken probe",
                "findings": [
                    {
                        "id": "F1",
                        "axis": "spec",
                        "severity": "important",
                        "file_path": "feature.py",
                        "line": 1,
                        "requirement_ref": "R1",
                        "repro_command": command,
                        "message": "Claimed defect",
                    }
                ],
            }
        ]
    )

    def write_flag(attempt: int, ws: Path, prompt: str) -> None:
        del attempt, prompt
        (ws / "feature.py").write_text("FLAG = True\n", encoding="utf-8")

    snap = _retained_stage(
        store, tmp_path, _AttemptWorker([write_flag]), provider=provider
    ).run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.COMPLETED
    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.ACCEPTED
    assert task.attempt == 1
    assert state["calls"] == 1
    events = store.list_events(run_id)
    assert not [e for e in events if e.event_type == "task.requeued"]
    payload = next(
        e.payload for e in events if e.event_type == "review.verdict_verified"
    )
    assert payload["verified_important"] == 0
    assert payload["dropped_unverified"] == 1


def test_proven_repro_blocks_persists_and_short_circuits_next_cycle(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    repro = "python3 -c 'import feature; assert feature.GUARD'"
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["feature.py"],
            max_cycles=4,
            checks=[
                {
                    "id": "CHK-REV",
                    "kind": CheckKind.INDEPENDENT_REVIEW.value,
                    "description": "Audit the flag implementation",
                }
            ],
        ),
    )
    provider, state = _review_script(
        [
            {
                "task_id": "T1",
                "attempt": 1,
                "passed": False,
                "summary": "Missing guard",
                "findings": [
                    {
                        "id": "F1",
                        "axis": "spec",
                        "severity": "important",
                        "file_path": "feature.py",
                        "line": 2,
                        "requirement_ref": "R1",
                        "repro_command": repro,
                        "message": "GUARD is absent",
                    }
                ],
            },
            {
                "task_id": "T1",
                "attempt": 1,
                "passed": True,
                "summary": "Guard added",
                "findings": [],
            },
        ]
    )

    def repair(attempt: int, ws: Path, prompt: str) -> None:
        if attempt == 1:
            (ws / "feature.py").write_text("FLAG = True\n", encoding="utf-8")
        elif attempt == 2:
            # Cycle 2 leaves the proven defect in place: Chronos must requeue with
            # zero reviewer LLM calls instead of asking Argus again.
            assert repro in prompt
            (ws / "feature.py").write_text("FLAG = True\n", encoding="utf-8")
        else:
            assert repro in prompt
            (ws / "feature.py").write_text("FLAG = True\nGUARD = True\n", encoding="utf-8")

    snap = _retained_stage(
        store, tmp_path, _AttemptWorker([repair]), provider=provider
    ).run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.COMPLETED
    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.ACCEPTED
    assert task.attempt == 3
    assert task.review_cycle == 2
    assert state["calls"] == 2
    assert '"requirement_refs":["R1"]' in state["prompts"][0]
    assert "NOW PASS" in state["prompts"][1]

    events = store.list_events(run_id)
    rechecks = [e for e in events if e.event_type == "review.repro_rechecked"]
    assert [e.payload["passed"] for e in rechecks] == [False, True]
    assert [row["command"] for row in rechecks[0].payload["results"]] == [repro]
    assert rechecks[0].payload["short_circuited"] is True
    assert rechecks[0].payload["llm_calls"] == 0
    assert rechecks[1].payload["short_circuited"] is False
    recorded = [e for e in events if e.event_type == "review.verdict_recorded"]
    assert [e.payload["attempt"] for e in recorded] == [1, 3]


def test_verification_probe_side_effects_cannot_corrupt_worker_edits(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    repro = (
        "python3 -c \"from pathlib import Path; "
        "Path('helper.py').write_text('CORRUPTED = True\\n'); "
        "Path('probe-junk.txt').write_text('junk')\" ; "
        "python3 -c 'import feature; assert feature.GUARD'"
    )
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["feature.py", "helper.py"],
            max_cycles=3,
            checks=[
                {
                    "id": "CHK-REV",
                    "kind": CheckKind.INDEPENDENT_REVIEW.value,
                    "description": "Audit the flag implementation",
                }
            ],
        ),
    )
    provider, state = _review_script(
        [
            {
                "task_id": "T1",
                "attempt": 1,
                "passed": False,
                "summary": "Missing guard",
                "findings": [
                    {
                        "id": "F1",
                        "axis": "spec",
                        "severity": "important",
                        "file_path": "feature.py",
                        "line": 2,
                        "requirement_ref": "R1",
                        "repro_command": repro,
                        "message": "GUARD is absent",
                    }
                ],
            },
            {
                "task_id": "T1",
                "attempt": 1,
                "passed": True,
                "summary": "Guard added",
                "findings": [],
            },
        ]
    )

    def repair(attempt: int, ws: Path, prompt: str) -> None:
        del prompt
        if attempt == 1:
            (ws / "helper.py").write_text("HELPER = True\n", encoding="utf-8")
            (ws / "feature.py").write_text("FLAG = True\n", encoding="utf-8")
        else:
            # The probe corrupted helper.py during cycle 1; Chronos must have
            # restored the worker's edit before this repair turn.
            assert (ws / "helper.py").read_text(encoding="utf-8") == "HELPER = True\n"
            (ws / "feature.py").write_text(
                "FLAG = True\nGUARD = True\n", encoding="utf-8"
            )

    snap = _retained_stage(
        store, tmp_path, _AttemptWorker([repair]), provider=provider
    ).run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.COMPLETED
    assert state["calls"] == 2
    events = store.list_events(run_id)
    assert not [e for e in events if e.event_type == "task.diff_gate_rejected"]
    overlay_dir = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay"
    feature = (overlay_dir / "feature.py").read_text(encoding="utf-8")
    helper = (overlay_dir / "helper.py").read_text(encoding="utf-8")
    assert "GUARD = True" in feature
    assert "HELPER = True" in helper and "CORRUPTED" not in helper


def test_resume_after_pause_reuses_retained_sandbox(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["feature.py"],
            max_cycles=3,
            checks=[_marker_command_check()],
        ),
    )

    def pause_then_repair(attempt: int, ws: Path, prompt: str) -> None:
        del prompt
        if attempt == 1:
            (ws / "feature.py").write_text("FLAG = 1\n", encoding="utf-8")
            store.set_run_status(run_id, RunStatus.PAUSED, actor="zeus")
            return
        assert (ws / "feature.py").read_text(encoding="utf-8") == "FLAG = 1\n"
        (ws / "feature.py").write_text("FLAG = 2\n", encoding="utf-8")

    worker = _AttemptWorker([pause_then_repair])
    stage = _retained_stage(store, tmp_path, worker)
    paused = stage.run_execution(run_id=run_id)

    assert paused["status"] == RunStatus.PAUSED
    sandbox = worker.sandboxes[0]
    assert sandbox.is_dir()
    assert (sandbox / "feature.py").is_file()

    store.set_run_status(run_id, RunStatus.RUNNING, actor="zeus")
    resumed = stage.run_execution(run_id=run_id)

    assert resumed["status"] == RunStatus.COMPLETED
    assert resumed["plan"].tasks[0].status == TaskStatus.ACCEPTED
    assert resumed["plan"].tasks[0].attempt == 2
    assert worker.sandboxes[0] == worker.sandboxes[1]
    assert _empty_dir(tmp_path / ".olympus" / "sandboxes")


def test_run_cancellation_reclaims_retained_sandbox_and_temp_dir(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["feature.py"],
            max_cycles=1,
            checks=[_marker_command_check()],
        ),
    )

    def stray(attempt: int, ws: Path, prompt: str) -> None:
        del attempt, prompt
        build_sandbox_env(ws, host_root=tmp_path)
        (ws / "feature.py").write_text("FLAG = True\n", encoding="utf-8")
        (ws / "scratch.py").write_text("stray\n", encoding="utf-8")

    worker = _AttemptWorker([stray])
    stage = _retained_stage(store, tmp_path, worker)
    snap = stage.run_execution(run_id=run_id, strict_scope=True)
    assert snap["status"] == RunStatus.AWAITING_HUMAN
    sandbox = worker.sandboxes[0]
    tmp_dir = tmp_path / ".olympus" / "sandbox_tmp" / sandbox.name
    assert sandbox.is_dir()
    assert tmp_dir.is_dir()

    store.set_run_status(run_id, RunStatus.CANCELLED, actor="zeus")
    cancelled = stage.run_execution(run_id=run_id)

    assert cancelled["status"] == RunStatus.CANCELLED
    assert not sandbox.exists()
    assert not tmp_dir.exists()


def test_execution_routes_role_drivers_and_repairs_external_review(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    plan = _plan_payload(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        file_name="feature.py",
        checks=[
            {
                "id": "CHK-REV",
                "kind": "independent_review",
                "description": "Audit the diff",
            }
        ],
    )
    provider = MockLLMProvider(responses=[LLMResponse(content=json.dumps(plan))])

    class ScriptedDriver:
        def __init__(self, name: str, responses: list[str]) -> None:
            self.name = name
            self.requests: list[DriverTurnRequest] = []
            self.responses = iter(responses)

        def run_turn(self, request: DriverTurnRequest, *, on_event=None, on_handle=None):
            self.requests.append(request)
            if self.name == "writer":
                (request.workspace_path / "feature.py").write_text("FLAG = True\n")
                assert on_event is not None
                assert request.trace_path is not None
                raw = {
                    "task_id": "spoofed",
                    "attempt": 99,
                    "actor": "spoofed",
                    "driver": "spoofed",
                    "step": 1,
                    "tool": "run_bash",
                    "target_hint": "PRIVATE_COMMAND",
                    "arguments": {"command": "PRIVATE_COMMAND"},
                    "output": "PRIVATE_STDOUT\nPRIVATE_STDERR",
                    "prompt": request.user_prompt,
                }
                on_event("agent.tool_started", raw)
                trace = [{"event": "text_delta", "text": "PRIVATE_DELTA"}] * 45
                for line in trace:
                    on_event("text_delta", line)
                finished = {
                    **raw, "is_error": True, "duration_ms": 9, "error_signature": raw["output"],
                }
                on_event("agent.tool_finished", finished)
                on_event("agent.steered", {"source": "watchdog", "reason": "PRIVATE_GUIDANCE"})
                trace.append({"event": "tool_end", **finished})
                request.trace_path.write_text(
                    "".join(json.dumps(line) + "\n" for line in trace), encoding="utf-8"
                )
            return TurnOutcome(
                True, "final_response", next(self.responses), 1, driver_name=self.name
            )

    writer = ScriptedDriver("writer", ["worker-only content"])
    verifier = ScriptedDriver(
        "verifier",
        [
            '{"task_id":"bad","attempt":1,"passed":true,"summary":"bad","extra":1}',
            'Result:\n```json\n{"task_id":"wrong","attempt":99,"passed":true,"summary":"OK"}\n```',
        ],
    )
    stage = OrchestrateStage(
        store=store,
        provider=provider,
        workspace_root=tmp_path,
        execution_drivers=ExecutionDriversConfig(
            worker_driver="writer@worker-model",
            reviewer_driver="verifier@different-model",
        ),
        driver_registry=DriverRegistry({"writer": writer, "verifier": verifier}),
    )
    stage.propose_plan(run_id=run_id, auto_approve=True)
    snap = stage.run_execution(run_id=run_id)

    assert len(provider.recorded_requests) == 1
    assert snap["status"] == RunStatus.COMPLETED
    assert len(writer.requests) == 1 and len(verifier.requests) == 2
    assert writer.requests[0].write_scope == ["feature.py"]
    assert writer.requests[0].model == "worker-model" and not writer.requests[0].read_only
    assert all(req.read_only and req.model == "different-model" for req in verifier.requests)
    assert "worker-only content" not in verifier.requests[0].user_prompt
    assert "VALIDATION" in verifier.requests[1].user_prompt
    events = store.list_events(run_id)
    milestones = [event.payload for event in events if event.event_type.startswith("agent.")]
    assert milestones[0] == {
        "task_id": "T1", "attempt": 1, "actor": "hephaestus:T1", "driver": "writer",
        "step": 1, "tool": "run_bash", "target_hint": "shell command",
    }
    assert milestones[1]["status"] == "error"
    assert milestones[1]["error_signature"].startswith("sha256:")
    assert milestones[2] == {
        "task_id": "T1", "attempt": 1, "actor": "hephaestus:T1", "source": "watchdog",
        "reason": "Watchdog guidance",
    }
    assert "PRIVATE_" not in json.dumps(milestones)
    assert not any(event.event_type == "text_delta" for event in events)
    node = build_run_projection(store, tmp_path, run_id)["dag"]["nodes"][0]
    assert node["artifacts"]["tool_calls"][0]["output_preview"] == "PRIVATE_STDOUT\nPRIVATE_STDERR"
    verdict = json.loads(
        (
            tmp_path
            / ".olympus"
            / "runs"
            / run_id
            / "artifacts"
            / "T1"
            / "attempt-1"
            / "verdict-cycle-1.json"
        ).read_text()
    )
    assert verdict["task_id"] == "T1" and verdict["attempt"] == 1


def test_live_worker_milestones_reach_ariadne_before_receipt(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    plan = _plan_payload(
        spec_id=spec.spec_id, spec_hash=spec.content_hash, file_name="feature.py",
        checks=[{"id": "CHK-REV", "kind": "independent_review", "description": "Audit the diff"}],
    )
    handles: list[ActiveTurnHandle] = []
    started, steered, release_tool, release_turn = (Event() for _ in range(4))

    def observe(kind: str, payload: dict[str, Any]) -> None:
        if kind == "agent.tool_started":
            started.set()
            release_tool.wait(5)
        elif kind == "agent.tool_finished":
            handles[0].steer("PRIVATE_STEERING_BODY", source="human")
        elif kind == "agent.steered":
            steered.set()
            release_turn.wait(5)

    app = AriadneApp(store=store, workspace_root=tmp_path)
    stage = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(
            responder=_stage_responder(plan=plan, file_name="feature.py", review_passes=[True])
        ),
        workspace_root=tmp_path,
        on_driver_event=observe,
        on_driver_handle=handles.append,
        live_activity=app.supervisor.live_activity,
    )
    stage.propose_plan(run_id=run_id, auto_approve=True)

    def task(payload: dict[str, Any]) -> dict[str, Any]:
        return payload["projection"]["dag"]["nodes"][0]

    with ThreadPoolExecutor() as pool:
        result = pool.submit(stage.run_execution, run_id=run_id)
        try:
            assert started.wait(3)
            initial = task(app.get_run_payload(run_id))["live_progress"]
            assert initial["driver_name"] == "native"
            assert initial["current_tool"]["tool"] == "write_file"
            assert initial["last_event_at"]
            assert initial["preview"] == "writing feature.py"
            release_tool.set()
            assert steered.wait(3)
            events = app.get_events_payload(run_id)["events"]
            worker_events = [e for e in events if e["actor"] == "hephaestus:T1"]
            milestones = [e for e in worker_events if e["event_type"].startswith("agent.")]
            assert [e["event_type"] for e in milestones] == [
                "agent.tool_started", "agent.tool_finished", "agent.steered",
            ]
            assert not any(e["event_type"] == "turn.receipt" for e in worker_events)
            assert not result.done()
            assert "PRIVATE_STEERING_BODY" not in json.dumps(milestones)
            assert "FLAG = True" not in json.dumps(milestones)
            assert str(tmp_path) not in json.dumps(milestones)

            node = task(app.get_run_payload(run_id))
            progress = node["live_progress"]
            assert progress["current_tool"] is None
            assert progress["recent_tools"][0]["status"] == "ok"
            assert progress["steer_count"] == 1
            assert progress["watchdog_verdict"] is None
            assert progress["watchdog_summary"] is None
            reopened = RunStore(store.db_path)
            durable = build_run_projection(reopened, tmp_path, run_id)["dag"]["nodes"][0][
                "live_progress"
            ]
            assert durable == {key: value for key, value in progress.items() if key != "preview"}
            assert not any(e["event_type"] == "text_delta" for e in worker_events)
            assert node["artifacts"]["tool_calls"][0]["tool"] == "write_file"
            assert node["artifacts"]["tool_calls"][0]["duration_ms"] >= 0
            worker_trace = Path(node["artifacts"]["attempt_dir"]) / "worker.jsonl"
            assert '"event": "text_delta"' in worker_trace.read_text(encoding="utf-8")

            with TestClient(create_fastapi_app(app)) as client:
                response = client.get(
                    f"/api/runs/{run_id}/stream"
                    f"?once=1&after_seq={milestones[0]['event_seq']}",
                )
                assert response.status_code == 200
                sse = json.loads(response.text.split("data: ", 1)[1])
                assert [e for e in sse["events"] if e["event_type"].startswith("agent.")] == (
                    milestones[1:]
                )
                with client.stream(
                    "GET",
                    f"/api/runs/{run_id}/stream"
                    f"?after_seq={milestones[-1]['event_seq']}",
                ) as stream:
                    lines = stream.iter_lines()
                    for line in lines:
                        if line == "event: activity":
                            activity = json.loads(next(lines).removeprefix("data: "))
                            break
                    else:
                        raise AssertionError("Expected an activity-only SSE event")
                assert activity["previews"]["T1"] == {"attempt": 1, "preview": "writing feature.py"}
        finally:
            release_tool.set()
            release_turn.set()
        assert result.result()["status"] == RunStatus.COMPLETED
        assert "preview" not in task(app.get_run_payload(run_id))["live_progress"]
        assert app.supervisor.live_activity.snapshot(run_id) == {}


def test_cli_run_combines_flags_and_env_for_role_models(tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from olympus.cli import app as cli_app

    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    plan = _plan_payload(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        file_name="feature.py",
        checks=[{"id": "review", "kind": "independent_review", "description": "Audit the diff"}],
    )
    planner = MockLLMProvider(responses=[LLMResponse(content=json.dumps(plan))])
    stage = OrchestrateStage(store=store, provider=planner, workspace_root=tmp_path)
    stage.propose_plan(run_id=run_id, auto_approve=True)
    built_models: list[str | None] = []

    def build_provider(*, model=None, **kwargs):
        built_models.append(model)
        return MockLLMProvider(
            model_name=model or "primary",
            responder=_stage_responder(plan=plan, file_name="feature.py", review_passes=[True]),
        )

    monkeypatch.setenv("OLYMPUS_WORKER_DRIVER", "native@ignored-worker-env")
    monkeypatch.setenv("OLYMPUS_REVIEWER_DRIVER", "native@reviewer-env")
    monkeypatch.setattr(cli_app, "build_provider", build_provider)
    result = CliRunner().invoke(
        cli_app.app,
        [
            "run",
            run_id,
            "--workspace",
            str(tmp_path),
            "--worker-driver",
            "native@worker-flag",
            "--monitor-model",
            "monitor-flag",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "completed"
    assert set(built_models) == {None, "worker-flag", "reviewer-env"}
    config_event = next(
        event
        for event in store.list_events(run_id)
        if event.event_type == "run.status_changed" and "execution_drivers" in event.payload
    )
    assert config_event.payload["execution_drivers"]["monitor_model"] == "monitor-flag"


def test_diff_gate_rejects_tracked_ignored_dir_and_posix_backslash_filename(
    tmp_path: Path,
) -> None:
    for rel, tracked in [("__pycache__/tracked.txt", True), (r"src\rogue.py", False)]:
        ws = tmp_path / ("case_tracked" if tracked else "case_backslash")
        ws.mkdir()
        _init_repo(ws)
        if tracked:
            (ws / rel).parent.mkdir()
            (ws / rel).write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "add", "-f", rel], cwd=ws, check=True)
            subprocess.run(["git", "commit", "-qm", "track ignored file"], cwd=ws, check=True)
        store, run_id = _confirmed_run(ws)
        plan = _single_task_plan(
            spec=store.get_run(run_id)["spec"],
            write_scope=["src/"],
            checks=[_marker_command_check()],
        )
        _approve_plan(
            store,
            run_id,
            plan.model_copy(update={"tasks": [plan.tasks[0].model_copy(update={"max_cycles": 1})]}),
        )

        class OutOfScopeDriver:
            name = "audit"

            def __init__(self, target_rel: str) -> None:
                self._target_rel = target_rel

            def run_turn(
                self,
                request: DriverTurnRequest,
                *,
                on_event: Any = None,
                on_handle: Any = None,
            ) -> TurnOutcome:
                del on_event, on_handle
                target = request.workspace_path / self._target_rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("mutated\n", encoding="utf-8")
                return TurnOutcome(
                    completed=True,
                    stop_reason="final_response",
                    final_content="done",
                    steps_taken=1,
                    driver_name="audit",
                )

        stage = OrchestrateStage(
            store=store,
            provider=MockLLMProvider(),
            workspace_root=ws,
            execution_drivers=ExecutionDriversConfig(worker_driver="audit"),
            driver_registry=DriverRegistry({"audit": OutOfScopeDriver(rel)}),
        )
        out = stage.run_execution(run_id=run_id, strict_scope=True)
        assert out["plan"].tasks[0].status == TaskStatus.BLOCKED
        rejections = [
            e for e in store.list_events(run_id) if e.event_type == "task.diff_gate_rejected"
        ]
        assert len(rejections) == 1
        assert rel in rejections[0].payload["violating_paths"]
        assert not (ws / ".olympus" / "runs" / run_id / "accepted_overlay" / rel).exists()


def test_argus_prompt_enforces_hard_write_scope_boundary(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    plan = _single_task_plan(
        spec=store.get_run(run_id)["spec"],
        write_scope=["feature.py"],
        checks=[_marker_command_check()],
    )
    _approve_plan(store, run_id, plan)
    seen_messages: list[tuple[str, str]] = []

    class CapturingReviewer(MockLLMProvider):
        def complete(self, *args: Any, **kwargs: Any) -> LLMResponse:
            del args
            messages = kwargs["messages"]
            seen_messages.append((messages[0].content, messages[1].content))
            return LLMResponse(
                content=json.dumps(
                    {
                        "task_id": "T1",
                        "attempt": 1,
                        "passed": True,
                        "summary": "ok",
                        "findings": [],
                    }
                )
            )

    class ScopedWorker:
        name = "scoped"
        supports_steer = False

        def steer(self, guidance: str, *, source: str) -> bool:
            del guidance, source
            return False

        def cancel(self, *, reason: str) -> None:
            del reason

        def run_turn(
            self,
            request: DriverTurnRequest,
            *,
            on_event: Any = None,
            on_handle: Any = None,
        ) -> TurnOutcome:
            del on_event, on_handle
            (request.workspace_path / "feature.py").write_text("FLAG = True\n", encoding="utf-8")
            return TurnOutcome(
                completed=True,
                stop_reason="final_response",
                final_content="done",
                steps_taken=1,
                driver_name="scoped",
            )

    stage = OrchestrateStage(
        store=store,
        workspace_root=tmp_path,
        provider=CapturingReviewer(),
        execution_drivers=ExecutionDriversConfig(worker_driver="scoped"),
        driver_registry=DriverRegistry({"scoped": ScopedWorker()}),
    )
    out = stage.run_execution(run_id=run_id)
    assert out["status"] == RunStatus.COMPLETED
    assert len(seen_messages) == 1
    sys_prompt, user_prompt = seen_messages[0]
    assert "TaskNode.write_scope as a hard boundary" in sys_prompt
    assert "outside its write_scope" in sys_prompt
    expected_scope_line = (
        "Hard write_scope boundary (do NOT require edits to files outside these paths): "
        '["feature.py"]'
    )
    assert expected_scope_line in user_prompt


def test_multiwave_diff_and_argus_prompt_exclude_unchanged_upstream_overlay(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    plan = TaskPlan.model_validate(
        {
            "plan_id": "plan-scoped-diff",
            "spec_id": spec.spec_id,
            "spec_hash": spec.content_hash,
            "tasks": [
                {
                    "task_id": "T0",
                    "title": "Publish the upstream module",
                    "objective": "Create the upstream module",
                    "requirement_refs": ["R1"],
                    "write_scope": ["upstream.py"],
                    "acceptance_checks": [
                        {
                            "id": "REV-0",
                            "kind": CheckKind.INDEPENDENT_REVIEW.value,
                            "description": "Review the upstream module",
                        }
                    ],
                },
                {
                    "task_id": "T1",
                    "title": "Publish the downstream module",
                    "objective": "Create the downstream module",
                    "requirement_refs": ["R1"],
                    "depends_on": ["T0"],
                    "write_scope": ["child.py"],
                    "acceptance_checks": [
                        {
                            "id": "REV-1",
                            "kind": CheckKind.INDEPENDENT_REVIEW.value,
                            "description": "Review the downstream module",
                        }
                    ],
                },
            ],
        }
    )
    _approve_plan(store, run_id, plan)
    upstream_body = "".join(f"UPSTREAM_{i} = {i}\n" for i in range(300))
    prompts: dict[str, str] = {}
    stage = OrchestrateStage(
        store=store,
        workspace_root=tmp_path,
        provider=_capturing_reviewer(prompts),
        execution_drivers=ExecutionDriversConfig(worker_driver="direct"),
        driver_registry=DriverRegistry(
            {
                "direct": _DirectWriteWorker(
                    {
                        "T0": {"upstream.py": upstream_body},
                        "T1": {"child.py": "CHILD = True\n"},
                    }
                )
            }
        ),
    )
    out = stage.run_execution(run_id=run_id)

    assert out["status"] == RunStatus.COMPLETED
    assert set(prompts) == {"T0", "T1"}
    assert "UPSTREAM_299" in prompts["T0"]

    downstream_prompt = prompts["T1"]
    assert "child.py" in downstream_prompt
    assert "CHILD = True" in downstream_prompt
    assert "UPSTREAM_" not in downstream_prompt
    assert "upstream.py" not in downstream_prompt

    diff_path = (
        tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "T1" / "attempt-1" / "diff.patch"
    )
    downstream_diff = diff_path.read_text(encoding="utf-8")
    assert "child.py" in downstream_diff and "CHILD = True" in downstream_diff
    assert "UPSTREAM_" not in downstream_diff
    assert "upstream.py" not in downstream_diff


def test_oversized_task_diff_is_bounded_in_argus_prompt_with_patch_pointer(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    plan = _single_task_plan(
        spec=store.get_run(run_id)["spec"],
        write_scope=["big_feature.py"],
        checks=[
            {
                "id": "REV-1",
                "kind": CheckKind.INDEPENDENT_REVIEW.value,
                "description": "Review the oversized module",
            }
        ],
    )
    _approve_plan(store, run_id, plan)
    body = "".join(f"ROW_{i} = {i}\n" for i in range(4000)) + "TAIL_MARKER = True\n"
    prompts: dict[str, str] = {}
    stage = OrchestrateStage(
        store=store,
        workspace_root=tmp_path,
        provider=_capturing_reviewer(prompts),
        execution_drivers=ExecutionDriversConfig(worker_driver="direct"),
        driver_registry=DriverRegistry(
            {"direct": _DirectWriteWorker({"T1": {"big_feature.py": body}})}
        ),
    )
    out = stage.run_execution(run_id=run_id)

    assert out["status"] == RunStatus.COMPLETED
    prompt = prompts["T1"]
    assert "new: big_feature.py" in prompt
    assert "+4001" in prompt
    assert "diff.patch" in prompt
    assert "truncated" in prompt.lower()
    assert "read-only inspection tools" in prompt
    assert "TAIL_MARKER" not in prompt
    assert "ROW_3999" not in prompt
    assert len(prompt) < 60_000

    diff_path = (
        tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "T1" / "attempt-1" / "diff.patch"
    )
    full_diff = diff_path.read_text(encoding="utf-8")
    assert "TAIL_MARKER" in full_diff
    assert "ROW_3999" in full_diff


def test_default_worker_max_steps_is_unlimited() -> None:
    import inspect

    from olympus.graph.loader import WorkflowRuntime

    sig = inspect.signature(OrchestrateStage.run_execution)
    assert sig.parameters["worker_max_steps"].default is None
    assert WorkflowRuntime.worker_max_steps is None


def test_max_steps_exceeded_with_valid_diff_still_verifies_and_accepts(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    plan = _single_task_plan(
        spec=store.get_run(run_id)["spec"],
        write_scope=["feature.py"],
        checks=[_marker_command_check()],
    )
    _approve_plan(store, run_id, plan)
    seen_max_steps: list[int | None] = []

    class StepCappedWorker:
        name = "capped"
        supports_steer = False

        def steer(self, guidance: str, *, source: str) -> bool:
            del guidance, source
            return False

        def cancel(self, *, reason: str) -> None:
            del reason

        def run_turn(
            self,
            request: DriverTurnRequest,
            *,
            on_event: Any = None,
            on_handle: Any = None,
        ) -> TurnOutcome:
            del on_event, on_handle
            seen_max_steps.append(request.max_steps)
            (request.workspace_path / "feature.py").write_text("FLAG = True\n", encoding="utf-8")
            return TurnOutcome(
                completed=False,
                stop_reason="max_steps_exceeded",
                final_content="",
                steps_taken=5,
                driver_name="capped",
            )

    verdict = json.dumps(
        {"task_id": "T1", "attempt": 1, "passed": True, "summary": "ok", "findings": []}
    )
    stage = OrchestrateStage(
        store=store,
        workspace_root=tmp_path,
        provider=MockLLMProvider(responses=[LLMResponse(content=verdict)]),
        execution_drivers=ExecutionDriversConfig(worker_driver="capped"),
        driver_registry=DriverRegistry({"capped": StepCappedWorker()}),
    )
    out = stage.run_execution(run_id=run_id, worker_max_steps=5)
    assert out["status"] == RunStatus.COMPLETED
    assert seen_max_steps == [5]
    assert out["plan"].tasks[0].status == TaskStatus.ACCEPTED


def test_agent_loop_supports_unlimited_max_steps() -> None:
    from olympus.harness.loop import AgentLoop
    from olympus.providers.base import ToolCall

    responses = [
        LLMResponse(
            content="",
            tool_calls=(ToolCall(id=f"c{i}", name="unknown_tool", arguments={}),),
        )
        for i in range(25)
    ] + [LLMResponse(content="finished after 26 steps")]
    loop = AgentLoop(
        provider=MockLLMProvider(responses=responses),
        max_steps=None,
    )
    res = loop.run(system_prompt="sys", user_prompt="user")
    assert res.completed is True
    assert res.stop_reason == "final_response"
    assert res.steps_taken == 26
    assert res.final_content == "finished after 26 steps"


class _GatedDriver:
    """Hold each worker turn until released, recording lifecycle ordering per task."""

    name = "gated"

    def __init__(self, files: dict[str, str]) -> None:
        self._files = files
        self._lock = Lock()
        self.started = {task_id: Event() for task_id in files}
        self.finished = {task_id: Event() for task_id in files}
        self.release = {task_id: Event() for task_id in files}
        self.start_times: dict[str, float] = {}
        self.end_times: dict[str, float] = {}
        self.prompts: dict[str, str] = {}

    def run_turn(
        self,
        request: DriverTurnRequest,
        *,
        on_event: Any = None,
        on_handle: Any = None,
    ) -> TurnOutcome:
        del on_event, on_handle
        task_id = request.task_id
        assert task_id is not None and task_id in self._files
        with self._lock:
            self.start_times[task_id] = time.monotonic()
            self.prompts[task_id] = request.user_prompt
        self.started[task_id].set()
        assert self.release[task_id].wait(10), f"{task_id} was never released"
        target = request.workspace_path / self._files[task_id]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{task_id}\n", encoding="utf-8")
        with self._lock:
            self.end_times[task_id] = time.monotonic()
        self.finished[task_id].set()
        return TurnOutcome(
            completed=True,
            stop_reason="final_response",
            final_content=f"{task_id} done",
            steps_taken=1,
            driver_name=self.name,
        )


def _parallel_plan(spec: GoalSpec, *, scopes: dict[str, list[str]]) -> TaskPlan:
    return TaskPlan.model_validate(
        {
            "plan_id": "plan-continuous",
            "spec_id": spec.spec_id,
            "spec_hash": spec.content_hash,
            "tasks": [
                {
                    "task_id": task_id,
                    "title": f"Run {task_id}",
                    "objective": f"Execute {task_id} independently",
                    "requirement_refs": ["R1"],
                    "write_scope": write_scope,
                    "acceptance_checks": [
                        {
                            "id": f"{task_id}-REV",
                            "kind": CheckKind.INDEPENDENT_REVIEW.value,
                            "description": "Cold review",
                        }
                    ],
                }
                for task_id, write_scope in scopes.items()
            ],
        }
    )


def _passing_reviewer(
    messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
) -> LLMResponse:
    del messages, tools
    return LLMResponse(
        content=json.dumps(
            {"task_id": "T1", "attempt": 1, "passed": True, "summary": "ok", "findings": []}
        )
    )


def _start_stage(
    stage: OrchestrateStage, run_id: str, **kwargs: Any
) -> tuple[Thread, list[dict[str, Any]], list[Exception]]:
    results: list[dict[str, Any]] = []
    errors: list[Exception] = []

    def execute() -> None:
        try:
            results.append(stage.run_execution(run_id=run_id, **kwargs))
        except Exception as exc:
            errors.append(exc)

    thread = Thread(target=execute, daemon=True)
    thread.start()
    return thread, results, errors


def test_continuous_dispatch_refills_slots_without_barrier_sync(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(
        tmp_path, budget_policy=BudgetPolicy(max_concurrent_tasks=2)
    )
    plan = _parallel_plan(
        store.get_run(run_id)["spec"],
        scopes={task_id: [f"{task_id.lower()}.txt"] for task_id in ("T1", "T2", "T3", "T4")},
    )
    _approve_plan(store, run_id, plan)
    driver = _GatedDriver(
        {task_id: f"{task_id.lower()}.txt" for task_id in ("T1", "T2", "T3", "T4")}
    )
    driver.release["T1"].set()
    driver.release["T3"].set()
    driver.release["T4"].set()
    stage = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=_passing_reviewer),
        workspace_root=tmp_path,
        execution_drivers=ExecutionDriversConfig(worker_driver="gated"),
        driver_registry=DriverRegistry({"gated": driver}),
    )

    thread, results, errors = _start_stage(stage, run_id)
    try:
        assert driver.started["T2"].wait(5), "blocked task never started"
        assert driver.started["T3"].wait(5), "Task C did not start while Task B was blocked"
        assert driver.finished["T3"].wait(5), "Task C did not finish while Task B was blocked"
        assert driver.finished["T4"].wait(5), "Task D did not finish while Task B was blocked"
        assert not driver.release["T2"].is_set()
    finally:
        driver.release["T2"].set()
        thread.join(10)
    assert not thread.is_alive()
    assert errors == []
    assert results[0]["status"] == RunStatus.COMPLETED
    assert driver.end_times["T3"] < driver.end_times["T2"]


def test_continuous_dispatch_serializes_overlapping_write_scopes(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(
        tmp_path, budget_policy=BudgetPolicy(max_concurrent_tasks=2)
    )
    plan = _parallel_plan(
        store.get_run(run_id)["spec"],
        scopes={"T1": ["shared/"], "T2": ["shared/"]},
    )
    _approve_plan(store, run_id, plan)
    driver = _GatedDriver({"T1": "shared/t1.txt", "T2": "shared/t2.txt"})
    stage = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=_passing_reviewer),
        workspace_root=tmp_path,
        execution_drivers=ExecutionDriversConfig(worker_driver="gated"),
        driver_registry=DriverRegistry({"gated": driver}),
    )

    thread, results, errors = _start_stage(stage, run_id)
    assert driver.started["T1"].wait(5)
    assert not driver.started["T2"].wait(0.3), "overlapping write_scope ran concurrently"
    driver.release["T1"].set()
    assert driver.finished["T1"].wait(5)
    assert driver.started["T2"].wait(5)
    driver.release["T2"].set()
    thread.join(10)
    assert not thread.is_alive()
    assert errors == []
    assert results[0]["status"] == RunStatus.COMPLETED
    assert driver.end_times["T1"] < driver.start_times["T2"]


def test_serial_execution_stays_sequential_and_keeps_handoff_context(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    plan = _parallel_plan(
        store.get_run(run_id)["spec"],
        scopes={"T1": ["t1.txt"], "T2": ["t2.txt"]},
    )
    _approve_plan(store, run_id, plan)
    driver = _GatedDriver({"T1": "t1.txt", "T2": "t2.txt"})
    stage = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=_passing_reviewer),
        workspace_root=tmp_path,
        execution_drivers=ExecutionDriversConfig(worker_driver="gated"),
        driver_registry=DriverRegistry({"gated": driver}),
    )

    thread, results, errors = _start_stage(stage, run_id, serial_execution=True)
    assert driver.started["T1"].wait(5)
    assert not driver.started["T2"].wait(0.3), "serial execution ran two tasks together"
    driver.release["T1"].set()
    assert driver.finished["T1"].wait(5)
    assert driver.started["T2"].wait(5)
    driver.release["T2"].set()
    thread.join(10)
    assert not thread.is_alive()
    assert errors == []
    assert results[0]["status"] == RunStatus.COMPLETED
    assert driver.end_times["T1"] < driver.start_times["T2"]
    assert "Prior Accepted Handoff Summary (serial_handoff)" in driver.prompts["T2"]
    assert "T1" in driver.prompts["T2"]
    status_events = [
        event
        for event in store.list_events(run_id)
        if event.event_type == "run.status_changed"
    ]
    assert any(event.payload.get("serial_execution") is True for event in status_events)


def test_retained_sandbox_diff_gate_blocks_worker_committed_out_of_scope_file(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=store.get_run(run_id)["spec"],
            write_scope=["feature.py"],
            max_cycles=2,
            checks=[_marker_command_check()],
        ),
    )

    def worker(attempt: int, ws: Path, prompt: str) -> None:
        del prompt
        if attempt == 1:
            (ws / "feature.py").write_text("FLAG = 1\n", encoding="utf-8")
            (ws / "out-of-scope.txt").write_text("leaked\n", encoding="utf-8")
            subprocess.run(["git", "add", "out-of-scope.txt"], cwd=ws, check=True)
            subprocess.run(["git", "commit", "-m", "commit leak"], cwd=ws, check=True)
        else:
            assert (ws / "out-of-scope.txt").is_file()

    snap = _retained_stage(store, tmp_path, _AttemptWorker([worker])).run_execution(
        run_id=run_id, strict_scope=True
    )
    rejections = [
        (event.payload["attempt"], event.payload["violating_paths"])
        for event in _gate_rejections(store, run_id)
    ]
    assert rejections == [(1, ["out-of-scope.txt"]), (2, ["out-of-scope.txt"])]
    assert snap["plan"].tasks[0].status == TaskStatus.BLOCKED


def test_continuous_dispatch_waits_for_upstream_overlay_publish_before_starting_dependent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import olympus.stages.orchestrate as orch_module

    _init_repo(tmp_path)
    store, run_id = _confirmed_run(
        tmp_path, budget_policy=BudgetPolicy(max_concurrent_tasks=2)
    )
    files = {"T1": "t1.txt", "T2": "t2.txt", "T3": "t3.txt"}
    plan = _parallel_plan(store.get_run(run_id)["spec"], scopes={k: [v] for k, v in files.items()})
    plan = plan.model_copy(
        update={
            "tasks": [
                *plan.tasks[:2],
                plan.tasks[2].model_copy(update={"depends_on": ["T1"]}),
            ]
        }
    )
    _approve_plan(store, run_id, plan)
    observations: list[bool] = []

    class _ObserveDriver(_GatedDriver):
        def run_turn(
            self,
            request: DriverTurnRequest,
            *,
            on_event: Any = None,
            on_handle: Any = None,
        ) -> TurnOutcome:
            if request.task_id == "T3":
                observations.append((request.workspace_path / "t1.txt").exists())
            return super().run_turn(request, on_event=on_event, on_handle=on_handle)

    driver = _ObserveDriver(files)
    stage = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=_passing_reviewer),
        workspace_root=tmp_path,
        execution_drivers=ExecutionDriversConfig(worker_driver="gated"),
        driver_registry=DriverRegistry({"gated": driver}),
    )
    original_copy = orch_module._copy_matching_scope
    before_publish, allow_publish = Event(), Event()

    def gated_copy(src: Path, dst: Path, write_scope: list[str] | None) -> None:
        if write_scope == ["t1.txt"] and dst.name == "accepted_overlay":
            before_publish.set()
            assert allow_publish.wait(10)
        original_copy(src, dst, write_scope=write_scope)

    monkeypatch.setattr(orch_module, "_copy_matching_scope", gated_copy)
    thread, results, errors = _start_stage(stage, run_id)
    try:
        assert driver.started["T1"].wait(5) and driver.started["T2"].wait(5)
        driver.release["T1"].set()
        assert before_publish.wait(5)
        driver.release["T2"].set()
        driver.release["T3"].set()
        assert not driver.started["T3"].wait(0.3), "T3 started before T1 overlay publish finished"
        allow_publish.set()
        assert driver.started["T3"].wait(5)
    finally:
        allow_publish.set()
        for event in driver.release.values():
            event.set()
        thread.join(10)

    assert not thread.is_alive() and not errors
    assert results[0]["status"] == RunStatus.COMPLETED
    assert observations == [True]


def test_worker_system_prompt_is_harness_agnostic_seam_first_and_anti_bloat() -> None:
    prompt = _WORKER_SYSTEM_PROMPT
    lowered = prompt.lower()
    assert "run_bash" not in prompt
    assert "failing seam test" in lowered
    assert "minimal clean" in lowered and "next criterion" in lowered
    assert "speculative" in lowered and "tautological" in lowered
    assert "batch" in lowered and "native" in lowered
    assert "shell execution" in lowered
    for probe in ("`cat`", "`sed`", "`grep`", "`find`"):
        assert probe in prompt


class _SliceDriver:
    """Write each vertical slice's files into its sandbox without LLM turns."""

    name = "slices"

    def __init__(self, files: dict[str, dict[str, str]]) -> None:
        self._files = files
        self.turns: list[str] = []

    def run_turn(
        self,
        request: DriverTurnRequest,
        *,
        on_event: Any = None,
        on_handle: Any = None,
    ) -> TurnOutcome:
        del on_event, on_handle
        task_id = request.task_id
        assert task_id is not None and task_id in self._files
        self.turns.append(task_id)
        for rel_path, content in self._files[task_id].items():
            target = request.workspace_path / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return TurnOutcome(
            completed=True,
            stop_reason="final_response",
            final_content=f"{task_id} implemented",
            steps_taken=1,
            driver_name=self.name,
        )


def _vertical_slice_fixture() -> tuple[GoalSpec, DesignSpec, TaskPlan]:
    spec = GoalSpec(
        spec_id="spec-vertical",
        goal="Grow an inventory through vertical slices",
        requirements=[
            {
                "id": requirement_id,
                "description": f"Deliver capability {requirement_id}",
                "acceptance_criteria": ["Works end to end"],
            }
            for requirement_id in ("R1", "R2", "R3", "R4")
        ],
    )
    design = DesignSpec.model_validate(
        {
            "design_id": "design-vertical",
            "spec_id": spec.spec_id,
            "spec_hash": spec.content_hash,
            "architecture_summary": "One storage core with two feature modules.",
            "modules": [
                {
                    "id": "M1",
                    "name": "Core",
                    "responsibility": "Storage and entrypoint",
                    "requirement_refs": ["R1"],
                    "target_paths": ["store.py", "app.py"],
                },
                {
                    "id": "M2",
                    "name": "Items",
                    "responsibility": "Item capability",
                    "requirement_refs": ["R2", "R3"],
                    "target_paths": ["feature_items.py"],
                },
                {
                    "id": "M3",
                    "name": "Count",
                    "responsibility": "Count capability",
                    "requirement_refs": ["R4"],
                    "target_paths": ["feature_count.py"],
                },
            ],
        }
    )
    plan = TaskPlan.model_validate(
        {
            "plan_id": "plan-vertical",
            "spec_id": spec.spec_id,
            "spec_hash": spec.content_hash,
            "design_id": design.design_id,
            "design_hash": design.content_hash,
            "tasks": [
                {
                    "task_id": "T1-skeleton",
                    "title": "Runnable skeleton",
                    "objective": "Wire storage into one observable seam",
                    "requirement_refs": ["R1"],
                    "design_refs": ["M1"],
                    "write_scope": ["store.py", "app.py", "test_app.py"],
                    "acceptance_checks": [
                        {
                            "id": "T1-C1",
                            "kind": CheckKind.COMMAND.value,
                            "description": "Seam returns a missing item",
                            "command": (
                                "python3 -c 'import app; assert app.get_item(\"missing\") is None'"
                            ),
                        }
                    ],
                },
                {
                    "task_id": "T2-items",
                    "title": "Items slice",
                    "objective": "Deliver the item capability end to end",
                    "requirement_refs": ["R2", "R3"],
                    "design_refs": ["M2"],
                    "depends_on": ["T1-skeleton"],
                    "write_scope": ["feature_items.py", "test_feature_items.py"],
                    "acceptance_checks": [
                        {
                            "id": "T2-C1",
                            "kind": CheckKind.COMMAND.value,
                            "description": "Item round trips through storage",
                            "command": (
                                "python3 -c 'import feature_items; "
                                "assert feature_items.add_item(\"a\", 1) == 1'"
                            ),
                        }
                    ],
                },
                {
                    "task_id": "T3-count",
                    "title": "Count slice",
                    "objective": "Deliver the count capability end to end",
                    "requirement_refs": ["R4"],
                    "design_refs": ["M3"],
                    "depends_on": ["T1-skeleton"],
                    "write_scope": ["feature_count.py", "test_feature_count.py"],
                    "acceptance_checks": [
                        {
                            "id": "T3-C1",
                            "kind": CheckKind.COMMAND.value,
                            "description": "Count reads the storage core",
                            "command": (
                                "python3 -c 'import feature_count; "
                                "assert feature_count.count() == 0'"
                            ),
                        }
                    ],
                },
            ],
        }
    )
    return spec, design, plan


def test_vertical_three_slice_tracer_bullet_dag_completes_end_to_end(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store = RunStore(tmp_path / ".olympus" / "olympus.db")
    run_id = store.create_run(title="Vertical tracer bullet", mode=RunMode.EXECUTION)
    spec, design, plan = _vertical_slice_fixture()
    store.save_spec(run_id, spec)
    align = AlignStage(store=store, provider=MockLLMProvider(), workspace_root=tmp_path)
    align.confirm_spec(run_id=run_id, approver="zeus")
    assert validate_plan(plan, spec=spec, design=design, topology="vertical_slices") == []
    _approve_plan(store, run_id, plan, design=design)

    driver = _SliceDriver(
        {
            "T1-skeleton": {
                "store.py": "ITEMS: dict[str, int] = {}\n",
                "app.py": "from store import ITEMS\n\n\ndef get_item(key: str) -> int | None:\n"
                "    return ITEMS.get(key)\n",
                "test_app.py": "def test_skeleton() -> None:\n    assert True\n",
            },
            "T2-items": {
                "feature_items.py": "from store import ITEMS\n\n\ndef add_item("
                "key: str, value: int) -> int:\n    ITEMS[key] = value\n"
                "    return ITEMS[key]\n",
                "test_feature_items.py": "def test_items() -> None:\n    assert True\n",
            },
            "T3-count": {
                "feature_count.py": "from store import ITEMS\n\n\ndef count() -> int:\n"
                "    return len(ITEMS)\n",
                "test_feature_count.py": "def test_count() -> None:\n    assert True\n",
            },
        }
    )
    stage = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=_passing_reviewer),
        workspace_root=tmp_path,
        execution_drivers=ExecutionDriversConfig(worker_driver="slices"),
        driver_registry=DriverRegistry({"slices": driver}),
    )
    final = stage.run_execution(run_id=run_id)
    assert final["status"] == RunStatus.COMPLETED
    assert {task.status for task in final["plan"].tasks} == {TaskStatus.ACCEPTED}
    assert set(driver.turns) == {"T1-skeleton", "T2-items", "T3-count"}
    assert driver.turns[0] == "T1-skeleton"


def test_exit_zero_verification_probe_cannot_inject_or_mutate_delivered_files(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    (tmp_path / "clean.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "clean.py"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "add clean file"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    store, run_id = _confirmed_run(tmp_path, mode=RunMode.EXECUTION)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    repro = (
        "python3 -c \"from pathlib import Path; "
        "Path('injected.py').write_text('LEAK = True\\n'); "
        "Path('clean.py').write_text('VALUE = 999\\n')\""
    )
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["*.py"],
            max_cycles=2,
            checks=[
                {
                    "id": "CHK-REV",
                    "kind": CheckKind.INDEPENDENT_REVIEW.value,
                    "description": "Audit implementation",
                }
            ],
        ),
    )
    provider, _ = _review_script(
        [
            {
                "task_id": "T1",
                "attempt": 1,
                "passed": False,
                "summary": "Claimed defect with exit-0 mutating probe",
                "findings": [
                    {
                        "id": "F1",
                        "axis": "spec",
                        "severity": "important",
                        "file_path": "feature.py",
                        "line": 1,
                        "requirement_ref": "R1",
                        "repro_command": repro,
                        "message": "False positive probe that writes files and exits 0",
                    }
                ],
            }
        ]
    )

    def write_feature(attempt: int, ws: Path, prompt: str) -> None:
        del attempt, prompt
        (ws / "feature.py").write_text("FLAG = True\n", encoding="utf-8")

    snap = _retained_stage(
        store, tmp_path, _AttemptWorker([write_feature]), provider=provider
    ).run_execution(run_id=run_id, apply_to_workspace=True)

    assert snap["status"] == RunStatus.COMPLETED
    overlay = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay"
    assert (tmp_path / "feature.py").read_text(encoding="utf-8") == "FLAG = True\n"
    assert (tmp_path / "clean.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not (tmp_path / "injected.py").exists()
    assert not (overlay / "injected.py").exists()
    assert not (overlay / "clean.py").exists() or (
        (overlay / "clean.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    )


def test_worker_rename_removes_old_path_from_overlay_and_workspace(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "pkg").mkdir(parents=True, exist_ok=True)
    (tmp_path / "pkg" / "old.py").write_text("OLD = True\n", encoding="utf-8")
    subprocess.run(["git", "add", "pkg/old.py"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "add pkg/old.py"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    store, run_id = _confirmed_run(tmp_path, mode=RunMode.EXECUTION)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["pkg/old.py"],
            max_cycles=2,
            checks=[
                {
                    "id": "CHK-1",
                    "kind": CheckKind.COMMAND.value,
                    "description": "Renamed module exists",
                    "command": "python3 -c 'import pkg.new as m; assert m.NEW'",
                }
            ],
        ),
    )
    overlay_old = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay" / "pkg" / "old.py"
    overlay_old.parent.mkdir(parents=True, exist_ok=True)
    overlay_old.write_text("OLD = True\n", encoding="utf-8")

    def rename_module(attempt: int, ws: Path, prompt: str) -> None:
        del attempt, prompt
        (ws / "pkg" / "old.py").unlink()
        (ws / "pkg" / "new.py").write_text("NEW = True\n", encoding="utf-8")

    snap = _retained_stage(
        store,
        tmp_path,
        _AttemptWorker([rename_module]),
        provider=MockLLMProvider(responder=_passing_reviewer),
    ).run_execution(run_id=run_id, apply_to_workspace=True)

    assert snap["status"] == RunStatus.COMPLETED
    overlay = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay"
    assert not (tmp_path / "pkg" / "old.py").exists()
    assert not (overlay / "pkg" / "old.py").exists()
    assert (tmp_path / "pkg" / "new.py").read_text(encoding="utf-8") == "NEW = True\n"
    assert (overlay / "pkg" / "new.py").read_text(encoding="utf-8") == "NEW = True\n"


def test_human_in_place_retry_still_rechecks_verified_repro_before_argus(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    repro = "python3 -c 'import feature; assert feature.GUARD'"
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["feature.py"],
            max_cycles=1,
            checks=[
                {
                    "id": "CHK-REV",
                    "kind": CheckKind.INDEPENDENT_REVIEW.value,
                    "description": "Audit the guard",
                }
            ],
        ),
    )
    provider, state = _review_script(
        [
            {
                "task_id": "T1",
                "attempt": 1,
                "passed": False,
                "summary": "Missing GUARD",
                "findings": [
                    {
                        "id": "F1",
                        "axis": "spec",
                        "severity": "important",
                        "file_path": "feature.py",
                        "line": 1,
                        "requirement_ref": "R1",
                        "repro_command": repro,
                        "message": "GUARD attribute missing",
                    }
                ],
            },
            {
                "task_id": "T1",
                "attempt": 3,
                "passed": True,
                "summary": "GUARD present",
                "findings": [],
            },
        ]
    )

    def repair(attempt: int, ws: Path, prompt: str) -> None:
        if attempt == 1:
            (ws / "feature.py").write_text("FLAG = True\n", encoding="utf-8")
        elif attempt == 2:
            assert "[Human hint]" in prompt
            # Still fails the proven repro; Chronos must catch it with 0 Argus calls.
            (ws / "feature.py").write_text("FLAG = True\n", encoding="utf-8")
        else:
            (ws / "feature.py").write_text("FLAG = True\nGUARD = True\n", encoding="utf-8")

    stage = _retained_stage(store, tmp_path, _AttemptWorker([repair]), provider=provider)
    first = stage.run_execution(run_id=run_id)
    assert first["status"] == RunStatus.AWAITING_HUMAN
    assert first["plan"].tasks[0].status == TaskStatus.BLOCKED
    assert first["plan"].tasks[0].verified_repro_commands == [repro]
    assert state["calls"] == 1

    # Human retries in place: review_cycle resets to 0, verified_repro_commands stays set.
    RunControl(store).retry_task(
        run_id,
        "T1",
        operation_id="op-retry-1",
        expected_state_revision=first["state_revision"],
        hint="Please define GUARD = True",
        actor="human",
    )
    retried = store.get_run(run_id)["plan"].tasks[0]
    assert retried.review_cycle == 0
    assert retried.verified_repro_commands == [repro]

    second = stage.run_execution(run_id=run_id)
    assert second["status"] == RunStatus.AWAITING_HUMAN
    assert second["plan"].tasks[0].status == TaskStatus.BLOCKED
    assert state["calls"] == 1  # Short-circuited at 0 reviewer LLM tokens!
    rechecks = [
        e for e in store.list_events(run_id) if e.event_type == "review.repro_rechecked"
    ]
    assert len(rechecks) == 1
    assert rechecks[0].payload["passed"] is False
    assert rechecks[0].payload["short_circuited"] is True

    RunControl(store).retry_task(
        run_id,
        "T1",
        operation_id="op-retry-2",
        expected_state_revision=second["state_revision"],
        hint="Actually add GUARD = True now",
        actor="human",
    )
    third = stage.run_execution(run_id=run_id)
    assert third["status"] == RunStatus.COMPLETED
    assert third["plan"].tasks[0].status == TaskStatus.ACCEPTED
    assert third["plan"].tasks[0].verified_repro_commands == []
    assert state["calls"] == 2
    assert "NOW PASS" in state["prompts"][1]


def test_probe_isolated_preserves_exec_mode_and_hides_parent_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stat

    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path, mode=RunMode.EXECUTION)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["feature.py"],
            max_cycles=1,
            checks=[
                {
                    "id": "CHK-REV",
                    "kind": CheckKind.INDEPENDENT_REVIEW.value,
                    "description": "Audit implementation",
                }
            ],
        ),
    )

    def write_exec(attempt: int, ws: Path, prompt: str) -> None:
        del attempt, prompt
        target = ws / "feature.py"
        target.write_text("#!/usr/bin/env python3\nFLAG = True\n", encoding="utf-8")
        target.chmod(0o755)

    monkeypatch.setenv("OLYMPUS_REVIEW_FIXTURE_TOKEN", "synthetic-fixture")
    provider, _ = _review_script(
        [
            {
                "task_id": "T1",
                "attempt": 1,
                "passed": False,
                "summary": "Mutating mode probe and env leak probe",
                "findings": [
                    {
                        "id": "F1",
                        "axis": "spec",
                        "severity": "important",
                        "file_path": "feature.py",
                        "line": 1,
                        "requirement_ref": "R1",
                        "repro_command": "chmod a-x feature.py",
                        "message": "Mode-mutating exit-0 probe",
                    },
                    {
                        "id": "F2",
                        "axis": "spec",
                        "severity": "important",
                        "file_path": "feature.py",
                        "line": 1,
                        "requirement_ref": "R1",
                        "repro_command": (
                            'if [ "${OLYMPUS_REVIEW_FIXTURE_TOKEN:-}" = synthetic-fixture ]; '
                            "then exit 1; fi"
                        ),
                        "message": "Env secret should not be visible to probe",
                    },
                ],
            }
        ]
    )
    snap = _retained_stage(
        store, tmp_path, _AttemptWorker([write_exec]), provider=provider
    ).run_execution(run_id=run_id, apply_to_workspace=True)

    assert snap["status"] == RunStatus.COMPLETED
    mode = stat.S_IMODE((tmp_path / "feature.py").stat().st_mode)
    assert mode == 0o755


def test_accepted_rename_deletion_persists_into_downstream_dag_and_recreation_clears_tombstone(
    tmp_path: Path,
) -> None:
    from olympus.core.contracts import AcceptanceCheck

    _init_repo(tmp_path)
    (tmp_path / "old.py").write_text("OLD = True\n", encoding="utf-8")
    subprocess.run(["git", "add", "old.py"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "add old.py"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    store, run_id = _confirmed_run(tmp_path, mode=RunMode.EXECUTION)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    base_plan = _parallel_plan(spec, scopes={"T1": ["old.py", "new.py"], "T2": ["second.py"]})
    t2_check = AcceptanceCheck(
        id="C2",
        kind=CheckKind.COMMAND,
        description="Old module stays gone in T2 sandbox",
        command="test ! -e old.py && test -f new.py",
    )
    t3_task = base_plan.tasks[1].model_copy(
        update={
            "task_id": "T3",
            "title": "Recreate old.py",
            "depends_on": ["T2"],
            "write_scope": ["old.py"],
            "acceptance_checks": [
                AcceptanceCheck(
                    id="C3",
                    kind=CheckKind.COMMAND,
                    description="Recreated old.py exists",
                    command="python3 -c 'import old; assert old.RECREATED'",
                )
            ],
            "max_cycles": 1,
        }
    )
    plan = base_plan.model_copy(
        update={
            "tasks": [
                base_plan.tasks[0],
                base_plan.tasks[1].model_copy(
                    update={"depends_on": ["T1"], "acceptance_checks": [t2_check], "max_cycles": 1}
                ),
                t3_task,
            ]
        }
    )
    _approve_plan(store, run_id, plan)

    seen_in_t2: list[tuple[bool, bool]] = []

    class _RenameDagWorker(_DirectWriteWorker):
        def run_turn(self, request: DriverTurnRequest, **kw: Any) -> TurnOutcome:
            ws = request.workspace_path
            if request.task_id == "T1":
                (ws / "old.py").rename(ws / "new.py")
            elif request.task_id == "T2":
                seen_in_t2.append(((ws / "old.py").exists(), (ws / "new.py").exists()))
            return super().run_turn(request, **kw)

    worker = _RenameDagWorker(
        {
            "T1": {},
            "T2": {"second.py": "OK = True\n"},
            "T3": {"old.py": "RECREATED = True\n"},
        }
    )
    stage = OrchestrateStage(
        store=store,
        workspace_root=tmp_path,
        provider=MockLLMProvider(responder=_passing_reviewer),
        execution_drivers=ExecutionDriversConfig(worker_driver="direct"),
        driver_registry=DriverRegistry({"direct": worker}),
    )
    snap = stage.run_execution(run_id=run_id, apply_to_workspace=True)
    assert snap["status"] == RunStatus.COMPLETED
    assert seen_in_t2 == [(False, True)]
    assert (tmp_path / "new.py").exists()
    assert (tmp_path / "old.py").read_text(encoding="utf-8") == "RECREATED = True\n"
    tombstones = tmp_path / ".olympus" / "runs" / run_id / "accepted_deletions.json"
    assert not tombstones.exists()


def test_serial_task_blocks_edits_to_spec_out_of_scope_and_frozen_test_paths(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    frozen = tmp_path / "tests" / "test_frozen.py"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text("def test_frozen(): assert False\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "tests/test_frozen.py"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "add frozen test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    store, run_id = _confirmed_run(tmp_path, mode=RunMode.EXECUTION)
    original = store.get_run(run_id)["spec"]
    assert original is not None
    spec = original.model_copy(
        update={
            "out_of_scope": ["tests/test_frozen.py"],
            "constraints": ["Test freeze: do not modify tests/test_frozen.py"],
        }
    )
    store.save_spec(run_id, spec)
    AlignStage(store=store, provider=MockLLMProvider(), workspace_root=tmp_path).confirm_spec(
        run_id=run_id, approver="zeus"
    )
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["feature.py"],
            max_cycles=1,
            checks=[
                {
                    "id": "CHK-REV",
                    "kind": CheckKind.INDEPENDENT_REVIEW.value,
                    "description": "Review",
                }
            ],
        ),
    )

    def tamper_frozen(attempt: int, ws: Path, prompt: str) -> None:
        del attempt, prompt
        (ws / "feature.py").write_text("FLAG = True\n", encoding="utf-8")
        (ws / "tests" / "test_frozen.py").write_text(
            "def test_frozen(): assert True\n", encoding="utf-8"
        )

    snap = _retained_stage(
        store,
        tmp_path,
        _AttemptWorker([tamper_frozen]),
        provider=MockLLMProvider(responder=_passing_reviewer),
    ).run_execution(run_id=run_id, apply_to_workspace=True)

    assert snap["plan"].tasks[0].status == TaskStatus.BLOCKED
    assert "assert False" in frozen.read_text(encoding="utf-8")
    rejections = [
        e for e in store.list_events(run_id) if e.event_type == "task.diff_gate_rejected"
    ]
    assert rejections
    assert "tests/test_frozen.py" in rejections[0].payload["violating_paths"]


def test_propose_plan_strips_planner_populated_execution_state_fields(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path, mode=RunMode.EXECUTION)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    raw = _single_task_plan(
        spec=spec,
        write_scope=["feature.py"],
        max_cycles=1,
        checks=[
            {
                "id": "CHK-REV",
                "kind": CheckKind.INDEPENDENT_REVIEW.value,
                "description": "Review",
            }
        ],
    ).model_dump(mode="json")
    raw["tasks"][0].update(
        verified_repro_commands=["test -f planner-injected.fixture"],
        in_place_recoveries=2,
        attempt=4,
        review_cycle=1,
    )

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        if "Independent Verifier" in messages[0].content:
            return _passing_reviewer(messages, tools)
        return LLMResponse(content=json.dumps(raw))

    def write_feature(attempt: int, ws: Path, prompt: str) -> None:
        del attempt, prompt
        (ws / "feature.py").write_text("FLAG = True\n", encoding="utf-8")

    stage = _retained_stage(
        store,
        tmp_path,
        _AttemptWorker([write_feature]),
        provider=MockLLMProvider(responder=responder),
    )
    proposed = stage.propose_plan(run_id=run_id, auto_approve=True)
    task = proposed.tasks[0]
    assert task.verified_repro_commands == []
    assert task.in_place_recoveries == 0
    assert task.attempt == 0
    assert task.review_cycle == 0

    snap = stage.run_execution(run_id=run_id)
    assert snap["status"] == RunStatus.COMPLETED
    assert snap["plan"].tasks[0].status == TaskStatus.ACCEPTED


def test_recheck_drops_broken_prior_repro_and_reinvokes_argus(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path, mode=RunMode.EXECUTION)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["feature.py"],
            max_cycles=2,
            checks=[
                {
                    "id": "CHK-REV",
                    "kind": CheckKind.INDEPENDENT_REVIEW.value,
                    "description": "Review",
                }
            ],
        ),
    )
    command = "if grep -q OLD feature.py; then exit 1; else exit 127; fi"
    provider, state = _review_script(
        [
            {
                "task_id": "T1",
                "attempt": 1,
                "passed": False,
                "summary": "OLD marker present",
                "findings": [
                    {
                        "id": "F1",
                        "axis": "spec",
                        "severity": "important",
                        "file_path": "feature.py",
                        "line": 1,
                        "requirement_ref": "R1",
                        "repro_command": command,
                        "message": "OLD marker fails",
                    }
                ],
            },
            {
                "task_id": "T1",
                "attempt": 2,
                "passed": True,
                "summary": "Now fixed",
                "findings": [],
            },
        ]
    )

    def worker(attempt: int, ws: Path, prompt: str) -> None:
        del prompt
        (ws / "feature.py").write_text(
            "OLD = True\n" if attempt == 1 else "FIXED = True\n", encoding="utf-8"
        )

    snap = _retained_stage(
        store, tmp_path, _AttemptWorker([worker]), provider=provider
    ).run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.COMPLETED
    assert snap["plan"].tasks[0].status == TaskStatus.ACCEPTED
    assert state["calls"] == 2
    recheck = next(
        e.payload for e in store.list_events(run_id) if e.event_type == "review.repro_rechecked"
    )
    assert recheck["short_circuited"] is False
    assert recheck["results"][0]["broken"] is True


def test_missing_or_unknown_requirement_ref_cannot_block_acceptance(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path, mode=RunMode.EXECUTION)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["feature.py"],
            max_cycles=1,
            checks=[
                {
                    "id": "CHK-REV",
                    "kind": CheckKind.INDEPENDENT_REVIEW.value,
                    "description": "Review",
                }
            ],
        ),
    )
    provider, _ = _review_script(
        [
            {
                "task_id": "T1",
                "attempt": 1,
                "passed": False,
                "summary": "Unanchored findings",
                "findings": [
                    {
                        "id": "F1",
                        "axis": "spec",
                        "severity": "important",
                        "file_path": "feature.py",
                        "line": 1,
                        "requirement_ref": None,
                        "repro_command": "exit 1",
                        "message": "Missing requirement_ref",
                    },
                    {
                        "id": "F2",
                        "axis": "spec",
                        "severity": "important",
                        "file_path": "feature.py",
                        "line": 1,
                        "requirement_ref": "R999",
                        "repro_command": "exit 1",
                        "message": "Unknown requirement_ref",
                    },
                ],
            }
        ]
    )

    def write_feature(attempt: int, ws: Path, prompt: str) -> None:
        del attempt, prompt
        (ws / "feature.py").write_text("FLAG = True\n", encoding="utf-8")

    snap = _retained_stage(
        store, tmp_path, _AttemptWorker([write_feature]), provider=provider
    ).run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.COMPLETED
    event = next(
        e.payload for e in store.list_events(run_id) if e.event_type == "review.verdict_verified"
    )
    assert event["verified_important"] == 0
    assert event["dropped_unverified"] == 2


def test_tracked_ephemeral_suffix_deletion_is_promoted_to_workspace(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "legacy.bak").write_text("tracked file\n", encoding="utf-8")
    subprocess.run(["git", "add", "legacy.bak"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "add tracked bak"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    store, run_id = _confirmed_run(tmp_path, mode=RunMode.EXECUTION)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["legacy.bak"],
            max_cycles=1,
            checks=[
                {
                    "id": "CHK-REV",
                    "kind": CheckKind.INDEPENDENT_REVIEW.value,
                    "description": "Review",
                }
            ],
        ),
    )

    def delete_bak(attempt: int, ws: Path, prompt: str) -> None:
        del attempt, prompt
        (ws / "legacy.bak").unlink()

    snap = _retained_stage(
        store,
        tmp_path,
        _AttemptWorker([delete_bak]),
        provider=MockLLMProvider(responder=_passing_reviewer),
    ).run_execution(run_id=run_id, apply_to_workspace=True)

    assert snap["status"] == RunStatus.COMPLETED
    assert not (tmp_path / "legacy.bak").exists()


def test_probe_strips_external_symlink_and_keeps_worker_syntax_error_as_real_defect(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("SAFE\n", encoding="utf-8")
    _init_repo(repo)
    store, run_id = _confirmed_run(repo, mode=RunMode.EXECUTION)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["feature.py", "ext_link.txt"],
            max_cycles=1,
            checks=[
                {
                    "id": "CHK-REV",
                    "kind": CheckKind.INDEPENDENT_REVIEW.value,
                    "description": "Review",
                }
            ],
        ),
    )

    def write_broken(attempt: int, ws: Path, prompt: str) -> None:
        del attempt, prompt
        (ws / "feature.py").write_text("def broken(:\n", encoding="utf-8")
        (ws / "ext_link.txt").symlink_to(outside)

    provider, _ = _review_script(
        [
            {
                "task_id": "T1",
                "attempt": 1,
                "passed": False,
                "summary": "Syntax error in worker module and symlink probe",
                "findings": [
                    {
                        "id": "F1",
                        "axis": "spec",
                        "severity": "important",
                        "file_path": "ext_link.txt",
                        "line": 1,
                        "requirement_ref": "R1",
                        "repro_command": "printf MUTATED > ext_link.txt",
                        "message": "Probe attempting to write through external symlink",
                    },
                    {
                        "id": "F2",
                        "axis": "spec",
                        "severity": "important",
                        "file_path": "feature.py",
                        "line": 1,
                        "requirement_ref": "R1",
                        "repro_command": "python3 -c 'import feature'",
                        "message": "feature.py has a genuine SyntaxError",
                    },
                ],
            }
        ]
    )
    snap = _retained_stage(
        store, repo, _AttemptWorker([write_broken]), provider=provider
    ).run_execution(run_id=run_id)

    assert outside.read_text(encoding="utf-8") == "SAFE\n"
    assert snap["plan"].tasks[0].status == TaskStatus.BLOCKED
    event = next(
        e.payload for e in store.list_events(run_id) if e.event_type == "review.verdict_verified"
    )
    assert event["verified_important"] == 1
    assert event["verified_repro_commands"] == ["python3 -c 'import feature'"]


def test_delivery_checks_respect_accepted_deletions_and_file_to_dir_replacement(
    tmp_path: Path,
) -> None:
    from olympus.core.contracts import DeliveryCheck

    _init_repo(tmp_path)
    (tmp_path / "old.py").write_text("OLD = True\n", encoding="utf-8")
    (tmp_path / "foo").write_text("LEGACY_FILE = True\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "old.py", "foo"], cwd=tmp_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "add old.py and foo"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    store, run_id = _confirmed_run(tmp_path, mode=RunMode.EXECUTION)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    base = _single_task_plan(
        spec=spec,
        write_scope=["old.py", "foo", "foo/**"],
        max_cycles=1,
        checks=[
            {
                "id": "CHK-REV",
                "kind": CheckKind.INDEPENDENT_REVIEW.value,
                "description": "Review",
            }
        ],
    )
    plan = base.model_copy(
        update={
            "delivery_checks": [
                DeliveryCheck(
                    requirement_id="R1",
                    criterion_index=1,
                    command="test ! -e old.py && test -f foo/bar.py",
                )
            ]
        }
    )
    _approve_plan(store, run_id, plan)

    def replace_paths(attempt: int, ws: Path, prompt: str) -> None:
        del attempt, prompt
        (ws / "old.py").unlink()
        (ws / "foo").unlink()
        (ws / "foo").mkdir()
        (ws / "foo" / "bar.py").write_text("BAR = True\n", encoding="utf-8")

    snap = _retained_stage(
        store,
        tmp_path,
        _AttemptWorker([replace_paths]),
        provider=MockLLMProvider(responder=_passing_reviewer),
    ).run_execution(run_id=run_id, apply_to_workspace=True)

    assert snap["status"] == RunStatus.COMPLETED
    assert not (tmp_path / "old.py").exists()
    assert (tmp_path / "foo" / "bar.py").read_text(encoding="utf-8") == "BAR = True\n"
    delivery = next(
        e.payload for e in store.list_events(run_id) if e.event_type == "delivery.checks_completed"
    )
    assert delivery["results"][0]["exit_code"] == 0


def test_backticked_nonexistent_frozen_path_blocked_and_untouched_scope_not_overwritten(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    (tmp_path / "other.py").write_text("BASE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "other.py"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "add other.py"], cwd=tmp_path, check=True, capture_output=True
    )
    # Uncommitted host edit in other.py must NOT be overwritten when worker only edits feature.py.
    (tmp_path / "other.py").write_text("LOCAL_EDIT = 2\n", encoding="utf-8")
    store, run_id = _confirmed_run(tmp_path, mode=RunMode.EXECUTION)
    original = store.get_run(run_id)["spec"]
    assert original is not None
    spec = original.model_copy(
        update={"constraints": ["Do not modify `new.lock` under any circumstances"]}
    )
    store.save_spec(run_id, spec)
    AlignStage(store=store, provider=MockLLMProvider(), workspace_root=tmp_path).confirm_spec(
        run_id=run_id, approver="zeus"
    )
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["feature.py", "other.py"],
            max_cycles=2,
            checks=[
                {
                    "id": "CHK-REV",
                    "kind": CheckKind.INDEPENDENT_REVIEW.value,
                    "description": "Review",
                }
            ],
        ),
    )

    def worker(attempt: int, ws: Path, prompt: str) -> None:
        del prompt
        (ws / "feature.py").write_text("FLAG = True\n", encoding="utf-8")
        if attempt == 1:
            (ws / "new.lock").write_text("LOCKED\n", encoding="utf-8")
        else:
            (ws / "new.lock").unlink(missing_ok=True)

    snap = _retained_stage(
        store,
        tmp_path,
        _AttemptWorker([worker]),
        provider=MockLLMProvider(responder=_passing_reviewer),
    ).run_execution(run_id=run_id, apply_to_workspace=True)

    assert snap["status"] == RunStatus.COMPLETED
    rejections = [
        e for e in store.list_events(run_id) if e.event_type == "task.diff_gate_rejected"
    ]
    assert rejections and "new.lock" in rejections[0].payload["violating_paths"]
    assert not (tmp_path / "new.lock").exists()
    assert (tmp_path / "other.py").read_text(encoding="utf-8") == "LOCAL_EDIT = 2\n"


def test_propose_plan_strips_prefilled_acceptance_check_outcomes(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path, mode=RunMode.EXECUTION)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    raw = _single_task_plan(
        spec=spec,
        write_scope=["feature.py"],
        max_cycles=1,
        checks=[
            {
                "id": "CHK-H",
                "kind": CheckKind.HUMAN.value,
                "description": "Manual verification",
                "passed": True,
                "evidence_ref": "forged://evidence",
            }
        ],
    ).model_dump(mode="json")

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del messages, tools
        return LLMResponse(content=json.dumps(raw))

    stage = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=responder),
        workspace_root=tmp_path,
    )
    proposed = stage.propose_plan(run_id=run_id, auto_approve=True)
    check = proposed.tasks[0].acceptance_checks[0]
    assert check.passed is False
    assert check.evidence_ref is None


def test_promotion_rolls_back_overlay_and_deletions_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_repo(tmp_path)
    (tmp_path / "keep.py").write_text("KEEP = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "keep.py"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "add keep.py"], cwd=tmp_path, check=True, capture_output=True
    )
    store, run_id = _confirmed_run(tmp_path, mode=RunMode.EXECUTION)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["keep.py", "added.py"],
            max_cycles=1,
            checks=[
                {
                    "id": "CHK-REV",
                    "kind": CheckKind.INDEPENDENT_REVIEW.value,
                    "description": "Review",
                }
            ],
        ),
    )

    def mutate(attempt: int, ws: Path, prompt: str) -> None:
        del attempt, prompt
        (ws / "keep.py").unlink()
        (ws / "added.py").write_text("ADDED = True\n", encoding="utf-8")

    real_copy = orchestrate_module._copy_matching_scope

    def fail_on_overlay(src: Path, dst: Path, write_scope: list[str] | None) -> None:
        if dst.name == "accepted_overlay":
            raise OSError("simulated overlay write error")
        real_copy(src, dst, write_scope)

    monkeypatch.setattr(orchestrate_module, "_copy_matching_scope", fail_on_overlay)
    snap = _retained_stage(
        store,
        tmp_path,
        _AttemptWorker([mutate]),
        provider=MockLLMProvider(responder=_passing_reviewer),
    ).run_execution(run_id=run_id, apply_to_workspace=True)

    assert snap["plan"].tasks[0].status == TaskStatus.BLOCKED
    overlay = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay"
    assert (tmp_path / "keep.py").read_text(encoding="utf-8") == "KEEP = 1\n"
    assert not (tmp_path / "added.py").exists()
    assert not (overlay / "added.py").exists()
    assert not (tmp_path / ".olympus" / "runs" / run_id / "accepted_deletions.json").exists()


def test_downstream_dag_task_starts_cleanly_after_upstream_file_to_dir_replacement(
    tmp_path: Path,
) -> None:
    from olympus.core.contracts import AcceptanceCheck

    _init_repo(tmp_path)
    (tmp_path / "foo").write_text("LEGACY_FILE = True\n", encoding="utf-8")
    subprocess.run(["git", "add", "foo"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "add foo file"], cwd=tmp_path, check=True, capture_output=True
    )
    store, run_id = _confirmed_run(tmp_path, mode=RunMode.EXECUTION)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    base_plan = _parallel_plan(
        spec, scopes={"T1": ["foo", "foo/**"], "T2": ["downstream.py"]}
    )
    plan = base_plan.model_copy(
        update={
            "tasks": [
                base_plan.tasks[0],
                base_plan.tasks[1].model_copy(
                    update={
                        "depends_on": ["T1"],
                        "max_cycles": 1,
                        "acceptance_checks": [
                            AcceptanceCheck(
                                id="C2",
                                kind=CheckKind.COMMAND,
                                description="Downstream sees foo/bar.py",
                                command="test -f foo/bar.py && test -f downstream.py",
                            )
                        ],
                    }
                ),
            ]
        }
    )
    _approve_plan(store, run_id, plan)

    class _ReplaceWorker(_DirectWriteWorker):
        def run_turn(self, request: DriverTurnRequest, **kw: Any) -> TurnOutcome:
            ws = request.workspace_path
            if request.task_id == "T1":
                (ws / "foo").unlink()
                (ws / "foo").mkdir()
                (ws / "foo" / "bar.py").write_text("BAR = True\n", encoding="utf-8")
            return super().run_turn(request, **kw)

    stage = OrchestrateStage(
        store=store,
        workspace_root=tmp_path,
        provider=MockLLMProvider(responder=_passing_reviewer),
        execution_drivers=ExecutionDriversConfig(worker_driver="direct"),
        driver_registry=DriverRegistry(
            {"direct": _ReplaceWorker({"T1": {}, "T2": {"downstream.py": "OK = True\n"}})}
        ),
    )
    snap = stage.run_execution(run_id=run_id, apply_to_workspace=True)
    assert snap["status"] == RunStatus.COMPLETED
    assert (tmp_path / "foo" / "bar.py").read_text(encoding="utf-8") == "BAR = True\n"
    assert (tmp_path / "downstream.py").read_text(encoding="utf-8") == "OK = True\n"


def test_literal_bracket_path_promotes_without_glob_colliding_siblings(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "app" / "[id]").mkdir(parents=True)
    (tmp_path / "app" / "i").mkdir(parents=True)
    (tmp_path / "app" / "[id]" / "page.tsx").write_text("OLD_DYNAMIC\n", encoding="utf-8")
    (tmp_path / "app" / "i" / "page.tsx").write_text("BASE_I\n", encoding="utf-8")
    subprocess.run(["git", "add", "app"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "add bracket and sibling routes"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    # Uncommitted host edit in app/i/page.tsx must not be clobbered by literal [id] promotion.
    (tmp_path / "app" / "i" / "page.tsx").write_text("LOCAL_I_EDIT\n", encoding="utf-8")
    store, run_id = _confirmed_run(tmp_path, mode=RunMode.EXECUTION)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["app/**"],
            max_cycles=1,
            checks=[
                {
                    "id": "CHK-REV",
                    "kind": CheckKind.INDEPENDENT_REVIEW.value,
                    "description": "Review",
                }
            ],
        ),
    )

    def edit_dynamic(attempt: int, ws: Path, prompt: str) -> None:
        del attempt, prompt
        (ws / "app" / "[id]" / "page.tsx").write_text("NEW_DYNAMIC\n", encoding="utf-8")

    snap = _retained_stage(
        store,
        tmp_path,
        _AttemptWorker([edit_dynamic]),
        provider=MockLLMProvider(responder=_passing_reviewer),
    ).run_execution(run_id=run_id, apply_to_workspace=True)

    assert snap["status"] == RunStatus.COMPLETED
    assert (tmp_path / "app" / "[id]" / "page.tsx").read_text(encoding="utf-8") == "NEW_DYNAMIC\n"
    assert (tmp_path / "app" / "i" / "page.tsx").read_text(encoding="utf-8") == "LOCAL_I_EDIT\n"


def test_stdin_python_probe_syntax_error_is_classified_as_broken(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path, mode=RunMode.EXECUTION)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["feature.py"],
            max_cycles=1,
            checks=[
                {
                    "id": "CHK-REV",
                    "kind": CheckKind.INDEPENDENT_REVIEW.value,
                    "description": "Review",
                }
            ],
        ),
    )
    provider, _ = _review_script(
        [
            {
                "task_id": "T1",
                "attempt": 1,
                "passed": False,
                "summary": "Broken stdin probe",
                "findings": [
                    {
                        "id": "F1",
                        "axis": "spec",
                        "severity": "important",
                        "file_path": "feature.py",
                        "line": 1,
                        "requirement_ref": "R1",
                        "repro_command": "printf 'def broken(:\\n' | python3 -",
                        "message": "Probe with stdin SyntaxError",
                    }
                ],
            }
        ]
    )

    def write_feature(attempt: int, ws: Path, prompt: str) -> None:
        del attempt, prompt
        (ws / "feature.py").write_text("FLAG = True\n", encoding="utf-8")

    snap = _retained_stage(
        store, tmp_path, _AttemptWorker([write_feature]), provider=provider
    ).run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.COMPLETED
    event = next(
        e.payload for e in store.list_events(run_id) if e.event_type == "review.verdict_verified"
    )
    assert event["verified_important"] == 0
    assert event["dropped_unverified"] == 1


def test_dotfile_literal_promotion_and_anchored_syntax_traceback(tmp_path: Path) -> None:
    from olympus.stages.orchestrate import (
        _copy_matching_scope,
        _probe_is_broken,
        _restore_tree_paths,
        _snapshot_tree_paths,
    )

    # 1. Dotfile ".config" must not collide with sibling "config".
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / ".config").write_text("DOTFILE\n", encoding="utf-8")
    (src / "config").write_text("SIBLING\n", encoding="utf-8")
    _copy_matching_scope(src, dst, [".config"])
    assert (dst / ".config").read_text(encoding="utf-8") == "DOTFILE\n"
    assert not (dst / "config").exists()

    # 2. Symlink snapshot & restore preserves existing symlink target on rollback.
    target_file = dst / "real.txt"
    target_file.write_text("TARGET\n", encoding="utf-8")
    link = dst / "link.txt"
    link.symlink_to("real.txt")
    snap = _snapshot_tree_paths(dst, ["link.txt"])
    link.unlink()
    link.write_text("OVERWRITTEN\n", encoding="utf-8")
    _restore_tree_paths(dst, snap)
    assert link.is_symlink() and str(link.readlink()) == "real.txt"

    # 3. Anchored traceback header ignores source-line literals of File "...".
    stdin_probe_with_literal = (
        '  File "<stdin>", line 1\n'
        '    msg = \'File "worker.py"\' def broken(:\n'
        "SyntaxError: invalid syntax\n"
    )
    assert _probe_is_broken(command="python3 -", returncode=1, output=stdin_probe_with_literal)
    worker_err_with_literal = (
        '  File "<string>", line 1, in <module>\n'
        "    import worker\n"
        '  File "/ws/worker.py", line 1\n'
        '    msg = \'File "<stdin>"\' def broken(:\n'
        "SyntaxError: invalid syntax\n"
    )
    assert not _probe_is_broken(
        command="python3 -c 'import worker'", returncode=1, output=worker_err_with_literal
    )


def test_replacing_symlink_during_promotion_never_overwrites_target_on_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_repo(tmp_path)
    (tmp_path / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "link.py").symlink_to("target.py")
    subprocess.run(
        ["git", "add", "target.py", "link.py"], cwd=tmp_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "add target and symlink"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    store, run_id = _confirmed_run(tmp_path, mode=RunMode.EXECUTION)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    _approve_plan(
        store,
        run_id,
        _single_task_plan(
            spec=spec,
            write_scope=["link.py"],
            max_cycles=1,
            checks=[
                {
                    "id": "CHK-REV",
                    "kind": CheckKind.INDEPENDENT_REVIEW.value,
                    "description": "Review",
                }
            ],
        ),
    )

    def replace_link(attempt: int, ws: Path, prompt: str) -> None:
        del attempt, prompt
        (ws / "link.py").unlink()
        (ws / "link.py").write_text("VALUE = 2\n", encoding="utf-8")

    real_copy = orchestrate_module._copy_matching_scope

    def fail_on_overlay(src: Path, dst: Path, write_scope: list[str] | None) -> None:
        if dst.name == "accepted_overlay":
            raise OSError("simulated overlay error")
        real_copy(src, dst, write_scope)

    monkeypatch.setattr(orchestrate_module, "_copy_matching_scope", fail_on_overlay)
    snap = _retained_stage(
        store,
        tmp_path,
        _AttemptWorker([replace_link]),
        provider=MockLLMProvider(responder=_passing_reviewer),
    ).run_execution(run_id=run_id, apply_to_workspace=True)

    assert snap["plan"].tasks[0].status == TaskStatus.BLOCKED
    assert (tmp_path / "link.py").is_symlink()
    assert str((tmp_path / "link.py").readlink()) == "target.py"
    assert (tmp_path / "target.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_propose_plan_lists_allowed_design_refs_in_modular_prompt(tmp_path: Path) -> None:
    from olympus.core.contracts import (
        ApprovalDecision,
        ApprovalRecord,
        ApprovalTarget,
        DataEntity,
        DesignModule,
        DesignSpec,
        InterfaceContract,
    )

    store, run_id = _confirmed_run(tmp_path)
    snap = store.get_run(run_id)
    spec = snap["spec"]
    assert spec is not None
    design = DesignSpec(
        design_id="design-allowlist",
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        architecture_summary="Two-module design",
        modules=[
            DesignModule(
                id="M-CORE",
                name="Core",
                responsibility="Core logic",
                target_paths=["feature.py"],
                requirement_refs=["R1"],
            )
        ],
        interfaces=[
            InterfaceContract(
                id="IF-RUN",
                provider_module_id="M-CORE",
                consumer_module_ids=[],
                signature="run() -> int",
                error_cases=["ValueError"],
            )
        ],
        data_models=[
            DataEntity(
                name="InternalRowModel",
                fields=["id: str"],
                owner_module_id="M-CORE",
            )
        ],
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
            reason="approved",
        )
    )

    user_prompts: list[str] = []

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        user_prompts.append(messages[-1].content)
        payload = _plan_payload(
            spec_id=spec.spec_id,
            spec_hash=spec.content_hash,
            file_name="feature.py",
            checks=[
                {
                    "id": "CHK-1",
                    "kind": CheckKind.COMMAND.value,
                    "description": "Check feature",
                    "command": "python -c 'print(1)'",
                }
            ],
        )
        payload["design_id"] = design.design_id
        payload["design_hash"] = design.content_hash
        payload["tasks"][0]["design_refs"] = ["M-CORE", "IF-RUN"]
        return LLMResponse(content=json.dumps(payload))

    orch = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=responder),
        workspace_root=tmp_path,
    )
    plan = orch.propose_plan(run_id=run_id, topology="modular_product")
    assert len(user_prompts) == 1
    assert "Allowed task.design_refs IDs" in user_prompts[0]
    assert "IF-RUN, M-CORE" in user_prompts[0]
    assert plan.tasks[0].design_refs == ["M-CORE", "IF-RUN"]


def test_format_spec_and_design_context_expand_referenced_ids() -> None:
    from olympus.core.contracts import (
        AcceptanceCheck,
        AcceptanceCriterion,
        CheckKind,
        DesignModule,
        DesignSpec,
        EarsPattern,
        GoalSpec,
        RequirementItem,
        TaskNode,
    )
    from olympus.stages.orchestrate import _format_design_context, _format_spec_context

    spec = GoalSpec(
        goal="Compute division safely",
        requirements=[
            RequirementItem(
                id="R1",
                description="Reject division by zero with a clear error message",
                acceptance_criteria=[
                    "IF divisor is zero THEN THE SYSTEM SHALL raise ValueError('division by zero')",
                    "Each division completes in under 5 ms",
                ],
                ears_criteria=[
                    AcceptanceCriterion(
                        id="R1-C1",
                        pattern=EarsPattern.UNWANTED_BEHAVIOR,
                        condition="divisor is zero",
                        outcome="raise ValueError('division by zero')",
                    )
                ],
            ),
            RequirementItem(
                id="R2",
                description="Unreferenced requirement",
                acceptance_criteria=["Never included for T1"],
            ),
        ],
    )
    design = DesignSpec(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        architecture_summary="Single calculator module",
        modules=[
            DesignModule(
                id="M-CALC",
                name="Calculator",
                responsibility="Arithmetic operations",
                requirement_refs=["R1"],
                target_paths=["calc.py"],
            )
        ],
    )
    task = TaskNode(
        task_id="T1",
        title="Implement R1",
        objective="Implement R1 in M-CALC.",
        write_scope=["calc.py"],
        requirement_refs=["R1"],
        design_refs=["M-CALC"],
        acceptance_checks=[
            AcceptanceCheck(
                id="C1",
                kind=CheckKind.COMMAND,
                description="Verify zero check",
                command="python -c 'print(1)'",
            )
        ],
    )

    spec_ctx = _format_spec_context(spec, task=task)
    assert "R1: Reject division by zero with a clear error message" in spec_ctx
    assert (
        "R1-C1: IF divisor is zero THEN THE SYSTEM SHALL raise ValueError('division by zero')"
        in spec_ctx
    )
    assert "Each division completes in under 5 ms" in spec_ctx
    assert spec_ctx.count("raise ValueError('division by zero')") == 1
    assert "Unreferenced requirement" not in spec_ctx

    design_ctx = _format_design_context(design, task=task)
    assert "M-CALC (Calculator): Arithmetic operations" in design_ctx


def test_spec_forbidden_paths_ignores_bare_pkg_names_symbols_and_prose_slashes(
    tmp_path: Path,
) -> None:
    from olympus.core.contracts import GoalSpec, RequirementItem
    from olympus.stages.orchestrate import _spec_forbidden_paths

    (tmp_path / "mypkg").mkdir()
    (tmp_path / "mypkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_frozen.py").write_text("", encoding="utf-8")
    (tmp_path / "legacy").mkdir()

    spec = GoalSpec(
        goal="Add feature to mypkg",
        requirements=[
            RequirementItem(
                id="R1",
                description="Add feature",
                acceptance_criteria=["Works"],
            )
        ],
        out_of_scope=[
            "CLI (`mypkg` command) flags or output formatting",
            "Modifying existing `mypkg.OldClass` or `stdlib.mod.Helper` behavior",
            "Changing unrelated Headers/URL semantics",
            "Do not touch `legacy/` or tests/test_frozen.py",
            "`mypkg`",
            "`mypkg.OldClass`",
            "Headers/URL",
        ],
        constraints=[
            "Test freeze: do not modify `tests/test_frozen.py` or `tests/frozen_*.py`",
            "Do not create `docs/README` or `types.d.ts`",
            "Do not modify docs/CHANGELOG or `.env`",
        ],
    )

    forbidden = _spec_forbidden_paths(spec, tmp_path)
    assert "mypkg" not in forbidden
    assert "mypkg.OldClass" not in forbidden
    assert "stdlib.mod.Helper" not in forbidden
    assert "Headers/URL" not in forbidden
    assert "legacy" in forbidden
    assert "tests/test_frozen.py" in forbidden
    assert "tests/frozen_*.py" in forbidden
    assert "docs/README" in forbidden
    assert "types.d.ts" in forbidden
    assert "docs/CHANGELOG" in forbidden
    assert ".env" in forbidden
