"""Stage 2 Orchestrate & Verify MANIFEST.md integrity artifact tests."""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import SimpleQueue
from threading import Event, Lock, Thread
from threading import enumerate as enumerate_threads
from typing import Any

import pytest
from fastapi.testclient import TestClient

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
from olympus.core.store import RunStore
from olympus.harness.driver import (
    ActiveTurnHandle,
    DriverRegistry,
    DriverTurnRequest,
    ExecutionDriversConfig,
    TurnOutcome,
)
from olympus.harness.watchdog import WatchdogConfig
from olympus.providers.base import (
    ChatMessage,
    LLMResponse,
    MockLLMProvider,
    ToolCall,
    ToolSpec,
)
from olympus.stages.align import AlignStage
from olympus.stages.orchestrate import OrchestrateStage
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
) -> tuple[RunStore, str]:
    store = RunStore(tmp_path / ".olympus" / "olympus.db")
    run_id = store.create_run(
        title="Manifest run", mode=RunMode.SIMULATION, budget_policy=budget_policy
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
    snap = orch.run_execution(run_id=run_id)

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
    snap = orch.run_execution(run_id=run_id)

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
    snap = orch.run_execution(run_id=run_id)

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
    snap = orch.run_execution(run_id=run_id)

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
            (ws / "scratch.db").write_text("stray\n", encoding="utf-8")
            return
        assert (ws / "feature.py").read_text(encoding="utf-8") == "FLAG = 1\n"
        assert (ws / "scratch.db").is_file()
        assert "IN-PLACE REPAIR" in prompt
        (ws / "scratch.db").unlink()
        (ws / "feature.py").write_text("FLAG = 2\n", encoding="utf-8")

    worker = _AttemptWorker([repair])
    snap = _retained_stage(store, tmp_path, worker).run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.COMPLETED
    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.ACCEPTED
    assert task.attempt == 2
    assert len(worker.sandboxes) == 2
    assert worker.sandboxes[0] == worker.sandboxes[1]
    rejections = _gate_rejections(store, run_id)
    assert [(e.payload["attempt"], e.payload["violating_paths"]) for e in rejections] == [
        (1, ["scratch.db"])
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
        (ws / "scratch.db").write_text("stray\n", encoding="utf-8")

    def leave_stray(attempt: int, ws: Path, prompt: str) -> None:
        del attempt, prompt
        assert (ws / "scratch.db").is_file()

    worker = _AttemptWorker([stray, leave_stray])
    snap = _retained_stage(store, tmp_path, worker).run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.AWAITING_HUMAN
    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.BLOCKED
    assert task.review_cycle == 2
    assert worker.sandboxes[0] == worker.sandboxes[1]
    assert [(e.payload["attempt"], e.payload["violating_paths"])
            for e in _gate_rejections(store, run_id)] == [
        (1, ["scratch.db"]),
        (2, ["scratch.db"]),
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
            "severity": "important",
            "file_path": "feature.py",
            "line": 1,
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
        (ws / "scratch.db").write_text("stray\n", encoding="utf-8")

    worker = _AttemptWorker([stray])
    stage = _retained_stage(store, tmp_path, worker)
    snap = stage.run_execution(run_id=run_id)
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


def test_watchdog_repeated_native_errors_steer_and_reach_ariadne(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    _approve_plan(
        store, run_id, _single_task_plan(
            spec=spec, write_scope=["feature.py"], checks=[{
                "id": "CHK-REV", "kind": "independent_review", "description": "Audit",
            }],
        )
    )
    monitor_checked, monitor_idle = Event(), Event()

    def observe(kind: str, payload: dict[str, Any]) -> None:
        if kind == "watchdog.checked":
            monitor_checked.set()

    def wait(condition, timeout):
        if monitor_checked.is_set():
            monitor_idle.set()
        condition.wait(timeout)

    def respond(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        if "Independent Verifier" in messages[0].content:
            return LLMResponse(content=json.dumps({
                "task_id": "T1", "attempt": 1, "passed": True, "summary": "Verified.",
                "findings": [],
            }))
        count = sum(message.role == "tool" for message in messages)
        if count < 3:
            return LLMResponse(
                content="PRIVATE_WORKER_TRANSCRIPT",
                tool_calls=(
                    ToolCall(str(count), "read_file", {"path": str(tmp_path / "private.txt")}),
                ),
            )
        if count == 3 and not any("Use the scoped path." in m.content for m in messages):
            assert monitor_idle.wait(3)
            return LLMResponse(content="Waiting for the next step.", tool_calls=(
                ToolCall("write", "glob_files", {"pattern": "*.py"}),
            ))
        assert any("Use the scoped path." in m.content for m in messages)
        return LLMResponse(content="Finished.")

    monitor = MockLLMProvider(responses=[LLMResponse(content=json.dumps({
        "verdict": "STUCK_LOOP", "summary": "Repeated file errors.",
        "suggested_steer": "Use the scoped path.",
    }))])
    app = AriadneApp(store=store, workspace_root=tmp_path)
    stage = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=respond),
        workspace_root=tmp_path,
        monitor_provider=monitor,
        watchdog_config=WatchdogConfig(wait=wait),
        on_driver_event=observe,
    )
    assert stage.run_execution(run_id=run_id)["status"] == RunStatus.COMPLETED
    assert len(monitor.recorded_requests) == 1
    (messages, tools), = monitor.recorded_requests
    assert not tools
    snapshot = json.loads(messages[-1].content)
    assert set(snapshot) == {"objective", "write_scope", "milestones", "diff_stat"}
    assert snapshot["write_scope"] == ["feature.py"]
    assert "PRIVATE_WORKER_TRANSCRIPT" not in messages[-1].content
    assert "private.txt" not in messages[-1].content
    events = store.list_events(run_id)
    assert any(
        e.event_type == "agent.steered" and e.payload["source"] == "watchdog" for e in events
    )
    progress = app.get_run_payload(run_id)["projection"]["dag"]["nodes"][0]["live_progress"]
    assert progress["watchdog_verdict"] == "STUCK_LOOP"
    assert progress["watchdog_summary"] == "Repeated file errors."


@pytest.mark.parametrize(("verdicts", "archive_fails"), [
    (("SCOPE_DRIFT", "STUCK_LOOP"), False),
    (("BLOCKED_EXTERNAL",), False),
    (("BLOCKED_EXTERNAL",), True),
])
def test_watchdog_pause_preserves_work_and_skips_review(
    tmp_path: Path, verdicts: tuple[str, ...], archive_fails: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if archive_fails:
        def fail_archive(*args):
            raise PermissionError("Cannot write patch")

        monkeypatch.setattr("olympus.stages.orchestrate.archive_sandbox_recovery", fail_archive)
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    _approve_plan(store, run_id, _single_task_plan(
        spec=spec, write_scope=["feature.py"], checks=[_marker_command_check()],
    ))
    checked = Event()
    cancelled = Event()
    observer_events: list[tuple[str, dict[str, Any]]] = []

    class JsonDriver:
        name = "test-json"
        supports_steer = False

        def steer(self, guidance, *, source):
            pytest.fail("JSON-only driver cannot steer")

        def cancel(self, *, reason):
            assert reason == "watchdog_paused"
            cancelled.set()

        def run_turn(self, request, *, on_event, on_handle):
            on_handle(self)
            (request.workspace_path / "feature.py").write_bytes(b"\x00unfinished\n")
            (request.workspace_path / "README.md").write_text("PRIVATE_PATCH_BODY\n")
            request.trace_path.write_text('{"event":"worker-progress"}\n', encoding="utf-8")
            for i in range(len(verdicts)):
                for step in range(3):
                    on_event("agent.tool_finished", {
                        "step": step, "tool": "write", "status": "error",
                        "error_signature": "repeat",
                    })
                if i + 1 < len(verdicts):
                    assert checked.wait(3)
                    assert not cancelled.is_set()
            assert cancelled.wait(3)
            return TurnOutcome(False, "cancelled", "", 1)

    def observe(kind: str, payload: dict[str, Any]) -> None:
        observer_events.append((kind, payload))
        if kind == "watchdog.checked":
            checked.set()

    monitor = MockLLMProvider(responses=[LLMResponse(content=json.dumps({
        "verdict": verdict, "summary": f"Diagnosis {verdict}.",
        "suggested_steer": "Try the scoped implementation.",
    })) for verdict in verdicts])
    primary = MockLLMProvider()
    stage = OrchestrateStage(
        store=store, provider=primary, workspace_root=tmp_path,
        monitor_provider=monitor, on_driver_event=observe,
        execution_drivers=ExecutionDriversConfig(worker_driver="test-json"),
        driver_registry=DriverRegistry({"test-json": JsonDriver()}),
    )
    result = stage.run_execution(run_id=run_id)
    assert result["status"] == RunStatus.PAUSED
    assert cancelled.is_set()
    assert not any(t.name == "olympus-watchdog" for t in enumerate_threads())
    assert not primary.recorded_requests
    assert len(monitor.recorded_requests) == len(verdicts)
    for messages, _ in monitor.recorded_requests:
        assert "README.md" in json.loads(messages[-1].content)["diff_stat"]
        assert "PRIVATE_PATCH_BODY" not in messages[-1].content
    assert [payload["severity"] for kind, payload in observer_events
            if kind == "watchdog.checked"] == ["warning"] * (len(verdicts) - 1)
    task = result["plan"].tasks[0]
    assert task.status == TaskStatus.BLOCKED
    assert task.review_cycle == 0
    assert task.last_error == f"Diagnosis {verdicts[-1]}."
    attempt = tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "T1" / "attempt-1"
    assert (attempt / "worker.jsonl").is_file()
    assert json.loads((attempt / "watchdog.json").read_text())["summary"] == task.last_error
    events = store.list_events(run_id)
    paused, = [event for event in events if event.event_type == "watchdog.paused"]
    # A paused attempt keeps its sandbox on disk so retry_task can resume in place.
    assert task.worktree_path == paused.payload["retained_sandbox"]
    retained = Path(task.worktree_path)
    assert retained.is_dir()
    assert (retained / "feature.py").read_bytes() == b"\x00unfinished\n"
    if archive_fails:
        assert paused.payload["archive_error"] == "PermissionError"
    else:
        patch = attempt / "diff.patch"
        assert "GIT binary patch" in patch.read_text(encoding="utf-8")
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=tmp_path, check=True)
    assert any(e.event_type == "run.paused" for e in events)
    progress = AriadneApp(store=store, workspace_root=tmp_path).get_run_payload(run_id)[
        "projection"
    ]["dag"]["nodes"][0]["live_progress"]
    assert progress["watchdog_verdict"] == verdicts[-1]
    assert progress["watchdog_summary"] == task.last_error


def test_watchdog_recovery_resets_streak_without_resetting_steer_cap(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    _approve_plan(store, run_id, _single_task_plan(
        spec=spec, write_scope=["feature.py"], checks=[{
            "id": "CHK-REV", "kind": "independent_review", "description": "Audit the change",
        }],
    ))
    verdicts = ["STUCK_LOOP", "HEALTHY", "SCOPE_DRIFT", "HEALTHY", "STUCK_LOOP"]
    checked = SimpleQueue()
    steers = []

    class RecoveringDriver:
        name = "recovering"
        supports_steer = True

        def steer(self, guidance, *, source):
            steers.append((guidance, source))
            return True

        def cancel(self, *, reason):
            pytest.fail(f"Recovering task unexpectedly cancelled: {reason}")

        def run_turn(self, request, *, on_event, on_handle):
            on_handle(self)
            for expected in verdicts:
                on_event("agent.tool_finished", {
                    "tool": "read", "step": 1, "status": "error", "error_signature": "retry",
                })
                assert checked.get(timeout=3) == expected
            return TurnOutcome(True, "completed", "", 1)

    stage = OrchestrateStage(
        store=store, workspace_root=tmp_path,
        provider=MockLLMProvider(responses=[LLMResponse(content=(
            '{"task_id":"T1","attempt":1,"passed":true,"summary":"Verified."}'
        ))]),
        driver_registry=DriverRegistry({"recovering": RecoveringDriver()}),
        execution_drivers=ExecutionDriversConfig(worker_driver="recovering"),
        monitor_provider=MockLLMProvider(responses=[LLMResponse(content=json.dumps({
            "verdict": verdict, "summary": f"Diagnosis {verdict}.",
            "suggested_steer": "Review the scoped paths.",
        })) for verdict in verdicts]),
        watchdog_config=WatchdogConfig(repeat_error_threshold=1),
        on_driver_event=lambda kind, fields: (
            checked.put(fields["verdict"]) if kind == "watchdog.checked" else None
        ),
    )
    assert stage.run_execution(run_id=run_id)["status"] == RunStatus.COMPLETED
    assert steers == [("Review the scoped paths.", "watchdog")] * 2
    assert not any(t.name == "olympus-watchdog" for t in enumerate_threads())


@pytest.mark.parametrize(("trigger", "provider_reply", "event_type"), [
    (None, None, None),
    ("worker_failure", None, None),
    ("silence", TimeoutError("secret timeout detail"), "watchdog.error"),
    ("interval", RuntimeError("secret provider detail"), "watchdog.error"),
    ("silence", '{"verdict":"HEALTHY","summary":"two\\nlines"}', "watchdog.error"),
    ("interval", '{"verdict":"HEALTHY","summary":"Progress is steady."}', "watchdog.checked"),
    ("interval_then_silence", '{"verdict":"HEALTHY","summary":"OK"}', "watchdog.checked"),
])
def test_watchdog_timers_fail_open_through_stage(
    tmp_path: Path, trigger: str | None, provider_reply: str | Exception | None,
    event_type: str | None,
) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    _approve_plan(store, run_id, _single_task_plan(
        spec=spec, write_scope=["feature.py"], checks=[{
            "id": "CHK-REV", "kind": "independent_review", "description": "Audit the change",
        }],
    ))
    observed = Event()
    observed_triggers: list[str] = []
    now = 0.0
    pending = int(event_type is not None)
    expected_triggers = [trigger] if pending else []
    if trigger == "interval_then_silence":
        expected_triggers = ["interval", "silence"]
        pending = 2

    def wait(condition, timeout):
        nonlocal now, pending
        if pending:
            now += timeout
            pending -= 1
        else:
            condition.wait()

    def observe(kind, fields):
        if kind == event_type:
            observed_triggers.append(fields["trigger"])
            if len(observed_triggers) == len(expected_triggers):
                observed.set()

    def monitor_response(messages, tools):
        if isinstance(provider_reply, Exception):
            raise provider_reply
        assert provider_reply is not None, "A fast turn must leave L2 idle"
        return LLMResponse(content=provider_reply)

    class FastDriver:
        name = "fast"
        supports_steer = True

        def steer(self, guidance, *, source):
            pytest.fail("Unexpected steering")

        def cancel(self, *, reason):
            pytest.fail("Unexpected cancellation")

        def run_turn(self, request, *, on_event, on_handle):
            on_handle(self)
            if trigger == "worker_failure":
                raise ValueError("worker failed")
            if provider_reply is not None:
                assert observed.wait(3)
            return TurnOutcome(True, "completed", "", 1)

    monitor = MockLLMProvider(responder=monitor_response)
    stage = OrchestrateStage(
        store=store, workspace_root=tmp_path,
        provider=MockLLMProvider(responses=[LLMResponse(content=(
            '{"task_id":"T1","attempt":1,"passed":true,"summary":"Verified."}'
        ))]),
        driver_registry=DriverRegistry({"fast": FastDriver()}),
        execution_drivers=ExecutionDriversConfig(worker_driver="fast", monitor_model="monitor"),
        model_provider=lambda model: {"monitor": monitor}[model],
        watchdog_config=WatchdogConfig(
            silence_seconds=15 if trigger and trigger.startswith("interval") else 10,
            interval_seconds=20 if trigger == "silence" else 10,
            clock=lambda: now, wait=wait,
        ),
        on_driver_event=observe,
    )
    if trigger == "worker_failure":
        with pytest.raises(ValueError, match="worker failed"):
            stage.run_execution(run_id=run_id)
    else:
        assert stage.run_execution(run_id=run_id)["status"] == RunStatus.COMPLETED
    watchdog_events = [
        e
        for e in store.list_events(run_id)
        if e.actor == "watchdog" and e.event_type.startswith("watchdog.")
    ]
    assert len(monitor.recorded_requests) == len(expected_triggers)
    assert [event.event_type for event in watchdog_events] == [event_type] * len(expected_triggers)
    assert [event.payload["trigger"] for event in watchdog_events] == expected_triggers
    assert all("secret" not in json.dumps(event.payload) for event in watchdog_events)
    assert not any(t.name == "olympus-watchdog" for t in enumerate_threads())



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
        out = stage.run_execution(run_id=run_id)
        assert out["plan"].tasks[0].status == TaskStatus.BLOCKED
        rejections = [
            e for e in store.list_events(run_id) if e.event_type == "task.diff_gate_rejected"
        ]
        assert len(rejections) == 1
        assert rel in rejections[0].payload["violating_paths"]
        assert not (ws / ".olympus" / "runs" / run_id / "accepted_overlay" / rel).exists()


def test_watchdog_slow_monitor_never_stalls_completed_worker(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    plan = _single_task_plan(
        spec=store.get_run(run_id)["spec"],
        write_scope=["feature.py"],
        checks=[_marker_command_check()],
    )
    _approve_plan(store, run_id, plan)
    entered, release, finished = Event(), Event(), Event()

    class FastDriver:
        name = "fast"
        supports_steer = False

        def cancel(self, *, reason: str) -> None:
            del reason

        def steer(self, guidance: str, *, source: str) -> bool:
            del guidance, source
            return False

        def run_turn(
            self,
            request: DriverTurnRequest,
            *,
            on_event: Any = None,
            on_handle: Any = None,
        ) -> TurnOutcome:
            del on_event
            if on_handle is not None:
                on_handle(self)
            assert entered.wait(3.0)
            (request.workspace_path / "feature.py").write_text("FLAG = True\n", encoding="utf-8")
            finished.set()
            return TurnOutcome(
                completed=True,
                stop_reason="final_response",
                final_content="done",
                steps_taken=1,
                driver_name="fast",
            )

    class SlowMonitor(MockLLMProvider):
        def complete(self, *args: Any, **kwargs: Any) -> LLMResponse:
            del args, kwargs
            entered.set()
            release.wait(5.0)
            return LLMResponse(
                content=json.dumps({"verdict": "BLOCKED_EXTERNAL", "summary": "Too late"})
            )

    verdict = json.dumps(
        {
            "task_id": "T1",
            "attempt": 1,
            "passed": True,
            "summary": "ok",
            "findings": [],
        }
    )
    stage = OrchestrateStage(
        store=store,
        workspace_root=tmp_path,
        provider=MockLLMProvider(responses=[LLMResponse(content=verdict)]),
        monitor_provider=SlowMonitor(),
        watchdog_config=WatchdogConfig(interval_seconds=0.01, stop_join_timeout_seconds=0.05),
        execution_drivers=ExecutionDriversConfig(worker_driver="fast"),
        driver_registry=DriverRegistry({"fast": FastDriver()}),
    )
    runner = Thread(target=lambda: stage.run_execution(run_id=run_id))
    runner.start()
    try:
        assert finished.wait(3.0)
        runner.join(1.5)
        assert not runner.is_alive()
        assert store.get_run(run_id)["plan"].tasks[0].status == TaskStatus.ACCEPTED
    finally:
        release.set()
        runner.join(5.0)


def test_watchdog_pause_blocks_concurrent_sibling_acceptance_and_overlay(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    plan = _single_task_plan(
        spec=store.get_run(run_id)["spec"],
        write_scope=["feature.py"],
        checks=[_marker_command_check()],
    )
    t1 = plan.tasks[0]
    t2 = t1.model_copy(
        update={"task_id": "T2", "objective": "Second worker", "write_scope": ["other.py"]}
    )
    _approve_plan(store, run_id, plan.model_copy(update={"tasks": [t1, t2]}))
    peer_running, t1_cancelled = Event(), Event()

    class PeerHandle:
        supports_steer = False

        def __init__(self, flag: Event) -> None:
            self._flag = flag

        def steer(self, guidance: str, *, source: str) -> bool:
            del guidance, source
            return False

        def cancel(self, *, reason: str) -> None:
            del reason
            self._flag.set()

    class ConcurrentDriver:
        name = "concurrent"

        def run_turn(
            self,
            request: DriverTurnRequest,
            *,
            on_event: Any = None,
            on_handle: Any = None,
        ) -> TurnOutcome:
            handle = PeerHandle(t1_cancelled if request.task_id == "T1" else Event())
            if on_handle is not None:
                on_handle(handle)
            if request.task_id == "T1":
                assert peer_running.wait(3.0)
                if on_event is not None:
                    on_event(
                        "agent.tool_finished",
                        {
                            "tool": "run_bash",
                            "step": 1,
                            "status": "error",
                            "error_signature": "sha256:" + "a" * 64,
                        },
                    )
                assert t1_cancelled.wait(3.0)
                return TurnOutcome(
                    completed=False,
                    stop_reason="cancelled",
                    final_content="",
                    steps_taken=1,
                    driver_name="concurrent",
                )
            peer_running.set()
            deadline = time.monotonic() + 3.0
            while (
                store.get_run(run_id)["status"] != RunStatus.PAUSED
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert store.get_run(run_id)["status"] == RunStatus.PAUSED
            (request.workspace_path / "other.py").write_text("late write\n", encoding="utf-8")
            return TurnOutcome(
                completed=True,
                stop_reason="final_response",
                final_content="done",
                steps_taken=1,
                driver_name="concurrent",
            )

    verdict = json.dumps(
        {"task_id": "T2", "attempt": 1, "passed": True, "summary": "ok", "findings": []}
    )
    stage = OrchestrateStage(
        store=store,
        workspace_root=tmp_path,
        provider=MockLLMProvider(responses=[LLMResponse(content=verdict)]),
        monitor_provider=MockLLMProvider(
            responses=[
                LLMResponse(
                    content=json.dumps({"verdict": "BLOCKED_EXTERNAL", "summary": "Needs operator"})
                )
            ]
        ),
        watchdog_config=WatchdogConfig(repeat_error_threshold=1),
        execution_drivers=ExecutionDriversConfig(worker_driver="concurrent"),
        driver_registry=DriverRegistry({"concurrent": ConcurrentDriver()}),
    )
    out = stage.run_execution(run_id=run_id)
    assert out["status"] == RunStatus.PAUSED
    assert not [e for e in store.list_events(run_id) if e.event_type == "task.accepted"]
    assert not (tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay" / "other.py").exists()



def test_watchdog_buckets_error_signatures_and_parses_fenced_json(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    plan = _single_task_plan(
        spec=store.get_run(run_id)["spec"],
        write_scope=["feature.py"],
        checks=[_marker_command_check()],
    )
    _approve_plan(store, run_id, plan)
    seen_payloads: list[dict[str, Any]] = []
    checked = Event()

    class ErrorThenFinishDriver:
        name = "err"
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
            if on_handle is not None:
                on_handle(self)
            assert on_event is not None
            on_event(
                "agent.tool_finished",
                {
                    "tool": "write",
                    "step": 1,
                    "status": "error",
                    "error_signature": "sha256:" + "9" * 64,
                },
            )
            assert checked.wait(3.0)
            (request.workspace_path / "feature.py").write_text("FLAG = True\n", encoding="utf-8")
            return TurnOutcome(
                completed=True,
                stop_reason="final_response",
                final_content="done",
                steps_taken=1,
                driver_name="err",
            )

    class FencedMonitor(MockLLMProvider):
        def complete(self, *args: Any, **kwargs: Any) -> LLMResponse:
            del args
            messages = kwargs["messages"]
            seen_payloads.append(json.loads(messages[-1].content))
            return LLMResponse(
                content='```json\n{"verdict":"HEALTHY","summary":"Recovered from first error"}\n```'
            )

    def on_event(kind: str, payload: dict[str, Any]) -> None:
        del payload
        if kind == "watchdog.checked":
            checked.set()

    verdict = json.dumps(
        {"task_id": "T1", "attempt": 1, "passed": True, "summary": "ok", "findings": []}
    )
    stage = OrchestrateStage(
        store=store,
        workspace_root=tmp_path,
        provider=MockLLMProvider(responses=[LLMResponse(content=verdict)]),
        monitor_provider=FencedMonitor(),
        watchdog_config=WatchdogConfig(repeat_error_threshold=1),
        execution_drivers=ExecutionDriversConfig(worker_driver="err"),
        driver_registry=DriverRegistry({"err": ErrorThenFinishDriver()}),
        on_driver_event=on_event,
    )
    out = stage.run_execution(run_id=run_id)
    assert out["status"] == RunStatus.COMPLETED
    assert seen_payloads
    raw_snapshot = json.dumps(seen_payloads[0])
    assert "sha256:" not in raw_snapshot
    assert seen_payloads[0]["milestones"][0]["error_bucket"] == "err#1"
    watchdog_events = [e for e in store.list_events(run_id) if e.event_type == "watchdog.checked"]
    assert len(watchdog_events) == 1
    assert watchdog_events[0].payload["verdict"] == "HEALTHY"


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

    snap = _retained_stage(store, tmp_path, _AttemptWorker([worker])).run_execution(run_id=run_id)
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


def test_watchdog_pause_cancels_late_registering_peer_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(
        tmp_path, budget_policy=BudgetPolicy(max_concurrent_tasks=2)
    )
    plan = _parallel_plan(
        store.get_run(run_id)["spec"], scopes={"T1": ["one.txt"], "T2": ["two.txt"]}
    )
    _approve_plan(store, run_id, plan)
    peer_entered, pause_processed = Event(), Event()
    peer_cancelled, source_cancelled = Event(), Event()

    class _Handle:
        supports_steer = False

        def __init__(self, event: Event) -> None:
            self._event = event

        def steer(self, guidance: str, *, source: str) -> bool:
            del guidance, source
            return False

        def cancel(self, *, reason: str) -> None:
            del reason
            self._event.set()

    class _PeerDriver:
        name = "late-peer"

        def run_turn(
            self,
            request: DriverTurnRequest,
            *,
            on_event: Any = None,
            on_handle: Any = None,
        ) -> TurnOutcome:
            if request.task_id == "T1":
                on_handle(_Handle(source_cancelled))
                assert peer_entered.wait(5)
                for step in range(3):
                    on_event(
                        "agent.tool_finished",
                        {"step": step, "tool": "write", "status": "error", "error_signature": "r"},
                    )
                assert source_cancelled.wait(5)
                return TurnOutcome(False, "cancelled", "", 1)
            peer_entered.set()
            assert pause_processed.wait(5)
            on_handle(_Handle(peer_cancelled))
            assert peer_cancelled.wait(2)
            return TurnOutcome(False, "cancelled", "", 1)

    stage = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=_passing_reviewer),
        monitor_provider=MockLLMProvider(
            responses=[
                LLMResponse(
                    content=json.dumps({"verdict": "BLOCKED_EXTERNAL", "summary": "Blocked."})
                )
            ]
        ),
        workspace_root=tmp_path,
        execution_drivers=ExecutionDriversConfig(worker_driver="late-peer"),
        driver_registry=DriverRegistry({"late-peer": _PeerDriver()}),
    )
    orig_exec = stage._execute_single_task

    def coordinated_exec(**kwargs: Any) -> Any:
        result = orig_exec(**kwargs)
        if kwargs["task_id"] == "T1":
            pause_processed.set()
        return result

    monkeypatch.setattr(stage, "_execute_single_task", coordinated_exec)
    snap = stage.run_execution(run_id=run_id)
    assert snap["status"] == RunStatus.PAUSED
    assert peer_cancelled.is_set()


def test_nongit_watchdog_pause_preserves_retained_sandbox_for_retry(tmp_path: Path) -> None:
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
    cancelled = Event()
    seen_prior: list[bool] = []

    class _NonGitDriver:
        name = "nongit"
        supports_steer = False

        def steer(self, guidance: str, *, source: str) -> bool:
            del guidance, source
            return False

        def cancel(self, *, reason: str) -> None:
            del reason
            cancelled.set()

        def run_turn(
            self,
            request: DriverTurnRequest,
            *,
            on_event: Any = None,
            on_handle: Any = None,
        ) -> TurnOutcome:
            ws = request.workspace_path
            on_handle(self)
            seen_prior.append((ws / "feature.py").exists())
            if len(seen_prior) == 1:
                (ws / "feature.py").write_text("FLAG = 1\n", encoding="utf-8")
                for step in range(3):
                    on_event(
                        "agent.tool_finished",
                        {"step": step, "tool": "write", "status": "error", "error_signature": "e"},
                    )
                assert cancelled.wait(3)
                return TurnOutcome(False, "cancelled", "", 1)
            return TurnOutcome(True, "final_response", "done", 1)

    stage = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=_passing_reviewer),
        monitor_provider=MockLLMProvider(
            responses=[
                LLMResponse(
                    content=json.dumps({"verdict": "BLOCKED_EXTERNAL", "summary": "Blocked."})
                )
            ]
        ),
        workspace_root=tmp_path,
        execution_drivers=ExecutionDriversConfig(worker_driver="nongit"),
        driver_registry=DriverRegistry({"nongit": _NonGitDriver()}),
    )
    paused = stage.run_execution(run_id=run_id)
    assert paused["status"] == RunStatus.PAUSED
    stage.retry_blocked_task(
        operation_id="retry-nongit",
        run_id=run_id,
        task_id="T1",
        expected_state_revision=paused["state_revision"],
    )
    store.set_run_status(run_id, RunStatus.RUNNING, actor="zeus")
    finished = stage.run_execution(run_id=run_id)
    assert finished["status"] == RunStatus.COMPLETED
    assert seen_prior == [False, True]





