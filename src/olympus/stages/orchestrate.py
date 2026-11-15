"""Stage 2 (Orchestrate & Verify): Adaptive Task DAG, sandbox execution, and cold-start review."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from olympus.core.contracts import (
    AcceptanceCheck,
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    CheckKind,
    GoalSpec,
    ReviewSeverity,
    ReviewVerdict,
    RunMode,
    RunStatus,
    TaskNode,
    TaskPlan,
    TaskStatus,
)
from olympus.core.dag import (
    PlanValidationError,
    compute_dispatchable_tasks,
    paths_overlap,
    validate_plan,
)
from olympus.core.store import RunStore
from olympus.harness.loop import AgentLoop
from olympus.harness.worktree import (
    SandboxHandle,
    build_sandbox_env,
    capture_sandbox_diff,
    create_sandbox,
    remove_sandbox,
)
from olympus.providers.base import LLMProvider
from olympus.stages.common import format_workspace_inventory, run_structured_loop
from olympus.tools.aci import _IGNORED_DIR_NAMES, WorkspaceToolkit

_PLANNER_SYSTEM_PROMPT = """You are the Stage 2 Task Planner of Olympus4SWE.
Decompose the confirmed GoalSpec into a minimal TaskPlan JSON.
Rules:
1. Keep simple tasks as a single TaskNode; split into a DAG ONLY when modules are truly separable.
2. Tasks that can run concurrently (neither depends on the other) MUST have disjoint write_scope.
3. Every requirement in GoalSpec must appear in at least one task's requirement_refs.
4. Keep tasks cohesive (1-3 checks per task, not redundant prose checks).
   Valid acceptance_checks[*].kind values are strictly: 'command', 'independent_review', or 'human'
   (NEVER invent values like 'review'). Whenever kind='command', supply a non-empty shell string
   in 'command' (e.g. 'python -m pytest ...'); if there is no shell command, use
   kind='independent_review' instead.
5. Acceptance check commands execute with cwd ALREADY set to an isolated sandbox copy of the repo.
   NEVER use 'cd <host_path>' or hardcoded host workspace paths inside check commands; use
   repo-relative paths only.
6. Honor all GoalSpec constraints and resolved open_ambiguities answers in task objectives/checks.
7. Output ONLY a single JSON object matching TaskPlan (no $comment, $schema, or extra keys)."""

_WORKER_SYSTEM_PROMPT = """You are the Stage 2 Execution Engineer (Hephaestus) of Olympus4SWE.
Implement the assigned TaskNode inside the isolated sandbox workspace.
Rules:
1. Read relevant files first using absolute paths under the sandbox root.
2. Only write or edit files inside your assigned write_scope.
3. Run tests/checks with run_bash and observe actual outputs before finishing.
4. Do NOT claim completion without verified tool outputs. If prior review findings are provided,
   fix every Important finding before concluding."""

_REVIEWER_SYSTEM_PROMPT = """You are the cold-start Independent Verifier (Argus) of Olympus4SWE.
You have ZERO prior conversation history with the implementer.
Inspect the GoalSpec requirements, out_of_scope boundaries, TaskNode objective, git diff, and actual
command check outputs (plus read-only workspace inspection if needed).
Rules:
1. Check for spec drift, out_of_scope additions, stub code, edge-case bugs, and security risks.
2. Classify each finding strictly as 'important' (blocks pass), 'nit', or 'pre_existing'.
3. Set passed=false whenever any 'important' finding exists; set passed=true only if implementation
   genuinely satisfies the task and no 'important' defects remain.
4. Output ONLY a single JSON object conforming to ReviewVerdict."""


def _copy_matching_scope(
    src_root: Path,
    dst_root: Path,
    write_scope: list[str] | None,
) -> None:
    """Copy non-ignored files from src_root to dst_root, optionally filtered by write_scope."""
    if not src_root.exists():
        return
    for p in sorted(src_root.rglob("*")):
        if not p.is_file():
            continue
        rel_path = p.relative_to(src_root)
        ignored = _IGNORED_DIR_NAMES | {".olympus"}
        if any(part in ignored for part in rel_path.parts):
            continue
        rel = rel_path.as_posix()
        if write_scope is not None and not any(paths_overlap(rel, scope) for scope in write_scope):
            continue
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)


def _format_spec_context(spec: GoalSpec) -> str:
    """Format constraints, assumptions, and resolved Stage 1 clarifications for Stage 2 roles."""
    clarifications = [
        {"id": amb.id, "question": amb.question, "answer": amb.answer}
        for amb in spec.open_ambiguities
        if amb.answer is not None and amb.answer.strip()
    ]
    unresolved = [
        {"id": amb.id, "question": amb.question, "blocking": amb.blocking}
        for amb in spec.open_ambiguities
        if amb.answer is None or not amb.answer.strip()
    ]
    lines = [
        f"Constraints: {json.dumps(spec.constraints, ensure_ascii=False)}",
        f"Assumptions: {json.dumps(spec.assumptions, ensure_ascii=False)}",
        f"Clarifications (resolved Q&A): {json.dumps(clarifications, ensure_ascii=False)}",
    ]
    if unresolved:
        lines.append(
            f"Unresolved ambiguities (accepted risk): {json.dumps(unresolved, ensure_ascii=False)}"
        )
    return "\n".join(lines)


class OrchestrateStage:
    """Deterministic Chronos scheduler driving isolated Hephaestus execution & cold-start review."""

    def __init__(
        self,
        *,
        store: RunStore,
        provider: LLMProvider,
        workspace_root: Path | str,
        artifacts_root: Path | None = None,
    ) -> None:
        self.store = store
        self.provider = provider
        self.workspace_root = Path(workspace_root).resolve()
        self.artifacts_root = (
            artifacts_root
            if artifacts_root is not None
            else self.workspace_root / ".olympus" / "runs"
        )

    def _run_dir(self, run_id: str) -> Path:
        return self.artifacts_root / run_id

    def _require_confirmed_spec(self, run_id: str) -> GoalSpec:
        snap = self.store.get_run(run_id)
        spec: GoalSpec | None = snap["spec"]
        if spec is None:
            raise ValueError(f"Run {run_id} has no GoalSpec; complete Stage 1 (Align) first.")
        if not self.store.is_approved(
            run_id=run_id,
            target=ApprovalTarget.GOAL_SPEC,
            target_id=spec.spec_id,
            target_hash=spec.content_hash,
        ):
            raise ValueError(
                f"GoalSpec {spec.spec_id} ({spec.content_hash}) is not human-approved."
            )
        return spec

    def propose_plan(
        self,
        *,
        run_id: str,
        auto_approve: bool = False,
        approver: str = "zeus",
        max_steps: int = 10,
    ) -> TaskPlan:
        """Generate and deterministically validate a TaskPlan for an approved GoalSpec."""
        spec = self._require_confirmed_spec(run_id)
        toolkit = WorkspaceToolkit(self.workspace_root, read_only=True)
        loop = AgentLoop(
            provider=self.provider,
            toolkit=toolkit,
            store=self.store,
            run_id=run_id,
            actor="orchestrate:planner",
            max_steps=max_steps,
            trace_path=self._run_dir(run_id) / "traces" / "planner.jsonl",
        )
        planner_toolkit = WorkspaceToolkit(self.workspace_root, read_only=True)
        inventory = format_workspace_inventory(planner_toolkit)
        schema_json = json.dumps(TaskPlan.model_json_schema(), ensure_ascii=False)
        base_prompt = (
            f"Workspace root: {self.workspace_root}\n"
            f"{inventory}\n"
            f"Note: acceptance_checks[*].command will run in an isolated sandbox worktree "
            f"(cwd = sandbox root). NEVER reference '{self.workspace_root}' inside commands.\n"
            f"Remember: acceptance_checks[*].kind MUST be one of "
            f"'command', 'independent_review', or 'human' (never 'review'), and every "
            f"kind='command' check MUST include a non-empty 'command' string.\n"
            f"Confirmed GoalSpec:\n{spec.model_dump_json(indent=2)}\n\n"
            f"Set spec_id='{spec.spec_id}' and spec_hash='{spec.content_hash}'.\n"
            f"Return TaskPlan JSON matching schema:\n{schema_json}"
        )

        prompt = base_prompt
        plan: TaskPlan | None = None
        last_violations: list[str] = []
        for attempt in range(2):
            candidate = run_structured_loop(
                loop=loop,
                system_prompt=_PLANNER_SYSTEM_PROMPT,
                user_prompt=prompt,
                model_cls=TaskPlan,
            ).model_copy(update={"spec_id": spec.spec_id, "spec_hash": spec.content_hash})
            violations = validate_plan(candidate, spec=spec, workspace_root=self.workspace_root)
            if not violations:
                plan = candidate
                break
            last_violations = violations
            if attempt == 0:
                prompt = (
                    f"{base_prompt}\n\n"
                    f"[PLAN VALIDATION FAILED]\n"
                    f"Violations: {'; '.join(violations)}\n"
                    "Fix all violations and return valid TaskPlan JSON."
                )
        if plan is None:
            raise PlanValidationError(last_violations)

        self.store.save_plan(run_id, plan, actor="orchestrate")
        if auto_approve:
            self.approve_plan(run_id=run_id, approver=approver, reason="auto_approve flag")
        else:
            self.store.set_run_status(
                run_id,
                RunStatus.AWAITING_HUMAN,
                actor="orchestrate",
                stage="orchestrate",
                detail={"plan_id": plan.plan_id, "plan_hash": plan.content_hash},
            )
        return plan

    def approve_plan(
        self,
        *,
        run_id: str,
        approver: str = "zeus",
        reason: str = "",
    ) -> ApprovalRecord:
        """Record human approval bound to the current TaskPlan content_hash."""
        spec = self._require_confirmed_spec(run_id)
        snap = self.store.get_run(run_id)
        plan: TaskPlan | None = snap["plan"]
        if plan is None:
            raise ValueError(f"Run {run_id} has no TaskPlan to approve.")
        if plan.spec_hash != spec.content_hash:
            raise ValueError(
                f"TaskPlan spec_hash ({plan.spec_hash}) does not match confirmed spec "
                f"({spec.content_hash}); regenerate plan first."
            )
        record = ApprovalRecord(
            run_id=run_id,
            target=ApprovalTarget.TASK_PLAN,
            target_id=plan.plan_id,
            target_hash=plan.content_hash,
            approver=approver,
            decision=ApprovalDecision.APPROVED,
            reason=reason,
        )
        self.store.record_approval(record)
        return record

    def _run_command_checks(
        self,
        *,
        run_id: str,
        task: TaskNode,
        sandbox: SandboxHandle,
        art_dir: Path,
    ) -> tuple[list[AcceptanceCheck], list[str]]:
        updated_checks: list[AcceptanceCheck] = []
        failures: list[str] = []
        for idx, check in enumerate(task.acceptance_checks, start=1):
            if check.kind != CheckKind.COMMAND:
                updated_checks.append(check)
                continue
            assert check.command is not None
            sandbox_env = build_sandbox_env(
                sandbox.workspace_path,
                host_root=self.workspace_root,
            )
            proc = subprocess.run(
                ["bash", "-lc", check.command],
                cwd=sandbox.workspace_path,
                env=sandbox_env,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            log_file = art_dir / f"check-{idx}-{check.id}.log"
            log_text = (
                f"command: {check.command}\n"
                f"exit_code: {proc.returncode}\n"
                f"--- stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}\n"
            )
            log_file.write_text(log_text, encoding="utf-8")
            passed = proc.returncode == 0
            updated_checks.append(
                check.model_copy(update={"passed": passed, "evidence_ref": str(log_file)})
            )
            if not passed:
                failures.append(
                    f"Command check {check.id} (`{check.command}`) failed with code "
                    f"{proc.returncode}:\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}"
                )
            self.store.append_event(
                run_id=run_id,
                stage="verify",
                event_type="check.command_completed",
                actor="chronos",
                payload={
                    "task_id": task.task_id,
                    "check_id": check.id,
                    "passed": passed,
                    "exit_code": proc.returncode,
                },
            )
        return updated_checks, failures

    def _run_independent_review(
        self,
        *,
        run_id: str,
        spec: GoalSpec,
        task: TaskNode,
        sandbox: SandboxHandle,
        diff_text: str,
        check_summary: str,
        art_dir: Path,
    ) -> ReviewVerdict:
        ro_toolkit = WorkspaceToolkit(sandbox.workspace_path, read_only=True)
        review_loop = AgentLoop(
            provider=self.provider,
            toolkit=ro_toolkit,
            store=self.store,
            run_id=run_id,
            actor=f"argus:{task.task_id}",
            max_steps=8,
            trace_path=art_dir / f"review-cycle-{task.review_cycle + 1}.jsonl",
        )
        relevant_reqs = [r for r in spec.requirements if r.id in task.requirement_refs]
        schema_json = json.dumps(ReviewVerdict.model_json_schema(), ensure_ascii=False)
        incremental = task.review_cycle > 0
        prior_ctx = (
            f"\nPrevious cycle rejection feedback:\n{task.last_error}\n"
            if incremental and task.last_error
            else ""
        )
        task_summary = task.model_dump_json(
            include={"task_id", "title", "objective", "write_scope"}
        )
        reqs_json = json.dumps(
            [r.model_dump(mode="json") for r in relevant_reqs], ensure_ascii=False
        )
        oos_json = json.dumps(spec.out_of_scope, ensure_ascii=False)
        spec_ctx = _format_spec_context(spec)
        user_prompt = (
            f"Sandbox root: {sandbox.workspace_path}\n"
            f"TaskNode: {task_summary}\n"
            f"Relevant requirements: {reqs_json}\n"
            f"{spec_ctx}\n"
            f"Out of scope (MUST NOT be added): {oos_json}\n"
            f"Incremental review cycle={task.review_cycle + 1} (incremental_only={incremental})\n"
            f"{prior_ctx}\n"
            f"Command check outputs:\n{check_summary}\n\n"
            f"Git diff / changes:\n{diff_text}\n\n"
            f"Set task_id='{task.task_id}', attempt={task.attempt}, "
            f"incremental_only={str(incremental).lower()}.\n"
            f"Return ReviewVerdict JSON matching schema:\n{schema_json}"
        )
        verdict = run_structured_loop(
            loop=review_loop,
            system_prompt=_REVIEWER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model_cls=ReviewVerdict,
        ).model_copy(
            update={
                "task_id": task.task_id,
                "attempt": task.attempt,
                "incremental_only": incremental,
            }
        )
        (art_dir / f"verdict-cycle-{task.review_cycle + 1}.json").write_text(
            verdict.model_dump_json(indent=2), encoding="utf-8"
        )
        self.store.append_event(
            run_id=run_id,
            stage="verify",
            event_type="review.verdict_recorded",
            actor="argus",
            payload={
                "task_id": task.task_id,
                "attempt": task.attempt,
                "passed": verdict.passed,
                "important_count": sum(
                    1 for f in verdict.findings if f.severity == ReviewSeverity.IMPORTANT
                ),
                "nit_count": sum(
                    1 for f in verdict.findings if f.severity == ReviewSeverity.NIT
                ),
            },
        )
        return verdict

    def _execute_single_task(
        self,
        *,
        run_id: str,
        spec: GoalSpec,
        task_id: str,
        mode: RunMode,
        apply_to_workspace: bool,
        worker_max_steps: int,
    ) -> TaskNode:
        leased = self.store.acquire_task_lease(run_id, task_id, actor="chronos")
        art_dir = self._run_dir(run_id) / "artifacts" / leased.task_id / f"attempt-{leased.attempt}"
        art_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir = self._run_dir(run_id) / "accepted_overlay"
        sandbox = create_sandbox(
            self.workspace_root,
            label=f"{run_id}-{leased.task_id}-a{leased.attempt}",
        )
        _copy_matching_scope(overlay_dir, sandbox.workspace_path, write_scope=None)

        try:
            toolkit = WorkspaceToolkit(
                sandbox.workspace_path,
                allowed_write_scope=leased.write_scope,
                read_only=False,
                env_root=self.workspace_root,
            )
            worker_loop = AgentLoop(
                provider=self.provider,
                toolkit=toolkit,
                store=self.store,
                run_id=run_id,
                actor=f"hephaestus:{leased.task_id}",
                max_steps=worker_max_steps,
                trace_path=art_dir / "worker.jsonl",
                synthesize_on_max_steps=True,
            )
            rework_ctx = (
                f"\n\n[PRIOR REJECTION / ERROR TO FIX]\n{leased.last_error}\n"
                if leased.last_error
                else ""
            )
            checks_json = json.dumps(
                [
                    c.model_dump(include={"id", "kind", "description", "command"}, mode="json")
                    for c in leased.acceptance_checks
                ],
                ensure_ascii=False,
            )
            spec_ctx = _format_spec_context(spec)
            worker_prompt = (
                f"Sandbox workspace root: {sandbox.workspace_path}\n"
                f"Task {leased.task_id}: {leased.title}\n"
                f"Objective: {leased.objective}\n"
                f"Step budget: {worker_max_steps} turns max (keep exploration focused <= 6 steps, "
                f"implement edits early, verify with targeted commands, and stop once passing).\n"
                f"Allowed write_scope (relative): {json.dumps(leased.write_scope)}\n"
                f"{spec_ctx}\n"
                f"Out of scope: {json.dumps(spec.out_of_scope, ensure_ascii=False)}\n"
                f"Acceptance checks: {checks_json}"
                f"{rework_ctx}"
            )
            loop_res = worker_loop.run(
                system_prompt=_WORKER_SYSTEM_PROMPT,
                user_prompt=worker_prompt,
            )

            # Transition lease to VERIFYING before running checks/review
            verifying = self.store.update_task_with_lease(
                run_id,
                leased.task_id,
                expected_lease_generation=leased.lease_generation,
                mutator=lambda t: t.model_copy(
                    update={
                        "status": TaskStatus.VERIFYING,
                        "worktree_path": str(sandbox.workspace_path),
                    }
                ),
                actor="chronos",
                event_type="task.verifying",
            )

            diff_text = capture_sandbox_diff(sandbox)
            diff_file = art_dir / "diff.patch"
            diff_file.write_text(diff_text, encoding="utf-8")

            if not loop_res.completed:
                return self._fail_or_requeue(
                    run_id=run_id,
                    task=verifying,
                    reason=f"Worker did not complete cleanly: {loop_res.stop_reason}",
                    checks=verifying.acceptance_checks,
                )

            updated_checks, cmd_failures = self._run_command_checks(
                run_id=run_id,
                task=verifying,
                sandbox=sandbox,
                art_dir=art_dir,
            )
            if cmd_failures:
                return self._fail_or_requeue(
                    run_id=run_id,
                    task=verifying,
                    reason="\n".join(cmd_failures),
                    checks=updated_checks,
                )

            # Run cold-start independent review if check or default policy requires it
            check_summary = "\n".join(
                f"- {c.id} ({c.kind.value}): passed={c.passed}" for c in updated_checks
            )
            verdict = self._run_independent_review(
                run_id=run_id,
                spec=spec,
                task=verifying,
                sandbox=sandbox,
                diff_text=diff_text,
                check_summary=check_summary,
                art_dir=art_dir,
            )
            if not verdict.passed:
                important_lines = [
                    f"[{f.severity.value}] {f.file_path}:{f.line or '-'} {f.message}"
                    for f in verdict.findings
                    if f.severity == ReviewSeverity.IMPORTANT
                ]
                err_msg = f"Review rejected: {verdict.summary}\n" + "\n".join(important_lines)
                return self._fail_or_requeue(
                    run_id=run_id,
                    task=verifying,
                    reason=err_msg,
                    checks=updated_checks,
                )

            # Always snapshot accepted outputs to run overlay for downstream DAG tasks
            _copy_matching_scope(sandbox.workspace_path, overlay_dir, verifying.write_scope)
            # Sync files to host workspace only in EXECUTION mode when explicitly requested
            if mode == RunMode.EXECUTION and apply_to_workspace:
                _copy_matching_scope(
                    sandbox.workspace_path, self.workspace_root, verifying.write_scope
                )

            verdict_path = str(art_dir / f"verdict-cycle-{verifying.review_cycle + 1}.json")
            final_checks = [
                c.model_copy(update={"passed": True, "evidence_ref": verdict_path})
                if c.kind == CheckKind.INDEPENDENT_REVIEW
                else c
                for c in updated_checks
            ]
            return self.store.update_task_with_lease(
                run_id,
                verifying.task_id,
                expected_lease_generation=verifying.lease_generation,
                mutator=lambda t: t.model_copy(
                    update={
                        "status": TaskStatus.ACCEPTED,
                        "acceptance_checks": final_checks,
                        "last_error": None,
                        "worktree_path": None,
                    }
                ),
                actor="chronos",
                event_type="task.accepted",
            )
        finally:
            remove_sandbox(sandbox)

    def _fail_or_requeue(
        self,
        *,
        run_id: str,
        task: TaskNode,
        reason: str,
        checks: list[AcceptanceCheck],
    ) -> TaskNode:
        next_cycle = task.review_cycle + 1
        exhausted = next_cycle >= task.max_cycles
        next_status = TaskStatus.BLOCKED if exhausted else TaskStatus.READY
        event_type = "task.blocked" if exhausted else "task.requeued"
        return self.store.update_task_with_lease(
            run_id,
            task.task_id,
            expected_lease_generation=task.lease_generation,
            mutator=lambda t: t.model_copy(
                update={
                    "status": next_status,
                    "review_cycle": next_cycle,
                    "acceptance_checks": checks,
                    "last_error": reason,
                    "worktree_path": None,
                }
            ),
            actor="chronos",
            event_type=event_type,
        )

    def run_execution(
        self,
        *,
        run_id: str,
        apply_to_workspace: bool = False,
        worker_max_steps: int = 20,
    ) -> dict[str, Any]:
        """Execute ready tasks respecting DAG dependencies, write_scope, and convergence rules."""
        spec = self._require_confirmed_spec(run_id)
        snap = self.store.get_run(run_id)
        plan: TaskPlan | None = snap["plan"]
        if plan is None:
            raise ValueError(f"Run {run_id} has no TaskPlan; propose plan first.")
        if not self.store.is_approved(
            run_id=run_id,
            target=ApprovalTarget.TASK_PLAN,
            target_id=plan.plan_id,
            target_hash=plan.content_hash,
        ):
            raise ValueError(
                f"TaskPlan {plan.plan_id} ({plan.content_hash}) is not human-approved."
            )

        self.store.set_run_status(run_id, RunStatus.RUNNING, actor="chronos", stage="orchestrate")

        while True:
            snap = self.store.get_run(run_id)
            if snap["status"] in (RunStatus.PAUSED, RunStatus.CANCELLED):
                return snap
            plan = snap["plan"]
            assert plan is not None
            policy = snap["budget_policy"]

            ready = compute_dispatchable_tasks(
                plan,
                max_concurrent=policy.max_concurrent_tasks,
            )
            if not ready:
                all_done = all(
                    t.status in (TaskStatus.ACCEPTED, TaskStatus.CANCELLED) for t in plan.tasks
                )
                target_status = RunStatus.COMPLETED if all_done else RunStatus.AWAITING_HUMAN
                self.store.set_run_status(
                    run_id,
                    target_status,
                    actor="chronos",
                    stage="orchestrate",
                    detail={
                        "accepted": sum(
                            1 for t in plan.tasks if t.status == TaskStatus.ACCEPTED
                        ),
                        "total": len(plan.tasks),
                    },
                )
                return self.store.get_run(run_id)

            if len(ready) == 1:
                self._execute_single_task(
                    run_id=run_id,
                    spec=spec,
                    task_id=ready[0].task_id,
                    mode=snap["mode"],
                    apply_to_workspace=apply_to_workspace,
                    worker_max_steps=worker_max_steps,
                )
            else:
                with ThreadPoolExecutor(max_workers=len(ready)) as pool:
                    futures = [
                        pool.submit(
                            self._execute_single_task,
                            run_id=run_id,
                            spec=spec,
                            task_id=t.task_id,
                            mode=snap["mode"],
                            apply_to_workspace=apply_to_workspace,
                            worker_max_steps=worker_max_steps,
                        )
                        for t in ready
                    ]
                    for fut in futures:
                        fut.result()

    def _apply_task_command(
        self,
        *,
        operation_id: str,
        run_id: str,
        task_id: str,
        expected_state_revision: int,
        action: str,
        event_type: str,
        actor: str,
        payload: dict[str, Any],
        task_mutator: Callable[[TaskNode], TaskNode],
        result_status: str,
    ) -> dict[str, Any]:
        """Shared idempotent task-mutation command helper."""

        def handler(conn: Any, cur_rev: int) -> dict[str, Any]:
            def mutate_plan(plan: TaskPlan) -> TaskPlan:
                updated_tasks: list[TaskNode] = []
                found = False
                for t in plan.tasks:
                    if t.task_id == task_id:
                        found = True
                        updated_tasks.append(task_mutator(t))
                    else:
                        updated_tasks.append(t)
                if not found:
                    raise ValueError(f"Task {task_id} not found")
                return plan.model_copy(update={"tasks": updated_tasks})

            new_rev = self.store.update_plan_in_tx(
                conn,
                run_id=run_id,
                cur_rev=cur_rev,
                mutator=mutate_plan,
                stage="control",
                event_type=event_type,
                actor=actor,
                payload=payload,
            )
            return {"task_id": task_id, "status": result_status, "state_revision": new_rev}

        return self.store.apply_idempotent_command(
            operation_id=operation_id,
            run_id=run_id,
            actor=actor,
            action=action,
            expected_state_revision=expected_state_revision,
            payload=payload,
            handler=handler,
        )

    def retry_blocked_task(
        self,
        *,
        operation_id: str,
        run_id: str,
        task_id: str,
        expected_state_revision: int,
        hint: str = "",
        actor: str = "zeus",
    ) -> dict[str, Any]:
        """Idempotently reset a BLOCKED/FAILED task back to READY for another cycle."""

        def mutate_retry(t: TaskNode) -> TaskNode:
            if t.status not in (TaskStatus.BLOCKED, TaskStatus.FAILED):
                raise ValueError(
                    f"Task {task_id} status is {t.status.value}, not blocked/failed"
                )
            new_err = (
                f"{t.last_error or ''}\n[Human hint]: {hint}".strip() if hint else t.last_error
            )
            return t.model_copy(
                update={"status": TaskStatus.READY, "review_cycle": 0, "last_error": new_err}
            )

        return self._apply_task_command(
            operation_id=operation_id,
            run_id=run_id,
            task_id=task_id,
            expected_state_revision=expected_state_revision,
            action="retry_task",
            event_type="task.retried_by_human",
            actor=actor,
            payload={"task_id": task_id, "hint": hint},
            task_mutator=mutate_retry,
            result_status="ready",
        )

    def accept_task_risk(
        self,
        *,
        operation_id: str,
        run_id: str,
        task_id: str,
        expected_state_revision: int,
        reason: str,
        actor: str = "zeus",
    ) -> dict[str, Any]:
        """Explicitly mark a blocked task accepted-with-risk (never marked verified)."""
        clean_reason = reason.strip()
        if not clean_reason:
            raise ValueError("accept_task_risk requires a non-empty reason")

        return self._apply_task_command(
            operation_id=operation_id,
            run_id=run_id,
            task_id=task_id,
            expected_state_revision=expected_state_revision,
            action="accept_risk",
            event_type="task.accepted_with_risk",
            actor=actor,
            payload={"task_id": task_id, "reason": clean_reason},
            task_mutator=lambda t: t.model_copy(
                update={"status": TaskStatus.ACCEPTED, "accepted_risk_reason": clean_reason}
            ),
            result_status="accepted_with_risk",
        )
