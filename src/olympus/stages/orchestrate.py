"""Stage 2 (Orchestrate & Verify): Adaptive Task DAG, sandbox execution, and cold-start review."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Event, Lock
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from olympus.core.contracts import (
    AcceptanceCheck,
    ApprovalRecord,
    ApprovalTarget,
    CheckKind,
    DesignSpec,
    ExecutionStrategy,
    GoalSpec,
    ReviewSeverity,
    ReviewVerdict,
    RunMode,
    RunStatus,
    TaskKind,
    TaskNode,
    TaskPlan,
    TaskStatus,
)
from olympus.core.control import RunControl, is_plan_bound_to_upstream
from olympus.core.dag import (
    PlanValidationError,
    compute_dispatchable_tasks,
    scopes_overlap,
    validate_plan,
)
from olympus.core.store import ConcurrencyConflictError, RunStore
from olympus.harness.activity import LiveActivity, milestone_fields
from olympus.harness.delivery import DELIVERY_EVENT, run_delivery_checks
from olympus.harness.diff_gate import run_diff_gate
from olympus.harness.driver import (
    ActiveTurnHandle,
    AgentDriver,
    DriverEvent,
    DriverRegistry,
    DriverTurnRequest,
    ExecutionDriversConfig,
    default_driver_factories,
)
from olympus.harness.loop import AgentLoop, BudgetExhaustedError, parse_model_json
from olympus.harness.native import NativeDriver
from olympus.harness.watchdog import Watchdog, WatchdogConfig
from olympus.harness.worktree import (
    SandboxHandle,
    archive_sandbox_recovery,
    build_sandbox_env,
    capture_sandbox_diff,
    create_sandbox,
    remove_sandbox,
    snapshot_sandbox_state,
)
from olympus.providers.base import LLMProvider
from olympus.providers.factory import build_provider
from olympus.stages.common import format_workspace_inventory
from olympus.tools.aci import IGNORED_DIR_NAMES, WorkspacePolicy, WorkspaceToolkit

if TYPE_CHECKING:
    from olympus.harness.driver import TurnOutcome

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
   repo-relative paths only. When writing inline 'python -c "..."' commands, NEVER chain
   '; try:', '; except:', '; for ', '; if ', or '; def ' on a single line (Python raises
   SyntaxError)—use real '\\n' newlines, simple expressions/asserts, or 'python -m pytest ...'.
6. Honor all GoalSpec constraints and resolved open_ambiguities answers in task objectives/checks.
7. Every TaskNode MUST declare non-empty `write_scope` (repo-relative paths/dirs owned by
   that task; NEVER omit `write_scope` or confuse it with `deliverables`), `task_id`, `title`,
   `objective`, `requirement_refs`, and `acceptance_checks`.
8. Exploration limit: inspect at most 3 existing files from Observed workspace files; if design
   paths do not exist yet (greenfield), emit TaskPlan JSON directly without running glob/grep.
9. Output ONLY a single raw JSON object starting with '{' matching TaskPlan (NEVER call an
   invented 'submit_plan' tool, and include no leading prose, $comment, $schema, or extra keys)."""

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
2. Respect TaskNode.write_scope as a hard boundary: judge ONLY work achievable within this
   TaskNode's objective and write_scope. Never reject a task for missing deliverables or test files
   outside its write_scope (those belong to other tasks in the DAG).
3. Classify each finding strictly as 'important' (blocks pass), 'nit', or 'pre_existing'.
4. Set passed=false whenever any 'important' finding exists; set passed=true only if implementation
   genuinely satisfies the task and no 'important' defects remain.
5. Output ONLY a single JSON object conforming to ReviewVerdict."""


def _plan_binding(
    *,
    design: DesignSpec | None,
    spec: GoalSpec | None,
) -> dict[str, Any]:
    """Force the plan binding to the exact upstream fingerprints the run froze."""
    update: dict[str, Any] = {}
    if design is not None:
        update["design_id"] = design.design_id
        update["design_hash"] = design.content_hash
    if spec is not None:
        update["spec_id"] = spec.spec_id
        update["spec_hash"] = spec.content_hash
    else:
        # Never let the model invent a spec binding the run does not own.
        update["spec_id"] = None
        update["spec_hash"] = None
    return update


def _copy_matching_scope(
    src_root: Path,
    dst_root: Path,
    write_scope: list[str] | None,
) -> None:
    """Copy non-ignored files from src_root to dst_root, optionally filtered by write_scope."""
    if not src_root.exists():
        return
    # Share the tool-gating matcher so glob scopes deliver exactly what they authorize.
    policy = WorkspacePolicy(allowed_write_scope=write_scope)
    for p in sorted(src_root.rglob("*")):
        if not p.is_file():
            continue
        rel_path = p.relative_to(src_root)
        ignored = IGNORED_DIR_NAMES | {".olympus"}
        if any(part in ignored for part in rel_path.parts):
            continue
        rel = rel_path.as_posix()
        if write_scope is not None and not policy.allows_relative_path(rel):
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


def _diff_stats(diff_text: str) -> tuple[int, int]:
    """Count added/removed lines, including raw contents of newly created files."""
    added = 0
    removed = 0
    for block in diff_text.split("\n\n"):
        lines = block.splitlines()
        is_new_file = (
            len(lines) >= 2
            and lines[0].startswith("--- /dev/null")
            and lines[1].startswith("+++ b/")
        )
        if is_new_file:
            added += sum(1 for line in lines[2:] if line.strip())
            continue
        for line in lines:
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
    return added, removed


def _write_manifest_md(
    *,
    art_dir: Path,
    task: TaskNode,
    diff_text: str,
    checks: Sequence[AcceptanceCheck],
    verdict: ReviewVerdict | None,
    receipt: dict[str, Any],
) -> Path:
    """Write a human-readable MANIFEST.md summarizing integrity artifacts for one attempt."""
    added, removed = _diff_stats(diff_text)
    passed_checks = sum(1 for check in checks if check.passed)
    failed_checks = [check for check in checks if not check.passed]
    if verdict is not None:
        outcome = "accepted" if verdict.passed else "rejected"
    elif any(check.evidence_ref is not None for check in failed_checks):
        outcome = "checks_failed"
    else:
        outcome = "incomplete"

    lines = [
        f"# MANIFEST — {task.task_id} (attempt {task.attempt})",
        "",
        f"- Task: {task.task_id} — {task.title}",
        f"- Outcome: {outcome}",
        f"- Task status: {task.status.value}",
        f"- Review cycle: {task.review_cycle}",
        f"- Checks: {passed_checks}/{len(checks)} passed",
        f"- Diff: +{added}/-{removed} lines",
        f"- Turn: {receipt.get('turn_id', '(none)')}",
        f"- Receipt: {receipt.get('one_line', '(none)')}",
        "",
        "## Objective",
        "",
        task.objective,
        "",
    ]
    if verdict is not None:
        lines.extend(
            [
                "## Review verdict",
                "",
                f"- passed: {verdict.passed}",
                f"- summary: {verdict.summary}",
                f"- findings: {len(verdict.findings)}",
                "",
            ]
        )
    lines.extend(["## Acceptance checks", ""])
    for check in checks:
        state = "passed" if check.passed else "not passed"
        lines.append(f"- {check.id} ({check.kind.value}): {state} — {check.description}")
    lines.append("")
    manifest_path = art_dir / "MANIFEST.md"
    manifest_path.write_text("\n".join(lines), encoding="utf-8")
    return manifest_path


def _format_design_context(design: DesignSpec | None, *, task: TaskNode) -> str:
    """Render relevant invariants, readonly contract paths, and typed interface seams."""
    if design is None:
        return ""
    lines: list[str] = ["DesignSpec Constraints & Seams:"]
    if design.execution_strategy is not None:
        lines.append(
            f"- Execution strategy: {design.execution_strategy.value} ({design.strategy_rationale})"
        )
    if design.shared_contract_paths:
        mode_note = (
            "Wave 0 owner (editable)"
            if task.kind == TaskKind.CONTRACT
            else "READ-ONLY shared contracts (DO NOT MODIFY; import only)"
        )
        lines.append(
            f"- Shared contract paths [{mode_note}]: {json.dumps(design.shared_contract_paths)}"
        )
    task_refs = set(task.design_refs)
    matched_invariants = [
        inv
        for inv in design.shared_invariants
        if not inv.applies_to or any(mod in task_refs for mod in inv.applies_to)
    ]
    if matched_invariants:
        lines.append("- Global invariants (MUST obey):")
        for inv in matched_invariants:
            hint = f" [hint: {inv.enforcement_hint}]" if inv.enforcement_hint else ""
            lines.append(f"  * {inv.id} [{inv.category}]: {inv.rule}{hint}")
    matched_interfaces = [
        iface
        for iface in design.interfaces
        if task.kind == TaskKind.CONTRACT
        or iface.id in task_refs
        or iface.provider_module_id in task_refs
        or any(cid in task_refs for cid in iface.consumer_module_ids)
    ]
    if matched_interfaces:
        lines.append("- Relevant interface contracts:")
        for iface in matched_interfaces:
            inputs = ", ".join(
                f"{f.name}:{f.type_expr}{'?' if not f.required else ''}"
                for f in iface.input_schema
            )
            outputs = ", ".join(
                f"{f.name}:{f.type_expr}{'?' if not f.required else ''}"
                for f in iface.output_schema
            )
            lines.append(
                f"  * {iface.id} [{iface.protocol}] {iface.signature} "
                f"({iface.provider_module_id} -> {iface.consumer_module_ids}) | "
                f"in=[{inputs}] out=[{outputs}] errors={iface.error_cases}"
            )
    return "\n".join(lines) + "\n"


def _format_handoff_context(plan: TaskPlan, *, current_task_id: str) -> str:
    """Build a compact handoff summary of already-accepted upstream steps for serial_handoff."""
    done = [
        t
        for t in plan.tasks
        if t.task_id != current_task_id and t.status == TaskStatus.ACCEPTED
    ]
    if not done:
        return ""
    lines = ["Prior Accepted Handoff Summary (serial_handoff):"]
    for t in done[-6:]:
        delivs = ", ".join(t.deliverables) if t.deliverables else ", ".join(t.write_scope)
        lines.append(f"- {t.task_id} ({t.title}): accepted; deliverables=[{delivs}]")
    return "\n".join(lines) + "\n"


_MODULAR_PLANNER_GUIDANCE = """
Modular product decomposition (topology={topology}):
- Emit Wave 0 contract tasks (kind='contract', role='architect') that exclusively own
  design.shared_contract_paths (plus any Wave 0 contract self-tests in that task's
  write_scope) and pin interface signatures/data models before any implementation.
- Emit Wave 1 implementation tasks (kind='implement', role='engineer'), one per design
  module, each with design_refs naming the module id it realizes. Wave 1/2 tasks MUST
  NOT list shared_contract_paths in write_scope (they are read-only during Wave 1+).
- Emit Wave 2 verification tasks (kind='test', role='qa') plus one integration task
  (kind='integrate', role='engineer') when more than one module exists.
- Every design module id MUST appear in at least one task's design_refs; `design_refs` may ONLY
  list DesignSpec module ids or interface ids (NEVER data_model names or invariant ids).
- Tasks in different modules MUST NOT share overlapping write_scope.
- Command acceptance_checks MUST only reference modules or test paths inside the task's own
  write_scope or its transitive depends_on ancestors (NEVER import a concurrent sibling task's
  modules in a task check; put cross-module integration checks in a Wave 2 task that depends_on
  all needed Wave 1 tasks), and MUST NOT silence failures with 2>/dev/null.
"""


class _ReviewOutputError(ValueError):
    pass


class _ReviewCancelled(Exception):
    pass


class OrchestrateStage:
    """Deterministic Chronos scheduler driving isolated Hephaestus execution & cold-start review."""

    def __init__(
        self,
        *,
        store: RunStore,
        provider: LLMProvider,
        workspace_root: Path | str,
        artifacts_root: Path | None = None,
        execution_drivers: ExecutionDriversConfig | None = None,
        driver_registry: DriverRegistry | None = None,
        model_provider: Callable[[str], LLMProvider] | None = None,
        monitor_provider: LLMProvider | None = None,
        watchdog_config: WatchdogConfig | None = None,
        on_driver_event: DriverEvent | None = None,
        on_driver_handle: Callable[[ActiveTurnHandle], None] | None = None,
        live_activity: LiveActivity | None = None,
    ) -> None:
        self.store = store
        self.provider = provider
        self.execution_drivers = execution_drivers or ExecutionDriversConfig.resolve()
        self.driver_registry = driver_registry or DriverRegistry()
        self.model_provider = model_provider
        self.monitor_provider = monitor_provider
        self.watchdog_config = watchdog_config
        self.on_driver_event = on_driver_event
        self.on_driver_handle = on_driver_handle
        self.live_activity = live_activity
        self.workspace_root = Path(workspace_root).resolve()
        self.artifacts_root = (
            artifacts_root
            if artifacts_root is not None
            else self.workspace_root / ".olympus" / "runs"
        )

    def _native_driver(
        self, run_id: str, model: str | None, *, review: bool = False
    ) -> NativeDriver:
        provider = self.provider
        if model is not None and model != provider.model_name:
            if self.model_provider is None:
                raise ValueError("A native model override requires a provider factory")
            provider = self.model_provider(model)
        return NativeDriver(
            provider=provider,
            store=self.store,
            run_id=run_id,
            response_model=ReviewVerdict if review else None,
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

    def _require_approved_design(self, run_id: str) -> DesignSpec:
        snap = self.store.get_run(run_id)
        design: DesignSpec | None = snap["design"]
        if design is None:
            raise ValueError(f"Run {run_id} has no DesignSpec; run Stage 1.5 (Design) first.")
        if not self.store.is_approved(
            run_id=run_id,
            target=ApprovalTarget.DESIGN_SPEC,
            target_id=design.design_id,
            target_hash=design.content_hash,
        ):
            raise ValueError(
                f"DesignSpec {design.design_id} ({design.content_hash}) is not human-approved."
            )
        return design

    def propose_plan(
        self,
        *,
        run_id: str,
        auto_approve: bool = False,
        approver: str = "zeus",
        max_steps: int = 10,
        topology: str = "minimal_patch",
        use_design: bool = True,
    ) -> TaskPlan:
        """Generate and deterministically validate a TaskPlan.

        Accepts an approved DesignSpec (with or without a GoalSpec) or, for backward
        compatibility, an approved GoalSpec alone. ``modular_product``/``tdd_first``
        topologies decompose the design's modules into Wave 0/1/2 tasks.
        """
        snap = self.store.get_run(run_id)
        design: DesignSpec | None = snap["design"] if use_design else None
        spec: GoalSpec | None = snap["spec"]
        if design is not None:
            design = self._require_approved_design(run_id)
            if spec is not None and not self.store.is_approved(
                run_id=run_id,
                target=ApprovalTarget.GOAL_SPEC,
                target_id=spec.spec_id,
                target_hash=spec.content_hash,
            ):
                # A design-only run may carry an unapproved draft spec; ignore its binding.
                spec = None
        else:
            spec = self._require_confirmed_spec(run_id)

        modular = topology in ("modular_product", "tdd_first") and design is not None
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
        inventory = format_workspace_inventory(toolkit)
        schema_json = json.dumps(TaskPlan.model_json_schema(), ensure_ascii=False)

        binding_lines: list[str] = []
        if design is not None:
            binding_lines.append(
                f"Set design_id='{design.design_id}' and design_hash='{design.content_hash}'."
            )
        if spec is not None:
            binding_lines.append(
                f"Set spec_id='{spec.spec_id}' and spec_hash='{spec.content_hash}'."
            )
        if modular:
            assert design is not None
            context_block = (
                f"Approved GoalSpec:\n{spec.model_dump_json(indent=2)}\n\n"
                if spec is not None
                else ""
            ) + f"Approved DesignSpec:\n{design.model_dump_json(indent=2)}\n\n"
            modular_block = _MODULAR_PLANNER_GUIDANCE.format(topology=topology)
            design_paths = [
                p for mod in design.modules for p in mod.target_paths
            ] + list(design.shared_contract_paths)
            if design_paths and not any(
                (self.workspace_root / p.rstrip("/")).exists() for p in design_paths
            ):
                context_block += (
                    "Greenfield workspace note: none of DesignSpec's target_paths exist on disk "
                    "yet — output TaskPlan JSON immediately without searching unrelated files.\n\n"
                )
        else:
            context_block = (
                f"Confirmed GoalSpec:\n{spec.model_dump_json(indent=2)}\n\n"
                if spec is not None
                else ""
            )
            modular_block = ""

        base_prompt = (
            f"Workspace root: {self.workspace_root}\n"
            f"{inventory}\n"
            f"Note: acceptance_checks[*].command will run in an isolated sandbox worktree "
            f"(cwd = sandbox root). NEVER reference '{self.workspace_root}' inside commands.\n"
            f"Remember: acceptance_checks[*].kind MUST be one of "
            f"'command', 'independent_review', or 'human' (never 'review'), and every "
            f"kind='command' check MUST include a non-empty 'command' string.\n"
            "When a GoalSpec is supplied, optional delivery_checks bind one specific end-to-end "
            "command per verifiable criterion by requirement_id and 1-based criterion_index. "
            "Run these commands against the composed delivery after task execution. "
            "Omit conditions without a reliable executable check. Task checks are separate.\n"
            f"{context_block}"
            f"{modular_block}"
            f"{chr(10).join(binding_lines)}\n"
            f"Return TaskPlan JSON matching schema:\n{schema_json}"
        )

        def stamp_binding(candidate: TaskPlan) -> TaskPlan:
            return candidate.model_copy(update=_plan_binding(design=design, spec=spec))

        def plan_gate(candidate: TaskPlan) -> list[str]:
            return validate_plan(
                candidate,
                spec=spec,
                design=design if modular else None,
                workspace_root=self.workspace_root,
            )

        def raise_exhausted(candidate: TaskPlan, violations: list[str]) -> TaskPlan:
            raise PlanValidationError(violations)

        plan = loop.run_structured(
            system_prompt=_PLANNER_SYSTEM_PROMPT + modular_block,
            user_prompt=base_prompt,
            model_cls=TaskPlan,
            max_validation_retries=2,
            validation_header="[PLAN VALIDATION FAILED]",
            validation_instruction="Fix all violations and return valid TaskPlan JSON.",
            transform=stamp_binding,
            validator=plan_gate,
            on_validation_exhausted=raise_exhausted,
        )

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
        expected_revision: int | None = None,
        expected_target_hash: str | None = None,
    ) -> ApprovalRecord:
        """Record human approval bound to the current TaskPlan content_hash."""
        return RunControl(self.store).approve_plan(
            run_id=run_id,
            approver=approver,
            reason=reason,
            expected_revision=expected_revision,
            expected_target_hash=expected_target_hash,
        )

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
        reviewer: tuple[AgentDriver, str | None],
        design: DesignSpec | None = None,
    ) -> ReviewVerdict:
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
        design_ctx = _format_design_context(design, task=task)
        user_prompt = (
            f"Sandbox root: {sandbox.workspace_path}\n"
            f"TaskNode: {task_summary}\n"
            f"Hard write_scope boundary (do NOT require edits to files outside these paths): "
            f"{json.dumps(task.write_scope, ensure_ascii=False)}\n"
            f"Relevant requirements: {reqs_json}\n"
            f"{spec_ctx}\n"
            f"{design_ctx}"
            f"Out of scope (MUST NOT be added): {oos_json}\n"
            f"Incremental review cycle={task.review_cycle + 1} (incremental_only={incremental})\n"
            f"{prior_ctx}\n"
            f"Command check outputs:\n{check_summary}\n\n"
            f"Git diff / changes:\n{diff_text}\n\n"
            f"Set task_id='{task.task_id}', attempt={task.attempt}, "
            f"incremental_only={str(incremental).lower()}.\n"
            f"Return ReviewVerdict JSON matching schema:\n{schema_json}"
        )
        driver, model = reviewer
        request = DriverTurnRequest(
            workspace_path=sandbox.workspace_path,
            env_root=sandbox.workspace_path,
            system_prompt=_REVIEWER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            actor=f"argus:{task.task_id}",
            task_id=task.task_id,
            lease_generation=task.lease_generation,
            read_only=True,
            max_steps=8,
            trace_path=art_dir / f"review-cycle-{task.review_cycle + 1}.jsonl",
            model=model,
        )
        for attempt in range(2):
            outcome = driver.run_turn(
                request, on_event=self.on_driver_event, on_handle=self.on_driver_handle
            )
            if outcome.stop_reason == "cancelled":
                raise _ReviewCancelled
            if outcome.stop_reason.startswith("budget_exhausted"):
                raise BudgetExhaustedError(outcome.stop_reason)
            if not outcome.completed:
                raise _ReviewOutputError(f"Reviewer stopped early: {outcome.stop_reason}")
            try:
                verdict = parse_model_json(outcome.final_content, ReviewVerdict)
            except (json.JSONDecodeError, ValidationError) as exc:
                if attempt == 1:
                    raise _ReviewOutputError(
                        "Reviewer returned invalid ReviewVerdict JSON"
                    ) from exc
                request = request.model_copy(
                    update={
                        "user_prompt": (
                            f"{user_prompt}\n\n[PREVIOUS OUTPUT FAILED VALIDATION]\n"
                            "Your previous response did not match ReviewVerdict. "
                            "Return only the corrected JSON object matching the schema above.\n"
                            f"Previous response:\n{outcome.final_content}"
                        ),
                        "trace_path": art_dir
                        / f"review-cycle-{task.review_cycle + 1}-repair.jsonl",
                    }
                )
            else:
                break

        verdict = verdict.model_copy(
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

    def _run_worker_turn(
        self,
        run_id: str,
        task: TaskNode,
        driver: AgentDriver,
        request: DriverTurnRequest,
        watchdog: Watchdog | None,
        *,
        on_handle_registered: Callable[[ActiveTurnHandle], None] | None = None,
        on_handle_finished: Callable[[ActiveTurnHandle], None] | None = None,
    ) -> TurnOutcome:
        activity = self.live_activity
        active_handle: list[ActiveTurnHandle] = []

        def observe(kind: str, payload: dict[str, Any]) -> None:
            fields = milestone_fields(kind, payload)
            if fields is not None:
                self.store.append_event(
                    run_id=run_id,
                    stage="orchestrate",
                    event_type=kind,
                    actor=request.actor,
                    payload={
                        "task_id": task.task_id,
                        "attempt": task.attempt,
                        "actor": request.actor,
                        **({"driver": driver.name} if kind != "agent.steered" else {}),
                        **fields,
                    },
                )
                if watchdog is not None:
                    watchdog.observe(kind, fields)
            if kind == "text_delta" and activity is not None:
                text = payload.get("text")
                if isinstance(text, str) and text:
                    activity.update(
                        run_id, task.task_id, task.attempt, text,
                        replace=payload.get("replace") is True,
                    )
            if self.on_driver_event is not None:
                self.on_driver_event(kind, payload)

        def register_handle(handle: ActiveTurnHandle) -> None:
            active_handle.append(handle)
            if on_handle_registered is not None:
                on_handle_registered(handle)
            if watchdog is not None:
                watchdog.start(handle)
            if self.on_driver_handle is not None:
                self.on_driver_handle(handle)

        try:
            return driver.run_turn(request, on_event=observe, on_handle=register_handle)
        finally:
            if watchdog is not None:
                watchdog.stop()
            if on_handle_finished is not None and active_handle:
                on_handle_finished(active_handle[-1])
            if activity is not None:
                activity.clear(run_id, task.task_id, task.attempt)

    def _execute_single_task(
        self,
        *,
        run_id: str,
        spec: GoalSpec,
        task_id: str,
        mode: RunMode,
        apply_to_workspace: bool,
        worker_max_steps: int | None,
        worker: tuple[AgentDriver, str | None],
        monitor_factory: Callable[[], LLMProvider] | None,
        reviewer: tuple[AgentDriver, str | None],
        interrupted: Event,
        design: DesignSpec | None = None,
        handoff_context: str = "",
        on_handle_registered: Callable[[ActiveTurnHandle], None] | None = None,
        on_handle_finished: Callable[[ActiveTurnHandle], None] | None = None,
        on_run_paused: Callable[[], None] | None = None,
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

        archive_failed = False
        try:
            worker_baseline = snapshot_sandbox_state(sandbox)
            forbidden_contracts = (
                design.shared_contract_paths
                if design is not None and leased.kind != TaskKind.CONTRACT
                else ()
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
            design_ctx = _format_design_context(design, task=leased)
            handoff_block = f"{handoff_context}\n" if handoff_context else ""
            step_guidance = (
                f"Step budget: {worker_max_steps} turns max (keep exploration focused <= 6 steps, "
                "implement edits early, verify with targeted commands, and stop once passing).\n"
                if worker_max_steps is not None
                else "Execution discipline: keep exploration focused, implement edits early, "
                "verify with targeted commands, and stop once all acceptance checks pass.\n"
            )
            worker_prompt = (
                f"Sandbox workspace root: {sandbox.workspace_path}\n"
                f"Task {leased.task_id}: {leased.title}\n"
                f"Objective: {leased.objective}\n"
                f"{step_guidance}"
                f"Allowed write_scope (relative): {json.dumps(leased.write_scope)}\n"
                f"{spec_ctx}\n"
                f"{design_ctx}"
                f"{handoff_block}"
                f"Out of scope: {json.dumps(spec.out_of_scope, ensure_ascii=False)}\n"
                f"Acceptance checks: {checks_json}"
                f"{rework_ctx}"
            )
            driver, model = worker
            request = DriverTurnRequest(
                workspace_path=sandbox.workspace_path,
                env_root=self.workspace_root,
                system_prompt=_WORKER_SYSTEM_PROMPT,
                user_prompt=worker_prompt,
                actor=f"hephaestus:{leased.task_id}",
                task_id=leased.task_id,
                lease_generation=leased.lease_generation,
                write_scope=leased.write_scope,
                forbidden_globs=list(forbidden_contracts),
                max_steps=worker_max_steps,
                trace_path=art_dir / "worker.jsonl",
                model=model,
            )
            watchdog = None
            if monitor_factory is not None:
                def emit_watchdog(kind: str, payload: dict[str, Any]) -> None:
                    self.store.append_event(
                        run_id=run_id,
                        stage="orchestrate",
                        event_type=kind,
                        actor="watchdog",
                        payload={"task_id": leased.task_id, "attempt": leased.attempt, **payload},
                    )
                    if self.on_driver_event is not None:
                        self.on_driver_event(kind, payload)

                watchdog = Watchdog(
                    config=self.watchdog_config or WatchdogConfig(),
                    provider_factory=monitor_factory,
                    sandbox=sandbox,
                    objective=leased.objective,
                    write_scope=leased.write_scope,
                    emit=emit_watchdog,
                )
            worker_error = None
            try:
                loop_res = self._run_worker_turn(
                    run_id,
                    leased,
                    driver,
                    request,
                    watchdog,
                    on_handle_registered=on_handle_registered,
                    on_handle_finished=on_handle_finished,
                )
            except Exception as exc:
                if watchdog is None or watchdog.pause_diagnosis is None:
                    raise
                worker_error = type(exc).__name__

            if watchdog is not None and watchdog.pause_diagnosis is not None:
                archive_failed = True
                diagnosis = dict(watchdog.pause_diagnosis)
                if worker_error is not None:
                    diagnosis["worker_error"] = worker_error
                try:
                    (art_dir / "watchdog.json").write_text(
                        json.dumps(diagnosis, ensure_ascii=False), encoding="utf-8",
                    )
                    (art_dir / "worker.jsonl").touch(exist_ok=True)
                    archive_sandbox_recovery(sandbox, art_dir)
                except (OSError, subprocess.SubprocessError) as exc:
                    diagnosis["archive_error"] = type(exc).__name__
                else:
                    archive_failed = False
                paused_task = RunControl(self.store).pause_for_watchdog(
                    run_id, task=leased, diagnosis=diagnosis,
                    retained_sandbox=str(sandbox.workspace_path) if archive_failed else None,
                )
                if on_run_paused is not None:
                    on_run_paused()
                return paused_task

            if self.store.get_run(run_id)["status"] in (RunStatus.PAUSED, RunStatus.CANCELLED):
                interrupted.set()
                return self._return_interrupted_task(run_id, leased)

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

            if not loop_res.completed and (
                loop_res.stop_reason == "cancelled"
                or loop_res.stop_reason.startswith("budget_exhausted")
            ):
                _write_manifest_md(
                    art_dir=art_dir,
                    task=verifying,
                    diff_text=diff_text,
                    checks=verifying.acceptance_checks,
                    verdict=None,
                    receipt=loop_res.receipt,
                )
                if loop_res.stop_reason == "cancelled":
                    interrupted.set()
                    return self._return_interrupted_task(run_id, verifying)
                return self._pause_for_budget(
                    run_id=run_id,
                    task=verifying,
                    reason=f"Paused due to budget limit: {loop_res.stop_reason}",
                )

            gate = run_diff_gate(
                sandbox,
                worker_baseline,
                allowed_write_scope=verifying.write_scope,
                forbidden_globs=forbidden_contracts,
            )
            if gate.rejected:
                self.store.append_event(
                    run_id=run_id,
                    stage="verify",
                    event_type="task.diff_gate_rejected",
                    actor="chronos",
                    payload={
                        "task_id": verifying.task_id,
                        "attempt": verifying.attempt,
                        "violating_paths": [v.path for v in gate.violations],
                        "violations": [
                            {
                                "path": v.path,
                                "reason": v.reason,
                                "matched_pattern": v.matched_pattern,
                            }
                            for v in gate.violations
                        ],
                        "allowed_write_scope": list(gate.allowed_write_scope),
                        "forbidden_globs": list(gate.forbidden_globs),
                        "changed_paths": list(gate.changed_paths),
                    },
                )
                _write_manifest_md(
                    art_dir=art_dir,
                    task=verifying,
                    diff_text=diff_text,
                    checks=verifying.acceptance_checks,
                    verdict=None,
                    receipt=loop_res.receipt,
                )
                return self._fail_or_requeue(
                    run_id=run_id,
                    task=verifying,
                    reason=gate.rejection_message(),
                    checks=verifying.acceptance_checks,
                )

            if not loop_res.completed and not (
                loop_res.stop_reason == "max_steps_exceeded" and gate.changed_paths
            ):
                _write_manifest_md(
                    art_dir=art_dir,
                    task=verifying,
                    diff_text=diff_text,
                    checks=verifying.acceptance_checks,
                    verdict=None,
                    receipt=loop_res.receipt,
                )
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
                _write_manifest_md(
                    art_dir=art_dir,
                    task=verifying,
                    diff_text=diff_text,
                    checks=updated_checks,
                    verdict=None,
                    receipt=loop_res.receipt,
                )
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
            try:
                verdict = self._run_independent_review(
                    run_id=run_id,
                    spec=spec,
                    task=verifying,
                    sandbox=sandbox,
                    diff_text=diff_text,
                    check_summary=check_summary,
                    art_dir=art_dir,
                    reviewer=reviewer,
                    design=design,
                )
            except BudgetExhaustedError as exc:
                return self._pause_for_budget(
                    run_id=run_id,
                    task=verifying,
                    reason=f"Paused due to budget limit: {exc}",
                )
            except (_ReviewCancelled, _ReviewOutputError) as exc:
                _write_manifest_md(
                    art_dir=art_dir,
                    task=verifying,
                    diff_text=diff_text,
                    checks=updated_checks,
                    verdict=None,
                    receipt=loop_res.receipt,
                )
                if isinstance(exc, _ReviewCancelled):
                    interrupted.set()
                    return self._return_interrupted_task(run_id, verifying)
                return self._fail_or_requeue(
                    run_id=run_id, task=verifying, reason=str(exc), checks=updated_checks
                )
            if not verdict.passed:
                important_lines = [
                    f"[{f.severity.value}] {f.file_path}:{f.line or '-'} {f.message}"
                    for f in verdict.findings
                    if f.severity == ReviewSeverity.IMPORTANT
                ]
                err_msg = f"Review rejected: {verdict.summary}\n" + "\n".join(important_lines)
                _write_manifest_md(
                    art_dir=art_dir,
                    task=verifying,
                    diff_text=diff_text,
                    checks=updated_checks,
                    verdict=verdict,
                    receipt=loop_res.receipt,
                )
                return self._fail_or_requeue(
                    run_id=run_id,
                    task=verifying,
                    reason=err_msg,
                    checks=updated_checks,
                )

            verdict_path = str(art_dir / f"verdict-cycle-{verifying.review_cycle + 1}.json")
            final_checks = [
                c.model_copy(update={"passed": True, "evidence_ref": verdict_path})
                if c.kind == CheckKind.INDEPENDENT_REVIEW
                else c
                for c in updated_checks
            ]
            try:
                accepted = self.store.update_task_with_lease(
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
                    reject_if_run_inactive=True,
                )
            except ConcurrencyConflictError:
                if self.store.get_run(run_id)["status"] in (
                    RunStatus.PAUSED,
                    RunStatus.CANCELLED,
                ):
                    interrupted.set()
                    return self._return_interrupted_task(run_id, verifying)
                raise

            _copy_matching_scope(sandbox.workspace_path, overlay_dir, verifying.write_scope)
            if mode == RunMode.EXECUTION and apply_to_workspace:
                _copy_matching_scope(
                    sandbox.workspace_path, self.workspace_root, verifying.write_scope
                )
            _write_manifest_md(
                art_dir=art_dir,
                task=accepted,
                diff_text=diff_text,
                checks=final_checks,
                verdict=verdict,
                receipt=loop_res.receipt,
            )
            return accepted
        finally:
            if not archive_failed:
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

    def _return_interrupted_task(self, run_id: str, task: TaskNode) -> TaskNode:
        return self.store.update_task_with_lease(
            run_id,
            task.task_id,
            expected_lease_generation=task.lease_generation,
            mutator=lambda t: t.model_copy(
                update={
                    "status": TaskStatus.READY,
                    "worktree_path": None,
                }
            ),
            actor="chronos",
            event_type="task.turn_cancelled",
        )

    def _pause_for_budget(self, *, run_id: str, task: TaskNode, reason: str) -> TaskNode:
        """Return a task to READY without burning a review cycle when budget is exhausted.

        Preserves any existing non-empty ``last_error`` (such as an unconsumed human hint
        or prior failure feedback) so a zero-call budget pause never wipes retry context.
        """
        return self.store.update_task_with_lease(
            run_id,
            task.task_id,
            expected_lease_generation=task.lease_generation,
            mutator=lambda t: t.model_copy(
                update={
                    "status": TaskStatus.READY,
                    "worktree_path": None,
                    "last_error": (
                        t.last_error if (t.last_error and t.last_error.strip()) else reason
                    ),
                }
            ),
            actor="chronos",
            event_type="task.budget_paused",
        )

    def _monitor_factory(self) -> Callable[[], LLMProvider] | None:
        if self.monitor_provider is not None:
            provider = self.monitor_provider
            return lambda: provider
        model = self.execution_drivers.monitor_model
        if model is None:
            return None
        if self.model_provider is not None:
            factory = self.model_provider
            return lambda: factory(model)
        return lambda: build_provider(
            model=model, project_providers_path=self.workspace_root / ".olympus" / "providers.json"
        )

    def run_execution(
        self,
        *,
        run_id: str,
        apply_to_workspace: bool = False,
        worker_max_steps: int | None = None,
        serial_execution: bool = False,
    ) -> dict[str, Any]:
        """Execute ready tasks respecting DAG dependencies, write_scope, and convergence rules."""
        spec = self._require_confirmed_spec(run_id)
        registry = self.driver_registry.with_factories(
            default_driver_factories(store=self.store, run_id=run_id)
        )
        worker = registry.resolve(
            self.execution_drivers.worker_driver,
            native_factory=lambda model: self._native_driver(run_id, model),
        )
        reviewer = registry.resolve(
            self.execution_drivers.reviewer_driver,
            native_factory=lambda model: self._native_driver(run_id, model, review=True),
        )
        monitor_factory = self._monitor_factory()
        interrupted = Event()
        while True:
            snap = self.store.get_run(run_id)
            if snap["status"] in (RunStatus.PAUSED, RunStatus.CANCELLED):
                return snap
            design: DesignSpec | None = snap.get("design")
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
            if not is_plan_bound_to_upstream(self.store, run_id, snap):
                raise ValueError(
                    f"TaskPlan {plan.plan_id} is stale: its spec_hash/design_hash no longer "
                    "match the current approved upstream artifacts; regenerate the plan first."
                )

            effective_serial = serial_execution or (
                design is not None and design.execution_strategy == ExecutionStrategy.SERIAL_HANDOFF
            )
            try:
                self.store.set_run_status(
                    run_id,
                    RunStatus.RUNNING,
                    expected_revision=int(snap["state_revision"]),
                    actor="chronos",
                    stage="orchestrate",
                    detail={
                        "serial_execution": effective_serial,
                        "execution_drivers": self.execution_drivers.model_dump(mode="json"),
                    },
                )
                break
            except ConcurrencyConflictError:
                continue

        wave_lock = Lock()
        wave_handles: set[ActiveTurnHandle] = set()

        def register_wave_handle(handle: ActiveTurnHandle) -> None:
            with wave_lock:
                wave_handles.add(handle)

        def unregister_wave_handle(handle: ActiveTurnHandle) -> None:
            with wave_lock:
                wave_handles.discard(handle)

        def cancel_wave_peers() -> None:
            with wave_lock:
                peers = list(wave_handles)
            for peer in peers:
                peer.cancel(reason="run_paused")

        limit = (
            1
            if effective_serial
            else self.store.get_run(run_id)["budget_policy"].max_concurrent_tasks
        )
        in_flight: dict[Future[TaskNode], TaskNode] = {}

        def pick_dispatchable(ready: Sequence[TaskNode], capacity: int) -> list[TaskNode]:
            occupied = list(in_flight.values())
            occupied_ids = {task.task_id for task in occupied}
            occupied_scopes = [scope for task in occupied for scope in task.write_scope]
            picked: list[TaskNode] = []
            for task in ready:
                if task.task_id in occupied_ids:
                    continue
                if scopes_overlap(task.write_scope, occupied_scopes):
                    continue
                picked.append(task)
                occupied_scopes.extend(task.write_scope)
                if len(picked) == capacity:
                    break
            return picked

        def drain_in_flight() -> None:
            for future in list(in_flight):
                in_flight.pop(future)
                future.result()

        with ThreadPoolExecutor(max_workers=limit) as pool:
            while True:
                snap = self.store.get_run(run_id)
                if snap["status"] in (RunStatus.PAUSED, RunStatus.CANCELLED):
                    if in_flight:
                        drain_in_flight()
                        continue
                    return snap
                if interrupted.is_set():
                    if in_flight:
                        drain_in_flight()
                        continue
                    try:
                        self.store.set_run_status(
                            run_id,
                            RunStatus.AWAITING_HUMAN,
                            expected_revision=int(snap["state_revision"]),
                            actor="chronos",
                            stage="orchestrate",
                            detail={"reason": "turn_cancelled"},
                        )
                    except ConcurrencyConflictError:
                        continue
                    return self.store.get_run(run_id)
                design = snap.get("design")
                plan = snap["plan"]
                assert plan is not None
                policy = snap["budget_policy"]

                ready: list[TaskNode] = []
                capacity = limit - len(in_flight)
                if capacity > 0:
                    ready = compute_dispatchable_tasks(plan, max_concurrent=limit)
                    candidates = pick_dispatchable(ready, capacity)
                    if candidates and snap["budget_ledger"].can_reserve(policy):
                        for task in candidates:
                            handoff = (
                                _format_handoff_context(plan, current_task_id=task.task_id)
                                if effective_serial
                                else ""
                            )
                            future = pool.submit(
                                self._execute_single_task,
                                run_id=run_id,
                                spec=spec,
                                task_id=task.task_id,
                                mode=snap["mode"],
                                apply_to_workspace=apply_to_workspace,
                                worker_max_steps=worker_max_steps,
                                worker=worker,
                                monitor_factory=monitor_factory,
                                reviewer=reviewer,
                                interrupted=interrupted,
                                design=design,
                                handoff_context=handoff,
                                on_handle_registered=register_wave_handle,
                                on_handle_finished=unregister_wave_handle,
                                on_run_paused=cancel_wave_peers,
                            )
                            in_flight[future] = task
                if in_flight:
                    done, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
                    for future in done:
                        in_flight.pop(future)
                        future.result()
                    continue

                if not ready:
                    all_done = all(
                        t.status in (TaskStatus.ACCEPTED, TaskStatus.CANCELLED)
                        for t in plan.tasks
                    )
                    if (
                        all_done
                        and plan.delivery_checks
                        and any(t.status == TaskStatus.ACCEPTED for t in plan.tasks)
                        and snap["spec"] is not None
                        and snap["spec"].content_hash == spec.content_hash
                        and (plan.spec_id, plan.spec_hash) == (spec.spec_id, spec.content_hash)
                        and not validate_plan(plan, spec=spec, workspace_root=self.workspace_root)
                        and self.store.is_approved(
                            run_id=run_id,
                            target=ApprovalTarget.GOAL_SPEC,
                            target_id=spec.spec_id,
                            target_hash=spec.content_hash,
                        )
                        and self.store.is_approved(
                            run_id=run_id,
                            target=ApprovalTarget.TASK_PLAN,
                            target_id=plan.plan_id,
                            target_hash=plan.content_hash,
                        )
                    ):
                        run_dir = self._run_dir(run_id)
                        receipt = run_delivery_checks(
                            plan,
                            workspace_root=self.workspace_root,
                            overlay_dir=run_dir / "accepted_overlay",
                            run_dir=run_dir,
                        )
                        try:
                            self.store.append_event(
                                run_id=run_id,
                                stage="verify",
                                event_type=DELIVERY_EVENT,
                                actor="chronos",
                                payload=receipt,
                                expected_state_revision=snap["state_revision"],
                            )
                        except ConcurrencyConflictError:
                            continue
                    target_status = (
                        RunStatus.COMPLETED if all_done else RunStatus.AWAITING_HUMAN
                    )
                    try:
                        self.store.set_run_status(
                            run_id,
                            target_status,
                            expected_revision=int(snap["state_revision"]),
                            actor="chronos",
                            stage="orchestrate",
                            detail={
                                "accepted": sum(
                                    1 for t in plan.tasks if t.status == TaskStatus.ACCEPTED
                                ),
                                "total": len(plan.tasks),
                            },
                        )
                    except ConcurrencyConflictError:
                        continue
                    return self.store.get_run(run_id)

                if not snap["budget_ledger"].can_reserve(policy):
                    try:
                        self.store.set_run_status(
                            run_id,
                            RunStatus.AWAITING_HUMAN,
                            expected_revision=int(snap["state_revision"]),
                            actor="chronos",
                            stage="orchestrate",
                            detail={
                                "reason": "budget_exhausted",
                                "accepted": sum(
                                    1 for t in plan.tasks if t.status == TaskStatus.ACCEPTED
                                ),
                                "total": len(plan.tasks),
                            },
                        )
                    except ConcurrencyConflictError:
                        continue
                    return self.store.get_run(run_id)

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
        return RunControl(self.store).retry_task(
            run_id,
            task_id,
            operation_id=operation_id,
            expected_state_revision=expected_state_revision,
            hint=hint,
            actor=actor,
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
        return RunControl(self.store).accept_task_risk(
            run_id,
            task_id,
            operation_id=operation_id,
            expected_state_revision=expected_state_revision,
            reason=reason,
            actor=actor,
        )
