"""Stage 2 (Orchestrate & Verify): Adaptive Task DAG, sandbox execution, and cold-start review."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from olympus.core.contracts import (
    AcceptanceCheck,
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    CheckKind,
    DesignSpec,
    ExecutionStrategy,
    GoalSpec,
    ReviewFinding,
    ReviewSeverity,
    ReviewVerdict,
    RunMode,
    RunStatus,
    TaskKind,
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
from olympus.harness.hooks import default_write_scope_hook
from olympus.harness.loop import AgentLoop
from olympus.harness.worktree import (
    SandboxHandle,
    build_sandbox_env,
    capture_sandbox_diff,
    commit_sandbox_baseline,
    create_sandbox,
    remove_sandbox,
)
from olympus.providers.base import LLMProvider, ToolSpec
from olympus.stages.common import (
    extract_json_payload,
    format_workspace_inventory,
    parse_model_json,
    run_structured_loop,
    strip_schema_meta_keys,
)
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
   repo-relative paths only. When writing inline 'python -c "..."' commands, NEVER chain
   '; try:', '; except:', '; for ', '; if ', or '; def ' on a single line (Python raises
   SyntaxError)—use real '\\n' newlines, simple expressions/asserts, or 'python -m pytest ...'.
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
command check outputs (plus read-only workspace inspection only if diff is truncated).
Rules:
1. Check for genuine spec drift, out_of_scope additions, stub code, edge-case bugs, security risks,
   or actual writes outside write_scope. Do NOT reject for stylistic preferences, harmless helper
   classes (e.g. default monotonic clock), or pre-existing / baseline files unchanged in git diff.
2. Classify each finding strictly as 'important' (blocks pass), 'nit', or 'pre_existing'.
3. Set passed=false whenever any 'important' finding exists; set passed=true when the implementation
   genuinely satisfies the task with passing tests and no 'important' defects remain.
4. Call the `submit_review` tool (strongly preferred) OR reply with concise JSON/Markdown stating
   passed, summary, and any important/nit findings."""

_SUBMIT_REVIEW_TOOL = ToolSpec(
    name="submit_review",
    description=(
        "Submit the final independent code review verdict for this task. Prefer calling this "
        "tool directly instead of hand-formatting raw JSON in text."
    ),
    parameters={
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "summary": {"type": "string"},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "severity": {
                            "type": "string",
                            "enum": ["important", "nit", "pre_existing"],
                        },
                        "file_path": {"type": "string"},
                        "line": {"type": ["integer", "null"]},
                        "message": {"type": "string"},
                        "suggestion": {"type": ["string", "null"]},
                    },
                    "required": ["severity", "message"],
                },
            },
        },
        "required": ["passed", "summary"],
    },
)

_NEGATIVE_PROSE_RE = re.compile(
    r"\b(fail|failed|reject|rejected|important|critical|bug|broken|violation|must fix)\b"
    r"|不通过|阻塞|严重问题",
    re.IGNORECASE,
)
_POSITIVE_PROSE_RE = re.compile(
    r"\b(pass|passed|lgtm|approved|looks good|no important|all checks pass|no defects)\b"
    r"|全部通过|审核通过",
    re.IGNORECASE,
)


def _normalize_review_payload(
    raw: dict[str, Any],
    *,
    task_id: str,
    attempt: int,
    incremental: bool,
) -> dict[str, Any]:
    raw_findings = raw.get("findings") or []
    cleaned: list[dict[str, Any]] = []
    if isinstance(raw_findings, list):
        for idx, item in enumerate(raw_findings, start=1):
            if not isinstance(item, dict):
                continue
            sev = str(item.get("severity") or "important").strip().lower()
            if sev not in ("important", "nit", "pre_existing"):
                sev = "important"
            msg = str(item.get("message") or item.get("summary") or "").strip()
            if not msg:
                continue
            line_raw = item.get("line")
            line_val = int(line_raw) if isinstance(line_raw, int) and line_raw >= 1 else None
            cleaned.append(
                {
                    "id": str(item.get("id") or f"F{idx}"),
                    "severity": sev,
                    "file_path": str(item.get("file_path") or ""),
                    "line": line_val,
                    "message": msg,
                    "suggestion": (
                        str(item["suggestion"]) if item.get("suggestion") is not None else None
                    ),
                }
            )
    passed_val = bool(raw.get("passed", False))
    summary_val = str(raw.get("summary") or "").strip() or (
        "Verified cleanly." if passed_val else "Independent review identified defects."
    )
    return {
        "task_id": task_id,
        "attempt": attempt,
        "passed": passed_val,
        "summary": summary_val,
        "findings": cleaned,
        "checked_commit": (
            str(raw["checked_commit"]) if raw.get("checked_commit") is not None else None
        ),
        "incremental_only": incremental,
    }


def _try_parse_review_verdict(
    raw_text: str,
    *,
    task_id: str,
    attempt: int,
    incremental: bool,
) -> ReviewVerdict | None:
    text = raw_text.strip()
    if not text:
        return None
    with suppress(json.JSONDecodeError, ValidationError):
        return parse_model_json(text, ReviewVerdict).model_copy(
            update={"task_id": task_id, "attempt": attempt, "incremental_only": incremental}
        )
    candidate = extract_json_payload(text)
    with suppress(json.JSONDecodeError, ValidationError, ValueError):
        data = strip_schema_meta_keys(json.loads(candidate))
        if isinstance(data, dict) and "passed" in data:
            normalized = _normalize_review_payload(
                data,
                task_id=task_id,
                attempt=attempt,
                incremental=incremental,
            )
            return ReviewVerdict.model_validate(normalized)
    return None


def _coerce_prose_verdict(
    raw_text: str,
    *,
    task_id: str,
    attempt: int,
    incremental: bool,
) -> ReviewVerdict:
    """Convert freeform Markdown/prose review output cleanly without crashing the DAG."""
    text = raw_text.strip()
    if not text:
        return ReviewVerdict(
            task_id=task_id,
            attempt=attempt,
            passed=False,
            summary="Independent verifier stopped without producing a final verdict.",
            findings=[
                ReviewFinding(
                    id="F1",
                    severity=ReviewSeverity.IMPORTANT,
                    file_path="",
                    message="Reviewer returned empty output; requeuing for verification.",
                )
            ],
            incremental_only=incremental,
        )
    has_positive = bool(_POSITIVE_PROSE_RE.search(text))
    has_negative = bool(_NEGATIVE_PROSE_RE.search(text))
    approved = has_positive and not has_negative
    summary = text if len(text) <= 1500 else text[:1497] + "..."
    findings = (
        []
        if approved
        else [
            ReviewFinding(
                id="F1",
                severity=ReviewSeverity.IMPORTANT,
                file_path="",
                message=summary[:600],
            )
        ]
    )
    return ReviewVerdict(
        task_id=task_id,
        attempt=attempt,
        passed=approved,
        summary=summary,
        findings=findings,
        incremental_only=incremental,
    )


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
- Every design module id MUST appear in at least one task's design_refs; never invent ids.
- Tasks in different modules MUST NOT share overlapping write_scope.
- Command acceptance_checks MUST only reference modules or test paths inside the task's own
  write_scope or its transitive depends_on ancestors (NEVER import a concurrent sibling task's
  modules in a task check; put cross-module integration checks in a Wave 2 task that depends_on
  all needed Wave 1 tasks), and MUST NOT silence failures with 2>/dev/null.
"""

_SERIAL_PLANNER_GUIDANCE = """
Cohesive sequential decomposition (topology={topology}, strategy=serial_handoff):
- Prefer 1-2 cohesive sequential TaskNodes (`depends_on` chaining them cleanly) rather than
  splitting every single file into micro-tasks. A single cohesive implement+test task or a
  2-step (`core+contract -> integration+tests`) chain is ideal for small repositories.
- Every design module id MUST appear in at least one task's design_refs; multiple design_refs
  may share the same cohesive task when running serially. If `design.shared_contract_paths`
  is non-empty, put those paths in a leading `kind='contract'` task's `write_scope`.
- Prefer `python -m pytest ...` checks against test files over fragile multi-line inline scripts.
- Command acceptance_checks MUST only reference modules or test paths inside the task's own
  write_scope or its transitive depends_on ancestors, and NEVER silence failures with 2>/dev/null.
"""


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
        planner_toolkit = WorkspaceToolkit(self.workspace_root, read_only=True)
        inventory = format_workspace_inventory(planner_toolkit)
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
            use_serial_guidance = (
                design.execution_strategy == ExecutionStrategy.SERIAL_HANDOFF
                or len(design.modules) <= 1
            )
            guidance = (
                _SERIAL_PLANNER_GUIDANCE if use_serial_guidance else _MODULAR_PLANNER_GUIDANCE
            )
            modular_block = guidance.format(topology=topology)
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
            f"{context_block}"
            f"{modular_block}"
            f"{chr(10).join(binding_lines)}\n"
            f"Return TaskPlan JSON matching schema:\n{schema_json}"
        )

        plan: TaskPlan | None = None
        last_violations: list[str] = []
        prompt = base_prompt
        for attempt in range(3):
            candidate = run_structured_loop(
                loop=loop,
                system_prompt=_PLANNER_SYSTEM_PROMPT + modular_block,
                user_prompt=prompt,
                model_cls=TaskPlan,
            ).model_copy(update=_plan_binding(design=design, spec=spec))
            violations = validate_plan(
                candidate,
                spec=spec,
                design=design if modular else None,
                workspace_root=self.workspace_root,
            )
            if not violations:
                plan = candidate
                break
            last_violations = violations
            if attempt < 2:
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
        snap = self.store.get_run(run_id)
        plan: TaskPlan | None = snap["plan"]
        if plan is None:
            raise ValueError(f"Run {run_id} has no TaskPlan to approve.")
        spec: GoalSpec | None = snap["spec"]
        design: DesignSpec | None = snap["design"]

        spec_bound = bool(
            spec
            and plan.spec_hash
            and plan.spec_hash == spec.content_hash
            and self.store.is_approved(
                run_id=run_id,
                target=ApprovalTarget.GOAL_SPEC,
                target_id=spec.spec_id,
                target_hash=spec.content_hash,
            )
        )
        design_bound = bool(
            design
            and plan.design_hash
            and plan.design_hash == design.content_hash
            and self.store.is_approved(
                run_id=run_id,
                target=ApprovalTarget.DESIGN_SPEC,
                target_id=design.design_id,
                target_hash=design.content_hash,
            )
        )
        if not (spec_bound or design_bound):
            if spec is None and design is None:
                raise ValueError(
                    f"Run {run_id} has no GoalSpec or DesignSpec; complete Stage 1 first."
                )
            raise ValueError(
                "TaskPlan does not match any approved upstream artifact "
                f"(spec_hash={plan.spec_hash}, design_hash={plan.design_hash}); "
                "regenerate the plan first."
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
        design: DesignSpec | None = None,
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
            synthesize_on_max_steps=True,
        )
        relevant_reqs = [r for r in spec.requirements if r.id in task.requirement_refs]
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
            f"Relevant requirements: {reqs_json}\n"
            f"{spec_ctx}\n"
            f"{design_ctx}"
            f"Out of scope (MUST NOT be added): {oos_json}\n"
            f"Incremental review cycle={task.review_cycle + 1} (incremental_only={incremental})\n"
            f"{prior_ctx}\n"
            f"Command check outputs:\n{check_summary}\n\n"
            f"Git diff / changes:\n{diff_text}\n\n"
            "Submit your verdict by calling the `submit_review` tool (preferred), "
            'or reply concisely with `{ "passed": bool, "summary": str, "findings": [...] }` '
            "or a brief Markdown review summary."
        )

        submitted: list[ReviewVerdict] = []

        def _handle_submit_review(args: dict[str, Any]) -> str:
            try:
                normalized = _normalize_review_payload(
                    args,
                    task_id=task.task_id,
                    attempt=task.attempt,
                    incremental=incremental,
                )
                verdict_obj = ReviewVerdict.model_validate(normalized)
            except (ValidationError, ValueError) as exc:
                return f"Error: invalid review verdict ({exc}); fix arguments and resubmit."
            submitted.append(verdict_obj)
            return verdict_obj.model_dump_json()

        res = review_loop.run(
            system_prompt=_REVIEWER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            extra_tools=(_SUBMIT_REVIEW_TOOL,),
            extra_tool_handlers={"submit_review": _handle_submit_review},
            terminal_tools={"submit_review"},
        )
        if not res.completed and res.stop_reason.startswith("budget_exhausted"):
            raise RuntimeError(
                f"AgentLoop stopped before producing final output: {res.stop_reason}"
            )

        verdict: ReviewVerdict | None = submitted[-1] if submitted else None
        if verdict is None and res.completed:
            verdict = _try_parse_review_verdict(
                res.final_content,
                task_id=task.task_id,
                attempt=task.attempt,
                incremental=incremental,
            )
        if (
            verdict is None
            and res.completed
            and not res.final_content.strip().startswith("{")
            and (
                _POSITIVE_PROSE_RE.search(res.final_content)
                or _NEGATIVE_PROSE_RE.search(res.final_content)
            )
        ):
            # Freeform Markdown/prose review from subagent: accept directly without nagging.
            verdict = _coerce_prose_verdict(
                res.final_content,
                task_id=task.task_id,
                attempt=task.attempt,
                incremental=incremental,
            )
        if verdict is None and res.messages:
            repair_res = review_loop.complete_repair(
                prior_messages=res.messages,
                correction=(
                    "Please reply with a concise valid JSON object containing only "
                    '`{"passed": bool, "summary": str, "findings": [...]}` (or plain text '
                    "stating PASSED or listing blocking defects)."
                ),
                step=1,
            )
            if not repair_res.completed and repair_res.stop_reason.startswith("budget_exhausted"):
                raise RuntimeError(
                    f"AgentLoop stopped before producing final output: {repair_res.stop_reason}"
                )
            verdict = _try_parse_review_verdict(
                repair_res.final_content,
                task_id=task.task_id,
                attempt=task.attempt,
                incremental=incremental,
            ) or _coerce_prose_verdict(
                repair_res.final_content,
                task_id=task.task_id,
                attempt=task.attempt,
                incremental=incremental,
            )
        if verdict is None:
            verdict = _coerce_prose_verdict(
                res.final_content,
                task_id=task.task_id,
                attempt=task.attempt,
                incremental=incremental,
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
        design: DesignSpec | None = None,
        handoff_context: str = "",
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
        commit_sandbox_baseline(sandbox)

        try:
            toolkit = WorkspaceToolkit(
                sandbox.workspace_path,
                allowed_write_scope=leased.write_scope,
                read_only=False,
                env_root=self.workspace_root,
            )
            forbidden_contracts = (
                design.shared_contract_paths
                if design is not None and leased.kind != TaskKind.CONTRACT
                else ()
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
                task_id=leased.task_id,
                pre_tool_hook=default_write_scope_hook(
                    leased.write_scope,
                    forbidden_globs=forbidden_contracts,
                    workspace_root=sandbox.workspace_path,
                ),
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
            worker_prompt = (
                f"Sandbox workspace root: {sandbox.workspace_path}\n"
                f"Task {leased.task_id}: {leased.title}\n"
                f"Objective: {leased.objective}\n"
                f"Step budget: {worker_max_steps} turns max (keep exploration focused <= 6 steps, "
                f"implement edits early, verify with targeted commands, and stop once passing).\n"
                f"Allowed write_scope (relative): {json.dumps(leased.write_scope)}\n"
                f"{spec_ctx}\n"
                f"{design_ctx}"
                f"{handoff_block}"
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
            verdict = self._run_independent_review(
                run_id=run_id,
                spec=spec,
                task=verifying,
                sandbox=sandbox,
                diff_text=diff_text,
                check_summary=check_summary,
                art_dir=art_dir,
                design=design,
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
        serial_execution: bool = False,
    ) -> dict[str, Any]:
        """Execute ready tasks respecting DAG dependencies, write_scope, and convergence rules."""
        spec = self._require_confirmed_spec(run_id)
        snap = self.store.get_run(run_id)
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

        effective_serial = serial_execution or (
            design is not None and design.execution_strategy == ExecutionStrategy.SERIAL_HANDOFF
        )
        self.store.set_run_status(
            run_id,
            RunStatus.RUNNING,
            actor="chronos",
            stage="orchestrate",
            detail={"serial_execution": effective_serial},
        )

        while True:
            snap = self.store.get_run(run_id)
            if snap["status"] in (RunStatus.PAUSED, RunStatus.CANCELLED):
                return snap
            design = snap.get("design")
            plan = snap["plan"]
            assert plan is not None
            policy = snap["budget_policy"]

            ready = compute_dispatchable_tasks(
                plan,
                max_concurrent=1 if effective_serial else policy.max_concurrent_tasks,
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
                handoff = (
                    _format_handoff_context(plan, current_task_id=ready[0].task_id)
                    if effective_serial
                    else ""
                )
                self._execute_single_task(
                    run_id=run_id,
                    spec=spec,
                    task_id=ready[0].task_id,
                    mode=snap["mode"],
                    apply_to_workspace=apply_to_workspace,
                    worker_max_steps=worker_max_steps,
                    design=design,
                    handoff_context=handoff,
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
                            design=design,
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
