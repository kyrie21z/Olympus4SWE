"""Stage 2 (Orchestrate & Verify): Adaptive Task DAG, sandbox execution, and cold-start review."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import Event, Lock
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError

from olympus.core.contracts import (
    AcceptanceCheck,
    ApprovalRecord,
    ApprovalTarget,
    CheckKind,
    DesignSpec,
    ExecutionStrategy,
    GoalSpec,
    ReviewAxis,
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
from olympus.core.control import RunControl, is_plan_bound_to_upstream
from olympus.core.dag import (
    VERTICAL_SLICE_TOPOLOGIES,
    PlanValidationError,
    compute_dispatchable_tasks,
    scopes_overlap,
    validate_plan,
)
from olympus.core.store import ConcurrencyConflictError, NotFoundError, RunStore
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
    resolve_driver_spec,
)
from olympus.harness.loop import AgentLoop, BudgetExhaustedError, parse_model_json
from olympus.harness.native import NativeDriver
from olympus.harness.structured import run_structured_turn
from olympus.harness.watchdog import Watchdog, WatchdogConfig
from olympus.harness.worktree import (
    SandboxHandle,
    absorb_verification_artifacts,
    archive_sandbox_recovery,
    build_sandbox_env,
    capture_sandbox_diff,
    clean_ephemeral_artifacts,
    create_sandbox,
    load_accepted_deletions,
    load_sandbox_baseline,
    remove_deleted_paths,
    remove_run_sandboxes,
    remove_sandbox,
    save_accepted_deletions,
    save_sandbox_baseline,
    snapshot_sandbox_state,
    sync_upstream_files,
)
from olympus.providers.base import LLMProvider
from olympus.providers.factory import build_provider
from olympus.stages.common import (
    default_artifacts_root,
    format_workspace_inventory,
    resolve_stage_driver,
)
from olympus.tools.aci import IGNORED_DIR_NAMES, WorkspacePolicy, WorkspaceToolkit

if TYPE_CHECKING:
    from olympus.harness.driver import TurnOutcome

_PLANNER_SYSTEM_PROMPT = """You are the Stage 2 Task Planner of Olympus4SWE.
Decompose the confirmed GoalSpec into a minimal TaskPlan JSON.
Rules:
1. Keep simple tasks as a single TaskNode; split into a DAG ONLY when modules are truly separable
   (unless vertical tracer-bullet slicing applies).
2. Tasks that can run concurrently (neither depends on the other) MUST have disjoint write_scope.
3. Every requirement in GoalSpec must appear in at least one task's requirement_refs.
4. Keep tasks cohesive (prefer 1-2 checks per task, at most 3; no redundant prose checks).
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
   invented 'submit_plan' tool, and include no leading prose, $comment, $schema, or extra keys).
   Keep the TaskPlan JSON concise: each task `objective` should be 2-4 crisp sentences referencing
   `requirement_refs` and `design_refs` (never copy-paste full GoalSpec/DesignSpec prose; workers
   already receive both specs).
10. Ground checks in observed reality: acceptance_checks commands MUST only use paths named by
    the Observed workspace files, the task's own write_scope, and tools/packages actually
    available in the environment. NEVER invent uninstalled third-party packages, never
    reference a test directory or file that neither exists nor is created inside the task's
    write_scope, and never assume a module/runner exists without observing it. Prefer standard
    repo test commands (e.g. `python -m pytest <existing_or_task_test_file> -q`). When using
    inline `python -c`, keep it to a short direct behavioral assertion; NEVER write multi-line
    AST, regex, or `git show` diff-inspection scripts (use `independent_review` for structural
    or diff-scope checks so checks cannot contradict each other)."""

_WORKER_SYSTEM_PROMPT = """You are the Stage 2 Execution Engineer (Hephaestus) of Olympus4SWE.
Implement the assigned TaskNode inside the isolated sandbox workspace.
Rules:
1. Work seam-first and vertically: write one failing seam test at the highest observable boundary
   (public API, CLI, or route) for the first acceptance criterion, add the minimal clean
   implementation that makes it pass, then repeat for the next criterion.
2. Keep the implementation deep and minimal: add no speculative abstractions, wrappers, or
   configuration layers for hypothetical callers, and write no implementation-coupled or
   tautological tests that merely restate the code under test.
3. Prefer native file read/search/edit tools, batching independent reads, searches, and edits in
   one turn, instead of serial shell `cat`/`sed`/`grep`/`find` probes; reserve shell execution for
   running tests, linters, and builds.
4. Prefer small, incremental file writes and targeted edits over emitting hundreds of lines in a
   single massive write call: scaffold a compact module first, then extend it in focused edits so
   each turn stays short, verifiable, and resilient against upstream stream drops.
5. Read relevant files first using absolute paths under the sandbox root, and only write or edit
   files inside your assigned write_scope unless the task prompt explicitly authorizes adjacent
   non-forbidden integration edits.
6. Do NOT claim completion without verified tool outputs. If prior review findings are provided,
   fix every Important finding before concluding. Once all acceptance checks pass cleanly, stop
   iterating and deliver your final summary immediately instead of over-polishing."""

_REVIEWER_SYSTEM_PROMPT = """You are the cold-start Independent Verifier (Argus) of Olympus4SWE.
You have ZERO prior conversation history with the implementer. You are strictly read-only: inspect,
probe, but never edit repository files.
Inspect the GoalSpec requirements, out_of_scope boundaries, TaskNode objective, git diff, and actual
command check outputs (plus read-only workspace inspection if needed).
Two-axis rules:
1. Every finding MUST set `axis`:
   - `axis="spec"`: a broken requirement, out-of-scope addition, or runtime/edge-case defect.
   - `axis="standards"`: repo conventions, smells, or style. Always `severity="nit"` and always a
     concrete `file_path` plus `line`.
2. An `axis="spec"` finding may be `severity="important"` ONLY when it includes:
   - `requirement_ref`: the exact GoalSpec requirement id it breaks, and
   - `repro_command`: one self-contained shell command that runs from the sandbox repo root,
     exits NON-ZERO while the defect exists, and exits ZERO once it is fixed.
   The command must not modify repository files. If you cannot produce such a command, report the
   finding as `severity="nit"` prose instead of inventing an unproven blocker.
3. Chronos deterministically re-runs every Axis A `important` `repro_command` in the sandbox and
   drops any finding whose command exits 0 (disproven), is missing, times out, hits a broken probe
   (SyntaxError in inline `python -c`, exit 126/127, command not found), or is on the standards
   axis.
4. Respect TaskNode.write_scope as a hard boundary for requested deliverables: judge ONLY work
   achievable within this TaskNode's objective and write_scope. Never reject a task for missing
   deliverables or test files outside its write_scope (those belong to other tasks in the DAG).
   Adjacent non-forbidden files listed as authorized out-of-declared-scope edits are intentional
   integration changes: audit them for regressions, never reject them for scope alone.
5. Classify each non-standards finding strictly as 'important' (blocks pass), 'nit', or
   'pre_existing'. Set passed=false whenever an 'important' Axis A defect exists; set passed=true
   only if implementation genuinely satisfies the task and no 'important' defects remain.
6. Output ONLY a single JSON object conforming to ReviewVerdict."""


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
    *,
    forbidden_write_scope: Sequence[str] = (),
) -> None:
    """Copy non-ignored files from src_root to dst_root, respecting write_scope and forbidden."""
    if not src_root.exists():
        return
    ignored = IGNORED_DIR_NAMES | {".olympus"}
    exact_rels: set[str] = set()
    glob_patterns: list[str] = []
    if write_scope is not None:
        for pattern in write_scope:
            norm = pattern.replace("\\", "/")
            while norm.startswith("./"):
                norm = norm[2:]
            norm = norm.lstrip("/")
            candidate = src_root / norm
            if candidate.is_file() or candidate.is_symlink():
                exact_rels.add(norm)
            else:
                glob_patterns.append(pattern)
    policy = WorkspacePolicy(
        allowed_write_scope=glob_patterns if write_scope is not None else None,
        forbidden_write_scope=list(forbidden_write_scope),
    )
    for p in sorted(src_root.rglob("*")):
        if not p.is_file():
            continue
        rel_path = p.relative_to(src_root)
        if any(part in ignored for part in rel_path.parts):
            continue
        rel = rel_path.as_posix()
        if forbidden_write_scope and policy.forbidden_match(rel) is not None:
            continue
        if write_scope is not None and rel not in exact_rels and (
            not glob_patterns or not policy.allows_relative_path(rel)
        ):
            continue
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink():
            dst.unlink()
        shutil.copy2(p, dst)


_TreeSnapshotEntry = tuple[Literal["file"], bytes, int] | tuple[Literal["symlink"], str] | None


def _snapshot_tree_paths(
    root: Path, rel_paths: Sequence[str]
) -> dict[str, _TreeSnapshotEntry]:
    """Record pre-promotion file/symlink state (or None if absent) for rollback on OSError."""
    saved: dict[str, _TreeSnapshotEntry] = {}
    for rel in sorted(set(rel_paths)):
        target = root / rel
        if target.is_symlink():
            saved[rel] = ("symlink", os.readlink(target))
        elif target.is_file():
            saved[rel] = ("file", target.read_bytes(), target.stat().st_mode)
        else:
            saved[rel] = None
    return saved


def _restore_tree_paths(
    root: Path, saved: Mapping[str, _TreeSnapshotEntry]
) -> None:
    """Restore pre-promotion file and symlink state on promotion failure."""
    # First remove entries that did not exist before promotion (including under new dirs).
    for rel, state in sorted(saved.items(), reverse=True):
        if state is not None:
            continue
        target = root / rel
        if target.is_symlink() or target.is_file():
            target.unlink(missing_ok=True)
            parent = target.parent
            while parent != root and parent.is_relative_to(root):
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
    # Then restore previously existing files, modes, and symlinks.
    for rel, state in sorted(saved.items()):
        if state is None:
            continue
        target = root / rel
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target, ignore_errors=True)
        elif target.is_symlink():
            target.unlink(missing_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        if state[0] == "symlink":
            if target.is_file() or target.is_symlink():
                target.unlink(missing_ok=True)
            target.symlink_to(state[1])
        else:
            _, data, mode = state
            target.write_bytes(data)
            target.chmod(mode)


_FREEZE_CONSTRAINT_RE = re.compile(
    r"\b(?:(?:do not|don't|must not|never|cannot)\s+"
    r"(?:modify|edit|touch|change|alter|overwrite|delete|remove|rewrite|create)"
    r"|test\s+freeze|frozen|immutable|read[- ]only|forbidden|prohibited|untouched|unchanged)\b",
    re.IGNORECASE,
)
_PATH_TOKEN_RE = re.compile(
    r"`([^`\s]+)`|"
    r"(?<![\w./-])([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.*?\[\]-]*)+|[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8})(?![\w./-])"
)
_IGNORED_PATH_LITERALS = frozenset({"e.g", "i.e", "etc"})


def _normalize_repo_rel_pattern(token: str) -> str | None:
    cleaned = token.strip().strip("`'\",;:()[]{}").replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    cleaned = cleaned.rstrip("/")
    if not cleaned or cleaned.startswith("/") or "://" in cleaned:
        return None
    if cleaned.lower() in _IGNORED_PATH_LITERALS:
        return None
    parts = PurePosixPath(cleaned).parts
    if not parts or any(part in ("..", "") for part in parts):
        return None
    return cleaned


def _is_explicit_spec_path(
    raw: str,
    norm: str,
    workspace_root: Path,
    *,
    backticked: bool,
    single_token: bool = False,
) -> bool:
    """Return True when a spec token is an explicit repo path/glob rather than prose/symbol."""
    raw_cleaned = raw.strip().strip("`'\",;:()[]{}").replace("\\", "/")
    raw_slash = "/" in raw_cleaned
    has_glob = any(ch in norm for ch in "*?[")
    target = workspace_root / norm
    suffix = PurePosixPath(norm).suffix
    has_file_ext = bool(re.fullmatch(r"\.[a-z][a-z0-9]{0,7}", suffix))
    if has_glob or target.is_file():
        return True
    if raw_slash:
        parts = PurePosixPath(norm).parts
        first_seg = parts[0] if parts else ""
        return bool(
            backticked
            or target.is_dir()
            or (first_seg and (workspace_root / first_seg).is_dir())
            or any(part[:1].islower() for part in parts)
            or has_file_ext
        )
    is_dotfile = norm.startswith(".") and len(norm) > 1 and norm[1].isalnum()
    return (backticked or single_token) and (has_file_ext or is_dotfile)


def _extract_spec_path_candidates(text: str, workspace_root: Path) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    candidates: list[str] = []
    if not any(ch.isspace() for ch in stripped):
        is_backticked = stripped.startswith("`") and stripped.endswith("`")
        norm = _normalize_repo_rel_pattern(stripped)
        if norm is not None and _is_explicit_spec_path(
            stripped, norm, workspace_root, backticked=is_backticked, single_token=True
        ):
            return [norm]
        return []
    for match in _PATH_TOKEN_RE.finditer(stripped):
        backticked = match.group(1)
        raw = backticked or match.group(2)
        if not raw:
            continue
        norm = _normalize_repo_rel_pattern(raw)
        if norm is None:
            continue
        if (
            _is_explicit_spec_path(
                raw, norm, workspace_root, backticked=backticked is not None
            )
            and norm not in candidates
        ):
            candidates.append(norm)
    return candidates


def _spec_forbidden_paths(spec: GoalSpec, workspace_root: Path) -> tuple[str, ...]:
    """Extract explicit repo-relative paths/globs frozen or excluded by the confirmed GoalSpec."""
    found: list[str] = []
    for entry in spec.out_of_scope:
        for candidate in _extract_spec_path_candidates(entry, workspace_root):
            if candidate not in found:
                found.append(candidate)
    for entry in spec.constraints:
        if not _FREEZE_CONSTRAINT_RE.search(entry):
            continue
        for candidate in _extract_spec_path_candidates(entry, workspace_root):
            if candidate not in found:
                found.append(candidate)
    return tuple(found)


def _format_spec_context(spec: GoalSpec, *, task: TaskNode | None = None) -> str:
    """Format constraints, assumptions, test seams, clarifications, and referenced requirements."""
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
    lines: list[str] = []
    if task is not None and task.requirement_refs:
        req_by_id = {req.id: req for req in spec.requirements}
        matched_reqs = [
            req_by_id[ref] for ref in dict.fromkeys(task.requirement_refs) if ref in req_by_id
        ]
        if matched_reqs:
            lines.append("Referenced requirements (MUST satisfy):")
            for req in matched_reqs:
                lines.append(f"- {req.id}: {req.description}")
                rendered_ears: set[str] = set()
                for crit in req.ears_criteria:
                    ears_text = crit.to_ears_text()
                    rendered_ears.add(ears_text)
                    lines.append(f"  * {crit.id}: {ears_text}")
                for criterion in req.acceptance_criteria:
                    if criterion not in rendered_ears:
                        lines.append(f"  * {criterion}")
    lines.extend(
        [
            f"Constraints: {json.dumps(spec.constraints, ensure_ascii=False)}",
            f"Assumptions: {json.dumps(spec.assumptions, ensure_ascii=False)}",
            "Test seams (verify behavior at these public boundaries): "
            f"{json.dumps(spec.test_seams, ensure_ascii=False)}",
            f"Clarifications (resolved Q&A): {json.dumps(clarifications, ensure_ascii=False)}",
        ]
    )
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


_INLINE_DIFF_BUDGET = 48_000


def _diff_file_blocks(diff_text: str) -> list[str]:
    if diff_text.startswith("diff --git "):
        blocks: list[str] = []
        current: list[str] = []
        for line in diff_text.splitlines():
            if line.startswith("diff --git ") and current:
                blocks.append("\n".join(current))
                current = []
            current.append(line)
        if current:
            blocks.append("\n".join(current))
        return blocks
    return [block for block in diff_text.split("\n\n") if block.strip()]


def _diff_block_label(block: str) -> str | None:
    rename_from = re.search(r"^rename from (.+)$", block, re.MULTILINE)
    rename_to = re.search(r"^rename to (.+)$", block, re.MULTILINE)
    if rename_from is not None and rename_to is not None:
        return f"renamed: {rename_to.group(1).strip()} (from {rename_from.group(1).strip()})"
    header = re.search(r"^diff --git a/(.+?) b/(.+)$", block, re.MULTILINE)
    if header is not None:
        path = header.group(2).strip()
    else:
        target = re.search(r"^\+\+\+ b/(.+)$", block, re.MULTILINE)
        path = target.group(1).strip() if target is not None else None
    if path is None:
        return None
    if "new file mode" in block or block.startswith("--- /dev/null") or "\n--- /dev/null" in block:
        return f"new: {path}"
    if "deleted file mode" in block or "\n+++ /dev/null" in block:
        return f"deleted: {path}"
    return f"modified: {path}"


def _diff_summary_lines(diff_text: str) -> list[str]:
    lines: list[str] = []
    for block in _diff_file_blocks(diff_text):
        label = _diff_block_label(block)
        if label is None:
            continue
        added, removed = _diff_stats(block)
        lines.append(f"- {label} (+{added}/-{removed})")
    return lines


def _bounded_inline_diff(
    diff_text: str, *, budget: int = _INLINE_DIFF_BUDGET
) -> tuple[str, bool]:
    if len(diff_text) <= budget:
        return diff_text, False
    cut = diff_text.rfind("\n", 0, budget)
    if cut <= 0:
        cut = budget
    return diff_text[:cut].rstrip(), True


def _reopen_or_recreate_sandbox(
    workspace_root: Path,
    *,
    label: str,
    reuse: bool,
    sandbox_root: Path | str | None = None,
) -> SandboxHandle:
    sandbox = create_sandbox(
        workspace_root, label=label, reuse=reuse, sandbox_root=sandbox_root
    )
    if sandbox.reused and load_sandbox_baseline(sandbox) is None:
        remove_sandbox(sandbox)
        return create_sandbox(workspace_root, label=label, sandbox_root=sandbox_root)
    return sandbox


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
                f"- dropped_unverified: {verdict.dropped_unverified_count}",
                "",
            ]
        )
        for finding in verdict.findings:
            lines.append(
                f"- [{finding.axis.value}/{finding.severity.value}] "
                f"{finding.file_path}:{finding.line or '-'} {finding.message} "
                f"(verified={finding.verified}, repro={finding.repro_command or '-'})"
            )
        lines.append("")
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
    matched_modules = [m for m in design.modules if m.id in task_refs]
    if matched_modules:
        lines.append("- Relevant modules:")
        for mod in matched_modules:
            lines.append(
                f"  * {mod.id} ({mod.name}): {mod.responsibility} | "
                f"target_paths={json.dumps(mod.target_paths, ensure_ascii=False)}"
            )
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


_VERTICAL_PLANNER_GUIDANCE = """
Vertical tracer-bullet decomposition (topology={topology}):
- Keep total slice count minimal (1-2 cohesive slices for single-package or focused changes, at
  most 3 for multi-module work) so execution avoids redundant cold-start exploration and review
  overhead across tiny serial slices; group closely related capabilities sharing the same files
  into one cohesive slice instead of emitting one task per requirement.
- If the approved DesignSpec declares non-empty `shared_contract_paths`, emit one Wave 0
  `kind='contract'` task owning those paths before Slice 1.
- Emit Slice 1 as ONE runnable end-to-end tracer bullet (kind='implement', role='engineer')
  that wires core storage/logic to an observable user-facing seam named by GoalSpec.test_seams
  or observable_surfaces, with at least one executable kind='command' seam check proving the
  path runs.
- Emit additional Slices 2..N only when distinct capabilities or modules genuinely merit separate
  deliveries (each carrying its own executable kind='command' seam test).
- NEVER slice horizontally: no models/schema-only wave, no backend-layer wave, no separate
  routes/templates/static wave, and no bulk test-only wave after unverified code waves.
- NEVER emit a catch-all mega-ticket spanning 3+ design modules alongside repo-root scope,
  4+ top-level scopes, or 4+ requirements across 4+ write scopes; split into vertical slices.
- Sequence shared entrypoints with explicit depends_on edges. Concurrent-eligible slices MUST
  have strictly disjoint write_scope.
"""


_REPRO_TIMEOUT_SECONDS = 30.0
_REPRO_OUTPUT_BUDGET = 2_000
_SECRET_ENV_NAME = re.compile(
    r"(?i)(token|secret|password|passwd|api[_-]?key|credential|private[_-]?key)"
)
_TRACEBACK_FILE_RE = re.compile(
    r'^(?:  )?File "([^"]+)", line \d+(?:, in .*)?$', re.MULTILINE
)
_BROKEN_PROBE_EXIT_CODES = frozenset({126, 127})
_PROBE_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "USER",
        "LOGNAME",
        "LANG",
        "TERM",
        "TZ",
        "SHELL",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "UV_CACHE_DIR",
        "UV_PYTHON",
        "PYTEST_ADDOPTS",
        "PYTHONUTF8",
        "PYTHONIOENCODING",
        "NO_COLOR",
        "FORCE_COLOR",
        "CI",
        "SYSTEMROOT",
        "COMSPEC",
        "PATHEXT",
    }
)


@dataclass(frozen=True)
class _ReproResult:
    """Outcome of one sandboxed reviewer repro command."""

    returncode: int | None
    output: str
    broken: bool


def _redact_probe_output(text: str, env: Mapping[str, str]) -> str:
    """Strip secret-looking environment values, then bound the captured output."""
    redacted = text
    for name, value in env.items():
        if len(value) >= 8 and _SECRET_ENV_NAME.search(name):
            redacted = redacted.replace(value, "***")
    if len(redacted) > _REPRO_OUTPUT_BUDGET:
        redacted = f"{redacted[:_REPRO_OUTPUT_BUDGET]}\n... (truncated)"
    return redacted


def _probe_is_broken(*, command: str, returncode: int | None, output: str) -> bool:
    """Recognize command failures that prove a broken probe, not a reproduced defect."""
    del command
    if returncode is None:
        return True
    if returncode in _BROKEN_PROBE_EXIT_CODES:
        return True
    if returncode == 2 and "syntax error" in output.lower():
        return True
    if "SyntaxError" in output:
        files = _TRACEBACK_FILE_RE.findall(output)
        return bool(files) and files[-1] in ("<string>", "<stdin>")
    return False


def _build_probe_env(
    probe_ws: Path,
    *,
    sandbox_root: Path,
    host_root: Path,
    tmp_dir: Path,
    home_dir: Path,
) -> dict[str, str]:
    """Build a minimal allowlisted environment for isolated reviewer repro probes."""
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in _PROBE_ENV_ALLOWLIST or key.startswith("LC_"):
            env[key] = value
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)
    env["HOME"] = str(home_dir)

    venv_candidates: list[Path] = [sandbox_root / ".venv", host_root / ".venv"]
    current_prefix = Path(sys.executable).resolve().parent.parent
    if (
        (current_prefix / "pyvenv.cfg").exists()
        and (current_prefix / "bin" / "pytest").exists()
        and current_prefix not in venv_candidates
    ):
        venv_candidates.append(current_prefix)
    for venv_dir in venv_candidates:
        bin_dir = venv_dir / "bin"
        if bin_dir.is_dir():
            env["VIRTUAL_ENV"] = str(venv_dir)
            existing_path = env.get("PATH", "")
            env["PATH"] = f"{bin_dir}{os.pathsep}{existing_path}" if existing_path else str(bin_dir)
            break

    python_roots: list[str] = []
    src_dir = probe_ws / "src"
    if src_dir.is_dir():
        python_roots.append(str(src_dir))
    python_roots.append(str(probe_ws))
    env["PYTHONPATH"] = os.pathsep.join(python_roots)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for var in ("TMPDIR", "TEMP", "TMP", "SQLITE_TMPDIR"):
        env[var] = str(tmp_dir)
    return env


def _format_rejection_reason(verdict: ReviewVerdict) -> str:
    """Render verified blockers with the exact failing command and observed output."""
    lines = [f"Review rejected: {verdict.summary}"]
    for finding in verdict.findings:
        if finding.severity != ReviewSeverity.IMPORTANT:
            continue
        header = (
            f"[important/{finding.axis.value}] {finding.file_path}:"
            f"{finding.line or '-'} {finding.message}"
        )
        if finding.requirement_ref:
            header += f" (requirement_ref={finding.requirement_ref})"
        lines.append(header)
        if finding.repro_command:
            lines.append(f"  failing repro_command: {finding.repro_command}")
        if finding.verification_output:
            lines.append("  observed output:")
            lines.extend(
                f"    {line}" for line in finding.verification_output.splitlines()
            )
    return "\n".join(lines)


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
        sandbox_root: Path | str | None = None,
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
            else default_artifacts_root(store, self.workspace_root)
        )
        self.sandbox_root = Path(sandbox_root).resolve() if sandbox_root is not None else None
        self._promotion_lock = Lock()

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

    def _latest_human_guidance(self, run_id: str) -> str:
        """Return the most recent human recovery/revision hint recorded on the run."""
        try:
            events = self.store.list_events(run_id)
        except NotFoundError:
            return ""
        for event in reversed(events):
            if event.event_type != "session.continued":
                continue
            feedback = event.payload.get("feedback")
            if isinstance(feedback, str) and feedback.strip():
                return feedback.strip()
        return ""

    def propose_plan(
        self,
        *,
        run_id: str,
        auto_approve: bool = False,
        approver: str = "zeus",
        max_steps: int = 16,
        topology: str = "minimal_patch",
        use_design: bool = True,
    ) -> TaskPlan:
        """Generate and deterministically validate a TaskPlan.

        Accepts an approved DesignSpec (with or without a GoalSpec) or, for backward
        compatibility, an approved GoalSpec alone. ``modular_product`` decomposes the design's
        modules into Wave 0/1/2 tasks, while ``tdd_first``/``vertical_slices`` slice the design
        into vertical tracer-bullet user stories.
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

        vertical = topology in VERTICAL_SLICE_TOPOLOGIES
        modular = design is not None and (topology == "modular_product" or vertical)
        if vertical:
            guidance_block = _VERTICAL_PLANNER_GUIDANCE.format(topology=topology)
        elif modular:
            guidance_block = _MODULAR_PLANNER_GUIDANCE.format(topology=topology)
        else:
            guidance_block = ""
        toolkit = WorkspaceToolkit(self.workspace_root, read_only=True)
        driver, driver_model = resolve_stage_driver(
            settings=self.execution_drivers,
            registry=self.driver_registry,
            provider=self.provider,
            store=self.store,
            run_id=run_id,
        )
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
            if modular:
                allowed_refs = sorted(
                    {m.id for m in design.modules} | {i.id for i in design.interfaces}
                )
                allowed_refs_csv = ", ".join(allowed_refs) if allowed_refs else "(none)"
                binding_lines.append(
                    "Allowed task.design_refs IDs (use ONLY IDs from this exact set; NEVER "
                    f"data_models or invariant names): {allowed_refs_csv}."
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

        guidance = self._latest_human_guidance(run_id)
        human_guidance_block = (
            "Latest human recovery guidance (honor it and ground checks in it):\n"
            f"{guidance}\n\n"
            if guidance
            else ""
        )
        base_prompt = (
            f"Workspace root: {self.workspace_root}\n"
            f"{inventory}\n"
            f"{human_guidance_block}"
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
            f"{guidance_block}"
            f"{chr(10).join(binding_lines)}\n"
            f"Return TaskPlan JSON matching schema:\n{schema_json}"
        )

        def stamp_binding(candidate: TaskPlan) -> TaskPlan:
            clean_tasks = [
                task.model_copy(
                    update={
                        "status": TaskStatus.WAITING,
                        "attempt": 0,
                        "review_cycle": 0,
                        "in_place_recoveries": 0,
                        "lease_generation": 0,
                        "worktree_path": None,
                        "commit_sha": None,
                        "last_error": None,
                        "accepted_risk_reason": None,
                        "verified_repro_commands": [],
                        "acceptance_checks": [
                            check.model_copy(update={"passed": False, "evidence_ref": None})
                            for check in task.acceptance_checks
                        ],
                    }
                )
                for task in candidate.tasks
            ]
            update = _plan_binding(design=design, spec=spec)
            update["tasks"] = clean_tasks
            return candidate.model_copy(update=update)

        def plan_gate(candidate: TaskPlan) -> list[str]:
            return validate_plan(
                candidate,
                spec=spec,
                design=design if modular else None,
                workspace_root=self.workspace_root,
                topology=topology,
            )

        def raise_exhausted(candidate: TaskPlan, violations: list[str]) -> TaskPlan:
            raise PlanValidationError(violations)

        plan = run_structured_turn(
            loop=loop,
            driver=driver,
            driver_model=driver_model,
            store=self.store,
            run_id=run_id,
            workspace_root=self.workspace_root,
            actor="orchestrate:planner",
            system_prompt=_PLANNER_SYSTEM_PROMPT + guidance_block,
            user_prompt=base_prompt,
            model_cls=TaskPlan,
            trace_path=self._run_dir(run_id) / "traces" / "planner.jsonl",
            max_steps=max_steps,
            max_validation_retries=2,
            validation_header="[PLAN VALIDATION FAILED]",
            validation_instruction="Fix all violations and return valid TaskPlan JSON.",
            transform=stamp_binding,
            validator=plan_gate,
            on_validation_exhausted=raise_exhausted,
            on_event=self.on_driver_event,
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
        diff_path: Path,
        check_summary: str,
        art_dir: Path,
        reviewer: tuple[AgentDriver, str | None],
        design: DesignSpec | None = None,
        advisory_out_of_scope: Sequence[str] = (),
        resolved_repro_commands: Sequence[str] = (),
    ) -> ReviewVerdict:
        relevant_reqs = [r for r in spec.requirements if r.id in task.requirement_refs]
        schema_json = json.dumps(ReviewVerdict.model_json_schema(), ensure_ascii=False)
        incremental = task.review_cycle > 0 or bool(resolved_repro_commands)
        prior_ctx = (
            f"\nPrevious cycle rejection feedback:\n{task.last_error}\n"
            if incremental and task.last_error
            else ""
        )
        resolved_ctx = (
            "\nChronos 0-token re-check: these previously verified failing repro commands "
            f"NOW PASS after the repair: {json.dumps(list(resolved_repro_commands))}\n"
            "Audit only for REGRESSIONS in the incremental diff. Do not re-report these "
            "resolved defects unless you supply a new repro_command that fails right now.\n"
            if resolved_repro_commands
            else ""
        )
        task_summary = task.model_dump_json(
            include={"task_id", "title", "objective", "requirement_refs", "write_scope"}
        )
        reqs_json = json.dumps(
            [r.model_dump(mode="json") for r in relevant_reqs], ensure_ascii=False
        )
        oos_json = json.dumps(spec.out_of_scope, ensure_ascii=False)
        spec_ctx = _format_spec_context(spec)
        design_ctx = _format_design_context(design, task=task)
        diff_summary = "\n".join(_diff_summary_lines(diff_text)) or "- (none detected)"
        inline_diff, diff_truncated = _bounded_inline_diff(diff_text)
        diff_ctx = (
            f"Changed files (per-file summary):\n{diff_summary}\n\n"
            f"Diff (inline{' truncated' if diff_truncated else ''}):\n{inline_diff}\n"
        )
        advisory_ctx = (
            "\nAuthorized out-of-declared-scope adjacent edits (non-forbidden; this task ran "
            "without concurrent siblings; audit for regressions, do not reject for scope alone): "
            f"{json.dumps(list(advisory_out_of_scope), ensure_ascii=False)}\n"
            if advisory_out_of_scope
            else ""
        )
        if diff_truncated:
            diff_ctx += (
                f"\nThe inline diff above was truncated at {_INLINE_DIFF_BUDGET} characters. "
                f"The complete diff is saved at: {diff_path}\n"
                "Inspect only the specific files you need with your read-only tools "
                "(read_file, grep_files, glob_files) before deciding; do not assume the "
                "inline diff is complete.\n"
            )
        user_prompt = (
            f"Sandbox root: {sandbox.workspace_path}\n"
            f"TaskNode: {task_summary}\n"
            f"Hard write_scope boundary (do NOT require edits to files outside these paths): "
            f"{json.dumps(task.write_scope, ensure_ascii=False)}\n"
            f"{advisory_ctx}"
            f"Relevant requirements: {reqs_json}\n"
            f"{spec_ctx}\n"
            f"{design_ctx}"
            f"Out of scope (MUST NOT be added): {oos_json}\n"
            f"Incremental review cycle={task.review_cycle + 1} (incremental_only={incremental})\n"
            f"{prior_ctx}\n"
            f"{resolved_ctx}"
            f"Command check outputs:\n{check_summary}\n\n"
            f"{diff_ctx}\n"
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
            max_steps=12,
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
        (art_dir / f"argus-verdict-cycle-{task.review_cycle + 1}.json").write_text(
            verdict.model_dump_json(indent=2), encoding="utf-8"
        )
        return verdict

    def _run_repro_command(
        self, *, command: str, sandbox: SandboxHandle
    ) -> _ReproResult:
        """Execute one reviewer repro command inside an isolated temp snapshot."""
        try:
            with tempfile.TemporaryDirectory(prefix="olympus-probe-") as tmp_root:
                probe_ws = Path(tmp_root) / "ws"
                probe_tmp = Path(tmp_root) / "tmp"
                probe_home = Path(tmp_root) / "home"
                probe_tmp.mkdir(parents=True, exist_ok=True)
                probe_home.mkdir(parents=True, exist_ok=True)
                shutil.copytree(
                    sandbox.workspace_path,
                    probe_ws,
                    symlinks=True,
                    ignore_dangling_symlinks=True,
                    ignore=shutil.ignore_patterns(".git", ".olympus", ".venv", "__pycache__"),
                )
                # Strip symlinks pointing outside the isolated probe snapshot so relative writes
                # inside probe_ws can never follow a link out to the real sandbox or host disk.
                resolved_probe = probe_ws.resolve()
                for dirpath, dirnames, filenames in os.walk(probe_ws):
                    root_path = Path(dirpath)
                    for name in list(dirnames) + list(filenames):
                        entry = root_path / name
                        if not entry.is_symlink():
                            continue
                        try:
                            target = entry.resolve(strict=False)
                        except OSError:
                            entry.unlink(missing_ok=True)
                            continue
                        if not target.is_relative_to(resolved_probe):
                            entry.unlink(missing_ok=True)
                            if name in dirnames:
                                dirnames.remove(name)
                env = _build_probe_env(
                    probe_ws,
                    sandbox_root=sandbox.workspace_path,
                    host_root=self.workspace_root,
                    tmp_dir=probe_tmp,
                    home_dir=probe_home,
                )
                try:
                    proc = subprocess.run(
                        ["bash", "--norc", "--noprofile", "-c", command],
                        cwd=probe_ws,
                        env=env,
                        capture_output=True,
                        text=True,
                        errors="replace",
                        timeout=_REPRO_TIMEOUT_SECONDS,
                        check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    captured = "\n".join(
                        part
                        for part in (
                            exc.stdout if isinstance(exc.stdout, str) else "",
                            exc.stderr if isinstance(exc.stderr, str) else "",
                        )
                        if part
                    )
                    output = _redact_probe_output(
                        f"probe timed out after {_REPRO_TIMEOUT_SECONDS:g}s\n{captured}".strip(),
                        env,
                    )
                    return _ReproResult(returncode=None, output=output, broken=True)
                combined = "\n".join(part for part in (proc.stdout, proc.stderr) if part).strip()
                return _ReproResult(
                    returncode=proc.returncode,
                    output=_redact_probe_output(combined, env),
                    broken=_probe_is_broken(
                        command=command, returncode=proc.returncode, output=combined
                    ),
                )
        except (OSError, shutil.Error, UnicodeError) as exc:
            return _ReproResult(
                returncode=None,
                output=f"probe failed to execute: {type(exc).__name__}",
                broken=True,
            )

    def _verify_review_verdict(
        self,
        *,
        run_id: str,
        spec: GoalSpec,
        task: TaskNode,
        sandbox: SandboxHandle,
        verdict: ReviewVerdict,
        art_dir: Path,
    ) -> ReviewVerdict:
        """Deterministically keep only run-proven Axis A blockers from an Argus verdict."""
        valid_req_ids = frozenset(req.id for req in spec.requirements)
        findings: list[ReviewFinding] = []
        verified_commands: list[str] = []
        dropped = 0
        standards_count = 0
        for index, finding in enumerate(verdict.findings, start=1):
            if finding.severity != ReviewSeverity.IMPORTANT:
                findings.append(finding)
                continue
            if finding.axis == ReviewAxis.STANDARDS:
                standards_count += 1
                findings.append(
                    finding.model_copy(
                        update={
                            "severity": ReviewSeverity.NIT,
                            "verified": None,
                            "verification_output": None,
                        }
                    )
                )
                continue
            req_ref = (finding.requirement_ref or "").strip()
            if not req_ref or req_ref not in valid_req_ids:
                dropped += 1
                findings.append(
                    finding.model_copy(
                        update={
                            "severity": ReviewSeverity.NIT,
                            "verified": False,
                            "verification_output": (
                                "dropped: missing or unknown requirement_ref "
                                f"({req_ref or 'none'})"
                            ),
                        }
                    )
                )
                continue
            command = (finding.repro_command or "").strip()
            if not command:
                dropped += 1
                findings.append(
                    finding.model_copy(
                        update={
                            "severity": ReviewSeverity.NIT,
                            "verified": False,
                            "verification_output": "dropped: no repro_command supplied",
                        }
                    )
                )
                continue
            result = self._run_repro_command(command=command, sandbox=sandbox)
            (art_dir / f"verify-cycle-{task.review_cycle + 1}-{index}.log").write_text(
                f"finding: {finding.id}\ncommand: {command}\n"
                f"exit_code: {result.returncode}\nbroken_probe: {result.broken}\n"
                f"--- output ---\n{result.output}\n",
                encoding="utf-8",
            )
            if result.broken or result.returncode == 0:
                dropped += 1
                reason = (
                    "dropped: broken reviewer probe"
                    if result.broken
                    else "dropped: repro_command exited 0 (disproven)"
                )
                findings.append(
                    finding.model_copy(
                        update={
                            "severity": ReviewSeverity.NIT,
                            "verified": False,
                            "verification_output": f"{reason}\n{result.output}".strip(),
                        }
                    )
                )
                continue
            findings.append(
                finding.model_copy(
                    update={"verified": True, "verification_output": result.output}
                )
            )
            if command not in verified_commands:
                verified_commands.append(command)
        passed = not any(f.severity == ReviewSeverity.IMPORTANT for f in findings)
        verified_verdict = verdict.model_copy(
            update={
                "findings": findings,
                "passed": passed,
                "dropped_unverified_count": dropped,
                "summary": (
                    verdict.summary
                    if not dropped and not standards_count
                    else f"{verdict.summary} [chronos: dropped={dropped}, "
                    f"standards={standards_count}]"
                ),
            }
        )
        self.store.append_event(
            run_id=run_id,
            stage="verify",
            event_type="review.verdict_verified",
            actor="chronos",
            payload={
                "task_id": task.task_id,
                "attempt": task.attempt,
                "review_cycle": task.review_cycle + 1,
                "passed": passed,
                "verified_important": sum(
                    1
                    for f in findings
                    if f.severity == ReviewSeverity.IMPORTANT and f.verified is True
                ),
                "dropped_unverified": dropped,
                "standards_count": standards_count,
                "verified_repro_commands": list(verified_commands),
            },
        )
        return verified_verdict

    def _recheck_verified_repro_commands(
        self,
        *,
        run_id: str,
        task: TaskNode,
        sandbox: SandboxHandle,
        art_dir: Path,
    ) -> tuple[list[str], list[str], list[str]]:
        """0-token pre-check: re-run the previous cycle's proven failing commands."""
        failures: list[str] = []
        still_verified: list[str] = []
        resolved: list[str] = []
        results: list[dict[str, Any]] = []
        for index, command in enumerate(task.verified_repro_commands, start=1):
            result = self._run_repro_command(command=command, sandbox=sandbox)
            passed = result.returncode == 0
            (art_dir / f"recheck-cycle-{task.review_cycle + 1}-{index}.log").write_text(
                f"command: {command}\nexit_code: {result.returncode}\n"
                f"broken_probe: {result.broken}\n"
                f"--- output ---\n{result.output}\n",
                encoding="utf-8",
            )
            results.append(
                {
                    "command": command,
                    "exit_code": result.returncode,
                    "passed": passed,
                    "broken": result.broken,
                }
            )
            if passed:
                resolved.append(command)
                continue
            if result.broken:
                continue
            observed = (
                f"exit {result.returncode}" if result.returncode is not None else "timeout"
            )
            still_verified.append(command)
            failures.append(
                f"Previously verified defect still reproduces: `{command}` "
                f"({observed})\n{result.output[-800:]}"
            )
        self.store.append_event(
            run_id=run_id,
            stage="verify",
            event_type="review.repro_rechecked",
            actor="chronos",
            payload={
                "task_id": task.task_id,
                "attempt": task.attempt,
                "review_cycle": task.review_cycle + 1,
                "passed": not failures,
                "short_circuited": bool(failures),
                "llm_calls": 0,
                "results": results,
            },
        )
        return failures, still_verified, resolved

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
        strict_scope: bool = True,
        sibling_write_scopes: Sequence[str] = (),
    ) -> TaskNode:
        leased = self.store.acquire_task_lease(run_id, task_id, actor="chronos")
        art_dir = self._run_dir(run_id) / "artifacts" / leased.task_id / f"attempt-{leased.attempt}"
        art_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir = self._run_dir(run_id) / "accepted_overlay"
        label = f"{run_id}-{leased.task_id}"
        sandbox = _reopen_or_recreate_sandbox(
            self.workspace_root,
            label=label,
            reuse=leased.attempt > 1,
            sandbox_root=self.sandbox_root,
        )

        retain_sandbox = False

        def retain(node: TaskNode) -> TaskNode:
            nonlocal retain_sandbox
            retain_sandbox = True
            return node

        try:
            forbidden_contracts = (
                design.shared_contract_paths
                if design is not None and leased.kind != TaskKind.CONTRACT
                else ()
            )
            spec_forbidden = _spec_forbidden_paths(spec, self.workspace_root)
            hard_forbidden = tuple(
                dict.fromkeys((*forbidden_contracts, *sibling_write_scopes, *spec_forbidden))
            )
            with self._promotion_lock:
                accepted_deletions = sorted(load_accepted_deletions(self._run_dir(run_id)))
            if sandbox.reused:
                task_baseline = load_sandbox_baseline(sandbox)
                assert task_baseline is not None
                task_baseline = sync_upstream_files(
                    sandbox,
                    overlay_dir,
                    task_baseline,
                    write_scope=leased.write_scope,
                    deleted_paths=accepted_deletions,
                )
            else:
                # Apply upstream deletions FIRST so file-to-directory replacements (e.g.
                # foo -> foo/bar.py) unlink the old HEAD file before copying overlay files.
                remove_deleted_paths(sandbox.workspace_path, accepted_deletions)
                _copy_matching_scope(overlay_dir, sandbox.workspace_path, write_scope=None)
                task_baseline = snapshot_sandbox_state(sandbox)
            save_sandbox_baseline(sandbox, task_baseline)
            rework_ctx = (
                f"\n\n[PRIOR REJECTION / ERROR TO FIX]\n{leased.last_error}\n"
                if leased.last_error
                else ""
            )
            repair_ctx = (
                "\n\n[IN-PLACE REPAIR]\n"
                "This sandbox was retained from your previous attempt: the files you already "
                "changed are still present on disk. Inspect their current contents and apply "
                "targeted edits that resolve the reported rejection instead of rewriting "
                "unrelated files from scratch. Remove or relocate any paths the rejection "
                "reports as outside write_scope; nothing was reverted for you.\n"
                if sandbox.reused
                else ""
            )
            checks_json = json.dumps(
                [
                    c.model_dump(include={"id", "kind", "description", "command"}, mode="json")
                    for c in leased.acceptance_checks
                ],
                ensure_ascii=False,
            )
            spec_ctx = _format_spec_context(spec, task=leased)
            design_ctx = _format_design_context(design, task=leased)
            handoff_block = f"{handoff_context}\n" if handoff_context else ""
            step_guidance = (
                f"Step budget: {worker_max_steps} turns max (keep exploration focused <= 6 steps, "
                "implement edits early, verify with targeted commands, and stop once passing).\n"
                if worker_max_steps is not None
                else "Execution discipline: keep exploration focused, implement edits early, "
                "verify with targeted commands, and stop once all acceptance checks pass.\n"
            )
            scope_note = (
                "\nScope note: no concurrent sibling task shares this run wave. Besides "
                "write_scope, adjacent NON-FORBIDDEN repo files needed for integration "
                "(e.g. a package `__init__.py` re-export) may be created or edited; workspace "
                "escapes and forbidden paths stay hard-blocked.\n"
                if not strict_scope
                else "\nScope note: concurrent wave active. Strictly stay inside write_scope; "
                "sibling task scopes and forbidden shared contracts are hard-blocked.\n"
            )
            worker_prompt = (
                f"Sandbox workspace root: {sandbox.workspace_path}\n"
                f"Task {leased.task_id}: {leased.title}\n"
                f"Objective: {leased.objective}\n"
                f"{step_guidance}"
                f"Allowed write_scope (relative): {json.dumps(leased.write_scope)}\n"
                f"{scope_note}"
                f"{spec_ctx}\n"
                f"{design_ctx}"
                f"{handoff_block}"
                f"Out of scope: {json.dumps(spec.out_of_scope, ensure_ascii=False)}\n"
                f"Acceptance checks: {checks_json}"
                f"{repair_ctx}"
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
                strict_scope=strict_scope,
                write_scope=leased.write_scope,
                forbidden_globs=list(hard_forbidden),
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
                    store=self.store,
                    run_id=run_id,
                    baseline=task_baseline,
                    forbidden_globs=list(hard_forbidden),
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
                retained = (
                    str(sandbox.workspace_path) if sandbox.workspace_path.exists() else None
                )
                retain_sandbox = retained is not None
                paused_task = RunControl(self.store).pause_for_watchdog(
                    run_id, task=leased, diagnosis=diagnosis,
                    retained_sandbox=retained,
                )
                if on_run_paused is not None:
                    on_run_paused()
                return paused_task

            run_status = self.store.get_run(run_id)["status"]
            if run_status in (RunStatus.PAUSED, RunStatus.CANCELLED):
                interrupted.set()
                retain_sandbox = run_status == RunStatus.PAUSED
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

            diff_text = capture_sandbox_diff(sandbox, task_baseline)
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
                    return retain(self._return_interrupted_task(run_id, verifying))
                return retain(
                    self._pause_for_budget(
                        run_id=run_id,
                        task=verifying,
                        reason=f"Paused due to budget limit: {loop_res.stop_reason}",
                    )
                )

            removed_artifacts = clean_ephemeral_artifacts(sandbox, task_baseline)
            if removed_artifacts:
                self.store.append_event(
                    run_id=run_id,
                    stage="verify",
                    event_type="task.ephemeral_artifacts_cleaned",
                    actor="chronos",
                    payload={
                        "task_id": verifying.task_id,
                        "attempt": verifying.attempt,
                        "removed_paths": removed_artifacts,
                    },
                )
                diff_text = capture_sandbox_diff(sandbox, task_baseline)
                diff_file.write_text(diff_text, encoding="utf-8")

            gate = run_diff_gate(
                sandbox,
                task_baseline,
                allowed_write_scope=verifying.write_scope,
                forbidden_globs=hard_forbidden,
                strict_scope=strict_scope,
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
                        "strict_scope": gate.strict_scope,
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
                        "advisory_out_of_scope": list(gate.advisory_out_of_scope),
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
                return retain(
                    self._fail_or_requeue(
                        run_id=run_id,
                        task=verifying,
                        reason=gate.rejection_message(),
                        checks=verifying.acceptance_checks,
                    )
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
                return retain(
                    self._fail_or_requeue(
                        run_id=run_id,
                        task=verifying,
                        reason=f"Worker did not complete cleanly: {loop_res.stop_reason}",
                        checks=verifying.acceptance_checks,
                    )
                )

            updated_checks, cmd_failures = self._run_command_checks(
                run_id=run_id,
                task=verifying,
                sandbox=sandbox,
                art_dir=art_dir,
            )
            task_baseline = absorb_verification_artifacts(
                sandbox, task_baseline, write_scope=verifying.write_scope
            )
            save_sandbox_baseline(sandbox, task_baseline)
            if cmd_failures:
                _write_manifest_md(
                    art_dir=art_dir,
                    task=verifying,
                    diff_text=diff_text,
                    checks=updated_checks,
                    verdict=None,
                    receipt=loop_res.receipt,
                )
                return retain(
                    self._fail_or_requeue(
                        run_id=run_id,
                        task=verifying,
                        reason="\n".join(cmd_failures),
                        checks=updated_checks,
                    )
                )

            # 0-token pre-check: re-run any previously proven failing repro commands
            # (including after human in-place retry_task which resets review_cycle to 0)
            # before spending any reviewer tokens.
            resolved_repro_commands: list[str] = []
            if verifying.verified_repro_commands:
                (
                    repro_failures,
                    still_verified_repros,
                    resolved_repro_commands,
                ) = self._recheck_verified_repro_commands(
                    run_id=run_id,
                    task=verifying,
                    sandbox=sandbox,
                    art_dir=art_dir,
                )
                if repro_failures:
                    _write_manifest_md(
                        art_dir=art_dir,
                        task=verifying,
                        diff_text=diff_text,
                        checks=updated_checks,
                        verdict=None,
                        receipt=loop_res.receipt,
                    )
                    return retain(
                        self._fail_or_requeue(
                            run_id=run_id,
                            task=verifying,
                            reason="\n".join(repro_failures),
                            checks=updated_checks,
                            verified_repro_commands=still_verified_repros,
                        )
                    )
                verifying = self.store.update_task_with_lease(
                    run_id,
                    verifying.task_id,
                    expected_lease_generation=verifying.lease_generation,
                    mutator=lambda t: t.model_copy(
                        update={"verified_repro_commands": []}
                    ),
                    actor="chronos",
                    event_type="review.repro_commands_cleared",
                )

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
                    diff_path=diff_file,
                    check_summary=check_summary,
                    art_dir=art_dir,
                    reviewer=reviewer,
                    design=design,
                    advisory_out_of_scope=gate.advisory_out_of_scope,
                    resolved_repro_commands=resolved_repro_commands,
                )
            except BudgetExhaustedError as exc:
                return retain(
                    self._pause_for_budget(
                        run_id=run_id,
                        task=verifying,
                        reason=f"Paused due to budget limit: {exc}",
                    )
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
                    return retain(self._return_interrupted_task(run_id, verifying))
                return retain(
                    self._fail_or_requeue(
                        run_id=run_id, task=verifying, reason=str(exc), checks=updated_checks
                    )
                )

            verdict = self._verify_review_verdict(
                run_id=run_id,
                spec=spec,
                task=verifying,
                sandbox=sandbox,
                verdict=verdict,
                art_dir=art_dir,
            )
            (art_dir / f"verdict-cycle-{verifying.review_cycle + 1}.json").write_text(
                verdict.model_dump_json(indent=2), encoding="utf-8"
            )
            self.store.append_event(
                run_id=run_id,
                stage="verify",
                event_type="review.verdict_recorded",
                actor="argus",
                payload={
                    "task_id": verifying.task_id,
                    "attempt": verifying.attempt,
                    "passed": verdict.passed,
                    "important_count": sum(
                        1
                        for f in verdict.findings
                        if f.severity == ReviewSeverity.IMPORTANT
                    ),
                    "nit_count": sum(
                        1 for f in verdict.findings if f.severity == ReviewSeverity.NIT
                    ),
                    "dropped_unverified_count": verdict.dropped_unverified_count,
                },
            )
            if not verdict.passed:
                _write_manifest_md(
                    art_dir=art_dir,
                    task=verifying,
                    diff_text=diff_text,
                    checks=updated_checks,
                    verdict=verdict,
                    receipt=loop_res.receipt,
                )
                return retain(
                    self._fail_or_requeue(
                        run_id=run_id,
                        task=verifying,
                        reason=_format_rejection_reason(verdict),
                        checks=updated_checks,
                        verified_repro_commands=[
                            f.repro_command
                            for f in verdict.findings
                            if f.severity == ReviewSeverity.IMPORTANT
                            and f.verified is True
                            and f.repro_command
                        ],
                    )
                )

            verdict_path = str(art_dir / f"verdict-cycle-{verifying.review_cycle + 1}.json")
            final_checks = [
                c.model_copy(update={"passed": True, "evidence_ref": verdict_path})
                if c.kind == CheckKind.INDEPENDENT_REVIEW
                else c
                for c in updated_checks
            ]
            removed_artifact_set = set(removed_artifacts)
            promotion_paths: list[str] = []
            seen_paths: set[str] = set()
            deleted_paths: list[str] = []
            recreated_paths: set[str] = set()
            promotion_policy = WorkspacePolicy(
                allowed_write_scope=verifying.write_scope,
                forbidden_write_scope=hard_forbidden,
            )
            for rel in gate.changed_paths:
                if rel in removed_artifact_set:
                    continue
                if promotion_policy.forbidden_match(rel) is not None:
                    continue
                candidate = sandbox.workspace_path / rel
                if not candidate.is_file() and not candidate.is_symlink():
                    deleted_paths.append(rel)
                    continue
                recreated_paths.add(rel)
                if rel not in seen_paths:
                    seen_paths.add(rel)
                    promotion_paths.append(rel)

            with self._promotion_lock:
                run_status = self.store.get_run(run_id)["status"]
                if run_status in (RunStatus.PAUSED, RunStatus.CANCELLED):
                    interrupted.set()
                    retain_sandbox = run_status == RunStatus.PAUSED
                    return self._return_interrupted_task(run_id, verifying)
                touched_paths = [*promotion_paths, *deleted_paths]
                prev_deletions = load_accepted_deletions(self._run_dir(run_id))
                overlay_backup = _snapshot_tree_paths(overlay_dir, touched_paths)
                workspace_backup = (
                    _snapshot_tree_paths(self.workspace_root, touched_paths)
                    if mode == RunMode.EXECUTION and apply_to_workspace
                    else None
                )
                try:
                    if mode == RunMode.EXECUTION and apply_to_workspace:
                        remove_deleted_paths(
                            self.workspace_root,
                            deleted_paths,
                            forbidden_write_scope=hard_forbidden,
                            ignore_errors=False,
                        )
                        if promotion_paths:
                            _copy_matching_scope(
                                sandbox.workspace_path,
                                self.workspace_root,
                                promotion_paths,
                            )
                    remove_deleted_paths(
                        overlay_dir,
                        deleted_paths,
                        forbidden_write_scope=hard_forbidden,
                        ignore_errors=False,
                    )
                    if promotion_paths:
                        _copy_matching_scope(
                            sandbox.workspace_path,
                            overlay_dir,
                            promotion_paths,
                        )
                    next_deletions = set(prev_deletions)
                    next_deletions.difference_update(recreated_paths)
                    next_deletions.update(deleted_paths)
                    save_accepted_deletions(self._run_dir(run_id), next_deletions)
                except OSError as exc:
                    rollback_errors: list[str] = []
                    if workspace_backup is not None:
                        try:
                            _restore_tree_paths(self.workspace_root, workspace_backup)
                        except OSError as rb_exc:
                            rollback_errors.append(f"workspace rollback: {rb_exc}")
                    try:
                        _restore_tree_paths(overlay_dir, overlay_backup)
                    except OSError as rb_exc:
                        rollback_errors.append(f"overlay rollback: {rb_exc}")
                    try:
                        save_accepted_deletions(self._run_dir(run_id), prev_deletions)
                    except OSError as rb_exc:
                        rollback_errors.append(f"deletions rollback: {rb_exc}")
                    reason = f"Failed to promote task changes: {exc}"
                    if rollback_errors:
                        reason = f"{reason} (rollback errors: {'; '.join(rollback_errors)})"
                    _write_manifest_md(
                        art_dir=art_dir,
                        task=verifying,
                        diff_text=diff_text,
                        checks=final_checks,
                        verdict=verdict,
                        receipt=loop_res.receipt,
                    )
                    return retain(
                        self._fail_or_requeue(
                            run_id=run_id,
                            task=verifying,
                            reason=reason,
                            checks=final_checks,
                        )
                    )

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
                            "verified_repro_commands": [],
                        }
                    ),
                    actor="chronos",
                    event_type="task.accepted",
                    reject_if_run_inactive=True,
                )
            except ConcurrencyConflictError:
                run_status = self.store.get_run(run_id)["status"]
                if run_status in (RunStatus.PAUSED, RunStatus.CANCELLED):
                    interrupted.set()
                    retain_sandbox = run_status == RunStatus.PAUSED
                    return self._return_interrupted_task(run_id, verifying)
                raise

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
            if not retain_sandbox:
                remove_sandbox(sandbox)

    def _fail_or_requeue(
        self,
        *,
        run_id: str,
        task: TaskNode,
        reason: str,
        checks: list[AcceptanceCheck],
        verified_repro_commands: Sequence[str] | None = None,
    ) -> TaskNode:
        next_cycle = task.review_cycle + 1
        exhausted = next_cycle >= task.max_cycles
        next_status = TaskStatus.BLOCKED if exhausted else TaskStatus.READY
        event_type = "task.blocked" if exhausted else "task.requeued"
        update: dict[str, Any] = {
            "status": next_status,
            "review_cycle": next_cycle,
            "acceptance_checks": checks,
            "last_error": reason,
            "worktree_path": None,
        }
        if verified_repro_commands is not None:
            update["verified_repro_commands"] = list(verified_repro_commands)
        return self.store.update_task_with_lease(
            run_id,
            task.task_id,
            expected_lease_generation=task.lease_generation,
            mutator=lambda t: t.model_copy(update=update),
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
        strict_scope: bool | None = None,
    ) -> dict[str, Any]:
        """Execute ready tasks respecting DAG dependencies, write_scope, and convergence rules.

        ``strict_scope`` defaults to ``None`` (auto): enforced only for waves with concurrent
        sibling tasks. Explicit ``True``/``False`` overrides that heuristic.
        """
        spec = self._require_confirmed_spec(run_id)
        registry = self.driver_registry.with_factories(
            default_driver_factories(
                store=self.store,
                run_id=run_id,
                provider=self.provider,
                execution_drivers=self.execution_drivers,
            )
        )
        worker = registry.resolve(
            resolve_driver_spec(self.execution_drivers.worker_driver, provider=self.provider),
            native_factory=lambda model: self._native_driver(run_id, model),
        )
        reviewer = registry.resolve(
            resolve_driver_spec(self.execution_drivers.reviewer_driver, provider=self.provider),
            native_factory=lambda model: self._native_driver(run_id, model, review=True),
        )
        monitor_factory = self._monitor_factory()
        interrupted = Event()
        while True:
            snap = self.store.get_run(run_id)
            if snap["status"] in (RunStatus.PAUSED, RunStatus.CANCELLED):
                if snap["status"] == RunStatus.CANCELLED:
                    remove_run_sandboxes(
                        self.workspace_root, run_id, sandbox_root=self.sandbox_root
                    )
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
        wave_paused = False

        def register_wave_handle(handle: ActiveTurnHandle) -> None:
            with wave_lock:
                already_paused = wave_paused
                if not already_paused:
                    wave_handles.add(handle)
            if already_paused:
                handle.cancel(reason="run_paused")

        def unregister_wave_handle(handle: ActiveTurnHandle) -> None:
            with wave_lock:
                wave_handles.discard(handle)

        def cancel_wave_peers() -> None:
            nonlocal wave_paused
            with wave_lock:
                wave_paused = True
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
                if any(dep in occupied_ids for dep in task.depends_on):
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
                    if snap["status"] == RunStatus.CANCELLED:
                        remove_run_sandboxes(self.workspace_root, run_id)
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
                            wave_peers = [*in_flight.values(), *candidates]
                            task_strict_scope = (
                                strict_scope
                                if strict_scope is not None
                                else not effective_serial and len(wave_peers) > 1
                            )
                            sibling_write_scopes = tuple(
                                scope
                                for peer in wave_peers
                                if peer.task_id != task.task_id
                                for scope in peer.write_scope
                            )
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
                                strict_scope=task_strict_scope,
                                sibling_write_scopes=sibling_write_scopes,
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
                            sandbox_root=self.sandbox_root,
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
                    if target_status == RunStatus.COMPLETED:
                        remove_run_sandboxes(
                            self.workspace_root, run_id, sandbox_root=self.sandbox_root
                        )
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
