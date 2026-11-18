"""Deterministic DAG validation, write_scope conflict detection, and ready-task scheduling."""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path, PurePosixPath

from olympus.core.contracts import (
    CheckKind,
    DesignSpec,
    ExecutionStrategy,
    GoalSpec,
    TaskKind,
    TaskNode,
    TaskPlan,
    TaskStatus,
)


class PlanValidationError(ValueError):
    """Raised when a TaskPlan violates structural, dependency, coverage, or scope invariants."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


# Buzzwords that describe an intention rather than a verifiable behavior.
_VAGUE_TERMS: tuple[str, ...] = (
    "美观",
    "体验流畅",
    "界面友好",
    "合理处理",
    "等等",
    "及相关功能",
    "user-friendly",
    "user friendly",
    "seamlessly",
    "handle appropriately",
    "etc.",
)

# A vague term is tolerated only when the same sentence carries a measurable detail.
_CONCRETE_DETAIL_RE = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s*(?:ms|s|sec|secs|seconds|m|min|h|%|percent|px|kb|mb|gb|"
    r"qps|rps|req/s|requests|items|records|rows|lines|bytes|次|条|个|行|秒|毫秒|分钟)\b)"
    r"|(?:\b(?:HTTP\s*)?[1-5]\d{2}\b)"
    r"|(?:`[^`]+`)"
    r"|(?:\b[\w./-]+\.(?:py|ts|tsx|js|jsx|md|json|ya?ml|sql|toml|html|css)\b)"
    r"|(?:\b(?:GET|POST|PUT|PATCH|DELETE)\s+/[^\s]*)",
    re.IGNORECASE,
)

_IMPLEMENTATION_LEAK_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bCREATE\s+TABLE\b", re.IGNORECASE),
    re.compile(r"\bALTER\s+TABLE\b", re.IGNORECASE),
    re.compile(r"\bsrc/[A-Za-z0-9_./-]+\.py\b"),
)


def _vague_terms(text: str) -> list[str]:
    if not text:
        return []
    lowered = text.lower()
    hits = [term for term in _VAGUE_TERMS if term.lower() in lowered]
    if not hits:
        return []
    if _CONCRETE_DETAIL_RE.search(text):
        return []
    return hits


def _implementation_leaks(text: str, constraints: list[str]) -> list[str]:
    if not text:
        return []
    allowed = " | ".join(constraints).lower()
    leaks: list[str] = []
    for pattern in _IMPLEMENTATION_LEAK_RES:
        for match in pattern.finditer(text):
            token = match.group(0)
            if token.lower() in allowed:
                continue
            if token not in leaks:
                leaks.append(token)
    return leaks


def normalize_scope_path(raw: str) -> str:
    """Normalize a repo-relative POSIX scope pattern and reject escapes."""
    cleaned = raw.strip().replace("\\", "/")
    if not cleaned:
        raise ValueError("Empty write_scope path is not allowed")
    if cleaned.startswith("/"):
        raise ValueError(f"write_scope must be repo-relative, got absolute path: {raw}")
    is_dir_hint = cleaned.endswith("/")
    pure = PurePosixPath(cleaned.rstrip("/") or ".")
    if ".." in pure.parts:
        raise ValueError(f"write_scope cannot contain '..': {raw}")
    norm = pure.as_posix()
    if norm == ".":
        return "."
    return f"{norm}/" if is_dir_hint else norm


def paths_overlap(a_raw: str, b_raw: str) -> bool:
    """Return True if two normalized repo-relative paths or directory prefixes can overlap."""
    a = normalize_scope_path(a_raw).rstrip("/")
    b = normalize_scope_path(b_raw).rstrip("/")
    if a == "." or b == ".":
        return True
    if a == b:
        return True
    return a.startswith(f"{b}/") or b.startswith(f"{a}/")


def scopes_overlap(scope_a: list[str], scope_b: list[str]) -> list[tuple[str, str]]:
    """Return all conflicting pairs between two write_scope lists."""
    conflicts: list[tuple[str, str]] = []
    for pa in scope_a:
        for pb in scope_b:
            if paths_overlap(pa, pb):
                conflicts.append((pa, pb))
    return conflicts


def _build_ancestors(tasks: list[TaskNode]) -> dict[str, set[str]]:
    by_id = {t.task_id: t for t in tasks}
    ancestors: dict[str, set[str]] = {}

    def visit(tid: str, visiting: set[str]) -> set[str]:
        if tid in ancestors:
            return ancestors[tid]
        if tid in visiting:
            return set()
        visiting.add(tid)
        acc: set[str] = set()
        for dep in by_id[tid].depends_on:
            if dep in by_id:
                acc.add(dep)
                acc |= visit(dep, visiting)
        visiting.remove(tid)
        ancestors[tid] = acc
        return acc

    for t in tasks:
        visit(t.task_id, set())
    return ancestors


def validate_goal_spec(
    spec: GoalSpec,
    *,
    strict_ears: bool = True,
    require_out_of_scope: bool = True,
) -> list[str]:
    """Zero-token gate for the 5-element + EARS GoalSpec contract.

    Covers deliverable kind, observable surfaces, actors, per-requirement EARS criteria,
    anti-vague lexical hygiene, premature implementation leaks, out-of-scope boundaries,
    and unresolved blocking ambiguities.
    """
    violations: list[str] = []
    actor_pool = {actor.strip().lower() for actor in spec.actors if actor.strip()}

    if strict_ears:
        if spec.deliverable_kind is None:
            violations.append("GoalSpec is missing deliverable_kind (web_app/rest_api/cli/...)")
        if not [s for s in spec.observable_surfaces if s.strip()]:
            violations.append("GoalSpec is missing observable_surfaces")
        if not actor_pool:
            violations.append("GoalSpec is missing actors")

    for req in spec.requirements:
        if strict_ears:
            if not req.actor or not req.actor.strip():
                violations.append(f"Requirement {req.id} is missing actor")
            elif req.actor.strip().lower() not in actor_pool:
                violations.append(
                    f"Requirement {req.id} actor '{req.actor}' is not declared in GoalSpec.actors"
                )
            if not req.ears_criteria:
                violations.append(f"Requirement {req.id} has no ears_criteria")

        scanned: list[tuple[str, str]] = [
            (f"description of {req.id}", req.description),
            *[
                (f"acceptance criterion {idx} of {req.id}", text)
                for idx, text in enumerate(req.acceptance_criteria, start=1)
            ],
            *[
                (f"EARS outcome {crit.id} of {req.id}", crit.outcome)
                for crit in req.ears_criteria
            ],
        ]
        for label, text in scanned:
            vague = _vague_terms(text)
            if vague:
                violations.append(
                    f"Vague unquantified wording in {label}: "
                    f"{', '.join(vague)} (add a measurable threshold or concrete artifact)"
                )
            leaks = _implementation_leaks(text, spec.constraints)
            if leaks:
                violations.append(
                    f"Premature internal implementation leak in {label}: {', '.join(leaks)}"
                )

    if require_out_of_scope and not [item for item in spec.out_of_scope if item.strip()]:
        violations.append("GoalSpec is missing out_of_scope boundaries")

    blockers = spec.unresolved_blockers()
    if blockers:
        violations.append(f"GoalSpec still has unresolved blockers: {', '.join(blockers)}")
    return violations


def validate_design_spec(
    design: DesignSpec,
    spec: GoalSpec | None = None,
    *,
    strict_v2: bool = False,
) -> list[str]:
    """Zero-token gate for invariants, shared contract seams, module cycles, and coverage."""
    violations: list[str] = []

    module_ids: set[str] = set()
    for module in design.modules:
        if module.id in module_ids:
            violations.append(f"Duplicate design module id: {module.id}")
        module_ids.add(module.id)

    normalized: dict[str, list[str]] = {}
    for module in design.modules:
        paths: list[str] = []
        for raw in module.target_paths:
            try:
                paths.append(normalize_scope_path(raw))
            except ValueError as exc:
                violations.append(f"Design module {module.id} invalid target_path ({raw}): {exc}")
        normalized[module.id] = paths

    shared_norm: list[str] = []
    for raw in design.shared_contract_paths:
        try:
            shared_norm.append(normalize_scope_path(raw))
        except ValueError as exc:
            violations.append(f"DesignSpec invalid shared_contract_path ({raw}): {exc}")

    ordered_ids = list(module_ids)
    for i in range(len(ordered_ids)):
        for j in range(i + 1, len(ordered_ids)):
            left, right = ordered_ids[i], ordered_ids[j]
            try:
                conflicts = scopes_overlap(normalized.get(left, []), normalized.get(right, []))
            except ValueError:
                conflicts = []
            if conflicts:
                pairs = ", ".join(f"{a}<->{b}" for a, b in conflicts)
                violations.append(
                    f"Design modules {left} and {right} have overlapping target_paths: {pairs}"
                )

    if shared_norm:
        for mod_id in ordered_ids:
            try:
                shared_conflicts = scopes_overlap(shared_norm, normalized.get(mod_id, []))
            except ValueError:
                shared_conflicts = []
            if shared_conflicts:
                pairs = ", ".join(f"{a}<->{b}" for a, b in shared_conflicts)
                violations.append(
                    f"shared_contract_paths overlaps module {mod_id} target_paths: {pairs}"
                )

    invariant_ids: set[str] = set()
    for inv in design.shared_invariants:
        if inv.id in invariant_ids:
            violations.append(f"Duplicate global invariant id: {inv.id}")
        invariant_ids.add(inv.id)
        for mod_id in inv.applies_to:
            if mod_id not in module_ids:
                violations.append(
                    f"GlobalInvariant {inv.id} applies_to unknown module '{mod_id}'"
                )

    entity_names = {entity.name for entity in design.data_models}
    for entity in design.data_models:
        if entity.owner_module_id not in module_ids:
            violations.append(
                f"DataEntity {entity.name} owner_module_id '{entity.owner_module_id}' "
                "is not a declared design module"
            )

    interface_ids: set[str] = set()
    mod_edges: dict[str, set[str]] = {mid: set() for mid in module_ids}
    mod_in_degree: dict[str, int] = {mid: 0 for mid in module_ids}
    for iface in design.interfaces:
        if iface.id in interface_ids:
            violations.append(f"Duplicate interface contract id: {iface.id}")
        interface_ids.add(iface.id)
        if iface.provider_module_id not in module_ids:
            violations.append(
                f"Interface {iface.id} provider_module_id '{iface.provider_module_id}' "
                "is not a declared design module"
            )
        for consumer in iface.consumer_module_ids:
            if consumer not in module_ids:
                violations.append(
                    f"Interface {iface.id} consumer_module_id '{consumer}' "
                    "is not a declared design module"
                )
            elif consumer == iface.provider_module_id:
                violations.append(
                    f"Interface {iface.id} introduces a self-dependency cycle on {consumer}"
                )
            elif (
                iface.provider_module_id in module_ids
                and iface.provider_module_id not in mod_edges[consumer]
            ):
                mod_edges[consumer].add(iface.provider_module_id)
                mod_in_degree[iface.provider_module_id] += 1
        for ref in iface.data_entity_refs:
            if ref not in entity_names:
                violations.append(
                    f"Interface {iface.id} references unknown DataEntity '{ref}'"
                )

    # Kahn topological sort on consumer -> provider dependencies to forbid module cycles
    mod_queue = deque(mid for mid, deg in mod_in_degree.items() if deg == 0)
    visited_mods = 0
    while mod_queue:
        cur = mod_queue.popleft()
        visited_mods += 1
        for dep_provider in mod_edges[cur]:
            mod_in_degree[dep_provider] -= 1
            if mod_in_degree[dep_provider] == 0:
                mod_queue.append(dep_provider)
    if visited_mods < len(module_ids):
        cyclic = sorted(mid for mid, deg in mod_in_degree.items() if deg > 0)
        violations.append(
            f"DesignSpec module dependency graph contains a cycle involving: {', '.join(cyclic)}"
        )

    if len(design.modules) > 1 and not design.interfaces:
        violations.append(
            "Multi-module design requires at least one interface contract (seam closure)"
        )

    if strict_v2:
        if not design.shared_invariants:
            violations.append(
                "DesignSpec v2 requires at least one shared_invariants rule "
                "(e.g. error_envelope, id_format, time_encoding)"
            )
        if design.execution_strategy is None:
            violations.append(
                "DesignSpec v2 requires execution_strategy "
                "('contract_first_parallel' or 'serial_handoff')"
            )
        elif len(design.strategy_rationale.strip()) < 10:
            violations.append(
                "DesignSpec v2 requires a non-trivial strategy_rationale (>= 10 characters)"
            )
        if len(design.modules) > 1:
            for iface in design.interfaces:
                if not iface.input_schema:
                    violations.append(
                        f"Interface {iface.id} is missing typed input_schema fields"
                    )
                if not iface.output_schema:
                    violations.append(
                        f"Interface {iface.id} is missing typed output_schema fields"
                    )
                if not iface.error_cases:
                    violations.append(
                        f"Interface {iface.id} is missing explicit error_cases"
                    )
            if (
                design.execution_strategy == ExecutionStrategy.CONTRACT_FIRST_PARALLEL
                and not shared_norm
            ):
                violations.append(
                    "execution_strategy='contract_first_parallel' with multiple modules "
                    "requires non-empty shared_contract_paths for Wave 0 contract lock"
                )

    if spec is not None:
        if design.spec_id is None:
            violations.append("DesignSpec is not bound to a GoalSpec (spec_id is empty)")
        elif design.spec_id != spec.spec_id:
            violations.append(
                f"DesignSpec spec_id ({design.spec_id}) does not match GoalSpec ({spec.spec_id})"
            )
        if design.spec_hash is None:
            violations.append("DesignSpec is not bound to a GoalSpec hash (spec_hash is empty)")
        elif design.spec_hash != spec.content_hash:
            violations.append(
                f"DesignSpec spec_hash ({design.spec_hash}) does not match GoalSpec "
                f"({spec.content_hash})"
            )

        valid_reqs = {req.id for req in spec.requirements}
        covered: set[str] = set()
        for module in design.modules:
            for ref in module.requirement_refs:
                if ref not in valid_reqs:
                    violations.append(
                        f"Design module {module.id} references unknown requirement: {ref}"
                    )
                else:
                    covered.add(ref)
        missing = sorted(valid_reqs - covered)
        if missing:
            violations.append(
                f"Requirements not covered by any design module: {', '.join(missing)}"
            )

    return violations


def validate_plan(
    plan: TaskPlan,
    spec: GoalSpec | None = None,
    *,
    design: DesignSpec | None = None,
    workspace_root: Path | str | None = None,
) -> list[str]:
    """Validate structural DAG properties, spec coverage, and concurrent write_scope isolation."""
    violations: list[str] = []
    host_root_pattern: re.Pattern[str] | None = None
    host_root_display = ""
    if workspace_root is not None:
        host_root_display = str(Path(workspace_root).resolve())
        host_root_pattern = re.compile(
            rf"(?:^|[\s\"'=:;]){re.escape(host_root_display)}(?:$|[\s\"'/:;])"
        )

    by_id: dict[str, TaskNode] = {}
    for task in plan.tasks:
        if task.task_id in by_id:
            violations.append(f"Duplicate task_id: {task.task_id}")
        by_id[task.task_id] = task
        for scope in task.write_scope:
            try:
                normalize_scope_path(scope)
            except ValueError as exc:
                violations.append(f"Task {task.task_id} invalid write_scope ({scope}): {exc}")
        if host_root_pattern is not None:
            for check in task.acceptance_checks:
                if (
                    check.kind == CheckKind.COMMAND
                    and check.command
                    and host_root_pattern.search(check.command)
                ):
                    violations.append(
                        f"Task {task.task_id} check {check.id} hardcodes host workspace path "
                        f"'{host_root_display}'; command checks run inside isolated sandbox cwd "
                        "and must use relative paths."
                    )

    # Dependency existence & self-loops
    in_degree: dict[str, int] = {tid: 0 for tid in by_id}
    children: dict[str, list[str]] = {tid: [] for tid in by_id}
    for task in plan.tasks:
        for dep in task.depends_on:
            if dep == task.task_id:
                violations.append(f"Task {task.task_id} cannot depend on itself")
            elif dep not in by_id:
                violations.append(f"Task {task.task_id} depends on unknown task: {dep}")
            else:
                in_degree[task.task_id] += 1
                children[dep].append(task.task_id)

    # Kahn topological sort for cycle detection
    queue = deque(tid for tid, deg in in_degree.items() if deg == 0)
    visited = 0
    while queue:
        cur = queue.popleft()
        visited += 1
        for nxt in children[cur]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)
    if visited != len(by_id):
        violations.append("TaskPlan contains a dependency cycle")

    # Spec traceability coverage
    if spec is not None:
        valid_reqs = {r.id for r in spec.requirements}
        covered_reqs: set[str] = set()
        for task in plan.tasks:
            for ref in task.requirement_refs:
                if ref not in valid_reqs:
                    violations.append(f"Task {task.task_id} references unknown requirement: {ref}")
                else:
                    covered_reqs.add(ref)
        missing_reqs = sorted(valid_reqs - covered_reqs)
        if missing_reqs:
            violations.append(f"Requirements not covered by any task: {', '.join(missing_reqs)}")

    # Design traceability coverage
    if design is not None:
        if plan.design_id != design.design_id:
            violations.append(
                f"TaskPlan design_id ({plan.design_id}) does not match DesignSpec "
                f"({design.design_id})"
            )
        if plan.design_hash != design.content_hash:
            violations.append(
                f"TaskPlan design_hash ({plan.design_hash}) does not match DesignSpec "
                f"({design.content_hash})"
            )
        known_design_refs = {m.id for m in design.modules} | {i.id for i in design.interfaces}
        referenced: set[str] = set()
        for task in plan.tasks:
            for ref in task.design_refs:
                if ref not in known_design_refs:
                    violations.append(
                        f"Task {task.task_id} references unknown design element: {ref}"
                    )
                else:
                    referenced.add(ref)
        orphan_modules = sorted({m.id for m in design.modules} - referenced)
        if orphan_modules:
            violations.append(
                "Design modules not referenced by any task: " + ", ".join(orphan_modules)
            )
        if design.shared_contract_paths:
            for task in plan.tasks:
                if task.kind == TaskKind.CONTRACT:
                    continue
                try:
                    c_hits = scopes_overlap(task.write_scope, design.shared_contract_paths)
                except ValueError:
                    c_hits = []
                if c_hits:
                    pairs = ", ".join(f"{a}<->{b}" for a, b in c_hits)
                    violations.append(
                        f"Task {task.task_id} ({task.kind.value}) write_scope overlaps "
                        f"shared_contract_paths ({pairs}); only kind='contract' tasks "
                        "may write shared contracts."
                    )

    # Independent sibling tasks must have disjoint write_scopes
    if visited == len(by_id):
        ancestors = _build_ancestors(plan.tasks)
        tasks_list = list(by_id.values())
        for i in range(len(tasks_list)):
            for j in range(i + 1, len(tasks_list)):
                ta, tb = tasks_list[i], tasks_list[j]
                ordered = tb.task_id in ancestors[ta.task_id] or ta.task_id in ancestors[tb.task_id]
                if ordered:
                    continue
                try:
                    conflicts = scopes_overlap(ta.write_scope, tb.write_scope)
                except ValueError:
                    conflicts = []
                if conflicts:
                    pairs = ", ".join(f"{a}<->{b}" for a, b in conflicts)
                    violations.append(
                        f"Concurrent-eligible tasks {ta.task_id} and {tb.task_id} "
                        f"have overlapping write_scope: {pairs}"
                    )

    return violations


def assert_valid_plan(
    plan: TaskPlan,
    spec: GoalSpec | None = None,
    *,
    design: DesignSpec | None = None,
    workspace_root: Path | str | None = None,
) -> None:
    violations = validate_plan(plan, spec=spec, design=design, workspace_root=workspace_root)
    if violations:
        raise PlanValidationError(violations)


def compute_dispatchable_tasks(
    plan: TaskPlan,
    *,
    max_concurrent: int = 3,
) -> list[TaskNode]:
    """Return tasks ready to run now without write_scope or concurrency conflicts."""
    by_id = {t.task_id: t for t in plan.tasks}
    active = [t for t in plan.tasks if t.status in (TaskStatus.RUNNING, TaskStatus.VERIFYING)]
    available_slots = max(0, max_concurrent - len(active))
    if available_slots == 0:
        return []

    active_scopes = [s for t in active for s in t.write_scope]
    dispatchable: list[TaskNode] = []

    for task in plan.tasks:
        if task.status not in (TaskStatus.WAITING, TaskStatus.READY):
            continue
        deps_accepted = all(
            dep in by_id and by_id[dep].status == TaskStatus.ACCEPTED for dep in task.depends_on
        )
        if not deps_accepted:
            continue
        if scopes_overlap(task.write_scope, active_scopes):
            continue
        dispatchable.append(task)
        active_scopes.extend(task.write_scope)
        if len(dispatchable) >= available_slots:
            break

    return dispatchable
