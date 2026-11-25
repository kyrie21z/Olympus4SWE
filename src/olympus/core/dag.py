"""Deterministic DAG validation, write_scope conflict detection, and ready-task scheduling."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Mapping
from itertools import combinations
from pathlib import Path

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
from olympus.core.scope_paths import normalize_scope_path as normalize_scope_path
from olympus.core.scope_paths import normalized_scopes, overlap_messages
from olympus.core.scope_paths import paths_overlap as paths_overlap
from olympus.core.scope_paths import scopes_overlap as scopes_overlap


class PlanValidationError(ValueError):
    """Raised when a TaskPlan violates structural, dependency, coverage, or scope invariants."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


def _dupes(kind: str, ids: Collection[str]) -> list[str]:
    return [f"Duplicate {kind} id: {item}" for item, n in Counter(ids).items() if n > 1]


def _build_ancestors(tasks: list[TaskNode]) -> dict[str, set[str]]:
    """Return the transitive dependency ids of every task; self-dependencies are ignored."""
    known = {t.task_id for t in tasks}
    direct = {t.task_id: {d for d in t.depends_on if d != t.task_id and d in known} for t in tasks}
    ancestors: dict[str, set[str]] = {tid: set() for tid in direct}
    for _ in direct:
        for tid, deps in direct.items():
            ancestors[tid] |= {d for dep in deps for d in {dep} | ancestors.get(dep, set())}
    return ancestors


def _residual_cycles(nodes: Collection[str], deps_of: Mapping[str, Collection[str]]) -> list[str]:
    """Return sorted nodes left after peeling every node whose dependencies are all peeled."""
    remaining = {node: {dep for dep in deps_of[node] if dep in nodes} for node in nodes}
    while free := [n for n, deps in remaining.items() if not deps & remaining.keys()]:
        for node in free:
            del remaining[node]
    return sorted(remaining)


def validate_goal_spec(
    spec: GoalSpec,
    *,
    strict_ears: bool = True,
    require_out_of_scope: bool = True,
) -> list[str]:
    """Zero-token gate for the 5-element + EARS GoalSpec contract and unresolved blockers."""
    issues: list[str] = []
    if strict_ears:
        actors = {actor.strip().lower() for actor in spec.actors if actor.strip()}
        required = [
            (spec.deliverable_kind is not None, "deliverable_kind"),
            (any(s.strip() for s in spec.observable_surfaces), "observable_surfaces"),
            (bool(actors), "actors"),
            (any(s.strip() for s in spec.test_seams), "test_seams (public boundary seams)"),
        ]
        issues += [f"GoalSpec is missing {name}" for ok, name in required if not ok]
        for req in spec.requirements:
            actor = (req.actor or "").strip()
            if not actor:
                issues.append(f"Requirement {req.id} is missing actor")
            elif actor.lower() not in actors:
                issues.append(f"Requirement {req.id} actor is not a declared GoalSpec actor")
            if not req.ears_criteria:
                issues.append(f"Requirement {req.id} has no ears_criteria")
    if require_out_of_scope and not any(item.strip() for item in spec.out_of_scope):
        issues.append("GoalSpec is missing out_of_scope boundaries")
    if blockers := spec.unresolved_blockers():
        issues.append(f"GoalSpec still has unresolved blockers: {', '.join(blockers)}")
    return issues


def validate_design_spec(
    design: DesignSpec,
    spec: GoalSpec | None = None,
    *,
    strict_v2: bool = False,
) -> list[str]:
    """Zero-token gate for invariants, shared contract seams, module cycles, and coverage."""
    issues: list[str] = []
    issues += _dupes("design module", [m.id for m in design.modules])
    issues += _dupes("global invariant", [i.id for i in design.shared_invariants])
    issues += _dupes("interface contract", [i.id for i in design.interfaces])
    known = {m.id for m in design.modules}
    paths = {m.id: normalized_scopes(m.target_paths, m.id, issues) for m in design.modules}
    shared = normalized_scopes(design.shared_contract_paths, "DesignSpec", issues)
    for left, right in combinations(known, 2):
        label = f"Modules {left}/{right} have overlapping target_paths"
        issues += overlap_messages(label, paths[left], paths[right])
    for mod_id in known:
        label = f"shared_contract_paths overlaps module {mod_id}"
        issues += overlap_messages(label, shared, paths[mod_id])
    for inv in design.shared_invariants:
        issues += [
            f"Invariant {inv.id} applies_to unknown '{mod}'"
            for mod in inv.applies_to
            if mod not in known
        ]
    entities = {e.name for e in design.data_models}
    refs = [(f"DataEntity {e.name} owner_module_id", e.owner_module_id) for e in design.data_models]
    deps: dict[str, set[str]] = {mod_id: set() for mod_id in known}
    for iface in design.interfaces:
        provider = iface.provider_module_id
        refs.append((f"Interface {iface.id} provider", provider))
        refs += [(f"Interface {iface.id} consumer", c) for c in iface.consumer_module_ids]
        unknown = [ref for ref in iface.data_entity_refs if ref not in entities]
        issues += [f"Interface {iface.id} unknown DataEntity '{ref}'" for ref in unknown]
        for consumer in iface.consumer_module_ids:
            if consumer == provider:
                issues.append(f"Interface {iface.id} self-dependency cycle on {consumer}")
            elif consumer in known and provider in known:
                deps[consumer].add(provider)
    issues += [f"{name} references unknown module '{r}'" for name, r in refs if r not in known]
    if cyclic := _residual_cycles(known, deps):
        issues.append(f"DesignSpec module dependency cycle: {', '.join(cyclic)}")
    if len(design.modules) > 1 and not design.interfaces:
        issues.append("Multi-module design requires an interface contract (seam closure)")
    if strict_v2:
        required = [
            (bool(design.shared_invariants), "shared_invariants"),
            (design.execution_strategy is not None, "execution_strategy"),
            (len(design.strategy_rationale.strip()) >= 10, "strategy_rationale (>= 10 chars)"),
        ]
        issues += [f"DesignSpec v2 requires {name}" for ok, name in required if not ok]
        for iface in design.interfaces if len(design.modules) > 1 else []:
            fields = ("input_schema", "output_schema", "error_cases")
            absent = [name for name in fields if not getattr(iface, name)]
            if absent:
                issues.append(f"Interface {iface.id} is missing {', '.join(absent)}")
        locked = design.execution_strategy == ExecutionStrategy.CONTRACT_FIRST_PARALLEL
        if locked and len(design.modules) > 1 and not shared:
            issues.append("contract_first_parallel requires shared_contract_paths")
    if spec is not None:
        for value, expected, label in (
            (design.spec_id, spec.spec_id, "spec_id"),
            (design.spec_hash, spec.content_hash, "spec_hash"),
        ):
            if value is None:
                issues.append(f"DesignSpec is not bound to a GoalSpec ({label} is empty)")
            elif value != expected:
                issues.append(f"DesignSpec {label} ({value}) != GoalSpec")
        valid = {req.id for req in spec.requirements}
        owners = {m.id: m.requirement_refs for m in design.modules}
        for name, wanted in owners.items():
            issues += [f"Design module {name} unknown ref: {r}" for r in wanted if r not in valid]
        covered = {ref for refs in owners.values() for ref in refs if ref in valid}
        if missing := sorted(valid - covered):
            issues.append(f"Requirements not covered by any design module: {', '.join(missing)}")
    return issues


VERTICAL_SLICE_TOPOLOGIES = frozenset({"tdd_first", "vertical_slices"})


def _has_executable_command(task: TaskNode) -> bool:
    """Return True when a task carries a command check with an executable non-comment token."""
    for check in task.acceptance_checks:
        lines = [line.strip() for line in (check.command or "").splitlines()]
        if check.kind == CheckKind.COMMAND and any(
            line.strip(" \t;|&()") for line in lines if line and line[0] != "#"
        ):
            return True
    return False


_MONOLITHIC_SLICE_FLOOR = 4


def _is_human_gate_task(task: TaskNode) -> bool:
    return task.role.strip().lower() == "human" and all(
        check.kind == CheckKind.HUMAN for check in task.acceptance_checks
    )


def _validate_vertical_slices(
    plan: TaskPlan,
    ancestors: dict[str, set[str]],
    spec: GoalSpec | None = None,
) -> list[str]:
    issues: list[str] = []
    by_id = {task.task_id: task for task in plan.tasks}
    spec_req_ids = [req.id for req in spec.requirements] if spec is not None else []
    total_reqs = len(spec_req_ids)
    guard_monolith = total_reqs >= _MONOLITHIC_SLICE_FLOOR
    for task in plan.tasks:
        slices = task.kind in (TaskKind.IMPLEMENT, TaskKind.INTEGRATE)
        if slices and not _is_human_gate_task(task) and not _has_executable_command(task):
            issues.append(f"Vertical slice {task.task_id} needs an executable command check")
        if guard_monolith and task.kind == TaskKind.IMPLEMENT and not _is_human_gate_task(task):
            task_refs = set(task.requirement_refs)
            covered = sum(1 for req_id in spec_req_ids if req_id in task_refs)
            if covered >= _MONOLITHIC_SLICE_FLOOR and covered * 4 >= total_reqs * 3:
                issues.append(
                    f"Vertical slice {task.task_id} collapses {covered}/{total_reqs} "
                    "requirements into one implement task; split into thin Slice 1 plus "
                    "parallel vertical feature slices"
                )
        if task.kind != TaskKind.TEST:
            continue
        upstream = [by_id.get(dep) for dep in sorted(ancestors.get(task.task_id, set()))]
        deferred = [
            up.task_id
            for up in upstream
            if up is not None
            and up.kind in (TaskKind.IMPLEMENT, TaskKind.INTEGRATE)
            and not _is_human_gate_task(up)
            and not _has_executable_command(up)
        ]
        if deferred:
            issues.append(f"Task {task.task_id} horizontally defers verification to {deferred}")
    return issues


def validate_plan_topology(
    plan: TaskPlan,
    spec: GoalSpec | None = None,
    *,
    design: DesignSpec | None = None,
    topology: str | None = None,
) -> list[str]:
    """Run only the topology-specific structural gate for ``plan``."""
    del design
    if topology not in VERTICAL_SLICE_TOPOLOGIES:
        return []
    return _validate_vertical_slices(plan, _build_ancestors(plan.tasks), spec=spec)


def validate_plan(
    plan: TaskPlan,
    spec: GoalSpec | None = None,
    *,
    design: DesignSpec | None = None,
    workspace_root: Path | str | None = None,
    topology: str | None = None,
) -> list[str]:
    """Validate structural DAG properties, spec coverage, and concurrent write_scope isolation."""
    issues: list[str] = []
    root = str(Path(workspace_root).resolve()) if workspace_root is not None else None
    host = re.compile(rf"(?:^|[\s\"'=:;]){re.escape(root)}(?:$|[\s\"'/:;])") if root else None
    reqs = spec.requirements if spec is not None else []
    for delivery in plan.delivery_checks:
        key = (delivery.requirement_id, delivery.criterion_index)
        matches = [req for req in reqs if req.id == delivery.requirement_id]
        if len(matches) != 1 or delivery.criterion_index > len(matches[0].acceptance_criteria):
            issues.append(f"Delivery check references an unknown criterion: {key}")
        if host is not None and host.search(delivery.command):
            issues.append(f"Delivery check {key} hardcodes host path '{root}'")
    issues += _dupes("task", [task.task_id for task in plan.tasks])
    issues += _dupes(
        "delivery check", [f"{d.requirement_id}:{d.criterion_index}" for d in plan.delivery_checks]
    )
    by_id = {task.task_id: task for task in plan.tasks}
    scopes = {tid: normalized_scopes(t.write_scope, tid, issues) for tid, t in by_id.items()}
    for task in plan.tasks:
        for check in task.acceptance_checks:
            if check.kind == CheckKind.COMMAND and host and host.search(check.command or ""):
                issues.append(f"Task {task.task_id} check hardcodes host workspace path")
    deps = {tid: {d for d in task.depends_on if d != tid} for tid, task in by_id.items()}
    for task in plan.tasks:
        for dep in task.depends_on:
            if dep == task.task_id:
                issues.append(f"Task {task.task_id} cannot depend on itself")
            elif dep not in by_id:
                issues.append(f"Task {task.task_id} depends on unknown task: {dep}")
    if cyclic := _residual_cycles(by_id, deps):
        issues.append(f"TaskPlan contains a dependency cycle: {', '.join(cyclic)}")
    if spec is not None:
        valid = {req.id for req in spec.requirements}
        owners = {task.task_id: task.requirement_refs for task in plan.tasks}
        for tid, wanted in owners.items():
            issues += [f"Task {tid} unknown requirement: {r}" for r in wanted if r not in valid]
        covered = {ref for refs in owners.values() for ref in refs if ref in valid}
        if missing := sorted(valid - covered):
            issues.append(f"Requirements not covered by any task: {', '.join(missing)}")
    if design is not None:
        if plan.design_id != design.design_id:
            issues.append(f"TaskPlan design_id ({plan.design_id}) != DesignSpec")
        if plan.design_hash != design.content_hash:
            issues.append(f"TaskPlan design_hash ({plan.design_hash}) != DesignSpec")
        known = {m.id for m in design.modules} | {i.id for i in design.interfaces}
        allowed = ", ".join(sorted(known)) or "none"
        refs = [(task.task_id, ref) for task in plan.tasks for ref in task.design_refs]
        for tid, ref in refs:
            if ref not in known:
                note = f"allowed design_refs: {allowed}"
                issues.append(f"Task {tid} unknown design element: {ref} ({note})")
        if orphans := sorted({m.id for m in design.modules} - {ref for _, ref in refs}):
            issues.append(f"Design modules not referenced by any task: {orphans}")
        shared = normalized_scopes(design.shared_contract_paths, "DesignSpec", issues)
        for task in plan.tasks:
            hits = scopes_overlap(scopes[task.task_id], shared)
            if hits and task.kind != TaskKind.CONTRACT:
                pairs = ", ".join(f"{a}<->{b}" for a, b in hits)
                note = "only kind='contract' tasks may write shared contracts"
                issues.append(f"Task {task.task_id} overlaps shared paths ({pairs}); {note}")
    ancestors = _build_ancestors(plan.tasks)
    if not cyclic:
        for task_a, task_b in combinations(by_id.values(), 2):
            ordered = task_b.task_id in ancestors[task_a.task_id]
            ordered = ordered or task_a.task_id in ancestors[task_b.task_id]
            hits = [] if ordered else scopes_overlap(scopes[task_a.task_id], scopes[task_b.task_id])
            if hits:
                pairs = ", ".join(f"{a}<->{b}" for a, b in hits)
                label = f"{task_a.task_id}/{task_b.task_id}"
                issues.append(f"Concurrent-eligible {label} have overlapping write_scope: {pairs}")
    if topology in VERTICAL_SLICE_TOPOLOGIES:
        issues.extend(_validate_vertical_slices(plan, ancestors, spec=spec))
    return issues


def assert_valid_plan(
    plan: TaskPlan,
    spec: GoalSpec | None = None,
    *,
    design: DesignSpec | None = None,
    workspace_root: Path | str | None = None,
    topology: str | None = None,
) -> None:
    """Raise PlanValidationError when the plan breaks a structural invariant."""
    issues = validate_plan(
        plan, spec, design=design, workspace_root=workspace_root, topology=topology
    )
    if issues:
        raise PlanValidationError(issues)


def compute_dispatchable_tasks(
    plan: TaskPlan,
    *,
    max_concurrent: int = 3,
) -> list[TaskNode]:
    """Return tasks ready to run now without write_scope or concurrency conflicts."""
    by_id = {t.task_id: t for t in plan.tasks}
    active = [t for t in plan.tasks if t.status in (TaskStatus.RUNNING, TaskStatus.VERIFYING)]
    slots = max(0, max_concurrent - len(active))
    active_scopes = [s for t in active for s in t.write_scope]
    ready: list[TaskNode] = []
    for task in plan.tasks:
        if len(ready) >= slots:
            break
        if task.status not in (TaskStatus.WAITING, TaskStatus.READY):
            continue
        deps = [by_id.get(dep) for dep in task.depends_on]
        blocked = any(dep is None or dep.status != TaskStatus.ACCEPTED for dep in deps)
        if blocked or scopes_overlap(task.write_scope, active_scopes):
            continue
        ready.append(task)
        active_scopes.extend(task.write_scope)
    return ready
