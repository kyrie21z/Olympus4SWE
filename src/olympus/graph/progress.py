"""Single source of truth for workflow gate, resume, and completion state.

The loader, the run supervisor, and the read-only Ariadne projection all derive
gate decisions from this module so a paused gate, an approved artifact, and a
resumed entry point can never disagree.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from olympus.core.contracts import (
    ApprovalTarget,
    RunEvent,
    RunStatus,
    TaskNode,
    TaskPlan,
    TaskStatus,
)
from olympus.core.control import (
    has_unfulfilled_revision,
    is_design_bound_to_spec,
    is_plan_bound_to_upstream,
)
from olympus.core.dag import validate_plan_topology
from olympus.core.store import RunStore

if TYPE_CHECKING:
    from olympus.graph.loader import WorkflowDefinition


class GateSpec(TypedDict):
    """Canonical metadata for one human approval gate."""

    node: str
    snapshot_key: str
    target: ApprovalTarget
    id_attr: str
    label: str


GATE_SPECS: dict[str, GateSpec] = {
    "spec": {
        "node": "gate_spec",
        "snapshot_key": "spec",
        "target": ApprovalTarget.GOAL_SPEC,
        "id_attr": "spec_id",
        "label": "GoalSpec",
    },
    "design": {
        "node": "gate_design",
        "snapshot_key": "design",
        "target": ApprovalTarget.DESIGN_SPEC,
        "id_attr": "design_id",
        "label": "DesignSpec",
    },
    "plan": {
        "node": "gate_plan",
        "snapshot_key": "plan",
        "target": ApprovalTarget.TASK_PLAN,
        "id_attr": "plan_id",
        "label": "TaskPlan",
    },
}


def is_artifact_approved(
    store: RunStore,
    run_id: str,
    gate: str,
    snap: Mapping[str, Any],
) -> bool:
    """Return whether the run's current artifact for ``gate`` carries a valid approval."""
    spec = GATE_SPECS[gate]
    artifact = snap.get(spec["snapshot_key"])
    if artifact is None:
        return False
    return store.is_approved(
        run_id=run_id,
        target=spec["target"],
        target_id=getattr(artifact, spec["id_attr"]),
        target_hash=artifact.content_hash,
    )


def resolve_resume_node(
    definition: WorkflowDefinition,
    snap: Mapping[str, Any],
    *,
    store: RunStore,
) -> str | None:
    """Pick the graph entry point implied by the run's confirmed artifacts."""
    names = {node.name for node in definition.nodes}
    run_id = str(snap["run_id"])

    align_node = next(
        (node.name for node in definition.nodes if node.type == "align_goal"), None
    )
    if align_node is not None and has_unfulfilled_revision(store.list_events(run_id)):
        return align_node

    spec = snap.get("spec")
    if spec is not None and not is_artifact_approved(store, run_id, "spec", snap):
        if "gate_spec" in names:
            return "gate_spec"
        if "align" in names:
            return "align"

    if (
        snap["plan"] is not None
        and is_artifact_approved(store, run_id, "plan", snap)
        and is_plan_bound_to_upstream(store, run_id, snap)
    ):
        plan_design = snap.get("design") if snap["plan"].design_id is not None else None
        if "validate_plan" in names and validate_plan_topology(
            snap["plan"],
            snap.get("spec"),
            design=plan_design,
            topology=definition.topology,
        ):
            return "validate_plan"
        if definition.stop_after == "execute" and "execute" in names:
            return "execute"
        # Re-enter the terminal gate so the graph records its final routed/END events.
        if "gate_plan" in names:
            return "gate_plan"
        return None

    if snap["design"] is not None:
        if is_design_bound_to_spec(store, run_id, snap):
            return "plan" if "plan" in names else None
        spec = snap.get("spec")
        design = snap["design"]
        design_stale_for_spec = (
            spec is not None
            and is_artifact_approved(store, run_id, "spec", snap)
            and design.spec_hash is not None
            and design.spec_hash != spec.content_hash
        )
        if design_stale_for_spec:
            for candidate in ("design", "plan"):
                if candidate in names:
                    return candidate
            return None
        for candidate in ("gate_design", "validate_design", "design"):
            if candidate in names:
                return candidate
        return None

    if snap["spec"] is not None:
        if is_artifact_approved(store, run_id, "spec", snap):
            for candidate in ("design", "plan"):
                if candidate in names:
                    return candidate
            return None
        if "gate_spec" in names:
            return "gate_spec"
        return "align" if "align" in names else None

    return "intake" if "intake" in names else ("align" if "align" in names else None)


_VISITED_STATUSES = frozenset({"running", "completed", "paused", "error", "failed"})


def _declared_edges(description: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """Return the workflow graph's declared ``(source, target)`` pairs in order."""
    if not description:
        return []
    declared: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for edge in description.get("edges") or []:
        if not isinstance(edge, Mapping):
            continue
        source, target = edge.get("source"), edge.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        if (source, target) in seen:
            continue
        seen.add((source, target))
        declared.append((source, target))
    return declared


def _node_names(value: Any) -> list[str]:
    """Normalize a ``graph.routed`` ``from``/``to`` payload into node names."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [item for item in value if isinstance(item, str)]
    return []


def _derive_traversed_edges(
    events: Sequence[RunEvent],
    declared: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    """Collect the executed trajectory against the declared workflow edges.

    Live runs persist one ``graph.routed`` event per wave, so the ``from`` x
    ``to`` product is the exact trajectory. Legacy and seeded runs never wrote
    them; for those, consecutive ``graph.node_finished`` -> ``graph.node_started``
    transitions are matched against the declared edges instead.
    """
    allowed = set(declared)
    traversed: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def record(source: str, target: str) -> None:
        if (source, target) not in allowed or (source, target) in seen:
            return
        seen.add((source, target))
        traversed.append({"source": source, "target": target})

    if any(event.event_type == "graph.routed" for event in events):
        paused_node: str | None = None
        for event in events:
            if event.event_type == "graph.routed":
                paused_node = None
                for source in _node_names(event.payload.get("from")):
                    for target in _node_names(event.payload.get("to")):
                        record(source, target)
            elif event.event_type == "graph.paused":
                node = event.payload.get("node")
                if isinstance(node, str):
                    paused_node = node
            elif event.event_type == "graph.started" and paused_node is not None:
                start_from = event.payload.get("start_from")
                if isinstance(start_from, str):
                    record(paused_node, start_from)
                    paused_node = None
            elif event.event_type == "graph.node_started" and paused_node is not None:
                node = event.payload.get("node")
                if isinstance(node, str) and node != paused_node:
                    record(paused_node, node)
                    paused_node = None
        return traversed

    last_finished: str | None = None
    for event in events:
        node = event.payload.get("node")
        if not isinstance(node, str):
            continue
        if event.event_type in ("graph.node_finished", "graph.paused"):
            last_finished = node
        elif event.event_type == "graph.node_started" and last_finished is not None:
            record(last_finished, node)
    return traversed


def _forward_reachable(start: str, adjacency: Mapping[str, Sequence[str]]) -> set[str]:
    """Return every node reachable downstream of ``start`` via forward declared edges."""
    reached: set[str] = set()
    stack = [start]
    while stack:
        for target in adjacency.get(stack.pop(), ()):
            if target == start or target in reached:
                continue
            reached.add(target)
            stack.append(target)
    return reached


def _acyclic_forward_adjacency(
    nodes: Sequence[str],
    declared: Sequence[tuple[str, str]],
    starts: Sequence[str],
) -> dict[str, list[str]]:
    """Build forward adjacency stripping only DFS back-edges from workflow entry roots.

    On acyclic graphs every declared edge survives regardless of YAML node declaration order;
    on cyclic graphs retry back-edges are removed so unreached downstream nodes cannot loop
    back to visited ancestors.
    """
    known = set(nodes)
    raw: dict[str, list[str]] = {name: [] for name in nodes}
    indegree: dict[str, int] = {name: 0 for name in nodes}
    for source, target in declared:
        if source in known and target in known and source != target and target not in raw[source]:
            raw[source].append(target)
            indegree[target] += 1

    roots = [name for name in starts if name in known]
    if not roots:
        roots = [name for name in nodes if indegree[name] == 0]
    order_seeds = list(dict.fromkeys([*roots, *nodes]))

    adjacency: dict[str, list[str]] = {name: [] for name in nodes}
    state: dict[str, int] = {}  # 1 = on DFS stack, 2 = finished
    for seed in order_seeds:
        if state.get(seed):
            continue
        stack: list[tuple[str, int]] = [(seed, 0)]
        state[seed] = 1
        while stack:
            current, idx = stack[-1]
            neighbors = raw[current]
            if idx >= len(neighbors):
                state[current] = 2
                stack.pop()
                continue
            stack[-1] = (current, idx + 1)
            nxt = neighbors[idx]
            nxt_state = state.get(nxt, 0)
            if nxt_state == 1:
                continue
            adjacency[current].append(nxt)
            if nxt_state == 0:
                state[nxt] = 1
                stack.append((nxt, 0))
    return adjacency


def _derive_skipped(
    statuses: Mapping[str, str],
    declared: Sequence[tuple[str, str]],
    traversed: Sequence[Mapping[str, str]],
    starts: Sequence[str] = (),
) -> set[str]:
    """Return never-visited nodes the executed trajectory passed straight over."""
    known = set(statuses)
    adjacency = _acyclic_forward_adjacency(list(statuses), declared, starts)

    visited = {name for name, status in statuses.items() if status in _VISITED_STATUSES}
    if not visited:
        return set()

    reach = {name: _forward_reachable(name, adjacency) for name in known}
    downstream_of_visited: set[str] = set()
    for name in visited:
        downstream_of_visited |= reach[name]
    skip_edges = [
        (str(edge["source"]), str(edge["target"]))
        for edge in traversed
        if edge["source"] in known and edge["target"] in known
    ]

    skipped: set[str] = set()
    for name, status in statuses.items():
        if status != "pending":
            continue
        between_visited = name in downstream_of_visited and bool(reach[name] & visited)
        bypassed = any(
            name in reach[source] and target in reach[name] for source, target in skip_edges
        )
        if between_visited or bypassed:
            skipped.add(name)
    return skipped


_GRAPH_LIFECYCLE_EVENTS = frozenset(
    {
        "graph.node_started",
        "graph.node_finished",
        "graph.paused",
        "graph.node_error",
        "graph.failed",
        "graph.finished",
    }
)
_SESSION_LIFECYCLE_EVENTS = frozenset(
    {
        "session.started",
        "session.continued",
        "session.paused",
        "session.completed",
        "session.failed",
    }
)
_SPEC_DOMAIN_EVENTS = frozenset(
    {"spec.saved", "spec.aligned", "spec.resynthesized", "spec.clarified"}
)
_CRITIC_DOMAIN_EVENTS = frozenset(
    {"spec.critic_recorded", "critic.completed", "critic.recorded"}
)
_EXECUTE_DOMAIN_EVENTS = frozenset(
    {
        "task.leased",
        "task.verifying",
        "task.accepted",
        "task.accepted_with_risk",
        "task.blocked",
        "task.requeued",
        "delivery.checks_completed",
        "product.verified",
        "review.verdict_recorded",
        "review.verdict_verified",
        "post_dag.repair_turn_completed",
        "post_dag.repair_applied",
        "post_dag.repair_rejected",
        "post_dag.repro_rechecked",
        "session.completed",
    }
)


def _reconcile_domain_workflow_state(
    events: Sequence[RunEvent],
    statuses: dict[str, str],
    run_status: RunStatus,
    *,
    spec_approved: bool,
    design_approved: bool,
    plan_approved: bool,
) -> tuple[str | None, dict[str, Any] | None, str | None, str | None, bool]:
    """Derive per-node workflow statuses from domain events when no graph.* events exist.

    Headless ``SessionRunner`` runs persist stage artifacts, approvals, and ``session.*``
    events without emitting ``graph.node_*`` events. Reconciling those deterministic
    milestones keeps the workflow view aligned with ``stage_pipeline``.
    """
    has_intake = False
    has_spec = spec_approved
    has_critic = False
    has_design = design_approved
    has_plan = plan_approved
    has_execute = False
    has_session_completed = False
    last_pause: RunEvent | None = None
    last_fail: RunEvent | None = None

    for event in events:
        kind = event.event_type
        if kind == "intake.triaged":
            has_intake = True
        elif kind in _SPEC_DOMAIN_EVENTS:
            has_spec = True
        elif kind in _CRITIC_DOMAIN_EVENTS:
            has_critic = True
        elif kind == "design.saved":
            has_design = True
        elif kind == "plan.saved":
            has_plan = True
        elif kind in _EXECUTE_DOMAIN_EVENTS:
            has_execute = True
            if kind == "session.completed":
                has_session_completed = True
        elif kind == "run.revision_requested":
            # A revision returns the pipeline to Stage 1; downstream flags are re-established
            # only by subsequent events (or still-valid approvals passed in by the caller).
            has_spec = spec_approved
            has_critic = False
            has_design = design_approved
            has_plan = plan_approved
            has_execute = False
            has_session_completed = False
            last_pause = None
            last_fail = None
        elif kind == "session.paused":
            last_pause = event
        elif kind == "session.continued":
            last_pause = None
            last_fail = None
        elif kind == "session.failed":
            last_fail = event

    def mark(node: str, status: str) -> None:
        if node in statuses:
            statuses[node] = status

    if has_intake or has_spec or has_design or has_plan or has_execute:
        mark("intake", "completed")
    if has_spec:
        mark("align", "completed")
    if has_critic:
        mark("critic", "completed")

    if spec_approved:
        mark("gate_spec", "completed")
    elif (
        has_spec
        and run_status in (RunStatus.AWAITING_HUMAN, RunStatus.PAUSED)
        and not (has_design or has_plan or has_execute)
    ):
        mark("gate_spec", "paused")

    if has_design:
        mark("design", "completed")
        mark("validate_design", "completed")
    if design_approved:
        mark("gate_design", "completed")
    elif (
        has_design
        and run_status in (RunStatus.AWAITING_HUMAN, RunStatus.PAUSED)
        and not (has_plan or has_execute)
    ):
        mark("gate_design", "paused")

    if has_plan:
        mark("plan", "completed")
        mark("validate_plan", "completed")
    if plan_approved:
        mark("gate_plan", "completed")
    elif (
        has_plan
        and run_status in (RunStatus.AWAITING_HUMAN, RunStatus.PAUSED)
        and not has_execute
    ):
        mark("gate_plan", "paused")

    current_node: str | None = None
    paused: dict[str, Any] | None = None
    failed_node: str | None = None
    last_error: str | None = None
    finished = has_session_completed or run_status == RunStatus.COMPLETED

    if "execute" in statuses:
        if run_status == RunStatus.COMPLETED or has_session_completed:
            if has_execute or plan_approved or run_status == RunStatus.COMPLETED:
                mark("execute", "completed")
        elif has_execute or (plan_approved and run_status == RunStatus.RUNNING):
            if run_status == RunStatus.FAILED:
                mark("execute", "failed")
                failed_node = "execute"
            elif run_status in (RunStatus.AWAITING_HUMAN, RunStatus.PAUSED):
                mark("execute", "paused")
            elif run_status == RunStatus.RUNNING:
                mark("execute", "running")
                current_node = "execute"
            else:
                mark("execute", "paused")

    if run_status in (RunStatus.AWAITING_HUMAN, RunStatus.PAUSED):
        pause_node: str | None = None
        pause_reason: Any = None
        if last_pause is not None:
            raw_gate = last_pause.payload.get("gate")
            raw_node = last_pause.payload.get("node")
            pause_reason = last_pause.payload.get("reason") or last_pause.payload.get("pause_kind")
            if isinstance(raw_gate, str) and raw_gate in GATE_SPECS:
                candidate = GATE_SPECS[raw_gate]["node"]
                if candidate in statuses:
                    pause_node = candidate
            elif isinstance(raw_node, str) and raw_node in statuses:
                pause_node = raw_node
        if pause_node is None:
            for candidate in ("execute", "gate_plan", "gate_design", "gate_spec"):
                if statuses.get(candidate) == "paused":
                    pause_node = candidate
                    break
        if pause_node is not None:
            statuses[pause_node] = "paused"
            paused = {"node": pause_node, "reason": pause_reason}

    if run_status == RunStatus.FAILED:
        if last_fail is not None:
            err = last_fail.payload.get("error") or last_fail.payload.get("reason")
            if err:
                last_error = str(err)
        if failed_node is None:
            if plan_approved and "execute" in statuses:
                failed_node = "execute"
            elif design_approved and "plan" in statuses:
                failed_node = "plan"
            elif spec_approved and "design" in statuses:
                failed_node = "design"
            elif (has_intake or has_spec) and "align" in statuses:
                failed_node = "align"
            elif "intake" in statuses:
                failed_node = "intake"
            if failed_node is not None:
                statuses[failed_node] = "failed"

    return current_node, paused, failed_node, last_error, finished


def build_workflow_view(
    workflow_name: str,
    events: Sequence[RunEvent],
    run_status: RunStatus,
    *,
    spec_approved: bool = False,
    design_approved: bool = False,
    plan_approved: bool = False,
) -> dict[str, Any]:
    """Describe the run's workflow topology plus live per-node statuses from graph.* events."""
    from olympus.graph.loader import build_graph, load_workflow

    topology: str | None = None
    stop_after: str | None = None
    description: dict[str, Any] | None = None
    node_names: list[str] = []
    try:
        definition = load_workflow(workflow_name)
        description = build_graph(definition).describe()
        topology = definition.topology
        stop_after = definition.stop_after
        node_names = [str(n["name"]) for n in description.get("nodes", [])]
    except Exception:  # noqa: BLE001 - a legacy/unknown workflow must not break the console
        description = None

    statuses: dict[str, str] = {name: "pending" for name in node_names}
    current_node: str | None = None
    paused: dict[str, Any] | None = None
    failed_node: str | None = None
    last_error: str | None = None
    finished = False
    has_graph_events = any(ev.event_type in _GRAPH_LIFECYCLE_EVENTS for ev in events)
    has_session_events = any(ev.event_type in _SESSION_LIFECYCLE_EVENTS for ev in events)
    if has_graph_events:
        for event in events:
            payload = event.payload
            node = payload.get("node")
            if event.event_type == "graph.started":
                finished = False
            elif event.event_type == "graph.node_started" and isinstance(node, str):
                finished = False
                paused = None
                failed_node = None
                last_error = None
                current_node = node
                if node in statuses:
                    statuses[node] = "running"
            elif event.event_type == "graph.node_finished" and isinstance(node, str):
                if node in statuses:
                    statuses[node] = "completed"
            elif event.event_type == "graph.paused":
                paused = {
                    "node": node,
                    "reason": payload.get("reason"),
                }
                current_node = None
                if isinstance(node, str) and node in statuses:
                    statuses[node] = "paused"
            elif event.event_type == "graph.finished":
                current_node = None
                paused = None
                finished = True
            elif (
                event.event_type == "graph.node_error"
                and isinstance(node, str)
                and node in statuses
            ):
                statuses[node] = "error"
                failed_node = node
                if payload.get("error"):
                    last_error = str(payload.get("error"))
            elif event.event_type == "graph.failed":
                last_error = str(payload.get("error") or "工作流执行异常中断")
                failed_node = current_node or (node if isinstance(node, str) else None)
                if failed_node and failed_node in statuses:
                    statuses[failed_node] = "failed"
                current_node = None
    elif has_session_events and statuses:
        current_node, paused, failed_node, last_error, finished = (
            _reconcile_domain_workflow_state(
                events,
                statuses,
                run_status,
                spec_approved=spec_approved,
                design_approved=design_approved,
                plan_approved=plan_approved,
            )
        )

    # Reconcile gates that were approved before resuming directly downstream.
    approved_by_gate = {
        "spec": spec_approved,
        "design": design_approved,
        "plan": plan_approved,
    }
    for gate, gate_spec in GATE_SPECS.items():
        if approved_by_gate[gate] and statuses.get(gate_spec["node"]) == "paused":
            statuses[gate_spec["node"]] = "completed"

    if run_status == RunStatus.COMPLETED:
        paused = None
        for gate_spec in GATE_SPECS.values():
            if statuses.get(gate_spec["node"]) == "paused":
                statuses[gate_spec["node"]] = "completed"

    if run_status == RunStatus.FAILED:
        for name, status in list(statuses.items()):
            if status == "running":
                statuses[name] = "failed"
                failed_node = failed_node or name
        if last_error is None:
            last_error = "工作流执行异常中断"

    declared_edges = _declared_edges(description)
    start_nodes = [
        name for name in ((description or {}).get("start") or ()) if isinstance(name, str)
    ]
    traversed_edges = _derive_traversed_edges(events, declared_edges)
    if not has_graph_events and has_session_events and not traversed_edges:
        traversed_edges = [
            {"source": source, "target": target}
            for source, target in declared_edges
            if statuses.get(source) == "completed"
            and statuses.get(target) in _VISITED_STATUSES
        ]
    for name in _derive_skipped(statuses, declared_edges, traversed_edges, start_nodes):
        statuses[name] = "skipped"

    return {
        "name": workflow_name,
        "topology": topology,
        "stop_after": stop_after,
        "graph": description,
        "nodes": [{"name": name, "status": status} for name, status in statuses.items()],
        "node_statuses": dict(statuses),
        "traversed_edges": traversed_edges,
        "current_node": current_node,
        "paused": paused,
        "failed_node": failed_node,
        "last_error": last_error,
        "finished": finished or run_status == RunStatus.COMPLETED,
    }


def is_blueprint_completed(
    *,
    plan: TaskPlan | None,
    plan_approved: bool,
    run_status: RunStatus,
    stop_after: str | None,
) -> bool:
    """Return whether a blueprint-only workflow froze a plan without executing code."""
    tasks = plan.tasks if plan is not None else []
    all_terminal = bool(tasks) and all(
        task.status in (TaskStatus.ACCEPTED, TaskStatus.CANCELLED) for task in tasks
    )
    return bool(
        plan is not None
        and plan_approved
        and run_status == RunStatus.COMPLETED
        and (stop_after or "execute") != "execute"
        and not all_terminal
    )


BallHolder = Literal["human", "agent", "done"]
PantheonGod = Literal["prometheus", "athena", "hephaestus", "momus", "argus"]
PantheonStage = Literal["align", "design", "execute", "review", "verify"]
PendingAction = Literal[
    "clarify",
    "approve_spec",
    "approve_design",
    "approve_plan",
    "resolve_blocker",
    "review_delivery",
]
SeatState = Literal["idle", "working", "waiting_on_human"]

PANTHEON_STAGE_BINDING: tuple[tuple[PantheonGod, PantheonStage, str], ...] = (
    ("prometheus", "align", "Align"),
    ("athena", "design", "Design/Plan"),
    ("hephaestus", "execute", "Build"),
    ("momus", "review", "Review"),
    ("argus", "verify", "Verify"),
)

PANTHEON_SEATS: tuple[tuple[PantheonGod, str], ...] = tuple(
    (god, label) for god, _stage, label in PANTHEON_STAGE_BINDING
)

_NODE_GODS: dict[str, PantheonGod] = {
    "intake": "prometheus",
    "align": "prometheus",
    "critic": "prometheus",
    "gate_spec": "prometheus",
    "design": "athena",
    "validate_design": "athena",
    "gate_design": "athena",
    "plan": "athena",
    "validate_plan": "athena",
    "gate_plan": "athena",
    "execute": "hephaestus",
    "verify": "argus",
}

_AGENT_BADGES: dict[PantheonGod, str] = {
    "prometheus": "aligning",
    "athena": "planning",
    "hephaestus": "running",
    "momus": "reviewing",
    "argus": "verifying",
}


@dataclass(frozen=True)
class BoardClassification:
    ball_holder: BallHolder
    active_god: PantheonGod
    pending_action: PendingAction | None
    worktree_branch: str | None
    status_badge: str
    is_archived: bool


def _approval_flags(
    snap: Mapping[str, Any],
    store: RunStore | None,
) -> tuple[bool, bool, bool]:
    """Read hash-bound gate approvals from the snapshot, falling back to the store."""
    recorded = snap.get("artifact_approved")
    if isinstance(recorded, Mapping):
        return (
            bool(recorded.get(ApprovalTarget.GOAL_SPEC)),
            bool(recorded.get(ApprovalTarget.DESIGN_SPEC)),
            bool(recorded.get(ApprovalTarget.TASK_PLAN)),
        )
    run_id = snap.get("run_id")
    if store is not None and isinstance(run_id, str) and run_id:
        return (
            is_artifact_approved(store, run_id, "spec", snap),
            is_artifact_approved(store, run_id, "design", snap),
            is_artifact_approved(store, run_id, "plan", snap),
        )
    return False, False, False


def _trailing_node(
    events: Sequence[RunEvent], workflow_view: Mapping[str, Any]
) -> str | None:
    paused = workflow_view.get("paused")
    paused_node = paused.get("node") if isinstance(paused, Mapping) else None
    candidates = (
        workflow_view.get("failed_node"),
        paused_node,
        workflow_view.get("current_node"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    for event in reversed(events):
        if event.event_type == "graph.routed":
            targets = _node_names(event.payload.get("to"))
            if targets:
                return targets[-1]
            continue
        node = event.payload.get("node")
        if isinstance(node, str) and node and event.event_type in (
            "graph.node_started",
            "graph.node_finished",
            "graph.paused",
            "graph.node_error",
            "graph.failed",
        ):
            return node
    return None


def _reviewed_attempts(events: Sequence[RunEvent]) -> set[tuple[str, int]]:
    reviewed: set[tuple[str, int]] = set()
    for event in events:
        if event.event_type != "review.verdict_recorded":
            continue
        task_id = event.payload.get("task_id")
        attempt = event.payload.get("attempt")
        if isinstance(task_id, str) and isinstance(attempt, int):
            reviewed.add((task_id, attempt))
    return reviewed


def _execution_god(tasks: Sequence[TaskNode], events: Sequence[RunEvent]) -> PantheonGod:
    """Attribute the execute phase to the worker, reviewer, or verifier."""
    if any(task.status == TaskStatus.RUNNING for task in tasks):
        return "hephaestus"
    verifying = [task for task in tasks if task.status == TaskStatus.VERIFYING]
    if verifying:
        reviewed = _reviewed_attempts(events)
        if any((task.task_id, task.attempt) in reviewed for task in verifying):
            return "momus"
        return "argus"
    blocked = [
        task for task in tasks
        if task.status in (TaskStatus.BLOCKED, TaskStatus.FAILED)
    ]
    if blocked:
        return "momus" if any(task.review_cycle > 0 for task in blocked) else "argus"
    if tasks and all(
        task.status in (TaskStatus.ACCEPTED, TaskStatus.CANCELLED) for task in tasks
    ):
        return "argus"
    return "hephaestus"


def _stage_god(
    tasks: Sequence[TaskNode],
    events: Sequence[RunEvent],
    workflow_view: Mapping[str, Any],
) -> PantheonGod:
    node = _trailing_node(events, workflow_view)
    if node == "execute":
        return _execution_god(tasks, events)
    if node is not None:
        return _NODE_GODS.get(node, "argus")
    if tasks:
        return _execution_god(tasks, events)
    return "prometheus"


def _phase_god(
    tasks: Sequence[TaskNode],
    events: Sequence[RunEvent],
    workflow_view: Mapping[str, Any],
) -> PantheonGod:
    node = _trailing_node(events, workflow_view)
    if node == "execute":
        return _execution_god(tasks, events)
    if node is not None:
        god = _NODE_GODS.get(node)
        if god is not None:
            return god
    if tasks:
        return _execution_god(tasks, events)
    return "prometheus"


def _worktree_branch(tasks: Sequence[TaskNode]) -> str | None:
    """Return the checked-out Git branch of the first active sandbox worktree, if attached."""
    for task in tasks:
        worktree = getattr(task, "worktree_path", None)
        if not isinstance(worktree, str) or not worktree.strip():
            continue
        path = Path(worktree)
        if not path.is_dir():
            continue
        try:
            proc = subprocess.run(
                ["git", "-C", str(path), "symbolic-ref", "-q", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        branch = proc.stdout.strip()
        if proc.returncode == 0 and branch:
            return branch
    return None


def classify_run_board(
    snap: Mapping[str, Any],
    events: Sequence[RunEvent],
    *,
    store: RunStore | None = None,
    workflow_view: Mapping[str, Any] | None = None,
) -> BoardClassification:
    """Classify one Oracle into its Ball-in-Court and Pantheon state.

    This is the single source of truth behind ``GET /api/runs``,
    ``GET /api/board/stream``, the Pantheon Dock aggregates, and the L2
    cockpit handoff semantics, so those surfaces can never disagree.
    """
    raw_status = snap.get("status")
    status = raw_status if isinstance(raw_status, RunStatus) else RunStatus(str(raw_status))
    plan: TaskPlan | None = snap.get("plan")
    spec = snap.get("spec")
    design = snap.get("design")
    tasks: list[TaskNode] = list(plan.tasks) if plan is not None else []

    spec_approved, design_approved, plan_approved = _approval_flags(snap, store)
    view = workflow_view
    if view is None:
        view = build_workflow_view(
            str(snap.get("workflow_name") or "product_blueprint"),
            events,
            status,
            spec_approved=spec_approved,
            design_approved=design_approved,
            plan_approved=plan_approved,
        )
    stop_after = view.get("stop_after")
    worktree = _worktree_branch(tasks)

    if status == RunStatus.CANCELLED:
        return BoardClassification(
            "done",
            _stage_god(tasks, events, view),
            None,
            worktree,
            "cancelled",
            True,
        )

    if status == RunStatus.COMPLETED:
        blueprint = (stop_after or "execute") != "execute" or is_blueprint_completed(
            plan=plan,
            plan_approved=plan_approved,
            run_status=status,
            stop_after=stop_after,
        )
        return BoardClassification(
            "done",
            "athena" if blueprint else "argus",
            None,
            worktree,
            "blueprint" if blueprint else "delivered",
            False,
        )

    if status in (RunStatus.PAUSED, RunStatus.FAILED):
        return BoardClassification(
            "human",
            _stage_god(tasks, events, view),
            "resolve_blocker",
            worktree,
            "paused" if status == RunStatus.PAUSED else "failed",
            False,
        )

    if status == RunStatus.AWAITING_HUMAN:
        if spec is not None and not spec_approved:
            if spec.unresolved_blockers():
                return BoardClassification(
                    "human", "prometheus", "clarify", worktree, "clarify", False
                )
            return BoardClassification(
                "human", "prometheus", "approve_spec", worktree, "approve_spec", False
            )
        if design is not None and not design_approved:
            return BoardClassification(
                "human", "athena", "approve_design", worktree, "approve_design", False
            )
        if plan is not None and not plan_approved:
            return BoardClassification(
                "human", "athena", "approve_plan", worktree, "approve_plan", False
            )
        if tasks and all(
            task.status in (TaskStatus.ACCEPTED, TaskStatus.CANCELLED) for task in tasks
        ):
            return BoardClassification(
                "human", "argus", "review_delivery", worktree, "review_delivery", False
            )
        return BoardClassification(
            "human",
            _stage_god(tasks, events, view),
            "resolve_blocker",
            worktree,
            "blocked",
            False,
        )

    god = _phase_god(tasks, events, view)
    if status == RunStatus.CREATED:
        badge = "created"
    elif god == "athena" and status == RunStatus.DESIGNING:
        badge = "designing"
    else:
        badge = _AGENT_BADGES[god]
    return BoardClassification("agent", god, None, worktree, badge, False)


__all__ = [
    "GATE_SPECS",
    "PANTHEON_SEATS",
    "BallHolder",
    "BoardClassification",
    "GateSpec",
    "PantheonGod",
    "PendingAction",
    "SeatState",
    "build_workflow_view",
    "classify_run_board",
    "is_artifact_approved",
    "is_blueprint_completed",
    "resolve_resume_node",
]