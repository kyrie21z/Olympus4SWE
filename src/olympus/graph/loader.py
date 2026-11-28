"""Declarative YAML workflow loading bound to Olympus stage node handlers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from olympus.core.contracts import (
    DesignSpec,
    GoalSpec,
    InputMaturity,
    RunStatus,
    SpecStatus,
    StrictModel,
)
from olympus.core.control import RunControl
from olympus.core.dag import validate_design_spec, validate_plan
from olympus.core.store import RunStore
from olympus.graph.engine import END, Graph, GraphPaused, Node
from olympus.graph.progress import GATE_SPECS, is_artifact_approved
from olympus.harness.activity import LiveActivity
from olympus.harness.driver import ActiveTurnHandle, DriverRegistry, ExecutionDriversConfig
from olympus.providers.base import LLMProvider
from olympus.stages.align import AlignStage
from olympus.stages.craft import CraftPause, CraftStage
from olympus.stages.design import DesignStage, triage_input
from olympus.stages.orchestrate import OrchestrateStage

WORKFLOW_DIR = Path(__file__).resolve().parent / "workflows"

# A gate decider returns "auto_approve", "await_human", or "block" for one artifact gate.
GateDecider = Callable[[str, Any, "WorkflowRuntime", str], str]

_NODE_TYPES: tuple[str, ...] = (
    "intake_triage",
    "align_goal",
    "critic_parallel",
    "design_spec",
    "validate_design",
    "pm_decompose",
    "validate_plan",
    "execute_task_dag",
    "human_gate",
    "explore_reproduce",
    "architect_sketch",
    "build_direct",
    "hygiene_pass",
    "interrogate_review",
    "verify_surface",
)


WorkflowAutonomy = Literal["gated", "autonomous_hard_stops", "read_only"]
WorkflowStopAfter = Literal["goal", "design", "plan", "execute", "verify", "report"]
WorkflowGod = Literal["prometheus", "athena", "hephaestus", "momus", "argus"]


def _is_none(value: object) -> bool:
    return value is None


def _is_empty_seq(value: object) -> bool:
    return not value


def _is_default_family(value: object) -> bool:
    return value == "sdlc"


def _is_default_autonomy(value: object) -> bool:
    return value == "gated"


class WorkflowNodeDef(StrictModel):
    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    max_visits: int = Field(default=1, ge=1)
    on_error: str | None = None
    god: WorkflowGod | None = Field(default=None, exclude_if=_is_none)
    phase: str | None = Field(default=None, exclude_if=_is_none)
    role: str | None = Field(default=None, exclude_if=_is_none)
    skill_refs: list[str] = Field(default_factory=list, exclude_if=_is_empty_seq)


class WorkflowEdgeDef(StrictModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)


class WorkflowRouterDef(StrictModel):
    at: str = Field(min_length=1)
    key: str = "route"
    rules: dict[str, str] = Field(default_factory=dict)
    default: str = END


class WorkflowDefinition(StrictModel):
    name: str = Field(min_length=1)
    description: str = ""
    family: str = Field(default="sdlc", exclude_if=_is_default_family)
    autonomy: WorkflowAutonomy = Field(default="gated", exclude_if=_is_default_autonomy)
    applicable_scenarios: list[str] = Field(default_factory=list, exclude_if=_is_empty_seq)
    anti_patterns: list[str] = Field(default_factory=list, exclude_if=_is_empty_seq)
    stop_after: WorkflowStopAfter = "plan"
    topology: str = "modular_product"
    nodes: list[WorkflowNodeDef] = Field(min_length=1)
    edges: list[WorkflowEdgeDef] = Field(default_factory=list)
    routers: list[WorkflowRouterDef] = Field(default_factory=list)
    gates: dict[str, bool] = Field(default_factory=dict)

    def node(self, name: str) -> WorkflowNodeDef | None:
        return next((n for n in self.nodes if n.name == name), None)

    def align_strict_canonical(self) -> bool:
        """Return the ``align_goal`` node's canonical-contract switch."""
        for node in self.nodes:
            if node.type == "align_goal":
                return bool(node.params.get("strict_canonical", True))
        return True


@dataclass
class WorkflowRuntime:
    """Everything node handlers may touch while a workflow executes."""

    store: RunStore
    workspace_root: Path
    artifacts_root: Path
    provider: LLMProvider
    auto_approve: bool = False
    approver: str = "zeus"
    enable_critique: bool = False
    max_steps: int = 16
    worker_max_steps: int | None = None
    apply_to_workspace: bool = False
    topology: str = "modular_product"
    extra: dict[str, Any] = field(default_factory=dict)
    gate_policy: GateDecider | None = None
    sandbox_root: Path | None = None
    execution_drivers: ExecutionDriversConfig = field(
        default_factory=ExecutionDriversConfig.resolve
    )
    driver_registry: DriverRegistry = field(default_factory=DriverRegistry)
    model_provider: Callable[[str], LLMProvider] | None = None
    on_driver_handle: Callable[[ActiveTurnHandle], None] | None = None
    live_activity: LiveActivity | None = None


def _runtime(state: Mapping[str, Any]) -> WorkflowRuntime:
    runtime = state.get("_runtime")
    if not isinstance(runtime, WorkflowRuntime):
        raise RuntimeError("Workflow node requires a '_runtime' WorkflowRuntime in state")
    return runtime


def _run_id(state: Mapping[str, Any]) -> str:
    run_id = state.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("Workflow node requires a 'run_id' in state")
    return run_id


def _align_stage(runtime: WorkflowRuntime) -> AlignStage:
    return AlignStage(
        store=runtime.store,
        provider=runtime.provider,
        workspace_root=runtime.workspace_root,
        trace_dir=runtime.artifacts_root,
        execution_drivers=runtime.execution_drivers,
        driver_registry=runtime.driver_registry,
    )


def _design_stage(runtime: WorkflowRuntime) -> DesignStage:
    return DesignStage(
        store=runtime.store,
        provider=runtime.provider,
        workspace_root=runtime.workspace_root,
        artifacts_root=runtime.artifacts_root,
        execution_drivers=runtime.execution_drivers,
        driver_registry=runtime.driver_registry,
    )


def _orchestrate_stage(runtime: WorkflowRuntime) -> OrchestrateStage:
    return OrchestrateStage(
        store=runtime.store,
        provider=runtime.provider,
        workspace_root=runtime.workspace_root,
        artifacts_root=runtime.artifacts_root,
        sandbox_root=runtime.sandbox_root,
        execution_drivers=runtime.execution_drivers,
        driver_registry=runtime.driver_registry,
        model_provider=runtime.model_provider,
        on_driver_handle=runtime.on_driver_handle,
        live_activity=runtime.live_activity,
    )


def _node_intake(state: dict[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    del params
    runtime = _runtime(state)
    run_id = _run_id(state)
    workspace_mode = str(state.get("workspace_mode") or "brownfield")
    task_type = str(state.get("task_type") or "feature")
    raw_base_ref = state.get("base_ref")
    base_ref = str(raw_base_ref) if isinstance(raw_base_ref, str) else None
    scope_paths = [str(item) for item in (state.get("scope_paths") or [])]
    scope_payload: dict[str, Any] = {
        "workspace_id": state.get("workspace_id"),
        "workspace_mode": workspace_mode,
        "task_type": task_type,
        "base_ref": base_ref,
        "scope_paths": scope_paths,
    }
    runtime.store.append_event(
        run_id=run_id,
        stage="graph",
        event_type="graph.scope_bound",
        actor="intake",
        payload=scope_payload,
    )
    result = triage_input(
        state.get("raw_input", ""),
        provider=runtime.provider,
        store=runtime.store,
        run_id=run_id,
        workspace_mode=workspace_mode,
        task_type=task_type,
        base_ref=base_ref,
        scope_paths=scope_paths,
    )
    updates: dict[str, Any] = {"triage": result.model_dump(mode="json")}

    if result.maturity == InputMaturity.DESIGN_SPEC_READY and result.extracted_design is not None:
        design = result.extracted_design.model_copy(update={"status": SpecStatus.AWAITING_HUMAN})
        runtime.store.save_design(run_id, design, actor="intake")
        updates["design"] = design.model_dump(mode="json")
        updates["route"] = "design_ready"
        return updates

    if result.maturity == InputMaturity.GOAL_SPEC_READY and result.extracted_goal is not None:
        spec = result.extracted_goal.model_copy(update={"status": SpecStatus.AWAITING_HUMAN})
        runtime.store.save_spec(run_id, spec, actor="intake")
        updates["spec"] = spec.model_dump(mode="json")
        updates["route"] = "blocked" if spec.unresolved_blockers() else "goal_ready"
        return updates

    updates["route"] = "raw"
    return updates


def _node_align(state: dict[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _runtime(state)
    run_id = _run_id(state)
    stage = _align_stage(runtime)
    raw_rev_seq = state.get("revision_event_seq")
    expected_rev_seq = (
        raw_rev_seq
        if isinstance(raw_rev_seq, int) and not isinstance(raw_rev_seq, bool)
        else None
    )
    spec = stage.propose_spec(
        run_id=run_id,
        raw_goal=str(state.get("raw_input", "")),
        constraints=list(state.get("constraints") or []),
        enable_critique=False,
        max_steps=runtime.max_steps,
        strict_canonical=bool(params.get("strict_canonical", True)),
        workspace_mode=state.get("workspace_mode"),
        task_type=state.get("task_type"),
        base_ref=state.get("base_ref"),
        scope_paths=list(state.get("scope_paths") or []),
        expected_revision_event_seq=expected_rev_seq,
    )
    if bool(params.get("auto_approve", False)) and not spec.unresolved_blockers():
        spec = stage.confirm_spec(run_id=run_id, approver=runtime.approver)
    return {
        "spec": spec.model_dump(mode="json"),
        "route": "blocked" if spec.unresolved_blockers() else "ready",
    }


def _node_critique(state: dict[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    del params
    runtime = _runtime(state)
    run_id = _run_id(state)
    if not runtime.enable_critique:
        return {"route": "ready"}
    spec = _align_stage(runtime).critique_spec(run_id=run_id)
    return {
        "spec": spec.model_dump(mode="json"),
        "route": "blocked" if spec.unresolved_blockers() else "ready",
    }


def _node_design(state: dict[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    del params
    runtime = _runtime(state)
    run_id = _run_id(state)
    try:
        design = _design_stage(runtime).propose_design(
            run_id,
            auto_approve=False,
            approver=runtime.approver,
            max_steps=runtime.max_steps,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a clean graph pause
        raise GraphPaused(
            "design_spec",
            str(exc),
            {"route": "blocked", "pause_kind": "blocked"},
        ) from exc
    return {"design": design.model_dump(mode="json"), "route": "ready"}


def _node_validate_design(state: dict[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    del params
    runtime = _runtime(state)
    run_id = _run_id(state)
    snap = runtime.store.get_run(run_id)
    design: DesignSpec | None = snap["design"]
    spec: GoalSpec | None = snap["spec"]
    if design is None:
        raise GraphPaused(
            "validate_design",
            "Run has no DesignSpec to validate",
            {"route": "blocked", "pause_kind": "blocked"},
        )
    violations = validate_design_spec(design, spec)
    if violations:
        raise GraphPaused(
            "validate_design",
            "; ".join(violations),
            {"route": "blocked", "pause_kind": "blocked"},
        )
    return {"route": "ok"}


def _node_pm_decompose(state: dict[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _runtime(state)
    run_id = _run_id(state)
    use_design = bool(params.get("use_design", True))
    auto_approve = bool(params.get("auto_approve", runtime.auto_approve))
    stage = _orchestrate_stage(runtime)
    try:
        plan = stage.propose_plan(
            run_id=run_id,
            auto_approve=auto_approve,
            approver=runtime.approver,
            max_steps=runtime.max_steps,
            topology=runtime.topology,
            use_design=use_design,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a clean graph pause
        raise GraphPaused(
            "pm_decompose",
            str(exc),
            {"route": "blocked", "pause_kind": "blocked"},
        ) from exc
    return {"plan": plan.model_dump(mode="json"), "route": "ready"}


def _node_validate_plan(state: dict[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _runtime(state)
    run_id = _run_id(state)
    topology = state.get("topology") or params.get("topology") or runtime.topology
    snap = runtime.store.get_run(run_id)
    plan = snap["plan"]
    if plan is None:
        raise GraphPaused(
            "validate_plan",
            "Run has no TaskPlan to validate",
            {"route": "blocked", "pause_kind": "blocked"},
        )
    violations = validate_plan(
        plan,
        snap["spec"],
        design=snap["design"],
        workspace_root=runtime.workspace_root,
        topology=topology,
    )
    if violations:
        raise GraphPaused(
            "validate_plan",
            "; ".join(violations),
            {"route": "blocked", "pause_kind": "blocked"},
        )
    return {"route": "ok"}


def _gate_auto(runtime: WorkflowRuntime) -> bool:
    return runtime.auto_approve


def _auto_approve_gate(
    runtime: WorkflowRuntime,
    gate: str,
    run_id: str,
    artifact: Any,
    *,
    expected_revision: int,
) -> None:
    control = RunControl(runtime.store)
    if gate == "spec":
        blockers = artifact.unresolved_blockers()
        control.confirm_spec(
            run_id=run_id,
            approver=runtime.approver,
            accept_risk=bool(blockers),
            reason="workflow auto-approve" if blockers else "",
            expected_revision=expected_revision,
        )
    elif gate == "design":
        control.approve_design(
            run_id=run_id,
            approver=runtime.approver,
            reason="workflow auto-approve",
            expected_revision=expected_revision,
        )
    else:
        control.approve_plan(
            run_id=run_id,
            approver=runtime.approver,
            reason="workflow auto-approve",
            expected_revision=expected_revision,
        )


def _node_human_gate(state: dict[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _runtime(state)
    run_id = _run_id(state)
    gate = str(params.get("gate", "spec"))
    enabled = bool(params.get("enabled", True))
    auto = _gate_auto(runtime) or not enabled
    snap = runtime.store.get_run(run_id)

    gate_spec = GATE_SPECS.get(gate)
    if gate_spec is None:
        pause_node = str(params.get("_node_name") or f"gate_{gate}")
        raise GraphPaused(pause_node, f"Unknown human gate '{gate}'")

    pause_node = str(params.get("_node_name") or gate_spec["node"])
    artifact = snap[gate_spec["snapshot_key"]]
    if artifact is None:
        raise GraphPaused(pause_node, f"Run has no {gate_spec['label']} to approve")

    approved = is_artifact_approved(runtime.store, run_id, gate, snap)
    policy = runtime.gate_policy
    if not approved and auto:
        _auto_approve_gate(
            runtime,
            gate,
            run_id,
            artifact,
            expected_revision=int(snap["state_revision"]),
        )
        approved = True
    elif not approved and policy is not None:
        decision = policy(gate, artifact, runtime, run_id)
        if decision == "auto_approve":
            _auto_approve_gate(
                runtime,
                gate,
                run_id,
                artifact,
                expected_revision=int(snap["state_revision"]),
            )
            approved = True
        elif decision == "block":
            raise GraphPaused(
                pause_node,
                f"{gate_spec['label']} failed deterministic validation",
                {"route": "blocked", "pause_kind": "blocked"},
            )
        else:
            pause_kind = "clarify" if gate == "spec" and _has_blockers(artifact) else "approval"
            raise GraphPaused(
                pause_node,
                f"{gate_spec['label']} is awaiting human approval",
                {"route": "awaiting", "pause_kind": pause_kind},
            )
    if not approved:
        target_id = getattr(artifact, gate_spec["id_attr"])
        raise GraphPaused(
            pause_node,
            f"{gate_spec['label']} {target_id} is awaiting human approval",
            {"route": "awaiting"},
        )
    return {"route": "approved"}


def _has_blockers(artifact: Any) -> bool:
    resolver = getattr(artifact, "unresolved_blockers", None)
    if callable(resolver):
        return bool(resolver())
    return False


def _node_execute(state: dict[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    del params
    runtime = _runtime(state)
    run_id = _run_id(state)
    snap = _orchestrate_stage(runtime).run_execution(
        run_id=run_id,
        apply_to_workspace=runtime.apply_to_workspace,
        worker_max_steps=runtime.worker_max_steps,
    )
    plan = snap["plan"]
    return {
        "execution": {
            "status": snap["status"].value,
            "tasks": [
                {"task_id": t.task_id, "status": t.status.value}
                for t in (plan.tasks if plan is not None else [])
            ],
        },
        "route": "done",
    }


def _craft_stage(runtime: WorkflowRuntime) -> CraftStage:
    return CraftStage(
        store=runtime.store,
        workspace_root=runtime.workspace_root,
        provider=runtime.provider,
        artifacts_root=runtime.artifacts_root,
        sandbox_root=runtime.sandbox_root,
        execution_drivers=runtime.execution_drivers,
        driver_registry=runtime.driver_registry,
        model_provider=runtime.model_provider,
        on_driver_handle=runtime.on_driver_handle,
    )


def _craft_step_budget(
    params: Mapping[str, Any],
    runtime: WorkflowRuntime,
    default_cap: int,
    *,
    use_worker_steps: bool = False,
) -> int:
    """Resolve a craft node's step budget without altering builtin workflow YAML hashes."""
    if params.get("max_steps") is not None:
        return int(params["max_steps"])
    if use_worker_steps:
        if runtime.worker_max_steps is not None:
            return int(runtime.worker_max_steps)
        return default_cap if runtime.max_steps == 16 else runtime.max_steps
    return min(runtime.max_steps, default_cap)


def _node_explore_reproduce(
    state: dict[str, Any], params: Mapping[str, Any]
) -> dict[str, Any]:
    runtime = _runtime(state)
    run_id = _run_id(state)
    try:
        return _craft_stage(runtime).explore_reproduce(
            run_id=run_id,
            raw_input=str(state.get("raw_input", "")),
            constraints=list(state.get("constraints", [])),
            scope_paths=list(state.get("scope_paths", [])),
            node_name=str(params.get("_node_name") or "explore"),
            role=str(params.get("_node_role") or "explorer"),
            skill_refs=tuple(params.get("_node_skill_refs") or ()),
            require_repro=bool(params.get("require_repro", False)),
            max_steps=_craft_step_budget(params, runtime, 8),
        )
    except CraftPause as paused:
        raise GraphPaused(paused.node_name, paused.reason, paused.state_updates) from paused


def _node_architect_sketch(
    state: dict[str, Any], params: Mapping[str, Any]
) -> dict[str, Any]:
    runtime = _runtime(state)
    run_id = _run_id(state)
    try:
        return _craft_stage(runtime).architect_sketch(
            run_id=run_id,
            raw_input=str(state.get("raw_input", "")),
            state=state,
            constraints=list(state.get("constraints", [])),
            scope_paths=list(state.get("scope_paths", [])),
            node_name=str(params.get("_node_name") or "architect"),
            role=str(params.get("_node_role") or "architect"),
            skill_refs=tuple(params.get("_node_skill_refs") or ()),
            max_steps=_craft_step_budget(params, runtime, 6),
        )
    except CraftPause as paused:
        raise GraphPaused(paused.node_name, paused.reason, paused.state_updates) from paused


def _node_build_direct(state: dict[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _runtime(state)
    run_id = _run_id(state)
    try:
        return _craft_stage(runtime).build_direct(
            run_id=run_id,
            raw_input=str(state.get("raw_input", "")),
            state=state,
            node_name=str(params.get("_node_name") or "build"),
            role=str(params.get("_node_role") or "builder"),
            skill_refs=tuple(params.get("_node_skill_refs") or ()),
            strict_scope=bool(params.get("strict_scope", True)),
            max_diff_gate_retries=int(params.get("max_diff_gate_retries", 2)),
            max_steps=_craft_step_budget(params, runtime, 24, use_worker_steps=True),
        )
    except CraftPause as paused:
        raise GraphPaused(paused.node_name, paused.reason, paused.state_updates) from paused


def _node_hygiene_pass(state: dict[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _runtime(state)
    run_id = _run_id(state)
    try:
        return _craft_stage(runtime).hygiene_pass(
            run_id=run_id,
            state=state,
            node_name=str(params.get("_node_name") or "hygiene"),
            role=str(params.get("_node_role") or "hygiene"),
            skill_refs=tuple(params.get("_node_skill_refs") or ()),
            strict_scope=bool(params.get("strict_scope", True)),
            max_steps=_craft_step_budget(params, runtime, 4),
        )
    except CraftPause as paused:
        raise GraphPaused(paused.node_name, paused.reason, paused.state_updates) from paused


def _node_interrogate_review(
    state: dict[str, Any], params: Mapping[str, Any]
) -> dict[str, Any]:
    runtime = _runtime(state)
    run_id = _run_id(state)
    visit_cap = int(params.get("_node_loop_visit_cap") or params.get("_node_max_visits") or 3)
    requested_cycles = int(params.get("max_cycles") or visit_cap)
    max_cycles = max(1, min(requested_cycles, visit_cap))
    try:
        return _craft_stage(runtime).interrogate_review(
            run_id=run_id,
            raw_input=str(state.get("raw_input", "")),
            state=state,
            node_name=str(params.get("_node_name") or "interrogate"),
            role=str(params.get("_node_role") or "interrogator"),
            skill_refs=tuple(params.get("_node_skill_refs") or ()),
            enable_thermo=bool(params.get("enable_thermo", False)),
            max_cycles=max_cycles,
            max_steps=_craft_step_budget(params, runtime, 6),
        )
    except CraftPause as paused:
        raise GraphPaused(paused.node_name, paused.reason, paused.state_updates) from paused


def _node_verify_surface(state: dict[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _runtime(state)
    run_id = _run_id(state)
    raw_commands = params.get("commands")
    commands = list(raw_commands) if isinstance(raw_commands, list) else None
    try:
        return _craft_stage(runtime).verify_surface(
            run_id=run_id,
            state=state,
            node_name=str(params.get("_node_name") or "verify_surface"),
            role=str(params.get("_node_role") or "verifier"),
            commands=commands,
            apply_to_workspace=runtime.apply_to_workspace,
            timeout_s=int(params.get("timeout_s", 120)),
        )
    except CraftPause as paused:
        raise GraphPaused(paused.node_name, paused.reason, paused.state_updates) from paused


_HANDLERS: dict[str, Callable[[dict[str, Any], Mapping[str, Any]], dict[str, Any]]] = {
    "intake_triage": _node_intake,
    "align_goal": _node_align,
    "critic_parallel": _node_critique,
    "design_spec": _node_design,
    "validate_design": _node_validate_design,
    "pm_decompose": _node_pm_decompose,
    "validate_plan": _node_validate_plan,
    "execute_task_dag": _node_execute,
    "human_gate": _node_human_gate,
    "explore_reproduce": _node_explore_reproduce,
    "architect_sketch": _node_architect_sketch,
    "build_direct": _node_build_direct,
    "hygiene_pass": _node_hygiene_pass,
    "interrogate_review": _node_interrogate_review,
    "verify_surface": _node_verify_surface,
}


def _bind(
    handler: Callable[[dict[str, Any], Mapping[str, Any]], dict[str, Any]],
    params: Mapping[str, Any],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _call(state: dict[str, Any]) -> dict[str, Any]:
        return handler(state, params)

    return _call


def _bind_router(definition: WorkflowRouterDef) -> Callable[[Mapping[str, Any]], str]:
    rules = dict(definition.rules)
    default = definition.default
    key = definition.key

    def _route(state: Mapping[str, Any]) -> str:
        raw = state.get(key)
        route_key = str(raw) if raw is not None else default
        return rules.get(route_key, default)

    return _route


def build_graph(definition: WorkflowDefinition, *, runtime: WorkflowRuntime | None = None) -> Graph:
    """Bind one declarative definition into an executable wave graph."""
    del runtime  # runtime travels inside state['_runtime']
    graph = Graph(name=definition.name)
    gate_flags = dict(definition.gates)
    for node_def in definition.nodes:
        handler = _HANDLERS.get(node_def.type)
        if handler is None:
            raise ValueError(
                f"Unknown node type '{node_def.type}' in workflow '{definition.name}'. "
                f"Known types: {', '.join(_NODE_TYPES)}"
            )
        params: dict[str, Any] = dict(node_def.params)
        if node_def.type == "human_gate":
            gate_name = str(params.get("gate", "spec"))
            params.setdefault("enabled", gate_flags.get(f"approve_{gate_name}", True))
            params.setdefault("_node_name", node_def.name)
        elif node_def.type in (
            "explore_reproduce",
            "architect_sketch",
            "build_direct",
            "hygiene_pass",
            "interrogate_review",
            "verify_surface",
        ):
            params.setdefault("_node_name", node_def.name)
            if node_def.role is not None:
                params.setdefault("_node_role", node_def.role)
            if node_def.skill_refs:
                params.setdefault("_node_skill_refs", tuple(node_def.skill_refs))
            params.setdefault("_node_max_visits", node_def.max_visits)
            if node_def.type == "interrogate_review":
                loop_caps = [node_def.max_visits]
                for router in definition.routers:
                    if router.at != node_def.name:
                        continue
                    findings_target = router.rules.get("findings")
                    if findings_target:
                        target_def = definition.node(findings_target)
                        if target_def is not None:
                            loop_caps.append(target_def.max_visits)
                params.setdefault("_node_loop_visit_cap", min(loop_caps))
        graph.add_node(
            Node(
                name=node_def.name,
                fn=_bind(handler, params),
                kind=node_def.type,
                max_visits=node_def.max_visits,
                on_error=node_def.on_error,
                god=node_def.god,
                phase=node_def.phase,
                role=node_def.role,
                skill_refs=tuple(node_def.skill_refs),
            )
        )
    for edge in definition.edges:
        graph.add_edge(edge.source, edge.target)
    for router in definition.routers:
        graph.add_router(
            router.at,
            _bind_router(router),
            targets=[*router.rules.values(), router.default],
        )
    return graph


def _resolve_path(name_or_path: str | Path) -> Path:
    candidate = Path(name_or_path)
    if isinstance(name_or_path, Path) or candidate.suffix in (".yaml", ".yml"):
        if not candidate.is_file():
            raise FileNotFoundError(f"Workflow file not found: {candidate}")
        return candidate
    builtin = WORKFLOW_DIR / f"{name_or_path}.yaml"
    if builtin.is_file():
        return builtin
    raise FileNotFoundError(
        f"Unknown workflow '{name_or_path}'. Builtin workflows: "
        f"{', '.join(list_builtin_workflows())}"
    )


def load_workflow(name_or_path: str | Path) -> WorkflowDefinition:
    """Load and schema-validate a builtin name or a ``.yaml``/``.yml`` path."""
    path = _resolve_path(name_or_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Workflow file {path} must contain a YAML mapping")
    definition = WorkflowDefinition.model_validate(raw)
    unknown = sorted({n.type for n in definition.nodes} - set(_NODE_TYPES))
    if unknown:
        raise ValueError(
            f"Workflow '{definition.name}' declares unknown node types: {', '.join(unknown)}"
        )
    return definition


def list_builtin_workflows() -> list[str]:
    """Return the names of the YAML workflows shipped inside the package."""
    return sorted(p.stem for p in WORKFLOW_DIR.glob("*.yaml"))


def describe_workflow(definition: WorkflowDefinition) -> dict[str, Any]:
    """Human/API-facing summary combining metadata with the bound graph topology."""
    graph = build_graph(definition)
    return {
        "name": definition.name,
        "description": definition.description,
        "family": definition.family,
        "autonomy": definition.autonomy,
        "applicable_scenarios": list(definition.applicable_scenarios),
        "anti_patterns": list(definition.anti_patterns),
        "stop_after": definition.stop_after,
        "topology": definition.topology,
        "gates": dict(definition.gates),
        "graph": graph.describe(),
    }


def workflow_status_seed(definition: WorkflowDefinition) -> dict[str, str]:
    """Initial per-node status map for projections."""
    return {node.name: "pending" for node in definition.nodes}


__all__ = [
    "END",
    "WORKFLOW_DIR",
    "WorkflowDefinition",
    "WorkflowEdgeDef",
    "WorkflowNodeDef",
    "WorkflowRouterDef",
    "WorkflowRuntime",
    "build_graph",
    "describe_workflow",
    "list_builtin_workflows",
    "load_workflow",
    "workflow_status_seed",
    "RunStatus",
]
