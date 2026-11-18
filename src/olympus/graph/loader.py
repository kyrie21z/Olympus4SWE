"""Declarative YAML workflow loading bound to Olympus stage node handlers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from olympus.core.contracts import (
    ApprovalTarget,
    DesignSpec,
    GoalSpec,
    InputMaturity,
    RunStatus,
    SpecStatus,
    StrictModel,
)
from olympus.core.dag import validate_design_spec, validate_plan
from olympus.core.store import RunStore
from olympus.graph.engine import END, Graph, GraphPaused, Node
from olympus.providers.base import LLMProvider
from olympus.stages.align import AlignStage
from olympus.stages.design import DesignStage, triage_input
from olympus.stages.orchestrate import OrchestrateStage

WORKFLOW_DIR = Path(__file__).resolve().parent / "workflows"

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
)


class WorkflowNodeDef(StrictModel):
    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    max_visits: int = Field(default=1, ge=1)
    on_error: str | None = None


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
    stop_after: Literal["goal", "design", "plan", "execute"] = "plan"
    topology: str = "modular_product"
    nodes: list[WorkflowNodeDef] = Field(min_length=1)
    edges: list[WorkflowEdgeDef] = Field(default_factory=list)
    routers: list[WorkflowRouterDef] = Field(default_factory=list)
    gates: dict[str, bool] = Field(default_factory=dict)

    def node(self, name: str) -> WorkflowNodeDef | None:
        return next((n for n in self.nodes if n.name == name), None)


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
    max_steps: int = 10
    worker_max_steps: int = 20
    apply_to_workspace: bool = False
    topology: str = "modular_product"
    extra: dict[str, Any] = field(default_factory=dict)


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
    )


def _design_stage(runtime: WorkflowRuntime) -> DesignStage:
    return DesignStage(
        store=runtime.store,
        provider=runtime.provider,
        workspace_root=runtime.workspace_root,
        artifacts_root=runtime.artifacts_root,
    )


def _orchestrate_stage(runtime: WorkflowRuntime) -> OrchestrateStage:
    return OrchestrateStage(
        store=runtime.store,
        provider=runtime.provider,
        workspace_root=runtime.workspace_root,
        artifacts_root=runtime.artifacts_root,
    )


def _node_intake(state: dict[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    del params
    runtime = _runtime(state)
    run_id = _run_id(state)
    result = triage_input(
        state.get("raw_input", ""),
        provider=runtime.provider,
        store=runtime.store,
        run_id=run_id,
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
    spec = stage.propose_spec(
        run_id=run_id,
        raw_goal=str(state.get("raw_input", "")),
        constraints=list(state.get("constraints") or []),
        enable_critique=False,
        max_steps=runtime.max_steps,
        strict_canonical=bool(params.get("strict_canonical", True)),
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
        raise GraphPaused("design_spec", str(exc)) from exc
    return {"design": design.model_dump(mode="json"), "route": "ready"}


def _node_validate_design(state: dict[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    del params
    runtime = _runtime(state)
    run_id = _run_id(state)
    snap = runtime.store.get_run(run_id)
    design: DesignSpec | None = snap["design"]
    spec: GoalSpec | None = snap["spec"]
    if design is None:
        raise GraphPaused("validate_design", "Run has no DesignSpec to validate")
    violations = validate_design_spec(design, spec)
    if violations:
        raise GraphPaused("validate_design", "; ".join(violations))
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
        raise GraphPaused("pm_decompose", str(exc)) from exc
    return {"plan": plan.model_dump(mode="json"), "route": "ready"}


def _node_validate_plan(state: dict[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    del params
    runtime = _runtime(state)
    run_id = _run_id(state)
    snap = runtime.store.get_run(run_id)
    plan = snap["plan"]
    if plan is None:
        raise GraphPaused("validate_plan", "Run has no TaskPlan to validate")
    violations = validate_plan(
        plan,
        snap["spec"],
        design=snap["design"],
        workspace_root=runtime.workspace_root,
    )
    if violations:
        raise GraphPaused("validate_plan", "; ".join(violations))
    return {"route": "ok"}


def _gate_auto(runtime: WorkflowRuntime) -> bool:
    return runtime.auto_approve


def _node_human_gate(state: dict[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _runtime(state)
    run_id = _run_id(state)
    gate = str(params.get("gate", "spec"))
    enabled = bool(params.get("enabled", True))
    auto = _gate_auto(runtime) or not enabled
    snap = runtime.store.get_run(run_id)

    if gate == "spec":
        spec: GoalSpec | None = snap["spec"]
        if spec is None:
            raise GraphPaused("gate_spec", "Run has no GoalSpec to approve")
        approved = runtime.store.is_approved(
            run_id=run_id,
            target=ApprovalTarget.GOAL_SPEC,
            target_id=spec.spec_id,
            target_hash=spec.content_hash,
        )
        if not approved and auto:
            blockers = spec.unresolved_blockers()
            _align_stage(runtime).confirm_spec(
                run_id=run_id,
                approver=runtime.approver,
                accept_risk=bool(blockers),
                reason="workflow auto-approve" if blockers else "",
            )
            approved = True
        if not approved:
            raise GraphPaused(
                "gate_spec",
                f"GoalSpec {spec.spec_id} is awaiting human approval",
                {"route": "awaiting"},
            )
        return {"route": "approved"}

    if gate == "design":
        design: DesignSpec | None = snap["design"]
        if design is None:
            raise GraphPaused("gate_design", "Run has no DesignSpec to approve")
        approved = runtime.store.is_approved(
            run_id=run_id,
            target=ApprovalTarget.DESIGN_SPEC,
            target_id=design.design_id,
            target_hash=design.content_hash,
        )
        if not approved and auto:
            _design_stage(runtime).approve_design(
                run_id=run_id,
                approver=runtime.approver,
                reason="workflow auto-approve",
            )
            approved = True
        if not approved:
            raise GraphPaused(
                "gate_design",
                f"DesignSpec {design.design_id} is awaiting human approval",
                {"route": "awaiting"},
            )
        return {"route": "approved"}

    if gate == "plan":
        plan = snap["plan"]
        if plan is None:
            raise GraphPaused("gate_plan", "Run has no TaskPlan to approve")
        approved = runtime.store.is_approved(
            run_id=run_id,
            target=ApprovalTarget.TASK_PLAN,
            target_id=plan.plan_id,
            target_hash=plan.content_hash,
        )
        if not approved and auto:
            _orchestrate_stage(runtime).approve_plan(
                run_id=run_id,
                approver=runtime.approver,
                reason="workflow auto-approve",
            )
            approved = True
        if not approved:
            raise GraphPaused(
                "gate_plan",
                f"TaskPlan {plan.plan_id} is awaiting human approval",
                {"route": "awaiting"},
            )
        return {"route": "approved"}

    raise GraphPaused(f"gate_{gate}", f"Unknown human gate '{gate}'")


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
        graph.add_node(
            Node(
                name=node_def.name,
                fn=_bind(handler, params),
                kind=node_def.type,
                max_visits=node_def.max_visits,
                on_error=node_def.on_error,
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
