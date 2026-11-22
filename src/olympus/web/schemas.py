"""Strict Pydantic v2 request/response contracts for the Ariadne HTTP API.

Every payload that crosses the HTTP seam is modeled here so that
``create_fastapi_app(...).openapi()`` exposes concrete schemas and the
checked-in ``openapi.json`` snapshot can guard against contract drift.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
)

from olympus.core.contracts import (
    AcceptanceCheck,
    DesignSpec,
    GoalSpec,
    ProviderTimeoutRetry,
    ReviewVerdict,
    RunEvent,
    TaskPlan,
)


def _omit_none(value: Any) -> bool:
    """Serialization predicate for fields that were absent in the legacy wire shape."""
    return value is None


class StrictModel(BaseModel):
    """Base model that rejects unexpected fields at the API boundary."""

    model_config = ConfigDict(extra="forbid")


class ErrorResponse(StrictModel):
    ok: Literal[False] = False
    error: str


class ConflictResponse(StrictModel):
    ok: Literal[False] = False
    conflict: Literal[True] = True
    error: str
    projection: RunProjection | None = Field(default=None, exclude_if=_omit_none)


class HealthResponse(StrictModel):
    ok: Literal[True] = True
    service: str
    version: str


class RunSummary(StrictModel):
    run_id: str
    title: str
    mode: str
    status: str
    state_revision: int
    workflow_name: str
    created_at: str
    updated_at: str


class RunListResponse(StrictModel):
    ok: Literal[True] = True
    runs: list[RunSummary]
    default_run_id: str | None = None


class PauseInfo(StrictModel):
    node: str | None = None
    reason: str | None = None


class ProviderConfig(StrictModel):
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    env_file: str | None = None
    providers_file: str | None = None


class CreateRunRequest(StrictModel):
    raw_input: str | None = None
    goal: str | None = None
    demo_id: str | None = None
    title: str | None = None
    workflow_name: str | None = None
    provider_config: ProviderConfig = Field(default_factory=ProviderConfig)
    worker_driver: str | None = None
    reviewer_driver: str | None = None
    monitor_model: str | None = None
    auto_approve: bool = False
    sync: bool = False
    constraints: list[str] | None = None
    workspace_root: str | None = None


class WorkflowRunResult(StrictModel):
    run_id: str | None = None
    workflow: str | None = None
    topology: str | None = None
    stop_after: str | None = None
    status: str | None = None
    state_revision: int | None = None
    paused: PauseInfo | None = None
    error: str | None = None
    coalesced: bool | None = None
    deduplicated: bool | None = None
    complete: bool | None = Field(default=None, exclude_if=_omit_none)


class CreateRunResponse(StrictModel):
    ok: Literal[True] = True
    run_id: str
    workflow: str | None = Field(default=None, exclude_if=_omit_none)
    topology: str | None = Field(default=None, exclude_if=_omit_none)
    stop_after: str | None = Field(default=None, exclude_if=_omit_none)
    status: str | None = Field(default=None, exclude_if=_omit_none)
    state_revision: int | None = Field(default=None, exclude_if=_omit_none)
    paused: PauseInfo | None = None
    error: str | None = Field(default=None, exclude_if=_omit_none)
    coalesced: bool | None = Field(default=None, exclude_if=_omit_none)
    complete: bool | None = Field(default=None, exclude_if=_omit_none)


class SeedDemoResponse(StrictModel):
    ok: Literal[True] = True
    seeded: list[str]
    runs: list[RunSummary]
    default_run_id: str | None = None


class EventsResponse(StrictModel):
    ok: Literal[True] = True
    run_id: str
    state_revision: int
    status: str
    events: list[RunEvent]


class AttemptArtifactsResponse(StrictModel):
    ok: Literal[True] = True
    artifacts: RecordedAttemptArtifacts


def _strip_non_empty(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("value must not be blank")
    return cleaned


class _CommandBase(StrictModel):
    action: str
    operation_id: str | None = None
    actor: str | None = None
    expected_state_revision: int | None = None
    target_hash: str | None = None
    auto_advance: bool = True
    sync: bool = False


class PauseCommand(_CommandBase):
    action: Literal["pause"]


class ResumeCommand(_CommandBase):
    action: Literal["resume"]


class ApproveSpecCommand(_CommandBase):
    action: Literal["approve_spec"]
    accept_risk: bool = False
    reason: str = ""


class ApproveDesignCommand(_CommandBase):
    action: Literal["approve_design"]
    reason: str = ""


class ApprovePlanCommand(_CommandBase):
    action: Literal["approve_plan"]
    reason: str = ""


class ClarifyCommand(_CommandBase):
    action: Literal["clarify"]
    answers: dict[str, str] = Field(default_factory=dict)
    resolutions: dict[str, str] = Field(default_factory=dict)
    out_of_scope: list[str] = Field(default_factory=list)


class RetryTaskCommand(_CommandBase):
    action: Literal["retry_task"]
    task_id: str = Field(min_length=1)
    hint: str = ""

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        return _strip_non_empty(value)


class AcceptRiskCommand(_CommandBase):
    action: Literal["accept_risk"]
    task_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)

    @field_validator("task_id", "reason")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        return _strip_non_empty(value)


class ReassignHumanCommand(_CommandBase):
    action: Literal["reassign_human"]
    task_id: str = Field(min_length=1)
    note: str = ""

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        return _strip_non_empty(value)


class RetryWorkflowCommand(_CommandBase):
    action: Literal["retry_workflow"]


class PromoteToExecuteCommand(_CommandBase):
    action: Literal["promote_to_execute"]
    workflow_name: str | None = None
    max_llm_calls: int | None = Field(default=None, ge=1)


CommandRequest = Annotated[
    PauseCommand
    | ResumeCommand
    | ApproveSpecCommand
    | ApproveDesignCommand
    | ApprovePlanCommand
    | ClarifyCommand
    | RetryTaskCommand
    | AcceptRiskCommand
    | ReassignHumanCommand
    | RetryWorkflowCommand
    | PromoteToExecuteCommand,
    Field(discriminator="action"),
]


class CommandResult(StrictModel):
    """Union of recorded HITL command results; serializes only fields present in the outcome."""

    model_config = ConfigDict(extra="allow")

    run_id: str | None = None
    status: str | None = None
    state_revision: int | None = None
    task_id: str | None = None
    spec_id: str | None = None
    spec_hash: str | None = None
    version: int | None = None
    remaining_blockers: list[str] | None = None
    approval_id: str | None = None
    target: str | None = None
    target_id: str | None = None
    target_hash: str | None = None
    approver: str | None = None
    decision: str | None = None
    reason: str | None = None
    created_at: str | None = None
    from_workflow: str | None = None
    workflow: WorkflowRunResult | str | None = None
    budget_max_calls: int | None = None
    deduplicated: bool | None = None

    @model_serializer(mode="wrap")
    def _serialize_present_fields(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        rendered: dict[str, Any] = dict(handler(self))
        present = self.model_fields_set | set((self.model_extra or {}).keys())
        return {key: value for key, value in rendered.items() if key in present}


class ReviewEntry(StrictModel):
    ref: str
    text: str
    label: str | None = Field(default=None, exclude_if=_omit_none)
    cost: str | None = Field(default=None, exclude_if=_omit_none)
    provenance: Literal["agent_high_impact", "unknown"] | None = Field(
        default=None, exclude_if=_omit_none
    )
    change: Literal["changed", "added", "removed", "affected"] | None = Field(
        default=None, exclude_if=_omit_none
    )
    previous: ReviewEntry | None = Field(default=None, exclude_if=_omit_none)
    related: bool | None = Field(default=None, exclude_if=_omit_none)


class ReviewSection(StrictModel):
    title: str
    entries: list[ReviewEntry]
    empty_text: str | None = Field(default=None, exclude_if=_omit_none)


class ApprovalReview(StrictModel):
    artifact: Literal["spec", "design", "plan"]
    target_hash: str
    sections: list[ReviewSection]
    baseline: GoalSpec | DesignSpec | TaskPlan | None = Field(default=None, exclude_if=_omit_none)
    unchanged_sections: list[ReviewSection] | None = Field(default=None, exclude_if=_omit_none)
    comparison_gap: str | None = Field(default=None, exclude_if=_omit_none)


class StagePipelineItem(StrictModel):
    id: str
    label: str
    status: str


class BudgetSummary(StrictModel):
    calls_used: int
    calls_reserved: int
    max_calls: int | None = None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int
    max_tokens: int | None = None
    unknown_outcome_calls: int


class ProgressSummary(StrictModel):
    accepted_tasks: int
    verified_tasks: int
    total_tasks: int
    ratio_text: str


class HitlBanner(StrictModel):
    kind: str
    tone: str
    task_id: str | None = None
    failed_node: str | None = Field(default=None, exclude_if=_omit_none)
    headline: str
    detail: str
    actions: list[str]
    affected_task_ids: list[str] | None = Field(default=None, exclude_if=_omit_none)


class TriageView(StrictModel):
    maturity: str
    reasons: list[str]
    has_goal: bool = False
    has_design: bool = False


class StageActivityItem(StrictModel):
    event_seq: int
    stage: str
    event_type: str
    actor: str
    summary: str


class RunProgressItem(StrictModel):
    event_seq: int
    created_at: str
    event_type: str
    node: str | None = None
    task_id: str | None = None
    to: str | None = None
    summary: str | None = Field(default=None, exclude_if=_omit_none)
    detail_available: bool | None = Field(default=None, exclude_if=_omit_none)


class KeyOutputItem(StrictModel):
    event_seq: int
    created_at: str
    event_type: str
    node: str | None = None
    task_id: str | None = None
    to: str | None = None
    key: str
    kind: str
    artifact_id: str
    version: int | None = None
    content_hash: str | None = Field(default=None, exclude_if=_omit_none)
    current: bool
    summary: str
    detail_available: bool | None = Field(default=None, exclude_if=_omit_none)


class WorkflowGraphNode(StrictModel):
    name: str
    kind: str
    max_visits: int
    on_error: str | None = None
    routed: bool


class WorkflowGraphEdge(StrictModel):
    source: str
    target: str
    conditional: bool


class WorkflowGraph(StrictModel):
    name: str
    start: list[str]
    nodes: list[WorkflowGraphNode]
    edges: list[WorkflowGraphEdge]
    routers: list[str]


class WorkflowNodeStatus(StrictModel):
    name: str
    status: str


class TraversedEdge(StrictModel):
    source: str
    target: str


class WorkflowPause(StrictModel):
    node: str | None = None
    reason: str | None = None


class WorkflowView(StrictModel):
    name: str
    topology: str | None = None
    stop_after: str | None = None
    graph: WorkflowGraph | None = None
    nodes: list[WorkflowNodeStatus]
    node_statuses: dict[str, str]
    traversed_edges: list[TraversedEdge]
    current_node: str | None = None
    paused: WorkflowPause | None = None
    failed_node: str | None = None
    last_error: str | None = None
    finished: bool


class CurrentTool(StrictModel):
    step: int
    tool: str
    target_hint: str


class RecentTool(StrictModel):
    step: int
    tool: str
    status: str
    duration_ms: int


class LiveProgress(StrictModel):
    driver_name: str | None = None
    current_tool: CurrentTool | None = None
    last_event_at: str | None = None
    recent_tools: list[RecentTool]
    steer_count: int
    watchdog_verdict: str | None = None
    watchdog_summary: str | None = None
    preview: str | None = Field(default=None, exclude_if=_omit_none)


class ProviderRetryRecord(ProviderTimeoutRetry):
    event_seq: int
    actor: str


class TaskBlocker(StrictModel):
    task_id: str
    headline: str
    detail: str
    affected_task_ids: list[str]
    actions: list[str]


class SubagentStep(StrictModel):
    tool: str
    target: str
    status: str
    duration_ms: int


class ReceiptLLM(StrictModel):
    calls: int
    input_tokens: int
    output_tokens: int


class ReceiptTool(StrictModel):
    tool: str
    status: str
    duration_ms: int


class ReceiptArtifacts(StrictModel):
    diff_added: int
    diff_removed: int
    checks_passed: int
    checks_total: int


class TurnReceipt(StrictModel):
    turn_id: str
    actor: str
    task_id: str | None = None
    llm: ReceiptLLM
    tools: list[ReceiptTool]
    artifacts: ReceiptArtifacts
    one_line: str


class ToolCallItem(StrictModel):
    step: int
    tool: str
    target: str
    status: str
    is_error: bool
    duration_ms: int
    output_preview: str


class TaskArtifacts(StrictModel):
    attempt_dir: str | None = None
    diff_patch: str
    check_logs: dict[str, str]
    verdicts: list[ReviewVerdict]
    tool_calls: list[ToolCallItem]
    manifest_md: str


class RecordedAttemptArtifacts(StrictModel):
    diff_patch: str
    manifest_md: str
    check_logs: dict[str, str]
    verdicts: list[ReviewVerdict]


class DagNode(StrictModel):
    task_id: str
    title: str
    objective: str
    status: str
    layer: int
    critical: bool
    attempt: int
    review_cycle: int
    max_cycles: int
    requirement_refs: list[str]
    design_refs: list[str]
    role: str
    kind: str
    deliverables: list[str]
    depends_on: list[str]
    write_scope: list[str]
    read_scope: list[str]
    last_error: str | None = None
    provider_retry: ProviderTimeoutRetry | None = None
    provider_retry_history: list[ProviderRetryRecord]
    live_progress: LiveProgress
    blocked_by: list[str]
    blocker: TaskBlocker | None = None
    accepted_risk_reason: str | None = None
    executor: str
    diagnostic_tool: str
    diagnostic_output: str
    diagnostic_attribution: str
    acceptance_checks: list[AcceptanceCheck]
    subagent_stream: list[SubagentStep]
    terminal_lines: list[str]
    receipt: TurnReceipt
    manifest_md: str
    artifacts: TaskArtifacts


class DagEdge(StrictModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source: str = Field(alias="from")
    target: str = Field(alias="to")
    critical: bool
    blocked_edge: bool


class DagView(StrictModel):
    nodes: list[DagNode]
    edges: list[DagEdge]


class SwimlaneItem(StrictModel):
    id: str
    task_id: str | None = None
    label: str
    status: str
    attempts: int
    summary: str
    actor: str


class SwimlaneRow(StrictModel):
    actor: str
    items: list[SwimlaneItem]


class WaterfallTurn(StrictModel):
    turn_id: str
    actor: str
    task_id: str | None = None
    title: str
    status: str
    summary: str
    duration_ms: int
    steps: int
    offset_pct: float
    width_pct: float
    receipt: TurnReceipt
    manifest_md: str


class DesignModuleRef(StrictModel):
    id: str
    name: str


class TraceabilityMatrix(StrictModel):
    requirement_id: str
    description: str
    source_ref: str
    acceptance_criteria: list[str]
    design_module_ids: list[str]
    design_modules: list[DesignModuleRef]
    task_ids: list[str]
    verification_status: str
    verification_label: str
    evidence_note: str


class DeliveryEvidence(StrictModel):
    delivery_version: str
    event_seq: int
    receipt_id: str
    check_position: int
    log_hash: str
    exit_code: int
    log: str


class DeliveryCriterion(StrictModel):
    requirement_id: str
    criterion_index: int
    text: str
    status: str
    reason: str
    risk_accepted: bool
    evidence: DeliveryEvidence | None = None


class DeliveryCounts(StrictModel):
    passed: int
    failed: int
    unverified: int


class DeliveryAcceptance(StrictModel):
    spec_id: str
    spec_version: int
    spec_hash: str
    counts: DeliveryCounts
    criteria: list[DeliveryCriterion]


class RunProjection(StrictModel):
    run_id: str
    title: str
    raw_input: str
    workspace_root: str
    mode: str
    status: str
    state_revision: int
    created_at: str
    updated_at: str
    workflow_name: str
    workflow: WorkflowView
    blueprint_completed: bool
    triage: TriageView | None = None
    stage_activity: list[StageActivityItem]
    recent_progress: RunProgressItem | None = None
    key_outputs: list[KeyOutputItem]
    spec_approved: bool
    design_approved: bool
    plan_approved: bool
    stage_pipeline: list[StagePipelineItem]
    budget: BudgetSummary
    progress: ProgressSummary
    hitl_banner: HitlBanner | None = None
    approval_reviews: dict[str, ApprovalReview]
    spec: GoalSpec | None = None
    design: DesignSpec | None = None
    plan: TaskPlan | None = None
    dag: DagView
    swimlanes: list[SwimlaneRow]
    waterfall_turns: list[WaterfallTurn]
    traceability_matrix: list[TraceabilityMatrix]
    delivery_acceptance: DeliveryAcceptance | None = None
    events: list[RunEvent]
    latest_event_seq: int


class RunProjectionResponse(StrictModel):
    ok: Literal[True] = True
    projection: RunProjection


class CommandResponse(StrictModel):
    ok: Literal[True] = True
    command_result: CommandResult
    projection: RunProjection


class ProviderStatus(StrictModel):
    id: str
    dialect: str
    default_model: str
    configured: bool
    env_vars: list[str]


class ProvidersResponse(StrictModel):
    ok: Literal[True] = True
    providers: list[ProviderStatus]


class WorkflowSummary(StrictModel):
    name: str
    description: str
    stop_after: str
    topology: str
    gates: dict[str, bool]
    graph: WorkflowGraph


class WorkflowsResponse(StrictModel):
    ok: Literal[True] = True
    workflows: list[WorkflowSummary]


class DemoPreset(StrictModel):
    id: str
    case_id: str
    label: str
    title: str
    raw_input: str
    constraints: list[str]
    default_workflow: str
    workspace_summary: str
    clarification_hints: list[str]


class DemosResponse(StrictModel):
    ok: Literal[True] = True
    demos: list[DemoPreset]


class ActivityPreviews(StrictModel):
    attempt: int
    preview: str


class ActivityFrame(StrictModel):
    run_id: str
    previews: dict[str, ActivityPreviews]


# Forward references declared before their targets (ConflictResponse,
# AttemptArtifactsResponse) need an explicit rebuild once the module is complete.
ConflictResponse.model_rebuild()
AttemptArtifactsResponse.model_rebuild()


__all__ = [
    "AcceptRiskCommand",
    "ActivityFrame",
    "ActivityPreviews",
    "ApprovalReview",
    "ApproveDesignCommand",
    "ApprovePlanCommand",
    "ApproveSpecCommand",
    "AttemptArtifactsResponse",
    "BudgetSummary",
    "ClarifyCommand",
    "CommandRequest",
    "CommandResponse",
    "CommandResult",
    "ConflictResponse",
    "CreateRunRequest",
    "CreateRunResponse",
    "DagEdge",
    "DagNode",
    "DagView",
    "DeliveryAcceptance",
    "DemoPreset",
    "DemosResponse",
    "ErrorResponse",
    "EventsResponse",
    "HealthResponse",
    "HitlBanner",
    "KeyOutputItem",
    "LiveProgress",
    "PauseCommand",
    "PauseInfo",
    "PromoteToExecuteCommand",
    "ProviderConfig",
    "ProviderStatus",
    "ProvidersResponse",
    "ReassignHumanCommand",
    "ResumeCommand",
    "RetryTaskCommand",
    "RetryWorkflowCommand",
    "RunListResponse",
    "RunProjection",
    "RunProjectionResponse",
    "RunProgressItem",
    "RunSummary",
    "SeedDemoResponse",
    "StageActivityItem",
    "StagePipelineItem",
    "SwimlaneRow",
    "TraceabilityMatrix",
    "WaterfallTurn",
    "WorkflowSummary",
    "WorkflowView",
    "WorkflowsResponse",
]
