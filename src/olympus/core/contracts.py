"""Strong-typed domain contracts for Stage 1 (Align), Stage 2 (Orchestrate), and audit state."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from olympus.core.fingerprint import content_hash


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FingerprintedModel(StrictModel):
    """Base model whose content_hash stays consistent across validation and model_copy."""

    content_hash: str = ""

    def _hash_payload(self) -> dict[str, Any]:
        raise NotImplementedError

    def _refresh_hash(self) -> Self:
        object.__setattr__(self, "content_hash", content_hash(self._hash_payload()))
        return self

    @model_validator(mode="after")
    def _compute_hash_after_validate(self) -> Self:
        return self._refresh_hash()

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        copied = super().model_copy(update=update, deep=deep)
        return copied._refresh_hash()


class SpecStatus(StrEnum):
    DRAFT = "draft"
    AWAITING_HUMAN = "awaiting_human"
    CONFIRMED = "confirmed"
    SUPERSEDED = "superseded"


class CritiquePerspective(StrEnum):
    PRODUCT = "product"
    ENGINEERING = "engineering"
    QA = "qa"


class CritiqueSeverity(StrEnum):
    BLOCKING = "blocking"
    WARNING = "warning"


class DeliverableKind(StrEnum):
    WEB_APP = "web_app"
    REST_API = "rest_api"
    CLI = "cli"
    LIBRARY = "library"
    DESKTOP = "desktop"
    MULTI_SURFACE = "multi_surface"


class EarsPattern(StrEnum):
    UBIQUITOUS = "ubiquitous"
    EVENT_DRIVEN = "event_driven"
    STATE_DRIVEN = "state_driven"
    UNWANTED_BEHAVIOR = "unwanted_behavior"
    OPTIONAL_FEATURE = "optional_feature"


class AcceptanceCriterion(StrictModel):
    """One EARS-pattern acceptance criterion with a deterministic rendering."""

    id: str = Field(min_length=1)
    pattern: EarsPattern
    condition: str = ""
    outcome: str = Field(min_length=1)

    @model_validator(mode="after")
    def _require_condition_for_conditional_patterns(self) -> AcceptanceCriterion:
        if self.pattern != EarsPattern.UBIQUITOUS and not self.condition.strip():
            raise ValueError(f"EARS pattern '{self.pattern.value}' requires a non-empty condition")
        return self

    def to_ears_text(self) -> str:
        outcome = self.outcome.strip()
        condition = self.condition.strip()
        if self.pattern == EarsPattern.UBIQUITOUS:
            return f"THE SYSTEM SHALL {outcome}"
        if self.pattern == EarsPattern.EVENT_DRIVEN:
            return f"WHEN {condition} THE SYSTEM SHALL {outcome}"
        if self.pattern == EarsPattern.STATE_DRIVEN:
            return f"WHILE {condition} THE SYSTEM SHALL {outcome}"
        if self.pattern == EarsPattern.UNWANTED_BEHAVIOR:
            return f"IF {condition} THEN THE SYSTEM SHALL {outcome}"
        return f"WHERE {condition} THE SYSTEM SHALL {outcome}"


class RequirementItem(StrictModel):
    id: str
    description: str
    # Always a list[str] so existing scorers/projection keep iterating plain strings.
    acceptance_criteria: list[str] = Field(default_factory=list)
    source_ref: str | None = None
    actor: str | None = None
    ears_criteria: list[AcceptanceCriterion] = Field(default_factory=list)

    @model_validator(mode="after")
    def _materialize_ears_text(self) -> RequirementItem:
        if self.ears_criteria and not self.acceptance_criteria:
            self.acceptance_criteria = [c.to_ears_text() for c in self.ears_criteria]
        if not self.acceptance_criteria:
            raise ValueError(
                f"RequirementItem {self.id} requires at least one acceptance criterion "
                "(acceptance_criteria or ears_criteria)"
            )
        return self


class AmbiguityItem(StrictModel):
    id: str
    question: str
    impact: str
    options: list[str] = Field(default_factory=list)
    default_assumption: str | None = None
    blocking: bool = True
    answer: str | None = None


class CritiqueNote(StrictModel):
    id: str
    perspective: CritiquePerspective
    severity: CritiqueSeverity
    summary: str
    evidence_refs: list[str] = Field(default_factory=list)
    resolved: bool = False
    resolution_note: str | None = None


class GoalSpec(FingerprintedModel):
    spec_id: str = Field(default_factory=lambda: new_id("spec"))
    version: int = Field(default=1, ge=1)
    goal: str = Field(min_length=1)
    requirements: list[RequirementItem] = Field(min_length=1)
    out_of_scope: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    open_ambiguities: list[AmbiguityItem] = Field(default_factory=list)
    minority_notes: list[CritiqueNote] = Field(default_factory=list)
    status: SpecStatus = SpecStatus.DRAFT
    deliverable_kind: DeliverableKind | None = None
    observable_surfaces: list[str] = Field(default_factory=list)
    actors: list[str] = Field(default_factory=list)

    def _hash_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude={"status", "content_hash"}, mode="python")

    def to_markdown(self) -> str:
        kind = self.deliverable_kind.value if self.deliverable_kind else "(unspecified)"
        lines = [
            f"# GoalSpec {self.spec_id} (v{self.version})",
            "",
            f"- Status: {self.status.value}",
            f"- Deliverable: {kind}",
            f"- Observable surfaces: {', '.join(self.observable_surfaces) or '(none)'}",
            f"- Actors: {', '.join(self.actors) or '(none)'}",
            "",
            "## Goal",
            "",
            self.goal,
            "",
            "## Requirements",
            "",
        ]
        for req in self.requirements:
            actor = f" [{req.actor}]" if req.actor else ""
            lines.append(f"### {req.id}{actor}")
            lines.append("")
            lines.append(req.description)
            lines.append("")
            for idx, criterion in enumerate(req.acceptance_criteria, start=1):
                lines.append(f"{idx}. {criterion}")
            lines.append("")
        lines.extend(["## Out of scope", ""])
        lines.extend(f"- {item}" for item in self.out_of_scope)
        if not self.out_of_scope:
            lines.append("- (none declared)")
        lines.extend(["", "## Constraints", ""])
        lines.extend(f"- {item}" for item in self.constraints)
        if not self.constraints:
            lines.append("- (none)")
        if self.assumptions:
            lines.extend(["", "## Assumptions", ""])
            lines.extend(f"- {item}" for item in self.assumptions)
        return "\n".join(lines).rstrip() + "\n"

    def unresolved_blockers(self) -> list[str]:
        blockers: list[str] = []
        for amb in self.open_ambiguities:
            if amb.blocking and not amb.answer:
                blockers.append(f"ambiguity:{amb.id}")
        for note in self.minority_notes:
            if note.severity == CritiqueSeverity.BLOCKING and not note.resolved:
                blockers.append(f"critique:{note.id}")
        return blockers


class GlobalInvariant(StrictModel):
    """Cross-cutting rule locked before parallel or sequential module implementation."""

    id: str = Field(min_length=1)
    category: Literal[
        "error_envelope",
        "id_format",
        "time_encoding",
        "concurrency",
        "naming_convention",
        "config",
        "validation",
        "security",
    ]
    rule: str = Field(min_length=1)
    enforcement_hint: str = ""
    applies_to: list[str] = Field(default_factory=list)


class SchemaField(StrictModel):
    """Typed parameter or payload field on a module interface seam."""

    name: str = Field(min_length=1)
    type_expr: str = Field(min_length=1)
    required: bool = True
    constraints: str = ""


class ExecutionStrategy(StrEnum):
    """Explicit concurrency vs. sequential-handoff verdict for Stage 2 execution."""

    SERIAL_HANDOFF = "serial_handoff"
    CONTRACT_FIRST_PARALLEL = "contract_first_parallel"


class DesignModule(StrictModel):
    """One white-box module boundary with an exclusive target path scope."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    responsibility: str = Field(min_length=1)
    requirement_refs: list[str] = Field(min_length=1)
    target_paths: list[str] = Field(min_length=1)


class DataEntity(StrictModel):
    name: str = Field(min_length=1)
    fields: list[str] = Field(min_length=1)
    owner_module_id: str = Field(min_length=1)
    notes: str = ""


class InterfaceContract(StrictModel):
    """A named seam between a provider module and its consumers."""

    id: str = Field(min_length=1)
    provider_module_id: str = Field(min_length=1)
    consumer_module_ids: list[str] = Field(default_factory=list)
    protocol: str = "http_or_fn"
    signature: str = Field(min_length=1)
    input_schema: list[SchemaField] = Field(default_factory=list)
    output_schema: list[SchemaField] = Field(default_factory=list)
    data_entity_refs: list[str] = Field(default_factory=list)
    error_cases: list[str] = Field(default_factory=list)


class DesignSpec(FingerprintedModel):
    """White-box design: invariants, contract seams, modules, data ownership, and strategy."""

    design_id: str = Field(default_factory=lambda: new_id("design"))
    version: int = Field(default=1, ge=1)
    spec_id: str | None = None
    spec_hash: str | None = None
    architecture_summary: str = Field(min_length=1)
    shared_invariants: list[GlobalInvariant] = Field(default_factory=list)
    shared_contract_paths: list[str] = Field(default_factory=list)
    modules: list[DesignModule] = Field(min_length=1)
    data_models: list[DataEntity] = Field(default_factory=list)
    interfaces: list[InterfaceContract] = Field(default_factory=list)
    execution_strategy: ExecutionStrategy | None = None
    strategy_rationale: str = ""
    risks_and_tradeoffs: list[str] = Field(default_factory=list)
    status: SpecStatus = SpecStatus.DRAFT

    def _hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(exclude={"status", "content_hash"}, mode="python")
        # Preserve exact v1 content_hash when v2-only fields are empty defaults.
        if not payload.get("shared_invariants"):
            payload.pop("shared_invariants", None)
        if not payload.get("shared_contract_paths"):
            payload.pop("shared_contract_paths", None)
        if payload.get("execution_strategy") is None:
            payload.pop("execution_strategy", None)
        elif isinstance(self.execution_strategy, ExecutionStrategy):
            payload["execution_strategy"] = self.execution_strategy.value
        if not payload.get("strategy_rationale"):
            payload.pop("strategy_rationale", None)
        for iface in payload.get("interfaces", []):
            if not iface.get("input_schema"):
                iface.pop("input_schema", None)
            if not iface.get("output_schema"):
                iface.pop("output_schema", None)
            if not iface.get("data_entity_refs"):
                iface.pop("data_entity_refs", None)
        return payload

    def to_markdown(self) -> str:
        lines = [
            f"# DesignSpec {self.design_id} (v{self.version})",
            "",
            f"- Status: {self.status.value}",
            f"- Bound spec: {self.spec_id or '(none)'}",
            f"- Spec hash: {self.spec_hash or '(none)'}",
        ]
        if self.execution_strategy is not None:
            lines.append(f"- Execution strategy: {self.execution_strategy.value}")
        if self.strategy_rationale:
            lines.append(f"- Strategy rationale: {self.strategy_rationale}")
        if self.shared_contract_paths:
            lines.append(
                f"- Shared contract paths (W0 write / W1 read-only): "
                f"{', '.join(self.shared_contract_paths)}"
            )
        lines.extend(
            [
                "",
                "## Architecture",
                "",
                self.architecture_summary,
                "",
            ]
        )
        if self.shared_invariants:
            lines.extend(["## Global invariants", ""])
            for inv in self.shared_invariants:
                scope = ", ".join(inv.applies_to) if inv.applies_to else "global"
                hint = f" — hint: {inv.enforcement_hint}" if inv.enforcement_hint else ""
                lines.append(f"- {inv.id} [{inv.category}] ({scope}): {inv.rule}{hint}")
            lines.append("")
        lines.extend(["## Modules", ""])
        for module in self.modules:
            lines.extend(
                [
                    f"### {module.id} — {module.name}",
                    "",
                    module.responsibility,
                    "",
                    f"- Requirements: {', '.join(module.requirement_refs)}",
                    f"- Target paths: {', '.join(module.target_paths)}",
                    "",
                ]
            )
        if self.data_models:
            lines.extend(["## Data models", ""])
            for entity in self.data_models:
                lines.append(
                    f"- {entity.name} (owner: {entity.owner_module_id}): "
                    f"{', '.join(entity.fields)}"
                )
            lines.append("")
        if self.interfaces:
            lines.extend(["## Interfaces", ""])
            for iface in self.interfaces:
                consumers = ", ".join(iface.consumer_module_ids) or "(external)"
                lines.append(
                    f"- {iface.id} [{iface.protocol}] {iface.signature} "
                    f"({iface.provider_module_id} -> {consumers})"
                )
                if iface.input_schema:
                    inputs = ", ".join(
                        f"{f.name}: {f.type_expr}{'?' if not f.required else ''}"
                        for f in iface.input_schema
                    )
                    lines.append(f"  - Input: {inputs}")
                if iface.output_schema:
                    outputs = ", ".join(
                        f"{f.name}: {f.type_expr}{'?' if not f.required else ''}"
                        for f in iface.output_schema
                    )
                    lines.append(f"  - Output: {outputs}")
                if iface.data_entity_refs:
                    lines.append(f"  - Data entities: {', '.join(iface.data_entity_refs)}")
                if iface.error_cases:
                    lines.append(f"  - Errors: {', '.join(iface.error_cases)}")
            lines.append("")
        if self.risks_and_tradeoffs:
            lines.extend(["## Risks and tradeoffs", ""])
            lines.extend(f"- {item}" for item in self.risks_and_tradeoffs)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


class InputMaturity(StrEnum):
    RAW_INTENT = "raw_intent"
    GOAL_SPEC_READY = "goal_spec_ready"
    DESIGN_SPEC_READY = "design_spec_ready"


class TriageResult(StrictModel):
    maturity: InputMaturity
    reasons: list[str] = Field(default_factory=list)
    extracted_goal: GoalSpec | None = None
    extracted_design: DesignSpec | None = None


class TaskStatus(StrEnum):
    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    VERIFYING = "verifying"
    ACCEPTED = "accepted"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    STALE = "stale"


class CheckKind(StrEnum):
    COMMAND = "command"
    INDEPENDENT_REVIEW = "independent_review"
    HUMAN = "human"


class AcceptanceCheck(StrictModel):
    id: str
    kind: CheckKind
    description: str = Field(min_length=1)
    command: str | None = None
    passed: bool = False
    evidence_ref: str | None = None

    @model_validator(mode="after")
    def _validate_command_check(self) -> AcceptanceCheck:
        if self.kind == CheckKind.COMMAND and not self.command:
            raise ValueError("command check requires non-empty command")
        return self


class ReviewSeverity(StrEnum):
    IMPORTANT = "important"
    NIT = "nit"
    PRE_EXISTING = "pre_existing"


class ReviewFinding(StrictModel):
    id: str
    severity: ReviewSeverity
    file_path: str
    line: int | None = Field(default=None, ge=1)
    message: str = Field(min_length=1)
    suggestion: str | None = None


class ReviewVerdict(StrictModel):
    verdict_id: str = Field(default_factory=lambda: new_id("verdict"))
    task_id: str
    attempt: int = Field(ge=1)
    passed: bool
    summary: str = Field(min_length=1)
    findings: list[ReviewFinding] = Field(default_factory=list)
    checked_commit: str | None = None
    incremental_only: bool = False

    @model_validator(mode="after")
    def _enforce_important_blocker(self) -> ReviewVerdict:
        has_important = any(f.severity == ReviewSeverity.IMPORTANT for f in self.findings)
        if self.passed and has_important:
            raise ValueError("ReviewVerdict cannot pass while Important findings remain")
        return self


class TaskKind(StrEnum):
    CONTRACT = "contract"
    IMPLEMENT = "implement"
    TEST = "test"
    INTEGRATE = "integrate"


class TaskNode(StrictModel):
    task_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    requirement_refs: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    write_scope: list[str] = Field(min_length=1)
    read_scope: list[str] = Field(default_factory=list)
    acceptance_checks: list[AcceptanceCheck] = Field(min_length=1)
    max_cycles: int = Field(default=3, ge=1, le=5)
    status: TaskStatus = TaskStatus.WAITING
    attempt: int = Field(default=0, ge=0)
    review_cycle: int = Field(default=0, ge=0)
    lease_generation: int = Field(default=0, ge=0)
    worktree_path: str | None = None
    commit_sha: str | None = None
    last_error: str | None = None
    accepted_risk_reason: str | None = None
    design_refs: list[str] = Field(default_factory=list)
    role: str = "engineer"
    kind: TaskKind = TaskKind.IMPLEMENT
    deliverables: list[str] = Field(default_factory=list)


class TaskPlan(FingerprintedModel):
    plan_id: str = Field(default_factory=lambda: new_id("plan"))
    spec_id: str | None = None
    spec_hash: str | None = None
    design_id: str | None = None
    design_hash: str | None = None
    version: int = Field(default=1, ge=1)
    tasks: list[TaskNode] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_upstream_binding(self) -> TaskPlan:
        spec_bound = bool(self.spec_id and self.spec_hash)
        design_bound = bool(self.design_id and self.design_hash)
        if not (spec_bound or design_bound):
            raise ValueError(
                "TaskPlan must bind to an approved GoalSpec (spec_id+spec_hash) "
                "or an approved DesignSpec (design_id+design_hash)"
            )
        return self

    def _hash_payload(self) -> dict[str, Any]:
        # Hash structural plan definition (not mutable runtime execution counters)
        structural_tasks = [
            t.model_dump(
                include={
                    "task_id",
                    "title",
                    "objective",
                    "requirement_refs",
                    "depends_on",
                    "write_scope",
                    "read_scope",
                    "max_cycles",
                    "design_refs",
                    "role",
                    "kind",
                    "deliverables",
                },
                mode="python",
            )
            | {
                "acceptance_checks": [
                    c.model_dump(include={"id", "kind", "description", "command"}, mode="python")
                    for c in t.acceptance_checks
                ]
            }
            for t in self.tasks
        ]
        return {
            "plan_id": self.plan_id,
            "spec_id": self.spec_id,
            "spec_hash": self.spec_hash,
            "design_id": self.design_id,
            "design_hash": self.design_hash,
            "version": self.version,
            "tasks": structural_tasks,
        }


class RunMode(StrEnum):
    SIMULATION = "simulation"
    EXECUTION = "execution"


class RunStatus(StrEnum):
    CREATED = "created"
    ALIGNING = "aligning"
    DESIGNING = "designing"
    PLANNING = "planning"
    AWAITING_HUMAN = "awaiting_human"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BudgetPolicy(StrictModel):
    max_llm_calls: int = Field(default=100, ge=1)
    max_total_tokens: int = Field(default=2_000_000, ge=1)
    max_concurrent_tasks: int = Field(default=3, ge=1)


class BudgetLedger(StrictModel):
    llm_calls_used: int = Field(default=0, ge=0)
    llm_calls_reserved: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    unknown_outcome_calls: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def can_reserve(self, policy: BudgetPolicy) -> bool:
        return (
            self.llm_calls_used + self.llm_calls_reserved < policy.max_llm_calls
            and self.total_tokens < policy.max_total_tokens
        )

    def reserve(self, policy: BudgetPolicy) -> BudgetLedger:
        if not self.can_reserve(policy):
            raise ValueError("Budget exhausted: cannot reserve additional LLM call")
        return self.model_copy(update={"llm_calls_reserved": self.llm_calls_reserved + 1})

    def settle(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        *,
        unknown: bool = False,
    ) -> BudgetLedger:
        if self.llm_calls_reserved <= 0:
            raise ValueError("No reserved LLM call to settle")
        return self.model_copy(
            update={
                "llm_calls_reserved": self.llm_calls_reserved - 1,
                "llm_calls_used": self.llm_calls_used + 1,
                "input_tokens": self.input_tokens + max(0, input_tokens),
                "output_tokens": self.output_tokens + max(0, output_tokens),
                "unknown_outcome_calls": self.unknown_outcome_calls + (1 if unknown else 0),
            }
        )


class ApprovalTarget(StrEnum):
    GOAL_SPEC = "goal_spec"
    DESIGN_SPEC = "design_spec"
    TASK_PLAN = "task_plan"
    HIGH_RISK_ACTION = "high_risk_action"
    FINAL_ACCEPTANCE = "final_acceptance"
    RISK_ACCEPTANCE = "risk_acceptance"


class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    ACCEPT_RISK = "accept_risk"


class ApprovalRecord(StrictModel):
    approval_id: str = Field(default_factory=lambda: new_id("appr"))
    run_id: str
    target: ApprovalTarget
    target_id: str
    target_hash: str
    commit_sha: str | None = None
    approver: str = Field(min_length=1)
    decision: ApprovalDecision
    reason: str = ""
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _validate_risk_reason(self) -> ApprovalRecord:
        if self.decision == ApprovalDecision.ACCEPT_RISK and not self.reason.strip():
            raise ValueError("ACCEPT_RISK requires an explicit non-empty reason")
        return self


class RunEvent(StrictModel):
    event_seq: int = Field(ge=1)
    event_id: str = Field(default_factory=lambda: new_id("evt"))
    run_id: str
    stage: str
    event_type: str
    actor: str
    payload: dict[str, Any] = Field(default_factory=dict)
    state_revision: int = Field(ge=0)
    created_at: datetime = Field(default_factory=utc_now)
