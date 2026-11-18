"""Input triage and Stage 1.5 (Design): white-box DesignSpec synthesis and freezing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from olympus.core.contracts import (
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    DesignSpec,
    GoalSpec,
    InputMaturity,
    RunStatus,
    SpecStatus,
    StrictModel,
    TriageResult,
)
from olympus.core.dag import validate_design_spec, validate_goal_spec
from olympus.core.store import RunStore
from olympus.harness.loop import AgentLoop
from olympus.providers.base import LLMProvider
from olympus.stages.common import format_workspace_inventory, run_structured_loop
from olympus.tools.aci import WorkspaceToolkit

# Free-form prose below this length can never carry a frozen contract.
_MIN_STRUCTURED_CHARS = 400

# Vocabulary that signals the author already reasoned about architecture seams.
_SEAM_MARKERS: tuple[str, ...] = (
    "module",
    "interface",
    "endpoint",
    "schema",
    "database",
    "table",
    "api ",
    "模块",
    "接口",
    "表结构",
    "数据库",
    "服务",
)


class DesignValidationError(ValueError):
    """Raised when a DesignSpec fails the zero-token validation gate twice in a row."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


class ExtractionProbe(StrictModel):
    """Permissive extraction envelope so a partial LLM answer never crashes triage."""

    goal: dict[str, Any] | None = None
    design: dict[str, Any] | None = None


_DESIGN_SYSTEM_PROMPT = """You are the Stage 1.5 White-Box Solution Architect (Metis) of
Olympus4SWE. Turn the frozen GoalSpec into a DesignSpec v2 JSON object that prevents
implicit decision drift and module boundary conflicts.
Rules:
1. Global invariants (`shared_invariants`): lock at least 1 cross-cutting rule BEFORE
   module slicing (error envelope format, ID scheme, UTC ISO-8601 time encoding, config,
   validation). Categories: `error_envelope`, `id_format`, `time_encoding`, `concurrency`,
   `naming_convention`, `config`, `validation`, `security`.
2. Shared contract paths (`shared_contract_paths`): declare repo-relative paths/dirs
   for Wave 0 contract code/schemas (owned exclusively by `kind='contract'` tasks in Wave 0,
   strictly read-only for Wave 1+ workers). `shared_contract_paths` MUST NOT overlap any
   module's `target_paths`.
3. Module boundaries: each module owns an exclusive `target_paths` list (repo-relative
   files or directory prefixes). Paths of any two modules MUST NOT overlap.
4. Full coverage: every GoalSpec requirement id appears in at least one module's
   `requirement_refs`; never invent requirement ids.
5. Typed acyclic interface seams: whenever two or more modules exist, declare at least
   one InterfaceContract with provider, consumers, signature, non-empty `input_schema`
   (`SchemaField` list), non-empty `output_schema`, `data_entity_refs`, and non-empty
   `error_cases`. Module dependencies (`consumer -> provider`) MUST be acyclic (no cycles).
6. Data ownership: every DataEntity names the module id that owns it; fields are concrete.
7. Explicit execution strategy: prefer `'serial_handoff'` with 1-2 cohesive modules for small
   or tightly coupled tasks (e.g. <= 6 workspace files or single-package features) so execution
   finishes cleanly in fewer sequential steps; reserve `'contract_first_parallel'` (2-6
   orthogonal modules with shared_contract_paths) for larger multi-subsystem changes where
   parallel sandboxes give real speedup. Always provide a concrete `strategy_rationale`
   (>= 10 chars).
8. Stay white-box & cohesive: name real files/directories, keep module count 1-2 for small
   repos and 2-6 orthogonal modules for larger systems, and set `spec_id` and `spec_hash` to
   the frozen prompt values.
9. Language Mirroring & Greenfield Paths: Match the natural language of the GoalSpec for
   human-facing text (`architecture_summary`, `name`, `responsibility`, `notes`,
   `strategy_rationale`, `risks_and_tradeoffs`); for greenfield products unrelated to the
   host workspace, choose clean standalone repo-relative `target_paths`.
10. Output ONLY a single JSON object conforming to DesignSpec (no $comment/$schema/extra keys)."""


def _parse_structured_payload(raw_input: str | dict[str, Any]) -> dict[str, Any] | None:
    """Deterministically try JSON then YAML before falling back to prose handling."""
    if isinstance(raw_input, dict):
        return raw_input
    text = raw_input.strip()
    if not text:
        return None
    if text.startswith(("{", "[")):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    if ":" in text:
        try:
            parsed_yaml = yaml.safe_load(text)
        except yaml.YAMLError:
            parsed_yaml = None
        if isinstance(parsed_yaml, dict):
            return parsed_yaml
    return None


def _missing_seam_reasons(goal: GoalSpec) -> list[str]:
    reasons = [
        "Input describes WHAT the product must do but not HOW it is carved up: "
        "no module boundaries or exclusive target_paths.",
        "No interface contracts: provider/consumer seams, signatures, and error cases are "
        "undefined.",
        "No data ownership map: entities and their owning module are unspecified.",
    ]
    if goal.deliverable_kind is None:
        reasons.append("deliverable_kind is not frozen, so observable surfaces stay ambiguous.")
    if not goal.observable_surfaces:
        reasons.append("observable_surfaces are not frozen, so acceptance cannot be bound.")
    return reasons


def _classify_payload(payload: dict[str, Any]) -> TriageResult | None:
    design: DesignSpec | None = None
    try:
        design = DesignSpec.model_validate(payload)
    except ValidationError:
        design = None
    if design is not None and not validate_design_spec(design):
        return TriageResult(
            maturity=InputMaturity.DESIGN_SPEC_READY,
            reasons=[
                f"Valid DesignSpec with {len(design.modules)} module(s) and "
                f"{len(design.interfaces)} interface contract(s).",
                "Module target_paths are disjoint and requirement coverage is complete.",
            ],
            extracted_design=design,
        )

    try:
        goal = GoalSpec.model_validate(payload)
    except ValidationError:
        return None

    violations = validate_goal_spec(goal, strict_ears=False, require_out_of_scope=False)
    if violations:
        return None
    return TriageResult(
        maturity=InputMaturity.GOAL_SPEC_READY,
        reasons=_missing_seam_reasons(goal),
        extracted_goal=goal,
    )


def _looks_structured(text: str) -> bool:
    if len(text) < _MIN_STRUCTURED_CHARS:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _SEAM_MARKERS)


def _probe_with_provider(
    text: str,
    *,
    provider: LLMProvider,
    workspace_root: Path,
) -> TriageResult | None:
    """One structured extraction probe; any failure downgrades instead of crashing."""
    toolkit = WorkspaceToolkit(workspace_root, read_only=True)
    loop = AgentLoop(
        provider=provider,
        toolkit=toolkit,
        actor="intake:probe",
        max_steps=2,
        synthesize_on_max_steps=True,
    )
    goal_schema = json.dumps(GoalSpec.model_json_schema(), ensure_ascii=False)
    design_schema = json.dumps(DesignSpec.model_json_schema(), ensure_ascii=False)
    user_prompt = (
        "Decide whether the input already contains a frozen DesignSpec (modules, target "
        "paths, interfaces) or a frozen GoalSpec (requirements with verifiable acceptance "
        "criteria). Return ONLY JSON: {\"design\": <DesignSpec|null>, \"goal\": <GoalSpec|null>}. "
        "Never invent module boundaries that the input does not support; leave both null "
        "when the text is just a raw idea.\n"
        f"DesignSpec schema: {design_schema}\n"
        f"GoalSpec schema: {goal_schema}\n\n"
        f"Input:\n{text}"
    )
    try:
        probe = run_structured_loop(
            loop=loop,
            system_prompt=(
                "You are the Olympus4SWE intake triager. Never trust filenames or UI claims; "
                "judge only the actual content. Prefer returning null over inventing detail."
            ),
            user_prompt=user_prompt,
            model_cls=ExtractionProbe,
        )
    except Exception:  # noqa: BLE001 - any provider/parse failure downgrades to raw intent
        return None

    if probe.design is not None:
        try:
            design = DesignSpec.model_validate(probe.design)
        except ValidationError:
            design = None
        if design is not None and not validate_design_spec(design):
            return TriageResult(
                maturity=InputMaturity.DESIGN_SPEC_READY,
                reasons=[
                    "Provider extraction produced a valid DesignSpec with disjoint module "
                    "paths and at least one seam.",
                ],
                extracted_design=design,
            )

    if probe.goal is not None:
        try:
            goal = GoalSpec.model_validate(probe.goal)
        except ValidationError:
            goal = None
        if goal is not None and not validate_goal_spec(
            goal, strict_ears=False, require_out_of_scope=False
        ):
            return TriageResult(
                maturity=InputMaturity.GOAL_SPEC_READY,
                reasons=[
                    "Provider extraction produced a valid GoalSpec, but the design seams "
                    "below are still missing.",
                    *_missing_seam_reasons(goal),
                ],
                extracted_goal=goal,
            )
    return None


def triage_input(
    raw_input: str | dict[str, Any],
    *,
    provider: LLMProvider | None = None,
    store: RunStore | None = None,
    run_id: str | None = None,
    workspace_root: Path | str | None = None,
) -> TriageResult:
    """Classify an input as raw intent, a frozen GoalSpec, or a frozen DesignSpec.

    Filenames and UI labels are never trusted: only deterministic schema validation of the
    actual content decides maturity.
    """
    result = _classify(raw_input, provider=provider, workspace_root=workspace_root)
    if store is not None and run_id is not None:
        store.append_event(
            run_id=run_id,
            stage="align",
            event_type="intake.triaged",
            actor="intake",
            payload={
                "maturity": result.maturity.value,
                "reasons": result.reasons,
                "has_goal": result.extracted_goal is not None,
                "has_design": result.extracted_design is not None,
            },
        )
    return result


def _classify(
    raw_input: str | dict[str, Any],
    *,
    provider: LLMProvider | None,
    workspace_root: Path | str | None,
) -> TriageResult:
    payload = _parse_structured_payload(raw_input)
    if payload is not None:
        classified = _classify_payload(payload)
        if classified is not None:
            return classified
        return TriageResult(
            maturity=InputMaturity.RAW_INTENT,
            reasons=[
                "Structured payload does not satisfy either frozen contract: it is missing "
                "verifiable acceptance criteria, exclusive module paths, or interface seams.",
            ],
        )

    text = raw_input if isinstance(raw_input, str) else json.dumps(raw_input, ensure_ascii=False)
    if not _looks_structured(text):
        return TriageResult(
            maturity=InputMaturity.RAW_INTENT,
            reasons=[
                "Free-form intent without frozen requirements, module boundaries, or "
                "interface contracts.",
                "Stage 1 Align must synthesize an EARS GoalSpec before any design work.",
            ],
        )

    if provider is not None:
        root = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd()
        probed = _probe_with_provider(text, provider=provider, workspace_root=root)
        if probed is not None:
            return probed
        return TriageResult(
            maturity=InputMaturity.RAW_INTENT,
            reasons=[
                "Input is detailed but the extraction probe could not produce a fully valid "
                "GoalSpec or DesignSpec.",
                "Fall back to Stage 1 Align so blocking ambiguities can be surfaced first.",
            ],
        )

    return TriageResult(
        maturity=InputMaturity.RAW_INTENT,
        reasons=[
            "Detailed prose without a machine-checkable contract; no provider available to "
            "probe for one.",
        ],
    )


class DesignStage:
    """White-box design synthesis bound to a frozen GoalSpec."""

    def __init__(
        self,
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
        approved = self.store.is_approved(
            run_id=run_id,
            target=ApprovalTarget.GOAL_SPEC,
            target_id=spec.spec_id,
            target_hash=spec.content_hash,
        )
        if not approved and spec.status != SpecStatus.CONFIRMED:
            raise ValueError(
                f"GoalSpec {spec.spec_id} ({spec.content_hash}) is not human-approved."
            )
        return spec

    def propose_design(
        self,
        run_id: str,
        auto_approve: bool = False,
        approver: str = "zeus",
        max_steps: int = 10,
    ) -> DesignSpec:
        """Draft a DesignSpec, repair it up to two times, then freeze it for review."""
        spec = self._require_confirmed_spec(run_id)
        self.store.set_run_status(
            run_id,
            RunStatus.DESIGNING,
            actor="design",
            stage="design",
            detail={"spec_id": spec.spec_id, "spec_hash": spec.content_hash},
        )
        toolkit = WorkspaceToolkit(self.workspace_root, read_only=True)
        inventory = format_workspace_inventory(toolkit)
        schema_json = json.dumps(DesignSpec.model_json_schema(), ensure_ascii=False)
        trace_path = self._run_dir(run_id) / "traces" / "designer.jsonl"
        base_prompt = (
            f"Workspace root: {self.workspace_root}\n"
            f"{inventory}\n"
            f"Frozen GoalSpec:\n{spec.model_dump_json(indent=2)}\n\n"
            f"GoalSpec markdown:\n{spec.to_markdown()}\n"
            f"Set spec_id='{spec.spec_id}' and spec_hash='{spec.content_hash}'.\n"
            f"Return DesignSpec JSON matching schema:\n{schema_json}"
        )

        design: DesignSpec | None = None
        last_violations: list[str] = []
        prompt = base_prompt
        for attempt in range(3):
            loop = AgentLoop(
                provider=self.provider,
                toolkit=WorkspaceToolkit(self.workspace_root, read_only=True),
                store=self.store,
                run_id=run_id,
                actor="design:architect",
                max_steps=max_steps,
                trace_path=trace_path,
            )
            candidate = run_structured_loop(
                loop=loop,
                system_prompt=_DESIGN_SYSTEM_PROMPT,
                user_prompt=prompt,
                model_cls=DesignSpec,
            ).model_copy(
                update={
                    "spec_id": spec.spec_id,
                    "spec_hash": spec.content_hash,
                    "status": SpecStatus.AWAITING_HUMAN,
                }
            )
            violations = validate_design_spec(candidate, spec, strict_v2=True)
            if not violations:
                design = candidate
                break
            last_violations = violations
            if attempt < 2:
                prompt = (
                    f"{base_prompt}\n\n"
                    "[DESIGN VALIDATION FAILED]\n"
                    f"Violations: {'; '.join(violations)}\n"
                    "Fix every violation and return a valid DesignSpec JSON object."
                )
        if design is None:
            raise DesignValidationError(last_violations)

        self.store.save_design(run_id, design, actor="design")
        self.store.set_run_status(
            run_id,
            RunStatus.AWAITING_HUMAN,
            actor="design",
            stage="design",
            detail={
                "design_id": design.design_id,
                "design_hash": design.content_hash,
                "module_count": len(design.modules),
            },
        )
        if auto_approve:
            self.approve_design(
                run_id=run_id,
                approver=approver,
                reason="auto_approve flag",
            )
        return design

    def approve_design(
        self,
        run_id: str,
        approver: str = "zeus",
        reason: str = "",
    ) -> ApprovalRecord:
        """Freeze the current DesignSpec and record a hash-bound approval."""
        snap = self.store.get_run(run_id)
        design: DesignSpec | None = snap["design"]
        if design is None:
            raise ValueError(f"Run {run_id} has no DesignSpec to approve.")
        confirmed = design.model_copy(update={"status": SpecStatus.CONFIRMED})
        self.store.save_design(run_id, confirmed, actor=approver)
        record = ApprovalRecord(
            run_id=run_id,
            target=ApprovalTarget.DESIGN_SPEC,
            target_id=confirmed.design_id,
            target_hash=confirmed.content_hash,
            approver=approver,
            decision=ApprovalDecision.APPROVED,
            reason=reason,
        )
        self.store.record_approval(record)
        return record


__all__ = [
    "DesignStage",
    "DesignValidationError",
    "ExtractionProbe",
    "triage_input",
]
