"""Input triage and Stage 1.5 (Design): white-box DesignSpec synthesis and freezing."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from olympus.core.contracts import (
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
from olympus.core.control import RunControl
from olympus.core.dag import validate_design_spec, validate_goal_spec
from olympus.core.store import RunStore
from olympus.harness.driver import DriverRegistry, ExecutionDriversConfig
from olympus.harness.loop import AgentLoop
from olympus.harness.structured import run_structured_turn
from olympus.providers.base import LLMProvider
from olympus.stages.common import (
    default_artifacts_root,
    format_workspace_inventory,
    resolve_stage_driver,
)
from olympus.tools.aci import WorkspaceToolkit

# Free-form prose below this length can never carry a frozen contract.
_MIN_STRUCTURED_CHARS = 400


class DesignVariant(StrEnum):
    """Independent lenses a candidate DesignSpec is drafted from.

    Each variant runs its own cold-start AgentLoop and never sees the other drafts; the merge
    happens afterwards in a separate synthesis turn so the trade-off stays auditable.
    """

    PRODUCT = "product_led"
    ENGINEERING = "engineering_led"
    RISK = "risk_led"


DEFAULT_DESIGN_VARIANTS: tuple[DesignVariant, ...] = (
    DesignVariant.PRODUCT,
    DesignVariant.ENGINEERING,
    DesignVariant.RISK,
)

# Per-variant mandates appended to the shared architect prompt. They deliberately pull in
# different directions so the panel produces genuinely different seams instead of three
# paraphrases of one design.
_VARIANT_BRIEFS: dict[DesignVariant, str] = {
    DesignVariant.PRODUCT: (
        "MANDATE (product-led): optimise for the observable surfaces in the GoalSpec. Keep the "
        "smallest module set that still delivers every acceptance criterion, make behaviour at "
        "the seam explicit, and prefer deferring anything out_of_scope over speculative "
        "generality. State which requirement each module exists for."
    ),
    DesignVariant.ENGINEERING: (
        "MANDATE (engineering-led): optimise for deep modules with narrow interfaces. Put "
        "shared invariants and error envelopes in one place, make each interface hard to misuse, "
        "and justify every module boundary by information hiding rather than by file layout. "
        "Call out seams that would force callers to change when internals change."
    ),
    DesignVariant.RISK: (
        "MANDATE (risk-led): optimise for failure containment. Identify the concurrency, "
        "ordering, and partial-write hazards implied by the GoalSpec, give each a module or "
        "invariant that owns it, and refuse any design whose failure mode is a silent wrong "
        "write. Prefer an extra explicit module over an implicit assumption."
    ),
}

_ARBITER_SYSTEM_PROMPT = """You are the Stage 1.5 Design Arbiter (Momus-of-design) of
Olympus4SWE. You receive several independently drafted DesignSpec candidates for the SAME
frozen GoalSpec, each with the verdict of the zero-token structural gate. Produce the single
DesignSpec that will be frozen.

Rules:
1. Never invent a module, interface, or data model that no candidate proposed unless two
   candidates disagree and the GoalSpec forces a choice.
2. Prefer the candidate whose seams survive the gate with zero violations; when several do,
   keep the module set that is smallest while still covering every requirement.
3. When candidates conflict, resolve the conflict and record why in `strategy_rationale`.
4. Preserve every shared invariant that any candidate declared; dropping one is a regression.
5. Set spec_id and spec_hash to the values given in the prompt, and return one DesignSpec JSON
   object matching the schema."""


@dataclass(frozen=True)
class DesignCandidate:
    """One variant's draft plus the gate verdict that decided whether it can be used."""

    variant: str
    design: DesignSpec | None = None
    violations: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.design is not None and not self.violations

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
2. Deep cohesive modules: default to 1-3 modules that hide substantial behavior behind a
   narrow interface. Keep a single-module design when the product is small and cohesive;
   never split into speculative horizontal layers (routes/schemas/services/wrappers).
3. The Deletion Test: do not emit single-caller pass-through wrappers (`ports`, `adapters`,
   `dto`, `facade`) or any layer whose deletion would merely inline one call site. A
   wrapper earns its existence only with >= 2 real callers/adapters or a genuine external
   boundary; otherwise inline it into its caller.
4. Real vs. hypothetical seams: emit `shared_contract_paths` and set
   `execution_strategy='contract_first_parallel'` ONLY when multiple independent modules
   genuinely build in parallel against frozen Wave 0 contracts. For a single deep module
   or a tightly coupled vertical core, choose `serial_handoff` and leave
   `shared_contract_paths` empty; never pay for a speculative Wave 0 contract layer.
   `shared_contract_paths` MUST NOT overlap any module's `target_paths`.
5. Module boundaries: each module owns an exclusive `target_paths` list (repo-relative
   files or directory prefixes). Paths of any two modules MUST NOT overlap.
6. Full coverage: every GoalSpec requirement id appears in at least one module's
   `requirement_refs`; never invent requirement ids.
7. Typed acyclic interface seams: whenever two or more modules exist, declare at least
   one InterfaceContract with provider, consumers, signature, non-empty `input_schema`
   (`SchemaField` list), non-empty `output_schema`, `data_entity_refs`, and non-empty
   `error_cases`. Align each interface directly with the GoalSpec `test_seams` (falling
   back to `observable_surfaces` when none are declared): expose the observable seam the
   product is verified through instead of internal plumbing. Module dependencies
   (`consumer -> provider`) MUST be acyclic (no cycles).
8. Data ownership: every DataEntity names the module id that owns it; fields are concrete.
9. Explicit execution strategy: set `execution_strategy` to `'contract_first_parallel'`
   or `'serial_handoff'` per rule 4, plus a concrete `strategy_rationale` (>= 10 chars).
10. Stay white-box & cohesive: name real files/directories, make each module deep rather
    than wide, and set `spec_id` and `spec_hash` to the frozen prompt values.
11. Language Mirroring & Greenfield Paths: Match the natural language of the GoalSpec for
    human-facing text (`architecture_summary`, `name`, `responsibility`, `notes`,
    `strategy_rationale`, `risks_and_tradeoffs`); for greenfield products unrelated to the
    host workspace, choose clean standalone repo-relative `target_paths`.
12. Human design review: write `architecture_summary` in one or two sentences. In
    `design_tradeoffs`, record each consequential decision affecting the existing system,
    deployment or rollback. Use impact `existing_system`, `deployment` or `rollback`, and
    state both the selected `choice` and its concrete `cost` (what gets harder or more
    expensive, or which capability is reduced). Match the GoalSpec language. Base entries on
    the proposed design and known constraints. Use [] if no such decision applies.
13. Honor every resolved `open_ambiguities` answer and the declared `deliverable_kind` /
    `observable_surfaces`: when `deliverable_kind='web_app'`, at least one module MUST own a
    user-facing web UI surface in its `target_paths` (served entrypoint, page, or static
    asset); never design backend-only modules for a `web_app`.
14. Output ONLY a single JSON object conforming to DesignSpec (no $comment/$schema/extra keys)."""


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
        "when the text is just a raw idea. Preserve design_tradeoffs only when the input "
        "explicitly supports both the selected choice and its cost.\n"
        f"DesignSpec schema: {design_schema}\n"
        f"GoalSpec schema: {goal_schema}\n\n"
        f"Input:\n{text}"
    )
    try:
        probe = loop.run_structured(
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
    workspace_mode: str | None = None,
    task_type: str | None = None,
    base_ref: str | None = None,
    scope_paths: list[str] | None = None,
) -> TriageResult:
    """Classify an input as raw intent, a frozen GoalSpec, or a frozen DesignSpec.

    Filenames and UI labels are never trusted: only deterministic schema validation of the
    actual content decides maturity. Explicit scope bindings are accepted so intake and
    downstream stages receive identical deterministic context.
    """
    del workspace_mode, task_type, base_ref, scope_paths
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
        *,
        execution_drivers: ExecutionDriversConfig | None = None,
        driver_registry: DriverRegistry | None = None,
    ) -> None:
        self.store = store
        self.provider = provider
        self.workspace_root = Path(workspace_root).resolve()
        self.execution_drivers = execution_drivers or ExecutionDriversConfig.resolve()
        self.driver_registry = driver_registry
        self.artifacts_root = (
            artifacts_root
            if artifacts_root is not None
            else default_artifacts_root(store, self.workspace_root)
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
        max_steps: int = 16,
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
        driver, driver_model = resolve_stage_driver(
            settings=self.execution_drivers,
            registry=self.driver_registry,
            provider=self.provider,
            store=self.store,
            run_id=run_id,
        )
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

        loop = AgentLoop(
            provider=self.provider,
            toolkit=toolkit,
            store=self.store,
            run_id=run_id,
            actor="design:architect",
            max_steps=max_steps,
            trace_path=trace_path,
        )

        def stamp_binding(candidate: DesignSpec) -> DesignSpec:
            return candidate.model_copy(
                update={
                    "spec_id": spec.spec_id,
                    "spec_hash": spec.content_hash,
                    "status": SpecStatus.AWAITING_HUMAN,
                }
            )

        def design_gate(candidate: DesignSpec) -> list[str]:
            return validate_design_spec(candidate, spec, strict_v2=True)

        def raise_exhausted(candidate: DesignSpec, violations: list[str]) -> DesignSpec:
            raise DesignValidationError(violations)

        design = run_structured_turn(
            loop=loop,
            driver=driver,
            driver_model=driver_model,
            store=self.store,
            run_id=run_id,
            workspace_root=self.workspace_root,
            actor="design:architect",
            system_prompt=_DESIGN_SYSTEM_PROMPT,
            user_prompt=base_prompt,
            model_cls=DesignSpec,
            trace_path=trace_path,
            max_steps=max_steps,
            max_validation_retries=2,
            validation_header="[DESIGN VALIDATION FAILED]",
            validation_instruction=(
                "Fix every violation and return a valid DesignSpec JSON object."
            ),
            transform=stamp_binding,
            validator=design_gate,
            on_validation_exhausted=raise_exhausted,
        )

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

    def propose_design_panel(
        self,
        run_id: str,
        *,
        variants: Sequence[DesignVariant | str] | None = None,
        synthesize: bool = True,
        auto_approve: bool = False,
        approver: str = "zeus",
        max_steps: int = 16,
    ) -> DesignSpec:
        """Draft one DesignSpec per variant in parallel, archive every draft, then freeze one.

        Unlike :meth:`propose_design`, each variant runs its own cold-start ``AgentLoop`` with
        its own trace and never observes its peers. Every draft (usable or not) is written to
        ``design_candidates/`` and recorded as an event, then a separate synthesis turn picks or
        merges the winner. The deliberation stays auditable while the control plane still
        freezes exactly one DesignSpec.
        """
        spec = self._require_confirmed_spec(run_id)
        chosen = _normalize_variants(variants)
        self.store.set_run_status(
            run_id,
            RunStatus.DESIGNING,
            actor="design",
            stage="design",
            detail={"panel_variants": [variant.value for variant in chosen]},
        )
        toolkit = WorkspaceToolkit(self.workspace_root, read_only=True)
        inventory = format_workspace_inventory(toolkit)
        schema_json = json.dumps(DesignSpec.model_json_schema(), ensure_ascii=False)
        run_dir = self._run_dir(run_id)

        def stamp_binding(candidate: DesignSpec) -> DesignSpec:
            return candidate.model_copy(
                update={
                    "spec_id": spec.spec_id,
                    "spec_hash": spec.content_hash,
                    "status": SpecStatus.AWAITING_HUMAN,
                }
            )

        def design_gate(candidate: DesignSpec) -> list[str]:
            return validate_design_spec(candidate, spec, strict_v2=True)

        def draft(variant: DesignVariant) -> DesignCandidate:
            driver, driver_model = resolve_stage_driver(
                settings=self.execution_drivers,
                registry=self.driver_registry,
                provider=self.provider,
                store=self.store,
                run_id=run_id,
            )
            brief = _VARIANT_BRIEFS[variant]
            trace_path = run_dir / "traces" / f"designer-{variant.value}.jsonl"
            actor = f"design:{variant.value}"
            loop = AgentLoop(
                provider=self.provider,
                toolkit=toolkit,
                store=self.store,
                run_id=run_id,
                actor=actor,
                max_steps=max_steps,
                trace_path=trace_path,
            )
            prompt = (
                f"Workspace root: {self.workspace_root}\n"
                f"{inventory}\n"
                f"Frozen GoalSpec:\n{spec.model_dump_json(indent=2)}\n\n"
                f"GoalSpec markdown:\n{spec.to_markdown()}\n"
                f"Set spec_id='{spec.spec_id}' and spec_hash='{spec.content_hash}'.\n"
                f"Return DesignSpec JSON matching schema:\n{schema_json}\n\n{brief}"
            )
            try:
                design = run_structured_turn(
                    loop=loop,
                    driver=driver,
                    driver_model=driver_model,
                    store=self.store,
                    run_id=run_id,
                    workspace_root=self.workspace_root,
                    actor=actor,
                    system_prompt=f"{_DESIGN_SYSTEM_PROMPT}\n\n{brief}",
                    user_prompt=prompt,
                    model_cls=DesignSpec,
                    trace_path=trace_path,
                    max_steps=max_steps,
                    max_validation_retries=2,
                    validation_header="[DESIGN VALIDATION FAILED]",
                    validation_instruction=(
                        "Fix every violation and return a valid DesignSpec JSON object."
                    ),
                    transform=stamp_binding,
                    validator=design_gate,
                )
            except Exception as exc:  # noqa: BLE001 - one bad draft must not kill the panel
                return DesignCandidate(variant=variant.value, error=str(exc))
            return DesignCandidate(
                variant=variant.value,
                design=design,
                violations=design_gate(design),
            )

        with ThreadPoolExecutor(max_workers=len(chosen)) as pool:
            candidates = list(pool.map(draft, chosen))

        archived = _archive_candidates(run_dir, candidates)
        self.store.append_event(
            run_id=run_id,
            stage="design",
            event_type="design.candidates_produced",
            actor="design:panel",
            payload={
                "variants": [candidate.variant for candidate in candidates],
                "usable": [candidate.variant for candidate in candidates if candidate.usable],
                "violations": {
                    candidate.variant: candidate.violations
                    for candidate in candidates
                    if candidate.violations
                },
                "errors": {
                    candidate.variant: candidate.error
                    for candidate in candidates
                    if candidate.error
                },
                "archived": archived,
            },
        )

        usable = [candidate for candidate in candidates if candidate.usable]
        if not usable:
            raise DesignValidationError(
                ["no usable design candidate from the panel", *_panel_failure_summary(candidates)]
            )

        if synthesize and len(usable) > 1:
            design = self._synthesize_design(
                run_id=run_id,
                spec=spec,
                candidates=usable,
                inventory=inventory,
                schema_json=schema_json,
                run_dir=run_dir,
                max_steps=max_steps,
                transform=stamp_binding,
                validator=design_gate,
            )
        else:
            only = usable[0].design
            if only is None:  # pragma: no cover - `usable` already guarantees a design
                raise DesignValidationError(["design panel produced no design"])
            design = only

        self.store.save_design(run_id, design, actor="design:panel")
        self.store.set_run_status(
            run_id,
            RunStatus.AWAITING_HUMAN,
            actor="design",
            stage="design",
            detail={
                "design_id": design.design_id,
                "design_hash": design.content_hash,
                "module_count": len(design.modules),
                "panel_variants": [candidate.variant for candidate in candidates],
                "panel_synthesized": bool(synthesize and len(usable) > 1),
            },
        )
        if auto_approve:
            self.approve_design(run_id=run_id, approver=approver, reason="auto_approve flag")
        return design

    def _synthesize_design(
        self,
        *,
        run_id: str,
        spec: GoalSpec,
        candidates: Sequence[DesignCandidate],
        inventory: str,
        schema_json: str,
        run_dir: Path,
        max_steps: int,
        transform: Callable[[DesignSpec], DesignSpec],
        validator: Callable[[DesignSpec], list[str]],
    ) -> DesignSpec:
        """Fold the usable drafts into one DesignSpec with a cold-start arbiter turn."""
        driver, driver_model = resolve_stage_driver(
            settings=self.execution_drivers,
            registry=self.driver_registry,
            provider=self.provider,
            store=self.store,
            run_id=run_id,
        )
        trace_path = run_dir / "traces" / "design-arbiter.jsonl"
        loop = AgentLoop(
            provider=self.provider,
            toolkit=WorkspaceToolkit(self.workspace_root, read_only=True),
            store=self.store,
            run_id=run_id,
            actor="design:arbiter",
            max_steps=max_steps,
            trace_path=trace_path,
        )
        blocks: list[str] = []
        for candidate in candidates:
            payload = candidate.design.model_dump_json(indent=2) if candidate.design else "{}"
            blocks.append(f"### Candidate {candidate.variant}\ngate: clean\n{payload}")
        prompt = (
            f"Workspace root: {self.workspace_root}\n"
            f"{inventory}\n"
            f"Frozen GoalSpec:\n{spec.model_dump_json(indent=2)}\n\n"
            f"GoalSpec markdown:\n{spec.to_markdown()}\n"
            f"Set spec_id='{spec.spec_id}' and spec_hash='{spec.content_hash}'.\n"
            f"Return DesignSpec JSON matching schema:\n{schema_json}\n\n"
            f"{len(candidates)} independent candidates follow. Produce the single design to "
            f"freeze.\n\n" + "\n\n".join(blocks)
        )
        return run_structured_turn(
            loop=loop,
            driver=driver,
            driver_model=driver_model,
            store=self.store,
            run_id=run_id,
            workspace_root=self.workspace_root,
            actor="design:arbiter",
            system_prompt=_ARBITER_SYSTEM_PROMPT,
            user_prompt=prompt,
            model_cls=DesignSpec,
            trace_path=trace_path,
            max_steps=max_steps,
            max_validation_retries=2,
            validation_header="[DESIGN VALIDATION FAILED]",
            validation_instruction=(
                "Fix every violation and return a valid DesignSpec JSON object."
            ),
            transform=transform,
            validator=validator,
            on_validation_exhausted=lambda candidate, violations: _raise_panel_exhausted(
                violations
            ),
        )

    def approve_design(
        self,
        run_id: str,
        approver: str = "zeus",
        reason: str = "",
        *,
        expected_revision: int | None = None,
        expected_target_hash: str | None = None,
    ) -> ApprovalRecord:
        """Freeze the current DesignSpec and record a hash-bound approval."""
        return RunControl(self.store).approve_design(
            run_id=run_id,
            approver=approver,
            reason=reason,
            expected_revision=expected_revision,
            expected_target_hash=expected_target_hash,
        )


def _normalize_variants(
    variants: Sequence[DesignVariant | str] | None,
) -> tuple[DesignVariant, ...]:
    """Coerce workflow params into a de-duplicated, ordered variant tuple."""
    if variants is None:
        return DEFAULT_DESIGN_VARIANTS
    normalized: list[DesignVariant] = []
    for raw in variants:
        if isinstance(raw, DesignVariant):
            normalized.append(raw)
            continue
        try:
            normalized.append(DesignVariant(str(raw).strip().lower()))
        except ValueError as exc:
            allowed = ", ".join(variant.value for variant in DesignVariant)
            raise ValueError(f"unknown design variant {raw!r}; allowed: {allowed}") from exc
    unique = tuple(dict.fromkeys(normalized))
    if not unique:
        raise ValueError("design panel requires at least one variant")
    return unique


def _archive_candidates(run_dir: Path, candidates: Iterable[DesignCandidate]) -> list[str]:
    """Persist every draft verbatim so the rejected ones stay inspectable after the fact."""
    target = run_dir / "design_candidates"
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for candidate in candidates:
        payload = {
            "variant": candidate.variant,
            "violations": candidate.violations,
            "error": candidate.error,
            "design": candidate.design.model_dump(mode="json") if candidate.design else None,
        }
        path = target / f"{candidate.variant}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(str(path))
    return written


def _panel_failure_summary(candidates: Sequence[DesignCandidate]) -> list[str]:
    return [
        f"{candidate.variant}: "
        f"{candidate.error or '; '.join(candidate.violations) or 'no design'}"
        for candidate in candidates
    ]


def _raise_panel_exhausted(violations: list[str]) -> DesignSpec:
    raise DesignValidationError(violations)


__all__ = [
    "DEFAULT_DESIGN_VARIANTS",
    "DesignCandidate",
    "DesignStage",
    "DesignValidationError",
    "DesignVariant",
    "ExtractionProbe",
    "triage_input",
]
