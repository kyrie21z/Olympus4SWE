"""Unit tests for the 5-element EARS GoalSpec, DesignSpec gate, and input triage."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from olympus.core.contracts import (
    AcceptanceCriterion,
    DesignSpec,
    EarsPattern,
    GoalSpec,
    InputMaturity,
    RequirementItem,
    TaskPlan,
)
from olympus.core.dag import validate_design_spec, validate_goal_spec, validate_plan
from olympus.stages.design import triage_input


def _ears(
    criterion_id: str = "C1",
    pattern: EarsPattern = EarsPattern.UBIQUITOUS,
    condition: str = "",
    outcome: str = "return HTTP 200 with the stored item",
) -> AcceptanceCriterion:
    return AcceptanceCriterion(
        id=criterion_id,
        pattern=pattern,
        condition=condition,
        outcome=outcome,
    )


def _canonical_spec() -> GoalSpec:
    return GoalSpec(
        spec_id="spec-canon",
        goal="Operators manage an inventory catalogue through a REST API",
        deliverable_kind="rest_api",
        observable_surfaces=["HTTP API", "OpenAPI document"],
        actors=["operator", "admin"],
        requirements=[
            RequirementItem(
                id="R1",
                description="Create an item through POST /items",
                actor="operator",
                ears_criteria=[
                    _ears(
                        "R1-C1",
                        EarsPattern.EVENT_DRIVEN,
                        "an operator posts a new item",
                        "persist it and return HTTP 201 within 200 ms",
                    )
                ],
            ),
            RequirementItem(
                id="R2",
                description="Reject malformed payloads",
                actor="operator",
                ears_criteria=[
                    _ears(
                        "R2-C1",
                        EarsPattern.UNWANTED_BEHAVIOR,
                        "the payload omits the name field",
                        "return HTTP 422 and never write a row",
                    )
                ],
            ),
        ],
        out_of_scope=["multi-tenant isolation", "bulk import"],
        constraints=["python 3.13"],
    )


def _design(spec: GoalSpec | None = None, **overrides: object) -> DesignSpec:
    data: dict[str, object] = {
        "architecture_summary": "HTTP edge over a storage core",
        "spec_id": spec.spec_id if spec else None,
        "spec_hash": spec.content_hash if spec else None,
        "modules": [
            {
                "id": "M1",
                "name": "API",
                "responsibility": "HTTP routing and validation",
                "requirement_refs": ["R1", "R2"],
                "target_paths": ["src/api/"],
            },
            {
                "id": "M2",
                "name": "Store",
                "responsibility": "Persistence",
                "requirement_refs": ["R1"],
                "target_paths": ["src/core/"],
            },
        ],
        "data_models": [
            {"name": "Item", "fields": ["id", "name"], "owner_module_id": "M2"},
        ],
        "interfaces": [
            {
                "id": "IF1",
                "provider_module_id": "M2",
                "consumer_module_ids": ["M1"],
                "signature": "create_item(name: str) -> Item",
                "error_cases": ["duplicate name"],
            }
        ],
    }
    data.update(overrides)
    return DesignSpec.model_validate(data)


# --------------------------------------------------------------------------- EARS


def test_ears_rendering_covers_every_pattern() -> None:
    assert _ears(outcome="log every request").to_ears_text() == (
        "THE SYSTEM SHALL log every request"
    )
    assert _ears(
        pattern=EarsPattern.EVENT_DRIVEN, condition="a request arrives", outcome="log it"
    ).to_ears_text() == "WHEN a request arrives THE SYSTEM SHALL log it"
    assert _ears(
        pattern=EarsPattern.STATE_DRIVEN, condition="the queue is full", outcome="reject writes"
    ).to_ears_text() == "WHILE the queue is full THE SYSTEM SHALL reject writes"
    assert _ears(
        pattern=EarsPattern.UNWANTED_BEHAVIOR, condition="the token expires", outcome="return 401"
    ).to_ears_text() == "IF the token expires THEN THE SYSTEM SHALL return 401"
    assert _ears(
        pattern=EarsPattern.OPTIONAL_FEATURE, condition="a plugin is enabled", outcome="run it"
    ).to_ears_text() == "WHERE a plugin is enabled THE SYSTEM SHALL run it"


def test_conditional_ears_pattern_requires_condition() -> None:
    with pytest.raises(ValidationError):
        AcceptanceCriterion(
            id="C1",
            pattern=EarsPattern.EVENT_DRIVEN,
            condition="   ",
            outcome="do something",
        )


def test_requirement_materializes_plain_string_criteria_from_ears() -> None:
    req = RequirementItem(
        id="R1",
        description="Issue tokens",
        actor="operator",
        ears_criteria=[
            _ears(
                "C1",
                EarsPattern.EVENT_DRIVEN,
                "a valid login arrives",
                "issue a JWT valid for 15 minutes",
            )
        ],
    )
    assert req.acceptance_criteria == [
        "WHEN a valid login arrives THE SYSTEM SHALL issue a JWT valid for 15 minutes"
    ]
    assert all(isinstance(item, str) for item in req.acceptance_criteria)


def test_requirement_without_any_criterion_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RequirementItem(id="R1", description="Nothing verifiable")


# --------------------------------------------------------------- GoalSpec gate


def test_canonical_goal_spec_passes_the_zero_token_gate() -> None:
    assert validate_goal_spec(_canonical_spec()) == []


def test_goal_spec_gate_requires_five_elements_and_ears() -> None:
    bare = GoalSpec(
        goal="Build something",
        requirements=[
            RequirementItem(
                id="R1",
                description="Do the thing",
                acceptance_criteria=["HTTP 200 returned"],
            )
        ],
        out_of_scope=["nothing"],
    )
    violations = validate_goal_spec(bare)
    joined = " ".join(violations)
    assert "deliverable_kind" in joined
    assert "observable_surfaces" in joined
    assert "actors" in joined
    assert "ears_criteria" in joined
    assert "actor" in joined


def test_goal_spec_gate_requires_out_of_scope_by_default() -> None:
    spec = _canonical_spec().model_copy(update={"out_of_scope": []})
    assert any("out_of_scope" in v for v in validate_goal_spec(spec))
    assert validate_goal_spec(
        spec, strict_ears=False, require_out_of_scope=False
    ) == []


def test_goal_spec_gate_rejects_vague_unquantified_wording() -> None:
    spec = _canonical_spec()
    vague_req = RequirementItem(
        id="R3",
        description="The UI must be user-friendly and handle appropriately",
        actor="operator",
        ears_criteria=[
            _ears("R3-C1", EarsPattern.UBIQUITOUS, outcome="make the interface 美观")
        ],
    )
    spec = spec.model_copy(update={"requirements": [*spec.requirements, vague_req]})
    violations = validate_goal_spec(spec)
    assert any("Vague unquantified wording" in v for v in violations)


def test_goal_spec_gate_accepts_vague_word_with_a_measurable_detail() -> None:
    spec = _canonical_spec()
    concrete_req = RequirementItem(
        id="R4",
        description="The interface stays user-friendly: first paint within 300 ms",
        actor="operator",
        ears_criteria=[
            _ears(
                "R4-C1",
                EarsPattern.UBIQUITOUS,
                outcome="keep the settings page 界面友好 with a 300 ms p95 budget",
            )
        ],
    )
    spec = spec.model_copy(update={"requirements": [*spec.requirements, concrete_req]})
    assert validate_goal_spec(spec) == []


def test_goal_spec_gate_rejects_premature_implementation_leaks() -> None:
    spec = _canonical_spec()
    leaky = RequirementItem(
        id="R5",
        description="Create the table with CREATE TABLE items and edit src/api/app.py",
        actor="operator",
        ears_criteria=[
            _ears("R5-C1", EarsPattern.UBIQUITOUS, outcome="persist rows with 1 ms latency")
        ],
    )
    spec = spec.model_copy(update={"requirements": [*spec.requirements, leaky]})
    violations = validate_goal_spec(spec)
    assert any("implementation leak" in v for v in violations)

    allowed = spec.model_copy(
        update={
            "constraints": [
                "python 3.13",
                "CREATE TABLE items",
                "src/api/app.py",
            ]
        }
    )
    assert validate_goal_spec(allowed) == []


def test_goal_spec_gate_reports_unresolved_blockers() -> None:
    from olympus.core.contracts import AmbiguityItem

    base = _canonical_spec()
    spec = GoalSpec(
        spec_id=base.spec_id,
        goal=base.goal,
        deliverable_kind=base.deliverable_kind,
        observable_surfaces=base.observable_surfaces,
        actors=base.actors,
        requirements=base.requirements,
        out_of_scope=base.out_of_scope,
        constraints=base.constraints,
        open_ambiguities=[
            AmbiguityItem(
                id="AMB-1",
                question="Which database?",
                impact="storage",
                blocking=True,
            )
        ],
    )
    assert any("unresolved blockers" in v for v in validate_goal_spec(spec))


# --------------------------------------------------------------- DesignSpec gate


def test_design_spec_passes_with_full_coverage_and_seams() -> None:
    spec = _canonical_spec()
    assert validate_design_spec(_design(spec), spec) == []


def test_design_spec_requires_bound_spec_and_full_requirement_coverage() -> None:
    spec = _canonical_spec()
    unbound = _design()
    assert any("not bound to a GoalSpec" in v for v in validate_design_spec(unbound, spec))

    partial = _design(
        spec,
        modules=[
            {
                "id": "M1",
                "name": "API",
                "responsibility": "HTTP",
                "requirement_refs": ["R1"],
                "target_paths": ["src/api/"],
            }
        ],
        interfaces=[],
        data_models=[],
    )
    violations = validate_design_spec(partial, spec)
    assert any("not covered by any design module" in v for v in violations)


def test_design_spec_rejects_overlapping_module_paths() -> None:
    spec = _canonical_spec()
    overlapping = _design(
        spec,
        modules=[
            {
                "id": "M1",
                "name": "API",
                "responsibility": "HTTP",
                "requirement_refs": ["R1", "R2"],
                "target_paths": ["src/"],
            },
            {
                "id": "M2",
                "name": "Store",
                "responsibility": "Persistence",
                "requirement_refs": ["R1"],
                "target_paths": ["src/core/"],
            },
        ],
    )
    violations = validate_design_spec(overlapping, spec)
    assert any("overlapping target_paths" in v for v in violations)


def test_design_spec_rejects_missing_interface_seam_and_dangling_owner() -> None:
    spec = _canonical_spec()
    unseamed = _design(spec, interfaces=[])
    violations = validate_design_spec(unseamed, spec)
    assert any("seam closure" in v for v in violations)

    dangling = _design(
        spec,
        data_models=[
            {"name": "Item", "fields": ["id"], "owner_module_id": "M9"},
        ],
    )
    assert any("owner_module_id" in v for v in validate_design_spec(dangling, spec))


# ------------------------------------------------------------------ Plan binding


def _task(task_id: str, **overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "task_id": task_id,
        "title": task_id,
        "objective": f"implement {task_id}",
        "write_scope": [f"src/{task_id.lower()}/"],
        "acceptance_checks": [
            {"id": f"{task_id}-c", "kind": "command", "description": "run", "command": "true"}
        ],
    }
    data.update(overrides)
    return data


def test_task_plan_accepts_design_only_binding() -> None:
    plan = TaskPlan(
        design_id="design-1",
        design_hash="sha256:abc",
        tasks=[
            _task(
                "T1",
                requirement_refs=[],
                design_refs=["M1"],
            )
        ],
    )
    assert plan.spec_id is None
    assert plan.tasks[0].kind.value == "implement"
    assert plan.tasks[0].role == "engineer"


def test_task_plan_requires_at_least_one_upstream_binding() -> None:
    with pytest.raises(ValidationError):
        TaskPlan(tasks=[_task("T1")])


def test_validate_plan_checks_design_refs_and_coverage() -> None:
    spec = _canonical_spec()
    design = _design(spec)
    plan = TaskPlan(
        plan_id="plan-design",
        design_id=design.design_id,
        design_hash=design.content_hash,
        tasks=[
            _task("T1", requirement_refs=["R1", "R2"], design_refs=["M1", "IF1"]),
            _task("T2", requirement_refs=["R1"], design_refs=["M2"], depends_on=["T1"]),
        ],
    )
    assert validate_plan(plan, spec, design=design) == []

    bad_ref = TaskPlan(
        plan_id="plan-design-bad",
        design_id=design.design_id,
        design_hash=design.content_hash,
        tasks=[
            _task("T1", requirement_refs=["R1", "R2"], design_refs=["M9"]),
            _task("T2", requirement_refs=["R1"], design_refs=["M2"], depends_on=["T1"]),
        ],
    )
    assert any("unknown design element" in v for v in validate_plan(bad_ref, spec, design=design))

    orphan = TaskPlan(
        plan_id="plan-design-orphan",
        design_id=design.design_id,
        design_hash=design.content_hash,
        tasks=[
            _task("T1", requirement_refs=["R1", "R2"], design_refs=["M1", "IF1"]),
            _task("T2", requirement_refs=["R1"], design_refs=["M1"], depends_on=["T1"]),
        ],
    )
    assert any(
        "not referenced by any task" in v for v in validate_plan(orphan, spec, design=design)
    )


# ---------------------------------------------------------------------- Triage


def test_triage_downgrades_prose_that_claims_to_be_a_design_document() -> None:
    prose = (
        "# design_spec.md\n\n"
        "This is a design document. The team wants a nice interface for users and a "
        "storage layer. It should feel modern."
    )
    result = triage_input(prose)
    assert result.maturity == InputMaturity.RAW_INTENT
    assert result.extracted_design is None


def test_triage_recognizes_a_frozen_goal_spec() -> None:
    spec = _canonical_spec()
    result = triage_input(spec.model_dump(mode="json"))
    assert result.maturity == InputMaturity.GOAL_SPEC_READY
    assert result.extracted_goal is not None
    assert result.extracted_design is None
    assert any("interface" in reason for reason in result.reasons)


def test_triage_recognizes_a_frozen_design_spec_and_emits_event(tmp_path) -> None:
    from olympus.core.store import RunStore

    spec = _canonical_spec()
    design = _design(spec)
    store = RunStore(tmp_path / ".olympus" / "olympus.db")
    run_id = store.create_run(title="triage")

    as_json = triage_input(json.dumps(design.model_dump(mode="json")))
    assert as_json.maturity == InputMaturity.DESIGN_SPEC_READY
    assert as_json.extracted_design is not None

    as_dict = triage_input(
        design.model_dump(mode="json"),
        store=store,
        run_id=run_id,
    )
    assert as_dict.maturity == InputMaturity.DESIGN_SPEC_READY
    events = [e for e in store.list_events(run_id) if e.event_type == "intake.triaged"]
    assert len(events) == 1
    assert events[0].payload["maturity"] == "design_spec_ready"


def test_triage_downgrades_incomplete_design_payloads() -> None:
    # "modules" is present but empty: schema-invalid, so it can only be raw intent.
    partial = {
        "architecture_summary": "one blob",
        "modules": [],
    }
    result = triage_input(partial)
    assert result.maturity == InputMaturity.RAW_INTENT
    assert result.reasons

    # Two modules that share every path also fail the seam/disjointness gate.
    overlapping = {
        "architecture_summary": "one blob",
        "modules": [
            {
                "id": "M1",
                "name": "A",
                "responsibility": "all of it",
                "requirement_refs": ["R1"],
                "target_paths": ["src/"],
            },
            {
                "id": "M2",
                "name": "B",
                "responsibility": "all of it",
                "requirement_refs": ["R1"],
                "target_paths": ["src/"],
            },
        ],
    }
    assert triage_input(overlapping).maturity == InputMaturity.RAW_INTENT


def test_design_spec_v2_invariants_cycles_contracts_and_strategy() -> None:
    from olympus.core.contracts import ExecutionStrategy, TaskKind

    spec = _canonical_spec()
    v1_design = _design(spec)
    assert validate_design_spec(v1_design, spec, strict_v2=False) == []
    # Reloading a stripped v1 JSON row preserves exact v1 content_hash.
    v1_raw = json.dumps(
        {
            "design_id": v1_design.design_id,
            "version": 1,
            "spec_id": spec.spec_id,
            "spec_hash": spec.content_hash,
            "architecture_summary": "HTTP edge over a storage core",
            "modules": [m.model_dump(mode="json") for m in v1_design.modules],
            "data_models": [d.model_dump(mode="json") for d in v1_design.data_models],
            "interfaces": [
                {
                    "id": i.id,
                    "provider_module_id": i.provider_module_id,
                    "consumer_module_ids": i.consumer_module_ids,
                    "protocol": i.protocol,
                    "signature": i.signature,
                    "error_cases": i.error_cases,
                }
                for i in v1_design.interfaces
            ],
        }
    )
    assert DesignSpec.model_validate_json(v1_raw).content_hash == v1_design.content_hash
    # Strict v2 gate rejects missing invariants/strategy/typed schemas.
    strict_v1_errors = validate_design_spec(v1_design, spec, strict_v2=True)
    assert any("shared_invariants" in e for e in strict_v1_errors)
    assert any("execution_strategy" in e for e in strict_v1_errors)
    assert any("input_schema" in e for e in strict_v1_errors)

    v2_design = _design(
        spec,
        shared_invariants=[
            {
                "id": "INV-ERR-01",
                "category": "error_envelope",
                "rule": "Return JSON {error: {code, message}} on every non-2xx path.",
                "enforcement_hint": "shared exception mapper",
                "applies_to": ["M1", "M2"],
            }
        ],
        shared_contract_paths=["src/contracts/"],
        execution_strategy=ExecutionStrategy.CONTRACT_FIRST_PARALLEL.value,
        strategy_rationale="M1 and M2 communicate strictly via src/contracts/ pinned in Wave 0.",
        interfaces=[
            {
                "id": "IF1",
                "provider_module_id": "M2",
                "consumer_module_ids": ["M1"],
                "protocol": "python_fn",
                "signature": "create_item(name: str) -> Item",
                "input_schema": [
                    {"name": "name", "type_expr": "str", "required": True, "constraints": "1..80"}
                ],
                "output_schema": [
                    {"name": "id", "type_expr": "str", "required": True},
                    {"name": "name", "type_expr": "str", "required": True},
                ],
                "data_entity_refs": ["Item"],
                "error_cases": ["duplicate name"],
            }
        ],
    )
    assert validate_design_spec(v2_design, spec, strict_v2=True) == []
    md = v2_design.to_markdown()
    assert "INV-ERR-01" in md
    assert "contract_first_parallel" in md
    assert "src/contracts/" in md

    # Shared contract path overlapping module target_paths is rejected.
    bad_overlap = _design(spec, shared_contract_paths=["src/core/contracts.py"])
    assert any("shared_contract_paths overlaps" in e for e in validate_design_spec(bad_overlap))

    # Cyclic module dependency graph (M1 -> M2 -> M1) is caught via Kahn check.
    cyclic = _design(
        spec,
        interfaces=[
            {
                "id": "IF1",
                "provider_module_id": "M2",
                "consumer_module_ids": ["M1"],
                "signature": "a() -> None",
                "error_cases": ["err"],
            },
            {
                "id": "IF2",
                "provider_module_id": "M1",
                "consumer_module_ids": ["M2"],
                "signature": "b() -> None",
                "error_cases": ["err"],
            },
        ],
    )
    assert any("cycle" in e for e in validate_design_spec(cyclic, spec))

    # Non-contract TaskNode cannot declare shared_contract_paths in write_scope.
    bad_plan = TaskPlan(
        plan_id="plan-bad-contract-scope",
        design_id=v2_design.design_id,
        design_hash=v2_design.content_hash,
        tasks=[
            _task(
                "T0",
                requirement_refs=["R1"],
                design_refs=["IF1"],
                write_scope=["src/contracts/models.py"],
                kind=TaskKind.CONTRACT,
            ),
            _task(
                "T1",
                requirement_refs=["R1", "R2"],
                design_refs=["M1", "M2"],
                depends_on=["T0"],
                write_scope=["src/api/", "src/core/", "src/contracts/models.py"],
                kind=TaskKind.IMPLEMENT,
            ),
        ],
    )
    assert any(
        "only kind='contract' tasks may write shared contracts" in v
        for v in validate_plan(bad_plan, spec, design=v2_design)
    )
