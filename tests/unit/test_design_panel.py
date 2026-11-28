"""Multi-agent design panel: parallel candidates, archived verdicts, one synthesized design."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from olympus.core.contracts import (
    AcceptanceCriterion,
    EarsPattern,
    ExecutionStrategy,
    GoalSpec,
    RequirementItem,
    SpecStatus,
)
from olympus.core.dag import validate_design_spec
from olympus.core.store import RunStore
from olympus.providers.base import ChatMessage, LLMResponse, MockLLMProvider, ToolSpec
from olympus.stages.design import DesignStage, DesignValidationError

VARIANTS = ("product_led", "engineering_led", "risk_led")


def _goal_spec() -> GoalSpec:
    return GoalSpec(
        spec_id="spec-panel",
        goal="Operators manage notes through a REST API",
        deliverable_kind="rest_api",
        observable_surfaces=["HTTP API"],
        actors=["operator"],
        requirements=[
            RequirementItem(
                id="R1",
                description="Create a note through POST /notes",
                actor="operator",
                ears_criteria=[
                    AcceptanceCriterion(
                        id="R1-C1",
                        pattern=EarsPattern.EVENT_DRIVEN,
                        condition="an operator posts a note",
                        outcome="return HTTP 201 within 200 ms",
                    )
                ],
            ),
            RequirementItem(
                id="R2",
                description="Reject malformed payloads",
                actor="operator",
                ears_criteria=[
                    AcceptanceCriterion(
                        id="R2-C1",
                        pattern=EarsPattern.UNWANTED_BEHAVIOR,
                        condition="the name field is missing",
                        outcome="return HTTP 422 and never write a row",
                    )
                ],
            ),
        ],
        out_of_scope=["multi-tenant isolation"],
        constraints=["python 3.13"],
    )


def _design_payload(spec: GoalSpec, *, summary: str) -> dict[str, object]:
    # `architecture_summary` carries a minimum length in the structural gate, so the variant
    # marker is a prefix on real prose instead of a bare token like "risk_led".
    headline = f"{summary}: one deep notes module behind the HTTP seam"
    return {
        "architecture_summary": headline,
        "spec_id": spec.spec_id,
        "spec_hash": spec.content_hash,
        "shared_invariants": [
            {
                "id": "INV-ERR",
                "category": "error_envelope",
                "rule": "Non-2xx responses use {error: {code, message}}.",
            }
        ],
        "modules": [
            {
                "id": "M1",
                "name": "Notes core",
                "responsibility": "Validation, storage, and the HTTP seam for notes",
                "requirement_refs": ["R1", "R2"],
                "target_paths": ["src/notes/"],
            }
        ],
        "data_models": [{"name": "Note", "fields": ["id", "name"], "owner_module_id": "M1"}],
        "interfaces": [],
        "execution_strategy": ExecutionStrategy.SERIAL_HANDOFF.value,
        "strategy_rationale": f"{summary}: one cohesive module, sequential delivery.",
    }


def _seeded_run(tmp_path: Path) -> tuple[RunStore, str, GoalSpec]:
    spec = _goal_spec()
    store = RunStore(tmp_path / ".olympus" / "olympus.db")
    run_id = store.create_run(title="design panel")
    store.save_spec(run_id, spec.model_copy(update={"status": SpecStatus.CONFIRMED}))
    return store, run_id, spec


def _panel_provider(
    spec: GoalSpec,
    *,
    fail_variants: frozenset[str] = frozenset(),
    arbiter_summary: str = "arbiter merge",
) -> MockLLMProvider:
    """Route each prompt to its variant by the mandate marker in the system prompt."""

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        system = messages[0].content
        for variant in VARIANTS:
            dash = variant.replace("_", "-")
            if f"MANDATE ({dash})" in system or f"MANDATE ({variant})" in system:
                if variant in fail_variants:
                    raise RuntimeError(f"{variant} exploded")
                return LLMResponse(content=json.dumps(_design_payload(spec, summary=variant)))
        if "Design Arbiter" in system:
            return LLMResponse(content=json.dumps(_design_payload(spec, summary=arbiter_summary)))
        raise AssertionError(f"unexpected prompt: {system[:120]}")

    return MockLLMProvider(responder=responder)


def test_panel_drafts_three_archives_them_and_synthesizes_one(tmp_path: Path) -> None:
    store, run_id, spec = _seeded_run(tmp_path)
    provider = _panel_provider(spec)
    stage = DesignStage(store, provider, tmp_path)

    design = stage.propose_design_panel(run_id)

    systems = [request[0][0].content for request in provider.recorded_requests]
    # A variant may be invoked more than once (structured-output repair), so assert coverage
    # rather than an exact call count.
    for variant in VARIANTS:
        marker = f"MANDATE ({variant.replace('_', '-')})"
        assert any(marker in system for system in systems), f"variant {variant} never drafted"
    assert any("Design Arbiter" in system for system in systems), "expected a synthesis turn"

    candidate_dir = stage.artifacts_root / run_id / "design_candidates"
    archived = sorted(path.stem for path in candidate_dir.glob("*.json"))
    assert archived == sorted(VARIANTS)
    archived_payload = json.loads((candidate_dir / "product_led.json").read_text("utf-8"))
    assert archived_payload["design"]["architecture_summary"].startswith("product_led")

    produced = [
        event
        for event in store.list_events(run_id)
        if event.event_type == "design.candidates_produced"
    ]
    assert len(produced) == 1
    payload = produced[0].payload
    assert sorted(payload["usable"]) == sorted(VARIANTS)
    assert payload["errors"] == {}

    assert design.architecture_summary.startswith("arbiter merge")
    assert validate_design_spec(design, spec, strict_v2=True) == []
    saved = store.get_run(run_id)["design"]
    assert saved is not None
    assert saved.design_id == design.design_id


def test_panel_keeps_going_when_one_variant_raises(tmp_path: Path) -> None:
    store, run_id, spec = _seeded_run(tmp_path)
    provider = _panel_provider(spec, fail_variants=frozenset({"risk_led"}))
    stage = DesignStage(store, provider, tmp_path)

    design = stage.propose_design_panel(run_id)

    produced = [
        event
        for event in store.list_events(run_id)
        if event.event_type == "design.candidates_produced"
    ]
    payload = produced[0].payload
    assert sorted(payload["usable"]) == ["engineering_led", "product_led"]
    assert "risk_led" in payload["errors"]
    assert design.architecture_summary.startswith("arbiter merge")


def test_panel_without_synthesis_takes_the_first_usable_candidate(tmp_path: Path) -> None:
    store, run_id, spec = _seeded_run(tmp_path)
    provider = _panel_provider(spec)
    stage = DesignStage(store, provider, tmp_path)

    design = stage.propose_design_panel(run_id, synthesize=False)

    systems = [request[0][0].content for request in provider.recorded_requests]
    assert not any("Design Arbiter" in system for system in systems)
    assert design.architecture_summary.startswith("product_led")
    assert validate_design_spec(design, spec, strict_v2=True) == []


def test_panel_raises_when_every_variant_fails(tmp_path: Path) -> None:
    store, run_id, spec = _seeded_run(tmp_path)
    provider = _panel_provider(spec, fail_variants=frozenset(VARIANTS))
    stage = DesignStage(store, provider, tmp_path)

    with pytest.raises(DesignValidationError) as excinfo:
        stage.propose_design_panel(run_id)

    assert "no usable design candidate" in str(excinfo.value)
    assert store.get_run(run_id)["design"] is None


def test_panel_rejects_an_unknown_variant(tmp_path: Path) -> None:
    store, run_id, spec = _seeded_run(tmp_path)
    stage = DesignStage(store, _panel_provider(spec), tmp_path)

    with pytest.raises(ValueError, match="unknown design variant"):
        stage.propose_design_panel(run_id, variants=["product_led", "nope_led"])
