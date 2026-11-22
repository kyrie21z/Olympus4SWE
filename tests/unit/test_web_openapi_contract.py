"""Contract seam: the checked-in OpenAPI snapshot must match the live FastAPI app."""

from __future__ import annotations

from pathlib import Path

from olympus.core.store import RunStore
from olympus.web.openapi import (
    OPENAPI_SNAPSHOT_PATH,
    build_openapi_schema,
    render_openapi_schema,
)
from olympus.web.server import AriadneApp, create_fastapi_app

EXPECTED_COMMANDS = {
    "pause",
    "resume",
    "approve_spec",
    "approve_design",
    "approve_plan",
    "clarify",
    "retry_task",
    "accept_risk",
    "reassign_human",
    "retry_workflow",
    "promote_to_execute",
}


def test_checked_in_openapi_snapshot_matches_live_schema(tmp_path: Path) -> None:
    app = AriadneApp(store=RunStore(tmp_path / "openapi.db"), workspace_root=tmp_path)
    live = render_openapi_schema(create_fastapi_app(app).openapi())
    assert OPENAPI_SNAPSHOT_PATH.is_file(), "Run `uv run python -m olympus.web.openapi`"
    assert live == OPENAPI_SNAPSHOT_PATH.read_text(encoding="utf-8")


def test_openapi_projection_and_command_contracts_are_concrete() -> None:
    schema = build_openapi_schema()
    components = schema["components"]["schemas"]

    projection = components["RunProjection"]
    assert projection["additionalProperties"] is False
    assert {"dag", "workflow", "approval_reviews", "events", "budget", "progress"} <= set(
        projection["properties"]
    )

    command = schema["paths"]["/api/runs/{run_id}/command"]["post"]
    body = command["requestBody"]["content"]["application/json"]["schema"]
    assert body["discriminator"]["propertyName"] == "action"
    assert set(body["discriminator"]["mapping"]) == EXPECTED_COMMANDS
    assert set(command["responses"]) >= {"200", "400", "404", "409"}
    assert command["responses"]["409"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "ConflictResponse"
    )
    assert command["responses"]["400"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "ErrorResponse"
    )
    assert "ErrorResponse" in components
    assert "ConflictResponse" in components
    assert "HTTPValidationError" not in components
    assert "ProviderConfig" in components
    assert "anyOf" in components["ApprovalReview"]["properties"]["baseline"]

    budget = components["BudgetSummary"]
    assert {"cache_read_tokens", "cache_write_tokens"} <= set(budget["properties"])
    assert not ({"cache_read_tokens", "cache_write_tokens"} & set(budget["required"]))

    for path_item in schema["paths"].values():
        for operation in path_item.values():
            assert "422" not in operation.get("responses", {})

    stream_responses = schema["paths"]["/api/runs/{run_id}/stream"]["get"]["responses"]
    assert set(stream_responses["200"]["content"]) == {"text/event-stream"}
    assert set(stream_responses["404"]["content"]) == {"application/json"}
    assert stream_responses["404"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "ErrorResponse"
    )

    log_responses = schema["paths"]["/api/runs/{run_id}/delivery-log/{event_seq}/{position}"][
        "get"
    ]["responses"]
    assert set(log_responses["200"]["content"]) == {"text/plain"}
    assert set(log_responses["404"]["content"]) == {"application/json"}
    assert log_responses["404"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "ErrorResponse"
    )


def test_openapi_covers_every_ariadne_route() -> None:
    paths = set(build_openapi_schema()["paths"])
    assert {
        "/healthz",
        "/api/runs",
        "/api/runs/{run_id}",
        "/api/runs/{run_id}/events",
        "/api/runs/{run_id}/command",
        "/api/runs/{run_id}/stream",
        "/api/runs/{run_id}/task-artifacts/{task_id}/{attempt}",
        "/api/runs/{run_id}/delivery-log/{event_seq}/{position}",
        "/api/providers",
        "/api/workflows",
        "/api/demos",
        "/api/seed-demo",
    } <= paths


def test_budget_summary_accepts_legacy_wire_payload_without_cache_fields() -> None:
    from olympus.web.schemas import BudgetSummary

    summary = BudgetSummary.model_validate(
        {
            "calls_used": 1,
            "calls_reserved": 0,
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "unknown_outcome_calls": 0,
        }
    )
    assert summary.cache_read_tokens == 0
    assert summary.cache_write_tokens == 0

