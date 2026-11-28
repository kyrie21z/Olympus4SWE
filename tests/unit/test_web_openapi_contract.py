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
    "request_revision",
    "reopen_run",
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
    assert {
        "dag",
        "workflow",
        "approval_reviews",
        "events",
        "budget",
        "progress",
        "pending_resynthesis",
        "resynthesis_error",
        "clarification_notes",
    } <= set(projection["properties"])
    assert "ClarificationNoteView" in components

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
        "/api/board/stream",
        "/api/runs/{run_id}/task-artifacts/{task_id}/{attempt}",
        "/api/runs/{run_id}/delivery-log/{event_seq}/{position}",
        "/api/runs/{run_id}/product-artifact/{kind}",
        "/api/providers",
        "/api/workflows",
        "/api/demos",
        "/api/seed-demo",
        "/api/filesystem/browse",
        "/api/workspaces/clone",
    } <= paths


def test_openapi_covers_filesystem_browse_and_workspace_clone_contracts() -> None:
    schema = build_openapi_schema()
    components = schema["components"]["schemas"]

    browse = schema["paths"]["/api/filesystem/browse"]["get"]
    assert {parameter["name"] for parameter in browse["parameters"]} == {
        "partial_path",
        "show_hidden",
    }
    assert set(browse["responses"]) >= {"200", "400"}
    assert browse["responses"]["200"]["content"]["application/json"]["schema"][
        "$ref"
    ].endswith("FilesystemBrowseResponse")
    assert browse["responses"]["400"]["content"]["application/json"]["schema"][
        "$ref"
    ].endswith("ErrorResponse")

    browse_response = components["FilesystemBrowseResponse"]
    assert browse_response["additionalProperties"] is False
    assert {
        "current_path",
        "parent_path",
        "prefix",
        "path_missing",
        "quick_roots",
        "entries",
    } <= set(browse_response["properties"])
    assert browse_response["properties"]["entries"]["items"]["$ref"].endswith(
        "FilesystemBrowseEntry"
    )
    assert browse_response["properties"]["quick_roots"]["items"]["$ref"].endswith("QuickRoot")

    entry = components["FilesystemBrowseEntry"]
    assert entry["additionalProperties"] is False
    assert {
        "name",
        "path",
        "is_git_repo",
        "default_branch",
        "already_registered",
    } <= set(entry["properties"])
    assert set(components["QuickRoot"]["properties"]) == {"label", "path"}

    clone = schema["paths"]["/api/workspaces/clone"]["post"]
    body = clone["requestBody"]["content"]["application/json"]["schema"]
    assert body["$ref"].endswith("CloneWorkspaceRequest")
    assert set(clone["responses"]) >= {"200", "400"}
    assert clone["responses"]["200"]["content"]["application/json"]["schema"][
        "$ref"
    ].endswith("WorkspaceResponse")
    assert clone["responses"]["400"]["content"]["application/json"]["schema"][
        "$ref"
    ].endswith("ErrorResponse")

    clone_request = components["CloneWorkspaceRequest"]
    assert clone_request["additionalProperties"] is False
    assert set(clone_request["required"]) == {"remote_url", "parent_dir"}
    assert {"folder_name", "name"} <= set(clone_request["properties"])


def test_openapi_covers_v01_projection_and_product_artifact_contracts() -> None:
    schema = build_openapi_schema()
    components = schema["components"]["schemas"]

    projection = components["RunProjection"]
    assert "stage_activity" not in projection["properties"]
    assert projection["properties"]["post_dag"]["anyOf"][1] == {"type": "null"}

    node = components["DagNode"]
    assert {"review_deferred", "escape_hatch_rejections", "escape_hatch_findings"} <= set(
        node["properties"]
    )
    assert "review_deferred" not in node.get("required", [])

    product = components["ProductVerificationView"]
    assert product["additionalProperties"] is False
    assert set(product["properties"]) == {
        "passed",
        "summary",
        "failures",
        "event_seq",
        "verified_at",
        "checks",
        "has_report",
        "has_boot_log",
        "has_screenshot",
        "screenshot_sha256",
    }
    assert product["properties"]["checks"]["items"]["$ref"].endswith("ProductCheckItem")

    post_dag = components["PostDagSummary"]
    assert post_dag["additionalProperties"] is False
    round_item = components["PostDagRoundItem"]
    assert {"sequence", "reported_round", "phase", "outcome"} <= set(round_item["properties"])
    assert set(round_item["properties"]["phase"]["enum"]) == {
        "verify",
        "recheck",
        "review",
        "repair",
    }

    waterfall = components["WaterfallTurn"]
    assert waterfall["properties"]["receipt"]["anyOf"][1] == {"type": "null"}

    artifact = schema["paths"]["/api/runs/{run_id}/product-artifact/{kind}"]["get"]
    assert set(artifact["responses"]) == {"200", "400", "404"}
    assert set(artifact["responses"]["200"]["content"]) == {
        "application/json",
        "text/plain",
        "image/png",
    }
    assert artifact["responses"]["200"]["content"]["application/json"]["schema"]["type"] == "object"
    kind = next(p for p in artifact["parameters"] if p["name"] == "kind")
    assert set(kind["schema"]["enum"]) == {"report", "boot_log", "screenshot"}
    assert artifact["responses"]["404"]["content"]["application/json"]["schema"][
        "$ref"
    ].endswith("ErrorResponse")


def test_openapi_board_projection_contract_is_concrete() -> None:
    schema = build_openapi_schema()
    components = schema["components"]["schemas"]

    summary = components["RunSummary"]
    assert summary["additionalProperties"] is False
    assert {
        "workspace_id",
        "workspace_name",
        "workspace_mode",
        "task_type",
        "ball_holder",
        "active_god",
        "pending_action",
        "worktree_branch",
        "status_badge",
        "is_archived",
    } <= set(summary["properties"])
    assert set(summary["properties"]["ball_holder"]["enum"]) == {"human", "agent", "done"}
    assert set(summary["properties"]["active_god"]["enum"]) == {
        "prometheus",
        "athena",
        "hephaestus",
        "momus",
        "argus",
    }
    pending = summary["properties"]["pending_action"]["anyOf"]
    assert {"clarify", "approve_spec", "approve_design", "approve_plan",
            "resolve_blocker", "review_delivery"} <= set(pending[0]["enum"])
    assert set(summary["properties"]["workspace_mode"]["enum"]) == {
        "brownfield",
        "greenfield",
        "spec_only",
    }
    assert set(summary["properties"]["task_type"]["enum"]) == {
        "feature",
        "fix",
        "refactor",
        "blueprint",
    }

    seat = components["PantheonSeatSummary"]
    assert seat["additionalProperties"] is False
    assert {
        "god",
        "role_label",
        "state",
        "active_count",
        "waiting_count",
        "oracle_ids",
    } == set(seat["properties"])
    assert set(seat["properties"]["state"]["enum"]) == {
        "idle",
        "working",
        "waiting_on_human",
    }

    listing = components["RunListResponse"]
    assert {
        "pantheon",
        "pending_ruling_total",
        "active_workspace_id",
    } <= set(listing["properties"])
    assert listing["properties"]["pantheon"]["items"]["$ref"].endswith(
        "PantheonSeatSummary"
    )
    assert {"pantheon", "pending_ruling_total"} <= set(listing["required"])

    workspace = components["WorkspaceSummary"]
    assert {"active_oracle_count", "pending_ruling_count"} <= set(workspace["properties"])

    parameters = {
        parameter["name"]
        for parameter in schema["paths"]["/api/runs"]["get"]["parameters"]
    }
    assert {
        "workspace_id",
        "workspace_mode",
        "task_type",
        "ball_holder",
        "include_archived",
    } <= parameters

    board = schema["paths"]["/api/board/stream"]["get"]
    assert set(board["responses"]) == {"200", "400"}
    assert set(board["responses"]["200"]["content"]) == {"text/event-stream"}
    assert board["responses"]["400"]["content"]["application/json"]["schema"][
        "$ref"
    ].endswith("ErrorResponse")
    assert {parameter["name"] for parameter in board["parameters"]} == {
        "workspace_id",
        "once",
    }


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
