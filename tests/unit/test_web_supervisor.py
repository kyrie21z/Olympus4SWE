"""HTTP + supervisor tests for workflow runs, provider status, and graph SSE events."""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

import httpx
import pytest

from olympus.core.contracts import ApprovalDecision, ApprovalRecord, ApprovalTarget, TaskPlan
from olympus.core.store import RunStore
from olympus.providers.base import ChatMessage, LLMResponse, MockLLMProvider, ToolSpec
from olympus.web.server import AriadneApp, create_http_server
from olympus.web.supervisor import RunSupervisor

GOAL_SPEC: dict[str, object] = {
    "spec_id": "spec-inventory",
    "version": 1,
    "goal": "Build an inventory REST API for operators",
    "deliverable_kind": "rest_api",
    "observable_surfaces": ["HTTP API"],
    "actors": ["operator"],
    "requirements": [
        {
            "id": "R1",
            "description": "Create an item through POST /items",
            "actor": "operator",
            "ears_criteria": [
                {
                    "id": "R1-C1",
                    "pattern": "event_driven",
                    "condition": "a POST /items request arrives",
                    "outcome": "persist the item and return HTTP 201",
                }
            ],
        },
        {
            "id": "R2",
            "description": "Read a stored item",
            "actor": "operator",
            "ears_criteria": [
                {
                    "id": "R2-C1",
                    "pattern": "ubiquitous",
                    "outcome": "return HTTP 200 with the stored item",
                }
            ],
        },
    ],
    "out_of_scope": ["authentication", "bulk import"],
    "constraints": ["python 3.13"],
}

DESIGN_SPEC: dict[str, object] = {
    "architecture_summary": "Thin HTTP edge over a storage core",
    "spec_id": "spec-inventory",
    "spec_hash": "sha256:placeholder",
    "shared_invariants": [
        {
            "id": "INV-ERR-1",
            "category": "error_envelope",
            "rule": "Return JSON {'error': {'code': str, 'message': str}} on every 4xx/5xx",
            "enforcement_hint": "shared error handler",
        }
    ],
    "shared_contract_paths": ["src/contracts.py"],
    "execution_strategy": "contract_first_parallel",
    "strategy_rationale": (
        "API and Store share only src/contracts.py pinned in Wave 0, enabling safe parallel Wave 1."
    ),
    "modules": [
        {
            "id": "M1",
            "name": "API",
            "responsibility": "HTTP routes",
            "requirement_refs": ["R1"],
            "target_paths": ["src/api/"],
        },
        {
            "id": "M2",
            "name": "Store",
            "responsibility": "Persistence",
            "requirement_refs": ["R2"],
            "target_paths": ["src/core/"],
        },
    ],
    "data_models": [{"name": "Item", "fields": ["id", "name"], "owner_module_id": "M2"}],
    "interfaces": [
        {
            "id": "IF1",
            "provider_module_id": "M2",
            "consumer_module_ids": ["M1"],
            "signature": "create_item(name: str) -> Item",
            "input_schema": [
                {"name": "name", "type_expr": "str", "required": True, "constraints": "non-empty"}
            ],
            "output_schema": [
                {"name": "id", "type_expr": "str", "required": True},
                {"name": "name", "type_expr": "str", "required": True},
            ],
            "data_entity_refs": ["Item"],
            "error_cases": ["duplicate name"],
        }
    ],
}

TASK_PLAN: dict[str, object] = {
    "plan_id": "plan-inventory",
    "spec_id": "spec-inventory",
    "spec_hash": "sha256:placeholder",
    "design_id": "design-placeholder",
    "design_hash": "sha256:placeholder",
    "tasks": [
        {
            "task_id": "T0",
            "title": "Interface contracts",
            "objective": "Pin the storage interface",
            "requirement_refs": ["R1", "R2"],
            "design_refs": ["IF1"],
            "write_scope": ["src/contracts.py"],
            "kind": "contract",
            "role": "architect",
            "deliverables": ["src/contracts.py"],
            "acceptance_checks": [
                {
                    "id": "T0-c",
                    "kind": "command",
                    "description": "import contracts",
                    "command": "python -c 'pass'",
                }
            ],
        },
        {
            "task_id": "T1",
            "title": "HTTP API",
            "objective": "Expose POST /items",
            "requirement_refs": ["R1"],
            "design_refs": ["M1"],
            "depends_on": ["T0"],
            "write_scope": ["src/api/"],
            "acceptance_checks": [
                {
                    "id": "T1-c",
                    "kind": "command",
                    "description": "api tests",
                    "command": "python -c 'pass'",
                }
            ],
        },
        {
            "task_id": "T2",
            "title": "Storage core",
            "objective": "Persist items",
            "requirement_refs": ["R2"],
            "design_refs": ["M2"],
            "depends_on": ["T0"],
            "write_scope": ["src/core/"],
            "acceptance_checks": [
                {
                    "id": "T2-c",
                    "kind": "command",
                    "description": "store tests",
                    "command": "python -c 'pass'",
                }
            ],
        },
        {
            "task_id": "T3",
            "title": "Acceptance suite",
            "objective": "Prove both requirements",
            "requirement_refs": ["R1", "R2"],
            "design_refs": ["M1", "M2"],
            "depends_on": ["T1", "T2"],
            "write_scope": ["tests/"],
            "kind": "test",
            "role": "qa",
            "deliverables": ["tests/test_acceptance.py"],
            "acceptance_checks": [
                {
                    "id": "T3-c",
                    "kind": "command",
                    "description": "acceptance tests",
                    "command": "python -c 'pass'",
                }
            ],
        },
    ],
}


def _init_git_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# inventory\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Dev"], cwd=root, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)


def _responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
    del tools
    system = messages[0].content
    if "Requirements Alignment Architect" in system:
        return LLMResponse(content=json.dumps(GOAL_SPEC))
    if "White-Box Solution Architect" in system:
        return LLMResponse(content=json.dumps(DESIGN_SPEC))
    if "Stage 2 Task Planner" in system:
        return LLMResponse(content=json.dumps(TASK_PLAN))
    raise AssertionError(f"Unexpected system prompt: {system[:120]}")


@pytest.fixture()
def supervisor_app(tmp_path: Path) -> Iterator[tuple[AriadneApp, RunSupervisor, str]]:
    _init_git_repo(tmp_path)
    store = RunStore(tmp_path / ".olympus" / "olympus.db")
    supervisor = RunSupervisor(
        store=store,
        workspace_root=tmp_path,
        provider_builder=lambda **kwargs: MockLLMProvider(responder=_responder),
    )
    app = AriadneApp(store=store, workspace_root=tmp_path, supervisor=supervisor)
    httpd = create_http_server(app, host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield app, supervisor, f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        supervisor.shutdown()


def test_provider_and_workflow_discovery_endpoints(
    supervisor_app: tuple[AriadneApp, RunSupervisor, str],
) -> None:
    _, _, base_url = supervisor_app
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        providers = client.get("/api/providers")
        assert providers.status_code == 200
        payload = providers.json()
        assert payload["ok"] is True
        ids = {provider["id"] for provider in payload["providers"]}
        assert {"openai", "anthropic"} <= ids
        for provider in payload["providers"]:
            assert isinstance(provider["configured"], bool)
            # Only metadata is exposed: identifiers, dialect, model, and env var NAMES.
            assert set(provider) == {
                "id",
                "dialect",
                "default_model",
                "configured",
                "env_vars",
            }
            # env_vars are variable NAMES only, never the configured secret values.
            assert all(
                name.isupper() and name.replace("_", "").isalnum()
                for name in provider["env_vars"]
            )

        workflows = client.get("/api/workflows")
        assert workflows.status_code == 200
        data = workflows.json()
        assert data["ok"] is True
        names = {workflow["name"] for workflow in data["workflows"]}
        assert {"product_blueprint", "full_sdlc_tdd", "repo_fast_patch"} <= names
        blueprint = next(w for w in data["workflows"] if w["name"] == "product_blueprint")
        assert blueprint["topology"] == "modular_product"
        assert blueprint["stop_after"] == "plan"
        assert blueprint["graph"]["nodes"]


def test_post_runs_requires_raw_input(
    supervisor_app: tuple[AriadneApp, RunSupervisor, str],
) -> None:
    _, _, base_url = supervisor_app
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        res = client.post("/api/runs", json={"title": "empty"})
        assert res.status_code == 400
        assert "raw_input" in res.json()["error"]


def test_full_product_blueprint_flow_over_http(
    supervisor_app: tuple[AriadneApp, RunSupervisor, str],
) -> None:
    app, _, base_url = supervisor_app
    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        created = client.post(
            "/api/runs",
            json={
                "title": "Inventory API",
                "raw_input": "Build an inventory REST API so operators can create and read items",
                "workflow_name": "product_blueprint",
                "provider_config": {"provider": "mock"},
                "sync": True,
            },
        )
        assert created.status_code == 200, created.text
        body = created.json()
        run_id = body["run_id"]
        assert body["status"] == "awaiting_human"
        assert body["paused"]["node"] == "gate_spec"

        proj = app.get_run_payload(run_id)["projection"]
        assert proj["triage"]["maturity"] == "raw_intent"
        assert proj["spec"]["status"] == "awaiting_human"
        assert proj["spec_approved"] is False

        # Gate 1: clarify then approve.
        clarified = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "clarify",
                "answers": {},
                "sync": True,
            },
        )
        assert clarified.status_code == 200, clarified.text
        assert clarified.json()["command_result"]["version"] == 2

        approved_spec = client.post(
            f"/api/runs/{run_id}/command",
            json={"action": "approve_spec", "sync": True},
        )
        assert approved_spec.status_code == 200, approved_spec.text
        workflow = approved_spec.json()["command_result"]["workflow"]
        assert workflow["paused"]["node"] == "gate_design"

        proj = app.get_run_payload(run_id)["projection"]
        assert proj["spec_approved"] is True
        assert proj["design"] is not None
        assert proj["design_approved"] is False
        assert proj["design"]["modules"]

        # Gate 1.5: approve the design, planner decomposes the modules.
        approved_design = client.post(
            f"/api/runs/{run_id}/command",
            json={"action": "approve_design", "sync": True},
        )
        assert approved_design.status_code == 200, approved_design.text
        assert approved_design.json()["command_result"]["workflow"]["paused"]["node"] == "gate_plan"

        proj = app.get_run_payload(run_id)["projection"]
        assert proj["design_approved"] is True
        plan = proj["plan"]
        assert plan["design_id"] == proj["design"]["design_id"]
        assert [task["task_id"] for task in plan["tasks"]] == ["T0", "T1", "T2", "T3"]
        projected = {node["task_id"]: node for node in proj["dag"]["nodes"]}
        assert projected["T1"]["design_refs"] == ["M1"]
        assert projected["T0"]["kind"] == "contract"
        assert projected["T3"]["role"] == "qa"
        assert projected["T3"]["deliverables"] == ["tests/test_acceptance.py"]

        # Gate 2: approving the plan completes the blueprint workflow.
        approved_plan = client.post(
            f"/api/runs/{run_id}/command",
            json={"action": "approve_plan", "sync": True},
        )
        assert approved_plan.status_code == 200, approved_plan.text
        assert approved_plan.json()["projection"]["status"] == "completed"

        proj = app.get_run_payload(run_id)["projection"]
        assert proj["plan_approved"] is True
        assert proj["workflow"]["finished"] is True
        node_status = {node["name"]: node["status"] for node in proj["workflow"]["nodes"]}
        assert node_status["intake"] == "completed"
        assert node_status["gate_plan"] == "completed"


def test_direct_design_spec_bypass(
    supervisor_app: tuple[AriadneApp, RunSupervisor, str],
) -> None:
    app, supervisor, base_url = supervisor_app
    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        created = client.post(
            "/api/runs",
            json={
                "title": "Bypass blueprint",
                "raw_input": json.dumps(DESIGN_SPEC),
                "provider_config": {"provider": "mock"},
                "sync": True,
            },
        )
        assert created.status_code == 200, created.text
        run_id = created.json()["run_id"]
        # A frozen DesignSpec skips align entirely and lands on the design gate.
        assert created.json()["paused"]["node"] == "gate_design"

        proj = app.get_run_payload(run_id)["projection"]
        assert proj["triage"]["maturity"] == "design_spec_ready"
        assert proj["spec"] is None
        assert proj["design"] is not None

        approved = client.post(
            f"/api/runs/{run_id}/command",
            json={"action": "approve_design", "sync": True},
        )
        assert approved.status_code == 200, approved.text
        proj = app.get_run_payload(run_id)["projection"]
        assert proj["plan"] is not None
        assert proj["plan"]["design_id"] == proj["design"]["design_id"]
        assert proj["plan"]["spec_id"] is None

        assert supervisor.is_supervised(run_id) is True


def test_sse_stream_carries_graph_events(
    supervisor_app: tuple[AriadneApp, RunSupervisor, str],
) -> None:
    _, _, base_url = supervisor_app
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        created = client.post(
            "/api/runs",
            json={
                "title": "SSE run",
                "raw_input": "Build an inventory REST API for operators",
                "provider_config": {"provider": "mock"},
                "sync": True,
            },
        )
        run_id = created.json()["run_id"]

        stream = client.get(f"/api/runs/{run_id}/stream", params={"once": "1"})
        assert stream.status_code == 200
        text = stream.text
        assert "event: state" in text
        data_line = next(
            line for line in text.splitlines() if line.startswith("data: ")
        )
        payload = json.loads(data_line[len("data: ") :])
        event_types = [event["event_type"] for event in payload["events"]]
        assert "graph.started" in event_types
        assert "graph.node_started" in event_types
        assert "graph.paused" in event_types
        assert "run.created" in event_types


def test_supervisor_restart_recovery_and_env_file_provider_detection(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    env_file = tmp_path / "custom.env"
    env_file.write_text("CUSTOM_SECRET_TOKEN=dummy-local-test-value\n", encoding="utf-8")
    olympus_dir = tmp_path / ".olympus"
    olympus_dir.mkdir(parents=True, exist_ok=True)
    (olympus_dir / "providers.json").write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "id": "custom_env_provider",
                        "dialect": "openai-chat",
                        "default_model": "custom-model",
                        "base_url": "https://example.invalid/v1",
                        "api_key_env": "CUSTOM_SECRET_TOKEN",
                        "env_file": str(env_file),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    store = RunStore(olympus_dir / "olympus.db")
    sup1 = RunSupervisor(
        store=store,
        workspace_root=tmp_path,
        provider_builder=lambda **kwargs: MockLLMProvider(responder=_responder),
    )
    started = sup1.start_run(
        title="Short Title",
        raw_input="Build an inventory REST API so operators can create and read items",
        workflow_name="product_blueprint",
        sync=True,
    )
    run_id = started["run_id"]
    assert started["paused"]["node"] == "gate_spec"
    sup1.shutdown()

    # Simulate process restart with a brand-new RunSupervisor instance.
    sup2 = RunSupervisor(
        store=store,
        workspace_root=tmp_path,
        provider_builder=lambda **kwargs: MockLLMProvider(responder=_responder),
    )
    try:
        providers = {p["id"]: p for p in sup2.list_providers_status()}
        assert providers["custom_env_provider"]["configured"] is True
        assert sup2.is_supervised(run_id) is True

        app2 = AriadneApp(store=store, workspace_root=tmp_path, supervisor=sup2)
        res = app2.execute_command(run_id, {"action": "approve_spec", "sync": True})
        assert res["command_result"]["workflow"]["paused"]["node"] == "gate_design"
        assert res["projection"]["design"] is not None
    finally:
        sup2.shutdown()


def test_promote_completed_blueprint_to_execute_and_blocked_status(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    store = RunStore(tmp_path / ".olympus" / "olympus.db")

    def _exec_responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        system = messages[0].content
        if "Execution Engineer (Hephaestus)" in system:
            return LLMResponse(content="Implemented task changes cleanly.")
        if "Independent Verifier (Argus)" in system:
            return LLMResponse(
                content=json.dumps(
                    {
                        "task_id": "T0",
                        "attempt": 1,
                        "passed": True,
                        "summary": "All checks verified.",
                        "findings": [],
                    }
                )
            )
        return _responder(messages, tools)

    supervisor = RunSupervisor(
        store=store,
        workspace_root=tmp_path,
        provider_builder=lambda **kwargs: MockLLMProvider(responder=_exec_responder),
    )
    app = AriadneApp(store=store, workspace_root=tmp_path, supervisor=supervisor)
    httpd = create_http_server(app, host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(base_url=base_url, timeout=60.0) as client:
            created = client.post(
                "/api/runs",
                json={
                    "title": "Blueprint Promote Flow",
                    "raw_input": (
                        "Build an inventory REST API so operators can create and read items"
                    ),
                    "workflow_name": "product_blueprint",
                    "provider_config": {"provider": "mock"},
                    "sync": True,
                },
            ).json()
            run_id = created["run_id"]
            assert 'id="btn-promote-execute"' in client.get("/").text
            client.post(
                f"/api/runs/{run_id}/command", json={"action": "approve_spec", "sync": True}
            )
            client.post(
                f"/api/runs/{run_id}/command", json={"action": "approve_design", "sync": True}
            )
            plan_res = client.post(
                f"/api/runs/{run_id}/command", json={"action": "approve_plan", "sync": True}
            ).json()["projection"]
            assert plan_res["status"] == "completed"
            assert plan_res["blueprint_completed"] is True
            assert [s["status"] for s in plan_res["stage_pipeline"]] == [
                "done",
                "done",
                "done",
                "skipped",
                "skipped",
            ]
            assert plan_res["hitl_banner"]["kind"] == "blueprint_completed"

            # Promote completed blueprint directly into full_sdlc_tdd sandbox execution.
            promoted = client.post(
                f"/api/runs/{run_id}/command",
                json={
                    "operation_id": "op-promote-1",
                    "action": "promote_to_execute",
                    "workflow_name": "full_sdlc_tdd",
                    "expected_state_revision": plan_res["state_revision"],
                    "sync": True,
                },
            )
            assert promoted.status_code == 200, promoted.text
            exec_proj = promoted.json()["projection"]
            assert exec_proj["workflow_name"] == "full_sdlc_tdd"
            assert exec_proj["status"] == "completed"
            assert exec_proj["budget"]["max_calls"] >= 100
            assert [s["status"] for s in exec_proj["stage_pipeline"]] == ["done"] * 5
            assert all(n["status"] == "accepted" for n in exec_proj["dag"]["nodes"])

            # Verify a failing execution preserves AWAITING_HUMAN instead of COMPLETED.
            failing_plan_dict = json.loads(json.dumps(TASK_PLAN))
            failing_plan_dict["tasks"] = [
                {
                    **failing_plan_dict["tasks"][0],
                    "max_cycles": 1,
                    "acceptance_checks": [
                        {
                            "id": "T0-fail",
                            "kind": "command",
                            "description": "always fail",
                            "command": "python -c 'raise SystemExit(1)'",
                        }
                    ],
                }
            ]
            failing_store_run = store.create_run(
                title="Blocked Exec Preserve",
                workflow_name="full_sdlc_tdd",
                provider_config={"provider": "mock"},
            )
            snap_done = store.get_run(run_id)
            assert snap_done["spec"] is not None
            assert snap_done["design"] is not None

            store.save_spec(failing_store_run, snap_done["spec"], actor="align")
            store.record_approval(
                ApprovalRecord(
                    run_id=failing_store_run,
                    target=ApprovalTarget.GOAL_SPEC,
                    target_id=snap_done["spec"].spec_id,
                    target_hash=snap_done["spec"].content_hash,
                    approver="zeus",
                    decision=ApprovalDecision.APPROVED,
                )
            )
            store.save_design(failing_store_run, snap_done["design"], actor="design")
            store.record_approval(
                ApprovalRecord(
                    run_id=failing_store_run,
                    target=ApprovalTarget.DESIGN_SPEC,
                    target_id=snap_done["design"].design_id,
                    target_hash=snap_done["design"].content_hash,
                    approver="zeus",
                    decision=ApprovalDecision.APPROVED,
                )
            )
            fail_plan = TaskPlan.model_validate(
                {
                    **failing_plan_dict,
                    "spec_hash": snap_done["spec"].content_hash,
                    "design_id": snap_done["design"].design_id,
                    "design_hash": snap_done["design"].content_hash,
                }
            )
            store.save_plan(failing_store_run, fail_plan, actor="orchestrate")
            store.record_approval(
                ApprovalRecord(
                    run_id=failing_store_run,
                    target=ApprovalTarget.TASK_PLAN,
                    target_id=fail_plan.plan_id,
                    target_hash=fail_plan.content_hash,
                    approver="zeus",
                    decision=ApprovalDecision.APPROVED,
                )
            )
            store.append_event(
                run_id=failing_store_run,
                stage="graph",
                event_type="graph.node_finished",
                actor="supervisor",
                payload={"node": "gate_plan"},
            )
            blocked_res = supervisor.resume_run(failing_store_run, sync=True)
            assert blocked_res["status"] == "awaiting_human"
            blocked_proj = app.get_run_payload(failing_store_run)["projection"]
            assert blocked_proj["status"] == "awaiting_human"
            assert blocked_proj["hitl_banner"]["kind"] == "task_blocked"
    finally:
        httpd.shutdown()
        httpd.server_close()
        supervisor.shutdown()
