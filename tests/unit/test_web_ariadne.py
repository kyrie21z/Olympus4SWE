"""Unit and HTTP/SSE integration tests for the M4 Ariadne Web Console."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import httpx

from olympus.core.contracts import (
    AcceptanceCheck,
    AmbiguityItem,
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    CheckKind,
    DataEntity,
    DeliverableKind,
    DesignModule,
    DesignSpec,
    EarsPattern,
    GoalSpec,
    InterfaceContract,
    RequirementItem,
    RunStatus,
    SpecStatus,
    TaskKind,
    TaskNode,
    TaskPlan,
    TaskStatus,
)
from olympus.core.store import RunStore
from olympus.graph.loader import build_graph, load_workflow
from olympus.providers.types import LLMResponse, MockLLMProvider
from olympus.stages.design import DesignStage
from olympus.web.projection import build_run_projection, seed_demo_runs
from olympus.web.server import AriadneApp, create_http_server
from olympus.web.supervisor import RunSupervisor


def _seed_gated_run(store: RunStore) -> str:
    """A blue-print run that sits before every human gate with one open ambiguity."""
    run_id = store.create_run(
        title="受控编排闸门测试",
        workflow_name="product_blueprint",
        provider_config={"provider": "mock"},
    )
    spec = GoalSpec(
        spec_id="spec-gated-1",
        version=1,
        goal="实现候补递补服务并验证四关口人工批准链路。",
        requirements=[
            RequirementItem(
                id="FR-100",
                description="已报名者取消后自动递补最早的有效候补者",
                acceptance_criteria=["递补在同一事务内完成且状态可观测"],
                source_ref="REQ-Test / D-SM-01",
            )
        ],
        out_of_scope=["不发送短信或邮件通知"],
        constraints=["报名、取消与递补共用单一写事务"],
        open_ambiguities=[
            AmbiguityItem(
                id="AMB-1",
                question="同一账号取消后是否允许重新报名？",
                impact="影响唯一索引与状态迁移设计",
                options=["允许重新入队并重新排号", "禁止再次报名"],
                default_assumption="允许重新入队并重新排号",
                blocking=True,
                answer=None,
            )
        ],
        status=SpecStatus.DRAFT,
    )
    store.save_spec(run_id, spec, actor="align")

    design = DesignSpec(
        design_id="design-gated-1",
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        architecture_summary="单写服务 + 显式事务边界，接口与数据所有权清晰。",
        modules=[
            DesignModule(
                id="MOD-WAITLIST",
                name="候补递补核心",
                responsibility="容量校验、候补入队与取消递补",
                requirement_refs=["FR-100"],
                target_paths=["src/waitlist/service.py"],
            ),
            DesignModule(
                id="MOD-API",
                name="查询与操作接口",
                responsibility="对外暴露报名状态与排位查询",
                requirement_refs=["FR-100"],
                target_paths=["src/waitlist/api.py"],
            ),
        ],
        data_models=[
            DataEntity(
                name="Seat",
                fields=["seat_id", "status", "queued_at"],
                owner_module_id="MOD-WAITLIST",
            )
        ],
        interfaces=[
            InterfaceContract(
                id="IF-1",
                provider_module_id="MOD-WAITLIST",
                consumer_module_ids=["MOD-API"],
                protocol="fn",
                signature="promote_next(activity_id: str) -> Seat",
                error_cases=["NoWaitlistedCandidate"],
            )
        ],
        risks_and_tradeoffs=["单一 SQLite 写锁串行化吞吐"],
    )
    store.save_design(run_id, design, actor="design")

    plan = TaskPlan(
        plan_id="plan-gated-1",
        design_id=design.design_id,
        design_hash=design.content_hash,
        tasks=[
            TaskNode(
                task_id="T01",
                title="实现取消后自动递补",
                objective="在单事务内完成状态迁移与候补递补",
                requirement_refs=["FR-100"],
                write_scope=["src/waitlist/service.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T01-C1",
                        kind=CheckKind.COMMAND,
                        description="pytest 验证并发递补",
                        command="pytest -q",
                    )
                ],
                design_refs=["MOD-WAITLIST"],
                role="backend",
                kind=TaskKind.IMPLEMENT,
                deliverables=["src/waitlist/service.py"],
                status=TaskStatus.READY,
            )
        ],
    )
    store.save_plan(run_id, plan, actor="orchestrate")
    return run_id


def test_seed_demo_and_projection_invariants(tmp_path: Path) -> None:
    db_path = tmp_path / ".olympus" / "olympus.db"
    store = RunStore(db_path)

    seeded = seed_demo_runs(store, tmp_path)
    assert seeded == ["run-campus-waitlist"]
    # Second call is idempotent
    assert seed_demo_runs(store, tmp_path) == []

    proj = build_run_projection(store, tmp_path, "run-campus-waitlist")
    assert proj["run_id"] == "run-campus-waitlist"
    assert proj["spec_approved"] is True
    assert proj["plan_approved"] is True
    assert proj["budget"]["calls_used"] == 41
    assert proj["budget"]["max_calls"] == 120
    assert proj["progress"]["accepted_tasks"] == 4
    assert proj["progress"]["total_tasks"] == 9

    assert proj["hitl_banner"] is not None
    assert proj["hitl_banner"]["task_id"] == "T03-R1"
    assert "阻塞：T03-R1" in proj["hitl_banner"]["headline"]

    nodes_by_id = {n["task_id"]: n for n in proj["dag"]["nodes"]}
    assert nodes_by_id["T01"]["layer"] == 0
    assert nodes_by_id["T03"]["layer"] == 1
    assert nodes_by_id["T03-R1"]["layer"] == 2
    assert nodes_by_id["T06"]["layer"] == 3
    assert nodes_by_id["T08"]["layer"] == 4
    assert "promote_next_waitlisted" in nodes_by_id["T03-R1"]["artifacts"]["diff_patch"]
    assert "同一候补者被递补 2 次" in nodes_by_id["T03-R1"]["artifacts"]["check_logs"]["T03-R1-cmd"]
    assert len(nodes_by_id["T03-R1"]["artifacts"]["verdicts"]) == 1

    blocked_edges = [e for e in proj["dag"]["edges"] if e["blocked_edge"]]
    assert any(e["from"] == "T03-R1" and e["to"] == "T06" for e in blocked_edges)

    matrix_by_req = {r["requirement_id"]: r for r in proj["traceability_matrix"]}
    assert matrix_by_req["FR-003"]["verification_status"] == "failed"
    assert matrix_by_req["FR-005"]["verification_status"] == "pending"

    # Every node carries the cockpit's per-task panels
    for node in proj["dag"]["nodes"]:
        assert isinstance(node["subagent_stream"], list)
        assert isinstance(node["terminal_lines"], list)
        assert len(node["terminal_lines"]) <= 4
        assert node["receipt"]["turn_id"]
        assert node["receipt"]["one_line"]
        assert node["manifest_md"].startswith("# MANIFEST")

    blocked_node = nodes_by_id["T03-R1"]
    assert [step["tool"] for step in blocked_node["subagent_stream"]] == [
        "read_file",
        "read_file",
        "edit_file",
        "run_bash",
        "read_file",
    ]
    assert blocked_node["subagent_stream"][3]["status"] == "error"
    assert blocked_node["receipt"]["task_id"] == "T03-R1"
    assert blocked_node["receipt"]["artifacts"]["checks_total"] == 2
    assert "MANIFEST — T03-R1" in blocked_node["manifest_md"]
    assert blocked_node["terminal_lines"][-1].startswith("1 failed, 3 passed")

    # Seeds write the real trace + manifest so the frontend renders recorded data
    t03_art = tmp_path / ".olympus" / "runs" / "run-campus-waitlist" / "artifacts" / "T03-R1"
    assert (t03_art / "attempt-2" / "worker.jsonl").exists()
    assert (t03_art / "attempt-2" / "MANIFEST.md").exists()

    # Waterfall: chronological Align -> Hephaestus -> Momus -> Argus turns
    turns = proj["waterfall_turns"]
    assert [turn["actor"] for turn in turns][0] == "Align"
    assert [turn["actor"] for turn in turns][-1] == "Argus"
    actors = {turn["actor"] for turn in turns}
    assert {"Align", "Hephaestus-1", "Hephaestus-2", "Momus", "Argus"} <= actors
    assert {turn["task_id"] for turn in turns} >= {"T03-R1", "T05"}
    for turn in turns:
        assert turn["receipt"]["turn_id"] == turn["turn_id"]
        assert turn["steps"] >= 1
        assert turn["duration_ms"] > 0


def test_ariadne_http_sse_and_idempotent_hitl_commands(tmp_path: Path) -> None:
    db_path = tmp_path / ".olympus" / "olympus.db"
    store = RunStore(db_path)
    seed_demo_runs(store, tmp_path)
    app = AriadneApp(store=store, workspace_root=tmp_path)

    httpd = create_http_server(app, host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"

    try:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            # 1. Index HTML & healthz
            html_res = client.get("/")
            assert html_res.status_code == 200
            assert 'closedby="any"' in html_res.text
            assert "需求追踪矩阵" in html_res.text
            assert 'href="/static/tokens.css"' in html_res.text
            assert 'href="/static/style.css"' in html_res.text
            assert 'src="/static/ui.js"' in html_res.text
            assert 'src="/static/app.js"' in html_res.text

            # 1b. Static assets are served with the right MIME types and no caching
            tokens_res = client.get("/static/tokens.css")
            assert tokens_res.status_code == 200
            assert tokens_res.headers["content-type"].startswith("text/css")
            assert tokens_res.headers["cache-control"] == "no-store"
            assert "--surface-bg" in tokens_res.text

            css_res = client.get("/static/style.css")
            assert css_res.status_code == 200
            assert css_res.headers["content-type"].startswith("text/css")
            assert css_res.headers["cache-control"] == "no-store"
            assert "scrollbar-color" in css_res.text

            ui_res = client.get("/static/ui.js")
            assert ui_res.status_code == 200
            assert "javascript" in ui_res.headers["content-type"]
            assert "function watchSlots(" in ui_res.text

            app_res = client.get("/static/app.js")
            assert app_res.status_code == 200
            assert "javascript" in app_res.headers["content-type"]
            assert "watchSlots();" in app_res.text

            assert client.get("/static/missing.css").status_code == 404
            assert client.get("/static/..%2fserver.py").status_code == 404

            health = client.get("/healthz").json()
            assert health["ok"] is True

            # 2. List runs & get projection
            runs_res = client.get("/api/runs").json()
            assert runs_res["default_run_id"] == "run-campus-waitlist"

            run_res = client.get("/api/runs/run-campus-waitlist").json()
            assert run_res["ok"] is True
            rev0 = int(run_res["projection"]["state_revision"])

            # 2b. The projection powers toolRow / subPane / receiptBlock / waterfall
            proj0 = run_res["projection"]
            t03r1 = next(n for n in proj0["dag"]["nodes"] if n["task_id"] == "T03-R1")
            assert t03r1["subagent_stream"][0]["tool"] == "read_file"
            assert t03r1["receipt"]["turn_id"] == "turn-t03-r1-a2"
            assert "MANIFEST — T03-R1" in t03r1["manifest_md"]
            assert len(t03r1["terminal_lines"]) <= 4
            assert proj0["waterfall_turns"][0]["actor"] == "Align"
            assert any(turn["actor"] == "Argus" for turn in proj0["waterfall_turns"])

            # 3. SSE stream (once=1)
            sse_res = client.get("/api/runs/run-campus-waitlist/stream?after_seq=0&once=1")
            assert sse_res.status_code == 200
            assert "text/event-stream" in sse_res.headers["content-type"]
            assert "event: state" in sse_res.text

            # 4. Pause & idempotency check
            pause_body = {
                "operation_id": "op-pause-1",
                "action": "pause",
                "expected_state_revision": rev0,
                "actor": "zeus",
            }
            p1 = client.post("/api/runs/run-campus-waitlist/command", json=pause_body)
            assert p1.status_code == 200
            rev1 = int(p1.json()["projection"]["state_revision"])
            assert rev1 == rev0 + 1
            assert p1.json()["projection"]["status"] == "paused"

            # Re-sending identical operation_id returns cached response without bumping revision
            p1_dup = client.post("/api/runs/run-campus-waitlist/command", json=pause_body)
            assert p1_dup.status_code == 200
            assert int(p1_dup.json()["projection"]["state_revision"]) == rev1

            # Stale state_revision with a new operation_id is rejected with 409 Conflict
            stale_res = client.post(
                "/api/runs/run-campus-waitlist/command",
                json={
                    "operation_id": "op-stale-99",
                    "action": "resume",
                    "expected_state_revision": rev0,
                },
            )
            assert stale_res.status_code == 409

            # 5. Reject accept_risk without reason
            bad_risk = client.post(
                "/api/runs/run-campus-waitlist/command",
                json={
                    "operation_id": "op-bad-risk",
                    "action": "accept_risk",
                    "task_id": "T03-R1",
                    "reason": "   ",
                    "expected_state_revision": rev1,
                },
            )
            assert bad_risk.status_code == 400

            # 6. Test accept_risk distinguishing unverified_risk_accepted from verified
            risk_res = client.post(
                "/api/runs/run-campus-waitlist/command",
                json={
                    "operation_id": "op-accept-risk-t03r1",
                    "action": "accept_risk",
                    "task_id": "T03-R1",
                    "reason": "首期单管理员手动取消，接受并发风险放行",
                    "expected_state_revision": rev1,
                },
            )
            assert risk_res.status_code == 200
            proj_after_risk = risk_res.json()["projection"]
            t03_node = next(n for n in proj_after_risk["dag"]["nodes"] if n["task_id"] == "T03-R1")
            assert t03_node["status"] == TaskStatus.ACCEPTED.value
            assert t03_node["accepted_risk_reason"] == "首期单管理员手动取消，接受并发风险放行"

            # Mark T06 & T08 accepted to verify FR-003/FR-004 show unverified_risk_accepted
            snap = store.get_run("run-campus-waitlist")
            plan = snap["plan"]
            assert plan is not None
            updated_tasks = [
                t.model_copy(update={"status": TaskStatus.ACCEPTED})
                if t.task_id in ("T06", "T08")
                else t
                for t in plan.tasks
            ]
            store.save_plan(
                "run-campus-waitlist",
                plan.model_copy(update={"tasks": updated_tasks}),
                actor="test",
            )
            final_proj = client.get("/api/runs/run-campus-waitlist").json()["projection"]
            matrix = {r["requirement_id"]: r for r in final_proj["traceability_matrix"]}
            assert matrix["FR-003"]["verification_status"] == "unverified_risk_accepted"
            assert matrix["FR-003"]["verification_label"] == "未验证 · 人工接受"
            assert "首期单管理员手动取消" in matrix["FR-003"]["evidence_note"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2.0)


def test_projection_workflow_traceability_and_triage(tmp_path: Path) -> None:
    store = RunStore(tmp_path / ".olympus" / "olympus.db")
    run_id = _seed_gated_run(store)

    store.append_event(
        run_id=run_id,
        stage="align",
        event_type="intake.triaged",
        actor="intake",
        payload={
            "maturity": "goal_spec_ready",
            "reasons": [
                "输入已包含目标、可观察面与验收标准",
                "伪 DesignSpec 片段已降级为 GoalSpec 证据",
            ],
            "has_goal": True,
            "has_design": False,
        },
    )
    store.append_event(
        run_id=run_id,
        stage="graph",
        event_type="graph.node_started",
        actor="supervisor",
        payload={"node": "intake"},
    )
    store.append_event(
        run_id=run_id,
        stage="graph",
        event_type="graph.node_finished",
        actor="supervisor",
        payload={"node": "intake"},
    )
    store.append_event(
        run_id=run_id,
        stage="graph",
        event_type="graph.paused",
        actor="supervisor",
        payload={"node": "gate_spec", "reason": "human_gate"},
    )

    proj = build_run_projection(store, tmp_path, run_id)

    assert proj["triage"]["maturity"] == "goal_spec_ready"
    assert any("降级" in reason for reason in proj["triage"]["reasons"])

    workflow = proj["workflow"]
    assert workflow["name"] == "product_blueprint"
    assert workflow["topology"] == "modular_product"
    assert [node["name"] for node in workflow["graph"]["nodes"]][:4] == [
        "intake",
        "align",
        "critic",
        "gate_spec",
    ]
    assert workflow["node_statuses"]["intake"] == "completed"
    assert workflow["node_statuses"]["gate_spec"] == "paused"
    assert workflow["paused"]["reason"] == "human_gate"
    assert any(edge["conditional"] for edge in workflow["graph"]["edges"])
    assert all(isinstance(edge["conditional"], bool) for edge in workflow["graph"]["edges"])

    row = proj["traceability_matrix"][0]
    assert row["requirement_id"] == "FR-100"
    assert row["design_module_ids"] == ["MOD-WAITLIST", "MOD-API"]
    assert [module["name"] for module in row["design_modules"]] == [
        "候补递补核心",
        "查询与操作接口",
    ]
    assert row["task_ids"] == ["T01"]
    assert row["verification_status"] == "pending"

    design = proj["design"]
    assert design["modules"][0]["target_paths"] == ["src/waitlist/service.py"]
    assert design["data_models"][0]["owner_module_id"] == "MOD-WAITLIST"
    assert design["interfaces"][0]["provider_module_id"] == "MOD-WAITLIST"
    assert design["interfaces"][0]["consumer_module_ids"] == ["MOD-API"]


def test_ariadne_v3_workbench_http_flow(tmp_path: Path) -> None:
    db_path = tmp_path / ".olympus" / "olympus.db"
    store = RunStore(db_path)
    run_id = _seed_gated_run(store)
    app = AriadneApp(store=store, workspace_root=tmp_path)

    httpd = create_http_server(app, host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            # 1. Every v3 workbench control is present in the shell
            html = client.get("/").text
            for marker in (
                'id="btn-new-run"',
                'id="thread-list"',
                'id="new-run-dialog"',
                'id="new-run-workflow"',
                'id="new-run-provider"',
                'id="new-run-model"',
                'id="new-run-title"',
                'id="new-run-input"',
                'id="new-run-file"',
                'id="triage-banner"',
                'id="clarify-panel"',
                'id="workflow-svg"',
                'id="view-design"',
                'id="btn-approve-spec"',
                'id="btn-approve-design"',
                'id="btn-approve-plan"',
                'data-view="design"',
            ):
                assert marker in html, f"index.html missing {marker}"

            # 2. Workflow catalog and provider status feed the new-run modal
            workflows = client.get("/api/workflows").json()["workflows"]
            by_name = {workflow["name"]: workflow for workflow in workflows}
            assert {"product_blueprint", "full_sdlc_tdd", "repo_fast_patch"} <= set(by_name)
            blueprint = by_name["product_blueprint"]
            assert blueprint["topology"] == "modular_product"
            assert blueprint["stop_after"] == "plan"
            assert blueprint["description"]
            assert any(edge.get("conditional") for edge in blueprint["graph"]["edges"])

            providers = client.get("/api/providers").json()["providers"]
            assert isinstance(providers, list)
            for provider in providers:
                assert {"id", "dialect", "default_model", "configured"} <= set(provider)

            # 3. The seeded run starts with every gate open
            proj = client.get(f"/api/runs/{run_id}").json()["projection"]
            assert proj["hitl_banner"]["kind"] == "spec_gate"
            assert proj["spec_approved"] is False
            assert proj["design_approved"] is False
            assert proj["plan_approved"] is False
            assert proj["workflow"]["node_statuses"]["intake"] == "pending"
            assert proj["traceability_matrix"][0]["design_module_ids"] == [
                "MOD-WAITLIST",
                "MOD-API",
            ]

            # 4. clarify advances the GoalSpec to v2 with the answer recorded
            clarify = client.post(
                f"/api/runs/{run_id}/command",
                json={
                    "operation_id": "op-clarify-1",
                    "action": "clarify",
                    "answers": {"AMB-1": "允许重新入队并重新排号"},
                    "expected_state_revision": proj["state_revision"],
                    "auto_advance": False,
                },
            )
            assert clarify.status_code == 200
            clarified = clarify.json()["projection"]
            assert clarified["spec"]["version"] == 2
            assert clarified["spec"]["open_ambiguities"][0]["answer"] == "允许重新入队并重新排号"
            assert clarified["spec_approved"] is False

            # 5. approve_spec -> approve_design -> approve_plan through HTTP
            approve_spec = client.post(
                f"/api/runs/{run_id}/command",
                json={
                    "operation_id": "op-approve-spec",
                    "action": "approve_spec",
                    "expected_state_revision": clarified["state_revision"],
                    "auto_advance": False,
                },
            )
            assert approve_spec.status_code == 200
            spec_proj = approve_spec.json()["projection"]
            assert spec_proj["spec_approved"] is True
            assert spec_proj["hitl_banner"]["kind"] == "design_gate"

            approve_design = client.post(
                f"/api/runs/{run_id}/command",
                json={
                    "operation_id": "op-approve-design",
                    "action": "approve_design",
                    "expected_state_revision": spec_proj["state_revision"],
                    "auto_advance": False,
                },
            )
            assert approve_design.status_code == 200
            design_proj = approve_design.json()["projection"]
            assert design_proj["design_approved"] is True
            assert design_proj["hitl_banner"]["kind"] == "plan_gate"

            approve_plan = client.post(
                f"/api/runs/{run_id}/command",
                json={
                    "operation_id": "op-approve-plan",
                    "action": "approve_plan",
                    "expected_state_revision": design_proj["state_revision"],
                    "auto_advance": False,
                },
            )
            assert approve_plan.status_code == 200
            plan_proj = approve_plan.json()["projection"]
            assert plan_proj["plan_approved"] is True
            assert plan_proj["hitl_banner"] is None
            assert plan_proj["stage_pipeline"][1]["status"] == "done"

            # 6. New-run creation rejects an empty brief and queues a real thread
            empty = client.post("/api/runs", json={"raw_input": "   "})
            assert empty.status_code == 400

            created = client.post(
                "/api/runs",
                json={
                    "raw_input": "为候补服务补充 JSON 导出与排位查询",
                    "title": "候补导出",
                    "workflow_name": "repo_fast_patch",
                    "provider_config": {"provider": "mock"},
                },
            )
            assert created.status_code == 200
            created_body = created.json()
            assert created_body["ok"] is True
            assert created_body["run_id"].startswith("run")
            assert created_body["workflow"] == "repo_fast_patch"
            app.supervisor.wait(created_body["run_id"], timeout=10.0)

            new_proj = client.get(f"/api/runs/{created_body['run_id']}").json()["projection"]
            assert new_proj["workflow_name"] == "repo_fast_patch"
            assert new_proj["workflow"]["graph"]["nodes"]
            assert new_proj["progress"]["total_tasks"] == 0

            runs = client.get("/api/runs").json()["runs"]
            assert any(
                run["run_id"] == created_body["run_id"]
                and run["workflow_name"] == "repo_fast_patch"
                for run in runs
            )
    finally:
        app.supervisor.shutdown()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2.0)


def test_direct_design_spec_projection_pipeline_and_matrix(tmp_path: Path) -> None:
    db_path = tmp_path / ".olympus" / "olympus.db"
    store = RunStore(db_path)
    run_id = store.create_run(
        title="直接上传 DesignSpec 投影测试",
        workflow_name="product_blueprint",
        provider_config={"provider": "mock"},
    )
    design = DesignSpec(
        design_id="design-direct-1",
        architecture_summary="核心域与 API 域隔离，直接由设计规格进入分解。",
        modules=[
            DesignModule(
                id="MOD-CORE",
                name="领域核心",
                responsibility="核心规则与数据持久化",
                requirement_refs=["REQ-1"],
                target_paths=["src/core/domain.py"],
            ),
            DesignModule(
                id="MOD-API",
                name="HTTP 接口",
                responsibility="对外暴露查询与命令端点",
                requirement_refs=["REQ-1", "REQ-2"],
                target_paths=["src/api/routes.py"],
            ),
        ],
        data_models=[
            DataEntity(
                name="Record",
                fields=["id", "status"],
                owner_module_id="MOD-CORE",
            )
        ],
        interfaces=[
            InterfaceContract(
                id="IF-DIRECT-1",
                provider_module_id="MOD-CORE",
                consumer_module_ids=["MOD-API"],
                protocol="fn",
                signature="get_record(record_id: str) -> Record",
                error_cases=["RecordNotFound"],
            )
        ],
    )
    store.save_design(run_id, design, actor="design")
    DesignStage(store, MockLLMProvider(), tmp_path).approve_design(
        run_id=run_id,
        approver="zeus",
        reason="direct design approved",
    )
    snap = store.get_run(run_id)
    confirmed_design = snap["design"]
    assert confirmed_design is not None

    plan = TaskPlan(
        plan_id="plan-direct-1",
        design_id=confirmed_design.design_id,
        design_hash=confirmed_design.content_hash,
        tasks=[
            TaskNode(
                task_id="T01",
                title="实现核心领域服务",
                objective="落地领域规则与存储",
                requirement_refs=["REQ-1"],
                write_scope=["src/core/domain.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T01-C1",
                        kind=CheckKind.COMMAND,
                        description="core tests",
                        command="pytest -q",
                    )
                ],
                design_refs=["MOD-CORE"],
                role="backend",
                kind=TaskKind.IMPLEMENT,
                deliverables=["src/core/domain.py"],
                status=TaskStatus.READY,
            ),
            TaskNode(
                task_id="T02",
                title="实现 HTTP 路由",
                objective="对接核心领域并暴露端点",
                requirement_refs=["REQ-2"],
                write_scope=["src/api/routes.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T02-C1",
                        kind=CheckKind.COMMAND,
                        description="api tests",
                        command="pytest -q",
                    )
                ],
                design_refs=["MOD-API"],
                role="backend",
                kind=TaskKind.IMPLEMENT,
                deliverables=["src/api/routes.py"],
                status=TaskStatus.READY,
            ),
        ],
    )
    store.save_plan(run_id, plan, actor="orchestrate")

    proj = build_run_projection(store, tmp_path, run_id)
    assert proj["spec"] is None
    assert proj["design_approved"] is True
    assert [s["status"] for s in proj["stage_pipeline"]] == [
        "done",
        "done",
        "active",
        "pending",
        "pending",
    ]
    assert proj["hitl_banner"] is not None
    assert proj["hitl_banner"]["kind"] == "plan_gate"
    assert [r["requirement_id"] for r in proj["traceability_matrix"]] == ["REQ-1", "REQ-2"]
    assert proj["traceability_matrix"][0]["source_ref"] == "DesignSpec-Direct"
    assert proj["traceability_matrix"][0]["design_module_ids"] == ["MOD-CORE", "MOD-API"]
    assert proj["traceability_matrix"][0]["task_ids"] == ["T01"]
    assert proj["traceability_matrix"][1]["design_module_ids"] == ["MOD-API"]
    assert proj["traceability_matrix"][1]["task_ids"] == ["T02"]


def test_workflow_graph_describe_includes_router_targets() -> None:
    definition = load_workflow("product_blueprint")
    described = build_graph(definition).describe()
    edges = {(edge["source"], edge["target"]): edge for edge in described["edges"]}
    for pair in (
        ("intake", "align"),
        ("intake", "gate_spec"),
        ("intake", "validate_design"),
        ("align", "critic"),
        ("critic", "gate_spec"),
        ("validate_design", "gate_design"),
        ("validate_plan", "gate_plan"),
    ):
        assert pair in edges, f"Missing router target edge {pair}"
        assert edges[pair]["conditional"] is True


def test_failed_workflow_projection_and_retry_workflow(tmp_path: Path) -> None:
    db_path = tmp_path / ".olympus" / "olympus.db"
    store = RunStore(db_path)
    run_id = store.create_run(
        title="RedNote Dev",
        workflow_name="product_blueprint",
        provider_config={"provider": "mock"},
    )
    store.append_event(
        run_id=run_id,
        stage="graph",
        event_type="graph.input_recorded",
        actor="supervisor",
        payload={"raw_input": "我想开发一款小红书这样的产品"},
    )
    store.append_event(
        run_id=run_id,
        stage="graph",
        event_type="graph.node_started",
        actor="supervisor",
        payload={"node": "align", "kind": "align_goal"},
    )
    store.append_event(
        run_id=run_id,
        stage="graph",
        event_type="graph.failed",
        actor="supervisor",
        payload={"error": "provider_error: The read operation timed out"},
    )
    store.set_run_status(
        run_id,
        RunStatus.FAILED,
        actor="supervisor",
        stage="graph",
        detail={"error": "provider_error: The read operation timed out"},
    )

    proj = build_run_projection(store, tmp_path, run_id)
    assert proj["raw_input"] == "我想开发一款小红书这样的产品"
    assert proj["workflow"]["failed_node"] == "align"
    assert proj["workflow"]["node_statuses"]["align"] == "failed"
    assert proj["hitl_banner"] is not None
    assert proj["hitl_banner"]["kind"] == "run_failed"
    assert proj["hitl_banner"]["actions"] == ["retry_workflow"]

    canned_goal = GoalSpec(
        spec_id="spec-retry-1",
        goal="构建图文内容社区与瀑布流浏览服务",
        deliverable_kind=DeliverableKind.MULTI_SURFACE,
        observable_surfaces=["REST API /notes", "Web UI 信息流页"],
        actors=["内容创作者", "普通读者"],
        requirements=[
            RequirementItem(
                id="REQ-1",
                description="支持发布带图笔记并按时间倒序浏览信息流",
                ears_criteria=[
                    {
                        "id": "AC-1",
                        "pattern": EarsPattern.EVENT_DRIVEN,
                        "condition": "When a user submits a valid note",
                        "outcome": "the system persists the note and returns HTTP 201",
                    }
                ],
                acceptance_criteria=["POST /notes returns 201 and GET /notes lists the note"],
            )
        ],
        out_of_scope=["不接入即时音视频直播"],
    ).model_dump(mode="json")

    def _responder(messages: object, _tools: object) -> LLMResponse:
        sys_msg = str(getattr(messages[0], "content", "")) if messages else ""  # type: ignore[index]
        if "Alignment Architect" in sys_msg or "GoalSpec" in sys_msg:
            return LLMResponse(content=json.dumps(canned_goal, ensure_ascii=False))
        return LLMResponse(content=json.dumps({"notes": []}, ensure_ascii=False))

    supervisor = RunSupervisor(
        store=store,
        workspace_root=tmp_path,
        provider_builder=lambda **_: MockLLMProvider(responder=_responder),
    )
    app = AriadneApp(store=store, workspace_root=tmp_path, supervisor=supervisor)
    try:
        res = app.execute_command(
            run_id,
            {
                "operation_id": "op-retry-wf-1",
                "action": "retry_workflow",
                "expected_state_revision": proj["state_revision"],
                "sync": True,
            },
        )
        assert res["ok"] is True
        after = res["projection"]
        assert after["status"] != "failed"
        assert after["spec"] is not None
    finally:
        app.supervisor.shutdown()


def test_recover_orphaned_supervised_runs(tmp_path: Path) -> None:
    db_path = tmp_path / ".olympus" / "olympus.db"
    store = RunStore(db_path)

    # Supervised run left mid-flight in ALIGNING
    orphan_id = store.create_run(
        title="Orphan Run",
        workflow_name="product_blueprint",
        provider_config={"provider": "mock"},
    )
    store.append_event(
        run_id=orphan_id,
        stage="graph",
        event_type="graph.node_started",
        actor="supervisor",
        payload={"node": "align", "kind": "align_goal"},
    )
    store.set_run_status(orphan_id, RunStatus.ALIGNING, actor="align", stage="align")

    # Unsupervised run in RUNNING (e.g. demo or CLI without graph events) stays untouched
    manual_id = store.create_run(title="Manual Run", workflow_name="repo_fast_patch")
    store.set_run_status(manual_id, RunStatus.RUNNING, actor="zeus", stage="orchestrate")

    supervisor = RunSupervisor(store=store, workspace_root=tmp_path)
    try:
        recovered = supervisor.recover_orphaned_runs()
        assert recovered == [orphan_id]

        proj = build_run_projection(store, tmp_path, orphan_id)
        assert proj["status"] == "failed"
        assert proj["workflow"]["failed_node"] == "align"
        assert proj["hitl_banner"] is not None
        assert proj["hitl_banner"]["kind"] == "run_failed"
        assert store.get_run(manual_id)["status"] == RunStatus.RUNNING
    finally:
        supervisor.shutdown()


def test_clarify_settles_gate_sync_and_409_returns_projection(tmp_path: Path) -> None:
    db_path = tmp_path / ".olympus" / "olympus.db"
    store = RunStore(db_path)
    run_id = store.create_run(
        title="Clarify Gate Revision Sync",
        workflow_name="product_blueprint",
        provider_config={"provider": "mock"},
    )
    spec = GoalSpec(
        spec_id="spec-clarify-sync",
        version=1,
        goal="构建社区笔记分享核心能力并验证澄清到关口的状态修订一致性。",
        requirements=[
            RequirementItem(
                id="FR-101",
                description="支持发布与浏览图文笔记",
                acceptance_criteria=["POST /notes 创建笔记并可在 GET /notes 列表中查到"],
                source_ref="REQ-101",
            )
        ],
        out_of_scope=["不接入直播间打赏"],
        constraints=["使用单一 SQLite 事务落盘"],
        open_ambiguities=[
            AmbiguityItem(
                id="AMB-1",
                question="首期是否支持视频笔记？",
                impact="决定媒体上传接口契约",
                options=["首期仅支持图文", "图文与短视频"],
                default_assumption="首期仅支持图文",
                blocking=True,
                answer=None,
            )
        ],
        status=SpecStatus.DRAFT,
    )
    store.save_spec(run_id, spec, actor="align")
    store.append_event(
        run_id=run_id,
        stage="graph",
        event_type="graph.node_finished",
        actor="supervisor",
        payload={"node": "align", "kind": "align_goal"},
    )
    store.set_run_status(
        run_id,
        RunStatus.AWAITING_HUMAN,
        actor="supervisor",
        stage="graph",
        detail={"paused_node": "gate_spec"},
    )

    app = AriadneApp(store=store, workspace_root=tmp_path)
    httpd = create_http_server(app, host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            initial = client.get(f"/api/runs/{run_id}").json()["projection"]

            # Default clarify (auto_advance omitted -> True, sync omitted -> False)
            # must enter gate_spec synchronously so returned state_revision is settled.
            clarify_res = client.post(
                f"/api/runs/{run_id}/command",
                json={
                    "operation_id": "op-clarify-sync-1",
                    "action": "clarify",
                    "answers": {"AMB-1": "首期仅支持图文"},
                    "expected_state_revision": initial["state_revision"],
                },
            )
            assert clarify_res.status_code == 200
            clarify_proj = clarify_res.json()["projection"]
            assert clarify_proj["status"] == "awaiting_human"
            assert clarify_proj["workflow"]["paused"]["node"] == "gate_spec"
            assert clarify_proj["spec"]["version"] == 2
            assert clarify_proj["spec_approved"] is False

            # Stale revision -> 409 with conflict=True and live projection for client resync.
            stale_rev = initial["state_revision"]
            conflict_res = client.post(
                f"/api/runs/{run_id}/command",
                json={
                    "operation_id": "op-approve-stale-1",
                    "action": "approve_spec",
                    "expected_state_revision": stale_rev,
                    "auto_advance": False,
                },
            )
            assert conflict_res.status_code == 409
            conflict_body = conflict_res.json()
            assert conflict_body["ok"] is False
            assert conflict_body["conflict"] is True
            assert conflict_body["projection"]["state_revision"] == clarify_proj["state_revision"]

            # Immediate approve_spec using clarify's returned revision succeeds (never 409).
            approve_res = client.post(
                f"/api/runs/{run_id}/command",
                json={
                    "operation_id": "op-approve-fresh-1",
                    "action": "approve_spec",
                    "expected_state_revision": clarify_proj["state_revision"],
                    "auto_advance": False,
                },
            )
            assert approve_res.status_code == 200
            approve_proj = approve_res.json()["projection"]
            assert approve_proj["spec_approved"] is True

            # SSE once=1 explicitly uses Connection: close under HTTP/1.1.
            sse_res = client.get(f"/api/runs/{run_id}/stream?after_seq=0&once=1")
            assert sse_res.status_code == 200
            assert sse_res.headers.get("connection") == "close"
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.supervisor.shutdown()


def test_completed_blueprint_projection_and_task_diagnostics(tmp_path: Path) -> None:
    db_path = tmp_path / ".olympus" / "olympus.db"
    store = RunStore(db_path)
    run_id = _seed_gated_run(store)

    # Approve every gate and finish product_blueprint with historical gate.paused events.
    snap = store.get_run(run_id)
    assert snap["spec"] is not None
    assert snap["design"] is not None
    assert snap["plan"] is not None
    for event_type, payload in (
        ("graph.paused", {"node": "gate_spec", "reason": "human_gate"}),
        ("graph.paused", {"node": "gate_design", "reason": "human_gate"}),
        ("graph.paused", {"node": "gate_plan", "reason": "human_gate"}),
        ("graph.finished", {"graph": "product_blueprint", "steps": 1}),
    ):
        store.append_event(
            run_id=run_id,
            stage="graph",
            event_type=event_type,
            actor="supervisor",
            payload=payload,
        )

    store.record_approval(
        ApprovalRecord(
            run_id=run_id,
            target=ApprovalTarget.GOAL_SPEC,
            target_id=snap["spec"].spec_id,
            target_hash=snap["spec"].content_hash,
            approver="zeus",
            decision=ApprovalDecision.APPROVED,
        )
    )
    store.record_approval(
        ApprovalRecord(
            run_id=run_id,
            target=ApprovalTarget.DESIGN_SPEC,
            target_id=snap["design"].design_id,
            target_hash=snap["design"].content_hash,
            approver="zeus",
            decision=ApprovalDecision.APPROVED,
        )
    )
    # Reset tasks to initial WAITING state and include a dependent downstream task.
    root_waiting = snap["plan"].tasks[0].model_copy(update={"status": TaskStatus.WAITING})
    dep_waiting = TaskNode(
        task_id="T02",
        title="对外接口接入",
        objective="在 API 层接入递补状态查询",
        requirement_refs=["FR-100"],
        design_refs=["MOD-API"],
        depends_on=["T01"],
        write_scope=["src/waitlist/api.py"],
        acceptance_checks=[
            AcceptanceCheck(
                id="T02-C1",
                kind=CheckKind.COMMAND,
                description="pytest 验证接口查询",
                command="pytest -q",
            )
        ],
        role="backend",
        kind=TaskKind.IMPLEMENT,
        deliverables=["src/waitlist/api.py"],
        status=TaskStatus.WAITING,
    )
    updated_plan = snap["plan"].model_copy(update={"tasks": [root_waiting, dep_waiting]})
    store.save_plan(run_id, updated_plan, actor="orchestrate")
    store.record_approval(
        ApprovalRecord(
            run_id=run_id,
            target=ApprovalTarget.TASK_PLAN,
            target_id=updated_plan.plan_id,
            target_hash=updated_plan.content_hash,
            approver="zeus",
            decision=ApprovalDecision.APPROVED,
        )
    )
    store.set_run_status(run_id, RunStatus.COMPLETED, actor="supervisor", stage="graph")

    proj = build_run_projection(store, tmp_path, run_id)
    assert proj["blueprint_completed"] is True
    assert proj["workflow"]["paused"] is None
    assert proj["workflow"]["node_statuses"]["gate_spec"] == "completed"
    assert proj["workflow"]["node_statuses"]["gate_design"] == "completed"
    assert proj["workflow"]["node_statuses"]["gate_plan"] == "completed"
    assert [step["status"] for step in proj["stage_pipeline"]] == [
        "done",
        "done",
        "done",
        "skipped",
        "skipped",
    ]
    assert proj["hitl_banner"] is not None
    assert proj["hitl_banner"]["kind"] == "blueprint_completed"
    assert proj["hitl_banner"]["actions"] == ["promote_to_execute"]
    for node in proj["dag"]["nodes"]:
        assert node["diagnostic_output"] == "已冻结（未启动代码执行）"
        assert "等待前置产物解锁" not in node["diagnostic_attribution"]

    # During active execution (full_sdlc_tdd), only tasks with unmet dependencies wait on upstream.
    store.set_run_status(run_id, RunStatus.RUNNING, actor="supervisor", stage="graph")
    with store._tx() as conn:
        conn.execute(
            "UPDATE runs SET workflow_name = 'full_sdlc_tdd' WHERE run_id = ?",
            (run_id,),
        )
    exec_proj = build_run_projection(store, tmp_path, run_id)
    assert exec_proj["blueprint_completed"] is False
    by_task = {n["task_id"]: n for n in exec_proj["dag"]["nodes"]}
    root_task = next(n for n in by_task.values() if not n["depends_on"])
    dep_task = next(n for n in by_task.values() if n["depends_on"])
    assert root_task["diagnostic_output"] == "就绪待调度"
    assert "等待前置产物解锁" in dep_task["diagnostic_attribution"]
