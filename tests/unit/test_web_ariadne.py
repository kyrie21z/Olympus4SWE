"""Unit and HTTP/SSE integration tests for the M4 Ariadne Web Console."""

from __future__ import annotations

import threading
from pathlib import Path

import httpx

from olympus.core.contracts import (
    AcceptanceCheck,
    AmbiguityItem,
    CheckKind,
    DataEntity,
    DesignModule,
    DesignSpec,
    GoalSpec,
    InterfaceContract,
    RequirementItem,
    SpecStatus,
    TaskKind,
    TaskNode,
    TaskPlan,
    TaskStatus,
)
from olympus.core.store import RunStore
from olympus.web.projection import build_run_projection, seed_demo_runs
from olympus.web.server import AriadneApp, create_http_server


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
