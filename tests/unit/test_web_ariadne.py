"""Unit and HTTP/SSE integration tests for the M4 Ariadne Web Console."""

from __future__ import annotations

import threading
from pathlib import Path

import httpx

from olympus.core.contracts import TaskStatus
from olympus.core.store import RunStore
from olympus.web.projection import build_run_projection, seed_demo_runs
from olympus.web.server import AriadneApp, create_http_server


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
