"""Run with uv run --with playwright pytest tests/unit/test_web_run_progress_browser.py."""

import importlib
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest
import uvicorn

from olympus.core.contracts import (
    AcceptanceCheck,
    AmbiguityItem,
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    CheckKind,
    DeliveryCheck,
    DesignSpec,
    GoalSpec,
    RequirementItem,
    RunStatus,
    TaskNode,
    TaskPlan,
    TaskStatus,
)
from olympus.core.store import RunStore
from olympus.web.demos import seed_demo_runs
from olympus.web.server import AriadneApp, create_fastapi_app

playwright = (
    importlib.import_module("playwright.sync_api")
    if os.environ.get("OLYMPUS_REQUIRE_BROWSER_TESTS") == "1"
    else pytest.importorskip("playwright.sync_api")
)


@pytest.fixture
def workbench(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    run_id = "run-campus-waitlist"
    assert run_id in seed_demo_runs(store, tmp_path)
    app = AriadneApp(store=store, workspace_root=tmp_path)
    server = uvicorn.Server(
        uvicorn.Config(
            create_fastapi_app(app),
            host="127.0.0.1",
            port=0,
            log_level="warning",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn server did not start within 10s")
        time.sleep(0.01)
    assert server.servers
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield store, run_id, f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)


@pytest.fixture
def browser():
    with playwright.sync_playwright() as runtime:
        chrome_path = os.environ.get("CHROME_PATH")
        if chrome_path:
            launched = runtime.chromium.launch(executable_path=chrome_path)
        else:
            try:
                launched = runtime.chromium.launch()
            except playwright.Error:
                if os.environ.get("OLYMPUS_REQUIRE_BROWSER_TESTS") == "1":
                    raise
                existing = sorted(
                    Path.home().glob(".cache/ms-playwright/chromium-*/chrome-linux*/chrome")
                )
                if not existing:
                    raise
                launched = runtime.chromium.launch(executable_path=str(existing[-1]))
        try:
            yield launched
        finally:
            launched.close()


def test_task_replay_freezes_horizon_and_reconstructs_only_observed_tasks(workbench, browser):
    store, run_id, url = workbench
    events = store.list_events(run_id)
    blocked = next(e for e in events if e.event_type == "agent.swimlane"
                   and e.payload.get("task_id") == "T03-R1" and e.payload["status"] == "blocked")
    end = events[-1].event_seq
    page = browser.new_page()
    writes = []
    page.on("request", lambda req: writes.append(req.url) if req.method == "POST" else None)
    page.goto(f"{url}/?run={run_id}&lang=zh")
    page.get_by_role("button", name="故障复盘").click()
    dialog = page.locator("#run-replay")
    incident = dialog.get_by_role("article", name="T03-R1")
    expect = playwright.expect
    expect(incident).to_contain_text("同一候补者被递补 2 次")
    expect(incident).to_contain_text("暂无重试或改派记录")
    expect(incident).to_contain_text("尚未记录该任务后续通过验收")
    incident.get_by_role("button", name="查看尝试证据").click()
    attempt = incident.get_by_role("region", name="第 2 次尝试")
    expect(attempt.get_by_role("listitem").filter(has_text="T03-R1-cmd")).to_contain_text("未通过")
    expect(attempt).to_contain_text("src/waitlist/waitlist.py")
    expect(incident).to_contain_text("截至回放终点，未记录后续尝试。")
    expect(incident.get_by_role("region")).to_have_count(1)
    expect(dialog.get_by_role("slider")).to_have_count(0)
    # Both disclosures expand and collapse at their own controls, including via keyboard.
    incident.get_by_role("button", name="收起尝试证据").press("Enter")
    expect(incident.get_by_role("region")).to_have_count(0)
    source = dialog.get_by_role("button", name=re.compile(r"^(查看|收起)原始事件$"))
    source.press("Enter")
    expect(source).to_have_attribute("aria-expanded", "true")
    expect(dialog.get_by_role("slider")).to_have_value(str(end))
    source.press("Space")
    expect(source).to_have_attribute("aria-expanded", "false")
    expect(dialog.get_by_role("slider")).to_have_count(0)
    incident.get_by_role("button", name=f"任务执行受阻 · #{blocked.event_seq}").click()
    expect(dialog.get_by_role("button", name="查看尝试证据")).to_have_count(0)
    tasks = dialog.get_by_role("region", name="已观测任务")
    expect(tasks).to_contain_text("T03-R1")
    expect(dialog.get_by_role("slider")).to_have_value(str(blocked.event_seq))
    dialog.get_by_role("button", name="上一步").click()
    expect(tasks).not_to_contain_text("T03-R1")
    expect(incident).to_have_count(0)
    source.click()
    expect(incident).to_contain_text("同一候补者被递补 2 次")
    source.click()
    expect(incident).to_have_count(0)
    dialog.get_by_role("button", name="下一步").click()
    row = tasks.get_by_role("listitem").filter(has_text="T03-R1")
    expect(row).to_contain_text("阻塞")
    expect(row).to_contain_text("第 2 次尝试")
    dialog.get_by_role("slider").press("End")
    expect(dialog.get_by_role("slider")).to_have_value(str(end))
    row.filter(has_text="Hephaestus").get_by_role("button").click()
    expect(dialog.get_by_role("slider")).to_have_value(str(blocked.event_seq))
    store.append_event(
        run_id=run_id, stage="orchestrate", actor="chronos", event_type="task.leased",
        payload={"task_id": "FUTURE-ONLY", "attempt": 1},
    )
    expect(dialog).to_contain_text("实时新增 1 条")
    expect(dialog).not_to_contain_text("FUTURE-ONLY")
    expect(dialog.get_by_role("slider")).to_have_value(str(blocked.event_seq))
    dialog.get_by_role("button", name="回到实时").click()
    expect(dialog).to_have_count(0)
    assert not writes


def test_failure_review_groups_retries_and_requires_task_acceptance(workbench, browser, tmp_path):
    store, run_id, url = workbench
    later = tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "T03-R1" / "attempt-3"
    later.mkdir()
    (later / "check-T03-R1-cmd.log").write_text(
        "command: pytest\nexit_code: 0\n--- stdout ---\nFAKE_PRIVATE_OUTPUT\n", encoding="utf-8"
    )
    (later / "diff.patch").write_text(
        "--- a/fix.py\n+++ b/fix.py\n@@ -1 +1 @@\n-old\n+FAKE_PRIVATE_OUTPUT\n", encoding="utf-8"
    )
    (later / "check-sk-fixture-private-label.log").write_text("test only", encoding="utf-8")
    for kind, payload in [
        ("task.retried_by_human", {"hint": "private hint"}),
        ("task.leased", {"attempt": 3, "lease_generation": 3}),
        ("review.verdict_recorded", {"attempt": 3, "passed": True}),
        ("task.accepted", {"status": "accepted", "lease_generation": 3}),
    ]:
        store.append_event(run_id=run_id, stage="verify", actor="tester", event_type=kind,
                           payload={"task_id": "T03-R1", **payload})
    page = browser.new_page()
    page.goto(f"{url}/?run={run_id}&lang=en")
    page.get_by_role("button", name="Failure review").click()
    dialog = page.get_by_role("dialog", name="Failure review")
    row = dialog.get_by_role("article", name="T03-R1")
    expect = playwright.expect
    expect(dialog.get_by_role("article")).to_have_count(1)
    expect(row).to_contain_text("Human retry requested")
    expect(row).to_contain_text("New execution lease acquired")
    expect(row).to_contain_text("Task accepted")
    row.get_by_role("button", name="Inspect attempt evidence").click()
    before = row.get_by_role("region", name="Attempt 2")
    after = row.get_by_role("region", name="Attempt 3")
    expect(before.get_by_role("listitem").filter(has_text="T03-R1-cmd")).to_contain_text(
        "Not passed"
    )
    expect(after.get_by_role("listitem").filter(has_text="T03-R1-cmd")).to_contain_text("Passed")
    expect(after).to_contain_text("fix.py")
    expect(after).to_contain_text("+1 / −1")
    expect(dialog).not_to_contain_text("FAKE_PRIVATE_OUTPUT")
    expect(dialog).not_to_contain_text("sk-fixture")
    expect(dialog).not_to_contain_text("private hint")
    before.get_by_role("button").click()
    expect(row).not_to_contain_text("Task accepted")
    expect(row).not_to_contain_text("New execution lease acquired")
    dialog.get_by_role("button", name="Hide source events").click()
    row.get_by_role("button", name="Task accepted", exact=False).click()
    expect(page.get_by_role("region", name="Selected event")).to_contain_text("task.accepted")


def test_milestone_navigation_skips_routine_events_and_freezes_results(workbench, browser):
    store, run_id, url = workbench

    def append(kind, **payload):
        return store.append_event(run_id=run_id, stage="verify", actor="tester",
                                  event_type=kind, payload=payload).event_seq

    task = {"task_id": "T03-R1"}
    request = append("task.retried_by_human", **task, hint="not for display")
    append("task.leased", task_id="other-task", attempt=1)
    leased = append("task.leased", **task, attempt=3)
    append("turn.receipt", **task)
    review = append("review.verdict_recorded", **task, passed=True)
    accepted = append("task.accepted", **task, status="accepted")
    append("post_dag.repro_rechecked", checked_count=0, still_failing_count=0)
    rechecked = append("post_dag.repro_rechecked", checked_count=1, still_failing_count=0)
    final = append("approval.recorded", target="final_acceptance", target_id="approval-3",
                   decision="approved")
    page = browser.new_page()
    page.goto(f"{url}/?run={run_id}&lang=en")
    page.get_by_role("button", name="Failure review").click()
    dialog = page.get_by_role("dialog", name="Failure review")
    dialog.get_by_role("button", name="Inspect source events").click()
    prev = dialog.get_by_role("button", name="Previous milestone")
    nxt = dialog.get_by_role("button", name="Next milestone")
    selected = dialog.get_by_role("region", name="Selected event")
    cursor = dialog.get_by_role("slider")
    expect = playwright.expect
    expect(selected).to_contain_text("approval-3")
    expect(selected).to_contain_text("Final acceptance approved")
    append("task.failed", task_id="FUTURE-ONLY")
    expect(dialog).to_contain_text("1 new live events")
    expect(nxt).to_be_disabled()
    cursor.press("Shift+ArrowLeft")
    expect(cursor).to_have_value(str(rechecked))
    expect(selected).to_contain_text("Delivery recheck passed")
    prev.click()
    expect(cursor).to_have_value(str(accepted))
    prev.click()
    expect(cursor).to_have_value(str(review))
    expect(selected).to_contain_text("Independent review passed")
    card = dialog.get_by_role("article", name="T03-R1")
    expect(card).to_contain_text("No acceptance recorded")
    prev.click()
    expect(cursor).to_have_value(str(leased))
    expect(selected).to_contain_text("Retry started")
    prev.press("Enter")
    expect(cursor).to_have_value(str(request))
    expect(selected).to_contain_text("Human retry requested")
    expect(card).not_to_contain_text("New execution lease acquired")
    cursor.press("ArrowRight")
    expect(cursor).to_have_value(str(request + 1))
    cursor.press("Shift+ArrowRight")
    expect(cursor).to_have_value(str(leased))
    for seq in (review, accepted, rechecked, final):
        nxt.click()
        expect(cursor).to_have_value(str(seq))
    expect(nxt).to_be_disabled()
    expect(dialog).not_to_contain_text("FUTURE-ONLY")


@pytest.mark.parametrize("one_stop", [False, True])
def test_empty_and_single_milestone_boundaries(workbench, browser, one_stop):
    store, _, url = workbench
    run_id = store.create_run(title="Milestone boundaries")
    if one_stop:
        stop = store.append_event(
            run_id=run_id, stage="plan", actor="zeus", event_type="approval.recorded",
            payload={
                "target": "task_plan", "target_id": "sk-fixture-hidden", "decision": "approved",
            },
        )
    page = browser.new_page()
    page.goto(f"{url}/?run={run_id}&lang=en")
    page.get_by_role("button", name="Failure review").click()
    dialog = page.get_by_role("dialog", name="Failure review")
    dialog.get_by_role("button", name="Inspect source events").click()
    prev = dialog.get_by_role("button", name="Previous milestone")
    nxt = dialog.get_by_role("button", name="Next milestone")
    cursor = dialog.get_by_role("slider")
    expect = playwright.expect
    expect(prev).to_be_disabled()
    expect(nxt).to_be_disabled()
    if one_stop:
        expect(dialog).to_contain_text("Task plan approved")
        expect(dialog).not_to_contain_text("sk-fixture-hidden")
        cursor.press("Home")
        expect(prev).to_be_disabled()
        nxt.press("Space")
        expect(cursor).to_have_value(str(stop.event_seq))
        expect(nxt).to_be_disabled()
    else:
        expect(dialog).to_contain_text("No milestones recorded in this history.")
        cursor.press("Shift+ArrowLeft")
        expect(cursor).to_have_value(str(store.list_events(run_id)[-1].event_seq))


def test_attempt_evidence_handles_absent_empty_and_retried_reads(workbench, browser, tmp_path):
    store, run_id, url = workbench
    for task, number in [("T01", 6), ("T02", 0)]:
        store.append_event(run_id=run_id, stage="verify", actor="tester", event_type="task.failed",
                           payload={"task_id": task, "attempt": number})
    page = browser.new_page()
    page.goto(f"{url}/?run={run_id}&lang=en")
    page.get_by_role("button", name="Failure review").click()
    dialog = page.get_by_role("dialog", name="Failure review")
    invalid = dialog.get_by_role("article", name="T02")
    playwright.expect(invalid).to_contain_text("No valid attempt number")
    missing_button = invalid.get_by_role("button", name="Inspect attempt evidence")
    playwright.expect(missing_button).to_have_count(0)

    # Hold another task's read while switching evidence panels.
    pending = []
    page.route("**/task-artifacts/T03-R1/2", lambda route: pending.append(route))
    original = dialog.get_by_role("article", name="T03-R1")
    toggle = original.get_by_role("button", name=re.compile("attempt evidence", re.I))
    with page.expect_request("**/task-artifacts/T03-R1/2"):
        toggle.click()
    toggle.click()
    row = dialog.get_by_role("article", name="T01")
    button = row.get_by_role("button", name=re.compile("attempt evidence", re.I))
    button.click()
    playwright.expect(row).to_contain_text("No artifacts were archived")
    assert pending
    pending[0].fulfill(json={"ok": True, "artifacts": {
        "manifest_md": "", "verdicts": [], "diff_patch": "", "check_logs": {"STALE": ""},
    }})
    playwright.expect(dialog).not_to_contain_text("STALE")
    button.click()
    attempt_dir = tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "T01" / "attempt-6"
    attempt_dir.mkdir(parents=True)
    page.route("**/task-artifacts/T01/6", lambda route: route.fulfill(
        status=503, json={"ok": False, "error": "temporary"},
    ), times=1)
    button.click()
    playwright.expect(row.get_by_role("alert")).to_contain_text("Could not load")
    row.get_by_role("button", name="Retry", exact=True).click()
    playwright.expect(row).to_contain_text("No check results recorded")
    playwright.expect(row).to_contain_text("No patch recorded")


def test_goal_rereview_shows_permission_impact_after_live_update(workbench, browser):
    store, _, url = workbench
    run_id = store.create_run(title="导出权限返审")
    approved = GoalSpec(
        goal="导出报表",
        requirements=[RequirementItem(
            id="EXPORT", description="仅管理员可以导出",
            acceptance_criteria=["导出包含时间戳"],
        )],
        out_of_scope=["定时导出"],
    )
    store.save_spec(run_id, approved)
    store.record_approval(ApprovalRecord(
        run_id=run_id, target=ApprovalTarget.GOAL_SPEC, target_id=approved.spec_id,
        target_hash=approved.content_hash, approver="human", decision=ApprovalDecision.APPROVED,
    ))

    page = browser.new_page()
    page.goto(f"{url}/?run={run_id}&lang=zh")
    expect = playwright.expect
    expect(page.get_by_text("导出权限返审", exact=True).first).to_be_visible()

    revised = approved.model_copy(update={
        "version": 2, "requirements": [approved.requirements[0].model_copy(update={
            "description": "所有成员都可以导出",
        })],
    })
    store.save_spec(run_id, revised)
    review = page.get_by_role("article", name="GoalSpec · v2")
    expect(review.get_by_text("所有成员都可以导出", exact=True)).to_be_visible()
    expect(review.get_by_text("仅管理员可以导出", exact=True)).to_be_visible()
    expect(review.get_by_text("导出包含时间戳", exact=True)).to_be_visible()
    unchanged = review.get_by_text("查看已确认未变的条目", exact=True)
    expect(review.get_by_text("定时导出", exact=True)).to_be_hidden()
    unchanged.click()
    expect(review.get_by_text("定时导出", exact=True)).to_be_visible()
    review.get_by_text("查看上次批准的 GoalSpec", exact=True).click()
    expect(review.locator("[data-review-baseline] pre")).to_contain_text(approved.content_hash)


def test_legacy_rereview_shows_candidate_when_the_approved_original_is_missing(workbench, browser):
    store, _, url = workbench
    run_id = store.create_run(title="旧库返审")
    approved = GoalSpec(
        goal="导出报表",
        requirements=[RequirementItem(
            id="EXPORT", description="仅管理员可以导出", acceptance_criteria=["导出保留时间戳"],
        )],
        out_of_scope=["定时导出"],
    )
    store.save_spec(run_id, approved)
    store.record_approval(ApprovalRecord(
        run_id=run_id, target=ApprovalTarget.GOAL_SPEC, target_id=approved.spec_id,
        target_hash=approved.content_hash, approver="human", decision=ApprovalDecision.APPROVED,
    ))
    store.save_spec(run_id, approved.model_copy(update={"version": 2, "goal": "导出当前报表"}))
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("DROP TABLE approval_artifacts")
    RunStore(store.db_path)

    page = browser.new_page()
    page.goto(f"{url}/?run={run_id}&lang=zh")
    expect = playwright.expect
    review = page.get_by_role("article", name="GoalSpec · v2")
    expect(review.get_by_text("缺少上次批准版本的可靠原文", exact=False)).to_be_visible()
    for commitment in ("导出当前报表", "仅管理员可以导出", "导出保留时间戳", "定时导出"):
        expect(review.get_by_text(commitment, exact=True)).to_be_visible()
    expect(review.locator("[data-review-baseline]")).to_have_count(0)


def test_design_approval_shows_tradeoffs_and_reopens_review_after_conflict(workbench, browser):
    store, _, url = workbench
    run_id = store.create_run(title="架构审阅", provider_config={"provider": "mock"})
    design = DesignSpec.model_validate(
        {
            "architecture_summary": "保留现有入口。写入切到新表。迁移步骤见接口。",
            "modules": [
                {
                    "id": "API",
                    "name": "报名入口",
                    "responsibility": "接收请求",
                    "requirement_refs": ["R1"],
                    "target_paths": ["src/api.py"],
                },
                {
                    "id": "STORE",
                    "name": "席位存储",
                    "responsibility": "写入席位",
                    "requirement_refs": ["R1"],
                    "target_paths": ["src/store.py"],
                },
            ],
            "data_models": [{"name": "Seat", "fields": ["id"], "owner_module_id": "STORE"}],
            "interfaces": [
                {
                    "id": "IF-SEAT",
                    "provider_module_id": "STORE",
                    "consumer_module_ids": ["API"],
                    "signature": "reserve_seat(id: str) -> Seat",
                }
            ],
            "risks_and_tradeoffs": ["迁移窗口长"],
        }
    )
    store.save_design(run_id, design)
    page = browser.new_page()
    page.goto(f"{url}/?run={run_id}&lang=zh")
    expect = playwright.expect
    review = page.locator('[data-review-action="approve_design"]')
    expect(review.get_by_text("规格未记录结构化取舍", exact=False)).to_be_visible()
    expect(review.get_by_text("迁移窗口长", exact=False)).to_be_hidden()

    candidate = DesignSpec.model_validate(
        {
            **design.model_dump(),
            "version": 2,
            "design_tradeoffs": [
                {"impact": "rollback", "choice": "保留旧列", "cost": "延后清理旧数据"}
            ],
        }
    )
    store.save_design(run_id, candidate)
    expect(review.get_by_role("heading", name="DesignSpec · v2")).to_be_visible()
    expect(review.get_by_text("保留现有入口。写入切到新表。", exact=True)).to_be_visible()
    full_summary = "保留现有入口。写入切到新表。迁移步骤见接口。"
    expect(page.get_by_text(full_summary, exact=True)).to_be_hidden()
    expect(review.get_by_text("选择：保留旧列", exact=True)).to_be_visible()
    expect(review.get_by_text("代价：延后清理旧数据", exact=True)).to_be_visible()
    structure = review.get_by_text("查看模块、数据模型与接口", exact=True)
    modules = review.get_by_role("table", name="模块图 · 模块边界与独占写入路径")
    expect(structure.locator("..")).to_have_js_property("open", False)
    structure.click()
    expect(
        modules.get_by_role("row", name="API 报名入口 接收请求 R1 src/api.py STORE")
    ).to_be_visible()
    expect(review.get_by_role("heading", name="数据模型")).to_be_visible()
    expect(review.get_by_role("heading", name="接口契约")).to_be_visible()
    expect(review.get_by_text("reserve_seat(id: str) -> Seat", exact=True)).to_be_visible()
    review.get_by_role("button", name="定位 回滚").click()
    expect(review.get_by_text("/design_tradeoffs/0", exact=True)).to_be_focused()
    expect(review.locator("pre")).to_contain_text(candidate.content_hash)

    newer = candidate.model_copy(
        update={
            "version": 3,
            "architecture_summary": "先验证再切流。",
            "modules": [
                candidate.modules[0].model_copy(update={"name": "新版报名入口"}),
                candidate.modules[1],
            ],
        }
    )

    def supersede_before_command(route):
        assert route.request.post_data_json["target_hash"] == candidate.content_hash
        store.save_design(run_id, newer)
        route.continue_()

    page.route(f"{url}/api/runs/{run_id}/command", supersede_before_command, times=1)
    with page.expect_response(
        lambda response: response.url.endswith("/command") and response.status == 409
    ):
        review.get_by_role("button", name="批准 DesignSpec").click()
    expect(page.get_by_text("请确认后重试", exact=False)).to_be_visible()
    expect(review.get_by_role("heading", name="DesignSpec · v3")).to_be_visible()
    expect(review.get_by_text("先验证再切流。", exact=True)).to_be_visible()
    expect(structure.locator("..")).to_have_js_property("open", True)
    expect(modules.get_by_text("新版报名入口", exact=True)).to_be_visible()
    expect(modules.get_by_text("报名入口", exact=True)).to_have_count(0)
    expect(review.get_by_text("查看完整 DesignSpec", exact=True).locator("..")).to_have_js_property(
        "open", True
    )
    expect(review.locator("pre")).to_contain_text(newer.content_hash)


@pytest.fixture
def export_plan(workbench):
    store, _, url = workbench
    run_id = store.create_run(title="演示 · 先核对权限，再开放导出")
    spec = GoalSpec(
        goal="只有通过权限检查的成员才能导出报表",
        requirements=[RequirementItem(
            id="EXPORT", description="按成员权限开放导出",
            acceptance_criteria=["无权限成员无法导出报表"],
        )],
    )
    store.save_spec(run_id, spec)
    store.record_approval(ApprovalRecord(
        run_id=run_id, target=ApprovalTarget.GOAL_SPEC, target_id=spec.spec_id,
        target_hash=spec.content_hash, approver="zeus", decision=ApprovalDecision.APPROVED,
    ))
    review = AcceptanceCheck(id="review", kind=CheckKind.INDEPENDENT_REVIEW, description="复核实现")
    plan = TaskPlan(
        spec_id=spec.spec_id, spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="EXPORT", title="开放导出入口", objective="向有权限成员开放导出",
                depends_on=["AUTH"], write_scope=["export.py"], deliverables=["报表导出入口"],
                acceptance_checks=[AcceptanceCheck(
                    id="member-preview", kind=CheckKind.HUMAN, description="人工试用成员导出入口",
                )],
            ),
            TaskNode(
                task_id="AUTH", title="校验成员权限", objective="拦截无权限成员的导出请求",
                write_scope=["auth.py"], deliverables=["成员权限检查"], acceptance_checks=[review],
            ),
            TaskNode(
                task_id="NOTES", title="整理内部记录", objective="整理开发记录",
                depends_on=["AUTH"], write_scope=["notes.md"], acceptance_checks=[review],
            ),
        ],
        delivery_checks=[DeliveryCheck(
            requirement_id="EXPORT", criterion_index=1,
            command="pytest tests/test_export.py -k denied_member",
        )],
    )
    store.save_plan(run_id, plan)
    return store, run_id, url, plan


def test_plan_review_explains_export_batches_and_opens_evidence(export_plan, browser):
    _, run_id, url, plan = export_plan
    page = browser.new_page()
    expect = playwright.expect
    page.goto(f"{url}/?run={run_id}&lang=zh")

    review = page.get_by_role("article", name="任务计划 · v1")
    batches = review.get_by_role("heading", name="交付批次（按任务依赖）").locator("..")
    expect(batches.get_by_text("第 1 批 · AUTH 校验成员权限")).to_be_visible()
    expect(batches.get_by_text("第 2 批 · EXPORT 开放导出入口")).to_be_visible()
    expect(batches.get_by_text("成员权限检查", exact=False)).to_be_visible()
    expect(batches.get_by_text("报表导出入口", exact=False)).to_be_visible()
    expect(batches.get_by_text("未声明交付物", exact=False)).to_be_visible()

    deps = review.get_by_role("heading", name="影响交付的依赖").locator("..")
    expect(deps).to_contain_text("验收「AUTH 校验成员权限」后，才能开始「EXPORT 开放导出入口」")
    expect(deps).not_to_contain_text("NOTES")
    human = review.get_by_role("heading", name="人工介入点").locator("..")
    expect(human).to_contain_text("人工试用成员导出入口")
    expect(human).not_to_contain_text("复核实现")

    graph = page.get_by_role("heading", name="任务依赖图")
    expect(graph).to_be_hidden()
    expect(review.locator("pre")).to_be_hidden()
    deps.get_by_role("button", name="定位 影响交付的依赖").click()
    expect(review.get_by_text("/tasks/0/depends_on/0", exact=True)).to_be_focused()
    expect(review.locator("pre")).to_contain_text(plan.content_hash)
    expect(review.locator("pre")).to_contain_text(plan.delivery_checks[0].command)
    human.get_by_role("button", name="定位 EXPORT 开放导出入口").click()
    expect(review.get_by_text("/tasks/0/acceptance_checks/0", exact=True)).to_be_focused()

    page.get_by_role("link", name="架构规格").click()
    review.get_by_role("button", name="查看任务依赖图").click()
    expect(graph).to_be_focused()
    page.get_by_role("button", name="任务 EXPORT", exact=False).click()
    expect(page.get_by_role("heading", name="EXPORT · 开放导出入口")).to_be_visible()
    expect(page.locator("#dock-body").get_by_text("AUTH", exact=True)).to_be_visible()


def test_plan_review_rejects_stale_approval_and_retains_approved_checks(export_plan, browser):
    store, run_id, url, plan = export_plan
    page = browser.new_page()
    expect = playwright.expect
    page.goto(f"{url}/?run={run_id}&nosse=1&lang=zh")

    old_review = page.get_by_role("article", name="任务计划 · v1")
    expect(old_review).to_be_visible()
    old_revision = store.get_run(run_id)["state_revision"]
    new_command = "pytest tests/test_export.py -k denied_and_allowed_members"
    revised = plan.model_copy(update={
        "version": 2,
        "delivery_checks": [plan.delivery_checks[0].model_copy(update={"command": new_command})],
    })
    store.save_plan(run_id, revised)

    endpoint = f"{url}/api/runs/{run_id}/command"
    approve = page.get_by_role("button", name="批准任务计划", exact=True)
    with page.expect_response(endpoint) as stale:
        approve.click()
    assert stale.value.status == 409
    assert stale.value.request.post_data_json["target_hash"] == plan.content_hash
    assert stale.value.request.post_data_json["expected_state_revision"] == old_revision
    expect(page.get_by_text("请确认后重试", exact=False)).to_be_visible()
    assert not store.is_approved(
        run_id=run_id, target=ApprovalTarget.TASK_PLAN,
        target_id=revised.plan_id, target_hash=revised.content_hash,
    )

    current = page.get_by_role("article", name="任务计划 · v2")
    current.get_by_text("查看完整 任务计划", exact=True).click()
    expect(current.locator("pre")).to_contain_text(new_command)
    expect(current.locator("pre")).not_to_contain_text(plan.delivery_checks[0].command)
    with page.expect_response(endpoint) as approved:
        approve.click()
    assert approved.value.status == 200
    assert approved.value.request.post_data_json["target_hash"] == revised.content_hash
    assert approved.value.json()["projection"]["plan_approved"]
    expect(approve).to_be_hidden()

    page.get_by_text("查看完整任务计划 · v2", exact=True).click()
    full_plan = page.get_by_role("region", name="任务依赖图").locator("pre")
    expect(full_plan).to_contain_text(revised.content_hash)
    expect(full_plan).to_contain_text(new_command)


def test_design_and_plan_rereview_show_previous_tradeoff_and_dependency(export_plan, browser):
    store, run_id, url, plan = export_plan
    design = DesignSpec.model_validate({
        "architecture_summary": "同步导出。",
        "modules": [{
            "id": "EXPORT", "name": "导出", "responsibility": "管理权限与导出",
            "target_paths": ["src/export.py"], "requirement_refs": ["EXPORT"],
        }],
        "design_tradeoffs": [{"impact": "rollback", "choice": "保留旧列", "cost": "占用空间"}],
    })
    store.save_design(run_id, design)
    for target, artifact_id, fingerprint in (
        (ApprovalTarget.DESIGN_SPEC, design.design_id, design.content_hash),
        (ApprovalTarget.TASK_PLAN, plan.plan_id, plan.content_hash),
    ):
        store.record_approval(ApprovalRecord(
            run_id=run_id, target=target, target_id=artifact_id, target_hash=fingerprint,
            approver="human", decision=ApprovalDecision.APPROVED,
        ))
    store.save_design(run_id, design.model_copy(update={"version": 2, "design_tradeoffs": []}))
    store.save_plan(run_id, plan.model_copy(update={
        "version": 2, "tasks": [task.model_copy(update={"depends_on": []}) for task in plan.tasks],
    }))

    page = browser.new_page()
    page.goto(f"{url}/?run={run_id}&lang=zh")
    expect = playwright.expect
    design_review = page.get_by_role("article", name="DesignSpec · v2")
    old_choice = design_review.get_by_text("选择：保留旧列", exact=True)
    expect(old_choice).to_be_visible()
    expect(old_choice.locator("xpath=ancestor::li").get_by_text("移除", exact=True)).to_be_visible()
    design_review.get_by_role("button", name="定位 上次批准的 回滚").click()
    expect(design_review.get_by_text("/design_tradeoffs/0", exact=True)).to_be_focused()
    expect(design_review.locator("[data-review-baseline] pre")).to_contain_text(design.content_hash)
    plan_review = page.get_by_role("article", name="任务计划 · v2")
    dependency = plan_review.get_by_text("才能开始", exact=False)
    expect(dependency.first).to_be_visible()
    expect(dependency.first.locator("xpath=ancestor::li").get_by_text("移除", exact=True)
           ).to_be_visible()
    expect(plan_review.get_by_role("button", name="查看任务依赖图")).to_be_visible()


def test_plan_history_survives_task_graph_and_updates_over_sse(export_plan, browser):
    store, _, url, plan = export_plan
    run_id = store.create_run(title="等待计划")
    page = browser.new_page()
    page.goto(f"{url}/?run={run_id}&lang=zh")
    expect = playwright.expect
    expect(page.locator("#run-recent-progress")).to_have_text("最近确认 · 已接收编排请求")
    expect(page.locator("#stat-band")).to_be_empty()
    expect(page.get_by_role("heading", name="任务依赖图")).to_be_hidden()
    store.save_plan(run_id, plan)
    graph = page.get_by_role("heading", name="任务依赖图")
    expect(page.locator("#run-output-summary")).to_contain_text("任务计划 v1")
    expect(graph).to_be_hidden()
    page.locator("#run-output-detail").click()
    expect(graph).to_be_visible()
    history = page.get_by_text("产物历史", exact=True)
    history.click()
    history.focus()

    store.save_plan(run_id, plan.model_copy(update={"version": 2}))

    expect(page.locator("#run-output-summary")).to_contain_text("任务计划 v2 已就绪，包含 3 个任务")
    expect(history).to_be_focused()
    expect(history.locator("..")).to_have_attribute("open", "")
    expect(page.get_by_text(plan.content_hash)).to_be_visible()
    expect(graph).to_be_hidden()
    page.locator("#run-output-detail").click()
    expect(graph).to_be_focused()


def test_reconnect_keeps_last_confirmation_and_rejects_late_snapshot(workbench, browser):
    store, run_id, url = workbench
    context = browser.new_context()
    page = context.new_page()
    snapshot_url = f"{url}/api/runs/{run_id}"
    expect = playwright.expect
    page.goto(f"{url}/?run={run_id}&lang=zh")
    time = page.locator("#run-progress-time")
    expect(time).to_be_visible()
    last_time = time.get_attribute("datetime")
    plan = store.get_run(run_id)["plan"]

    context.set_offline(True)
    expect(page.get_by_role("status").filter(has_text="正在重连")).to_be_visible()
    expect(time).to_have_attribute("datetime", last_time)
    store.save_plan(run_id, plan.model_copy(update={"version": 2}))
    context.set_offline(False)
    expect(page.locator("#run-output-summary")).to_contain_text("任务计划 v2")
    expect(page.get_by_text("正在重连", exact=False)).to_be_hidden()

    held = []
    store.save_plan(run_id, plan.model_copy(update={"version": 3}))
    old_snapshot = httpx.get(snapshot_url).json()

    def delay_snapshot(route):
        if not held:
            held.append(route)
            page.evaluate("window.snapshotHeld = true")
        else:
            route.continue_()

    page.route(snapshot_url, delay_snapshot)
    page.wait_for_function("window.snapshotHeld")
    assert held
    expect(page.locator("#run-output-summary")).to_contain_text("任务计划 v2")
    store.save_plan(run_id, plan.model_copy(update={"version": 4}))
    expect(page.locator("#run-output-summary")).to_contain_text("任务计划 v4")
    with page.expect_response(snapshot_url):
        held[0].fulfill(json=old_snapshot)
    page.evaluate("new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)))")
    expect(page.locator("#run-output-summary")).to_contain_text("任务计划 v4")
    context.close()


@pytest.mark.parametrize("motion", ["no-preference", "reduce"])
def test_progress_motion_and_reading_position(workbench, browser, motion):
    store, run_id, url = workbench
    page = browser.new_page(viewport={"width": 1440, "height": 700}, reduced_motion=motion)
    page.add_init_script("""
        window.progressAnimations = [];
        const animate = Element.prototype.animate;
        Element.prototype.animate = function (...args) {
          const animation = animate.apply(this, args);
          if (["run-current-stage", "run-key-output"].includes(this.id)) {
            window.progressAnimations.push(this.id);
          }
          return animation;
        };
    """)
    page.goto(f"{url}/?run={run_id}&lang=zh")
    expect = playwright.expect
    expect(page.locator("#run-current-stage")).to_contain_text("有任务受阻")
    page.get_by_text("产物历史", exact=True).click()
    target = page.locator("#dag-graph [data-task-id='T01']")
    target.evaluate("""node => {
      const stage = document.querySelector("#stage");
      stage.scrollTop += document.querySelector("#topology-title").getBoundingClientRect().top
        - document.querySelector(".stage-head").getBoundingClientRect().bottom - 12;
      node.focus({preventScroll: true});
    }""")
    anchor = page.locator("#topology-title")
    top = anchor.evaluate("node => node.getBoundingClientRect().top")
    assert page.locator("#stage").evaluate("node => node.scrollTop") > 100
    plan = store.get_run(run_id)["plan"]
    store.save_plan(run_id, plan.model_copy(update={"version": 2}))
    expect(page.locator("#run-output-summary")).to_contain_text("任务计划 v2")
    expect(target).to_be_focused()
    assert abs(anchor.evaluate("node => node.getBoundingClientRect().top") - top) <= 2

    store.save_plan(run_id, store.get_run(run_id)["plan"])
    store.append_event(run_id=run_id, stage="system", event_type="budget.settled", actor="test")
    last = store.list_events(run_id)[-1].event_seq
    expect(page.locator("#foot-note")).to_contain_text(f"#{last}")
    before = page.evaluate("window.progressAnimations.slice()")
    assert before.count("run-key-output") == (1 if motion == "no-preference" else 0)

    store.append_event(
        run_id=run_id, stage="graph", event_type="graph.node_started", actor="test",
        payload={"node": "execute"},
    )
    expect(page.locator("#run-current-stage")).to_contain_text("并发执行")
    after = page.evaluate("window.progressAnimations.slice()")
    expected = before + (["run-current-stage"] if motion == "no-preference" else [])
    assert after == expected
    page.close()


def test_submit_feedback_precedes_http_response(workbench, browser):
    _, run_id, url = workbench
    page = browser.new_page()
    page.goto(f"{url}/?run={run_id}&lang=zh")
    page.locator("#nav-oracle-board").click()
    page.locator("#btn-issue-oracle").click()
    page.get_by_label("需求原文").fill("检查提交反馈")
    held = []
    page.route(f"{url}/api/runs", lambda route: held.append(route))
    page.get_by_role("button", name="创建并启动").click()
    expect = playwright.expect
    expect(page.locator("#run-connection")).to_have_text("正在提交…")
    assert len(held) == 1 and held[0].request.method == "POST"
    held[0].fulfill(json={"ok": False, "error": "测试用失败回执"})
    expect(page.get_by_text("创建请求失败：测试用失败回执")).to_be_visible()
    page.close()


def test_historical_execution_attempt_opens_recorded_patch(workbench, browser):
    store, run_id, url = workbench
    root = store.db_path.parent / ".olympus" / "runs" / run_id
    attempt = root / "artifacts" / "T03-R1" / "attempt-1"
    attempt.mkdir(parents=True, exist_ok=True)
    (attempt / "diff.patch").write_text("+recorded attempt one\n", encoding="utf-8")
    store.append_event(
        run_id=run_id, stage="orchestrate", event_type="task.leased", actor="test",
        payload={"task_id": "T03-R1", "lease_generation": 1, "attempt": 1},
    )
    store.append_event(
        run_id=run_id, stage="orchestrate", event_type="task.verifying", actor="test",
        payload={"task_id": "T03-R1", "lease_generation": 1},
    )
    page = browser.new_page()
    page.goto(f"{url}/?run={run_id}&lang=zh")
    page.get_by_role("button", name="查看代码变更").click(timeout=2200)
    expect = playwright.expect
    expect(page.get_by_text("+recorded attempt one", exact=True)).to_be_visible()
    store.append_event(
        run_id=run_id, stage="verify", event_type="review.verdict_recorded", actor="argus",
        payload={"task_id": "T03-R1", "attempt": 2, "passed": False},
    )
    expect(page.locator("#run-output-summary")).to_have_text("T03-R1 · 独立复核 · 未通过")
    page.locator("#run-output-detail").click()
    expect(page.locator("#dock").get_by_role("heading", name="Argus 独立复核")).to_be_visible()
    page.close()


def test_eventsource_error_reopens_from_snapshot(workbench, browser):
    store, run_id, url = workbench
    page = browser.new_page()
    requests = []

    def close_first_stream(route):
        requests.append(route.request.url)
        if len(requests) == 1:
            route.fulfill(status=503, body="stream interrupted")
        else:
            route.continue_()

    page.route("**/stream?after_seq=*", close_first_stream)
    page.goto(f"{url}/?run={run_id}&lang=zh")
    expect = playwright.expect
    expect(page.locator("#run-connection")).to_contain_text("正在重连")
    plan = store.get_run(run_id)["plan"]
    store.save_plan(run_id, plan.model_copy(update={"version": 2}))
    latest = store.list_events(run_id)[-1].event_seq
    expect(page.locator("#run-output-summary")).to_contain_text("任务计划 v2")
    expect(page.locator("#run-connection")).to_be_hidden()
    assert len(requests) == 2 and requests[1].endswith(f"after_seq={latest}")
    page.close()


# ---------------------------------------------------------------------------
# Task Dependency Graph island (React Flow + ELK) — real-browser geometry seams
# ---------------------------------------------------------------------------

SKIP_DEPS = {
    "A": [], "B": [],
    "C": ["A"], "D": ["B"],
    "E": ["C", "A"], "F": ["D"],
    "G": ["E", "B", "D"], "H": ["F"],
}
SKIP_WAVES = {"A": 0, "B": 0, "C": 1, "D": 1, "E": 2, "F": 2, "G": 3, "H": 3}

GRAPH_GEOMETRY = """(host) => {
  const screen = (m, p) => [m.a * p.x + m.c * p.y + m.e, m.b * p.x + m.d * p.y + m.f];
  const cards = Array.from(host.querySelectorAll('[data-task-id]')).map((card) => {
    const r = card.getBoundingClientRect();
    return {id: card.dataset.taskId, x: r.left, y: r.top, w: r.width, h: r.height};
  });
  const edges = Array.from(host.querySelectorAll('.ag-edge')).map((g) => {
    const path = g.querySelector('path');
    const m = path.getScreenCTM();
    const total = path.getTotalLength();
    const points = [];
    for (let s = 0; s <= total; s += 2) points.push(screen(m, path.getPointAtLength(s)));
    const ref = (path.getAttribute('marker-end') || '').match(/#([^)'"]+)/);
    const marker = ref && document.getElementById(ref[1]);
    const style = getComputedStyle(path);
    return {
      source: g.dataset.edgeSource, target: g.dataset.edgeTarget, tone: g.dataset.tone,
      points, end: screen(m, path.getPointAtLength(total)),
      stroke: style.stroke, dash: style.strokeDasharray,
      markerUnits: marker && marker.getAttribute('markerUnits'),
      markerFill: marker && getComputedStyle(marker.querySelector('path')).fill,
    };
  });
  const waves = Array.from(host.querySelectorAll('[data-wave]')).map((w) => {
    const r = w.getBoundingClientRect();
    return {layer: Number(w.dataset.wave), text: w.textContent, x: r.left, bottom: r.bottom};
  });
  const viewport = host.querySelector('.react-flow__viewport');
  return {
    cards, edges, waves,
    transform: viewport ? getComputedStyle(viewport).transform : null,
    hostWidth: host.getBoundingClientRect().width,
  };
}"""

TOKEN_COLOUR = """(name) => {
  const probe = document.createElement('span');
  probe.style.color = `var(${name})`;
  document.body.append(probe);
  const value = getComputedStyle(probe).color;
  probe.remove();
  return value;
}"""


def _token(page, name):
    return page.evaluate(TOKEN_COLOUR, name)


def _wait_for_layout(page, mode="elk"):
    island = page.locator(f"#dag-graph .ag-island[data-layout='{mode}']")
    playwright.expect(island).to_be_visible()
    page.wait_for_function(
        "() => document.querySelectorAll('#dag-graph .ag-edge path').length > 0"
    )
    return page.locator("#dag-graph").evaluate(GRAPH_GEOMETRY)


GRAPH_STYLESHEET = """() => {
  const links = Array.from(document.querySelectorAll('link[rel="stylesheet"]'))
    .filter((node) => node.href.includes('ariadne-graph.css'))
    .map((node) => {
      let rules = 0;
      try {
        rules = node.sheet ? node.sheet.cssRules.length : 0;
      } catch {
        rules = 0;
      }
      return {href: node.href, crossOrigin: node.crossOrigin, rules};
    });
  const viewport = document.querySelector('#dag-graph .react-flow__viewport');
  return {
    links,
    viewportPosition: viewport ? getComputedStyle(viewport).position : null,
  };
}"""


def _assert_graph_stylesheet_applied(page, *, allow_retry=False):
    """Regression #79: ariadne-graph.css must be applied cleanly with a single stylesheet link."""
    state = page.evaluate(GRAPH_STYLESHEET)
    assert len(state["links"]) == 1, state
    stylesheet = state["links"][0]
    parsed = urlparse(stylesheet["href"])
    assert parsed.path.endswith("/static/assets/ariadne-graph.css"), stylesheet
    if not allow_retry:
        assert not parsed.query, stylesheet
    assert stylesheet["crossOrigin"] == "anonymous", stylesheet
    assert stylesheet["rules"] > 0, stylesheet
    assert state["viewportPosition"] == "absolute", state


def _inside(point, card, inset=1.0):
    x, y = point
    return (card["x"] + inset < x < card["x"] + card["w"] - inset
            and card["y"] + inset < y < card["y"] + card["h"] - inset)


@pytest.fixture
def skip_dag(workbench):
    store, _, url = workbench
    run_id = store.create_run(title="跨层依赖 DAG")
    spec = GoalSpec(goal="跨层依赖", requirements=[RequirementItem(
        id="R1", description="多层跳跃依赖", acceptance_criteria=["路径不穿过卡片"],
    )])
    store.save_spec(run_id, spec)
    store.record_approval(ApprovalRecord(
        run_id=run_id, target=ApprovalTarget.GOAL_SPEC, target_id=spec.spec_id,
        target_hash=spec.content_hash, approver="human", decision=ApprovalDecision.APPROVED,
    ))
    review = AcceptanceCheck(id="review", kind=CheckKind.INDEPENDENT_REVIEW, description="复核")
    plan = TaskPlan(spec_id=spec.spec_id, spec_hash=spec.content_hash, tasks=[
        TaskNode(task_id=task_id, title=f"任务 {task_id}", objective="跨层", depends_on=deps,
                 write_scope=[f"{task_id.lower()}.py"], acceptance_checks=[review])
        for task_id, deps in SKIP_DEPS.items()
    ])
    store.save_plan(run_id, plan)
    store.record_approval(ApprovalRecord(
        run_id=run_id, target=ApprovalTarget.TASK_PLAN, target_id=plan.plan_id,
        target_hash=plan.content_hash, approver="human", decision=ApprovalDecision.APPROVED,
    ))
    return store, run_id, url


def test_task_dag_island_keeps_fixed_scale_and_routes_around_cards(skip_dag, browser):
    _, run_id, url = skip_dag
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.goto(f"{url}/?run={run_id}")
    geo = _wait_for_layout(page)
    _assert_graph_stylesheet_applied(page)
    cards = {card["id"]: card for card in geo["cards"]}
    assert set(cards) == set(SKIP_DEPS)

    # Fixed 1:1 physical card size, never stretched to the (much wider) frame.
    assert geo["transform"] in ("none", "matrix(1, 0, 0, 1, 0, 0)") or geo["transform"].startswith(
        "matrix(1, 0, 0, 1,"
    ), geo["transform"]
    for card in cards.values():
        assert abs(card["w"] - 184) <= 0.5 and abs(card["h"] - 76) <= 0.5, card
    assert geo["hostWidth"] > 4 * 184 + 3 * 96

    # Wave columns: one shared x per wave, ordered left to right, under a matching header.
    wave_x = {}
    for task_id, wave in SKIP_WAVES.items():
        wave_x.setdefault(wave, cards[task_id]["x"])
        assert abs(cards[task_id]["x"] - wave_x[wave]) <= 0.5, (task_id, wave)
    assert [wave_x[w] for w in sorted(wave_x)] == sorted(wave_x.values())
    headers = {header["layer"]: header for header in geo["waves"]}
    assert sorted(headers) == [0, 1, 2, 3]
    for wave, header in headers.items():
        assert header["text"] == f"Wave {wave}"
        assert abs(header["x"] - wave_x[wave]) <= 1
        assert header["bottom"] <= min(cards[t]["y"] for t, w in SKIP_WAVES.items() if w == wave)

    # Every dependency renders; no edge passes through a non-endpoint card.
    edges = {(edge["source"], edge["target"]): edge for edge in geo["edges"]}
    assert set(edges) == {(dep, task) for task, deps in SKIP_DEPS.items() for dep in deps}
    for (source, target), edge in edges.items():
        for card_id, card in cards.items():
            if card_id in (source, target):
                continue
            hits = [p for p in edge["points"] if _inside(p, card)]
            assert not hits, f"{source}->{target} crosses {card_id} at {hits[:3]}"

    # Multi-parent G: incoming heads land on G's west side at separated y coordinates.
    incoming = [edges[(dep, "G")]["end"] for dep in SKIP_DEPS["G"]]
    for x, _ in incoming:
        assert abs(x - cards["G"]["x"]) <= 2
    ys = sorted(y for _, y in incoming)
    assert all(b - a >= 6 for a, b in zip(ys, ys[1:], strict=False)), ys

    # Fixed-size heads whose fill matches each edge's stroke tone.
    ink, rule = _token(page, "--text-ink"), _token(page, "--rule-hard")
    for edge in edges.values():
        assert edge["markerUnits"] == "userSpaceOnUse"
        assert edge["markerFill"] == edge["stroke"]
        assert edge["stroke"] == (ink if edge["tone"] == "critical" else rule)
    assert edges[("E", "G")]["tone"] == "critical"
    assert edges[("B", "G")]["tone"] == "default"
    page.close()


def test_task_dag_island_separates_selection_from_critical_path(workbench, browser):
    _, run_id, url = workbench
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(f"{url}/?run={run_id}")
    expect = playwright.expect
    geo = _wait_for_layout(page)
    edges = {(edge["source"], edge["target"]): edge for edge in geo["edges"]}

    ink, bad, live, accent = (
        _token(page, name) for name in ("--text-ink", "--bad", "--live", "--accent")
    )
    assert accent != ink
    assert edges[("T01", "T03")]["tone"] == "critical"
    assert edges[("T01", "T03")]["stroke"] == ink
    blocked = edges[("T03-R1", "T06")]
    assert blocked["tone"] == "blocked" and blocked["stroke"] == bad
    assert blocked["dash"] not in ("none", "") and blocked["markerFill"] == bad
    running = edges[("T03", "T05")]
    assert running["tone"] == "running" and running["stroke"] == live

    def ring(task_id):
        return expect(page.locator(f"#dag-graph [data-task-id='{task_id}']"))

    blocked_card = page.locator("#dag-graph [data-task-id='T03-R1']")
    expect(blocked_card).to_have_attribute("aria-selected", "true")
    expect(blocked_card).to_have_attribute("data-status", "blocked")
    expect(blocked_card).to_have_attribute("role", "button")
    expect(blocked_card).to_have_attribute("tabindex", "0")
    ring("T03-R1").to_have_css("outline-color", accent)
    ring("T03-R1").to_have_css("outline-style", "solid")
    ring("T03-R1").to_have_css("outline-width", "2px")
    ring("T01").not_to_have_css("outline-color", accent)

    card = page.locator("#dag-graph [data-task-id='T05']")
    card.focus()
    page.keyboard.press("Enter")
    expect(card).to_have_attribute("aria-selected", "true")
    expect(blocked_card).to_have_attribute("aria-selected", "false")
    expect(page.locator("body")).to_have_attribute("data-dock", "open")
    expect(page.locator("#dock-title")).to_contain_text("T05")
    expect(card).to_be_focused()

    first = page.locator("#dag-graph [data-task-id='T01']")
    first.focus()
    page.keyboard.press(" ")
    expect(first).to_have_attribute("aria-selected", "true")
    expect(page.locator("#dock-title")).to_contain_text("T01")
    page.locator("#dag-graph [data-task-id='T02']").click()
    expect(page.locator("#dock-title")).to_contain_text("T02")
    ring("T02").to_have_css("outline-color", accent)
    ring("T03").not_to_have_css("outline-color", accent)

    # Dark theme: cards and edges re-read tokens in place, without remounting.
    light_surface = first.evaluate("node => getComputedStyle(node).backgroundColor")
    page.locator("#theme-toggle").click()
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    assert _token(page, "--ok-soft") != light_surface
    expect(first).to_have_css("background-color", _token(page, "--ok-soft"))
    dark = page.locator("#dag-graph").evaluate(GRAPH_GEOMETRY)
    dark_edges = {(edge["source"], edge["target"]): edge for edge in dark["edges"]}
    assert dark_edges[("T01", "T03")]["stroke"] == _token(page, "--text-ink") != ink
    ring("T02").to_have_css("outline-color", _token(page, "--accent"))
    assert len(dark["cards"]) == len(geo["cards"])
    assert first.evaluate("node => node.isConnected")
    assert not errors, errors
    page.close()


def test_task_dag_island_preserves_viewport_on_live_updates(workbench, browser):
    store, run_id, url = workbench
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.goto(f"{url}/?run={run_id}&lang=zh")
    _wait_for_layout(page)
    host = page.locator("#dag-graph")
    viewport = host.locator(".react-flow__viewport")
    before = viewport.evaluate("node => getComputedStyle(node).transform")
    host.scroll_into_view_if_needed()
    box = host.bounding_box()
    page.mouse.move(box["x"] + 12, box["y"] + box["height"] - 12)
    page.mouse.down()
    page.mouse.move(box["x"] + 72, box["y"] + box["height"] - 32, steps=6)
    page.mouse.up()
    panned = viewport.evaluate("node => getComputedStyle(node).transform")
    assert panned != before
    card = host.locator("[data-task-id='T05']").element_handle()

    task = next(task for task in store.get_run(run_id)["plan"].tasks if task.task_id == "T05")
    store.append_event(
        run_id=run_id, stage="orchestrate", event_type="agent.tool_started", actor="hephaestus:T05",
        payload={
            "task_id": "T05", "attempt": task.attempt, "actor": "hephaestus:T05",
            "driver": "native", "step": 2, "tool": "read_file", "target_hint": "workspace file",
        },
    )
    playwright.expect(host.locator("[data-task-id='T05']")).to_contain_text("native · read_file")
    assert viewport.evaluate("node => getComputedStyle(node).transform") == panned
    host.locator("[data-task-id='T05']").click()
    playwright.expect(page.locator("#dock")).to_contain_text("当前工具")
    playwright.expect(page.locator("#dock")).to_contain_text("read_file")

    plan = store.get_run(run_id)["plan"]
    store.save_plan(run_id, plan.model_copy(update={"version": 2}))
    playwright.expect(page.locator("#run-output-summary")).to_contain_text("任务计划 v2")
    assert viewport.evaluate("node => getComputedStyle(node).transform") == panned
    assert card.evaluate("node => node.isConnected")
    page.close()


def test_task_dag_shows_inline_titles_within_fixed_cards(workbench, browser):
    _, run_id, url = workbench
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    long_titles = {
        "T02": "实现报名注册以及同账号唯一校验和容量上限事务锁",
        "T04": "Implement the registration status endpoint and the waiting list position display",
    }

    def set_graph_titles(route):
        response = route.fetch()
        data = response.json()
        nodes = {node["task_id"]: node for node in data["projection"]["dag"]["nodes"]}
        for task_id, title in long_titles.items():
            nodes[task_id].update(title=title, attempt=2)
        nodes["T03"]["title"] = "   "
        nodes["T05"].pop("title")
        route.fulfill(response=response, json=data)

    page.route(f"**/api/runs/{run_id}", set_graph_titles)
    page.goto(f"{url}/?run={run_id}&nosse=1&lang=zh")
    _wait_for_layout(page)
    expect = playwright.expect
    first = page.locator("#dag-graph .ag-task-card[data-task-id='T01']")
    expect(first.locator(".ag-task-title")).to_have_text("· 接口类型与测试夹具")
    expect(first).to_have_attribute("aria-label", re.compile("接口类型与测试夹具"))
    assert first.locator(".ag-task-title").evaluate("(el) => el.scrollWidth <= el.clientWidth")

    for task_id, title in long_titles.items():
        card = page.locator(f"#dag-graph .ag-task-card[data-task-id='{task_id}']")
        label = card.locator(".ag-task-title")
        expect(label).to_have_text(f"· {title}")
        expect(label).to_have_css("white-space", "nowrap")
        expect(label).to_have_css("text-overflow", "ellipsis")
        expect(card).to_have_attribute("title", title)
        expect(card.locator(".ag-card-pill")).to_have_text("×2")
        expect(card.locator(".ag-card-status")).to_be_visible()
        assert label.evaluate("(el) => el.scrollWidth > el.clientWidth")
        label_box = label.bounding_box()
        badge_box = card.locator(".ag-card-pill").bounding_box()
        assert label_box["x"] + label_box["width"] <= badge_box["x"] + 1
        assert abs(card.bounding_box()["width"] - 184) <= 0.5

    for task_id in ("T03", "T05"):
        untitled = page.locator(f"#dag-graph .ag-task-card[data-task-id='{task_id}']")
        expect(untitled.locator(".ag-task-heading")).to_have_text(task_id)
        expect(untitled.locator(".ag-task-title")).to_have_count(0)
        assert untitled.get_attribute("title") is None

    page.locator("#dag-graph .ag-task-card[data-task-id='T02']").click()
    expect(page.locator("#dock-title")).to_have_text(f"T02 · {long_titles['T02']}")

    first.focus()
    first.press("Enter")
    expect(page.locator("#dock-title")).to_contain_text("T01")
    page.close()


def test_task_dag_island_empty_cycle_fallback_and_missing_bundle(workbench, browser):
    _, run_id, url = workbench
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(f"{url}/?run={run_id}&lang=zh")
    _wait_for_layout(page)
    _assert_graph_stylesheet_applied(page)
    expect = playwright.expect

    page.evaluate("""() => {
      const make = (id) => {
        const host = document.createElement('div');
        host.id = id;
        host.style.width = '900px';
        document.querySelector('#view-topology').prepend(host);
        return host;
      };
      const view = (id, layer) => ({
        id, layer, status: 'waiting', tone: 'muted', meta: 'engineer · implement',
        statusText: '◷ 等待', ariaLabel: `任务 ${id} ◷ 等待`,
      });
      window.emptyHandle = window.AriadneGraph.mountDag(make('graph-empty'), {
        runId: 'empty', nodes: [], edges: [], emptyMessage: '当前运行尚未生成任务 DAG',
      });
      window.cycleHandle = window.AriadneGraph.mountDag(make('graph-cycle'), {
        runId: 'cycle',
        nodes: [view('X', 0), view('Y', 1), view('Z', 2)],
        edges: [
          {source: 'X', target: 'Y'}, {source: 'Y', target: 'Z'}, {source: 'Z', target: 'X'},
          {source: 'Z', target: 'ghost'},
        ],
      });
    }""")
    empty = page.locator("#graph-empty")
    expect(empty.get_by_text("当前运行尚未生成任务 DAG", exact=True)).to_be_visible()
    expect(empty.locator("[data-task-id]")).to_have_count(0)

    cycle = page.locator("#graph-cycle")
    expect(cycle.locator(".ag-island[data-layout='fallback']")).to_be_visible()
    expect(cycle.locator("[data-task-id]")).to_have_count(3)
    expect(cycle.locator(".ag-edge path")).to_have_count(3)
    for task_id in ("X", "Y", "Z"):
        box = cycle.locator(f"[data-task-id='{task_id}']").bounding_box()
        assert abs(box["width"] - 184) <= 0.5 and abs(box["height"] - 76) <= 0.5

    page.evaluate("""() => window.cycleHandle.update({
      runId: 'cycle', selectedId: 'Y',
      nodes: ['X', 'Y', 'Z'].map((id, layer) => ({
        id, layer, status: 'running', tone: 'live', meta: 'engineer · implement',
        statusText: '▶ 运行中', ariaLabel: `任务 ${id} ▶ 运行中`,
      })),
      edges: [{source: 'X', target: 'Y'}, {source: 'Y', target: 'Z'}, {source: 'Z', target: 'X'}],
    })""")
    expect(cycle.locator("[data-task-id='Y']")).to_have_attribute("aria-selected", "true")
    expect(cycle.locator("[data-task-id='Y']")).to_have_attribute("data-status", "running")
    page.evaluate("() => { window.cycleHandle.destroy(); window.emptyHandle.destroy(); }")
    expect(cycle.locator(".ag-island")).to_have_count(0)
    page.close()

    broken = browser.new_page()
    broken.route("**/static/assets/ariadne-graph*.js", lambda route: route.abort())
    broken.goto(f"{url}/?run={run_id}&lang=zh")
    expect(broken.locator("#dag-graph")).to_contain_text("任务依赖图组件未能加载")
    expect(broken.locator("#workflow-graph[data-workflow-view='staged']")).to_be_visible()
    expect(broken.locator("#workflow-graph [data-workflow-node]")).to_have_count(10)
    expect(broken.locator("#workflow-badge")).to_contain_text("产品蓝图")
    expect(broken.locator("#run-current-stage")).to_contain_text("有任务受阻")
    broken.unroute("**/static/assets/ariadne-graph*.js")
    broken.locator("#nav-oracle-board").click()
    broken.locator(f".oracle-card:not([data-run-id='{run_id}']) .oracle-card-trigger").first.click()
    expect(broken.locator("#dag-graph .ag-island")).to_be_visible()
    expect(broken.locator("#workflow-graph[data-workflow-view='staged']")).to_be_visible()
    _assert_graph_stylesheet_applied(broken)
    broken.close()

    for label, handler in (
        ("http-503", lambda route: route.fulfill(status=503, body="css unavailable")),
        ("abort", lambda route: route.abort()),
    ):
        css_broken = browser.new_page()
        css_broken.route("**/static/assets/ariadne-graph*.css*", handler)
        css_broken.goto(f"{url}/?run={run_id}&lang=zh")
        expect(css_broken.locator("#dag-graph")).to_contain_text("任务依赖图组件未能加载")
        expect(css_broken.locator("#workflow-graph[data-workflow-view='staged']")).to_be_visible()
        assert css_broken.locator("#dag-graph .ag-island").count() == 0, label
        css_broken.unroute("**/static/assets/ariadne-graph*.css*")
        css_broken.locator("#nav-oracle-board").click()
        css_broken.locator(
            f".oracle-card:not([data-run-id='{run_id}']) .oracle-card-trigger"
        ).first.click()
        expect(css_broken.locator("#dag-graph .ag-island")).to_be_visible()
        expect(css_broken.locator("#workflow-graph[data-workflow-view='staged']")).to_be_visible()
        _assert_graph_stylesheet_applied(css_broken, allow_retry=True)
        css_broken.close()

    stalled_css = browser.new_page()
    stalled_css.route("**/static/assets/ariadne-graph*.css*", lambda _route: None)
    stalled_css.goto(f"{url}/?run={run_id}&lang=zh")
    expect(stalled_css.locator("#dag-graph")).to_contain_text(
        "ariadne-graph.css load timed out", timeout=7000
    )
    expect(stalled_css.locator("#workflow-graph[data-workflow-view='staged']")).to_be_visible()
    assert stalled_css.locator("#dag-graph .ag-island").count() == 0
    stalled_css.unroute("**/static/assets/ariadne-graph*.css*")
    stalled_css.locator("#nav-oracle-board").click()
    stalled_css.locator(
        f".oracle-card:not([data-run-id='{run_id}']) .oracle-card-trigger"
    ).first.click()
    expect(stalled_css.locator("#dag-graph .ag-island")).to_be_visible()
    expect(stalled_css.locator("#workflow-graph[data-workflow-view='staged']")).to_be_visible()
    _assert_graph_stylesheet_applied(stalled_css, allow_retry=True)
    stalled_css.close()

    held_js = []
    held_retry_css = []
    late_js = browser.new_page()
    late_js.clock.install()
    initial_js_re = re.compile(r"/static/assets/ariadne-graph\.js$")
    initial_css_re = re.compile(r"/static/assets/ariadne-graph\.css$")
    retry_css_re = re.compile(r"/static/assets/ariadne-graph\.css\?retry=")
    late_js.route(
        initial_css_re,
        lambda route: route.fulfill(status=200, content_type="text/css", body=""),
    )
    late_js.route(initial_js_re, lambda route: held_js.append(route))
    late_js.route(retry_css_re, lambda route: held_retry_css.append(route))
    late_js.goto(f"{url}/?run={run_id}&nosse=1&lang=zh")
    _wait_for_intercepted(late_js, held_js)
    late_js.clock.fast_forward(20_500)
    expect(late_js.locator("#dag-graph")).to_contain_text(
        "任务依赖图组件未能加载", timeout=7000
    )
    late_js.locator("#nav-oracle-board").click()
    late_js.locator(
        f".oracle-card:not([data-run-id='{run_id}']) .oracle-card-trigger"
    ).first.click()
    late_js.wait_for_function(
        "() => Array.from(document.querySelectorAll('link[rel=\"stylesheet\"]'))"
        ".some((node) => node.href.includes('ariadne-graph.css?retry='))"
    )
    late_js.unroute(initial_css_re)
    with late_js.expect_response(initial_js_re):
        for route in held_js:
            route.continue_()
    late_js.unroute(initial_js_re)
    late_js.wait_for_timeout(200)
    assert late_js.evaluate(
        "() => Array.from(document.querySelectorAll('link[rel=\"stylesheet\"]'))"
        ".some((node) => node.href.includes('ariadne-graph.css?retry='))"
    )
    for route in held_retry_css:
        route.continue_()
    late_js.unroute(retry_css_re)
    expect(late_js.locator("#dag-graph .ag-island")).to_be_visible()
    expect(late_js.locator("#workflow-graph[data-workflow-view='staged']")).to_be_visible()
    _assert_graph_stylesheet_applied(late_js, allow_retry=True)
    late_js.close()


def _wait_for_intercepted(page, routes: list, *, timeout_s: float = 7.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not routes and time.monotonic() < deadline:
        page.wait_for_timeout(50)
    assert routes, "expected intercepted request was not observed before timeout"


def test_task_dag_bundle_slow_download_survives_stylesheet_timeout_and_retries_in_place(
    workbench, browser
):
    _, run_id, url = workbench
    expect = playwright.expect

    # 1. Slow JS bundle (> 4s CSS timeout, < 20s bundle timeout) must succeed once CSS loads.
    slow = browser.new_page()
    slow.clock.install()
    held_slow_js = []
    initial_js_re = re.compile(r"/static/assets/ariadne-graph\.js$")
    slow.route(initial_js_re, lambda route: held_slow_js.append(route))
    slow.goto(f"{url}/?run={run_id}&nosse=1&lang=zh")
    slow.wait_for_function(
        "() => Array.from(document.querySelectorAll('link[rel=\"stylesheet\"]'))"
        ".some((node) => node.href.includes('ariadne-graph.css')"
        " && node.sheet && node.sheet.cssRules.length > 0)"
    )
    _wait_for_intercepted(slow, held_slow_js)
    slow.clock.fast_forward(5_000)
    expect(slow.locator("#dag-graph")).not_to_contain_text("任务依赖图组件未能加载")
    for route in held_slow_js:
        route.continue_()
    slow.unroute(initial_js_re)
    expect(slow.locator("#dag-graph .ag-island")).to_be_visible()
    _assert_graph_stylesheet_applied(slow)
    slow.close()

    # 2. JS bundle stalling past 20s surfaces fallback Notice + one-click retry that recovers
    #    in place (even across a failed retry attempt) without switching runs.
    timed_out = browser.new_page()
    timed_out.clock.install()
    stalled_js = []
    timed_out.route(initial_js_re, lambda route: stalled_js.append(route))
    timed_out.goto(f"{url}/?run={run_id}&nosse=1&lang=zh")
    timed_out.wait_for_function(
        "() => Array.from(document.querySelectorAll('link[rel=\"stylesheet\"]'))"
        ".some((node) => node.href.includes('ariadne-graph.css')"
        " && node.sheet && node.sheet.cssRules.length > 0)"
    )
    _wait_for_intercepted(timed_out, stalled_js)
    timed_out.clock.fast_forward(5_000)
    expect(timed_out.locator("#dag-graph")).not_to_contain_text("任务依赖图组件未能加载")
    timed_out.clock.fast_forward(16_000)
    expect(timed_out.locator("#dag-graph")).to_contain_text("任务依赖图组件未能加载")
    expect(timed_out.locator("#dag-graph")).to_contain_text("ariadne-graph bundle load timed out")
    retry_btn = timed_out.locator("#dag-graph").get_by_role("button", name="重试加载")
    expect(retry_btn).to_be_visible()

    # First retry fails over the network: boundary must stay recoverable and clear pending state.
    retry_js_re = re.compile(r"/static/assets/ariadne-graph\.js\?retry=")
    timed_out.route(retry_js_re, lambda route: route.abort())
    retry_btn.click()
    expect(timed_out.locator("#dag-graph")).to_contain_text("任务依赖图组件未能加载")
    expect(retry_btn).to_be_visible()

    # Second retry succeeds in place on the same run without navigating away.
    timed_out.unroute(retry_js_re)
    timed_out.unroute(initial_js_re)
    retry_btn.click()
    expect(timed_out.locator("#dag-graph .ag-island")).to_be_visible()
    _assert_graph_stylesheet_applied(timed_out, allow_retry=True)
    timed_out.close()


# ---------------------------------------------------------------------------
# Staged Orchestration Workflow board — phases, kinds, shortcuts & themes
# ---------------------------------------------------------------------------

WORKFLOW_STAGES_GEOMETRY = """(host) => {
  const stages = Array.from(host.querySelectorAll('.wf-stage-col')).map((col) => {
    const r = col.getBoundingClientRect();
    return {
      id: col.dataset.stageId,
      state: col.dataset.stageState,
      text: col.textContent,
      nodes: Array.from(col.querySelectorAll('[data-workflow-node]')).map(
        (el) => el.dataset.workflowNode
      ),
      x: r.left, y: r.top, w: r.width, h: r.height,
    };
  });
  const cards = Array.from(host.querySelectorAll('[data-workflow-node]')).map((card) => {
    const r = card.getBoundingClientRect();
    const style = getComputedStyle(card);
    const badge = card.querySelector('.wf-step-kind');
    const routePill = card.querySelector('.wf-route-pill');
    return {
      id: card.dataset.workflowNode, status: card.dataset.status, kind: card.dataset.nodeKind,
      text: card.textContent, x: r.left, y: r.top, w: r.width, h: r.height,
      borderStyle: style.borderLeftStyle, borderTopStyle: style.borderTopStyle,
      background: style.backgroundColor, outline: style.outlineColor,
      badge: badge ? badge.textContent : null,
      badgeBorder: badge ? getComputedStyle(badge).borderTopColor : null,
      routePill: routePill ? routePill.textContent : null,
      marker: getComputedStyle(card, '::before').backgroundColor,
    };
  });
  const box = host.getBoundingClientRect();
  return {
    stages, cards,
    scrollWidth: host.scrollWidth,
    clientWidth: host.clientWidth,
    host: {x: box.left, y: box.top, w: box.width, h: box.height},
  };
}"""


def _wait_for_workflow(page, host="#workflow-graph"):
    playwright.expect(page.locator(f"{host}[data-workflow-view='staged']")).to_be_visible()
    playwright.expect(page.locator(f"{host} .wf-stage-col").first).to_be_visible()
    return page.locator(host).evaluate(WORKFLOW_STAGES_GEOMETRY)


def _unscaled(transform):
    return transform in ("none",) or transform.startswith("matrix(1, 0, 0, 1,")


@pytest.fixture
def skipped_workflow(workbench):
    """full_sdlc_tdd run whose router jumped align -> gate_spec, bypassing critic."""
    store, _, url = workbench
    run_id = store.create_run(title="跳过评审的全流程", workflow_name="full_sdlc_tdd")
    for event_type, payload in (
        ("graph.started", {"graph": "full_sdlc_tdd", "start_from": None}),
        ("graph.node_started", {"node": "intake"}),
        ("graph.node_finished", {"node": "intake"}),
        ("graph.routed", {"from": ["intake"], "to": ["align"], "wave": 1}),
        ("graph.node_started", {"node": "align"}),
        ("graph.node_finished", {"node": "align"}),
        ("graph.routed", {"from": ["align"], "to": ["gate_spec"], "wave": 2}),
        ("graph.node_started", {"node": "gate_spec"}),
        ("graph.paused", {"node": "gate_spec", "reason": "human_gate"}),
    ):
        store.append_event(
            run_id=run_id, stage="graph", event_type=event_type, actor="graph", payload=payload,
        )
    return store, run_id, url


def test_staged_workflow_groups_phases_without_horizontal_overflow(workbench, browser):
    store, run_id, url = workbench
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.goto(f"{url}/?run={run_id}&lang=zh")
    dag = _wait_for_layout(page)
    flow = _wait_for_workflow(page)
    cards = {card["id"]: card for card in flow["cards"]}
    assert len(cards) == 10

    assert _unscaled(dag["transform"]), dag["transform"]
    for card in dag["cards"]:
        assert abs(card["w"] - 184) <= 0.5 and abs(card["h"] - 76) <= 0.5, card

    assert [stage["id"] for stage in flow["stages"]] == ["align", "design", "plan"]
    assert flow["stages"][0]["nodes"] == ["intake", "align", "critic", "gate_spec"]
    assert flow["stages"][1]["nodes"] == ["design", "validate_design", "gate_design"]
    assert flow["stages"][2]["nodes"] == ["plan", "validate_plan", "gate_plan"]
    assert flow["scrollWidth"] <= flow["clientWidth"] + 1
    assert page.locator("#workflow-graph [data-graph-control]").count() == 0
    for card in cards.values():
        assert card["x"] >= flow["host"]["x"] - 1
        assert card["x"] + card["w"] <= flow["host"]["x"] + flow["host"]["w"] + 1

    for name in ("gate_spec", "gate_design", "gate_plan"):
        assert cards[name]["kind"] == "gate" and cards[name]["badge"] == "关口"
    for name in ("validate_design", "validate_plan"):
        assert cards[name]["kind"] == "validator" and cards[name]["badge"] == "校验"
        assert cards[name]["borderTopStyle"] == "double"
    for name in ("intake", "align", "critic", "design", "plan"):
        assert cards[name]["kind"] == "stage" and cards[name]["badge"] is None
    accent = _token(page, "--accent")
    assert cards["gate_spec"]["marker"] == accent
    assert cards["gate_spec"]["badgeBorder"] == accent
    assert cards["align"]["marker"] != accent

    fast_id = store.create_run(title="快速补丁阶段适配", workflow_name="repo_fast_patch")
    page.goto(f"{url}/?run={fast_id}&lang=zh")
    fast_flow = _wait_for_workflow(page)
    assert [stage["id"] for stage in fast_flow["stages"]] == ["align", "plan", "execute"]
    assert "STAGE 01" in fast_flow["stages"][0]["text"]
    assert "STAGE 03" in fast_flow["stages"][1]["text"]
    assert "STAGE 04" in fast_flow["stages"][2]["text"]
    assert len(fast_flow["cards"]) == 6
    page.close()


def test_staged_workflow_marks_skipped_critic_and_shortcut_badge(skipped_workflow, browser):
    _, run_id, url = skipped_workflow
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(f"{url}/?run={run_id}&lang=zh")
    flow = _wait_for_workflow(page)
    cards = {card["id"]: card for card in flow["cards"]}
    assert len(cards) == 11
    assert [stage["id"] for stage in flow["stages"]] == ["align", "design", "plan", "execute"]
    assert flow["stages"][0]["state"] == "waiting"

    statuses = {name: card["status"] for name, card in cards.items()}
    assert statuses["intake"] == statuses["align"] == "completed"
    assert statuses["critic"] == "skipped"
    assert statuses["gate_spec"] == "paused"
    assert {statuses[n] for n in ("design", "plan", "execute")} == {"pending"}
    assert "— 已跳过" in cards["critic"]["text"]
    assert "◷ 待确认" in cards["gate_spec"]["text"]
    assert cards["align"]["routePill"] == "直达关口"

    # Skipped collapses into a shorter recessed dashed pill; waiting stays solid.
    assert cards["critic"]["borderStyle"] == "dashed"
    assert cards["critic"]["background"] == _token(page, "--surface-sunk")
    assert cards["critic"]["h"] < cards["align"]["h"]
    assert cards["design"]["borderStyle"] == "solid"
    assert cards["design"]["background"] == _token(page, "--surface-raised")

    # All 11 nodes fit simultaneously inside the host frame without panning.
    host = flow["host"]
    for card in cards.values():
        assert host["x"] - 1 <= card["x"] and card["x"] + card["w"] <= host["x"] + host["w"] + 1
    expect = playwright.expect
    expect(page.locator("#workflow-badge")).to_contain_text("当前 需求确认 (gate_spec)")
    page.close()


def test_staged_workflow_selection_opens_dock_and_follows_dark_theme(skipped_workflow, browser):
    _, run_id, url = skipped_workflow
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(f"{url}/?run={run_id}&lang=zh")
    _wait_for_workflow(page)
    expect = playwright.expect
    accent = _token(page, "--accent")

    def node(name):
        return page.locator(f"#workflow-graph [data-workflow-node='{name}']")

    critic = node("critic")
    expect(critic).to_have_attribute("role", "button")
    expect(critic).to_have_attribute("tabindex", "0")
    expect(critic).to_have_attribute("aria-selected", "false")
    critic.click()
    expect(critic).to_have_attribute("aria-selected", "true")
    expect(critic).to_have_css("outline-color", accent)
    expect(critic).to_have_css("outline-style", "solid")
    expect(page.locator("body")).to_have_attribute("data-dock", "open")
    expect(page.locator("#dock-title")).to_have_text("多维评审 · critic")

    gate = node("gate_spec")
    gate.focus()
    page.keyboard.press("Enter")
    expect(gate).to_have_attribute("aria-selected", "true")
    expect(critic).to_have_attribute("aria-selected", "false")
    expect(page.locator("#dock-title")).to_have_text("需求确认 · gate_spec")
    expect(gate).to_be_focused()

    align = node("align")
    align.focus()
    page.keyboard.press(" ")
    expect(align).to_have_attribute("aria-selected", "true")
    expect(page.locator("#dock-title")).to_have_text("需求对齐 · align")
    expect(align).to_have_css("outline-color", accent)
    expect(gate).not_to_have_css("outline-color", accent)

    # Dark theme: stage cards and selection ring re-read tokens in place without remounting.
    light_sunk = _token(page, "--surface-sunk")
    page.locator("#theme-toggle").click()
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    assert _token(page, "--surface-sunk") != light_sunk
    expect(critic).to_have_css("background-color", _token(page, "--surface-sunk"))
    expect(align).to_have_css("outline-color", _token(page, "--accent"))
    assert critic.evaluate("node => node.isConnected")
    assert not errors, errors
    page.close()


def test_staged_workflow_empty_unmapped_and_failed_state(workbench, browser):
    store, _, url = workbench
    run_id = store.create_run(title="旧版工作流", workflow_name="legacy_unmapped_flow")
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(f"{url}/?run={run_id}&lang=zh")
    expect = playwright.expect
    host = page.locator("#workflow-graph")
    expect(host.get_by_text("当前工作流未提供拓扑描述", exact=True)).to_be_visible()
    expect(host.locator("[data-workflow-node]")).to_have_count(0)
    expect(page.locator("#workflow-badge")).to_contain_text("拓扑不可用")

    failed_id = store.create_run(title="计划校验失败", workflow_name="full_sdlc_tdd")
    for event_type, payload in (
        ("graph.started", {"graph": "full_sdlc_tdd", "start_from": None}),
        ("graph.node_started", {"node": "plan"}),
        ("graph.node_finished", {"node": "plan"}),
        ("graph.routed", {"from": ["plan"], "to": ["validate_plan"], "wave": 1}),
        ("graph.node_started", {"node": "validate_plan"}),
        ("graph.node_error", {"node": "validate_plan", "error": "scope overlap"}),
    ):
        store.append_event(
            run_id=failed_id, stage="graph", event_type=event_type, actor="graph", payload=payload,
        )
    page.goto(f"{url}/?run={failed_id}&lang=zh")
    _wait_for_workflow(page)
    val_plan = page.locator("#workflow-graph [data-workflow-node='validate_plan']")
    expect(val_plan).to_have_attribute("data-status", "error")
    expect(val_plan).to_have_css("background-color", _token(page, "--bad-soft"))
    expect(page.locator("#workflow-graph .wf-stage-col[data-stage-id='plan']")).to_have_attribute(
        "data-stage-state", "failed"
    )
    assert not errors, errors
    page.close()


def test_staged_workflow_custom_nodes_display_order_and_narrow_containers(workbench, browser):
    _, run_id, url = workbench
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))

    long_target = "very_long_custom_verification_target_node_name_for_narrow_overflow_guard"

    def patch_custom_workflow(route):
        resp = route.fetch()
        data = resp.json()
        proj = data.get("projection")
        if isinstance(proj, dict) and proj.get("run_id") == run_id:
            proj["workflow"] = {
                **proj["workflow"],
                "graph": {
                    "nodes": [
                        {"name": "plan", "kind": "pm_decompose"},
                        {"name": "intake", "kind": "intake_triage"},
                        {"name": "align", "kind": "align_goal"},
                        {"name": "critic", "kind": "critic_parallel"},
                        {"name": "design_spec_draft", "kind": "custom_step"},
                        {"name": "toString", "kind": "custom_step"},
                        {"name": long_target, "kind": "custom_step"},
                    ],
                    "edges": [
                        {"source": "intake", "target": long_target, "conditional": True},
                        {"source": "plan", "target": "intake", "conditional": True},
                        {"source": "intake", "target": "plan", "conditional": True},
                    ],
                },
                "node_statuses": {
                    "intake": "completed",
                    "align": "completed",
                    "critic": "completed",
                    "design_spec_draft": "pending",
                    "plan": "completed",
                    "toString": "pending",
                    long_target: "pending",
                },
                "traversed_edges": [
                    {"source": "intake", "target": long_target},
                    {"source": "plan", "target": "intake"},
                ],
            }
            route.fulfill(response=resp, json=data)
            return
        route.fulfill(response=resp)

    page.route(f"**/api/runs/{run_id}*", patch_custom_workflow)
    page.goto(f"{url}/?run={run_id}&lang=zh")
    flow = _wait_for_workflow(page)
    by_stage = {stage["id"]: stage for stage in flow["stages"]}
    assert "design_spec_draft" in by_stage["design"]["nodes"]
    assert "toString" in by_stage["other"]["nodes"]

    cards = {card["id"]: card for card in flow["cards"]}
    assert cards["plan"]["routePill"] == "回退输入分流"
    assert cards["intake"]["routePill"] == f"直达{long_target}"

    align_connectors = page.locator(
        "#workflow-graph .wf-stage-col[data-stage-id='align'] .wf-step-connector"
    ).evaluate_all("els => els.map(el => el.dataset.connector)")
    assert align_connectors == ["pending", "pending"]

    narrow = page.locator("#workflow-graph").evaluate("""(host) => {
      host.style.width = '290px';
      return { scrollWidth: host.scrollWidth, clientWidth: host.clientWidth };
    }""")
    assert narrow["scrollWidth"] <= narrow["clientWidth"] + 1
    page.unroute(f"**/api/runs/{run_id}*")

    def patch_two_stage(route):
        resp = route.fetch()
        data = resp.json()
        proj = data.get("projection")
        if isinstance(proj, dict) and proj.get("run_id") == run_id:
            proj["workflow"] = {
                **proj["workflow"],
                "graph": {
                    "nodes": [
                        {"name": "intake", "kind": "intake_triage"},
                        {"name": "execute", "kind": "execute_task_dag"},
                    ],
                    "edges": [{"source": "intake", "target": "execute"}],
                },
                "node_statuses": {"intake": "completed", "execute": "running"},
                "traversed_edges": [{"source": "intake", "target": "execute"}],
            }
            route.fulfill(response=resp, json=data)
            return
        route.fulfill(response=resp)

    page.route(f"**/api/runs/{run_id}*", patch_two_stage)
    page.goto(f"{url}/?run={run_id}&lang=zh")
    _wait_for_workflow(page)
    two_col_widths = page.locator("#workflow-graph").evaluate("""(host) => {
      host.style.width = '600px';
      return Array.from(host.querySelectorAll('.wf-stage-col')).map(
        (el) => el.getBoundingClientRect().width
      );
    }""")
    assert len(two_col_widths) == 2
    assert abs(two_col_widths[0] - two_col_widths[1]) <= 2.0
    assert two_col_widths[0] > 240
    assert not errors, errors
    page.close()


# ---------------------------------------------------------------------------
# Graph interactions: scroll-safe controls, lineage highlight & SSE viewport
# ---------------------------------------------------------------------------

VIEWPORT_MATRIX = """(host) => {
  const raw = getComputedStyle(host.querySelector('.react-flow__viewport')).transform;
  const match = raw.match(/matrix\\(([^)]+)\\)/);
  if (!match) return { zoom: 1, x: 0, y: 0, raw };
  const [a, , , , e, f] = match[1].split(',').map(Number);
  return { zoom: a, x: e, y: f, raw };
}"""


def test_graph_islands_scroll_safe_controls_and_readonly_cards(workbench, browser):
    _, run_id, url = workbench
    page = browser.new_page(viewport={"width": 1200, "height": 720})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(f"{url}/?run={run_id}")
    _wait_for_workflow(page)
    _wait_for_layout(page)

    stage = page.locator("#stage")
    wf_host = page.locator("#workflow-graph")
    dag_host = page.locator("#dag-graph")

    wf_host.scroll_into_view_if_needed()
    wf_box = wf_host.bounding_box()
    scroll_before = stage.evaluate("el => el.scrollTop")
    page.mouse.move(wf_box["x"] + wf_box["width"] / 2, wf_box["y"] + wf_box["height"] / 2)
    page.mouse.wheel(0, 260)
    page.wait_for_function(
        "(prev) => document.getElementById('stage').scrollTop > prev",
        arg=scroll_before,
    )

    dag_host.scroll_into_view_if_needed()
    dag_before = dag_host.evaluate(VIEWPORT_MATRIX)
    dag_box = dag_host.bounding_box()
    scroll_mid = stage.evaluate("el => el.scrollTop")
    page.mouse.move(dag_box["x"] + dag_box["width"] / 2, dag_box["y"] + dag_box["height"] / 2)
    page.mouse.wheel(0, -180)
    page.wait_for_function(
        "(prev) => document.getElementById('stage').scrollTop < prev",
        arg=scroll_mid,
    )
    assert abs(dag_host.evaluate(VIEWPORT_MATRIX)["zoom"] - dag_before["zoom"]) < 0.001

    # Ctrl + Wheel zooms the DAG smoothly around the pointer.
    dag_host.locator(".ag-island").evaluate(
        """el => el.dispatchEvent(new WheelEvent('wheel', {
          deltaY: -120, ctrlKey: true, bubbles: true, cancelable: true,
          clientX: el.getBoundingClientRect().left + 120,
          clientY: el.getBoundingClientRect().top + 80,
        }))"""
    )
    assert dag_host.evaluate(VIEWPORT_MATRIX)["zoom"] > dag_before["zoom"] + 0.05

    # Corner toolbar (+, -, fit, 100%) drives the DAG cleanly.
    dag_host.locator("[data-graph-control='reset']").click()
    assert abs(dag_host.evaluate(VIEWPORT_MATRIX)["zoom"] - 1.0) < 0.01
    dag_host.locator("[data-graph-control='zoom-in']").click()
    zoomed_in = dag_host.evaluate(VIEWPORT_MATRIX)["zoom"]
    assert zoomed_in > 1.1
    dag_host.locator("[data-graph-control='zoom-out']").click()
    assert abs(dag_host.evaluate(VIEWPORT_MATRIX)["zoom"] - 1.0) < 0.02

    # Read-only semantic layout: dragging a node card does not displace its coordinates.
    dag_host.scroll_into_view_if_needed()
    node_wrap = dag_host.locator(".react-flow__node[data-id='T01']")
    before_transform = node_wrap.evaluate("el => el.style.transform")
    card = dag_host.locator("[data-task-id='T01']")
    before_card = card.bounding_box()
    page.mouse.move(
        before_card["x"] + before_card["width"] / 2,
        before_card["y"] + before_card["height"] / 2,
    )
    page.mouse.down()
    page.mouse.move(before_card["x"] + 160, before_card["y"] + 80, steps=6)
    page.mouse.up()
    assert node_wrap.evaluate("el => el.style.transform") == before_transform
    after_card = card.bounding_box()
    assert abs(after_card["x"] - before_card["x"]) <= 4.0
    assert abs(after_card["y"] - before_card["y"]) <= 4.0
    assert not errors, errors
    page.close()


def test_graph_islands_hover_lineage_highlighting(skip_dag, browser):
    _, run_id, url = skip_dag
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(f"{url}/?run={run_id}")
    expect = playwright.expect
    _wait_for_workflow(page)
    _wait_for_layout(page)

    dag = page.locator("#dag-graph")

    def card(tid):
        return dag.locator(f"[data-task-id='{tid}']")

    def edge(s, t):
        return dag.locator(f".ag-edge[data-edge-source='{s}'][data-edge-target='{t}']")

    # Select B first so we also prove hover lineage leaves selection ring untouched.
    card("B").click()
    expect(card("B")).to_have_attribute("aria-selected", "true")

    # Hover C: ancestors={A}, focus={C}, descendants={E, G}; siblings B, D, F, H are dimmed.
    card("C").hover()
    expect(card("C")).to_have_attribute("data-lineage", "focus")
    for related in ("A", "E", "G"):
        expect(card(related)).to_have_attribute("data-lineage", "related")
    for dimmed in ("B", "D", "F", "H"):
        expect(card(dimmed)).to_have_attribute("data-lineage", "dim")
    expect(card("B")).to_have_css("opacity", "0.26")
    expect(card("B")).to_have_attribute("aria-selected", "true")

    for src, dst in (("A", "C"), ("C", "E"), ("E", "G")):
        expect(edge(src, dst)).to_have_attribute("data-lineage", "related")
    for src, dst in (("A", "E"), ("B", "D"), ("D", "F"), ("F", "H"), ("B", "G")):
        expect(edge(src, dst)).to_have_attribute("data-lineage", "dim")

    # Moving pointer off the card clears hover dimming while keeping selection.
    page.mouse.move(10, 10)
    for tid in ("A", "B", "C", "D", "E", "F", "G", "H"):
        expect(card(tid)).not_to_have_attribute("data-lineage", "dim")
    expect(card("B")).to_have_attribute("aria-selected", "true")

    # Keyboard :focus-visible lineage survives a pointer enter+leave pass.
    page.keyboard.press("Tab")
    card("C").focus()
    assert card("C").evaluate('node => node.matches(":focus-visible")')
    expect(card("C")).to_have_attribute("data-lineage", "focus")
    card("C").hover()
    page.mouse.move(10, 10)
    expect(card("C")).to_have_attribute("data-lineage", "focus")
    assert card("C").evaluate("node => node === document.activeElement")
    assert not errors, errors
    page.close()


def test_task_dag_live_sse_status_update_preserves_zoom_and_pan_and_resets_on_run_switch(
    workbench, browser
):
    store, run_id, url = workbench
    second_id = store.create_run(title="第二个独立拓扑")
    spec2 = GoalSpec(goal="独立拓扑", requirements=[RequirementItem(
        id="R2", description="单任务", acceptance_criteria=["可切换"],
    )])
    store.save_spec(second_id, spec2)
    store.record_approval(ApprovalRecord(
        run_id=second_id, target=ApprovalTarget.GOAL_SPEC, target_id=spec2.spec_id,
        target_hash=spec2.content_hash, approver="human", decision=ApprovalDecision.APPROVED,
    ))
    plan2 = TaskPlan(spec_id=spec2.spec_id, spec_hash=spec2.content_hash, tasks=[
        TaskNode(
            task_id="SOLO-1", title="单独任务", objective="验证切线程重置视口",
            depends_on=[], write_scope=["solo.py"],
            acceptance_checks=[
                AcceptanceCheck(id="c1", kind=CheckKind.INDEPENDENT_REVIEW, description="ok")
            ],
        ),
    ])
    store.save_plan(second_id, plan2)
    store.record_approval(ApprovalRecord(
        run_id=second_id, target=ApprovalTarget.TASK_PLAN, target_id=plan2.plan_id,
        target_hash=plan2.content_hash, approver="human", decision=ApprovalDecision.APPROVED,
    ))

    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.goto(f"{url}/?run={run_id}&lang=zh")
    expect = playwright.expect
    _wait_for_layout(page)
    dag = page.locator("#dag-graph")
    dag.scroll_into_view_if_needed()

    # Manually zoom in AND drag-pan the Task DAG viewport.
    dag.locator("[data-graph-control='zoom-in']").click()
    box = dag.bounding_box()
    page.mouse.move(box["x"] + 20, box["y"] + box["height"] - 20)
    page.mouse.down()
    page.mouse.move(box["x"] + 90, box["y"] + box["height"] - 50, steps=6)
    page.mouse.up()
    pinned = dag.evaluate(VIEWPORT_MATRIX)
    assert pinned["zoom"] > 1.1
    card_handle = dag.locator("[data-task-id='T05']").element_handle()

    # Live SSE mutation: update T05 status RUNNING -> ACCEPTED on the same topology.
    plan = store.get_run(run_id)["plan"]
    updated_tasks = [
        t.model_copy(update={"status": TaskStatus.ACCEPTED}) if t.task_id == "T05" else t
        for t in plan.tasks
    ]
    store.save_plan(run_id, plan.model_copy(update={"tasks": updated_tasks}))
    expect(dag.locator("[data-task-id='T05']")).to_have_attribute("data-status", "accepted")
    expect(dag.locator("[data-task-id='T05']")).to_contain_text("已验收")
    assert dag.evaluate(VIEWPORT_MATRIX)["raw"] == pinned["raw"]
    assert card_handle.evaluate("node => node.isConnected")

    # Switching runs changes topology signature -> triggers fresh ELK layout and 1:1 auto-fit.
    page.locator("#nav-oracle-board").click()
    page.locator(f".oracle-card[data-run-id='{second_id}'] .oracle-card-trigger").click()
    expect(dag.locator("[data-task-id='SOLO-1']")).to_be_visible()
    assert abs(dag.evaluate(VIEWPORT_MATRIX)["zoom"] - 1.0) < 0.01
    page.close()


def test_task_dag_handles_arrow_ids_and_high_fanin_port_spacing(workbench, browser):
    store, _, url = workbench
    run_id = store.create_run(title="箭头字符与高扇入回归")
    goal = GoalSpec(
        goal="回归验证",
        requirements=[
            RequirementItem(id="R1", description="保留全部边", acceptance_criteria=["不丢边"])
        ],
    )
    store.save_spec(run_id, goal)
    inputs = {"A": [], "A→B": [], "B→C": ["A"], "C": ["A→B"]}
    review = AcceptanceCheck(id="review", kind=CheckKind.INDEPENDENT_REVIEW, description="ok")
    plan = TaskPlan(
        spec_id=goal.spec_id,
        spec_hash=goal.content_hash,
        tasks=[
            TaskNode(
                task_id=task_id,
                title=task_id,
                objective="test",
                depends_on=deps,
                write_scope=[f"task{index}.py"],
                acceptance_checks=[review],
            )
            for index, (task_id, deps) in enumerate(inputs.items())
        ],
    )
    store.save_plan(run_id, plan)
    store.record_approval(
        ApprovalRecord(
            run_id=run_id,
            target=ApprovalTarget.TASK_PLAN,
            target_id=plan.plan_id,
            target_hash=plan.content_hash,
            approver="human",
            decision=ApprovalDecision.APPROVED,
        )
    )

    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.goto(f"{url}/?run={run_id}")
    _wait_for_layout(page)
    rendered = page.locator("#dag-graph .ag-edge[data-edge-source]").evaluate_all(
        "els => els.map(el => [el.dataset.edgeSource, el.dataset.edgeTarget])"
    )
    assert sorted(rendered) == [["A", "B→C"], ["A→B", "C"]]

    # High fan-in (12 edges into T): card expands cleanly so fixed 7.2px arrowheads never overlap.
    page.evaluate(
        """() => {
          const host = document.createElement('div');
          host.id = 'fanin-check';
          host.style.width = '1100px';
          document.querySelector('#view-topology').prepend(host);
          const nodes = [...Array(12)].map((_, i) => ({
            id: 'S' + i, layer: 0, status: 'accepted', tone: 'ok',
            meta: 'engineer', statusText: '✓ 已验收', ariaLabel: 'S' + i,
          })).concat({
            id: 'T', layer: 1, status: 'waiting', tone: 'muted',
            meta: 'engineer', statusText: '◷ 等待', ariaLabel: 'T',
          });
          const edges = nodes.slice(0, -1).map(n => ({ source: n.id, target: 'T' }));
          window.faninHandle = window.AriadneGraph.mountDag(host, { runId: 'fanin', nodes, edges });
        }"""
    )
    page.locator("#fanin-check [data-layout='elk']").wait_for()
    fanin = page.evaluate(
        """() => {
          const card = document.querySelector('#fanin-check [data-task-id="T"]');
          const wrap = card.closest('.react-flow__node');
          const matrix = new DOMMatrix(getComputedStyle(wrap).transform);
          const ys = [...document.querySelectorAll(
            '#fanin-check .ag-edge[data-edge-target="T"] path.ag-edge-path'
          )].map(p => p.getPointAtLength(p.getTotalLength()).y - matrix.f).sort((a, b) => a - b);
          const marker = document.querySelector('#fanin-check marker[data-tone="default"]');
          const arrowH = marker.querySelector('path').getBBox().height
            * Number(marker.getAttribute('markerHeight')) / marker.viewBox.baseVal.height;
          const minGap = Math.min(...ys.slice(1).map((y, i) => y - ys[i]));
          return { count: ys.length, minGap, arrowH };
        }"""
    )
    assert fanin["count"] == 12
    assert fanin["minGap"] >= fanin["arrowH"], fanin
    page.evaluate("() => window.faninHandle.destroy()")
    page.close()


# ---------------------------------------------------------------------------
# Oracle Board (L1), Two-Tier Sidebar, Workspace Switcher & Open Ariadne Tabs
# ---------------------------------------------------------------------------


def test_oracle_board_landing_columns_mode_filters_and_deep_link(workbench, browser):
    store, demo_id, url = workbench
    human_id = store.create_run(
        title="Clarify Auth Rate Limit",
        workspace_mode="greenfield",
        task_type="feature",
    )
    store.save_spec(
        human_id,
        GoalSpec(
            goal="Rate limit API",
            requirements=[
                RequirementItem(
                    id="R1", description="Limit requests", acceptance_criteria=["Returns 429"]
                )
            ],
            open_ambiguities=[
                AmbiguityItem(
                    id="A1",
                    question="Use SQLite or in-memory counters?",
                    impact="Storage persistence",
                    blocking=True,
                )
            ],
        ),
    )
    store.set_run_status(human_id, RunStatus.AWAITING_HUMAN)

    agent_id = store.create_run(
        title="Refactor Router Adapters",
        workspace_mode="brownfield",
        task_type="refactor",
    )
    store.set_run_status(agent_id, RunStatus.RUNNING)

    done_id = store.create_run(
        title="API Spec Blueprint",
        workspace_mode="spec_only",
        task_type="blueprint",
    )
    store.set_run_status(done_id, RunStatus.COMPLETED)

    archived_id = store.create_run(
        title="Retired Legacy Migration",
        workspace_mode="brownfield",
        task_type="fix",
    )
    store.set_run_status(archived_id, RunStatus.CANCELLED)

    page = browser.new_page(viewport={"width": 1440, "height": 900})
    expect = playwright.expect
    page.goto(f"{url}/")

    # Lands on L1 Oracle Board with 3 Ball-in-Court columns and data-god attributes.
    expect(page.locator("body")).to_have_attribute("data-surface", "board")
    expect(page.get_by_role("heading", name="Oracle Board", exact=True)).to_be_visible()

    human_col = page.locator("[data-column='human']")
    agent_col = page.locator("[data-column='agent']")
    done_col = page.locator("[data-column='done']")

    expect(human_col.locator(f".oracle-card[data-run-id='{human_id}']")).to_have_attribute(
        "data-god", "prometheus"
    )
    expect(agent_col.locator(f".oracle-card[data-run-id='{agent_id}']")).to_be_visible()
    expect(done_col.locator(f".oracle-card[data-run-id='{done_id}']")).to_have_attribute(
        "data-god", "athena"
    )
    expect(page.locator(f".oracle-card[data-run-id='{archived_id}']")).to_have_count(0)

    # Sidebar pending-ruling badge reflects live human-held count.
    pending_count = httpx.get(f"{url}/api/runs").json()["pending_ruling_total"]
    assert pending_count >= 1
    expect(page.locator("#nav-pending-badge")).to_have_text(str(pending_count))

    # Mode filter pills filter across columns.
    page.locator("[data-mode-filter='greenfield']").click()
    expect(human_col.locator(f".oracle-card[data-run-id='{human_id}']")).to_be_visible()
    expect(agent_col.locator(f".oracle-card[data-run-id='{agent_id}']")).to_have_count(0)
    expect(done_col.locator(f".oracle-card[data-run-id='{done_id}']")).to_have_count(0)

    page.locator("[data-mode-filter='spec_only']").click()
    expect(done_col.locator(f".oracle-card[data-run-id='{done_id}']")).to_be_visible()
    expect(human_col.locator(f".oracle-card[data-run-id='{human_id}']")).to_have_count(0)

    page.locator("[data-mode-filter='all']").click()
    expect(human_col.locator(f".oracle-card[data-run-id='{human_id}']")).to_be_visible()

    # Archive toggle surfaces cancelled/completed history and returns cleanly to active board.
    page.locator("#nav-archive").click()
    expect(page.locator("#oracle-board-archive")).to_be_visible()
    expect(page.locator(f".oracle-card[data-run-id='{archived_id}']")).to_be_visible()
    page.locator("#nav-oracle-board").click()
    expect(page.locator("#oracle-board-columns")).to_be_visible()

    # Legacy deep link (?run=<id>#design) lands directly in L2 Ariadne Cockpit with tab open.
    page.goto(f"{url}/?run={demo_id}#design")
    expect(page.locator("body")).to_have_attribute("data-surface", "oracle")
    expect(page.locator("#view-design")).to_be_visible()
    expect(page.locator(f"#open-tabs .ariadne-tab[data-run-id='{demo_id}']")).to_have_attribute(
        "data-active", "true"
    )
    page.close()


def test_workspace_register_unregister_and_switching_preserves_cross_project_tabs(
    workbench, browser, tmp_path
):
    store, _, url = workbench
    second_repo = tmp_path / "voice-cli"
    second_repo.mkdir()
    marker_file = second_repo / "README.md"
    marker_file.write_text("# voice-cli\n", encoding="utf-8")

    default_ws = httpx.get(f"{url}/api/workspaces").json()["workspaces"][0]
    ws1_run = store.create_run(
        title="Default Repo Task",
        workspace_id=default_ws["workspace_id"],
        workspace_root=default_ws["root_path"],
        workspace_mode="brownfield",
        task_type="fix",
    )

    page = browser.new_page(viewport={"width": 1440, "height": 900})
    expect = playwright.expect
    page.goto(f"{url}/")

    # Open a Workspace 1 Oracle as an L2 Ariadne tab, then return to Oracle Board.
    page.locator(f".oracle-card[data-run-id='{ws1_run}'] .oracle-card-trigger").click()
    expect(page.locator("body")).to_have_attribute("data-surface", "oracle")
    expect(page.locator(f"#open-tabs .ariadne-tab[data-run-id='{ws1_run}']")).to_be_visible()
    page.locator("#nav-oracle-board").click()
    expect(page.locator("body")).to_have_attribute("data-surface", "board")

    # Open Workspace Manager via sidebar and test invalid path rejection first.
    page.locator("#nav-workspaces").click()
    dialog = page.locator("#workspace-manager-dialog")
    expect(dialog).to_be_visible()
    dialog.locator("#workspace-root-input").fill(str(tmp_path / "does-not-exist"))
    dialog.locator("#btn-register-workspace").click()
    expect(dialog.locator("#workspace-dialog-error")).to_be_visible()

    # Register the real second_repo directory.
    dialog.locator("#workspace-root-input").fill(str(second_repo))
    dialog.locator("#workspace-name-input").fill("voice-cli")
    dialog.locator("#btn-register-workspace").click()
    expect(dialog.locator("#workspace-list")).to_contain_text("voice-cli")
    dialog.get_by_role("button", name="Done").click()
    expect(dialog).to_be_hidden()

    # Seed a run bound to voice-cli and switch workspace filter to voice-cli.
    workspaces = httpx.get(f"{url}/api/workspaces").json()["workspaces"]
    ws2 = next(item for item in workspaces if item["name"] == "voice-cli")
    ws2_run = store.create_run(
        title="Offline Voice Slicer",
        workspace_id=ws2["workspace_id"],
        workspace_root=ws2["root_path"],
        workspace_mode="brownfield",
        task_type="feature",
    )

    page.locator("#workspace-switcher").select_option(ws2["workspace_id"])
    expect(page.locator(f".oracle-card[data-run-id='{ws2_run}']")).to_be_visible()
    expect(page.locator(f".oracle-card[data-run-id='{ws1_run}']")).to_have_count(0)

    # Invariant: switching Workspace NEVER closes already-open cross-project Ariadne tabs
    # and surfaces the tab's workspace badge for disambiguation.
    cross_tab = page.locator(f"#open-tabs .ariadne-tab[data-run-id='{ws1_run}']")
    expect(cross_tab).to_be_visible()
    expect(cross_tab.locator(".ariadne-tab-workspace")).to_have_text(default_ws["name"])

    # Unregister voice-cli via Workspace Manager: resets filter to All Projects, keeps disk files.
    page.locator("#nav-workspaces").click()
    expect(dialog).to_be_visible()
    dialog.locator(f"[data-workspace-remove='{ws2['workspace_id']}']").click()
    expect(dialog.locator(f"[data-workspace-id='{ws2['workspace_id']}']")).to_have_count(0)
    dialog.get_by_role("button", name="Done").click()

    assert second_repo.is_dir() and marker_file.read_text(encoding="utf-8") == "# voice-cli\n"
    expect(page.locator("#workspace-switcher")).to_have_value("")
    expect(page.locator(f".oracle-card[data-run-id='{ws1_run}']")).to_be_visible()
    page.close()


def test_open_and_close_ariadne_tabs_persist_without_deleting_runs(workbench, browser):
    store, demo_id, url = workbench
    extra_id = store.create_run(title="Keep On Board After Close")

    page = browser.new_page(viewport={"width": 1440, "height": 900})
    expect = playwright.expect
    page.goto(f"{url}/")

    # Seed a stale ID in localStorage alongside opening a real card.
    page.evaluate(
        "() => localStorage.setItem('olympus-open-tabs', JSON.stringify(['run-stale-gone']))"
    )
    page.reload()
    expect(page.locator("#open-tabs .ariadne-tab[data-run-id='run-stale-gone']")).to_have_count(0)

    # Open card from Oracle Board -> appears in Open Ariadne Tabs and persists across reload.
    page.locator(f".oracle-card[data-run-id='{extra_id}'] .oracle-card-trigger").click()
    expect(page.locator("body")).to_have_attribute("data-surface", "oracle")
    expect(page.locator(f"#open-tabs .ariadne-tab[data-run-id='{extra_id}']")).to_be_visible()
    stored = page.evaluate("() => JSON.parse(localStorage.getItem('olympus-open-tabs') || '[]')")
    assert extra_id in stored

    # Open a second tab, reload to verify persistence of both tabs.
    page.locator("#nav-oracle-board").click()
    page.locator(f".oracle-card[data-run-id='{demo_id}'] .oracle-card-trigger").click()
    expect(page.locator("#open-tabs .ariadne-tab")).to_have_count(2)
    page.reload()
    expect(page.locator("#open-tabs .ariadne-tab")).to_have_count(2)

    # Close both tabs with × -> returns to Oracle Board and never deletes backend runs.
    page.locator(f"[data-close-tab='{demo_id}']").click()
    expect(page.locator(f"#open-tabs .ariadne-tab[data-run-id='{demo_id}']")).to_have_count(0)
    expect(page.locator("body")).to_have_attribute("data-surface", "oracle")

    page.locator(f"[data-close-tab='{extra_id}']").click()
    expect(page.locator("#open-tabs .ariadne-tab")).to_have_count(0)
    expect(page.locator("body")).to_have_attribute("data-surface", "board")
    expect(page.locator(f".oracle-card[data-run-id='{extra_id}']")).to_be_visible()
    expect(page.locator(f".oracle-card[data-run-id='{demo_id}']")).to_be_visible()
    assert store.get_run(extra_id)["run_id"] == extra_id
    page.close()


def test_live_board_sse_updates_pending_ruling_badge_while_on_l2_tab(workbench, browser):
    store, demo_id, url = workbench
    watcher_id = store.create_run(title="Background In-Flight Oracle")
    store.set_run_status(watcher_id, RunStatus.RUNNING)

    page = browser.new_page(viewport={"width": 1440, "height": 900})
    expect = playwright.expect
    page.goto(f"{url}/?run={demo_id}")
    expect(page.locator("body")).to_have_attribute("data-surface", "oracle")

    initial_pending = httpx.get(f"{url}/api/runs").json()["pending_ruling_total"]
    if initial_pending > 0:
        expect(page.locator("#nav-pending-badge")).to_have_text(str(initial_pending))
    else:
        expect(page.locator("#nav-pending-badge")).to_have_count(0)

    # Transition the other Oracle to ball_holder="human" while staying on demo_id's L2 tab.
    store.save_spec(
        watcher_id,
        GoalSpec(
            goal="Need human ruling",
            requirements=[
                RequirementItem(id="R1", description="Clarify target", acceptance_criteria=["Done"])
            ],
            open_ambiguities=[
                AmbiguityItem(
                    id="AMB-1",
                    question="Which branch is the base?",
                    impact="Worktree target",
                    blocking=True,
                )
            ],
        ),
    )
    store.set_run_status(watcher_id, RunStatus.AWAITING_HUMAN)

    # Sidebar Oracle Board pending badge updates live via /api/board/stream without leaving L2.
    expect(page.locator("#nav-pending-badge")).to_have_text(str(initial_pending + 1))
    expect(page.locator("body")).to_have_attribute("data-surface", "oracle")

    # Switching back to Oracle Board immediately shows the Oracle in Awaiting Your Ruling.
    page.locator("#nav-oracle-board").click()
    card = page.locator(f"[data-column='human'] .oracle-card[data-run-id='{watcher_id}']")
    expect(card).to_be_visible()
    expect(card).to_have_attribute("data-god", "prometheus")
    page.close()
