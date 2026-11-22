"""Unit and HTTP/SSE integration tests for the M4 Ariadne Web Console."""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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
from olympus.core.fingerprint import content_hash
from olympus.core.store import ConcurrencyConflictError, RunStore
from olympus.graph.loader import build_graph, load_workflow
from olympus.providers.types import LLMResponse, MockLLMProvider
from olympus.stages.design import DesignStage
from olympus.web.demos import seed_demo_runs
from olympus.web.projection import build_run_projection
from olympus.web.server import AriadneApp, create_fastapi_app
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


def test_goal_approval_review_snapshot_links_commitments_to_candidate(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "test.db")
    run_id = _seed_gated_run(store)
    spec = store.get_run(run_id)["spec"].model_copy(update={"assumptions": ["最多 200 QPS"]})
    store.save_spec(run_id, spec)

    projection = AriadneApp(store=store, workspace_root=tmp_path).get_run_payload(run_id)[
        "projection"
    ]
    review = projection["approval_reviews"]["approve_spec"]
    assert review["target_hash"] == projection["spec"]["content_hash"]
    sections = {section["title"]: section["entries"] for section in review["sections"]}
    assert sections["目标"][0]["ref"] == "/goal"
    assert sections["范围"] == [
        {
            "ref": "/requirements/0/description",
            "label": "FR-100",
            "text": spec.requirements[0].description,
        }
    ]
    assert sections["排除项"][0]["text"] == "不发送短信或邮件通知"
    assert sections["验收条件"][0]["ref"] == "/requirements/0/acceptance_criteria/0"
    assert sections["验收条件"][0]["text"] == "递补在同一事务内完成且状态可观测"
    assert sections["假设"][0] == {
        "ref": "/assumptions/0",
        "text": "最多 200 QPS",
        "provenance": "unknown",
    }


def test_goal_rereview_uses_last_approval_and_surfaces_unchanged_criteria(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "test.db")
    run_id = _seed_gated_run(store)
    approved = store.get_run(run_id)["spec"].model_copy(update={
        "requirements": [RequirementItem(
            id="EXPORT", actor="管理员", description="仅管理员可以导出",
            acceptance_criteria=["导出包含时间戳", "拒绝未登录访问"],
        )],
        "open_ambiguities": [],
    })
    store.save_spec(run_id, approved)
    app = AriadneApp(store=store, workspace_root=tmp_path)
    current = app.get_run_payload(run_id)["projection"]
    app.execute_command(run_id, {
        "action": "approve_spec", "target_hash": approved.content_hash,
        "expected_state_revision": current["state_revision"], "auto_advance": False,
    })
    store.save_spec(run_id, approved.model_copy(update={"goal": "未批准的中间稿"}))
    revised = approved.model_copy(update={
        "version": 2, "requirements": [approved.requirements[0].model_copy(update={
            "actor": "所有成员", "description": "所有成员都可以导出",
        })],
    })
    store.save_spec(run_id, revised)
    projection = app.get_run_payload(run_id)["projection"]
    review = projection["approval_reviews"]["approve_spec"]
    assert review["target_hash"] == revised.content_hash
    assert review["baseline"]["content_hash"] == approved.content_hash
    entries = [entry for section in review["sections"] for entry in section["entries"]]
    assert any(
        entry["text"] == "所有成员都可以导出"
        and entry["previous"]["text"] == "仅管理员可以导出"
        for entry in entries
    )
    assert {entry["text"] for entry in entries if entry["change"] == "affected"} == {
        "导出包含时间戳", "拒绝未登录访问",
    }
    assert any(section["title"] == "排除项" for section in review["unchanged_sections"])
    assert set(projection["approval_reviews"]) == {"approve_spec", "approve_design", "approve_plan"}


@pytest.mark.parametrize("original_available", [False, True])
def test_rereview_migrates_legacy_database(tmp_path: Path, original_available: bool) -> None:
    db = tmp_path / "legacy.db"
    store = RunStore(db)
    run_id = _seed_gated_run(store)
    old = store.get_run(run_id)["spec"]
    approved = ApprovalRecord(
        run_id=run_id, target=ApprovalTarget.GOAL_SPEC, target_id=old.spec_id,
        target_hash=old.content_hash, approver="human", decision=ApprovalDecision.APPROVED,
    )
    store.record_approval(approved)
    candidate = old.model_copy(update={"version": 2, "goal": "当前完整目标"})
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE approval_artifacts")
        if not original_available:
            conn.execute(
                "UPDATE runs SET spec_json = ? WHERE run_id = ?",
                (candidate.model_dump_json(), run_id),
            )

    store = RunStore(db)
    if original_available:
        store.save_spec(run_id, candidate)
    review = build_run_projection(store, tmp_path, run_id)["approval_reviews"]["approve_spec"]
    if original_available:
        assert review["baseline"]["content_hash"] == old.content_hash
    else:
        assert "缺少上次批准版本的可靠原文" in review["comparison_gap"]
        assert "baseline" not in review and "unchanged_sections" not in review
        assert {
            entry["text"] for section in review["sections"] for entry in section["entries"]
        } >= {"当前完整目标", "已报名者取消后自动递补最早的有效候补者",
              "不发送短信或邮件通知", "递补在同一事务内完成且状态可观测",
              '["报名、取消与递补共用单一写事务"]'}
    assert store.is_approved(
        run_id=run_id, target=approved.target,
        target_id=approved.target_id, target_hash=approved.target_hash,
    )
    assert any(event.payload.get("approval_id") == approved.approval_id
               for event in store.list_events(run_id))


def test_rereview_does_not_skip_a_newer_approval_with_an_invalid_original(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "test.db")
    run_id = _seed_gated_run(store)
    first = store.get_run(run_id)["spec"]
    latest = first.model_copy(update={"version": 2, "goal": "已批准的新版目标"})
    rejected = latest.model_copy(update={"version": 3, "goal": "被否决的目标"})
    for artifact, decision in (
        (first, ApprovalDecision.APPROVED), (latest, ApprovalDecision.APPROVED),
        (rejected, ApprovalDecision.REJECTED),
    ):
        store.save_spec(run_id, artifact)
        store.record_approval(ApprovalRecord(
            run_id=run_id, target=ApprovalTarget.GOAL_SPEC, target_id=artifact.spec_id,
            target_hash=artifact.content_hash, approver="human", decision=decision,
        ))
    app = AriadneApp(store=store, workspace_root=tmp_path)
    review = app.get_run_payload(run_id)["projection"]["approval_reviews"]["approve_spec"]
    assert review["baseline"]["content_hash"] == latest.content_hash
    invalid = latest.model_dump(mode="json") | {"goal": "与已批准指纹不符的正文"}
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE approval_artifacts SET artifact_json = ? WHERE approval_id IN "
            "(SELECT approval_id FROM approvals WHERE target_hash = ?)",
            (json.dumps(invalid), latest.content_hash),
        )
    review = app.get_run_payload(run_id)["projection"]["approval_reviews"]["approve_spec"]
    assert "baseline" not in review
    assert "缺少上次批准版本的可靠原文" in review["comparison_gap"]
    assert review["sections"][0]["entries"][0]["text"] == rejected.goal


def test_rereview_all_gates_http_snapshot_sse_and_version_binding(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "test.db")
    run_id = _seed_gated_run(store)
    baseline = store.get_run(run_id)
    spec = baseline["spec"].model_copy(update={"open_ambiguities": []})
    design = DesignSpec.model_validate({
        **baseline["design"].model_dump(), "spec_hash": spec.content_hash,
        "design_tradeoffs": [
            {"impact": "rollback", "choice": "保留旧列", "cost": "延后清理"},
            {"impact": "deployment", "choice": "同步切流", "cost": "集中值班"},
        ],
    })
    plan = baseline["plan"].model_copy(update={
        "design_hash": design.content_hash, "spec_id": spec.spec_id,
        "spec_hash": spec.content_hash,
    })
    root = plan.tasks[0]
    downstream = root.model_copy(update={
        "task_id": "T02", "title": "更新查询", "depends_on": [root.task_id],
        "write_scope": ["src/waitlist/api.py"], "deliverables": ["更新查询 API"],
    })
    plan = plan.model_copy(update={"tasks": [root, downstream]})
    store.save_spec(run_id, spec)
    store.save_design(run_id, design)
    store.save_plan(run_id, plan)
    app = AriadneApp(store=store, workspace_root=tmp_path)
    path = f"/api/runs/{run_id}"
    with TestClient(create_fastapi_app(app)) as client:
        def approve(action, target_hash, projection):
            return client.post(f"{path}/command", json={
                "action": action, "target_hash": target_hash,
                "expected_state_revision": projection["state_revision"], "auto_advance": False,
            })

        initial = client.get(path).json()["projection"]
        projection = initial
        for action, artifact in (
            ("approve_spec", spec), ("approve_design", design), ("approve_plan", plan)
        ):
            response = approve(action, artifact.content_hash, projection)
            assert response.status_code == 200, response.text
            projection = response.json()["projection"]

        store.acquire_task_lease(run_id, root.task_id)
        updated = client.get(path).json()["projection"]
        assert updated["plan"]["content_hash"] == plan.content_hash
        assert updated["plan_approved"] and "approve_plan" not in updated["approval_reviews"]

        revised_spec = spec.model_copy(update={
            "version": 2, "requirements": [spec.requirements[0].model_copy(update={
                "description": "所有成员可以取消自己的报名",
            })],
        })
        revised_design = DesignSpec.model_validate({
            **design.model_dump(), "version": 2, "spec_hash": revised_spec.content_hash,
            "design_tradeoffs": [
                {"impact": "rollback", "choice": "保留旧列", "cost": "延后清理一个月"},
                {"impact": "existing_system", "choice": "沿用现有入口", "cost": "保留兼容层"},
            ],
        })
        revised_plan = plan.model_copy(update={
            "version": 2, "design_hash": revised_design.content_hash,
            "spec_hash": revised_spec.content_hash, "tasks": [
                root.model_copy(update={"deliverables": ["新的权限处理器"]}),
                downstream.model_copy(update={"depends_on": []}),
            ],
        })
        store.save_spec(run_id, revised_spec)
        store.save_design(run_id, revised_design)
        store.save_plan(run_id, revised_plan)
        rereview = client.get(path).json()["projection"]
        stream = client.get(f"{path}/stream?after_seq={updated['latest_event_seq']}&once=1")
        assert stream.status_code == 200
        state = json.loads(next(line[6:] for line in stream.text.splitlines()
                                if line.startswith("data: ")))
        assert state["state_revision"] == rereview["state_revision"]
        assert [event["payload"]["content_hash"] for event in state["events"]] == [
            revised_spec.content_hash, revised_design.content_hash, revised_plan.content_hash,
        ]
        for action, old, new in (
            ("approve_spec", spec, revised_spec), ("approve_design", design, revised_design),
            ("approve_plan", plan, revised_plan),
        ):
            review = rereview["approval_reviews"][action]
            assert review["baseline"]["content_hash"] == old.content_hash
            assert review["target_hash"] == new.content_hash
            conflict = approve(action, old.content_hash, rereview)
            assert conflict.status_code == 409 and conflict.json()["conflict"]
            assert conflict.json()["projection"]["approval_reviews"][action] == review

        reviews = rereview["approval_reviews"]
        design_entries = [entry for section in reviews["approve_design"]["sections"]
                          if section["title"] == "关键取舍" for entry in section["entries"]]
        assert {entry["change"]: entry["text"] for entry in design_entries} == {
            "changed": "保留旧列", "added": "沿用现有入口", "removed": "同步切流",
        }
        plan_review = reviews["approve_plan"]
        assert plan_review["baseline"]["tasks"][0]["status"] == "ready"
        plan_entries = [entry for section in plan_review["sections"]
                        for entry in section["entries"]]
        assert any(entry["change"] == "removed" and "T01" in entry["text"]
                   and "T02" in entry["text"] for entry in plan_entries)
        assert any(entry["change"] == "affected" and entry["text"] ==
                   spec.requirements[0].acceptance_criteria[0] for entry in plan_entries)
        for key, artifact in (
            ("spec", revised_spec), ("design", revised_design), ("plan", revised_plan),
        ):
            response = approve(f"approve_{key}", artifact.content_hash, rereview)
            assert response.status_code == 200, response.text
            rereview = response.json()["projection"]
            assert rereview[f"{key}_approved"]
            assert rereview[key]["content_hash"] == artifact.content_hash
        assert not rereview["approval_reviews"]
        assert sum(event.event_type == "approval.recorded"
                   for event in store.list_events(run_id)) == 6


def test_design_approval_review_snapshot_projects_the_candidate_tradeoffs(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "test.db")
    run_id = _seed_gated_run(store)
    original = store.get_run(run_id)["design"].model_dump()
    tradeoffs = [
        {"impact": "existing_system", "choice": "保留报名接口", "cost": "维护旧字段映射"},
        {"impact": "deployment", "choice": "双写后切流", "cost": "部署期额外占用写入容量"},
        {"impact": "rollback", "choice": "保留旧列一周", "cost": "延后清理旧数据"},
    ]
    design = DesignSpec.model_validate(
        {
            **original,
            "architecture_summary": "入口仍由旧服务接收。写入使用新表。详细迁移步骤见模块。",
            "design_tradeoffs": tradeoffs,
        }
    )
    store.save_design(run_id, design)

    snapshot = AriadneApp(store=store, workspace_root=tmp_path).get_run_payload(run_id)[
        "projection"
    ]
    review = snapshot["approval_reviews"]["approve_design"]
    assert review["artifact"] == "design"
    assert review["target_hash"] == snapshot["design"]["content_hash"]
    assert review["target_hash"] != original["content_hash"]
    assert review["sections"] == [
        {
            "title": "架构概述",
            "entries": [
                {"ref": "/architecture_summary", "text": "入口仍由旧服务接收。写入使用新表。"}
            ],
        },
        {
            "title": "关键取舍",
            "entries": [
                {
                    "ref": f"/design_tradeoffs/{index}",
                    "label": label,
                    "text": item["choice"],
                    "cost": item["cost"],
                }
                for index, (label, item) in enumerate(
                    zip(("现有系统", "部署", "回滚"), tradeoffs, strict=True)
                )
            ],
            "empty_text": "规格未记录结构化取舍，可查看完整 DesignSpec 核对。",
        },
    ]
    assert snapshot["design"]["design_tradeoffs"] == tradeoffs
    tradeoffs[1]["cost"] = "切流需要两次值班"
    revised = DesignSpec.model_validate({**design.model_dump(), "design_tradeoffs": tradeoffs})
    assert revised.content_hash != design.content_hash


def test_legacy_design_approval_keeps_its_original_fingerprint(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "test.db")
    run_id = _seed_gated_run(store)
    design = store.get_run(run_id)["design"]
    legacy_json = design.model_dump_json(exclude={"design_tradeoffs"})
    legacy_hash = "sha256:a31b6e6ee7b280949bc86bf6bb0fcf1c830ab1536a9caab8630f0d16f0fc8624"
    assert DesignSpec.model_validate_json(legacy_json).content_hash == legacy_hash

    app = AriadneApp(store=store, workspace_root=tmp_path)
    snapshot = app.get_run_payload(run_id)["projection"]
    review = snapshot["approval_reviews"]["approve_design"]
    assert review["target_hash"] == legacy_hash
    assert review["sections"][1] == {
        "title": "关键取舍",
        "empty_text": "规格未记录结构化取舍，可查看完整 DesignSpec 核对。",
        "entries": [],
    }
    approved = app.execute_command(
        run_id,
        {
            "action": "approve_design",
            "target_hash": legacy_hash,
            "expected_state_revision": snapshot["state_revision"],
            "auto_advance": False,
        },
    )["projection"]
    assert approved["design_approved"]
    assert approved["design"]["content_hash"] == legacy_hash


def test_approval_rejects_viewed_hash_after_two_candidate_updates(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "test.db")
    run_id = _seed_gated_run(store)
    app = AriadneApp(store=store, workspace_root=tmp_path)
    snapshot = app.get_run_payload(run_id)["projection"]
    viewed_hash = snapshot["approval_reviews"]["approve_spec"]["target_hash"]
    spec = store.get_run(run_id)["spec"]
    for version in (2, 3):
        store.save_spec(
            run_id, spec.model_copy(update={"version": version, "goal": f"候选 v{version}"})
        )
    latest = app.get_run_payload(run_id)["projection"]
    command = {
        "action": "approve_spec",
        "target_hash": viewed_hash,
        "expected_state_revision": latest["state_revision"],
        "accept_risk": True,
        "reason": "接受已审阅版本的风险",
        "auto_advance": False,
    }
    with pytest.raises(ConcurrencyConflictError, match="hash"):
        app.execute_command(run_id, command)
    assert not app.get_run_payload(run_id)["projection"]["spec_approved"]

    command["target_hash"] = latest["approval_reviews"]["approve_spec"]["target_hash"]
    result = app.execute_command(run_id, command)["projection"]
    assert result["spec_approved"]
    assert result["spec"]["content_hash"] == command["target_hash"]
    assert "approve_spec" not in result["approval_reviews"]


def test_review_marks_explicit_agent_assumptions_and_preserves_legacy_fingerprints(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "test.db")
    run_id = _seed_gated_run(store)
    payload = store.get_run(run_id)["spec"].model_dump(
        exclude={"content_hash", "status", "agent_high_impact_assumptions"}
    )
    payload["assumptions"] = ["首版只开放给管理员", "高峰 200 QPS"]
    legacy = GoalSpec.model_validate(payload)
    assert legacy.content_hash == content_hash(payload)
    spec = GoalSpec.model_validate(
        {**payload, "agent_high_impact_assumptions": ["首版只开放给管理员"]}
    )
    store.save_spec(run_id, spec)
    app = AriadneApp(store=store, workspace_root=tmp_path)
    projection = app.get_run_payload(run_id)["projection"]
    review = projection["approval_reviews"]["approve_spec"]
    assert review["target_hash"] != legacy.content_hash
    assumptions = next(section for section in review["sections"] if section["title"] == "假设")
    assert [entry["provenance"] for entry in assumptions["entries"]] == [
        "agent_high_impact",
        "unknown",
    ]


def test_approval_write_conflicts_if_candidate_changes_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path / "test.db")
    run_id = _seed_gated_run(store)
    app = AriadneApp(store=store, workspace_root=tmp_path)
    viewed = app.get_run_payload(run_id)["projection"]
    save_spec = store.save_spec
    candidate = store.get_run(run_id)["spec"].model_copy(update={"version": 2, "goal": "新版"})
    real_apply = store.apply_idempotent_command
    injected = False

    def apply_after_update(*args, **kwargs):
        """Model a concurrent write landing right before the approval transaction."""
        nonlocal injected
        if not injected:
            injected = True
            save_spec(run_id, candidate)
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(store, "apply_idempotent_command", apply_after_update)
    with pytest.raises(ConcurrencyConflictError, match="revision"):
        app.execute_command(
            run_id,
            {
                "action": "approve_spec",
                "target_hash": viewed["approval_reviews"]["approve_spec"]["target_hash"],
                "expected_state_revision": viewed["state_revision"],
                "accept_risk": True,
                "reason": "接受已审阅版本的风险",
                "auto_advance": False,
            },
        )
    fresh = app.get_run_payload(run_id)["projection"]
    assert fresh["spec"]["content_hash"] == candidate.content_hash
    assert fresh["spec"]["status"] == "draft"
    assert not fresh["spec_approved"]
    assert not any(event["event_type"] == "approval.recorded" for event in fresh["events"])


def test_delivery_demo_reseeds_in_git_workspace_with_retained_artifacts(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
            "commit", "--allow-empty", "--no-gpg-sign", "-qm", "Fixture",
        ],
        cwd=workspace,
        check=True,
    )

    evidence_logs = workspace / ".olympus" / "runs" / "run-delivery-acceptance" / "artifacts"
    for generation in (1, 2):
        store = RunStore(tmp_path / f"generation-{generation}.db")
        seed_demo_runs(store, workspace)
        review = build_run_projection(store, workspace, "run-delivery-acceptance")[
            "delivery_acceptance"
        ]
        assert review["counts"] == {"passed": 1, "failed": 1, "unverified": 1}
        assert "Ready" in review["criteria"][0]["evidence"]["log"]
        assert len(list(evidence_logs.glob("delivery-*/check-1.log"))) == generation


def test_seed_demo_and_projection_invariants(tmp_path: Path) -> None:
    db_path = tmp_path / ".olympus" / "olympus.db"
    store = RunStore(db_path)

    seeded = seed_demo_runs(store, tmp_path)
    assert seeded == ["run-delivery-acceptance", "run-campus-waitlist"]
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
    assert matrix_by_req["FR-003"]["verification_status"] == "pending"
    assert matrix_by_req["FR-005"]["verification_status"] == "pending"
    assert proj["delivery_acceptance"]["counts"] == {"passed": 0, "failed": 0, "unverified": 5}

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


def test_projection_reads_tool_calls_from_legacy_worker_trace(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    seed_demo_runs(store, tmp_path)
    run_id = "run-campus-waitlist"
    node = next(
        node for node in build_run_projection(store, tmp_path, run_id)["dag"]["nodes"]
        if node["task_id"] == "T03-R1"
    )
    trace = Path(node["artifacts"]["attempt_dir"]) / "worker.jsonl"
    trace.write_text(
        json.dumps({
            "step": 3,
            "tool_calls": [
                {"id": "read", "name": "read_file", "arguments": {"path": "source.py"}},
            ],
            "tool_outputs": [{"id": "read", "name": "read_file", "output": "VALUE = 1"}],
        }) + "\n",
        encoding="utf-8",
    )
    node = next(
        node for node in build_run_projection(store, tmp_path, run_id)["dag"]["nodes"]
        if node["task_id"] == "T03-R1"
    )
    assert node["artifacts"]["tool_calls"][0]["target"] == "source.py"
    assert node["artifacts"]["tool_calls"][0]["output_preview"] == "VALUE = 1"


def test_ariadne_http_sse_and_idempotent_hitl_commands(tmp_path: Path) -> None:
    db_path = tmp_path / ".olympus" / "olympus.db"
    store = RunStore(db_path)
    seed_demo_runs(store, tmp_path)
    app = AriadneApp(store=store, workspace_root=tmp_path)


    with TestClient(create_fastapi_app(app)) as client:
        html_res = client.get("/")
        assert html_res.status_code == 200
        assert '<div id="root"></div>' in html_res.text
        assert 'src="/static/assets/ariadne.js"' in html_res.text
        assert 'href="/static/assets/ariadne.css"' in html_res.text
        assert 'src="/static/assets/ariadne-graph.js"' not in html_res.text

        css_res = client.get("/static/assets/ariadne.css")
        assert css_res.status_code == 200
        assert css_res.headers["content-type"].startswith("text/css")
        assert css_res.headers["cache-control"] == "no-store"
        assert "--surface-bg" in css_res.text
        assert "scrollbar-color" in css_res.text

        app_res = client.get("/static/assets/ariadne.js")
        assert app_res.status_code == 200
        assert "javascript" in app_res.headers["content-type"]
        assert app_res.headers["cache-control"] == "no-store"
        # Regression #79: the lazy graph import must pass a non-empty __vite__mapDeps list
        # containing assets/ariadne-graph.css, otherwise React Flow renders unstyled on first load.
        assert re.search(
            r'\[[^\]]*[`"\']assets/ariadne-graph\.css[`"\'][^\]]*\]',
            app_res.text,
        )
        assert re.search(
            r'import\([`"\']\./ariadne-graph\.js[`"\']\)\s*,\s*__vite__mapDeps\(\[\d+(?:,\d+)*\]\)',
            app_res.text,
        )

        graph_js = client.get("/static/assets/ariadne-graph.js")
        assert graph_js.status_code == 200
        assert "javascript" in graph_js.headers["content-type"]
        graph_css = client.get("/static/assets/ariadne-graph.css")
        assert graph_css.status_code == 200
        assert graph_css.headers["content-type"].startswith("text/css")
        assert ".ag-task-card" in graph_css.text
        assert ".ag-wf-card" in graph_css.text

        assert client.get("/static/app.js").status_code == 404
        assert client.get("/static/ui.js").status_code == 404
        assert client.get("/static/missing.css").status_code == 404
        assert client.get("/static/..%2fserver.py").status_code == 404

        health = client.get("/healthz").json()
        assert health["ok"] is True

        runs_res = client.get("/api/runs").json()
        assert runs_res["default_run_id"] == "run-campus-waitlist"
        assert client.get("/api/runs?limit=abc").status_code == 400
        assert (
            client.post(
                "/api/seed-demo",
                content="{",
                headers={"Content-Type": "application/json"},
            ).status_code
            == 400
        )
        demo = client.get("/api/runs/run-delivery-acceptance").json()["projection"]
        acceptance = demo["delivery_acceptance"]
        assert acceptance["counts"] == {"passed": 1, "failed": 1, "unverified": 1}
        passed, failed, unverified = acceptance["criteria"]
        assert "Ready" in passed["evidence"]["log"]
        assert failed["reason"] and failed["evidence"]
        assert unverified["risk_accepted"] and "人工接受风险" in unverified["reason"]
        assert unverified["evidence"] is None
        seq = passed["evidence"]["event_seq"]
        full_log = client.get(f"/api/runs/run-delivery-acceptance/delivery-log/{seq}/1")
        assert full_log.status_code == 200
        assert full_log.headers["content-type"].startswith("text/plain")
        assert "--- stdout ---\nReady\n" in full_log.text
        assert client.get(
            f"/api/runs/run-delivery-acceptance/delivery-log/{seq}/3"
        ).status_code == 404

        run_res = client.get("/api/runs/run-campus-waitlist").json()
        assert run_res["ok"] is True
        rev0 = int(run_res["projection"]["state_revision"])

        proj0 = run_res["projection"]
        t03r1 = next(n for n in proj0["dag"]["nodes"] if n["task_id"] == "T03-R1")
        assert t03r1["subagent_stream"][0]["tool"] == "read_file"
        assert t03r1["receipt"]["turn_id"] == "turn-t03-r1-a2"
        assert "MANIFEST — T03-R1" in t03r1["manifest_md"]
        assert len(t03r1["terminal_lines"]) <= 4
        assert proj0["waterfall_turns"][0]["actor"] == "Align"
        assert any(turn["actor"] == "Argus" for turn in proj0["waterfall_turns"])

        sse_res = client.get("/api/runs/run-campus-waitlist/stream?after_seq=0&once=1")
        assert sse_res.status_code == 200
        assert "text/event-stream" in sse_res.headers["content-type"]
        assert "event: state" in sse_res.text

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

        p1_dup = client.post("/api/runs/run-campus-waitlist/command", json=pause_body)
        assert p1_dup.status_code == 200
        assert int(p1_dup.json()["projection"]["state_revision"]) == rev1

        stale_res = client.post(
            "/api/runs/run-campus-waitlist/command",
            json={
                "operation_id": "op-stale-99",
                "action": "resume",
                "expected_state_revision": rev0,
            },
        )
        assert stale_res.status_code == 409
        conflict_edges = stale_res.json()["projection"]["dag"]["edges"]
        assert conflict_edges and all("from" in edge and "to" in edge for edge in conflict_edges)

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

        retry_completed = client.post(
            "/api/runs/run-delivery-acceptance/command",
            json={
                "operation_id": "op-retry-completed",
                "action": "retry_workflow",
                "sync": True,
            },
        )
        assert retry_completed.status_code == 200
        assert retry_completed.json()["command_result"]["workflow"]["complete"] is True
        retry_replay = client.post(
            "/api/runs/run-delivery-acceptance/command",
            json={
                "operation_id": "op-retry-completed",
                "action": "retry_workflow",
                "sync": True,
            },
        )
        assert retry_replay.status_code == 200
        assert retry_replay.json()["command_result"]["workflow"]["complete"] is True
        proj_after_risk = risk_res.json()["projection"]
        t03_node = next(n for n in proj_after_risk["dag"]["nodes"] if n["task_id"] == "T03-R1")
        assert t03_node["status"] == TaskStatus.ACCEPTED.value
        assert t03_node["accepted_risk_reason"] == "首期单管理员手动取消，接受并发风险放行"
        assert proj_after_risk["progress"]["accepted_tasks"] == 5
        assert proj_after_risk["progress"]["verified_tasks"] == 4

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
        risk_review = proj_after_risk["delivery_acceptance"]
        assert risk_review["counts"] == {"passed": 0, "failed": 0, "unverified": 5}
        risk_row = next(
            row for row in risk_review["criteria"] if row["requirement_id"] == "FR-003"
        )
        assert risk_row["risk_accepted"] and risk_row["evidence"] is None
        assert t03_node["accepted_risk_reason"] in risk_row["reason"]


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


def test_projection_derives_workflow_trajectory_and_skipped_nodes(tmp_path: Path) -> None:
    store = RunStore(tmp_path / ".olympus" / "olympus.db")
    run_id = store.create_run(title="轨迹投影测试", workflow_name="full_sdlc_tdd")
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
            run_id=run_id,
            stage="graph",
            event_type=event_type,
            actor="supervisor",
            payload=payload,
        )
    store.set_run_status(run_id, RunStatus.AWAITING_HUMAN, actor="supervisor", stage="graph")

    proj = build_run_projection(store, tmp_path, run_id)
    workflow = proj["workflow"]
    assert {"source": "align", "target": "gate_spec"} in workflow["traversed_edges"]
    assert {"source": "intake", "target": "align"} in workflow["traversed_edges"]
    assert workflow["node_statuses"]["critic"] == "skipped"
    assert next(n for n in workflow["nodes"] if n["name"] == "critic")["status"] == "skipped"
    assert workflow["node_statuses"]["gate_spec"] == "paused"
    for downstream in (
        "design",
        "validate_design",
        "gate_design",
        "plan",
        "validate_plan",
        "gate_plan",
        "execute",
    ):
        assert workflow["node_statuses"][downstream] == "pending"

    # Once the run executes to completion the bypassed node stays skipped.
    for event_type, payload in (
        ("graph.node_finished", {"node": "gate_spec"}),
        ("graph.node_started", {"node": "design"}),
        ("graph.node_finished", {"node": "design"}),
        ("graph.routed", {"from": ["design"], "to": ["execute"], "wave": 3}),
        ("graph.node_started", {"node": "execute"}),
        ("graph.node_finished", {"node": "execute"}),
        ("graph.finished", {"graph": "full_sdlc_tdd", "steps": 3}),
    ):
        store.append_event(
            run_id=run_id,
            stage="graph",
            event_type=event_type,
            actor="supervisor",
            payload=payload,
        )
    store.set_run_status(run_id, RunStatus.COMPLETED, actor="supervisor", stage="graph")

    finished = build_run_projection(store, tmp_path, run_id)["workflow"]
    assert finished["node_statuses"]["critic"] == "skipped"
    assert finished["node_statuses"]["execute"] == "completed"


def test_projection_skips_alignment_chain_on_design_ready_triage(tmp_path: Path) -> None:
    store = RunStore(tmp_path / ".olympus" / "olympus.db")
    run_id = store.create_run(title="分流投影测试", workflow_name="full_sdlc_tdd")
    for event_type, payload in (
        ("graph.node_started", {"node": "intake"}),
        ("graph.node_finished", {"node": "intake"}),
        ("graph.routed", {"from": ["intake"], "to": ["validate_design"], "wave": 1}),
        ("graph.node_started", {"node": "validate_design"}),
        ("graph.node_finished", {"node": "validate_design"}),
        ("graph.routed", {"from": ["validate_design"], "to": ["gate_design"], "wave": 2}),
        ("graph.node_started", {"node": "gate_design"}),
        ("graph.paused", {"node": "gate_design", "reason": "human_gate"}),
    ):
        store.append_event(
            run_id=run_id,
            stage="graph",
            event_type=event_type,
            actor="supervisor",
            payload=payload,
        )
    store.set_run_status(run_id, RunStatus.AWAITING_HUMAN, actor="supervisor", stage="graph")

    workflow = build_run_projection(store, tmp_path, run_id)["workflow"]
    assert {"source": "intake", "target": "validate_design"} in workflow["traversed_edges"]
    for bypassed in ("align", "critic", "gate_spec", "design"):
        assert workflow["node_statuses"][bypassed] == "skipped"
    assert workflow["node_statuses"]["gate_design"] == "paused"
    for unreached in ("plan", "validate_plan", "gate_plan", "execute"):
        assert workflow["node_statuses"][unreached] == "pending"


def test_projection_derives_legacy_trajectory_without_routed_events(tmp_path: Path) -> None:
    store = RunStore(tmp_path / ".olympus" / "olympus.db")
    run_id = store.create_run(title="遗留运行", workflow_name="repo_fast_patch")
    for event_type, payload in (
        ("graph.node_started", {"node": "intake"}),
        ("graph.node_finished", {"node": "intake"}),
        ("graph.node_started", {"node": "align"}),
        ("graph.node_finished", {"node": "align"}),
        ("graph.node_started", {"node": "gate_spec"}),
    ):
        store.append_event(
            run_id=run_id,
            stage="graph",
            event_type=event_type,
            actor="supervisor",
            payload=payload,
        )
    store.set_run_status(run_id, RunStatus.AWAITING_HUMAN, actor="supervisor", stage="graph")

    workflow = build_run_projection(store, tmp_path, run_id)["workflow"]
    assert workflow["traversed_edges"] == [
        {"source": "intake", "target": "align"},
        {"source": "align", "target": "gate_spec"},
    ]
    assert "skipped" not in workflow["node_statuses"].values()
    assert workflow["node_statuses"]["plan"] == "pending"

    # Seeded demo runs predate graph.* events entirely: derivation stays empty, never raises.
    for seeded_id in seed_demo_runs(store, tmp_path):
        seeded = build_run_projection(store, tmp_path, seeded_id)["workflow"]
        assert seeded["traversed_edges"] == []
        assert "skipped" not in seeded["node_statuses"].values()


def test_ariadne_v3_workbench_http_flow(tmp_path: Path) -> None:
    db_path = tmp_path / ".olympus" / "olympus.db"
    store = RunStore(db_path)
    run_id = _seed_gated_run(store)
    app = AriadneApp(store=store, workspace_root=tmp_path)

    try:
        with TestClient(create_fastapi_app(app)) as client:
            html = client.get("/").text
            assert 'src="/static/assets/ariadne.js"' in html
            bundle = client.get("/static/assets/ariadne.js").text
            for marker in (
                "btn-new-run",
                "thread-list",
                "new-run-dialog",
                "new-run-workflow",
                "new-run-provider",
                "new-run-model",
                "new-run-title",
                "new-run-input",
                "new-run-file",
                "triage-banner",
                "clarify-panel",
                "workflow-graph",
                "view-design",
                "btn-approve-spec",
                "btn-approve-design",
                "btn-approve-plan",
            ):
                assert marker in bundle, f"ariadne.js missing {marker}"

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
            design_review = spec_proj["approval_reviews"]["approve_design"]
            assert design_review["target_hash"] == spec_proj["design"]["content_hash"]

            approve_design = client.post(
                f"/api/runs/{run_id}/command",
                json={
                    "operation_id": "op-approve-design",
                    "action": "approve_design",
                    "target_hash": design_review["target_hash"],
                    "expected_state_revision": spec_proj["state_revision"],
                    "auto_advance": False,
                },
            )
            assert approve_design.status_code == 200
            design_proj = approve_design.json()["projection"]
            assert design_proj["design_approved"] is True
            assert design_proj["hitl_banner"]["kind"] == "plan_gate"
            plan_review = design_proj["approval_reviews"]["approve_plan"]
            assert plan_review["target_hash"] == design_proj["plan"]["content_hash"]

            approve_plan = client.post(
                f"/api/runs/{run_id}/command",
                json={
                    "operation_id": "op-approve-plan",
                    "action": "approve_plan",
                    "target_hash": plan_review["target_hash"],
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
    try:
        with TestClient(create_fastapi_app(app)) as client:
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


def test_run_snapshot_exposes_recorded_output_versions_with_task_graph(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    run_id = _seed_gated_run(store)
    original = store.get_run(run_id)["plan"]
    revised = original.model_copy(update={"version": 2})
    store.save_plan(run_id, revised)
    change = store.list_events(run_id)[-1]
    store.append_event(run_id=run_id, stage="system", event_type="budget.settled", actor="test")

    app = AriadneApp(store=store, workspace_root=tmp_path)
    snapshot = app.get_run_payload(run_id)["projection"]
    assert snapshot["dag"]["nodes"]
    assert snapshot["recent_progress"]["event_seq"] == change.event_seq
    assert snapshot["recent_progress"]["created_at"] == change.created_at.isoformat()
    versions = [output for output in snapshot["key_outputs"] if output["kind"] == "plan"]
    assert [(item["version"], item["content_hash"], item["current"]) for item in versions] == [
        (2, revised.content_hash, True),
        (1, original.content_hash, False),
    ]
    assert versions[0]["summary"] == "任务计划 v2 已就绪，包含 1 个任务"
