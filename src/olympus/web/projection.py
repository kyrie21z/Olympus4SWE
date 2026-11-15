"""Deterministic event-sourced projection and demo seeding for the Ariadne Web Console."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from olympus.core.contracts import (
    AcceptanceCheck,
    AmbiguityItem,
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    BudgetLedger,
    BudgetPolicy,
    CheckKind,
    CritiqueNote,
    CritiquePerspective,
    CritiqueSeverity,
    GoalSpec,
    RequirementItem,
    ReviewFinding,
    ReviewSeverity,
    ReviewVerdict,
    RunEvent,
    RunMode,
    RunStatus,
    SpecStatus,
    TaskNode,
    TaskPlan,
    TaskStatus,
)
from olympus.core.store import RunStore


def _compute_dag_layers_and_critical_path(
    tasks: list[TaskNode],
) -> tuple[dict[str, int], set[str], set[tuple[str, str]]]:
    """Assign topological layers and highlight critical/blocking path edges."""
    by_id = {t.task_id: t for t in tasks}
    layers: dict[str, int] = {}

    def get_layer(tid: str, visiting: set[str]) -> int:
        if tid in layers:
            return layers[tid]
        if tid in visiting or tid not in by_id:
            return 0
        visiting.add(tid)
        node = by_id[tid]
        dep_layers = [get_layer(dep, visiting) for dep in node.depends_on if dep in by_id]
        visiting.remove(tid)
        layer = (max(dep_layers) + 1) if dep_layers else 0
        layers[tid] = layer
        return layer

    for t in tasks:
        get_layer(t.task_id, set())

    # Compute longest path weights favoring blocked/running/unaccepted chains
    dist: dict[str, int] = {}
    parent: dict[str, str | None] = {}
    ordered = sorted(tasks, key=lambda item: (layers.get(item.task_id, 0), item.task_id))
    for t in ordered:
        weight = 2 if t.status in (TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.RUNNING) else 1
        best_prev: str | None = None
        best_score = -1
        for dep in t.depends_on:
            if dep in dist and dist[dep] > best_score:
                best_score = dist[dep]
                best_prev = dep
        dist[t.task_id] = (best_score if best_score >= 0 else 0) + weight
        parent[t.task_id] = best_prev

    crit_nodes: set[str] = set()
    crit_edges: set[tuple[str, str]] = set()
    if ordered:
        end_node = max(ordered, key=lambda item: (dist.get(item.task_id, 0), layers[item.task_id]))
        cur: str | None = end_node.task_id
        while cur is not None:
            crit_nodes.add(cur)
            prv = parent.get(cur)
            if prv is not None:
                crit_edges.add((prv, cur))
            cur = prv

    return layers, crit_nodes, crit_edges


def _read_task_artifacts(
    run_dir: Path,
    task: TaskNode,
) -> dict[str, Any]:
    """Load recorded diff, command check logs, review verdicts, and worker trace summary."""
    task_art_root = run_dir / "artifacts" / task.task_id
    attempt_idx = max(1, task.attempt)
    art_dir = task_art_root / f"attempt-{attempt_idx}"
    if not art_dir.exists() and task_art_root.exists():
        candidates = sorted(task_art_root.glob("attempt-*"))
        if candidates:
            art_dir = candidates[-1]

    diff_text = ""
    diff_path = art_dir / "diff.patch"
    if diff_path.exists():
        diff_text = diff_path.read_text(encoding="utf-8", errors="replace")[:8000]

    check_logs: dict[str, str] = {}
    if art_dir.exists():
        for log_file in sorted(art_dir.glob("check-*.log")):
            check_logs[log_file.stem.removeprefix("check-")] = log_file.read_text(
                encoding="utf-8", errors="replace"
            )[:4000]

    verdicts: list[dict[str, Any]] = []
    if art_dir.exists():
        for v_file in sorted(art_dir.glob("verdict-cycle-*.json")):
            try:
                verdicts.append(json.loads(v_file.read_text(encoding="utf-8")))
            except Exception:
                continue

    tool_calls: list[dict[str, Any]] = []
    trace_file = art_dir / "worker.jsonl"
    if trace_file.exists():
        for raw_line in trace_file.read_text(encoding="utf-8", errors="replace").splitlines()[:40]:
            try:
                entry = json.loads(raw_line)
            except Exception:
                continue
            if entry.get("event") == "tool_call":
                tool_calls.append(
                    {
                        "step": entry.get("step"),
                        "tool": entry.get("tool"),
                        "is_error": bool(entry.get("is_error", False)),
                        "output_preview": str(entry.get("output", ""))[:240],
                    }
                )

    return {
        "attempt_dir": str(art_dir) if art_dir.exists() else None,
        "diff_patch": diff_text,
        "check_logs": check_logs,
        "verdicts": verdicts,
        "tool_calls": tool_calls,
    }


def _build_stage_pipeline(
    snap: dict[str, Any],
    spec_approved: bool,
    plan_approved: bool,
) -> list[dict[str, str]]:
    spec: GoalSpec | None = snap["spec"]
    plan: TaskPlan | None = snap["plan"]
    status: RunStatus = snap["status"]
    tasks = plan.tasks if plan else []

    align_state = "done" if spec_approved else ("active" if spec is not None else "pending")
    design_state = "done" if spec_approved else "pending"
    plan_state = (
        "done" if plan_approved else ("active" if plan is not None and spec_approved else "pending")
    )

    all_accepted = bool(tasks) and all(
        t.status in (TaskStatus.ACCEPTED, TaskStatus.CANCELLED) for t in tasks
    )
    any_blocked = any(t.status in (TaskStatus.BLOCKED, TaskStatus.FAILED) for t in tasks)

    if all_accepted and status == RunStatus.COMPLETED:
        exec_state = "done"
        verify_state = "done"
    elif plan_approved:
        exec_state = "blocked" if any_blocked else "active"
        verify_state = (
            "active" if any(t.status == TaskStatus.ACCEPTED for t in tasks) else "pending"
        )
    else:
        exec_state = "pending"
        verify_state = "pending"

    return [
        {"id": "align", "label": "要求", "status": align_state},
        {"id": "design", "label": "设计", "status": design_state},
        {"id": "plan", "label": "分解", "status": plan_state},
        {"id": "execute", "label": "执行", "status": exec_state},
        {"id": "verify", "label": "验收", "status": verify_state},
    ]


def _build_swimlanes(
    tasks: list[TaskNode],
    events: list[RunEvent],
) -> list[dict[str, Any]]:
    """Build chronological agent execution lanes from tasks and append-only events."""
    lanes_by_actor: dict[str, list[dict[str, Any]]] = {}

    # Check if events carry explicit lane assignments (e.g., Hephaestus-1, Hephaestus-2, Momus)
    explicit_items = [e for e in events if e.event_type == "agent.swimlane" and "lane" in e.payload]
    if explicit_items:
        for ev in explicit_items:
            lane_name = str(ev.payload["lane"])
            lanes_by_actor.setdefault(lane_name, []).append(
                {
                    "id": str(ev.payload.get("item_id", ev.event_id)),
                    "task_id": ev.payload.get("task_id"),
                    "label": str(ev.payload.get("label", ev.payload.get("task_id", ""))),
                    "status": str(ev.payload.get("status", "accepted")),
                    "attempts": int(ev.payload.get("attempts", 1)),
                    "summary": str(ev.payload.get("summary", "")),
                    "actor": lane_name,
                }
            )
    else:
        for idx, t in enumerate(tasks):
            if t.status == TaskStatus.WAITING and t.attempt == 0:
                continue
            worker_lane = f"Hephaestus-{(idx % 2) + 1}"
            suffix = f" ×{t.attempt}" if t.attempt > 1 else ""
            status_text = "运行中" if t.status == TaskStatus.RUNNING else ""
            label = f"{t.task_id}{suffix}" + (f" {status_text}" if status_text else "")
            lanes_by_actor.setdefault(worker_lane, []).append(
                {
                    "id": f"lane-{t.task_id}",
                    "task_id": t.task_id,
                    "label": label.strip(),
                    "status": t.status.value,
                    "attempts": max(1, t.attempt),
                    "summary": t.objective,
                    "actor": worker_lane,
                }
            )
        # Add Argus / Momus review lane if any task reached verification/review/rework
        reviewed = [t for t in tasks if t.attempt > 0 or t.review_cycle > 0]
        if reviewed:
            lanes_by_actor["Argus / Momus"] = [
                {
                    "id": f"review-{t.task_id}",
                    "task_id": t.task_id,
                    "label": f"审查 {t.task_id}",
                    "status": (
                        "accepted"
                        if t.status == TaskStatus.ACCEPTED
                        else ("blocked" if t.status == TaskStatus.BLOCKED else "running")
                    ),
                    "attempts": max(1, t.review_cycle),
                    "summary": t.last_error or "冷启动独立审查通过",
                    "actor": "Argus / Momus",
                }
                for t in reviewed[:4]
            ]

    return [{"actor": actor, "items": items} for actor, items in lanes_by_actor.items()]


def _build_traceability_matrix(
    spec: GoalSpec | None,
    tasks: list[TaskNode],
) -> list[dict[str, Any]]:
    """Build Requirement x Design/Source x Task x Verification Status rows."""
    if spec is None:
        return []

    rows: list[dict[str, Any]] = []
    for req in spec.requirements:
        covering = [t for t in tasks if req.id in t.requirement_refs]
        if not covering:
            v_status = "pending"
            v_label = "待验证"
            evidence_note = "尚未分配执行任务"
        elif any(t.status in (TaskStatus.BLOCKED, TaskStatus.FAILED) for t in covering):
            v_status = "failed"
            v_label = "失败"
            failed_t = next(
                t for t in covering if t.status in (TaskStatus.BLOCKED, TaskStatus.FAILED)
            )
            evidence_note = failed_t.last_error or f"{failed_t.task_id} 验证未通过"
        elif all(t.status == TaskStatus.ACCEPTED for t in covering):
            risk_tasks = [t for t in covering if t.accepted_risk_reason]
            if risk_tasks:
                v_status = "unverified_risk_accepted"
                v_label = "未验证 · 人工接受"
                evidence_note = f"人工接受风险：{risk_tasks[0].accepted_risk_reason}"
            else:
                v_status = "verified"
                v_label = "已验证"
                evidence_note = "命令门禁 + Argus 独立复核通过"
        elif any(t.status in (TaskStatus.RUNNING, TaskStatus.VERIFYING) for t in covering):
            v_status = "running"
            v_label = "验证中"
            evidence_note = "正在隔离沙盒中执行或待复核"
        else:
            v_status = "pending"
            v_label = "待验证"
            evidence_note = "前置任务待完成：" + ", ".join(t.task_id for t in covering)

        rows.append(
            {
                "requirement_id": req.id,
                "description": req.description,
                "source_ref": req.source_ref or "REQ-Baseline",
                "acceptance_criteria": req.acceptance_criteria,
                "task_ids": [t.task_id for t in covering],
                "verification_status": v_status,
                "verification_label": v_label,
                "evidence_note": evidence_note,
            }
        )
    return rows


def build_run_projection(
    store: RunStore,
    workspace_root: Path,
    run_id: str,
    *,
    artifacts_root: Path | None = None,
) -> dict[str, Any]:
    """Derive a complete read-only Ariadne console view from SQLite events and run artifacts."""
    snap = store.get_run(run_id)
    spec: GoalSpec | None = snap["spec"]
    plan: TaskPlan | None = snap["plan"]
    events = store.list_events(run_id, after_seq=0)
    run_dir = (
        artifacts_root
        if artifacts_root is not None
        else Path(workspace_root).resolve() / ".olympus" / "runs"
    ) / run_id

    spec_approved = bool(
        spec
        and store.is_approved(
            run_id=run_id,
            target=ApprovalTarget.GOAL_SPEC,
            target_id=spec.spec_id,
            target_hash=spec.content_hash,
        )
    )
    plan_approved = bool(
        plan
        and store.is_approved(
            run_id=run_id,
            target=ApprovalTarget.TASK_PLAN,
            target_id=plan.plan_id,
            target_hash=plan.content_hash,
        )
    )

    tasks = plan.tasks if plan else []
    layers, crit_nodes, crit_edges = _compute_dag_layers_and_critical_path(tasks)
    task_by_id = {t.task_id: t for t in tasks}

    dag_nodes: list[dict[str, Any]] = []
    dag_edges: list[dict[str, Any]] = []
    for t in tasks:
        art = _read_task_artifacts(run_dir, t)
        # Extract structured diagnostic fields if present in last_error or events
        diag_tool = ""
        diag_output = ""
        diag_attribution = ""
        for c in t.acceptance_checks:
            if c.command and not diag_tool:
                diag_tool = f"{c.command} → {'通过' if c.passed else '失败'}"
        if t.last_error:
            parts = [line.strip() for line in t.last_error.splitlines() if line.strip()]
            diag_output = parts[0] if parts else t.last_error
            if len(parts) > 1:
                diag_attribution = "；".join(parts[1:])
        if not diag_attribution and t.accepted_risk_reason:
            diag_attribution = f"未验证 · 人工接受：{t.accepted_risk_reason}"

        dag_nodes.append(
            {
                "task_id": t.task_id,
                "title": t.title,
                "objective": t.objective,
                "status": t.status.value,
                "layer": layers.get(t.task_id, 0),
                "critical": t.task_id in crit_nodes,
                "attempt": t.attempt,
                "review_cycle": t.review_cycle,
                "max_cycles": t.max_cycles,
                "requirement_refs": t.requirement_refs,
                "depends_on": t.depends_on,
                "write_scope": t.write_scope,
                "read_scope": t.read_scope,
                "last_error": t.last_error,
                "accepted_risk_reason": t.accepted_risk_reason,
                "executor": "Hephaestus-1"
                if t.task_id.endswith(("1", "2", "3"))
                else "Hephaestus-2",
                "diagnostic_tool": diag_tool or "pytest -q",
                "diagnostic_output": diag_output
                or ("全部检查通过" if t.status == TaskStatus.ACCEPTED else "等待执行"),
                "diagnostic_attribution": diag_attribution
                or (
                    "Argus 冷启动独立复核通过"
                    if t.status == TaskStatus.ACCEPTED
                    else "等待前置产物解锁"
                ),
                "acceptance_checks": [c.model_dump(mode="json") for c in t.acceptance_checks],
                "artifacts": art,
            }
        )
        for dep in t.depends_on:
            dep_node = task_by_id.get(dep)
            blocked_edge = bool(
                dep_node and dep_node.status in (TaskStatus.BLOCKED, TaskStatus.FAILED)
            )
            dag_edges.append(
                {
                    "from": dep,
                    "to": t.task_id,
                    "critical": (dep, t.task_id) in crit_edges,
                    "blocked_edge": blocked_edge,
                }
            )

    blocked_tasks = [t for t in tasks if t.status in (TaskStatus.BLOCKED, TaskStatus.FAILED)]
    hitl_banner: dict[str, Any] | None = None
    if blocked_tasks:
        bt = blocked_tasks[0]
        hitl_banner = {
            "kind": "task_blocked",
            "tone": "danger",
            "task_id": bt.task_id,
            "headline": f"阻塞：{bt.task_id} 重试 {max(1, bt.review_cycle)} 次仍失败，需要人工处理",
            "detail": bt.last_error or "验收未通过且重试额度已用尽",
            "actions": ["inspect", "retry", "hint", "reassign", "accept_risk"],
        }
    elif spec is not None and not spec_approved:
        blockers = spec.unresolved_blockers()
        hitl_banner = {
            "kind": "spec_gate",
            "tone": "warn",
            "task_id": None,
            "headline": (
                f"待澄清与确认：GoalSpec ({spec.spec_id}) 含 {len(blockers)} 项未决阻断项"
                if blockers
                else f"关口 1 待批准：需求规格 {spec.spec_id} 已就绪，等待真人确认冻结"
            ),
            "detail": spec.goal,
            "actions": ["approve_spec"],
        }
    elif plan is not None and not plan_approved:
        hitl_banner = {
            "kind": "plan_gate",
            "tone": "warn",
            "task_id": None,
            "headline": (
                f"关口 2 待批准：任务计划 {plan.plan_id}（共 {len(tasks)} 个任务）等待批准开工"
            ),
            "detail": f"绑定需求指纹：{plan.spec_hash[:16]}…",
            "actions": ["approve_plan"],
        }
    elif snap["status"] == RunStatus.PAUSED:
        hitl_banner = {
            "kind": "run_paused",
            "tone": "info",
            "task_id": None,
            "headline": "运行已暂停：当前停止派发新任务，等待人工恢复",
            "detail": "点击右上角「恢复运行」继续调度就绪任务",
            "actions": ["resume"],
        }

    accepted_count = sum(1 for t in tasks if t.status == TaskStatus.ACCEPTED)
    ledger: BudgetLedger = snap["budget_ledger"]
    policy: BudgetPolicy = snap["budget_policy"]

    return {
        "run_id": snap["run_id"],
        "title": snap["title"],
        "mode": snap["mode"].value,
        "status": snap["status"].value,
        "state_revision": snap["state_revision"],
        "created_at": snap["created_at"],
        "updated_at": snap["updated_at"],
        "spec_approved": spec_approved,
        "plan_approved": plan_approved,
        "stage_pipeline": _build_stage_pipeline(snap, spec_approved, plan_approved),
        "budget": {
            "calls_used": ledger.llm_calls_used,
            "calls_reserved": ledger.llm_calls_reserved,
            "max_calls": policy.max_llm_calls,
            "input_tokens": ledger.input_tokens,
            "output_tokens": ledger.output_tokens,
            "total_tokens": ledger.total_tokens,
            "max_tokens": policy.max_total_tokens,
            "unknown_outcome_calls": ledger.unknown_outcome_calls,
        },
        "progress": {
            "accepted_tasks": accepted_count,
            "total_tasks": len(tasks),
            "ratio_text": f"{accepted_count} / {len(tasks)}",
        },
        "hitl_banner": hitl_banner,
        "spec": spec.model_dump(mode="json") if spec else None,
        "plan": plan.model_dump(mode="json") if plan else None,
        "dag": {
            "nodes": dag_nodes,
            "edges": dag_edges,
        },
        "swimlanes": _build_swimlanes(tasks, events),
        "traceability_matrix": _build_traceability_matrix(spec, tasks),
        "events": [e.model_dump(mode="json") for e in events[-80:]],
        "latest_event_seq": events[-1].event_seq if events else 0,
    }


def seed_demo_runs(
    store: RunStore,
    workspace_root: Path,
    *,
    artifacts_root: Path | None = None,
) -> list[str]:
    """Seed deterministic walkthrough runs (Campus Activity Waitlist & cattrs) if missing."""
    existing_ids = {r["run_id"] for r in store.list_runs(limit=100)}
    seeded: list[str] = []
    art_base = (
        artifacts_root
        if artifacts_root is not None
        else Path(workspace_root).resolve() / ".olympus" / "runs"
    )

    demo_id = "run-campus-waitlist"
    if demo_id not in existing_ids:
        store.create_run(
            title="社团活动报名与候补自动递补系统 (I001-v2)",
            mode=RunMode.EXECUTION,
            budget_policy=BudgetPolicy(
                max_llm_calls=120,
                max_total_tokens=2_000_000,
                max_concurrent_tasks=3,
            ),
            run_id=demo_id,
        )
        spec = GoalSpec(
            spec_id="spec-i001-v2",
            version=2,
            goal="实现社团活动报名与满员候补自动递补工具，保证并发名额不超限且取消后跳过已退出记录自动递补。",
            requirements=[
                RequirementItem(
                    id="FR-001",
                    description="并发争抢最后名额时不超活动容量上限",
                    acceptance_criteria=["同一活动事务级加锁校验余量，满员自动转入 WAITLISTED"],
                    source_ref="I001-v2 / D-CON-01",
                ),
                RequirementItem(
                    id="FR-002",
                    description="同一账号在同一活动内不重复占位",
                    acceptance_criteria=["活动 ID + 账号 ID 唯一约束，重复提交返回幂等状态"],
                    source_ref="H-01 / D-DATA-01",
                ),
                RequirementItem(
                    id="FR-003",
                    description="已报名者取消后自动递补最早的有效候补者",
                    acceptance_criteria=[
                        "按服务端入队序号递补首个状态仍为 WAITLISTED 的记录，跳过已取消记录"
                    ],
                    source_ref="I001-v2 / D-SM-01",
                ),
                RequirementItem(
                    id="FR-004",
                    description="候补者取消后立即退出有效候补队列",
                    acceptance_criteria=["状态迁移至 CANCELLED，后续名额释放时不再被递补"],
                    source_ref="I001-v2 / D-SM-01",
                ),
                RequirementItem(
                    id="FR-005",
                    description="支持主动查询当前报名/候补排位并展示无推送提示",
                    acceptance_criteria=[
                        "查询接口返回状态与排位，界面固定展示“仅能主动查询，不推送通知”"
                    ],
                    source_ref="I001-v2 / D-UI-01",
                ),
            ],
            out_of_scope=["不接入真实统一身份认证", "不收报名费", "不发送短信或邮件推送通知"],
            constraints=["报名、取消与自动递补必须在同一 SQLite 事务内锁定活动记录"],
            assumptions=["单活动并发请求峰值 <= 200 QPS"],
            open_ambiguities=[
                AmbiguityItem(
                    id="H-01",
                    question="同一账号取消后是否允许重新报名同一活动？",
                    impact="影响唯一索引与状态迁移表设计",
                    options=["允许重新入队并重新排号", "禁止再次报名"],
                    blocking=True,
                    answer="允许重新入队并重新排号，但同一时刻仅保留一条有效记录",
                )
            ],
            minority_notes=[
                CritiqueNote(
                    id="CRIT-QA-01",
                    perspective=CritiquePerspective.QA,
                    severity=CritiqueSeverity.WARNING,
                    summary="Momus 指出：最早候补者若已取消，递补游标必须跳过 CANCELLED 记录",
                    evidence_refs=["src/waitlist/service.py"],
                    resolved=True,
                    resolution_note="已纳入 FR-003 与 T03-R1 验收条件",
                )
            ],
            status=SpecStatus.CONFIRMED,
        )
        store.save_spec(demo_id, spec, actor="align")
        store.record_approval(
            ApprovalRecord(
                run_id=demo_id,
                target=ApprovalTarget.GOAL_SPEC,
                target_id=spec.spec_id,
                target_hash=spec.content_hash,
                approver="zeus",
                decision=ApprovalDecision.APPROVED,
                reason="需求规格 I001-v2 已核对确认",
            )
        )

        tasks = [
            TaskNode(
                task_id="T01",
                title="接口类型与测试夹具",
                objective=(
                    "定义活动、报名记录状态机 (CONFIRMED/WAITLISTED/CANCELLED) 与 SQLite 事务夹具。"
                ),
                requirement_refs=["FR-001", "FR-002"],
                depends_on=[],
                write_scope=["src/waitlist/models.py", "tests/conftest.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T01-cmd",
                        kind=CheckKind.COMMAND,
                        description="校验数据模型与枚举导入",
                        command="pytest tests/test_models.py -q",
                        passed=True,
                    ),
                    AcceptanceCheck(
                        id="T01-rev",
                        kind=CheckKind.INDEPENDENT_REVIEW,
                        description="Argus 独立复核接口契约",
                        passed=True,
                    ),
                ],
                status=TaskStatus.ACCEPTED,
                attempt=1,
                review_cycle=0,
            ),
            TaskNode(
                task_id="T02",
                title="报名与容量锁服务",
                objective="实现报名注册、同账号唯一校验与容量上限事务锁。",
                requirement_refs=["FR-001", "FR-002"],
                depends_on=["T01"],
                write_scope=["src/waitlist/registration.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T02-cmd",
                        kind=CheckKind.COMMAND,
                        description="运行报名与唯一性检查",
                        command="pytest tests/test_registration.py -q",
                        passed=True,
                    )
                ],
                status=TaskStatus.ACCEPTED,
                attempt=1,
                review_cycle=0,
            ),
            TaskNode(
                task_id="T03",
                title="候补入队与基础递补服务",
                objective="满员时按服务端时间戳进入候补队列，并在名额释放时触发递补。",
                requirement_refs=["FR-003", "FR-004"],
                depends_on=["T01"],
                write_scope=["src/waitlist/promotion.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T03-cmd",
                        kind=CheckKind.COMMAND,
                        description="运行基础候补递补单测",
                        command="pytest tests/test_promotion_basic.py -q",
                        passed=True,
                    )
                ],
                status=TaskStatus.ACCEPTED,
                attempt=1,
                review_cycle=0,
            ),
            TaskNode(
                task_id="T04",
                title="主动状态查询与免推送提示",
                objective="实现报名状态与候补顺位查询接口，附带固定提示“仅能主动查询，不推送通知”。",
                requirement_refs=["FR-005"],
                depends_on=["T01"],
                write_scope=["src/waitlist/query.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T04-cmd",
                        kind=CheckKind.COMMAND,
                        description="校验查询返回与免推送提示文案",
                        command="pytest tests/test_query.py -q",
                        passed=True,
                    )
                ],
                status=TaskStatus.ACCEPTED,
                attempt=1,
                review_cycle=0,
            ),
            TaskNode(
                task_id="T05",
                title="并发抢报与恢复交互联调",
                objective="基于契约夹具执行多线程并发争抢最后 1 个名额与异常恢复校验。",
                requirement_refs=["FR-001", "FR-002"],
                depends_on=["T03"],
                write_scope=["src/waitlist/concurrency.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T05-cmd",
                        kind=CheckKind.COMMAND,
                        description="20 线程并发压测最后名额",
                        command="pytest tests/test_concurrency.py -q",
                        passed=False,
                    )
                ],
                status=TaskStatus.RUNNING,
                attempt=1,
                review_cycle=0,
            ),
            TaskNode(
                task_id="T03-R1",
                title="修复已取消候补者重复递补缺陷",
                objective="确保递补查询在同一事务内过滤 CANCELLED 记录并原子更新被递补者状态。",
                requirement_refs=["FR-003", "FR-004"],
                depends_on=["T03"],
                write_scope=["src/waitlist/waitlist.py"],
                max_cycles=2,
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T03-R1-cmd",
                        kind=CheckKind.COMMAND,
                        description="验证候补取消与连续递补边界用例",
                        command="pytest tests/test_waitlist.py",
                        passed=False,
                    ),
                    AcceptanceCheck(
                        id="T03-R1-rev",
                        kind=CheckKind.INDEPENDENT_REVIEW,
                        description="Argus 审查事务边界",
                        passed=False,
                    ),
                ],
                status=TaskStatus.BLOCKED,
                attempt=2,
                review_cycle=2,
                last_error=(
                    "同一候补者被递补 2 次\n递补未在同一事务内校验队列（Momus 建议，待人工判断）"
                ),
            ),
            TaskNode(
                task_id="T07",
                title="报名与并发异常验收测试",
                objective="覆盖 FR-001、FR-002 与 FR-005 的端到端独立验收测试集。",
                requirement_refs=["FR-001", "FR-002", "FR-005"],
                depends_on=["T02", "T03", "T04", "T05"],
                write_scope=["tests/test_acceptance_reg.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T07-cmd",
                        kind=CheckKind.COMMAND,
                        description="执行报名与查询验收套件",
                        command="pytest tests/test_acceptance_reg.py -q",
                        passed=False,
                    )
                ],
                status=TaskStatus.WAITING,
                attempt=0,
                review_cycle=0,
            ),
            TaskNode(
                task_id="T06",
                title="候补取消与递补链验收测试",
                objective="覆盖 FR-003 与 FR-004 的端到端候补递补与取消跳过验收测试。",
                requirement_refs=["FR-003", "FR-004"],
                depends_on=["T02", "T03", "T03-R1"],
                write_scope=["tests/test_waitlist.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T06-cmd",
                        kind=CheckKind.COMMAND,
                        description="执行候补与递补验收测试",
                        command="pytest tests/test_waitlist.py -q",
                        passed=False,
                    )
                ],
                status=TaskStatus.WAITING,
                attempt=0,
                review_cycle=0,
            ),
            TaskNode(
                task_id="T08",
                title="全量集成与追踪矩阵合拢",
                objective="汇总 T06 与 T07 验收产物，生成最终交付包与追踪证据。",
                requirement_refs=["FR-001", "FR-002", "FR-003", "FR-004", "FR-005"],
                depends_on=["T06", "T07"],
                write_scope=["src/waitlist/__init__.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T08-cmd",
                        kind=CheckKind.COMMAND,
                        description="全量回归测试",
                        command="pytest -q",
                        passed=False,
                    )
                ],
                status=TaskStatus.WAITING,
                attempt=0,
                review_cycle=0,
            ),
        ]
        plan = TaskPlan(
            plan_id="plan-i001-v1",
            spec_id=spec.spec_id,
            spec_hash=spec.content_hash,
            version=1,
            tasks=tasks,
        )
        store.save_plan(demo_id, plan, actor="orchestrate")
        store.record_approval(
            ApprovalRecord(
                run_id=demo_id,
                target=ApprovalTarget.TASK_PLAN,
                target_id=plan.plan_id,
                target_hash=plan.content_hash,
                approver="zeus",
                decision=ApprovalDecision.APPROVED,
                reason="计划 DAG 与 write_scope 静态校验通过",
            )
        )
        store.set_run_status(
            demo_id,
            RunStatus.AWAITING_HUMAN,
            actor="chronos",
            stage="orchestrate",
            detail={"blocked_task": "T03-R1", "accepted": 4, "total": 9},
        )

        # Record explicit swimlane events matching the walkthrough diagram
        swimlane_events = [
            ("Hephaestus-1", "T02", "T02", "accepted", 1, "报名与容量锁服务已验收"),
            ("Hephaestus-1", "T03", "T03", "accepted", 1, "候补入队基础服务已验收"),
            ("Hephaestus-1", "T03-R1", "T03-R1 ×2", "blocked", 2, "同一候补者被递补 2 次"),
            ("Hephaestus-2", "T04", "T04", "accepted", 1, "状态查询与免推送提示已验收"),
            ("Hephaestus-2", "T05", "T05 运行中", "running", 1, "20 线程并发争抢最后名额压测中"),
            (
                "Momus",
                "T03-R1",
                "审查 DS-v1",
                "accepted",
                1,
                "指出最早候补者已取消边界并给出事务归因",
            ),
        ]
        for lane, tid, label, st, att, summ in swimlane_events:
            store.append_event(
                run_id=demo_id,
                stage="orchestrate",
                event_type="agent.swimlane",
                actor=lane,
                payload={
                    "lane": lane,
                    "task_id": tid,
                    "label": label,
                    "status": st,
                    "attempts": att,
                    "summary": summ,
                },
            )

        # Populate realistic budget usage (41 / 120 calls)
        for _ in range(41):
            store.reserve_llm_call(demo_id, actor="hephaestus:T03-R1", purpose="coding_step")
            store.settle_llm_call(
                demo_id, actor="hephaestus:T03-R1", input_tokens=2850, output_tokens=620
            )

        # Write realistic artifacts for T03-R1 attempt-2
        t03_art = art_base / demo_id / "artifacts" / "T03-R1" / "attempt-2"
        t03_art.mkdir(parents=True, exist_ok=True)
        (t03_art / "diff.patch").write_text(
            "--- a/src/waitlist/waitlist.py\n"
            "+++ b/src/waitlist/waitlist.py\n"
            "@@ -42,8 +42,14 @@ def promote_next_waitlisted(\n"
            "     conn: sqlite3.Connection, activity_id: str\n"
            " ) -> str |\n"
            "-    row = conn.execute(\n"
            '-        "SELECT account_id FROM registrations "\n'
            '-        "WHERE activity_id = ? ORDER BY seq ASC LIMIT 1",\n'
            "-        (activity_id,),\n"
            "-    ).fetchone()\n"
            "+    row = conn.execute(\n"
            '+        "SELECT id, account_id FROM registrations "\n'
            "+        \"WHERE activity_id = ? AND status = 'WAITLISTED' \"\n"
            '+        "ORDER BY seq ASC LIMIT 1",\n'
            "+        (activity_id,),\n"
            "+    ).fetchone()\n"
            "     if row is None:\n"
            "         return None\n"
            "+    # BUG: missing atomic status update inside same IMMEDIATE transaction\n"
            "     return str(row[1])\n",
            encoding="utf-8",
        )
        (t03_art / "check-T03-R1-cmd.log").write_text(
            "$ pytest tests/test_waitlist.py\n"
            "============================= test session starts ==============================\n"
            "collected 4 items\n\n"
            "tests/test_waitlist.py ..F.                                              [100%]\n\n"
            "=================================== FAILURES ===================================\n"
            "_________________ test_concurrent_cancel_promotes_once_only ____________________\n"
            "AssertionError: 同一候补者被递补 2 次 "
            "(expected 1 promotion for account 'stu-042', got 2)\n"
            "=========================== short test summary info ============================\n"
            "FAILED tests/test_waitlist.py::test_concurrent_cancel_promotes_once_only\n"
            "1 failed, 3 passed in 0.19s\n",
            encoding="utf-8",
        )
        verdict = ReviewVerdict(
            task_id="T03-R1",
            attempt=2,
            passed=False,
            summary="递补查询虽过滤了 WAITLISTED，但未在同一事务内将命中记录原子更新为 CONFIRMED。",
            findings=[
                ReviewFinding(
                    id="F-01",
                    severity=ReviewSeverity.IMPORTANT,
                    file_path="src/waitlist/waitlist.py",
                    line=49,
                    message=(
                        "SELECT 与 UPDATE 分属两个事务，并发取消两个名额时同一候补者被递补 2 次。"
                    ),
                    suggestion=(
                        "在 BEGIN IMMEDIATE 事务内完成 SELECT ... WHERE status='WAITLISTED' "
                        "与 UPDATE status='CONFIRMED'。"
                    ),
                )
            ],
        )
        (t03_art / "verdict-cycle-2.json").write_text(
            verdict.model_dump_json(indent=2),
            encoding="utf-8",
        )
        seeded.append(demo_id)

    return seeded
