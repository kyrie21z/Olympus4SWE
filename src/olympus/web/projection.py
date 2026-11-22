"""Deterministic event-sourced projection and demo seeding for the Ariadne Web Console."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TypedDict

import olympus.web.acceptance as delivery_review
from olympus.core.contracts import (
    ApprovalTarget,
    BudgetLedger,
    BudgetPolicy,
    CheckKind,
    DesignSpec,
    GoalSpec,
    ProviderTimeoutRetry,
    RunEvent,
    RunStatus,
    TaskNode,
    TaskPlan,
    TaskStatus,
)
from olympus.core.dag import compute_dispatchable_tasks
from olympus.core.store import RunStore
from olympus.graph import load_workflow
from olympus.graph.progress import build_workflow_view, is_blueprint_completed
from olympus.harness.activity import tool_output_status
from olympus.harness.receipt import build_turn_receipt
from olympus.web.approval_review import (
    APPROVAL_ARTIFACTS,
    ApprovalReview,
    ReviewSection,
    compare_approval_review,
    design_review_sections,
    expand_candidate_review,
    goal_review_sections,
    plan_review_sections,
)

# Log rules such as "===== test session starts =====" or "____ test_x ____" carry no signal.
_NOISE_LINE = re.compile(r"[=\-_*·~]{4,}(?:.*[=\-_*·~]{4,})?")

_TOOL_STATUSES = ("ok", "error", "denied")

# Waterfall timeline, as percent of the shared track. Align holds [0, 16];
# worker layers fan out in cumulative waves; Momus and Argus close the tail.
_WF_ALIGN_WIDTH = 16.0
_WF_WORKERS_START = 18.0
_WF_WORKERS_SPAN = 66.0
_WF_WAVE_SLOTS = 5
_WF_FAN_OUT = 2.0
_WF_MIN_WIDTH = 8.0
_WF_MAX_OFFSET = 90.0
_WF_REVIEW_WIDTH = 8.0


class TaskBlocker(TypedDict):
    task_id: str
    headline: str
    detail: str
    affected_task_ids: list[str]
    actions: list[str]


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


def _parse_tool_trace_line(raw_line: str) -> dict[str, Any] | None:
    """Normalize one worker trace line into a tool step, or None for non-tool lines."""
    try:
        entry = json.loads(raw_line)
    except (TypeError, ValueError):
        return None
    if not isinstance(entry, dict) or entry.get("event") not in ("tool_call", "tool_end"):
        return None
    return _tool_trace_step(entry)


def _tool_trace_step(entry: dict[str, Any]) -> dict[str, Any]:
    status = str(entry.get("status") or "")
    if status not in _TOOL_STATUSES:
        if entry.get("denied"):
            status = "denied"
        elif entry.get("is_error"):
            status = "error"
        else:
            status = "ok"
    arguments = entry.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    target = (
        entry.get("target") or entry.get("path") or entry.get("command")
        or arguments.get("path") or arguments.get("command") or arguments.get("pattern") or ""
    )
    step = entry.get("step")
    duration = entry.get("duration_ms")
    return {
        "step": step if type(step) is int else 0,
        "tool": str(entry.get("tool") or "unknown"),
        "target": str(target)[:160],
        "status": status,
        "is_error": status != "ok",
        "duration_ms": duration if type(duration) is int else 0,
        "output_preview": str(entry.get("output") or "")[:240],
    }


def _legacy_tool_trace_steps(raw_line: str) -> list[dict[str, Any]]:
    try:
        entry = json.loads(raw_line)
    except ValueError:
        return []
    if not isinstance(entry, dict):
        return []
    calls, outputs = entry.get("tool_calls"), entry.get("tool_outputs")
    if not isinstance(calls, list) or not isinstance(outputs, list):
        return []
    calls_by_id = {
        call["id"]: call for call in calls
        if isinstance(call, dict) and isinstance(call.get("id"), str)
    }
    steps = []
    for result in outputs:
        if not isinstance(result, dict) or not isinstance(result.get("id"), str):
            continue
        call = calls_by_id.get(result["id"], {})
        tool = str(call.get("name") or result.get("name") or "unknown")
        output = str(result.get("output") or "")
        steps.append(_tool_trace_step({
            "step": entry.get("step"),
            "tool": tool,
            "arguments": call.get("arguments"),
            "status": tool_output_status(tool, output),
            "output": output,
        }))
    return steps


def _read_task_artifacts(
    run_dir: Path,
    task: TaskNode,
    *,
    attempt: int | None = None,
) -> dict[str, Any]:
    """Load recorded diff, command check logs, review verdicts, manifest and worker trace."""
    task_art_root = run_dir / "artifacts" / task.task_id
    attempt_idx = attempt if attempt is not None else max(1, task.attempt)
    art_dir = task_art_root / f"attempt-{attempt_idx}"
    if attempt is None and not art_dir.exists() and task_art_root.exists():
        candidates = sorted(task_art_root.glob("attempt-*"))
        if candidates:
            art_dir = candidates[-1]

    def readable(path: Path) -> bool:
        return path.is_file() and (attempt is None or not path.is_symlink())

    diff_text = ""
    diff_path = art_dir / "diff.patch"
    if readable(diff_path):
        diff_text = diff_path.read_text(encoding="utf-8", errors="replace")[:8000]

    manifest_md = ""
    manifest_path = art_dir / "MANIFEST.md"
    if readable(manifest_path):
        manifest_md = manifest_path.read_text(encoding="utf-8", errors="replace")[:12000]

    check_logs: dict[str, str] = {}
    if art_dir.exists():
        for log_file in sorted(art_dir.glob("check-*.log")):
            if not readable(log_file):
                continue
            check_logs[log_file.stem.removeprefix("check-")] = log_file.read_text(
                encoding="utf-8", errors="replace"
            )[:4000]

    verdicts: list[dict[str, Any]] = []
    if art_dir.exists():
        for v_file in sorted(art_dir.glob("verdict-cycle-*.json")):
            if not readable(v_file):
                continue
            try:
                verdicts.append(json.loads(v_file.read_text(encoding="utf-8")))
            except Exception:
                continue

    tool_calls: list[dict[str, Any]] = []
    trace_file = art_dir / "worker.jsonl"
    if attempt is None and readable(trace_file):
        with trace_file.open(encoding="utf-8", errors="replace") as trace:
            for raw_line in trace:
                step = _parse_tool_trace_line(raw_line)
                if step is not None:
                    tool_calls.append(step)
                elif '"tool_outputs"' in raw_line:
                    tool_calls.extend(_legacy_tool_trace_steps(raw_line))
                if len(tool_calls) >= 40:
                    break

    return {
        "attempt_dir": str(art_dir) if art_dir.exists() else None,
        "diff_patch": diff_text,
        "check_logs": check_logs,
        "verdicts": verdicts,
        "tool_calls": tool_calls[:40],
        "manifest_md": manifest_md,
    }


def _diff_line_stats(diff_text: str) -> tuple[int, int]:
    """Count added/removed patch lines; the +++/--- file headers are not content."""
    added = 0
    removed = 0
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def _tail_lines(text: str, *, limit: int = 4) -> list[str]:
    """Return the clean tail of a check log: the command line plus the last lines."""
    meaningful = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not _NOISE_LINE.fullmatch(line.strip())
    ]
    if not meaningful:
        return []
    command = next((line for line in meaningful if line.startswith("$ ")), None)
    tail = meaningful[-3:]
    if command is not None and command not in tail:
        tail = [command, *tail]
    return tail[-limit:]


def _synthesize_tool_stream(task: TaskNode) -> list[dict[str, Any]]:
    """Build contextual tool steps for tasks that carry no recorded worker trace."""
    if task.status in (TaskStatus.WAITING, TaskStatus.CANCELLED):
        return []
    read_target = task.write_scope[0] if task.write_scope else "README.md"
    write_target = task.write_scope[0] if task.write_scope else f"src/{task.task_id.lower()}.py"
    command = next((c.command for c in task.acceptance_checks if c.command), "pytest -q")
    failed = task.status in (TaskStatus.BLOCKED, TaskStatus.FAILED)
    attempt = max(1, task.attempt)
    return [
        {
            "tool": "read_file",
            "target": read_target,
            "status": "ok",
            "duration_ms": 320 + 40 * attempt,
        },
        {
            "tool": "edit_file",
            "target": write_target,
            "status": "ok",
            "duration_ms": 1180 + 120 * attempt,
        },
        {
            "tool": "run_bash",
            "target": command,
            "status": "error" if failed else "ok",
            "duration_ms": 2140 + 90 * attempt,
        },
    ]


def _build_subagent_stream(task: TaskNode, art: dict[str, Any]) -> list[dict[str, Any]]:
    """Compact tool events for the dock: recorded trace first, contextual synthesis otherwise."""
    recorded = art.get("tool_calls") or []
    if recorded:
        return [
            {
                "tool": str(call.get("tool") or "unknown"),
                "target": str(call.get("target") or ""),
                "status": str(call.get("status") or "ok"),
                "duration_ms": int(call.get("duration_ms") or 0),
            }
            for call in recorded[:8]
        ]
    return _synthesize_tool_stream(task)


def _build_terminal_lines(
    task: TaskNode,
    art: dict[str, Any],
    *,
    blueprint_completed: bool = False,
    unmet_deps: list[str] | None = None,
) -> list[str]:
    """Up to four clean lines for the dock's mini terminal."""
    for log_text in (art.get("check_logs") or {}).values():
        lines = _tail_lines(str(log_text))
        if lines:
            return lines
    if task.last_error:
        error_lines = [line.strip() for line in task.last_error.splitlines() if line.strip()]
        if error_lines:
            return error_lines[-4:]
    if task.status == TaskStatus.ACCEPTED:
        command = next((c.command for c in task.acceptance_checks if c.command), "pytest -q")
        return [f"$ {command}", "全部检查通过", "Argus 冷启动独立复核通过"]
    if task.status in (TaskStatus.RUNNING, TaskStatus.VERIFYING):
        command = next((c.command for c in task.acceptance_checks if c.command), "pytest -q")
        return [f"$ {command}", "执行中，等待命令门禁输出…"]
    if task.accepted_risk_reason:
        return ["⚠ 未验证 · 人工接受风险放行", task.accepted_risk_reason]
    if task.status == TaskStatus.READY:
        return ["◷ 任务已就绪，等待调度器认领"]
    if task.status == TaskStatus.WAITING:
        if blueprint_completed:
            return ["✓ 已冻结于 TaskPlan", "当前工作流止步于计划阶段，未启动代码执行"]
        if unmet_deps:
            return [f"◷ 等待前置产物解锁：{', '.join(unmet_deps)}"]
        return ["◷ 无前置依赖阻塞，等待调度器认领"]
    return ["◷ 等待执行"]


def _receipt_events(
    stream: list[dict[str, Any]],
    *,
    input_tokens: int,
    output_tokens: int,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        {
            "kind": "llm",
            "calls": 1,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    ]
    for step in stream:
        events.append(
            {
                "kind": "tool_end",
                "tool": step["tool"],
                "is_error": step["status"] == "error",
                "duration_ms": step["duration_ms"],
            }
        )
    return events


def _build_receipt(
    task: TaskNode,
    stream: list[dict[str, Any]],
    art: dict[str, Any],
    *,
    order: int,
) -> dict[str, Any]:
    """Deterministic receipt for a task whose turn.receipt event was never recorded."""
    added, removed = _diff_line_stats(str(art.get("diff_patch") or ""))
    checks_total = len(task.acceptance_checks)
    checks_passed = sum(1 for check in task.acceptance_checks if check.passed)
    return build_turn_receipt(
        turn_id=f"turn-{task.task_id.lower()}-a{max(1, task.attempt)}",
        actor=f"hephaestus:{task.task_id}",
        task_id=task.task_id,
        events=_receipt_events(
            stream,
            input_tokens=2480 + 140 * order,
            output_tokens=560 + 40 * order,
        ),
        diff_lines=(added, removed),
        checks_passed=checks_passed,
        checks_total=checks_total,
    )


def _synthesize_manifest_md(
    task: TaskNode,
    art: dict[str, Any],
    receipt: dict[str, Any],
) -> str:
    """Clean Markdown summary used when an attempt has no recorded MANIFEST.md."""
    added, removed = _diff_line_stats(str(art.get("diff_patch") or ""))
    checks_total = len(task.acceptance_checks)
    checks_passed = sum(1 for check in task.acceptance_checks if check.passed)
    if task.status == TaskStatus.ACCEPTED:
        outcome = "accepted"
    elif task.status in (TaskStatus.BLOCKED, TaskStatus.FAILED):
        outcome = "checks_failed"
    else:
        outcome = "in_progress"
    lines = [
        f"# MANIFEST — {task.task_id} (attempt {max(1, task.attempt)})",
        "",
        f"- Task: {task.task_id} — {task.title}",
        f"- Outcome: {outcome}",
        f"- Task status: {task.status.value}",
        f"- Review cycle: {task.review_cycle}",
        f"- Checks: {checks_passed}/{checks_total} passed",
        f"- Diff: +{added}/-{removed} lines",
        f"- Turn: {receipt.get('turn_id', '(none)')}",
        f"- Receipt: {receipt.get('one_line', '(none)')}",
        "",
        "## Objective",
        "",
        task.objective,
        "",
        "## Write scope",
        "",
    ]
    lines.extend(f"- {path}" for path in task.write_scope)
    lines.extend(["", "## Acceptance checks", ""])
    for check in task.acceptance_checks:
        state = "passed" if check.passed else "not passed"
        lines.append(f"- {check.id} ({check.kind.value}): {state} — {check.description}")
    lines.append("")
    return "\n".join(lines)


def _build_stage_pipeline(
    snap: dict[str, Any],
    spec_approved: bool,
    plan_approved: bool,
    design_approved: bool = False,
    *,
    stop_after: str | None = None,
) -> list[dict[str, str]]:
    spec: GoalSpec | None = snap["spec"]
    design: DesignSpec | None = snap.get("design")
    plan: TaskPlan | None = snap["plan"]
    status: RunStatus = snap["status"]
    tasks = plan.tasks if plan else []
    workflow_stop = stop_after
    if workflow_stop is None:
        try:
            workflow_stop = load_workflow(
                str(snap.get("workflow_name") or "product_blueprint")
            ).stop_after
        except Exception:  # noqa: BLE001 - fallback for legacy/unknown workflows
            workflow_stop = "execute"

    align_state = (
        "done"
        if spec_approved or (spec is None and design is not None)
        else ("active" if spec is not None else "pending")
    )
    design_state = (
        "done"
        if design_approved or plan_approved
        else ("active" if design is not None or spec_approved else "pending")
    )
    plan_state = (
        "done"
        if plan_approved
        else (
            "active"
            if plan is not None and (spec_approved or design_approved)
            else "pending"
        )
    )

    all_accepted = bool(tasks) and all(
        t.status in (TaskStatus.ACCEPTED, TaskStatus.CANCELLED) for t in tasks
    )
    any_blocked = any(t.status in (TaskStatus.BLOCKED, TaskStatus.FAILED) for t in tasks)

    if all_accepted and status == RunStatus.COMPLETED:
        exec_state = "done"
        verify_state = "done"
    elif status == RunStatus.COMPLETED and workflow_stop == "plan" and plan_approved:
        exec_state = "skipped"
        verify_state = "skipped"
    elif status == RunStatus.COMPLETED and workflow_stop == "design" and design_approved:
        plan_state = "skipped"
        exec_state = "skipped"
        verify_state = "skipped"
    elif status == RunStatus.COMPLETED and workflow_stop == "goal" and spec_approved:
        design_state = "skipped"
        plan_state = "skipped"
        exec_state = "skipped"
        verify_state = "skipped"
    elif plan_approved:
        executing = status == RunStatus.RUNNING or (
            status == RunStatus.PAUSED
            and any(t.status in (TaskStatus.RUNNING, TaskStatus.VERIFYING) for t in tasks)
        )
        exec_state = "blocked" if any_blocked and not executing else "active"
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
    design: DesignSpec | None = None,
    *,
    delivery_acceptance: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build Requirement x DesignModule x Task x Verification Status rows."""
    if spec is None and design is None:
        return []

    if spec is not None:
        req_entries = [
            (
                req.id,
                req.description,
                req.source_ref or "REQ-Baseline",
                req.acceptance_criteria,
            )
            for req in spec.requirements
        ]
    else:
        assert design is not None
        seen_reqs: list[str] = []
        for module in design.modules:
            for ref in module.requirement_refs:
                if ref not in seen_reqs:
                    seen_reqs.append(ref)
        req_entries = [
            (
                req_id,
                "直接输入 DesignSpec 关联模块："
                + ", ".join(m.name for m in design.modules if req_id in m.requirement_refs),
                "DesignSpec-Direct",
                [],
            )
            for req_id in seen_reqs
        ]

    rows: list[dict[str, Any]] = []
    for req_id, req_desc, req_source_ref, req_criteria in req_entries:
        covering = [t for t in tasks if req_id in t.requirement_refs]
        design_modules = [
            {"id": module.id, "name": module.name}
            for module in (design.modules if design is not None else [])
            if req_id in module.requirement_refs
        ]
        if spec is not None:
            criteria = [
                row for row in (delivery_acceptance or {}).get("criteria", [])
                if row["requirement_id"] == req_id
            ]
            if any(row["status"] == "failed" for row in criteria):
                v_status, v_label, evidence_note = "failed", "失败", "有验收条件失败"
            elif any(row["risk_accepted"] for row in criteria):
                v_status, v_label = "unverified_risk_accepted", "未验证 · 人工接受"
                evidence_note = next(row["reason"] for row in criteria if row["risk_accepted"])
            elif criteria and all(row["status"] == "passed" for row in criteria):
                v_status, v_label, evidence_note = "verified", "通过", "所有验收条件均已通过"
            else:
                v_status, v_label, evidence_note = "pending", "未验证", "缺少可核验的逐条验收结果"
        elif not covering:
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
                "requirement_id": req_id,
                "description": req_desc,
                "source_ref": req_source_ref,
                "acceptance_criteria": req_criteria,
                "design_module_ids": [module["id"] for module in design_modules],
                "design_modules": design_modules,
                "task_ids": [t.task_id for t in covering],
                "verification_status": v_status,
                "verification_label": v_label,
                "evidence_note": evidence_note,
            }
        )
    return rows


def _build_waterfall_turns(
    spec: GoalSpec | None,
    tasks: list[TaskNode],
    nodes_by_id: dict[str, dict[str, Any]],
    receipt_by_actor: dict[str, dict[str, Any]],
    *,
    spec_approved: bool,
) -> list[dict[str, Any]]:
    """Flatten Align / Hephaestus / Momus / Argus turns into one chronological list.

    Computes duration-proportional ``offset_pct`` and ``width_pct`` over the
    cumulative timeline span so longer stages (e.g. Align 18.4s) render visibly
    wider than short worker turns (4.8s) and wave layers stagger left-to-right.
    """
    turns: list[dict[str, Any]] = []
    ordered = sorted(tasks, key=lambda item: (nodes_by_id[item.task_id]["layer"], item.task_id))
    executed = [task for task in ordered if task.attempt > 0]

    # Collect raw wave durations first so offsets and widths scale strictly by ms.
    align_dur = 18420 if spec is not None else 0
    layer_ids = sorted({int(nodes_by_id[task.task_id]["layer"]) for task in executed})
    wave_dur_by_layer: dict[int, int] = {lid: 0 for lid in layer_ids}
    task_durs: dict[str, int] = {}
    for task in executed:
        node = nodes_by_id[task.task_id]
        stream = node["subagent_stream"]
        dur = sum(int(step["duration_ms"]) for step in stream) + 900
        task_durs[task.task_id] = dur
        lid = int(node["layer"])
        wave_dur_by_layer[lid] = max(wave_dur_by_layer[lid], dur)

    reviewed = [t for t in ordered if t.review_cycle > 0]
    momus_dur = 9360 if reviewed else 0
    verified = [
        t
        for t in ordered
        if t.status == TaskStatus.ACCEPTED
        or any(c.kind == CheckKind.INDEPENDENT_REVIEW for c in t.acceptance_checks)
    ]
    argus_dur = 12680 if verified else 0

    wave_start_ms: dict[int, int] = {}
    cursor_ms = align_dur
    for lid in layer_ids:
        wave_start_ms[lid] = cursor_ms
        cursor_ms += wave_dur_by_layer[lid]
    momus_start_ms = cursor_ms
    if reviewed:
        cursor_ms += momus_dur
    argus_start_ms = cursor_ms
    total_span_ms = max(1, cursor_ms + argus_dur)

    def _span(start_ms: int, dur_ms: int) -> tuple[float, float]:
        off = min(92.0, round(start_ms / total_span_ms * 100.0, 1))
        wid = max(6.0, min(100.0 - off, round(dur_ms / total_span_ms * 100.0, 1)))
        return off, wid

    if spec is not None:
        align_receipt = (
            receipt_by_actor.get("align")
            or receipt_by_actor.get("Align")
            or build_turn_receipt(
                turn_id="turn-align-1",
                actor="Align",
                events=[
                    {
                        "kind": "llm",
                        "calls": max(1, len(spec.requirements)),
                        "input_tokens": 8400,
                        "output_tokens": 2150,
                    }
                ],
            )
        )
        align_manifest = "\n".join(
            [
                f"# MANIFEST — Align ({spec.spec_id})",
                "",
                f"- Requirements: {len(spec.requirements)}",
                f"- Open ambiguities: {len(spec.open_ambiguities)}",
                f"- Blocking ambiguities: {len(spec.unresolved_blockers())}",
                f"- Status: {spec.status.value}",
                "",
                "## Goal",
                "",
                spec.goal,
                "",
            ]
        )
        off, wid = _span(0, align_dur)
        turns.append(
            {
                "turn_id": str(align_receipt.get("turn_id") or "turn-align-1"),
                "actor": "Align",
                "task_id": None,
                "title": f"需求对齐 · {spec.spec_id}",
                "status": "accepted" if spec_approved else "running",
                "summary": spec.goal,
                "duration_ms": align_dur,
                "steps": max(1, len(spec.requirements)),
                "offset_pct": off,
                "width_pct": wid,
                "receipt": align_receipt,
                "manifest_md": align_manifest,
            }
        )

    layer_slot: dict[int, int] = {}
    for task in executed:
        node = nodes_by_id[task.task_id]
        stream = node["subagent_stream"]
        error_lines = [
            line.strip() for line in (task.last_error or "").splitlines() if line.strip()
        ]
        layer = int(node["layer"])
        slot = layer_slot.get(layer, 0)
        layer_slot[layer] = slot + 1
        dur = task_durs[task.task_id]
        # Slight horizontal stagger within parallel sibling waves so overlaps stay readable
        start = wave_start_ms.get(layer, align_dur) + slot * 400
        off, wid = _span(start, dur)
        turns.append(
            {
                "turn_id": str(node["receipt"].get("turn_id") or f"turn-{task.task_id.lower()}"),
                "actor": str(node["executor"]),
                "task_id": task.task_id,
                "title": f"{task.task_id} · {task.title}",
                "status": task.status.value,
                "summary": error_lines[0] if error_lines else task.objective,
                "duration_ms": dur,
                "steps": max(1, len(stream)),
                "offset_pct": off,
                "width_pct": wid,
                "receipt": node["receipt"],
                "manifest_md": node["manifest_md"],
            }
        )

    if reviewed:
        reviewed_task = reviewed[0]
        momus_receipt = build_turn_receipt(
            turn_id="turn-momus-1",
            actor="Momus",
            task_id=reviewed_task.task_id,
            events=[
                {"kind": "llm", "calls": 1, "input_tokens": 5120, "output_tokens": 980},
                {"kind": "tool_end", "tool": "read_file", "is_error": False, "duration_ms": 460},
                {"kind": "tool_end", "tool": "read_file", "is_error": False, "duration_ms": 520},
            ],
        )
        off, wid = _span(momus_start_ms, momus_dur)
        turns.append(
            {
                "turn_id": "turn-momus-1",
                "actor": "Momus",
                "task_id": reviewed_task.task_id,
                "title": f"对抗式审查 · {reviewed_task.task_id}",
                "status": "accepted",
                "summary": (
                    "指出最早候补者已取消边界并给出事务归因"
                    if reviewed_task.status == TaskStatus.BLOCKED
                    else "审查未发现新的阻断性风险"
                ),
                "duration_ms": momus_dur,
                "steps": 2,
                "offset_pct": off,
                "width_pct": wid,
                "receipt": momus_receipt,
                "manifest_md": "\n".join(
                    [
                        f"# MANIFEST — Momus ({reviewed_task.task_id})",
                        "",
                        "- Perspective: QA / adversarial review",
                        "- Findings: 1 important",
                        "",
                        reviewed_task.last_error or "无阻断性发现",
                        "",
                    ]
                ),
            }
        )

    if verified:
        argus_receipt = build_turn_receipt(
            turn_id="turn-argus-1",
            actor="Argus",
            events=[
                {"kind": "llm", "calls": 1, "input_tokens": 4200, "output_tokens": 760},
                {"kind": "tool_end", "tool": "run_bash", "is_error": False, "duration_ms": 2480},
            ],
        )
        off, wid = _span(argus_start_ms, argus_dur)
        turns.append(
            {
                "turn_id": "turn-argus-1",
                "actor": "Argus",
                "task_id": None,
                "title": "冷启动独立复核 · 全量验收",
                "status": (
                    "accepted"
                    if all(t.status == TaskStatus.ACCEPTED for t in verified)
                    else "running"
                ),
                "summary": f"独立复核 {len(verified)} 个任务的命令门禁与产物指纹",
                "duration_ms": argus_dur,
                "steps": 1,
                "offset_pct": off,
                "width_pct": wid,
                "receipt": argus_receipt,
                "manifest_md": "\n".join(
                    [
                        "# MANIFEST — Argus",
                        "",
                        f"- Reviewed tasks: {', '.join(t.task_id for t in verified)}",
                        "- Cold-start review: no worker context reused",
                        "",
                    ]
                ),
            }
        )

    return turns


def read_recorded_attempt(
    run_dir: Path, plan: TaskPlan | None, task_id: str, attempt: int
) -> dict[str, Any] | None:
    task = next((task for task in plan.tasks if task.task_id == task_id), None) if plan else None
    root = (run_dir / "artifacts").resolve()
    target = (root / task_id / f"attempt-{attempt}").resolve()
    if task is None or attempt < 1 or not target.is_relative_to(root) or not target.is_dir():
        return None
    art = _read_task_artifacts(run_dir, task, attempt=attempt)
    return {name: art[name] for name in ("diff_patch", "manifest_md", "check_logs", "verdicts")}


def _build_run_progress(
    snap: dict[str, Any], events: list[RunEvent], dag_nodes: list[dict[str, Any]], run_dir: Path
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    publications = {
        "spec.saved": ("spec", "GoalSpec", "spec_id"),
        "design.saved": ("design", "DesignSpec", "design_id"),
        "plan.saved": ("plan", "任务计划", "plan_id"),
    }
    milestones = {
        "run.created", "intake.triaged", "spec.clarified", "approval.recorded",
        "graph.node_started", "graph.node_finished", "graph.paused", "graph.failed",
        "graph.finished", "run.status_changed", "run.workflow_promoted", "task.requeued",
        "task.blocked", "task.accepted", "task.accepted_with_risk",
    }
    tasks = {node["task_id"]: node for node in dag_nodes}
    attempts: dict[str, tuple[int, int]] = {}
    last_publication: dict[str, str] = {}
    outputs: dict[str, dict[str, Any]] = {}
    recent = None
    for event in events:
        kind = event.event_type
        payload = event.payload
        entry = {
            "event_seq": event.event_seq,
            "created_at": event.created_at.isoformat(),
            "event_type": kind,
            "node": payload.get("node"),
            "task_id": payload.get("task_id"),
            "to": payload.get("to"),
        }
        output = None
        if kind in publications:
            category, label, id_field = publications[kind]
            version = payload.get("version")
            fingerprint = payload.get("content_hash")
            artifact_id = payload.get(id_field)
            if not fingerprint or not artifact_id or version is None:
                continue
            key = f"{category}:{artifact_id}:{version}:{fingerprint}"
            if last_publication.get(category) == key:
                continue
            last_publication[category] = key
            outputs.pop(key, None)
            artifact = snap[category]
            current = artifact is not None and artifact.content_hash == fingerprint
            summary = f"{label} v{version} 已就绪"
            if category == "spec" and current:
                summary += f"，{artifact.goal}"
            elif category == "plan" and isinstance(payload.get("task_count"), int):
                summary += f"，包含 {payload['task_count']} 个任务"
            elif category == "design" and isinstance(payload.get("module_count"), int):
                summary += f"，包含 {payload['module_count']} 个模块"
            output = {
                "key": key, "kind": category, "artifact_id": artifact_id,
                "version": version, "content_hash": fingerprint, "current": current,
                "summary": summary,
            }
        elif kind == "task.leased":
            task_id, lease, attempt = (
                payload.get("task_id"), payload.get("lease_generation"), payload.get("attempt")
            )
            if isinstance(task_id, str) and isinstance(lease, int) and isinstance(attempt, int):
                attempts[task_id] = (lease, attempt)
        elif kind in {"task.verifying", "review.verdict_recorded", "check.command_completed"}:
            task_id = payload.get("task_id")
            if not isinstance(task_id, str) or task_id not in tasks:
                continue
            task = tasks[task_id]
            category = "execution" if kind == "task.verifying" else "verification"
            lease, leased_attempt = attempts.get(task_id, (None, None))
            attempt = payload.get("attempt")
            if attempt is None and lease == payload.get("lease_generation"):
                attempt = leased_attempt
            folder = run_dir / "artifacts" / task_id / f"attempt-{attempt}"
            detail_available = False
            if kind == "task.verifying":
                diff = folder / "diff.patch"
                if not attempt or not diff.is_file() or not diff.stat().st_size:
                    continue
                key = f"execution:{task_id}:{attempt}"
                summary = f"{task_id} 第 {attempt} 次执行的代码变更已就绪"
                detail_available = True
            else:
                key = f"verification:{event.event_seq}"
                name = "独立复核" if kind == "review.verdict_recorded" else payload.get("check_id")
                if not name or not isinstance(payload.get("passed"), bool):
                    continue
                result = "通过" if payload.get("passed") is True else "未通过"
                summary = f"{task_id} · {name} · {result}"
                if attempt and folder.is_dir():
                    detail_available = (
                        any(folder.glob("verdict-cycle-*.json"))
                        if kind == "review.verdict_recorded"
                        else (folder / f"check-{name}.log").is_file()
                    )
            output = {
                "key": key, "kind": category, "artifact_id": task_id,
                "version": attempt, "current": detail_available and attempt == task["attempt"],
                "detail_available": detail_available, "summary": summary,
            }
        if output and output["key"] not in outputs:
            outputs[output["key"]] = entry | output
            recent = entry | {"summary": output["summary"]}
        elif not output and kind in milestones:
            if kind == "run.status_changed" and payload.get("from") == payload.get("to"):
                continue
            recent = entry
    return recent, list(reversed(outputs.values()))


def _live_task_progress(
    tasks: list[TaskNode], events: list[RunEvent]
) -> dict[str, dict[str, Any]]:
    attempts = {task.task_id: task.attempt for task in tasks}
    progress: dict[str, dict[str, Any]] = {
        task.task_id: {
            "driver_name": None,
            "current_tool": None,
            "last_event_at": None,
            "recent_tools": [],
            "steer_count": 0,
            "watchdog_verdict": None,
            "watchdog_summary": None,
        }
        for task in tasks
    }
    configured_driver = None
    for event in events:
        payload = event.payload
        if event.event_type == "run.status_changed" and "execution_drivers" in payload:
            configured_driver = payload["execution_drivers"]["worker_driver"].split("@", 1)[0]
        task_id = payload.get("task_id")
        if not isinstance(task_id, str) or task_id not in progress:
            continue
        if payload.get("attempt") != attempts[task_id]:
            continue
        entry = progress[task_id]
        kind = event.event_type
        if kind in ("agent.tool_started", "agent.tool_finished"):
            entry["driver_name"] = payload["driver"]
            tool = {"step": payload["step"], "tool": payload["tool"]}
            if kind == "agent.tool_started":
                entry["current_tool"] = {**tool, "target_hint": payload["target_hint"]}
            else:
                current = entry["current_tool"]
                if current and all(current[key] == value for key, value in tool.items()):
                    entry["current_tool"] = None
                entry["recent_tools"] = [
                    *entry["recent_tools"][-4:],
                    {**tool, "status": payload["status"], "duration_ms": payload["duration_ms"]},
                ]
        elif kind == "agent.steered":
            entry["steer_count"] += 1
        elif kind in ("watchdog.checked", "watchdog.paused"):
            entry["watchdog_verdict"] = payload.get("verdict")
            entry["watchdog_summary"] = payload.get("summary")
        elif kind != "watchdog.error":
            continue
        entry["last_event_at"] = event.created_at.isoformat()

    for task in tasks:
        entry = progress[task.task_id]
        if task.status == TaskStatus.RUNNING:
            entry["driver_name"] = entry["driver_name"] or configured_driver
        else:
            entry["current_tool"] = None
    return progress


def build_run_projection(
    store: RunStore,
    workspace_root: Path,
    run_id: str,
    *,
    artifacts_root: Path | None = None,
) -> dict[str, Any]:
    """Derive a complete read-only Ariadne console view from SQLite events and run artifacts."""
    snap = store.get_run(run_id, include_approval_baselines=True)
    spec: GoalSpec | None = snap["spec"]
    plan: TaskPlan | None = snap["plan"]
    design: DesignSpec | None = snap.get("design")
    events = [
        event for event in store.list_events(run_id, after_seq=0)
        if event.state_revision <= snap["state_revision"]
    ]
    run_dir = (
        artifacts_root
        if artifacts_root is not None
        else Path(workspace_root).resolve() / ".olympus" / "runs"
    ) / run_id

    approved = snap["artifact_approved"]
    spec_approved = approved[ApprovalTarget.GOAL_SPEC]
    design_approved = approved[ApprovalTarget.DESIGN_SPEC]
    plan_approved = approved[ApprovalTarget.TASK_PLAN]

    approval_reviews: dict[str, ApprovalReview] = {}
    targets = {
        "spec": ApprovalTarget.GOAL_SPEC, "design": ApprovalTarget.DESIGN_SPEC,
        "plan": ApprovalTarget.TASK_PLAN,
    }
    baselines = snap["approval_baselines"]
    tasks = plan.tasks if plan else []
    layers, crit_nodes, crit_edges = _compute_dag_layers_and_critical_path(tasks)
    for action, key in APPROVAL_ARTIFACTS.items():
        artifact = snap[key]
        target = targets[key]
        if artifact and not approved[target]:
            sections: list[ReviewSection]
            if key == "spec":
                sections = goal_review_sections(artifact)
            elif key == "design":
                sections = design_review_sections(artifact)
            else:
                sections = plan_review_sections(artifact, layers)
            review: ApprovalReview = {
                "artifact": key,
                "target_hash": artifact.content_hash,
                "sections": sections,
            }
            baseline = baselines.get(target)
            if baseline is not None:
                compare_approval_review(
                    review, artifact, baseline,
                    plan_layers=lambda tasks: _compute_dag_layers_and_critical_path(tasks)[0],
                    goals=[item for item in (spec, baselines.get(ApprovalTarget.GOAL_SPEC))
                           if item is not None],
                    designs=[item for item in (design, baselines.get(ApprovalTarget.DESIGN_SPEC))
                             if item is not None],
                )
            elif target in baselines:
                review["comparison_gap"] = (
                    "缺少上次批准版本的可靠原文。以下展示本次候选的全部关键承诺。"
                )
                expand_candidate_review(review, artifact)
            else:
                review["comparison_gap"] = "暂无已批准版本可供对照。"
            approval_reviews[action] = review

    task_by_id = {t.task_id: t for t in tasks}
    blocked_by: dict[str, list[str]] = {}
    for task in sorted(tasks, key=lambda task: layers[task.task_id]):
        if task.status in (TaskStatus.BLOCKED, TaskStatus.FAILED):
            blocked_by[task.task_id] = [task.task_id]
        elif task.status in (TaskStatus.WAITING, TaskStatus.READY):
            blocked_by[task.task_id] = sorted(
                {blocker for dep in task.depends_on for blocker in blocked_by.get(dep, [])}
            )
    task_blockers: dict[str, TaskBlocker] = {
        task.task_id: {
            "task_id": task.task_id,
            "headline": f"阻塞：{task.task_id} 需要人工处理",
            "detail": (task.last_error or "").strip() or "任务当前无法继续，等待人工处理",
            "affected_task_ids": [
                affected.task_id
                for affected in tasks
                if task.task_id in blocked_by.get(affected.task_id, [])
            ],
            "actions": ["inspect", "retry", "hint", "reassign", "accept_risk"],
        }
        for task in tasks
        if task.status in (TaskStatus.BLOCKED, TaskStatus.FAILED)
    }

    delivery_acceptance = delivery_review.build_delivery_acceptance(
        spec if spec_approved else None,
        plan if plan_approved else None,
        [event for event in events if event.state_revision <= snap["state_revision"]],
        run_dir,
    )

    receipt_by_task: dict[str, dict[str, Any]] = {}
    receipt_by_actor: dict[str, dict[str, Any]] = {}
    retries_by_task: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        if ev.event_type == ProviderTimeoutRetry.event_type:
            retry = ProviderTimeoutRetry.model_validate(ev.payload)
            retries_by_task.setdefault(retry.task_id, []).append(
                {**retry.model_dump(mode="json"), "event_seq": ev.event_seq, "actor": ev.actor}
            )
        if ev.event_type != "turn.receipt":
            continue
        payload = dict(ev.payload)
        recorded_task = payload.get("task_id")
        if isinstance(recorded_task, str) and recorded_task:
            receipt_by_task.setdefault(recorded_task, payload)
        receipt_by_actor.setdefault(ev.actor, payload)

    workflow_name = str(snap.get("workflow_name") or "product_blueprint")
    workflow_view = build_workflow_view(
        workflow_name,
        events,
        snap["status"],
        spec_approved=spec_approved,
        design_approved=design_approved,
        plan_approved=plan_approved,
    )
    blueprint_completed = is_blueprint_completed(
        plan=plan,
        plan_approved=plan_approved,
        run_status=snap["status"],
        stop_after=workflow_view.get("stop_after"),
    )

    live_progress = _live_task_progress(tasks, events)
    dag_nodes: list[dict[str, Any]] = []
    dag_edges: list[dict[str, Any]] = []
    for order, t in enumerate(tasks):
        blocker = task_blockers.get(t.task_id)
        dependency_blockers = [] if blocker else blocked_by.get(t.task_id, [])
        retry_history = retries_by_task.get(t.task_id, [])
        provider_retry = retry_history[-1] if retry_history else None
        if provider_retry is not None and (
            provider_retry["phase"] != "started"
            or provider_retry["lease_generation"] != t.lease_generation
            or t.status not in (TaskStatus.RUNNING, TaskStatus.VERIFYING)
            or snap["status"] not in (RunStatus.RUNNING, RunStatus.PAUSED)
        ):
            provider_retry = None
        art = _read_task_artifacts(run_dir, t)
        stream = _build_subagent_stream(t, art)
        receipt = receipt_by_task.get(t.task_id) or _build_receipt(t, stream, art, order=order)
        manifest_md = str(art.get("manifest_md") or "") or _synthesize_manifest_md(t, art, receipt)
        unmet_deps = [
            dep
            for dep in t.depends_on
            if not (task_by_id.get(dep) and task_by_id[dep].status == TaskStatus.ACCEPTED)
        ]
        diag_tool = ""
        diag_output = ""
        diag_attribution = ""
        for c in t.acceptance_checks:
            if c.command and not diag_tool:
                diag_tool = f"{c.command} → {'通过' if c.passed else '失败'}"
        if blocker:
            parts = blocker["detail"].splitlines()
            diag_output = parts[0]
            diag_attribution = "；".join(parts[1:]) or "等待人工处理"
        if not diag_attribution and t.accepted_risk_reason:
            diag_attribution = f"未验证 · 人工接受：{t.accepted_risk_reason}"

        if not diag_output:
            if t.status == TaskStatus.ACCEPTED:
                diag_output = "全部检查通过"
            elif blueprint_completed and t.status == TaskStatus.WAITING:
                diag_output = "已冻结（未启动代码执行）"
            elif t.status == TaskStatus.READY or (
                t.status == TaskStatus.WAITING and not unmet_deps
            ):
                diag_output = "就绪待调度"
            elif t.status == TaskStatus.RUNNING:
                diag_output = "正在执行"
            elif t.status == TaskStatus.VERIFYING:
                diag_output = "正在验收"
            else:
                diag_output = "等待执行"

        if not diag_attribution:
            if t.status == TaskStatus.ACCEPTED:
                diag_attribution = "Argus 冷启动独立复核通过"
            elif blueprint_completed and t.status == TaskStatus.WAITING:
                diag_attribution = (
                    f"{workflow_name} 止步于计划阶段，可点击上方「转入沙盒执行」开始写代码"
                )
            elif t.status == TaskStatus.READY or (
                t.status == TaskStatus.WAITING and not unmet_deps
            ):
                diag_attribution = "无阻塞前置依赖，等待调度器认领执行"
            elif t.status in (TaskStatus.RUNNING, TaskStatus.VERIFYING):
                diag_attribution = "调度器已认领任务"
            elif unmet_deps:
                diag_attribution = f"等待前置产物解锁（{', '.join(unmet_deps)}）"
            else:
                diag_attribution = "等待前置产物解锁"

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
                "design_refs": t.design_refs,
                "role": t.role,
                "kind": t.kind.value,
                "deliverables": t.deliverables,
                "depends_on": t.depends_on,
                "write_scope": t.write_scope,
                "read_scope": t.read_scope,
                "last_error": t.last_error,
                "provider_retry": provider_retry,
                "provider_retry_history": retry_history,
                "live_progress": live_progress[t.task_id],
                "blocked_by": dependency_blockers,
                "blocker": blocker,
                "accepted_risk_reason": t.accepted_risk_reason,
                "executor": "Hephaestus-1"
                if t.task_id.endswith(("1", "2", "3"))
                else "Hephaestus-2",
                "diagnostic_tool": diag_tool or "pytest -q",
                "diagnostic_output": diag_output,
                "diagnostic_attribution": diag_attribution,
                "acceptance_checks": [c.model_dump(mode="json") for c in t.acceptance_checks],
                "subagent_stream": stream,
                "terminal_lines": _build_terminal_lines(
                    t,
                    art,
                    blueprint_completed=blueprint_completed,
                    unmet_deps=unmet_deps,
                ),
                "receipt": receipt,
                "manifest_md": manifest_md,
                "artifacts": art,
            }
        )
        for dep in t.depends_on:
            dag_edges.append(
                {
                    "from": dep,
                    "to": t.task_id,
                    "critical": (dep, t.task_id) in crit_edges,
                    "blocked_edge": bool(dependency_blockers and blocked_by.get(dep)),
                }
            )

    ledger: BudgetLedger = snap["budget_ledger"]
    policy: BudgetPolicy = snap["budget_policy"]
    hitl_banner: dict[str, Any] | None = None
    if task_blockers:
        first_blocker = next(iter(task_blockers.values()))
        hitl_banner = {
            **first_blocker,
            "kind": "task_blocked",
            "tone": "danger",
        }
    elif snap["status"] == RunStatus.FAILED:
        f_node = workflow_view.get("failed_node") or "workflow"
        err_msg = workflow_view.get("last_error") or "模型调用超时或执行异常中断"
        hitl_banner = {
            "kind": "run_failed",
            "tone": "danger",
            "task_id": None,
            "failed_node": f_node,
            "headline": f"节点 {f_node} 执行中断，可从当前断点直接重试",
            "detail": err_msg,
            "actions": ["retry_workflow"],
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
    elif design is not None and not design_approved:
        hitl_banner = {
            "kind": "design_gate",
            "tone": "warn",
            "task_id": None,
            "headline": (
                f"关口 2 待批准：DesignSpec {design.design_id} "
                f"（{len(design.modules)} 个模块）等待确认后再分解任务"
            ),
            "detail": design.architecture_summary,
            "actions": ["approve_design"],
        }
    elif plan is not None and not plan_approved:
        hitl_banner = {
            "kind": "plan_gate",
            "tone": "warn",
            "task_id": None,
            "headline": (
                f"关口 3 待批准：任务计划 {plan.plan_id}（共 {len(tasks)} 个任务）等待批准开工"
            ),
            "detail": (
                f"绑定需求指纹：{(plan.spec_hash or plan.design_hash or '')[:16]}…"
            ),
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
    elif blueprint_completed:
        hitl_banner = {
            "kind": "blueprint_completed",
            "tone": "ok",
            "task_id": None,
            "headline": f"产品蓝图已冻结完成：共 {len(tasks)} 个模块任务（止步于计划阶段）",
            "detail": (
                f"工作流 {workflow_name} 不含代码执行阶段；"
                "如需按此蓝图在隔离 git worktree 沙盒中编写代码并独立验收，"
                "可一键转入 full_sdlc_tdd。"
            ),
            "actions": ["promote_to_execute"],
        }
    elif (
        snap["status"] == RunStatus.AWAITING_HUMAN
        and plan is not None
        and plan_approved
        and not blueprint_completed
    ):
        if not ledger.can_reserve(policy):
            hitl_banner = {
                "kind": "budget_exhausted",
                "tone": "warn",
                "task_id": None,
                "headline": "已达到本轮显式预算上限，任务派发已暂停",
                "detail": "当前运行已耗尽显式配置的调用或 Token 上限，未继续消耗任务重试次数。",
                "actions": [],
            }
        elif compute_dispatchable_tasks(plan, max_concurrent=1):
            hitl_banner = {
                "kind": "ready_to_resume",
                "tone": "info",
                "task_id": None,
                "headline": "存在就绪任务待继续执行",
                "detail": "点击「从断点重试」即可继续调度剩余就绪任务。",
                "actions": ["retry_workflow"],
            }

    accepted_count = sum(1 for t in tasks if t.status == TaskStatus.ACCEPTED)
    verified_count = sum(
        1 for t in tasks if t.status == TaskStatus.ACCEPTED and not t.accepted_risk_reason
    )
    raw_input = spec.goal if spec is not None else str(snap["title"])
    triage_payload: dict[str, Any] | None = None
    stage_activity: list[dict[str, Any]] = []
    stage_event_types = {
        "intake.triaged",
        "spec.drafted",
        "spec.clarified",
        "spec.confirmed",
        "design.generated",
        "design.approved",
        "plan.generated",
        "plan.approved",
        "run.workflow_promoted",
        "graph.failed",
    }
    for event in events:
        if event.event_type == "graph.input_recorded":
            recorded = str(event.payload.get("raw_input") or "").strip()
            if recorded:
                raw_input = recorded
        elif event.event_type == "intake.triaged":
            triage_payload = dict(event.payload)
        if event.event_type in stage_event_types or (
            event.event_type == "turn.receipt" and event.payload.get("task_id") is None
        ):
            stage_activity.append(
                {
                    "event_seq": event.event_seq,
                    "stage": event.stage,
                    "event_type": event.event_type,
                    "actor": event.actor,
                    "summary": str(
                        event.payload.get("one_line")
                        or event.payload.get("error")
                        or event.event_type
                    ),
                }
            )

    recent_progress, key_outputs = _build_run_progress(snap, events, dag_nodes, run_dir)
    return {
        "run_id": snap["run_id"],
        "title": snap["title"],
        "raw_input": raw_input,
        "workspace_root": snap.get("workspace_root") or str(Path(workspace_root).resolve()),
        "mode": snap["mode"].value,
        "status": snap["status"].value,
        "state_revision": snap["state_revision"],
        "created_at": snap["created_at"],
        "updated_at": snap["updated_at"],
        "workflow_name": workflow_name,
        "workflow": workflow_view,
        "blueprint_completed": blueprint_completed,
        "triage": triage_payload,
        "stage_activity": stage_activity[-12:],
        "recent_progress": recent_progress,
        "key_outputs": key_outputs,
        "spec_approved": spec_approved,
        "design_approved": design_approved,
        "plan_approved": plan_approved,
        "stage_pipeline": _build_stage_pipeline(
            snap,
            spec_approved,
            plan_approved,
            design_approved,
            stop_after=workflow_view.get("stop_after"),
        ),
        "budget": {
            "calls_used": ledger.llm_calls_used,
            "calls_reserved": ledger.llm_calls_reserved,
            "max_calls": policy.max_llm_calls,
            "input_tokens": ledger.input_tokens,
            "output_tokens": ledger.output_tokens,
            "cache_read_tokens": ledger.cache_read_tokens,
            "cache_write_tokens": ledger.cache_write_tokens,
            "total_tokens": ledger.total_tokens,
            "max_tokens": policy.max_total_tokens,
            "unknown_outcome_calls": ledger.unknown_outcome_calls,
        },
        "progress": {
            "accepted_tasks": accepted_count,
            "verified_tasks": verified_count,
            "total_tasks": len(tasks),
            "ratio_text": f"{accepted_count} / {len(tasks)}",
        },
        "hitl_banner": hitl_banner,
        "approval_reviews": approval_reviews,
        "spec": spec.model_dump(mode="json") if spec else None,
        "design": design.model_dump(mode="json") if design else None,
        "plan": plan.model_dump(mode="json") if plan else None,
        "dag": {
            "nodes": dag_nodes,
            "edges": dag_edges,
        },
        "swimlanes": _build_swimlanes(tasks, events),
        "waterfall_turns": _build_waterfall_turns(
            spec,
            tasks,
            {node["task_id"]: node for node in dag_nodes},
            receipt_by_actor,
            spec_approved=spec_approved,
        ),
        "traceability_matrix": _build_traceability_matrix(
            spec, tasks, design, delivery_acceptance=delivery_acceptance
        ),
        "delivery_acceptance": delivery_acceptance,
        "events": [e.model_dump(mode="json") for e in events[-80:]],
        "latest_event_seq": events[-1].event_seq if events else 0,
    }

# Backward-compatible re-export: demo seeding historically lived in this module.
from olympus.web.demos import seed_demo_runs  # noqa: E402, F401 - lazy bottom re-export avoids circular import with demos.py
