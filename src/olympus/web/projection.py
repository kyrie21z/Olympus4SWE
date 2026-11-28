"""Deterministic event-sourced projection and demo seeding for the Ariadne Web Console."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from pydantic import ValidationError

import olympus.web.acceptance as delivery_review
from olympus.core.contracts import (
    ApprovalTarget,
    BudgetLedger,
    BudgetPolicy,
    DesignSpec,
    GoalSpec,
    ProviderTimeoutRetry,
    ReviewVerdict,
    RunEvent,
    RunStatus,
    TaskNode,
    TaskPlan,
    TaskStatus,
)
from olympus.core.dag import compute_dispatchable_tasks
from olympus.core.store import NotFoundError, RunStore
from olympus.graph import load_workflow
from olympus.graph.progress import (
    PANTHEON_SEATS,
    build_workflow_view,
    classify_run_board,
    is_blueprint_completed,
)
from olympus.harness.activity import tool_output_status
from olympus.harness.receipt import build_turn_receipt
from olympus.stages.align import clarification_note_states, has_pending_resynthesis
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
from olympus.web.schemas import TurnReceipt

# Log rules such as "===== test session starts =====" or "____ test_x ____" carry no signal.
_NOISE_LINE = re.compile(r"[=\-_*·~]{4,}(?:.*[=\-_*·~]{4,})?")

_TOOL_STATUSES = ("ok", "error", "denied")


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
    """Compact tool events for the dock, strictly from the recorded worker trace."""
    recorded = art.get("tool_calls") or []
    return [
        {
            "tool": str(call.get("tool") or "unknown"),
            "target": str(call.get("target") or ""),
            "status": str(call.get("status") or "ok"),
            "duration_ms": int(call.get("duration_ms") or 0),
        }
        for call in recorded[:8]
    ]


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


def _safe_turn_receipt(payload: Any) -> dict[str, Any] | None:
    """Validate one recorded ``turn.receipt`` payload; drop malformed legacy rows."""
    if not isinstance(payload, Mapping):
        return None
    try:
        return TurnReceipt.model_validate(payload).model_dump(mode="json")
    except ValidationError:
        return None


def _receipt_tool_duration_ms(receipt: dict[str, Any] | None) -> int:
    """Sum real recorded tool durations from one turn receipt."""
    if not receipt:
        return 0
    tools = receipt.get("tools")
    if not isinstance(tools, list):
        return 0
    total = 0
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        duration = tool.get("duration_ms")
        if isinstance(duration, int) and not isinstance(duration, bool):
            total += max(0, duration)
    return total


def _receipt_tool_count(receipt: dict[str, Any] | None) -> int:
    if not receipt:
        return 0
    tools = receipt.get("tools")
    return len(tools) if isinstance(tools, list) else 0


def _stage_spans_ms(events: list[RunEvent]) -> dict[str, int]:
    """Real elapsed milliseconds per workflow node, from ISO event timestamps."""
    started: dict[str, datetime] = {}
    spans: dict[str, int] = {}
    for event in events:
        node = event.payload.get("node")
        if not isinstance(node, str):
            continue
        if event.event_type == "graph.node_started":
            started[node] = event.created_at
        elif event.event_type == "graph.node_finished":
            began = started.pop(node, None)
            if began is not None:
                elapsed = (event.created_at - began).total_seconds() * 1000
                spans[node] = max(0, int(elapsed))
    return spans


def _build_waterfall_turns(
    spec: GoalSpec | None,
    tasks: list[TaskNode],
    nodes_by_id: dict[str, dict[str, Any]],
    receipt_by_task: dict[str, dict[str, Any]],
    events: list[RunEvent],
    *,
    spec_approved: bool,
    post_dag: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Flatten real recorded turns into one chronological list.

    Every duration comes from a recorded ``turn.receipt`` tool trace or from
    ``graph.node_started`` -> ``graph.node_finished`` timestamps; nothing is
    synthesized. Offsets and widths scale by each turn's share of the real
    total duration, or fall back to even slots when no duration is recorded.
    """
    turns: list[dict[str, Any]] = []
    ordered = sorted(tasks, key=lambda item: (nodes_by_id[item.task_id]["layer"], item.task_id))
    executed = [task for task in ordered if task.attempt > 0]
    spans = _stage_spans_ms(events)

    receipts_by_actor: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.event_type == "turn.receipt":
            validated = _safe_turn_receipt(event.payload)
            if validated is not None:
                receipts_by_actor[event.actor] = validated

    def _actor_receipt(*prefixes: str) -> dict[str, Any] | None:
        found: dict[str, Any] | None = None
        for actor, payload in receipts_by_actor.items():
            if actor.split(":", 1)[0].strip().lower() in prefixes:
                found = payload
        return found

    def _add(
        *,
        turn_id: str,
        actor: str,
        task_id: str | None,
        title: str,
        status: str,
        summary: str,
        duration_ms: int,
        steps: int,
        receipt: dict[str, Any] | None,
        manifest_md: str,
    ) -> None:
        turns.append(
            {
                "turn_id": turn_id,
                "actor": actor,
                "task_id": task_id,
                "title": title,
                "status": status,
                "summary": summary,
                "duration_ms": max(0, duration_ms),
                "steps": max(0, steps),
                "offset_pct": 0.0,
                "width_pct": 0.0,
                "receipt": receipt,
                "manifest_md": manifest_md,
            }
        )

    align_receipt = _actor_receipt("align")
    if spec is not None or align_receipt is not None:
        align_duration = (
            _receipt_tool_duration_ms(align_receipt)
            if align_receipt is not None
            else spans.get("align", 0)
        )
        align_manifest = (
            "\n".join(
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
            if spec is not None
            else "# MANIFEST — Align"
        )
        _add(
            turn_id=str((align_receipt or {}).get("turn_id") or "turn-align-1"),
            actor="Align",
            task_id=None,
            title=f"需求对齐 · {spec.spec_id}" if spec is not None else "需求对齐",
            status=("accepted" if spec_approved else "running") if spec is not None else "accepted",
            summary=spec.goal
            if spec is not None
            else str((align_receipt or {}).get("one_line") or ""),
            duration_ms=align_duration,
            steps=_receipt_tool_count(align_receipt) or (len(spec.requirements) if spec else 1),
            receipt=align_receipt,
            manifest_md=align_manifest,
        )

    for prefix, actor, title, node_name in (
        ("design", "Athena", "白盒架构设计", "design"),
        ("orchestrate", "Orchestrate", "任务分解与计划", "plan"),
    ):
        receipt = _actor_receipt(prefix)
        if receipt is None and node_name not in spans:
            continue
        duration = _receipt_tool_duration_ms(receipt) or spans.get(node_name, 0)
        _add(
            turn_id=str((receipt or {}).get("turn_id") or f"turn-{prefix}"),
            actor=actor,
            task_id=None,
            title=title,
            status="accepted",
            summary=str((receipt or {}).get("one_line") or ""),
            duration_ms=duration,
            steps=_receipt_tool_count(receipt),
            receipt=receipt,
            manifest_md=f"# MANIFEST — {actor}",
        )

    for task in executed:
        node = nodes_by_id[task.task_id]
        receipt = receipt_by_task.get(task.task_id)
        stream = node["subagent_stream"]
        trace_duration = sum(int(step["duration_ms"]) for step in stream)
        duration = _receipt_tool_duration_ms(receipt) if receipt is not None else trace_duration
        error_lines = [
            line.strip() for line in (task.last_error or "").splitlines() if line.strip()
        ]
        _add(
            turn_id=str((receipt or {}).get("turn_id") or f"turn-{task.task_id.lower()}"),
            actor=str(node["executor"]),
            task_id=task.task_id,
            title=f"{task.task_id} · {task.title}",
            status=task.status.value,
            summary=(
                str(receipt.get("one_line"))
                if receipt is not None and receipt.get("one_line")
                else (error_lines[0] if error_lines else task.objective)
            ),
            duration_ms=duration,
            steps=_receipt_tool_count(receipt) if receipt is not None else len(stream),
            receipt=receipt,
            manifest_md=str(node["manifest_md"]),
        )

    momus_receipt = _actor_receipt("momus")
    if momus_receipt is not None:
        reviewed = next((task for task in ordered if task.review_cycle > 0), None)
        reviewed_id = reviewed.task_id if reviewed is not None else None
        _add(
            turn_id=str(momus_receipt.get("turn_id") or "turn-momus-1"),
            actor="Momus",
            task_id=reviewed_id,
            title=f"对抗式审查 · {reviewed_id}" if reviewed_id else "对抗式审查",
            status="accepted",
            summary=str(momus_receipt.get("one_line") or ""),
            duration_ms=_receipt_tool_duration_ms(momus_receipt) or spans.get("critic", 0),
            steps=_receipt_tool_count(momus_receipt),
            receipt=momus_receipt,
            manifest_md="\n".join(
                [
                    f"# MANIFEST — Momus ({reviewed_id or 'delivery'})",
                    "",
                    "- Perspective: QA / adversarial review",
                    "",
                    (reviewed.last_error or "") if reviewed is not None else "",
                    "",
                ]
            ),
        )

    post_dag_events = [
        event
        for event in events
        if event.event_type.startswith("post_dag.")
        or event.event_type == "product.verified"
        or (
            event.event_type == "review.verdict_recorded"
            and event.payload.get("task_id") == "post-dag"
        )
    ]
    argus_receipt = _actor_receipt("argus")
    if argus_receipt is not None or post_dag_events:
        if post_dag_events:
            elapsed = (
                post_dag_events[-1].created_at - post_dag_events[0].created_at
            ).total_seconds() * 1000
            duration = _receipt_tool_duration_ms(argus_receipt) or max(0, int(elapsed))
            steps = _receipt_tool_count(argus_receipt) or len(post_dag_events)
            title = "交付级独立复核 · Post-DAG"
            pd_status = post_dag["status"] if post_dag is not None else "verifying"
            latest_round_outcome = (
                post_dag["rounds"][-1]["outcome"]
                if post_dag is not None and post_dag["rounds"]
                else None
            )
            if pd_status == "passed":
                status = "accepted"
            elif pd_status == "paused":
                status = "paused"
            elif pd_status in ("failed", "awaiting_human") or latest_round_outcome in (
                "failed",
                "rejected",
            ):
                status = "blocked"
            else:
                status = "running"
        else:
            duration = _receipt_tool_duration_ms(argus_receipt)
            steps = _receipt_tool_count(argus_receipt)
            title = "冷启动独立复核 · 全量验收"
            status = "accepted"
        _add(
            turn_id=str((argus_receipt or {}).get("turn_id") or "turn-argus-1"),
            actor="Argus",
            task_id="post-dag" if post_dag_events else None,
            title=title,
            status=status,
            summary=(
                str(argus_receipt.get("one_line"))
                if argus_receipt is not None and argus_receipt.get("one_line")
                else "交付级产品验证与独立复核"
            ),
            duration_ms=duration,
            steps=steps,
            receipt=argus_receipt,
            manifest_md="# MANIFEST — Argus\n",
        )

    total_ms = sum(int(turn["duration_ms"]) for turn in turns)
    if turns and total_ms > 0:
        cursor = 0.0
        for turn in turns:
            offset = min(92.0, round(cursor / total_ms * 100.0, 1))
            width = max(
                6.0, min(100.0 - offset, round(turn["duration_ms"] / total_ms * 100.0, 1))
            )
            turn["offset_pct"] = offset
            turn["width_pct"] = width
            cursor += int(turn["duration_ms"])
    elif turns:
        slot = round(100.0 / len(turns), 1)
        for index, turn in enumerate(turns):
            offset = min(92.0, round(index * slot, 1))
            turn["offset_pct"] = offset
            turn["width_pct"] = max(6.0, min(100.0 - offset, slot))

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
        "spec.clarification_note", "spec.resynthesized", "spec.resynthesis_failed",
        "run.revision_requested",
        "run.reopened", "task.ephemeral_artifacts_cleaned", "task.diff_gate_rejected",
        "task.escape_hatch_rejected", "task.review_deferred",
        "task.deferred_review_passed", "review.verdict_verified",
        "review.repro_rechecked", "review.repro_commands_cleared",
        "product.verified",
        "post_dag.repro_rechecked", "post_dag.repair_turn_completed",
        "post_dag.repair_applied", "post_dag.repair_rejected",
    }
    tasks = {node["task_id"]: node for node in dag_nodes}
    attempts: dict[str, tuple[int, int]] = {}
    last_publication: dict[str, str] = {}
    outputs: dict[str, dict[str, Any]] = {}
    recent = None
    for event in events:
        kind = event.event_type
        payload = event.payload
        raw_node = payload.get("node")
        raw_task_id = payload.get("task_id")
        raw_to = payload.get("to")
        entry = {
            "event_seq": event.event_seq,
            "created_at": event.created_at.isoformat(),
            "event_type": kind,
            "node": raw_node if isinstance(raw_node, str) else None,
            "task_id": raw_task_id if isinstance(raw_task_id, str) else None,
            "to": raw_to if isinstance(raw_to, str) else None,
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
        elif kind == "product.verified":
            verification = delivery_review.latest_product_verification([event], run_dir)
            if verification is None:
                continue
            passed = bool(verification["passed"])
            failure_count = len(verification["failures"])
            summary = (
                "产品端到端验证通过"
                if passed
                else f"产品端到端验证未通过（{failure_count} 项失败）"
            )
            output = {
                "key": f"verification:product:{event.event_seq}",
                "kind": "verification",
                "artifact_id": "product",
                "version": None,
                "current": False,
                "detail_available": False,
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
        else:
            continue
        entry["last_event_at"] = event.created_at.isoformat()

    for task in tasks:
        entry = progress[task.task_id]
        if task.status == TaskStatus.RUNNING:
            entry["driver_name"] = entry["driver_name"] or configured_driver
        else:
            entry["current_tool"] = None
    return progress


def _task_review_flags(
    tasks: list[TaskNode], events: list[RunEvent]
) -> tuple[dict[str, bool], dict[str, int], dict[str, list[str]]]:
    """Track per-task deferred-review state and bounded escape-hatch rejections."""
    by_id = {task.task_id: task for task in tasks}
    deferred: dict[str, bool] = {}
    deferred_seq: dict[str, int] = {}
    rejections: dict[str, int] = {}
    findings: dict[str, list[str]] = {}
    for event in events:
        task_id = event.payload.get("task_id")
        if not isinstance(task_id, str) or task_id not in by_id:
            continue
        kind = event.event_type
        if kind == "task.review_deferred":
            attempt = event.payload.get("attempt")
            if (
                isinstance(attempt, int)
                and not isinstance(attempt, bool)
                and attempt != by_id[task_id].attempt
            ):
                continue
            deferred[task_id] = True
            deferred_seq[task_id] = event.event_seq
        elif kind in ("task.deferred_review_passed", "task.leased"):
            if event.event_seq > deferred_seq.get(task_id, 0):
                deferred[task_id] = False
        elif kind == "task.escape_hatch_rejected":
            rejections[task_id] = rejections.get(task_id, 0) + 1
            raw = event.payload.get("violations")
            if not isinstance(raw, list):
                continue
            latest_findings: list[str] = []
            for item in raw:
                if len(latest_findings) >= 10:
                    break
                if not isinstance(item, dict):
                    continue
                rule = item.get("rule") if isinstance(item.get("rule"), str) else ""
                message = item.get("message") if isinstance(item.get("message"), str) else ""
                path = item.get("file_path") if isinstance(item.get("file_path"), str) else ""
                line = item.get("line")
                position = (
                    line if isinstance(line, int) and not isinstance(line, bool) else None
                )
                location = f"{path}:{position}" if path and position is not None else path
                prefix = f"{location} {rule}".strip()
                text = f"{prefix}: {message}".strip(": ").strip()
                if text:
                    latest_findings.append(text[:240])
            if latest_findings:
                findings[task_id] = latest_findings
    for task in tasks:
        if task.status not in (TaskStatus.ACCEPTED, TaskStatus.VERIFYING):
            deferred[task.task_id] = False
    return deferred, rejections, findings


def _post_dag_round_entry(event: RunEvent, sequence: int) -> dict[str, Any] | None:
    """Whitelist one post-DAG event into a round row (never pass raw payloads on)."""
    kind = event.event_type
    payload = event.payload
    raw_round = payload.get("repair_round")
    if raw_round is None and kind == "review.verdict_recorded":
        raw_round = payload.get("review_cycle")
    reported_round = (
        raw_round if isinstance(raw_round, int) and not isinstance(raw_round, bool) else 0
    )
    entry: dict[str, Any] = {
        "sequence": sequence,
        "reported_round": reported_round,
        "event_seq": event.event_seq,
        "created_at": event.created_at.isoformat(),
        "phase": "verify",
        "outcome": "completed",
        "summary": "",
        "changed_paths": [],
        "repro_checked": None,
        "repro_failed": None,
    }
    if kind == "product.verified":
        passed = payload.get("passed")
        if not isinstance(passed, bool):
            return None
        raw_summary = payload.get("summary")
        entry["phase"] = "verify"
        entry["outcome"] = "passed" if passed else "failed"
        entry["summary"] = (
            raw_summary
            if isinstance(raw_summary, str) and raw_summary
            else ("产品端到端验证通过" if passed else "产品端到端验证未通过")
        )
    elif kind == "post_dag.repro_rechecked":
        checked = payload.get("checked_count")
        failing = payload.get("still_failing_count")
        checked_val = (
            checked
            if isinstance(checked, int) and not isinstance(checked, bool) and checked >= 0
            else None
        )
        failing_val = (
            failing
            if isinstance(failing, int) and not isinstance(failing, bool) and failing >= 0
            else None
        )
        entry["phase"] = "recheck"
        entry["repro_checked"] = checked_val
        entry["repro_failed"] = failing_val
        entry["outcome"] = (
            "passed" if checked_val is not None and failing_val == 0 else "failed"
        )
        entry["summary"] = f"复检 {checked_val or 0} 条复现命令"
    elif kind == "post_dag.repair_turn_completed":
        completed = payload.get("completed")
        stop_reason = payload.get("stop_reason")
        entry["phase"] = "repair"
        entry["outcome"] = "completed" if completed is True else "failed"
        entry["summary"] = (
            f"修复轮次完成（{stop_reason}）"
            if isinstance(stop_reason, str) and stop_reason
            else ("修复轮次完成" if completed is True else "修复轮次未完成")
        )
    elif kind == "post_dag.repair_applied":
        raw_paths = payload.get("changed_paths")
        changed = (
            [item for item in raw_paths if isinstance(item, str)]
            if isinstance(raw_paths, list)
            else []
        )
        entry["phase"] = "repair"
        entry["outcome"] = "applied"
        entry["changed_paths"] = changed
        entry["summary"] = f"修复已应用（{len(changed)} 个路径变更）"
    elif kind == "post_dag.repair_rejected":
        reason = payload.get("reason")
        entry["phase"] = "repair"
        entry["outcome"] = "rejected"
        entry["summary"] = (
            str(reason) if isinstance(reason, str) and reason else "修复被确定性门禁拒绝"
        )
    elif kind == "review.verdict_recorded":
        if payload.get("task_id") != "post-dag":
            return None
        passed = payload.get("passed")
        entry["phase"] = "review"
        entry["outcome"] = "passed" if passed is True else "failed"
        entry["summary"] = f"交付级 Argus 复核{'通过' if passed is True else '未通过'}"
    else:
        return None
    return entry


def _latest_post_dag_verdict(
    events: list[RunEvent], run_dir: Path, *, artifacts_root: Path
) -> ReviewVerdict | None:
    latest: RunEvent | None = None
    for event in events:
        if (
            event.event_type == "review.verdict_recorded"
            and event.payload.get("task_id") == "post-dag"
        ):
            latest = event
    if latest is None:
        return None
    cycle = latest.payload.get("review_cycle")
    if not isinstance(cycle, int) or isinstance(cycle, bool) or cycle < 1:
        return None
    raw = delivery_review.open_safe_artifact_bytes(
        artifacts_root, run_dir.name, "artifacts", "post_dag", f"round-{cycle}", "verdict.json"
    )
    if raw is None:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            return None
        return ReviewVerdict.model_validate(data)
    except (UnicodeDecodeError, ValueError):
        return None


def _build_post_dag_summary(
    events: list[RunEvent], run_status: RunStatus, run_dir: Path, *, artifacts_root: Path
) -> dict[str, Any] | None:
    round_events = [
        event
        for event in events
        if event.event_type.startswith("post_dag.")
        or event.event_type == "product.verified"
        or (
            event.event_type == "review.verdict_recorded"
            and event.payload.get("task_id") == "post-dag"
        )
    ]
    has_deferred = any(
        event.event_type in ("task.review_deferred", "task.deferred_review_passed")
        for event in events
    )
    if not round_events and not has_deferred:
        return None
    rounds: list[dict[str, Any]] = []
    for event in round_events:
        entry = _post_dag_round_entry(event, len(rounds) + 1)
        if entry is not None:
            rounds.append(entry)
    latest = round_events[-1] if round_events else None
    latest_round = rounds[-1] if rounds else None
    latest_product = next(
        (event for event in reversed(round_events) if event.event_type == "product.verified"),
        None,
    )
    latest_repro = next(
        (
            row
            for row in reversed(rounds)
            if row["phase"] == "recheck"
        ),
        None,
    )
    latest_verdict = next(
        (
            event
            for event in reversed(round_events)
            if event.event_type == "review.verdict_recorded"
            and event.payload.get("task_id") == "post-dag"
        ),
        None,
    )
    product_passed = (
        latest_product.payload.get("passed") is True if latest_product is not None else None
    )
    repro_failed = bool(
        latest_repro is not None
        and latest_repro["outcome"] == "failed"
        and (
            latest_product is None
            or latest_repro["event_seq"] >= latest_product.event_seq
        )
    )
    verdict_passed = (
        latest_verdict.payload.get("passed") is True if latest_verdict is not None else None
    )
    if run_status == RunStatus.AWAITING_HUMAN:
        status = "awaiting_human"
    elif run_status == RunStatus.PAUSED:
        status = "paused"
    elif run_status == RunStatus.FAILED:
        status = "failed"
    elif latest is None:
        status = "idle"
    elif run_status == RunStatus.COMPLETED:
        status = (
            "passed"
            if (
                product_passed is not False
                and not repro_failed
                and verdict_passed is not False
            )
            else "failed"
        )
    elif (
        latest.event_type == "review.verdict_recorded"
        and verdict_passed is True
        and product_passed is True
        and not repro_failed
    ):
        status = "passed"
    elif latest.event_type in (
        "post_dag.repair_turn_completed",
        "post_dag.repair_rejected",
        "post_dag.repair_applied",
    ):
        status = "repairing"
    elif latest_round is not None:
        status = "verifying"
    else:
        status = "idle"
    return {
        "status": status,
        "total_rounds": len(rounds),
        "rounds": rounds,
        "latest_argus_verdict": _latest_post_dag_verdict(
            events, run_dir, artifacts_root=artifacts_root
        ),
    }


def build_run_projection(
    store: RunStore,
    workspace_root: Path,
    run_id: str,
    *,
    artifacts_root: Path | None = None,
    supervised: bool | None = None,
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
    resolved_artifacts_root = (
        artifacts_root
        if artifacts_root is not None
        else Path(workspace_root).resolve() / ".olympus" / "runs"
    )
    run_dir = resolved_artifacts_root / run_id

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
        artifacts_root=resolved_artifacts_root,
    )
    product_verification = delivery_review.latest_product_verification(
        events, run_dir, artifacts_root=resolved_artifacts_root
    )
    post_dag = _build_post_dag_summary(
        events, snap["status"], run_dir, artifacts_root=resolved_artifacts_root
    )

    receipt_by_task: dict[str, dict[str, Any]] = {}
    retries_by_task: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        if ev.event_type == ProviderTimeoutRetry.event_type:
            retry = ProviderTimeoutRetry.model_validate(ev.payload)
            retries_by_task.setdefault(retry.task_id, []).append(
                {**retry.model_dump(mode="json"), "event_seq": ev.event_seq, "actor": ev.actor}
            )
        if ev.event_type != "turn.receipt":
            continue
        validated_receipt = _safe_turn_receipt(ev.payload)
        if validated_receipt is None:
            continue
        recorded_task = validated_receipt.get("task_id")
        actor_prefix = ev.actor.split(":", 1)[0].strip().lower()
        if (
            isinstance(recorded_task, str)
            and recorded_task
            and actor_prefix not in ("momus", "argus")
        ):
            receipt_by_task[recorded_task] = validated_receipt

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

    is_supervised = (
        any(ev.stage == "graph" for ev in events) if supervised is None else supervised
    )
    try:
        strict_canonical = load_workflow(workflow_name).align_strict_canonical()
    except Exception:  # noqa: BLE001 - fallback for legacy/unknown workflows
        strict_canonical = True
    pending_resynthesis = is_supervised and has_pending_resynthesis(
        events,
        spec,
        spec_approved=spec_approved,
        strict_canonical=strict_canonical,
    )
    clarification_notes: list[dict[str, Any]] = []
    resynthesis_error: str | None = None
    if spec is not None:
        clarification_notes = [
            {
                "event_seq": event.event_seq,
                "created_at": event.created_at.isoformat(),
                "feedback": str(event.payload.get("feedback") or "").strip(),
                "consumed": consumed,
            }
            for event, consumed in clarification_note_states(events, spec_id=spec.spec_id)
        ]
        for event in reversed(events):
            if event.event_type == "spec.resynthesis_failed":
                if event.payload.get("spec_id") in (None, "", spec.spec_id):
                    resynthesis_error = str(event.payload.get("error") or "").strip() or None
                    break
                continue
            if event.event_type in ("spec.resynthesized", "spec.saved") and (
                event.payload.get("spec_id") in (None, "", spec.spec_id)
            ):
                break
    else:
        last_craft_progress_seq = max(
            (
                event.event_seq
                for event in events
                if event.event_type
                in (
                    "craft.evidence_recorded",
                    "craft.architect_recorded",
                    "craft.build_completed",
                )
            ),
            default=0,
        )
        clarification_notes = [
            {
                "event_seq": event.event_seq,
                "created_at": event.created_at.isoformat(),
                "feedback": str(event.payload.get("note") or "").strip(),
                "consumed": event.event_seq < last_craft_progress_seq,
            }
            for event in events
            if event.event_type == "craft.clarification_recorded"
            and str(event.payload.get("note") or "").strip()
        ]

    live_progress = _live_task_progress(tasks, events)
    review_deferred, escape_rejections, escape_findings = _task_review_flags(tasks, events)
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
                "review_deferred": review_deferred.get(t.task_id, False),
                "escape_hatch_rejections": escape_rejections.get(t.task_id, 0),
                "escape_hatch_findings": escape_findings.get(t.task_id, []),
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
    elif snap["status"] == RunStatus.PAUSED:
        hitl_banner = {
            "kind": "run_paused",
            "tone": "info",
            "task_id": None,
            "headline": "运行已暂停：当前停止派发新任务，等待人工恢复",
            "detail": "点击右上角「恢复运行」继续调度就绪任务",
            "actions": ["resume"],
        }
    elif pending_resynthesis and resynthesis_error:
        hitl_banner = {
            "kind": "resynthesis_failed",
            "tone": "danger",
            "task_id": None,
            "headline": "需求融合（Resynthesize）失败，可原地重试或补充说明",
            "detail": resynthesis_error,
            "actions": ["retry_workflow"],
        }
    elif pending_resynthesis:
        hitl_banner = {
            "kind": "resynthesizing",
            "tone": "info",
            "task_id": None,
            "headline": "正在将澄清与补充约束融合进 GoalSpec…",
            "detail": "融合完成后将生成新版本供您确认，期间暂不可批准旧版规格。",
            "actions": [],
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
        elif tasks and all(
            task.status in (TaskStatus.ACCEPTED, TaskStatus.CANCELLED) for task in tasks
        ):
            product_failed = (
                product_verification is not None and not product_verification["passed"]
            )
            post_dag_failed = bool(
                post_dag is not None
                and post_dag["rounds"]
                and post_dag["rounds"][-1]["outcome"] in ("failed", "rejected")
            )
            latest_status = next(
                (
                    event
                    for event in reversed(events)
                    if event.event_type == "run.status_changed"
                    and event.payload.get("to") == RunStatus.AWAITING_HUMAN.value
                ),
                None,
            )
            status_failure = ""
            status_reason = ""
            if latest_status is not None:
                raw_failure = latest_status.payload.get("failure")
                if isinstance(raw_failure, str) and raw_failure.strip():
                    status_failure = raw_failure.strip()
                raw_reason = latest_status.payload.get("reason") or latest_status.payload.get(
                    "detail"
                )
                if isinstance(raw_reason, str) and raw_reason.strip():
                    status_reason = raw_reason.strip()
            blocking_status = bool(status_failure) or status_reason in (
                "post_dag_verification_failed",
                "review_output_invalid",
                "turn_cancelled",
            )
            failed = product_failed or post_dag_failed or blocking_status
            pieces: list[str] = []
            if status_failure:
                pieces.append(status_failure)
            elif blocking_status and status_reason:
                pieces.append(status_reason)
            if product_verification is not None:
                if not blocking_status and product_verification["summary"]:
                    pieces.append(product_verification["summary"])
                pieces.extend(product_verification["failures"][:2])
            if post_dag is not None and post_dag["rounds"]:
                latest_round = post_dag["rounds"][-1]
                if not blocking_status or latest_round["outcome"] in ("failed", "rejected"):
                    pieces.append(latest_round["summary"])
            if not pieces:
                pieces.append(status_reason or "全部任务已验收，等待人工确认交付验收结果")
            deduped_pieces = list(dict.fromkeys(piece for piece in pieces if piece))
            hitl_banner = {
                "kind": "review_delivery",
                "tone": "danger" if failed else "warn",
                "task_id": None,
                "headline": (
                    "交付验收未通过：产品端到端验证或交付级复核仍需人工决策"
                    if failed
                    else "交付验收待人工确认：全部任务已验收，等待放行交付"
                ),
                "detail": "；".join(deduped_pieces),
                "actions": ["retry_workflow"],
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
    for event in events:
        if event.event_type == "graph.input_recorded":
            recorded = str(event.payload.get("raw_input") or "").strip()
            if recorded:
                raw_input = recorded
        elif event.event_type == "intake.triaged":
            triage_payload = dict(event.payload)

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
            "reasoning_tokens": ledger.reasoning_tokens,
            "cache_read_tokens": ledger.cache_read_tokens,
            "cache_write_tokens": ledger.cache_write_tokens,
            "cache_hit_rate": ledger.cache_hit_rate,
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
        "pending_resynthesis": pending_resynthesis,
        "resynthesis_error": resynthesis_error,
        "clarification_notes": clarification_notes,
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
            receipt_by_task,
            events,
            spec_approved=spec_approved,
            post_dag=post_dag,
        ),
        "traceability_matrix": _build_traceability_matrix(
            spec, tasks, design, delivery_acceptance=delivery_acceptance
        ),
        "delivery_acceptance": delivery_acceptance,
        "post_dag": post_dag,
        "events": [e.model_dump(mode="json") for e in events[-80:]],
        "latest_event_seq": events[-1].event_seq if events else 0,
    }

def _resolve_workspace_meta(
    store: RunStore,
    snap: Mapping[str, Any],
    workspaces: Mapping[str, Mapping[str, Any]] | None,
    workspaces_by_root: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[str | None, str]:
    workspace_id = snap.get("workspace_id")
    if isinstance(workspace_id, str) and workspace_id:
        workspace = workspaces.get(workspace_id) if workspaces is not None else None
        if workspace is None:
            workspace = store.get_workspace(workspace_id)
        if workspace is not None:
            return workspace_id, str(workspace["name"])
    root = snap.get("workspace_root")
    if isinstance(root, str) and root.strip():
        canonical = store.canonical_root(root)
        matched = (
            workspaces_by_root.get(canonical)
            if workspaces_by_root is not None
            else store.get_workspace_by_root(canonical)
        )
        if matched is not None:
            return str(matched["workspace_id"]), str(matched["name"])
        return None, Path(root).name or "Workspace"
    return None, "Unassigned"


def build_run_summary(
    store: RunStore,
    run_id: str,
    *,
    workspaces: Mapping[str, Mapping[str, Any]] | None = None,
    workspaces_by_root: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Derive one Oracle Board card from the canonical deterministic classifier."""
    snap = store.get_run(run_id, include_approval_baselines=True)
    events = store.list_events(run_id, after_seq=0)
    approved = snap["artifact_approved"]
    workflow_view = build_workflow_view(
        str(snap.get("workflow_name") or "product_blueprint"),
        events,
        snap["status"],
        spec_approved=approved[ApprovalTarget.GOAL_SPEC],
        design_approved=approved[ApprovalTarget.DESIGN_SPEC],
        plan_approved=approved[ApprovalTarget.TASK_PLAN],
    )
    state = classify_run_board(snap, events, workflow_view=workflow_view)
    resolved_ws_id, resolved_ws_name = _resolve_workspace_meta(
        store, snap, workspaces, workspaces_by_root
    )
    return {
        "run_id": str(snap["run_id"]),
        "title": str(snap["title"]),
        "mode": snap["mode"].value,
        "status": snap["status"].value,
        "state_revision": int(snap["state_revision"]),
        "workflow_name": str(snap.get("workflow_name") or "product_blueprint"),
        "created_at": str(snap["created_at"]),
        "updated_at": str(snap["updated_at"]),
        "workspace_id": resolved_ws_id,
        "workspace_name": resolved_ws_name,
        "workspace_mode": str(snap.get("workspace_mode") or "brownfield"),
        "task_type": str(snap.get("task_type") or "feature"),
        "ball_holder": state.ball_holder,
        "active_god": state.active_god,
        "pending_action": state.pending_action,
        "worktree_branch": state.worktree_branch,
        "status_badge": state.status_badge,
        "is_archived": state.is_archived,
    }


def aggregate_pantheon(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seats: list[dict[str, Any]] = []
    for god, role_label in PANTHEON_SEATS:
        held = [
            run
            for run in runs
            if run["active_god"] == god and run["ball_holder"] in ("agent", "human")
        ]
        active_count = sum(1 for run in held if run["ball_holder"] == "agent")
        waiting_count = sum(1 for run in held if run["ball_holder"] == "human")
        seats.append(
            {
                "god": god,
                "role_label": role_label,
                "state": (
                    "waiting_on_human"
                    if waiting_count > 0
                    else ("working" if active_count > 0 else "idle")
                ),
                "active_count": active_count,
                "waiting_count": waiting_count,
                "oracle_ids": [str(run["run_id"]) for run in held],
            }
        )
    return seats


def build_board_payload(
    store: RunStore,
    *,
    limit: int | None = 50,
    workspace_id: str | None = None,
    workspace_mode: str | None = None,
    task_type: str | None = None,
    ball_holder: str | None = None,
    include_archived: bool = False,
) -> dict[str, Any]:
    workspace_list = store.list_workspaces()
    workspaces = {str(item["workspace_id"]): item for item in workspace_list}
    workspaces_by_root = {str(item["root_path"]): item for item in workspace_list}
    rows = store.list_runs(
        limit=None,
        workspace_id=workspace_id,
        workspace_mode=workspace_mode,
        task_type=task_type,
        include_archived=include_archived,
    )
    summaries: list[dict[str, Any]] = []
    for row in rows:
        try:
            summaries.append(
                build_run_summary(
                    store,
                    str(row["run_id"]),
                    workspaces=workspaces,
                    workspaces_by_root=workspaces_by_root,
                )
            )
        except NotFoundError:
            continue
    pending_ruling_total = sum(
        1 for summary in summaries if summary["ball_holder"] == "human"
    )
    runs = [
        summary
        for summary in summaries
        if ball_holder is None or summary["ball_holder"] == ball_holder
    ]
    if limit is not None:
        cap = max(limit, 0)
        if ball_holder is None and cap > 0:
            human_runs = [item for item in runs if item["ball_holder"] == "human"]
            other_runs = [item for item in runs if item["ball_holder"] != "human"]
            remaining = max(cap - len(human_runs), 0)
            runs = [*human_runs, *other_runs[:remaining]]
        else:
            runs = runs[:cap]
    return {
        "ok": True,
        "runs": runs,
        "default_run_id": runs[0]["run_id"] if runs else None,
        "pantheon": aggregate_pantheon(summaries),
        "pending_ruling_total": pending_ruling_total,
        "active_workspace_id": workspace_id,
    }


def build_workspace_counts(store: RunStore) -> dict[str, dict[str, int]]:
    """Count active/waiting Oracles per registered workspace via the same classifier."""
    counts = {
        str(workspace["workspace_id"]): {
            "active_oracle_count": 0,
            "pending_ruling_count": 0,
        }
        for workspace in store.list_workspaces()
    }
    for run in build_board_payload(store, limit=None, include_archived=False)["runs"]:
        workspace_id = run["workspace_id"]
        if not isinstance(workspace_id, str) or workspace_id not in counts:
            continue
        if run["ball_holder"] == "agent":
            counts[workspace_id]["active_oracle_count"] += 1
        elif run["ball_holder"] == "human":
            counts[workspace_id]["pending_ruling_count"] += 1
    return counts


# Backward-compatible re-export: demo seeding historically lived in this module.
from olympus.web.demos import seed_demo_runs  # noqa: E402, F401 - lazy bottom re-export avoids circular import with demos.py
