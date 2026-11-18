"""Deterministic event-sourced projection and demo seeding for the Ariadne Web Console."""

from __future__ import annotations

import json
import re
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
    DesignSpec,
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
from olympus.graph import build_graph, load_workflow
from olympus.harness.receipt import build_turn_receipt

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

    status = str(entry.get("status") or "")
    if status not in _TOOL_STATUSES:
        if entry.get("denied"):
            status = "denied"
        elif entry.get("is_error"):
            status = "error"
        else:
            status = "ok"
    target = entry.get("target") or entry.get("path") or entry.get("command") or ""
    return {
        "step": int(entry.get("step") or 0),
        "tool": str(entry.get("tool") or "unknown"),
        "target": str(target)[:160],
        "status": status,
        "is_error": status != "ok",
        "duration_ms": int(entry.get("duration_ms") or 0),
        "output_preview": str(entry.get("output") or "")[:240],
    }


def _read_task_artifacts(
    run_dir: Path,
    task: TaskNode,
) -> dict[str, Any]:
    """Load recorded diff, command check logs, review verdicts, manifest and worker trace."""
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

    manifest_md = ""
    manifest_path = art_dir / "MANIFEST.md"
    if manifest_path.exists():
        manifest_md = manifest_path.read_text(encoding="utf-8", errors="replace")[:12000]

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
            step = _parse_tool_trace_line(raw_line)
            if step is not None:
                tool_calls.append(step)

    return {
        "attempt_dir": str(art_dir) if art_dir.exists() else None,
        "diff_patch": diff_text,
        "check_logs": check_logs,
        "verdicts": verdicts,
        "tool_calls": tool_calls,
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


def _build_terminal_lines(task: TaskNode, art: dict[str, Any]) -> list[str]:
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
    return ["◷ 等待前置产物解锁"]


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
) -> list[dict[str, str]]:
    spec: GoalSpec | None = snap["spec"]
    plan: TaskPlan | None = snap["plan"]
    status: RunStatus = snap["status"]
    tasks = plan.tasks if plan else []

    align_state = "done" if spec_approved else ("active" if spec is not None else "pending")
    design_state = (
        "done"
        if design_approved or plan_approved
        else ("active" if spec_approved else "pending")
    )
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
    design: DesignSpec | None = None,
) -> list[dict[str, Any]]:
    """Build Requirement x DesignModule x Task x Verification Status rows."""
    if spec is None:
        return []

    rows: list[dict[str, Any]] = []
    for req in spec.requirements:
        covering = [t for t in tasks if req.id in t.requirement_refs]
        design_modules = [
            {"id": module.id, "name": module.name}
            for module in (design.modules if design is not None else [])
            if req.id in module.requirement_refs
        ]
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


def _build_workflow_view(
    workflow_name: str,
    events: list[RunEvent],
    run_status: RunStatus,
) -> dict[str, Any]:
    """Describe the run's workflow topology plus live per-node statuses from graph.* events."""
    topology: str | None = None
    stop_after: str | None = None
    description: dict[str, Any] | None = None
    node_names: list[str] = []
    try:
        definition = load_workflow(workflow_name)
        description = build_graph(definition).describe()
        topology = definition.topology
        stop_after = definition.stop_after
        node_names = [str(n["name"]) for n in description.get("nodes", [])]
    except Exception:  # noqa: BLE001 - a legacy/unknown workflow must not break the console
        description = None

    statuses: dict[str, str] = {name: "pending" for name in node_names}
    current_node: str | None = None
    paused: dict[str, Any] | None = None
    finished = False
    for event in events:
        payload = event.payload
        node = payload.get("node")
        if event.event_type == "graph.node_started" and isinstance(node, str):
            current_node = node
            if node in statuses:
                statuses[node] = "running"
        elif event.event_type == "graph.node_finished" and isinstance(node, str):
            if node in statuses:
                statuses[node] = "completed"
        elif event.event_type == "graph.paused":
            paused = {
                "node": node,
                "reason": payload.get("reason"),
            }
            current_node = None
            if isinstance(node, str) and node in statuses:
                statuses[node] = "paused"
        elif event.event_type == "graph.finished":
            current_node = None
            finished = True
        elif event.event_type == "graph.node_error" and isinstance(node, str) and node in statuses:
            statuses[node] = "error"

    return {
        "name": workflow_name,
        "topology": topology,
        "stop_after": stop_after,
        "graph": description,
        "nodes": [{"name": name, "status": status} for name, status in statuses.items()],
        "node_statuses": dict(statuses),
        "current_node": current_node,
        "paused": paused,
        "finished": finished or run_status == RunStatus.COMPLETED,
    }


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
    design: DesignSpec | None = snap.get("design")
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
    design_approved = bool(
        design
        and store.is_approved(
            run_id=run_id,
            target=ApprovalTarget.DESIGN_SPEC,
            target_id=design.design_id,
            target_hash=design.content_hash,
        )
    )

    tasks = plan.tasks if plan else []
    layers, crit_nodes, crit_edges = _compute_dag_layers_and_critical_path(tasks)
    task_by_id = {t.task_id: t for t in tasks}

    receipt_by_task: dict[str, dict[str, Any]] = {}
    receipt_by_actor: dict[str, dict[str, Any]] = {}
    for ev in events:
        if ev.event_type != "turn.receipt":
            continue
        payload = dict(ev.payload)
        recorded_task = payload.get("task_id")
        if isinstance(recorded_task, str) and recorded_task:
            receipt_by_task.setdefault(recorded_task, payload)
        receipt_by_actor.setdefault(ev.actor, payload)

    dag_nodes: list[dict[str, Any]] = []
    dag_edges: list[dict[str, Any]] = []
    for order, t in enumerate(tasks):
        art = _read_task_artifacts(run_dir, t)
        stream = _build_subagent_stream(t, art)
        receipt = receipt_by_task.get(t.task_id) or _build_receipt(t, stream, art, order=order)
        manifest_md = str(art.get("manifest_md") or "") or _synthesize_manifest_md(t, art, receipt)
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
                "design_refs": t.design_refs,
                "role": t.role,
                "kind": t.kind.value,
                "deliverables": t.deliverables,
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
                "subagent_stream": stream,
                "terminal_lines": _build_terminal_lines(t, art),
                "receipt": receipt,
                "manifest_md": manifest_md,
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

    accepted_count = sum(1 for t in tasks if t.status == TaskStatus.ACCEPTED)
    ledger: BudgetLedger = snap["budget_ledger"]
    policy: BudgetPolicy = snap["budget_policy"]
    triage_payload: dict[str, Any] | None = None
    for event in events:
        if event.event_type == "intake.triaged":
            triage_payload = dict(event.payload)

    return {
        "run_id": snap["run_id"],
        "title": snap["title"],
        "mode": snap["mode"].value,
        "status": snap["status"].value,
        "state_revision": snap["state_revision"],
        "created_at": snap["created_at"],
        "updated_at": snap["updated_at"],
        "workflow_name": snap.get("workflow_name", "product_blueprint"),
        "workflow": _build_workflow_view(
            str(snap.get("workflow_name") or "product_blueprint"),
            events,
            snap["status"],
        ),
        "triage": triage_payload,
        "spec_approved": spec_approved,
        "design_approved": design_approved,
        "plan_approved": plan_approved,
        "stage_pipeline": _build_stage_pipeline(
            snap, spec_approved, plan_approved, design_approved
        ),
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
        "traceability_matrix": _build_traceability_matrix(spec, tasks, design),
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
        t03_diff = (
            "--- a/src/waitlist/waitlist.py\n"
            "+++ b/src/waitlist/waitlist.py\n"
            "@@ -42,8 +42,14 @@ def promote_next_waitlisted(\n"
            "     conn: sqlite3.Connection, activity_id: str\n"
            " ) -> str | None:\n"
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
            "     return str(row[1])\n"
        )
        (t03_art / "diff.patch").write_text(t03_diff, encoding="utf-8")
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

        def _append_receipt(
            *,
            turn_id: str,
            actor: str,
            task_id: str | None,
            tools: list[tuple[str, bool, int]],
            calls: int,
            input_tokens: int,
            output_tokens: int,
            diff_lines: tuple[int, int] = (0, 0),
            checks_passed: int = 0,
            checks_total: int = 0,
        ) -> dict[str, Any]:
            turn_events: list[dict[str, Any]] = [
                {
                    "kind": "llm",
                    "calls": calls,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            ]
            for tool_name, is_error, duration_ms in tools:
                turn_events.append(
                    {
                        "kind": "tool_end",
                        "tool": tool_name,
                        "is_error": is_error,
                        "duration_ms": duration_ms,
                    }
                )
            receipt = build_turn_receipt(
                turn_id=turn_id,
                actor=actor,
                task_id=task_id,
                events=turn_events,
                diff_lines=diff_lines,
                checks_passed=checks_passed,
                checks_total=checks_total,
            )
            store.append_event(
                run_id=demo_id,
                stage="orchestrate",
                event_type="turn.receipt",
                actor=actor,
                payload=receipt,
            )
            return receipt

        # Real worker trace so the dock renders recorded tool steps, not synthetic ones
        worker_trace = [
            (
                1,
                "read_file",
                "src/waitlist/waitlist.py",
                "ok",
                384,
                "def promote_next_waitlisted(conn, activity_id): ...",
            ),
            (
                2,
                "read_file",
                "tests/test_waitlist.py",
                "ok",
                296,
                "def test_concurrent_cancel_promotes_once_only(): ...",
            ),
            (
                3,
                "edit_file",
                "src/waitlist/waitlist.py",
                "ok",
                1420,
                "patched promote_next_waitlisted: added status='WAITLISTED' filter",
            ),
            (
                4,
                "run_bash",
                "pytest tests/test_waitlist.py",
                "error",
                2210,
                "1 failed, 3 passed in 0.19s",
            ),
            (
                5,
                "read_file",
                "src/waitlist/models.py",
                "ok",
                212,
                "class RegistrationStatus(StrEnum): ...",
            ),
        ]
        (t03_art / "worker.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "event": "tool_call",
                        "step": step,
                        "tool": tool,
                        "target": target,
                        "status": status,
                        "duration_ms": duration,
                        "output": output,
                    },
                    ensure_ascii=False,
                )
                for step, tool, target, status, duration, output in worker_trace
            )
            + "\n",
            encoding="utf-8",
        )

        t03_receipt = _append_receipt(
            turn_id="turn-t03-r1-a2",
            actor="hephaestus:T03-R1",
            task_id="T03-R1",
            tools=[
                (tool, status != "ok", duration)
                for _, tool, _, status, duration, _ in worker_trace
            ],
            calls=3,
            input_tokens=12840,
            output_tokens=2410,
            diff_lines=_diff_line_stats(t03_diff),
            checks_passed=0,
            checks_total=2,
        )
        t03_added, t03_removed = _diff_line_stats(t03_diff)
        (t03_art / "MANIFEST.md").write_text(
            "\n".join(
                [
                    "# MANIFEST — T03-R1 (attempt 2)",
                    "",
                    "- Task: T03-R1 — 修复已取消候补者重复递补缺陷",
                    "- Outcome: checks_failed",
                    "- Task status: blocked",
                    "- Review cycle: 2",
                    "- Checks: 0/2 passed",
                    f"- Diff: +{t03_added}/-{t03_removed} lines",
                    f"- Turn: {t03_receipt['turn_id']}",
                    f"- Receipt: {t03_receipt['one_line']}",
                    "",
                    "## Objective",
                    "",
                    "确保递补查询在同一事务内过滤 CANCELLED 记录并原子更新被递补者状态。",
                    "",
                    "## Review verdict",
                    "",
                    "- passed: False",
                    (
                        "- summary: 递补查询虽过滤了 WAITLISTED，"
                        "但未在同一事务内将命中记录原子更新为 CONFIRMED。"
                    ),
                    "- findings: 1",
                    "",
                    "## Acceptance checks",
                    "",
                    "- T03-R1-cmd (command): not passed — 验证候补取消与连续递补边界用例",
                    "- T03-R1-rev (independent_review): not passed — Argus 审查事务边界",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        # Align turn receipt: the one non-worker actor in the waterfall
        _append_receipt(
            turn_id="turn-align-1",
            actor="align",
            task_id=None,
            tools=[("read_file", False, 412), ("read_file", False, 508)],
            calls=2,
            input_tokens=8400,
            output_tokens=2150,
        )
        # Worker turn receipts for every task that actually executed
        for order, task in enumerate(tasks):
            if task.attempt <= 0 or task.task_id == "T03-R1":
                continue
            synth_stream = _synthesize_tool_stream(task)
            if not synth_stream:
                continue
            _append_receipt(
                turn_id=f"turn-{task.task_id.lower()}-a{max(1, task.attempt)}",
                actor=f"hephaestus:{task.task_id}",
                task_id=task.task_id,
                tools=[
                    (str(step["tool"]), step["status"] != "ok", int(step["duration_ms"]))
                    for step in synth_stream
                ],
                calls=1,
                input_tokens=2480 + 140 * order,
                output_tokens=560 + 40 * order,
                diff_lines=(0, 0),
                checks_passed=sum(1 for c in task.acceptance_checks if c.passed),
                checks_total=len(task.acceptance_checks),
            )

        seeded.append(demo_id)

    return seeded
