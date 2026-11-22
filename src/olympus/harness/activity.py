"""Privacy boundary for durable milestones and the ephemeral activity preview."""

from __future__ import annotations

import re
from threading import Lock
from typing import Any, Literal

from olympus.core.fingerprint import content_hash

_TOOL_TARGETS = {
    "read_file": "workspace file",
    "write_file": "workspace file",
    "edit_file": "workspace file",
    "glob_files": "workspace search",
    "grep_files": "workspace search",
    "run_bash": "shell command",
    "read": "workspace file",
    "write": "workspace file",
    "edit": "workspace file",
    "find": "workspace search",
    "grep": "workspace search",
    "ls": "workspace search",
    "bash": "shell command",
}
_STEER_REASONS = {"human": "Operator guidance", "watchdog": "Watchdog guidance"}
_PREVIEW_LIMIT = 240


def tool_output_status(tool: str, output: str) -> Literal["ok", "error", "denied"]:
    if output.startswith("Error: [PRE_TOOL_HOOK DENIED]"):
        return "denied"
    if output.startswith(("Error:", "Error executing ")):
        return "error"
    if tool == "run_bash" and output.startswith("exit_code="):
        return "ok" if output.startswith("exit_code=0\n") else "error"
    return "ok"


def milestone_fields(kind: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Select safe milestone fields without copying arguments, output, or guidance."""
    if kind == "agent.steered":
        source = payload.get("source")
        if isinstance(source, str) and source in _STEER_REASONS:
            return {"source": source, "reason": _STEER_REASONS[source]}
        return None
    if kind == "agent.compaction":
        phase = payload.get("phase")
        if phase in ("start", "end"):
            return {"phase": phase}
        return None
    if kind not in ("agent.tool_started", "agent.tool_finished"):
        return None

    tool = payload.get("tool")
    step = payload.get("step")
    if type(step) is not int or step < 0:
        return None
    if not isinstance(tool, str) or not re.fullmatch(r"[a-zA-Z][\w.-]{0,63}", tool, re.ASCII):
        tool = "unknown"
    fields: dict[str, Any] = {"step": step, "tool": tool}
    if kind == "agent.tool_started":
        return {**fields, "target_hint": _TOOL_TARGETS.get(tool, "tool invocation")}

    status = payload.get("status")
    if status is None and isinstance(payload.get("is_error"), bool):
        status = "error" if payload["is_error"] else "ok"
    duration = payload.get("duration_ms")
    if status not in ("ok", "error", "denied"):
        return None
    signature = payload.get("error_signature")
    if status == "ok":
        signature = ""
    elif not isinstance(signature, str) or not signature:
        signature = status
    elif not re.fullmatch(r"sha256:[0-9a-f]{64}", signature):
        signature = content_hash(signature)
    return {
        **fields,
        "status": status,
        "duration_ms": max(0, duration) if type(duration) is int else 0,
        "error_signature": signature,
    }


class LiveActivity:
    def __init__(self) -> None:
        self._lock = Lock()
        self._previews: dict[str, dict[str, tuple[int, str]]] = {}

    def update(self, run_id: str, task_id: str, attempt: int, text: str, *, replace: bool) -> None:
        with self._lock:
            previews = self._previews.setdefault(run_id, {})
            old_attempt, old_text = previews.get(task_id, (attempt, ""))
            prefix = old_text if old_attempt == attempt and not replace else ""
            previews[task_id] = (attempt, (prefix + text[-_PREVIEW_LIMIT:])[-_PREVIEW_LIMIT:])

    def clear(self, run_id: str, task_id: str, attempt: int) -> None:
        with self._lock:
            previews = self._previews.get(run_id)
            if previews is None:
                return
            if task_id in previews and previews[task_id][0] == attempt:
                del previews[task_id]
            if not previews:
                del self._previews[run_id]

    def snapshot(self, run_id: str) -> dict[str, dict[str, Any]]:
        with self._lock:
            previews = self._previews.get(run_id, {}).copy()
        result = {}
        for task_id, (attempt, text) in previews.items():
            lines = text.rstrip().splitlines()
            if lines:
                result[task_id] = {"attempt": attempt, "preview": " ".join(lines[-1].split())}
        return result
