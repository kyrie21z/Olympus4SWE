"""Privacy-safe per-turn receipts: aggregate counts only, never prompt or tool text."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

RECEIPT_KEYS = ("turn_id", "actor", "task_id", "llm", "tools", "artifacts", "one_line")

_TOOL_NAME_RE = re.compile(r"[A-Za-z0-9_.:-]{1,64}")


def _format_k(n: int) -> str:
    if n < 1000:
        return str(n)
    return f"{n / 1000:.1f}k"


def _safe_tool_name(name: Any) -> str:
    if isinstance(name, str) and _TOOL_NAME_RE.fullmatch(name):
        return name
    return "unknown"


def _collapse_tool_names(tools: Sequence[dict[str, Any]]) -> str:
    counts: list[tuple[str, int]] = []
    for entry in tools:
        name = str(entry["tool"])
        if counts and counts[-1][0] == name:
            counts[-1] = (name, counts[-1][1] + 1)
        else:
            counts.append((name, 1))
    return " · ".join(name if count == 1 else f"{name} ×{count}" for name, count in counts)


def build_turn_receipt(
    *,
    turn_id: str,
    actor: str,
    task_id: str | None = None,
    events: Sequence[dict[str, Any]] = (),
    diff_lines: tuple[int, int] = (0, 0),
    checks_passed: int = 0,
    checks_total: int = 0,
) -> dict[str, Any]:
    """Aggregate turn events into a receipt containing only counts and sanitized names."""
    calls = 0
    input_tokens = 0
    output_tokens = 0
    tools: list[dict[str, Any]] = []
    for event in events:
        kind = event.get("kind")
        if kind == "llm":
            calls += int(event.get("calls", 1))
            input_tokens += int(event.get("input_tokens", 0))
            output_tokens += int(event.get("output_tokens", 0))
        elif kind == "tool_end":
            tools.append(
                {
                    "tool": _safe_tool_name(event.get("tool")),
                    "status": "error" if event.get("is_error") else "ok",
                    "duration_ms": int(event.get("duration_ms", 0)),
                }
            )
        elif kind == "pre_tool_blocked":
            tools.append(
                {
                    "tool": _safe_tool_name(event.get("tool")),
                    "status": "denied",
                    "duration_ms": 0,
                }
            )

    one_line = (
        f"{actor} · {calls} calls · "
        f"{_format_k(input_tokens)} in / {_format_k(output_tokens)} out"
    )
    if tools:
        one_line += f" | {_collapse_tool_names(tools)}"
    if any(diff_lines):
        one_line += f" | diff +{diff_lines[0]}/-{diff_lines[1]}"

    receipt: dict[str, Any] = {
        "turn_id": turn_id,
        "actor": actor,
        "task_id": task_id,
        "llm": {
            "calls": calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "tools": tools,
        "artifacts": {
            "diff_added": diff_lines[0],
            "diff_removed": diff_lines[1],
            "checks_passed": checks_passed,
            "checks_total": checks_total,
        },
        "one_line": one_line,
    }
    return receipt
