"""Deterministic pre-tool-use authorization gates (write_scope enforcement)."""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

PreToolUseHook = Callable[[str, dict[str, Any]], tuple[bool, str]]

_READ_ONLY_TOOLS = frozenset({"read_file", "glob_files", "grep_files", "run_bash"})
_WRITE_TOOLS = frozenset({"write_file", "edit_file"})


def _normalize_patterns(allowed_globs: Sequence[str]) -> list[str]:
    patterns: list[str] = []
    for raw in allowed_globs:
        pattern = raw.strip()
        if not pattern:
            continue
        if pattern.endswith("/"):
            pattern += "*"
        patterns.append(pattern)
    return patterns


def _relative_target(raw: str, workspace_root: Path | None) -> str:
    rel = raw
    if workspace_root is not None:
        path = Path(raw)
        if path.is_absolute():
            with_root = workspace_root.expanduser().resolve()
            try:
                rel = path.resolve().relative_to(with_root).as_posix()
            except ValueError:
                rel = raw
    while rel.startswith("./"):
        rel = rel[2:]
    return rel


def default_write_scope_hook(
    allowed_globs: Sequence[str],
    *,
    forbidden_globs: Sequence[str] = (),
    workspace_root: Path | None = None,
) -> PreToolUseHook:
    """Build a hook that denies writes outside ``allowed_globs`` or inside ``forbidden_globs``."""
    patterns = _normalize_patterns(allowed_globs)
    forbidden_patterns = _normalize_patterns(forbidden_globs)

    def _matches(rel: str, pats: list[str]) -> bool:
        for pattern in pats:
            if fnmatch.fnmatch(rel, pattern):
                return True
            if not pattern.endswith("*") and rel.startswith(f"{pattern.rstrip('/')}/"):
                return True
        return False

    def hook(tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        if tool_name in _READ_ONLY_TOOLS:
            return True, ""
        if not patterns:
            return False, "No write_scope authorized for this worker."
        if tool_name not in _WRITE_TOOLS:
            return False, f"Tool '{tool_name}' is not authorized for this worker."

        raw_path = arguments.get("path") or arguments.get("file_path") or ""
        rel = _relative_target(str(raw_path), workspace_root)
        if forbidden_patterns and _matches(rel, forbidden_patterns):
            return (
                False,
                f"Target path '{rel}' is a read-only shared contract path "
                f"({list(forbidden_globs)}): only contract tasks may modify it.",
            )
        if _matches(rel, patterns):
            return True, ""
        return (
            False,
            f"Target path '{rel}' is outside authorized write_scope: {list(allowed_globs)}",
        )

    return hook
