"""Poka-Yoke filesystem & sandbox command tools with write_scope and output caps."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from olympus.core.dag import paths_overlap
from olympus.harness.worktree import build_sandbox_env
from olympus.providers.base import ToolSpec

_IGNORED_DIR_NAMES = frozenset(
    {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".olympus"}
)


def truncate_output(text: str, *, max_chars: int, hint: str) -> str:
    """Truncate oversized tool output while giving an actionable recovery hint."""
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    omitted = len(text) - max_chars
    return f"{head}\n\n[TRUNCATED: {omitted} chars omitted. {hint}]"


@dataclass(frozen=True)
class RegisteredTool:
    spec: ToolSpec
    handler: Callable[[dict[str, Any]], str]


class WorkspaceToolkit:
    """Workspace-scoped Poka-Yoke toolset enforcing absolute paths and write_scope."""

    def __init__(
        self,
        workspace_root: Path | str,
        *,
        allowed_write_scope: list[str] | None = None,
        read_only: bool = False,
        max_output_chars: int = 12_000,
        env_root: Path | str | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.env_root = Path(env_root).resolve() if env_root is not None else self.workspace_root
        self.allowed_write_scope = allowed_write_scope
        self.read_only = read_only
        self.max_output_chars = max_output_chars
        self._tools = self._build_tools()

    def tool_specs(self) -> list[ToolSpec]:
        return [self._tools[name].spec for name in sorted(self._tools)]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(sorted(self._tools))
            return f"Error: unknown tool '{name}'. Available tools: {available}"
        try:
            return tool.handler(arguments)
        except Exception as exc:  # noqa: BLE001 - return actionable error text to model
            return f"Error executing {name}: {exc}"

    def _resolve_abs_path(self, raw_path: Any) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(
                f"path must be a non-empty absolute path inside '{self.workspace_root}'"
            )
        p = Path(raw_path)
        if not p.is_absolute():
            raise ValueError(
                f"Relative path '{raw_path}' is not allowed. "
                f"Pass an absolute path starting with '{self.workspace_root}/'."
            )
        resolved = p.resolve()
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError(
                f"Path '{resolved}' escapes workspace root '{self.workspace_root}'."
            ) from exc
        return resolved

    def _assert_write_allowed(self, resolved: Path) -> None:
        if self.read_only:
            raise PermissionError("Toolkit is read-only; write operations are disabled.")
        if self.allowed_write_scope is None:
            return
        rel = resolved.relative_to(self.workspace_root).as_posix()
        if any(paths_overlap(rel, scope) for scope in self.allowed_write_scope):
            return
        allowed_str = ", ".join(self.allowed_write_scope)
        raise PermissionError(
            f"Write to '{rel}' is outside allowed write_scope [{allowed_str}]."
        )

    def _iter_workspace_files(self, base: Path) -> list[Path]:
        if base.is_file():
            return [base]
        files: list[Path] = []
        for root, dirs, names in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d not in _IGNORED_DIR_NAMES)
            for fname in sorted(names):
                files.append(Path(root) / fname)
        return files

    def list_relative_files(self, *, max_files: int = 40) -> tuple[list[str], int]:
        """Return up to ``max_files`` sorted repo-relative paths and total file count."""
        all_files = self._iter_workspace_files(self.workspace_root)
        rels = [p.resolve().relative_to(self.workspace_root).as_posix() for p in all_files]
        return rels[: max(0, max_files)], len(rels)

    def _read_file(self, args: dict[str, Any]) -> str:
        target = self._resolve_abs_path(args.get("path"))
        if not target.is_file():
            raise FileNotFoundError(
                f"File not found: '{target}'. Use glob_files under '{self.workspace_root}'."
            )
        start = int(args.get("start_line", 1))
        end = int(args.get("end_line", start + 199))
        if start < 1 or end < start:
            raise ValueError("Require 1 <= start_line <= end_line.")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        sliced = lines[start - 1 : end]
        numbered = [f"{idx}: {line}" for idx, line in enumerate(sliced, start=start)]
        header = f"# {target} (lines {start}-{min(end, total)} of {total})\n"
        body = header + "\n".join(numbered)
        return truncate_output(
            body,
            max_chars=self.max_output_chars,
            hint=(
                f"Call read_file with a smaller range, e.g. "
                f"start_line={start}, end_line={start + 80}."
            ),
        )

    def _glob_files(self, args: dict[str, Any]) -> str:
        pattern = str(args.get("pattern") or "").strip()
        if not pattern:
            raise ValueError("pattern must be non-empty, e.g. '**/*.py'")
        base = (
            self._resolve_abs_path(args["path"])
            if args.get("path") is not None
            else self.workspace_root
        )
        if not base.exists():
            raise FileNotFoundError(f"Search base does not exist: '{base}'")
        matches: list[Path] = []
        for p in sorted(base.glob(pattern)):
            rel_parts = p.resolve().relative_to(self.workspace_root).parts
            if not any(part in _IGNORED_DIR_NAMES for part in rel_parts):
                matches.append(p.resolve())
        max_items = 100
        lines = [str(p) for p in matches[:max_items]]
        if len(matches) > max_items:
            lines.append(
                f"... ({len(matches) - max_items} more matches omitted; "
                "narrow glob pattern or path)"
            )
        return "\n".join(lines) if lines else "(no matching files)"

    def _grep_files(self, args: dict[str, Any]) -> str:
        raw_pat = str(args.get("pattern") or "")
        if not raw_pat:
            raise ValueError("pattern must be a non-empty regex string")
        regex = re.compile(raw_pat)
        base = (
            self._resolve_abs_path(args["path"])
            if args.get("path") is not None
            else self.workspace_root
        )
        if not base.exists():
            raise FileNotFoundError(f"Search path does not exist: '{base}'")
        max_matches = min(int(args.get("max_matches", 50)), 200)
        hits: list[str] = []
        for fpath in self._iter_workspace_files(base):
            try:
                content = fpath.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for lineno, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    hits.append(f"{fpath}:{lineno}: {line}")
                    if len(hits) >= max_matches:
                        hits.append(
                            f"[STOPPED at {max_matches} matches; narrow path or pattern]"
                        )
                        return truncate_output(
                            "\n".join(hits),
                            max_chars=self.max_output_chars,
                            hint="Narrow grep_files to a subdirectory or stricter regex.",
                        )
        return truncate_output(
            "\n".join(hits) if hits else "(no matches)",
            max_chars=self.max_output_chars,
            hint="Narrow grep_files to a subdirectory or stricter regex.",
        )

    def _write_file(self, args: dict[str, Any]) -> str:
        target = self._resolve_abs_path(args.get("path"))
        self._assert_write_allowed(target)
        content = args.get("content")
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {target}"

    def _edit_file(self, args: dict[str, Any]) -> str:
        target_path = self._resolve_abs_path(args.get("path"))
        self._assert_write_allowed(target_path)
        if not target_path.is_file():
            raise FileNotFoundError(
                f"Cannot edit missing file '{target_path}'. Use write_file to create new files."
            )
        old_text = args.get("target")
        new_text = args.get("replacement")
        if not isinstance(old_text, str) or not old_text:
            raise ValueError("target must be a non-empty exact string present in the file")
        if not isinstance(new_text, str):
            raise ValueError("replacement must be a string")
        content = target_path.read_text(encoding="utf-8")
        count = content.count(old_text)
        if count == 0:
            raise ValueError(
                f"target text not found in '{target_path}'. "
                "Call read_file first to verify exact whitespace and lines."
            )
        if count > 1:
            raise ValueError(
                f"target text matched {count} times in '{target_path}'. "
                "Include more surrounding lines so target is unique."
            )
        updated = content.replace(old_text, new_text, 1)
        target_path.write_text(updated, encoding="utf-8")
        return f"Updated {target_path} (1 replacement)"

    def _run_bash(self, args: dict[str, Any]) -> str:
        if self.read_only:
            raise PermissionError("Toolkit is read-only; run_bash is disabled.")
        cmd = args.get("command")
        if not isinstance(cmd, str) or not cmd.strip():
            raise ValueError("command must be a non-empty shell string")
        timeout = min(int(args.get("timeout_seconds", 60)), 300)
        proc = subprocess.run(
            ["bash", "-lc", cmd],
            cwd=self.workspace_root,
            env=build_sandbox_env(self.workspace_root, host_root=self.env_root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        combined = (
            f"exit_code={proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout.rstrip()}\n"
            f"--- stderr ---\n{proc.stderr.rstrip()}"
        )
        return truncate_output(
            combined,
            max_chars=self.max_output_chars,
            hint="Pipe noisy output through `head -n 80` or `pytest -q --tb=short`.",
        )

    def _build_tools(self) -> dict[str, RegisteredTool]:
        tools: dict[str, RegisteredTool] = {
            "read_file": RegisteredTool(
                spec=ToolSpec(
                    name="read_file",
                    description="Read lines from a file inside workspace using an absolute path.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Absolute file path."},
                            "start_line": {"type": "integer", "minimum": 1},
                            "end_line": {"type": "integer", "minimum": 1},
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                ),
                handler=self._read_file,
            ),
            "glob_files": RegisteredTool(
                spec=ToolSpec(
                    name="glob_files",
                    description="List files matching a glob pattern inside the workspace.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {"type": "string", "description": "Optional absolute dir."},
                        },
                        "required": ["pattern"],
                        "additionalProperties": False,
                    },
                ),
                handler=self._glob_files,
            ),
            "grep_files": RegisteredTool(
                spec=ToolSpec(
                    name="grep_files",
                    description="Search workspace files with a Python regex.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {"type": "string", "description": "Optional absolute path."},
                            "max_matches": {"type": "integer", "minimum": 1, "maximum": 200},
                        },
                        "required": ["pattern"],
                        "additionalProperties": False,
                    },
                ),
                handler=self._grep_files,
            ),
        }
        if not self.read_only:
            tools["write_file"] = RegisteredTool(
                spec=ToolSpec(
                    name="write_file",
                    description="Create or overwrite a file within allowed write_scope.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Absolute file path."},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                        "additionalProperties": False,
                    },
                ),
                handler=self._write_file,
            )
            tools["edit_file"] = RegisteredTool(
                spec=ToolSpec(
                    name="edit_file",
                    description="Replace an exact unique substring inside an existing file.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Absolute file path."},
                            "target": {"type": "string"},
                            "replacement": {"type": "string"},
                        },
                        "required": ["path", "target", "replacement"],
                        "additionalProperties": False,
                    },
                ),
                handler=self._edit_file,
            )
            tools["run_bash"] = RegisteredTool(
                spec=ToolSpec(
                    name="run_bash",
                    description="Execute a bash command inside the isolated workspace root.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300},
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                ),
                handler=self._run_bash,
            )
        return tools
