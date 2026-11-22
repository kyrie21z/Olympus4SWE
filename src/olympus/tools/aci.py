"""Poka-Yoke filesystem & sandbox command tools with write_scope and output caps."""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from olympus.providers.base import ToolSpec

IGNORED_DIR_NAMES = frozenset(
    {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".olympus"}
)

SANDBOX_TMP_ENV_VARS = ("TMPDIR", "TEMP", "TMP", "SQLITE_TMPDIR")

_GLOB_CHARS = ("*", "?", "[")


def sandbox_tmp_path(
    workspace_root: Path | str,
    host_root: Path | str | None = None,
) -> Path:
    """Return the per-sandbox temp directory, kept outside the sandbox git worktree."""
    ws = Path(workspace_root).resolve()
    if host_root is not None:
        host = Path(host_root).resolve()
    else:
        parent = ws.parent
        if parent.name == "sandboxes" and parent.parent.name == ".olympus":
            host = parent.parent.parent
        else:
            host = next((c for c in (ws, *ws.parents) if (c / ".git").is_dir()), ws)
    return host / ".olympus" / "sandbox_tmp" / ws.name


def build_sandbox_env(
    workspace_root: Path | str,
    host_root: Path | str | None = None,
) -> dict[str, str]:
    """Build an isolated subprocess environment bound to sandbox paths and target .venv."""
    ws = Path(workspace_root).resolve()
    host = Path(host_root).resolve() if host_root is not None else None
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)

    venv_candidates = [ws / ".venv"]
    if host is not None and host != ws:
        venv_candidates.append(host / ".venv")
    current_prefix = Path(sys.executable).resolve().parent.parent
    if (
        (current_prefix / "pyvenv.cfg").exists()
        and (current_prefix / "bin" / "pytest").exists()
        and current_prefix not in venv_candidates
    ):
        venv_candidates.append(current_prefix)
    for venv_dir in venv_candidates:
        bin_dir = venv_dir / "bin"
        if bin_dir.is_dir():
            env["VIRTUAL_ENV"] = str(venv_dir)
            existing_path = env.get("PATH", "")
            env["PATH"] = f"{bin_dir}{os.pathsep}{existing_path}" if existing_path else str(bin_dir)
            break

    python_roots: list[str] = []
    src_dir = ws / "src"
    if src_dir.is_dir():
        python_roots.append(str(src_dir))
    python_roots.append(str(ws))
    existing_pp = env.get("PYTHONPATH", "")
    if existing_pp:
        for entry in existing_pp.split(os.pathsep):
            if entry and entry not in python_roots:
                python_roots.append(entry)
    env["PYTHONPATH"] = os.pathsep.join(python_roots)
    tmp_dir = sandbox_tmp_path(ws, host)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for name in SANDBOX_TMP_ENV_VARS:
        env[name] = str(tmp_dir)
    return env


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


@dataclass
class WorkspacePolicy:
    """Canonical authorization policy shared by tool handlers and pre-tool hooks."""

    workspace_root: Path | None = None
    allowed_write_scope: Sequence[str] | None = None
    forbidden_write_scope: Sequence[str] = ()
    read_only: bool = False

    READ_ONLY_TOOLS: ClassVar[frozenset[str]] = frozenset(
        {"read_file", "glob_files", "grep_files"}
    )
    WRITE_TOOLS: ClassVar[frozenset[str]] = frozenset({"write_file", "edit_file"})

    def __post_init__(self) -> None:
        if self.workspace_root is not None:
            self.workspace_root = Path(self.workspace_root).expanduser().resolve()
        if self.allowed_write_scope is not None:
            self.allowed_write_scope = tuple(self.allowed_write_scope)
        self.forbidden_write_scope = tuple(self.forbidden_write_scope)

    @staticmethod
    def _strip_dot_prefix(text: str) -> str:
        while text.startswith("./"):
            text = text[2:]
        return text

    def _relative_target(self, raw_path: Any) -> str:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("Write tools require a non-empty 'path' argument.")
        # Preserve the caller's exact path characters (only separators are normalized):
        # a trailing space names a different file and must not be silently trimmed.
        text = self._strip_dot_prefix(raw_path.replace("\\", "/"))
        candidate = Path(raw_path)
        if self.workspace_root is None:
            if candidate.is_absolute():
                raise ValueError(
                    f"Absolute path '{raw_path}' cannot be validated without a workspace_root."
                )
            pure = PurePosixPath(text or ".")
            if ".." in pure.parts:
                raise ValueError(f"Path '{raw_path}' must not contain '..' traversal.")
            return pure.as_posix()
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            pure = PurePosixPath(text or ".")
            if ".." in pure.parts:
                raise ValueError(f"Path '{raw_path}' must not contain '..' traversal.")
            resolved = (self.workspace_root / pure).resolve()
        try:
            return resolved.relative_to(self.workspace_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Path '{resolved}' escapes workspace root '{self.workspace_root}'."
            ) from exc

    def _scope_pattern(self, raw: str) -> str | None:
        if not raw or not raw.strip():
            return None
        pattern = self._strip_dot_prefix(raw.replace("\\", "/"))
        if not pattern:
            return None
        if pattern.startswith("/"):
            if self.workspace_root is None:
                return None
            try:
                pattern = Path(pattern).resolve().relative_to(self.workspace_root).as_posix()
            except ValueError:
                return None
        if ".." in PurePosixPath(pattern.rstrip("/") or ".").parts:
            return None
        return pattern

    def _scope_matches(self, rel: str, raw_pattern: str) -> bool:
        """Exact glob match or symmetric ancestor/descendant match for forbidden scopes."""
        pattern = self._scope_pattern(raw_pattern)
        if pattern is None:
            return False
        norm_rel = self._strip_dot_prefix(rel).lstrip("/")
        if any(char in pattern for char in _GLOB_CHARS):
            return fnmatch.fnmatch(norm_rel, pattern)
        norm_pat = pattern.rstrip("/")
        if norm_pat in ("", "."):
            return True
        return (
            norm_rel == norm_pat
            or norm_rel.startswith(f"{norm_pat}/")
            or norm_pat.startswith(f"{norm_rel}/")
        )

    def _scope_contains(self, rel: str, raw_pattern: str) -> bool:
        """True when ``rel`` sits at or below an authorized scope; never grants ancestors."""
        pattern = self._scope_pattern(raw_pattern)
        if pattern is None:
            return False
        if any(char in pattern for char in _GLOB_CHARS):
            return fnmatch.fnmatch(rel, pattern)
        norm_rel = self._strip_dot_prefix(rel).lstrip("/")
        norm_pattern = pattern.rstrip("/")
        if norm_pattern in ("", "."):
            return True
        return norm_rel == norm_pattern or norm_rel.startswith(f"{norm_pattern}/")

    def forbidden_match(self, rel: str) -> str | None:
        """Return the forbidden pattern matching ``rel``, or None when the path is allowed."""
        for pattern in self.forbidden_write_scope:
            if self._scope_contains(rel, pattern) or self._scope_matches(rel, pattern):
                return pattern
        return None

    def allows_relative_path(self, rel: str) -> bool:
        """Return True when a repo-relative path is inside the allowed write scope."""
        if self.allowed_write_scope is None:
            return True
        raw = str(rel) if os.sep == "/" else str(rel).replace(os.sep, "/")
        norm_rel = self._strip_dot_prefix(raw).lstrip("/")
        return any(self._scope_contains(norm_rel, pattern) for pattern in self.allowed_write_scope)

    def authorize_tool(self, tool_name: str, arguments: Mapping[str, Any]) -> tuple[bool, str]:
        if tool_name in self.READ_ONLY_TOOLS:
            return True, ""
        if self.read_only:
            return False, f"Toolkit is read-only; {tool_name} is disabled."
        if self.allowed_write_scope is not None and len(self.allowed_write_scope) == 0:
            return False, "No write_scope authorized for this worker."
        if tool_name == "run_bash":
            return True, ""
        if tool_name not in self.WRITE_TOOLS:
            return False, f"Tool '{tool_name}' is not authorized for this worker."

        raw_path = arguments.get("path") or arguments.get("file_path") or ""
        try:
            rel = self._relative_target(raw_path)
        except ValueError as exc:
            return False, str(exc)
        if self.forbidden_match(rel) is not None:
            return (
                False,
                f"Target path '{rel}' is a read-only shared contract path "
                f"({list(self.forbidden_write_scope)}): only contract tasks may modify it.",
            )
        if self.allowed_write_scope is None or any(
            self._scope_contains(rel, pattern) for pattern in self.allowed_write_scope
        ):
            return True, ""
        return (
            False,
            f"Target path '{rel}' is outside authorized write_scope: "
            f"{list(self.allowed_write_scope)}",
        )

    def assert_write_allowed(self, resolved_path: Path) -> None:
        if self.read_only:
            raise PermissionError("Toolkit is read-only; write operations are disabled.")
        resolved = resolved_path.resolve()
        if self.workspace_root is None:
            rel = resolved.as_posix()
        else:
            try:
                rel = resolved.relative_to(self.workspace_root).as_posix()
            except ValueError as exc:
                raise PermissionError(
                    f"Path '{resolved}' escapes workspace root '{self.workspace_root}'."
                ) from exc
        if self.forbidden_match(rel) is not None:
            raise PermissionError(
                f"Target path '{rel}' is a read-only shared contract path "
                f"({list(self.forbidden_write_scope)}): only contract tasks may modify it."
            )
        if self.allowed_write_scope is None:
            return
        if any(self._scope_contains(rel, pattern) for pattern in self.allowed_write_scope):
            return
        allowed_str = ", ".join(self.allowed_write_scope)
        raise PermissionError(f"Write to '{rel}' is outside allowed write_scope [{allowed_str}].")

    def as_pre_tool_hook(self) -> Callable[[str, dict[str, Any]], tuple[bool, str]]:
        def hook(tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
            return self.authorize_tool(tool_name, arguments)

        return hook


class WorkspaceToolkit:
    """Workspace-scoped Poka-Yoke toolset enforcing absolute paths and write_scope."""

    def __init__(
        self,
        workspace_root: Path | str,
        *,
        allowed_write_scope: Sequence[str] | None = None,
        forbidden_write_scope: Sequence[str] = (),
        read_only: bool = False,
        max_output_chars: int = 12_000,
        env_root: Path | str | None = None,
        env_builder: Callable[[Path, Path], dict[str, str]] | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.env_root = Path(env_root).resolve() if env_root is not None else self.workspace_root
        self.policy = WorkspacePolicy(
            workspace_root=self.workspace_root,
            allowed_write_scope=allowed_write_scope,
            forbidden_write_scope=forbidden_write_scope,
            read_only=read_only,
        )
        self.max_output_chars = max_output_chars
        self._env_builder = env_builder or build_sandbox_env
        self._tools = self._build_tools()

    def as_pre_tool_hook(self) -> Callable[[str, dict[str, Any]], tuple[bool, str]]:
        return self.policy.as_pre_tool_hook()

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

    def _iter_workspace_files(self, base: Path) -> list[Path]:
        if base.is_file():
            return [base]
        files: list[Path] = []
        for root, dirs, names in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIR_NAMES)
            for fname in sorted(names):
                candidate = Path(root) / fname
                try:
                    resolved = candidate.resolve()
                except OSError:
                    continue
                # Skip symlinks (or junctions) whose target escapes the workspace.
                if not resolved.is_relative_to(self.workspace_root):
                    continue
                files.append(candidate)
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
            try:
                resolved = p.resolve()
                rel_parts = resolved.relative_to(self.workspace_root).parts
            except (OSError, ValueError):
                # Broken links and symlinks escaping the workspace are never reported.
                continue
            if not any(part in IGNORED_DIR_NAMES for part in rel_parts):
                matches.append(resolved)
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
        self.policy.assert_write_allowed(target)
        content = args.get("content")
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {target}"

    def _edit_file(self, args: dict[str, Any]) -> str:
        target_path = self._resolve_abs_path(args.get("path"))
        self.policy.assert_write_allowed(target_path)
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
        allowed, reason = self.policy.authorize_tool("run_bash", args)
        if not allowed:
            raise PermissionError(reason)
        cmd = args.get("command")
        if not isinstance(cmd, str) or not cmd.strip():
            raise ValueError("command must be a non-empty shell string")
        timeout = min(int(args.get("timeout_seconds", 60)), 300)
        proc = subprocess.run(
            ["bash", "-lc", cmd],
            cwd=self.workspace_root,
            env=self._env_builder(self.workspace_root, self.env_root),
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
        if not self.policy.read_only:
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
