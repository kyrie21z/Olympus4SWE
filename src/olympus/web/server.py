"""FastAPI + uvicorn backend for the Ariadne Web Console (Milestone 4)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

import olympus.web.acceptance as delivery_review
from olympus.core.contracts import ApprovalTarget, new_id
from olympus.core.control import RunControl, has_unfulfilled_revision
from olympus.core.store import ConcurrencyConflictError, NotFoundError, RunStore
from olympus.graph.loader import load_workflow
from olympus.stages.align import has_pending_resynthesis
from olympus.web.demos import list_benchmark_presets, materialize_benchmark_demo, seed_demo_runs
from olympus.web.projection import (
    build_board_payload,
    build_run_projection,
    build_workspace_counts,
    read_recorded_attempt,
)
from olympus.web.schemas import (
    AttemptArtifactsResponse,
    BallHolder,
    CloneWorkspaceRequest,
    CommandRequest,
    CommandResponse,
    ConflictResponse,
    CreateRunRequest,
    CreateRunResponse,
    CreateWorkspaceRequest,
    DeleteWorkspaceResponse,
    DemosResponse,
    ErrorResponse,
    EventsResponse,
    FilesystemBrowseQuery,
    FilesystemBrowseResponse,
    HealthResponse,
    ProvidersResponse,
    RunListResponse,
    RunProjection,
    RunProjectionResponse,
    SeedDemoResponse,
    TaskType,
    WorkflowsResponse,
    WorkspaceListResponse,
    WorkspaceMode,
    WorkspaceResponse,
)
from olympus.web.supervisor import RunSupervisor

_STATIC_DIR = Path(__file__).resolve().parent / "static"

# Only these extensions are ever served from /static/; everything else is a 404.
_STATIC_MIME_TYPES: dict[str, str] = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".webp": "image/webp",
}


@dataclass(frozen=True)
class RunScope:
    """Deterministic Scope x Type binding resolved before any run or LLM call."""

    workspace_id: str | None
    workspace_root: Path | None
    workspace_mode: str
    task_type: str
    base_ref: str | None
    scope_paths: list[str]
    workspace_source: Literal["explicit", "server_default", "none"] = "none"


def _detect_git_repo(root: Path) -> tuple[bool, str | None]:
    """Read-only git metadata; returns ``(is_repo, current_branch)``."""
    if not (root / ".git").exists():
        return False, None
    for args in (
        ("rev-parse", "--abbrev-ref", "HEAD"),
        ("symbolic-ref", "--short", "HEAD"),
    ):
        try:
            proc = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return True, None
        branch = proc.stdout.strip()
        if proc.returncode == 0 and branch and branch != "HEAD":
            return True, branch
    return True, None


def _read_git_head_state(worktree_path: str) -> str | None:
    root = Path(worktree_path)
    dot_git = root / ".git"
    head_file: Path | None = None
    try:
        if dot_git.is_dir():
            head_file = dot_git / "HEAD"
        elif dot_git.is_file():
            pointer = dot_git.read_text(encoding="utf-8", errors="ignore").strip()
            if pointer.startswith("gitdir:"):
                git_dir = Path(pointer.removeprefix("gitdir:").strip())
                if not git_dir.is_absolute():
                    git_dir = (root / git_dir).resolve()
                head_file = git_dir / "HEAD"
        if head_file is not None and head_file.is_file():
            return head_file.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None
    return None


_MAX_BROWSE_PATH_LENGTH = 512
_MAX_BROWSE_ENTRIES = 200
_MAX_CLONE_URL_LENGTH = 512
_MAX_FOLDER_NAME_LENGTH = 120
_CLONE_URL_SCHEMES = frozenset({"https", "ssh"})
_LOCAL_PATH_PREFIXES = ("/", "~", "./", "../", ".\\", "..\\", "\\\\")
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)
_SCP_GIT_URL = re.compile(r"^(?P<user>[A-Za-z0-9._-]+)@(?P<host>[A-Za-z0-9.-]+):(?P<path>[^:\s]+)$")
_SHORTHAND_GIT_URL = re.compile(r"^(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)$")
_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _expand_user_path(raw: str) -> Path:
    try:
        return Path(raw).expanduser()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Cannot resolve path '{raw}': {exc}") from exc


def _branch_name_from_git_head(head_state: str | None) -> str | None:
    if head_state is None or not head_state.startswith("ref: refs/heads/"):
        return None
    return head_state.removeprefix("ref: refs/heads/").strip() or None


def _clone_repo_name(path: str, *, min_segments: int = 2) -> str:
    segments = [segment for segment in path.strip("/").split("/") if segment]
    if len(segments) < min_segments:
        if min_segments == 1:
            raise ValueError("remote_url is missing a repository name")
        raise ValueError("remote_url must include a repository owner and name")
    if any(segment in {".", ".."} for segment in segments):
        raise ValueError("remote_url must not contain path traversal segments")
    repo = segments[-1]
    name = repo[:-4] if repo.endswith(".git") else repo
    if not name:
        raise ValueError("remote_url is missing a repository name")
    return name


def normalize_clone_url(raw: str) -> tuple[str, str]:
    """Validate a Git remote and return ``(normalized_url, inferred_repo_name)``."""
    value = raw.strip()
    if not value:
        raise ValueError("remote_url is required")
    if len(value) > _MAX_CLONE_URL_LENGTH:
        raise ValueError(f"remote_url exceeds {_MAX_CLONE_URL_LENGTH} characters")
    if value.startswith("-"):
        raise ValueError("remote_url must not start with '-'")
    if "\x00" in value or any(
        ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in value
    ):
        raise ValueError("remote_url contains whitespace or control characters")
    if value in {".", ".."} or value.startswith(_LOCAL_PATH_PREFIXES) or _DRIVE_PATH.match(value):
        raise ValueError("remote_url must be a Git remote, not a local filesystem path")
    if "://" in value:
        parts = urlsplit(value)
        scheme = parts.scheme.lower()
        if scheme not in _CLONE_URL_SCHEMES:
            raise ValueError(f"Unsupported clone URL scheme: {parts.scheme or 'unknown'}")
        if not parts.hostname:
            raise ValueError("remote_url is missing a host")
        repo_name = _clone_repo_name(parts.path, min_segments=1 if scheme == "ssh" else 2)
        return urlunsplit((scheme, parts.netloc, parts.path, parts.query, "")), repo_name
    scp = _SCP_GIT_URL.match(value)
    if scp is not None:
        return value.rstrip("/"), _clone_repo_name(scp.group("path"), min_segments=1)
    shorthand = _SHORTHAND_GIT_URL.match(value)
    if shorthand is not None:
        repo = shorthand.group("repo")
        repo_slug = repo[:-4] if repo.endswith(".git") else repo
        if not repo_slug:
            raise ValueError("remote_url is missing a repository name")
        return f"https://github.com/{shorthand.group('owner')}/{repo_slug}.git", repo_slug
    raise ValueError(
        "remote_url must be an https/ssh Git URL, an SCP-style remote, or owner/repo shorthand"
    )


def _validate_clone_folder_name(raw: str) -> str:
    value = raw.strip()
    if not value:
        raise ValueError("Clone folder name is required")
    if value in {".", ".."}:
        raise ValueError("Clone folder name must not be '.' or '..'")
    if "/" in value or "\\" in value:
        raise ValueError("Clone folder name must not contain path separators")
    if value.startswith("-"):
        raise ValueError("Clone folder name must not start with '-'")
    if value != value.rstrip(" ."):
        raise ValueError("Clone folder name must not end with a space or dot")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError("Clone folder name contains control characters")
    if len(value) > _MAX_FOLDER_NAME_LENGTH:
        raise ValueError(f"Clone folder name exceeds {_MAX_FOLDER_NAME_LENGTH} characters")
    if value.split(".", 1)[0].lower() in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"Clone folder name '{value}' is reserved on Windows")
    return value


def _default_workflow_for(workspace_mode: str, task_type: str) -> str:
    if workspace_mode == "spec_only" or task_type == "blueprint":
        return "product_blueprint"
    if task_type == "fix":
        return "repo_fast_patch"
    return "full_sdlc_tdd"


_WORKSPACE_MODES: tuple[str, ...] = ("brownfield", "greenfield", "spec_only")
_TASK_TYPES: tuple[str, ...] = ("feature", "fix", "refactor", "blueprint")


def default_workflow_matrix() -> dict[str, str]:
    """Single-source default workflow mapping exposed via ``GET /api/workflows``."""
    return {
        f"{mode}:{task_type}": _default_workflow_for(mode, task_type)
        for mode in _WORKSPACE_MODES
        for task_type in _TASK_TYPES
    }



class AriadneApp:
    """Pure-Python request dispatcher backing the Ariadne HTTP/SSE console."""

    def __init__(
        self,
        *,
        store: RunStore,
        workspace_root: Path | str,
        artifacts_root: Path | None = None,
        supervisor: RunSupervisor | None = None,
    ) -> None:
        self.store = store
        self.control = RunControl(self.store)
        self.workspace_root = Path(workspace_root).resolve()
        self.artifacts_root = (
            artifacts_root
            if artifacts_root is not None
            else self.workspace_root / ".olympus" / "runs"
        )
        self.supervisor = (
            supervisor
            if supervisor is not None
            else RunSupervisor(
                store=self.store,
                workspace_root=self.workspace_root,
                artifacts_root=self.artifacts_root,
            )
        )
        self._retry_guard = threading.Lock()
        self._retry_locks: dict[str, threading.Lock] = {}
        self._clone_locks: dict[str, threading.Lock] = {}
        self.default_workspace_id: str | None = None
        self.ensure_default_workspace()

    def _clone_lock_for(self, key: str) -> threading.Lock:
        with self._retry_guard:
            return self._clone_locks.setdefault(key, threading.Lock())

    def ensure_default_workspace(self) -> dict[str, Any] | None:
        """Idempotently register the server's default ``workspace_root`` on startup."""
        root = self.workspace_root
        if not root.is_dir():
            return None
        is_git_repo, default_branch = _detect_git_repo(root)
        workspace = self.store.register_workspace(
            name=root.name or "workspace",
            root_path=root,
            is_git_repo=is_git_repo,
            default_branch=default_branch,
            exist_ok=True,
        )
        self.default_workspace_id = str(workspace["workspace_id"])
        return workspace

    def list_workspaces_payload(self) -> dict[str, Any]:
        counts = build_workspace_counts(self.store)
        workspaces = [
            {**workspace, **counts.get(str(workspace["workspace_id"]), {})}
            for workspace in self.store.list_workspaces()
        ]
        return {"ok": True, "workspaces": workspaces}

    def create_workspace(self, payload: CreateWorkspaceRequest) -> dict[str, Any]:
        """Register a local repository; never mutates an existing workspace tree."""
        root = _expand_user_path(payload.root_path).resolve()
        if self.store.get_workspace_by_root(root) is not None:
            raise ValueError(f"Workspace already registered: {root}")
        if not root.exists():
            if not payload.create_if_missing:
                raise ValueError(f"workspace_root not found: {payload.root_path}")
            if not root.parent.is_dir():
                raise ValueError(f"Workspace parent directory does not exist: {root.parent}")
            try:
                root.mkdir()
            except OSError as exc:
                raise ValueError(f"Cannot create workspace directory {root}: {exc}") from exc
        elif not root.is_dir():
            raise ValueError(f"workspace_root is not a directory: {root}")

        is_git_repo, default_branch = _detect_git_repo(root)
        if payload.init_git and not is_git_repo:
            self._init_git_repo(root)
            is_git_repo, default_branch = _detect_git_repo(root)
        workspace = self.store.register_workspace(
            name=(payload.name or "").strip() or root.name or "workspace",
            root_path=root,
            is_git_repo=is_git_repo,
            default_branch=default_branch,
            exist_ok=False,
        )
        return {"ok": True, "workspace": workspace}

    def clone_workspace(self, payload: CloneWorkspaceRequest) -> dict[str, Any]:
        """Clone a validated Git remote into an empty directory and register it."""
        remote_url, inferred_name = normalize_clone_url(payload.remote_url)
        folder_name = _validate_clone_folder_name(payload.folder_name or inferred_name)
        parent = _expand_user_path(payload.parent_dir)
        if not parent.is_dir():
            raise ValueError(f"Clone parent directory does not exist: {payload.parent_dir}")
        parent_dir = parent.resolve()
        dest_entry = parent_dir / folder_name
        if dest_entry.is_symlink():
            raise ValueError(f"Clone destination must not be a symlink: {dest_entry}")
        destination = dest_entry.resolve()
        if destination.parent != parent_dir:
            raise ValueError("Clone destination escapes the selected parent directory")
        with self._clone_lock_for(str(destination)):
            if dest_entry.is_symlink():
                raise ValueError(f"Clone destination must not be a symlink: {dest_entry}")
            existed = destination.exists()
            if existed:
                if not destination.is_dir():
                    raise ValueError(f"Clone destination is not a directory: {destination}")
                try:
                    populated = next(destination.iterdir(), None) is not None
                except OSError as exc:
                    raise ValueError(
                        f"Cannot inspect clone destination {destination}: {exc}"
                    ) from exc
                if populated:
                    raise ValueError(f"Clone destination is not empty: {destination}")
            if self.store.get_workspace_by_root(destination) is not None:
                raise ValueError(f"Workspace already registered: {destination}")
            try:
                self._run_git_clone(remote_url, destination)
            except ValueError:
                if not existed and self.store.get_workspace_by_root(destination) is None:
                    shutil.rmtree(destination, ignore_errors=True)
                raise
            is_git_repo, default_branch = _detect_git_repo(destination)
            workspace = self.store.register_workspace(
                name=(payload.name or "").strip() or folder_name,
                root_path=destination,
                is_git_repo=is_git_repo,
                default_branch=default_branch,
                exist_ok=False,
            )
            return {"ok": True, "workspace": workspace}

    def _run_git_clone(self, remote_url: str, destination: Path) -> None:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        try:
            completed = subprocess.run(
                ["git", "clone", "--", remote_url, str(destination)],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                env=env,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(f"git clone failed: {exc}") from exc
        if completed.returncode != 0:
            lines = (completed.stderr or completed.stdout or "").strip().splitlines()
            summary = lines[-1].strip() if lines else f"exit code {completed.returncode}"
            raise ValueError(f"git clone failed: {summary}")

    def browse_filesystem(
        self, *, partial_path: str | None = None, show_hidden: bool = False
    ) -> dict[str, Any]:
        """List child directories for the T3-style ``partial_path`` picker."""
        raw = (partial_path or "").strip()
        if "\x00" in raw:
            raise ValueError("partial_path must not contain NUL bytes")
        if len(raw) > _MAX_BROWSE_PATH_LENGTH:
            raise ValueError(f"partial_path exceeds {_MAX_BROWSE_PATH_LENGTH} characters")
        if raw:
            resolved = _expand_user_path(raw).resolve()
            if raw.endswith("/") or resolved.is_dir():
                browse_dir, prefix = resolved, ""
            else:
                browse_dir, prefix = resolved.parent, resolved.name
        else:
            browse_dir, prefix = self._default_browse_root(), ""

        parent_path = str(browse_dir.parent) if browse_dir.parent != browse_dir else None
        if not browse_dir.is_dir():
            return {
                "ok": True,
                "current_path": str(browse_dir),
                "parent_path": parent_path,
                "prefix": prefix,
                "path_missing": True,
                "quick_roots": self._quick_roots(),
                "entries": [],
            }

        registered = {str(workspace["root_path"]) for workspace in self.store.list_workspaces()}
        prefix_lower = prefix.lower()
        ranked: list[tuple[tuple[int, int, str], dict[str, Any]]] = []
        try:
            children = sorted(browse_dir.iterdir(), key=lambda child: child.name.lower())
        except OSError:
            children = []
        for child in children:
            name = child.name
            if not show_hidden and not prefix.startswith(".") and name.startswith("."):
                continue
            if prefix_lower and prefix_lower not in name.lower():
                continue
            try:
                if not child.is_dir():
                    continue
                canonical = str(child.resolve())
                dot_git = child / ".git"
                head_file = dot_git / "HEAD"
                is_git_repo = dot_git.is_dir() and head_file.is_file()
            except (OSError, RuntimeError):
                continue
            default_branch: str | None = None
            if is_git_repo:
                try:
                    head_text = head_file.read_text(encoding="utf-8", errors="ignore").strip()
                except OSError:
                    head_text = None
                default_branch = _branch_name_from_git_head(head_text)
            prefix_rank = 0
            if prefix_lower:
                prefix_rank = 0 if name.lower().startswith(prefix_lower) else 1
            key = (prefix_rank, 0 if is_git_repo else 1, name.lower())
            ranked.append(
                (
                    key,
                    {
                        "name": name,
                        "path": canonical,
                        "is_git_repo": is_git_repo,
                        "default_branch": default_branch,
                        "already_registered": canonical in registered,
                    },
                )
            )
        ranked.sort(key=lambda item: item[0])
        return {
            "ok": True,
            "current_path": str(browse_dir),
            "parent_path": parent_path,
            "prefix": prefix,
            "path_missing": False,
            "quick_roots": self._quick_roots(),
            "entries": [entry for _, entry in ranked[:_MAX_BROWSE_ENTRIES]],
        }

    def _default_browse_root(self) -> Path:
        parent = self.workspace_root.parent
        if parent.is_dir() and os.access(parent, os.R_OK | os.X_OK):
            return parent
        return Path.home().resolve()

    def _quick_roots(self) -> list[dict[str, str]]:
        home = Path.home().resolve()
        roots: list[dict[str, str]] = []
        seen: set[str] = set()
        candidates = (
            ("Projects", self.workspace_root.parent),
            ("Home", home),
            ("Workspace", self.workspace_root),
        )
        for label, path in candidates:
            if label == "Projects" and path == home:
                continue
            canonical = str(path)
            if canonical in seen or not path.is_dir():
                continue
            seen.add(canonical)
            roots.append({"label": label, "path": canonical})
        return roots

    @staticmethod
    def _init_git_repo(root: Path) -> None:
        try:
            subprocess.run(
                ["git", "init"],
                cwd=root,
                check=True,
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(f"git init failed in {root}: {exc}") from exc

    def delete_workspace(self, workspace_id: str) -> dict[str, Any]:
        """Unregister a workspace (metadata only); disk files and runs are untouched."""
        if not self.store.delete_workspace(workspace_id):
            raise NotFoundError(f"Unknown workspace_id: {workspace_id}")
        return {"ok": True, "workspace_id": workspace_id, "deleted": True}

    def prepare_run_scope(
        self,
        *,
        workspace_id: str | None,
        workspace_root: str | None,
        workspace_mode: str | None,
        task_type: str | None,
        base_ref: str | None,
        scope_paths: list[str] | None,
        init_git: bool = False,
    ) -> RunScope:
        """Resolve and validate Scope x Type before any run row or LLM call exists."""
        mode = workspace_mode or "brownfield"
        kind = task_type or "feature"
        target: Path | None = None
        bound_id: str | None = None
        workspace_source: Literal["explicit", "server_default", "none"] = "none"
        if workspace_id:
            workspace = self.store.get_workspace(workspace_id)
            if workspace is None:
                raise ValueError(f"Unknown workspace_id: {workspace_id}")
            bound_id = str(workspace["workspace_id"])
            target = Path(str(workspace["root_path"]))
            workspace_source = "explicit"
        if workspace_root:
            candidate = _expand_user_path(workspace_root).resolve()
            if target is not None and target.resolve() != candidate:
                raise ValueError(
                    f"workspace_id '{workspace_id}' does not match workspace_root {candidate}"
                )
            target = candidate
            workspace_source = "explicit"

        if mode == "greenfield":
            if target is None:
                raise ValueError(
                    "greenfield requires a workspace_root target or a workspace_id to bind"
                )
            self._validate_greenfield_target(target)
            normalized_scope = self._normalize_scope_paths(scope_paths, target)
            target = self._materialize_greenfield_target(target)
            is_git_repo, default_branch = _detect_git_repo(target)
            if init_git and not is_git_repo:
                self._init_git_repo(target)
                is_git_repo, default_branch = _detect_git_repo(target)
            registered = self.store.register_workspace(
                name=target.name or "workspace",
                root_path=target,
                is_git_repo=is_git_repo,
                default_branch=default_branch,
                exist_ok=True,
            )
            bound_id = str(registered["workspace_id"])
        else:
            if target is not None and not target.is_dir():
                raise ValueError(f"workspace_root not found: {target}")
            if target is not None and bound_id is None:
                existing = self.store.get_workspace_by_root(target)
                if existing is not None:
                    bound_id = str(existing["workspace_id"])
            elif target is None and mode == "brownfield" and self.workspace_root.is_dir():
                bound_id = self.default_workspace_id
                target = self.workspace_root
                workspace_source = "server_default"
            containment_root = target if target is not None else self.workspace_root
            normalized_scope = self._normalize_scope_paths(scope_paths, containment_root)

        return RunScope(
            workspace_id=bound_id,
            workspace_root=target,
            workspace_mode=mode,
            task_type=kind,
            base_ref=(base_ref or "").strip() or None,
            scope_paths=normalized_scope,
            workspace_source=workspace_source,
        )

    @staticmethod
    def _validate_greenfield_target(target: Path) -> None:
        if target.exists():
            if not target.is_dir():
                raise ValueError(f"greenfield target is not a directory: {target}")
            if any(target.iterdir()):
                raise ValueError(
                    f"greenfield target directory must be empty: {target}"
                )
            return
        if not target.parent.is_dir():
            raise ValueError(
                f"greenfield parent directory does not exist: {target.parent}"
            )

    @staticmethod
    def _materialize_greenfield_target(target: Path) -> Path:
        if target.exists():
            return target
        try:
            target.mkdir()
        except OSError as exc:
            raise ValueError(f"Cannot create greenfield workspace {target}: {exc}") from exc
        return target

    @staticmethod
    def _normalize_scope_paths(scope_paths: list[str] | None, root: Path) -> list[str]:
        """Validate containment and return canonical workspace-relative POSIX paths."""
        if not scope_paths:
            return []
        resolved_root = Path(root).expanduser().resolve()
        normalized: list[str] = []
        for raw in scope_paths:
            candidate = raw.strip()
            if not candidate:
                raise ValueError("scope_paths entries must not be blank")
            resolved = (resolved_root / candidate).resolve()
            if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
                raise ValueError(
                    f"scope path '{raw}' must resolve strictly inside {resolved_root}"
                )
            relative = resolved.relative_to(resolved_root).as_posix()
            if relative not in normalized:
                normalized.append(relative)
        return normalized

    def _retry_lock_for(self, operation_id: str) -> threading.Lock:
        with self._retry_guard:
            return self._retry_locks.setdefault(operation_id, threading.Lock())

    def list_runs_payload(
        self,
        *,
        limit: int | None = 50,
        workspace_id: str | None = None,
        workspace_mode: str | None = None,
        task_type: str | None = None,
        ball_holder: str | None = None,
        include_archived: bool = False,
    ) -> dict[str, Any]:
        return build_board_payload(
            self.store,
            limit=limit,
            workspace_id=workspace_id,
            workspace_mode=workspace_mode,
            task_type=task_type,
            ball_holder=ball_holder,
            include_archived=include_archived,
        )

    def validated_board_payload(self, *, workspace_id: str | None = None) -> dict[str, Any]:
        """Validate one board snapshot against RunListResponse before SSE emission."""
        payload = self.list_runs_payload(workspace_id=workspace_id)
        return RunListResponse.model_validate(payload).model_dump(mode="json")

    def board_change_token(self, *, workspace_id: str | None = None) -> tuple[Any, ...]:
        """Combine SQLite board state with active worktree Git HEAD states for SSE polling."""
        head_states = tuple(
            (wt, _read_git_head_state(wt)) for wt in self.store.active_worktree_paths()
        )
        return (self.store.board_fingerprint(workspace_id=workspace_id), head_states)

    def _workspace_for(self, run_id: str) -> Path:
        return self.supervisor.resolve_workspace(run_id)

    def _projection_payload(self, run_id: str) -> dict[str, Any]:
        """Build and validate one projection before it crosses the HTTP seam."""
        proj = build_run_projection(
            self.store,
            self._workspace_for(run_id),
            run_id,
            artifacts_root=self.artifacts_root,
            supervised=self.supervisor.is_supervised(run_id),
        )
        return RunProjection.model_validate(proj).model_dump(mode="json", by_alias=True)

    def get_run_payload(self, run_id: str) -> dict[str, Any]:
        proj = self._projection_payload(run_id)
        if proj["status"] == "running":
            previews = self.supervisor.live_activity.snapshot(run_id)
            for task in proj["dag"]["nodes"]:
                entry = previews.get(task["task_id"])
                if task["status"] == "running" and entry and entry["attempt"] == task["attempt"]:
                    task["live_progress"]["preview"] = entry["preview"]
        return {"ok": True, "projection": proj}

    def get_events_payload(self, run_id: str, *, after_seq: int = 0) -> dict[str, Any]:
        snap = self.store.get_run(run_id)
        events = [
            e.model_dump(mode="json") for e in self.store.list_events(run_id, after_seq=after_seq)
        ]
        return {
            "ok": True,
            "run_id": run_id,
            "state_revision": snap["state_revision"],
            "status": snap["status"].value,
            "events": events,
        }

    def get_attempt_payload(self, run_id: str, task_id: str, attempt: int) -> dict[str, Any]:
        plan = self.store.get_run(run_id)["plan"]
        artifacts = read_recorded_attempt(self.artifacts_root / run_id, plan, task_id, attempt)
        if artifacts is None:
            raise NotFoundError("Recorded task attempt not found")
        return {"ok": True, "artifacts": artifacts}

    def execute_command(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Execute an idempotent HITL command and return updated projection."""
        action = str(body.get("action", "")).strip()
        if not action:
            raise ValueError("Missing required 'action' field")

        operation_id = str(body.get("operation_id") or new_id("op"))
        actor = str(body.get("actor") or "zeus")
        snap = self.store.get_run(run_id)
        expected_rev_raw = body.get("expected_state_revision")
        expected_rev = (
            int(expected_rev_raw) if expected_rev_raw is not None else int(snap["state_revision"])
        )
        target_hash: str | None = None
        if "target_hash" in body:
            raw_hash = body["target_hash"]
            target_hash = str(raw_hash) if raw_hash is not None else ""
        auto_advance = bool(body.get("auto_advance", True))
        sync_flag = bool(body.get("sync", False))
        # Omit default flags so legacy command rows replay cleanly while non-default
        # flag changes under the same operation_id still raise a payload conflict.
        dispatch_meta: dict[str, Any] = {}
        if not auto_advance:
            dispatch_meta["auto_advance"] = False
        if sync_flag:
            dispatch_meta["sync"] = True

        if action in ("pause", "resume"):
            cmd_result = self.control.set_paused(
                run_id,
                paused=action == "pause",
                operation_id=operation_id,
                expected_state_revision=expected_rev,
                actor=actor,
            )
            if action == "pause" and not cmd_result.get("deduplicated"):
                self.supervisor.cancel_active_turns(run_id, reason="human_paused")
        elif action == "approve_spec":
            supervised_run = self.supervisor.is_supervised(run_id)

            def _check_spec_ready_for_approval(
                current: Mapping[str, Any], conn: sqlite3.Connection | None
            ) -> None:
                pending_spec = current["spec"]
                if pending_spec is None:
                    return
                events = (
                    self.store.list_events_in_tx(conn, run_id)
                    if conn is not None
                    else self.store.list_events(run_id)
                )
                if has_unfulfilled_revision(events):
                    raise ValueError(
                        "GoalSpec is awaiting revision realignment before approval"
                    )
                if not supervised_run:
                    return
                spec_approved = self.store.is_approved(
                    run_id=run_id,
                    target=ApprovalTarget.GOAL_SPEC,
                    target_id=pending_spec.spec_id,
                    target_hash=pending_spec.content_hash,
                )
                try:
                    strict = load_workflow(
                        str(current["workflow_name"])
                    ).align_strict_canonical()
                except Exception:  # noqa: BLE001 - fallback for legacy workflows
                    strict = True
                if has_pending_resynthesis(
                    events,
                    pending_spec,
                    spec_approved=spec_approved,
                    strict_canonical=strict,
                ):
                    raise ValueError(
                        "GoalSpec is awaiting clarification resynthesis before approval"
                    )

            spec_result = self.control.confirm_spec(
                run_id=run_id,
                approver=actor,
                accept_risk=bool(body.get("accept_risk", False)),
                reason=str(body.get("reason", "")),
                expected_revision=expected_rev,
                expected_target_hash=target_hash,
                operation_id=operation_id,
                command_meta=dispatch_meta,
                pre_confirm_check=_check_spec_ready_for_approval,
            )
            cmd_result = (
                spec_result
                if isinstance(spec_result, dict)
                else {
                    "run_id": run_id,
                    "spec_id": spec_result.spec_id,
                    "spec_hash": spec_result.content_hash,
                    "status": spec_result.status.value,
                }
            )
        elif action == "approve_design":
            design_result = self.control.approve_design(
                run_id=run_id,
                approver=actor,
                reason=str(body.get("reason", "")),
                expected_revision=expected_rev,
                expected_target_hash=target_hash,
                operation_id=operation_id,
                command_meta=dispatch_meta,
            )
            cmd_result = (
                design_result
                if isinstance(design_result, dict)
                else design_result.model_dump(mode="json")
            )
        elif action == "approve_plan":
            plan_result = self.control.approve_plan(
                run_id=run_id,
                approver=actor,
                reason=str(body.get("reason", "")),
                expected_revision=expected_rev,
                expected_target_hash=target_hash,
                operation_id=operation_id,
                command_meta=dispatch_meta,
            )
            cmd_result = (
                plan_result
                if isinstance(plan_result, dict)
                else plan_result.model_dump(mode="json")
            )
        elif action == "clarify":
            run_snap = self.store.get_run(run_id)
            if run_snap.get("spec") is None:
                cmd_result = self.control.clarify_craft(
                    run_id=run_id,
                    answers=dict(body.get("answers") or {}),
                    resolutions=dict(body.get("resolutions") or {}),
                    clarification_note=str(body.get("clarification_note") or ""),
                    actor=actor,
                    expected_revision=expected_rev,
                    operation_id=operation_id,
                    command_meta=dispatch_meta,
                )
            else:
                clarify_result = self.control.clarify_spec(
                    run_id=run_id,
                    answers=dict(body.get("answers") or {}),
                    resolutions=dict(body.get("resolutions") or {}),
                    extra_out_of_scope=list(body.get("out_of_scope") or []),
                    clarification_note=str(body.get("clarification_note") or ""),
                    actor=actor,
                    expected_revision=expected_rev,
                    operation_id=operation_id,
                    command_meta=dispatch_meta,
                )
                cmd_result = (
                    clarify_result
                    if isinstance(clarify_result, dict)
                    else {
                        "run_id": run_id,
                        "version": clarify_result.version,
                        "spec_hash": clarify_result.content_hash,
                    }
                )
        elif action == "retry_task":
            task_id = str(body.get("task_id", "")).strip()
            if not task_id:
                raise ValueError("retry_task requires non-empty 'task_id'")
            cmd_result = self.control.retry_task(
                run_id,
                task_id,
                operation_id=operation_id,
                expected_state_revision=expected_rev,
                hint=str(body.get("hint", "")).strip(),
                actor=actor,
                command_meta=dispatch_meta,
            )
        elif action == "accept_risk":
            task_id = str(body.get("task_id", "")).strip()
            reason = str(body.get("reason", "")).strip()
            if not task_id:
                raise ValueError("accept_risk requires non-empty 'task_id'")
            if not reason:
                raise ValueError("accept_risk requires a non-empty 'reason'")
            cmd_result = self.control.accept_task_risk(
                run_id,
                task_id,
                operation_id=operation_id,
                expected_state_revision=expected_rev,
                reason=reason,
                actor=actor,
            )
        elif action == "reassign_human":
            task_id = str(body.get("task_id", "")).strip()
            if not task_id:
                raise ValueError("reassign_human requires non-empty 'task_id'")
            cmd_result = self.control.reassign_task_human(
                run_id,
                task_id,
                operation_id=operation_id,
                expected_state_revision=expected_rev,
                note=str(body.get("note", "")).strip(),
                actor=actor,
            )
        elif action == "request_revision":
            feedback = str(body.get("feedback", "")).strip()
            if not feedback:
                raise ValueError("request_revision requires non-empty 'feedback'")
            if auto_advance and self.supervisor.is_supervised(run_id):
                wf_def = load_workflow(snap["workflow_name"])
                if not any(node.type == "align_goal" for node in wf_def.nodes):
                    raise ValueError(f"Workflow {wf_def.name} has no align stage node")
            cmd_result = self.control.request_revision(
                run_id,
                feedback=feedback,
                expected_revision=expected_rev,
                operation_id=operation_id,
                actor=actor,
                command_meta=dispatch_meta,
            )
            if (
                auto_advance
                and not cmd_result.get("deduplicated")
                and self.supervisor.is_supervised(run_id)
            ):
                cmd_result = {
                    **cmd_result,
                    "workflow": self.supervisor.revise_run(
                        run_id, feedback=feedback, sync=sync_flag
                    ),
                }
        elif action == "reopen_run":
            feedback = str(body.get("feedback", "")).strip()
            if not feedback:
                raise ValueError("reopen_run requires non-empty 'feedback'")
            if auto_advance and self.supervisor.is_supervised(run_id):
                wf_def = load_workflow(snap["workflow_name"])
                if not any(node.type == "align_goal" for node in wf_def.nodes):
                    raise ValueError(f"Workflow {wf_def.name} has no align stage node")
            cmd_result = self.control.reopen_run(
                run_id,
                feedback=feedback,
                expected_revision=expected_rev,
                operation_id=operation_id,
                actor=actor,
                command_meta=dispatch_meta,
            )
            if (
                auto_advance
                and not cmd_result.get("deduplicated")
                and self.supervisor.is_supervised(run_id)
            ):
                cmd_result = {
                    **cmd_result,
                    "workflow": self.supervisor.revise_run(
                        run_id, feedback=feedback, sync=sync_flag
                    ),
                }
        elif action == "retry_workflow":
            retry_payload: dict[str, Any] = {"action": "retry_workflow"}
            if sync_flag:
                retry_payload["sync"] = True
            with self._retry_lock_for(operation_id):
                replay = self.store.lookup_command(
                    operation_id=operation_id,
                    run_id=run_id,
                    action="retry_workflow",
                    payload=retry_payload,
                )
                if replay is not None:
                    cmd_result = replay
                else:
                    pre_snap = self.store.get_run(run_id, expected_revision=expected_rev)
                    pre_rev = int(pre_snap["state_revision"])
                    workflow = self.supervisor.resume_run(run_id, sync=sync_flag)
                    cmd_result = self.control.complete_workflow_retry(
                        run_id,
                        operation_id=operation_id,
                        workflow=workflow,
                        pre_dispatch_revision=pre_rev,
                        actor=actor,
                        sync=sync_flag,
                    )
        elif action == "promote_to_execute":
            if snap.get("workspace_mode") == "spec_only" and not snap.get("workspace_root"):
                raise ValueError(
                    "spec_only runs require an explicit bound target workspace "
                    "before promoting to execute"
                )
            target_workflow = str(body.get("workflow_name") or "full_sdlc_tdd").strip()
            target_def = load_workflow(target_workflow)
            if target_def.stop_after != "execute" or target_def.node("execute") is None:
                raise ValueError(
                    f"Target workflow '{target_workflow}' must support stop_after='execute'"
                )
            cur_def = load_workflow(snap["workflow_name"])
            max_llm_calls = (
                int(body["max_llm_calls"]) if body.get("max_llm_calls") is not None else None
            )
            cmd_result = self.control.promote_to_execute(
                run_id,
                target_workflow=target_workflow,
                current_stop_after=cur_def.stop_after,
                operation_id=operation_id,
                expected_state_revision=expected_rev,
                max_llm_calls=max_llm_calls,
                auto_advance=auto_advance,
                sync=sync_flag,
                actor=actor,
            )
        else:
            raise ValueError(f"Unsupported HITL action: {action}")

        if (
            action
            in (
                "clarify",
                "approve_spec",
                "approve_design",
                "approve_plan",
                "promote_to_execute",
                "retry_task",
                "resume",
            )
            and auto_advance
            and not cmd_result.get("deduplicated")
            and (action == "promote_to_execute" or self.supervisor.is_supervised(run_id))
        ):
            cmd_result = {
                **cmd_result,
                "workflow": self.supervisor.resume_run(run_id, sync=sync_flag),
            }

        projection = self._projection_payload(run_id)
        return {
            "ok": True,
            "command_result": cmd_result,
            "projection": projection,
        }


class _NoStoreJSONResponse(JSONResponse):
    """JSON response that never allows a browser or proxy to cache API state."""

    def __init__(
        self,
        content: Any,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        media_type: str | None = None,
        background: Any = None,
    ) -> None:
        merged = dict(headers or {})
        merged.setdefault("Cache-Control", "no-store")
        super().__init__(
            content=content,
            status_code=status_code,
            headers=merged,
            media_type=media_type,
            background=background,
        )


def _error_response(status_code: int, message: str) -> _NoStoreJSONResponse:
    return _NoStoreJSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=message).model_dump(mode="json"),
    )


def _validation_summary(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "Invalid request payload"
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
    message = str(first.get("msg") or "Invalid request payload")
    return f"Invalid request: {location}: {message}" if location else f"Invalid request: {message}"


def _sse_frame(event_name: str, data: dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _serve_static(relpath: str) -> Response:
    """Serve one whitelisted asset from static/, refusing any escape of the root."""
    root = _STATIC_DIR.resolve()
    try:
        candidate: Path | None = (root / relpath).resolve()
    except (OSError, ValueError):
        candidate = None
    if candidate is None or not candidate.is_relative_to(root) or not candidate.is_file():
        raise NotFoundError(f"Unknown static asset: {relpath}")
    content_type = _STATIC_MIME_TYPES.get(candidate.suffix.lower())
    if content_type is None:
        raise NotFoundError(f"Unsupported static type: {candidate.suffix}")
    return Response(
        content=candidate.read_bytes(),
        headers={"Content-Type": content_type, "Cache-Control": "no-store"},
    )


def _index_response() -> Response:
    index_file = _STATIC_DIR / "index.html"
    if not index_file.exists():
        raise NotFoundError("Console static/index.html not found")
    return Response(
        content=index_file.read_bytes(),
        headers={"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store"},
    )


class _EventStreamResponse(StreamingResponse):
    media_type = "text/event-stream"


class _AriadneFastAPI(FastAPI):
    def openapi(self) -> dict[str, Any]:
        if self.openapi_schema is not None:
            return self.openapi_schema
        schema = super().openapi()
        for path_item in schema.get("paths", {}).values():
            if not isinstance(path_item, dict):
                continue
            for operation in path_item.values():
                if not isinstance(operation, dict):
                    continue
                responses = operation.get("responses")
                if isinstance(responses, dict):
                    responses.pop("422", None)
                    for code in ("400", "404", "409"):
                        err = responses.get(code)
                        if not isinstance(err, dict):
                            continue
                        content = err.get("content")
                        if (
                            isinstance(content, dict)
                            and "application/json" not in content
                            and len(content) == 1
                        ):
                            err["content"] = {"application/json": next(iter(content.values()))}
        components = schema.get("components", {}).get("schemas")
        if isinstance(components, dict):
            components.pop("HTTPValidationError", None)
            components.pop("ValidationError", None)
        self.openapi_schema = schema
        return schema


def create_fastapi_app(app: AriadneApp) -> FastAPI:
    """Create the ASGI application serving the Ariadne console over one service layer."""
    service = app
    api = _AriadneFastAPI(
        title="Olympus4SWE Ariadne API",
        version="0.1.0",
        default_response_class=_NoStoreJSONResponse,
        redirect_slashes=False,
    )

    @api.exception_handler(RequestValidationError)
    async def _handle_request_validation(
        request: Request, exc: RequestValidationError
    ) -> _NoStoreJSONResponse:
        return _error_response(400, _validation_summary(exc))

    @api.exception_handler(ValidationError)
    async def _handle_validation_error(
        request: Request, exc: ValidationError
    ) -> _NoStoreJSONResponse:
        return _error_response(500, str(exc))

    @api.exception_handler(NotFoundError)
    async def _handle_not_found(request: Request, exc: NotFoundError) -> _NoStoreJSONResponse:
        return _error_response(404, str(exc))

    @api.exception_handler(FileNotFoundError)
    async def _handle_file_not_found(
        request: Request, exc: FileNotFoundError
    ) -> _NoStoreJSONResponse:
        return _error_response(400, str(exc))

    @api.exception_handler(ConcurrencyConflictError)
    async def _handle_conflict(
        request: Request, exc: ConcurrencyConflictError
    ) -> _NoStoreJSONResponse:
        projection: dict[str, Any] | None = None
        parts = [part for part in request.url.path.split("/") if part]
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "runs":
            with contextlib.suppress(Exception):
                projection = service.get_run_payload(parts[2])["projection"]
        payload = ConflictResponse(
            error=str(exc),
            projection=(
                RunProjection.model_validate(projection) if projection is not None else None
            ),
        )
        return _NoStoreJSONResponse(
            status_code=409,
            content=payload.model_dump(mode="json", by_alias=True),
        )

    @api.exception_handler(ValueError)
    async def _handle_value_error(request: Request, exc: ValueError) -> _NoStoreJSONResponse:
        return _error_response(400, str(exc))

    @api.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> _NoStoreJSONResponse:
        return _error_response(500, str(exc))

    @api.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> _NoStoreJSONResponse:
        message = (
            f"Unknown route: {request.url.path}"
            if exc.status_code == 404
            else str(exc.detail)
        )
        return _error_response(exc.status_code, message)

    error_responses: dict[int | str, dict[str, Any]] = {400: {"model": ErrorResponse}}
    read_responses: dict[int | str, dict[str, Any]] = {
        **error_responses,
        404: {"model": ErrorResponse},
    }
    write_responses: dict[int | str, dict[str, Any]] = {
        **read_responses,
        409: {"model": ConflictResponse},
    }

    @api.get("/healthz", response_model=HealthResponse)
    def healthz() -> dict[str, Any]:
        return {"ok": True, "service": "olympus-ariadne", "version": "0.1.0"}

    @api.get("/api/runs", response_model=RunListResponse, responses=error_responses)
    def list_runs(
        limit: int = 50,
        workspace_id: str | None = None,
        workspace_mode: WorkspaceMode | None = None,
        task_type: TaskType | None = None,
        ball_holder: BallHolder | None = None,
        include_archived: bool = False,
    ) -> dict[str, Any]:
        return service.list_runs_payload(
            limit=limit,
            workspace_id=workspace_id,
            workspace_mode=workspace_mode,
            task_type=task_type,
            ball_holder=ball_holder,
            include_archived=include_archived,
        )

    @api.get("/api/providers", response_model=ProvidersResponse, responses=error_responses)
    def list_providers() -> dict[str, Any]:
        return {"ok": True, "providers": service.supervisor.list_providers_status()}

    @api.get("/api/workflows", response_model=WorkflowsResponse, responses=error_responses)
    def list_workflows() -> dict[str, Any]:
        return {
            "ok": True,
            "workflows": service.supervisor.list_workflows(),
            "default_workflows": default_workflow_matrix(),
        }

    @api.get("/api/demos", response_model=DemosResponse, responses=error_responses)
    def list_demos() -> dict[str, Any]:
        return {"ok": True, "demos": list_benchmark_presets()}

    @api.get(
        "/api/runs/{run_id}",
        response_model=RunProjectionResponse,
        responses=read_responses,
    )
    def get_run(run_id: str) -> dict[str, Any]:
        return service.get_run_payload(run_id)

    @api.get(
        "/api/runs/{run_id}/events",
        response_model=EventsResponse,
        responses=read_responses,
    )
    def get_run_events(run_id: str, after_seq: int = 0) -> dict[str, Any]:
        return service.get_events_payload(run_id, after_seq=after_seq)

    @api.get(
        "/api/runs/{run_id}/task-artifacts/{task_id}/{attempt}",
        response_model=AttemptArtifactsResponse,
        responses=read_responses,
    )
    def get_task_artifacts(run_id: str, task_id: str, attempt: int) -> dict[str, Any]:
        return service.get_attempt_payload(run_id, task_id, attempt)

    @api.get(
        "/api/runs/{run_id}/delivery-log/{event_seq}/{position}",
        response_class=PlainTextResponse,
        responses=read_responses,
    )
    def get_delivery_log(run_id: str, event_seq: str, position: str) -> Response:
        projection = service.get_run_payload(run_id)["projection"]
        review = projection["delivery_acceptance"] or {}
        for row in review.get("criteria", []):
            evidence = row["evidence"]
            if (
                evidence
                and str(evidence["event_seq"]) == event_seq
                and str(evidence["check_position"]) == position
            ):
                log = delivery_review.read_delivery_log(service.artifacts_root / run_id, evidence)
                if log is not None:
                    return Response(
                        content=log,
                        headers={
                            "Content-Type": "text/plain; charset=utf-8",
                            "Cache-Control": "no-store",
                        },
                    )
        raise NotFoundError("No matching current delivery evidence")

    @api.get(
        "/api/runs/{run_id}/product-artifact/{kind}",
        response_class=Response,
        responses={
            **read_responses,
            200: {
                "description": "Allowlisted product verification artifact",
                "content": {
                    "application/json": {
                        "schema": {"type": "object", "additionalProperties": True}
                    },
                    "text/plain": {"schema": {"type": "string"}},
                    "image/png": {"schema": {"type": "string", "format": "binary"}},
                },
            },
        },
    )
    def get_product_artifact(
        run_id: str,
        kind: Literal["report", "boot_log", "screenshot"],
        v: int | None = None,
        sha: str | None = None,
    ) -> Response:
        service.store.get_run(run_id)
        expected_sha = sha.strip().lower() if isinstance(sha, str) and sha.strip() else None
        if v is not None or kind == "screenshot":
            events = service.store.list_events(run_id)
            pv = delivery_review.latest_product_verification(
                events,
                service.artifacts_root / run_id,
                artifacts_root=service.artifacts_root,
            )
            if v is not None and (pv is None or pv["event_seq"] != v):
                raise NotFoundError("Product verification artifact does not match requested event")
            if kind == "screenshot" and pv is not None:
                if not pv["has_screenshot"]:
                    raise NotFoundError("Product verification screenshot is not available")
                recorded_sha = pv.get("screenshot_sha256")
                if isinstance(recorded_sha, str) and recorded_sha:
                    if expected_sha is not None and expected_sha != recorded_sha:
                        raise NotFoundError("Product verification screenshot has changed")
                    expected_sha = recorded_sha
        artifact = delivery_review.read_product_artifact(
            service.artifacts_root / run_id,
            kind,
            artifacts_root=service.artifacts_root,
            expected_sha256=expected_sha,
        )
        if artifact is None:
            raise NotFoundError("Product verification artifact is not available")
        content, media_type = artifact
        return Response(
            content=content,
            media_type=media_type,
            headers={"Cache-Control": "no-store"},
        )

    @api.get("/api/workspaces", response_model=WorkspaceListResponse, responses=error_responses)
    def list_workspaces() -> dict[str, Any]:
        return service.list_workspaces_payload()

    @api.post("/api/workspaces", response_model=WorkspaceResponse, responses=error_responses)
    def create_workspace(payload: CreateWorkspaceRequest) -> dict[str, Any]:
        return service.create_workspace(payload)

    @api.post(
        "/api/workspaces/clone", response_model=WorkspaceResponse, responses=error_responses
    )
    def clone_workspace(payload: CloneWorkspaceRequest) -> dict[str, Any]:
        return service.clone_workspace(payload)

    @api.get(
        "/api/filesystem/browse",
        response_model=FilesystemBrowseResponse,
        responses=error_responses,
    )
    def browse_filesystem(
        request: Request,
        partial_path: str | None = None,
        show_hidden: bool = False,
    ) -> dict[str, Any]:
        del partial_path, show_hidden
        try:
            query = FilesystemBrowseQuery.model_validate(dict(request.query_params))
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(part) for part in first.get("loc", ()))
            detail = f"{loc}: {first['msg']}" if loc else str(first["msg"])
            raise ValueError(f"Invalid browse query ({detail})") from exc
        return service.browse_filesystem(
            partial_path=query.partial_path, show_hidden=query.show_hidden
        )

    @api.delete(
        "/api/workspaces/{workspace_id}",
        response_model=DeleteWorkspaceResponse,
        responses=read_responses,
    )
    def delete_workspace(workspace_id: str) -> dict[str, Any]:
        return service.delete_workspace(workspace_id)

    @api.post(
        "/api/runs",
        response_model=CreateRunResponse,
        responses=write_responses,
    )
    def create_run(payload: CreateRunRequest) -> dict[str, Any]:
        demo_id = (payload.demo_id or "").strip()
        preset: dict[str, Any] | None = None
        requested_root = payload.workspace_root
        if demo_id:
            preset, demo_ws = materialize_benchmark_demo(
                demo_id,
                workspace_parent=service.workspace_root / ".olympus" / "workspaces",
            )
            if not requested_root:
                requested_root = str(demo_ws)

        default_goal = str(preset["raw_input"]) if preset else ""
        raw_input = str(payload.raw_input or payload.goal or default_goal).strip()
        if not raw_input:
            raise ValueError("POST /api/runs requires a non-empty 'raw_input'")
        if payload.constraints is not None:
            constraints = [item.strip() for item in payload.constraints if item.strip()]
        elif preset:
            constraints = list(preset.get("constraints") or [])
        else:
            constraints = []
        default_title = str(preset["title"]) if preset else raw_input[:60]
        resolved_mode = payload.workspace_mode or "brownfield"
        resolved_type = payload.task_type or "feature"
        if payload.workflow_name:
            resolved_workflow = payload.workflow_name
        elif (
            payload.workspace_mode is not None
            or payload.task_type is not None
            or payload.workspace_id is not None
        ):
            resolved_workflow = _default_workflow_for(resolved_mode, resolved_type)
        elif preset:
            resolved_workflow = str(preset["default_workflow"])
        else:
            resolved_workflow = "product_blueprint"
        workflow_def = load_workflow(resolved_workflow)
        if (
            resolved_mode == "spec_only"
            and payload.workspace_id is None
            and not requested_root
            and workflow_def.stop_after == "execute"
        ):
            raise ValueError(
                "spec_only runs cannot enter an execution workflow "
                "without binding an explicit target workspace"
            )

        scope = service.prepare_run_scope(
            workspace_id=payload.workspace_id,
            workspace_root=requested_root,
            workspace_mode=payload.workspace_mode,
            task_type=payload.task_type,
            base_ref=payload.base_ref,
            scope_paths=payload.scope_paths,
            init_git=payload.init_git,
        )
        started = service.supervisor.start_run(
            title=str(payload.title or default_title),
            raw_input=raw_input,
            workflow_name=resolved_workflow,
            provider_config=payload.provider_config.model_dump(exclude_none=True),
            worker_driver=payload.worker_driver,
            reviewer_driver=payload.reviewer_driver,
            auto_approve=payload.auto_approve,
            sync=payload.sync,
            constraints=constraints,
            workspace_root=scope.workspace_root,
            workspace_id=scope.workspace_id,
            workspace_mode=scope.workspace_mode,
            task_type=scope.task_type,
            base_ref=scope.base_ref,
            scope_paths=scope.scope_paths,
        )
        return {"ok": True, **started}

    @api.post(
        "/api/runs/{run_id}/command",
        response_model=CommandResponse,
        responses=write_responses,
    )
    def run_command(run_id: str, payload: CommandRequest) -> dict[str, Any]:
        body = payload.model_dump(mode="json", exclude_unset=True)
        return service.execute_command(run_id, body)

    @api.post("/api/seed-demo", response_model=SeedDemoResponse, responses=write_responses)
    async def seed_demo(request: Request) -> dict[str, Any]:
        raw_body = await request.body()
        if raw_body.strip():
            try:
                parsed = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid JSON payload: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError("POST /api/seed-demo expects a JSON object")
        seeded = seed_demo_runs(
            service.store,
            service.workspace_root,
            artifacts_root=service.artifacts_root,
        )
        return {"ok": True, "seeded": seeded, **service.list_runs_payload(include_archived=True)}

    @api.get(
        "/api/runs/{run_id}/stream",
        response_class=_EventStreamResponse,
        responses={
            200: {
                "description": "Server-Sent Events stream",
                "content": {"text/event-stream": {"schema": {"type": "string"}}},
            },
            **read_responses,
        },
    )
    async def stream_run(
        run_id: str, request: Request, after_seq: int = 0, once: int = 0
    ) -> _EventStreamResponse:
        initial = service.get_events_payload(run_id, after_seq=after_seq)
        return _EventStreamResponse(
            _sse_stream(
                service,
                request,
                run_id,
                initial=initial,
                after_seq=after_seq,
                once=once == 1,
            ),
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "close",
                "X-Accel-Buffering": "no",
            },
        )

    @api.get(
        "/api/board/stream",
        response_class=_EventStreamResponse,
        responses={
            200: {
                "description": "Server-Sent Events board stream",
                "content": {"text/event-stream": {"schema": {"type": "string"}}},
            },
            **error_responses,
        },
    )
    async def stream_board(
        request: Request, workspace_id: str | None = None, once: int = 0
    ) -> _EventStreamResponse:
        return _EventStreamResponse(
            _board_sse_stream(
                service,
                request,
                workspace_id=workspace_id,
                once=once == 1,
            ),
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "close",
                "X-Accel-Buffering": "no",
            },
        )

    @api.get("/static/{relpath:path}", include_in_schema=False)
    def static_asset(relpath: str) -> Response:
        return _serve_static(relpath)

    @api.get("/", include_in_schema=False)
    @api.get("/index.html", include_in_schema=False)
    def index() -> Response:
        return _index_response()

    @api.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(
            content=b"",
            headers={"Content-Type": "image/x-icon", "Cache-Control": "no-store"},
        )

    return api


async def _sse_stream(
    service: AriadneApp,
    request: Request,
    run_id: str,
    *,
    initial: dict[str, Any],
    after_seq: int,
    once: bool,
) -> AsyncIterator[str]:
    """Async SSE frames matching the legacy wire contract exactly."""
    yield _sse_frame("state", initial)
    if once:
        return

    last_rev = int(initial["state_revision"])
    events = initial["events"]
    cur_seq = int(events[-1]["event_seq"]) if events else after_seq
    last_previews: dict[str, Any] = {}
    ticks = 0
    while True:
        await asyncio.sleep(0.5)
        if await request.is_disconnected():
            return
        ticks += 1
        nxt = await asyncio.to_thread(service.get_events_payload, run_id, after_seq=cur_seq)
        if int(nxt["state_revision"]) != last_rev or nxt["events"]:
            last_rev = int(nxt["state_revision"])
            if nxt["events"]:
                cur_seq = int(nxt["events"][-1]["event_seq"])
            yield _sse_frame("state", nxt)
        elif ticks % 10 == 0:
            yield ": heartbeat\n\n"
        previews = (
            service.supervisor.live_activity.snapshot(run_id)
            if nxt["status"] == "running"
            else {}
        )
        if previews != last_previews:
            yield _sse_frame("activity", {"run_id": run_id, "previews": previews})
            last_previews = previews


async def _board_sse_stream(
    service: AriadneApp,
    request: Request,
    *,
    workspace_id: str | None,
    once: bool,
) -> AsyncIterator[str]:
    """Board snapshot stream: 0.5s SQLite polls, aggregate-change emissions, 5s heartbeats."""
    last_token = await asyncio.to_thread(
        service.board_change_token, workspace_id=workspace_id
    )
    initial = await asyncio.to_thread(
        service.validated_board_payload, workspace_id=workspace_id
    )
    yield _sse_frame("board", initial)
    if once:
        return

    last_signature = json.dumps(initial, ensure_ascii=False, sort_keys=True)
    ticks = 0
    while True:
        await asyncio.sleep(0.5)
        if await request.is_disconnected():
            return
        ticks += 1
        token = await asyncio.to_thread(
            service.board_change_token, workspace_id=workspace_id
        )
        if token == last_token:
            if ticks % 10 == 0:
                yield ": heartbeat\n\n"
            continue
        last_token = token
        nxt = await asyncio.to_thread(
            service.validated_board_payload, workspace_id=workspace_id
        )
        signature = json.dumps(nxt, ensure_ascii=False, sort_keys=True)
        if signature != last_signature:
            last_signature = signature
            yield _sse_frame("board", nxt)
        elif ticks % 10 == 0:
            yield ": heartbeat\n\n"


def serve_ariadne(
    *,
    store: RunStore,
    workspace_root: Path | str,
    host: str = "127.0.0.1",
    port: int = 8765,
    seed_demo: bool = False,
) -> None:
    """Run the Ariadne console uvicorn server in the foreground."""
    ws = Path(workspace_root).resolve()
    if seed_demo:
        seed_demo_runs(store, ws)
    app = AriadneApp(store=store, workspace_root=ws)
    app.supervisor.recover_orphaned_runs()
    uvicorn.run(create_fastapi_app(app), host=host, port=port)


__all__ = [
    "AriadneApp",
    "create_fastapi_app",
    "serve_ariadne",
]
