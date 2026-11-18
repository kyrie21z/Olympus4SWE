"""Isolated git worktree sandbox management and diff/commit inspection."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


def build_sandbox_env(
    workspace_root: Path | str,
    *,
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
    return env


@dataclass(frozen=True)
class SandboxHandle:
    repo_root: Path
    workspace_path: Path
    base_commit: str | None
    is_git_worktree: bool


def _run_git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def is_git_repo(path: Path) -> bool:
    proc = _run_git(path, "rev-parse", "--is-inside-work-tree")
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def create_sandbox(repo_root: Path | str, *, label: str) -> SandboxHandle:
    """Create an isolated git worktree when inside a git repo, else copy workspace."""
    root = Path(repo_root).resolve()
    safe_label = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)
    sandbox_dir = root / ".olympus" / "sandboxes" / safe_label
    if sandbox_dir.exists():
        remove_sandbox(SandboxHandle(root, sandbox_dir, None, is_git_repo(root)))

    if is_git_repo(root):
        head_proc = _run_git(root, "rev-parse", "HEAD")
        if head_proc.returncode == 0:
            base_commit = head_proc.stdout.strip()
            sandbox_dir.parent.mkdir(parents=True, exist_ok=True)
            add_proc = _run_git(
                root,
                "worktree",
                "add",
                "--detach",
                str(sandbox_dir),
                base_commit,
            )
            if add_proc.returncode == 0:
                return SandboxHandle(
                    repo_root=root,
                    workspace_path=sandbox_dir,
                    base_commit=base_commit,
                    is_git_worktree=True,
                )

    # Fallback for uninitialized or unborn-HEAD repos
    sandbox_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        root,
        sandbox_dir,
        ignore=shutil.ignore_patterns(".git", ".venv", ".olympus", "__pycache__"),
    )
    return SandboxHandle(
        repo_root=root,
        workspace_path=sandbox_dir,
        base_commit=None,
        is_git_worktree=False,
    )


def capture_sandbox_diff(handle: SandboxHandle) -> str:
    """Return tracked diff plus untracked file list/contents for cold-start review."""
    ws = handle.workspace_path
    if not is_git_repo(ws):
        return "(non-git sandbox; inspect modified files directly)"
    tracked = _run_git(ws, "diff", "HEAD").stdout
    untracked_proc = _run_git(ws, "ls-files", "--others", "--exclude-standard")
    parts: list[str] = []
    if tracked.strip():
        parts.append(tracked.rstrip())
    untracked = [line.strip() for line in untracked_proc.stdout.splitlines() if line.strip()]
    for rel in untracked:
        fpath = ws / rel
        if fpath.is_file():
            try:
                content = fpath.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = "<binary file>"
            parts.append(f"--- /dev/null\n+++ b/{rel}\n{content}")
    return "\n\n".join(parts) if parts else "(no changes)"


def remove_sandbox(handle: SandboxHandle) -> None:
    """Clean up a worktree or copied sandbox directory."""
    if handle.is_git_worktree and is_git_repo(handle.repo_root):
        _run_git(handle.repo_root, "worktree", "remove", "--force", str(handle.workspace_path))
    if handle.workspace_path.exists():
        shutil.rmtree(handle.workspace_path, ignore_errors=True)
