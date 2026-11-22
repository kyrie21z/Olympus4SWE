"""Isolated git worktree sandbox management and diff/commit inspection."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from olympus.tools.aci import IGNORED_DIR_NAMES
from olympus.tools.aci import build_sandbox_env as build_sandbox_env


@dataclass(frozen=True)
class SandboxHandle:
    repo_root: Path
    workspace_path: Path
    base_commit: str | None
    is_git_worktree: bool


@dataclass(frozen=True)
class SandboxBaseline:
    """Fingerprints of the sandbox state handed to a worker before its turn."""

    is_git: bool
    entries: tuple[tuple[str, str | None], ...]

    def mapping(self) -> dict[str, str | None]:
        return dict(self.entries)


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


def capture_sandbox_diff(
    handle: SandboxHandle, baseline: SandboxBaseline | None = None
) -> str:
    """Return tracked diff plus untracked file contents for cold-start review.

    Supplying the task's pre-worker ``baseline`` restricts the output to paths
    mutated since that snapshot, so unchanged upstream ``accepted_overlay``
    files never leak into a downstream task's diff. Without a baseline the
    legacy ``HEAD`` + untracked behavior is preserved for direct callers.
    """
    ws = handle.workspace_path
    if not is_git_repo(ws):
        return "(non-git sandbox; inspect modified files directly)"
    if baseline is None:
        return _head_scoped_diff(ws)
    return _baseline_scoped_diff(handle, baseline)


def _head_scoped_diff(ws: Path) -> str:
    tracked = _run_git(ws, "diff", "HEAD").stdout
    untracked_proc = _run_git(ws, "ls-files", "--others", "--exclude-standard")
    parts: list[str] = []
    if tracked.strip():
        parts.append(tracked.rstrip())
    untracked = [line.strip() for line in untracked_proc.stdout.splitlines() if line.strip()]
    for rel in untracked:
        fpath = ws / rel
        if fpath.is_file():
            parts.append(f"--- /dev/null\n+++ b/{rel}\n{_read_text(fpath)}")
    return "\n\n".join(parts) if parts else "(no changes)"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "<binary file>"


def _prefixed_content(path: Path) -> str:
    text = _read_text(path)
    lines = text.splitlines()
    if not lines:
        return ""
    return "\n".join(f"+{line}" for line in lines) + "\n"


def _new_file_block(ws: Path, rel: str) -> str:
    return (
        f"diff --git a/{rel} b/{rel}\n"
        "new file mode 100644\n"
        f"--- /dev/null\n+++ b/{rel}\n"
        f"{_prefixed_content(ws / rel)}"
    )


def _deleted_file_block(ws: Path, rel: str) -> str:
    tracked = _run_git(ws, "diff", "HEAD", "--", rel).stdout.rstrip()
    if tracked:
        return tracked
    return (
        f"diff --git a/{rel} b/{rel}\n"
        "deleted file mode 100644\n"
        f"--- a/{rel}\n+++ /dev/null"
    )


def _modified_file_block(ws: Path, rel: str) -> str:
    tracked = _run_git(ws, "ls-files", "--error-unmatch", "--", rel)
    if tracked.returncode == 0:
        return _run_git(ws, "diff", "HEAD", "--", rel).stdout.rstrip()
    # Baseline-only (overlay) file: no recorded baseline content, so show the full
    # current file as added lines rather than dropping the mutation from review.
    return (
        f"diff --git a/{rel} b/{rel}\n"
        f"--- a/{rel}\n+++ b/{rel}\n"
        f"{_prefixed_content(ws / rel)}"
    )


def _baseline_scoped_diff(handle: SandboxHandle, baseline: SandboxBaseline) -> str:
    ws = handle.workspace_path
    before = baseline.mapping()
    changed = collect_sandbox_changes(handle, baseline)
    if not changed:
        return "(no changes)"

    def present(rel: str) -> bool:
        candidate = ws / rel
        return candidate.is_file() or candidate.is_symlink()

    added = {rel for rel in changed if rel not in before and present(rel)}
    deleted = {rel for rel in changed if rel in before and not present(rel)}
    modified = set(changed) - added - deleted

    renames: dict[str, str] = {}
    for old in sorted(deleted):
        fingerprint = before[old]
        if fingerprint is None:
            continue
        match = next(
            (new for new in sorted(added) if _file_fingerprint(ws / new) == fingerprint),
            None,
        )
        if match is not None:
            renames[old] = match
            added.discard(match)

    blocks: list[str] = []
    for rel in sorted(changed):
        if rel in renames:
            new = renames[rel]
            blocks.append(
                f"diff --git a/{rel} b/{new}\n"
                "similarity index 100%\n"
                f"rename from {rel}\nrename to {new}"
            )
        elif rel in renames.values():
            continue
        elif rel in added:
            blocks.append(_new_file_block(ws, rel))
        elif rel in deleted:
            blocks.append(_deleted_file_block(ws, rel))
        elif rel in modified:
            blocks.append(_modified_file_block(ws, rel))
    return "\n\n".join(blocks) if blocks else "(no changes)"


def archive_sandbox_recovery(handle: SandboxHandle, attempt_dir: Path) -> None:
    """Record recoverable binary and text changes from a paused attempt."""
    patch = attempt_dir / "diff.patch"
    if handle.base_commit is None:
        patch.write_text("Non-git workspace preserved in sandbox/\n", encoding="utf-8")
        shutil.move(handle.workspace_path, attempt_dir / "sandbox")
        return

    ws = handle.workspace_path
    diff = ["git", "diff", "--binary", "--no-ext-diff", "--no-textconv"]
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=ws, capture_output=True, check=True,
    ).stdout
    with patch.open("wb") as output:
        subprocess.run([*diff, handle.base_commit, "--"], cwd=ws, stdout=output, check=True)
        for raw_path in untracked.split(b"\0"):
            if not raw_path:
                continue
            result = subprocess.run(
                [*diff, "--no-index", "--", "/dev/null", os.fsdecode(raw_path)],
                cwd=ws, stdout=output, check=False,
            )
            if result.returncode != 1:
                result.check_returncode()


def _is_ignored_rel(rel: str) -> bool:
    return any(part in IGNORED_DIR_NAMES for part in PurePosixPath(rel).parts)


def _file_fingerprint(path: Path) -> str | None:
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if stat.S_ISLNK(info.st_mode):
        try:
            return f"link:{os.readlink(path)}"
        except OSError:
            return None
    if not stat.S_ISREG(info.st_mode):
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)
    except OSError:
        return None
    return f"{info.st_mode & 0o777:o}:{digest.hexdigest()}"


def _workspace_fingerprints(root: Path) -> dict[str, str | None]:
    fingerprints: dict[str, str | None] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIR_NAMES)
        for name in sorted(filenames):
            candidate = Path(dirpath) / name
            rel = candidate.relative_to(root).as_posix()
            if _is_ignored_rel(rel):
                continue
            fingerprints[rel] = _file_fingerprint(candidate)
    return fingerprints


def _git_changed_fingerprints(ws: Path, base_commit: str) -> dict[str, str | None]:
    proc = _run_git(ws, "diff", "--name-status", "-z", "-M", base_commit)
    if proc.returncode != 0:
        raise RuntimeError(
            f"git diff against {base_commit} failed: {proc.stderr.strip()[:200]}"
        )
    entries: dict[str, str | None] = {}
    tokens = proc.stdout.split("\0")
    index = 0
    while index < len(tokens):
        status = tokens[index]
        index += 1
        if not status:
            continue
        if status[0] in ("R", "C"):
            old_path, new_path = tokens[index], tokens[index + 1]
            index += 2
            if status[0] == "R" and old_path:
                # The rename deletes the old path; the deletion is itself a mutation.
                entries[old_path] = _file_fingerprint(ws / old_path)
            if new_path:
                entries[new_path] = _file_fingerprint(ws / new_path)
            continue
        rel = tokens[index]
        index += 1
        if rel:
            entries[rel] = _file_fingerprint(ws / rel)

    untracked = _run_git(ws, "ls-files", "--others", "--exclude-standard", "-z")
    if untracked.returncode != 0:
        raise RuntimeError(
            f"git ls-files failed: {untracked.stderr.strip()[:200]}"
        )
    for rel in untracked.stdout.split("\0"):
        if rel and not _is_ignored_rel(rel):
            entries[rel] = _file_fingerprint(ws / rel)
    return entries


def snapshot_sandbox_state(handle: SandboxHandle) -> SandboxBaseline:
    """Capture the sandbox state before a worker runs so later mutations are attributable."""
    ws = handle.workspace_path
    if handle.is_git_worktree and handle.base_commit is not None and is_git_repo(ws):
        try:
            entries = _git_changed_fingerprints(ws, handle.base_commit)
        except RuntimeError:
            entries = _workspace_fingerprints(ws)
            return SandboxBaseline(is_git=False, entries=tuple(sorted(entries.items())))
        return SandboxBaseline(is_git=True, entries=tuple(sorted(entries.items())))
    return SandboxBaseline(
        is_git=False, entries=tuple(sorted(_workspace_fingerprints(ws).items()))
    )


def collect_sandbox_changes(handle: SandboxHandle, baseline: SandboxBaseline) -> list[str]:
    """Return repo-relative paths mutated since ``baseline`` (adds, edits, deletes, renames)."""
    ws = handle.workspace_path
    if baseline.is_git and handle.base_commit is not None and is_git_repo(ws):
        try:
            current = _git_changed_fingerprints(ws, handle.base_commit)
        except RuntimeError:
            # A tampered sandbox (e.g. deleted .git) fails closed: every file looks changed.
            current = _workspace_fingerprints(ws)
    else:
        current = _workspace_fingerprints(ws)

    before = baseline.mapping()
    changed = {
        rel
        for rel, fingerprint in current.items()
        if rel not in before or before[rel] != fingerprint
    }
    changed.update(rel for rel in before if rel not in current)
    return sorted(changed)


def remove_sandbox(handle: SandboxHandle) -> None:
    """Clean up a worktree or copied sandbox directory."""
    if handle.is_git_worktree and is_git_repo(handle.repo_root):
        _run_git(handle.repo_root, "worktree", "remove", "--force", str(handle.workspace_path))
    if handle.workspace_path.exists():
        shutil.rmtree(handle.workspace_path, ignore_errors=True)
