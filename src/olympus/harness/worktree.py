"""Isolated git worktree sandbox management and diff/commit inspection."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
import stat
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

from olympus.tools.aci import IGNORED_DIR_NAMES, WorkspacePolicy, sandbox_tmp_path
from olympus.tools.aci import build_sandbox_env as build_sandbox_env

EPHEMERAL_ARTIFACT_PATTERNS: tuple[str, ...] = (
    "*.backup",
    "*.bak",
    "*.orig",
    "*.tmp",
    "*~",
    ".*.swp",
    ".DS_Store",
    "**/__pycache__/**",
    "*.pyc",
    "**/.pytest_cache/**",
)
_RUNTIME_ARTIFACT_SUFFIXES = (
    ".db", ".db-wal", ".db-shm", ".db-journal",
    ".sqlite", ".sqlite-wal", ".sqlite-shm", ".sqlite-journal",
    ".sqlite3", ".sqlite3-wal", ".sqlite3-shm", ".sqlite3-journal",
    ".log",
)
_UPLOAD_ARTIFACT_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".pdf", ".bin"}
)


@dataclass(frozen=True)
class SandboxHandle:
    repo_root: Path
    workspace_path: Path
    base_commit: str | None
    is_git_worktree: bool
    reused: bool = False

    @property
    def tmp_dir(self) -> Path:
        return sandbox_tmp_path(self.workspace_path, self.repo_root)

    @property
    def baseline_path(self) -> Path:
        if not self.workspace_path.resolve().is_relative_to(self.repo_root.resolve()):
            return (
                self.workspace_path.resolve().parent
                / ".sandbox_baselines"
                / f"{self.workspace_path.name}.json"
            )
        return (
            self.repo_root
            / ".olympus"
            / "sandbox_baselines"
            / f"{self.workspace_path.name}.json"
        )


@dataclass(frozen=True)
class SandboxBaseline:
    """Fingerprints of the sandbox state handed to a worker before its turn."""

    is_git: bool
    entries: tuple[tuple[str, str | None], ...]
    base_commit: str | None = None
    preexisting_ephemeral: tuple[str, ...] = ()
    preexisting_dirs: tuple[str, ...] | None = None
    text_snapshots: tuple[tuple[str, str], ...] = ()

    def mapping(self) -> dict[str, str | None]:
        return dict(self.entries)

    def text_mapping(self) -> dict[str, str]:
        return dict(self.text_snapshots)


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


def _safe_label(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in label)


def _open_retained_sandbox(root: Path, sandbox_dir: Path) -> SandboxHandle | None:
    if is_git_repo(root):
        if not is_git_repo(sandbox_dir):
            return None
        probe = SandboxHandle(root, sandbox_dir, None, is_git_worktree=True, reused=True)
        saved = load_sandbox_baseline(probe)
        if saved is not None and saved.base_commit is not None:
            base_commit = saved.base_commit
        else:
            head_proc = _run_git(sandbox_dir, "rev-parse", "HEAD")
            if head_proc.returncode != 0:
                return None
            base_commit = head_proc.stdout.strip()
        return SandboxHandle(
            root, sandbox_dir, base_commit, is_git_worktree=True, reused=True
        )
    return SandboxHandle(root, sandbox_dir, None, is_git_worktree=False, reused=True)


def create_sandbox(
    repo_root: Path | str,
    *,
    label: str,
    reuse: bool = False,
    sandbox_root: Path | str | None = None,
    base_ref: str | None = None,
) -> SandboxHandle:
    """Create an isolated git worktree when inside a git repo, else copy workspace.

    With ``reuse=True`` a healthy retained sandbox is reopened as-is so a retried
    attempt keeps every uncommitted file from its previous turn. ``sandbox_root``
    keeps sandbox worktrees (and their baselines/tmp dirs) outside the target
    repository so headless sessions never pollute the repo with ``.olympus``.
    When ``base_ref`` is provided, creation fails closed if ``repo_root`` is not a
    git repository, ``base_ref`` cannot be resolved to a commit, or ``git worktree
    add`` fails.
    """
    root = Path(repo_root).resolve()
    safe_label = _safe_label(label)
    sandbox_dir = (
        Path(sandbox_root).resolve() / safe_label
        if sandbox_root is not None
        else root / ".olympus" / "sandboxes" / safe_label
    )
    explicit_ref = base_ref.strip() if base_ref is not None and base_ref.strip() else None
    git_repo = is_git_repo(root)
    if explicit_ref is not None and not git_repo:
        raise ValueError(
            f"Explicit base_ref {explicit_ref!r} cannot be resolved because "
            f"{root} is not a git repository"
        )

    if sandbox_dir.exists():
        if reuse:
            retained = _open_retained_sandbox(root, sandbox_dir)
            if retained is not None:
                return retained
        remove_sandbox(SandboxHandle(root, sandbox_dir, None, git_repo))

    if git_repo:
        target_ref = explicit_ref or "HEAD"
        rev_arg = f"{target_ref}^{{commit}}" if explicit_ref is not None else "HEAD"
        head_proc = _run_git(root, "rev-parse", "--verify", rev_arg)
        if head_proc.returncode == 0:
            base_commit = head_proc.stdout.strip()
            sandbox_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                add_proc = _run_git(
                    root,
                    "worktree",
                    "add",
                    "--detach",
                    str(sandbox_dir),
                    base_commit,
                )
            except BaseException:
                _cleanup_partial_sandbox(root, sandbox_dir, is_git_worktree=True)
                raise
            if add_proc.returncode == 0:
                return SandboxHandle(
                    repo_root=root,
                    workspace_path=sandbox_dir,
                    base_commit=base_commit,
                    is_git_worktree=True,
                )
            _cleanup_partial_sandbox(root, sandbox_dir, is_git_worktree=True)
            if explicit_ref is not None:
                detail = (add_proc.stderr or add_proc.stdout).strip() or "git worktree add failed"
                raise RuntimeError(
                    f"Failed to create git worktree at base_ref {explicit_ref!r}: {detail}"
                )
        elif explicit_ref is not None:
            detail = (
                (head_proc.stderr or head_proc.stdout).strip()
                or f"unknown ref {explicit_ref!r}"
            )
            raise ValueError(f"Invalid base_ref {explicit_ref!r}: {detail}")

    # Fallback for uninitialized or unborn-HEAD repos when no explicit base_ref was requested.
    sandbox_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(
            root,
            sandbox_dir,
            symlinks=True,
            ignore=shutil.ignore_patterns(".git", ".venv", ".olympus", "__pycache__"),
        )
    except BaseException:
        _cleanup_partial_sandbox(root, sandbox_dir, is_git_worktree=False)
        raise
    return SandboxHandle(
        repo_root=root,
        workspace_path=sandbox_dir,
        base_commit=None,
        is_git_worktree=False,
    )


def _cleanup_partial_sandbox(root: Path, sandbox_dir: Path, *, is_git_worktree: bool) -> None:
    """Best-effort cleanup if sandbox creation fails midway."""
    if is_git_worktree and is_git_repo(root):
        _run_git(root, "worktree", "remove", "--force", str(sandbox_dir))
    if sandbox_dir.exists():
        shutil.rmtree(sandbox_dir, ignore_errors=True)
    if is_git_worktree and is_git_repo(root):
        _run_git(root, "worktree", "prune")


def save_sandbox_baseline(handle: SandboxHandle, baseline: SandboxBaseline) -> None:
    """Persist the task's Attempt-1 baseline next to its retained sandbox."""
    path = handle.baseline_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "is_git": baseline.is_git,
                "base_commit": baseline.base_commit,
                "entries": [list(e) for e in baseline.entries],
                "preexisting_ephemeral": list(baseline.preexisting_ephemeral),
                "preexisting_dirs": baseline.preexisting_dirs,
                "text_snapshots": [list(item) for item in baseline.text_snapshots],
            },
        ),
        encoding="utf-8",
    )


def load_sandbox_baseline(handle: SandboxHandle) -> SandboxBaseline | None:
    """Load the persisted Attempt-1 baseline; ``None`` when absent or unreadable."""
    try:
        payload = json.loads(handle.baseline_path.read_text(encoding="utf-8"))
        entries = tuple(
            (str(rel), None if fingerprint is None else str(fingerprint))
            for rel, fingerprint in payload["entries"]
        )
        raw_commit = payload.get("base_commit")
        base_commit = str(raw_commit) if isinstance(raw_commit, str) and raw_commit else None
        raw_ephemeral = payload.get("preexisting_ephemeral", [])
        preexisting_ephemeral = (
            tuple(str(rel) for rel in raw_ephemeral if isinstance(rel, str) and rel)
            if isinstance(raw_ephemeral, list)
            else ()
        )
        raw_dirs = payload.get("preexisting_dirs")
        raw_texts = payload.get("text_snapshots", [])
        text_snapshots = (
            tuple(
                (str(item[0]), str(item[1]))
                for item in raw_texts
                if isinstance(item, list) and len(item) == 2
            )
            if isinstance(raw_texts, list)
            else ()
        )
        return SandboxBaseline(
            is_git=bool(payload["is_git"]),
            entries=entries,
            base_commit=base_commit,
            preexisting_ephemeral=preexisting_ephemeral,
            preexisting_dirs=tuple(raw_dirs) if isinstance(raw_dirs, list) else None,
            text_snapshots=text_snapshots,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def capture_sandbox_diff(
    handle: SandboxHandle, baseline: SandboxBaseline | None = None
) -> str:
    """Return tracked diff plus untracked file contents for cold-start review.

    Supplying the task's pre-worker ``baseline`` restricts the output to paths
    mutated since that snapshot, so unchanged upstream ``accepted_overlay``
    files never leak into a downstream task's diff. Without a baseline the
    legacy ``HEAD`` + untracked behavior is preserved for direct callers.
    """
    if baseline is not None:
        return _baseline_scoped_diff(handle, baseline)
    ws = handle.workspace_path
    if not is_git_repo(ws):
        return "(non-git sandbox; inspect modified files directly)"
    return _head_scoped_diff(ws)


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


def _prefixed_content(path: Path) -> tuple[str, int]:
    text = _read_text(path)
    lines = text.splitlines()
    if not lines:
        return "", 0
    return "\n".join(f"+{line}" for line in lines) + "\n", len(lines)


def _is_tracked_at_base(ws: Path, base_commit: str | None, rel: str) -> bool:
    if base_commit is None or not is_git_repo(ws):
        return False
    return _run_git(ws, "cat-file", "-e", f"{base_commit}:{rel}").returncode == 0


def _no_index_new_file_diff(ws: Path, rel: str) -> str:
    proc = _run_git(
        ws,
        "diff",
        "--binary",
        "--no-ext-diff",
        "--no-textconv",
        "--no-index",
        "--",
        "/dev/null",
        rel,
    )
    if proc.returncode in (0, 1) and proc.stdout.strip():
        lines = proc.stdout.rstrip().splitlines()
        if lines and lines[0].startswith("diff --git "):
            lines[0] = f"diff --git a/{rel} b/{rel}"
        return "\n".join(lines)
    body, count = _prefixed_content(ws / rel)
    hunk = f"@@ -0,0 +1,{count} @@\n{body}" if count else ""
    return (
        f"diff --git a/{rel} b/{rel}\n"
        "new file mode 100644\n"
        f"--- /dev/null\n+++ b/{rel}\n"
        f"{hunk}"
    ).rstrip()


def _new_file_block(ws: Path, rel: str) -> str:
    return _no_index_new_file_diff(ws, rel)


def _deleted_file_block(ws: Path, rel: str, base_commit: str | None) -> str:
    if base_commit is not None and is_git_repo(ws):
        tracked = _run_git(
            ws, "diff", "--binary", "--no-ext-diff", "--no-textconv", base_commit, "--", rel
        ).stdout.rstrip()
        if tracked:
            return tracked
    return (
        f"diff --git a/{rel} b/{rel}\n"
        "deleted file mode 100644\n"
        f"--- a/{rel}\n+++ /dev/null"
    )


def _overlay_file_block(ws: Path, rel: str) -> str:
    body, count = _prefixed_content(ws / rel)
    hunk = f"@@ -0,0 +1,{count} @@\n{body}" if count else ""
    return (
        f"diff --git a/{rel} b/{rel}\n"
        f"--- a/{rel}\n+++ b/{rel}\n"
        f"{hunk}"
    ).rstrip()


_MAX_TEXT_SNAPSHOT_BYTES = 4 * 1024 * 1024


def _unified_text_diff_from_strings(before_text: str, after_path: Path, rel: str) -> str | None:
    try:
        after_raw = after_path.read_bytes()
    except OSError:
        return None
    if b"\x00" in after_raw:
        return None
    try:
        after_text = after_raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    diff_lines = list(
        difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            lineterm="\n",
        )
    )
    if not diff_lines:
        return None
    rendered = "".join(line if line.endswith("\n") else f"{line}\n" for line in diff_lines)
    return f"diff --git a/{rel} b/{rel}\n{rendered}".rstrip()


def _unified_text_diff_block(before_path: Path, after_path: Path, rel: str) -> str | None:
    try:
        before_raw = before_path.read_bytes()
    except OSError:
        return None
    if b"\x00" in before_raw:
        return None
    try:
        before_text = before_raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return _unified_text_diff_from_strings(before_text, after_path, rel)


def _modified_file_block(
    ws: Path,
    rel: str,
    base_commit: str | None,
    *,
    repo_root: Path | None = None,
    baseline_text: str | None = None,
) -> str:
    after_path = ws / rel
    if baseline_text is not None and after_path.is_file():
        unified = _unified_text_diff_from_strings(baseline_text, after_path, rel)
        if unified is not None:
            return unified
    if _is_tracked_at_base(ws, base_commit, rel):
        assert base_commit is not None
        diff_out = _run_git(
            ws, "diff", "--binary", "--no-ext-diff", "--no-textconv", base_commit, "--", rel
        ).stdout.rstrip()
        if diff_out:
            return diff_out
    if repo_root is not None and repo_root != ws:
        before_path = repo_root / rel
        if before_path.is_file() and not before_path.is_symlink() and after_path.is_file():
            unified = _unified_text_diff_block(before_path, after_path, rel)
            if unified is not None:
                return unified
    return _overlay_file_block(ws, rel)


def _baseline_scoped_diff(handle: SandboxHandle, baseline: SandboxBaseline) -> str:
    ws = handle.workspace_path
    before = baseline.mapping()
    base_commit = baseline.base_commit or handle.base_commit
    changed = collect_sandbox_changes(handle, baseline)
    if not changed:
        return "(no changes)"

    def present(rel: str) -> bool:
        candidate = ws / rel
        return candidate.is_file() or candidate.is_symlink()

    added = {
        rel
        for rel in changed
        if rel not in before and not _is_tracked_at_base(ws, base_commit, rel) and present(rel)
    }
    deleted = {
        rel
        for rel in changed
        if (rel in before or _is_tracked_at_base(ws, base_commit, rel)) and not present(rel)
    }
    modified = set(changed) - added - deleted

    renames: dict[str, str] = {}
    for old in sorted(deleted):
        fingerprint = before.get(old)
        if fingerprint is None:
            continue
        match = next(
            (new for new in sorted(added) if _file_fingerprint(ws / new) == fingerprint),
            None,
        )
        if match is not None:
            renames[old] = match
            added.discard(match)

    before_texts = baseline.text_mapping()
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
            blocks.append(_deleted_file_block(ws, rel, base_commit))
        elif rel in modified:
            blocks.append(
                _modified_file_block(
                    ws,
                    rel,
                    base_commit,
                    repo_root=handle.repo_root,
                    baseline_text=before_texts.get(rel),
                )
            )
    return "\n\n".join(blocks) if blocks else "(no changes)"


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
    base_commit = handle.base_commit
    if handle.is_git_worktree and base_commit is not None and is_git_repo(ws):
        try:
            entries = _git_changed_fingerprints(ws, base_commit)
        except RuntimeError:
            base_commit = None
            entries = _workspace_fingerprints(ws)
    else:
        base_commit = None
        entries = _workspace_fingerprints(ws)

    untracked, untracked_complete = _scan_untracked_paths(handle)
    ephemeral = tuple(
        sorted(rel for rel in untracked if _is_cleanup_candidate(rel))
    )
    directories: list[str] = []
    walk_complete = True

    def _on_walk_error(_exc: OSError) -> None:
        nonlocal walk_complete
        walk_complete = False

    for dirpath, dirnames, _ in os.walk(ws, onerror=_on_walk_error):
        directories.extend((Path(dirpath) / name).relative_to(ws).as_posix() for name in dirnames)
        dirnames[:] = [name for name in dirnames if name not in _EPHEMERAL_WALK_SKIP]
    preexisting_dirs = (
        tuple(sorted(directories)) if (untracked_complete and walk_complete) else None
    )
    text_snapshots: list[tuple[str, str]] = []
    for rel in sorted(entries):
        candidate = ws / rel
        if not candidate.is_file() or candidate.is_symlink():
            continue
        try:
            if candidate.stat().st_size > _MAX_TEXT_SNAPSHOT_BYTES:
                continue
            raw = candidate.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw:
            continue
        try:
            text_snapshots.append((rel, raw.decode("utf-8")))
        except UnicodeDecodeError:
            continue
    return SandboxBaseline(
        is_git=base_commit is not None,
        entries=tuple(sorted(entries.items())),
        base_commit=base_commit,
        preexisting_ephemeral=ephemeral,
        preexisting_dirs=preexisting_dirs,
        text_snapshots=tuple(text_snapshots),
    )


def _current_fingerprints(
    handle: SandboxHandle, baseline: SandboxBaseline | None = None
) -> dict[str, str | None]:
    ws = handle.workspace_path
    use_git = baseline.is_git if baseline is not None else handle.is_git_worktree
    base_commit = (
        baseline.base_commit
        if baseline is not None and baseline.base_commit is not None
        else handle.base_commit
    )
    if use_git and base_commit is not None and is_git_repo(ws):
        try:
            return _git_changed_fingerprints(ws, base_commit)
        except RuntimeError:
            # A tampered sandbox (e.g. deleted .git) fails closed: every file looks changed.
            return _workspace_fingerprints(ws)
    return _workspace_fingerprints(ws)


def collect_sandbox_changes(handle: SandboxHandle, baseline: SandboxBaseline) -> list[str]:
    """Return repo-relative paths mutated since ``baseline`` (adds, edits, deletes, renames)."""
    current = _current_fingerprints(handle, baseline)
    before = baseline.mapping()
    changed = {
        rel
        for rel, fingerprint in current.items()
        if rel not in before or before[rel] != fingerprint
    }
    changed.update(rel for rel in before if rel not in current)
    return sorted(changed)


def is_ephemeral_artifact(rel: str) -> bool:
    """Return True when ``rel`` matches a conservative temporary/backup junk pattern."""
    parts = PurePosixPath(rel).parts
    if not parts:
        return False
    name = parts[-1]
    for pattern in EPHEMERAL_ARTIFACT_PATTERNS:
        if pattern.startswith("**/") and pattern.endswith("/**"):
            segment = pattern[3:-3]
            if segment and segment in parts[:-1]:
                return True
        elif fnmatchcase(name, pattern):
            return True
    return False


_EPHEMERAL_WALK_SKIP = frozenset({".git", ".venv", ".olympus", "node_modules"})


def _is_cleanup_candidate(rel: str, *, include_runtime: bool = True) -> bool:
    if is_ephemeral_artifact(rel):
        return True
    if not include_runtime:
        return False
    path = PurePosixPath(rel)
    return path.name.endswith(_RUNTIME_ARTIFACT_SUFFIXES) or (
        "uploads" in path.parts[:-1] and path.suffix.lower() in _UPLOAD_ARTIFACT_SUFFIXES
    )


def _scan_untracked_paths(handle: SandboxHandle) -> tuple[list[str], bool]:
    """List untracked files (including ignored dirs) and whether the scan was complete."""
    ws = handle.workspace_path
    git_failed = False
    if handle.is_git_worktree:
        if is_git_repo(ws):
            proc = _run_git(ws, "ls-files", "--others", "-z")
            if proc.returncode == 0:
                paths = [rel for rel in proc.stdout.split("\0") if rel]
                return paths, not bool(proc.stderr.strip())
        git_failed = True
    paths = []
    complete = not git_failed

    def _on_error(_exc: OSError) -> None:
        nonlocal complete
        complete = False

    for dirpath, dirnames, filenames in os.walk(ws, onerror=_on_error):
        dirnames[:] = sorted(d for d in dirnames if d not in _EPHEMERAL_WALK_SKIP)
        for name in sorted(filenames):
            rel = (Path(dirpath) / name).relative_to(ws).as_posix()
            paths.append(rel)
    return paths, complete


def _untracked_paths(handle: SandboxHandle) -> list[str]:
    """List untracked files, including ignored dirs such as ``__pycache__``."""
    paths, _ = _scan_untracked_paths(handle)
    return paths


def clean_ephemeral_artifacts(
    handle: SandboxHandle, baseline: SandboxBaseline, *, write_scope: Sequence[str] | None = None
) -> list[str]:
    """Delete new untracked junk and optionally out-of-scope runtime files.

    A supplied write scope protects all its files. Legacy or incomplete baselines
    (where ``preexisting_dirs is None``) fall back to conservative junk cleanup only.
    Return sorted paths for the cleanup event and prune only directories absent from
    the pre-turn snapshot.
    """
    before = baseline.mapping()
    preexisting_ephemeral = set(baseline.preexisting_ephemeral)
    base_commit = baseline.base_commit or handle.base_commit
    if handle.workspace_path.is_symlink():
        return []
    ws = handle.workspace_path.resolve()
    policy = (
        WorkspacePolicy(allowed_write_scope=list(write_scope)) if write_scope is not None else None
    )
    include_runtime = policy is not None and baseline.preexisting_dirs is not None

    preexisting_dirs = set(baseline.preexisting_dirs or ())
    removed: list[str] = []
    for rel in sorted(_untracked_paths(handle)):
        if rel in before or rel in preexisting_ephemeral:
            continue
        if not _is_cleanup_candidate(rel, include_runtime=include_runtime):
            continue
        if policy is not None and policy.allows_relative_path(rel):
            continue
        if _is_tracked_at_base(ws, base_commit, rel):
            continue
        target = ws / rel
        try:
            if target.parent.resolve() != target.parent:
                continue
            if target.is_symlink():
                if not is_ephemeral_artifact(rel):
                    continue
            elif not target.is_file():
                continue
            target.unlink()
        except (OSError, RuntimeError):
            continue
        removed.append(rel)
        if baseline.preexisting_dirs is not None and not any(
            part in _EPHEMERAL_WALK_SKIP for part in Path(rel).parts[:-1]
        ):
            for parent in target.parents:
                if parent == ws or parent.relative_to(ws).as_posix() in preexisting_dirs:
                    break
                try:
                    parent.rmdir()
                except OSError:
                    break
    return removed


def sync_upstream_files(
    handle: SandboxHandle,
    overlay_dir: Path,
    baseline: SandboxBaseline,
    *,
    write_scope: Sequence[str],
    deleted_paths: Sequence[str] = (),
) -> SandboxBaseline:
    """Refresh newly accepted upstream files/deletions without clobbering in-progress work.

    Only paths the worker left exactly at their baseline content are synced. Worker
    mutations outside ``write_scope`` stay untouched so the Diff Gate still audits them,
    and the returned baseline records fingerprints for the synced upstream paths only.
    """
    if not overlay_dir.is_dir() and not deleted_paths:
        return baseline
    policy = WorkspacePolicy(allowed_write_scope=list(write_scope))
    merged = baseline.mapping()
    current_map = _current_fingerprints(handle, baseline)
    base_commit = baseline.base_commit or handle.base_commit
    ws = handle.workspace_path
    synced_rels: list[str] = []
    # Apply upstream deletions FIRST so file-to-directory replacements (e.g. foo -> foo/bar.py)
    # unlink the old regular file before copying overlay children into the new directory.
    for rel in sorted(set(deleted_paths)):
        if any(
            part in IGNORED_DIR_NAMES or part == ".olympus"
            for part in PurePosixPath(rel).parts
        ):
            continue
        if policy.allows_relative_path(rel):
            continue
        if rel in current_map and current_map[rel] != merged.get(rel):
            continue
        if rel in merged and rel not in current_map:
            continue
        current = _file_fingerprint(ws / rel)
        if current is None:
            continue
        if rel in merged:
            if current != merged[rel]:
                continue
        elif not _is_tracked_at_base(ws, base_commit, rel):
            continue
        target = ws / rel
        if target.is_symlink() or target.is_file():
            target.unlink()
            synced_rels.append(rel)
    if overlay_dir.is_dir():
        for src in sorted(overlay_dir.rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(overlay_dir).as_posix()
            if any(
                part in IGNORED_DIR_NAMES or part == ".olympus"
                for part in PurePosixPath(rel).parts
            ):
                continue
            if policy.allows_relative_path(rel):
                continue
            if rel in current_map and current_map[rel] != merged.get(rel):
                continue
            if rel in merged and rel not in current_map:
                continue
            current = _file_fingerprint(ws / rel)
            if rel in merged:
                if current != merged[rel]:
                    continue
            elif (current is not None) != _is_tracked_at_base(ws, base_commit, rel):
                continue
            dst = ws / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            synced_rels.append(rel)
    if synced_rels:
        updated_map = _current_fingerprints(handle, baseline)
        for rel in synced_rels:
            if rel in updated_map:
                merged[rel] = updated_map[rel]
            else:
                merged.pop(rel, None)
    return replace(baseline, entries=tuple(sorted(merged.items())))


def absorb_verification_artifacts(
    handle: SandboxHandle,
    baseline: SandboxBaseline,
    *,
    write_scope: Sequence[str],
) -> SandboxBaseline:
    """Fold newly created untracked out-of-scope check-command side effects into the baseline.

    Tracked or pre-existing baseline files modified or deleted by a check command are never
    absorbed so tampered inputs remain visible to the Diff Gate on retry.
    """
    policy = WorkspacePolicy(allowed_write_scope=list(write_scope))
    current = _current_fingerprints(handle, baseline)
    merged = baseline.mapping()
    base_commit = baseline.base_commit or handle.base_commit
    ws = handle.workspace_path
    for rel, fingerprint in current.items():
        if policy.allows_relative_path(rel):
            continue
        if rel in merged or _is_tracked_at_base(ws, base_commit, rel):
            continue
        if fingerprint is not None:
            merged[rel] = fingerprint
    return replace(baseline, entries=tuple(sorted(merged.items())))


def _is_registered_worktree(repo_root: Path, workspace_path: Path) -> bool:
    """Return True if ``workspace_path`` is still registered in ``git worktree list``."""
    proc = _run_git(repo_root, "worktree", "list", "--porcelain")
    if proc.returncode != 0:
        return False
    target = workspace_path.resolve(strict=False)
    for line in proc.stdout.splitlines():
        if not line.startswith("worktree "):
            continue
        listed = Path(line.removeprefix("worktree ").strip()).resolve(strict=False)
        if listed == target:
            return True
    return False


def remove_sandbox(handle: SandboxHandle, *, strict: bool = False) -> None:
    """Clean up a worktree or copied sandbox directory plus its temp dir and baseline.

    When ``strict=True``, verify that both the sandbox directory and any git worktree
    registration are completely removed, attempting ``git worktree prune`` if needed,
    and raise ``RuntimeError`` instead of silently swallowing failures.
    """
    errors: list[str] = []
    git_host = is_git_repo(handle.repo_root)
    git_rm_failed = False
    if handle.is_git_worktree and git_host:
        rm_proc = _run_git(
            handle.repo_root, "worktree", "remove", "--force", str(handle.workspace_path)
        )
        if rm_proc.returncode != 0:
            git_rm_failed = True
            detail = (rm_proc.stderr or rm_proc.stdout).strip()
            if detail:
                errors.append(f"git worktree remove exit={rm_proc.returncode}: {detail}")
    if handle.workspace_path.exists():
        if strict:
            try:
                shutil.rmtree(handle.workspace_path)
            except OSError as exc:
                errors.append(f"rmtree({handle.workspace_path}): {exc}")
        else:
            shutil.rmtree(handle.workspace_path, ignore_errors=True)
    if git_host and (
        git_rm_failed or _is_registered_worktree(handle.repo_root, handle.workspace_path)
    ):
        prune_proc = _run_git(handle.repo_root, "worktree", "prune")
        if prune_proc.returncode != 0:
            detail = (prune_proc.stderr or prune_proc.stdout).strip()
            if detail:
                errors.append(f"git worktree prune exit={prune_proc.returncode}: {detail}")
    shutil.rmtree(handle.tmp_dir, ignore_errors=True)
    handle.baseline_path.unlink(missing_ok=True)
    if strict:
        still_exists = handle.workspace_path.exists()
        still_registered = git_host and _is_registered_worktree(
            handle.repo_root, handle.workspace_path
        )
        if still_exists or still_registered:
            reasons: list[str] = []
            if still_exists:
                reasons.append(f"directory still exists: {handle.workspace_path}")
            if still_registered:
                reasons.append(f"still registered in git worktree list: {handle.workspace_path}")
            reasons.extend(errors)
            raise RuntimeError(
                f"Failed to remove sandbox {handle.workspace_path}: " + "; ".join(reasons)
            )


def remove_run_sandboxes(
    repo_root: Path | str,
    run_id: str,
    *,
    sandbox_root: Path | str | None = None,
) -> None:
    """Reclaim every retained sandbox, baseline, and temp dir belonging to ``run_id``."""
    root = Path(repo_root).resolve()
    prefix = f"{_safe_label(run_id)}-"
    git_host = is_git_repo(root)
    if sandbox_root is not None:
        sandboxes = Path(sandbox_root).resolve()
        external = not sandboxes.is_relative_to(root)
        baselines = (
            sandboxes / ".sandbox_baselines"
            if external
            else root / ".olympus" / "sandbox_baselines"
        )
        tmp_root = sandboxes / ".sandbox_tmp" if external else root / ".olympus" / "sandbox_tmp"
    else:
        sandboxes = root / ".olympus" / "sandboxes"
        baselines = root / ".olympus" / "sandbox_baselines"
        tmp_root = root / ".olympus" / "sandbox_tmp"
    if sandboxes.is_dir():
        for entry in sorted(sandboxes.iterdir()):
            if entry.is_dir() and entry.name.startswith(prefix):
                if git_host:
                    _run_git(root, "worktree", "remove", "--force", str(entry))
                shutil.rmtree(entry, ignore_errors=True)
    if baselines.is_dir():
        for entry in sorted(baselines.iterdir()):
            if entry.name.startswith(prefix):
                entry.unlink(missing_ok=True)
    if tmp_root.is_dir():
        for entry in sorted(tmp_root.iterdir()):
            if entry.is_dir() and entry.name.startswith(prefix):
                shutil.rmtree(entry, ignore_errors=True)


ACCEPTED_DELETIONS_FILE = "accepted_deletions.json"


def load_accepted_deletions(run_dir: Path) -> set[str]:
    """Load the persisted set of repo-relative paths deleted by accepted tasks."""
    path = run_dir / ACCEPTED_DELETIONS_FILE
    if not path.is_file():
        return set()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(raw, list):
        return set()
    return {item for item in raw if isinstance(item, str) and item}


def save_accepted_deletions(run_dir: Path, paths: set[str]) -> None:
    """Persist repo-relative paths deleted by accepted tasks so downstream sandboxes apply them."""
    path = run_dir / ACCEPTED_DELETIONS_FILE
    if not paths:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(paths), ensure_ascii=False), encoding="utf-8")


def remove_deleted_paths(
    dst_root: Path,
    rel_paths: Sequence[str],
    *,
    forbidden_write_scope: Sequence[str] = (),
    ignore_errors: bool = True,
) -> None:
    """Remove deleted or renamed source paths from a sandbox, overlay, or workspace root."""
    if not rel_paths or not dst_root.exists():
        return
    resolved_root = dst_root.resolve()
    policy = (
        WorkspacePolicy(forbidden_write_scope=list(forbidden_write_scope))
        if forbidden_write_scope
        else None
    )
    for rel in rel_paths:
        if policy is not None and policy.forbidden_match(rel) is not None:
            continue
        target = dst_root / rel
        try:
            if not target.resolve(strict=False).is_relative_to(resolved_root):
                continue
            if target.is_symlink() or target.is_file():
                target.unlink()
                parent = target.parent
                while parent != dst_root and parent.is_relative_to(dst_root):
                    try:
                        parent.rmdir()
                    except OSError:
                        break
                    parent = parent.parent
        except OSError:
            if not ignore_errors:
                raise
            continue
