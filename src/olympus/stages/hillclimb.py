"""Frozen-harness metric hill-climbing optimization stage (`pstack_hillclimb`).

Implements the ``hillclimb_loop`` composite deterministic primitive:
1. Snapshots SHA-256 digests of every file in ``frozen_harness_paths`` before iteration 1
   and verifies byte-identity after every edit and every metric execution (anti-tamper guard).
2. Measures baseline metric inside an isolated git worktree sandbox using ``shell=False``
   (plus ``bwrap`` read-only host FS isolation when available), failing closed on parse failure.
3. Iteratively proposes and applies one grounded hypothesis inside ``write_scope`` checked by
   ``DiffGate`` + anti-tamper guard, then deterministically ``git commit``s on strict metric
   improvement (``keep``) or runs ``git reset --hard`` + ``git clean -fd`` (``revert``).
4. Persists every ``HillclimbIterationRow`` to SQLite (``hillclimb.iteration_recorded``,
   ``hillclimb.completed``) and ``artifacts/<run_id>/hillclimb/decision_trail.json``.
5. Stops cleanly on ``target_value``, ``plateau_patience``, or ``max_iterations``, and supports
   atomic 2-phase workspace promotion with rollback.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from olympus.core.contracts import (
    EvidenceBrief,
    HillclimbConfig,
    HillclimbIterationRow,
    HillclimbReport,
    SurfaceCheckResult,
    SurfaceVerificationReport,
)
from olympus.core.store import RunStore
from olympus.harness.diff_gate import run_diff_gate
from olympus.harness.driver import (
    ActiveTurnHandle,
    AgentDriver,
    DriverRegistry,
    DriverTurnRequest,
    ExecutionDriversConfig,
    TurnOutcome,
    default_driver_factories,
    resolve_driver_spec,
)
from olympus.harness.native import NativeDriver
from olympus.harness.worktree import (
    SandboxBaseline,
    SandboxHandle,
    clean_ephemeral_artifacts,
    collect_sandbox_changes,
    create_sandbox,
    load_sandbox_baseline,
    remove_sandbox,
    save_sandbox_baseline,
    snapshot_sandbox_state,
)
from olympus.providers.base import LLMProvider
from olympus.stages.common import default_artifacts_root
from olympus.stages.craft import load_skill_pack
from olympus.tools.aci import WorkspacePolicy

_MAX_EXCERPT_CHARS = 2400
_TRIVIAL_NOOP_CMD_RE = re.compile(
    r"^\s*(?::|true|false|exit(?:\s+\d+)?|return(?:\s+\d+)?)\s*(?:;+\s*)*$",
    re.IGNORECASE,
)
_SHELL_META_RE = re.compile(r"(?:[;&|<>`]|\$\()")
_HYPOTHESIS_RE = re.compile(r"(?im)^\s*hypothesis\s*:\s*(.+?)\s*$")
_UNTRUSTED_PATH_PREFIXES: tuple[Path, ...] = (
    Path("/tmp"),
    Path("/var/tmp"),
    Path("/dev/shm"),
)
_GIT_COMMIT_ENV_ARGS: tuple[str, ...] = (
    "-c",
    "user.name=Olympus Hillclimb",
    "-c",
    "user.email=hillclimb@olympus.local",
)
_FORBIDDEN_CMD_FLAGS: frozenset[str] = frozenset(
    {
        "-c",
        "-e",
        "-p",
        "--eval",
        "--exec",
        "--fix",
        "--fix-only",
        "--unsafe-fixes",
        "--add-noqa",
        "-w",
        "--write",
        "--in-place",
        "-o",
        "--output",
        "--output-file",
        "--basetemp",
        "--junitxml",
        "--junit-xml",
        "--cache-dir",
        "--cov-report",
        "--log-file",
        "--resultlog",
        "--html",
        "--json-report",
        "--config",
        "--config-file",
        "--confcutdir",
        "--rootdir",
        "--override-ini",
        "--with",
        "--from",
        "--script",
        "--directory",
        "--project",
        "--python-executable",
        "--shadow-file",
        "--custom-typeshed-dir",
    }
)
_FORBIDDEN_SHORT_FLAGS: frozenset[str] = frozenset({"c", "e", "o", "p", "w"})
_SAFE_ENV_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TZ",
    "USER",
    "LOGNAME",
    "VIRTUAL_ENV",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "SYSTEMROOT",
)


class _SandboxIsolationUnavailableError(RuntimeError):
    """Raised when OS-level read-only FS isolation (bwrap) is required but unavailable."""


class HillclimbPause(Exception):
    """Typed hard-stop raised by ``HillclimbStage`` and mapped to ``GraphPaused``."""

    def __init__(
        self,
        node_name: str,
        reason: str,
        *,
        route: Literal["clarify", "blocked"] = "blocked",
        pause_kind: str = "blocked",
        state_updates: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.node_name = node_name
        self.reason = reason
        self.route = route
        self.pause_kind = pause_kind
        self.state_updates = dict(state_updates or {})
        self.state_updates.setdefault("route", route)
        self.state_updates.setdefault("pause_kind", pause_kind)


def _trim_output(text: str, limit: int = _MAX_EXCERPT_CHARS) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "\n...[truncated]"


def _compose_system_prompt(base_prompt: str, skill_refs: Sequence[str]) -> str:
    pack = load_skill_pack(skill_refs)
    if not pack:
        return base_prompt
    return f"{base_prompt}\n\n---\n{pack}"


_MINIMAL_SYSTEM_RO_MOUNTS: tuple[str, ...] = (
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
    "/etc/alternatives",
    "/etc/ssl",
    "/etc/ca-certificates",
)
_FORBIDDEN_RO_PREFIXES: frozenset[Path] = frozenset(
    {
        Path("/"),
        Path("/home"),
        Path("/Users"),
        Path("/root"),
        Path("/tmp"),
        Path("/var"),
        Path("/var/tmp"),
        Path("/opt"),
        Path("/etc"),
    }
)


def _is_safe_ro_prefix(path: Path, *, cwd_resolved: Path, metric_tmp: Path) -> bool:
    """Return True when ``path`` is a narrow install/runtime prefix safe to mount read-only."""
    norm = Path(os.path.abspath(str(path)))
    home = Path(os.path.abspath(str(Path.home())))
    home_res = Path.home().resolve(strict=False)
    if norm in _FORBIDDEN_RO_PREFIXES or norm in {home, home_res}:
        return False
    # Never mount /home/<user> or /Users/<user> directly.
    if len(norm.parts) == 3 and norm.parts[1] in {"home", "Users"}:
        return False
    if norm == cwd_resolved or norm.is_relative_to(cwd_resolved):
        return False
    if norm == metric_tmp or norm.is_relative_to(metric_tmp):
        return False
    return not (cwd_resolved.is_relative_to(norm) or metric_tmp.is_relative_to(norm))


def _system_ro_mount_args() -> list[str]:
    """Build read-only system mounts, preserving usrmerge root symlinks (/bin -> usr/bin)."""
    args: list[str] = []
    for raw in _MINIMAL_SYSTEM_RO_MOUNTS:
        p = Path(raw)
        if p.is_symlink():
            args.extend(["--symlink", os.readlink(p), raw])
        elif p.exists():
            args.extend(["--ro-bind", raw, raw])
    return args


def _resolve_posix_namei(
    raw_path: str | Path,
    *,
    max_hops: int = 64,
) -> tuple[Path, list[tuple[Path, str]], list[Path], list[Path]]:
    """Resolve ``raw_path`` component-by-component following Linux ``namei`` semantics.

    Never collapses ``..`` before dereferencing preceding symlinks (so targets like
    ``bridge/../second`` dereference ``bridge`` first, record ``bridge`` as a symlink,
    record ``bridge``'s resolved target directory as a traversed directory for ``--dir``,
    and only then step to ``..``).

    Returns ``(final_resolved_path, symlinks, hop_parents, traversed_dirs)``.
    """
    text = str(raw_path)
    if not text.startswith("/"):
        text = str(Path.cwd() / text)
    cur = Path("/")
    queue: deque[str] = deque(part for part in text.split("/") if part)
    symlinks: dict[Path, str] = {}
    hop_parents: list[Path] = []
    traversed_dirs: set[Path] = set()
    hops = 0

    while queue:
        part = queue.popleft()
        if part == ".":
            continue
        if part == "..":
            if cur != Path("/"):
                traversed_dirs.add(cur)
                cur = cur.parent
            continue
        nxt = cur / part
        if os.path.islink(nxt):
            raw_target = os.readlink(nxt)
            symlinks.setdefault(nxt, raw_target)
            if not queue:
                # Leaf symlink (e.g. venv/bin/python) lives inside cur; mount cur's prefix.
                hop_parents.append(cur)
            hops += 1
            if hops > max_hops:
                break
            target_parts = [p for p in raw_target.split("/") if p]
            if raw_target.startswith("/"):
                cur = Path("/")
            for tpart in reversed(target_parts):
                queue.appendleft(tpart)
            continue
        cur = nxt

    return (
        cur,
        sorted(symlinks.items(), key=lambda item: len(item[0].parts)),
        hop_parents,
        sorted(traversed_dirs, key=lambda item: len(item.parts)),
    )


def _minimal_bwrap_mount_args(
    argv: Sequence[str],
    *,
    cwd_resolved: Path,
    metric_tmp: Path,
) -> list[str]:
    """Build minimal read-only system + POSIX ``namei`` symlink-resolved bwrap mount args.

    Resolves ``argv[0]``, ``sys.executable``, ``sys.prefix``, and ``sys.base_prefix``
    component-by-component without pre-collapsing ``..`` so multi-hop and relative
    ``bridge/../second`` symlink chains mount every hop cleanly without exposing ``/``
    or host home.
    """
    symlinks: dict[Path, str] = {}
    ro_dirs: list[Path] = []
    seen_dirs: set[Path] = set()
    traversed_dirs: set[Path] = set()

    def _add_real_ro_dir(real_dir: Path) -> None:
        if not real_dir.exists() or real_dir.is_symlink():
            return
        if not _is_safe_ro_prefix(real_dir, cwd_resolved=cwd_resolved, metric_tmp=metric_tmp):
            return
        if real_dir not in seen_dirs:
            seen_dirs.add(real_dir)
            ro_dirs.append(real_dir)

    def _trace_dir(dir_path: Path) -> None:
        if not (dir_path.exists() or dir_path.is_symlink()):
            return
        final_dir, links, _hop_parents, tdirs = _resolve_posix_namei(dir_path)
        for link_p, target in links:
            symlinks.setdefault(link_p, target)
        traversed_dirs.update(tdirs)
        _add_real_ro_dir(final_dir)

    def _trace_executable(exe_path: Path) -> None:
        if not (exe_path.exists() or exe_path.is_symlink()):
            return
        final_exe, links, hop_parents, tdirs = _resolve_posix_namei(exe_path)
        for link_p, target in links:
            symlinks.setdefault(link_p, target)
        traversed_dirs.update(tdirs)
        for parent_dir in (*hop_parents, final_exe.parent):
            _add_real_ro_dir(parent_dir)
            if parent_dir.name in {"bin", "Scripts"}:
                _add_real_ro_dir(parent_dir.parent)

    for prefix in (sys.prefix, sys.base_prefix):
        if prefix:
            _trace_dir(Path(prefix))

    _trace_executable(Path(sys.executable))
    if argv:
        bin_path = Path(argv[0])
        if bin_path.is_absolute():
            _trace_executable(bin_path)

    args: list[str] = _system_ro_mount_args()
    # Emit real directory ro-binds first (skipping already-covered subdirs).
    system_dirs = {
        Path(raw).resolve(strict=False)
        for raw in _MINIMAL_SYSTEM_RO_MOUNTS
        if Path(raw).exists()
    }
    mounted_dirs: list[Path] = list(system_dirs)
    for ro_dir in sorted(ro_dirs, key=lambda item: len(item.parts)):
        if any(ro_dir == existing or ro_dir.is_relative_to(existing) for existing in mounted_dirs):
            continue
        args.extend(["--ro-bind", str(ro_dir), str(ro_dir)])
        mounted_dirs.append(ro_dir)
    # Create empty stub dirs for directories traversed solely to step up via ``..``
    # (e.g. bridge -> other/sub followed by ..) without exposing any host files inside them.
    for tdir in sorted(traversed_dirs, key=lambda item: len(item.parts)):
        if tdir == Path("/") or any(
            tdir == existing or tdir.is_relative_to(existing) for existing in mounted_dirs
        ):
            continue
        if tdir in symlinks:
            continue
        args.extend(["--dir", str(tdir)])
    system_links = {Path(raw) for raw in _MINIMAL_SYSTEM_RO_MOUNTS}
    for link_path, target in sorted(symlinks.items(), key=lambda item: len(item[0].parts)):
        if link_path in system_links or any(
            link_path == existing or link_path.is_relative_to(existing)
            for existing in mounted_dirs
        ):
            continue
        args.extend(["--symlink", target, str(link_path)])
    return args


@functools.lru_cache(maxsize=1)
def _probe_bwrap() -> str | None:
    """Return the path to ``bwrap`` when Linux user-namespace minimal mounting works."""
    bwrap_bin = shutil.which("bwrap", path="/usr/bin:/bin:/usr/local/bin")
    if bwrap_bin is None:
        return None
    cmd: list[str] = [
        bwrap_bin,
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        *_system_ro_mount_args(),
        "--",
        "/bin/true",
    ]
    try:
        probe = subprocess.run(
            cmd,
            shell=False,
            capture_output=True,
            timeout=5,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    return bwrap_bin if probe.returncode == 0 else None


@functools.lru_cache(maxsize=1)
def _probe_bwrap_unshare_net() -> bool:
    """Return True when ``bwrap --unshare-net`` works with minimal read-only mounts."""
    bwrap_bin = _probe_bwrap()
    if bwrap_bin is None:
        return False
    cmd: list[str] = [
        bwrap_bin,
        "--unshare-net",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        *_system_ro_mount_args(),
        "--",
        "/bin/true",
    ]
    try:
        probe = subprocess.run(
            cmd,
            shell=False,
            capture_output=True,
            timeout=5,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return probe.returncode == 0


@functools.lru_cache(maxsize=1)
def _probe_bwrap_unshare_pid() -> bool:
    """Return True when ``bwrap --unshare-pid`` works alongside ``--proc /proc``."""
    bwrap_bin = _probe_bwrap()
    if bwrap_bin is None:
        return False
    cmd: list[str] = [
        bwrap_bin,
        "--unshare-pid",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        *_system_ro_mount_args(),
        "--",
        "/bin/true",
    ]
    try:
        probe = subprocess.run(
            cmd,
            shell=False,
            capture_output=True,
            timeout=5,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return probe.returncode == 0


def _has_forbidden_flag(arg: str) -> str | None:
    """Detect forbidden long, attached-value, or bundled short flags in commands."""
    if not arg.startswith("-") or arg == "-":
        return None
    flag_name = arg.split("=", 1)[0].lower()
    if flag_name in _FORBIDDEN_CMD_FLAGS:
        return f"flag {flag_name!r} is forbidden in sandboxed command"
    if arg.startswith("--"):
        return None
    short_body = arg[1:].lower()
    if not short_body:
        return None
    if short_body[0] in _FORBIDDEN_SHORT_FLAGS:
        return f"short flag -{short_body[0]} (in {arg!r}) is forbidden in sandboxed command"
    if short_body[0] not in {"k", "m"}:
        for ch in short_body:
            if ch in _FORBIDDEN_SHORT_FLAGS:
                return f"bundled short flag -{ch} (in {arg!r}) is forbidden in sandboxed command"
    return None


def _minimal_sandbox_env(
    *,
    safe_path: str,
    metric_tmp: Path,
    metric_home: Path,
) -> dict[str, str]:
    """Build a minimal sanitized subprocess env without leaking host secrets."""
    env: dict[str, str] = {}
    for key in _SAFE_ENV_PASSTHROUGH_KEYS:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val
    env["PATH"] = safe_path
    env["HOME"] = str(metric_home)
    env["CI"] = "1"
    env["TMPDIR"] = str(metric_tmp)
    env["TEMP"] = str(metric_tmp)
    env["TMP"] = str(metric_tmp)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["RUFF_NO_CACHE"] = "1"
    env["RUFF_CACHE_DIR"] = str(metric_tmp / "ruff_cache")
    env["MYPY_CACHE_DIR"] = str(metric_tmp / "mypy_cache")
    env["PYTEST_ADDOPTS"] = f"-o cache_dir={metric_tmp / 'pytest_cache'} -p no:cacheprovider"
    return env


def _trusted_bin_dirs(
    *, host_root: Path | None = None, sandbox_root: Path | None = None
) -> list[Path]:
    """Return trusted executable directories excluding repo/sandbox/tmp paths."""
    untrusted: list[Path] = list(_UNTRUSTED_PATH_PREFIXES)
    for root in (host_root, sandbox_root):
        if root is not None:
            untrusted.append(root.resolve())
    candidates: list[Path] = [
        Path(sys.executable).parent,
        Path(sys.executable).resolve().parent,
        Path("/usr/local/bin"),
        Path("/usr/bin"),
        Path("/bin"),
        Path("/opt/homebrew/bin"),
        Path.home() / ".local" / "bin",
        Path.home() / ".cargo" / "bin",
    ]
    seen: set[Path] = set()
    trusted: list[Path] = []
    for raw_dir in candidates:
        resolved = raw_dir.resolve(strict=False)
        if not resolved.is_dir():
            continue
        if any(resolved == bad or resolved.is_relative_to(bad) for bad in untrusted):
            continue
        if resolved not in seen:
            seen.add(resolved)
            trusted.append(resolved)
    return trusted


def _sanitized_path(*, host_root: Path | None = None, sandbox_root: Path | None = None) -> str:
    return os.pathsep.join(
        str(path) for path in _trusted_bin_dirs(host_root=host_root, sandbox_root=sandbox_root)
    )


def _resolve_trusted_executable(
    token: str,
    *,
    host_root: Path,
    sandbox_root: Path,
) -> tuple[str | None, str | None]:
    """Resolve an executable strictly to a trusted directory outside sandbox/tmp."""
    if not token:
        return None, "empty executable token"
    bare = token.lower()
    if bare in {"python", "python3"}:
        token = sys.executable

    trusted_dirs = _trusted_bin_dirs(host_root=host_root, sandbox_root=sandbox_root)
    trusted_set = {path.resolve(strict=False) for path in trusted_dirs}
    sandbox_resolved = sandbox_root.resolve()
    untrusted_roots = (sandbox_resolved, *_UNTRUSTED_PATH_PREFIXES)

    if "/" in token or "\\" in token:
        candidate = Path(token)
        if not candidate.is_absolute():
            return None, f"relative executable path is forbidden: {token!r}"
        resolved = candidate.resolve(strict=False)
        for bad in untrusted_roots:
            if (
                candidate == bad
                or candidate.is_relative_to(bad)
                or resolved == bad
                or resolved.is_relative_to(bad)
            ):
                return None, f"executable inside untrusted directory is forbidden: {token!r}"
        if candidate.parent not in trusted_set and resolved.parent not in trusted_set:
            # Allow the exact active Python interpreter even if hosted inside a repo .venv.
            sys_exe = Path(sys.executable)
            if candidate != sys_exe and resolved != sys_exe.resolve(strict=False):
                return None, f"executable outside trusted bin directories is forbidden: {token!r}"
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            return None, f"executable not found or not executable: {token!r}"
        return str(candidate), None

    found = shutil.which(token, path=os.pathsep.join(str(path) for path in trusted_dirs))
    if found is None:
        return None, f"executable not found in trusted PATH: {token!r}"
    found_raw = Path(found)
    found_resolved = found_raw.resolve(strict=False)
    for bad in untrusted_roots:
        if (
            found_raw == bad
            or found_raw.is_relative_to(bad)
            or found_resolved == bad
            or found_resolved.is_relative_to(bad)
        ):
            return None, f"resolved binary inside untrusted directory is forbidden: {found!r}"
    return found, None


def _parse_safe_argv(
    command: str,
    *,
    host_root: Path,
    sandbox_root: Path,
) -> tuple[list[str] | None, str | None]:
    """Tokenize and validate a ``shell=False`` command without shell metacharacters."""
    stripped = command.strip()
    if not stripped:
        return None, "empty command"
    if _TRIVIAL_NOOP_CMD_RE.match(stripped):
        return None, f"trivial no-op command is forbidden: {stripped!r}"
    if _SHELL_META_RE.search(stripped):
        return None, f"shell metacharacters are forbidden in command: {stripped!r}"
    try:
        argv = shlex.split(stripped)
    except ValueError as exc:
        return None, f"malformed command syntax: {exc}"
    if not argv:
        return None, "empty tokenized command"
    trusted0, exe_issue = _resolve_trusted_executable(
        argv[0], host_root=host_root, sandbox_root=sandbox_root
    )
    if exe_issue is not None or trusted0 is None:
        return None, exe_issue or "untrusted executable"
    for arg in argv[1:]:
        flag_issue = _has_forbidden_flag(arg)
        if flag_issue is not None:
            return None, flag_issue
    return [trusted0, *argv[1:]], None


def _run_argv_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_s: int = 60,
    host_root: Path | None = None,
    require_fs_isolation: bool = True,
    read_only_files: Sequence[Path] = (),
) -> tuple[int | None, str]:
    """Execute one tokenized command (``shell=False``) with hardened ``bwrap`` isolation.

    Returns ``(returncode, raw_combined_output)`` untruncated so callers can parse metrics
    against the complete output before trimming excerpts for storage/display.
    """
    bwrap_bin = _probe_bwrap()
    if bwrap_bin is None and require_fs_isolation:
        raise _SandboxIsolationUnavailableError(
            "Read-only OS filesystem sandbox (bwrap) is unavailable; refusing unsandboxed "
            "metric execution (set allow_unsandboxed_metric=True or "
            "OLYMPUS_ALLOW_UNSANDBOXED_REPRO=1 to opt in explicitly)"
        )
    cwd_resolved = cwd.resolve()
    metric_tmp = (cwd_resolved.parent / f".{cwd_resolved.name}-hillclimb-tmp").resolve()
    metric_home = metric_tmp / "home"
    metric_home.mkdir(parents=True, exist_ok=True)
    safe_path = _sanitized_path(host_root=host_root, sandbox_root=cwd_resolved)
    env = _minimal_sandbox_env(
        safe_path=safe_path,
        metric_tmp=metric_tmp,
        metric_home=metric_home,
    )

    cmd: list[str] = list(argv)
    if bwrap_bin is not None:
        unshare_net_ok = _probe_bwrap_unshare_net()
        unshare_pid_ok = _probe_bwrap_unshare_pid()
        if (not unshare_net_ok or not unshare_pid_ok) and require_fs_isolation:
            shutil.rmtree(metric_tmp, ignore_errors=True)
            missing = [
                flag
                for flag, ok in (
                    ("--unshare-net", unshare_net_ok),
                    ("--unshare-pid", unshare_pid_ok),
                )
                if not ok
            ]
            raise _SandboxIsolationUnavailableError(
                f"bwrap {' '.join(missing)} is unavailable; refusing weakly isolated metric "
                "execution (set allow_unsandboxed_metric=True or "
                "OLYMPUS_ALLOW_UNSANDBOXED_REPRO=1 to opt in explicitly)"
            )
        bwrap_cmd: list[str] = [bwrap_bin]
        if unshare_net_ok:
            bwrap_cmd.append("--unshare-net")
        if unshare_pid_ok:
            bwrap_cmd.append("--unshare-pid")
        bwrap_cmd.extend(["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"])
        bwrap_cmd.extend(
            _minimal_bwrap_mount_args(argv, cwd_resolved=cwd_resolved, metric_tmp=metric_tmp)
        )
        bwrap_cmd.extend(["--bind", str(cwd_resolved), str(cwd_resolved)])
        # Overlay frozen harness files read-only inside the sandbox mount namespace.
        for ro_file in read_only_files:
            ro_resolved = ro_file.resolve()
            if ro_resolved.is_file() and ro_resolved.is_relative_to(cwd_resolved):
                bwrap_cmd.extend(["--ro-bind", str(ro_resolved), str(ro_resolved)])
        bwrap_cmd.extend(
            [
                "--bind",
                str(metric_tmp),
                str(metric_tmp),
                "--chdir",
                str(cwd_resolved),
                "--",
                *argv,
            ]
        )
        cmd = bwrap_cmd
    try:
        proc = subprocess.run(
            cmd,
            shell=False,
            cwd=str(cwd_resolved),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            stdin=subprocess.DEVNULL,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        err = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        combined = "\n".join(part for part in (out, err) if part).strip()
        return None, f"Command timed out after {timeout_s}s.\n{combined}".strip()
    except FileNotFoundError:
        return 127, f"Command not found: {argv[0]}"
    except PermissionError:
        return 126, f"Command not executable: {argv[0]}"
    except OSError as exc:
        return None, f"Command execution error: {exc}"
    finally:
        shutil.rmtree(metric_tmp, ignore_errors=True)
    combined = "\n".join(part for part in (proc.stdout, proc.stderr) if part and part.strip())
    return proc.returncode, combined


def _run_git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        shell=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )


def _git_head_sha(cwd: Path) -> str:
    proc = _run_git(cwd, "rev-parse", "HEAD")
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"git rev-parse HEAD failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _git_has_changes(cwd: Path, *, include_ignored: bool = False) -> bool:
    args = ["status", "--porcelain", "--untracked-files=all"]
    if include_ignored:
        args.append("--ignored")
    proc = _run_git(cwd, *args)
    if proc.returncode != 0:
        raise RuntimeError(f"git status failed: {proc.stderr.strip()}")
    return bool(proc.stdout.strip())


def _host_path_fingerprint(path: Path) -> str | None:
    """Compute host file fingerprint matching ``worktree._file_fingerprint`` semantics."""
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if stat.S_ISLNK(info.st_mode):
        return f"symlink:{os.readlink(path)}"
    if not stat.S_ISREG(info.st_mode):
        return f"other:{stat.S_IFMT(info.st_mode):o}"
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return f"{stat.S_IMODE(info.st_mode):o}:{digest.hexdigest()}"


def _git_blob_fingerprint(repo_root: Path, base_commit: str, rel: str) -> str | None:
    """Return expected ``mode:sha256`` or ``symlink:<target>`` from ``base_commit`` in git."""
    tree_proc = _run_git(repo_root, "ls-tree", base_commit, "--", rel)
    if tree_proc.returncode != 0:
        raise RuntimeError(f"git ls-tree {base_commit} -- {rel} failed: {tree_proc.stderr.strip()}")
    line = tree_proc.stdout.strip()
    if not line:
        return None
    header = line.split("\t", 1)[0].split()
    if len(header) < 3:
        return None
    git_mode = header[0]
    blob_proc = subprocess.run(
        ["git", "cat-file", "-p", f"{base_commit}:{rel}"],
        cwd=str(repo_root),
        shell=False,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    if blob_proc.returncode != 0:
        raise RuntimeError(f"git cat-file -p {base_commit}:{rel} failed")
    raw = blob_proc.stdout
    if git_mode == "120000":
        return f"symlink:{raw.decode('utf-8', errors='surrogateescape')}"
    digest = hashlib.sha256(raw).hexdigest()
    # Compare digest ignoring umask differences on tracked non-exec files while keeping exec bit.
    is_exec = git_mode == "100755"
    return f"git:{'x' if is_exec else 'f'}:{digest}"


_HOST_WALK_SKIP = frozenset({".git", ".venv", ".olympus", "__pycache__", "node_modules"})


def snapshot_host_write_scope(
    ws_root: Path,
    write_scope: Sequence[str],
    forbidden_paths: Sequence[str] = (),
) -> dict[str, str | None]:
    """Snapshot exact host ``mode:sha256`` fingerprints for all files under ``write_scope``."""
    policy = WorkspacePolicy(
        allowed_write_scope=list(write_scope),
        forbidden_write_scope=list(forbidden_paths),
    )
    root = ws_root.resolve()
    fingerprints: dict[str, str | None] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in _HOST_WALK_SKIP)
        for name in sorted(filenames):
            candidate = Path(dirpath) / name
            rel = candidate.relative_to(root).as_posix()
            if policy.forbidden_match(rel) is not None or not policy.allows_relative_path(rel):
                continue
            fingerprints[rel] = _host_path_fingerprint(candidate)
    return fingerprints


def _host_matches_baseline(
    *,
    ws_root: Path,
    rel: str,
    baseline: SandboxBaseline,
    host_baseline_fingerprints: Mapping[str, str | None] | None = None,
) -> bool:
    """Return True iff host ``ws_root / rel`` matches BOTH git ``base_commit`` and exact mode.

    Checks both:
    - Exact fingerprint (including ``stat.S_IMODE``) against the start-time snapshot so local
      mode drift (e.g. ``chmod 0o600``) fails closed.
    - Existence, type, and content against git ``base_commit`` so pre-existing uncommitted
      local host edits are never silently overwritten by a worktree-derived promotion.
    """
    dst = ws_root / rel
    current = _host_path_fingerprint(dst)
    if host_baseline_fingerprints is not None:
        expected_fp = host_baseline_fingerprints.get(rel)
        if current != expected_fp:
            return False
    if baseline.is_git and baseline.base_commit is not None:
        expected = _git_blob_fingerprint(ws_root, baseline.base_commit, rel)
        if expected is None:
            return current is None
        if expected.startswith("symlink:"):
            return current == expected
        if current is None or not expected.startswith("git:"):
            return False
        if ":" not in current or current.startswith(("symlink:", "other:")):
            return False
        cur_mode_str, cur_digest = current.split(":", 1)
        _, exp_kind, exp_digest = expected.split(":", 2)
        if cur_digest != exp_digest:
            return False
        cur_is_exec = bool(int(cur_mode_str, 8) & 0o111)
        return cur_is_exec == (exp_kind == "x")
    before_map = baseline.mapping()
    if rel in before_map:
        return current == before_map[rel]
    # Non-git fallback: missing from baseline.mapping() means file did not exist at baseline.
    return current is None


def _git_commit_all(cwd: Path, *, iteration: int, hypothesis: str) -> str:
    add_proc = _run_git(cwd, "add", "-A")
    if add_proc.returncode != 0:
        raise RuntimeError(f"git add -A failed: {add_proc.stderr.strip()}")
    short_hyp = " ".join(hypothesis.split())[:72]
    msg = f"hillclimb(iter {iteration}): {short_hyp}"
    commit_proc = _run_git(
        cwd,
        *_GIT_COMMIT_ENV_ARGS,
        "commit",
        "--no-gpg-sign",
        "-m",
        msg,
    )
    if commit_proc.returncode != 0:
        raise RuntimeError(
            f"git commit failed: {(commit_proc.stderr or commit_proc.stdout).strip()}"
        )
    return _git_head_sha(cwd)


def _git_revert_to_commit(cwd: Path, commit_sha: str) -> None:
    reset_proc = _run_git(cwd, "reset", "--hard", commit_sha)
    if reset_proc.returncode != 0:
        raise RuntimeError(f"git reset --hard {commit_sha} failed: {reset_proc.stderr.strip()}")
    clean_proc = _run_git(cwd, "clean", "-fdx")
    if clean_proc.returncode != 0:
        raise RuntimeError(f"git clean -fdx failed: {clean_proc.stderr.strip()}")


def _sha256_file_checked(workspace_root: Path, rel: str) -> str:
    """Compute SHA-256 of a regular non-symlink file strictly inside ``workspace_root``."""
    ws_resolved = workspace_root.resolve()
    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise ValueError(f"Unsafe frozen harness path: {rel!r}")
    candidate = workspace_root / rel_path
    # Reject symlinked ancestors or symlinked target files to prevent indirection tampering.
    cur = workspace_root
    for part in rel_path.parts:
        cur = cur / part
        if cur.is_symlink():
            raise ValueError(f"Frozen harness path must not traverse or be a symlink: {rel!r}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Frozen harness path missing or not a regular file: {rel!r}")
    resolved = candidate.resolve()
    if resolved == ws_resolved or not resolved.is_relative_to(ws_resolved):
        raise ValueError(f"Frozen harness path escapes workspace root: {rel!r}")
    digest = hashlib.sha256()
    with candidate.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_frozen_harness(
    workspace_root: Path,
    frozen_harness_paths: Sequence[str],
) -> dict[str, str]:
    """Snapshot baseline SHA-256 digests of every frozen harness file before iteration 1."""
    return {rel: _sha256_file_checked(workspace_root, rel) for rel in frozen_harness_paths}


def detect_frozen_harness_tamper(
    workspace_root: Path,
    frozen_hashes: Mapping[str, str],
) -> list[str]:
    """Return sorted frozen harness paths whose bytes, type, or location changed."""
    tampered: list[str] = []
    for rel, expected_digest in frozen_hashes.items():
        try:
            actual_digest = _sha256_file_checked(workspace_root, rel)
        except (OSError, ValueError):
            tampered.append(rel)
            continue
        if actual_digest != expected_digest:
            tampered.append(rel)
    return sorted(tampered)


def parse_metric_value(output: str, metric_regex: str) -> tuple[float | None, str | None]:
    """Parse an unambiguous finite numeric metric from untruncated command output."""
    pattern = re.compile(metric_regex)
    matches = list(pattern.finditer(output))
    if not matches:
        return None, f"metric_regex {metric_regex!r} did not match command output"
    parsed: list[float] = []
    for match in matches:
        raw = match.group(1) if match.lastindex else match.group(0)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None, f"matched metric token {raw!r} is not a valid float"
        if not math.isfinite(value):
            return None, f"matched metric value {raw!r} is not finite"
        parsed.append(value)
    distinct = list(dict.fromkeys(parsed))
    if len(distinct) > 1:
        return (
            None,
            f"ambiguous metric_regex {metric_regex!r}: matched conflicting values {distinct}",
        )
    return distinct[0], None


def is_strictly_better(
    candidate: float,
    incumbent: float,
    direction: Literal["minimize", "maximize"],
) -> bool:
    return candidate < incumbent if direction == "minimize" else candidate > incumbent


def is_target_reached(
    value: float,
    target_value: float | None,
    direction: Literal["minimize", "maximize"],
) -> bool:
    if target_value is None:
        return False
    return value <= target_value if direction == "minimize" else value >= target_value


def _extract_hypothesis(final_text: str, iteration: int) -> str:
    match = _HYPOTHESIS_RE.search(final_text or "")
    if match:
        cleaned = " ".join(match.group(1).split())
        if cleaned:
            return cleaned[:240]
    for line in (final_text or "").splitlines():
        cleaned = " ".join(line.split())
        if cleaned:
            return cleaned[:240]
    return f"Iteration {iteration} optimization hypothesis"


def _atomic_replace_from_src(src: Path, dst: Path) -> None:
    """Copy ``src`` to a temp sibling and ``os.replace`` so hard-linked aliases stay intact."""
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".olympus-promote-{dst.name}.",
        dir=str(dst.parent),
    )
    os.close(fd)
    tmp_path = Path(raw_tmp)
    try:
        shutil.copy2(src, tmp_path)
        os.replace(tmp_path, dst)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise


def _atomic_replace_bytes(dst: Path, data: bytes, mode: int) -> None:
    """Atomically write ``data`` + ``mode`` via temp sibling without mutating hard links."""
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".olympus-rollback-{dst.name}.",
        dir=str(dst.parent),
    )
    tmp_path = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, dst)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise


def _frozen_ro_files(sandbox_path: Path, frozen_paths: Sequence[str]) -> list[Path]:
    """Resolve existing regular frozen harness files strictly inside ``sandbox_path``."""
    ws_resolved = sandbox_path.resolve()
    ro_files: list[Path] = []
    for rel in frozen_paths:
        candidate = sandbox_path / rel
        if candidate.is_symlink() or not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if resolved != ws_resolved and resolved.is_relative_to(ws_resolved):
            ro_files.append(resolved)
    return ro_files


class HillclimbStage:
    """Cohesive stage implementing ``hillclimb_loop`` and hillclimb-aware ``verify_surface``."""

    def __init__(
        self,
        *,
        store: RunStore,
        workspace_root: Path,
        provider: LLMProvider,
        artifacts_root: Path | None = None,
        sandbox_root: Path | None = None,
        execution_drivers: ExecutionDriversConfig | None = None,
        driver_registry: DriverRegistry | None = None,
        model_provider: Callable[[str], LLMProvider] | None = None,
        on_driver_handle: Callable[[ActiveTurnHandle], None] | None = None,
        command_runner: Callable[..., tuple[int | None, str]] | None = None,
        allow_unsandboxed_metric: bool | None = None,
    ) -> None:
        self.store = store
        self.workspace_root = Path(workspace_root).resolve()
        self.provider = provider
        self.artifacts_root = (
            Path(artifacts_root).resolve()
            if artifacts_root is not None
            else default_artifacts_root(store, self.workspace_root)
        )
        self.sandbox_root = Path(sandbox_root).resolve() if sandbox_root is not None else None
        self.execution_drivers = execution_drivers or ExecutionDriversConfig.resolve()
        self.driver_registry = driver_registry or DriverRegistry()
        self.model_provider = model_provider
        self.on_driver_handle = on_driver_handle
        self._custom_runner = command_runner
        self._allow_unsandboxed_metric = (
            allow_unsandboxed_metric
            if allow_unsandboxed_metric is not None
            else os.environ.get("OLYMPUS_ALLOW_UNSANDBOXED_REPRO") == "1"
        )

    # ------------------------------------------------------------------ state persistence

    def _hillclimb_dir(self, run_id: str) -> Path:
        path = self.artifacts_root / run_id / "hillclimb"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _state_file(self, run_id: str) -> Path:
        return self._hillclimb_dir(run_id) / "state.json"

    def _decision_trail_file(self, run_id: str) -> Path:
        return self._hillclimb_dir(run_id) / "decision_trail.json"

    def _load_persisted(self, run_id: str) -> dict[str, Any]:
        path = self._state_file(run_id)
        if not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _save_persisted(self, run_id: str, updates: Mapping[str, Any]) -> dict[str, Any]:
        current = self._load_persisted(run_id)
        current.update(updates)
        self._state_file(run_id).write_text(
            json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return current

    def _write_decision_trail(
        self,
        run_id: str,
        *,
        config: HillclimbConfig,
        frozen_hashes: Mapping[str, str],
        baseline_metric: float,
        best_metric: float,
        rows: Sequence[HillclimbIterationRow],
        stopped_reason: str | None = None,
    ) -> Path:
        trail_path = self._decision_trail_file(run_id)
        tmp_path = trail_path.with_suffix(".json.tmp")
        payload = {
            "config": config.model_dump(mode="json"),
            "frozen_hashes": dict(frozen_hashes),
            "baseline_metric": baseline_metric,
            "best_metric": best_metric,
            "kept_iterations": sum(1 for row in rows if row.decision == "keep"),
            "reverted_iterations": sum(1 for row in rows if row.decision != "keep"),
            "stopped_reason": stopped_reason,
            "rows": [row.model_dump(mode="json") for row in rows],
        }
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(trail_path)
        return trail_path

    def _hydrate_model[T: BaseModel](
        self,
        run_id: str,
        state: Mapping[str, Any],
        key: str,
        model_cls: type[T],
    ) -> T | None:
        raw = state.get(key)
        if raw is None:
            raw = self._load_persisted(run_id).get(key)
        if raw is None:
            return None
        if isinstance(raw, model_cls):
            return raw
        return model_cls.model_validate(raw)

    # ------------------------------------------------------------------ driver & sandbox

    def _provider_for_model(self, model: str | None) -> LLMProvider:
        if model and self.model_provider is not None:
            return self.model_provider(model)
        return self.provider

    def _resolve_role_driver(
        self,
        *,
        run_id: str,
        node_name: str,
        role: str,
        cold_start: bool = True,
    ) -> tuple[AgentDriver | None, str | None, str]:
        raw_spec = self.execution_drivers.resolve_driver_for_role(role)
        resolved_spec = resolve_driver_spec(raw_spec, provider=self.provider)
        self.store.append_event(
            run_id=run_id,
            stage="hillclimb",
            event_type="hillclimb.role_dispatched",
            actor=f"hillclimb:{role}",
            payload={
                "node": node_name,
                "role": role,
                "driver_spec": resolved_spec,
                "cold_start": cold_start,
            },
        )
        name, _, model_override = resolved_spec.partition("@")
        model = model_override.strip() or None
        if name == "native":
            return None, model, resolved_spec
        registry = self.driver_registry.with_factories(
            default_driver_factories(
                store=self.store,
                run_id=run_id,
                provider=self.provider,
                execution_drivers=self.execution_drivers,
            )
        )
        driver, ext_model = registry.resolve_external(resolved_spec)
        return driver, ext_model, resolved_spec

    def _run_worker_turn(
        self,
        *,
        run_id: str,
        node_name: str,
        role: str,
        actor: str,
        workspace_path: Path,
        write_scope: Sequence[str],
        forbidden_globs: Sequence[str],
        system_prompt: str,
        user_prompt: str,
        max_steps: int,
        trace_name: str,
    ) -> tuple[TurnOutcome, str]:
        driver, driver_model, resolved_spec = self._resolve_role_driver(
            run_id=run_id,
            node_name=node_name,
            role=role,
            cold_start=True,
        )
        active_driver: AgentDriver = (
            driver
            if driver is not None
            else NativeDriver(
                provider=self._provider_for_model(driver_model),
                store=self.store,
                run_id=run_id,
            )
        )
        trace_path = self._hillclimb_dir(run_id) / f"{trace_name}.jsonl"
        request = DriverTurnRequest(
            workspace_path=workspace_path,
            env_root=self.workspace_root,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            actor=actor,
            read_only=False,
            strict_scope=True,
            write_scope=list(write_scope),
            forbidden_globs=list(forbidden_globs),
            max_steps=max_steps,
            trace_path=trace_path,
            model=driver_model,
        )
        outcome = active_driver.run_turn(request, on_handle=self.on_driver_handle)
        return outcome, resolved_spec

    def _ensure_sandbox(self, run_id: str) -> tuple[SandboxHandle, SandboxBaseline]:
        handle = create_sandbox(
            self.workspace_root,
            label=f"{run_id}-craft",
            reuse=True,
            sandbox_root=self.sandbox_root,
        )
        baseline = load_sandbox_baseline(handle)
        if baseline is None:
            baseline = snapshot_sandbox_state(handle)
            save_sandbox_baseline(handle, baseline)
        return handle, baseline

    def _run_safe_command(
        self,
        command: str,
        *,
        sandbox_path: Path,
        timeout_s: int,
        node_name: str,
        read_only_files: Sequence[Path] = (),
    ) -> tuple[int | None, str, str | None]:
        argv, argv_issue = _parse_safe_argv(
            command,
            host_root=self.workspace_root,
            sandbox_root=sandbox_path,
        )
        if argv_issue is not None or argv is None:
            return None, "", argv_issue or "invalid command"
        if self._custom_runner is not None:
            code, output = self._custom_runner(command, cwd=sandbox_path, timeout_s=timeout_s)
            return code, output, None
        try:
            code, output = _run_argv_command(
                argv,
                cwd=sandbox_path,
                timeout_s=timeout_s,
                host_root=self.workspace_root,
                require_fs_isolation=not self._allow_unsandboxed_metric,
                read_only_files=read_only_files,
            )
        except _SandboxIsolationUnavailableError as exc:
            raise HillclimbPause(
                node_name,
                str(exc),
                route="blocked",
                pause_kind="blocked",
            ) from exc
        return code, output, None

    def _measure_metric(
        self,
        *,
        sandbox_path: Path,
        config: HillclimbConfig,
        timeout_s: int,
        node_name: str,
    ) -> tuple[float | None, str | None, str]:
        ro_files = _frozen_ro_files(sandbox_path, config.frozen_harness_paths)
        code, raw_output, cmd_issue = self._run_safe_command(
            config.metric_command,
            sandbox_path=sandbox_path,
            timeout_s=timeout_s,
            node_name=node_name,
            read_only_files=ro_files,
        )
        excerpt = _trim_output(raw_output)
        if cmd_issue is not None:
            return None, f"Rejected metric_command: {cmd_issue}", excerpt
        tail = excerpt[-400:] if len(excerpt) > 400 else excerpt
        if code is None:
            return None, f"metric_command failed or timed out: {tail}", excerpt
        if code != 0:
            return None, f"metric_command exited {code}: {tail}", excerpt
        # Parse against full untruncated stdout+stderr before trimming excerpts.
        value, parse_err = parse_metric_value(raw_output, config.metric_regex)
        if parse_err is not None or value is None:
            return None, f"{parse_err} (output={tail!r})", excerpt
        return value, None, excerpt

    def _has_verified_argus_record(
        self,
        *,
        run_id: str,
        expected_commit_sha: str,
        expected_config_hash: str,
    ) -> bool:
        """Return True iff the LATEST Argus verify_surface event passed for commit+config."""
        argus_events = [
            event
            for event in self.store.list_events(run_id)
            if event.event_type == "craft.verify_recorded" and event.actor == "argus"
        ]
        if not argus_events:
            return False
        latest = max(argus_events, key=lambda event: event.event_seq)
        payload = latest.payload
        if not isinstance(payload, Mapping):
            return False
        return (
            payload.get("passed") is True
            and payload.get("winning_commit_sha") == expected_commit_sha
            and payload.get("config_hash") == expected_config_hash
        )

    def _promote_sandbox_to_workspace(
        self,
        handle: SandboxHandle,
        baseline: SandboxBaseline,
        config: HillclimbConfig,
        *,
        frozen_hashes: Mapping[str, str],
        expected_commit_sha: str | None = None,
        run_id: str | None = None,
        require_verified: bool = False,
        host_baseline_fingerprints: Mapping[str, str | None] | None = None,
        node_name: str = "hillclimb",
    ) -> list[str]:
        if require_verified:
            if not run_id or not expected_commit_sha:
                raise HillclimbPause(
                    node_name,
                    "Refusing to promote: run_id and expected_commit_sha are required for "
                    "Argus verification binding",
                    route="blocked",
                    pause_kind="blocked",
                )
            if not self._has_verified_argus_record(
                run_id=run_id,
                expected_commit_sha=expected_commit_sha,
                expected_config_hash=config.content_hash,
            ):
                raise HillclimbPause(
                    node_name,
                    "Refusing to promote: missing passing Argus verify_surface record in "
                    f"SQLite for commit {expected_commit_sha[:8]} and config_hash "
                    f"{config.content_hash[:12]}",
                    route="blocked",
                    pause_kind="blocked",
                )
        if expected_commit_sha is not None:
            if not handle.is_git_worktree:
                raise HillclimbPause(
                    node_name,
                    "Refusing to promote: sandbox is not a git worktree",
                    route="blocked",
                    pause_kind="blocked",
                )
            head_sha = _git_head_sha(handle.workspace_path)
            if head_sha != expected_commit_sha:
                raise HillclimbPause(
                    node_name,
                    f"Refusing to promote: sandbox HEAD ({head_sha}) does not match "
                    f"recorded winning commit ({expected_commit_sha})",
                    route="blocked",
                    pause_kind="blocked",
                )
            if _git_has_changes(handle.workspace_path, include_ignored=True):
                raise HillclimbPause(
                    node_name,
                    "Refusing to promote: sandbox working tree has uncommitted or ignored "
                    "mutations",
                    route="blocked",
                    pause_kind="blocked",
                )
        tampered = detect_frozen_harness_tamper(handle.workspace_path, frozen_hashes)
        if tampered:
            raise HillclimbPause(
                node_name,
                f"Refusing to promote: frozen harness tamper detected on {', '.join(tampered)}",
                route="blocked",
                pause_kind="blocked",
            )
        strict_gate = run_diff_gate(
            handle,
            baseline,
            allowed_write_scope=config.write_scope,
            forbidden_globs=config.frozen_harness_paths,
            strict_scope=True,
        )
        if strict_gate.rejected:
            raise HillclimbPause(
                node_name,
                strict_gate.rejection_message(),
                route="blocked",
                pause_kind="blocked",
            )
        changed = collect_sandbox_changes(handle, baseline)
        policy = WorkspacePolicy(
            allowed_write_scope=list(config.write_scope),
            forbidden_write_scope=list(config.frozen_harness_paths),
        )
        ws_root = self.workspace_root.resolve()
        # Phase 1: validate every mutation before touching any host file.
        copy_plan: list[tuple[Path, Path, str | None]] = []
        delete_plan: list[Path] = []
        applied: list[str] = []
        for rel in changed:
            if policy.forbidden_match(rel) is not None or not policy.allows_relative_path(rel):
                raise HillclimbPause(
                    node_name,
                    f"Refusing to promote unauthorized path outside write_scope: {rel}",
                    route="blocked",
                    pause_kind="blocked",
                )
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise HillclimbPause(
                    node_name,
                    f"Refusing to promote unsafe relative path: {rel}",
                    route="blocked",
                    pause_kind="blocked",
                )
            src = handle.workspace_path / rel_path
            dst = ws_root / rel_path
            parent_resolved = dst.parent.resolve(strict=False)
            if parent_resolved != ws_root and not parent_resolved.is_relative_to(ws_root):
                raise HillclimbPause(
                    node_name,
                    f"Refusing to promote path escaping workspace root: {rel}",
                    route="blocked",
                    pause_kind="blocked",
                )
            if not src.exists() and not src.is_symlink():
                if dst.exists() and not dst.is_symlink() and dst.is_dir():
                    raise HillclimbPause(
                        node_name,
                        f"Refusing to delete directory via file promotion: {rel}",
                        route="blocked",
                        pause_kind="blocked",
                    )
                delete_plan.append(dst)
                applied.append(rel)
                continue
            link_target: str | None = None
            if src.is_symlink():
                link_target = os.readlink(src)
                resolved_target = (dst.parent / link_target).resolve(strict=False)
                if resolved_target != ws_root and not resolved_target.is_relative_to(ws_root):
                    raise HillclimbPause(
                        node_name,
                        f"Refusing to promote symlink escaping workspace root: "
                        f"{rel} -> {link_target}",
                        route="blocked",
                        pause_kind="blocked",
                    )
            elif src.is_dir() or not src.is_file():
                raise HillclimbPause(
                    node_name,
                    f"Refusing to promote non-regular path/directory: {rel}",
                    route="blocked",
                    pause_kind="blocked",
                )
            if dst.exists() and not dst.is_symlink() and dst.is_dir():
                raise HillclimbPause(
                    node_name,
                    f"Refusing to overwrite host directory with file: {rel}",
                    route="blocked",
                    pause_kind="blocked",
                )
            copy_plan.append((src, dst, link_target))
            applied.append(rel)

        # Reject any symlinked ancestor directory on delete targets (prevents deleting through
        # an in-repo symlinked directory even if its target sits inside ws_root).
        for target in delete_plan:
            ancestor = target.parent
            while ancestor != ws_root and ancestor.is_relative_to(ws_root):
                if ancestor.is_symlink():
                    raise HillclimbPause(
                        node_name,
                        f"Refusing to delete path through symlinked ancestor directory: "
                        f"{target.relative_to(ws_root)}",
                        route="blocked",
                        pause_kind="blocked",
                    )
                ancestor = ancestor.parent

        delete_set = set(delete_plan)
        for _src, dst, _link_target in copy_plan:
            ancestor = dst.parent
            while ancestor != ws_root and ancestor.is_relative_to(ws_root):
                if (
                    ancestor.is_symlink() or (ancestor.exists() and not ancestor.is_dir())
                ) and ancestor not in delete_set:
                    raise HillclimbPause(
                        node_name,
                        f"Refusing to promote path whose ancestor is a non-directory: "
                        f"{dst.relative_to(ws_root)}",
                        route="blocked",
                        pause_kind="blocked",
                    )
                ancestor = ancestor.parent

        # Phase 1b: verify host workspace has NOT drifted (content, type, or mode) since baseline.
        drifted_paths = [
            rel
            for rel in changed
            if not _host_matches_baseline(
                ws_root=ws_root,
                rel=rel,
                baseline=baseline,
                host_baseline_fingerprints=host_baseline_fingerprints,
            )
        ]
        if drifted_paths:
            raise HillclimbPause(
                node_name,
                "Refusing to promote: host workspace drifted since hillclimb baseline on "
                + ", ".join(sorted(drifted_paths)),
                route="blocked",
                pause_kind="blocked",
            )

        # Phase 2: transactional apply with full backup+mode+hardlink rollback on any OSError.
        backups: list[
            tuple[Path, str, tuple[bytes, int] | tuple[Path, bytes, int] | str | int | None]
        ] = []
        temp_backup_links: list[Path] = []
        created_dirs: list[Path] = []
        try:
            for target in delete_plan:
                if target.is_symlink():
                    backups.append((target, "symlink", os.readlink(target)))
                    target.unlink()
                elif target.is_file():
                    st = target.stat()
                    mode = stat.S_IMODE(st.st_mode)
                    data = target.read_bytes()
                    if st.st_nlink > 1:
                        hl_bak = target.parent / f".olympus-hlbak-{target.name}.{uuid4().hex[:8]}"
                        os.link(target, hl_bak)
                        temp_backup_links.append(hl_bak)
                        backups.append((target, "hardlink_file", (hl_bak, data, mode)))
                    else:
                        backups.append((target, "file", (data, mode)))
                    target.unlink()
                parent = target.parent
                while parent != ws_root and parent.is_relative_to(ws_root):
                    try:
                        dir_mode = stat.S_IMODE(parent.stat().st_mode)
                        parent.rmdir()
                    except OSError:
                        break
                    backups.append((parent, "removed_dir", dir_mode))
                    parent = parent.parent
            for src, dst, link_target in copy_plan:
                missing_parents: list[Path] = []
                cur_parent = dst.parent
                while cur_parent != ws_root and not cur_parent.exists():
                    missing_parents.append(cur_parent)
                    cur_parent = cur_parent.parent
                if missing_parents:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    created_dirs.extend(missing_parents)
                if dst.is_symlink():
                    backups.append((dst, "symlink", os.readlink(dst)))
                    dst.unlink()
                elif dst.is_file():
                    st = dst.stat()
                    mode = stat.S_IMODE(st.st_mode)
                    data = dst.read_bytes()
                    if st.st_nlink > 1:
                        # Retain original inode hard link so rollback restores alias identity.
                        hl_bak = dst.parent / f".olympus-hlbak-{dst.name}.{uuid4().hex[:8]}"
                        os.link(dst, hl_bak)
                        temp_backup_links.append(hl_bak)
                        backups.append((dst, "hardlink_file", (hl_bak, data, mode)))
                    else:
                        backups.append((dst, "file", (data, mode)))
                else:
                    backups.append((dst, "missing", None))
                if link_target is not None:
                    if dst.is_file() or dst.is_symlink():
                        dst.unlink()
                    os.symlink(link_target, dst)
                else:
                    # Atomic temp-sibling replace avoids mutating hard-linked aliases in place.
                    _atomic_replace_from_src(src, dst)
        except OSError as exc:
            rollback_errors: list[str] = []
            restore_dir_modes: list[tuple[Path, int]] = []
            for path, kind, saved in reversed(backups):
                try:
                    if kind == "missing":
                        if path.is_symlink() or path.is_file():
                            path.unlink(missing_ok=True)
                    elif kind == "removed_dir" and isinstance(saved, int):
                        path.mkdir(parents=True, exist_ok=True)
                        # Keep writable until deleted children are restored, then restore mode.
                        os.chmod(path, saved | 0o700)
                        restore_dir_modes.append((path, saved))
                    elif kind == "symlink" and isinstance(saved, str):
                        if path.is_symlink() and os.readlink(path) == saved:
                            continue
                        if path.is_symlink() or path.is_file():
                            path.unlink(missing_ok=True)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        os.symlink(saved, path)
                    elif kind == "hardlink_file" and isinstance(saved, tuple) and len(saved) == 3:
                        hl_bak, data, saved_mode = saved
                        path.parent.mkdir(parents=True, exist_ok=True)
                        if hl_bak.exists():
                            os.replace(hl_bak, path)
                            os.chmod(path, saved_mode)
                        else:
                            _atomic_replace_bytes(path, data, saved_mode)
                    elif kind == "file" and isinstance(saved, tuple) and len(saved) == 2:
                        data, saved_mode = saved
                        if path.is_symlink():
                            path.unlink(missing_ok=True)
                        elif (
                            path.is_file()
                            and path.read_bytes() == data
                            and stat.S_IMODE(path.stat().st_mode) == saved_mode
                        ):
                            continue
                        path.parent.mkdir(parents=True, exist_ok=True)
                        _atomic_replace_bytes(path, data, saved_mode)
                except OSError as rb_exc:
                    rollback_errors.append(f"{path.name}: {rb_exc}")
            for hl_bak in temp_backup_links:
                hl_bak.unlink(missing_ok=True)
            for dir_path, dir_mode in sorted(
                restore_dir_modes, key=lambda item: len(item[0].parts), reverse=True
            ):
                try:
                    os.chmod(dir_path, dir_mode)
                except OSError as mode_exc:
                    rollback_errors.append(f"{dir_path.name}: {mode_exc}")
            for directory in sorted(
                created_dirs, key=lambda item: len(item.parts), reverse=True
            ):
                try:
                    directory.rmdir()
                except OSError as dir_exc:
                    rollback_errors.append(f"{directory.name}: {dir_exc}")
            extra = (
                f" (partial rollback errors: {'; '.join(rollback_errors)})"
                if rollback_errors
                else ""
            )
            raise HillclimbPause(
                node_name,
                f"Atomic workspace promotion failed and was rolled back: {exc}{extra}",
                route="blocked",
                pause_kind="blocked",
            ) from exc
        for hl_bak in temp_backup_links:
            hl_bak.unlink(missing_ok=True)
        return sorted(set(applied))

    def promote_winner(
        self,
        *,
        run_id: str,
        state: Mapping[str, Any] | None = None,
        config: HillclimbConfig | Mapping[str, Any] | None = None,
        node_name: str = "verify_surface",
    ) -> list[str]:
        """Atomically promote the SQLite-bound winning commit AFTER Argus verification."""
        state_map: Mapping[str, Any] = state or {}
        cfg = self._resolve_config(
            run_id=run_id,
            state=state_map,
            explicit_config=config,
            node_name=node_name,
        )
        auth = self._load_authoritative_sqlite_state(run_id, cfg, node_name=node_name)
        handle, baseline = self._ensure_sandbox(run_id)
        applied_paths = self._promote_sandbox_to_workspace(
            handle,
            baseline,
            cfg,
            frozen_hashes=auth["frozen_hashes"],
            expected_commit_sha=auth["expected_commit_sha"],
            run_id=run_id,
            require_verified=True,
            host_baseline_fingerprints=auth.get("host_baseline_fingerprints"),
            node_name=node_name,
        )
        remove_sandbox(handle)
        return applied_paths

    # ------------------------------------------------------------------ config resolution

    def _resolve_config(
        self,
        *,
        run_id: str,
        state: Mapping[str, Any],
        explicit_config: HillclimbConfig | Mapping[str, Any] | None,
        node_name: str,
    ) -> HillclimbConfig:
        if isinstance(explicit_config, HillclimbConfig):
            return explicit_config
        merged: dict[str, Any] = {}
        persisted = self._load_persisted(run_id).get("hillclimb_config")
        if isinstance(persisted, Mapping):
            merged.update(persisted)
        for event in self.store.list_events(run_id):
            if event.event_type != "session.started" or not isinstance(event.payload, Mapping):
                continue
            ev_hc = event.payload.get("hillclimb_config")
            if isinstance(ev_hc, Mapping):
                merged.update(ev_hc)
            ev_meta = event.payload.get("metadata")
            if isinstance(ev_meta, Mapping) and isinstance(
                ev_meta.get("hillclimb_config"), Mapping
            ):
                merged.update(ev_meta["hillclimb_config"])
        if isinstance(explicit_config, Mapping):
            merged.update(explicit_config)
        state_meta = state.get("metadata")
        if isinstance(state_meta, Mapping) and isinstance(
            state_meta.get("hillclimb_config"), Mapping
        ):
            merged.update(state_meta["hillclimb_config"])
        state_cfg = state.get("hillclimb_config")
        if isinstance(state_cfg, HillclimbConfig):
            return state_cfg
        if isinstance(state_cfg, Mapping):
            merged.update(state_cfg)
        if not merged:
            raise HillclimbPause(
                node_name,
                "HillclimbConfig is required in state['hillclimb_config'], metadata, or params",
                route="blocked",
                pause_kind="blocked",
            )
        try:
            return HillclimbConfig.model_validate(merged)
        except ValidationError as exc:
            raise HillclimbPause(
                node_name,
                f"Invalid HillclimbConfig: {exc}",
                route="blocked",
                pause_kind="blocked",
            ) from exc

    def _load_sqlite_history(
        self, run_id: str
    ) -> tuple[dict[str, Any] | None, list[HillclimbIterationRow]]:
        """Load baseline payload and recorded iteration rows from SQLite for resume/verification."""
        baseline_payload: dict[str, Any] | None = None
        prior_rows: list[HillclimbIterationRow] = []
        row_fields = set(HillclimbIterationRow.model_fields)
        for event in self.store.list_events(run_id):
            if (
                event.event_type == "hillclimb.baseline_recorded"
                and baseline_payload is None
                and isinstance(event.payload, Mapping)
            ):
                baseline_payload = dict(event.payload)
            elif (
                event.event_type == "hillclimb.iteration_recorded"
                and isinstance(event.payload, Mapping)
            ):
                row_data = {k: v for k, v in event.payload.items() if k in row_fields}
                prior_rows.append(HillclimbIterationRow.model_validate(row_data))
        return baseline_payload, prior_rows

    def _load_authoritative_sqlite_state(
        self,
        run_id: str,
        cfg: HillclimbConfig,
        *,
        node_name: str,
    ) -> dict[str, Any]:
        """Reconstruct authoritative metrics, frozen hashes, and winning commit from SQLite."""
        baseline_payload, sqlite_rows = self._load_sqlite_history(run_id)
        if baseline_payload is None:
            raise HillclimbPause(
                node_name,
                "Missing SQLite hillclimb.baseline_recorded audit event",
                route="blocked",
                pause_kind="blocked",
            )
        recorded_hash = baseline_payload.get("config_hash")
        if recorded_hash != cfg.content_hash:
            raise HillclimbPause(
                node_name,
                f"HillclimbConfig hash ({cfg.content_hash[:12]}) does not match SQLite "
                f"recorded baseline config_hash ({str(recorded_hash)[:12]})",
                route="blocked",
                pause_kind="blocked",
            )
        if (
            baseline_payload.get("direction") != cfg.direction
            or baseline_payload.get("metric_command") != cfg.metric_command
            or baseline_payload.get("metric_regex") != cfg.metric_regex
        ):
            raise HillclimbPause(
                node_name,
                "HillclimbConfig metric parameters disagree with SQLite recorded baseline",
                route="blocked",
                pause_kind="blocked",
            )
        raw_hashes = baseline_payload.get("frozen_harness_hashes") or baseline_payload.get(
            "frozen_hashes"
        )
        if not isinstance(raw_hashes, Mapping) or not raw_hashes:
            raise HillclimbPause(
                node_name,
                "SQLite hillclimb.baseline_recorded event is missing frozen_harness_hashes",
                route="blocked",
                pause_kind="blocked",
            )
        frozen_hashes = {str(k): str(v) for k, v in raw_hashes.items()}
        if sorted(frozen_hashes) != sorted(cfg.frozen_harness_paths):
            raise HillclimbPause(
                node_name,
                "HillclimbConfig frozen_harness_paths disagree with SQLite recorded baseline",
                route="blocked",
                pause_kind="blocked",
            )
        raw_base_metric = baseline_payload.get("baseline_metric")
        raw_base_commit = baseline_payload.get("base_commit")
        if not isinstance(raw_base_metric, (int, float)) or not isinstance(raw_base_commit, str):
            raise HillclimbPause(
                node_name,
                "SQLite hillclimb.baseline_recorded event is missing baseline_metric/base_commit",
                route="blocked",
                pause_kind="blocked",
            )
        baseline_metric = float(raw_base_metric)
        best_metric = baseline_metric
        expected_commit_sha = raw_base_commit.strip()
        consecutive_non_improving = 0
        for row in sqlite_rows:
            if row.decision == "keep":
                assert row.metric_after is not None and row.commit_sha is not None
                best_metric = row.metric_after
                expected_commit_sha = row.commit_sha
                consecutive_non_improving = 0
            else:
                consecutive_non_improving += 1
        raw_host_fps = baseline_payload.get("host_baseline_fingerprints")
        host_baseline_fingerprints: dict[str, str | None] | None = (
            {str(k): (None if v is None else str(v)) for k, v in raw_host_fps.items()}
            if isinstance(raw_host_fps, Mapping)
            else None
        )
        return {
            "baseline_payload": baseline_payload,
            "rows": sqlite_rows,
            "frozen_hashes": frozen_hashes,
            "baseline_metric": baseline_metric,
            "best_metric": best_metric,
            "expected_commit_sha": expected_commit_sha,
            "consecutive_non_improving": consecutive_non_improving,
            "host_baseline_fingerprints": host_baseline_fingerprints,
        }

    # ------------------------------------------------------------------ composite primitive

    def hillclimb_loop(
        self,
        *,
        run_id: str,
        raw_input: str,
        state: Mapping[str, Any] | None = None,
        config: HillclimbConfig | Mapping[str, Any] | None = None,
        node_name: str = "hillclimb",
        role: str = "builder",
        skill_refs: Sequence[str] = ("pstack/hillclimb",),
        max_steps: int = 16,
        metric_timeout_s: int = 60,
        apply_to_workspace: bool = False,
    ) -> dict[str, Any]:
        """Run the deterministic frozen-harness keep/revert optimization loop."""
        state_map: Mapping[str, Any] = state or {}
        cfg = self._resolve_config(
            run_id=run_id,
            state=state_map,
            explicit_config=config,
            node_name=node_name,
        )
        handle, baseline = self._ensure_sandbox(run_id)
        if not handle.is_git_worktree:
            raise HillclimbPause(
                node_name,
                "hillclimb_loop requires a git worktree sandbox for deterministic keep/revert",
                route="blocked",
                pause_kind="blocked",
            )

        prior_baseline, prior_rows = self._load_sqlite_history(run_id)
        rows: list[HillclimbIterationRow] = list(prior_rows)
        consecutive_non_improving = 0

        if prior_baseline is not None:
            # Resume/retry on an existing run_id: enforce config_hash parity and preserve rows.
            auth = self._load_authoritative_sqlite_state(run_id, cfg, node_name=node_name)
            frozen_hashes = auth["frozen_hashes"]
            baseline_metric = auth["baseline_metric"]
            best_metric = auth["best_metric"]
            last_kept_sha = auth["expected_commit_sha"]
            consecutive_non_improving = auth["consecutive_non_improving"]
            # Reconcile worktree cleanly back to last_kept_sha before resuming.
            if (
                _git_head_sha(handle.workspace_path) != last_kept_sha
                or _git_has_changes(handle.workspace_path, include_ignored=True)
            ):
                _git_revert_to_commit(handle.workspace_path, last_kept_sha)
            resume_tamper = detect_frozen_harness_tamper(handle.workspace_path, frozen_hashes)
            if resume_tamper:
                raise HillclimbPause(
                    node_name,
                    "Frozen harness does not match recorded baseline on resume: "
                    + ", ".join(resume_tamper),
                    route="blocked",
                    pause_kind="blocked",
                )
        else:
            # 1. Freeze harness SHA-256 fingerprints BEFORE iteration 1.
            try:
                frozen_hashes = snapshot_frozen_harness(
                    handle.workspace_path, cfg.frozen_harness_paths
                )
            except (OSError, ValueError) as exc:
                raise HillclimbPause(
                    node_name,
                    f"Failed to freeze harness before iteration 1: {exc}",
                    route="blocked",
                    pause_kind="blocked",
                ) from exc

            last_kept_sha = _git_head_sha(handle.workspace_path)
            # Snapshot exact host write_scope modes+hashes BEFORE baseline metric execution so
            # local host chmod/edits during baseline measurement are caught on promotion.
            host_baseline_fingerprints = snapshot_host_write_scope(
                self.workspace_root, cfg.write_scope, cfg.frozen_harness_paths
            )

            # 2. Measure baseline metric inside the isolated worktree (fail closed).
            measured_base, base_err, _base_output = self._measure_metric(
                sandbox_path=handle.workspace_path,
                config=cfg,
                timeout_s=metric_timeout_s,
                node_name=node_name,
            )
            post_base_tamper = detect_frozen_harness_tamper(handle.workspace_path, frozen_hashes)
            if post_base_tamper:
                raise HillclimbPause(
                    node_name,
                    "Frozen harness mutated during baseline measurement: "
                    + ", ".join(post_base_tamper),
                    route="blocked",
                    pause_kind="blocked",
                )
            if base_err is not None or measured_base is None:
                raise HillclimbPause(
                    node_name,
                    f"Baseline metric measurement failed: {base_err}",
                    route="blocked",
                    pause_kind="blocked",
                )
            baseline_metric = measured_base

            self.store.append_event(
                run_id=run_id,
                stage="hillclimb",
                event_type="hillclimb.baseline_recorded",
                actor="hephaestus",
                payload={
                    "config_id": cfg.config_id,
                    "config_hash": cfg.content_hash,
                    "baseline_metric": baseline_metric,
                    "best_metric": baseline_metric,
                    "direction": cfg.direction,
                    "target_value": cfg.target_value,
                    "metric_command": cfg.metric_command,
                    "metric_regex": cfg.metric_regex,
                    "write_scope": list(cfg.write_scope),
                    "base_commit": last_kept_sha,
                    "frozen_hashes": dict(frozen_hashes),
                    "frozen_harness_hashes": dict(frozen_hashes),
                    "host_baseline_fingerprints": host_baseline_fingerprints,
                },
            )
            best_metric = baseline_metric

        stopped_reason: Literal[
            "target_reached", "plateau", "max_iterations", "error", "blocked"
        ] = "max_iterations"

        brief = self._hydrate_model(run_id, state_map, "evidence_brief", EvidenceBrief)
        evidence_block = (
            f"\nExploration evidence summary:\n{brief.summary}\n"
            if brief is not None
            else ""
        )
        system_prompt = _compose_system_prompt(
            "You are Hephaestus running one iteration of a frozen-harness hill-climbing loop. "
            "Propose ONE grounded optimization hypothesis, apply a clean edit strictly inside "
            "`write_scope`, never touch `frozen_harness_paths`, and start your final summary "
            "with `HYPOTHESIS: <concise description>`.",
            skill_refs,
        )

        # Persist current trail (preserving any resumed SQLite rows) before running new iterations.
        trail_path = self._write_decision_trail(
            run_id,
            config=cfg,
            frozen_hashes=frozen_hashes,
            baseline_metric=baseline_metric,
            best_metric=best_metric,
            rows=rows,
            stopped_reason=None,
        )

        start_iteration = (rows[-1].iteration + 1) if rows else 1
        if is_target_reached(best_metric, cfg.target_value, cfg.direction):
            stopped_reason = "target_reached"
        elif consecutive_non_improving >= cfg.plateau_patience:
            stopped_reason = "plateau"
        elif start_iteration > cfg.max_iterations:
            stopped_reason = "max_iterations"
        else:
            for iteration in range(start_iteration, cfg.max_iterations + 1):
                iter_t0 = time.monotonic()
                metric_before = best_metric
                history_lines = (
                    "\n".join(
                        f"- iter {r.iteration}: [{r.decision}] "
                        f"{r.metric_before} -> {r.metric_after} | {r.hypothesis} ({r.reason})"
                        for r in rows
                    )
                    if rows
                    else "(none yet)"
                )
                user_prompt = (
                    f"Goal:\n{raw_input}\n"
                    f"{evidence_block}\n"
                    f"Hillclimb Iteration: {iteration}/{cfg.max_iterations}\n"
                    f"Direction: {cfg.direction}\n"
                    f"Baseline metric: {baseline_metric}\n"
                    f"Current best metric: {best_metric}\n"
                    f"Target value: {cfg.target_value}\n"
                    f"Metric command: {cfg.metric_command}\n"
                    f"Metric regex: {cfg.metric_regex}\n"
                    f"Allowed write_scope: {json.dumps(cfg.write_scope)}\n"
                    f"FROZEN harness paths (DO NOT EDIT): "
                    f"{json.dumps(cfg.frozen_harness_paths)}\n\n"
                    f"Prior iterations:\n{history_lines}\n\n"
                    "Make ONE focused optimization edit strictly inside write_scope. "
                    "End with `HYPOTHESIS: <what you changed and why it improves the metric>`."
                )

                outcome, _resolved_spec = self._run_worker_turn(
                    run_id=run_id,
                    node_name=node_name,
                    role=role,
                    actor=f"hephaestus:hillclimb_iter_{iteration}",
                    workspace_path=handle.workspace_path,
                    write_scope=cfg.write_scope,
                    forbidden_globs=cfg.frozen_harness_paths,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_steps=max_steps,
                    trace_name=f"iter_{iteration}",
                )
                hypothesis = _extract_hypothesis(outcome.final_content, iteration)

                decision: Literal["keep", "revert", "rejected_tamper", "error"]
                metric_after: float | None = None
                commit_sha: str | None = None
                reason = ""

                # Step A: Deterministic anti-tamper guard BEFORE running any metric command.
                tampered = detect_frozen_harness_tamper(handle.workspace_path, frozen_hashes)
                if tampered:
                    _git_revert_to_commit(handle.workspace_path, last_kept_sha)
                    decision = "rejected_tamper"
                    reason = (
                        "Rejected and reverted: frozen harness modified "
                        f"({', '.join(tampered)})"
                    )
                    consecutive_non_improving += 1
                else:
                    # Step B: Clean ephemeral junk and enforce write_scope via DiffGate.
                    clean_ephemeral_artifacts(
                        handle,
                        baseline,
                        write_scope=[*cfg.write_scope, *cfg.frozen_harness_paths],
                    )
                    gate = run_diff_gate(
                        handle,
                        baseline,
                        allowed_write_scope=cfg.write_scope,
                        forbidden_globs=cfg.frozen_harness_paths,
                        strict_scope=True,
                    )
                    head_after_turn = _git_head_sha(handle.workspace_path)
                    has_iter_delta = (head_after_turn != last_kept_sha) or _git_has_changes(
                        handle.workspace_path
                    )

                    if gate.rejected:
                        violating_frozen = [
                            violation.path
                            for violation in gate.violations
                            if violation.path in frozen_hashes
                        ]
                        _git_revert_to_commit(handle.workspace_path, last_kept_sha)
                        if violating_frozen:
                            decision = "rejected_tamper"
                            reason = (
                                "Rejected and reverted: frozen harness touched "
                                f"({', '.join(sorted(set(violating_frozen)))})"
                            )
                        else:
                            decision = "error"
                            reason = f"Reverted: {gate.rejection_message()}"
                        consecutive_non_improving += 1
                    elif not has_iter_delta:
                        _git_revert_to_commit(handle.workspace_path, last_kept_sha)
                        if not outcome.completed:
                            decision = "error"
                            reason = (
                                f"Worker turn did not complete without edits "
                                f"(stop_reason={outcome.stop_reason})"
                            )
                        else:
                            decision = "revert"
                            reason = "No edits produced inside write_scope"
                        consecutive_non_improving += 1
                    else:
                        # Re-verify frozen harness right before metric execution (fail closed).
                        pre_metric_tamper = detect_frozen_harness_tamper(
                            handle.workspace_path, frozen_hashes
                        )
                        if pre_metric_tamper:
                            _git_revert_to_commit(handle.workspace_path, last_kept_sha)
                            decision = "rejected_tamper"
                            reason = (
                                "Rejected and reverted before metric run: frozen harness modified "
                                f"({', '.join(pre_metric_tamper)})"
                            )
                            consecutive_non_improving += 1
                        else:
                            # Step C: Measure candidate metric safely and re-check anti-tamper.
                            measured, measure_err, _measure_out = self._measure_metric(
                                sandbox_path=handle.workspace_path,
                                config=cfg,
                                timeout_s=metric_timeout_s,
                                node_name=node_name,
                            )
                            runtime_tamper = detect_frozen_harness_tamper(
                                handle.workspace_path, frozen_hashes
                            )
                            if runtime_tamper:
                                _git_revert_to_commit(handle.workspace_path, last_kept_sha)
                                decision = "rejected_tamper"
                                reason = (
                                    "Rejected and reverted: frozen harness modified during "
                                    f"metric run ({', '.join(runtime_tamper)})"
                                )
                                consecutive_non_improving += 1
                            elif measure_err is not None or measured is None:
                                _git_revert_to_commit(handle.workspace_path, last_kept_sha)
                                decision = "error"
                                reason = f"Reverted after metric error: {measure_err}"
                                consecutive_non_improving += 1
                            elif is_strictly_better(measured, best_metric, cfg.direction):
                                # Re-clean ephemeral artifacts created during metric run.
                                clean_ephemeral_artifacts(
                                    handle,
                                    baseline,
                                    write_scope=[*cfg.write_scope, *cfg.frozen_harness_paths],
                                )
                                post_gate = run_diff_gate(
                                    handle,
                                    baseline,
                                    allowed_write_scope=cfg.write_scope,
                                    forbidden_globs=cfg.frozen_harness_paths,
                                    strict_scope=True,
                                )
                                if post_gate.rejected:
                                    _git_revert_to_commit(handle.workspace_path, last_kept_sha)
                                    decision = "error"
                                    reason = (
                                        "Reverted: metric run produced out-of-scope mutations "
                                        f"({post_gate.rejection_message()})"
                                    )
                                    consecutive_non_improving += 1
                                else:
                                    if _git_has_changes(handle.workspace_path):
                                        commit_sha = _git_commit_all(
                                            handle.workspace_path,
                                            iteration=iteration,
                                            hypothesis=hypothesis,
                                        )
                                    else:
                                        commit_sha = _git_head_sha(handle.workspace_path)
                                    last_kept_sha = commit_sha
                                    best_metric = measured
                                    metric_after = measured
                                    decision = "keep"
                                    delta = measured - metric_before
                                    reason = (
                                        f"Improved ({cfg.direction}): "
                                        f"{metric_before:g} -> {measured:g} (delta={delta:+g})"
                                    )
                                    consecutive_non_improving = 0
                            else:
                                _git_revert_to_commit(handle.workspace_path, last_kept_sha)
                                metric_after = measured
                                decision = "revert"
                                delta = measured - metric_before
                                reason = (
                                    f"No improvement ({cfg.direction}): "
                                    f"{metric_before:g} -> {measured:g} (delta={delta:+g})"
                                )
                                consecutive_non_improving += 1

                elapsed_s = round(max(0.0, time.monotonic() - iter_t0), 6)
                row = HillclimbIterationRow(
                    iteration=iteration,
                    hypothesis=hypothesis,
                    metric_before=metric_before,
                    metric_after=metric_after,
                    decision=decision,
                    commit_sha=commit_sha,
                    reason=reason,
                    elapsed_s=elapsed_s,
                )
                rows.append(row)

                iter_payload = {
                    **row.model_dump(mode="json"),
                    "config_hash": cfg.content_hash,
                    "baseline_metric": baseline_metric,
                    "best_metric": best_metric,
                    "direction": cfg.direction,
                    "frozen_harness_hashes": dict(frozen_hashes),
                    "metric_command": cfg.metric_command,
                    "metric_regex": cfg.metric_regex,
                }
                self.store.append_event(
                    run_id=run_id,
                    stage="hillclimb",
                    event_type="hillclimb.iteration_recorded",
                    actor="hephaestus",
                    payload=iter_payload,
                )
                trail_path = self._write_decision_trail(
                    run_id,
                    config=cfg,
                    frozen_hashes=frozen_hashes,
                    baseline_metric=baseline_metric,
                    best_metric=best_metric,
                    rows=rows,
                    stopped_reason=None,
                )

                if is_target_reached(best_metric, cfg.target_value, cfg.direction):
                    stopped_reason = "target_reached"
                    break
                if consecutive_non_improving >= cfg.plateau_patience:
                    stopped_reason = "plateau"
                    break
                if iteration >= cfg.max_iterations:
                    stopped_reason = "max_iterations"
                    break

        kept_count = sum(1 for row in rows if row.decision == "keep")
        reverted_count = sum(1 for row in rows if row.decision != "keep")
        report = HillclimbReport(
            baseline_metric=baseline_metric,
            best_metric=best_metric,
            kept_iterations=kept_count,
            reverted_iterations=reverted_count,
            stopped_reason=stopped_reason,
            rows=rows,
        )
        trail_path = self._write_decision_trail(
            run_id,
            config=cfg,
            frozen_hashes=frozen_hashes,
            baseline_metric=baseline_metric,
            best_metric=best_metric,
            rows=rows,
            stopped_reason=stopped_reason,
        )
        dumped_cfg = cfg.model_dump(mode="json")
        dumped_report = report.model_dump(mode="json")
        self._save_persisted(
            run_id,
            {
                "hillclimb_config": dumped_cfg,
                "hillclimb_report": dumped_report,
                "frozen_hashes": dict(frozen_hashes),
                "last_kept_sha": last_kept_sha,
                "decision_trail_path": str(trail_path),
            },
        )
        self.store.append_event(
            run_id=run_id,
            stage="hillclimb",
            event_type="hillclimb.completed",
            actor="hephaestus",
            payload={
                "report_id": report.report_id,
                "content_hash": report.content_hash,
                "baseline_metric": report.baseline_metric,
                "best_metric": report.best_metric,
                "kept_iterations": report.kept_iterations,
                "reverted_iterations": report.reverted_iterations,
                "stopped_reason": report.stopped_reason,
                "decision_trail_path": str(trail_path),
            },
        )

        result_state: dict[str, Any] = {
            "hillclimb_config": dumped_cfg,
            "hillclimb_report": dumped_report,
            "hillclimb_frozen_hashes": dict(frozen_hashes),
            "hillclimb_winning_commit_sha": last_kept_sha,
            "hillclimb_decision_trail_path": str(trail_path),
            "applied_paths": [],
            "route": "done",
        }
        if apply_to_workspace:
            # Never bypass Argus verify_surface before promoting to host workspace.
            verify_out = self.verify_surface(
                run_id=run_id,
                state=result_state,
                node_name=node_name,
                apply_to_workspace=True,
                timeout_s=metric_timeout_s,
            )
            result_state["surface_verification"] = verify_out["surface_verification"]
            result_state["applied_paths"] = verify_out["applied_paths"]
        return result_state

    # ------------------------------------------------------------------ verify_surface

    def verify_surface(
        self,
        *,
        run_id: str,
        state: Mapping[str, Any],
        node_name: str = "verify_surface",
        role: str = "verifier",
        commands: Sequence[str] | None = None,
        apply_to_workspace: bool = False,
        timeout_s: int = 120,
    ) -> dict[str, Any]:
        """Real-surface Argus verification after ``hillclimb_loop`` plus atomic promotion."""
        cfg = self._resolve_config(
            run_id=run_id,
            state=state,
            explicit_config=None,
            node_name=node_name,
        )
        auth = self._load_authoritative_sqlite_state(run_id, cfg, node_name=node_name)
        frozen_hashes: dict[str, str] = auth["frozen_hashes"]
        auth_baseline_metric: float = auth["baseline_metric"]
        auth_best_metric: float = auth["best_metric"]
        expected_commit_sha: str = auth["expected_commit_sha"]
        sqlite_rows: list[HillclimbIterationRow] = auth["rows"]

        # Cross-check optional caller-supplied report against SQLite authoritative state so
        # forged or stale in-memory state cannot override metric thresholds or commits.
        supplied_report = self._hydrate_model(run_id, state, "hillclimb_report", HillclimbReport)
        if supplied_report is not None:
            kept_sqlite = [row for row in sqlite_rows if row.decision == "keep"]
            kept_report = [row for row in supplied_report.rows if row.decision == "keep"]
            if (
                len(kept_report) != len(kept_sqlite)
                or (kept_report and kept_report[-1].commit_sha != expected_commit_sha)
                or supplied_report.baseline_metric != auth_baseline_metric
                or supplied_report.best_metric != auth_best_metric
            ):
                raise HillclimbPause(
                    node_name,
                    "Supplied HillclimbReport disagrees with authoritative SQLite audit trail",
                    route="blocked",
                    pause_kind="blocked",
                )

        handle, _baseline = self._ensure_sandbox(run_id)
        if not handle.is_git_worktree:
            raise HillclimbPause(
                node_name,
                "Hillclimb verification requires a git worktree sandbox",
                route="blocked",
                pause_kind="blocked",
            )
        self._resolve_role_driver(
            run_id=run_id,
            node_name=node_name,
            role=role,
            cold_start=True,
        )

        checks: list[SurfaceCheckResult] = []
        short_circuit = False

        # 1. Verify frozen harness SHA-256 anti-tamper integrity BEFORE running any command.
        tampered = detect_frozen_harness_tamper(handle.workspace_path, frozen_hashes)
        if tampered:
            short_circuit = True
            _git_revert_to_commit(handle.workspace_path, expected_commit_sha)
        checks.append(
            SurfaceCheckResult(
                surface="frozen_harness_integrity",
                command="sha256(frozen_harness_paths)",
                status="fail" if tampered else "pass",
                exit_code=1 if tampered else 0,
                evidence=(
                    f"Tampered frozen harness paths (reverted to {expected_commit_sha[:8]}; "
                    f"skipped metric execution): {', '.join(tampered)}"
                    if tampered
                    else f"All {len(frozen_hashes)} frozen harness file(s) match baseline SHA-256."
                ),
            )
        )

        # 2. Bind sandbox explicitly to the SQLite-recorded winning commit & clean tree.
        if not short_circuit:
            current_head = _git_head_sha(handle.workspace_path)
            dirty_before = _git_has_changes(handle.workspace_path, include_ignored=True)
            binding_ok = (current_head == expected_commit_sha) and not dirty_before
            if not binding_ok:
                short_circuit = True
            checks.append(
                SurfaceCheckResult(
                    surface="winning_commit_binding",
                    command="git rev-parse HEAD && git status --porcelain --ignored",
                    status="pass" if binding_ok else "fail",
                    exit_code=0 if binding_ok else 1,
                    evidence=(
                        f"Sandbox HEAD matches SQLite winning commit {expected_commit_sha[:8]} "
                        "with clean working tree."
                        if binding_ok
                        else (
                            f"Sandbox state mismatch before verification: HEAD={current_head[:8]} "
                            f"(expected {expected_commit_sha[:8]}), dirty={dirty_before}."
                        )
                    ),
                )
            )

        # 3. Re-run frozen metric command (with --ro-bind frozen harness) & check vs SQLite best.
        if not short_circuit:
            measured, measure_err, measure_out = self._measure_metric(
                sandbox_path=handle.workspace_path,
                config=cfg,
                timeout_s=timeout_s,
                node_name=node_name,
            )
            post_tamper = detect_frozen_harness_tamper(handle.workspace_path, frozen_hashes)
            if post_tamper:
                short_circuit = True
                checks.append(
                    SurfaceCheckResult(
                        surface="hillclimb_metric",
                        command=cfg.metric_command,
                        status="fail",
                        exit_code=1,
                        evidence=(
                            "Frozen harness modified during verification metric run: "
                            + ", ".join(post_tamper)
                        ),
                    )
                )
            elif measure_err is not None or measured is None:
                checks.append(
                    SurfaceCheckResult(
                        surface="hillclimb_metric",
                        command=cfg.metric_command,
                        status="inconclusive",
                        exit_code=None,
                        evidence=measure_err
                        or "Metric failed to parse; inconclusive never passes.",
                    )
                )
            else:
                regressed_vs_best = is_strictly_better(
                    auth_best_metric, measured, cfg.direction
                )
                regressed_vs_baseline = is_strictly_better(
                    auth_baseline_metric, measured, cfg.direction
                )
                failed_metric = regressed_vs_best or regressed_vs_baseline
                checks.append(
                    SurfaceCheckResult(
                        surface="hillclimb_metric",
                        command=cfg.metric_command,
                        status="fail" if failed_metric else "pass",
                        exit_code=1 if failed_metric else 0,
                        evidence=(
                            f"Verified metric={measured:g} vs best={auth_best_metric:g} "
                            f"and baseline={auth_baseline_metric:g} ({cfg.direction}); "
                            f"output={measure_out[:400]}"
                        ),
                    )
                )

        # 4. Run any additional verification commands (with frozen harness read-only).
        if not short_circuit:
            extra_cmds: list[str] = []
            if commands is not None:
                extra_cmds.extend(str(item) for item in commands)
            else:
                state_cmds = state.get("verification_commands")
                if isinstance(state_cmds, list):
                    extra_cmds.extend(str(item) for item in state_cmds)

            ro_files = _frozen_ro_files(handle.workspace_path, cfg.frozen_harness_paths)
            for idx, raw_cmd in enumerate(extra_cmds, start=1):
                code, raw_out, cmd_issue = self._run_safe_command(
                    raw_cmd,
                    sandbox_path=handle.workspace_path,
                    timeout_s=timeout_s,
                    node_name=node_name,
                    read_only_files=ro_files,
                )
                output = _trim_output(raw_out)
                status: Literal["pass", "fail", "inconclusive"]
                if cmd_issue is not None or code is None:
                    status = "inconclusive"
                    evidence = (
                        f"Rejected verification command ({cmd_issue}); inconclusive never passes."
                        if cmd_issue is not None
                        else output or "Command failed to complete; inconclusive never passes."
                    )
                elif code == 0:
                    status = "pass"
                    evidence = output
                else:
                    status = "fail"
                    evidence = output
                checks.append(
                    SurfaceCheckResult(
                        surface=f"check_{idx}",
                        command=raw_cmd,
                        status=status,
                        exit_code=code,
                        evidence=evidence,
                    )
                )

            # 5. Reject if any check mutated tracked, untracked, or gitignored sandbox files.
            post_head = _git_head_sha(handle.workspace_path)
            dirty_after = _git_has_changes(handle.workspace_path, include_ignored=True)
            post_all_tamper = detect_frozen_harness_tamper(handle.workspace_path, frozen_hashes)
            clean_ok = (
                post_head == expected_commit_sha
                and not dirty_after
                and not post_all_tamper
            )
            checks.append(
                SurfaceCheckResult(
                    surface="sandbox_cleanliness",
                    command="git rev-parse HEAD && git status --porcelain --ignored",
                    status="pass" if clean_ok else "fail",
                    exit_code=0 if clean_ok else 1,
                    evidence=(
                        f"Sandbox remained clean at winning commit {expected_commit_sha[:8]} "
                        "after all verification checks."
                        if clean_ok
                        else (
                            f"Verification mutated sandbox: HEAD={post_head[:8]} "
                            f"(expected {expected_commit_sha[:8]}), dirty={dirty_after}, "
                            f"tampered={post_all_tamper}."
                        )
                    ),
                )
            )

        passed = bool(checks) and all(item.status == "pass" for item in checks)
        failing_summary = "; ".join(
            f"{c.surface}={c.status} (exit={c.exit_code})"
            for c in checks
            if c.status != "pass"
        )
        verification = SurfaceVerificationReport(
            passed=passed,
            checks=checks,
            summary=(
                f"All {len(checks)} hillclimb surface verification check(s) passed."
                if passed
                else f"Hillclimb surface verification did not pass: {failing_summary}"
            ),
        )
        dumped = verification.model_dump(mode="json")
        self._save_persisted(run_id, {"surface_verification": dumped})
        self.store.append_event(
            run_id=run_id,
            stage="hillclimb",
            event_type="craft.verify_recorded",
            actor="argus",
            payload={
                "verification_id": verification.verification_id,
                "passed": verification.passed,
                "check_count": len(verification.checks),
                "winning_commit_sha": expected_commit_sha,
                "config_hash": cfg.content_hash,
                "summary": verification.summary,
            },
        )
        if not verification.passed:
            raise HillclimbPause(
                node_name,
                verification.summary,
                route="blocked",
                pause_kind="blocked",
                state_updates={"surface_verification": dumped},
            )

        applied_paths: list[str] = []
        if apply_to_workspace:
            applied_paths = self.promote_winner(
                run_id=run_id,
                state=state,
                config=cfg,
                node_name=node_name,
            )

        return {
            "surface_verification": dumped,
            "applied_paths": applied_paths,
            "route": "pass",
        }


def hillclimb_loop(
    *,
    store: RunStore,
    workspace_root: Path,
    provider: LLMProvider,
    run_id: str,
    raw_input: str,
    config: HillclimbConfig | Mapping[str, Any],
    state: Mapping[str, Any] | None = None,
    artifacts_root: Path | None = None,
    sandbox_root: Path | None = None,
    execution_drivers: ExecutionDriversConfig | None = None,
    driver_registry: DriverRegistry | None = None,
    model_provider: Callable[[str], LLMProvider] | None = None,
    on_driver_handle: Callable[[ActiveTurnHandle], None] | None = None,
    command_runner: Callable[..., tuple[int | None, str]] | None = None,
    allow_unsandboxed_metric: bool | None = None,
    node_name: str = "hillclimb",
    role: str = "builder",
    skill_refs: Sequence[str] = ("pstack/hillclimb",),
    max_steps: int = 16,
    metric_timeout_s: int = 60,
    apply_to_workspace: bool = False,
) -> dict[str, Any]:
    """Composite deterministic primitive entry point for frozen-harness hill-climbing."""
    stage = HillclimbStage(
        store=store,
        workspace_root=workspace_root,
        provider=provider,
        artifacts_root=artifacts_root,
        sandbox_root=sandbox_root,
        execution_drivers=execution_drivers,
        driver_registry=driver_registry,
        model_provider=model_provider,
        on_driver_handle=on_driver_handle,
        command_runner=command_runner,
        allow_unsandboxed_metric=allow_unsandboxed_metric,
    )
    return stage.hillclimb_loop(
        run_id=run_id,
        raw_input=raw_input,
        state=state,
        config=config,
        node_name=node_name,
        role=role,
        skill_refs=skill_refs,
        max_steps=max_steps,
        metric_timeout_s=metric_timeout_s,
        apply_to_workspace=apply_to_workspace,
    )


__all__ = [
    "HillclimbPause",
    "HillclimbStage",
    "detect_frozen_harness_tamper",
    "hillclimb_loop",
    "is_strictly_better",
    "is_target_reached",
    "parse_metric_value",
    "snapshot_frozen_harness",
]
