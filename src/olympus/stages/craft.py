"""Cohesive evidence-first craft stage implementing six orthogonal pstack primitives.

Primitives:
1. ``explore_reproduce`` (Prometheus, read-only): repo/runtime evidence + optional failing repro.
2. ``architect_sketch`` (Athena, read-only): >=2 candidate shapes + write_scope + verify plan.
3. ``build_direct`` (Hephaestus, worktree): isolated build/repair + DiffGate, no TaskPlan needed.
4. ``hygiene_pass`` (Hephaestus, worktree): diff-scoped deslop/comment cleanup + DiffGate check.
5. ``interrogate_review`` (Momus, cold-start): zero-history adversarial audit + run-proven filter.
6. ``verify_surface`` (Argus): real-surface execution where ``inconclusive`` never passes.
"""

from __future__ import annotations

import functools
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from olympus.core.contracts import (
    ArchitectSketch,
    EvidenceBrief,
    InterrogateFinding,
    InterrogateReport,
    SurfaceCheckResult,
    SurfaceVerificationReport,
)
from olympus.core.store import RunStore
from olympus.harness.diff_gate import DiffGateResult, run_diff_gate
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
from olympus.harness.loop import AgentLoop
from olympus.harness.native import NativeDriver
from olympus.harness.structured import run_structured_turn
from olympus.harness.worktree import (
    SandboxBaseline,
    SandboxHandle,
    capture_sandbox_diff,
    clean_ephemeral_artifacts,
    collect_sandbox_changes,
    create_sandbox,
    load_sandbox_baseline,
    remove_sandbox,
    save_sandbox_baseline,
    snapshot_sandbox_state,
)
from olympus.providers.base import LLMProvider
from olympus.stages.common import default_artifacts_root, format_workspace_inventory
from olympus.tools.aci import WorkspacePolicy, WorkspaceToolkit

SKILLS_ROOT = Path(__file__).resolve().parent.parent / "skills"
_CMD_TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]")
_TRIVIAL_NOOP_CMD_RE = re.compile(
    r"^\s*(?::|true|false|exit(?:\s+\d+)?|return(?:\s+\d+)?)\s*(?:;+\s*)*$",
    re.IGNORECASE,
)
_SHELL_META_RE = re.compile(r"(?:[;&|<>`]|\$\()")
_PYTHON_BIN_RE = re.compile(r"^python(?:\d+(?:\.\d+)?)?$")
_ALLOWED_PYTHON_MODULES: frozenset[str] = frozenset({"pytest", "unittest", "mypy", "ruff"})
_ALLOWED_DIRECT_RUNNERS: frozenset[str] = frozenset(
    {
        "pytest",
        "py.test",
        "mypy",
        "flake8",
        "pylint",
        "pyright",
        "vitest",
        "jest",
        "eslint",
        "tsc",
        "ctest",
    }
)
_ALLOWED_SUBCOMMAND_RUNNERS: dict[str, frozenset[str]] = {
    "ruff": frozenset({"check"}),
    "go": frozenset({"test", "vet"}),
    "cargo": frozenset({"test", "check", "clippy"}),
    "npm": frozenset({"test"}),
    "pnpm": frozenset({"test", "typecheck"}),
    "yarn": frozenset({"test"}),
}
_FORBIDDEN_REPRO_FLAGS: frozenset[str] = frozenset(
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
_TRUSTED_SYSTEM_BIN_DIRS: frozenset[Path] = frozenset(
    {Path("/usr/bin"), Path("/bin"), Path("/usr/local/bin")}
)
_UNTRUSTED_PATH_PREFIXES: tuple[Path, ...] = (Path("/tmp"), Path("/var/tmp"))
_FORBIDDEN_SHORT_FLAGS: frozenset[str] = frozenset({"c", "e", "o", "p", "w"})
_ABS_PATH_LITERAL_RE = re.compile(r"(?:^|[\s'\"=(,])/(?:etc|home|root|tmp|var|usr|opt|Users)/")
_NON_PROOF_EXIT_CODES: frozenset[int] = frozenset({126, 127})
_PYTEST_NON_FAILURE_EXIT_CODES: frozenset[int] = frozenset({2, 3, 4, 5})


class _SandboxIsolationUnavailableError(RuntimeError):
    """Raised when OS-level read-only FS isolation (bwrap) is required but unavailable."""
_MAX_EXCERPT_CHARS = 1500
_MAX_DIFF_PROMPT_CHARS = 24000


class CraftPause(Exception):
    """Signal raised by a craft primitive that should pause the workflow graph."""

    def __init__(
        self,
        node_name: str,
        reason: str,
        *,
        route: str = "blocked",
        pause_kind: str = "blocked",
        state_updates: Mapping[str, Any] | None = None,
    ) -> None:
        self.node_name = node_name
        self.reason = reason
        self.state_updates: dict[str, Any] = {
            "route": route,
            "pause_kind": pause_kind,
            **dict(state_updates or {}),
        }
        super().__init__(reason)


def load_skill_pack(skill_refs: Sequence[str], *, skills_root: Path = SKILLS_ROOT) -> str:
    """Load distilled markdown skill packs referenced by a workflow node."""
    if not skill_refs:
        return ""
    resolved_root = skills_root.resolve()
    sections: list[str] = []
    for raw_ref in skill_refs:
        cleaned = raw_ref.strip().strip("/")
        if not cleaned or ".." in Path(cleaned).parts:
            raise ValueError(f"Invalid skill_ref {raw_ref!r}")
        rel = cleaned if cleaned.endswith(".md") else f"{cleaned}.md"
        candidate = (resolved_root / rel).resolve()
        if not candidate.is_relative_to(resolved_root) or not candidate.is_file():
            raise FileNotFoundError(f"Skill pack not found for skill_ref {raw_ref!r}: {candidate}")
        sections.append(candidate.read_text(encoding="utf-8").strip())
    return "\n\n".join(part for part in sections if part)


def _has_executable_command(command: str | None) -> bool:
    if not isinstance(command, str):
        return False
    stripped = command.strip()
    if not stripped:
        return False
    return bool(_CMD_TOKEN_RE.search(stripped))


def _validate_substantive_command(command: str | None) -> str | None:
    """Return a rejection reason if ``command`` is empty or a vacuous shell no-op."""
    if not _has_executable_command(command):
        return "missing executable command"
    assert command is not None
    stripped = command.strip()
    if _TRIVIAL_NOOP_CMD_RE.match(stripped):
        return f"vacuous shell no-op command ({stripped!r})"
    return None


def _trusted_bin_dirs(
    *, host_root: Path | None = None, sandbox_root: Path | None = None
) -> list[Path]:
    """Return ordered trusted executable directories (venv bin, ~/.local/bin, system bins)."""
    blocked: list[Path] = list(_UNTRUSTED_PATH_PREFIXES)
    if host_root is not None:
        blocked.append(host_root.resolve())
    if sandbox_root is not None:
        blocked.append(sandbox_root.resolve())
    sys_exe = Path(sys.executable)
    candidates = [
        sys_exe.parent,
        sys_exe.resolve().parent,
        Path.home() / ".local" / "bin",
        Path("/usr/local/bin"),
        Path("/usr/bin"),
        Path("/bin"),
    ]
    seen: set[Path] = set()
    trusted: list[Path] = []
    for raw_dir in candidates:
        if not raw_dir.is_absolute():
            continue
        resolved_dir = raw_dir.resolve(strict=False)
        if any(
            raw_dir == b
            or raw_dir.is_relative_to(b)
            or resolved_dir == b
            or resolved_dir.is_relative_to(b)
            for b in blocked
        ):
            continue
        if raw_dir not in seen:
            seen.add(raw_dir)
            trusted.append(raw_dir)
    return trusted


def _sanitized_path(*, host_root: Path | None = None, sandbox_root: Path | None = None) -> str:
    """Return a PATH restricted exclusively to trusted executable directories."""
    return os.pathsep.join(
        str(d) for d in _trusted_bin_dirs(host_root=host_root, sandbox_root=sandbox_root)
    )


def _resolve_trusted_executable(
    raw_cmd0: str,
    *,
    host_root: Path,
    sandbox_root: Path,
) -> tuple[str | None, str | None]:
    """Ensure argv[0] resolves strictly inside trusted bin dirs, never workspace/tmp/PATH."""
    host_resolved = host_root.resolve()
    sandbox_resolved = sandbox_root.resolve()
    trusted_dirs = _trusted_bin_dirs(host_root=host_root, sandbox_root=sandbox_root)
    trusted_set = {d for d in trusted_dirs} | {d.resolve(strict=False) for d in trusted_dirs}
    if "/" in raw_cmd0 or "\\" in raw_cmd0:
        raw_path = Path(raw_cmd0)
        if not raw_path.is_absolute() or ".." in raw_path.parts:
            return None, f"relative or traversal executable path is forbidden: {raw_cmd0!r}"
        resolved = raw_path.resolve(strict=False)
        for bad in (host_resolved, sandbox_resolved, *_UNTRUSTED_PATH_PREFIXES):
            if (
                raw_path == bad
                or raw_path.is_relative_to(bad)
                or resolved == bad
                or resolved.is_relative_to(bad)
            ):
                return None, f"executable inside untrusted directory is forbidden: {raw_cmd0!r}"
        sys_exe = Path(sys.executable)
        sys_exe_resolved = sys_exe.resolve()
        raw_trusted = raw_path == sys_exe or raw_path.parent in trusted_set
        resolved_trusted = resolved == sys_exe_resolved or resolved.parent in trusted_set
        if not (raw_trusted and resolved_trusted):
            return None, f"executable path is outside trusted bin directories: {raw_cmd0!r}"
        if not raw_path.is_file() or not os.access(raw_path, os.X_OK):
            return None, f"trusted executable path not found or not executable: {raw_cmd0!r}"
        return str(raw_path), None

    safe_path = os.pathsep.join(str(d) for d in trusted_dirs)
    found = shutil.which(raw_cmd0, path=safe_path)
    if found is None:
        return raw_cmd0, None
    found_raw = Path(found)
    found_resolved = found_raw.resolve(strict=False)
    for bad in (host_resolved, sandbox_resolved, *_UNTRUSTED_PATH_PREFIXES):
        if (
            found_raw == bad
            or found_raw.is_relative_to(bad)
            or found_resolved == bad
            or found_resolved.is_relative_to(bad)
        ):
            return None, f"resolved binary inside untrusted directory is forbidden: {found!r}"
    if found_raw.parent not in trusted_set and found_resolved.parent not in trusted_set:
        return None, f"resolved binary is outside trusted bin directories: {found!r}"
    return found, None


def _effective_repro_runner(argv: Sequence[str]) -> str:
    """Return the canonical test/check runner name after unwrapping uv run / python -m."""
    if not argv:
        return ""
    binary = Path(argv[0]).name.lower()
    if binary == "uv" and len(argv) >= 3 and argv[1] == "run":
        idx = 2
        while idx < len(argv) and argv[idx] in {"--no-sync", "--frozen", "--offline", "--quiet"}:
            idx += 1
        if idx < len(argv):
            return _effective_repro_runner(argv[idx:])
    if _PYTHON_BIN_RE.match(binary) and len(argv) >= 3 and argv[1] == "-m":
        return argv[2].lower()
    return binary


def _is_proven_failure_exit(argv: Sequence[str], code: int | None) -> tuple[bool, str]:
    """Return (proven, detail) distinguishing real test failures from tool/signal errors."""
    if code is None:
        return False, "repro timed out or failed to execute"
    if code == 0:
        return False, "repro exited 0"
    if code < 0 or code >= 128:
        return False, f"repro terminated by signal, exit={code}"
    if code in _NON_PROOF_EXIT_CODES:
        return False, f"command not found/not executable, exit={code}"
    runner = _effective_repro_runner(argv)
    if runner in {"pytest", "py.test"} and code != 1:
        if code == 5:
            return False, "pytest collected no tests, exit=5"
        return False, f"pytest did not complete test execution, exit={code}"
    if runner == "unittest" and code == 2:
        return False, "unittest CLI/usage error, exit=2"
    return True, f"repro failed with exit={code}"


@functools.lru_cache(maxsize=1)
def _probe_bwrap() -> str | None:
    """Return the path to ``bwrap`` when Linux user-namespace read-only mounting works."""
    bwrap_bin = shutil.which("bwrap", path="/usr/bin:/bin:/usr/local/bin")
    if bwrap_bin is None:
        return None
    try:
        probe = subprocess.run(
            [bwrap_bin, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "/bin/true"],
            shell=False,
            capture_output=True,
            timeout=5,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    return bwrap_bin if probe.returncode == 0 else None


def _validate_allowed_repro_argv(
    argv: list[str],
    *,
    host_root: Path,
    sandbox_root: Path,
) -> tuple[list[str] | None, str | None]:
    """Enforce a strict allowlist of read-only runners and trusted executable paths."""
    if not argv:
        return None, "empty repro_command"
    trusted0, exe_issue = _resolve_trusted_executable(
        argv[0], host_root=host_root, sandbox_root=sandbox_root
    )
    if exe_issue is not None or trusted0 is None:
        return None, exe_issue or "untrusted executable"
    resolved_argv = [trusted0, *argv[1:]]
    binary = Path(argv[0]).name.lower()
    if binary == "uv":
        idx = 1
        if idx >= len(resolved_argv) or resolved_argv[idx] != "run":
            return None, "uv in repro_command only allows 'uv run <read-only-runner>'"
        idx += 1
        while idx < len(resolved_argv) and resolved_argv[idx] in {
            "--no-sync",
            "--frozen",
            "--offline",
            "--quiet",
        }:
            idx += 1
        if idx >= len(resolved_argv):
            return None, "uv run requires a read-only test runner command"
        nested, nested_issue = _validate_allowed_repro_argv(
            resolved_argv[idx:], host_root=host_root, sandbox_root=sandbox_root
        )
        if nested_issue is not None or nested is None:
            return None, nested_issue
        return [*resolved_argv[:idx], *nested], None
    if _PYTHON_BIN_RE.match(binary):
        if len(resolved_argv) < 3 or resolved_argv[1] != "-m":
            return None, f"{binary!r} in repro_command only allows '-m <allowed_test_module>'"
        mod = resolved_argv[2].lower()
        if mod not in _ALLOWED_PYTHON_MODULES:
            return None, f"python module {mod!r} is not in the read-only repro allowlist"
        if mod == "ruff" and (len(resolved_argv) < 4 or resolved_argv[3] != "check"):
            return None, "python -m ruff in repro_command only allows 'check'"
        return resolved_argv, None
    if binary in _ALLOWED_DIRECT_RUNNERS:
        return resolved_argv, None
    if binary in _ALLOWED_SUBCOMMAND_RUNNERS:
        allowed_subs = _ALLOWED_SUBCOMMAND_RUNNERS[binary]
        idx = 1
        if binary == "pnpm" and idx + 1 < len(resolved_argv) and resolved_argv[idx] == "--dir":
            idx += 2
        if idx >= len(resolved_argv) or resolved_argv[idx] not in allowed_subs:
            return (
                None,
                f"{binary!r} in repro_command only allows subcommands {sorted(allowed_subs)}",
            )
        return resolved_argv, None
    return None, f"binary {binary!r} is not in the read-only repro allowlist"


def _has_forbidden_flag(arg: str) -> str | None:
    """Detect forbidden long, attached-value, or bundled short flags."""
    if not arg.startswith("-") or arg == "-":
        return None
    flag_name = arg.split("=", 1)[0].lower()
    if flag_name in _FORBIDDEN_REPRO_FLAGS:
        return f"flag {flag_name!r} is forbidden in read-only repro_command"
    if arg.startswith("--"):
        return None
    short_body = arg[1:].lower()
    if not short_body:
        return None
    if short_body[0] in _FORBIDDEN_SHORT_FLAGS:
        return f"short flag -{short_body[0]} (in {arg!r}) is forbidden in read-only repro_command"
    if short_body[0] not in {"k", "m"}:
        for ch in short_body:
            if ch in _FORBIDDEN_SHORT_FLAGS:
                return (
                    f"bundled short flag -{ch} (in {arg!r}) is forbidden in read-only repro_command"
                )
    return None


def _parse_sandboxed_repro_argv(
    command: str,
    *,
    host_root: Path,
    sandbox_root: Path,
) -> tuple[list[str] | None, str | None]:
    """Validate and tokenize a read-only reproduction command for ``shell=False`` execution."""
    issue = _validate_substantive_command(command)
    if issue is not None:
        return None, issue
    if _SHELL_META_RE.search(command):
        return None, "shell metacharacters and redirections are forbidden in repro_command"
    try:
        raw_argv = shlex.split(command)
    except ValueError as exc:
        return None, f"malformed repro_command: {exc}"
    argv, allow_issue = _validate_allowed_repro_argv(
        raw_argv, host_root=host_root, sandbox_root=sandbox_root
    )
    if allow_issue is not None or argv is None:
        return None, allow_issue
    host_str = str(host_root.resolve())
    sandbox_resolved = sandbox_root.resolve()
    for arg in raw_argv[1:]:
        flag_issue = _has_forbidden_flag(arg)
        if flag_issue is not None:
            return None, flag_issue
        if host_str and host_str in arg:
            return None, f"repro_command argument references host workspace: {arg!r}"
        if arg.startswith("~") or ".." in Path(arg).parts:
            return None, f"repro_command argument attempts path traversal: {arg!r}"
        if arg.startswith("/"):
            candidate = Path(arg).resolve(strict=False)
            if candidate != sandbox_resolved and not candidate.is_relative_to(sandbox_resolved):
                return None, f"repro_command argument references external path: {arg!r}"
        if _ABS_PATH_LITERAL_RE.search(arg):
            return None, f"repro_command contains external absolute path literal: {arg!r}"
    return argv, None


def _run_argv_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_s: int = 60,
    host_root: Path | None = None,
    require_fs_isolation: bool = True,
) -> tuple[int | None, str]:
    """Execute one tokenized command directly (``shell=False``) with read-only host FS via bwrap."""
    bwrap_bin = _probe_bwrap()
    if bwrap_bin is None and require_fs_isolation:
        raise _SandboxIsolationUnavailableError(
            "Read-only OS filesystem sandbox (bwrap) is unavailable; refusing unsandboxed "
            "repro_command execution (set allow_unsandboxed_repro=True or "
            "OLYMPUS_ALLOW_UNSANDBOXED_REPRO=1 to opt in explicitly)"
        )
    cwd_resolved = cwd.resolve()
    repro_tmp = (cwd_resolved.parent / f".{cwd_resolved.name}-repro-tmp").resolve()
    repro_home = repro_tmp / "home"
    repro_home.mkdir(parents=True, exist_ok=True)
    safe_path = _sanitized_path(host_root=host_root, sandbox_root=cwd_resolved)
    if "/" not in argv[0] and "\\" not in argv[0] and shutil.which(argv[0], path=safe_path) is None:
        shutil.rmtree(repro_tmp, ignore_errors=True)
        return 127, f"Command not found: {argv[0]}"

    env = dict(os.environ)
    env["PATH"] = safe_path
    env["HOME"] = str(repro_home)
    env.setdefault("CI", "1")
    env["TMPDIR"] = str(repro_tmp)
    env["TEMP"] = str(repro_tmp)
    env["TMP"] = str(repro_tmp)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["RUFF_NO_CACHE"] = "1"
    env["RUFF_CACHE_DIR"] = str(repro_tmp / "ruff_cache")
    env["MYPY_CACHE_DIR"] = str(repro_tmp / "mypy_cache")
    env["PYTEST_ADDOPTS"] = f"-o cache_dir={repro_tmp / 'pytest_cache'} -p no:cacheprovider"

    cmd: list[str] = list(argv)
    if bwrap_bin is not None:
        cmd = [
            bwrap_bin,
            "--ro-bind",
            "/",
            "/",
            "--bind",
            str(cwd_resolved),
            str(cwd_resolved),
            "--bind",
            str(repro_tmp),
            str(repro_tmp),
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--chdir",
            str(cwd_resolved),
            "--",
            *argv,
        ]
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
        return None, _trim_output(f"Command timed out after {timeout_s}s.\n{combined}")
    except FileNotFoundError:
        return 127, f"Command not found: {argv[0]}"
    except PermissionError:
        return 126, f"Command not executable: {argv[0]}"
    except OSError as exc:
        return None, _trim_output(f"Command execution error: {exc}")
    finally:
        shutil.rmtree(repro_tmp, ignore_errors=True)
    combined = "\n".join(part for part in (proc.stdout, proc.stderr) if part and part.strip())
    return proc.returncode, _trim_output(combined)


def _trim_output(text: str, limit: int = _MAX_EXCERPT_CHARS) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "\n...[truncated]"


def _run_sandbox_command(
    command: str,
    *,
    cwd: Path,
    timeout_s: int = 60,
) -> tuple[int | None, str]:
    """Execute one bounded shell command without blocking on stdin."""
    env = dict(os.environ)
    env.setdefault("CI", "1")
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
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
        return None, _trim_output(f"Command timed out after {timeout_s}s.\n{combined}")
    except OSError as exc:
        return None, _trim_output(f"Command execution error: {exc}")
    combined = "\n".join(part for part in (proc.stdout, proc.stderr) if part and part.strip())
    return proc.returncode, _trim_output(combined)


def _compose_system_prompt(base_prompt: str, skill_refs: Sequence[str]) -> str:
    pack = load_skill_pack(skill_refs)
    if not pack:
        return base_prompt
    return f"{base_prompt}\n\n{pack}"


class CraftStage:
    """Cohesive handler for all six orthogonal evidence-first craft primitives."""

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
        command_runner: Callable[..., tuple[int | None, str]] = _run_sandbox_command,
        allow_unsandboxed_repro: bool | None = None,
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
        self._run_command = command_runner
        self._allow_unsandboxed_repro = (
            allow_unsandboxed_repro
            if allow_unsandboxed_repro is not None
            else os.environ.get("OLYMPUS_ALLOW_UNSANDBOXED_REPRO") == "1"
        )

    # ------------------------------------------------------------------ state persistence

    def _craft_dir(self, run_id: str) -> Path:
        path = self.artifacts_root / run_id / "craft"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _state_file(self, run_id: str) -> Path:
        return self._craft_dir(run_id) / "state.json"

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
            stage="craft",
            event_type="craft.role_dispatched",
            actor=f"craft:{role}",
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

    def _run_structured[T: BaseModel](
        self,
        *,
        run_id: str,
        node_name: str,
        role: str,
        actor: str,
        workspace_path: Path,
        system_prompt: str,
        user_prompt: str,
        model_cls: type[T],
        max_steps: int,
        trace_name: str,
    ) -> tuple[T, str]:
        driver, driver_model, resolved_spec = self._resolve_role_driver(
            run_id=run_id,
            node_name=node_name,
            role=role,
            cold_start=True,
        )
        active_provider = self._provider_for_model(driver_model)
        toolkit = WorkspaceToolkit(workspace_path, read_only=True, env_root=self.workspace_root)
        trace_path = self._craft_dir(run_id) / f"{trace_name}.jsonl"
        loop = AgentLoop(
            provider=active_provider,
            toolkit=toolkit,
            store=self.store,
            run_id=run_id,
            actor=actor,
            max_steps=max_steps,
            trace_path=trace_path,
            synthesize_on_max_steps=True,
        )
        result = run_structured_turn(
            loop=loop,
            driver=driver,
            driver_model=driver_model,
            store=self.store,
            run_id=run_id,
            workspace_root=workspace_path,
            actor=actor,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_cls=model_cls,
            trace_path=trace_path,
            max_steps=max_steps,
        )
        return result, resolved_spec

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
        strict_scope: bool,
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
        trace_path = self._craft_dir(run_id) / f"{trace_name}.jsonl"
        request = DriverTurnRequest(
            workspace_path=workspace_path,
            env_root=self.workspace_root,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            actor=actor,
            read_only=False,
            strict_scope=strict_scope,
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

    def _promote_sandbox_to_workspace(
        self,
        handle: SandboxHandle,
        baseline: SandboxBaseline,
        sketch: ArchitectSketch,
        *,
        node_name: str = "verify_surface",
    ) -> list[str]:
        strict_gate = run_diff_gate(
            handle,
            baseline,
            allowed_write_scope=sketch.write_scope,
            forbidden_globs=sketch.forbidden_globs,
            strict_scope=True,
        )
        if strict_gate.rejected:
            raise CraftPause(
                node_name,
                strict_gate.rejection_message(),
                route="blocked",
                pause_kind="blocked",
            )
        changed = collect_sandbox_changes(handle, baseline)
        policy = WorkspacePolicy(
            allowed_write_scope=list(sketch.write_scope),
            forbidden_write_scope=list(sketch.forbidden_globs),
        )
        ws_root = self.workspace_root.resolve()
        # Phase 1: validate every single mutation before writing or deleting anything on host.
        copy_plan: list[tuple[Path, Path, str | None]] = []
        delete_plan: list[Path] = []
        applied: list[str] = []
        for rel in changed:
            if policy.forbidden_match(rel) is not None or not policy.allows_relative_path(rel):
                raise CraftPause(
                    node_name,
                    f"Refusing to promote unauthorized path outside write_scope: {rel}",
                    route="blocked",
                    pause_kind="blocked",
                )
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise CraftPause(
                    node_name,
                    f"Refusing to promote unsafe relative path: {rel}",
                    route="blocked",
                    pause_kind="blocked",
                )
            src = handle.workspace_path / rel_path
            dst = ws_root / rel_path
            # Check parent containment without dereferencing dst itself so existing
            # broken or external symlinks can still be unlinked cleanly.
            parent_resolved = dst.parent.resolve(strict=False)
            if parent_resolved != ws_root and not parent_resolved.is_relative_to(ws_root):
                raise CraftPause(
                    node_name,
                    f"Refusing to promote path escaping workspace root: {rel}",
                    route="blocked",
                    pause_kind="blocked",
                )
            if not src.exists() and not src.is_symlink():
                if dst.exists() and not dst.is_symlink() and dst.is_dir():
                    raise CraftPause(
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
                    raise CraftPause(
                        node_name,
                        f"Refusing to promote symlink escaping workspace root: "
                        f"{rel} -> {link_target}",
                        route="blocked",
                        pause_kind="blocked",
                    )
            elif src.is_dir() or not src.is_file():
                raise CraftPause(
                    node_name,
                    f"Refusing to promote non-regular path/directory: {rel}",
                    route="blocked",
                    pause_kind="blocked",
                )
            if dst.exists() and not dst.is_symlink() and dst.is_dir():
                raise CraftPause(
                    node_name,
                    f"Refusing to overwrite host directory with file: {rel}",
                    route="blocked",
                    pause_kind="blocked",
                )
            copy_plan.append((src, dst, link_target))
            applied.append(rel)

        delete_set = set(delete_plan)
        for _src, dst, _link_target in copy_plan:
            ancestor = dst.parent
            while ancestor != ws_root and ancestor.is_relative_to(ws_root):
                if (
                    ancestor.is_symlink() or (ancestor.exists() and not ancestor.is_dir())
                ) and ancestor not in delete_set:
                    raise CraftPause(
                        node_name,
                        f"Refusing to promote path whose ancestor is a non-directory: "
                        f"{dst.relative_to(ws_root)}",
                        route="blocked",
                        pause_kind="blocked",
                    )
                ancestor = ancestor.parent

        # Phase 2: transactional apply with best-effort backup+mode rollback on any OSError.
        backups: list[tuple[Path, str, tuple[bytes, int] | str | None]] = []
        created_dirs: list[Path] = []
        try:
            for target in delete_plan:
                if target.is_symlink():
                    backups.append((target, "symlink", os.readlink(target)))
                    target.unlink()
                elif target.is_file():
                    mode = stat.S_IMODE(target.stat().st_mode)
                    backups.append((target, "file", (target.read_bytes(), mode)))
                    target.unlink()
                parent = target.parent
                while parent != ws_root and parent.is_relative_to(ws_root):
                    try:
                        parent.rmdir()
                    except OSError:
                        break
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
                    mode = stat.S_IMODE(dst.stat().st_mode)
                    backups.append((dst, "file", (dst.read_bytes(), mode)))
                else:
                    backups.append((dst, "missing", None))
                if link_target is not None:
                    if dst.is_file() or dst.is_symlink():
                        dst.unlink()
                    os.symlink(link_target, dst)
                else:
                    shutil.copy2(src, dst)
        except OSError as exc:
            rollback_errors: list[str] = []
            for path, kind, saved in reversed(backups):
                try:
                    if kind == "missing":
                        if path.is_symlink() or path.is_file():
                            path.unlink(missing_ok=True)
                    elif kind == "symlink" and isinstance(saved, str):
                        if path.is_symlink() and os.readlink(path) == saved:
                            continue
                        if path.is_symlink() or path.is_file():
                            path.unlink(missing_ok=True)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        os.symlink(saved, path)
                    elif kind == "file" and isinstance(saved, tuple):
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
                        path.write_bytes(data)
                        os.chmod(path, saved_mode)
                except OSError as rb_exc:
                    rollback_errors.append(f"{path.name}: {rb_exc}")
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
            raise CraftPause(
                node_name,
                f"Atomic workspace promotion failed and was rolled back: {exc}{extra}",
                route="blocked",
                pause_kind="blocked",
            ) from exc
        return sorted(set(applied))

    def _recorded_clarifications(self, run_id: str) -> list[str]:
        notes: list[str] = []
        for event in self.store.list_events(run_id):
            if event.event_type != "craft.clarification_recorded":
                continue
            note = event.payload.get("note")
            if isinstance(note, str) and note.strip():
                notes.append(note.strip())
        return notes

    def _active_review_cycle(self, run_id: str, state: Mapping[str, Any]) -> int:
        if "craft_review_cycle" in state and state["craft_review_cycle"] is not None:
            return int(state["craft_review_cycle"])
        cycle = 0
        for event in self.store.list_events(run_id):
            if event.event_type == "craft.recovery_reset":
                cycle = 0
            elif event.event_type == "craft.interrogate_recorded":
                raw = event.payload.get("cycle")
                if isinstance(raw, int):
                    cycle = raw
        return cycle

    # ------------------------------------------------------------------ 6 orthogonal primitives

    def explore_reproduce(
        self,
        *,
        run_id: str,
        raw_input: str,
        constraints: Sequence[str] = (),
        scope_paths: Sequence[str] = (),
        node_name: str = "explore",
        role: str = "explorer",
        skill_refs: Sequence[str] = (),
        require_repro: bool = False,
        max_steps: int = 16,
    ) -> dict[str, Any]:
        toolkit = WorkspaceToolkit(
            self.workspace_root, read_only=True, env_root=self.workspace_root
        )
        inventory = format_workspace_inventory(toolkit, scope_paths=scope_paths)
        system_prompt = _compose_system_prompt(
            "You are Prometheus (Evidence-First Explorer). Inspect the repository read-only and "
            "return ONLY valid EvidenceBrief JSON conforming strictly to the schema. "
            "Leave repro_verified=false (the deterministic control plane verifies repro commands).",
            skill_refs,
        )
        merged_constraints = [
            *constraints,
            *(
                note
                for note in self._recorded_clarifications(run_id)
                if note not in constraints
            ),
        ]
        constraints_block = (
            "\nConstraints / prior clarifications:\n- " + "\n- ".join(merged_constraints)
            if merged_constraints
            else ""
        )
        repro_note = (
            "\nYou MUST supply a concrete failing `repro_command` that reproduces the defect."
            if require_repro
            else ""
        )
        user_prompt = (
            f"Goal:\n{raw_input}\n\n{inventory}{constraints_block}{repro_note}\n\n"
            f"Return JSON matching schema:\n"
            f"{json.dumps(EvidenceBrief.model_json_schema(), ensure_ascii=False)}"
        )
        brief, _ = self._run_structured(
            run_id=run_id,
            node_name=node_name,
            role=role,
            actor="prometheus:explore",
            workspace_path=self.workspace_root,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_cls=EvidenceBrief,
            max_steps=max_steps,
            trace_name="explore",
        )
        # Never trust LLM-claimed repro verification; run repro inside a throwaway sandbox.
        repro_verified = False
        repro_exit_code = brief.repro_exit_code
        repro_excerpt = brief.repro_excerpt
        if require_repro:
            cmd_issue = _validate_substantive_command(brief.repro_command)
            if cmd_issue is not None:
                brief = brief.model_copy(
                    update={"repro_verified": False, "repro_exit_code": None}
                )
                dumped = brief.model_dump(mode="json")
                self._save_persisted(run_id, {"evidence_brief": dumped})
                raise CraftPause(
                    node_name,
                    f"Reproduction required before fixing ({cmd_issue})",
                    route="blocked",
                    pause_kind="blocked",
                    state_updates={"evidence_brief": dumped},
                )
            assert brief.repro_command is not None
            repro_handle = create_sandbox(
                self.workspace_root,
                label=f"{run_id}-repro",
                reuse=False,
                sandbox_root=self.sandbox_root,
            )
            try:
                argv, argv_issue = _parse_sandboxed_repro_argv(
                    brief.repro_command,
                    host_root=self.workspace_root,
                    sandbox_root=repro_handle.workspace_path,
                )
                if argv_issue is not None or argv is None:
                    brief = brief.model_copy(
                        update={"repro_verified": False, "repro_exit_code": None}
                    )
                    dumped = brief.model_dump(mode="json")
                    self._save_persisted(run_id, {"evidence_brief": dumped})
                    raise CraftPause(
                        node_name,
                        f"Unsafe repro_command rejected in read-only explore ({argv_issue})",
                        route="blocked",
                        pause_kind="blocked",
                        state_updates={"evidence_brief": dumped},
                    )
                if self._run_command is _run_sandbox_command:
                    try:
                        code, excerpt = _run_argv_command(
                            argv,
                            cwd=repro_handle.workspace_path,
                            timeout_s=60,
                            host_root=self.workspace_root,
                            require_fs_isolation=not self._allow_unsandboxed_repro,
                        )
                    except _SandboxIsolationUnavailableError as exc:
                        raise CraftPause(
                            node_name,
                            str(exc),
                            route="blocked",
                            pause_kind="blocked",
                        ) from exc
                else:
                    code, excerpt = self._run_command(
                        brief.repro_command, cwd=repro_handle.workspace_path, timeout_s=60
                    )
            finally:
                remove_sandbox(repro_handle)
            repro_exit_code = code
            repro_excerpt = excerpt
            repro_verified, repro_detail = _is_proven_failure_exit(argv, code)
            brief = brief.model_copy(
                update={
                    "repro_exit_code": repro_exit_code,
                    "repro_excerpt": repro_excerpt,
                    "repro_verified": repro_verified,
                }
            )
            if not repro_verified:
                dumped = brief.model_dump(mode="json")
                self._save_persisted(run_id, {"evidence_brief": dumped})
                raise CraftPause(
                    node_name,
                    f"Reproduction command did not prove failure on base workspace "
                    f"({repro_detail}): {brief.repro_command}",
                    route="blocked",
                    pause_kind="blocked",
                    state_updates={"evidence_brief": dumped},
                )
        else:
            brief = brief.model_copy(update={"repro_verified": False})

        dumped = brief.model_dump(mode="json")
        self._save_persisted(run_id, {"evidence_brief": dumped})
        self.store.append_event(
            run_id=run_id,
            stage="craft",
            event_type="craft.evidence_recorded",
            actor="prometheus",
            payload={
                "brief_id": brief.brief_id,
                "content_hash": brief.content_hash,
                "fact_count": len(brief.facts),
                "repro_verified": brief.repro_verified,
                "unresolved_questions": list(brief.unresolved_questions),
            },
        )
        if brief.unresolved_questions:
            raise CraftPause(
                node_name,
                "; ".join(brief.unresolved_questions),
                route="clarify",
                pause_kind="clarify",
                state_updates={"evidence_brief": dumped},
            )
        return {"evidence_brief": dumped, "route": "ready"}

    def architect_sketch(
        self,
        *,
        run_id: str,
        raw_input: str,
        state: Mapping[str, Any],
        constraints: Sequence[str] = (),
        scope_paths: Sequence[str] = (),
        node_name: str = "architect",
        role: str = "architect",
        skill_refs: Sequence[str] = (),
        max_steps: int = 16,
    ) -> dict[str, Any]:
        brief = self._hydrate_model(run_id, state, "evidence_brief", EvidenceBrief)
        toolkit = WorkspaceToolkit(
            self.workspace_root, read_only=True, env_root=self.workspace_root
        )
        inventory = format_workspace_inventory(toolkit, scope_paths=scope_paths)
        system_prompt = _compose_system_prompt(
            "You are Athena (Lightweight Architect). Compare at least two candidate shapes and "
            "return ONLY valid ArchitectSketch JSON with a tight write_scope and concrete "
            "executable verification_plan commands.",
            skill_refs,
        )
        brief_block = (
            json.dumps(brief.model_dump(mode="json"), ensure_ascii=False, indent=2)
            if brief is not None
            else "(none)"
        )
        constraints_block = (
            "\nConstraints:\n- " + "\n- ".join(constraints) if constraints else ""
        )
        user_prompt = (
            f"Goal:\n{raw_input}\n\n{inventory}{constraints_block}\n\n"
            f"EvidenceBrief:\n{brief_block}\n\n"
            f"Return JSON matching schema:\n"
            f"{json.dumps(ArchitectSketch.model_json_schema(), ensure_ascii=False)}"
        )
        sketch, _ = self._run_structured(
            run_id=run_id,
            node_name=node_name,
            role=role,
            actor="athena:architect",
            workspace_path=self.workspace_root,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_cls=ArchitectSketch,
            max_steps=max_steps,
            trace_name="architect",
        )
        dumped = sketch.model_dump(mode="json")
        self._save_persisted(run_id, {"architect_sketch": dumped})
        self.store.append_event(
            run_id=run_id,
            stage="craft",
            event_type="craft.architect_recorded",
            actor="athena",
            payload={
                "sketch_id": sketch.sketch_id,
                "content_hash": sketch.content_hash,
                "chosen_shape": sketch.chosen_shape,
                "write_scope": list(sketch.write_scope),
                "verification_plan": list(sketch.verification_plan),
            },
        )
        return {"architect_sketch": dumped, "route": "ready"}

    def build_direct(
        self,
        *,
        run_id: str,
        raw_input: str,
        state: Mapping[str, Any],
        node_name: str = "build",
        role: str = "builder",
        skill_refs: Sequence[str] = (),
        strict_scope: bool = True,
        max_diff_gate_retries: int = 2,
        max_steps: int = 40,
    ) -> dict[str, Any]:
        sketch = self._hydrate_model(run_id, state, "architect_sketch", ArchitectSketch)
        if sketch is None:
            raise CraftPause(
                node_name,
                "build_direct requires an ArchitectSketch before editing code",
                route="blocked",
                pause_kind="blocked",
            )
        brief = self._hydrate_model(run_id, state, "evidence_brief", EvidenceBrief)
        prior_review = self._hydrate_model(run_id, state, "interrogate_report", InterrogateReport)
        handle, baseline = self._ensure_sandbox(run_id)

        system_prompt = _compose_system_prompt(
            "You are Hephaestus (Direct Craft Builder). Implement the requested change inside "
            "the isolated worktree strictly within the authorized write_scope.",
            skill_refs,
        )
        proven_findings = (
            prior_review.proven_blocking_findings() if prior_review is not None else []
        )
        if proven_findings:
            findings_json = json.dumps(
                [item.model_dump(mode="json") for item in proven_findings],
                ensure_ascii=False,
                indent=2,
            )
            current_diff = _trim_output(
                capture_sandbox_diff(handle, baseline), _MAX_DIFF_PROMPT_CHARS
            )
            base_user_prompt = (
                f"Goal:\n{raw_input}\n\n"
                f"Authorized write_scope: {json.dumps(sketch.write_scope)}\n"
                f"Fix ONLY these run-proven blocking findings from Momus interrogation:\n"
                f"{findings_json}\n\nCurrent diff:\n```diff\n{current_diff}\n```"
            )
        else:
            brief_block = (
                json.dumps(brief.model_dump(mode="json"), ensure_ascii=False, indent=2)
                if brief is not None
                else "(none)"
            )
            base_user_prompt = (
                f"Goal:\n{raw_input}\n\n"
                f"ArchitectSketch:\n"
                f"{json.dumps(sketch.model_dump(mode='json'), ensure_ascii=False, indent=2)}\n\n"
                f"EvidenceBrief:\n{brief_block}\n\n"
                f"Implement the chosen shape cleanly inside write_scope and run verification."
            )

        user_prompt = base_user_prompt
        gate_result: DiffGateResult | None = None
        last_outcome: TurnOutcome | None = None
        for attempt in range(1, max(1, max_diff_gate_retries) + 1):
            last_outcome, _ = self._run_worker_turn(
                run_id=run_id,
                node_name=node_name,
                role=role,
                actor="hephaestus:build",
                workspace_path=handle.workspace_path,
                write_scope=sketch.write_scope,
                forbidden_globs=sketch.forbidden_globs,
                strict_scope=strict_scope,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_steps=max_steps,
                trace_name=f"build_attempt_{attempt}",
            )
            clean_ephemeral_artifacts(handle, baseline, write_scope=sketch.write_scope)
            gate_result = run_diff_gate(
                handle,
                baseline,
                allowed_write_scope=sketch.write_scope,
                forbidden_globs=sketch.forbidden_globs,
                strict_scope=strict_scope,
            )
            if not gate_result.rejected:
                break
            rejection = gate_result.rejection_message()
            self.store.append_event(
                run_id=run_id,
                stage="craft",
                event_type="craft.diff_gate_rejected",
                actor="diff_gate",
                payload={
                    "attempt": attempt,
                    "violations": [
                        {"path": v.path, "reason": v.reason} for v in gate_result.violations
                    ],
                },
            )
            user_prompt = f"{base_user_prompt}\n\n[DIFF GATE REJECTED]\n{rejection}"

        assert gate_result is not None
        if gate_result.rejected:
            raise CraftPause(
                node_name,
                gate_result.rejection_message(),
                route="blocked",
                pause_kind="blocked",
            )
        if (
            last_outcome is not None
            and not last_outcome.completed
            and not gate_result.changed_paths
        ):
            raise CraftPause(
                node_name,
                f"Build turn did not complete ({last_outcome.stop_reason}) "
                "and produced no workspace changes",
                route="blocked",
                pause_kind="blocked",
            )

        diff = capture_sandbox_diff(handle, baseline)
        (self._craft_dir(run_id) / "diff.patch").write_text(diff, encoding="utf-8")
        self._save_persisted(
            run_id,
            {
                "craft_changed_paths": list(gate_result.changed_paths),
                "sandbox_path": str(handle.workspace_path),
            },
        )
        self.store.append_event(
            run_id=run_id,
            stage="craft",
            event_type="craft.build_completed",
            actor="hephaestus",
            payload={
                "changed_paths": list(gate_result.changed_paths),
                "diff_bytes": len(diff.encode("utf-8")),
            },
        )
        return {
            "craft_diff": diff,
            "craft_changed_paths": list(gate_result.changed_paths),
            "route": "built",
        }

    def hygiene_pass(
        self,
        *,
        run_id: str,
        state: Mapping[str, Any],
        node_name: str = "hygiene",
        role: str = "hygiene",
        skill_refs: Sequence[str] = (),
        strict_scope: bool = True,
        max_steps: int = 16,
    ) -> dict[str, Any]:
        sketch = self._hydrate_model(run_id, state, "architect_sketch", ArchitectSketch)
        if sketch is None:
            return {"route": "clean"}
        handle, baseline = self._ensure_sandbox(run_id)
        diff = capture_sandbox_diff(handle, baseline)
        if not diff.strip():
            self.store.append_event(
                run_id=run_id,
                stage="craft",
                event_type="craft.hygiene_completed",
                actor="hephaestus",
                payload={"skipped": True, "changed_paths": []},
            )
            return {"craft_diff": diff, "route": "clean"}

        system_prompt = _compose_system_prompt(
            "You are Hephaestus (Diff Hygiene Pass). Clean AI slop and redundant comments in "
            "the current diff without changing runtime behavior or touching files outside scope.",
            skill_refs,
        )
        user_prompt = (
            f"Authorized write_scope: {json.dumps(sketch.write_scope)}\n"
            f"Inspect and clean only the diff below:\n```diff\n"
            f"{_trim_output(diff, _MAX_DIFF_PROMPT_CHARS)}\n```"
        )
        self._run_worker_turn(
            run_id=run_id,
            node_name=node_name,
            role=role,
            actor="hephaestus:hygiene",
            workspace_path=handle.workspace_path,
            write_scope=sketch.write_scope,
            forbidden_globs=sketch.forbidden_globs,
            strict_scope=strict_scope,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_steps=max_steps,
            trace_name="hygiene",
        )
        clean_ephemeral_artifacts(handle, baseline, write_scope=sketch.write_scope)
        gate_result = run_diff_gate(
            handle,
            baseline,
            allowed_write_scope=sketch.write_scope,
            forbidden_globs=sketch.forbidden_globs,
            strict_scope=strict_scope,
        )
        if gate_result.rejected:
            raise CraftPause(
                node_name,
                gate_result.rejection_message(),
                route="blocked",
                pause_kind="blocked",
            )
        updated_diff = capture_sandbox_diff(handle, baseline)
        (self._craft_dir(run_id) / "diff.patch").write_text(updated_diff, encoding="utf-8")
        self.store.append_event(
            run_id=run_id,
            stage="craft",
            event_type="craft.hygiene_completed",
            actor="hephaestus",
            payload={
                "skipped": False,
                "changed_paths": list(gate_result.changed_paths),
                "diff_bytes": len(updated_diff.encode("utf-8")),
            },
        )
        return {
            "craft_diff": updated_diff,
            "craft_changed_paths": list(gate_result.changed_paths),
            "route": "clean",
        }

    def interrogate_review(
        self,
        *,
        run_id: str,
        raw_input: str,
        state: Mapping[str, Any],
        node_name: str = "interrogate",
        role: str = "interrogator",
        skill_refs: Sequence[str] = (),
        enable_thermo: bool = False,
        max_cycles: int = 3,
        max_steps: int = 16,
    ) -> dict[str, Any]:
        sketch = self._hydrate_model(run_id, state, "architect_sketch", ArchitectSketch)
        handle, baseline = self._ensure_sandbox(run_id)
        diff = capture_sandbox_diff(handle, baseline)
        review_cycle = self._active_review_cycle(run_id, state) + 1

        sketch_block = (
            json.dumps(sketch.model_dump(mode="json"), ensure_ascii=False, indent=2)
            if sketch is not None
            else "(none)"
        )
        system_prompt = _compose_system_prompt(
            "You are Momus (Cold-Start Adversarial Interrogator). You have ZERO builder history. "
            "Attempt to break this diff and return ONLY valid InterrogateReport JSON. "
            "Every `important` finding MUST carry a runnable `repro_command` that fails (non-zero "
            "exit) in the sandbox; unproven claims will be filtered out deterministically.",
            skill_refs,
        )
        user_prompt = (
            f"Original Goal:\n{raw_input}\n\nArchitectSketch:\n{sketch_block}\n\n"
            f"Diff to interrogate:\n```diff\n{_trim_output(diff, _MAX_DIFF_PROMPT_CHARS)}\n```\n\n"
            f"Return JSON matching schema:\n"
            f"{json.dumps(InterrogateReport.model_json_schema(), ensure_ascii=False)}"
        )
        raw_report, resolved_spec = self._run_structured(
            run_id=run_id,
            node_name=node_name,
            role=role,
            actor="momus:interrogate",
            workspace_path=handle.workspace_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_cls=InterrogateReport,
            max_steps=max_steps,
            trace_name=f"interrogate_cycle_{review_cycle}",
        )

        all_findings: list[InterrogateFinding] = list(raw_report.findings)
        unproven_notes: list[str] = list(raw_report.unproven_notes)
        if enable_thermo:
            judo_prompt = _compose_system_prompt(
                "You are Momus (Code-Judo Quality Lane). Inspect the diff cold-start for "
                "structural complexity, thin wrappers, or missed dramatic simplifications and "
                "return ONLY valid InterrogateReport JSON.",
                skill_refs,
            )
            judo_report, _ = self._run_structured(
                run_id=run_id,
                node_name=node_name,
                role="thermo_reviewer",
                actor="momus:code_judo",
                workspace_path=handle.workspace_path,
                system_prompt=judo_prompt,
                user_prompt=user_prompt,
                model_cls=InterrogateReport,
                max_steps=max_steps,
                trace_name=f"thermo_judo_cycle_{review_cycle}",
            )
            seen_claims = {(f.file_path, f.claim.strip().lower()) for f in all_findings}
            for item in judo_report.findings:
                key = (item.file_path, item.claim.strip().lower())
                if key not in seen_claims:
                    seen_claims.add(key)
                    all_findings.append(item.model_copy(update={"lane": "code_judo"}))
            unproven_notes.extend(judo_report.unproven_notes)

        # Deterministic run-proven verification gate: important findings must fail a real command.
        verified_findings: list[InterrogateFinding] = []
        filtered_unproven = 0
        for finding in all_findings:
            if finding.severity != "important":
                verified_findings.append(finding)
                continue
            cmd_issue = _validate_substantive_command(finding.repro_command)
            if cmd_issue is not None:
                filtered_unproven += 1
                unproven_notes.append(
                    f"[filtered unproven] {finding.claim} ({cmd_issue})"
                )
                verified_findings.append(
                    finding.model_copy(update={"proven": False, "exit_code": None})
                )
                continue
            assert finding.repro_command is not None
            argv, argv_issue = _parse_sandboxed_repro_argv(
                finding.repro_command,
                host_root=self.workspace_root,
                sandbox_root=handle.workspace_path,
            )
            if argv_issue is not None or argv is None:
                filtered_unproven += 1
                unproven_notes.append(
                    f"[filtered unproven] {finding.claim} ({argv_issue})"
                )
                verified_findings.append(
                    finding.model_copy(update={"proven": False, "exit_code": None})
                )
                continue
            if self._run_command is _run_sandbox_command:
                try:
                    code, output = _run_argv_command(
                        argv,
                        cwd=handle.workspace_path,
                        timeout_s=30,
                        host_root=self.workspace_root,
                        require_fs_isolation=not self._allow_unsandboxed_repro,
                    )
                except _SandboxIsolationUnavailableError as exc:
                    raise CraftPause(
                        node_name,
                        str(exc),
                        route="blocked",
                        pause_kind="blocked",
                    ) from exc
            else:
                code, output = self._run_command(
                    finding.repro_command, cwd=handle.workspace_path, timeout_s=30
                )
            proven_failure, reason_detail = _is_proven_failure_exit(argv, code)
            if proven_failure:
                verified_findings.append(
                    finding.model_copy(
                        update={
                            "proven": True,
                            "exit_code": code,
                            "observed_output": output,
                        }
                    )
                )
            else:
                filtered_unproven += 1
                unproven_notes.append(
                    f"[filtered unproven] {finding.claim} ({reason_detail})"
                )
                verified_findings.append(
                    finding.model_copy(
                        update={
                            "proven": False,
                            "exit_code": code,
                            "observed_output": output,
                        }
                    )
                )

        proven_blocking = [
            item for item in verified_findings if item.severity == "important" and item.proven
        ]
        verdict: Literal["pass", "findings", "blocked"]
        if not proven_blocking:
            verdict = "pass"
            route = "pass"
        elif review_cycle < max(1, max_cycles):
            verdict = "findings"
            route = "findings"
        else:
            verdict = "blocked"
            route = "blocked"

        final_report = InterrogateReport(
            report_id=raw_report.report_id,
            verdict=verdict,
            findings=verified_findings,
            unproven_notes=unproven_notes,
            summary=raw_report.summary
            or f"Cycle {review_cycle}: {len(proven_blocking)} proven blocking finding(s).",
            reviewer_driver=resolved_spec,
            cold_start=True,
        )
        dumped = final_report.model_dump(mode="json")
        self._save_persisted(
            run_id,
            {"interrogate_report": dumped, "review_cycle": review_cycle},
        )
        self.store.append_event(
            run_id=run_id,
            stage="craft",
            event_type="craft.interrogate_recorded",
            actor="momus",
            payload={
                "report_id": final_report.report_id,
                "verdict": final_report.verdict,
                "cycle": review_cycle,
                "proven_blocking_count": len(proven_blocking),
                "filtered_unproven_count": filtered_unproven,
                "reviewer_driver": resolved_spec,
                "cold_start": True,
            },
        )
        if verdict == "blocked":
            raise CraftPause(
                node_name,
                f"Interrogate review exhausted {max_cycles} convergence cycles with "
                f"{len(proven_blocking)} proven blocking finding(s)",
                route="blocked",
                pause_kind="blocked",
                state_updates={
                    "interrogate_report": dumped,
                    "craft_review_cycle": review_cycle,
                },
            )
        return {
            "interrogate_report": dumped,
            "craft_review_cycle": review_cycle,
            "route": route,
        }

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
        sketch = self._hydrate_model(run_id, state, "architect_sketch", ArchitectSketch)
        handle, baseline = self._ensure_sandbox(run_id)
        # Emit role dispatch record so Argus verification is auditable in the dispatch ledger.
        self._resolve_role_driver(
            run_id=run_id,
            node_name=node_name,
            role=role,
            cold_start=True,
        )

        planned: list[str] = (
            list(commands)
            if commands is not None
            else (list(sketch.verification_plan) if sketch is not None else [])
        )
        checks: list[SurfaceCheckResult] = []
        if not planned:
            checks.append(
                SurfaceCheckResult(
                    surface="verification_plan",
                    command="",
                    status="inconclusive",
                    exit_code=None,
                    evidence="No verification commands were supplied; inconclusive never passes.",
                )
            )
        else:
            for idx, raw_cmd in enumerate(planned, start=1):
                surface = f"check_{idx}"
                cmd_issue = _validate_substantive_command(raw_cmd)
                if cmd_issue is not None:
                    checks.append(
                        SurfaceCheckResult(
                            surface=surface,
                            command=raw_cmd,
                            status="inconclusive",
                            exit_code=None,
                            evidence=(
                                f"Rejected verification command ({cmd_issue}); "
                                "inconclusive never passes."
                            ),
                        )
                    )
                    continue
                argv, argv_issue = _parse_sandboxed_repro_argv(
                    raw_cmd,
                    host_root=self.workspace_root,
                    sandbox_root=handle.workspace_path,
                )
                if argv_issue is not None or argv is None:
                    checks.append(
                        SurfaceCheckResult(
                            surface=surface,
                            command=raw_cmd,
                            status="inconclusive",
                            exit_code=None,
                            evidence=(
                                f"Rejected verification command ({argv_issue}); "
                                "inconclusive never passes."
                            ),
                        )
                    )
                    continue
                if self._run_command is _run_sandbox_command:
                    try:
                        code, output = _run_argv_command(
                            argv,
                            cwd=handle.workspace_path,
                            timeout_s=timeout_s,
                            host_root=self.workspace_root,
                            require_fs_isolation=not self._allow_unsandboxed_repro,
                        )
                    except _SandboxIsolationUnavailableError as exc:
                        raise CraftPause(
                            node_name,
                            str(exc),
                            route="blocked",
                            pause_kind="blocked",
                        ) from exc
                else:
                    code, output = self._run_command(
                        raw_cmd, cwd=handle.workspace_path, timeout_s=timeout_s
                    )
                status: Literal["pass", "fail", "inconclusive"]
                if code is None:
                    status = "inconclusive"
                elif code == 0:
                    status = "pass"
                else:
                    status = "fail"
                checks.append(
                    SurfaceCheckResult(
                        surface=surface,
                        command=raw_cmd,
                        status=status,
                        exit_code=code,
                        evidence=output,
                    )
                )

        passed = bool(checks) and all(item.status == "pass" for item in checks)
        failing_summary = "; ".join(
            f"{c.surface}={c.status} (exit={c.exit_code})"
            for c in checks
            if c.status != "pass"
        )
        report = SurfaceVerificationReport(
            passed=passed,
            checks=checks,
            summary=(
                f"All {len(checks)} surface verification check(s) passed."
                if passed
                else f"Surface verification did not pass: {failing_summary}"
            ),
        )
        dumped = report.model_dump(mode="json")
        self._save_persisted(run_id, {"surface_verification": dumped})
        self.store.append_event(
            run_id=run_id,
            stage="craft",
            event_type="craft.verify_recorded",
            actor="argus",
            payload={
                "verification_id": report.verification_id,
                "passed": report.passed,
                "check_count": len(report.checks),
                "summary": report.summary,
            },
        )
        if not report.passed:
            raise CraftPause(
                node_name,
                report.summary,
                route="blocked",
                pause_kind="blocked",
                state_updates={"surface_verification": dumped},
            )

        applied_paths: list[str] = []
        if apply_to_workspace and sketch is not None:
            clean_ephemeral_artifacts(handle, baseline, write_scope=sketch.write_scope)
            applied_paths = self._promote_sandbox_to_workspace(
                handle, baseline, sketch, node_name=node_name
            )
            remove_sandbox(handle)
        return {
            "surface_verification": dumped,
            "applied_paths": applied_paths,
            "route": "pass",
        }


__all__ = [
    "SKILLS_ROOT",
    "CraftPause",
    "CraftStage",
    "load_skill_pack",
]
