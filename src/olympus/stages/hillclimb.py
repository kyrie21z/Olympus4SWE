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
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

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
    return [trusted0, *argv[1:]], None


def _run_argv_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_s: int = 60,
    host_root: Path | None = None,
    require_fs_isolation: bool = True,
) -> tuple[int | None, str]:
    """Execute one tokenized command (``shell=False``) with read-only host FS via ``bwrap``."""
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

    env = dict(os.environ)
    env["PATH"] = safe_path
    env["HOME"] = str(metric_home)
    env.setdefault("CI", "1")
    env["TMPDIR"] = str(metric_tmp)
    env["TEMP"] = str(metric_tmp)
    env["TMP"] = str(metric_tmp)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["RUFF_NO_CACHE"] = "1"
    env["RUFF_CACHE_DIR"] = str(metric_tmp / "ruff_cache")
    env["MYPY_CACHE_DIR"] = str(metric_tmp / "mypy_cache")
    env["PYTEST_ADDOPTS"] = f"-o cache_dir={metric_tmp / 'pytest_cache'} -p no:cacheprovider"

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
            str(metric_tmp),
            str(metric_tmp),
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
        shutil.rmtree(metric_tmp, ignore_errors=True)
    combined = "\n".join(part for part in (proc.stdout, proc.stderr) if part and part.strip())
    return proc.returncode, _trim_output(combined)


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


def _git_has_changes(cwd: Path) -> bool:
    proc = _run_git(cwd, "status", "--porcelain", "--untracked-files=all")
    if proc.returncode != 0:
        raise RuntimeError(f"git status failed: {proc.stderr.strip()}")
    return bool(proc.stdout.strip())


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
    clean_proc = _run_git(cwd, "clean", "-fd")
    if clean_proc.returncode != 0:
        raise RuntimeError(f"git clean -fd failed: {clean_proc.stderr.strip()}")


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
    """Parse a finite numeric metric from command output via ``metric_regex``."""
    pattern = re.compile(metric_regex)
    match = pattern.search(output)
    if match is None:
        return None, f"metric_regex {metric_regex!r} did not match command output"
    raw = match.group(1) if match.lastindex else match.group(0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, f"matched metric token {raw!r} is not a valid float"
    if not math.isfinite(value):
        return None, f"matched metric value {raw!r} is not finite"
    return value, None


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
        code, output, cmd_issue = self._run_safe_command(
            config.metric_command,
            sandbox_path=sandbox_path,
            timeout_s=timeout_s,
            node_name=node_name,
        )
        if cmd_issue is not None:
            return None, f"Rejected metric_command: {cmd_issue}", output
        if code is None:
            return None, f"metric_command failed or timed out: {output[:200]}", output
        if code != 0:
            return None, f"metric_command exited {code}: {output[:200]}", output
        value, parse_err = parse_metric_value(output, config.metric_regex)
        if parse_err is not None or value is None:
            return None, f"{parse_err} (output={output[:200]!r})", output
        return value, None, output

    def _promote_sandbox_to_workspace(
        self,
        handle: SandboxHandle,
        baseline: SandboxBaseline,
        config: HillclimbConfig,
        *,
        frozen_hashes: Mapping[str, str],
        node_name: str = "hillclimb",
    ) -> list[str]:
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

        # Phase 2: transactional apply with full backup+mode rollback on any OSError.
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
            raise HillclimbPause(
                node_name,
                f"Atomic workspace promotion failed and was rolled back: {exc}{extra}",
                route="blocked",
                pause_kind="blocked",
            ) from exc
        return sorted(set(applied))

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
        if isinstance(explicit_config, Mapping):
            merged.update(explicit_config)
        state_cfg = state.get("hillclimb_config")
        if isinstance(state_cfg, HillclimbConfig):
            return state_cfg
        if isinstance(state_cfg, Mapping):
            merged.update(state_cfg)
        if not merged:
            raise HillclimbPause(
                node_name,
                "HillclimbConfig is required in state['hillclimb_config'] or node params",
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

        # 2. Measure baseline metric inside the isolated worktree (fail closed).
        baseline_metric, base_err, _base_output = self._measure_metric(
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
        if base_err is not None or baseline_metric is None:
            raise HillclimbPause(
                node_name,
                f"Baseline metric measurement failed: {base_err}",
                route="blocked",
                pause_kind="blocked",
            )

        self.store.append_event(
            run_id=run_id,
            stage="hillclimb",
            event_type="hillclimb.baseline_recorded",
            actor="hephaestus",
            payload={
                "config_id": cfg.config_id,
                "config_hash": cfg.content_hash,
                "baseline_metric": baseline_metric,
                "direction": cfg.direction,
                "target_value": cfg.target_value,
                "base_commit": last_kept_sha,
                "frozen_hashes": dict(frozen_hashes),
            },
        )

        best_metric = baseline_metric
        rows: list[HillclimbIterationRow] = []
        consecutive_non_improving = 0
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

        # Persist initial trail before iteration 1 so even early stops have an artifact.
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
        else:
            for iteration in range(1, cfg.max_iterations + 1):
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
                                "Rejected and reverted: frozen harness modified during metric run "
                                f"({', '.join(runtime_tamper)})"
                            )
                            consecutive_non_improving += 1
                        elif measure_err is not None or measured is None:
                            _git_revert_to_commit(handle.workspace_path, last_kept_sha)
                            decision = "error"
                            reason = f"Reverted after metric error: {measure_err}"
                            consecutive_non_improving += 1
                        elif is_strictly_better(measured, best_metric, cfg.direction):
                            # Re-clean ephemeral artifacts created during metric run before commit.
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

                self.store.append_event(
                    run_id=run_id,
                    stage="hillclimb",
                    event_type="hillclimb.iteration_recorded",
                    actor="hephaestus",
                    payload=row.model_dump(mode="json"),
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

        applied_paths: list[str] = []
        if apply_to_workspace:
            clean_ephemeral_artifacts(
                handle,
                baseline,
                write_scope=[*cfg.write_scope, *cfg.frozen_harness_paths],
            )
            applied_paths = self._promote_sandbox_to_workspace(
                handle,
                baseline,
                cfg,
                frozen_hashes=frozen_hashes,
                node_name=node_name,
            )
            remove_sandbox(handle)

        return {
            "hillclimb_config": dumped_cfg,
            "hillclimb_report": dumped_report,
            "hillclimb_frozen_hashes": dict(frozen_hashes),
            "hillclimb_decision_trail_path": str(trail_path),
            "applied_paths": applied_paths,
            "route": "done",
        }

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
        cfg = self._hydrate_model(run_id, state, "hillclimb_config", HillclimbConfig)
        report = self._hydrate_model(run_id, state, "hillclimb_report", HillclimbReport)
        if cfg is None or report is None:
            raise HillclimbPause(
                node_name,
                "Hillclimb verification requires hillclimb_config and hillclimb_report in state",
                route="blocked",
                pause_kind="blocked",
            )
        persisted = self._load_persisted(run_id)
        raw_hashes = state.get("hillclimb_frozen_hashes") or persisted.get("frozen_hashes")
        if not isinstance(raw_hashes, Mapping) or not raw_hashes:
            raise HillclimbPause(
                node_name,
                "Missing frozen harness SHA-256 baseline for hillclimb verification",
                route="blocked",
                pause_kind="blocked",
            )
        frozen_hashes = {str(k): str(v) for k, v in raw_hashes.items()}

        handle, baseline = self._ensure_sandbox(run_id)
        self._resolve_role_driver(
            run_id=run_id,
            node_name=node_name,
            role=role,
            cold_start=True,
        )

        checks: list[SurfaceCheckResult] = []

        # 1. Verify frozen harness SHA-256 anti-tamper integrity.
        tampered = detect_frozen_harness_tamper(handle.workspace_path, frozen_hashes)
        checks.append(
            SurfaceCheckResult(
                surface="frozen_harness_integrity",
                command="sha256(frozen_harness_paths)",
                status="fail" if tampered else "pass",
                exit_code=1 if tampered else 0,
                evidence=(
                    f"Tampered frozen harness paths: {', '.join(tampered)}"
                    if tampered
                    else f"All {len(frozen_hashes)} frozen harness file(s) match baseline SHA-256."
                ),
            )
        )

        # 2. Re-run frozen metric command inside the sandbox and ensure no regression vs baseline.
        measured, measure_err, measure_out = self._measure_metric(
            sandbox_path=handle.workspace_path,
            config=cfg,
            timeout_s=timeout_s,
            node_name=node_name,
        )
        post_tamper = detect_frozen_harness_tamper(handle.workspace_path, frozen_hashes)
        if post_tamper:
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
                    evidence=measure_err or "Metric failed to parse; inconclusive never passes.",
                )
            )
        else:
            regressed = is_strictly_better(report.baseline_metric, measured, cfg.direction)
            checks.append(
                SurfaceCheckResult(
                    surface="hillclimb_metric",
                    command=cfg.metric_command,
                    status="fail" if regressed else "pass",
                    exit_code=1 if regressed else 0,
                    evidence=(
                        f"Verified metric={measured:g} vs baseline={report.baseline_metric:g} "
                        f"({cfg.direction}); output={measure_out[:400]}"
                    ),
                )
            )

        # 3. Run any additional verification commands from params or state['verification_commands'].
        extra_cmds: list[str] = []
        if commands is not None:
            extra_cmds.extend(str(item) for item in commands)
        else:
            state_cmds = state.get("verification_commands")
            if isinstance(state_cmds, list):
                extra_cmds.extend(str(item) for item in state_cmds)

        for idx, raw_cmd in enumerate(extra_cmds, start=1):
            code, output, cmd_issue = self._run_safe_command(
                raw_cmd,
                sandbox_path=handle.workspace_path,
                timeout_s=timeout_s,
                node_name=node_name,
            )
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
            clean_ephemeral_artifacts(
                handle,
                baseline,
                write_scope=[*cfg.write_scope, *cfg.frozen_harness_paths],
            )
            applied_paths = self._promote_sandbox_to_workspace(
                handle,
                baseline,
                cfg,
                frozen_hashes=frozen_hashes,
                node_name=node_name,
            )
            remove_sandbox(handle)

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
