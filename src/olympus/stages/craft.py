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
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel

from olympus.core.contracts import (
    ArchitectSketch,
    BudgetLedger,
    BudgetPolicy,
    EvidenceBrief,
    InterrogateFinding,
    InterrogateReport,
    SurfaceCheckResult,
    SurfaceVerificationReport,
)
from olympus.core.fingerprint import content_hash
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
        "-h",
        "--help",
        "--version",
        "--version-verbose",
        "--collect-only",
        "--co",
        "--fixtures",
        "--funcargs",
        "--fixtures-per-test",
        "--markers",
        "--setup-plan",
        "--setup-only",
        "--cache-show",
        "--cache-clear",
        "--pastebin",
        "--list",
        "--list-tests",
        "--show-files",
        "--show-settings",
        "--show-fixes",
    }
)
_TRUSTED_SYSTEM_BIN_DIRS: frozenset[Path] = frozenset(
    {Path("/usr/bin"), Path("/bin"), Path("/usr/local/bin")}
)
_UNTRUSTED_PATH_PREFIXES: tuple[Path, ...] = (Path("/tmp"), Path("/var/tmp"))
_FORBIDDEN_SHORT_FLAGS: frozenset[str] = frozenset({"c", "e", "h", "o", "p", "w"})
_ABS_PATH_LITERAL_RE = re.compile(r"(?:^|[\s'\"=(,])/(?:etc|home|root|tmp|var|usr|opt|Users)/")
_NON_PROOF_EXIT_CODES: frozenset[int] = frozenset({126, 127})
_PYTEST_NON_FAILURE_EXIT_CODES: frozenset[int] = frozenset({2, 3, 4, 5})
_ZERO_TEST_OUTPUT_RE = re.compile(
    r"(?i)(?:"
    r"\bran\s+0\s+tests?\b"
    r"|\brunning\s+0\s+tests?\b"
    r"|\bno\s+tests?\s+ran\b"
    r"|\bcollected\s+0\s+items?\b"
    r"|\b0\s+tests?\s+run\b"
    r"|\bno\s+test\s+files?\s+found\b"
    r"|\[no\s+test\s+files\]"
    r"|\b0\s+specs?\b"
    r")"
)
_TEST_DIR_NAMES: frozenset[str] = frozenset({"test", "tests", "__tests__", "spec", "specs"})
_TEST_FILE_NAME_RE = re.compile(
    r"^(?:test_.*|.*_test\.[^.]+|.*\.test\.[^.]+|.*\.spec\.[^.]+|.*Tests?\.[^.]+|.*Spec\.[^.]+)$"
)
_TEST_DEF_RE = re.compile(r"(?m)^\s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\s*\(")
_UNITTEST_RAN_RE = re.compile(r"(?i)\bran\s+(\d+)\s+tests?\b")
_UNITTEST_OK_SKIPPED_RE = re.compile(r"(?i)\bOK\s*\([^)]*\bskipped=(\d+)\b[^)]*\)")
_SKIPPED_SUMMARY_RE = re.compile(
    r"(?i)\b([1-9]\d*)\s+(?:skipped|ignored|deselected|xfailed)\b"
)
_POSITIVE_PASSED_RE = re.compile(r"(?i)\b[1-9]\d*\s+passed\b")
_PYTEST_PROGRESS_WITH_PCT_RE = re.compile(r"^([.sSxXfFeE]+)\s+\[\s*\d+%\]$")
_PYTEST_SKIP_ONLY_PCT_RE = re.compile(r"^[sSxX]+\s+\[\s*\d+%\]$")
_PYTEST_SKIP_ONLY_LINE_RE = re.compile(r"^[sSxX]+$")
_PYTEST_PASS_DOT_LINE_RE = re.compile(r"^[.sSxXfFeE]*\.[.sSxXfFeE]*(?:\s+\[\s*\d+%\])?$")
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


class _SandboxIsolationUnavailableError(RuntimeError):
    """Raised when OS-level read-only FS isolation (bwrap) is required but unavailable."""
_MAX_EXCERPT_CHARS = 1500
_MAX_DIFF_PROMPT_CHARS = 24000
_SANDBOXED_COMMAND_RULES = (
    "Command execution contract (STRICT): commands execute directly from the sandbox repo root "
    "with shell=False inside a read-only allowlist. Use ONLY direct test/lint/typecheck runners "
    "such as `python -m unittest <test_file>`, `pytest -q`, `uv run pytest -q`, `ruff check .`, "
    "`mypy src`, `npm test`, `pnpm test`, `go test ./...`, or `cargo test` with repo-relative "
    "paths. NEVER use `cd`, `&&`, `;`, `|`, `>`, `python -c`, `grep`, mutating flags (`--fix`, "
    "`-w`, `-o`), or absolute host workspace paths."
)


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


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text) if text else ""


def _has_zero_test_output(output: str) -> bool:
    clean = _strip_ansi(output)
    return bool(clean and _ZERO_TEST_OUTPUT_RE.search(clean))


def _has_all_skipped_test_output(output: str, *, runner: str = "") -> str | None:
    """Detect test runs that exited 0 only because every test was skipped/ignored."""
    clean = _strip_ansi(output)
    if not clean:
        return None
    unittest_skip = _UNITTEST_OK_SKIPPED_RE.search(clean)
    if unittest_skip is not None:
        skipped_count = int(unittest_skip.group(1))
        ran_matches = _UNITTEST_RAN_RE.findall(clean)
        ran_count = int(ran_matches[-1]) if ran_matches else 0
        if skipped_count > 0 and (ran_count == 0 or skipped_count >= ran_count):
            return f"all unittest tests were skipped ({skipped_count}/{ran_count} executed)"
    if _SKIPPED_SUMMARY_RE.search(clean) and not _POSITIVE_PASSED_RE.search(clean):
        # Allow unittest suites that ran more tests than skipped (e.g. Ran 5 tests, OK (skipped=1)).
        ran_matches = _UNITTEST_RAN_RE.findall(clean)
        if (
            ran_matches
            and unittest_skip is not None
            and int(ran_matches[-1]) > int(unittest_skip.group(1))
        ):
            return None
        return "all tests were skipped/ignored (0 passed)"
    if runner in {"pytest", "py.test"} and not _POSITIVE_PASSED_RE.search(clean):
        non_empty = [line.strip() for line in clean.splitlines() if line.strip()]
        has_pass_dot = any(_PYTEST_PASS_DOT_LINE_RE.match(line) for line in non_empty)
        if not has_pass_dot and non_empty:
            pct_lines = [line for line in non_empty if _PYTEST_PROGRESS_WITH_PCT_RE.match(line)]
            if pct_lines and all(_PYTEST_SKIP_ONLY_PCT_RE.match(line) for line in pct_lines):
                return "all pytest tests were skipped/xfailed (0 passed)"
            if _PYTEST_SKIP_ONLY_LINE_RE.match(non_empty[-1]):
                return "all pytest tests were skipped/xfailed (0 passed)"
    return None


def _baseline_fingerprint(baseline: SandboxBaseline) -> str:
    """Compute a deterministic fingerprint binding receipts to one exact sandbox baseline."""
    return content_hash(
        {
            "is_git": baseline.is_git,
            "base_commit": baseline.base_commit,
            "entries": list(baseline.entries),
        }
    )


def _is_test_path(path: str) -> bool:
    posix = PurePosixPath(path.strip().replace("\\", "/"))
    if not posix.name:
        return False
    if any(part.lower() in _TEST_DIR_NAMES for part in posix.parts[:-1]):
        return True
    return bool(_TEST_FILE_NAME_RE.match(posix.name))


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_test_def_names(text: str) -> set[str]:
    return set(_TEST_DEF_RE.findall(text))


def _resolve_argv_test_files(workspace_path: Path, argv: Sequence[str]) -> list[str]:
    """Extract existing workspace-relative test file paths referenced by a test command."""
    found: list[str] = []
    for token in argv[1:]:
        if token.startswith("-"):
            continue
        raw_target = token.split("::", 1)[0].strip()
        if not raw_target or Path(raw_target).is_absolute() or ".." in Path(raw_target).parts:
            continue
        candidates = [raw_target]
        if "/" not in raw_target and "\\" not in raw_target and not raw_target.endswith(".py"):
            parts = raw_target.split(".")
            for prefix_len in range(len(parts), 0, -1):
                rel_py = "/".join(parts[:prefix_len]) + ".py"
                candidates.append(rel_py)
        for rel in candidates:
            norm = PurePosixPath(rel.replace("\\", "/")).as_posix()
            if (workspace_path / norm).is_file() and norm not in found:
                found.append(norm)
                break
    return found


def _is_proven_failure_exit(
    argv: Sequence[str], code: int | None, output: str = ""
) -> tuple[bool, str]:
    """Return (proven, detail) distinguishing real test failures from tool/signal errors."""
    clean_output = _strip_ansi(output)
    if code is None:
        return False, "repro timed out or failed to execute"
    if code == 0:
        return False, "repro exited 0"
    if code < 0 or code >= 128:
        return False, f"repro terminated by signal, exit={code}"
    if code in _NON_PROOF_EXIT_CODES:
        return False, f"command not found/not executable, exit={code}"
    if _has_zero_test_output(clean_output):
        return False, f"command ran zero tests, exit={code}"
    runner = _effective_repro_runner(argv)
    if runner in {"pytest", "py.test"} and code != 1:
        if code == 5:
            return False, "pytest collected no tests, exit=5"
        return False, f"pytest did not complete test execution, exit={code}"
    if runner == "unittest" and code == 2:
        return False, "unittest CLI/usage error, exit=2"
    return True, f"repro failed with exit={code}"


def _is_proven_green_exit(
    argv: Sequence[str], code: int | None, output: str = ""
) -> tuple[bool, str]:
    """Return (passed, detail) rejecting non-zero, zero-test, and all-skipped runs."""
    clean_output = _strip_ansi(output)
    if code is None:
        return False, "verification command timed out or failed to execute"
    if code != 0:
        return False, f"exit_code={code}: {_trim_output(clean_output, 600)}"
    if _has_zero_test_output(clean_output):
        return False, f"vacuous zero-test run (exit=0): {_trim_output(clean_output, 600)}"
    runner = _effective_repro_runner(argv)
    skipped_issue = _has_all_skipped_test_output(clean_output, runner=runner)
    if skipped_issue is not None:
        return False, f"{skipped_issue}: {_trim_output(clean_output, 600)}"
    return True, "green exit=0"


def _verify_red_tests_preserved(
    workspace_path: Path, red_receipt: Mapping[str, Any]
) -> str | None:
    """Verify locked red-phase test files match their exact SHA-256 snapshot byte-for-byte."""
    raw_hashes = red_receipt.get("red_test_file_hashes")
    hash_map = raw_hashes if isinstance(raw_hashes, Mapping) else {}
    for rel, expected_sha in hash_map.items():
        if not isinstance(rel, str) or not rel.strip() or not isinstance(expected_sha, str):
            continue
        target = workspace_path / rel
        if not target.is_file():
            return f"Locked red test file was deleted or is missing: {rel}"
        actual_sha = _file_sha256(target)
        if actual_sha != expected_sha:
            return (
                f"Locked red test file was modified after red phase (SHA-256 mismatch): {rel}"
            )

    raw_paths = red_receipt.get("required_test_paths")
    required_paths = (
        [str(p) for p in raw_paths if isinstance(p, str) and p.strip()]
        if isinstance(raw_paths, list)
        else []
    )
    baseline_hashes = red_receipt.get("red_baseline_hashes")
    baseline_map = baseline_hashes if isinstance(baseline_hashes, Mapping) else {}
    added_defs = red_receipt.get("added_test_defs")
    added_defs_map = added_defs if isinstance(added_defs, Mapping) else {}

    for rel in required_paths:
        target = workspace_path / rel
        if not target.is_file():
            return f"Required red test file was deleted or is missing: {rel}"
        raw_bytes = target.read_bytes()
        if not raw_bytes.strip():
            return f"Required red test file was emptied: {rel}"
        if rel in baseline_map:
            current_hash = hashlib.sha256(raw_bytes).hexdigest()
            if baseline_map[rel] is not None and current_hash == baseline_map[rel]:
                return f"Red-phase regression test file was reverted to pre-red baseline: {rel}"
        expected_defs = added_defs_map.get(rel)
        if isinstance(expected_defs, list) and expected_defs:
            current_defs = _extract_test_def_names(raw_bytes.decode("utf-8", errors="replace"))
            missing = [name for name in expected_defs if name not in current_defs]
            if missing:
                return (
                    f"Red-phase test definition(s) removed from {rel}: {', '.join(sorted(missing))}"
                )
    return None


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
    if arg == "-V" or (
        not arg.startswith("--")
        and len(arg) > 1
        and arg[1].lower() not in {"k", "m"}
        and "V" in arg[1:]
    ):
        return f"short flag -V (in {arg!r}) is forbidden in read-only repro_command"
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
    trim: bool = True,
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
    return proc.returncode, (_trim_output(combined) if trim else combined.strip())


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


def _format_investigation_report(brief: EvidenceBrief) -> str:
    """Format an ``EvidenceBrief`` into a deterministic read-only forensics report."""
    lines: list[str] = [
        "# Investigation Report",
        "",
        f"**Brief ID:** `{brief.brief_id}`  ",
        f"**Content Hash:** `{brief.content_hash}`",
        "",
        "## Root-Cause Summary",
        "",
        brief.summary.strip(),
        "",
        "## Verified Codebase Evidence",
        "",
    ]
    if brief.facts:
        for fact in brief.facts:
            loc = f"{fact.path}:{fact.line}" if fact.line is not None else fact.path
            lines.append(f"- `{loc}` — {fact.observation.strip()}")
    else:
        lines.append("- (no file:line facts recorded)")

    if brief.candidate_files:
        lines.extend(["", "## Implicated Files", ""])
        for path in brief.candidate_files:
            lines.append(f"- `{path}`")

    if brief.repro_command:
        lines.extend(["", "## Reproduction Signal", ""])
        lines.append(f"- **Command:** `{brief.repro_command}`")
        lines.append(f"- **Verified on base:** `{brief.repro_verified}`")
        if brief.repro_exit_code is not None:
            lines.append(f"- **Exit code:** `{brief.repro_exit_code}`")
        if brief.repro_excerpt.strip():
            lines.extend(["", "```text", brief.repro_excerpt.strip(), "```"])

    if brief.unresolved_questions:
        lines.extend(["", "## Open Questions", ""])
        for question in brief.unresolved_questions:
            lines.append(f"- {question.strip()}")

    return "\n".join(lines) + "\n"


def _checkout_base_ref(repo_root: Path, sandbox_path: Path, base_ref: str) -> str:
    """Resolve and detach-checkout ``base_ref`` inside a throwaway git worktree."""
    ref = base_ref.strip()
    if not ref or ref.startswith("-"):
        raise ValueError(f"Invalid base_ref {base_ref!r}")
    resolve_proc = subprocess.run(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if resolve_proc.returncode != 0:
        detail = (resolve_proc.stderr or resolve_proc.stdout).strip() or "unknown ref"
        raise ValueError(f"Unable to resolve base_ref {ref!r}: {detail}")
    resolved_sha = resolve_proc.stdout.strip()
    checkout_proc = subprocess.run(
        ["git", "checkout", "--detach", resolved_sha],
        cwd=str(sandbox_path),
        capture_output=True,
        text=True,
        check=False,
    )
    if checkout_proc.returncode != 0:
        detail = (checkout_proc.stderr or checkout_proc.stdout).strip() or "checkout failed"
        raise ValueError(f"Failed to checkout base_ref {ref!r} ({resolved_sha}): {detail}")
    return resolved_sha


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

    def _is_inside_workspace(self, path: Path) -> bool:
        resolved = path.resolve(strict=False)
        return resolved == self.workspace_root or resolved.is_relative_to(self.workspace_root)

    def _craft_dir(self, run_id: str, *, read_only_workspace: bool = False) -> Path:
        root = self.artifacts_root
        if read_only_workspace and self._is_inside_workspace(root):
            fallback_root = (
                Path(tempfile.gettempdir()).resolve(strict=False) / "olympus-read-only-artifacts"
            )
            if self._is_inside_workspace(fallback_root):
                raise CraftPause(
                    "explore",
                    "Refusing to write read-only investigation artifacts inside workspace_root",
                    route="blocked",
                    pause_kind="blocked",
                )
            root = fallback_root
        path = root / run_id / "craft"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _state_file(self, run_id: str, *, read_only_workspace: bool = False) -> Path:
        return self._craft_dir(run_id, read_only_workspace=read_only_workspace) / "state.json"

    def _load_persisted(
        self, run_id: str, *, read_only_workspace: bool = False
    ) -> dict[str, Any]:
        path = self._state_file(run_id, read_only_workspace=read_only_workspace)
        if not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _save_persisted(
        self,
        run_id: str,
        updates: Mapping[str, Any],
        *,
        read_only_workspace: bool = False,
    ) -> dict[str, Any]:
        current = self._load_persisted(run_id, read_only_workspace=read_only_workspace)
        current.update(updates)
        self._state_file(run_id, read_only_workspace=read_only_workspace).write_text(
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

    def _has_remaining_budget(self, run_id: str) -> bool:
        """Return True when the run ledger still permits reserving one more LLM call."""
        try:
            snap = self.store.get_run(run_id)
        except Exception:  # noqa: BLE001
            return True
        ledger = snap.get("budget_ledger")
        policy = snap.get("budget_policy")
        if isinstance(ledger, BudgetLedger) and isinstance(policy, BudgetPolicy):
            return ledger.can_reserve(policy)
        return True

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
        read_only_workspace: bool = False,
    ) -> tuple[T, str]:
        driver, driver_model, resolved_spec = self._resolve_role_driver(
            run_id=run_id,
            node_name=node_name,
            role=role,
            cold_start=True,
        )
        active_provider = self._provider_for_model(driver_model)
        toolkit = WorkspaceToolkit(workspace_path, read_only=True, env_root=self.workspace_root)
        trace_path = (
            self._craft_dir(run_id, read_only_workspace=read_only_workspace)
            / f"{trace_name}.jsonl"
        )
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
        baseline = load_sandbox_baseline(handle) if handle.reused else None
        baseline_invalidated = not handle.reused or baseline is None
        if handle.is_git_worktree:
            head_proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(handle.workspace_path),
                capture_output=True,
                text=True,
                check=False,
            )
            actual_head = head_proc.stdout.strip() if head_proc.returncode == 0 else None
            if actual_head:
                if handle.base_commit != actual_head:
                    handle = replace(handle, base_commit=actual_head)
                if baseline is not None and baseline.base_commit != actual_head:
                    baseline = None
                    baseline_invalidated = True
        if baseline is None:
            baseline = snapshot_sandbox_state(handle)
            save_sandbox_baseline(handle, baseline)
        if baseline_invalidated:
            persisted = self._load_persisted(run_id)
            if persisted.get("red_test_receipt") is not None:
                self._save_persisted(run_id, {"red_test_receipt": None})
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

    def _partition_verification_plan(
        self, commands: Sequence[str]
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """Split verification_plan into allowlisted commands and rejected (cmd, reason) pairs."""
        valid: list[str] = []
        rejected: list[tuple[str, str]] = []
        placeholder_sandbox = (self.sandbox_root or self.workspace_root.parent) / ".verify-plan"
        for raw_cmd in commands:
            cleaned = raw_cmd.strip()
            if not cleaned:
                continue
            argv, issue = _parse_sandboxed_repro_argv(
                cleaned,
                host_root=self.workspace_root,
                sandbox_root=placeholder_sandbox,
            )
            if issue is not None or argv is None:
                rejected.append((cleaned, issue or "rejected by sandbox allowlist"))
            else:
                valid.append(cleaned)
        return valid, rejected

    def _run_verified_command(
        self,
        command: str | None,
        *,
        cwd: Path,
        timeout_s: int,
        node_name: str,
    ) -> tuple[list[str] | None, int | None, str, str | None]:
        """Validate and execute one allowlisted command under the bwrap/sandbox boundary."""
        cmd_issue = _validate_substantive_command(command)
        if cmd_issue is not None:
            return None, None, "", cmd_issue
        assert command is not None
        argv, argv_issue = _parse_sandboxed_repro_argv(
            command,
            host_root=self.workspace_root,
            sandbox_root=cwd,
        )
        if argv_issue is not None or argv is None:
            return None, None, "", argv_issue or "command rejected by allowlist"
        if self._run_command is _run_sandbox_command:
            try:
                code, raw_output = _run_argv_command(
                    argv,
                    cwd=cwd,
                    timeout_s=timeout_s,
                    host_root=self.workspace_root,
                    require_fs_isolation=not self._allow_unsandboxed_repro,
                    trim=False,
                )
            except _SandboxIsolationUnavailableError as exc:
                raise CraftPause(
                    node_name,
                    str(exc),
                    route="blocked",
                    pause_kind="blocked",
                ) from exc
        else:
            code, raw_output = self._run_command(command, cwd=cwd, timeout_s=timeout_s)
        return argv, code, raw_output, None

    def _assert_read_only_store_outside_workspace(self, node_name: str) -> None:
        """Fail closed in read-only report mode if RunStore SQLite DB lives inside workspace."""
        raw_db_path = getattr(self.store, "db_path", None)
        if raw_db_path is None:
            return
        db_str = str(raw_db_path).strip()
        if not db_str or db_str == ":memory:" or db_str.startswith("file::memory:"):
            return
        if self._is_inside_workspace(Path(db_str)):
            raise CraftPause(
                node_name,
                "Read-only investigation refuses to write to RunStore inside workspace_root "
                f"({raw_db_path}); configure RunStore outside the target repository.",
                route="blocked",
                pause_kind="blocked",
            )

    # ------------------------------------------------------------------ 6 orthogonal primitives

    def explore_reproduce(
        self,
        *,
        run_id: str,
        raw_input: str,
        constraints: Sequence[str] = (),
        scope_paths: Sequence[str] = (),
        base_ref: str | None = None,
        node_name: str = "explore",
        role: str = "explorer",
        skill_refs: Sequence[str] = (),
        require_repro: bool = False,
        read_only_report: bool = False,
        max_steps: int = 8,
    ) -> dict[str, Any]:
        if read_only_report:
            self._assert_read_only_store_outside_workspace(node_name)
        toolkit = WorkspaceToolkit(
            self.workspace_root, read_only=True, env_root=self.workspace_root
        )
        inventory = format_workspace_inventory(toolkit, scope_paths=scope_paths)
        if read_only_report:
            base_system_prompt = (
                "You are Prometheus (Read-Only Root-Cause Forensics Explorer). Inspect the "
                "repository strictly read-only without modifying any files and return ONLY "
                "valid EvidenceBrief JSON conforming strictly to the schema. Explain the exact "
                "root-cause mechanism in `summary`, ground every claim in `file:line` entries "
                "inside `facts`, list implicated files in `candidate_files`, and leave "
                "repro_verified=false."
            )
        else:
            base_system_prompt = (
                "You are Prometheus (Evidence-First Explorer). Inspect the repository read-only "
                "(batching file reads efficiently) and return ONLY valid EvidenceBrief JSON "
                "conforming strictly to the schema. Leave repro_verified=false (the deterministic "
                "control plane verifies repro commands). Record solvable engineering defaults, "
                "tie-breakers, and environment notes in `assumptions` and keep "
                "`unresolved_questions` as `[]` unless execution is genuinely blocked on an "
                "unresolvable product/intent decision.\n"
                f"{_SANDBOXED_COMMAND_RULES}"
            )
        system_prompt = _compose_system_prompt(base_system_prompt, skill_refs)
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
        report_note = (
            "\nThis is a terminal read-only investigation (`stop_after: report`). "
            "Do not plan edits; produce a complete, self-contained root-cause EvidenceBrief."
            if read_only_report
            else ""
        )
        user_prompt = (
            f"Goal:\n{raw_input}\n\n{inventory}{constraints_block}{repro_note}{report_note}\n\n"
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
            read_only_workspace=read_only_report,
        )
        # Never trust LLM-claimed repro verification; run repro inside a throwaway sandbox.
        repro_verified = False
        repro_exit_code = brief.repro_exit_code
        repro_excerpt = brief.repro_excerpt
        repro_receipt: dict[str, Any] | None = None
        if require_repro:
            cmd_issue = _validate_substantive_command(brief.repro_command)
            if cmd_issue is not None:
                brief = brief.model_copy(
                    update={"repro_verified": False, "repro_exit_code": None}
                )
                dumped = brief.model_dump(mode="json")
                self._save_persisted(
                    run_id,
                    {"evidence_brief": dumped},
                    read_only_workspace=read_only_report,
                )
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
            resolved_base_ref = repro_handle.base_commit or "HEAD"
            try:
                if base_ref and not repro_handle.is_git_worktree:
                    brief = brief.model_copy(
                        update={"repro_verified": False, "repro_exit_code": None}
                    )
                    dumped = brief.model_dump(mode="json")
                    self._save_persisted(
                        run_id,
                        {"evidence_brief": dumped},
                        read_only_workspace=read_only_report,
                    )
                    raise CraftPause(
                        node_name,
                        f"Cannot verify explicit base_ref {base_ref!r} on a non-git copy sandbox",
                        route="blocked",
                        pause_kind="blocked",
                        state_updates={"evidence_brief": dumped},
                    )
                if base_ref and repro_handle.is_git_worktree:
                    try:
                        resolved_base_ref = _checkout_base_ref(
                            self.workspace_root,
                            repro_handle.workspace_path,
                            base_ref,
                        )
                    except ValueError as exc:
                        brief = brief.model_copy(
                            update={"repro_verified": False, "repro_exit_code": None}
                        )
                        dumped = brief.model_dump(mode="json")
                        self._save_persisted(
                            run_id,
                            {"evidence_brief": dumped},
                            read_only_workspace=read_only_report,
                        )
                        raise CraftPause(
                            node_name,
                            str(exc),
                            route="blocked",
                            pause_kind="blocked",
                            state_updates={"evidence_brief": dumped},
                        ) from exc
                argv, code, excerpt, issue = self._run_verified_command(
                    brief.repro_command,
                    cwd=repro_handle.workspace_path,
                    timeout_s=60,
                    node_name=node_name,
                )
                if issue is not None or argv is None:
                    brief = brief.model_copy(
                        update={"repro_verified": False, "repro_exit_code": None}
                    )
                    dumped = brief.model_dump(mode="json")
                    self._save_persisted(
                        run_id,
                        {"evidence_brief": dumped},
                        read_only_workspace=read_only_report,
                    )
                    raise CraftPause(
                        node_name,
                        f"Unsafe repro_command rejected in read-only explore ({issue})",
                        route="blocked",
                        pause_kind="blocked",
                        state_updates={"evidence_brief": dumped},
                    )
            finally:
                remove_sandbox(repro_handle)
            repro_exit_code = code
            repro_verified, repro_detail = _is_proven_failure_exit(argv, code, output=excerpt)
            repro_excerpt = _trim_output(excerpt)
            brief = brief.model_copy(
                update={
                    "repro_exit_code": repro_exit_code,
                    "repro_excerpt": repro_excerpt,
                    "repro_verified": repro_verified,
                }
            )
            repro_receipt = {
                "command": brief.repro_command,
                "exit_code": repro_exit_code,
                "excerpt": repro_excerpt,
                "verified": repro_verified,
                "base_ref": resolved_base_ref,
            }
            if not repro_verified:
                dumped = brief.model_dump(mode="json")
                self._save_persisted(
                    run_id,
                    {"evidence_brief": dumped, "repro_receipt": repro_receipt},
                    read_only_workspace=read_only_report,
                )
                raise CraftPause(
                    node_name,
                    f"Reproduction command did not prove failure on base workspace "
                    f"({repro_detail}): {brief.repro_command}",
                    route="blocked",
                    pause_kind="blocked",
                    state_updates={
                        "evidence_brief": dumped,
                        "repro_receipt": repro_receipt,
                    },
                )
        else:
            brief = brief.model_copy(update={"repro_verified": False})

        dumped = brief.model_dump(mode="json")
        persisted_updates: dict[str, Any] = {"evidence_brief": dumped}
        if repro_receipt is not None:
            persisted_updates["repro_receipt"] = repro_receipt
        report_md: str | None = None
        report_path: Path | None = None
        if read_only_report:
            report_md = _format_investigation_report(brief)
            report_path = (
                self._craft_dir(run_id, read_only_workspace=True) / "investigation_report.md"
            )
            report_path.write_text(report_md, encoding="utf-8")
            persisted_updates["investigation_report"] = report_md
        self._save_persisted(
            run_id,
            persisted_updates,
            read_only_workspace=read_only_report,
        )
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
                "repro_command": brief.repro_command,
                "repro_exit_code": brief.repro_exit_code,
                "read_only_report": read_only_report,
                "unresolved_questions": list(brief.unresolved_questions),
            },
        )
        if read_only_report:
            assert report_md is not None and report_path is not None
            self.store.append_event(
                run_id=run_id,
                stage="craft",
                event_type="craft.investigation_report_recorded",
                actor="prometheus",
                payload={
                    "brief_id": brief.brief_id,
                    "content_hash": brief.content_hash,
                    "fact_count": len(brief.facts),
                    "candidate_file_count": len(brief.candidate_files),
                    "read_only": True,
                    "report_path": str(report_path),
                },
            )
            return {
                "evidence_brief": dumped,
                "investigation_report": report_md,
                "read_only": True,
                "route": "report",
            }
        if brief.unresolved_questions:
            raise CraftPause(
                node_name,
                "; ".join(brief.unresolved_questions),
                route="clarify",
                pause_kind="clarify",
                state_updates={"evidence_brief": dumped},
            )
        result_state: dict[str, Any] = {"evidence_brief": dumped, "route": "ready"}
        if repro_receipt is not None:
            result_state["repro_receipt"] = repro_receipt
        return result_state

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
        max_steps: int = 6,
    ) -> dict[str, Any]:
        brief = self._hydrate_model(run_id, state, "evidence_brief", EvidenceBrief)
        toolkit = WorkspaceToolkit(
            self.workspace_root, read_only=True, env_root=self.workspace_root
        )
        inventory = format_workspace_inventory(toolkit, scope_paths=scope_paths)
        system_prompt = _compose_system_prompt(
            "You are Athena (Lightweight Architect). Compare at least two candidate shapes and "
            "return ONLY valid ArchitectSketch JSON with a tight write_scope of workspace-relative "
            "paths (never absolute paths) and concrete direct allowlisted verification_plan "
            "commands.\n"
            f"{_SANDBOXED_COMMAND_RULES}",
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
        normalized_scope: list[str] = []
        for raw_entry in sketch.write_scope:
            entry = raw_entry.strip()
            candidate = Path(entry)
            if candidate.is_absolute():
                resolved = candidate.resolve(strict=False)
                if resolved.is_relative_to(self.workspace_root):
                    rel = resolved.relative_to(self.workspace_root).as_posix()
                    entry = rel if rel and rel != "." else entry
            if entry and entry not in normalized_scope:
                normalized_scope.append(entry)
        if normalized_scope and normalized_scope != sketch.write_scope:
            sketch = ArchitectSketch.model_validate(
                {
                    **sketch.model_dump(mode="json", exclude={"content_hash"}),
                    "write_scope": normalized_scope,
                }
            )
        valid_cmds, rejected_cmds = self._partition_verification_plan(sketch.verification_plan)
        if rejected_cmds:
            if self._has_remaining_budget(run_id):
                repair_details = "\n".join(
                    f"- {cmd!r}: {reason}" for cmd, reason in rejected_cmds
                )
                repair_prompt = (
                    f"{user_prompt}\n\n"
                    f"Previous ArchitectSketch JSON had invalid verification_plan commands that "
                    f"fail the deterministic read-only sandbox allowlist:\n{repair_details}\n\n"
                    f"Keep your chosen architecture and write_scope, and replace ONLY the "
                    f"rejected verification_plan entries with valid allowlisted commands "
                    f"(e.g. `python -m unittest <test_file>`, `pytest -q`, `ruff check .`). "
                    f"Return ONLY valid ArchitectSketch JSON."
                )
                try:
                    repaired, _ = self._run_structured(
                        run_id=run_id,
                        node_name=node_name,
                        role=role,
                        actor="athena:architect",
                        workspace_path=self.workspace_root,
                        system_prompt=system_prompt,
                        user_prompt=repair_prompt,
                        model_cls=ArchitectSketch,
                        max_steps=3,
                        trace_name="architect_repair",
                    )
                    repaired_valid, repaired_rejected = self._partition_verification_plan(
                        repaired.verification_plan
                    )
                    if repaired_valid:
                        # Preserve all original valid commands and never let the repair turn
                        # mutate write_scope, forbidden_globs, or architecture fields.
                        valid_cmds = list(dict.fromkeys([*valid_cmds, *repaired_valid]))
                        rejected_cmds = repaired_rejected
                except Exception:  # noqa: BLE001 - fall back to any valid commands from initial sketch
                    pass
            if not valid_cmds:
                reasons = "; ".join(f"{cmd!r} ({reason})" for cmd, reason in rejected_cmds)
                raise CraftPause(
                    node_name,
                    f"ArchitectSketch.verification_plan has no allowlisted read-only commands: "
                    f"{reasons}",
                    route="blocked",
                    pause_kind="blocked",
                )
            sketch = ArchitectSketch.model_validate(
                {
                    **sketch.model_dump(mode="json", exclude={"content_hash"}),
                    "verification_plan": valid_cmds,
                }
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

    def _ensure_red_test_receipt(
        self,
        *,
        run_id: str,
        raw_input: str,
        state: Mapping[str, Any],
        brief: EvidenceBrief | None,
        sketch: ArchitectSketch,
        handle: SandboxHandle,
        baseline: SandboxBaseline,
        node_name: str,
        role: str,
        system_prompt: str,
        strict_scope: bool,
        max_steps: int,
    ) -> dict[str, Any]:
        """Establish deterministic pre-fix failing-test evidence on the target surface."""
        persisted = self._load_persisted(run_id)
        del state  # Never trust caller-supplied red receipts; only trust control-plane state.
        current_baseline_fp = _baseline_fingerprint(baseline)
        existing = persisted.get("red_test_receipt")
        if (
            isinstance(existing, Mapping)
            and existing.get("verified") is True
            and isinstance(existing.get("command"), str)
            and existing["command"].strip()
            and existing.get("baseline_fingerprint") == current_baseline_fp
            and isinstance(existing.get("red_test_file_hashes"), Mapping)
        ):
            return dict(existing)

        target_cmd = (
            brief.repro_command
            if brief is not None and _has_executable_command(brief.repro_command)
            else (sketch.verification_plan[0] if sketch.verification_plan else None)
        )
        cmd_issue = _validate_substantive_command(target_cmd)
        if cmd_issue is not None or target_cmd is None:
            raise CraftPause(
                node_name,
                f"failing_test_first requires a valid failing test command ({cmd_issue})",
                route="blocked",
                pause_kind="blocked",
            )
        argv, argv_issue = _parse_sandboxed_repro_argv(
            target_cmd,
            host_root=self.workspace_root,
            sandbox_root=handle.workspace_path,
        )
        if argv_issue is not None or argv is None:
            raise CraftPause(
                node_name,
                f"failing_test_first rejected unsafe test command ({argv_issue}): {target_cmd}",
                route="blocked",
                pause_kind="blocked",
            )

        # Always verify against the untouched production code of the current sandbox
        # rather than blindly trusting a potentially stale historical base_ref receipt.
        pre_argv, pre_code, pre_raw_output, pre_issue = self._run_verified_command(
            target_cmd,
            cwd=handle.workspace_path,
            timeout_s=60,
            node_name=node_name,
        )
        if pre_issue is not None or pre_argv is None:
            raise CraftPause(
                node_name,
                f"failing_test_first rejected test command ({pre_issue}): {target_cmd}",
                route="blocked",
                pause_kind="blocked",
            )
        required_test_paths: list[str] = []
        for candidate_path in sketch.write_scope:
            if (
                _is_test_path(candidate_path)
                and (handle.workspace_path / candidate_path).is_file()
                and candidate_path not in required_test_paths
            ):
                required_test_paths.append(candidate_path)
        for resolved_path in _resolve_argv_test_files(handle.workspace_path, pre_argv):
            if resolved_path not in required_test_paths:
                required_test_paths.append(resolved_path)

        pre_failed, _ = _is_proven_failure_exit(pre_argv, pre_code, output=pre_raw_output)
        if pre_failed:
            persisted_repro = persisted.get("repro_receipt")
            source = (
                "explore_repro"
                if (
                    isinstance(persisted_repro, Mapping)
                    and persisted_repro.get("verified") is True
                    and persisted_repro.get("command") == target_cmd
                    and (
                        handle.base_commit is None
                        or persisted_repro.get("base_ref") == handle.base_commit
                    )
                )
                else "sandbox_pre_fix"
            )
            red_test_file_hashes: dict[str, str] = {}
            for rel in required_test_paths:
                digest = _file_sha256(handle.workspace_path / rel)
                if digest is not None:
                    red_test_file_hashes[rel] = digest
            receipt: dict[str, Any] = {
                "command": target_cmd,
                "exit_code": pre_code,
                "excerpt": _trim_output(pre_raw_output),
                "verified": True,
                "source": source,
                "baseline_fingerprint": current_baseline_fp,
                "required_test_paths": required_test_paths,
                "red_test_file_hashes": red_test_file_hashes,
            }
            self._save_persisted(run_id, {"red_test_receipt": receipt})
            self.store.append_event(
                run_id=run_id,
                stage="craft",
                event_type="craft.red_test_recorded",
                actor="hephaestus",
                payload={
                    "command": target_cmd,
                    "exit_code": pre_code,
                    "source": source,
                    "baseline_fingerprint": current_baseline_fp,
                    "required_test_paths": required_test_paths,
                },
            )
            return receipt

        # Base test currently passes: require Hephaestus to write a failing regression test first
        # inside test-only write_scope so production code cannot be mutated during the red phase.
        test_write_scope = [path for path in sketch.write_scope if _is_test_path(path)]
        if not test_write_scope:
            raise CraftPause(
                node_name,
                "failing_test_first requires at least one test file in write_scope to author a "
                "failing regression test before editing production code",
                route="blocked",
                pause_kind="blocked",
            )
        pre_red_hashes: dict[str, str | None] = {
            rel: _file_sha256(handle.workspace_path / rel) for rel in test_write_scope
        }
        pre_red_defs: dict[str, set[str]] = {
            rel: _extract_test_def_names(
                (handle.workspace_path / rel).read_text(encoding="utf-8", errors="replace")
            )
            if (handle.workspace_path / rel).is_file()
            else set()
            for rel in test_write_scope
        }
        red_prompt = (
            f"Goal:\n{raw_input}\n\n"
            f"Authorized test-only write_scope: {json.dumps(test_write_scope)}\n"
            f"Target test command: `{target_cmd}` (currently exits {pre_code}).\n\n"
            "PHASE 1 (RED TEST FIRST): Add or update a regression test inside the test-only "
            "write_scope that reproduces the defect and FAILS against untouched production code "
            "when running the target test command. Do NOT edit production files yet."
        )
        self._run_worker_turn(
            run_id=run_id,
            node_name=node_name,
            role=role,
            actor="hephaestus:red_test",
            workspace_path=handle.workspace_path,
            write_scope=test_write_scope,
            forbidden_globs=sketch.forbidden_globs,
            strict_scope=True,
            system_prompt=system_prompt,
            user_prompt=red_prompt,
            max_steps=max(4, min(max_steps, 16)),
            trace_name="build_red_test",
        )
        clean_ephemeral_artifacts(handle, baseline, write_scope=test_write_scope)
        red_gate = run_diff_gate(
            handle,
            baseline,
            allowed_write_scope=test_write_scope,
            forbidden_globs=sketch.forbidden_globs,
            strict_scope=True,
        )
        if red_gate.rejected:
            raise CraftPause(
                node_name,
                red_gate.rejection_message(),
                route="blocked",
                pause_kind="blocked",
            )
        red_argv, red_code, red_raw_output, red_issue = self._run_verified_command(
            target_cmd,
            cwd=handle.workspace_path,
            timeout_s=60,
            node_name=node_name,
        )
        if red_issue is not None or red_argv is None:
            raise CraftPause(
                node_name,
                f"failing_test_first rejected test command ({red_issue}): {target_cmd}",
                route="blocked",
                pause_kind="blocked",
            )
        red_failed, red_detail = _is_proven_failure_exit(red_argv, red_code, output=red_raw_output)
        if not red_failed:
            raise CraftPause(
                node_name,
                f"failing_test_first requires a proven failing test before applying the fix "
                f"({red_detail}): {target_cmd}",
                route="blocked",
                pause_kind="blocked",
            )
        red_changed = list(red_gate.changed_paths)
        red_required_paths = sorted(set([*required_test_paths, *red_changed]))
        red_baseline_hashes: dict[str, str | None] = {
            rel: pre_red_hashes.get(rel) for rel in red_changed
        }
        red_test_file_hashes = {}
        for rel in red_required_paths:
            digest = _file_sha256(handle.workspace_path / rel)
            if digest is not None:
                red_test_file_hashes[rel] = digest
        added_test_defs: dict[str, list[str]] = {}
        for rel in red_changed:
            target_file = handle.workspace_path / rel
            if target_file.is_file():
                after_defs = _extract_test_def_names(
                    target_file.read_text(encoding="utf-8", errors="replace")
                )
                newly_added = sorted(after_defs - pre_red_defs.get(rel, set()))
                if newly_added:
                    added_test_defs[rel] = newly_added
        receipt = {
            "command": target_cmd,
            "exit_code": red_code,
            "excerpt": _trim_output(red_raw_output),
            "verified": True,
            "source": "red_test_turn",
            "baseline_fingerprint": current_baseline_fp,
            "changed_paths": red_changed,
            "required_test_paths": red_required_paths,
            "red_test_file_hashes": red_test_file_hashes,
            "red_baseline_hashes": red_baseline_hashes,
            "added_test_defs": added_test_defs,
        }
        self._save_persisted(run_id, {"red_test_receipt": receipt})
        self.store.append_event(
            run_id=run_id,
            stage="craft",
            event_type="craft.red_test_recorded",
            actor="hephaestus",
            payload={
                "command": target_cmd,
                "exit_code": red_code,
                "source": "red_test_turn",
                "baseline_fingerprint": current_baseline_fp,
                "changed_paths": red_changed,
                "required_test_paths": red_required_paths,
            },
        )
        return receipt

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
        failing_test_first: bool = False,
        max_diff_gate_retries: int = 2,
        max_steps: int = 24,
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
        red_test_receipt: dict[str, Any] | None = None
        if failing_test_first:
            red_test_receipt = self._ensure_red_test_receipt(
                run_id=run_id,
                raw_input=raw_input,
                state=state,
                brief=brief,
                sketch=sketch,
                handle=handle,
                baseline=baseline,
                node_name=node_name,
                role=role,
                system_prompt=system_prompt,
                strict_scope=strict_scope,
                max_steps=max_steps,
            )

        proven_findings = (
            prior_review.proven_blocking_findings() if prior_review is not None else []
        )
        red_context = ""
        if red_test_receipt is not None:
            red_context = (
                f"\nVerified failing test before fix:\n"
                f"- command: `{red_test_receipt['command']}`\n"
                f"- exit_code: {red_test_receipt['exit_code']}\n"
                f"- excerpt: {_trim_output(str(red_test_receipt.get('excerpt', '')), 600)}\n"
                "Fix the root cause inside write_scope so this exact test surface turns green "
                "(exit 0) without weakening assertions or adding defensive workarounds.\n"
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
                f"{red_context}"
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
                f"{red_context}"
                f"Implement the chosen shape cleanly inside write_scope and run verification."
            )

        user_prompt = base_user_prompt
        gate_result: DiffGateResult | None = None
        last_outcome: TurnOutcome | None = None
        green_test_receipt: dict[str, Any] | None = None
        last_green_failure: str | None = None
        locked_red_paths: list[str] = []
        if red_test_receipt is not None and isinstance(
            red_test_receipt.get("red_test_file_hashes"), Mapping
        ):
            locked_red_paths = [
                str(p)
                for p in red_test_receipt["red_test_file_hashes"]
                if isinstance(p, str) and p.strip()
            ]
        worker_forbidden_globs = list(dict.fromkeys([*sketch.forbidden_globs, *locked_red_paths]))
        for attempt in range(1, max(1, max_diff_gate_retries) + 1):
            last_outcome, _ = self._run_worker_turn(
                run_id=run_id,
                node_name=node_name,
                role=role,
                actor="hephaestus:build",
                workspace_path=handle.workspace_path,
                write_scope=sketch.write_scope,
                forbidden_globs=worker_forbidden_globs,
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
            if gate_result.rejected:
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
                continue

            if red_test_receipt is not None:
                target_cmd = str(red_test_receipt["command"])
                preserved_issue = _verify_red_tests_preserved(
                    handle.workspace_path, red_test_receipt
                )
                if preserved_issue is not None:
                    green_code = None
                    green_raw_output = ""
                    green_ok = False
                    failure_reason = preserved_issue
                else:
                    green_argv, green_code, green_raw_output, green_issue = (
                        self._run_verified_command(
                            target_cmd,
                            cwd=handle.workspace_path,
                            timeout_s=60,
                            node_name=node_name,
                        )
                    )
                    if green_issue is not None or green_argv is None:
                        green_ok = False
                        failure_reason = green_issue or "verification command rejected"
                    else:
                        green_ok, green_detail = _is_proven_green_exit(
                            green_argv, green_code, output=green_raw_output
                        )
                        failure_reason = green_detail
                if green_ok:
                    green_test_receipt = {
                        "command": target_cmd,
                        "exit_code": 0,
                        "excerpt": _trim_output(green_raw_output),
                        "verified": True,
                    }
                    last_green_failure = None
                    break
                last_green_failure = (
                    f"Post-fix verification failed on red test surface `{target_cmd}` "
                    f"({failure_reason})"
                )
                self.store.append_event(
                    run_id=run_id,
                    stage="craft",
                    event_type="craft.failing_test_still_red",
                    actor="hephaestus",
                    payload={
                        "attempt": attempt,
                        "command": target_cmd,
                        "exit_code": green_code,
                        "reason": failure_reason,
                    },
                )
                user_prompt = (
                    f"{base_user_prompt}\n\n"
                    f"[FAILING_TEST_FIRST POST-FIX CHECK FAILED]\n{last_green_failure}"
                )
                continue
            break

        assert gate_result is not None
        if gate_result.rejected:
            raise CraftPause(
                node_name,
                gate_result.rejection_message(),
                route="blocked",
                pause_kind="blocked",
            )
        if red_test_receipt is not None and green_test_receipt is None:
            raise CraftPause(
                node_name,
                last_green_failure
                or f"Post-fix verification did not turn green for `{red_test_receipt['command']}`",
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
        persisted_updates: dict[str, Any] = {
            "craft_changed_paths": list(gate_result.changed_paths),
            "sandbox_path": str(handle.workspace_path),
        }
        failing_test_receipt: dict[str, Any] | None = None
        if red_test_receipt is not None and green_test_receipt is not None:
            failing_test_receipt = {
                "red": red_test_receipt,
                "green": green_test_receipt,
                "same_surface": red_test_receipt["command"] == green_test_receipt["command"],
            }
            persisted_updates["failing_test_receipt"] = failing_test_receipt
            self.store.append_event(
                run_id=run_id,
                stage="craft",
                event_type="craft.failing_test_verified",
                actor="hephaestus",
                payload={
                    "command": red_test_receipt["command"],
                    "red_exit_code": red_test_receipt["exit_code"],
                    "green_exit_code": green_test_receipt["exit_code"],
                    "red_source": red_test_receipt.get("source"),
                    "same_surface": True,
                },
            )
        self._save_persisted(run_id, persisted_updates)
        self.store.append_event(
            run_id=run_id,
            stage="craft",
            event_type="craft.build_completed",
            actor="hephaestus",
            payload={
                "changed_paths": list(gate_result.changed_paths),
                "diff_bytes": len(diff.encode("utf-8")),
                "failing_test_first": failing_test_first,
            },
        )
        result_state: dict[str, Any] = {
            "craft_diff": diff,
            "craft_changed_paths": list(gate_result.changed_paths),
            "route": "built",
        }
        if failing_test_receipt is not None:
            result_state["failing_test_receipt"] = failing_test_receipt
        return result_state

    def hygiene_pass(
        self,
        *,
        run_id: str,
        state: Mapping[str, Any],
        node_name: str = "hygiene",
        role: str = "hygiene",
        skill_refs: Sequence[str] = (),
        strict_scope: bool = True,
        max_steps: int = 4,
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
            "the current diff without changing runtime behavior or touching files outside scope. "
            "If the diff is already clean and concise, finish in 1 step with zero edits.",
            skill_refs,
        )
        user_prompt = (
            f"Authorized write_scope: {json.dumps(sketch.write_scope)}\n"
            f"Inspect and clean only the diff below:\n```diff\n"
            f"{_trim_output(diff, _MAX_DIFF_PROMPT_CHARS)}\n```"
        )
        persisted_red = self._load_persisted(run_id).get("red_test_receipt")
        hygiene_locked_paths = (
            [
                str(p)
                for p in persisted_red["red_test_file_hashes"]
                if isinstance(p, str) and p.strip()
            ]
            if isinstance(persisted_red, Mapping)
            and isinstance(persisted_red.get("red_test_file_hashes"), Mapping)
            else []
        )
        hygiene_forbidden = list(dict.fromkeys([*sketch.forbidden_globs, *hygiene_locked_paths]))
        self._run_worker_turn(
            run_id=run_id,
            node_name=node_name,
            role=role,
            actor="hephaestus:hygiene",
            workspace_path=handle.workspace_path,
            write_scope=sketch.write_scope,
            forbidden_globs=hygiene_forbidden,
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
        if isinstance(persisted_red, Mapping) and hygiene_locked_paths:
            preserved_issue = _verify_red_tests_preserved(handle.workspace_path, persisted_red)
            if preserved_issue is not None:
                raise CraftPause(
                    node_name,
                    f"Hygiene pass violated red-test immutability: {preserved_issue}",
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
        max_steps: int = 6,
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
            "This turn is READ-ONLY (no shell tool): read the changed files at most once, do NOT "
            "loop re-reading the same file, and return ONLY valid InterrogateReport JSON. "
            "Every `important` finding MUST carry an allowlisted `repro_command` that fails "
            "(non-zero exit) when the deterministic control plane runs it in the sandbox; unproven "
            "claims will be filtered out deterministically.\n"
            f"{_SANDBOXED_COMMAND_RULES}",
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
                    code, raw_output = _run_argv_command(
                        argv,
                        cwd=handle.workspace_path,
                        timeout_s=30,
                        host_root=self.workspace_root,
                        require_fs_isolation=not self._allow_unsandboxed_repro,
                        trim=False,
                    )
                except _SandboxIsolationUnavailableError as exc:
                    raise CraftPause(
                        node_name,
                        str(exc),
                        route="blocked",
                        pause_kind="blocked",
                    ) from exc
            else:
                code, raw_output = self._run_command(
                    finding.repro_command, cwd=handle.workspace_path, timeout_s=30
                )
            proven_failure, reason_detail = _is_proven_failure_exit(argv, code, output=raw_output)
            trimmed_output = _trim_output(raw_output)
            if proven_failure:
                verified_findings.append(
                    finding.model_copy(
                        update={
                            "proven": True,
                            "exit_code": code,
                            "observed_output": trimmed_output,
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
                            "observed_output": trimmed_output,
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
        brief = self._hydrate_model(run_id, state, "evidence_brief", EvidenceBrief)
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
        if (
            commands is None
            and brief is not None
            and brief.repro_verified
            and brief.repro_command
            and brief.repro_command not in planned
        ):
            planned.append(brief.repro_command)
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
                        code, raw_output = _run_argv_command(
                            argv,
                            cwd=handle.workspace_path,
                            timeout_s=timeout_s,
                            host_root=self.workspace_root,
                            require_fs_isolation=not self._allow_unsandboxed_repro,
                            trim=False,
                        )
                    except _SandboxIsolationUnavailableError as exc:
                        raise CraftPause(
                            node_name,
                            str(exc),
                            route="blocked",
                            pause_kind="blocked",
                        ) from exc
                else:
                    code, raw_output = self._run_command(
                        raw_cmd, cwd=handle.workspace_path, timeout_s=timeout_s
                    )
                status: Literal["pass", "fail", "inconclusive"]
                runner = _effective_repro_runner(argv)
                if code is None:
                    status = "inconclusive"
                elif (
                    code == 0
                    and runner == "unittest"
                    and re.search(r"\bRan 0 tests\b", _strip_ansi(raw_output))
                ):
                    status = "fail"
                elif code == 0:
                    if (
                        _has_zero_test_output(raw_output)
                        or _has_all_skipped_test_output(raw_output, runner=runner) is not None
                    ):
                        status = "inconclusive"
                    else:
                        status = "pass"
                else:
                    status = "fail"
                checks.append(
                    SurfaceCheckResult(
                        surface=surface,
                        command=raw_cmd,
                        status=status,
                        exit_code=code,
                        evidence=_trim_output(raw_output),
                    )
                )

        passed = bool(checks) and all(item.status == "pass" for item in checks)
        failing_summary = "; ".join(
            f"{c.surface} ({c.command})={c.status} (exit={c.exit_code})"
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
