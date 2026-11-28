"""Cohesive Arena stage implementing the ``arena_race`` composite primitive.

Primitive:
- ``arena_race`` (Athena + Hephaestus): synthesize/validate an ``ArenaRubric``
  (3-6 criteria), spawn ``N >= 2`` isolated git worktrees from ``base_ref``,
  execute candidate builds concurrently across configured ``role_drivers`` with
  ``DiffGate`` + ``clean_ephemeral_artifacts`` + safe verification, run an
  independent cold-start Judge (``role: arena_judge``) to produce ``ArenaVerdict``,
  promote the winner's diff into the run's craft sandbox, optionally apply scoped
  ``graft_items`` verified by ``DiffGate``, and guarantee cleanup of all temporary
  candidate worktrees in ``finally:`` on both success and failure.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import os
import shutil
import stat
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from olympus.core.contracts import (
    ArchitectSketch,
    ArenaCandidateResult,
    ArenaGraftItem,
    ArenaRubric,
    ArenaVerdict,
    CandidateShape,
    EvidenceBrief,
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
from olympus.harness.loop import AgentLoop, TurnCancelledError
from olympus.harness.native import NativeDriver
from olympus.harness.structured import run_structured_turn
from olympus.harness.worktree import (
    SandboxBaseline,
    SandboxHandle,
    _safe_label,
    capture_sandbox_diff,
    clean_ephemeral_artifacts,
    collect_sandbox_changes,
    create_sandbox,
    is_git_repo,
    remove_sandbox,
    save_sandbox_baseline,
    snapshot_sandbox_state,
)
from olympus.providers.base import LLMProvider
from olympus.stages.common import default_artifacts_root, format_workspace_inventory
from olympus.stages.craft import (
    CraftPause,
    CraftStage,
    _compose_system_prompt,
    _parse_sandboxed_repro_argv,
    _run_argv_command,
    _run_sandbox_command,
    _SandboxIsolationUnavailableError,
    _trim_output,
    _validate_substantive_command,
)
from olympus.tools.aci import WorkspacePolicy, WorkspaceToolkit

_MAX_CANDIDATE_DIFF_CHARS = 10_000
_MAX_GRAFT_DIFF_CHARS = 8_000

_DEFAULT_STRATEGY_HINTS: tuple[str, ...] = (
    "Minimal surgical solution: prioritize simplicity, minimal diff footprint, "
    "direct edits, and zero unnecessary abstractions.",
    "Clean structural decomposition: prioritize crisp module boundaries, clear "
    "invariants, readability, and long-term maintainability.",
    "Edge-case & robustness hardening: prioritize explicit failure handling, "
    "defensive boundary contracts, and thorough test coverage.",
    "Performance & ergonomics focus: prioritize low overhead, clean ergonomics, "
    "and cohesive data flow.",
)


def _strategy_hint_for_index(idx: int) -> str:
    return _DEFAULT_STRATEGY_HINTS[(idx - 1) % len(_DEFAULT_STRATEGY_HINTS)]


def _summarize_diff(changed_paths: Sequence[str], diff_text: str) -> str:
    if not changed_paths:
        return "No files changed."
    added = sum(
        1
        for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    removed = sum(
        1
        for line in diff_text.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    preview = ", ".join(changed_paths[:6])
    suffix = f" (+{len(changed_paths) - 6} more)" if len(changed_paths) > 6 else ""
    return f"{len(changed_paths)} file(s) changed ({preview}{suffix}); +{added}/-{removed} lines."


def _normalize_write_scope(paths: Sequence[str], workspace_root: Path) -> list[str]:
    """Convert any workspace-prefixed absolute paths in write_scope to repo-relative POSIX paths."""
    ws = workspace_root.resolve()
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        cleaned = raw.strip()
        if not cleaned:
            continue
        candidate = Path(cleaned)
        if candidate.is_absolute():
            resolved = candidate.resolve(strict=False)
            if resolved == ws:
                cleaned = "."
            elif resolved.is_relative_to(ws):
                cleaned = resolved.relative_to(ws).as_posix()
        else:
            cleaned = cleaned.replace("\\", "/")
            while cleaned.startswith("./") and len(cleaned) > 2:
                cleaned = cleaned[2:]
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)
    return normalized


def _filter_allowed_verification_plan(
    commands: Sequence[str],
    *,
    workspace_root: Path,
) -> list[str]:
    """Keep sandbox-allowed test/check runner commands when at least one valid command exists.

    If zero commands pass the sandboxed runner allowlist, return the original list untouched so
    downstream verification still fails closed (inconclusive never passes).
    """
    valid: list[str] = []
    seen: set[str] = set()
    for raw in commands:
        cleaned = raw.strip()
        if not cleaned or cleaned in seen:
            continue
        if _validate_substantive_command(cleaned) is not None:
            continue
        argv, issue = _parse_sandboxed_repro_argv(
            cleaned,
            host_root=workspace_root,
            sandbox_root=workspace_root,
        )
        if issue is None and argv is not None:
            seen.add(cleaned)
            valid.append(cleaned)
    return valid if valid else [cmd.strip() for cmd in commands if cmd.strip()]


def _is_cancelled_stop_reason(stop_reason: str) -> bool:
    normalized = stop_reason.strip().lower()
    return any(token in normalized for token in ("cancel", "interrupt", "abort"))


def _planned_sandbox_handle(
    repo_root: Path,
    *,
    label: str,
    sandbox_root: Path | None,
    is_git_worktree: bool,
) -> SandboxHandle:
    """Compute the deterministic SandboxHandle before creation so partial failures clean up."""
    root = repo_root.resolve()
    safe_label = _safe_label(label)
    sandbox_dir = (
        sandbox_root.resolve() / safe_label
        if sandbox_root is not None
        else root / ".olympus" / "sandboxes" / safe_label
    )
    return SandboxHandle(
        repo_root=root,
        workspace_path=sandbox_dir,
        base_commit=None,
        is_git_worktree=is_git_worktree,
    )


def _resolve_git_base_commit(repo_root: Path, base_ref: str | None) -> str | None:
    """Lock ``base_ref`` (including default ``HEAD``) to one immutable commit SHA up front.

    Fail closed when an explicit ``base_ref`` is requested on a non-git directory or
    when the target ref cannot be resolved to a commit.
    """
    explicit_ref = base_ref.strip() if base_ref is not None and base_ref.strip() else None
    if not is_git_repo(repo_root):
        if explicit_ref is not None:
            raise ValueError(
                f"Explicit base_ref {explicit_ref!r} cannot be resolved because "
                f"{repo_root} is not a git repository"
            )
        return None
    target_ref = explicit_ref or "HEAD"
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", f"{target_ref}^{{commit}}"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        if explicit_ref is None:
            return None
        detail = (proc.stderr or proc.stdout).strip() or f"unknown ref {target_ref!r}"
        raise ValueError(f"Invalid base_ref {target_ref!r}: {detail}")
    return proc.stdout.strip()


def _mirror_sandbox_changes(
    *,
    source_handle: SandboxHandle,
    target_handle: SandboxHandle,
    base_baseline: SandboxBaseline,
    allowed_write_scope: Sequence[str],
    forbidden_globs: Sequence[str],
    node_name: str,
) -> list[str]:
    """Transactionally mirror ``source_handle`` changes onto ``target_handle`` with rollback.

    Covers edits, additions, deletions, symlinks, and file mode bits across the union of
    paths changed in either ``source_handle`` or ``target_handle`` relative to ``base_baseline``.
    On any ``OSError``, restores every touched file/mode/symlink and removes created dirs.
    """
    source_changed = collect_sandbox_changes(source_handle, base_baseline)
    target_changed = collect_sandbox_changes(target_handle, base_baseline)
    union_paths = sorted(set(source_changed) | set(target_changed))
    policy = WorkspacePolicy(
        allowed_write_scope=list(allowed_write_scope),
        forbidden_write_scope=list(forbidden_globs),
    )
    src_root = source_handle.workspace_path.resolve()
    dst_root = target_handle.workspace_path.resolve()

    copy_plan: list[tuple[Path, Path, str | None]] = []
    delete_plan: list[Path] = []
    for rel in union_paths:
        if policy.forbidden_match(rel) is not None or not policy.allows_relative_path(rel):
            raise CraftPause(
                node_name,
                f"Refusing to commit graft change outside write_scope: {rel}",
                route="blocked",
                pause_kind="blocked",
            )
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise CraftPause(
                node_name,
                f"Refusing to commit unsafe relative graft path: {rel}",
                route="blocked",
                pause_kind="blocked",
            )
        src = src_root / rel_path
        dst = dst_root / rel_path
        parent_resolved = dst.parent.resolve(strict=False)
        if parent_resolved != dst_root and not parent_resolved.is_relative_to(dst_root):
            raise CraftPause(
                node_name,
                f"Refusing to commit graft path escaping sandbox root: {rel}",
                route="blocked",
                pause_kind="blocked",
            )
        if not src.exists() and not src.is_symlink():
            if dst.exists() and not dst.is_symlink() and dst.is_dir():
                raise CraftPause(
                    node_name,
                    f"Refusing to delete directory via graft commit: {rel}",
                    route="blocked",
                    pause_kind="blocked",
                )
            delete_plan.append(dst)
            continue
        link_target: str | None = None
        if src.is_symlink():
            link_target = os.readlink(src)
            resolved_target = (dst.parent / link_target).resolve(strict=False)
            if resolved_target != dst_root and not resolved_target.is_relative_to(dst_root):
                raise CraftPause(
                    node_name,
                    f"Refusing to commit graft symlink escaping sandbox root: "
                    f"{rel} -> {link_target}",
                    route="blocked",
                    pause_kind="blocked",
                )
        elif src.is_dir() or not src.is_file():
            raise CraftPause(
                node_name,
                f"Refusing to commit non-regular graft path: {rel}",
                route="blocked",
                pause_kind="blocked",
            )
        if dst.exists() and not dst.is_symlink() and dst.is_dir():
            raise CraftPause(
                node_name,
                f"Refusing to overwrite directory via graft commit: {rel}",
                route="blocked",
                pause_kind="blocked",
            )
        copy_plan.append((src, dst, link_target))

    backups: list[tuple[Path, str, tuple[bytes, int] | str | None]] = []
    created_dirs: list[Path] = []
    created_dir_set: set[Path] = set()
    removed_dirs: list[tuple[Path, int]] = []
    removed_dir_set: set[Path] = set()
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
            while parent != dst_root and parent.is_relative_to(dst_root):
                try:
                    parent_mode = stat.S_IMODE(parent.stat().st_mode)
                    parent.rmdir()
                except OSError:
                    break
                if parent not in removed_dir_set:
                    removed_dirs.append((parent, parent_mode))
                    removed_dir_set.add(parent)
                parent = parent.parent
        for src, dst, link_target in copy_plan:
            missing_parents: list[Path] = []
            cur_parent = dst.parent
            while cur_parent != dst_root and not cur_parent.exists():
                missing_parents.append(cur_parent)
                cur_parent = cur_parent.parent
            for missing_dir in reversed(missing_parents):
                missing_dir.mkdir()
                if missing_dir not in removed_dir_set and missing_dir not in created_dir_set:
                    created_dirs.append(missing_dir)
                    created_dir_set.add(missing_dir)
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
        # Recreate removed directories shallow-to-deep first so nested files can be restored.
        for dir_path, _saved_mode in sorted(
            removed_dirs, key=lambda item: len(item[0].parts)
        ):
            with contextlib.suppress(OSError):
                dir_path.mkdir(parents=True, exist_ok=True)
        for path, kind, saved in reversed(backups):
            try:
                if kind == "missing":
                    if path.is_symlink() or path.is_file():
                        path.unlink(missing_ok=True)
                elif kind == "symlink" and isinstance(saved, str):
                    if path.is_symlink() or path.is_file():
                        path.unlink(missing_ok=True)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(saved, path)
                elif kind == "file" and isinstance(saved, tuple):
                    data, saved_mode = saved
                    if path.is_symlink():
                        path.unlink(missing_ok=True)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    os.chmod(path, saved_mode)
            except OSError:
                pass
        # Only remove directories that genuinely did not exist before mirroring started.
        for directory in sorted(created_dirs, key=lambda item: len(item.parts), reverse=True):
            if directory in removed_dir_set:
                continue
            with contextlib.suppress(OSError):
                directory.rmdir()
        # Restore original modes on removed directories deep-to-shallow (after file writes).
        for dir_path, saved_mode in sorted(
            removed_dirs, key=lambda item: len(item[0].parts), reverse=True
        ):
            with contextlib.suppress(OSError):
                os.chmod(dir_path, saved_mode)
        raise CraftPause(
            node_name,
            f"Transactional graft commit failed and winner sandbox was rolled back: {exc}",
            route="blocked",
            pause_kind="blocked",
        ) from exc
    return source_changed


class ArenaStage:
    """Deterministic multi-worktree competition, rubric judging, and grafting stage."""

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
        self._store_lock = threading.Lock()
        self._craft = CraftStage(
            store=self.store,
            workspace_root=self.workspace_root,
            provider=self.provider,
            artifacts_root=self.artifacts_root,
            sandbox_root=self.sandbox_root,
            execution_drivers=self.execution_drivers,
            driver_registry=self.driver_registry,
            model_provider=self.model_provider,
            on_driver_handle=self.on_driver_handle,
            command_runner=self._run_command,
            allow_unsandboxed_repro=self._allow_unsandboxed_repro,
        )

    # ------------------------------------------------------------------ paths & audit helpers

    def _arena_dir(self, run_id: str) -> Path:
        path = self.artifacts_root / run_id / "arena"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _append_event(
        self,
        *,
        run_id: str,
        event_type: str,
        actor: str,
        payload: Mapping[str, Any],
    ) -> None:
        with self._store_lock:
            self.store.append_event(
                run_id=run_id,
                stage="arena",
                event_type=event_type,
                actor=actor,
                payload=dict(payload),
            )

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
        normalized = role.strip().lower()
        if normalized in self.execution_drivers.role_drivers:
            raw_spec = self.execution_drivers.role_drivers[normalized]
        elif (
            normalized.startswith("arena_runner_")
            and "arena_runner" in self.execution_drivers.role_drivers
        ):
            raw_spec = self.execution_drivers.role_drivers["arena_runner"]
        else:
            raw_spec = self.execution_drivers.resolve_driver_for_role(normalized)
        resolved_spec = resolve_driver_spec(raw_spec, provider=self.provider)
        self._append_event(
            run_id=run_id,
            event_type="arena.role_dispatched",
            actor=f"arena:{normalized}",
            payload={
                "node": node_name,
                "role": normalized,
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
        trace_path = self._arena_dir(run_id) / f"{trace_name}.jsonl"
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
        trace_path = self._arena_dir(run_id) / f"{trace_name}.jsonl"
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

    # ------------------------------------------------------------------ safe verification

    def _verify_candidate_sandbox(
        self,
        *,
        handle: SandboxHandle,
        commands: Sequence[str],
        timeout_s: int,
        node_name: str,
    ) -> tuple[bool, int | None, str]:
        """Run rubric verification commands using CraftStage's sandboxed read-only primitives."""
        if not commands:
            return False, None, "No verification commands configured."
        excerpts: list[str] = []
        first_fail_code: int | None = None
        all_passed = True
        for raw_cmd in commands:
            cmd_issue = _validate_substantive_command(raw_cmd)
            if cmd_issue is not None:
                all_passed = False
                excerpts.append(f"[{raw_cmd}] rejected: {cmd_issue}")
                continue
            argv, argv_issue = _parse_sandboxed_repro_argv(
                raw_cmd,
                host_root=self.workspace_root,
                sandbox_root=handle.workspace_path,
            )
            if argv_issue is not None or argv is None:
                all_passed = False
                excerpts.append(f"[{raw_cmd}] rejected: {argv_issue}")
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
                    raw_cmd,
                    cwd=handle.workspace_path,
                    timeout_s=timeout_s,
                )
            if code != 0:
                all_passed = False
                if first_fail_code is None:
                    first_fail_code = code
            excerpts.append(f"[{raw_cmd}] exit={code}\n{output}")
        exit_code = 0 if all_passed else first_fail_code
        return all_passed, exit_code, _trim_output("\n\n".join(excerpts))

    # ------------------------------------------------------------------ rubric synthesis

    def _resolve_rubric(
        self,
        *,
        run_id: str,
        raw_input: str,
        state: Mapping[str, Any],
        evidence: EvidenceBrief | None,
        constraints: Sequence[str],
        scope_paths: Sequence[str],
        node_name: str,
        skill_refs: Sequence[str],
        max_steps: int,
    ) -> ArenaRubric:
        existing = state.get("arena_rubric")
        if existing is not None:
            rubric = (
                existing
                if isinstance(existing, ArenaRubric)
                else ArenaRubric.model_validate(existing)
            )
            rel_scope = _normalize_write_scope(rubric.write_scope, self.workspace_root)
            return (
                rubric.model_copy(update={"write_scope": rel_scope})
                if rel_scope and rel_scope != rubric.write_scope
                else rubric
            )

        toolkit = WorkspaceToolkit(
            self.workspace_root, read_only=True, env_root=self.workspace_root
        )
        inventory = format_workspace_inventory(toolkit, scope_paths=scope_paths)
        evidence_block = (
            json.dumps(evidence.model_dump(mode="json"), ensure_ascii=False, indent=2)
            if evidence is not None
            else "(no prior EvidenceBrief)"
        )
        constraints_block = (
            "\nConstraints:\n- " + "\n- ".join(constraints) if constraints else ""
        )
        system_prompt = _compose_system_prompt(
            "You are Athena (Arena Rubric Architect). Produce ONLY valid ArenaRubric JSON "
            "with 3 to 6 distinct weighted criteria, a tight repo-relative write_scope "
            "(NEVER absolute paths, e.g. ['src/app.py']), and 1 to 3 concrete executable "
            "verification_plan commands to judge competing candidate implementations. "
            "Every verification_plan command MUST ONLY invoke an allowlisted test/check "
            "runner directly (`pytest`, `python -m unittest`, `python -m pytest`, `ruff check`, "
            "`mypy`, `go test`, `cargo test`, `npm test`, `pnpm test`) with NO shell "
            "metacharacters (`;`, `&&`, `|`, `>`), NO `git`, NO `py_compile`, and NO inline "
            "code flags (`-c`, `-e`, `--eval`).",
            skill_refs,
        )
        user_prompt = (
            f"Goal:\n{raw_input}\n\n{inventory}{constraints_block}\n\n"
            f"EvidenceBrief:\n{evidence_block}\n\n"
            f"Return JSON matching schema:\n"
            f"{json.dumps(ArenaRubric.model_json_schema(), ensure_ascii=False)}"
        )
        rubric, _ = self._run_structured(
            run_id=run_id,
            node_name=node_name,
            role="architect",
            actor="athena:arena_rubric",
            workspace_path=self.workspace_root,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_cls=ArenaRubric,
            max_steps=max_steps,
            trace_name="rubric",
        )
        updates: dict[str, Any] = {}
        rel_scope = _normalize_write_scope(rubric.write_scope, self.workspace_root)
        if rel_scope and rel_scope != rubric.write_scope:
            updates["write_scope"] = rel_scope
        filtered_plan = _filter_allowed_verification_plan(
            rubric.verification_plan,
            workspace_root=self.workspace_root,
        )
        if filtered_plan and filtered_plan != rubric.verification_plan:
            updates["verification_plan"] = filtered_plan
        if updates:
            rubric = rubric.model_copy(update=updates)
        return rubric

    # ------------------------------------------------------------------ candidate execution

    def _execute_candidate(
        self,
        *,
        run_id: str,
        node_name: str,
        candidate_id: str,
        candidate_role: str,
        strategy_hint: str,
        handle: SandboxHandle,
        baseline: SandboxBaseline,
        raw_input: str,
        evidence: EvidenceBrief | None,
        rubric: ArenaRubric,
        skill_refs: Sequence[str],
        strict_scope: bool,
        max_steps: int,
        verify_timeout_s: int,
    ) -> tuple[ArenaCandidateResult, str]:
        evidence_block = (
            json.dumps(evidence.model_dump(mode="json"), ensure_ascii=False, indent=2)
            if evidence is not None
            else "(none)"
        )
        rubric_block = json.dumps(rubric.model_dump(mode="json"), ensure_ascii=False, indent=2)
        system_prompt = _compose_system_prompt(
            f"You are Hephaestus (Arena Candidate {candidate_id}). Implement a complete, "
            "self-contained solution inside your isolated git worktree strictly within "
            "write_scope. Do not edit files outside write_scope or leave temporary artifacts.",
            skill_refs,
        )
        user_prompt = (
            f"Goal:\n{raw_input}\n\n"
            f"Candidate ID: {candidate_id}\n"
            f"Isolated Worktree Root (use this directory for all file paths): "
            f"{handle.workspace_path}\n"
            f"Architectural Strategy Focus: {strategy_hint}\n\n"
            f"EvidenceBrief:\n{evidence_block}\n\n"
            f"ArenaRubric (your solution will be scored on these criteria):\n{rubric_block}\n\n"
            f"Authorized write_scope (relative to worktree root): "
            f"{json.dumps(rubric.write_scope)}\n"
            f"Forbidden globs: {json.dumps(rubric.forbidden_globs)}\n"
            f"Verification commands: {json.dumps(rubric.verification_plan)}"
        )

        driver_spec = resolve_driver_spec(
            self.execution_drivers.resolve_driver_for_role(candidate_role),
            provider=self.provider,
        )
        diff_text = ""
        try:
            outcome, driver_spec = self._run_worker_turn(
                run_id=run_id,
                node_name=node_name,
                role=candidate_role,
                actor=f"hephaestus:{candidate_id}",
                workspace_path=handle.workspace_path,
                write_scope=rubric.write_scope,
                forbidden_globs=rubric.forbidden_globs,
                strict_scope=strict_scope,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_steps=max_steps,
                trace_name=f"candidate_{candidate_id}",
            )
            if _is_cancelled_stop_reason(outcome.stop_reason):
                result = ArenaCandidateResult(
                    candidate_id=candidate_id,
                    role=candidate_role,
                    driver_spec=driver_spec,
                    strategy_hint=strategy_hint,
                    changed_paths=[],
                    diff_summary="Candidate execution cancelled.",
                    diff_bytes=0,
                    diff_gate_passed=False,
                    diff_gate_error="",
                    verification_passed=False,
                    verification_exit_code=None,
                    verification_excerpt="",
                    status="cancelled",
                    worktree_status="active",
                    error=f"Candidate turn cancelled (stop_reason={outcome.stop_reason!r})",
                )
            elif not outcome.completed:
                result = ArenaCandidateResult(
                    candidate_id=candidate_id,
                    role=candidate_role,
                    driver_spec=driver_spec,
                    strategy_hint=strategy_hint,
                    changed_paths=[],
                    diff_summary="Candidate execution did not complete.",
                    diff_bytes=0,
                    diff_gate_passed=False,
                    diff_gate_error="",
                    verification_passed=False,
                    verification_exit_code=None,
                    verification_excerpt="",
                    status="failed",
                    worktree_status="active",
                    error=f"Candidate turn incomplete (stop_reason={outcome.stop_reason!r})",
                )
            else:
                clean_ephemeral_artifacts(handle, baseline, write_scope=rubric.write_scope)
                gate_result = run_diff_gate(
                    handle,
                    baseline,
                    allowed_write_scope=rubric.write_scope,
                    forbidden_globs=rubric.forbidden_globs,
                    strict_scope=strict_scope,
                )
                diff_text = capture_sandbox_diff(handle, baseline)
                changed_paths = list(gate_result.changed_paths)
                if gate_result.rejected:
                    result = ArenaCandidateResult(
                        candidate_id=candidate_id,
                        role=candidate_role,
                        driver_spec=driver_spec,
                        strategy_hint=strategy_hint,
                        changed_paths=changed_paths,
                        diff_summary=_summarize_diff(changed_paths, diff_text),
                        diff_bytes=len(diff_text.encode("utf-8")),
                        diff_gate_passed=False,
                        diff_gate_error=gate_result.rejection_message(),
                        verification_passed=False,
                        verification_exit_code=None,
                        verification_excerpt="",
                        status="rejected",
                        worktree_status="active",
                        error=gate_result.rejection_message(),
                    )
                else:
                    verified, exit_code, excerpt = self._verify_candidate_sandbox(
                        handle=handle,
                        commands=rubric.verification_plan,
                        timeout_s=verify_timeout_s,
                        node_name=node_name,
                    )
                    clean_ephemeral_artifacts(handle, baseline, write_scope=rubric.write_scope)
                    diff_text = capture_sandbox_diff(handle, baseline)
                    changed_paths = collect_sandbox_changes(handle, baseline)
                    result = ArenaCandidateResult(
                        candidate_id=candidate_id,
                        role=candidate_role,
                        driver_spec=driver_spec,
                        strategy_hint=strategy_hint,
                        changed_paths=changed_paths,
                        diff_summary=_summarize_diff(changed_paths, diff_text),
                        diff_bytes=len(diff_text.encode("utf-8")),
                        diff_gate_passed=True,
                        diff_gate_error="",
                        verification_passed=verified,
                        verification_exit_code=exit_code,
                        verification_excerpt=excerpt,
                        status="completed",
                        worktree_status="active",
                        error="",
                    )
        except CraftPause:
            raise
        except (
            TurnCancelledError,
            concurrent.futures.CancelledError,
            asyncio.CancelledError,
        ) as exc:
            result = ArenaCandidateResult(
                candidate_id=candidate_id,
                role=candidate_role,
                driver_spec=driver_spec,
                strategy_hint=strategy_hint,
                changed_paths=[],
                diff_summary="Candidate execution cancelled.",
                diff_bytes=0,
                diff_gate_passed=False,
                diff_gate_error="",
                verification_passed=False,
                verification_exit_code=None,
                verification_excerpt="",
                status="cancelled",
                worktree_status="active",
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - isolate one candidate failure from peers
            result = ArenaCandidateResult(
                candidate_id=candidate_id,
                role=candidate_role,
                driver_spec=driver_spec,
                strategy_hint=strategy_hint,
                changed_paths=[],
                diff_summary="Candidate execution failed.",
                diff_bytes=0,
                diff_gate_passed=False,
                diff_gate_error="",
                verification_passed=False,
                verification_exit_code=None,
                verification_excerpt="",
                status="failed",
                worktree_status="active",
                error=f"{type(exc).__name__}: {exc}",
            )

        arena_dir = self._arena_dir(run_id)
        (arena_dir / f"{candidate_id}.patch").write_text(diff_text, encoding="utf-8")
        (arena_dir / f"{candidate_id}.json").write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._append_event(
            run_id=run_id,
            event_type="arena.candidate_completed",
            actor=f"hephaestus:{candidate_id}",
            payload={
                "candidate_id": result.candidate_id,
                "role": result.role,
                "driver_spec": result.driver_spec,
                "status": result.status,
                "diff_gate_passed": result.diff_gate_passed,
                "verification_passed": result.verification_passed,
                "verification_exit_code": result.verification_exit_code,
                "changed_paths": result.changed_paths,
                "diff_bytes": result.diff_bytes,
                "error": result.error,
            },
        )
        return result, diff_text

    # ------------------------------------------------------------------ judge & graft

    def _run_judge(
        self,
        *,
        run_id: str,
        node_name: str,
        raw_input: str,
        rubric: ArenaRubric,
        candidates: Sequence[ArenaCandidateResult],
        candidate_diffs: Mapping[str, str],
        eligible_ids: Sequence[str],
        skill_refs: Sequence[str],
        max_steps: int,
        max_graft_items: int,
    ) -> ArenaVerdict:
        criteria_ids = [item.criterion_id for item in rubric.criteria]
        candidate_ids = [item.candidate_id for item in candidates]
        dossier_parts: list[str] = []
        for cand in candidates:
            patch = _trim_output(
                candidate_diffs.get(cand.candidate_id, ""),
                _MAX_CANDIDATE_DIFF_CHARS,
            )
            dossier_parts.append(
                f"=== Candidate {cand.candidate_id} ===\n"
                f"role={cand.role} driver={cand.driver_spec} status={cand.status}\n"
                f"strategy={cand.strategy_hint}\n"
                f"diff_gate_passed={cand.diff_gate_passed} "
                f"verification_passed={cand.verification_passed} "
                f"verification_exit_code={cand.verification_exit_code}\n"
                f"changed_paths={json.dumps(cand.changed_paths)}\n"
                f"summary={cand.diff_summary}\n"
                f"verification_excerpt:\n{cand.verification_excerpt or '(none)'}\n"
                f"diff:\n```diff\n{patch}\n```"
            )
        system_prompt = _compose_system_prompt(
            "You are Athena (Independent Cold-Start Arena Judge). You have ZERO builder "
            "conversation history. Score every candidate objectively from 0.0 to 10.0 on "
            "every rubric criterion, select one eligible winner (must have passed DiffGate; "
            "strongly prefer verified passing candidates), and optionally specify up to "
            f"{max_graft_items} concrete graft_items from non-winning candidates to transplant "
            "onto the winner. Return ONLY valid ArenaVerdict JSON.",
            skill_refs,
        )
        rubric_json = json.dumps(rubric.model_dump(mode="json"), ensure_ascii=False, indent=2)
        user_prompt = (
            f"Goal:\n{raw_input}\n\n"
            f"ArenaRubric:\n{rubric_json}\n\n"
            f"Required candidate IDs in `scores`: {json.dumps(candidate_ids)}\n"
            f"Required criterion IDs per candidate: {json.dumps(criteria_ids)}\n"
            f"Eligible winner_id choices (completed, DiffGate passed, verification passed): "
            f"{json.dumps(list(eligible_ids))}\n\n"
            + "\n\n".join(dossier_parts)
            + "\n\nReturn JSON matching schema:\n"
            + json.dumps(ArenaVerdict.model_json_schema(), ensure_ascii=False)
        )
        verdict, judge_spec = self._run_structured(
            run_id=run_id,
            node_name=node_name,
            role="arena_judge",
            actor="athena:arena_judge",
            workspace_path=self.workspace_root,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_cls=ArenaVerdict,
            max_steps=max_steps,
            trace_name="judge",
        )
        verdict = verdict.model_copy(
            update={"judge_driver": judge_spec, "cold_start": True}
        )
        try:
            verdict.validate_against(
                rubric,
                candidate_ids,
                eligible_winner_ids=eligible_ids,
                max_graft_items=max_graft_items,
            )
        except ValueError as exc:
            raise CraftPause(
                node_name,
                f"Arena judge verdict failed deterministic validation: {exc}",
                route="blocked",
                pause_kind="blocked",
            ) from exc

        arena_dir = self._arena_dir(run_id)
        (arena_dir / "verdict.json").write_text(
            json.dumps(verdict.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._append_event(
            run_id=run_id,
            event_type="arena.judged",
            actor="athena:arena_judge",
            payload={
                "verdict_id": verdict.verdict_id,
                "winner_id": verdict.winner_id,
                "convergence_mode": verdict.convergence_mode,
                "scores": verdict.scores,
                "weighted_totals": verdict.weighted_totals(rubric),
                "graft_count": len(verdict.graft_items),
                "judge_driver": verdict.judge_driver,
                "cold_start": verdict.cold_start,
                "rationale": verdict.rationale,
            },
        )
        return verdict

    def _apply_graft_turn(
        self,
        *,
        run_id: str,
        node_name: str,
        raw_input: str,
        rubric: ArenaRubric,
        sketch: ArchitectSketch,
        verdict: ArenaVerdict,
        winner_handle: SandboxHandle,
        winner_baseline: SandboxBaseline,
        craft_handle: SandboxHandle,
        craft_baseline: SandboxBaseline,
        candidate_diffs: Mapping[str, str],
        created_handles: list[tuple[str, SandboxHandle]],
        resolved_base_sha: str | None,
        sandbox_source_root: Path,
        skill_refs: Sequence[str],
        strict_scope: bool,
        max_steps: int,
        verify_timeout_s: int,
        max_graft_items: int,
    ) -> tuple[str, list[str], list[ArenaGraftItem]]:
        if len(verdict.graft_items) > max_graft_items:
            raise CraftPause(
                node_name,
                f"ArenaVerdict proposed {len(verdict.graft_items)} graft_items, exceeding "
                f"max_graft_items={max_graft_items}",
                route="blocked",
                pause_kind="blocked",
            )
        active_grafts = list(verdict.graft_items)
        if not active_grafts:
            diff = capture_sandbox_diff(craft_handle, craft_baseline)
            changed = collect_sandbox_changes(craft_handle, craft_baseline)
            self._append_event(
                run_id=run_id,
                event_type="arena.grafted",
                actor="hephaestus:arena_graft",
                payload={
                    "skipped": True,
                    "winner_id": verdict.winner_id,
                    "graft_items": [],
                    "changed_paths": changed,
                    "diff_bytes": len(diff.encode("utf-8")),
                },
            )
            return diff, changed, []

        # Run graft in an isolated staging worktree initialized from the winner so any error,
        # cancellation, DiffGate violation, or verification regression leaves craft_handle intact.
        graft_label = f"{run_id}-arena-graft"
        planned_graft = _planned_sandbox_handle(
            self.workspace_root,
            label=graft_label,
            sandbox_root=self.sandbox_root,
            is_git_worktree=resolved_base_sha is not None,
        )
        created_handles.append(("graft", planned_graft))
        graft_idx = len(created_handles) - 1
        graft_handle = self._create_arena_sandbox(
            label=graft_label,
            resolved_base_sha=resolved_base_sha,
            sandbox_source_root=sandbox_source_root,
        )
        created_handles[graft_idx] = ("graft", graft_handle)
        graft_baseline = snapshot_sandbox_state(graft_handle)
        save_sandbox_baseline(graft_handle, graft_baseline)

        graft_promoter = CraftStage(
            store=self.store,
            workspace_root=graft_handle.workspace_path,
            provider=self.provider,
            artifacts_root=self.artifacts_root,
            sandbox_root=self.sandbox_root,
            execution_drivers=self.execution_drivers,
            driver_registry=self.driver_registry,
            model_provider=self.model_provider,
            on_driver_handle=self.on_driver_handle,
            command_runner=self._run_command,
            allow_unsandboxed_repro=self._allow_unsandboxed_repro,
        )
        graft_promoter._promote_sandbox_to_workspace(
            winner_handle,
            winner_baseline,
            sketch,
            node_name=node_name,
        )

        winner_diff = capture_sandbox_diff(graft_handle, graft_baseline)
        source_blocks: list[str] = []
        seen_sources: set[str] = set()
        for item in active_grafts:
            if item.source_candidate_id in seen_sources:
                continue
            seen_sources.add(item.source_candidate_id)
            src_patch = _trim_output(
                candidate_diffs.get(item.source_candidate_id, ""),
                _MAX_GRAFT_DIFF_CHARS,
            )
            source_blocks.append(
                f"--- Source candidate {item.source_candidate_id} diff ---\n"
                f"```diff\n{src_patch}\n```"
            )
        graft_plan = json.dumps(
            [item.model_dump(mode="json") for item in active_grafts],
            ensure_ascii=False,
            indent=2,
        )
        system_prompt = _compose_system_prompt(
            "You are Hephaestus (Arena Graft Surgeon). The winning candidate's diff is "
            "already checked out in this sandbox. Apply ONLY the specified graft_items "
            "from non-winning candidates cleanly without regressing the winner or "
            "touching files outside write_scope.",
            skill_refs,
        )
        user_prompt = (
            f"Goal:\n{raw_input}\n\n"
            f"Sandbox Root (use this directory for all file paths): "
            f"{graft_handle.workspace_path}\n"
            f"Winning candidate: {verdict.winner_id}\n"
            f"Authorized write_scope: {json.dumps(rubric.write_scope)}\n"
            f"Forbidden globs: {json.dumps(rubric.forbidden_globs)}\n\n"
            f"Graft items to apply:\n{graft_plan}\n\n"
            f"Current winning diff:\n```diff\n"
            f"{_trim_output(winner_diff, _MAX_GRAFT_DIFF_CHARS)}\n```\n\n"
            + "\n\n".join(source_blocks)
        )
        try:
            graft_outcome, _ = self._run_worker_turn(
                run_id=run_id,
                node_name=node_name,
                role="builder",
                actor="hephaestus:arena_graft",
                workspace_path=graft_handle.workspace_path,
                write_scope=rubric.write_scope,
                forbidden_globs=rubric.forbidden_globs,
                strict_scope=strict_scope,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_steps=max_steps,
                trace_name="graft",
            )
        except (
            TurnCancelledError,
            concurrent.futures.CancelledError,
            asyncio.CancelledError,
        ) as exc:
            raise CraftPause(
                node_name,
                f"Arena graft turn cancelled; winner craft sandbox preserved untouched: {exc}",
                route="blocked",
                pause_kind="blocked",
            ) from exc
        except CraftPause:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve winner sandbox on graft failure
            raise CraftPause(
                node_name,
                f"Arena graft turn failed ({type(exc).__name__}: {exc}); "
                "winner craft sandbox preserved untouched.",
                route="blocked",
                pause_kind="blocked",
            ) from exc

        if _is_cancelled_stop_reason(graft_outcome.stop_reason) or not graft_outcome.completed:
            raise CraftPause(
                node_name,
                f"Arena graft turn did not complete (stop_reason={graft_outcome.stop_reason!r}); "
                "winner craft sandbox preserved untouched.",
                route="blocked",
                pause_kind="blocked",
            )

        clean_ephemeral_artifacts(graft_handle, graft_baseline, write_scope=rubric.write_scope)
        gate_result = run_diff_gate(
            graft_handle,
            graft_baseline,
            allowed_write_scope=rubric.write_scope,
            forbidden_globs=rubric.forbidden_globs,
            strict_scope=strict_scope,
        )
        if gate_result.rejected:
            raise CraftPause(
                node_name,
                gate_result.rejection_message(),
                route="blocked",
                pause_kind="blocked",
            )
        verified, exit_code, excerpt = self._verify_candidate_sandbox(
            handle=graft_handle,
            commands=rubric.verification_plan,
            timeout_s=verify_timeout_s,
            node_name=node_name,
        )
        if not verified or exit_code != 0:
            raise CraftPause(
                node_name,
                f"Arena graft failed verification (exit={exit_code}); "
                f"winner craft sandbox preserved untouched.\n{excerpt}",
                route="blocked",
                pause_kind="blocked",
            )
        clean_ephemeral_artifacts(graft_handle, graft_baseline, write_scope=rubric.write_scope)

        # Persist grafted.patch BEFORE mutating craft_handle so an artifact write failure
        # never leaves uncommitted graft changes in the winning craft sandbox.
        grafted_diff = capture_sandbox_diff(graft_handle, graft_baseline)
        patch_path = self._arena_dir(run_id) / "grafted.patch"
        try:
            patch_path.write_text(grafted_diff, encoding="utf-8")
        except OSError as exc:
            raise CraftPause(
                node_name,
                f"Failed to persist grafted.patch ({exc}); "
                "winner craft sandbox preserved untouched.",
                route="blocked",
                pause_kind="blocked",
            ) from exc

        # Graft passed DiffGate + verification + artifact write: transactionally commit.
        try:
            changed = _mirror_sandbox_changes(
                source_handle=graft_handle,
                target_handle=craft_handle,
                base_baseline=craft_baseline,
                allowed_write_scope=rubric.write_scope,
                forbidden_globs=rubric.forbidden_globs,
                node_name=node_name,
            )
        except BaseException:
            with contextlib.suppress(OSError):
                patch_path.unlink(missing_ok=True)
            raise
        self._append_event(
            run_id=run_id,
            event_type="arena.grafted",
            actor="hephaestus:arena_graft",
            payload={
                "skipped": False,
                "winner_id": verdict.winner_id,
                "graft_items": [item.model_dump(mode="json") for item in active_grafts],
                "changed_paths": changed,
                "diff_bytes": len(grafted_diff.encode("utf-8")),
            },
        )
        return grafted_diff, changed, active_grafts

    # ------------------------------------------------------------------ main composite primitive

    def _create_arena_sandbox(
        self,
        *,
        label: str,
        resolved_base_sha: str | None,
        sandbox_source_root: Path,
    ) -> SandboxHandle:
        """Create a sandbox locked to ``resolved_base_sha`` or a frozen snapshot directory."""
        if resolved_base_sha is not None:
            return create_sandbox(
                self.workspace_root,
                label=label,
                reuse=False,
                sandbox_root=self.sandbox_root,
                base_ref=resolved_base_sha,
            )
        effective_sandbox_root = (
            self.sandbox_root
            if self.sandbox_root is not None
            else (self.workspace_root / ".olympus" / "sandboxes")
        )
        copied = create_sandbox(
            sandbox_source_root,
            label=label,
            reuse=False,
            sandbox_root=effective_sandbox_root,
            base_ref=None,
            force_copy=True,
        )
        return SandboxHandle(
            repo_root=self.workspace_root,
            workspace_path=copied.workspace_path,
            base_commit=None,
            is_git_worktree=False,
        )

    def arena_race(
        self,
        *,
        run_id: str,
        raw_input: str,
        state: Mapping[str, Any],
        constraints: Sequence[str] = (),
        scope_paths: Sequence[str] = (),
        node_name: str = "arena_race",
        skill_refs: Sequence[str] = (),
        num_candidates: int = 2,
        candidate_roles: Sequence[str] | None = None,
        base_ref: str | None = None,
        strict_scope: bool = True,
        max_steps: int = 24,
        verify_timeout_s: int = 120,
        max_graft_items: int = 2,
    ) -> dict[str, Any]:
        """Execute a full multi-worktree Arena competition, rubric judge, and graft."""
        if num_candidates < 2:
            raise ValueError(f"arena_race requires num_candidates >= 2, got {num_candidates}")

        resolved_roles = (
            [r.strip() for r in candidate_roles if r.strip()]
            if candidate_roles
            else [f"arena_runner_{idx}" for idx in range(1, num_candidates + 1)]
        )
        if len(resolved_roles) < num_candidates:
            for idx in range(len(resolved_roles) + 1, num_candidates + 1):
                resolved_roles.append(f"arena_runner_{idx}")
        resolved_roles = resolved_roles[:num_candidates]

        effective_base_ref = (
            base_ref
            if base_ref is not None
            else (str(state["base_ref"]) if isinstance(state.get("base_ref"), str) else None)
        )
        try:
            resolved_base_sha = _resolve_git_base_commit(self.workspace_root, effective_base_ref)
        except ValueError as exc:
            raise CraftPause(
                node_name,
                str(exc),
                route="blocked",
                pause_kind="blocked",
            ) from exc

        evidence = self._craft._hydrate_model(run_id, state, "evidence_brief", EvidenceBrief)
        rubric = self._resolve_rubric(
            run_id=run_id,
            raw_input=raw_input,
            state=state,
            evidence=evidence,
            constraints=constraints,
            scope_paths=scope_paths,
            node_name=node_name,
            skill_refs=skill_refs,
            max_steps=max_steps,
        )
        arena_dir = self._arena_dir(run_id)
        (arena_dir / "rubric.json").write_text(
            json.dumps(rubric.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._append_event(
            run_id=run_id,
            event_type="arena.started",
            actor="athena:arena",
            payload={
                "rubric_id": rubric.rubric_id,
                "criteria_ids": [item.criterion_id for item in rubric.criteria],
                "num_candidates": num_candidates,
                "candidate_roles": resolved_roles,
                "base_ref": effective_base_ref or "HEAD",
                "resolved_base_commit": resolved_base_sha,
                "write_scope": list(rubric.write_scope),
            },
        )

        # Spawn all temporary candidate worktrees sequentially up front to avoid git lock races.
        candidate_slots: list[tuple[str, str, str, SandboxHandle, SandboxBaseline]] = []
        created_handles: list[tuple[str, SandboxHandle]] = []
        cleanup_statuses: dict[str, Literal["active", "cleaned", "cleanup_failed"]] = {}
        cleanup_errors: list[str] = []
        candidate_results: dict[str, ArenaCandidateResult] = {}
        candidate_diffs: dict[str, str] = {}
        sandbox_source_root = self.workspace_root

        try:
            # When no commit SHA exists (unborn-HEAD git repo or non-git workspace), freeze one
            # immutable baseline snapshot directory so every candidate and craft sandbox sees the
            # exact same initial tree even if the host repo creates a commit mid-race.
            if resolved_base_sha is None:
                base_label = f"{run_id}-arena-base"
                planned_base = _planned_sandbox_handle(
                    self.workspace_root,
                    label=base_label,
                    sandbox_root=self.sandbox_root,
                    is_git_worktree=False,
                )
                created_handles.append(("base_snapshot", planned_base))
                base_dir = planned_base.workspace_path
                if base_dir.exists():
                    shutil.rmtree(base_dir)
                base_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(
                    self.workspace_root,
                    base_dir,
                    symlinks=True,
                    ignore=shutil.ignore_patterns(".git", ".venv", ".olympus", "__pycache__"),
                )
                sandbox_source_root = base_dir

            for idx in range(1, num_candidates + 1):
                cid = f"cand_{idx}"
                role = resolved_roles[idx - 1]
                hint = _strategy_hint_for_index(idx)
                label = f"{run_id}-arena-{cid}"
                # Register the target path BEFORE creation so mid-creation failures still clean up.
                planned = _planned_sandbox_handle(
                    self.workspace_root,
                    label=label,
                    sandbox_root=self.sandbox_root,
                    is_git_worktree=resolved_base_sha is not None,
                )
                created_handles.append((cid, planned))
                handle_idx = len(created_handles) - 1
                handle = self._create_arena_sandbox(
                    label=label,
                    resolved_base_sha=resolved_base_sha,
                    sandbox_source_root=sandbox_source_root,
                )
                created_handles[handle_idx] = (cid, handle)
                baseline = snapshot_sandbox_state(handle)
                save_sandbox_baseline(handle, baseline)
                candidate_slots.append((cid, role, hint, handle, baseline))

            # Execute candidate builds concurrently in their isolated worktrees.
            with ThreadPoolExecutor(max_workers=num_candidates) as pool:
                future_to_cid = {
                    pool.submit(
                        self._execute_candidate,
                        run_id=run_id,
                        node_name=node_name,
                        candidate_id=cid,
                        candidate_role=role,
                        strategy_hint=hint,
                        handle=handle,
                        baseline=baseline,
                        raw_input=raw_input,
                        evidence=evidence,
                        rubric=rubric,
                        skill_refs=skill_refs,
                        strict_scope=strict_scope,
                        max_steps=max_steps,
                        verify_timeout_s=verify_timeout_s,
                    ): cid
                    for cid, role, hint, handle, baseline in candidate_slots
                }
                for future in as_completed(future_to_cid):
                    cid = future_to_cid[future]
                    res, diff_text = future.result()
                    candidate_results[cid] = res
                    candidate_diffs[cid] = diff_text

            ordered_results = [candidate_results[cid] for cid, *_ in candidate_slots]
            eligible_ids = [
                cand.candidate_id
                for cand in ordered_results
                if cand.status == "completed"
                and cand.diff_gate_passed
                and cand.verification_passed
                and cand.verification_exit_code == 0
            ]
            if not eligible_ids:
                details = "; ".join(
                    f"{cand.candidate_id}={cand.status} "
                    f"(diff_gate={cand.diff_gate_passed}, "
                    f"verified={cand.verification_passed}, "
                    f"exit={cand.verification_exit_code}, "
                    f"{cand.error or 'no details'})"
                    for cand in ordered_results
                )
                raise CraftPause(
                    node_name,
                    f"No eligible arena candidates passed DiffGate and verification: {details}",
                    route="blocked",
                    pause_kind="blocked",
                )

            verdict = self._run_judge(
                run_id=run_id,
                node_name=node_name,
                raw_input=raw_input,
                rubric=rubric,
                candidates=ordered_results,
                candidate_diffs=candidate_diffs,
                eligible_ids=eligible_ids,
                skill_refs=skill_refs,
                max_steps=max_steps,
                max_graft_items=max_graft_items,
            )

            winner_slot = next(
                slot for slot in candidate_slots if slot[0] == verdict.winner_id
            )
            _w_cid, _w_role, _w_hint, winner_handle, winner_baseline = winner_slot

            # Synthesize canonical ArchitectSketch for atomic promotion + downstream hygiene/verify.
            sketch = ArchitectSketch(
                candidate_shapes=[
                    CandidateShape(
                        name=cand.candidate_id,
                        data_flow=cand.strategy_hint or cand.role,
                        tradeoffs=cand.diff_summary or cand.status,
                    )
                    for cand in ordered_results
                ],
                chosen_shape=verdict.winner_id,
                rationale=verdict.rationale,
                state_owners={"arena_winner": verdict.winner_id},
                write_scope=list(rubric.write_scope),
                forbidden_globs=list(rubric.forbidden_globs),
                blast_radius=f"Arena winner {verdict.winner_id} ({verdict.convergence_mode})",
                verification_plan=list(rubric.verification_plan),
            )

            # Create a clean craft sandbox locked at the exact same baseline; if any init step
            # (baseline snapshot/save or winner promotion) fails, immediately reclaim -craft.
            craft_handle: SandboxHandle | None = None
            try:
                craft_handle = self._create_arena_sandbox(
                    label=f"{run_id}-craft",
                    resolved_base_sha=resolved_base_sha,
                    sandbox_source_root=sandbox_source_root,
                )
                craft_baseline = snapshot_sandbox_state(craft_handle)
                save_sandbox_baseline(craft_handle, craft_baseline)

                # Reuse CraftStage's transactional atomic workspace promotion into craft sandbox.
                sandbox_promoter = CraftStage(
                    store=self.store,
                    workspace_root=craft_handle.workspace_path,
                    provider=self.provider,
                    artifacts_root=self.artifacts_root,
                    sandbox_root=self.sandbox_root,
                    execution_drivers=self.execution_drivers,
                    driver_registry=self.driver_registry,
                    model_provider=self.model_provider,
                    on_driver_handle=self.on_driver_handle,
                    command_runner=self._run_command,
                    allow_unsandboxed_repro=self._allow_unsandboxed_repro,
                )
                sandbox_promoter._promote_sandbox_to_workspace(
                    winner_handle,
                    winner_baseline,
                    sketch,
                    node_name=node_name,
                )
            except BaseException:
                if craft_handle is not None:
                    with contextlib.suppress(Exception):
                        remove_sandbox(craft_handle, strict=False)
                raise

            final_diff, final_changed, _applied_grafts = self._apply_graft_turn(
                run_id=run_id,
                node_name=node_name,
                raw_input=raw_input,
                rubric=rubric,
                sketch=sketch,
                verdict=verdict,
                winner_handle=winner_handle,
                winner_baseline=winner_baseline,
                craft_handle=craft_handle,
                craft_baseline=craft_baseline,
                candidate_diffs=candidate_diffs,
                created_handles=created_handles,
                resolved_base_sha=resolved_base_sha,
                sandbox_source_root=sandbox_source_root,
                skill_refs=skill_refs,
                strict_scope=strict_scope,
                max_steps=max_steps,
                verify_timeout_s=verify_timeout_s,
                max_graft_items=max_graft_items,
            )
        finally:
            # Guaranteed strict sequential cleanup of every temporary candidate & graft worktree.
            for cid, handle in created_handles:
                try:
                    remove_sandbox(handle, strict=True)
                    cleanup_statuses[cid] = "cleaned"
                except Exception as exc:  # noqa: BLE001 - continue cleaning remaining worktrees
                    cleanup_statuses[cid] = "cleanup_failed"
                    cleanup_errors.append(f"{cid} ({handle.workspace_path}): {exc}")
                    self._append_event(
                        run_id=run_id,
                        event_type="arena.cleanup_failed",
                        actor=f"arena:{cid}",
                        payload={
                            "candidate_id": cid,
                            "workspace_path": str(handle.workspace_path),
                            "error": str(exc),
                        },
                    )

        finalized_candidates = [
            candidate_results[cid].model_copy(
                update={"worktree_status": cleanup_statuses.get(cid, "cleanup_failed")}
            )
            for cid, *_ in candidate_slots
        ]
        dumped_rubric = rubric.model_dump(mode="json")
        dumped_candidates = [cand.model_dump(mode="json") for cand in finalized_candidates]
        dumped_verdict = verdict.model_dump(mode="json")
        dumped_sketch = sketch.model_dump(mode="json")

        (arena_dir / "candidates.json").write_text(
            json.dumps(dumped_candidates, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if cleanup_errors:
            raise CraftPause(
                node_name,
                "Failed to clean up temporary arena worktree(s): " + "; ".join(cleanup_errors),
                route="blocked",
                pause_kind="blocked",
                state_updates={"arena_candidates": dumped_candidates},
            )
        (arena_dir / "summary.json").write_text(
            json.dumps(
                {
                    "rubric": dumped_rubric,
                    "candidates": dumped_candidates,
                    "verdict": dumped_verdict,
                    "changed_paths": final_changed,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (self._craft._craft_dir(run_id) / "diff.patch").write_text(final_diff, encoding="utf-8")
        self._craft._save_persisted(
            run_id,
            {
                "architect_sketch": dumped_sketch,
                "arena_rubric": dumped_rubric,
                "arena_candidates": dumped_candidates,
                "arena_verdict": dumped_verdict,
            },
        )
        return {
            "architect_sketch": dumped_sketch,
            "arena_rubric": dumped_rubric,
            "arena_candidates": dumped_candidates,
            "arena_verdict": dumped_verdict,
            "craft_diff": final_diff,
            "craft_changed_paths": final_changed,
            "route": "pass",
        }


__all__ = [
    "ArenaStage",
]
