"""Resident run supervisor: dotenv loading, provider resolution, and graph workflows."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import Any

import olympus.harness.activity as activity
from olympus.core.contracts import RunStatus
from olympus.core.store import ConcurrencyConflictError, RunStore
from olympus.graph import (
    GraphPaused,
    WorkflowDefinition,
    build_graph,
    describe_workflow,
    list_builtin_workflows,
    load_workflow,
    run_graph,
)
from olympus.graph.loader import WorkflowRuntime
from olympus.graph.progress import resolve_resume_node
from olympus.harness.driver import (
    ActiveTurnHandle,
    DriverRegistry,
    ExecutionDriversConfig,
    default_driver_factories,
)
from olympus.providers.auth import resolve_env_mapping
from olympus.providers.base import LLMProvider, MockLLMProvider
from olympus.providers.factory import build_provider
from olympus.providers.registry import ProviderRegistry, ProviderSpec

ProviderBuilder = Callable[..., LLMProvider]

_ENGINE_EVENT_TO_STORE: dict[str, str] = {
    "graph_started": "graph.started",
    "node_started": "graph.node_started",
    "node_finished": "graph.node_finished",
    "node_error": "graph.node_error",
    "node_paused": "graph.node_paused",
    "routed": "graph.routed",
    "graph_paused": "graph.paused",
    "graph_finished": "graph.finished",
}

_PROVIDER_KWARGS: tuple[str, ...] = ("provider", "model", "base_url", "api_key")


class RunSupervisor:
    """Owns the thread pool, per-run locks, and the workflow <-> stage bridge."""

    def __init__(
        self,
        *,
        store: RunStore,
        workspace_root: Path | str,
        artifacts_root: Path | None = None,
        provider_builder: ProviderBuilder | None = None,
        driver_registry: DriverRegistry | None = None,
        max_workers: int = 4,
    ) -> None:
        self.store = store
        self.workspace_root = Path(workspace_root).resolve()
        self.artifacts_root = (
            artifacts_root
            if artifacts_root is not None
            else self.workspace_root / ".olympus" / "runs"
        )
        self.provider_builder = provider_builder
        self.driver_registry = driver_registry or DriverRegistry()
        self.live_activity = activity.LiveActivity()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._future_by_run: dict[str, Future[dict[str, Any]]] = {}
        self._run_locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._managed_runs: set[str] = set()
        self._active_tokens: dict[str, int] = {}
        self._next_token: int = 0
        self._pending_resume: dict[str, int] = {}
        self._providers: dict[str, LLMProvider] = {}
        self._active_turns: dict[str, list[ActiveTurnHandle]] = {}
        self.dotenv_loaded = self._load_dotenv()

    # ------------------------------------------------------------------ environment

    def _load_dotenv(self) -> bool:
        """Load ``<workspace>/.env`` when present. Values are never logged or returned."""
        env_path = self.workspace_root / ".env"
        if not env_path.is_file():
            return False
        try:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path=env_path, override=False)
        except Exception:  # noqa: BLE001 - a broken .env must never block the console
            return False
        return True

    def _load_discovered_workflows(self) -> list[WorkflowDefinition]:
        definitions: list[WorkflowDefinition] = []
        for name in list_builtin_workflows():
            definitions.append(load_workflow(name))
        discovered_dir = self.workspace_root / ".olympus" / "workflows"
        if discovered_dir.is_dir():
            known = {d.name for d in definitions}
            for path in sorted(discovered_dir.glob("*.y*ml")):
                try:
                    definition = load_workflow(path)
                except Exception:  # noqa: BLE001 - skip unreadable community workflows
                    continue
                if definition.name not in known:
                    definitions.append(definition)
        return definitions

    def list_workflows(self) -> list[dict[str, Any]]:
        """Summaries of every builtin/discovered YAML workflow plus graph topology."""
        return [describe_workflow(definition) for definition in self._load_discovered_workflows()]

    # -------------------------------------------------------------------- providers

    def _configured(self, spec: ProviderSpec) -> bool:
        if any(ep.api_key and ep.api_key.strip() for ep in spec.endpoints):
            return True
        names: list[str] = []
        if spec.api_key_env:
            names.append(spec.api_key_env)
        if spec.api_keys_env:
            names.append(spec.api_keys_env)
        for endpoint in spec.endpoints:
            if endpoint.api_key_env:
                names.append(endpoint.api_key_env)
            if endpoint.api_keys_env:
                names.append(endpoint.api_keys_env)
        if not names:
            return False
        env = resolve_env_mapping(
            provider_env_file=Path(spec.env_file) if spec.env_file else None,
            cwd=self.workspace_root,
        )
        return any(env.get(name, "").strip() for name in names)

    def list_providers_status(self) -> list[dict[str, Any]]:
        """Registry view with configuration booleans only — secrets never leave the process."""
        registry = ProviderRegistry.load(
            project_providers_path=self.workspace_root / ".olympus" / "providers.json"
        )
        statuses: list[dict[str, Any]] = []
        for provider_id in registry.list_ids():
            spec = registry.get(provider_id)
            if spec is None:
                continue
            statuses.append(
                {
                    "id": spec.id,
                    "dialect": spec.dialect,
                    "default_model": spec.default_model,
                    "configured": self._configured(spec),
                    "env_vars": [
                        name
                        for name in (spec.api_key_env, spec.api_keys_env)
                        if name is not None
                    ],
                }
            )
        return statuses

    def _builder(self, config: dict[str, Any]) -> LLMProvider:
        requested = str(config.get("provider") or "").strip().lower()
        if requested == "mock" and self.provider_builder is None:
            return (
                MockLLMProvider(model_name=str(config["model"]))
                if config.get("model")
                else MockLLMProvider()
            )
        kwargs: dict[str, Any] = {
            key: value
            for key, value in config.items()
            if key in _PROVIDER_KWARGS and value not in (None, "")
        }
        kwargs.setdefault("provider", "auto")
        builder = self.provider_builder or build_provider
        if builder is build_provider:
            kwargs.setdefault(
                "project_providers_path",
                self.workspace_root / ".olympus" / "providers.json",
            )
        return builder(**kwargs)

    def resolve_provider(self, run_id: str) -> LLMProvider:
        """Resolve the run's provider, falling back to Mock only for mock/unconfigured runs."""
        with self._guard:
            cached = self._providers.get(run_id)
        if cached is not None:
            return cached
        snap = self.store.get_run(run_id)
        raw_cfg = snap.get("provider_config")
        config = dict(raw_cfg or {})
        provider: LLMProvider
        if self.provider_builder is not None:
            # An injected builder is authoritative (tests, custom routers, replays).
            kwargs = {
                key: value
                for key, value in config.items()
                if key in _PROVIDER_KWARGS and value not in (None, "")
            }
            try:
                provider = self.provider_builder(**kwargs)
            except Exception:  # noqa: BLE001 - never let provider wiring break a run
                provider = MockLLMProvider()
        elif raw_cfg is None or str(config.get("provider") or "").strip().lower() == "mock":
            provider = MockLLMProvider()
        else:
            try:
                provider = self._builder(config)
            except Exception:  # noqa: BLE001 - unconfigured provider => deterministic mock
                provider = MockLLMProvider()
        with self._guard:
            self._providers[run_id] = provider
        return provider

    # ----------------------------------------------------------------------- runs

    def _lock_for(self, run_id: str) -> threading.Lock:
        with self._guard:
            return self._run_locks.setdefault(run_id, threading.Lock())

    def is_supervised(self, run_id: str) -> bool:
        with self._guard:
            if run_id in self._managed_runs:
                return True
        events = self.store.list_events(run_id, after_seq=0)
        supervised = any(ev.stage == "graph" for ev in events)
        if supervised:
            with self._guard:
                self._managed_runs.add(run_id)
        return supervised

    def _observer(self, run_id: str) -> Callable[[str, dict[str, Any]], None]:
        def _observe(event: str, payload: dict[str, Any]) -> None:
            event_type = _ENGINE_EVENT_TO_STORE.get(event)
            if event_type is None:
                return
            self.store.append_event(
                run_id=run_id,
                stage="graph",
                event_type=event_type,
                actor="supervisor",
                payload=dict(payload),
            )

        return _observe

    def resolve_workspace(self, run_id: str) -> Path:
        """Return the run's dedicated workspace_root when configured, else supervisor root."""
        snap = self.store.get_run(run_id)
        raw = snap.get("workspace_root")
        if isinstance(raw, str) and raw.strip():
            candidate = Path(raw).expanduser().resolve()
            if candidate.exists():
                return candidate
        return self.workspace_root

    def _register_turn(self, run_id: str, handle: ActiveTurnHandle) -> None:
        with self._guard:
            self._active_turns.setdefault(run_id, []).append(handle)
        if self.store.get_run(run_id)["status"] in (RunStatus.PAUSED, RunStatus.CANCELLED):
            handle.cancel(reason="run_interrupted")

    def cancel_active_turns(self, run_id: str, *, reason: str) -> None:
        with self._guard:
            handles = tuple(self._active_turns.get(run_id, ()))
        for handle in handles:
            handle.cancel(reason=reason)

    def _advance(
        self,
        run_id: str,
        *,
        raw_input: str,
        start_from: str | None,
        auto_approve: bool,
        recompute_entry: bool = False,
    ) -> dict[str, Any]:
        with self._lock_for(run_id):
            snap = self.store.get_run(run_id)
            ws = self.resolve_workspace(run_id)
            definition = load_workflow(snap["workflow_name"])
            effective_start = start_from
            if recompute_entry:
                if snap["status"] in (RunStatus.PAUSED, RunStatus.CANCELLED):
                    return {
                        "run_id": run_id,
                        "workflow": definition.name,
                        "topology": definition.topology,
                        "stop_after": definition.stop_after,
                        "status": snap["status"].value,
                        "state_revision": snap["state_revision"],
                        "paused": None,
                    }
                resolved = self.resume_node(definition, snap)
                if snap["status"] == RunStatus.COMPLETED and resolved in (
                    None,
                    "execute",
                    "gate_plan",
                ):
                    return {
                        "run_id": run_id,
                        "workflow": definition.name,
                        "topology": definition.topology,
                        "stop_after": definition.stop_after,
                        "status": snap["status"].value,
                        "state_revision": snap["state_revision"],
                        "paused": None,
                        "complete": True,
                    }
                if resolved is None:
                    with suppress(ConcurrencyConflictError):
                        self.store.set_run_status(
                            run_id,
                            RunStatus.COMPLETED,
                            expected_revision=int(snap["state_revision"]),
                            actor="supervisor",
                            stage="graph",
                            detail={
                                "workflow": definition.name,
                                "stop_after": definition.stop_after,
                            },
                        )
                    final = self.store.get_run(run_id)
                    return {
                        "run_id": run_id,
                        "workflow": definition.name,
                        "topology": definition.topology,
                        "stop_after": definition.stop_after,
                        "status": final["status"].value,
                        "state_revision": final["state_revision"],
                        "paused": None,
                        "complete": final["status"] == RunStatus.COMPLETED,
                    }
                effective_start = resolved

            graph = build_graph(definition)
            recorded_drivers = snap.get("execution_drivers")
            driver_config = (
                ExecutionDriversConfig.model_validate(recorded_drivers)
                if recorded_drivers is not None
                else ExecutionDriversConfig.resolve()
            )
            provider_config = snap.get("provider_config") or {"provider": "mock"}
            active_provider = self.resolve_provider(run_id)
            runtime = WorkflowRuntime(
                store=self.store,
                workspace_root=ws,
                artifacts_root=self.artifacts_root,
                provider=active_provider,
                auto_approve=auto_approve,
                topology=definition.topology,
                execution_drivers=driver_config,
                driver_registry=self.driver_registry.with_factories(
                    default_driver_factories(
                        store=self.store,
                        run_id=run_id,
                        provider=active_provider,
                        execution_drivers=driver_config,
                    )
                ),
                model_provider=lambda model: self._builder({**provider_config, "model": model}),
                on_driver_handle=lambda handle: self._register_turn(run_id, handle),
                live_activity=self.live_activity,
            )
            state: dict[str, Any] = {
                "run_id": run_id,
                "raw_input": raw_input,
                "constraints": self._recorded_constraints(run_id),
                "topology": definition.topology,
                "route": None,
                "workspace_id": snap.get("workspace_id"),
                "workspace_mode": snap.get("workspace_mode") or "brownfield",
                "task_type": snap.get("task_type") or "feature",
                "base_ref": snap.get("base_ref"),
                "scope_paths": list(snap.get("scope_paths") or []),
                "_runtime": runtime,
            }
            if snap["spec"] is not None:
                state["spec"] = snap["spec"].model_dump(mode="json")
            if snap["design"] is not None:
                state["design"] = snap["design"].model_dump(mode="json")
            if snap["plan"] is not None:
                state["plan"] = snap["plan"].model_dump(mode="json")

            try:
                result = run_graph(
                    graph,
                    state,
                    observer=self._observer(run_id),
                    max_steps=runtime.max_steps * 2,
                    start_from=effective_start,
                )
            finally:
                with self._guard:
                    self._active_turns.pop(run_id, None)
            paused_node = result.get("_paused_node")
            while True:
                latest = self.store.get_run(run_id)
                if paused_node:
                    if latest["status"] in (RunStatus.PAUSED, RunStatus.CANCELLED):
                        break
                    try:
                        self.store.set_run_status(
                            run_id,
                            RunStatus.AWAITING_HUMAN,
                            expected_revision=int(latest["state_revision"]),
                            actor="supervisor",
                            stage="graph",
                            detail={
                                "paused_node": paused_node,
                                "pause_reason": result.get("_pause_reason"),
                            },
                        )
                        break
                    except ConcurrencyConflictError:
                        continue
                if (
                    not (
                        latest["status"] == RunStatus.AWAITING_HUMAN
                        and isinstance(result.get("execution"), dict)
                    )
                    and latest["status"]
                    not in (
                        RunStatus.PAUSED,
                        RunStatus.CANCELLED,
                        RunStatus.COMPLETED,
                    )
                ):
                    try:
                        self.store.set_run_status(
                            run_id,
                            RunStatus.COMPLETED,
                            expected_revision=int(latest["state_revision"]),
                            actor="supervisor",
                            stage="graph",
                            detail={
                                "workflow": definition.name,
                                "stop_after": definition.stop_after,
                            },
                        )
                        break
                    except ConcurrencyConflictError:
                        continue
                break
            final = self.store.get_run(run_id)
            return {
                "run_id": run_id,
                "workflow": definition.name,
                "topology": definition.topology,
                "stop_after": definition.stop_after,
                "status": final["status"].value,
                "state_revision": final["state_revision"],
                "paused": (
                    {"node": paused_node, "reason": result.get("_pause_reason")}
                    if paused_node
                    else None
                ),
            }

    def _release_token(self, run_id: str, token: int, *, clear_pending: bool = False) -> None:
        """Release active ownership only if ``token`` still owns ``run_id``."""
        with self._guard:
            if self._active_tokens.get(run_id) == token:
                self._active_tokens.pop(run_id, None)
                if clear_pending:
                    self._pending_resume.pop(run_id, None)

    def _drain_advance_loop(
        self,
        run_id: str,
        *,
        token: int,
        raw_input: str,
        start_from: str | None,
        auto_approve: bool,
        recompute_entry: bool,
    ) -> dict[str, Any]:
        """Run ``_advance`` and drain any coalesced mid-flight resumes for ``token``."""
        current_start = start_from
        current_recompute = recompute_entry
        while True:
            start_rev = int(self.store.get_run(run_id)["state_revision"])
            result = self._advance(
                run_id,
                raw_input=raw_input,
                start_from=current_start,
                auto_approve=auto_approve,
                recompute_entry=current_recompute,
            )
            with self._guard:
                if self._active_tokens.get(run_id) != token:
                    return result
                if run_id not in self._pending_resume:
                    self._active_tokens.pop(run_id, None)
                    return result
                pending_rev = self._pending_resume.pop(run_id)
            snap = self.store.get_run(run_id)
            definition = load_workflow(snap["workflow_name"])
            entry = self.resume_node(definition, snap)
            entry_node = definition.node(entry) if entry is not None else None
            paused_node = (result.get("paused") or {}).get("node")
            has_new_state = (
                pending_rev > start_rev
                or int(snap["state_revision"]) > int(result["state_revision"])
            )
            already_at_same_gate = (
                entry_node is not None
                and entry_node.type == "human_gate"
                and paused_node == entry
                and int(snap["state_revision"]) <= int(result["state_revision"])
            )
            still_completed = snap["status"] == RunStatus.COMPLETED and entry in (
                None,
                "execute",
                "gate_plan",
            )
            if (
                entry is None
                or snap["status"] in (RunStatus.PAUSED, RunStatus.CANCELLED)
                or still_completed
                or not has_new_state
                or already_at_same_gate
            ):
                with self._guard:
                    if self._active_tokens.get(run_id) == token and run_id in self._pending_resume:
                        continue
                    if self._active_tokens.get(run_id) == token:
                        self._active_tokens.pop(run_id, None)
                return result
            current_start = entry
            current_recompute = True

    def _launch(
        self,
        run_id: str,
        *,
        raw_input: str,
        start_from: str | None,
        auto_approve: bool,
        sync: bool,
        recompute_entry: bool = False,
    ) -> dict[str, Any]:
        snap = self.store.get_run(run_id)
        if sync:
            active_future: Future[dict[str, Any]] | None = None
            sync_future: Future[dict[str, Any]] | None = None
            token = 0
            with self._guard:
                if run_id in self._active_tokens:
                    prev_rev = self._pending_resume.get(run_id, -1)
                    self._pending_resume[run_id] = max(prev_rev, int(snap["state_revision"]))
                    active_future = self._future_by_run.get(run_id)
                else:
                    self._next_token += 1
                    token = self._next_token
                    self._active_tokens[run_id] = token
                    sync_future = Future()
                    self._future_by_run[run_id] = sync_future
            if active_future is not None:
                return active_future.result()
            assert sync_future is not None
            try:
                result = self._drain_advance_loop(
                    run_id,
                    token=token,
                    raw_input=raw_input,
                    start_from=start_from,
                    auto_approve=auto_approve,
                    recompute_entry=recompute_entry,
                )
                sync_future.set_result(result)
                return result
            except BaseException as exc:
                self._release_token(run_id, token, clear_pending=True)
                if not sync_future.done():
                    sync_future.set_exception(exc)
                raise
            finally:
                self._release_token(run_id, token)

        # Async: check ownership, mint token, submit future, and publish _future_by_run
        # atomically under _guard so neither active tokens nor futures can race or invert.
        coalesced = False
        with self._guard:
            if run_id in self._active_tokens:
                prev_rev = self._pending_resume.get(run_id, -1)
                self._pending_resume[run_id] = max(prev_rev, int(snap["state_revision"]))
                coalesced = True
            else:
                self._next_token += 1
                token = self._next_token
                self._active_tokens[run_id] = token
                try:
                    future = self._executor.submit(
                        self._advance_guarded,
                        run_id,
                        token=token,
                        raw_input=raw_input,
                        start_from=start_from,
                        auto_approve=auto_approve,
                        recompute_entry=recompute_entry,
                    )
                except Exception:
                    if self._active_tokens.get(run_id) == token:
                        self._active_tokens.pop(run_id, None)
                    raise
                self._future_by_run[run_id] = future
        if coalesced:
            return {
                "run_id": run_id,
                "workflow": snap["workflow_name"],
                "status": "queued",
                "coalesced": True,
                "state_revision": snap["state_revision"],
                "paused": None,
            }
        return {
            "run_id": run_id,
            "workflow": snap["workflow_name"],
            "status": "queued",
            "state_revision": snap["state_revision"],
            "paused": None,
        }

    def _advance_guarded(
        self,
        run_id: str,
        *,
        token: int,
        raw_input: str,
        start_from: str | None,
        auto_approve: bool,
        recompute_entry: bool = False,
    ) -> dict[str, Any]:
        try:
            return self._drain_advance_loop(
                run_id,
                token=token,
                raw_input=raw_input,
                start_from=start_from,
                auto_approve=auto_approve,
                recompute_entry=recompute_entry,
            )
        except Exception as exc:  # noqa: BLE001 - background failures must be auditable
            self.store.append_event(
                run_id=run_id,
                stage="graph",
                event_type="graph.failed",
                actor="supervisor",
                payload={"error": str(exc)},
            )
            while True:
                latest = self.store.get_run(run_id)
                if latest["status"] in (RunStatus.PAUSED, RunStatus.CANCELLED):
                    break
                try:
                    self.store.set_run_status(
                        run_id,
                        RunStatus.FAILED,
                        expected_revision=int(latest["state_revision"]),
                        actor="supervisor",
                        stage="graph",
                        detail={"error": str(exc)},
                    )
                    break
                except ConcurrencyConflictError:
                    continue
            final = self.store.get_run(run_id)
            self._release_token(run_id, token, clear_pending=True)
            return {
                "run_id": run_id,
                "status": final["status"].value,
                "error": str(exc),
                "paused": None,
            }
        finally:
            self._release_token(run_id, token)

    def start_run(
        self,
        *,
        title: str,
        raw_input: str,
        workflow_name: str = "product_blueprint",
        provider_config: dict[str, Any] | None = None,
        worker_driver: str | None = None,
        reviewer_driver: str | None = None,
        monitor_model: str | None = None,
        auto_approve: bool = False,
        sync: bool = False,
        constraints: list[str] | None = None,
        workspace_root: Path | str | None = None,
        workspace_id: str | None = None,
        workspace_mode: str = "brownfield",
        task_type: str = "feature",
        base_ref: str | None = None,
        scope_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a run and drive its workflow until a human gate or completion."""
        definition = load_workflow(workflow_name)
        config = dict(provider_config) if provider_config else {"provider": "auto"}
        driver_config = ExecutionDriversConfig.resolve(
            worker_driver=worker_driver,
            reviewer_driver=reviewer_driver,
            monitor_model=monitor_model,
        )
        normalized_scope = [str(item) for item in (scope_paths or [])]
        run_id = self.store.create_run(
            title=title,
            workflow_name=definition.name,
            provider_config=config,
            execution_drivers=driver_config.model_dump(mode="json"),
            workspace_root=workspace_root,
            workspace_id=workspace_id,
            workspace_mode=workspace_mode,
            task_type=task_type,
            base_ref=base_ref,
            scope_paths=normalized_scope,
            actor="supervisor",
        )
        cleaned_constraints = [str(c).strip() for c in (constraints or []) if str(c).strip()]
        self.store.append_event(
            run_id=run_id,
            stage="graph",
            event_type="graph.input_recorded",
            actor="supervisor",
            payload={
                "raw_input": raw_input,
                "constraints": cleaned_constraints,
                "workspace_id": workspace_id,
                "workspace_mode": workspace_mode,
                "task_type": task_type,
                "base_ref": base_ref,
                "scope_paths": normalized_scope,
            },
        )
        with self._guard:
            self._managed_runs.add(run_id)
        return self._launch(
            run_id,
            raw_input=raw_input,
            start_from=None,
            auto_approve=auto_approve,
            sync=sync,
        )

    def resume_node(
        self,
        definition: WorkflowDefinition,
        snap: dict[str, Any],
    ) -> str | None:
        """Pick the graph entry point implied by the run's confirmed artifacts."""
        return resolve_resume_node(definition, snap, store=self.store)

    def _recorded_raw_input(self, run_id: str, fallback: str) -> str:
        for ev in reversed(self.store.list_events(run_id, after_seq=0)):
            if ev.event_type == "graph.input_recorded":
                raw = ev.payload.get("raw_input")
                if isinstance(raw, str) and raw.strip():
                    return raw
        return fallback

    def _recorded_constraints(self, run_id: str) -> list[str]:
        for ev in reversed(self.store.list_events(run_id, after_seq=0)):
            if ev.event_type == "graph.input_recorded":
                items = ev.payload.get("constraints")
                if isinstance(items, list):
                    return [str(item).strip() for item in items if str(item).strip()]
        return []

    def resume_run(self, run_id: str, *, sync: bool = False) -> dict[str, Any]:
        """Advance a paused/awaiting run to its next gate or completion."""
        snap = self.store.get_run(run_id)
        if not sync:
            with self._guard:
                if run_id in self._active_tokens:
                    # An advance is already in flight; coalesce instead of submitting a
                    # duplicate thread with a stale start_from. The running pass will
                    # recompute a fresh entry node after it finishes.
                    prev_rev = self._pending_resume.get(run_id, -1)
                    self._pending_resume[run_id] = max(prev_rev, int(snap["state_revision"]))
                    return {
                        "run_id": run_id,
                        "workflow": snap["workflow_name"],
                        "status": "queued",
                        "coalesced": True,
                        "state_revision": snap["state_revision"],
                        "paused": None,
                    }
        definition = load_workflow(snap["workflow_name"])
        entry = self.resume_node(definition, snap)
        if entry is None:
            latest = snap
            if snap["status"] not in (
                RunStatus.PAUSED,
                RunStatus.CANCELLED,
                RunStatus.COMPLETED,
            ):
                with suppress(ConcurrencyConflictError):
                    self.store.set_run_status(
                        run_id,
                        RunStatus.COMPLETED,
                        expected_revision=int(snap["state_revision"]),
                        actor="supervisor",
                        stage="graph",
                        detail={
                            "workflow": definition.name,
                            "stop_after": definition.stop_after,
                        },
                    )
                latest = self.store.get_run(run_id)
            return {
                "run_id": run_id,
                "workflow": definition.name,
                "status": latest["status"].value,
                "state_revision": latest["state_revision"],
                "paused": None,
                "complete": latest["status"] == RunStatus.COMPLETED,
            }
        entry_node = definition.node(entry)
        effective_sync = sync or (entry_node is not None and entry_node.type == "human_gate")
        return self._launch(
            run_id,
            raw_input=self._recorded_raw_input(run_id, str(snap["title"])),
            start_from=entry,
            auto_approve=False,
            sync=effective_sync,
            recompute_entry=True,
        )

    def wait(self, run_id: str, *, timeout: float | None = None) -> dict[str, Any] | None:
        """Await a background run future (used by CLI/tests, never by the HTTP handler)."""
        with self._guard:
            future = self._future_by_run.get(run_id)
        if future is None:
            return None
        return future.result(timeout=timeout)

    def recover_orphaned_runs(self) -> list[str]:
        """Mark interrupted supervised runs as FAILED on startup so UI can retry cleanly."""
        recovered: list[str] = []
        active_statuses = ("created", "aligning", "designing", "planning", "running")
        for row in self.store.list_runs(limit=100):
            if row.get("status") not in active_statuses:
                continue
            run_id = str(row["run_id"])
            if not self.is_supervised(run_id):
                continue
            with self._guard:
                future = self._future_by_run.get(run_id)
            if future is not None and not future.done():
                continue
            msg = "服务重启导致阶段执行中断，可从当前断点直接重试"
            self.store.append_event(
                run_id=run_id,
                stage="graph",
                event_type="graph.failed",
                actor="supervisor",
                payload={"error": msg},
            )
            self.store.set_run_status(
                run_id,
                RunStatus.FAILED,
                actor="supervisor",
                stage="graph",
                detail={"reason": "orphan_recovered", "error": msg},
            )
            recovered.append(run_id)
        return recovered

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


__all__ = ["ProviderBuilder", "RunSupervisor", "GraphPaused"]
