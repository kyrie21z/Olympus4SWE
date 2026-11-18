"""Resident run supervisor: dotenv loading, provider resolution, and graph workflows."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from olympus.core.contracts import ApprovalTarget, DesignSpec, GoalSpec, RunStatus, TaskPlan
from olympus.core.store import RunStore
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
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._future_by_run: dict[str, Future[dict[str, Any]]] = {}
        self._run_locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._managed_runs: set[str] = set()
        self._providers: dict[str, LLMProvider] = {}
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
        if requested == "mock":
            return MockLLMProvider()
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

    def _advance(
        self,
        run_id: str,
        *,
        raw_input: str,
        start_from: str | None,
        auto_approve: bool,
    ) -> dict[str, Any]:
        with self._lock_for(run_id):
            snap = self.store.get_run(run_id)
            definition = load_workflow(snap["workflow_name"])
            graph = build_graph(definition)
            runtime = WorkflowRuntime(
                store=self.store,
                workspace_root=self.workspace_root,
                artifacts_root=self.artifacts_root,
                provider=self.resolve_provider(run_id),
                auto_approve=auto_approve,
                topology=definition.topology,
            )
            state: dict[str, Any] = {
                "run_id": run_id,
                "raw_input": raw_input,
                "constraints": [],
                "topology": definition.topology,
                "route": None,
                "_runtime": runtime,
            }
            if snap["spec"] is not None:
                state["spec"] = snap["spec"].model_dump(mode="json")
            if snap["design"] is not None:
                state["design"] = snap["design"].model_dump(mode="json")
            if snap["plan"] is not None:
                state["plan"] = snap["plan"].model_dump(mode="json")

            result = run_graph(
                graph,
                state,
                observer=self._observer(run_id),
                max_steps=runtime.max_steps * 2,
                start_from=start_from,
            )
            paused_node = result.get("_paused_node")
            latest = self.store.get_run(run_id)
            if paused_node:
                if latest["status"] not in (RunStatus.PAUSED, RunStatus.CANCELLED):
                    self.store.set_run_status(
                        run_id,
                        RunStatus.AWAITING_HUMAN,
                        actor="supervisor",
                        stage="graph",
                        detail={
                            "paused_node": paused_node,
                            "pause_reason": result.get("_pause_reason"),
                        },
                    )
            elif latest["status"] not in (
                RunStatus.PAUSED,
                RunStatus.CANCELLED,
                RunStatus.COMPLETED,
            ):
                self.store.set_run_status(
                    run_id,
                    RunStatus.COMPLETED,
                    actor="supervisor",
                    stage="graph",
                    detail={"workflow": definition.name, "stop_after": definition.stop_after},
                )
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

    def _launch(
        self,
        run_id: str,
        *,
        raw_input: str,
        start_from: str | None,
        auto_approve: bool,
        sync: bool,
    ) -> dict[str, Any]:
        if sync:
            return self._advance(
                run_id,
                raw_input=raw_input,
                start_from=start_from,
                auto_approve=auto_approve,
            )
        future = self._executor.submit(
            self._advance_guarded,
            run_id,
            raw_input=raw_input,
            start_from=start_from,
            auto_approve=auto_approve,
        )
        with self._guard:
            self._future_by_run[run_id] = future
        snap = self.store.get_run(run_id)
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
        raw_input: str,
        start_from: str | None,
        auto_approve: bool,
    ) -> dict[str, Any]:
        try:
            return self._advance(
                run_id,
                raw_input=raw_input,
                start_from=start_from,
                auto_approve=auto_approve,
            )
        except Exception as exc:  # noqa: BLE001 - background failures must be auditable
            self.store.append_event(
                run_id=run_id,
                stage="graph",
                event_type="graph.failed",
                actor="supervisor",
                payload={"error": str(exc)},
            )
            self.store.set_run_status(
                run_id,
                RunStatus.FAILED,
                actor="supervisor",
                stage="graph",
                detail={"error": str(exc)},
            )
            return {
                "run_id": run_id,
                "status": RunStatus.FAILED.value,
                "error": str(exc),
                "paused": None,
            }

    def start_run(
        self,
        *,
        title: str,
        raw_input: str,
        workflow_name: str = "product_blueprint",
        provider_config: dict[str, Any] | None = None,
        auto_approve: bool = False,
        sync: bool = False,
    ) -> dict[str, Any]:
        """Create a run and drive its workflow until a human gate or completion."""
        definition = load_workflow(workflow_name)
        config = dict(provider_config) if provider_config else {"provider": "auto"}
        run_id = self.store.create_run(
            title=title,
            workflow_name=definition.name,
            provider_config=config,
            actor="supervisor",
        )
        self.store.append_event(
            run_id=run_id,
            stage="graph",
            event_type="graph.input_recorded",
            actor="supervisor",
            payload={"raw_input": raw_input},
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

    def _is_approved(self, run_id: str, target: ApprovalTarget, target_id: str, hash_: str) -> bool:
        return self.store.is_approved(
            run_id=run_id,
            target=target,
            target_id=target_id,
            target_hash=hash_,
        )

    def resume_node(
        self,
        definition: WorkflowDefinition,
        snap: dict[str, Any],
    ) -> str | None:
        """Pick the graph entry point implied by the run's confirmed artifacts."""
        names = {node.name for node in definition.nodes}
        run_id = str(snap["run_id"])
        spec: GoalSpec | None = snap["spec"]
        design: DesignSpec | None = snap["design"]
        plan: TaskPlan | None = snap["plan"]

        if plan is not None and self._is_approved(
            run_id, ApprovalTarget.TASK_PLAN, plan.plan_id, plan.content_hash
        ):
            if definition.stop_after == "execute" and "execute" in names:
                return "execute"
            # Re-enter the terminal gate so the graph records its final routed/END events.
            if "gate_plan" in names:
                return "gate_plan"
            return None

        if design is not None:
            if self._is_approved(
                run_id, ApprovalTarget.DESIGN_SPEC, design.design_id, design.content_hash
            ):
                return "plan" if "plan" in names else None
            for candidate in ("gate_design", "validate_design", "design"):
                if candidate in names:
                    return candidate
            return None

        if spec is not None:
            if self._is_approved(
                run_id, ApprovalTarget.GOAL_SPEC, spec.spec_id, spec.content_hash
            ):
                for candidate in ("design", "plan"):
                    if candidate in names:
                        return candidate
                return None
            if "gate_spec" in names:
                return "gate_spec"
            return "align" if "align" in names else None

        return "intake" if "intake" in names else ("align" if "align" in names else None)

    def _recorded_raw_input(self, run_id: str, fallback: str) -> str:
        for ev in reversed(self.store.list_events(run_id, after_seq=0)):
            if ev.event_type == "graph.input_recorded":
                raw = ev.payload.get("raw_input")
                if isinstance(raw, str) and raw.strip():
                    return raw
        return fallback

    def resume_run(self, run_id: str, *, sync: bool = False) -> dict[str, Any]:
        """Advance a paused/awaiting run to its next gate or completion."""
        snap = self.store.get_run(run_id)
        definition = load_workflow(snap["workflow_name"])
        entry = self.resume_node(definition, snap)
        if entry is None:
            latest = snap
            if snap["status"] not in (
                RunStatus.PAUSED,
                RunStatus.CANCELLED,
                RunStatus.COMPLETED,
            ):
                self.store.set_run_status(
                    run_id,
                    RunStatus.COMPLETED,
                    actor="supervisor",
                    stage="graph",
                    detail={"workflow": definition.name, "stop_after": definition.stop_after},
                )
                latest = self.store.get_run(run_id)
            return {
                "run_id": run_id,
                "workflow": definition.name,
                "status": latest["status"].value,
                "state_revision": latest["state_revision"],
                "paused": None,
                "complete": True,
            }
        return self._launch(
            run_id,
            raw_input=self._recorded_raw_input(run_id, str(snap["title"])),
            start_from=entry,
            auto_approve=False,
            sync=sync,
        )

    def wait(self, run_id: str, *, timeout: float | None = None) -> dict[str, Any] | None:
        """Await a background run future (used by CLI/tests, never by the HTTP handler)."""
        with self._guard:
            future = self._future_by_run.get(run_id)
        if future is None:
            return None
        return future.result(timeout=timeout)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


__all__ = ["ProviderBuilder", "RunSupervisor", "GraphPaused"]
