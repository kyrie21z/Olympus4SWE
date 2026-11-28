"""Synchronous headless session driver over the existing workflow graph runtime.

The runner owns no harness-specific concepts: it composes ``RunStore``,
``RunControl``, declarative workflows, the wave graph, and stage handlers exactly
as the Web supervisor does, but exposes one blocking call per CLI invocation.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import subprocess
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from olympus.core.contracts import (
    ApprovalTarget,
    BudgetLedger,
    DesignSpec,
    GoalSpec,
    RunMode,
    RunStatus,
    TaskNode,
    TaskPlan,
    TaskStatus,
)
from olympus.core.control import (
    RunControl,
    is_design_bound_to_spec,
    is_plan_bound_to_upstream,
)
from olympus.core.store import ConcurrencyConflictError, NotFoundError, RunStore
from olympus.graph.engine import END, run_graph
from olympus.graph.loader import (
    WorkflowDefinition,
    WorkflowRuntime,
    build_graph,
    load_workflow,
)
from olympus.graph.progress import GATE_SPECS, resolve_resume_node
from olympus.harness.driver import (
    DriverRegistry,
    ExecutionDriversConfig,
    default_driver_factories,
)
from olympus.providers.base import LLMProvider, MockLLMProvider
from olympus.providers.factory import build_provider
from olympus.session.contracts import (
    ActorTokenUsage,
    GatePolicyName,
    PauseKind,
    SessionEnvelope,
    SessionStatus,
    StepMode,
    TokenUsageSummary,
)
from olympus.session.policy import InteractiveMinimalPolicy
from olympus.session.router import parse_reply
from olympus.stages.align import (
    AlignStage,
    _normalize_clarification_text,
    has_pending_resynthesis,
    unconsumed_clarification_notes,
)

ProviderBuilder = Callable[..., LLMProvider]

_GATE_BY_NODE: dict[str, str] = {spec["node"]: gate for gate, spec in GATE_SPECS.items()}
_EXECUTE_TYPES = frozenset(
    {
        "execute_task_dag",
        "build_direct",
        "hygiene_pass",
        "interrogate_review",
        "verify_surface",
        "hillclimb_loop",
    }
)
_PLAN_ONLY_STOP = "plan"
_MAX_IN_PLACE_RECOVERIES = 2
_MAX_PLAN_RECOVERIES = 3
_EXPLICIT_REVISION_RE = re.compile(
    r"^\s*(?:revise|reopen)\s*[:：]\s*(.+)$", re.IGNORECASE | re.DOTALL
)


def _parse_explicit_revision_directive(text: str) -> str | None:
    """Return the revision body when the human explicitly issued ``revise:`` or ``reopen:``."""
    match = _EXPLICIT_REVISION_RE.match(text)
    if match is None:
        return None
    body = match.group(1).strip()
    return body or None


_PURE_RETRY_TOKENS = frozenset(
    {
        "retry",
        "try again",
        "continue",
        "proceed",
        "resume",
        "go ahead",
        "重试",
        "继续",
        "恢复",
        "再试一次",
    }
)


def _is_pure_retry_message(text: str) -> bool:
    normalized = _normalize_clarification_text(text).strip(".!。！？")
    return normalized in _PURE_RETRY_TOKENS


def _workflow_strict_canonical(definition: WorkflowDefinition) -> bool:
    return definition.align_strict_canonical()


@dataclass(frozen=True)
class SessionConfig:
    """Everything one session invocation needs; providers stay injectable."""

    workspace_root: Path
    state_dir: Path | None = None
    workflow: str = "full_sdlc_tdd"
    workflow_explicit: bool = True
    gate_policy: GatePolicyName | None = None
    step_mode: StepMode = StepMode.EXECUTE
    provider: str = "auto"
    model: str | None = None
    base_url: str | None = None
    stage_driver: str | None = None
    worker_driver: str | None = None
    reviewer_driver: str | None = None
    role_drivers: dict[str, str] = field(default_factory=dict)
    monitor_model: str | None = None
    pi_provider: str | None = None
    thinking_level: str | None = None
    context_window: int | None = None
    worker_max_steps: int | None = 40
    workspace_id: str | None = None
    workspace_mode: str = "brownfield"
    task_type: str = "feature"
    base_ref: str | None = None
    scope_paths: list[str] = field(default_factory=list)
    provider_builder: ProviderBuilder | None = None
    extra_provider_kwargs: dict[str, Any] = field(default_factory=dict)
    hillclimb_config: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _assert_state_outside_workspace(
    workspace: Path | str, state_dir: Path | str
) -> Path:
    root = Path(workspace).expanduser().resolve()
    candidate = Path(state_dir).expanduser().resolve()
    toplevel = _git_toplevel(root)
    targets = (
        ("--state-dir", candidate),
        ("--state-dir/sandboxes", (candidate / "sandboxes").resolve()),
        ("--state-dir/runs", (candidate / "runs").resolve()),
        ("--state-dir/olympus.db", (candidate / "olympus.db").resolve()),
    )
    for boundary in (root, toplevel):
        if boundary is None:
            continue
        scope = (
            "the target workspace"
            if boundary == root
            else f"the enclosing git repository {boundary} of the target workspace"
        )
        for label, resolved in targets:
            if resolved == boundary or resolved.is_relative_to(boundary):
                raise ValueError(
                    f"{label} {resolved} is inside {scope}; "
                    "headless session state must live outside the repository"
                )
    return candidate


def resolve_state_dir(workspace: Path | str, state_dir: Path | str | None) -> Path:
    """Return an external state directory, never nested inside the target repo."""
    root = Path(workspace).resolve()
    if state_dir is None:
        digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
        candidate = Path(tempfile.gettempdir()) / "olympus-session" / digest
    else:
        candidate = Path(state_dir).expanduser().resolve()
    candidate = _assert_state_outside_workspace(root, candidate)
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _git_toplevel(path: Path | str) -> Path | None:
    """Return the enclosing git work tree root for ``path``, when there is one."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    raw = proc.stdout.strip()
    if not raw:
        return None
    return Path(raw).resolve()


def _yaml_workflow_path(value: str | Path) -> Path | None:
    """Return the absolute path of a custom ``.yaml``/``.yml`` workflow argument."""
    candidate = Path(value).expanduser()
    if candidate.suffix.lower() in (".yaml", ".yml"):
        return candidate.resolve()
    return None


_ACTIONABLE_CHARS = re.compile(r"[0-9A-Za-z\u3400-\u9fff]")


def _looks_actionable(text: str) -> bool:
    """True when a completed-run reply carries a concrete follow-up request."""
    return len(_ACTIONABLE_CHARS.findall(text)) >= 3


def resolve_workspace_binding(
    *,
    requested: Path | str | None,
    state_dir: Path | str | None,
    run_id: str | None,
) -> Path | None:
    """Bind one CLI invocation to the repository that owns the run.

    ``start`` has no recorded run yet and just resolves ``--workspace``. A ``reply``
    must never resume inside a different repository: when ``--workspace`` is omitted
    the recorded root wins, and an explicit mismatch raises ``ValueError`` before any
    runner, database, or trace file is created.
    """
    requested_path = (
        Path(requested).expanduser().resolve() if requested is not None else None
    )
    if requested_path is not None and state_dir is not None:
        _assert_state_outside_workspace(requested_path, state_dir)
    if run_id is None:
        return requested_path
    recorded = _recorded_workspace(state_dir, run_id)
    if recorded is None:
        return requested_path
    if state_dir is not None:
        _assert_state_outside_workspace(recorded, state_dir)
    if requested_path is None:
        return recorded
    if requested_path != recorded:
        raise ValueError(
            f"--workspace {requested_path} does not match the run's recorded workspace "
            f"{recorded}; a reply must resume in the repository that started the run"
        )
    return recorded


def _recorded_workspace(state_dir: Path | str | None, run_id: str) -> Path | None:
    if state_dir is None:
        return None
    db_path = Path(state_dir).expanduser().resolve() / "olympus.db"
    if not db_path.is_file() or db_path.stat().st_size == 0:
        return None
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        row = conn.execute(
            "SELECT workspace_root FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        if conn is not None:
            conn.close()
    if row is None or not row[0]:
        return None
    return Path(str(row[0])).resolve()


def plan_only_definition(definition: WorkflowDefinition) -> WorkflowDefinition:
    """Drop execution nodes/edges so a plan-only run can never mutate the repo."""
    execute_nodes = {node.name for node in definition.nodes if node.type in _EXECUTE_TYPES}
    nodes = [node for node in definition.nodes if node.name not in execute_nodes]
    edges = []
    for edge in definition.edges:
        if edge.source in execute_nodes:
            continue
        target = END if edge.target in execute_nodes else edge.target
        edges.append(edge.model_copy(update={"target": target}))
    routers = []
    for router in definition.routers:
        if router.at in execute_nodes:
            continue
        rules = {
            key: (END if target in execute_nodes else target)
            for key, target in router.rules.items()
        }
        default = END if router.default in execute_nodes else router.default
        routers.append(router.model_copy(update={"rules": rules, "default": default}))
    return definition.model_copy(
        update={"nodes": nodes, "edges": edges, "routers": routers, "stop_after": _PLAN_ONLY_STOP}
    )


class SessionRunner:
    """Drive one ``olympus session start`` / ``reply`` invocation to an envelope."""

    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self.workspace_root = Path(config.workspace_root).resolve()
        self.state_dir = resolve_state_dir(self.workspace_root, config.state_dir)
        self.artifacts_root = self.state_dir / "runs"
        self.sandbox_root = self.state_dir / "sandboxes"
        self.store = RunStore(self.state_dir / "olympus.db")
        self.control = RunControl(self.store)

    # ------------------------------------------------------------------ public API

    def start(
        self,
        *,
        goal: str,
        title: str | None = None,
        constraints: list[str] | None = None,
        hillclimb_config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionEnvelope:
        """Create a run and drive it until a pause or completion."""
        clean_goal = goal.strip()
        if not clean_goal:
            return SessionEnvelope.failure(
                error="session start requires a non-empty goal",
                step_mode=self.config.step_mode,
            )
        try:
            definition = load_workflow(self.config.workflow)
        except Exception as exc:  # noqa: BLE001 - surfaced as a stable failed envelope
            return SessionEnvelope.failure(
                error=str(exc), step_mode=self.config.step_mode, workflow=self.config.workflow
            )
        workflow_path = _yaml_workflow_path(self.config.workflow)
        clean_constraints = [str(item).strip() for item in (constraints or []) if str(item).strip()]
        explicit_meta = dict(metadata or {})
        merged_metadata: dict[str, Any] = {
            **dict(self.config.metadata),
            **explicit_meta,
        }
        effective_hillclimb_cfg = (
            dict(hillclimb_config)
            if hillclimb_config is not None
            else (
                dict(explicit_meta["hillclimb_config"])
                if isinstance(explicit_meta.get("hillclimb_config"), dict)
                else (
                    dict(self.config.hillclimb_config)
                    if self.config.hillclimb_config is not None
                    else (
                        dict(self.config.metadata["hillclimb_config"])
                        if isinstance(self.config.metadata.get("hillclimb_config"), dict)
                        else None
                    )
                )
            )
        )
        if effective_hillclimb_cfg is not None:
            merged_metadata["hillclimb_config"] = effective_hillclimb_cfg
        drivers = self._drivers_config()
        provider_config = self._provider_config()
        run_id = self.store.create_run(
            title=title or clean_goal[:60],
            mode=RunMode.EXECUTION,
            workflow_name=definition.name,
            provider_config=provider_config,
            execution_drivers=drivers.model_dump(mode="json"),
            workspace_root=self.workspace_root,
            workspace_id=self.config.workspace_id,
            workspace_mode=self.config.workspace_mode,
            task_type=self.config.task_type,
            base_ref=self.config.base_ref,
            scope_paths=list(self.config.scope_paths),
            actor="session",
        )
        toplevel = _git_toplevel(self.workspace_root)
        started_payload: dict[str, Any] = {
            "raw_input": clean_goal,
            "constraints": clean_constraints,
            "workflow": definition.name,
            "workflow_path": str(workflow_path) if workflow_path is not None else None,
            "workflow_hash": definition.execution_hash(),
            "workspace_root": str(self.workspace_root),
            "git_toplevel": str(toplevel) if toplevel is not None else None,
            "step_mode": self.config.step_mode.value,
            "gate_policy": self._resolved_gate_policy(run_id).value,
        }
        if effective_hillclimb_cfg is not None:
            started_payload["hillclimb_config"] = effective_hillclimb_cfg
        if merged_metadata:
            started_payload["metadata"] = merged_metadata
        self.store.append_event(
            run_id=run_id,
            stage="session",
            event_type="session.started",
            actor="session",
            payload=started_payload,
        )
        return self._drive(
            run_id=run_id,
            raw_input=clean_goal,
            start_from=None,
            definition=definition,
        )

    def reply(self, *, run_id: str, message: str) -> SessionEnvelope:
        """Route one free-form reply and continue the same workflow."""
        text = message.strip()
        try:
            snap = self.store.get_run(run_id)
        except NotFoundError as exc:
            return SessionEnvelope.failure(
                error=str(exc), run_id=run_id, step_mode=self.config.step_mode
            )
        recorded = snap.get("workspace_root")
        if recorded:
            recorded_root = Path(str(recorded)).resolve()
            if recorded_root != self.workspace_root:
                return SessionEnvelope.failure(
                    error=(
                        f"--workspace {self.workspace_root} does not match the run's recorded "
                        f"workspace {recorded_root}; refusing to continue in a different "
                        "repository"
                    ),
                    run_id=run_id,
                    step_mode=self.config.step_mode,
                    workflow=snap["workflow_name"],
                    token_usage=self._token_usage(run_id),
                )
        try:
            definition = self._workflow_for_run(run_id, snap)
        except Exception as exc:  # noqa: BLE001 -- surface workflow load failure as SessionEnvelope
            return SessionEnvelope.failure(
                error=str(exc),
                run_id=run_id,
                step_mode=self.config.step_mode,
                workflow=snap["workflow_name"],
                token_usage=self._token_usage(run_id),
            )
        if snap["status"] == RunStatus.COMPLETED:
            return self._incremental_reopen(run_id, definition, snap, text)
        if not text:
            return self._current_pause_envelope(
                snap, message="Reply was empty; please answer the pending question."
            )

        pause_kind, gate, node = self._pause_state(run_id)
        explicit_revision = _parse_explicit_revision_directive(text)
        if explicit_revision is not None:
            return self._revise(run_id, definition, snap, explicit_revision)
        intent = parse_reply(
            provider=self._provider(run_id),
            message=text,
            snapshot=snap,
            pause_kind=pause_kind,
            gate=gate,
            completed=False,
        )
        if intent.action == "clarify":
            if snap.get("spec") is None:
                clarify_text = intent.feedback.strip() or text
                self.store.append_event(
                    run_id=run_id,
                    stage="craft",
                    event_type="craft.clarification_recorded",
                    actor="human",
                    payload={"note": clarify_text, "answers": dict(intent.answers)},
                )
                return self._continue(run_id, definition)
            try:
                self.control.clarify_spec(
                    run_id=run_id,
                    answers=intent.answers,
                    resolutions=intent.resolutions,
                    clarification_note=intent.feedback.strip(),
                    actor="human",
                )
            except (ValueError, ConcurrencyConflictError) as exc:
                return self._emit_pause(
                    run_id,
                    pause_kind or PauseKind.CLARIFY,
                    node=node,
                    gate=gate,
                    reason=str(exc),
                    message=str(exc),
                )
            return self._continue(run_id, definition)

        if intent.action == "approve":
            if pause_kind == PauseKind.PLAN_HANDOFF:
                if self.config.step_mode == StepMode.EXECUTE:
                    return self._continue(run_id, definition)
                return self._emit_pause(
                    run_id,
                    PauseKind.PLAN_HANDOFF,
                    node=node,
                    gate=gate,
                    reason="Plan already approved; re-run with --step-mode execute to implement.",
                )
            pending_gate = gate or self._gate_from_node(node, definition=definition)
            if pending_gate is None:
                if pause_kind == PauseKind.BLOCKED:
                    return self._recover_blocked(run_id, definition, snap, text)
                return self._emit_pause(
                    run_id,
                    PauseKind.APPROVAL,
                    node=node,
                    reason="No pending approval gate to confirm.",
                    message=(
                        "No pending approval gate to confirm; send clarification or "
                        "revision feedback."
                    ),
                )
            latest_snap = self.store.get_run(run_id)
            if pending_gate == "spec" and self._has_pending_resynthesis(
                run_id, latest_snap.get("spec"), definition=definition
            ):
                paused = self._ensure_clarified_spec_resynthesized(run_id, definition)
                if paused is not None:
                    return paused
                return self._continue(run_id, definition)
            try:
                self._approve_gate(
                    run_id,
                    pending_gate,
                    expected_revision=(
                        int(latest_snap["state_revision"]) if pending_gate == "spec" else None
                    ),
                )
            except (ValueError, ConcurrencyConflictError) as exc:
                return self._emit_pause(
                    run_id,
                    pause_kind or PauseKind.APPROVAL,
                    node=node,
                    gate=pending_gate,
                    reason=str(exc),
                    message=str(exc),
                )
            return self._continue(run_id, definition)

        if intent.action in ("revise", "reopen"):
            feedback = intent.feedback or text
            if pause_kind == PauseKind.BLOCKED and intent.action == "revise":
                return self._recover_blocked(run_id, definition, snap, feedback)
            return self._revise(run_id, definition, snap, feedback)

        fallback_kind = pause_kind or PauseKind.APPROVAL
        focus = self._pause_message(
            fallback_kind,
            snap,
            node=node,
            gate=gate,
            reason="Could not map the reply to a safe session action.",
        )
        return self._emit_pause(
            run_id,
            fallback_kind,
            node=node,
            gate=gate,
            reason="Could not map the reply to a safe session action.",
            message=(
                "I could not map that reply to a safe action. Please answer the pending "
                "question IDs explicitly, or say 'revise: <change>' / 'approve'.\n"
                f"{focus}"
            ),
        )

    # ------------------------------------------------------------------ internals

    def _drivers_config(self) -> ExecutionDriversConfig:
        return ExecutionDriversConfig.resolve(
            stage_driver=self.config.stage_driver,
            worker_driver=self.config.worker_driver,
            reviewer_driver=self.config.reviewer_driver,
            role_drivers=self.config.role_drivers or None,
            monitor_model=self.config.monitor_model,
            pi_provider=self.config.pi_provider,
            thinking_level=self.config.thinking_level,
            context_window=self.config.context_window,
        )

    def _resolved_gate_policy(self, run_id: str | None) -> GatePolicyName:
        if self.config.gate_policy is not None:
            return self.config.gate_policy
        if run_id is not None:
            try:
                events = self.store.list_events(run_id)
            except NotFoundError:
                events = []
            for event in reversed(events):
                if event.event_type != "session.started":
                    continue
                raw = event.payload.get("gate_policy")
                if raw:
                    try:
                        return GatePolicyName(str(raw))
                    except ValueError:
                        break
                break
        return GatePolicyName.INTERACTIVE_MINIMAL

    def _provider_config(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "provider": self.config.provider,
                "model": self.config.model,
                "base_url": self.config.base_url,
            }.items()
            if value not in (None, "")
        }

    def _provider(self, run_id: str | None = None) -> LLMProvider:
        provider = self.config.provider
        model = self.config.model
        base_url = self.config.base_url
        if (
            run_id is not None
            and provider in (None, "", "auto")
            and model is None
            and base_url is None
        ):
            try:
                recorded = self.store.get_run(run_id).get("provider_config") or {}
            except NotFoundError:
                recorded = {}
            provider = str(recorded.get("provider") or provider or "auto")
            model = recorded.get("model", model)
            base_url = recorded.get("base_url", base_url)
        if self.config.provider_builder is not None:
            return self.config.provider_builder(
                provider=provider or "auto",
                model=model,
                base_url=base_url,
            )
        if (provider or "").strip().lower() == "mock":
            return MockLLMProvider(model_name=model) if model else MockLLMProvider()
        return build_provider(
            provider=provider or "auto",
            model=model,
            base_url=base_url,
            **self.config.extra_provider_kwargs,
        )

    def _model_provider(self, model: str | None) -> LLMProvider:
        builder = self.config.provider_builder or build_provider
        kwargs: dict[str, Any] = {}
        if self.config.provider_builder is None:
            kwargs.update(self.config.extra_provider_kwargs)
        return builder(
            provider=self.config.provider or "auto",
            model=model or self.config.model,
            base_url=self.config.base_url,
            **kwargs,
        )

    def _runtime(self, run_id: str, definition: WorkflowDefinition) -> WorkflowRuntime:
        snap = self.store.get_run(run_id)
        recorded_drivers = snap.get("execution_drivers")
        base_drivers = (
            ExecutionDriversConfig.model_validate(recorded_drivers)
            if recorded_drivers
            else self._drivers_config()
        )
        explicit_overrides = {
            field: value
            for field, value in (
                ("stage_driver", self.config.stage_driver),
                ("worker_driver", self.config.worker_driver),
                ("reviewer_driver", self.config.reviewer_driver),
                ("monitor_model", self.config.monitor_model),
                ("pi_provider", self.config.pi_provider),
                ("thinking_level", self.config.thinking_level),
                ("context_window", self.config.context_window),
            )
            if value is not None
        }
        drivers = (
            ExecutionDriversConfig.model_validate(
                {**base_drivers.model_dump(mode="json"), **explicit_overrides}
            )
            if explicit_overrides
            else base_drivers
        )
        policy_name = self._resolved_gate_policy(run_id)
        policy = (
            InteractiveMinimalPolicy()
            if policy_name == GatePolicyName.INTERACTIVE_MINIMAL
            else None
        )
        active_provider = self._provider(run_id)
        return WorkflowRuntime(
            store=self.store,
            workspace_root=self.workspace_root,
            artifacts_root=self.artifacts_root,
            provider=active_provider,
            auto_approve=False,
            approver=f"session:{policy_name.value}",
            topology=definition.topology,
            worker_max_steps=self.config.worker_max_steps,
            apply_to_workspace=self.config.step_mode == StepMode.EXECUTE,
            gate_policy=policy,
            sandbox_root=self.sandbox_root,
            execution_drivers=drivers,
            driver_registry=DriverRegistry().with_factories(
                default_driver_factories(
                    store=self.store,
                    run_id=run_id,
                    provider=active_provider,
                    execution_drivers=drivers,
                )
            ),
            model_provider=self._model_provider,
        )

    def _effective_definition(self, definition: WorkflowDefinition) -> WorkflowDefinition:
        if self.config.step_mode == StepMode.PLAN_ONLY:
            return plan_only_definition(definition)
        return definition

    def _drive(
        self,
        *,
        run_id: str,
        raw_input: str,
        start_from: str | None,
        definition: WorkflowDefinition,
    ) -> SessionEnvelope:
        effective = self._effective_definition(definition)
        try:
            runtime = self._runtime(run_id, effective)
            graph = build_graph(effective)
        except Exception as exc:  # noqa: BLE001 -- surface graph/runtime setup failure as SessionEnvelope
            return self._fail(run_id, exc)
        snap = self.store.get_run(run_id)
        # Prefer the authoritative configuration persisted in session.started for this run_id
        # over self.config defaults so explicit start() overrides stay consistent across resume.
        started_meta: dict[str, Any] = {}
        started_hc: dict[str, Any] | None = None
        for event in self.store.list_events(run_id):
            if event.event_type != "session.started":
                continue
            ev_meta = event.payload.get("metadata")
            if isinstance(ev_meta, dict):
                started_meta.update(ev_meta)
            ev_hc = event.payload.get("hillclimb_config")
            if isinstance(ev_hc, dict):
                started_hc = dict(ev_hc)
        session_metadata: dict[str, Any] = {**dict(self.config.metadata), **started_meta}
        hillclimb_cfg: dict[str, Any] | None = (
            started_hc
            if started_hc is not None
            else (
                dict(started_meta["hillclimb_config"])
                if isinstance(started_meta.get("hillclimb_config"), dict)
                else (
                    dict(self.config.hillclimb_config)
                    if self.config.hillclimb_config is not None
                    else (
                        dict(session_metadata["hillclimb_config"])
                        if isinstance(session_metadata.get("hillclimb_config"), dict)
                        else None
                    )
                )
            )
        )
        state: dict[str, Any] = {
            "run_id": run_id,
            "raw_input": raw_input,
            "constraints": self._constraints(run_id),
            "topology": effective.topology,
            "route": None,
            "workspace_id": snap.get("workspace_id") or self.config.workspace_id,
            "workspace_mode": snap.get("workspace_mode") or self.config.workspace_mode,
            "task_type": snap.get("task_type") or self.config.task_type,
            "base_ref": snap.get("base_ref") or self.config.base_ref,
            "scope_paths": list(snap.get("scope_paths") or self.config.scope_paths),
            "_runtime": runtime,
        }
        if session_metadata:
            state["metadata"] = session_metadata
            if isinstance(session_metadata.get("verification_commands"), list):
                state["verification_commands"] = list(session_metadata["verification_commands"])
        if hillclimb_cfg is not None:
            state["hillclimb_config"] = hillclimb_cfg
        for key in ("spec", "design", "plan"):
            artifact = snap.get(key)
            if artifact is not None:
                state[key] = artifact.model_dump(mode="json")
        try:
            result = run_graph(
                graph,
                state,
                max_steps=max(12, runtime.max_steps * 3),
                start_from=start_from,
            )
        except Exception as exc:  # noqa: BLE001 - unrecoverable infrastructure error
            return self._fail(run_id, exc)

        paused_node = result.get("_paused_node")
        pause_reason = str(result.get("_pause_reason") or "")
        pause_gate = self._gate_from_node(
            str(paused_node) if paused_node else None,
            definition=effective,
            explicit_gate=result.get("pause_gate"),
        )
        final = self.store.get_run(run_id)
        if paused_node:
            kind = self._coerce_pause_kind(
                result.get("pause_kind"),
                str(paused_node),
                final,
                gate=pause_gate,
            )
            self._mark_awaiting_human(run_id, final, str(paused_node), pause_reason)
            return self._emit_pause(
                run_id,
                kind,
                node=str(paused_node),
                gate=pause_gate,
                reason=pause_reason,
            )

        spec: GoalSpec | None = final["spec"]
        plan: TaskPlan | None = final["plan"]
        if spec is not None and spec.unresolved_blockers():
            self._mark_awaiting_human(run_id, final, "gate_spec", "unresolved blockers")
            return self._emit_pause(
                run_id, PauseKind.CLARIFY, node="gate_spec", reason="unresolved blockers"
            )
        if self.config.step_mode == StepMode.PLAN_ONLY:
            return self._emit_plan_handoff(run_id, final, plan)
        blocked = [
            task
            for task in (plan.tasks if plan is not None else [])
            if task.status == TaskStatus.BLOCKED
        ]
        if (
            effective.stop_after in ("goal", "design", "plan")
            and not blocked
            and plan is not None
            and self.store.is_approved(
                run_id=run_id,
                target=ApprovalTarget.TASK_PLAN,
                target_id=plan.plan_id,
                target_hash=plan.content_hash,
            )
            and final["status"] != RunStatus.COMPLETED
        ):
            # Blueprint workflows intentionally stop after the approved plan.
            with suppress(ConcurrencyConflictError):
                self.store.set_run_status(
                    run_id,
                    RunStatus.COMPLETED,
                    expected_revision=int(final["state_revision"]),
                    actor="session",
                    stage="graph",
                    detail={"workflow": effective.name, "stop_after": effective.stop_after},
                )
            return self._completed_envelope(run_id, self.store.get_run(run_id))
        if blocked or final["status"] in (RunStatus.AWAITING_HUMAN, RunStatus.PAUSED):
            reason = self._blocked_reason(blocked)
            self._mark_awaiting_human(run_id, final, None, reason)
            return self._emit_pause(run_id, PauseKind.BLOCKED, node=None, reason=reason)
        if final["status"] != RunStatus.COMPLETED:
            with suppress(ConcurrencyConflictError):
                self.store.set_run_status(
                    run_id,
                    RunStatus.COMPLETED,
                    expected_revision=int(final["state_revision"]),
                    actor="session",
                    stage="graph",
                    detail={"workflow": effective.name, "step_mode": self.config.step_mode.value},
                )
            final = self.store.get_run(run_id)
        return self._completed_envelope(run_id, final)

    def _continue(self, run_id: str, definition: WorkflowDefinition) -> SessionEnvelope:
        paused = self._ensure_clarified_spec_resynthesized(run_id, definition)
        if paused is not None:
            return paused
        snap = self.store.get_run(run_id)
        effective = self._effective_definition(definition)
        entry = resolve_resume_node(effective, snap, store=self.store)
        if entry is None:
            if self.config.step_mode == StepMode.PLAN_ONLY:
                return self._emit_plan_handoff(run_id, snap, snap["plan"])
            if snap["status"] != RunStatus.COMPLETED:
                self.store.set_run_status(
                    run_id,
                    RunStatus.COMPLETED,
                    expected_revision=int(snap["state_revision"]),
                    actor="session",
                    stage="graph",
                    detail={"workflow": effective.name, "reason": "no_resume_entry"},
                )
                snap = self.store.get_run(run_id)
            return self._completed_envelope(run_id, snap)
        if snap["status"] == RunStatus.AWAITING_HUMAN:
            with suppress(ConcurrencyConflictError):
                self.store.set_run_status(
                    run_id,
                    RunStatus.RUNNING,
                    expected_revision=int(snap["state_revision"]),
                    actor="session",
                    stage="session",
                    detail={"resumed_at": entry},
                )
            snap = self.store.get_run(run_id)
        return self._drive(
            run_id=run_id,
            raw_input=self._current_input(run_id, snap),
            start_from=entry,
            definition=definition,
        )

    def _revise(
        self,
        run_id: str,
        definition: WorkflowDefinition,
        snap: dict[str, Any],
        feedback: str,
    ) -> SessionEnvelope:
        clean = feedback.strip()
        if not clean:
            return self._emit_pause(
                run_id,
                PauseKind.BLOCKED,
                node=None,
                reason="Revision feedback was empty.",
                message="Revision feedback was empty; describe the change you want.",
            )
        try:
            if snap["status"] == RunStatus.COMPLETED:
                self.control.reopen_run(run_id=run_id, feedback=clean, actor="human")
            else:
                self.control.request_revision(run_id=run_id, feedback=clean, actor="human")
        except (ValueError, ConcurrencyConflictError) as exc:
            return self._emit_pause(
                run_id, PauseKind.BLOCKED, node=None, reason=str(exc), message=str(exc)
            )
        augmented = self._augment_input(run_id, snap, clean, "User revision feedback")
        self.store.append_event(
            run_id=run_id,
            stage="session",
            event_type="session.continued",
            actor="session",
            payload={"raw_input": augmented, "feedback": clean},
        )
        return self._drive(
            run_id=run_id,
            raw_input=augmented,
            start_from=self._stage_node_name(definition, "align_goal"),
            definition=definition,
        )

    def _recover_blocked(
        self,
        run_id: str,
        definition: WorkflowDefinition,
        snap: dict[str, Any],
        hint: str,
    ) -> SessionEnvelope:
        """Apply a human hint received while blocked and resume at a safe node."""
        clean = hint.strip()
        if not clean:
            return self._emit_pause(
                run_id,
                PauseKind.BLOCKED,
                node=None,
                reason="Recovery hint was empty.",
                message="Recovery hint was empty; describe the path, symbol, or correction.",
            )
        spec: GoalSpec | None = snap.get("spec")
        if spec is not None and self._has_pending_resynthesis(
            run_id, spec, definition=definition
        ):
            existing_notes = self._clarification_notes(run_id, spec_id=spec.spec_id)
            if not _is_pure_retry_message(clean) and _normalize_clarification_text(clean) not in {
                _normalize_clarification_text(note) for note in existing_notes
            }:
                self.control.clarify_spec(
                    run_id=run_id,
                    answers={},
                    clarification_note=clean,
                    actor="human",
                )
            return self._continue(run_id, definition)
        blocked = self._execution_blocked_tasks(run_id, snap)
        if blocked:
            return self._recover_blocked_in_place(run_id, definition, snap, clean, blocked)
        if not definition.uses_sdlc_artifacts() and snap.get("plan") is None:
            return self._recover_blocked_craft(run_id, definition, snap, clean)
        if self._has_bound_approved_plan(run_id, snap):
            plan: TaskPlan | None = snap["plan"]
            pending = [
                task
                for task in (plan.tasks if plan is not None else [])
                if task.status
                in (
                    TaskStatus.READY,
                    TaskStatus.WAITING,
                    TaskStatus.RUNNING,
                    TaskStatus.VERIFYING,
                )
            ]
            if pending:
                return self._resume_execution_in_place(run_id, definition, snap, clean)
        return self._recover_blocked_planning(run_id, definition, snap, clean)

    def _recover_blocked_craft(
        self,
        run_id: str,
        definition: WorkflowDefinition,
        snap: dict[str, Any],
        hint: str,
    ) -> SessionEnvelope:
        """Resume a blocked non-SDLC craft run without re-running explore or stale cycles."""
        _kind, _gate, paused_node = self._pause_state(run_id)
        self.store.append_event(
            run_id=run_id,
            stage="craft",
            event_type="craft.clarification_recorded",
            actor="human",
            payload={"note": hint, "recovery": True},
        )
        self.store.append_event(
            run_id=run_id,
            stage="craft",
            event_type="craft.recovery_reset",
            actor="human",
            payload={"hint": hint, "paused_node": paused_node},
        )
        build_node = next(
            (n.name for n in definition.nodes if n.type == "build_direct"), None
        )
        has_architect = any(
            ev.event_type == "craft.architect_recorded"
            for ev in self.store.list_events(run_id)
        )
        if build_node is not None and has_architect:
            entry: str | None = build_node
        elif paused_node is not None and definition.node(paused_node) is not None:
            entry = paused_node
        else:
            entry = resolve_resume_node(definition, snap, store=self.store)
        if snap["status"] == RunStatus.AWAITING_HUMAN:
            with suppress(ConcurrencyConflictError):
                self.store.set_run_status(
                    run_id,
                    RunStatus.RUNNING,
                    expected_revision=int(snap["state_revision"]),
                    actor="session",
                    stage="session",
                    detail={"resumed_at": entry},
                )
            snap = self.store.get_run(run_id)
        augmented = self._augment_input(run_id, snap, hint, "User recovery hint")
        return self._drive(
            run_id=run_id,
            raw_input=augmented,
            start_from=entry,
            definition=definition,
        )

    def _has_bound_approved_plan(self, run_id: str, snap: dict[str, Any]) -> bool:
        """Return True when the run has an approved TaskPlan bound to approved upstream specs."""
        plan: TaskPlan | None = snap["plan"]
        if plan is None:
            return False
        if not self.store.is_approved(
            run_id=run_id,
            target=ApprovalTarget.TASK_PLAN,
            target_id=plan.plan_id,
            target_hash=plan.content_hash,
        ):
            return False
        return is_plan_bound_to_upstream(self.store, run_id, snap)

    def _execution_blocked_tasks(
        self, run_id: str, snap: dict[str, Any]
    ) -> list[TaskNode]:
        """Return BLOCKED tasks only for an approved, upstream-bound executable plan."""
        if not self._has_bound_approved_plan(run_id, snap):
            return []
        plan: TaskPlan | None = snap["plan"]
        if plan is None:
            return []
        return [task for task in plan.tasks if task.status == TaskStatus.BLOCKED]

    def _requeue_stranded_in_flight(
        self, run_id: str, snap: dict[str, Any]
    ) -> list[str]:
        """Reset any stranded RUNNING/VERIFYING tasks back to READY before resuming."""
        plan: TaskPlan | None = snap["plan"]
        requeued: list[str] = []
        if plan is None:
            return requeued
        for task in plan.tasks:
            if task.status not in (TaskStatus.RUNNING, TaskStatus.VERIFYING):
                continue
            with suppress(ValueError, ConcurrencyConflictError, NotFoundError):
                self.store.update_task_with_lease(
                    run_id,
                    task.task_id,
                    expected_lease_generation=task.lease_generation,
                    mutator=lambda node: node.model_copy(
                        update={"status": TaskStatus.READY}
                    ),
                    actor="session",
                    event_type="task.requeued",
                )
                requeued.append(task.task_id)
        return requeued

    def _resume_execution_in_place(
        self,
        run_id: str,
        definition: WorkflowDefinition,
        snap: dict[str, Any],
        hint: str,
    ) -> SessionEnvelope:
        """Resume an execution pause (e.g. budget pause or crash recovery) in place."""
        requeued = self._requeue_stranded_in_flight(run_id, snap)
        if requeued:
            snap = self.store.get_run(run_id)
        if snap["status"] in (RunStatus.AWAITING_HUMAN, RunStatus.PAUSED):
            with suppress(ValueError, ConcurrencyConflictError):
                self.store.set_run_status(
                    run_id,
                    RunStatus.RUNNING,
                    expected_revision=int(snap["state_revision"]),
                    actor="human",
                    stage="session",
                    detail={"recovery_mode": "resume_execution", "feedback": hint},
                )
        execute_node = self._stage_node_name(definition, "execute_task_dag")
        self.store.append_event(
            run_id=run_id,
            stage="session",
            event_type="session.continued",
            actor="session",
            payload={
                "feedback": hint,
                "recovery_mode": "resume_execution",
                "recovery_node": execute_node,
                **({"requeued_task_ids": requeued} if requeued else {}),
            },
        )
        return self._drive(
            run_id=run_id,
            raw_input=self._current_input(run_id, snap),
            start_from=execute_node,
            definition=definition,
        )

    def _recover_blocked_in_place(
        self,
        run_id: str,
        definition: WorkflowDefinition,
        snap: dict[str, Any],
        hint: str,
        blocked: list[TaskNode],
    ) -> SessionEnvelope:
        """Retry blocked tasks on their retained worktrees and resume at ``execute``."""
        recoverable = [
            task for task in blocked if task.in_place_recoveries < _MAX_IN_PLACE_RECOVERIES
        ]
        if not recoverable:
            reason = (
                "Task execution is still blocked and the in-place recovery budget "
                f"(max_in_place_recoveries = {_MAX_IN_PLACE_RECOVERIES}) is exhausted for "
                f"{', '.join(task.task_id for task in blocked)}. The run stays paused without "
                "re-planning or retrying again; send 'revise: <change>' to request a requirement "
                "revision, or resolve the blocker manually."
            )
            self._mark_awaiting_human(run_id, snap, None, reason)
            return self._emit_pause(run_id, PauseKind.BLOCKED, node=None, reason=reason)

        requeued = self._requeue_stranded_in_flight(run_id, snap)
        retried: list[str] = []
        revision = int(self.store.get_run(run_id)["state_revision"])
        try:
            for task in recoverable:
                result = self.control.retry_task(
                    run_id,
                    task.task_id,
                    operation_id=(
                        f"in-place-recovery:{run_id}:{task.task_id}:"
                        f"{task.in_place_recoveries + 1}"
                    ),
                    expected_state_revision=revision,
                    hint=hint,
                    actor="human",
                )
                revision = int(result["state_revision"])
                retried.append(task.task_id)
        except (ValueError, ConcurrencyConflictError) as exc:
            return self._emit_pause(
                run_id, PauseKind.BLOCKED, node=None, reason=str(exc), message=str(exc)
            )

        execute_node = self._stage_node_name(definition, "execute_task_dag")
        self.store.append_event(
            run_id=run_id,
            stage="session",
            event_type="session.continued",
            actor="session",
            payload={
                "feedback": hint,
                "recovery_mode": "in_place",
                "recovery_node": execute_node,
                "retried_task_ids": retried,
                **({"requeued_task_ids": requeued} if requeued else {}),
            },
        )
        return self._drive(
            run_id=run_id,
            raw_input=self._current_input(run_id, snap),
            start_from=execute_node,
            definition=definition,
        )

    def _recover_blocked_planning(
        self,
        run_id: str,
        definition: WorkflowDefinition,
        snap: dict[str, Any],
        hint: str,
    ) -> SessionEnvelope:
        """Regenerate the plan (or alignment) for a block that happened before execution."""
        prior_recoveries = self._planning_recovery_count(run_id)
        if prior_recoveries >= _MAX_PLAN_RECOVERIES:
            reason = (
                "Planning is still blocked and the planning recovery budget "
                f"(max_plan_recoveries = {_MAX_PLAN_RECOVERIES}) is exhausted. The run stays "
                "paused without generating another plan; send 'revise: <change>' with a concrete "
                "requirement change, or resolve the blocker manually."
            )
            self._mark_awaiting_human(run_id, snap, None, reason)
            return self._emit_pause(run_id, PauseKind.BLOCKED, node=None, reason=reason)
        try:
            self.control.request_revision(run_id=run_id, feedback=hint, actor="human")
        except (ValueError, ConcurrencyConflictError) as exc:
            return self._emit_pause(
                run_id, PauseKind.BLOCKED, node=None, reason=str(exc), message=str(exc)
            )
        entry = self._recovery_entry(run_id, definition, snap)
        augmented = self._augment_input(run_id, snap, hint, "User recovery hint")
        self.store.append_event(
            run_id=run_id,
            stage="session",
            event_type="session.continued",
            actor="session",
            payload={"raw_input": augmented, "feedback": hint, "recovery_node": entry},
        )
        return self._drive(
            run_id=run_id,
            raw_input=augmented,
            start_from=entry,
            definition=definition,
        )

    def _planning_recovery_count(self, run_id: str) -> int:
        count = 0
        for event in self.store.list_events(run_id):
            if event.event_type != "session.continued":
                continue
            payload = event.payload
            if "recovery_node" not in payload:
                continue
            if payload.get("recovery_mode") in ("in_place", "resume_execution"):
                continue
            count += 1
        return count

    @staticmethod
    def _find_node_by_type(definition: WorkflowDefinition, node_type: str) -> str | None:
        return next((node.name for node in definition.nodes if node.type == node_type), None)

    @classmethod
    def _stage_node_name(cls, definition: WorkflowDefinition, node_type: str) -> str:
        found = cls._find_node_by_type(definition, node_type)
        if found is not None:
            return found
        return definition.nodes[0].name

    def _recovery_entry(
        self, run_id: str, definition: WorkflowDefinition, snap: dict[str, Any]
    ) -> str:
        """Resume from the plan node when upstream spec/design bindings are valid."""
        spec: GoalSpec | None = snap["spec"]
        design: DesignSpec | None = snap["design"]
        align_node = self._stage_node_name(definition, "align_goal")
        plan_node = self._find_node_by_type(definition, "pm_decompose")
        has_design_stage = self._find_node_by_type(definition, "design_spec") is not None
        if (
            plan_node is None
            or spec is None
            or not self.store.is_approved(
                run_id=run_id,
                target=ApprovalTarget.GOAL_SPEC,
                target_id=spec.spec_id,
                target_hash=spec.content_hash,
            )
        ):
            return align_node
        if has_design_stage and (
            design is None or not is_design_bound_to_spec(self.store, run_id, snap)
        ):
            return align_node
        return plan_node

    def _workflow_for_run(
        self, run_id: str, snap: dict[str, Any]
    ) -> WorkflowDefinition:
        """Reload the workflow the run was started with, custom YAML included.

        ``reply`` accepts an explicit ``--workflow`` path; it must name the same
        workflow the run recorded. When the flag is omitted, the workflow persisted by
        ``start`` (custom YAML path or built-in name) wins across pause/resume.
        """
        recorded_name = str(snap["workflow_name"])
        requested_path = _yaml_workflow_path(self.config.workflow)
        if requested_path is not None:
            candidate: str | Path = requested_path
        elif not self.config.workflow_explicit:
            recorded_path = self._recorded_workflow_path(run_id)
            candidate = recorded_path if recorded_path is not None else recorded_name
        else:
            candidate = self.config.workflow
        definition = load_workflow(candidate)
        if definition.name != recorded_name:
            raise ValueError(
                f"Workflow '{definition.name}' does not match the run's workflow "
                f"'{recorded_name}'"
            )
        recorded_hash = self._recorded_workflow_hash(run_id)
        if recorded_hash is not None:
            valid_hashes = {definition.execution_hash(), definition.content_hash}
            if recorded_hash not in valid_hashes:
                raise ValueError(
                    f"Workflow definition for '{definition.name}' does not match the "
                    "workflow that started this run"
                )
        return definition

    def _recorded_workflow_path(self, run_id: str) -> Path | None:
        try:
            events = self.store.list_events(run_id)
        except NotFoundError:
            return None
        for event in events:
            if event.event_type != "session.started":
                continue
            raw = event.payload.get("workflow_path")
            if not raw:
                return None
            path = Path(str(raw))
            return path if path.is_file() else None
        return None

    def _recorded_workflow_hash(self, run_id: str) -> str | None:
        try:
            events = self.store.list_events(run_id)
        except NotFoundError:
            return None
        for event in events:
            if event.event_type != "session.started":
                continue
            raw = event.payload.get("workflow_hash")
            return str(raw) if raw else None
        return None

    def _incremental_reopen(
        self,
        run_id: str,
        definition: WorkflowDefinition,
        snap: dict[str, Any],
        message: str,
    ) -> SessionEnvelope:
        if not message:
            return self._completed_envelope(run_id, snap)
        intent = parse_reply(
            provider=self._provider(run_id),
            message=message,
            snapshot=snap,
            pause_kind=None,
            gate=None,
            completed=True,
        )
        feedback = intent.feedback.strip() or message
        if intent.action not in ("reopen", "revise") or not _looks_actionable(feedback):
            # Empty, punctuation-only, or unparseable replies must never reopen a
            # finished run: keep the completed status and ask for a concrete request.
            return SessionEnvelope(
                run_id=run_id,
                status=SessionStatus.NEEDS_INPUT,
                pause_kind=None,
                step_mode=self.config.step_mode,
                workflow=snap["workflow_name"],
                message=(
                    "This run is already completed, and the reply did not describe a "
                    "concrete follow-up change. Send a specific correction (for example "
                    "'revise: <change>') to reopen it."
                ),
                unresolved_blockers=[],
                changed_files=self._changed_files(),
                error=None,
                token_usage=self._token_usage(run_id),
            )
        try:
            self.control.reopen_run(run_id=run_id, feedback=feedback, actor="human")
        except (ValueError, ConcurrencyConflictError) as exc:
            return SessionEnvelope.failure(
                error=str(exc),
                run_id=run_id,
                step_mode=self.config.step_mode,
                workflow=snap["workflow_name"],
                token_usage=self._token_usage(run_id),
            )
        augmented = self._augment_input(run_id, snap, feedback, "Follow-up instruction from user")
        self.store.append_event(
            run_id=run_id,
            stage="session",
            event_type="session.continued",
            actor="session",
            payload={"raw_input": augmented, "feedback": feedback, "reopened": True},
        )
        return self._drive(
            run_id=run_id,
            raw_input=augmented,
            start_from=self._stage_node_name(definition, "align_goal"),
            definition=definition,
        )

    def _approve_gate(
        self,
        run_id: str,
        gate: str,
        *,
        expected_revision: int | None = None,
    ) -> None:
        if gate == "spec":
            self.control.confirm_spec(
                run_id=run_id, approver="human", expected_revision=expected_revision
            )
        elif gate == "design":
            self.control.approve_design(run_id=run_id, approver="human")
        elif gate == "plan":
            self.control.approve_plan(run_id=run_id, approver="human")
        else:
            raise ValueError(f"Unknown approval gate: {gate}")

    # ------------------------------------------------------------------ envelope

    def _token_usage(self, run_id: str) -> TokenUsageSummary:
        """Run-wide token accounting from the BudgetLedger plus per-actor settled events."""
        try:
            ledger: BudgetLedger = self.store.get_run(run_id)["budget_ledger"]
        except NotFoundError:
            return TokenUsageSummary()
        summary = TokenUsageSummary(
            llm_calls=ledger.llm_calls_used,
            input_tokens=ledger.input_tokens,
            output_tokens=ledger.output_tokens,
            reasoning_tokens=ledger.reasoning_tokens,
            cache_read_tokens=ledger.cache_read_tokens,
            cache_write_tokens=ledger.cache_write_tokens,
            total_tokens=ledger.total_tokens,
            cache_hit_rate=ledger.cache_hit_rate,
        )
        try:
            events = self.store.list_events(run_id)
        except NotFoundError:
            return summary
        by_actor: dict[str, ActorTokenUsage] = {}
        for event in events:
            if event.event_type != "budget.settled":
                continue
            usage = by_actor.setdefault(event.actor, ActorTokenUsage())
            payload = event.payload
            usage.llm_calls += 1
            usage.input_tokens += max(0, int(payload.get("input_tokens") or 0))
            usage.output_tokens += max(0, int(payload.get("output_tokens") or 0))
            usage.reasoning_tokens += max(0, int(payload.get("reasoning_tokens") or 0))
            usage.cache_read_tokens += max(0, int(payload.get("cache_read_tokens") or 0))
            usage.cache_write_tokens += max(0, int(payload.get("cache_write_tokens") or 0))
        for usage in by_actor.values():
            usage.total_tokens = (
                usage.input_tokens
                + usage.output_tokens
                + usage.cache_read_tokens
                + usage.cache_write_tokens
            )
        summary.by_actor = by_actor
        return summary

    def _emit_pause(
        self,
        run_id: str,
        kind: PauseKind,
        *,
        node: str | None,
        gate: str | None = None,
        reason: str,
        message: str | None = None,
    ) -> SessionEnvelope:
        snap = self.store.get_run(run_id)
        spec: GoalSpec | None = snap["spec"]
        blockers = spec.unresolved_blockers() if spec is not None else []
        resolved_gate = gate if gate is not None else self._gate_from_node(node)
        text = message or self._pause_message(
            kind, snap, node=node, gate=resolved_gate, reason=reason
        )
        self.store.append_event(
            run_id=run_id,
            stage="session",
            event_type="session.paused",
            actor="session",
            payload={
                "pause_kind": kind.value,
                "node": node,
                "gate": resolved_gate,
                "reason": reason,
                "message": text,
                "unresolved_blockers": blockers,
            },
        )
        return SessionEnvelope(
            run_id=run_id,
            status=SessionStatus.NEEDS_INPUT,
            pause_kind=kind,
            step_mode=self.config.step_mode,
            workflow=snap["workflow_name"],
            message=text,
            unresolved_blockers=blockers,
            changed_files=self._changed_files(),
            error=None,
            token_usage=self._token_usage(run_id),
        )

    def _emit_plan_handoff(
        self,
        run_id: str,
        snap: dict[str, Any],
        plan: TaskPlan | None,
    ) -> SessionEnvelope:
        self._mark_awaiting_human(run_id, snap, "gate_plan", "plan_only handoff")
        return self._emit_pause(
            run_id,
            PauseKind.PLAN_HANDOFF,
            node="gate_plan",
            reason="plan_only step reached its terminal handoff",
            message=self._plan_message(plan),
        )

    def _current_pause_envelope(
        self, snap: dict[str, Any], *, message: str
    ) -> SessionEnvelope:
        pause_kind, gate, node = self._pause_state(str(snap["run_id"]))
        return self._emit_pause(
            str(snap["run_id"]),
            pause_kind or PauseKind.BLOCKED,
            node=node,
            gate=gate,
            reason=message,
            message=message,
        )

    def _completed_envelope(self, run_id: str, snap: dict[str, Any]) -> SessionEnvelope:
        changed = self._changed_files()
        message = (
            f"Session completed with {len(changed)} changed file(s)."
            if changed
            else "Session completed with no workspace changes."
        )
        self.store.append_event(
            run_id=run_id,
            stage="session",
            event_type="session.completed",
            actor="session",
            payload={"changed_files": changed, "message": message},
        )
        return SessionEnvelope(
            run_id=run_id,
            status=SessionStatus.COMPLETED,
            pause_kind=None,
            step_mode=self.config.step_mode,
            workflow=snap["workflow_name"],
            message=message,
            unresolved_blockers=[],
            changed_files=changed,
            error=None,
            token_usage=self._token_usage(run_id),
        )

    def _fail(self, run_id: str, exc: BaseException) -> SessionEnvelope:
        error = str(exc) or exc.__class__.__name__
        try:
            snap = self.store.get_run(run_id)
        except NotFoundError:
            return SessionEnvelope.failure(
                error=error, run_id=run_id, step_mode=self.config.step_mode
            )
        if snap["status"] not in (RunStatus.PAUSED, RunStatus.CANCELLED):
            with suppress(ConcurrencyConflictError):
                self.store.set_run_status(
                    run_id,
                    RunStatus.FAILED,
                    expected_revision=int(snap["state_revision"]),
                    actor="session",
                    stage="session",
                    detail={"error": error},
                )
        self.store.append_event(
            run_id=run_id,
            stage="session",
            event_type="session.failed",
            actor="session",
            payload={"error": error},
        )
        return SessionEnvelope(
            run_id=run_id,
            status=SessionStatus.FAILED,
            pause_kind=None,
            step_mode=self.config.step_mode,
            workflow=snap["workflow_name"],
            message=error,
            unresolved_blockers=[],
            changed_files=self._changed_files(),
            error=error,
            token_usage=self._token_usage(run_id),
        )

    # ------------------------------------------------------------------ helpers

    def _mark_awaiting_human(
        self,
        run_id: str,
        snap: dict[str, Any],
        node: str | None,
        reason: str,
    ) -> None:
        if snap["status"] in (RunStatus.PAUSED, RunStatus.CANCELLED):
            return
        if snap["status"] == RunStatus.AWAITING_HUMAN:
            return
        with suppress(ConcurrencyConflictError):
            self.store.set_run_status(
                run_id,
                RunStatus.AWAITING_HUMAN,
                expected_revision=int(snap["state_revision"]),
                actor="session",
                stage="session",
                detail={"paused_node": node, "pause_reason": reason},
            )

    def _pause_state(self, run_id: str) -> tuple[PauseKind | None, str | None, str | None]:
        try:
            events = self.store.list_events(run_id)
        except NotFoundError:
            return None, None, None
        for event in reversed(events):
            if event.event_type == "session.paused":
                kind_raw = event.payload.get("pause_kind")
                try:
                    kind = PauseKind(str(kind_raw)) if kind_raw else None
                except ValueError:
                    kind = None
                gate_raw = event.payload.get("gate")
                node_raw = event.payload.get("node")
                return (
                    kind,
                    str(gate_raw) if gate_raw else None,
                    str(node_raw) if node_raw else None,
                )
            if event.event_type in ("session.completed", "session.failed"):
                return None, None, None
        return None, None, None

    def _coerce_pause_kind(
        self,
        raw: Any,
        paused_node: str,
        snap: dict[str, Any],
        *,
        gate: str | None = None,
    ) -> PauseKind:
        if isinstance(raw, str):
            try:
                return PauseKind(raw)
            except ValueError:
                pass
        return self._infer_pause_kind(paused_node, snap, gate=gate)

    def _infer_pause_kind(
        self, paused_node: str, snap: dict[str, Any], *, gate: str | None = None
    ) -> PauseKind:
        resolved_gate = gate if gate is not None else self._gate_from_node(paused_node)
        if resolved_gate is not None:
            if resolved_gate == "spec":
                spec: GoalSpec | None = snap["spec"]
                if spec is not None and spec.unresolved_blockers():
                    return PauseKind.CLARIFY
                return PauseKind.APPROVAL
            return PauseKind.APPROVAL
        return PauseKind.BLOCKED

    @staticmethod
    def _gate_from_node(
        node: str | None,
        *,
        definition: WorkflowDefinition | None = None,
        explicit_gate: Any = None,
    ) -> str | None:
        if isinstance(explicit_gate, str) and explicit_gate in GATE_SPECS:
            return explicit_gate
        if node is None:
            return None
        if definition is not None:
            node_def = definition.node(node)
            if node_def is not None and node_def.type == "human_gate":
                declared = str(node_def.params.get("gate", "spec"))
                if declared in GATE_SPECS:
                    return declared
        return _GATE_BY_NODE.get(node)

    def _pause_message(
        self,
        kind: PauseKind,
        snap: dict[str, Any],
        *,
        node: str | None,
        gate: str | None = None,
        reason: str,
    ) -> str:
        spec: GoalSpec | None = snap["spec"]
        if kind == PauseKind.CLARIFY:
            lines: list[str] = []
            if spec is not None:
                for ambiguity in spec.open_ambiguities:
                    if ambiguity.blocking and not ambiguity.answer:
                        options = (
                            f" (options: {', '.join(ambiguity.options)})"
                            if ambiguity.options
                            else ""
                        )
                        lines.append(f"{ambiguity.id}: {ambiguity.question}{options}")
                for note in spec.minority_notes:
                    if note.severity.value == "blocking" and not note.resolved:
                        lines.append(f"{note.id}: {note.summary}")
            if lines:
                return "\n".join(lines)
            return reason or "Additional clarification is required."
        if kind == PauseKind.APPROVAL:
            resolved_gate = gate if gate is not None else self._gate_from_node(node)
            if resolved_gate is not None:
                label = GATE_SPECS[resolved_gate]["label"]
                return reason or f"{label} is awaiting human approval."
            return reason or "The run is awaiting human approval."
        if kind == PauseKind.PLAN_HANDOFF:
            return self._plan_message(snap["plan"])
        return reason or "The run is blocked and needs human guidance."

    @staticmethod
    def _plan_message(plan: TaskPlan | None) -> str:
        if plan is None:
            return "Plan step finished without a TaskPlan."
        tasks = "; ".join(f"{task.task_id}: {task.title}" for task in plan.tasks)
        return (
            f"Plan {plan.plan_id} is approved with {len(plan.tasks)} task(s): {tasks}. "
            "Re-run with --step-mode execute to implement it."
        )

    @staticmethod
    def _blocked_reason(blocked: list[TaskNode]) -> str:
        if not blocked:
            return "The workflow paused and needs human review."
        details = "; ".join(
            f"{task.task_id}: {task.last_error or 'blocked'}" for task in blocked
        )
        return f"Task execution blocked: {details}"

    def _unconsumed_clarification_note_events(
        self, run_id: str, *, spec_id: str
    ) -> list[tuple[str, str]]:
        """Return ``(event_id, feedback)`` pairs not yet consumed by ``spec.resynthesized``."""
        return unconsumed_clarification_notes(self.store.list_events(run_id), spec_id=spec_id)

    def _has_pending_resynthesis(
        self,
        run_id: str,
        spec: GoalSpec | None,
        *,
        definition: WorkflowDefinition | None = None,
    ) -> bool:
        """True when a clarified GoalSpec still needs body-level resynthesis before gate_spec."""
        if spec is None:
            return False
        strict_canonical = (
            _workflow_strict_canonical(self._effective_definition(definition))
            if definition is not None
            else False
        )
        return has_pending_resynthesis(
            self.store.list_events(run_id),
            spec,
            spec_approved=self.store.is_approved(
                run_id=run_id,
                target=ApprovalTarget.GOAL_SPEC,
                target_id=spec.spec_id,
                target_hash=spec.content_hash,
            ),
            strict_canonical=strict_canonical,
        )

    def _ensure_clarified_spec_resynthesized(
        self,
        run_id: str,
        definition: WorkflowDefinition,
    ) -> SessionEnvelope | None:
        """Complete any pending post-clarification resynthesis before entering gate_spec."""
        for _ in range(2):
            snap = self.store.get_run(run_id)
            spec: GoalSpec | None = snap.get("spec")
            if spec is None or not self._has_pending_resynthesis(
                run_id, spec, definition=definition
            ):
                return None
            note_events = self._unconsumed_clarification_note_events(run_id, spec_id=spec.spec_id)
            extra_notes = [feedback for _, feedback in note_events]
            consumed_ids = [event_id for event_id, _ in note_events]
            try:
                self._resynthesize_clarified_spec(
                    run_id,
                    definition,
                    snap,
                    extra_feedback="\n\n".join(extra_notes),
                    consumed_note_ids=consumed_ids,
                )
            except Exception as exc:  # noqa: BLE001 -- preserve resumable pause on stage/validation error
                latest = self.store.get_run(run_id)
                reason = f"Spec resynthesis failed: {exc}"
                self._mark_awaiting_human(run_id, latest, "align_resynthesize", reason)
                return self._emit_pause(
                    run_id,
                    PauseKind.BLOCKED,
                    node="align_resynthesize",
                    reason=reason,
                    message=(
                        f"Spec resynthesis failed ({exc}). Reply 'retry' to try again in place, "
                        "or add extra clarification guidance."
                    ),
                )
        snap = self.store.get_run(run_id)
        if self._has_pending_resynthesis(run_id, snap.get("spec"), definition=definition):
            reason = "New clarification notes arrived during resynthesis; retry to fuse them."
            self._mark_awaiting_human(run_id, snap, "align_resynthesize", reason)
            return self._emit_pause(
                run_id,
                PauseKind.BLOCKED,
                node="align_resynthesize",
                reason=reason,
                message=reason,
            )
        return None

    def _clarification_notes(self, run_id: str, *, spec_id: str) -> list[str]:
        """Collect unconsumed human clarification notes recorded for the active spec."""
        return [
            feedback
            for _, feedback in self._unconsumed_clarification_note_events(run_id, spec_id=spec_id)
        ]

    def _resynthesize_clarified_spec(
        self,
        run_id: str,
        definition: WorkflowDefinition,
        snap: dict[str, Any],
        *,
        extra_feedback: str,
        consumed_note_ids: list[str] | None = None,
    ) -> GoalSpec:
        effective = self._effective_definition(definition)
        runtime = self._runtime(run_id, effective)
        stage = AlignStage(
            store=runtime.store,
            provider=runtime.provider,
            workspace_root=runtime.workspace_root,
            trace_dir=runtime.artifacts_root,
            execution_drivers=runtime.execution_drivers,
            driver_registry=runtime.driver_registry,
        )
        return stage.resynthesize_after_clarification(
            run_id=run_id,
            raw_goal=self._current_input(run_id, snap),
            constraints=self._constraints(run_id),
            extra_feedback=extra_feedback,
            consumed_note_ids=consumed_note_ids,
            strict_canonical=_workflow_strict_canonical(effective),
        )

    def _current_input(self, run_id: str, snap: dict[str, Any]) -> str:
        for event in reversed(self.store.list_events(run_id)):
            if event.event_type in ("session.started", "session.continued"):
                raw = event.payload.get("raw_input")
                if isinstance(raw, str) and raw.strip():
                    return raw
        return str(snap["title"])

    def _constraints(self, run_id: str) -> list[str]:
        for event in reversed(self.store.list_events(run_id)):
            if event.event_type in ("session.started", "session.continued"):
                items = event.payload.get("constraints")
                if isinstance(items, list):
                    return [str(item).strip() for item in items if str(item).strip()]
        return []

    def _augment_input(
        self,
        run_id: str,
        snap: dict[str, Any],
        feedback: str,
        heading: str,
    ) -> str:
        base = self._current_input(run_id, snap)
        return f"{base}\n\n{heading}:\n{feedback}"

    def _changed_files(self) -> list[str]:
        if self.config.step_mode == StepMode.PLAN_ONLY:
            return []
        try:
            proc = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return []
        if proc.returncode != 0:
            return []
        paths: list[str] = []
        for line in proc.stdout.splitlines():
            if not line.strip():
                continue
            payload = line[3:].strip() if len(line) > 3 else ""
            if " -> " in payload:
                payload = payload.split(" -> ", 1)[1]
            payload = payload.strip().strip('"')
            if payload:
                paths.append(payload)
        return sorted(dict.fromkeys(paths))


__all__ = [
    "SessionConfig",
    "SessionRunner",
    "plan_only_definition",
    "resolve_state_dir",
    "resolve_workspace_binding",
]
