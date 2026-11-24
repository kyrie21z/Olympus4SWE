"""``olympus session`` headless CLI: stable JSON envelopes over the workflow runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from olympus.providers.base import LLMProvider, MockLLMProvider, ProviderError
from olympus.providers.factory import build_provider
from olympus.session.contracts import GatePolicyName, SessionEnvelope, SessionStatus, StepMode
from olympus.session.runner import (
    SessionConfig,
    SessionRunner,
    resolve_workspace_binding,
)

session_app = typer.Typer(
    name="session",
    help="Headless pause-and-resume session CLI emitting a stable JSON envelope.",
    no_args_is_help=True,
)

WorkspaceOpt = Annotated[
    Path | None,
    typer.Option("--workspace", "-w", help="Target repository root (defaults to cwd)."),
]
StateDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--state-dir",
        help="External state root for SQLite, traces, and sandboxes (defaults outside the repo).",
    ),
]
WorkflowOpt = Annotated[
    str,
    typer.Option("--workflow", help="Builtin workflow name or YAML path."),
]
ReplyWorkflowOpt = Annotated[
    str | None,
    typer.Option(
        "--workflow",
        help="Builtin workflow name or YAML path (defaults to the run's recorded workflow).",
    ),
]
GatePolicyOpt = Annotated[
    GatePolicyName | None,
    typer.Option(
        "--gate-policy",
        help="Gate austerity mode (defaults to the policy recorded at session start).",
    ),
]
StepModeOpt = Annotated[
    StepMode,
    typer.Option("--step-mode", help="Stop after planning or execute accepted task changes."),
]
ProviderOpt = Annotated[
    str | None,
    typer.Option("--provider", help="Provider ID, 'auto', or 'mock'."),
]
ModelOpt = Annotated[
    str | None,
    typer.Option("--model", help="Override LLM model name."),
]
BaseUrlOpt = Annotated[
    str | None,
    typer.Option("--base-url", help="Override OpenAI-compatible/Anthropic base URL."),
]
StageDriverOpt = Annotated[
    str | None,
    typer.Option(
        "--stage-driver",
        help="Planning driver for Align/Design/Plan: auto|native|pi (optionally driver@model).",
    ),
]
WorkerDriverOpt = Annotated[
    str | None,
    typer.Option(
        "--worker-driver",
        help="Stage 2 worker driver: auto|native|pi (optionally driver@model). "
        "Default auto selects Pi when available on eligible remote providers.",
    ),
]
ReviewerDriverOpt = Annotated[
    str | None,
    typer.Option(
        "--reviewer-driver",
        help="Stage 2 reviewer driver: auto|native|pi (optionally driver@model). "
        "Default auto selects Pi when available on eligible remote providers.",
    ),
]
MonitorModelOpt = Annotated[
    str | None,
    typer.Option("--monitor-model", help="Deprecated no-op. The LLM Watchdog was removed."),
]
PiProviderOpt = Annotated[
    str | None,
    typer.Option(
        "--pi-provider",
        help=(
            "Gateway provider ID for the native Pi driver "
            "(e.g. the project's OpenAI-compatible gateway)."
        ),
    ),
]
ThinkingLevelOpt = Annotated[
    str | None,
    typer.Option(
        "--thinking",
        help="Pi thinking effort: off|minimal|low|medium|high|xhigh|max.",
    ),
]
ContextWindowOpt = Annotated[
    int | None,
    typer.Option("--context-window", min=1024, help="Pi model context window override."),
]
MessageOpt = Annotated[
    str | None,
    typer.Option("--message", help="Natural-language goal or reply text."),
]


def build_session_provider(
    *, provider: str | None, model: str | None, base_url: str | None
) -> LLMProvider:
    """Resolve one provider instance for a session invocation.

    Tests inject a scripted ``MockLLMProvider`` by replacing this module-level builder.
    """
    requested = (provider or "auto").strip().lower()
    if requested == "mock":
        return MockLLMProvider(model_name=model) if model else MockLLMProvider()
    return build_provider(provider=provider or "auto", model=model, base_url=base_url)


def _emit(envelope: SessionEnvelope) -> None:
    typer.echo(envelope.model_dump_json(indent=2))
    if envelope.status == SessionStatus.FAILED:
        raise typer.Exit(code=1)


def _build_runner(
    *,
    workspace: Path | None,
    state_dir: Path | None,
    workflow: str | None,
    gate_policy: GatePolicyName | None,
    step_mode: StepMode,
    provider: str | None,
    model: str | None,
    base_url: str | None,
    stage_driver: str | None,
    worker_driver: str | None,
    reviewer_driver: str | None,
    monitor_model: str | None,
    pi_provider: str | None,
    thinking_level: str | None,
    context_window: int | None,
    run_id: str | None = None,
) -> SessionRunner:
    root = resolve_workspace_binding(
        requested=workspace, state_dir=state_dir, run_id=run_id
    )
    if root is None:
        root = Path.cwd().resolve()
    config = SessionConfig(
        workspace_root=root,
        state_dir=state_dir,
        workflow=workflow or "full_sdlc_tdd",
        workflow_explicit=workflow is not None,
        gate_policy=gate_policy,
        step_mode=step_mode,
        provider=provider or "auto",
        model=model,
        base_url=base_url,
        stage_driver=stage_driver,
        worker_driver=worker_driver,
        reviewer_driver=reviewer_driver,
        monitor_model=monitor_model,
        pi_provider=pi_provider,
        thinking_level=thinking_level,
        context_window=context_window,
        provider_builder=build_session_provider,
    )
    return SessionRunner(config)


@session_app.command("start")
def cmd_start(
    goal: Annotated[
        str | None, typer.Argument(help="Raw project/feature goal.")
    ] = None,
    message: MessageOpt = None,
    title: Annotated[str | None, typer.Option("--title", help="Optional run title.")] = None,
    constraint: Annotated[
        list[str] | None, typer.Option("--constraint", "-c", help="Explicit constraint.")
    ] = None,
    workspace: WorkspaceOpt = None,
    state_dir: StateDirOpt = None,
    workflow: WorkflowOpt = "full_sdlc_tdd",
    gate_policy: GatePolicyOpt = None,
    step_mode: StepModeOpt = StepMode.EXECUTE,
    provider: ProviderOpt = "auto",
    model: ModelOpt = None,
    base_url: BaseUrlOpt = None,
    stage_driver: StageDriverOpt = None,
    worker_driver: WorkerDriverOpt = None,
    reviewer_driver: ReviewerDriverOpt = None,
    monitor_model: MonitorModelOpt = None,
    pi_provider: PiProviderOpt = None,
    thinking: ThinkingLevelOpt = None,
    context_window: ContextWindowOpt = None,
) -> None:
    """Create a run and drive it until a pause, completion, or failure."""
    raw_goal = (message or goal or "").strip()
    try:
        runner = _build_runner(
            workspace=workspace,
            state_dir=state_dir,
            workflow=workflow,
            gate_policy=gate_policy,
            step_mode=step_mode,
            provider=provider,
            model=model,
            base_url=base_url,
            stage_driver=stage_driver,
            worker_driver=worker_driver,
            reviewer_driver=reviewer_driver,
            monitor_model=monitor_model,
            pi_provider=pi_provider,
            thinking_level=thinking,
            context_window=context_window,
        )
    except (ProviderError, ValueError, OSError) as exc:
        _emit(
            SessionEnvelope.failure(
                error=str(exc), step_mode=step_mode, workflow=workflow
            )
        )
        return
    try:
        envelope = runner.start(goal=raw_goal, title=title, constraints=list(constraint or []))
    except (ProviderError, ValueError, OSError) as exc:
        envelope = SessionEnvelope.failure(
            error=str(exc), step_mode=step_mode, workflow=workflow
        )
    _emit(envelope)


@session_app.command("reply")
def cmd_reply(
    run_id: Annotated[str, typer.Argument(help="Run ID returned by session start.")],
    message_arg: Annotated[
        str | None, typer.Argument(help="Free-form natural-language reply.")
    ] = None,
    message: MessageOpt = None,
    workspace: WorkspaceOpt = None,
    state_dir: StateDirOpt = None,
    workflow: ReplyWorkflowOpt = None,
    gate_policy: GatePolicyOpt = None,
    step_mode: StepModeOpt = StepMode.EXECUTE,
    provider: ProviderOpt = "auto",
    model: ModelOpt = None,
    base_url: BaseUrlOpt = None,
    stage_driver: StageDriverOpt = None,
    worker_driver: WorkerDriverOpt = None,
    reviewer_driver: ReviewerDriverOpt = None,
    monitor_model: MonitorModelOpt = None,
    pi_provider: PiProviderOpt = None,
    thinking: ThinkingLevelOpt = None,
    context_window: ContextWindowOpt = None,
) -> None:
    """Route a free-form reply into clarification, approval, revision, or reopen."""
    text = (message or message_arg or "").strip()
    try:
        runner = _build_runner(
            workspace=workspace,
            state_dir=state_dir,
            workflow=workflow,
            gate_policy=gate_policy,
            step_mode=step_mode,
            provider=provider,
            model=model,
            base_url=base_url,
            stage_driver=stage_driver,
            worker_driver=worker_driver,
            reviewer_driver=reviewer_driver,
            monitor_model=monitor_model,
            pi_provider=pi_provider,
            thinking_level=thinking,
            context_window=context_window,
            run_id=run_id,
        )
    except (ProviderError, ValueError, OSError) as exc:
        _emit(
            SessionEnvelope.failure(
                error=str(exc),
                run_id=run_id,
                step_mode=step_mode,
                workflow=workflow or "",
            )
        )
        return
    try:
        envelope = runner.reply(run_id=run_id, message=text)
    except (ProviderError, ValueError, OSError) as exc:
        envelope = SessionEnvelope.failure(
            error=str(exc),
            run_id=run_id,
            step_mode=step_mode,
            workflow=workflow or "",
        )
    _emit(envelope)


__all__ = ["build_session_provider", "session_app"]
