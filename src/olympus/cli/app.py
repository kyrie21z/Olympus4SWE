"""Typer CLI entry point driving the end-to-end Olympus4SWE closed loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from olympus.core.contracts import RunMode, new_id
from olympus.core.store import NotFoundError, RunStore
from olympus.providers.factory import build_provider
from olympus.stages.align import AlignStage
from olympus.stages.design import DesignStage
from olympus.stages.orchestrate import OrchestrateStage

app = typer.Typer(
    name="olympus",
    help="Olympus4SWE Agent-Native SWE system CLI.",
    no_args_is_help=True,
)

WorkspaceOpt = Annotated[
    Path | None,
    typer.Option("--workspace", "-w", help="Target repository root (defaults to cwd)."),
]
DbOpt = Annotated[
    Path | None,
    typer.Option(
        "--db",
        help="Path to SQLite state DB (defaults to <workspace>/.olympus/olympus.db).",
    ),
]
ProviderOpt = Annotated[
    str,
    typer.Option(
        "--provider",
        help="Registered provider ID, 'auto', or use --base-url for custom endpoints.",
    ),
]
ModelOpt = Annotated[
    str | None,
    typer.Option("--model", help="Override LLM model name."),
]
BaseUrlOpt = Annotated[
    str | None,
    typer.Option("--base-url", help="Override OpenAI-compatible/Anthropic base URL."),
]


def _resolve_workspace(workspace: Path | None) -> Path:
    return (workspace if workspace is not None else Path.cwd()).resolve()


def _resolve_db(workspace: Path, db: Path | None) -> Path:
    return db if db is not None else workspace / ".olympus" / "olympus.db"


def _parse_kv_pairs(items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"Expected KEY=VALUE format, got: {item}")
        key, val = item.split("=", 1)
        parsed[key.strip()] = val.strip()
    return parsed


@app.command("align")
def cmd_align(
    goal: Annotated[str, typer.Argument(help="Raw project/feature goal.")],
    title: Annotated[str | None, typer.Option("--title", help="Optional run title.")] = None,
    constraint: Annotated[list[str] | None, typer.Option("--constraint", "-c")] = None,
    critique: Annotated[bool, typer.Option("--critique", help="Enable parallel critique.")] = False,
    mode: Annotated[RunMode, typer.Option("--mode")] = RunMode.EXECUTION,
    workspace: WorkspaceOpt = None,
    db: DbOpt = None,
    provider: ProviderOpt = "auto",
    model: ModelOpt = None,
    base_url: BaseUrlOpt = None,
) -> None:
    """Stage 1: Probe repo, draft GoalSpec, and surface ambiguities/critiques."""
    ws = _resolve_workspace(workspace)
    store = RunStore(_resolve_db(ws, db))
    llm = build_provider(provider=provider, model=model, base_url=base_url)
    run_id = store.create_run(title=title or goal[:60], mode=mode)
    stage = AlignStage(
        store=store,
        provider=llm,
        workspace_root=ws,
        trace_dir=ws / ".olympus" / "runs",
    )
    spec = stage.propose_spec(
        run_id=run_id,
        raw_goal=goal,
        constraints=list(constraint or []),
        enable_critique=critique,
    )
    typer.echo(
        json.dumps(
            {
                "run_id": run_id,
                "spec": spec.model_dump(mode="json"),
                "unresolved_blockers": spec.unresolved_blockers(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("clarify")
def cmd_clarify(
    run_id: Annotated[str, typer.Argument(help="Run ID.")],
    answer: Annotated[
        list[str] | None, typer.Option("--answer", "-a", help="AMB_ID=answer")
    ] = None,
    resolve: Annotated[
        list[str] | None, typer.Option("--resolve", "-r", help="CRIT_ID=note")
    ] = None,
    out_of_scope: Annotated[list[str] | None, typer.Option("--out-of-scope")] = None,
    workspace: WorkspaceOpt = None,
    db: DbOpt = None,
) -> None:
    """Stage 1: Apply human answers to open ambiguities / blocking critiques."""
    ws = _resolve_workspace(workspace)
    store = RunStore(_resolve_db(ws, db))
    # Clarifications are pure deterministic updates; provider is unused here.
    from olympus.providers.base import MockLLMProvider

    stage = AlignStage(store=store, provider=MockLLMProvider(), workspace_root=ws)
    spec = stage.apply_clarifications(
        run_id=run_id,
        answers=_parse_kv_pairs(list(answer or [])),
        resolutions=_parse_kv_pairs(list(resolve or [])),
        extra_out_of_scope=list(out_of_scope or []),
    )
    typer.echo(
        json.dumps(
            {
                "run_id": run_id,
                "version": spec.version,
                "spec_hash": spec.content_hash,
                "unresolved_blockers": spec.unresolved_blockers(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("approve-spec")
def cmd_approve_spec(
    run_id: Annotated[str, typer.Argument(help="Run ID.")],
    approver: Annotated[str, typer.Option("--approver")] = "zeus",
    accept_risk: Annotated[bool, typer.Option("--accept-risk")] = False,
    reason: Annotated[str, typer.Option("--reason")] = "",
    workspace: WorkspaceOpt = None,
    db: DbOpt = None,
) -> None:
    """Gate 1: Freeze and approve GoalSpec."""
    ws = _resolve_workspace(workspace)
    store = RunStore(_resolve_db(ws, db))
    from olympus.providers.base import MockLLMProvider

    stage = AlignStage(store=store, provider=MockLLMProvider(), workspace_root=ws)
    spec = stage.confirm_spec(
        run_id=run_id,
        approver=approver,
        accept_risk=accept_risk,
        reason=reason,
    )
    typer.echo(
        json.dumps(
            {
                "run_id": run_id,
                "spec_id": spec.spec_id,
                "spec_hash": spec.content_hash,
                "status": spec.status.value,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("design")
def cmd_design(
    run_id: Annotated[str, typer.Argument(help="Run ID.")],
    auto_approve: Annotated[bool, typer.Option("--auto-approve")] = False,
    approver: Annotated[str, typer.Option("--approver")] = "zeus",
    workspace: WorkspaceOpt = None,
    db: DbOpt = None,
    provider: ProviderOpt = "auto",
    model: ModelOpt = None,
    base_url: BaseUrlOpt = None,
) -> None:
    """Stage 1.5: Synthesize and validate a white-box DesignSpec for a frozen GoalSpec."""
    ws = _resolve_workspace(workspace)
    store = RunStore(_resolve_db(ws, db))
    llm = build_provider(provider=provider, model=model, base_url=base_url)
    stage = DesignStage(
        store,
        llm,
        ws,
        artifacts_root=ws / ".olympus" / "runs",
    )
    design = stage.propose_design(run_id, auto_approve=auto_approve, approver=approver)
    typer.echo(json.dumps(design.model_dump(mode="json"), ensure_ascii=False, indent=2))


@app.command("approve-design")
def cmd_approve_design(
    run_id: Annotated[str, typer.Argument(help="Run ID.")],
    approver: Annotated[str, typer.Option("--approver")] = "zeus",
    reason: Annotated[str, typer.Option("--reason")] = "",
    workspace: WorkspaceOpt = None,
    db: DbOpt = None,
) -> None:
    """Gate 1.5: Freeze and approve the current DesignSpec."""
    ws = _resolve_workspace(workspace)
    store = RunStore(_resolve_db(ws, db))
    from olympus.providers.base import MockLLMProvider

    stage = DesignStage(store, MockLLMProvider(), ws)
    record = stage.approve_design(run_id, approver=approver, reason=reason)
    typer.echo(record.model_dump_json(indent=2))


@app.command("workflows")
def cmd_workflows(
    workspace: WorkspaceOpt = None,
) -> None:
    """List builtin YAML workflows with their bound graph topology."""
    from olympus.graph import describe_workflow, list_builtin_workflows, load_workflow

    _resolve_workspace(workspace)
    payload = [
        describe_workflow(load_workflow(name)) for name in list_builtin_workflows()
    ]
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("plan")
def cmd_plan(
    run_id: Annotated[str, typer.Argument(help="Run ID.")],
    auto_approve: Annotated[bool, typer.Option("--auto-approve")] = False,
    workspace: WorkspaceOpt = None,
    db: DbOpt = None,
    provider: ProviderOpt = "auto",
    model: ModelOpt = None,
    base_url: BaseUrlOpt = None,
) -> None:
    """Stage 2: Generate and validate TaskPlan DAG from confirmed GoalSpec."""
    ws = _resolve_workspace(workspace)
    store = RunStore(_resolve_db(ws, db))
    llm = build_provider(provider=provider, model=model, base_url=base_url)
    stage = OrchestrateStage(store=store, provider=llm, workspace_root=ws)
    plan = stage.propose_plan(run_id=run_id, auto_approve=auto_approve)
    typer.echo(json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2))


@app.command("approve-plan")
def cmd_approve_plan(
    run_id: Annotated[str, typer.Argument(help="Run ID.")],
    approver: Annotated[str, typer.Option("--approver")] = "zeus",
    reason: Annotated[str, typer.Option("--reason")] = "",
    workspace: WorkspaceOpt = None,
    db: DbOpt = None,
) -> None:
    """Gate 2: Approve TaskPlan before execution."""
    ws = _resolve_workspace(workspace)
    store = RunStore(_resolve_db(ws, db))
    from olympus.providers.base import MockLLMProvider

    stage = OrchestrateStage(store=store, provider=MockLLMProvider(), workspace_root=ws)
    record = stage.approve_plan(run_id=run_id, approver=approver, reason=reason)
    typer.echo(record.model_dump_json(indent=2))


@app.command("run")
def cmd_run(
    run_id: Annotated[str, typer.Argument(help="Run ID.")],
    apply: Annotated[
        bool, typer.Option("--apply", help="Sync verified files to workspace.")
    ] = False,
    workspace: WorkspaceOpt = None,
    db: DbOpt = None,
    provider: ProviderOpt = "auto",
    model: ModelOpt = None,
    base_url: BaseUrlOpt = None,
) -> None:
    """Stage 2: Execute approved TaskPlan with sandbox isolation and cold-start review."""
    ws = _resolve_workspace(workspace)
    store = RunStore(_resolve_db(ws, db))
    llm = build_provider(provider=provider, model=model, base_url=base_url)
    stage = OrchestrateStage(store=store, provider=llm, workspace_root=ws)
    snap = stage.run_execution(run_id=run_id, apply_to_workspace=apply)
    typer.echo(
        json.dumps(
            {
                "run_id": run_id,
                "status": snap["status"].value,
                "state_revision": snap["state_revision"],
                "tasks": [
                    {
                        "task_id": t.task_id,
                        "status": t.status.value,
                        "attempt": t.attempt,
                        "review_cycle": t.review_cycle,
                        "last_error": t.last_error,
                    }
                    for t in (snap["plan"].tasks if snap["plan"] else [])
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("retry")
def cmd_retry(
    run_id: Annotated[str, typer.Argument(help="Run ID.")],
    task_id: Annotated[str, typer.Argument(help="Blocked or failed task ID.")],
    expected_revision: Annotated[int, typer.Option("--revision", min=1)],
    hint: Annotated[str, typer.Option("--hint")] = "",
    operation_id: Annotated[str | None, typer.Option("--op-id")] = None,
    workspace: WorkspaceOpt = None,
    db: DbOpt = None,
) -> None:
    """Reset a blocked/failed task back to READY."""
    ws = _resolve_workspace(workspace)
    store = RunStore(_resolve_db(ws, db))
    from olympus.providers.base import MockLLMProvider

    stage = OrchestrateStage(store=store, provider=MockLLMProvider(), workspace_root=ws)
    res = stage.retry_blocked_task(
        operation_id=operation_id or new_id("op"),
        run_id=run_id,
        task_id=task_id,
        expected_state_revision=expected_revision,
        hint=hint,
    )
    typer.echo(json.dumps(res, ensure_ascii=False, indent=2))


@app.command("accept-risk")
def cmd_accept_risk(
    run_id: Annotated[str, typer.Argument(help="Run ID.")],
    task_id: Annotated[str, typer.Argument(help="Blocked task ID.")],
    expected_revision: Annotated[int, typer.Option("--revision", min=1)],
    reason: Annotated[str, typer.Option("--reason")],
    operation_id: Annotated[str | None, typer.Option("--op-id")] = None,
    workspace: WorkspaceOpt = None,
    db: DbOpt = None,
) -> None:
    """Accept risk to close a blocked task with an explicit audit reason."""
    ws = _resolve_workspace(workspace)
    store = RunStore(_resolve_db(ws, db))
    from olympus.providers.base import MockLLMProvider

    stage = OrchestrateStage(store=store, provider=MockLLMProvider(), workspace_root=ws)
    res = stage.accept_task_risk(
        operation_id=operation_id or new_id("op"),
        run_id=run_id,
        task_id=task_id,
        expected_state_revision=expected_revision,
        reason=reason,
    )
    typer.echo(json.dumps(res, ensure_ascii=False, indent=2))


@app.command("list")
def list_runs(
    workspace: WorkspaceOpt = None,
    db: DbOpt = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 20,
) -> None:
    """List recorded runs."""
    ws = _resolve_workspace(workspace)
    store = RunStore(_resolve_db(ws, db))
    runs = store.list_runs(limit=limit)
    typer.echo(json.dumps(runs, ensure_ascii=False, indent=2))


@app.command("status")
def run_status(
    run_id: Annotated[str, typer.Argument(help="Run ID to inspect.")],
    workspace: WorkspaceOpt = None,
    db: DbOpt = None,
) -> None:
    """Show current state snapshot for a run."""
    ws = _resolve_workspace(workspace)
    store = RunStore(_resolve_db(ws, db))
    try:
        snap = store.get_run(run_id)
    except NotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    out = {
        "run_id": snap["run_id"],
        "title": snap["title"],
        "mode": snap["mode"].value,
        "status": snap["status"].value,
        "state_revision": snap["state_revision"],
        "budget_ledger": snap["budget_ledger"].model_dump(mode="json"),
        "spec_hash": snap["spec"].content_hash if snap["spec"] else None,
        "plan_hash": snap["plan"].content_hash if snap["plan"] else None,
        "tasks": [
            {
                "task_id": t.task_id,
                "status": t.status.value,
                "attempt": t.attempt,
                "review_cycle": t.review_cycle,
                "accepted_risk_reason": t.accepted_risk_reason,
            }
            for t in (snap["plan"].tasks if snap["plan"] else [])
        ],
    }
    typer.echo(json.dumps(out, ensure_ascii=False, indent=2))


@app.command("events")
def run_events(
    run_id: Annotated[str, typer.Argument(help="Run ID to inspect.")],
    after_seq: Annotated[int, typer.Option("--after-seq", min=0)] = 0,
    workspace: WorkspaceOpt = None,
    db: DbOpt = None,
) -> None:
    """Replay append-only events for a run."""
    ws = _resolve_workspace(workspace)
    store = RunStore(_resolve_db(ws, db))
    events = [e.model_dump(mode="json") for e in store.list_events(run_id, after_seq=after_seq)]
    typer.echo(json.dumps(events, ensure_ascii=False, indent=2))


@app.command("serve")
def cmd_serve(
    host: Annotated[str, typer.Option("--host", help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", min=1, max=65535)] = 8765,
    seed_demo: Annotated[
        bool,
        typer.Option("--seed-demo", help="Seed walkthrough demo runs if missing."),
    ] = True,
    workspace: WorkspaceOpt = None,
    db: DbOpt = None,
) -> None:
    """M4 Ariadne: Launch the event-sourced Web Console and HITL control server."""
    from olympus.web.server import serve_ariadne

    ws = _resolve_workspace(workspace)
    store = RunStore(_resolve_db(ws, db))
    typer.echo(f"Ariadne Web Console listening on http://{host}:{port}")
    serve_ariadne(
        store=store,
        workspace_root=ws,
        host=host,
        port=port,
        seed_demo=seed_demo,
    )

