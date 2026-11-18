"""Lightweight stdlib HTTP + SSE server for the Ariadne Web Console (Milestone 4)."""

from __future__ import annotations

import contextlib
import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from olympus.core.contracts import (
    ApprovalTarget,
    BudgetLedger,
    BudgetPolicy,
    RunStatus,
    TaskNode,
    TaskStatus,
    new_id,
)
from olympus.core.store import ConcurrencyConflictError, NotFoundError, RunStore
from olympus.graph.loader import load_workflow
from olympus.stages.align import AlignStage
from olympus.stages.design import DesignStage
from olympus.stages.orchestrate import OrchestrateStage
from olympus.web.projection import build_run_projection, seed_demo_runs
from olympus.web.supervisor import RunSupervisor

_STATIC_DIR = Path(__file__).resolve().parent / "static"

# Only these extensions are ever served from /static/; everything else is a 404.
_STATIC_MIME_TYPES: dict[str, str] = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".svg": "image/svg+xml",
}


class AriadneApp:
    """Pure-Python request dispatcher backing the Ariadne HTTP/SSE console."""

    def __init__(
        self,
        *,
        store: RunStore,
        workspace_root: Path | str,
        artifacts_root: Path | None = None,
        supervisor: RunSupervisor | None = None,
    ) -> None:
        self.store = store
        self.workspace_root = Path(workspace_root).resolve()
        self.artifacts_root = (
            artifacts_root
            if artifacts_root is not None
            else self.workspace_root / ".olympus" / "runs"
        )
        self.supervisor = (
            supervisor
            if supervisor is not None
            else RunSupervisor(
                store=self.store,
                workspace_root=self.workspace_root,
                artifacts_root=self.artifacts_root,
            )
        )

    def list_runs_payload(self, *, limit: int = 50) -> dict[str, Any]:
        runs = self.store.list_runs(limit=limit)
        return {
            "ok": True,
            "runs": runs,
            "default_run_id": runs[0]["run_id"] if runs else None,
        }

    def get_run_payload(self, run_id: str) -> dict[str, Any]:
        proj = build_run_projection(
            self.store,
            self.workspace_root,
            run_id,
            artifacts_root=self.artifacts_root,
        )
        return {"ok": True, "projection": proj}

    def get_events_payload(self, run_id: str, *, after_seq: int = 0) -> dict[str, Any]:
        snap = self.store.get_run(run_id)
        events = [
            e.model_dump(mode="json") for e in self.store.list_events(run_id, after_seq=after_seq)
        ]
        return {
            "ok": True,
            "run_id": run_id,
            "state_revision": snap["state_revision"],
            "status": snap["status"].value,
            "events": events,
        }

    def execute_command(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Execute an idempotent HITL command and return updated projection."""
        action = str(body.get("action", "")).strip()
        if not action:
            raise ValueError("Missing required 'action' field")

        operation_id = str(body.get("operation_id") or new_id("op"))
        actor = str(body.get("actor") or "zeus")
        snap = self.store.get_run(run_id)
        expected_rev_raw = body.get("expected_state_revision")
        expected_rev = (
            int(expected_rev_raw) if expected_rev_raw is not None else int(snap["state_revision"])
        )

        if action in ("pause", "resume"):
            target_status = RunStatus.PAUSED if action == "pause" else RunStatus.RUNNING
            evt_type = "run.paused" if action == "pause" else "run.resumed"

            def _toggle_status(conn: Any, cur_rev: int) -> dict[str, Any]:
                new_rev = cur_rev + 1
                conn.execute(
                    "UPDATE runs SET status = ?, state_revision = ?, updated_at = datetime('now') "
                    "WHERE run_id = ?",
                    (target_status.value, new_rev, run_id),
                )
                self.store.append_event_in_tx(
                    conn,
                    run_id=run_id,
                    stage="orchestrate",
                    event_type=evt_type,
                    actor=actor,
                    payload={"status": target_status.value},
                    state_revision=new_rev,
                )
                return {
                    "run_id": run_id,
                    "status": target_status.value,
                    "state_revision": new_rev,
                }

            cmd_result = self.store.apply_idempotent_command(
                operation_id=operation_id,
                run_id=run_id,
                actor=actor,
                action=action,
                expected_state_revision=expected_rev,
                payload={"action": action},
                handler=_toggle_status,
            )
        elif action == "approve_spec":
            if snap["state_revision"] != expected_rev:
                raise ConcurrencyConflictError(
                    f"State revision mismatch: expected {expected_rev}, "
                    f"got {snap['state_revision']}"
                )
            align = AlignStage(
                store=self.store,
                provider=self.supervisor.resolve_provider(run_id),
                workspace_root=self.workspace_root,
                trace_dir=self.artifacts_root,
            )
            spec = align.confirm_spec(
                run_id=run_id,
                approver=actor,
                accept_risk=bool(body.get("accept_risk", False)),
                reason=str(body.get("reason", "")),
            )
            cmd_result = {
                "run_id": run_id,
                "spec_id": spec.spec_id,
                "spec_hash": spec.content_hash,
                "status": spec.status.value,
            }
        elif action == "approve_design":
            if snap["state_revision"] != expected_rev:
                raise ConcurrencyConflictError(
                    f"State revision mismatch: expected {expected_rev}, "
                    f"got {snap['state_revision']}"
                )
            design_stage = DesignStage(
                self.store,
                self.supervisor.resolve_provider(run_id),
                self.workspace_root,
                artifacts_root=self.artifacts_root,
            )
            record = design_stage.approve_design(
                run_id=run_id,
                approver=actor,
                reason=str(body.get("reason", "")),
            )
            cmd_result = record.model_dump(mode="json")
        elif action == "approve_plan":
            if snap["state_revision"] != expected_rev:
                raise ConcurrencyConflictError(
                    f"State revision mismatch: expected {expected_rev}, "
                    f"got {snap['state_revision']}"
                )
            orch = OrchestrateStage(
                store=self.store,
                provider=self.supervisor.resolve_provider(run_id),
                workspace_root=self.workspace_root,
                artifacts_root=self.artifacts_root,
            )
            rec = orch.approve_plan(
                run_id=run_id,
                approver=actor,
                reason=str(body.get("reason", "")),
            )
            cmd_result = rec.model_dump(mode="json")
        elif action == "clarify":
            if snap["state_revision"] != expected_rev:
                raise ConcurrencyConflictError(
                    f"State revision mismatch: expected {expected_rev}, "
                    f"got {snap['state_revision']}"
                )
            align = AlignStage(
                store=self.store,
                provider=self.supervisor.resolve_provider(run_id),
                workspace_root=self.workspace_root,
                trace_dir=self.artifacts_root,
            )
            spec = align.apply_clarifications(
                run_id=run_id,
                answers=dict(body.get("answers") or {}),
                resolutions=dict(body.get("resolutions") or {}),
                extra_out_of_scope=list(body.get("out_of_scope") or []),
            )
            cmd_result = {
                "run_id": run_id,
                "version": spec.version,
                "spec_hash": spec.content_hash,
            }
        elif action == "retry_task":
            task_id = str(body.get("task_id", "")).strip()
            if not task_id:
                raise ValueError("retry_task requires non-empty 'task_id'")
            hint = str(body.get("hint", "")).strip()
            orch = OrchestrateStage(
                store=self.store,
                provider=self.supervisor.resolve_provider(run_id),
                workspace_root=self.workspace_root,
                artifacts_root=self.artifacts_root,
            )
            cmd_result = orch.retry_blocked_task(
                operation_id=operation_id,
                run_id=run_id,
                task_id=task_id,
                expected_state_revision=expected_rev,
                hint=hint,
                actor=actor,
            )
        elif action == "accept_risk":
            task_id = str(body.get("task_id", "")).strip()
            reason = str(body.get("reason", "")).strip()
            if not task_id:
                raise ValueError("accept_risk requires non-empty 'task_id'")
            if not reason:
                raise ValueError("accept_risk requires a non-empty 'reason'")
            orch = OrchestrateStage(
                store=self.store,
                provider=self.supervisor.resolve_provider(run_id),
                workspace_root=self.workspace_root,
                artifacts_root=self.artifacts_root,
            )
            cmd_result = orch.accept_task_risk(
                operation_id=operation_id,
                run_id=run_id,
                task_id=task_id,
                expected_state_revision=expected_rev,
                reason=reason,
                actor=actor,
            )
        elif action == "reassign_human":
            task_id = str(body.get("task_id", "")).strip()
            note = (
                str(body.get("note", "")).strip()
                or "已改派给人工工程师处理（提交后仍须经独立验收）"
            )
            if not task_id:
                raise ValueError("reassign_human requires non-empty 'task_id'")
            orch = OrchestrateStage(
                store=self.store,
                provider=self.supervisor.resolve_provider(run_id),
                workspace_root=self.workspace_root,
                artifacts_root=self.artifacts_root,
            )

            def _mutate_reassign(t: TaskNode) -> TaskNode:
                if t.status not in (TaskStatus.READY, TaskStatus.BLOCKED, TaskStatus.FAILED):
                    raise ValueError(
                        f"Task {t.task_id} in state {t.status.value} cannot be reassigned"
                    )
                return t.model_copy(
                    update={
                        "status": TaskStatus.READY,
                        "last_error": f"[HUMAN_ASSIGNED] {note}",
                    }
                )

            cmd_result = orch._apply_task_command(
                operation_id=operation_id,
                run_id=run_id,
                task_id=task_id,
                expected_state_revision=expected_rev,
                action="reassign_human",
                event_type="task.reassigned_human",
                actor=actor,
                payload={"task_id": task_id, "note": note},
                task_mutator=_mutate_reassign,
                result_status=TaskStatus.READY.value,
            )
        elif action == "retry_workflow":
            if snap["state_revision"] != expected_rev:
                raise ConcurrencyConflictError(
                    f"State revision mismatch: expected {expected_rev}, "
                    f"got {snap['state_revision']}"
                )
            if snap["status"] == RunStatus.FAILED:
                self.store.set_run_status(
                    run_id,
                    RunStatus.RUNNING,
                    actor=actor,
                    stage="graph",
                    detail={"action": "retry_workflow"},
                )
            cmd_result = self.supervisor.resume_run(
                run_id, sync=bool(body.get("sync", False))
            )
        elif action == "promote_to_execute":
            target_workflow = str(body.get("workflow_name") or "full_sdlc_tdd").strip()
            target_def = load_workflow(target_workflow)
            if target_def.stop_after != "execute" or target_def.node("execute") is None:
                raise ValueError(
                    f"Target workflow '{target_workflow}' must support stop_after='execute'"
                )
            cur_def = load_workflow(snap["workflow_name"])
            if cur_def.stop_after == "execute" and snap["status"] == RunStatus.COMPLETED:
                raise ValueError(f"Run {run_id} already completed execution")
            if snap["status"] in (
                RunStatus.RUNNING,
                RunStatus.ALIGNING,
                RunStatus.DESIGNING,
                RunStatus.PLANNING,
            ):
                raise ValueError(f"Run {run_id} is currently active ({snap['status'].value})")
            plan = snap["plan"]
            if plan is None:
                raise ValueError(f"Run {run_id} has no TaskPlan to execute")
            if not self.store.is_approved(
                run_id=run_id,
                target=ApprovalTarget.TASK_PLAN,
                target_id=plan.plan_id,
                target_hash=plan.content_hash,
            ):
                raise ValueError(f"TaskPlan {plan.plan_id} must be approved before execution")
            spec = snap["spec"]
            if spec is None:
                raise ValueError(f"Run {run_id} has no GoalSpec required for execution")
            if not self.store.is_approved(
                run_id=run_id,
                target=ApprovalTarget.GOAL_SPEC,
                target_id=spec.spec_id,
                target_hash=spec.content_hash,
            ):
                raise ValueError(f"GoalSpec {spec.spec_id} must be approved before execution")

            policy: BudgetPolicy = snap["budget_policy"]
            ledger: BudgetLedger = snap["budget_ledger"]
            min_calls = ledger.llm_calls_used + max(60, len(plan.tasks) * 25)
            new_max_calls = max(policy.max_llm_calls, min_calls)
            if body.get("max_llm_calls") is not None:
                new_max_calls = max(new_max_calls, int(body["max_llm_calls"]))
            min_tokens = ledger.total_tokens + max(500_000, len(plan.tasks) * 250_000)
            updated_policy = policy.model_copy(
                update={
                    "max_llm_calls": new_max_calls,
                    "max_total_tokens": max(policy.max_total_tokens, min_tokens),
                }
            )
            auto_adv = bool(body.get("auto_advance", True))
            target_status = RunStatus.RUNNING if auto_adv else RunStatus.AWAITING_HUMAN

            def _promote_tx(conn: Any, cur_rev: int) -> dict[str, Any]:
                new_rev = self.store.promote_workflow_in_tx(
                    conn,
                    run_id=run_id,
                    cur_rev=cur_rev,
                    workflow_name=target_workflow,
                    status=target_status,
                    budget_policy=updated_policy,
                    actor=actor,
                    payload={
                        "from_workflow": snap["workflow_name"],
                        "to_workflow": target_workflow,
                        "budget_max_calls": updated_policy.max_llm_calls,
                        "budget_max_tokens": updated_policy.max_total_tokens,
                    },
                )
                return {
                    "run_id": run_id,
                    "from_workflow": snap["workflow_name"],
                    "workflow": target_workflow,
                    "status": target_status.value,
                    "budget_max_calls": updated_policy.max_llm_calls,
                    "state_revision": new_rev,
                }

            cmd_result = self.store.apply_idempotent_command(
                operation_id=operation_id,
                run_id=run_id,
                actor=actor,
                action="promote_to_execute",
                expected_state_revision=expected_rev,
                payload={"workflow_name": target_workflow},
                handler=_promote_tx,
            )
        else:
            raise ValueError(f"Unsupported HITL action: {action}")

        if (
            action
            in ("clarify", "approve_spec", "approve_design", "approve_plan", "promote_to_execute")
            and bool(body.get("auto_advance", True))
            and not cmd_result.get("deduplicated")
            and (action == "promote_to_execute" or self.supervisor.is_supervised(run_id))
        ):
            cmd_result = {
                **cmd_result,
                "workflow": self.supervisor.resume_run(
                    run_id, sync=bool(body.get("sync", False))
                ),
            }

        projection = build_run_projection(
            self.store,
            self.workspace_root,
            run_id,
            artifacts_root=self.artifacts_root,
        )
        return {
            "ok": True,
            "command_result": cmd_result,
            "projection": projection,
        }


def create_http_server(
    app: AriadneApp,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Create a ThreadingHTTPServer bound to (host, port) serving the Ariadne console."""

    class AriadneRequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _send_bytes(self, content_type: str, body: bytes) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html_bytes: bytes) -> None:
            self._send_bytes("text/html; charset=utf-8", html_bytes)

        def _serve_static(self, relpath: str) -> None:
            """Serve one whitelisted asset from static/, refusing any escape of the root."""
            root = _STATIC_DIR.resolve()
            requested = unquote(relpath)
            try:
                candidate: Path | None = (root / requested).resolve()
            except (OSError, ValueError):
                candidate = None
            if (
                candidate is None
                or not candidate.is_relative_to(root)
                or not candidate.is_file()
            ):
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": f"Unknown static asset: {requested}"},
                )
                return
            content_type = _STATIC_MIME_TYPES.get(candidate.suffix.lower())
            if content_type is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": f"Unsupported static type: {candidate.suffix}"},
                )
                return
            self._send_bytes(content_type, candidate.read_bytes())

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            qs = parse_qs(parsed.query)

            try:
                if path.startswith("/static/"):
                    self._serve_static(path[len("/static/") :])
                    return

                if path in ("/", "/index.html"):
                    index_file = _STATIC_DIR / "index.html"
                    if not index_file.exists():
                        self._send_json(
                            HTTPStatus.NOT_FOUND,
                            {"ok": False, "error": "Console static/index.html not found"},
                        )
                        return
                    self._send_html(index_file.read_bytes())
                    return

                if path == "/favicon.ico":
                    self._send_bytes("image/x-icon", b"")
                    return

                if path == "/healthz":
                    self._send_json(
                        HTTPStatus.OK,
                        {"ok": True, "service": "olympus-ariadne", "version": "0.1.0"},
                    )
                    return

                if path == "/api/runs":
                    limit = int(qs.get("limit", ["50"])[0])
                    self._send_json(HTTPStatus.OK, app.list_runs_payload(limit=limit))
                    return

                if path == "/api/providers":
                    self._send_json(
                        HTTPStatus.OK,
                        {"ok": True, "providers": app.supervisor.list_providers_status()},
                    )
                    return

                if path == "/api/workflows":
                    self._send_json(
                        HTTPStatus.OK,
                        {"ok": True, "workflows": app.supervisor.list_workflows()},
                    )
                    return

                if path.startswith("/api/runs/"):
                    parts = [p for p in path.split("/") if p]
                    # parts: ['api', 'runs', '<run_id>', ...]
                    if len(parts) == 3:
                        run_id = parts[2]
                        self._send_json(HTTPStatus.OK, app.get_run_payload(run_id))
                        return
                    if len(parts) == 4 and parts[3] == "events":
                        run_id = parts[2]
                        after_seq = int(qs.get("after_seq", ["0"])[0])
                        self._send_json(
                            HTTPStatus.OK,
                            app.get_events_payload(run_id, after_seq=after_seq),
                        )
                        return
                    if len(parts) == 4 and parts[3] == "stream":
                        run_id = parts[2]
                        after_seq = int(qs.get("after_seq", ["0"])[0])
                        once = qs.get("once", ["0"])[0] == "1"
                        self._handle_sse(run_id, after_seq=after_seq, once=once)
                        return

                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": f"Unknown route: {path}"},
                )
            except NotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": str(exc)})
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": str(exc)},
                )

        def _handle_sse(self, run_id: str, *, after_seq: int = 0, once: bool = False) -> None:
            # Validate run exists before starting stream
            payload = app.get_events_payload(run_id, after_seq=after_seq)
            self.close_connection = True
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def _write_sse(event_name: str, data: dict[str, Any]) -> None:
                chunk = f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()

            try:
                _write_sse("state", payload)
                if once:
                    return

                last_rev = int(payload["state_revision"])
                events = payload["events"]
                cur_seq = int(events[-1]["event_seq"]) if events else after_seq
                ticks = 0
                while True:
                    time.sleep(0.5)
                    ticks += 1
                    nxt = app.get_events_payload(run_id, after_seq=cur_seq)
                    if int(nxt["state_revision"]) != last_rev or nxt["events"]:
                        last_rev = int(nxt["state_revision"])
                        if nxt["events"]:
                            cur_seq = int(nxt["events"][-1]["event_seq"])
                        _write_sse("state", nxt)
                    elif ticks % 10 == 0:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            finally:
                self.close_connection = True

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/")
            try:
                content_len = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
                body = json.loads(raw_body.decode("utf-8") or "{}")

                if path == "/api/seed-demo":
                    seeded = seed_demo_runs(
                        app.store,
                        app.workspace_root,
                        artifacts_root=app.artifacts_root,
                    )
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "ok": True,
                            "seeded": seeded,
                            **app.list_runs_payload(),
                        },
                    )
                    return

                parts = [p for p in path.split("/") if p]
                if len(parts) == 2 and parts[0] == "api" and parts[1] == "runs":
                    raw_input = str(body.get("raw_input") or body.get("goal") or "").strip()
                    if not raw_input:
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {
                                "ok": False,
                                "error": "POST /api/runs requires a non-empty 'raw_input'",
                            },
                        )
                        return
                    started = app.supervisor.start_run(
                        title=str(body.get("title") or raw_input[:60]),
                        raw_input=raw_input,
                        workflow_name=str(body.get("workflow_name") or "product_blueprint"),
                        provider_config=dict(body.get("provider_config") or {}),
                        auto_approve=bool(body.get("auto_approve", False)),
                        sync=bool(body.get("sync", False)),
                    )
                    self._send_json(HTTPStatus.OK, {"ok": True, **started})
                    return

                if (
                    len(parts) == 4
                    and parts[0] == "api"
                    and parts[1] == "runs"
                    and parts[3] == "command"
                ):
                    run_id = parts[2]
                    res = app.execute_command(run_id, body)
                    self._send_json(HTTPStatus.OK, res)
                    return

                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": f"Unknown POST route: {path}"},
                )
            except NotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": str(exc)})
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            except ConcurrencyConflictError as exc:
                conflict_payload: dict[str, Any] = {
                    "ok": False,
                    "conflict": True,
                    "error": str(exc),
                }
                req_parts = [p for p in path.split("/") if p]
                if len(req_parts) == 4 and req_parts[0] == "api" and req_parts[1] == "runs":
                    with contextlib.suppress(Exception):
                        conflict_payload["projection"] = build_run_projection(
                            app.store,
                            app.workspace_root,
                            req_parts[2],
                            artifacts_root=app.artifacts_root,
                        )
                self._send_json(HTTPStatus.CONFLICT, conflict_payload)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": str(exc)},
                )

    return ThreadingHTTPServer((host, port), AriadneRequestHandler)


def serve_ariadne(
    *,
    store: RunStore,
    workspace_root: Path | str,
    host: str = "127.0.0.1",
    port: int = 8765,
    seed_demo: bool = False,
) -> None:
    """Run the Ariadne console HTTP server in the foreground."""
    ws = Path(workspace_root).resolve()
    if seed_demo:
        seed_demo_runs(store, ws)
    app = AriadneApp(store=store, workspace_root=ws)
    app.supervisor.recover_orphaned_runs()
    httpd = create_http_server(app, host=host, port=port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
