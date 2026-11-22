"""FastAPI + uvicorn backend for the Ariadne Web Console (Milestone 4)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

import olympus.web.acceptance as delivery_review
from olympus.core.contracts import new_id
from olympus.core.control import RunControl
from olympus.core.store import ConcurrencyConflictError, NotFoundError, RunStore
from olympus.graph.loader import load_workflow
from olympus.web.demos import list_benchmark_presets, materialize_benchmark_demo, seed_demo_runs
from olympus.web.projection import build_run_projection, read_recorded_attempt
from olympus.web.schemas import (
    AttemptArtifactsResponse,
    CommandRequest,
    CommandResponse,
    ConflictResponse,
    CreateRunRequest,
    CreateRunResponse,
    DemosResponse,
    ErrorResponse,
    EventsResponse,
    HealthResponse,
    ProvidersResponse,
    RunListResponse,
    RunProjection,
    RunProjectionResponse,
    SeedDemoResponse,
    WorkflowsResponse,
)
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
        self.control = RunControl(self.store)
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
        self._retry_guard = threading.Lock()
        self._retry_locks: dict[str, threading.Lock] = {}

    def _retry_lock_for(self, operation_id: str) -> threading.Lock:
        with self._retry_guard:
            return self._retry_locks.setdefault(operation_id, threading.Lock())

    def list_runs_payload(self, *, limit: int = 50) -> dict[str, Any]:
        runs = self.store.list_runs(limit=limit)
        return {
            "ok": True,
            "runs": runs,
            "default_run_id": runs[0]["run_id"] if runs else None,
        }

    def _workspace_for(self, run_id: str) -> Path:
        return self.supervisor.resolve_workspace(run_id)

    def _projection_payload(self, run_id: str) -> dict[str, Any]:
        """Build and validate one projection before it crosses the HTTP seam."""
        proj = build_run_projection(
            self.store,
            self._workspace_for(run_id),
            run_id,
            artifacts_root=self.artifacts_root,
        )
        return RunProjection.model_validate(proj).model_dump(mode="json", by_alias=True)

    def get_run_payload(self, run_id: str) -> dict[str, Any]:
        proj = self._projection_payload(run_id)
        if proj["status"] == "running":
            previews = self.supervisor.live_activity.snapshot(run_id)
            for task in proj["dag"]["nodes"]:
                entry = previews.get(task["task_id"])
                if task["status"] == "running" and entry and entry["attempt"] == task["attempt"]:
                    task["live_progress"]["preview"] = entry["preview"]
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

    def get_attempt_payload(self, run_id: str, task_id: str, attempt: int) -> dict[str, Any]:
        plan = self.store.get_run(run_id)["plan"]
        artifacts = read_recorded_attempt(self.artifacts_root / run_id, plan, task_id, attempt)
        if artifacts is None:
            raise NotFoundError("Recorded task attempt not found")
        return {"ok": True, "artifacts": artifacts}

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
        target_hash: str | None = None
        if "target_hash" in body:
            raw_hash = body["target_hash"]
            target_hash = str(raw_hash) if raw_hash is not None else ""
        auto_advance = bool(body.get("auto_advance", True))
        sync_flag = bool(body.get("sync", False))
        # Omit default flags so legacy command rows replay cleanly while non-default
        # flag changes under the same operation_id still raise a payload conflict.
        dispatch_meta: dict[str, Any] = {}
        if not auto_advance:
            dispatch_meta["auto_advance"] = False
        if sync_flag:
            dispatch_meta["sync"] = True

        if action in ("pause", "resume"):
            cmd_result = self.control.set_paused(
                run_id,
                paused=action == "pause",
                operation_id=operation_id,
                expected_state_revision=expected_rev,
                actor=actor,
            )
            if action == "pause" and not cmd_result.get("deduplicated"):
                self.supervisor.cancel_active_turns(run_id, reason="human_paused")
        elif action == "approve_spec":
            spec_result = self.control.confirm_spec(
                run_id=run_id,
                approver=actor,
                accept_risk=bool(body.get("accept_risk", False)),
                reason=str(body.get("reason", "")),
                expected_revision=expected_rev,
                expected_target_hash=target_hash,
                operation_id=operation_id,
                command_meta=dispatch_meta,
            )
            cmd_result = (
                spec_result
                if isinstance(spec_result, dict)
                else {
                    "run_id": run_id,
                    "spec_id": spec_result.spec_id,
                    "spec_hash": spec_result.content_hash,
                    "status": spec_result.status.value,
                }
            )
        elif action == "approve_design":
            design_result = self.control.approve_design(
                run_id=run_id,
                approver=actor,
                reason=str(body.get("reason", "")),
                expected_revision=expected_rev,
                expected_target_hash=target_hash,
                operation_id=operation_id,
                command_meta=dispatch_meta,
            )
            cmd_result = (
                design_result
                if isinstance(design_result, dict)
                else design_result.model_dump(mode="json")
            )
        elif action == "approve_plan":
            plan_result = self.control.approve_plan(
                run_id=run_id,
                approver=actor,
                reason=str(body.get("reason", "")),
                expected_revision=expected_rev,
                expected_target_hash=target_hash,
                operation_id=operation_id,
                command_meta=dispatch_meta,
            )
            cmd_result = (
                plan_result
                if isinstance(plan_result, dict)
                else plan_result.model_dump(mode="json")
            )
        elif action == "clarify":
            clarify_result = self.control.clarify_spec(
                run_id=run_id,
                answers=dict(body.get("answers") or {}),
                resolutions=dict(body.get("resolutions") or {}),
                extra_out_of_scope=list(body.get("out_of_scope") or []),
                actor=actor,
                expected_revision=expected_rev,
                operation_id=operation_id,
                command_meta=dispatch_meta,
            )
            cmd_result = (
                clarify_result
                if isinstance(clarify_result, dict)
                else {
                    "run_id": run_id,
                    "version": clarify_result.version,
                    "spec_hash": clarify_result.content_hash,
                }
            )
        elif action == "retry_task":
            task_id = str(body.get("task_id", "")).strip()
            if not task_id:
                raise ValueError("retry_task requires non-empty 'task_id'")
            cmd_result = self.control.retry_task(
                run_id,
                task_id,
                operation_id=operation_id,
                expected_state_revision=expected_rev,
                hint=str(body.get("hint", "")).strip(),
                actor=actor,
                command_meta=dispatch_meta,
            )
        elif action == "accept_risk":
            task_id = str(body.get("task_id", "")).strip()
            reason = str(body.get("reason", "")).strip()
            if not task_id:
                raise ValueError("accept_risk requires non-empty 'task_id'")
            if not reason:
                raise ValueError("accept_risk requires a non-empty 'reason'")
            cmd_result = self.control.accept_task_risk(
                run_id,
                task_id,
                operation_id=operation_id,
                expected_state_revision=expected_rev,
                reason=reason,
                actor=actor,
            )
        elif action == "reassign_human":
            task_id = str(body.get("task_id", "")).strip()
            if not task_id:
                raise ValueError("reassign_human requires non-empty 'task_id'")
            cmd_result = self.control.reassign_task_human(
                run_id,
                task_id,
                operation_id=operation_id,
                expected_state_revision=expected_rev,
                note=str(body.get("note", "")).strip(),
                actor=actor,
            )
        elif action == "retry_workflow":
            retry_payload: dict[str, Any] = {"action": "retry_workflow"}
            if sync_flag:
                retry_payload["sync"] = True
            with self._retry_lock_for(operation_id):
                replay = self.store.lookup_command(
                    operation_id=operation_id,
                    run_id=run_id,
                    action="retry_workflow",
                    payload=retry_payload,
                )
                if replay is not None:
                    cmd_result = replay
                else:
                    pre_snap = self.store.get_run(run_id, expected_revision=expected_rev)
                    pre_rev = int(pre_snap["state_revision"])
                    workflow = self.supervisor.resume_run(run_id, sync=sync_flag)
                    cmd_result = self.control.complete_workflow_retry(
                        run_id,
                        operation_id=operation_id,
                        workflow=workflow,
                        pre_dispatch_revision=pre_rev,
                        actor=actor,
                        sync=sync_flag,
                    )
        elif action == "promote_to_execute":
            target_workflow = str(body.get("workflow_name") or "full_sdlc_tdd").strip()
            target_def = load_workflow(target_workflow)
            if target_def.stop_after != "execute" or target_def.node("execute") is None:
                raise ValueError(
                    f"Target workflow '{target_workflow}' must support stop_after='execute'"
                )
            cur_def = load_workflow(snap["workflow_name"])
            max_llm_calls = (
                int(body["max_llm_calls"]) if body.get("max_llm_calls") is not None else None
            )
            cmd_result = self.control.promote_to_execute(
                run_id,
                target_workflow=target_workflow,
                current_stop_after=cur_def.stop_after,
                operation_id=operation_id,
                expected_state_revision=expected_rev,
                max_llm_calls=max_llm_calls,
                auto_advance=auto_advance,
                sync=sync_flag,
                actor=actor,
            )
        else:
            raise ValueError(f"Unsupported HITL action: {action}")

        if (
            action
            in (
                "clarify",
                "approve_spec",
                "approve_design",
                "approve_plan",
                "promote_to_execute",
                "retry_task",
                "resume",
            )
            and auto_advance
            and not cmd_result.get("deduplicated")
            and (action == "promote_to_execute" or self.supervisor.is_supervised(run_id))
        ):
            cmd_result = {
                **cmd_result,
                "workflow": self.supervisor.resume_run(run_id, sync=sync_flag),
            }

        projection = self._projection_payload(run_id)
        return {
            "ok": True,
            "command_result": cmd_result,
            "projection": projection,
        }


class _NoStoreJSONResponse(JSONResponse):
    """JSON response that never allows a browser or proxy to cache API state."""

    def __init__(
        self,
        content: Any,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        media_type: str | None = None,
        background: Any = None,
    ) -> None:
        merged = dict(headers or {})
        merged.setdefault("Cache-Control", "no-store")
        super().__init__(
            content=content,
            status_code=status_code,
            headers=merged,
            media_type=media_type,
            background=background,
        )


def _error_response(status_code: int, message: str) -> _NoStoreJSONResponse:
    return _NoStoreJSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=message).model_dump(mode="json"),
    )


def _validation_summary(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "Invalid request payload"
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
    message = str(first.get("msg") or "Invalid request payload")
    return f"Invalid request: {location}: {message}" if location else f"Invalid request: {message}"


def _sse_frame(event_name: str, data: dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _serve_static(relpath: str) -> Response:
    """Serve one whitelisted asset from static/, refusing any escape of the root."""
    root = _STATIC_DIR.resolve()
    try:
        candidate: Path | None = (root / relpath).resolve()
    except (OSError, ValueError):
        candidate = None
    if candidate is None or not candidate.is_relative_to(root) or not candidate.is_file():
        raise NotFoundError(f"Unknown static asset: {relpath}")
    content_type = _STATIC_MIME_TYPES.get(candidate.suffix.lower())
    if content_type is None:
        raise NotFoundError(f"Unsupported static type: {candidate.suffix}")
    return Response(
        content=candidate.read_bytes(),
        headers={"Content-Type": content_type, "Cache-Control": "no-store"},
    )


def _index_response() -> Response:
    index_file = _STATIC_DIR / "index.html"
    if not index_file.exists():
        raise NotFoundError("Console static/index.html not found")
    return Response(
        content=index_file.read_bytes(),
        headers={"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store"},
    )


class _EventStreamResponse(StreamingResponse):
    media_type = "text/event-stream"


class _AriadneFastAPI(FastAPI):
    def openapi(self) -> dict[str, Any]:
        if self.openapi_schema is not None:
            return self.openapi_schema
        schema = super().openapi()
        for path_item in schema.get("paths", {}).values():
            if not isinstance(path_item, dict):
                continue
            for operation in path_item.values():
                if not isinstance(operation, dict):
                    continue
                responses = operation.get("responses")
                if isinstance(responses, dict):
                    responses.pop("422", None)
                    for code in ("400", "404", "409"):
                        err = responses.get(code)
                        if not isinstance(err, dict):
                            continue
                        content = err.get("content")
                        if (
                            isinstance(content, dict)
                            and "application/json" not in content
                            and len(content) == 1
                        ):
                            err["content"] = {"application/json": next(iter(content.values()))}
        components = schema.get("components", {}).get("schemas")
        if isinstance(components, dict):
            components.pop("HTTPValidationError", None)
            components.pop("ValidationError", None)
        self.openapi_schema = schema
        return schema


def create_fastapi_app(app: AriadneApp) -> FastAPI:
    """Create the ASGI application serving the Ariadne console over one service layer."""
    service = app
    api = _AriadneFastAPI(
        title="Olympus4SWE Ariadne API",
        version="0.1.0",
        default_response_class=_NoStoreJSONResponse,
        redirect_slashes=False,
    )

    @api.exception_handler(RequestValidationError)
    async def _handle_request_validation(
        request: Request, exc: RequestValidationError
    ) -> _NoStoreJSONResponse:
        return _error_response(400, _validation_summary(exc))

    @api.exception_handler(ValidationError)
    async def _handle_validation_error(
        request: Request, exc: ValidationError
    ) -> _NoStoreJSONResponse:
        return _error_response(500, str(exc))

    @api.exception_handler(NotFoundError)
    async def _handle_not_found(request: Request, exc: NotFoundError) -> _NoStoreJSONResponse:
        return _error_response(404, str(exc))

    @api.exception_handler(FileNotFoundError)
    async def _handle_file_not_found(
        request: Request, exc: FileNotFoundError
    ) -> _NoStoreJSONResponse:
        return _error_response(400, str(exc))

    @api.exception_handler(ConcurrencyConflictError)
    async def _handle_conflict(
        request: Request, exc: ConcurrencyConflictError
    ) -> _NoStoreJSONResponse:
        projection: dict[str, Any] | None = None
        parts = [part for part in request.url.path.split("/") if part]
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "runs":
            with contextlib.suppress(Exception):
                projection = service.get_run_payload(parts[2])["projection"]
        payload = ConflictResponse(
            error=str(exc),
            projection=(
                RunProjection.model_validate(projection) if projection is not None else None
            ),
        )
        return _NoStoreJSONResponse(
            status_code=409,
            content=payload.model_dump(mode="json", by_alias=True),
        )

    @api.exception_handler(ValueError)
    async def _handle_value_error(request: Request, exc: ValueError) -> _NoStoreJSONResponse:
        return _error_response(400, str(exc))

    @api.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> _NoStoreJSONResponse:
        return _error_response(500, str(exc))

    @api.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> _NoStoreJSONResponse:
        message = (
            f"Unknown route: {request.url.path}"
            if exc.status_code == 404
            else str(exc.detail)
        )
        return _error_response(exc.status_code, message)

    error_responses: dict[int | str, dict[str, Any]] = {400: {"model": ErrorResponse}}
    read_responses: dict[int | str, dict[str, Any]] = {
        **error_responses,
        404: {"model": ErrorResponse},
    }
    write_responses: dict[int | str, dict[str, Any]] = {
        **read_responses,
        409: {"model": ConflictResponse},
    }

    @api.get("/healthz", response_model=HealthResponse)
    def healthz() -> dict[str, Any]:
        return {"ok": True, "service": "olympus-ariadne", "version": "0.1.0"}

    @api.get("/api/runs", response_model=RunListResponse, responses=error_responses)
    def list_runs(limit: int = 50) -> dict[str, Any]:
        return service.list_runs_payload(limit=limit)

    @api.get("/api/providers", response_model=ProvidersResponse, responses=error_responses)
    def list_providers() -> dict[str, Any]:
        return {"ok": True, "providers": service.supervisor.list_providers_status()}

    @api.get("/api/workflows", response_model=WorkflowsResponse, responses=error_responses)
    def list_workflows() -> dict[str, Any]:
        return {"ok": True, "workflows": service.supervisor.list_workflows()}

    @api.get("/api/demos", response_model=DemosResponse, responses=error_responses)
    def list_demos() -> dict[str, Any]:
        return {"ok": True, "demos": list_benchmark_presets()}

    @api.get(
        "/api/runs/{run_id}",
        response_model=RunProjectionResponse,
        responses=read_responses,
    )
    def get_run(run_id: str) -> dict[str, Any]:
        return service.get_run_payload(run_id)

    @api.get(
        "/api/runs/{run_id}/events",
        response_model=EventsResponse,
        responses=read_responses,
    )
    def get_run_events(run_id: str, after_seq: int = 0) -> dict[str, Any]:
        return service.get_events_payload(run_id, after_seq=after_seq)

    @api.get(
        "/api/runs/{run_id}/task-artifacts/{task_id}/{attempt}",
        response_model=AttemptArtifactsResponse,
        responses=read_responses,
    )
    def get_task_artifacts(run_id: str, task_id: str, attempt: int) -> dict[str, Any]:
        return service.get_attempt_payload(run_id, task_id, attempt)

    @api.get(
        "/api/runs/{run_id}/delivery-log/{event_seq}/{position}",
        response_class=PlainTextResponse,
        responses=read_responses,
    )
    def get_delivery_log(run_id: str, event_seq: str, position: str) -> Response:
        projection = service.get_run_payload(run_id)["projection"]
        review = projection["delivery_acceptance"] or {}
        for row in review.get("criteria", []):
            evidence = row["evidence"]
            if (
                evidence
                and str(evidence["event_seq"]) == event_seq
                and str(evidence["check_position"]) == position
            ):
                log = delivery_review.read_delivery_log(service.artifacts_root / run_id, evidence)
                if log is not None:
                    return Response(
                        content=log,
                        headers={
                            "Content-Type": "text/plain; charset=utf-8",
                            "Cache-Control": "no-store",
                        },
                    )
        raise NotFoundError("No matching current delivery evidence")

    @api.post(
        "/api/runs",
        response_model=CreateRunResponse,
        responses=write_responses,
    )
    def create_run(payload: CreateRunRequest) -> dict[str, Any]:
        demo_id = (payload.demo_id or "").strip()
        preset: dict[str, Any] | None = None
        resolved_ws: Path | str | None = payload.workspace_root
        if demo_id:
            preset, demo_ws = materialize_benchmark_demo(
                demo_id,
                workspace_parent=service.workspace_root / ".olympus" / "workspaces",
            )
            if not resolved_ws:
                resolved_ws = demo_ws
        elif resolved_ws:
            ws_candidate = Path(str(resolved_ws)).resolve()
            if not ws_candidate.is_dir():
                raise FileNotFoundError(f"workspace_root not found: {resolved_ws}")
            resolved_ws = ws_candidate

        default_goal = str(preset["raw_input"]) if preset else ""
        raw_input = str(payload.raw_input or payload.goal or default_goal).strip()
        if not raw_input:
            raise ValueError("POST /api/runs requires a non-empty 'raw_input'")
        if payload.constraints is not None:
            constraints = [item.strip() for item in payload.constraints if item.strip()]
        elif preset:
            constraints = list(preset.get("constraints") or [])
        else:
            constraints = []
        default_title = str(preset["title"]) if preset else raw_input[:60]
        default_workflow = str(preset["default_workflow"]) if preset else "product_blueprint"
        started = service.supervisor.start_run(
            title=str(payload.title or default_title),
            raw_input=raw_input,
            workflow_name=str(payload.workflow_name or default_workflow),
            provider_config=payload.provider_config.model_dump(exclude_none=True),
            worker_driver=payload.worker_driver,
            reviewer_driver=payload.reviewer_driver,
            monitor_model=payload.monitor_model,
            auto_approve=payload.auto_approve,
            sync=payload.sync,
            constraints=constraints,
            workspace_root=resolved_ws,
        )
        return {"ok": True, **started}

    @api.post(
        "/api/runs/{run_id}/command",
        response_model=CommandResponse,
        responses=write_responses,
    )
    def run_command(run_id: str, payload: CommandRequest) -> dict[str, Any]:
        body = payload.model_dump(mode="json", exclude_unset=True)
        return service.execute_command(run_id, body)

    @api.post("/api/seed-demo", response_model=SeedDemoResponse, responses=write_responses)
    async def seed_demo(request: Request) -> dict[str, Any]:
        raw_body = await request.body()
        if raw_body.strip():
            try:
                parsed = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid JSON payload: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError("POST /api/seed-demo expects a JSON object")
        seeded = seed_demo_runs(
            service.store,
            service.workspace_root,
            artifacts_root=service.artifacts_root,
        )
        return {"ok": True, "seeded": seeded, **service.list_runs_payload()}

    @api.get(
        "/api/runs/{run_id}/stream",
        response_class=_EventStreamResponse,
        responses={
            200: {
                "description": "Server-Sent Events stream",
                "content": {"text/event-stream": {"schema": {"type": "string"}}},
            },
            **read_responses,
        },
    )
    async def stream_run(
        run_id: str, request: Request, after_seq: int = 0, once: int = 0
    ) -> _EventStreamResponse:
        initial = service.get_events_payload(run_id, after_seq=after_seq)
        return _EventStreamResponse(
            _sse_stream(
                service,
                request,
                run_id,
                initial=initial,
                after_seq=after_seq,
                once=once == 1,
            ),
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "close",
                "X-Accel-Buffering": "no",
            },
        )

    @api.get("/static/{relpath:path}", include_in_schema=False)
    def static_asset(relpath: str) -> Response:
        return _serve_static(relpath)

    @api.get("/", include_in_schema=False)
    @api.get("/index.html", include_in_schema=False)
    def index() -> Response:
        return _index_response()

    @api.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(
            content=b"",
            headers={"Content-Type": "image/x-icon", "Cache-Control": "no-store"},
        )

    return api


async def _sse_stream(
    service: AriadneApp,
    request: Request,
    run_id: str,
    *,
    initial: dict[str, Any],
    after_seq: int,
    once: bool,
) -> AsyncIterator[str]:
    """Async SSE frames matching the legacy wire contract exactly."""
    yield _sse_frame("state", initial)
    if once:
        return

    last_rev = int(initial["state_revision"])
    events = initial["events"]
    cur_seq = int(events[-1]["event_seq"]) if events else after_seq
    last_previews: dict[str, Any] = {}
    ticks = 0
    while True:
        await asyncio.sleep(0.5)
        if await request.is_disconnected():
            return
        ticks += 1
        nxt = await asyncio.to_thread(service.get_events_payload, run_id, after_seq=cur_seq)
        if int(nxt["state_revision"]) != last_rev or nxt["events"]:
            last_rev = int(nxt["state_revision"])
            if nxt["events"]:
                cur_seq = int(nxt["events"][-1]["event_seq"])
            yield _sse_frame("state", nxt)
        elif ticks % 10 == 0:
            yield ": heartbeat\n\n"
        previews = (
            service.supervisor.live_activity.snapshot(run_id)
            if nxt["status"] == "running"
            else {}
        )
        if previews != last_previews:
            yield _sse_frame("activity", {"run_id": run_id, "previews": previews})
            last_previews = previews


def serve_ariadne(
    *,
    store: RunStore,
    workspace_root: Path | str,
    host: str = "127.0.0.1",
    port: int = 8765,
    seed_demo: bool = False,
) -> None:
    """Run the Ariadne console uvicorn server in the foreground."""
    ws = Path(workspace_root).resolve()
    if seed_demo:
        seed_demo_runs(store, ws)
    app = AriadneApp(store=store, workspace_root=ws)
    app.supervisor.recover_orphaned_runs()
    uvicorn.run(create_fastapi_app(app), host=host, port=port)


__all__ = [
    "AriadneApp",
    "create_fastapi_app",
    "serve_ariadne",
]
