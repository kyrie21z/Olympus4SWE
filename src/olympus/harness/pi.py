"""AgentDriver adapter for the Pi coding agent CLI (``--mode rpc`` / ``--mode json``).

Pi is driven as an isolated subprocess inside the task sandbox: stdout JSONL is
pumped on a dedicated reader thread, tool milestones are normalized onto the
``AgentDriver`` observer seam, usage is settled into the run ledger, and
``steer`` / ``cancel`` act over the RPC stdin pipe and the process group.
"""

from __future__ import annotations

import json
import os
import select
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from olympus.core.contracts import new_id, utc_now
from olympus.core.fingerprint import content_hash
from olympus.core.store import RunStore
from olympus.harness.driver import ActiveTurnHandle, DriverEvent, DriverTurnRequest, TurnOutcome
from olympus.harness.receipt import build_turn_receipt
from olympus.tools.aci import build_sandbox_env

_INSTALL_HINT = (
    "Install Pi with `npm install -g --ignore-scripts @earendil-works/pi-coding-agent` "
    "or point OLYMPUS_PI_BIN at the pi executable."
)
_READ_ONLY_TOOLS = "read,grep,find,ls"
_FORBIDDEN_WRITE_TOOLS = "bash,write,edit,powershell"
_HELP_PROBE_TIMEOUT_SECONDS = 10.0
_SCOPE_GUARD_PATH = Path(__file__).with_name("pi_scope_guard.ts")
_TRACE_MILESTONE_TYPES = frozenset(
    {
        "tool_execution_start",
        "tool_execution_end",
        "message_end",
        "compaction_start",
        "compaction_end",
        "agent_settled",
        "extension_error",
    }
)


def _is_trace_milestone(record: dict[str, Any]) -> bool:
    record_type = record.get("type")
    if record_type in _TRACE_MILESTONE_TYPES:
        return True
    return record_type == "response" and not bool(record.get("success"))


class PiNotAvailableError(RuntimeError):
    """Raised when the configured Pi CLI is missing or unusable."""


_help_cache: dict[str, bool] = {}
_help_cache_lock = threading.Lock()


def _resolve_binary(binary: str | None) -> str:
    requested = (binary or os.getenv("OLYMPUS_PI_BIN", "") or "pi").strip()
    resolved = shutil.which(requested)
    if resolved is None:
        raise PiNotAvailableError(
            f"PiDriver requires the Pi coding agent CLI but {requested!r} was not found. "
            + _INSTALL_HINT
        )
    return resolved


def _probe_rpc_support(binary: str, env: dict[str, str]) -> bool:
    """Negotiate Pi's output protocol from ``--help`` once per binary path."""
    with _help_cache_lock:
        cached = _help_cache.get(binary)
    if cached is not None:
        return cached
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            [binary, "--help"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=_HELP_PROBE_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PiNotAvailableError(
            f"PiDriver could not run {binary!r} --help: {exc}. {_INSTALL_HINT}"
        ) from exc
    finally:
        if proc is not None:
            with suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(proc.pid, signal.SIGKILL)
            with suppress(subprocess.TimeoutExpired, OSError):
                proc.wait(timeout=2)
    mode_lines = [
        line for line in f"{stdout}\n{stderr}".splitlines() if "--mode" in line
    ]
    mode_help = " ".join(mode_lines)
    if "json" not in mode_help:
        raise PiNotAvailableError(
            f"Pi CLI at {binary!r} does not advertise JSON output (`--mode json`) in --help; "
            "upgrade Pi (run `pi update`) so Olympus can stream events and token usage."
        )
    supports_rpc = "rpc" in mode_help
    with _help_cache_lock:
        _help_cache[binary] = supports_rpc
    return supports_rpc


def _assistant_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n".join(parts)


def _short_target(arguments: dict[str, Any]) -> str:
    for key in ("path", "file_path", "command", "pattern", "query"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:160]
    return ""


def _tool_result_preview(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    content = result.get("content")
    if isinstance(content, str):
        return content[:240]
    if not isinstance(content, list):
        return ""
    texts = [
        str(block.get("text") or "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(text for text in texts if text)[:240]


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _safe_argv(command: list[str]) -> list[str]:
    """Redact prompts/API keys so the on-disk trace stays introspectable but compact."""
    safe: list[str] = []
    redact_next = False
    after_separator = False
    for index, argument in enumerate(command):
        if index == 0:
            continue
        if after_separator:
            safe.append(f"<prompt: {len(argument)} chars>")
            after_separator = False
        elif redact_next:
            safe.append(f"<{len(argument)} chars>")
            redact_next = False
        elif argument in ("--system-prompt", "--append-system-prompt", "--api-key"):
            safe.append(argument)
            redact_next = True
        elif argument == "--":
            safe.append(argument)
            after_separator = True
        else:
            safe.append(argument)
    return safe


class _TraceWriter:
    """Append-only JSONL trace guarded for reader, stderr, and steer threads."""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._handle: Any = None

    def append(self, entry: dict[str, Any]) -> None:
        if self._path is None:
            return
        record = {"timestamp": utc_now().isoformat(), **entry}
        with self._lock:
            if self._handle is None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._handle = self._path.open("a", encoding="utf-8")
            self._handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                with suppress(Exception):
                    self._handle.close()
                self._handle = None


class _PiTurn:
    def __init__(
        self,
        *,
        request: DriverTurnRequest,
        binary: str,
        supports_rpc: bool,
        store: RunStore | None,
        run_id: str | None,
        cancel_grace_seconds: float,
        term_grace_seconds: float,
        scope_guard_path: Path,
        emit: DriverEvent | None,
    ) -> None:
        self._request = request
        self._binary = binary
        self._supports_rpc = supports_rpc
        self._store = store
        self._run_id = run_id
        self._cancel_grace_seconds = cancel_grace_seconds
        self._term_grace_seconds = term_grace_seconds
        self._scope_guard_path = scope_guard_path
        self._emit_callback = emit
        self._trace = _TraceWriter(request.trace_path)
        self._lock = threading.Lock()
        self._stdin_lock = threading.Lock()
        self._closed = threading.Event()
        self._process: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._prompt_thread: threading.Thread | None = None
        self._cancel_requested = False
        self._budget_exhausted_reason = ""
        self._max_steps_exceeded = False
        self._stdin_closed = False
        self._steps = 0
        self._tool_started_at: dict[str, float] = {}
        self._step_by_tool: dict[str, int] = {}
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._final_content = ""
        self._receipt_events: list[dict[str, Any]] = []
        self._aborted = False
        self._prompt_error = ""
        self._saw_error_message = False
        self._returncode: int | None = None

    @property
    def supports_steer(self) -> bool:
        with self._lock:
            return (
                self._supports_rpc
                and not self._stdin_closed
                and not self._cancel_requested
            )

    def steer(self, guidance: str, *, source: str) -> bool:
        text = guidance.strip()
        if not text:
            return False
        with self._lock:
            if (
                not self._supports_rpc
                or self._stdin_closed
                or self._cancel_requested
            ):
                return False
        if not self._write_command({"type": "steer", "message": text}):
            return False
        with self._lock:
            step = self._steps + 1
            self._trace.append({"event": "agent.steered", "step": step, "source": source})
        self._emit(
            "agent.steered",
            {
                "actor": self._request.actor,
                "task_id": self._request.task_id,
                "step": step,
                "source": source,
            },
        )
        return True

    def cancel(self, *, reason: str) -> None:
        with self._lock:
            if self._cancel_requested:
                return
            self._cancel_requested = True
            self._trace.append({"event": "agent.cancelled", "reason": reason})
            supports_rpc = self._supports_rpc
        if supports_rpc:
            with suppress(Exception):
                self._write_command({"type": "abort"}, nonblocking=True)
        self._close_stdin(nonblocking=True)
        if threading.current_thread() is self._reader:
            # Never block the stdout pump on its own pipe closing.
            threading.Thread(
                target=self._escalate_cancel, name="pi-cancel-escalation", daemon=True
            ).start()
            return
        self._escalate_cancel()

    def execute(
        self, on_handle: Callable[[ActiveTurnHandle], None] | None = None
    ) -> TurnOutcome:
        try:
            self._spawn()
            reader = threading.Thread(target=self._pump_stdout, name="pi-stdout", daemon=True)
            self._reader = reader
            reader.start()
            threading.Thread(target=self._pump_stderr, name="pi-stderr", daemon=True).start()
            if on_handle is not None:
                on_handle(self)
            if self._supports_rpc:
                prompt_thread = threading.Thread(
                    target=self._send_prompt, name="pi-stdin-prompt", daemon=True
                )
                self._prompt_thread = prompt_thread
                prompt_thread.start()
            self._await_stdout_close()
        finally:
            self._reap()
            self._trace.close()
        receipt = build_turn_receipt(
            turn_id=new_id("turn"),
            actor=self._request.actor,
            task_id=self._request.task_id,
            events=self._receipt_events,
        )
        completed, stop_reason, final_content = self._classify_outcome()
        self._persist_receipt(receipt)
        self._emit("receipt", receipt)
        return TurnOutcome(
            completed=completed,
            stop_reason=stop_reason,
            final_content=final_content,
            steps_taken=self._steps,
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            receipt=receipt,
            driver_name="pi",
        )

    def _spawn(self) -> None:
        env = build_sandbox_env(self._request.workspace_path, host_root=self._request.env_root)
        env["OLYMPUS_PI_WRITE_SCOPE"] = json.dumps(self._request.write_scope)
        env["OLYMPUS_PI_FORBIDDEN_GLOBS"] = json.dumps(self._request.forbidden_globs)
        command = self._build_command()
        self._trace.append(
            {
                "event": "pi.spawn",
                "mode": "rpc" if self._supports_rpc else "json",
                "argv": _safe_argv(command),
            }
        )
        try:
            self._process = subprocess.Popen(
                command,
                cwd=self._request.workspace_path,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise PiNotAvailableError(
                f"PiDriver could not launch {self._binary!r}: {exc}. {_INSTALL_HINT}"
            ) from exc

    def _build_command(self) -> list[str]:
        request = self._request
        command = [self._binary]
        if self._supports_rpc:
            command += ["--mode", "rpc"]
        else:
            command += ["-p", "--mode", "json"]
        command += [
            "--no-session",
            "--no-mcp",
            "--no-approve",
            "--no-extensions",
            "--no-context-files",
            "--no-skills",
        ]
        if request.system_prompt.strip():
            command += ["--system-prompt", request.system_prompt]
        if request.model:
            command += ["--model", request.model]
        if request.read_only:
            command += ["--tools", _READ_ONLY_TOOLS, "--exclude-tools", _FORBIDDEN_WRITE_TOOLS]
        elif self._scope_guard_path.is_file():
            command += ["--extension", str(self._scope_guard_path)]
        else:
            self._trace.append(
                {"event": "pi.scope_guard_missing", "path": str(self._scope_guard_path)}
            )
        if not self._supports_rpc:
            command += ["--", request.user_prompt]
        return command

    def _send_prompt(self) -> None:
        if not self._supports_rpc:
            return
        delivered = self._write_command(
            {
                "id": "olympus-prompt",
                "type": "prompt",
                "message": self._request.user_prompt,
            }
        )
        if not delivered:
            with self._lock:
                if not self._cancel_requested:
                    self._prompt_error = "failed to deliver the prompt to pi"
            self._close_stdin()

    def _await_stdout_close(self) -> None:
        process = self._process
        if process is None:
            return
        while not self._closed.wait(0.05):
            if process.poll() is not None:
                # Give the reader a moment to drain, then reap a pipe-holding grandchild.
                if self._closed.wait(0.5):
                    break
                self._signal_group(signal.SIGKILL)
                self._closed.wait(2.0)
                break

    def _reap(self) -> None:
        process = self._process
        if process is None:
            return
        with suppress(subprocess.TimeoutExpired):
            self._returncode = process.wait(timeout=5)
        if process.poll() is None:
            self._signal_group(signal.SIGKILL)
            with suppress(subprocess.TimeoutExpired):
                self._returncode = process.wait(timeout=5)
        self._signal_group(signal.SIGKILL)
        if self._reader is not None:
            self._reader.join(timeout=5.0)
        if self._prompt_thread is not None:
            self._prompt_thread.join(timeout=2.0)
        self._close_stdin(nonblocking=True)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                with suppress(Exception):
                    stream.close()

    def _escalate_cancel(self) -> None:
        if self._closed.wait(self._cancel_grace_seconds):
            return
        self._signal_group(signal.SIGTERM)
        if self._closed.wait(self._term_grace_seconds):
            return
        self._signal_group(signal.SIGKILL)
        self._closed.wait(5.0)

    def _signal_group(self, sig: signal.Signals) -> None:
        process = self._process
        if process is None:
            return
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, PermissionError):
            with suppress(ProcessLookupError, PermissionError):
                process.send_signal(sig)

    def _write_command(self, command: dict[str, Any], *, nonblocking: bool = False) -> bool:
        if not self._stdin_lock.acquire(blocking=not nonblocking):
            return False
        try:
            process = self._process
            with self._lock:
                closed = self._stdin_closed
            if process is None or process.stdin is None or closed:
                return False
            if nonblocking:
                try:
                    _, writable, _ = select.select([], [process.stdin.fileno()], [], 0.05)
                except (OSError, ValueError):
                    return False
                if not writable:
                    return False
            payload = (json.dumps(command, ensure_ascii=False) + "\n").encode("utf-8")
            try:
                process.stdin.write(payload)
                process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                self._close_stdin_locked()
                return False
            return True
        finally:
            self._stdin_lock.release()

    def _close_stdin(self, *, nonblocking: bool = False) -> None:
        with self._lock:
            self._stdin_closed = True
        if not self._stdin_lock.acquire(blocking=not nonblocking):
            return
        try:
            self._close_stdin_locked()
        finally:
            self._stdin_lock.release()

    def _close_stdin_locked(self) -> None:
        process = self._process
        with self._lock:
            self._stdin_closed = True
        if process is not None and process.stdin is not None:
            with suppress(Exception):
                process.stdin.close()

    def _pump_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            for raw in process.stderr:
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                self._trace.append({"source": "pi.stderr", "line": line[:2000]})
        except (OSError, ValueError):
            return

    def _pump_stdout(self) -> None:
        process = self._process
        try:
            if process is not None and process.stdout is not None:
                for raw in process.stdout:
                    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        self._trace.append({"source": "pi.unparsed", "line": line[:2000]})
                        continue
                    if not isinstance(record, dict):
                        continue
                    if _is_trace_milestone(record):
                        self._trace.append({"source": "pi", "record": record})
                    self._handle_record(record)
        except (OSError, ValueError) as exc:
            self._trace.append({"event": "pi.reader_error", "error": str(exc)[:300]})
        finally:
            self._close_stdin(nonblocking=True)
            self._closed.set()

    def _handle_record(self, record: dict[str, Any]) -> None:
        record_type = str(record.get("type") or "")
        if record_type == "tool_execution_start":
            self._on_tool_start(record)
        elif record_type == "tool_execution_end":
            self._on_tool_end(record)
        elif record_type == "message_update":
            delta = record.get("assistantMessageEvent")
            if isinstance(delta, dict) and delta.get("type") == "text_delta":
                text = delta.get("delta")
                if isinstance(text, str) and text:
                    self._emit("text_delta", {"text": text, "replace": False})
        elif record_type == "message_end":
            message = record.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                self._on_assistant_message(message)
        elif record_type in ("compaction_start", "compaction_end"):
            self._on_compaction(record_type)
        elif record_type == "agent_settled":
            with self._lock:
                self._aborted = bool(record.get("aborted"))
            self._close_stdin(nonblocking=True)
        elif record_type == "response":
            self._on_response(record)
        elif record_type == "extension_error":
            self._trace.append(
                {"event": "pi.extension_error", "error": str(record.get("error") or "")[:300]}
            )

    def _on_tool_start(self, record: dict[str, Any]) -> None:
        call_id = str(record.get("toolCallId") or "")
        tool = str(record.get("toolName") or "unknown")
        arguments = record.get("args")
        target = _short_target(arguments if isinstance(arguments, dict) else {})
        with self._lock:
            step = self._steps + 1
            self._tool_started_at[call_id] = time.monotonic()
            self._step_by_tool[call_id] = step
        self._trace.append(
            {"event": "tool_call", "step": step, "tool": tool, "target": target, "status": "ok"}
        )
        self._emit(
            "agent.tool_started",
            {
                "actor": self._request.actor,
                "task_id": self._request.task_id,
                "step": step,
                "tool": tool,
            },
        )

    def _on_tool_end(self, record: dict[str, Any]) -> None:
        call_id = str(record.get("toolCallId") or "")
        tool = str(record.get("toolName") or "unknown")
        is_error = bool(record.get("isError"))
        with self._lock:
            started = self._tool_started_at.pop(call_id, None)
            step = self._step_by_tool.pop(call_id, self._steps or 1)
        if record.get("durationMs") is not None:
            duration_ms = _non_negative_int(record.get("durationMs"))
        elif started is not None:
            duration_ms = int((time.monotonic() - started) * 1000)
        else:
            duration_ms = 0
        arguments = record.get("args")
        target = _short_target(arguments if isinstance(arguments, dict) else {})
        output_preview = _tool_result_preview(record.get("result"))
        error_signature = (
            content_hash({"tool": tool, "output": output_preview}) if is_error else ""
        )
        self._trace.append(
            {
                "event": "tool_end",
                "step": step,
                "tool": tool,
                "target": target,
                "status": "error" if is_error else "ok",
                "is_error": is_error,
                "duration_ms": duration_ms,
                "output": output_preview,
            }
        )
        self._receipt_events.append(
            {
                "kind": "tool_end",
                "tool": tool,
                "is_error": is_error,
                "duration_ms": duration_ms,
            }
        )
        self._emit(
            "agent.tool_finished",
            {
                "actor": self._request.actor,
                "task_id": self._request.task_id,
                "step": step,
                "tool": tool,
                "status": "error" if is_error else "ok",
                "is_error": is_error,
                "duration_ms": duration_ms,
                "error_signature": error_signature,
            },
        )

    def _on_compaction(self, record_type: str) -> None:
        phase = "start" if record_type == "compaction_start" else "end"
        self._trace.append({"event": "agent.compaction", "phase": phase})
        self._emit(
            "agent.compaction",
            {
                "actor": self._request.actor,
                "task_id": self._request.task_id,
                "phase": phase,
            },
        )

    def _on_assistant_message(self, message: dict[str, Any]) -> None:
        usage = message.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        input_tokens = _non_negative_int(usage.get("input"))
        output_tokens = _non_negative_int(usage.get("output"))
        cache_read_tokens = _non_negative_int(usage.get("cacheRead"))
        cache_write_tokens = _non_negative_int(usage.get("cacheWrite"))
        with self._lock:
            self._steps += 1
            step = self._steps
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            if message.get("stopReason") == "error":
                self._saw_error_message = True
        text = _assistant_text(message)
        if text:
            self._final_content = text
        self._trace.append(
            {
                "event": "llm",
                "step": step,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_tokens": cache_read_tokens,
                "cache_write_tokens": cache_write_tokens,
            }
        )
        self._receipt_events.append(
            {
                "kind": "llm",
                "actor": self._request.actor,
                "task_id": self._request.task_id,
                "step": step,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )
        over_budget = self._settle_usage(
            step, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
        )
        self._emit(
            "llm",
            {
                "actor": self._request.actor,
                "task_id": self._request.task_id,
                "step": step,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_tokens": cache_read_tokens,
                "cache_write_tokens": cache_write_tokens,
            },
        )
        if over_budget:
            with self._lock:
                self._budget_exhausted_reason = "budget_exhausted: Pi exceeded configured budget"
            self.cancel(reason="budget_exhausted")
        elif self._request.max_steps is not None and step > self._request.max_steps:
            with self._lock:
                self._max_steps_exceeded = True
            self.cancel(reason="max_steps_exceeded")

    def _on_response(self, record: dict[str, Any]) -> None:
        if bool(record.get("success")):
            return
        command = str(record.get("command") or "")
        error = str(record.get("error") or "unknown error")
        self._trace.append(
            {"event": "pi.command_error", "command": command, "error": error[:500]}
        )
        if command == "prompt":
            with self._lock:
                self._prompt_error = error[:500]
            self._close_stdin(nonblocking=True)

    def _settle_usage(
        self,
        step: int,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
    ) -> bool:
        if self._store is None or self._run_id is None:
            return False
        try:
            return self._store.record_external_llm_usage(
                self._run_id,
                actor=self._request.actor,
                purpose=f"pi_step_{step}",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - usage accounting must not kill the turn
            self._trace.append(
                {"event": "usage.settle_error", "step": step, "error": str(exc)[:300]}
            )
            return False

    def _classify_outcome(self) -> tuple[bool, str, str]:
        if self._budget_exhausted_reason:
            return False, self._budget_exhausted_reason, ""
        if self._max_steps_exceeded:
            return False, "max_steps_exceeded", ""
        if self._prompt_error:
            return False, "pi_error", ""
        if self._cancel_requested:
            return False, "cancelled", ""
        if self._aborted:
            return False, "pi_aborted", ""
        if self._saw_error_message:
            return False, "pi_error", ""
        if self._returncode not in (0, None):
            return False, "pi_error", ""
        if self._steps == 0:
            return False, "pi_error", ""
        return True, "final_response", self._final_content

    def _persist_receipt(self, receipt: dict[str, Any]) -> None:
        if self._store is None or self._run_id is None:
            return
        try:
            self._store.append_event(
                run_id=self._run_id,
                stage="orchestrate",
                event_type="turn.receipt",
                actor=self._request.actor,
                payload=receipt,
            )
        except Exception as exc:  # noqa: BLE001 - receipt persistence must not fail a completed turn
            self._trace.append(
                {"event": "receipt.persist_error", "error": str(exc)[:300]}
            )

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self._emit_callback is None:
            return
        try:
            self._emit_callback(kind, payload)
        except Exception as exc:  # noqa: BLE001 - observers must never abort a harness turn
            self._trace.append(
                {"event": "observer.error", "kind": kind, "error": str(exc)[:300]}
            )


class PiDriver:
    """Run one Pi turn inside the task sandbox behind the ``AgentDriver`` seam."""

    name = "pi"

    def __init__(
        self,
        *,
        binary: str | None = None,
        store: RunStore | None = None,
        run_id: str | None = None,
        cancel_grace_seconds: float = 3.0,
        term_grace_seconds: float = 2.0,
        scope_guard_path: Path | None = None,
    ) -> None:
        self._binary = _resolve_binary(binary)
        self._store = store
        self._run_id = run_id
        self._cancel_grace_seconds = cancel_grace_seconds
        self._term_grace_seconds = term_grace_seconds
        self._scope_guard_path = scope_guard_path or _SCOPE_GUARD_PATH

    def run_turn(
        self,
        request: DriverTurnRequest,
        *,
        on_event: DriverEvent | None = None,
        on_handle: Callable[[ActiveTurnHandle], None] | None = None,
    ) -> TurnOutcome:
        env = build_sandbox_env(request.workspace_path, host_root=request.env_root)
        supports_rpc = _probe_rpc_support(self._binary, env)
        turn = _PiTurn(
            request=request,
            binary=self._binary,
            supports_rpc=supports_rpc,
            store=self._store,
            run_id=self._run_id,
            cancel_grace_seconds=self._cancel_grace_seconds,
            term_grace_seconds=self._term_grace_seconds,
            scope_guard_path=self._scope_guard_path,
            emit=on_event,
        )
        return turn.execute(on_handle)