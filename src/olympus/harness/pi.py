"""AgentDriver adapter for the Pi coding agent CLI (``--mode rpc`` / ``--mode json``).

Pi is driven as an isolated subprocess inside the task sandbox: stdout JSONL is
pumped on a dedicated reader thread, tool milestones are normalized onto the
``AgentDriver`` observer seam, usage is settled into the run ledger, and
``steer`` / ``cancel`` act over the RPC stdin pipe and the process group.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import select
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from olympus.core.contracts import new_id, utc_now
from olympus.core.fingerprint import content_hash
from olympus.core.store import RunStore
from olympus.harness.driver import (
    VALID_THINKING_LEVELS,
    ActiveTurnHandle,
    DriverEvent,
    DriverTurnRequest,
    TurnOutcome,
)
from olympus.harness.receipt import build_turn_receipt
from olympus.providers.auth import ExternalGatewayBinding
from olympus.providers.base import LLMProvider
from olympus.providers.factory import resolve_external_gateway_from_env
from olympus.tools.aci import build_sandbox_env

_INSTALL_HINT = (
    "Install Pi with `npm install -g --ignore-scripts @earendil-works/pi-coding-agent` "
    "or point OLYMPUS_PI_BIN at the pi executable."
)
_READ_ONLY_TOOLS = "read,grep,find,ls"
_WRITABLE_TOOLS = "read,bash,edit,write,grep,find,ls"
_FORBIDDEN_WRITE_TOOLS = "bash,write,edit,powershell"
_HELP_PROBE_TIMEOUT_SECONDS = 10.0
_SCOPE_GUARD_PATH = Path(__file__).with_name("pi_scope_guard.ts")
_NET_KEEPALIVE_PATH = Path(__file__).with_name("pi_net_keepalive.cjs")
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


_BinaryCacheKey = tuple[str, int, int]
_help_cache: dict[_BinaryCacheKey, bool] = {}
_help_cache_lock = threading.Lock()
_availability_cache: dict[_BinaryCacheKey, bool] = {}
_availability_cache_lock = threading.Lock()
_SECRET_ENV_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|passwd|credential|authorization|auth)"
)


def _binary_cache_key(resolved: str) -> _BinaryCacheKey:
    """Bind cached binary probes to the concrete file identity on disk."""
    try:
        st = os.stat(resolved)
        return (resolved, st.st_mtime_ns, st.st_size)
    except OSError:
        return (resolved, -1, -1)


def _collect_redaction_secrets(
    env: Mapping[str, str], extra_secrets: tuple[str, ...] = ()
) -> tuple[str, ...]:
    values: set[str] = {s.strip() for s in extra_secrets if s and s.strip()}
    for key, val in env.items():
        if val and len(val) >= 8 and _SECRET_ENV_RE.search(key):
            values.add(val)
    return tuple(sorted(values, key=len, reverse=True))


def _redact_secrets(text: str, secrets: tuple[str, ...]) -> str:
    if not text or not secrets:
        return text
    redacted = text
    for secret in secrets:
        redacted = redacted.replace(secret, "***")
    return redacted


def _redact_value(value: Any, secrets: tuple[str, ...]) -> Any:
    if not secrets:
        return value
    if isinstance(value, str):
        return _redact_secrets(value, secrets)
    if isinstance(value, dict):
        return {k: _redact_value(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(item, secrets) for item in value]
    return value


def pi_available(*, binary: str | None = None) -> bool:
    """Return True when the Pi CLI resolves and passes its cached ``--help`` probe."""
    requested = (binary or os.getenv("OLYMPUS_PI_BIN", "") or "pi").strip()
    try:
        resolved = _resolve_binary(requested)
    except PiNotAvailableError:
        return False
    cache_key = _binary_cache_key(resolved)
    with _availability_cache_lock:
        cached = _availability_cache.get(cache_key)
    if cached is not None:
        return cached
    available = False
    try:
        _probe_rpc_support(resolved, dict(os.environ))
        available = True
    except PiNotAvailableError:
        available = False
    with _availability_cache_lock:
        _availability_cache[cache_key] = available
    return available


def _reset_pi_availability_cache() -> None:
    """Drop cached probes (used by tests that swap fake binaries per case)."""
    with _availability_cache_lock:
        _availability_cache.clear()
    with _help_cache_lock:
        _help_cache.clear()


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
    """Negotiate Pi's output protocol from ``--help`` once per concrete binary."""
    cache_key = _binary_cache_key(binary)
    with _help_cache_lock:
        cached = _help_cache.get(cache_key)
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
        _help_cache[cache_key] = supports_rpc
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


def _read_only_tmp_dir(request: DriverTurnRequest) -> Path | None:
    """Keep temp dirs outside the repository for read-only turns on the repo root."""
    if not request.read_only:
        return None
    ws = request.workspace_path.resolve()
    host = request.env_root.resolve()
    if ws != host and not ws.is_relative_to(host):
        return None
    safe_actor = "".join(c if c.isalnum() or c in "-_" else "-" for c in request.actor) or "stage"
    if request.trace_path is not None:
        trace_parent = request.trace_path.resolve().parent
        if not trace_parent.is_relative_to(ws):
            return trace_parent / ".pi_tmp" / safe_actor
    digest = hashlib.sha256(str(ws).encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / "olympus-pi-readonly-tmp" / f"{digest}-{safe_actor}"


def _inject_pi_node_keepalive(env: dict[str, str]) -> None:
    """Attach the TCP keep-alive & read-timeout preload to Pi's ``NODE_OPTIONS``."""
    if not _NET_KEEPALIVE_PATH.is_file():
        return
    preload = str(_NET_KEEPALIVE_PATH.resolve())
    if any(ch.isspace() or ch in ('"', "'") for ch in preload):
        return
    flag = f"--require={preload}"
    existing = (env.get("NODE_OPTIONS") or "").strip()
    if preload in existing:
        return
    env["NODE_OPTIONS"] = f"{existing} {flag}".strip() if existing else flag


def _build_pi_env(request: DriverTurnRequest) -> dict[str, str]:
    env = build_sandbox_env(
        request.workspace_path,
        host_root=request.env_root,
        tmp_dir=_read_only_tmp_dir(request),
    )
    _inject_pi_node_keepalive(env)
    return env


def _normalize_thinking_level(value: str | None) -> str | None:
    """Validate and lowercase a ``--thinking`` level against Pi's documented scale."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized not in VALID_THINKING_LEVELS:
        raise ValueError(
            f"Invalid thinking_level {value!r}; expected one of "
            f"{', '.join(sorted(VALID_THINKING_LEVELS))}"
        )
    return normalized


def _strip_provider_prefix(model: str | None, pi_provider: str | None) -> str:
    """Drop a leading ``<pi_provider>/`` prefix while preserving catalog IDs verbatim.

    For example ``--pi-provider openrouter --model openai/gpt-4.1`` preserves
    ``openai/gpt-4.1`` because ``openai/`` is part of OpenRouter's model ID,
    whereas ``openrouter/openai/gpt-4.1`` strips only ``openrouter/``.
    """
    cleaned = (model or "").strip()
    if not cleaned or not pi_provider or not pi_provider.strip():
        return cleaned
    prefix = f"{pi_provider.strip().lower()}/"
    if cleaned.lower().startswith(prefix) and cleaned[len(prefix) :].strip():
        return cleaned[len(prefix) :]
    return cleaned


def _resolve_ambient_pi_auth(
    pi_provider: str, env: Mapping[str, str]
) -> tuple[str | None, str | None, dict[str, str], list[str]]:
    """Read a saved Pi credential for ``pi_provider`` from ``~/.pi/agent/auth.json`` read-only.

    Returns ``(api_key_ref, literal_secret, extra_env, redaction_secrets)`` without ever
    copying, serializing, or symlinking ``auth.json`` into the ephemeral config dir.
    """
    home_raw = (env.get("HOME") or "").strip()
    auth_path = (Path(home_raw) if home_raw else Path.home()) / ".pi" / "agent" / "auth.json"
    if not auth_path.is_file():
        return None, None, {}, []
    try:
        raw = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None, {}, []
    if not isinstance(raw, dict):
        return None, None, {}, []
    entry = raw.get(pi_provider)
    if not isinstance(entry, dict):
        return None, None, {}, []
    extra_env: dict[str, str] = {}
    secrets: list[str] = []
    raw_entry_env = entry.get("env")
    if isinstance(raw_entry_env, dict):
        for env_key, env_val in raw_entry_env.items():
            if isinstance(env_key, str) and env_key.strip() and isinstance(env_val, str):
                extra_env[env_key.strip()] = env_val
                if env_val.strip():
                    secrets.append(env_val.strip())
    credential = ""
    for field in ("key", "apiKey", "access"):
        candidate = entry.get(field)
        if isinstance(candidate, str) and candidate.strip():
            credential = candidate.strip()
            break
    if not credential:
        return None, None, extra_env, secrets
    if credential.startswith(("!", "$")):
        if credential.startswith("$"):
            var_name = credential[1:].strip("{}")
            env_val = (env.get(var_name) or extra_env.get(var_name) or "").strip()
            if env_val:
                secrets.append(env_val)
        return credential, None, extra_env, secrets
    secrets.append(credential)
    return None, credential, extra_env, secrets


def _binding_secrets(binding: ExternalGatewayBinding, env: Mapping[str, str]) -> list[str]:
    """Collect every non-empty credential value referenced by a gateway binding."""
    secrets: list[str] = []
    if binding.api_key and binding.api_key.strip():
        secrets.append(binding.api_key.strip())
    if binding.api_key_env:
        env_val = (env.get(binding.api_key_env) or "").strip()
        if env_val:
            secrets.append(env_val)
    return secrets


def _prepare_gateway_config(
    env: dict[str, str],
    model: str | None,
    *,
    provider: LLMProvider | None = None,
    pi_provider: str | None = None,
    context_window: int | None = None,
) -> tuple[str | None, str | None, Path | None, tuple[str, ...]]:
    """Resolve the provider wiring for one Pi turn.

    Returns ``(effective_provider, model_id, cfg_dir, extra_secrets)``:

    * A caller-provided ``PI_CODING_AGENT_DIR`` is respected untouched.
    * A configured ``pi_provider`` passes through natively (no ``models.json``)
      unless an explicit custom gateway binding or a context-window override
      needs an ephemeral provider-override config.
    * Without a ``pi_provider`` the legacy ``olympus-gateway`` provider is
      synthesized as before.

    Credentials stay as ``$ENV_VAR`` references so no API keys are ever written
    to disk.
    """
    effective_pi_provider = (pi_provider or "").strip() or None
    effective_context_window = context_window
    if env.get("PI_CODING_AGENT_DIR"):
        # The caller owns the entire config dir: never overwrite it.
        return effective_pi_provider, model, None, ()

    if effective_pi_provider is not None:
        binding: ExternalGatewayBinding | None = None
        binder = getattr(provider, "external_gateway_binding", None)
        if callable(binder):
            maybe = binder(model)
            if isinstance(maybe, ExternalGatewayBinding) and maybe.is_custom_endpoint:
                binding = maybe
        if binding is None:
            env_binding = resolve_external_gateway_from_env(env, model)
            if env_binding is not None and env_binding.is_custom_endpoint:
                binding = env_binding
        raw_model = (
            (model or "").strip()
            or str(getattr(provider, "model_name", "") or "").strip()
            or (binding.model_id if binding is not None else "")
        )
        model_id = _strip_provider_prefix(raw_model, effective_pi_provider)
        if binding is None and effective_context_window is None:
            # Pure native Pi passthrough: Pi's built-in catalog for the provider
            # already carries baseUrl, apiKey wiring, model metadata, and limits.
            return effective_pi_provider, model_id, None, ()

        extra_secrets: list[str] = []
        override: dict[str, Any] = {}
        if binding is not None:
            if binding.api_key_env:
                api_key_ref = f"${binding.api_key_env}"
            else:
                env["OLYMPUS_PI_GATEWAY_KEY"] = binding.api_key or "keyless"
                api_key_ref = "$OLYMPUS_PI_GATEWAY_KEY"
            extra_secrets.extend(_binding_secrets(binding, env))
            override["baseUrl"] = binding.base_url
            override["apiKey"] = api_key_ref
        if effective_context_window is not None and model_id:
            override["modelOverrides"] = {
                model_id: {"contextWindow": effective_context_window}
            }
        if binding is None:
            (
                ambient_ref,
                ambient_secret,
                ambient_env,
                ambient_secrets,
            ) = _resolve_ambient_pi_auth(effective_pi_provider, env)
            for env_key, env_val in ambient_env.items():
                env.setdefault(env_key, env_val)
            extra_secrets.extend(ambient_secrets)
            if ambient_ref is not None:
                override["apiKey"] = ambient_ref
            elif ambient_secret is not None:
                env["OLYMPUS_PI_GATEWAY_KEY"] = ambient_secret
                override["apiKey"] = "$OLYMPUS_PI_GATEWAY_KEY"
        cfg_dir = Path(tempfile.mkdtemp(prefix="olympus-pi-cfg-"))
        os.chmod(cfg_dir, 0o700)
        models_doc = {"providers": {effective_pi_provider: override}}
        models_file = cfg_dir / "models.json"
        models_file.write_text(
            json.dumps(models_doc, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.chmod(models_file, 0o600)
        env["PI_CODING_AGENT_DIR"] = str(cfg_dir)
        env.setdefault("PI_OFFLINE", "1")
        env.setdefault("PI_SKIP_VERSION_CHECK", "1")
        return effective_pi_provider, model_id, cfg_dir, tuple(extra_secrets)

    legacy_binding: ExternalGatewayBinding | None = None
    binder = getattr(provider, "external_gateway_binding", None)
    if callable(binder):
        maybe = binder(model)
        if isinstance(maybe, ExternalGatewayBinding):
            legacy_binding = maybe
    if legacy_binding is None:
        legacy_binding = resolve_external_gateway_from_env(env, model)
    if legacy_binding is None:
        return None, model, None, ()

    legacy_secrets: list[str] = _binding_secrets(legacy_binding, env)
    if legacy_binding.api_key_env:
        api_key_ref = f"${legacy_binding.api_key_env}"
    else:
        env["OLYMPUS_PI_GATEWAY_KEY"] = legacy_binding.api_key or "keyless"
        api_key_ref = "$OLYMPUS_PI_GATEWAY_KEY"

    model_entry: dict[str, Any] = {"id": legacy_binding.model_id}
    if effective_context_window is not None:
        model_entry["contextWindow"] = effective_context_window
    cfg_dir = Path(tempfile.mkdtemp(prefix="olympus-pi-cfg-"))
    os.chmod(cfg_dir, 0o700)
    models_doc = {
        "providers": {
            "olympus-gateway": {
                "baseUrl": legacy_binding.base_url,
                "api": legacy_binding.api,
                "apiKey": api_key_ref,
                "models": [model_entry],
            }
        }
    }
    models_file = cfg_dir / "models.json"
    models_file.write_text(json.dumps(models_doc, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(models_file, 0o600)
    env["PI_CODING_AGENT_DIR"] = str(cfg_dir)
    env.setdefault("PI_OFFLINE", "1")
    env.setdefault("PI_SKIP_VERSION_CHECK", "1")
    return "olympus-gateway", legacy_binding.model_id, cfg_dir, tuple(legacy_secrets)


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
        self._secrets: tuple[str, ...] = ()

    def set_secrets(self, secrets: tuple[str, ...]) -> None:
        with self._lock:
            self._secrets = secrets

    def append(self, entry: dict[str, Any]) -> None:
        if self._path is None:
            return
        with self._lock:
            safe_entry = _redact_value(entry, self._secrets)
            record = {"timestamp": utc_now().isoformat(), **safe_entry}
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
        provider: LLMProvider | None = None,
        pi_provider: str | None = None,
        thinking_level: str | None = None,
        context_window: int | None = None,
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
        self._provider = provider
        # Request values override the driver defaults when explicitly set.
        self._pi_provider = (request.pi_provider or pi_provider or "").strip() or None
        self._thinking_level = _normalize_thinking_level(
            request.thinking_level if request.thinking_level is not None else thinking_level
        )
        self._context_window = (
            request.context_window if request.context_window is not None else context_window
        )
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
        self._budget_steer_steps: set[int] = set()
        self._stdin_closed = False
        self._steps = 0
        self._tool_started_at: dict[str, float] = {}
        self._step_by_tool: dict[str, int] = {}
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cache_read_tokens = 0
        self._total_cache_write_tokens = 0
        self._total_reasoning_tokens = 0
        self._final_content = ""
        self._receipt_events: list[dict[str, Any]] = []
        self._aborted = False
        self._prompt_error = ""
        self._saw_error_message = False
        self._last_error_message = ""
        self._last_stderr_line = ""
        self._returncode: int | None = None
        self._gateway_provider: str | None = None
        self._gateway_model: str | None = request.model
        self._gateway_config_dir: Path | None = None
        self._redaction_secrets: tuple[str, ...] = ()

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
            total_cache_read_tokens=self._total_cache_read_tokens,
            total_cache_write_tokens=self._total_cache_write_tokens,
            total_reasoning_tokens=self._total_reasoning_tokens,
            receipt=receipt,
            driver_name="pi",
        )

    def _spawn(self) -> None:
        env = _build_pi_env(self._request)
        env["OLYMPUS_PI_WRITE_SCOPE"] = json.dumps(self._request.write_scope)
        env["OLYMPUS_PI_FORBIDDEN_GLOBS"] = json.dumps(self._request.forbidden_globs)
        env["OLYMPUS_PI_STRICT_SCOPE"] = "1" if self._request.strict_scope else "0"
        provider_name, model_id, cfg_dir, extra_secrets = _prepare_gateway_config(
            env,
            self._request.model,
            provider=self._provider,
            pi_provider=self._pi_provider,
            context_window=self._context_window,
        )
        self._gateway_provider = provider_name
        self._gateway_model = model_id
        self._gateway_config_dir = cfg_dir
        self._redaction_secrets = _collect_redaction_secrets(env, extra_secrets)
        self._trace.set_secrets(self._redaction_secrets)
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
            command += ["--append-system-prompt", request.system_prompt]
        if self._gateway_provider:
            command += ["--provider", self._gateway_provider]
        if self._gateway_model:
            command += ["--model", self._gateway_model]
        if self._thinking_level:
            command += ["--thinking", self._thinking_level]
        if request.disable_tools:
            command += ["--no-tools"]
        elif request.read_only:
            command += ["--tools", _READ_ONLY_TOOLS, "--exclude-tools", _FORBIDDEN_WRITE_TOOLS]
        else:
            command += ["--tools", _WRITABLE_TOOLS]
            if self._scope_guard_path.is_file():
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
        try:
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
        finally:
            if self._gateway_config_dir is not None:
                shutil.rmtree(self._gateway_config_dir, ignore_errors=True)
                self._gateway_config_dir = None

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
                redacted = _redact_secrets(line, self._redaction_secrets)[:2000]
                with self._lock:
                    self._last_stderr_line = redacted[:300]
                self._trace.append({"source": "pi.stderr", "line": redacted})
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
                        redacted = _redact_secrets(line, self._redaction_secrets)[:2000]
                        self._trace.append({"source": "pi.unparsed", "line": redacted})
                        continue
                    if not isinstance(record, dict):
                        continue
                    if _is_trace_milestone(record):
                        self._trace.append({"source": "pi", "record": record})
                    self._handle_record(record)
        except (OSError, ValueError) as exc:
            err_text = _redact_secrets(str(exc), self._redaction_secrets)[:300]
            self._trace.append({"event": "pi.reader_error", "error": err_text})
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
        reasoning_tokens = _non_negative_int(usage.get("reasoning"))
        with self._lock:
            self._steps += 1
            step = self._steps
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_cache_read_tokens += cache_read_tokens
            self._total_cache_write_tokens += cache_write_tokens
            self._total_reasoning_tokens += reasoning_tokens
            # Track whether the *latest* assistant message errored so a transient
            # transport error that Pi retries and recovers from does not fail the turn.
            is_error_msg = message.get("stopReason") == "error"
            self._saw_error_message = is_error_msg
            if is_error_msg:
                raw_err = str(message.get("errorMessage") or message.get("error") or "").strip()
                self._last_error_message = (
                    _redact_secrets(raw_err, self._redaction_secrets)[:300]
                    if raw_err
                    else ""
                )
            else:
                self._last_error_message = ""
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
                "reasoning_tokens": reasoning_tokens,
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
                "cache_read_tokens": cache_read_tokens,
                "cache_write_tokens": cache_write_tokens,
                "reasoning_tokens": reasoning_tokens,
            }
        )
        over_budget = self._settle_usage(
            step,
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_tokens,
            reasoning_tokens,
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
                "reasoning_tokens": reasoning_tokens,
            },
        )
        stop_reason = str(message.get("stopReason") or "")
        if over_budget:
            with self._lock:
                self._budget_exhausted_reason = "budget_exhausted: Pi exceeded configured budget"
            self.cancel(reason="budget_exhausted")
        elif self._request.max_steps is not None:
            warn_steps = {
                s for s in (self._request.max_steps - 3, self._request.max_steps - 1) if s >= 1
            }
            if step in warn_steps and stop_reason != "stop":
                with self._lock:
                    should_steer = step not in self._budget_steer_steps
                    self._budget_steer_steps.add(step)
                if should_steer:
                    threading.Thread(
                        target=self.steer,
                        args=(
                            "Step budget is almost exhausted. Stop calling tools immediately and "
                            "emit your final response now.",
                        ),
                        kwargs={"source": "step_budget"},
                        daemon=True,
                    ).start()
            elif step > self._request.max_steps:
                allow_final_landing = (
                    step == self._request.max_steps + 1
                    and stop_reason == "stop"
                    and bool(self._final_content.strip())
                )
                if not allow_final_landing:
                    with self._lock:
                        self._max_steps_exceeded = True
                    self.cancel(reason="max_steps_exceeded")

    def _on_response(self, record: dict[str, Any]) -> None:
        if bool(record.get("success")):
            return
        command = str(record.get("command") or "")
        error = _redact_secrets(
            str(record.get("error") or "unknown error"), self._redaction_secrets
        )
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
        reasoning_tokens: int,
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
                reasoning_tokens=reasoning_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - usage accounting must not kill the turn
            self._trace.append(
                {"event": "usage.settle_error", "step": step, "error": str(exc)[:300]}
            )
            return False

    def _pi_error_reason(self, detail: str = "") -> str:
        clean = (detail or self._last_error_message or self._last_stderr_line).strip()
        return f"pi_error: {clean}" if clean else "pi_error"

    def _classify_outcome(self) -> tuple[bool, str, str]:
        if self._budget_exhausted_reason:
            return False, self._budget_exhausted_reason, ""
        if self._max_steps_exceeded:
            return False, "max_steps_exceeded", self._final_content
        if self._prompt_error:
            return False, self._pi_error_reason(self._prompt_error), ""
        if self._cancel_requested:
            return False, "cancelled", ""
        if self._aborted:
            return False, "pi_aborted", ""
        if self._saw_error_message:
            return False, self._pi_error_reason(self._last_error_message), ""
        if self._returncode not in (0, None):
            return False, self._pi_error_reason(), ""
        if self._steps == 0:
            return False, self._pi_error_reason(), ""
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
    records_usage_internally = True

    def __init__(
        self,
        *,
        binary: str | None = None,
        store: RunStore | None = None,
        run_id: str | None = None,
        provider: LLMProvider | None = None,
        pi_provider: str | None = None,
        thinking_level: str | None = None,
        context_window: int | None = None,
        cancel_grace_seconds: float = 3.0,
        term_grace_seconds: float = 2.0,
        scope_guard_path: Path | None = None,
    ) -> None:
        self._binary = _resolve_binary(binary)
        self._store = store
        self._run_id = run_id
        self._provider = provider
        self._pi_provider = (pi_provider or "").strip() or None
        self._thinking_level = _normalize_thinking_level(thinking_level)
        self._context_window = context_window
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
        env = _build_pi_env(request)
        supports_rpc = _probe_rpc_support(self._binary, env)
        turn = _PiTurn(
            request=request,
            binary=self._binary,
            supports_rpc=supports_rpc,
            store=self._store,
            run_id=self._run_id,
            provider=self._provider,
            pi_provider=self._pi_provider,
            thinking_level=self._thinking_level,
            context_window=self._context_window,
            cancel_grace_seconds=self._cancel_grace_seconds,
            term_grace_seconds=self._term_grace_seconds,
            scope_guard_path=self._scope_guard_path,
            emit=on_event,
        )
        return turn.execute(on_handle)