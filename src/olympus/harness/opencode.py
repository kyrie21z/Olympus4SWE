"""AgentDriver adapter for the OpenCode CLI (``opencode run --format json``).

OpenCode runs as an isolated subprocess inside the task workspace with an
ephemeral XDG state directory and declarative ``OPENCODE_CONFIG_CONTENT``
permissions. Stdout JSONL is pumped on a dedicated reader thread, tool and step
milestones are normalized onto the ``AgentDriver`` observer seam, token usage
is settled into the run ledger, and cancellation terminates the process group.

Note: Under CLI subprocess transport (without an HTTP proxy), ``agent.steps``
and atomic call reservations prevent planned extra steps and concurrent turn
oversubscription. OpenCode v2.4.2's internal ``finish_reason="length"``
continuation or transport retries may start before cancellation arrives;
``OpenCodeDriver`` settles every received ``step_finish`` honestly into the run
ledger and cancels immediately when budget or step limits are tripped.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from olympus.core.contracts import new_id, utc_now
from olympus.core.fingerprint import content_hash
from olympus.core.store import RunStore
from olympus.harness.driver import (
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
    "Install OpenCode (https://opencode.ai) or point OLYMPUS_OPENCODE_BIN "
    "at the opencode executable."
)
_AGENT_ID = "olympus_turn"
_GATEWAY_PROVIDER_ID = "olympus-gateway"
_GATEWAY_KEY_ENV = "OLYMPUS_OPENCODE_GATEWAY_KEY"
_HELP_PROBE_TIMEOUT_SECONDS = 10.0
_MAX_STDERR_TRACE_LINES = 50
_MAX_CHAR_CLASS_EXPANSIONS = 256
_MAX_PRE_RESERVE_STEPS = 64
_SECRET_ENV_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|passwd|credential|authorization|auth)"
)


class OpenCodeNotAvailableError(RuntimeError):
    """Raised when the configured OpenCode CLI is missing or unusable."""


_BinaryCacheKey = tuple[str, int, int]
_help_cache: dict[_BinaryCacheKey, bool] = {}
_help_cache_lock = threading.Lock()
_availability_cache: dict[_BinaryCacheKey, bool] = {}
_availability_cache_lock = threading.Lock()


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


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _short_target(arguments: Mapping[str, Any]) -> str:
    for key in ("path", "file_path", "filePath", "command", "pattern", "query"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:160]
    return ""


def _resolve_binary(binary: str | None) -> str:
    requested = (binary or os.getenv("OLYMPUS_OPENCODE_BIN", "") or "opencode").strip()
    resolved = shutil.which(requested)
    if resolved is None:
        raise OpenCodeNotAvailableError(
            f"OpenCodeDriver requires the OpenCode CLI but {requested!r} was not found. "
            + _INSTALL_HINT
        )
    return resolved


def _probe_run_json_support(binary: str, env: Mapping[str, str]) -> None:
    """Verify ``opencode run --help`` advertises ``--format`` JSON output."""
    cache_key = _binary_cache_key(binary)
    with _help_cache_lock:
        if _help_cache.get(cache_key) is True:
            return
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            [binary, "run", "--help"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=dict(env),
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=_HELP_PROBE_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OpenCodeNotAvailableError(
            f"OpenCodeDriver could not run {binary!r} run --help: {exc}. {_INSTALL_HINT}"
        ) from exc
    finally:
        if proc is not None:
            with suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(proc.pid, signal.SIGKILL)
            with suppress(subprocess.TimeoutExpired, OSError):
                proc.wait(timeout=2)
    combined = f"{stdout}\n{stderr}"
    format_lines = [line for line in combined.splitlines() if "--format" in line]
    if not format_lines or "json" not in " ".join(format_lines).lower():
        raise OpenCodeNotAvailableError(
            f"OpenCode CLI at {binary!r} does not advertise JSON output "
            "(`--format json`) in `opencode run --help`; upgrade OpenCode so "
            "Olympus can stream events and token usage."
        )
    with _help_cache_lock:
        _help_cache[cache_key] = True


def opencode_available(*, binary: str | None = None) -> bool:
    """Return True when the OpenCode CLI resolves and passes its cached help probe."""
    requested = (binary or os.getenv("OLYMPUS_OPENCODE_BIN", "") or "opencode").strip()
    try:
        resolved = _resolve_binary(requested)
    except OpenCodeNotAvailableError:
        return False
    cache_key = _binary_cache_key(resolved)
    with _availability_cache_lock:
        cached = _availability_cache.get(cache_key)
    if cached is not None:
        return cached
    available = False
    try:
        _probe_run_json_support(resolved, os.environ)
        available = True
    except OpenCodeNotAvailableError:
        available = False
    with _availability_cache_lock:
        _availability_cache[cache_key] = available
    return available


def _reset_opencode_availability_cache() -> None:
    """Drop cached binary probes (used by tests that swap fake binaries per case)."""
    with _availability_cache_lock:
        _availability_cache.clear()
    with _help_cache_lock:
        _help_cache.clear()


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
            return trace_parent / ".opencode_tmp" / safe_actor
    digest = hashlib.sha256(str(ws).encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / "olympus-opencode-readonly-tmp" / f"{digest}-{safe_actor}"


def _normalize_scope_pattern(raw: str) -> str:
    pattern = raw.strip().replace("\\", "/")
    while pattern.startswith("./"):
        pattern = pattern[2:]
    return pattern.rstrip("/")


def _matches_scope_entry(path: str, raw_pattern: str) -> bool:
    """Match a concrete workspace-relative path against one Olympus scope entry."""
    pattern = _normalize_scope_pattern(raw_pattern)
    if not pattern or pattern == ".":
        return True
    if any(ch in pattern for ch in ("*", "?", "[")):
        fn_pattern = pattern.replace("[^", "[!")
        if fnmatch.fnmatchcase(path, fn_pattern):
            return True
        if "**/" in fn_pattern:
            return fnmatch.fnmatchcase(path, fn_pattern.replace("**/", ""))
        return False
    return path == pattern or path.startswith(f"{pattern}/")


def _parse_char_class(
    pattern: str, start: int
) -> tuple[bool, list[str], int] | None:
    """Parse a POSIX glob ``[...]`` group into ``(negated, chars, next_index)``."""
    length = len(pattern)
    index = start + 1
    if index >= length:
        return None
    negated = pattern[index] in ("!", "^")
    if negated:
        index += 1
    body_start = index
    if index < length and pattern[index] == "]":
        index += 1
    close = pattern.find("]", index)
    if close == -1:
        return None
    chars: list[str] = []
    seen: set[str] = set()
    cursor = body_start
    while cursor < close:
        if cursor + 2 < close and pattern[cursor + 1] == "-":
            lo, hi = ord(pattern[cursor]), ord(pattern[cursor + 2])
            if lo <= hi:
                for code in range(lo, hi + 1):
                    ch = chr(code)
                    if ch not in seen:
                        seen.add(ch)
                        chars.append(ch)
                        if len(chars) > _MAX_CHAR_CLASS_EXPANSIONS:
                            raise ValueError(
                                f"OpenCode declarative permission rules cannot faithfully "
                                f"expand character class in glob {pattern!r} "
                                f"(exceeds {_MAX_CHAR_CLASS_EXPANSIONS} variants)."
                            )
                cursor += 3
                continue
        ch = pattern[cursor]
        if ch not in seen:
            seen.add(ch)
            chars.append(ch)
            if len(chars) > _MAX_CHAR_CLASS_EXPANSIONS:
                raise ValueError(
                    f"OpenCode declarative permission rules cannot faithfully expand "
                    f"character class in glob {pattern!r} "
                    f"(exceeds {_MAX_CHAR_CLASS_EXPANSIONS} variants)."
                )
        cursor += 1
    if not chars:
        raise ValueError(
            f"OpenCode declarative permission rules cannot represent empty character class "
            f"in glob {pattern!r}."
        )
    if any(ch in ("*", "?") for ch in chars):
        raise ValueError(
            f"OpenCode declarative permission rules cannot match literal '*' or '?' "
            f"from character class in glob {pattern!r}."
        )
    return negated, chars, close + 1


def _compile_glob_variants(pattern: str) -> tuple[list[str], list[str] | None]:
    """Compile one glob into ``(main_globs, excluded_globs)`` for OpenCode's ``*``/``?`` matcher.

    Positive ``[...]`` classes expand finitely (exact). A single negated ``[!...]``
    class returns ``?``-substituted ``main_globs`` paired with the exact
    ``excluded_globs`` so callers can emit paired allow/deny rules without
    approximating or widening scope. Multiple negated classes or expansions
    exceeding ``_MAX_CHAR_CLASS_EXPANSIONS`` raise ``ValueError``.
    """
    if "[" not in pattern:
        return [pattern], None
    main_variants = [""]
    excluded_variants: list[str] | None = None
    index = 0
    length = len(pattern)
    while index < length:
        if pattern[index] == "[":
            parsed = _parse_char_class(pattern, index)
            if parsed is not None:
                negated, choices, next_index = parsed
                if negated:
                    if excluded_variants is not None:
                        raise ValueError(
                            f"OpenCode declarative permission rules cannot faithfully "
                            f"represent multiple negated character classes in glob {pattern!r}."
                        )
                    if len(main_variants) * len(choices) > _MAX_CHAR_CLASS_EXPANSIONS:
                        raise ValueError(
                            f"OpenCode declarative permission rules cannot faithfully "
                            f"expand negated glob {pattern!r} "
                            f"(exceeds {_MAX_CHAR_CLASS_EXPANSIONS} variants)."
                        )
                    excluded_variants = [
                        prefix + ch for prefix in main_variants for ch in choices
                    ]
                    main_variants = [prefix + "?" for prefix in main_variants]
                else:
                    if len(main_variants) * len(choices) > _MAX_CHAR_CLASS_EXPANSIONS:
                        raise ValueError(
                            f"OpenCode declarative permission rules cannot faithfully "
                            f"expand glob {pattern!r} "
                            f"(exceeds {_MAX_CHAR_CLASS_EXPANSIONS} variants)."
                        )
                    main_variants = [
                        prefix + ch for prefix in main_variants for ch in choices
                    ]
                    if excluded_variants is not None:
                        if len(excluded_variants) * len(choices) > _MAX_CHAR_CLASS_EXPANSIONS:
                            raise ValueError(
                                f"OpenCode declarative permission rules cannot faithfully "
                                f"expand negated glob {pattern!r} "
                                f"(exceeds {_MAX_CHAR_CLASS_EXPANSIONS} variants)."
                            )
                        excluded_variants = [
                            prefix + ch for prefix in excluded_variants for ch in choices
                        ]
                index = next_index
                continue
        ch = pattern[index]
        main_variants = [prefix + ch for prefix in main_variants]
        if excluded_variants is not None:
            excluded_variants = [prefix + ch for prefix in excluded_variants]
        index += 1
    return main_variants, excluded_variants


def _expand_scope_entry(raw: str) -> tuple[list[str], list[str] | None]:
    """Normalize an Olympus write_scope / forbidden_globs entry to OpenCode rules."""
    pattern = _normalize_scope_pattern(raw)
    if not pattern or pattern == ".":
        return ["*"], None
    if any(ch in pattern for ch in ("*", "?", "[")):
        return _compile_glob_variants(pattern)
    return [pattern, f"{pattern}/**"], None


def _set_rule_last(rules: dict[str, str], key: str, effect: str) -> None:
    rules.pop(key, None)
    rules[key] = effect


def _is_concrete_path_allowed(
    path: str,
    *,
    strict_scope: bool,
    write_scope: list[str],
    forbidden_globs: list[str],
) -> bool:
    if any(_matches_scope_entry(path, pat) for pat in forbidden_globs):
        return False
    if not strict_scope:
        return True
    return any(_matches_scope_entry(path, pat) for pat in write_scope)


def _build_agent_permissions(request: DriverTurnRequest) -> dict[str, Any]:
    """Translate DriverTurnRequest scope flags into OpenCode declarative permissions."""
    if request.disable_tools:
        return {"*": "deny"}
    if request.read_only:
        return {
            "*": "deny",
            "read": "allow",
            "grep": "allow",
            "glob": "allow",
            "list": "allow",
            "external_directory": "deny",
        }

    edit_rules: dict[str, str] = {"*": "deny" if request.strict_scope else "allow"}
    if request.strict_scope:
        for entry in request.write_scope:
            main_globs, excluded_globs = _expand_scope_entry(entry)
            for glob in main_globs:
                _set_rule_last(edit_rules, glob, "allow")
            if excluded_globs is not None:
                for ex in excluded_globs:
                    if any(ch in ex for ch in ("*", "?")):
                        raise ValueError(
                            f"OpenCode declarative permission rules cannot combine wildcards "
                            f"with negated character class in write_scope {entry!r}."
                        )
                    if not any(_matches_scope_entry(ex, pat) for pat in request.write_scope):
                        _set_rule_last(edit_rules, ex, "deny")
    for entry in request.forbidden_globs:
        main_globs, excluded_globs = _expand_scope_entry(entry)
        for glob in main_globs:
            _set_rule_last(edit_rules, glob, "deny")
        if excluded_globs is not None:
            for ex in excluded_globs:
                if any(ch in ex for ch in ("*", "?")):
                    raise ValueError(
                        f"OpenCode declarative permission rules cannot combine wildcards "
                        f"with negated character class in forbidden_globs {entry!r}."
                    )
                if _is_concrete_path_allowed(
                    ex,
                    strict_scope=request.strict_scope,
                    write_scope=request.write_scope,
                    forbidden_globs=request.forbidden_globs,
                ):
                    _set_rule_last(edit_rules, ex, "allow")

    return {
        "*": "deny",
        "read": "allow",
        "grep": "allow",
        "glob": "allow",
        "list": "allow",
        "bash": "allow",
        "shell": "allow",
        "edit": edit_rules,
        "write": dict(edit_rules),
        "external_directory": "deny",
    }


def _binding_secrets(binding: ExternalGatewayBinding, env: Mapping[str, str]) -> list[str]:
    secrets: list[str] = []
    if binding.api_key and binding.api_key.strip():
        secrets.append(binding.api_key.strip())
    if binding.api_key_env:
        env_val = (env.get(binding.api_key_env) or "").strip()
        if env_val:
            secrets.append(env_val)
    return secrets


def _is_explicit_provider_model(model: str) -> bool:
    """Return True for OpenCode native ``<provider>/<model>[#<variant>]`` specs."""
    if "/" not in model:
        return False
    provider_part, _, model_part = model.partition("/")
    return bool(provider_part.strip() and model_part.strip())


def _resolve_gateway_binding(
    env: Mapping[str, str],
    model: str | None,
    *,
    provider: LLMProvider | None,
) -> ExternalGatewayBinding | None:
    binder = getattr(provider, "external_gateway_binding", None)
    if callable(binder):
        maybe = binder(model)
        if isinstance(maybe, ExternalGatewayBinding):
            return maybe
    return resolve_external_gateway_from_env(env, model)


def _prepare_opencode_runtime(
    request: DriverTurnRequest,
    *,
    provider: LLMProvider | None,
    context_window: int | None,
    step_cap: int | None = None,
) -> tuple[dict[str, str], str | None, Path, tuple[str, ...]]:
    """Build isolated env, ``OPENCODE_CONFIG_CONTENT``, model arg, and redaction secrets.

    Routing policy:
    * Explicit ``<provider>/<model>[#<variant>]`` passes directly to OpenCode.
    * Bare ``opencode`` or ``opencode@<model>`` (without ``/``) synthesizes an
      ephemeral ``olympus-gateway`` provider from Olympus's
      ``ExternalGatewayBinding`` using ``{env:...}`` references so API keys are
      never serialized into config.
    """
    env = build_sandbox_env(
        request.workspace_path,
        host_root=request.env_root,
        tmp_dir=_read_only_tmp_dir(request),
    )
    resolved_workspace = str(request.workspace_path.resolve())
    env["PWD"] = resolved_workspace
    env.pop("OLDPWD", None)
    for override_key in (
        "OPENCODE_DB",
        "OPENCODE_CONFIG",
        "OPENCODE_CONFIG_DIR",
        "OPENCODE_CONFIG_CONTENT",
        "OPENCODE_PERMISSION",
    ):
        env.pop(override_key, None)

    agent_doc: dict[str, Any] = {
        "mode": "primary",
        "permission": _build_agent_permissions(request),
    }
    if step_cap is not None:
        agent_doc["steps"] = step_cap
    if request.system_prompt.strip():
        agent_doc["prompt"] = request.system_prompt

    config_doc: dict[str, Any] = {
        "agent": {_AGENT_ID: agent_doc},
        "mcp": {},
    }

    requested_model = (request.model or "").strip()
    effective_model: str | None = None
    extra_secrets: list[str] = []

    if requested_model and _is_explicit_provider_model(requested_model):
        effective_model = requested_model
    else:
        binding = _resolve_gateway_binding(env, requested_model or None, provider=provider)
        if binding is None:
            raise OpenCodeNotAvailableError(
                "OpenCodeDriver could not resolve an Olympus ExternalGatewayBinding for "
                f"{requested_model or 'opencode'!r}. Configure an Olympus LLM provider/gateway "
                "or pass an explicit OpenCode provider/model spec (e.g. opencode@provider/model)."
            )
        extra_secrets.extend(_binding_secrets(binding, env))
        if binding.api_key_env:
            api_key_ref = f"{{env:{binding.api_key_env}}}"
        else:
            env[_GATEWAY_KEY_ENV] = binding.api_key or "keyless"
            api_key_ref = f"{{env:{_GATEWAY_KEY_ENV}}}"
        npm_pkg = (
            "@ai-sdk/anthropic"
            if binding.api == "anthropic-messages"
            else "@ai-sdk/openai-compatible"
        )
        model_entry: dict[str, Any] = {"name": binding.model_id}
        effective_window = request.context_window or context_window
        if effective_window is not None:
            model_entry["limit"] = {
                "context": effective_window,
                "output": min(16384, effective_window),
            }
        config_doc["provider"] = {
            _GATEWAY_PROVIDER_ID: {
                "npm": npm_pkg,
                "name": "Olympus Gateway",
                "options": {
                    "baseURL": binding.base_url,
                    "apiKey": api_key_ref,
                },
                "models": {binding.model_id: model_entry},
            }
        }
        effective_model = f"{_GATEWAY_PROVIDER_ID}/{binding.model_id}"

    state_dir = Path(tempfile.mkdtemp(prefix="olympus-opencode-state-"))
    os.chmod(state_dir, 0o700)
    for sub in ("data", "config", "state"):
        sub_path = state_dir / sub
        sub_path.mkdir(mode=0o700, exist_ok=True)
    env["XDG_DATA_HOME"] = str(state_dir / "data")
    env["XDG_CONFIG_HOME"] = str(state_dir / "config")
    env["XDG_STATE_HOME"] = str(state_dir / "state")
    env.setdefault("OPENCODE_DISABLE_AUTOUPDATE", "1")
    env["OPENCODE_DISABLE_CLAUDE_CODE"] = "1"
    env["OPENCODE_DISABLE_DEFAULT_PLUGINS"] = "1"
    env.setdefault("OPENCODE_DISABLE_LSP_DOWNLOAD", "1")
    env["OPENCODE_DISABLE_PROJECT_CONFIG"] = "1"
    env["OPENCODE_CONFIG_PROJECT_DISABLE"] = "1"
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config_doc, ensure_ascii=False)
    redaction_secrets = _collect_redaction_secrets(env, tuple(extra_secrets))
    return env, effective_model, state_dir, redaction_secrets


def _safe_argv(command: list[str]) -> list[str]:
    """Redact trailing prompt arguments so traces stay introspectable and compact."""
    safe: list[str] = []
    after_separator = False
    for index, argument in enumerate(command):
        if index == 0:
            continue
        if after_separator:
            safe.append(f"<prompt: {len(argument)} chars>")
            after_separator = False
        elif argument == "--":
            safe.append(argument)
            after_separator = True
        else:
            safe.append(argument)
    return safe


class _TraceWriter:
    """Append-only JSONL trace guarded across stdout/stderr/cancel threads."""

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


class _OpenCodeTurn:
    """Single-turn lifecycle coordinator implementing ``ActiveTurnHandle``."""

    def __init__(
        self,
        *,
        request: DriverTurnRequest,
        binary: str,
        store: RunStore | None,
        run_id: str | None,
        provider: LLMProvider | None,
        context_window: int | None,
        cancel_grace_seconds: float,
        term_grace_seconds: float,
        on_event: DriverEvent | None,
    ) -> None:
        self._request = request
        self._binary = binary
        self._store = store
        self._run_id = run_id
        self._provider = provider
        self._context_window = context_window
        self._cancel_grace_seconds = cancel_grace_seconds
        self._term_grace_seconds = term_grace_seconds
        self._emit_callback = on_event

        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._trace = _TraceWriter(request.trace_path)
        self._process: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._state_dir: Path | None = None
        self._effective_model: str | None = None
        self._redaction_secrets: tuple[str, ...] = ()

        self._cancel_requested = False
        self._max_steps_exceeded = False
        self._budget_exhausted_reason = ""
        self._reserved_slots = 0
        self._saw_error_event = False
        self._last_error_message = ""
        self._last_stderr_line = ""
        self._returncode: int | None = None

        self._steps = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cache_read_tokens = 0
        self._total_cache_write_tokens = 0
        self._total_reasoning_tokens = 0
        self._step_text_parts: list[str] = []
        self._final_content = ""
        self._started_tool_ids: set[str] = set()
        self._receipt_events: list[dict[str, Any]] = []

    @property
    def supports_steer(self) -> bool:
        return False

    def steer(self, guidance: str, *, source: str) -> bool:
        return False

    def cancel(self, *, reason: str) -> None:
        with self._lock:
            if self._cancel_requested:
                return
            self._cancel_requested = True
            self._trace.append({"event": "agent.cancelled", "reason": reason})
        if self._closed.is_set():
            return
        self._signal_group(signal.SIGTERM)
        if threading.current_thread() is self._reader:
            threading.Thread(
                target=self._escalate_cancel,
                name="opencode-cancel-escalation",
                daemon=True,
            ).start()
            return
        self._escalate_cancel()

    def _acquire_budget_and_step_cap(self) -> tuple[bool, int | None]:
        """Atomically reserve call slots so OpenCode can never exceed ``max_llm_calls``."""
        max_calls_for_turn: int | None = None
        if self._store is not None and self._run_id is not None:
            policy_max_calls: int | None = None
            try:
                snapshot = self._store.get_run(self._run_id)
                policy_max_calls = snapshot["budget_policy"].max_llm_calls
            except Exception as exc:  # noqa: BLE001 - preflight read errors fall back safely
                self._trace.append({"event": "budget.preflight_error", "error": str(exc)[:300]})

            if policy_max_calls is not None:
                desired_slots = (
                    min(self._request.max_steps, policy_max_calls)
                    if self._request.max_steps is not None
                    else min(policy_max_calls, _MAX_PRE_RESERVE_STEPS)
                )
                desired_slots = max(1, desired_slots)
            else:
                desired_slots = 1

            reserved = 0
            for slot_idx in range(1, desired_slots + 1):
                try:
                    self._store.reserve_llm_call(
                        self._run_id,
                        actor=self._request.actor,
                        purpose=f"opencode_step_{slot_idx}",
                    )
                except ValueError as exc:
                    if "budget" in str(exc).lower():
                        break
                    self._trace.append({"event": "budget.reserve_error", "error": str(exc)[:300]})
                    break
                except Exception as exc:  # noqa: BLE001 - reserve errors fall back safely
                    self._trace.append({"event": "budget.reserve_error", "error": str(exc)[:300]})
                    break
                reserved += 1

            with self._lock:
                self._reserved_slots = reserved
            if reserved == 0:
                return True, 0
            if policy_max_calls is not None:
                max_calls_for_turn = reserved

        caps = [cap for cap in (self._request.max_steps, max_calls_for_turn) if cap is not None]
        return False, (min(caps) if caps else None)

    def _release_reserved_slots(self) -> None:
        with self._lock:
            remaining = self._reserved_slots
            self._reserved_slots = 0
        if remaining <= 0 or self._store is None or self._run_id is None:
            return
        for _ in range(remaining):
            with suppress(Exception):
                self._store.settle_llm_call(
                    self._run_id,
                    actor=self._request.actor,
                    cancelled=True,
                )

    def execute(
        self, on_handle: Callable[[ActiveTurnHandle], None] | None = None
    ) -> TurnOutcome:
        preflight_exhausted, step_cap = self._acquire_budget_and_step_cap()
        if preflight_exhausted:
            self._budget_exhausted_reason = "budget_exhausted: OpenCode exceeded configured budget"
            self._trace.close()
        else:
            try:
                self._spawn(step_cap=step_cap)
                reader = threading.Thread(
                    target=self._pump_stdout, name="opencode-stdout", daemon=True
                )
                stderr_reader = threading.Thread(
                    target=self._pump_stderr, name="opencode-stderr", daemon=True
                )
                self._reader = reader
                self._stderr_reader = stderr_reader
                reader.start()
                stderr_reader.start()
                if on_handle is not None:
                    on_handle(self)
                self._await_stdout_close()
            finally:
                self._reap()
                self._release_reserved_slots()
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
            driver_name="opencode",
        )

    def _spawn(self, *, step_cap: int | None = None) -> None:
        env, effective_model, state_dir, secrets = _prepare_opencode_runtime(
            self._request,
            provider=self._provider,
            context_window=self._context_window,
            step_cap=step_cap,
        )
        self._effective_model = effective_model
        self._state_dir = state_dir
        self._redaction_secrets = secrets
        self._trace.set_secrets(secrets)
        command = self._build_command()
        self._trace.append({"event": "opencode.spawn", "argv": _safe_argv(command)})
        try:
            self._process = subprocess.Popen(
                command,
                cwd=self._request.workspace_path,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise OpenCodeNotAvailableError(
                f"OpenCodeDriver could not launch {self._binary!r}: {exc}. {_INSTALL_HINT}"
            ) from exc

    def _build_command(self) -> list[str]:
        command = [
            self._binary, "run", "--standalone", "--archived",
            "--format", "json", "--auto",
            "--title", f"olympus:{self._request.actor}",
            "--agent", _AGENT_ID,
        ]
        if self._effective_model:
            command += ["--model", self._effective_model]
        command += ["--", self._request.user_prompt]
        return command

    def _await_stdout_close(self) -> None:
        proc = self._process
        if proc is None:
            return
        while not self._closed.wait(0.05):
            if proc.poll() is not None:
                if self._closed.wait(0.5):
                    break
                self._signal_group(signal.SIGKILL)
                self._closed.wait(2.0)
                break

    def _escalate_cancel(self) -> None:
        if self._closed.wait(self._term_grace_seconds):
            return
        self._signal_group(signal.SIGKILL)
        self._closed.wait(self._cancel_grace_seconds)

    def _signal_group(self, sig: signal.Signals) -> None:
        proc = self._process
        if proc is None:
            return
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            with suppress(ProcessLookupError, PermissionError):
                proc.send_signal(sig)

    def _reap(self) -> None:
        try:
            proc = self._process
            if proc is not None:
                if proc.poll() is None:
                    self._signal_group(signal.SIGTERM)
                    with suppress(subprocess.TimeoutExpired):
                        proc.wait(timeout=self._term_grace_seconds)
                if proc.poll() is None:
                    self._signal_group(signal.SIGKILL)
                    with suppress(subprocess.TimeoutExpired):
                        proc.wait(timeout=self._cancel_grace_seconds)
                self._signal_group(signal.SIGKILL)
                self._returncode = proc.poll()
                if self._reader is not None:
                    self._reader.join(timeout=5.0)
                if self._stderr_reader is not None:
                    self._stderr_reader.join(timeout=2.0)
                for stream in (proc.stdout, proc.stderr):
                    if stream is not None:
                        with suppress(OSError):
                            stream.close()
        finally:
            if self._state_dir is not None:
                with suppress(OSError):
                    shutil.rmtree(self._state_dir, ignore_errors=True)
                self._state_dir = None

    def _pump_stderr(self) -> None:
        proc = self._process
        if proc is None or proc.stderr is None:
            return
        traced_lines = 0
        for raw_line in proc.stderr:
            text = raw_line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            redacted = _redact_secrets(text, self._redaction_secrets)[:300]
            with self._lock:
                self._last_stderr_line = redacted
            if traced_lines < _MAX_STDERR_TRACE_LINES:
                self._trace.append({"event": "opencode.stderr", "line": redacted})
                traced_lines += 1

    def _pump_stdout(self) -> None:
        try:
            proc = self._process
            if proc is None or proc.stdout is None:
                return
            for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    self._trace.append(
                        {
                            "event": "opencode.non_json_stdout",
                            "line": _redact_secrets(line, self._redaction_secrets)[:240],
                        }
                    )
                    continue
                if isinstance(record, dict):
                    self._handle_record(record)
        finally:
            self._closed.set()

    def _handle_record(self, record: dict[str, Any]) -> None:
        record_type = str(record.get("type") or "")
        if record_type == "step_start":
            with self._lock:
                self._step_text_parts = []
        elif record_type == "text":
            self._on_text(record)
        elif record_type == "tool_use":
            self._on_tool_use(record)
        elif record_type == "step_finish":
            self._on_step_finish(record)
        elif record_type == "error":
            self._on_error(record)

    def _on_text(self, record: dict[str, Any]) -> None:
        part = record.get("part")
        if not isinstance(part, dict):
            return
        text = part.get("text")
        if not isinstance(text, str) or not text:
            return
        with self._lock:
            self._step_text_parts.append(text)
            self._final_content = "\n".join(self._step_text_parts)
        self._trace.append({"event": "text_delta", "text": text})
        self._emit("text_delta", {"text": text, "replace": False})

    def _on_tool_use(self, record: dict[str, Any]) -> None:
        part = record.get("part")
        if not isinstance(part, dict):
            return
        tool = str(part.get("tool") or "unknown")
        call_id = str(part.get("id") or part.get("partID") or "")
        state = part.get("state")
        state = state if isinstance(state, dict) else {}
        status_raw = str(state.get("status") or "completed").lower()

        with self._lock:
            step = self._steps + 1
            emit_started = not call_id or call_id not in self._started_tool_ids
            if call_id:
                self._started_tool_ids.add(call_id)

        if emit_started:
            self._emit(
                "agent.tool_started",
                {
                    "actor": self._request.actor,
                    "task_id": self._request.task_id,
                    "step": step,
                    "tool": tool,
                },
            )

        if status_raw not in ("completed", "error"):
            return

        is_error = status_raw == "error"
        raw_arguments = state.get("input")
        arguments: dict[str, Any] = raw_arguments if isinstance(raw_arguments, dict) else {}
        target = _short_target(arguments)
        raw_error = str(state.get("error") or "")
        raw_output = raw_error if is_error else str(state.get("output") or "")
        output_preview = _redact_secrets(raw_output, self._redaction_secrets)[:240]
        is_permission_denied = is_error and raw_error.startswith("Permission denied:")

        raw_timing = state.get("time")
        timing: dict[str, Any] = raw_timing if isinstance(raw_timing, dict) else {}
        start_ms = _non_negative_int(timing.get("start"))
        end_ms = _non_negative_int(timing.get("end"))
        duration_ms = max(0, end_ms - start_ms) if end_ms >= start_ms else 0
        error_signature = (
            content_hash({"tool": tool, "output": output_preview}) if is_error else ""
        )

        self._trace.append(
            {
                "event": "tool_end", "step": step, "tool": tool, "target": target,
                "arguments": arguments, "status": "error" if is_error else "ok",
                "is_error": is_error, "duration_ms": duration_ms, "output": output_preview,
            }
        )
        if is_permission_denied:
            self._receipt_events.append({"kind": "pre_tool_blocked", "tool": tool})
        else:
            self._receipt_events.append(
                {
                    "kind": "tool_end", "tool": tool,
                    "is_error": is_error, "duration_ms": duration_ms,
                }
            )
        self._emit(
            "agent.tool_finished",
            {
                "actor": self._request.actor, "task_id": self._request.task_id,
                "step": step, "tool": tool, "status": "error" if is_error else "ok",
                "is_error": is_error, "duration_ms": duration_ms,
                "error_signature": error_signature,
            },
        )

    def _on_step_finish(self, record: dict[str, Any]) -> None:
        part = record.get("part")
        part = part if isinstance(part, dict) else {}
        tokens = part.get("tokens")
        tokens = tokens if isinstance(tokens, dict) else {}
        cache = tokens.get("cache")
        cache = cache if isinstance(cache, dict) else {}

        input_tokens = _non_negative_int(tokens.get("input"))
        reasoning_tokens = _non_negative_int(tokens.get("reasoning"))
        output_tokens = _non_negative_int(tokens.get("output")) + reasoning_tokens
        cache_read_tokens = _non_negative_int(cache.get("read"))
        cache_write_tokens = _non_negative_int(cache.get("write"))
        reason = str(part.get("reason") or "")

        with self._lock:
            self._steps += 1
            step = self._steps
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_cache_read_tokens += cache_read_tokens
            self._total_cache_write_tokens += cache_write_tokens
            self._total_reasoning_tokens += reasoning_tokens
            self._saw_error_event = False
            self._last_error_message = ""

        llm_payload = {
            "actor": self._request.actor,
            "task_id": self._request.task_id,
            "step": step,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "reasoning_tokens": reasoning_tokens,
        }
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
        self._receipt_events.append({"kind": "llm", **llm_payload})
        over_budget = self._settle_usage(
            step,
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_tokens,
            reasoning_tokens,
            will_continue=reason != "stop",
        )
        self._emit("llm", llm_payload)

        if over_budget:
            with self._lock:
                self._budget_exhausted_reason = (
                    "budget_exhausted: OpenCode exceeded configured budget"
                )
            self.cancel(reason="budget_exhausted")
            return

        if self._request.max_steps is not None and (
            (step >= self._request.max_steps and reason != "stop")
            or step > self._request.max_steps
        ):
            with self._lock:
                self._max_steps_exceeded = True
            self.cancel(reason="max_steps_exceeded")

    def _on_error(self, record: dict[str, Any]) -> None:
        raw_error = record.get("error")
        if isinstance(raw_error, dict):
            detail = str(raw_error.get("message") or raw_error.get("type") or "").strip()
        else:
            detail = str(raw_error or "").strip()
        redacted = _redact_secrets(detail, self._redaction_secrets)[:300]
        with self._lock:
            self._saw_error_event = True
            self._last_error_message = redacted
        self._trace.append({"event": "opencode.error", "error": redacted})

    def _settle_usage(
        self,
        step: int,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        reasoning_tokens: int,
        *,
        will_continue: bool = False,
    ) -> bool:
        if self._store is None or self._run_id is None:
            return False
        with self._lock:
            had_reservation = self._reserved_slots > 0
            if had_reservation:
                self._reserved_slots -= 1
            slots_left = self._reserved_slots
        try:
            if had_reservation:
                updated = self._store.settle_llm_call(
                    self._run_id,
                    actor=self._request.actor,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                    reasoning_tokens=reasoning_tokens,
                )
                policy = self._store.get_run(self._run_id)["budget_policy"]
                over_budget = (
                    policy.max_total_tokens is not None
                    and updated.total_tokens > policy.max_total_tokens
                )
            else:
                over_budget = self._store.record_external_llm_usage(
                    self._run_id,
                    actor=self._request.actor,
                    purpose=f"opencode_step_{step}",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                    reasoning_tokens=reasoning_tokens,
                )
            if not over_budget and will_continue and slots_left == 0:
                if self._request.max_steps is not None and step >= self._request.max_steps:
                    return False
                try:
                    self._store.reserve_llm_call(
                        self._run_id,
                        actor=self._request.actor,
                        purpose=f"opencode_step_{step + 1}",
                    )
                except ValueError as exc:
                    if "budget" in str(exc).lower():
                        return True
                    raise
                with self._lock:
                    self._reserved_slots += 1
            return over_budget
        except Exception as exc:  # noqa: BLE001 - usage accounting must not kill the turn
            self._trace.append(
                {"event": "usage.settle_error", "step": step, "error": str(exc)[:300]}
            )
            return False

    def _opencode_error_reason(self, detail: str = "") -> str:
        clean = (detail or self._last_error_message or self._last_stderr_line).strip()
        return f"opencode_error: {clean}" if clean else "opencode_error"

    def _classify_outcome(self) -> tuple[bool, str, str]:
        if self._budget_exhausted_reason:
            return False, self._budget_exhausted_reason, ""
        if self._max_steps_exceeded:
            return False, "max_steps_exceeded", self._final_content
        if self._cancel_requested:
            return False, "cancelled", ""
        if self._saw_error_event:
            return False, self._opencode_error_reason(self._last_error_message), ""
        if self._returncode not in (0, None):
            return False, self._opencode_error_reason(), ""
        if self._steps == 0:
            return False, self._opencode_error_reason(), ""
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


class OpenCodeDriver:
    """Run one OpenCode turn inside the task workspace behind the ``AgentDriver`` seam."""

    name = "opencode"
    records_usage_internally = True

    def __init__(
        self,
        *,
        binary: str | None = None,
        store: RunStore | None = None,
        run_id: str | None = None,
        provider: LLMProvider | None = None,
        context_window: int | None = None,
        cancel_grace_seconds: float = 3.0,
        term_grace_seconds: float = 2.0,
    ) -> None:
        self._binary = _resolve_binary(binary)
        self._store = store
        self._run_id = run_id
        self._provider = provider
        self._context_window = context_window
        self._cancel_grace_seconds = cancel_grace_seconds
        self._term_grace_seconds = term_grace_seconds

    def run_turn(
        self,
        request: DriverTurnRequest,
        *,
        on_event: DriverEvent | None = None,
        on_handle: Callable[[ActiveTurnHandle], None] | None = None,
    ) -> TurnOutcome:
        probe_env = build_sandbox_env(
            request.workspace_path,
            host_root=request.env_root,
            tmp_dir=_read_only_tmp_dir(request),
        )
        _probe_run_json_support(self._binary, probe_env)
        turn = _OpenCodeTurn(
            request=request,
            binary=self._binary,
            store=self._store,
            run_id=self._run_id,
            provider=self._provider,
            context_window=self._context_window,
            cancel_grace_seconds=self._cancel_grace_seconds,
            term_grace_seconds=self._term_grace_seconds,
            on_event=on_event,
        )
        return turn.execute(on_handle=on_handle)
