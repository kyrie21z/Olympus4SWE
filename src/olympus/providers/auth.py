"""Credential discovery, endpoint targeting, and sticky failover policy.

Everything here is vendor-neutral: a provider is an ordered pool of
``EndpointTarget`` values resolved from environment variables or files.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from olympus.providers.types import (
    ChatMessage,
    HttpStatusError,
    LLMResponse,
    ProviderError,
    ToolSpec,
)

if TYPE_CHECKING:
    from olympus.providers.dialects import Dialect

_FAILOVER_STATUS_CODES = frozenset({401, 402, 403, 429})
_QUOTA_BODY_MARKERS = (
    "quota",
    "insufficient",
    "arrearage",
    "allocation",
    "rate limit",
    "exceeded",
    "balance",
)


@dataclass(frozen=True)
class EndpointTarget:
    """One concrete API key + base URL pair a dialect can post to."""

    api_key: str
    base_url: str
    headers: dict[str, str] = field(default_factory=dict)
    is_custom_endpoint: bool = True


@dataclass(frozen=True)
class ExternalGatewayBinding:
    """Wire-protocol gateway descriptor exported to external agent drivers."""

    api: str
    base_url: str
    model_id: str
    api_key: str = ""
    api_key_env: str | None = None
    is_custom_endpoint: bool = True


def read_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE or ``export KEY=VALUE`` entries from a dotenv-style file."""
    expanded = path.expanduser()
    if not expanded.is_file():
        return {}
    parsed: dict[str, str] = {}
    for raw_line in expanded.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, value = line.split("=", 1)
        clean_key = key.strip()
        clean_val = value.strip()
        if len(clean_val) >= 2 and clean_val[0] == clean_val[-1] and clean_val[0] in ('"', "'"):
            clean_val = clean_val[1:-1]
        if clean_key:
            parsed[clean_key] = clean_val
    return parsed


def resolve_env_mapping(
    *,
    provider_env_file: Path | None = None,
    cwd: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Merge settings with the documented precedence.

    Lowest to highest: ``provider_env_file`` -> ``<cwd>/.env`` ->
    ``OLYMPUS_ENV_FILE`` -> process environment.
    """
    proc_env = os.environ if environ is None else environ
    merged: dict[str, str] = {}

    if provider_env_file is not None:
        merged.update(read_env_file(provider_env_file))

    merged.update(read_env_file((cwd or Path.cwd()) / ".env"))

    env_file_override = proc_env.get("OLYMPUS_ENV_FILE", "").strip()
    if env_file_override:
        merged.update(read_env_file(Path(env_file_override)))

    merged.update(proc_env)
    return merged


def apply_url_rewrites(raw_url: str, rewrites: Mapping[str, str]) -> str:
    """Replace matching URL suffixes, leaving unknown URLs untouched.

    Rewrites only exist so gateway quirks live in data, never in code.
    """
    result = raw_url.strip().rstrip("/")
    for suffix, replacement in rewrites.items():
        clean_suffix = suffix.strip().rstrip("/")
        if clean_suffix and result.endswith(clean_suffix):
            result = result[: -len(clean_suffix)] + replacement.strip().rstrip("/")
            result = result.rstrip("/")
    return result


class CredentialPool:
    """Ordered endpoints with a thread-safe sticky active index shared across calls."""

    def __init__(self, endpoints: Sequence[EndpointTarget]) -> None:
        self._endpoints = tuple(endpoints)
        self._active_index = 0
        self._lock = threading.Lock()

    @property
    def endpoints(self) -> tuple[EndpointTarget, ...]:
        return self._endpoints

    @property
    def active_index(self) -> int:
        with self._lock:
            return self._active_index

    @property
    def active(self) -> EndpointTarget | None:
        with self._lock:
            if 0 <= self._active_index < len(self._endpoints):
                return self._endpoints[self._active_index]
            return None

    def active_with_index(self) -> tuple[int, EndpointTarget | None]:
        """Atomically read the current active index and its endpoint."""
        with self._lock:
            idx = self._active_index
            if 0 <= idx < len(self._endpoints):
                return idx, self._endpoints[idx]
            return idx, None

    def advance(self) -> bool:
        """Move to the next endpoint; return False when the pool is exhausted."""
        with self._lock:
            if self._active_index + 1 < len(self._endpoints):
                self._active_index += 1
                return True
            return False

    def advance_from(self, failed_index: int) -> bool:
        """Compare-and-advance failover: advance only if ``failed_index`` is still active.

        If another thread already advanced past ``failed_index``, do not skip the newly
        selected endpoint; return True so the caller retries on the new active endpoint.
        """
        with self._lock:
            if self._active_index != failed_index:
                return 0 <= self._active_index < len(self._endpoints)
            if self._active_index + 1 < len(self._endpoints):
                self._active_index += 1
                return True
            return False

    def should_failover(self, exc: HttpStatusError) -> bool:
        """Return True when a failure looks like credential or quota exhaustion."""
        if exc.status_code in _FAILOVER_STATUS_CODES:
            return True
        if exc.status_code == 400:
            body = exc.body_preview.lower()
            return any(marker in body for marker in _QUOTA_BODY_MARKERS)
        return False


class PooledProvider:
    """``LLMProvider`` that drives a dialect over a credential pool."""

    def __init__(
        self,
        *,
        model_name: str,
        dialect: Dialect,
        pool: CredentialPool,
        timeout: float = 300.0,
    ) -> None:
        self.model_name = model_name
        self.dialect = dialect
        self.pool = pool
        self.timeout = timeout

    def external_gateway_binding(
        self, model: str | None = None
    ) -> ExternalGatewayBinding | None:
        """Return the active endpoint + wire dialect for external CLI drivers."""
        endpoint = self.pool.active
        wire_api = str(getattr(self.dialect, "wire_api", "") or "").strip()
        if endpoint is None or not endpoint.base_url.strip() or not wire_api:
            return None
        chosen = (model or self.model_name or "").strip()
        if not chosen:
            return None
        if "/" in chosen:
            prefix, rest = chosen.split("/", 1)
            if prefix.strip().lower() in {"openai", "anthropic"} and rest.strip():
                chosen = rest.strip()
        return ExternalGatewayBinding(
            api=wire_api,
            base_url=endpoint.base_url.strip(),
            model_id=chosen,
            api_key=endpoint.api_key,
            is_custom_endpoint=endpoint.is_custom_endpoint,
        )

    def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
        temperature: float = 0.0,
        enable_thinking: bool | None = None,
        stream: bool = False,
        timeout: float | None = None,
    ) -> LLMResponse:
        if not self.pool.endpoints:
            raise ProviderError(
                "No credentials resolved for this provider. Set an API key environment "
                "variable, configure providers.json, or pass --base-url."
            )
        effective_timeout = self.timeout if timeout is None else timeout
        while True:
            attempt_index, endpoint = self.pool.active_with_index()
            if endpoint is None:
                raise ProviderError("All configured provider endpoints are exhausted.")
            try:
                return self.dialect.send(
                    endpoint,
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    timeout=effective_timeout,
                    enable_thinking=enable_thinking,
                    stream=stream,
                )
            except HttpStatusError as exc:
                if not self.pool.should_failover(exc) or not self.pool.advance_from(attempt_index):
                    raise
